"""Fail-closed subprocess adapter for IOPaint's built-in LaMa model.

IOPaint is intentionally kept out of the hybrid package's Python environment.
The adapter invokes the supported ``iopaint run`` command with one private,
ordered image/mask batch, pins the process controls that the command exposes,
and keeps all user paths out of its receipt-facing metadata.

No model is downloaded here.  The caller must place and attest the exact
``big-lama.pt`` file that IOPaint will load before invoking the adapter.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

try:
    import resource as _resource
except ImportError:  # pragma: no cover - exercised through the explicit fallback
    _resource = None

import imageio.v3 as iio
import numpy as np
import numpy.typing as npt


IOPAINT_SUPPORTED_VERSION: Final = "1.6.0"
IOPAINT_PACKAGE_LICENSE_SPDX: Final = "Apache-2.0"
IOPAINT_LAMA_RELEASE: Final = "Sanster/models:add_big_lama"
IOPAINT_LAMA_FILENAME: Final = "big-lama.pt"
IOPAINT_LAMA_UPSTREAM_LICENSE_SPDX: Final = "Apache-2.0"
IOPAINT_LAMA_ARTIFACT_LICENSE_STATUS: Final = (
    "upstream LaMa repository is Apache-2.0; the exact converted IOPaint "
    "weight release does not state a separate artifact license"
)

_MODEL_NAME: Final = "lama"
_PRIVATE_IMAGE_INDEX_WIDTH: Final = 8
_PRIVATE_CONFIG_NAME: Final = "inpaint-config.json"
_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_ARTIFACT_ID_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/@+-]*$")
_VERSION_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+_-]*$")
_SAFE_RUNTIME_TEXT_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 .,+_/:()@-]*$")
_SAFE_DEVICE_NAME_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 .,+_()@-]*$")
_CUDA_UUID_BODY: Final = (
    r"(?:[0-9A-Fa-f]{1,32}|"
    r"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
    r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12})"
)
_CUDA_VISIBLE_TOKEN_RE: Final = re.compile(
    rf"^(?:0|[1-9][0-9]*|GPU-{_CUDA_UUID_BODY}|MIG-{_CUDA_UUID_BODY}|"
    rf"MIG-GPU-{_CUDA_UUID_BODY}/(?:0|[1-9][0-9]*)/(?:0|[1-9][0-9]*))$"
)
_CUDA_VISIBLE_MAX_LENGTH: Final = 4096
_CUDA_VISIBLE_MAX_TOKENS: Final = 128
_PROBE_OUTPUT_MAX_BYTES: Final = 1024 * 1024
# Seven days stays well below signed 32-bit millisecond wait conversions used
# by common subprocess backends while leaving ample room for real frame batches.
_SUBPROCESS_TIMEOUT_MAX_SECONDS: Final = 7 * 24 * 60 * 60
_REQUIRED_RUN_HELP: Final = (
    "--model",
    "--device",
    "--image",
    "--mask",
    "--output",
    "--config",
    "--model-dir",
    "--no-concat",
)
_RUNTIME_PROBE = """
import hashlib
import importlib.metadata as m
import json
import os
import platform

import torch

distribution = m.distribution("IOPaint")
source_digest = hashlib.sha256()
source_count = 0
for entry in sorted(distribution.files or (), key=lambda item: str(item)):
    name = str(entry)
    if not (name.startswith("iopaint/") and name.endswith(".py")):
        continue
    source_digest.update(name.encode("utf-8"))
    source_digest.update(b"\\0")
    source_digest.update(
        hashlib.sha256(distribution.locate_file(entry).read_bytes()).digest()
    )
    source_count += 1
if source_count == 0:
    raise RuntimeError("IOPaint distribution contains no Python sources")

cuda_available = torch.cuda.is_available()
cuda_device_names = []
if cuda_available:
    cuda_device_names = [
        torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
    ]
cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
if cuda_visible_devices is None:
    cuda_visible_devices = "unset"
elif cuda_visible_devices == "":
    cuda_visible_devices = "empty"
mps_backend = getattr(torch.backends, "mps", None)
mps_available = bool(
    mps_backend is not None
    and getattr(mps_backend, "is_available", lambda: False)()
)
mps_device_name = None
if mps_available:
    get_mps_name = getattr(mps_backend, "get_name", None)
    if get_mps_name is not None:
        mps_device_name = get_mps_name()

document = {
    "cudnn_benchmark": torch.backends.cudnn.benchmark,
    "cuda_available": cuda_available,
    "cuda_device_names": cuda_device_names,
    "cuda_runtime_version": torch.version.cuda,
    "cuda_visible_devices": cuda_visible_devices,
    "cudnn_version": torch.backends.cudnn.version(),
    "hip_runtime_version": getattr(torch.version, "hip", None),
    "iopaint_source_manifest_sha256": source_digest.hexdigest(),
    "iopaint_source_file_count": source_count,
    "iopaint_version": m.version("IOPaint"),
    "mps_available": mps_available,
    "mps_device_name": mps_device_name,
    "numpy_version": m.version("numpy"),
    "opencv_version": m.version("opencv-python"),
    "pillow_version": m.version("Pillow"),
    "platform_machine": platform.machine(),
    "platform_release": platform.release(),
    "platform_system": platform.system(),
    "pydantic_version": m.version("pydantic"),
    "python_implementation": platform.python_implementation(),
    "python_version": platform.python_version(),
    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    "torch_version": m.version("torch"),
    "typer_version": m.version("typer"),
}
print(json.dumps(document, sort_keys=True, separators=(",", ":")))
"""
_THREAD_ENV_KEYS: Final = (
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
_WINDOWS_RUNTIME_ENV_KEYS: Final = (
    "COMSPEC",
    "PATHEXT",
    "SystemRoot",
    "WINDIR",
)


UInt8RgbImage = npt.NDArray[np.uint8]
UInt8Mask = npt.NDArray[np.uint8]


class IOPaintError(RuntimeError):
    """Base class for adapter failures."""


class IOPaintUnavailableError(IOPaintError):
    """Raised when the isolated IOPaint runtime is absent or incompatible."""


class IOPaintArtifactError(IOPaintError):
    """Raised when the exact model artifact cannot be attested."""


class IOPaintExecutionError(IOPaintError):
    """Raised when IOPaint exits unsuccessfully or times out."""


class IOPaintOutputError(IOPaintError):
    """Raised when IOPaint does not produce the promised RGB uint8 crop."""


def _validated_config_timeout(value: int | float) -> float:
    try:
        normalized = float(value)
    except (OverflowError, ValueError):
        raise ValueError("timeout_seconds exceeds the subprocess-safe limit") from None
    if not math.isfinite(normalized) or normalized <= 0:
        raise ValueError("timeout_seconds must be finite and > 0")
    if normalized > _SUBPROCESS_TIMEOUT_MAX_SECONDS:
        raise ValueError("timeout_seconds exceeds the subprocess-safe limit")
    return normalized


def _validated_batch_timeout(per_item_seconds: int | float, item_count: int) -> float:
    """Compute one bounded batch deadline before any external runtime access."""

    try:
        per_item = float(per_item_seconds)
        if (
            item_count < 1
            or not math.isfinite(per_item)
            or per_item <= 0
            or item_count > _SUBPROCESS_TIMEOUT_MAX_SECONDS / per_item
        ):
            raise IOPaintExecutionError(
                "IOPaint batch timeout exceeds the subprocess-safe limit"
            )
        timeout = per_item * item_count
    except (OverflowError, ValueError):
        raise IOPaintExecutionError(
            "IOPaint batch timeout exceeds the subprocess-safe limit"
        ) from None
    if not math.isfinite(timeout) or timeout > _SUBPROCESS_TIMEOUT_MAX_SECONDS:
        raise IOPaintExecutionError(
            "IOPaint batch timeout exceeds the subprocess-safe limit"
        )
    return timeout


@dataclass(frozen=True)
class _IOPaintSourceMeasurement:
    """Private, independently measured identity of one installed source tree."""

    package_root: Path
    relative_names: tuple[str, ...]
    sha256: str

    @property
    def file_count(self) -> int:
        return len(self.relative_names)


def _validate_cuda_visible_metadata(value: object) -> str:
    if not isinstance(value, str):
        raise IOPaintUnavailableError(
            "IOPaint runtime probe returned invalid CUDA visibility metadata"
        )
    if value in ("unset", "empty", "-1"):
        return value
    if len(value) > _CUDA_VISIBLE_MAX_LENGTH:
        raise IOPaintUnavailableError(
            "CUDA_VISIBLE_DEVICES exceeds the supported length"
        )
    tokens = value.split(",")
    if (
        not tokens
        or len(tokens) > _CUDA_VISIBLE_MAX_TOKENS
        or any(not _CUDA_VISIBLE_TOKEN_RE.fullmatch(token) for token in tokens)
    ):
        raise IOPaintUnavailableError(
            "CUDA_VISIBLE_DEVICES has unsupported token syntax"
        )
    return value


def _host_cuda_visible_metadata() -> str:
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if value is None:
        return "unset"
    if value == "":
        return "empty"
    return _validate_cuda_visible_metadata(value)


@dataclass(frozen=True)
class ModelArtifactAttestation:
    """Receipt-safe identity for the exact model file to be loaded."""

    identifier: str
    sha256: str

    def __post_init__(self) -> None:
        _validate_artifact_identifier(self.identifier)
        if not _SHA256_RE.fullmatch(self.sha256):
            raise ValueError("model artifact SHA-256 must be 64 lowercase hex digits")


@dataclass(frozen=True)
class IOPaintLaMaConfig:
    """External-runtime and determinism controls for one adapter instance.

    ``mps`` is part of IOPaint's public device enum, so it is accepted at this
    boundary and probed honestly.  IOPaint 1.6.0 nevertheless marks LaMa as
    MPS-unsupported and silently substitutes CPU; the adapter rejects that
    model-specific fallback before inference.
    """

    iopaint_executable: Path
    python_executable: Path
    model_dir: Path
    weights_file: Path
    artifact: ModelArtifactAttestation
    expected_iopaint_source_manifest_sha256: str
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    thread_count: int = 1
    seed: int = 0
    timeout_seconds: float = 600.0
    temp_parent: Path | None = None
    required_iopaint_version: str = IOPAINT_SUPPORTED_VERSION

    def __post_init__(self) -> None:
        if self.device not in ("cpu", "cuda", "mps"):
            raise ValueError("device must be 'cpu', 'cuda', or 'mps' for IOPaint LaMa")
        if isinstance(self.thread_count, bool) or not isinstance(
            self.thread_count, int
        ):
            raise TypeError("thread_count must be an integer")
        if self.thread_count < 1:
            raise ValueError("thread_count must be >= 1")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("seed must be in [0, 2**32 - 1]")
        if isinstance(self.timeout_seconds, bool) or not isinstance(
            self.timeout_seconds, (int, float)
        ):
            raise TypeError("timeout_seconds must be numeric")
        _validated_config_timeout(self.timeout_seconds)
        if not _VERSION_RE.fullmatch(self.required_iopaint_version):
            raise ValueError("required_iopaint_version must be a sanitized version")
        if not _SHA256_RE.fullmatch(self.expected_iopaint_source_manifest_sha256):
            raise ValueError(
                "expected IOPaint source manifest SHA-256 must be 64 lowercase hex digits"
            )


@dataclass(frozen=True)
class IOPaintRuntimeMetadata:
    """Receipt-facing runtime and model provenance without local paths."""

    tool_name: str
    tool_version: str
    tool_license_spdx: str
    iopaint_source_manifest_sha256: str
    iopaint_source_file_count: int
    python_version: str
    python_implementation: str
    torch_version: str
    numpy_version: str
    pillow_version: str
    opencv_version: str
    pydantic_version: str
    typer_version: str
    platform_system: str
    platform_release: str
    platform_machine: str
    deterministic_algorithms: bool
    cudnn_benchmark: bool
    cuda_available: bool
    cuda_runtime_version: str | None
    cudnn_version: str | None
    hip_runtime_version: str | None
    cuda_device_names: tuple[str, ...]
    cuda_visible_devices: str
    mps_available: bool
    mps_device_name: str | None
    effective_environment_sha256: str
    model_name: str
    model_release: str
    model_artifact_identifier: str
    model_weights_sha256: str
    model_upstream_license_spdx: str
    model_artifact_license_status: str
    device: str
    thread_count: int
    seed: int
    seed_scope: str
    determinism_scope: str


@dataclass(frozen=True)
class IOPaintInvocationMetadata:
    """Receipt-facing record for one successful crop invocation."""

    runtime: IOPaintRuntimeMetadata
    sanitized_argv: tuple[str, ...]
    deterministic_environment: tuple[tuple[str, str], ...]
    config_document: tuple[tuple[str, object], ...]
    input_rgb8_raw_sha256: str
    mask_u8_raw_sha256: str
    output_rgb8_raw_sha256: str


def _validate_artifact_identifier(identifier: str) -> None:
    if not _ARTIFACT_ID_RE.fullmatch(identifier):
        raise ValueError(
            "model artifact identifier must be a sanitized relative identifier"
        )
    if identifier.startswith(("/", "~")) or ".." in identifier.split("/"):
        raise ValueError(
            "model artifact identifier must be a sanitized relative identifier"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        raise IOPaintArtifactError(
            "cannot read the attested LaMa weights file"
        ) from None
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def measure_model_artifact(
    weights_file: Path,
    *,
    identifier: str,
) -> ModelArtifactAttestation:
    """Measure a model file without placing its local path in the result."""

    _validate_artifact_identifier(identifier)
    path = Path(weights_file)
    if not path.is_file():
        raise IOPaintArtifactError("LaMa weights file is missing")
    return ModelArtifactAttestation(
        identifier=identifier,
        sha256=_sha256_file(path),
    )


def _resolve_executable(path: Path, *, label: str) -> Path:
    requested = os.fspath(path)
    resolved = shutil.which(requested)
    if resolved is None:
        raise IOPaintUnavailableError(f"{label} executable is unavailable")
    return Path(resolved)


def _require_cli_python_binding(
    iopaint_executable: Path,
    python_executable: Path,
) -> None:
    """Prove that version probing targets the CLI script's interpreter."""

    try:
        with iopaint_executable.open("rb") as handle:
            first_line = handle.readline(4097)
    except OSError:
        raise IOPaintUnavailableError("cannot inspect the IOPaint launcher") from None
    if len(first_line) > 4096 or not first_line.startswith(b"#!"):
        raise IOPaintUnavailableError("IOPaint launcher has no verifiable shebang")
    try:
        interpreter = first_line[2:].decode("utf-8").strip()
    except UnicodeDecodeError:
        raise IOPaintUnavailableError(
            "IOPaint launcher has an invalid shebang"
        ) from None
    # Console scripts generated by pip/uv use one absolute interpreter path.
    # Reject env-based or argument-bearing launchers because they cannot prove
    # which distribution the separate metadata probe is inspecting.
    if not interpreter.startswith("/") or any(
        character.isspace() for character in interpreter
    ):
        raise IOPaintUnavailableError(
            "IOPaint launcher is not bound to one verifiable Python runtime"
        )
    interpreter_path = Path(os.path.abspath(interpreter))
    configured_path = Path(os.path.abspath(os.fspath(python_executable)))
    # Resolve directory aliases such as macOS /tmp -> /private/tmp, but do not
    # resolve the final venv Python symlink: two venvs can share one base
    # interpreter while containing different installed IOPaint distributions.
    bound_interpreter = interpreter_path.parent.resolve() / interpreter_path.name
    configured_interpreter = configured_path.parent.resolve() / configured_path.name
    if bound_interpreter != configured_interpreter:
        raise IOPaintUnavailableError(
            "IOPaint launcher and metadata-probe Python runtimes differ"
        )


def _probe_output_limit_preexec() -> Callable[[], None]:
    """Return a POSIX child hook that hard-limits every probe output file.

    ``subprocess.run`` cannot stream into a bounded regular file itself.  On
    POSIX, ``RLIMIT_FSIZE`` gives the child a kernel-enforced ceiling while it
    is running.  Platforms without that primitive fail closed instead of
    falling back to an after-the-fact size check.
    """

    resource_module = _resource
    if (
        os.name != "posix"
        or resource_module is None
        or not hasattr(resource_module, "RLIMIT_FSIZE")
    ):
        raise IOPaintUnavailableError(
            "a hard IOPaint probe-output limit is unavailable on this platform"
        )

    def apply_limit() -> None:
        _soft_limit, hard_limit = resource_module.getrlimit(
            resource_module.RLIMIT_FSIZE
        )
        limit = _PROBE_OUTPUT_MAX_BYTES
        if hard_limit != resource_module.RLIM_INFINITY:
            limit = min(limit, hard_limit)
        resource_module.setrlimit(resource_module.RLIMIT_FSIZE, (limit, limit))

    return apply_limit


def _find_bound_iopaint_source_root(python_executable: Path) -> Path:
    """Locate exactly one non-symlinked ``iopaint`` tree inside the venv."""

    try:
        runtime_root = python_executable.absolute().parent.resolve(strict=True).parent
        site_package_parents: list[Path] = []
        for library_name in ("Lib", "lib", "lib64"):
            library_root = runtime_root / library_name
            if library_root.is_symlink() or not library_root.is_dir():
                continue
            version_roots = [library_root]
            version_roots.extend(
                entry
                for entry in library_root.iterdir()
                if not entry.is_symlink() and entry.is_dir()
            )
            for version_root in version_roots:
                for packages_name in ("site-packages", "dist-packages"):
                    packages_root = version_root / packages_name
                    if packages_root.is_symlink() or not packages_root.is_dir():
                        continue
                    site_package_parents.append(packages_root)

        candidates: dict[tuple[int, int], Path] = {}
        for packages_root in site_package_parents:
            package_root = packages_root / "iopaint"
            if package_root.is_symlink():
                raise IOPaintUnavailableError(
                    "the bound IOPaint source tree cannot contain path aliases"
                )
            if package_root.is_dir():
                resolved = package_root.resolve(strict=True)
                measured = resolved.stat()
                candidates[(measured.st_dev, measured.st_ino)] = resolved
    except IOPaintUnavailableError:
        raise
    except OSError:
        raise IOPaintUnavailableError(
            "cannot independently locate the bound IOPaint source tree"
        ) from None

    if len(candidates) != 1:
        raise IOPaintUnavailableError(
            "cannot independently locate exactly one bound IOPaint source tree"
        )
    return next(iter(candidates.values()))


def _enumerate_iopaint_python_sources(
    package_root: Path,
) -> tuple[tuple[str, Path], ...]:
    """Enumerate an exact, symlink-free installed Python source set."""

    pending = [package_root]
    entries: list[tuple[str, Path]] = []
    try:
        while pending:
            directory = pending.pop()
            with os.scandir(directory) as scanned:
                children = sorted(scanned, key=lambda item: item.name)
            for child in children:
                path = Path(child.path)
                if child.is_symlink():
                    raise IOPaintUnavailableError(
                        "the bound IOPaint source tree cannot contain path aliases"
                    )
                if child.is_dir(follow_symlinks=False):
                    pending.append(path)
                    continue
                if not child.name.endswith(".py"):
                    continue
                if not child.is_file(follow_symlinks=False):
                    raise IOPaintUnavailableError(
                        "the bound IOPaint source tree contains a non-regular source"
                    )
                relative = path.relative_to(package_root).as_posix()
                entries.append((f"iopaint/{relative}", path))
    except IOPaintUnavailableError:
        raise
    except (OSError, ValueError):
        raise IOPaintUnavailableError(
            "cannot independently enumerate the bound IOPaint source tree"
        ) from None

    entries.sort(key=lambda item: item[0])
    if not entries:
        raise IOPaintUnavailableError(
            "the bound IOPaint source tree contains no Python sources"
        )
    return tuple(entries)


def _sha256_regular_source(path: Path) -> bytes:
    """Hash one source through a no-follow descriptor and reject read races."""

    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    try:
        path_before = os.lstat(path)
        if not stat.S_ISREG(path_before.st_mode):
            raise OSError
        descriptor = os.open(path, flags)
        opened_before = os.fstat(descriptor)
        if not stat.S_ISREG(opened_before.st_mode) or (
            path_before.st_dev,
            path_before.st_ino,
        ) != (opened_before.st_dev, opened_before.st_ino):
            raise OSError
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        opened_after = os.fstat(descriptor)
        path_after = os.lstat(path)
        before_identity = (
            opened_before.st_dev,
            opened_before.st_ino,
            opened_before.st_size,
            opened_before.st_mtime_ns,
            opened_before.st_ctime_ns,
        )
        if before_identity != (
            opened_after.st_dev,
            opened_after.st_ino,
            opened_after.st_size,
            opened_after.st_mtime_ns,
            opened_after.st_ctime_ns,
        ) or before_identity != (
            path_after.st_dev,
            path_after.st_ino,
            path_after.st_size,
            path_after.st_mtime_ns,
            path_after.st_ctime_ns,
        ):
            raise OSError
        return digest.digest()
    except OSError:
        raise IOPaintUnavailableError(
            "cannot independently read a stable IOPaint source tree"
        ) from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _measure_iopaint_source_tree(package_root: Path) -> _IOPaintSourceMeasurement:
    entries = _enumerate_iopaint_python_sources(package_root)
    manifest = hashlib.sha256()
    try:
        for relative_name, path in entries:
            manifest.update(relative_name.encode("utf-8"))
            manifest.update(b"\0")
            manifest.update(_sha256_regular_source(path))
    except UnicodeEncodeError:
        raise IOPaintUnavailableError(
            "the bound IOPaint source tree contains an unsupported source name"
        ) from None
    return _IOPaintSourceMeasurement(
        package_root=package_root,
        relative_names=tuple(name for name, _path in entries),
        sha256=manifest.hexdigest(),
    )


def _measure_bound_iopaint_sources(
    python_executable: Path,
) -> _IOPaintSourceMeasurement:
    return _measure_iopaint_source_tree(
        _find_bound_iopaint_source_root(python_executable)
    )


def _require_unchanged_iopaint_sources(
    baseline: _IOPaintSourceMeasurement,
    *,
    failure_message: str,
) -> _IOPaintSourceMeasurement:
    try:
        current = _measure_iopaint_source_tree(baseline.package_root)
    except IOPaintUnavailableError:
        raise IOPaintUnavailableError(failure_message) from None
    if (
        current.relative_names != baseline.relative_names
        or current.sha256 != baseline.sha256
    ):
        raise IOPaintUnavailableError(failure_message)
    return current


def _validate_rgb_and_mask(
    rgb_crop: npt.ArrayLike,
    component_mask: npt.ArrayLike,
) -> tuple[UInt8RgbImage, UInt8Mask]:
    rgb = np.asarray(rgb_crop)
    mask = np.asarray(component_mask)
    if rgb.dtype != np.dtype(np.uint8):
        raise TypeError("rgb_crop must have dtype uint8")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb_crop must have shape HxWx3")
    if rgb.shape[0] == 0 or rgb.shape[1] == 0:
        raise ValueError("rgb_crop cannot have an empty dimension")
    if mask.dtype != np.dtype(np.uint8):
        raise TypeError("component_mask must have dtype uint8")
    if mask.shape != rgb.shape[:2]:
        raise ValueError("component_mask must have shape HxW matching rgb_crop")
    values = np.unique(mask)
    if not np.all(np.isin(values, np.array([0, 255], dtype=np.uint8))):
        raise ValueError("component_mask must contain only 0 and 255")
    if not np.any(mask == 255):
        raise ValueError("component_mask must contain at least one synthesized pixel")
    return (
        np.array(rgb, dtype=np.uint8, order="C", copy=True),
        np.array(mask, dtype=np.uint8, order="C", copy=True),
    )


def _validate_batch_inputs(
    rgb_crops: Sequence[npt.ArrayLike],
    component_masks: Sequence[npt.ArrayLike],
) -> tuple[tuple[UInt8RgbImage, UInt8Mask], ...]:
    """Copy and validate every ordered item before any runtime/model access."""

    try:
        rgb_count = len(rgb_crops)
        mask_count = len(component_masks)
    except Exception:
        raise TypeError(
            "rgb_crops and component_masks must be sized ordered sequences"
        ) from None
    if rgb_count == 0:
        raise ValueError("inpaint_batch requires at least one crop and mask")
    if rgb_count != mask_count:
        raise ValueError("rgb_crops and component_masks must have equal length")

    validated: list[tuple[UInt8RgbImage, UInt8Mask]] = []
    for index in range(rgb_count):
        try:
            rgb_value = rgb_crops[index]
            mask_value = component_masks[index]
        except Exception:
            raise TypeError(
                "rgb_crops and component_masks must support ordered integer indexing"
            ) from None
        try:
            validated.append(_validate_rgb_and_mask(rgb_value, mask_value))
        except (TypeError, ValueError) as error:
            raise type(error)(f"batch item {index}: {error}") from None
        except Exception:
            raise TypeError(f"batch item {index}: invalid array input") from None
    return tuple(validated)


def _private_image_names(count: int) -> tuple[str, ...]:
    return tuple(
        f"crop-{index:0{_PRIVATE_IMAGE_INDEX_WIDTH}d}.png" for index in range(count)
    )


def _canonical_config(seed: int) -> dict[str, object]:
    # LaMa is feed-forward and does not consume sd_seed.  It is still fixed in
    # the supported IOPaint request document, alongside Original strategy so
    # IOPaint cannot silently perform a second crop around our prepared crop.
    return {"hd_strategy": "Original", "sd_seed": seed}


def _write_private_png(path: Path, array: np.ndarray) -> None:
    try:
        iio.imwrite(path, array, extension=".png", plugin="pillow", compress_level=9)
        path.chmod(0o600)
    except Exception:
        raise IOPaintExecutionError(
            "cannot encode a private IOPaint PNG input"
        ) from None


class IOPaintLaMaAdapter:
    """Callable RGB crop inpainter backed by the official IOPaint batch CLI.

    Construction is deliberately inert.  Executable probing, weight access,
    and subprocess execution happen only on :meth:`probe` or :meth:`__call__`,
    so callers can construct the callback before composite context validation
    and still leave empty routing completely model-free.
    """

    def __init__(self, config: IOPaintLaMaConfig) -> None:
        self.config = config
        self._runtime: IOPaintRuntimeMetadata | None = None
        self._resolved_iopaint: Path | None = None
        self._resolved_python: Path | None = None
        self._source_measurement: _IOPaintSourceMeasurement | None = None
        self._invocations: list[IOPaintInvocationMetadata] = []

    @property
    def invocations(self) -> tuple[IOPaintInvocationMetadata, ...]:
        """Successful invocations in deterministic call order."""

        return tuple(self._invocations)

    @property
    def last_invocation(self) -> IOPaintInvocationMetadata | None:
        """The most recent successful invocation, if any."""

        if not self._invocations:
            return None
        return self._invocations[-1]

    def probe(self) -> IOPaintRuntimeMetadata:
        """Attest weights and inspect CLI/runtime metadata without inference."""

        weights_path = self._attest_weights()
        return self._probe(weights_path)

    def _expected_weights_path(self) -> Path:
        return (
            Path(self.config.model_dir)
            / "torch"
            / "hub"
            / "checkpoints"
            / IOPAINT_LAMA_FILENAME
        )

    def _attest_weights(self) -> Path:
        configured = Path(self.config.weights_file).expanduser()
        expected = self._expected_weights_path().expanduser()
        try:
            configured_resolved = configured.resolve(strict=True)
        except OSError:
            raise IOPaintArtifactError("LaMa weights file is missing") from None
        expected_resolved = expected.resolve(strict=False)
        if configured_resolved != expected_resolved:
            raise IOPaintArtifactError(
                "weights_file is not the exact IOPaint LaMa cache artifact"
            )
        measured = _sha256_file(configured_resolved)
        if measured != self.config.artifact.sha256:
            raise IOPaintArtifactError(
                "LaMa weights SHA-256 does not match attestation"
            )
        return configured_resolved

    def _run_probe_command(
        self,
        argv: list[str],
        *,
        cuda_visible_devices: str,
        failure_message: str,
    ) -> subprocess.CompletedProcess[str]:
        output_limit_preexec = _probe_output_limit_preexec()
        temp_parent = self.config.temp_parent
        if temp_parent is not None:
            temp_parent = Path(temp_parent)
            try:
                temp_parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            except OSError:
                raise IOPaintUnavailableError(
                    "cannot create the private IOPaint probe parent"
                ) from None
            if temp_parent.is_symlink() or not temp_parent.is_dir():
                raise IOPaintUnavailableError(
                    "private IOPaint probe parent is not a real directory"
                )
        try:
            with tempfile.TemporaryDirectory(
                prefix="fauxce-iopaint-probe-",
                dir=None if temp_parent is None else os.fspath(temp_parent),
            ) as temporary:
                probe_root = Path(temporary)
                probe_root.chmod(0o700)
                private_home = probe_root / "home"
                private_temp = probe_root / "tmp"
                private_home.mkdir(mode=0o700)
                private_temp.mkdir(mode=0o700)
                environment = self._process_environment(
                    private_home=private_home,
                    private_temp=private_temp,
                    cuda_visible_devices=cuda_visible_devices,
                )
                output_path = probe_root / "probe-stdout.bin"
                with output_path.open("xb") as output_handle:
                    output_path.chmod(0o600)
                    completed = subprocess.run(
                        argv,
                        check=False,
                        stdout=output_handle,
                        stderr=subprocess.DEVNULL,
                        timeout=min(float(self.config.timeout_seconds), 60.0),
                        cwd=probe_root,
                        env=environment,
                        preexec_fn=output_limit_preexec,
                        shell=False,
                    )
                    output_handle.flush()
                injected_output = completed.stdout
                if injected_output is None:
                    output_size = output_path.stat().st_size
                    if output_size > _PROBE_OUTPUT_MAX_BYTES or (
                        output_size >= _PROBE_OUTPUT_MAX_BYTES
                        and completed.returncode != 0
                    ):
                        raise IOPaintUnavailableError(
                            f"{failure_message}; probe output exceeded byte limit"
                        )
                    with output_path.open("rb") as output_handle:
                        output_payload = output_handle.read(_PROBE_OUTPUT_MAX_BYTES + 1)
                elif isinstance(injected_output, str):
                    output_payload = injected_output.encode("utf-8")
                elif isinstance(injected_output, bytes):
                    output_payload = injected_output
                else:
                    raise IOPaintUnavailableError(
                        f"{failure_message}; probe output type is invalid"
                    )
                if len(output_payload) > _PROBE_OUTPUT_MAX_BYTES:
                    raise IOPaintUnavailableError(
                        f"{failure_message}; probe output exceeded byte limit"
                    )
                try:
                    output_text = output_payload.decode("utf-8")
                except UnicodeDecodeError:
                    raise IOPaintUnavailableError(
                        f"{failure_message}; probe output is not UTF-8"
                    ) from None
                return subprocess.CompletedProcess(
                    completed.args,
                    completed.returncode,
                    stdout=output_text,
                    stderr=None,
                )
        except (OSError, OverflowError, subprocess.SubprocessError):
            raise IOPaintUnavailableError(failure_message) from None

    def _probe(self, weights_path: Path) -> IOPaintRuntimeMetadata:
        iopaint_executable = _resolve_executable(
            Path(self.config.iopaint_executable), label="IOPaint"
        )
        python_executable = _resolve_executable(
            Path(self.config.python_executable), label="IOPaint Python"
        )
        _require_cli_python_binding(iopaint_executable, python_executable)
        source_measurement = _measure_bound_iopaint_sources(python_executable)
        if (
            source_measurement.sha256
            != self.config.expected_iopaint_source_manifest_sha256
        ):
            raise IOPaintUnavailableError(
                "independently measured IOPaint source manifest does not match "
                "the required trust anchor"
            )
        expected_cuda_visible_devices = _host_cuda_visible_metadata()
        help_result = self._run_probe_command(
            [
                os.fspath(python_executable),
                "-I",
                "-m",
                "iopaint",
                "run",
                "--help",
            ],
            cuda_visible_devices=expected_cuda_visible_devices,
            failure_message="cannot probe the IOPaint run command",
        )
        source_measurement = _require_unchanged_iopaint_sources(
            source_measurement,
            failure_message="IOPaint source tree changed during runtime probing",
        )
        if help_result.returncode != 0:
            raise IOPaintUnavailableError("IOPaint run --help exited unsuccessfully")
        help_text = f"{help_result.stdout}\n{help_result.stderr}"
        missing_options = [item for item in _REQUIRED_RUN_HELP if item not in help_text]
        if missing_options:
            raise IOPaintUnavailableError(
                "IOPaint run command is missing required options: "
                + ", ".join(missing_options)
            )

        version_result = self._run_probe_command(
            [os.fspath(python_executable), "-I", "-c", _RUNTIME_PROBE],
            cuda_visible_devices=expected_cuda_visible_devices,
            failure_message="cannot inspect the IOPaint runtime",
        )
        source_measurement = _require_unchanged_iopaint_sources(
            source_measurement,
            failure_message="IOPaint source tree changed during runtime probing",
        )
        if version_result.returncode != 0:
            raise IOPaintUnavailableError("cannot inspect the IOPaint runtime")
        try:
            document = json.loads(version_result.stdout.strip().splitlines()[-1])
            if not isinstance(document, dict):
                raise TypeError
            iopaint_version = str(document["iopaint_version"])
            torch_version = str(document["torch_version"])
            python_version = str(document["python_version"])
            numpy_version = str(document["numpy_version"])
            pillow_version = str(document["pillow_version"])
            opencv_version = str(document["opencv_version"])
            pydantic_version = str(document["pydantic_version"])
            typer_version = str(document["typer_version"])
            python_implementation = str(document["python_implementation"])
            platform_system = str(document["platform_system"])
            platform_release = str(document["platform_release"])
            platform_machine = str(document["platform_machine"])
            cuda_visible_value = document["cuda_visible_devices"]
            source_manifest = str(document["iopaint_source_manifest_sha256"])
            source_file_count = document["iopaint_source_file_count"]
            deterministic_algorithms = document["deterministic_algorithms"]
            cudnn_benchmark = document["cudnn_benchmark"]
            cuda_available = document["cuda_available"]
            cuda_runtime_value = document["cuda_runtime_version"]
            cudnn_value = document["cudnn_version"]
            hip_runtime_value = document["hip_runtime_version"]
            cuda_names_value = document["cuda_device_names"]
            mps_available = document["mps_available"]
            mps_name_value = document["mps_device_name"]
        except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            raise IOPaintUnavailableError(
                "IOPaint runtime probe returned malformed metadata"
            ) from None
        version_values = (
            iopaint_version,
            torch_version,
            python_version,
            numpy_version,
            pillow_version,
            opencv_version,
            pydantic_version,
            typer_version,
        )
        if not all(_VERSION_RE.fullmatch(value) for value in version_values):
            raise IOPaintUnavailableError(
                "IOPaint runtime probe returned unsafe version metadata"
            )
        runtime_text_values = (
            python_implementation,
            platform_system,
            platform_release,
            platform_machine,
        )
        if not all(
            _SAFE_RUNTIME_TEXT_RE.fullmatch(value) for value in runtime_text_values
        ):
            raise IOPaintUnavailableError(
                "IOPaint runtime probe returned unsafe environment metadata"
            )
        cuda_visible_devices = _validate_cuda_visible_metadata(cuda_visible_value)
        if cuda_visible_devices != expected_cuda_visible_devices:
            raise IOPaintUnavailableError(
                "IOPaint runtime probe returned mismatched CUDA visibility metadata"
            )
        if not _SHA256_RE.fullmatch(source_manifest):
            raise IOPaintUnavailableError(
                "IOPaint runtime probe returned an invalid source manifest"
            )
        if source_manifest != source_measurement.sha256:
            raise IOPaintUnavailableError(
                "IOPaint runtime source report does not match the independent "
                "source measurement"
            )
        if (
            isinstance(source_file_count, bool)
            or not isinstance(source_file_count, int)
            or source_file_count < 1
        ):
            raise IOPaintUnavailableError(
                "IOPaint runtime probe returned an invalid source file count"
            )
        if source_file_count != source_measurement.file_count:
            raise IOPaintUnavailableError(
                "IOPaint runtime source count does not match the independent "
                "source measurement"
            )
        if not isinstance(cuda_available, bool):
            raise IOPaintUnavailableError(
                "IOPaint runtime probe returned an invalid CUDA state"
            )
        if not isinstance(deterministic_algorithms, bool):
            raise IOPaintUnavailableError(
                "IOPaint runtime probe returned an invalid deterministic-algorithms "
                "state"
            )
        if not isinstance(cudnn_benchmark, bool):
            raise IOPaintUnavailableError(
                "IOPaint runtime probe returned an invalid cuDNN benchmark state"
            )
        if cuda_runtime_value is not None and not isinstance(cuda_runtime_value, str):
            raise IOPaintUnavailableError(
                "IOPaint runtime probe returned an invalid CUDA version"
            )
        cuda_runtime_version = cuda_runtime_value
        if cuda_runtime_version is not None and not _VERSION_RE.fullmatch(
            cuda_runtime_version
        ):
            raise IOPaintUnavailableError(
                "IOPaint runtime probe returned an unsafe CUDA version"
            )
        if cudnn_value is not None and (
            isinstance(cudnn_value, bool) or not isinstance(cudnn_value, int)
        ):
            raise IOPaintUnavailableError(
                "IOPaint runtime probe returned an invalid cuDNN version"
            )
        cudnn_version = None if cudnn_value is None else str(cudnn_value)
        if hip_runtime_value is not None and not isinstance(hip_runtime_value, str):
            raise IOPaintUnavailableError(
                "IOPaint runtime probe returned an invalid HIP version"
            )
        hip_runtime_version = hip_runtime_value
        if hip_runtime_version is not None and not _VERSION_RE.fullmatch(
            hip_runtime_version
        ):
            raise IOPaintUnavailableError(
                "IOPaint runtime probe returned an unsafe HIP version"
            )
        if not isinstance(cuda_names_value, list) or not all(
            isinstance(value, str) and _SAFE_RUNTIME_TEXT_RE.fullmatch(value)
            for value in cuda_names_value
        ):
            raise IOPaintUnavailableError(
                "IOPaint runtime probe returned unsafe CUDA device metadata"
            )
        cuda_device_names = tuple(cuda_names_value)
        if not isinstance(mps_available, bool):
            raise IOPaintUnavailableError(
                "IOPaint runtime probe returned an invalid MPS state"
            )
        if mps_name_value is not None and not (
            isinstance(mps_name_value, str)
            and _SAFE_DEVICE_NAME_RE.fullmatch(mps_name_value)
        ):
            raise IOPaintUnavailableError(
                "IOPaint runtime probe returned unsafe MPS device metadata"
            )
        if not mps_available and mps_name_value is not None:
            raise IOPaintUnavailableError(
                "IOPaint runtime probe returned inconsistent MPS device metadata"
            )
        mps_device_name = mps_name_value
        if iopaint_version != self.config.required_iopaint_version:
            raise IOPaintUnavailableError(
                "IOPaint version mismatch: required "
                f"{self.config.required_iopaint_version}, got {iopaint_version}"
            )
        if self.config.device == "cuda" and not cuda_available:
            raise IOPaintUnavailableError(
                "CUDA was requested but is unavailable in the IOPaint runtime"
            )
        if self.config.device == "cuda" and hip_runtime_version is not None:
            # PyTorch intentionally reuses its torch.cuda API and the "cuda"
            # device spelling for ROCm.  This adapter's CUDA provenance and
            # CUBLAS controls are NVIDIA-specific, so never mislabel HIP/ROCm.
            # https://docs.pytorch.org/docs/stable/notes/hip.html
            raise IOPaintUnavailableError(
                "CUDA was requested but the IOPaint runtime reports ROCm/HIP"
            )
        if self.config.device == "cuda" and cuda_runtime_version is None:
            raise IOPaintUnavailableError(
                "CUDA was requested but the IOPaint runtime does not report "
                "an NVIDIA CUDA version"
            )
        if self.config.device == "mps" and not mps_available:
            raise IOPaintUnavailableError(
                "MPS was requested but is unavailable in the IOPaint runtime"
            )
        if self.config.device == "mps":
            # IOPaint 1.6.0 includes ``lama`` in MPS_UNSUPPORT_MODELS and its
            # switch_mps_device() helper silently returns torch.device("cpu").
            # A provenance-first adapter must not record that run as MPS.
            # https://github.com/Sanster/IOPaint/blob/main/iopaint/const.py
            # https://github.com/Sanster/IOPaint/blob/main/iopaint/helper.py
            raise IOPaintUnavailableError(
                f"IOPaint {iopaint_version} routes LaMa MPS requests to CPU; "
                "refusing the silent device fallback"
            )

        environment_document = {
            "cudnn_benchmark": cudnn_benchmark,
            "cuda_available": cuda_available,
            "cuda_device_names": list(cuda_device_names),
            "cuda_runtime_version": cuda_runtime_version,
            "cuda_visible_devices": cuda_visible_devices,
            "cudnn_version": cudnn_version,
            "deterministic_algorithms": deterministic_algorithms,
            "device": self.config.device,
            "iopaint_source_manifest_sha256": source_manifest,
            "iopaint_version": iopaint_version,
            "hip_runtime_version": hip_runtime_version,
            "model_weights_sha256": self.config.artifact.sha256,
            "mps_available": mps_available,
            "mps_device_name": mps_device_name,
            "numpy_version": numpy_version,
            "opencv_version": opencv_version,
            "pillow_version": pillow_version,
            "platform_machine": platform_machine,
            "platform_release": platform_release,
            "platform_system": platform_system,
            "pydantic_version": pydantic_version,
            "python_implementation": python_implementation,
            "python_version": python_version,
            "seed": self.config.seed,
            "thread_count": self.config.thread_count,
            "torch_version": torch_version,
            "typer_version": typer_version,
        }
        environment_fingerprint = hashlib.sha256(
            json.dumps(
                environment_document,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("ascii")
        ).hexdigest()
        probed_runtime = IOPaintRuntimeMetadata(
            tool_name="IOPaint",
            tool_version=iopaint_version,
            tool_license_spdx=IOPAINT_PACKAGE_LICENSE_SPDX,
            iopaint_source_manifest_sha256=source_manifest,
            iopaint_source_file_count=source_file_count,
            python_version=python_version,
            python_implementation=python_implementation,
            torch_version=torch_version,
            numpy_version=numpy_version,
            pillow_version=pillow_version,
            opencv_version=opencv_version,
            pydantic_version=pydantic_version,
            typer_version=typer_version,
            platform_system=platform_system,
            platform_release=platform_release,
            platform_machine=platform_machine,
            deterministic_algorithms=deterministic_algorithms,
            cudnn_benchmark=cudnn_benchmark,
            cuda_available=cuda_available,
            cuda_runtime_version=cuda_runtime_version,
            cudnn_version=cudnn_version,
            hip_runtime_version=hip_runtime_version,
            cuda_device_names=cuda_device_names,
            cuda_visible_devices=cuda_visible_devices,
            mps_available=mps_available,
            mps_device_name=mps_device_name,
            effective_environment_sha256=environment_fingerprint,
            model_name=_MODEL_NAME,
            model_release=IOPAINT_LAMA_RELEASE,
            model_artifact_identifier=self.config.artifact.identifier,
            model_weights_sha256=self.config.artifact.sha256,
            model_upstream_license_spdx=IOPAINT_LAMA_UPSTREAM_LICENSE_SPDX,
            model_artifact_license_status=IOPAINT_LAMA_ARTIFACT_LICENSE_STATUS,
            device=self.config.device,
            thread_count=self.config.thread_count,
            seed=self.config.seed,
            seed_scope=(
                "fixed IOPaint request; LaMa is feed-forward and does not expose "
                "a sampling seed"
            ),
            determinism_scope=(
                "no repeatability claim; fingerprint covers only the recorded "
                "IOPaint, torch, Python, platform, selected device facts, thread "
                "count, weights, deterministic-algorithm and cuDNN benchmark "
                "states, not the complete driver, OS, or hardware state"
            ),
        )
        if self._runtime is not None and self._runtime != probed_runtime:
            raise IOPaintUnavailableError(
                "IOPaint effective environment changed between invocations"
            )
        self._resolved_iopaint = iopaint_executable
        self._resolved_python = python_executable
        self._source_measurement = source_measurement
        self._runtime = probed_runtime
        return probed_runtime

    def _process_environment(
        self,
        *,
        private_home: Path,
        private_temp: Path,
        cuda_visible_devices: str,
        weights_path: Path | None = None,
        model_dir: Path | None = None,
        enable_cuda_cublas: bool = False,
    ) -> dict[str, str]:
        cuda_visible_devices = _validate_cuda_visible_metadata(cuda_visible_devices)
        resolved_home = private_home.resolve(strict=True)
        resolved_temp = private_temp.resolve(strict=True)
        if not resolved_home.is_dir() or not resolved_temp.is_dir():
            raise AssertionError("private HOME and temporary paths must be directories")
        environment = {
            "HOME": os.fspath(resolved_home),
            "LANG": "C",
            "LC_ALL": "C",
            "PATH": os.defpath,
            "TEMP": os.fspath(resolved_temp),
            "TMP": os.fspath(resolved_temp),
            "TMPDIR": os.fspath(resolved_temp),
            "TZ": "UTC",
            "USERPROFILE": os.fspath(resolved_home),
        }
        for key in _WINDOWS_RUNTIME_ENV_KEYS:
            value = os.environ.get(key)
            if value:
                environment[key] = value
        if cuda_visible_devices == "empty":
            environment["CUDA_VISIBLE_DEVICES"] = ""
        elif cuda_visible_devices != "unset":
            environment["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        thread_count = str(self.config.thread_count)
        for key in _THREAD_ENV_KEYS:
            environment[key] = thread_count
        environment.update(
            {
                "HF_HUB_OFFLINE": "1",
                "MKL_DYNAMIC": "FALSE",
                "NO_COLOR": "1",
                "OMP_DYNAMIC": "FALSE",
                "TERM": "dumb",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
        # CUBLAS_WORKSPACE_CONFIG controls CUDA's cuBLAS implementation only.
        # Keep it absent for CPU and MPS so their receipts do not claim a
        # backend-specific determinism control that cannot affect those runs.
        if enable_cuda_cublas:
            if self.config.device != "cuda":
                raise AssertionError("cuBLAS controls require a CUDA request")
            environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        if (weights_path is None) != (model_dir is None):
            raise AssertionError("weights_path and model_dir must be supplied together")
        if weights_path is not None and model_dir is not None:
            resolved_model_dir = model_dir.resolve()
            huggingface_home = resolved_model_dir / "huggingface"
            huggingface_hub = huggingface_home / "hub"
            environment.update(
                {
                    "DIFFUSERS_CACHE": os.fspath(huggingface_hub),
                    "HF_HOME": os.fspath(huggingface_home),
                    "HF_HUB_CACHE": os.fspath(huggingface_hub),
                    "HUGGINGFACE_HUB_CACHE": os.fspath(huggingface_hub),
                    "LAMA_MODEL_URL": os.fspath(weights_path),
                    "TORCH_HOME": os.fspath(resolved_model_dir / "torch"),
                    "TRANSFORMERS_CACHE": os.fspath(huggingface_hub),
                    "XDG_CACHE_HOME": os.fspath(resolved_model_dir),
                }
            )
        return environment

    def _deterministic_environment_metadata(self) -> tuple[tuple[str, str], ...]:
        values = {
            "DIFFUSERS_CACHE": "<private-model-cache>/huggingface/hub",
            "HF_HUB_OFFLINE": "1",
            "HF_HOME": "<private-model-cache>/huggingface",
            "HF_HUB_CACHE": "<private-model-cache>/huggingface/hub",
            "HUGGINGFACE_HUB_CACHE": "<private-model-cache>/huggingface/hub",
            "LAMA_MODEL_URL": "<private-model-artifact>",
            "MKL_DYNAMIC": "FALSE",
            "OMP_DYNAMIC": "FALSE",
            "TORCH_HOME": "<private-model-cache>/torch",
            "TRANSFORMERS_CACHE": "<private-model-cache>/huggingface/hub",
            "TRANSFORMERS_OFFLINE": "1",
            "XDG_CACHE_HOME": "<private-model-cache>",
        }
        if self.config.device == "cuda":
            values["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        for key in _THREAD_ENV_KEYS:
            values[key] = str(self.config.thread_count)
        return tuple(sorted(values.items()))

    def inpaint_batch(
        self,
        rgb_crops: Sequence[npt.ArrayLike],
        component_masks: Sequence[npt.ArrayLike],
    ) -> tuple[UInt8RgbImage, ...]:
        """Inpaint a non-empty ordered crop batch in one IOPaint process.

        Every crop and mask is copied and validated before weight access,
        executable probing, or subprocess execution.  The result and metadata
        commit atomically only after the complete output set is validated.
        ``timeout_seconds`` remains a per-crop budget and scales linearly for
        the one directory process, preserving the scalar call's semantics.
        """

        validated = _validate_batch_inputs(rgb_crops, component_masks)
        batch_timeout = _validated_batch_timeout(
            self.config.timeout_seconds,
            len(validated),
        )
        weights_path = self._attest_weights()
        runtime = self._probe(weights_path)
        assert self._resolved_iopaint is not None
        assert self._resolved_python is not None
        assert self._source_measurement is not None
        source_measurement = self._source_measurement

        temp_parent = self.config.temp_parent
        if temp_parent is not None:
            temp_parent = Path(temp_parent)
            try:
                temp_parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            except OSError:
                raise IOPaintExecutionError(
                    "cannot create the private IOPaint run parent"
                ) from None
        config_document = _canonical_config(self.config.seed)
        private_names = _private_image_names(len(validated))
        sanitized_argv = (
            "python",
            "-I",
            "-m",
            "iopaint",
            "run",
            "--model",
            _MODEL_NAME,
            "--device",
            self.config.device,
            "--image",
            "<private-input-dir>",
            "--mask",
            "<private-mask-dir>",
            "--output",
            "<private-output-dir>",
            "--config",
            "<private-config-json>",
            "--model-dir",
            "<model-cache-dir>",
            "--no-concat",
        )
        outputs: list[UInt8RgbImage] = []

        try:
            try:
                with tempfile.TemporaryDirectory(
                    prefix="fauxce-iopaint-",
                    dir=None if temp_parent is None else os.fspath(temp_parent),
                ) as temporary:
                    run_root = Path(temporary)
                    run_root.chmod(0o700)
                    input_dir = run_root / "input"
                    mask_dir = run_root / "mask"
                    output_dir = run_root / "output"
                    runtime_model_dir = run_root / "model-cache"
                    private_home = run_root / "home"
                    private_temp = run_root / "tmp"
                    for directory in (
                        input_dir,
                        mask_dir,
                        output_dir,
                        runtime_model_dir,
                        private_home,
                        private_temp,
                    ):
                        directory.mkdir(mode=0o700)
                    runtime_weights = (
                        runtime_model_dir
                        / "torch"
                        / "hub"
                        / "checkpoints"
                        / IOPAINT_LAMA_FILENAME
                    )
                    runtime_weights.parent.mkdir(mode=0o700, parents=True)
                    for directory in (
                        runtime_model_dir / "torch",
                        runtime_model_dir / "torch" / "hub",
                        runtime_weights.parent,
                    ):
                        directory.chmod(0o700)
                    try:
                        shutil.copyfile(weights_path, runtime_weights)
                        runtime_weights.chmod(0o600)
                    except OSError:
                        raise IOPaintArtifactError(
                            "cannot create a private LaMa weights snapshot"
                        ) from None
                    if _sha256_file(runtime_weights) != self.config.artifact.sha256:
                        raise IOPaintArtifactError(
                            "private LaMa weights snapshot does not match attestation"
                        )

                    for name, (rgb, mask) in zip(private_names, validated, strict=True):
                        _write_private_png(input_dir / name, rgb)
                        _write_private_png(mask_dir / name, mask)
                    config_path = run_root / _PRIVATE_CONFIG_NAME
                    try:
                        config_path.write_bytes(
                            (
                                json.dumps(
                                    config_document,
                                    sort_keys=True,
                                    separators=(",", ":"),
                                    ensure_ascii=True,
                                )
                                + "\n"
                            ).encode("ascii")
                        )
                        config_path.chmod(0o600)
                    except OSError:
                        raise IOPaintExecutionError(
                            "cannot write the private IOPaint request"
                        ) from None

                    argv = [
                        os.fspath(self._resolved_python),
                        "-I",
                        "-m",
                        "iopaint",
                        "run",
                        "--model",
                        _MODEL_NAME,
                        "--device",
                        self.config.device,
                        "--image",
                        os.fspath(input_dir),
                        "--mask",
                        os.fspath(mask_dir),
                        "--output",
                        os.fspath(output_dir),
                        "--config",
                        os.fspath(config_path),
                        "--model-dir",
                        os.fspath(runtime_model_dir),
                        "--no-concat",
                    ]
                    environment = self._process_environment(
                        private_home=private_home,
                        private_temp=private_temp,
                        cuda_visible_devices=runtime.cuda_visible_devices,
                        weights_path=runtime_weights,
                        model_dir=runtime_model_dir,
                        enable_cuda_cublas=(
                            runtime.device == "cuda"
                            and runtime.cuda_runtime_version is not None
                            and runtime.hip_runtime_version is None
                        ),
                    )
                    source_measurement = _require_unchanged_iopaint_sources(
                        source_measurement,
                        failure_message=(
                            "IOPaint source tree changed before IOPaint execution"
                        ),
                    )
                    try:
                        completed = subprocess.run(
                            argv,
                            check=False,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            timeout=batch_timeout,
                            cwd=run_root,
                            env=environment,
                            shell=False,
                        )
                    except subprocess.TimeoutExpired:
                        raise IOPaintExecutionError(
                            "IOPaint invocation timed out"
                        ) from None
                    except OverflowError:
                        raise IOPaintExecutionError(
                            "IOPaint invocation timeout is unsupported by the platform"
                        ) from None
                    except OSError:
                        raise IOPaintExecutionError(
                            "IOPaint invocation could not start"
                        ) from None
                    finally:
                        _require_unchanged_iopaint_sources(
                            source_measurement,
                            failure_message=(
                                "IOPaint source tree changed during IOPaint execution"
                            ),
                        )
                        if _sha256_file(runtime_weights) != self.config.artifact.sha256:
                            raise IOPaintArtifactError(
                                "private LaMa weights changed during IOPaint execution"
                            ) from None

                    if completed.returncode != 0:
                        raise IOPaintExecutionError(
                            f"IOPaint exited with status {completed.returncode}; "
                            "subprocess output was withheld to protect private paths"
                        )

                    try:
                        output_entries = tuple(output_dir.iterdir())
                    except OSError:
                        raise IOPaintOutputError(
                            "cannot inspect the private IOPaint batch outputs"
                        ) from None
                    expected_names = set(private_names)
                    actual_names = {entry.name for entry in output_entries}
                    if (
                        len(output_entries) != len(private_names)
                        or actual_names != expected_names
                    ):
                        missing_count = len(expected_names - actual_names)
                        unexpected_count = len(actual_names - expected_names)
                        raise IOPaintOutputError(
                            "IOPaint batch output set mismatch: "
                            f"{missing_count} missing, {unexpected_count} unexpected"
                        )

                    for index, (name, (rgb, _mask)) in enumerate(
                        zip(private_names, validated, strict=True)
                    ):
                        output_path = output_dir / name
                        if output_path.is_symlink() or not output_path.is_file():
                            raise IOPaintOutputError(
                                f"IOPaint batch output item {index} is not a regular file"
                            )
                        try:
                            output_value = iio.imread(output_path, plugin="pillow")
                        except Exception:
                            raise IOPaintOutputError(
                                f"IOPaint batch output item {index} is not a readable PNG"
                            ) from None
                        output = np.asarray(output_value)
                        if output.dtype != np.dtype(np.uint8):
                            raise IOPaintOutputError(
                                f"IOPaint batch output item {index} must have dtype uint8"
                            )
                        if output.shape != rgb.shape:
                            raise IOPaintOutputError(
                                f"IOPaint batch output item {index} must have shape "
                                f"{rgb.shape}, got {output.shape}"
                            )
                        outputs.append(
                            np.array(output, dtype=np.uint8, order="C", copy=True)
                        )
            except OSError:
                raise IOPaintExecutionError(
                    "private IOPaint workspace operation failed"
                ) from None
        finally:
            # The provenance file is never exposed to IOPaint.  Re-hash it on
            # every batch exit so concurrent mutation also invalidates failures.
            if _sha256_file(weights_path) != self.config.artifact.sha256:
                raise IOPaintArtifactError(
                    "external LaMa weights changed during IOPaint execution"
                ) from None

        for output in outputs:
            output.setflags(write=False)
        deterministic_environment = self._deterministic_environment_metadata()
        config_metadata = tuple(sorted(config_document.items()))
        pending_metadata = tuple(
            IOPaintInvocationMetadata(
                runtime=runtime,
                sanitized_argv=sanitized_argv,
                deterministic_environment=deterministic_environment,
                config_document=config_metadata,
                input_rgb8_raw_sha256=_sha256_array(rgb),
                mask_u8_raw_sha256=_sha256_array(mask),
                output_rgb8_raw_sha256=_sha256_array(output),
            )
            for (rgb, mask), output in zip(validated, outputs, strict=True)
        )
        self._invocations.extend(pending_metadata)
        return tuple(outputs)

    def __call__(
        self,
        rgb_crop: npt.ArrayLike,
        component_mask: npt.ArrayLike,
    ) -> UInt8RgbImage:
        """Inpaint one RGB uint8 crop through the hardened batch path."""

        return self.inpaint_batch((rgb_crop,), (component_mask,))[0]


__all__ = [
    "IOPAINT_LAMA_ARTIFACT_LICENSE_STATUS",
    "IOPAINT_LAMA_RELEASE",
    "IOPAINT_LAMA_UPSTREAM_LICENSE_SPDX",
    "IOPAINT_PACKAGE_LICENSE_SPDX",
    "IOPAINT_SUPPORTED_VERSION",
    "IOPaintArtifactError",
    "IOPaintError",
    "IOPaintExecutionError",
    "IOPaintInvocationMetadata",
    "IOPaintLaMaAdapter",
    "IOPaintLaMaConfig",
    "IOPaintOutputError",
    "IOPaintRuntimeMetadata",
    "IOPaintUnavailableError",
    "ModelArtifactAttestation",
    "measure_model_artifact",
]
