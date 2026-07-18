"""Deterministic, fail-closed cache for Portable Digital ICE diagnostics.

The cache is deliberately not a Python object serialization format.  Each
array is a separate non-object ``.npy`` file and a canonical JSON manifest
binds those files to the exact inputs, caller assertion, core source, profile,
and backend selection that produced them.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import tempfile
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import numpy.typing as npt
import portable_digital_ice
from portable_digital_ice import (
    DEFAULT_PROFILE,
    ProcessingDiagnostics,
    ProcessingResult,
)

CACHE_SCHEMA = "fauxce-hybrid-diagnostics-cache-v1"
MANIFEST_FILENAME = "diagnostics-cache.json"

_HEX_DIGITS = frozenset("0123456789abcdef")
_BACKEND_REQUESTS = frozenset(("auto", "cpu", "cpu-fast", "cuda"))
_BACKEND_RESULTS = frozenset(("cpu", "cpu-fast", "cuda"))

_ARTIFACT_SPECS = {
    "pure_output_rgb16": {
        "filename": "output.rgb16.npy",
        "dtype": "<u2",
        "source": "portable_digital_ice.ProcessingResult.output_rgb16",
    },
    "score_plane": {
        "filename": "score-plane.npy",
        "dtype": "<f4",
        "source": "portable_digital_ice.ProcessingDiagnostics.score_plane",
    },
    "at_floor_mask": {
        "filename": "at-floor-mask.npy",
        "dtype": "|b1",
        "source": "portable_digital_ice.ProcessingDiagnostics.at_floor_mask",
    },
    "changed_mask": {
        "filename": "changed-mask.npy",
        "dtype": "|b1",
        "source": "portable_digital_ice.ProcessingDiagnostics.changed_mask",
    },
}
_CACHE_FILENAMES = frozenset(
    {MANIFEST_FILENAME} | {str(spec["filename"]) for spec in _ARTIFACT_SPECS.values()}
)
_MAX_NPY_ENCODED_BYTES = 512 * 1024 * 1024
_MAX_NPY_DECODED_BYTES = 512 * 1024 * 1024
_MAX_NPY_HEADER_BYTES = 1024 * 1024
_MAX_CACHE_MANIFEST_BYTES = 1024 * 1024
_NPY_MAGIC = b"\x93NUMPY"

_HASH_CANONICALIZATION = {
    "algorithm": "sha256",
    "array_order": "C",
    "bool": "one byte per element, NumPy |b1",
    "float32": "IEEE-754 little-endian float32 bytes",
    "json": "UTF-8, sorted keys, compact separators, ensure_ascii=true, newline",
    "uint16": "little-endian unsigned 16-bit bytes",
}


class DiagnosticsCacheError(ValueError):
    """Raised when a diagnostics cache cannot be trusted."""


def canonical_backend_reason(requested_backend: str, used_backend: str) -> str:
    """Return the only path-free reason allowed for a backend selection."""

    reasons = {
        ("cpu", "cpu"): "explicit CPU request",
        (
            "cpu-fast",
            "cpu-fast",
        ): "explicit cpu-fast request; self-test passed byte parity",
        ("cuda", "cuda"): "explicit CUDA request; self-test passed",
        (
            "auto",
            "cpu-fast",
        ): "CUDA unavailable; cpu-fast startup self-test passed byte parity",
        ("auto", "cuda"): "startup self-test passed byte parity",
        (
            "auto",
            "cpu",
        ): "CUDA unavailable; complete job ran on exact CPU reference",
    }
    try:
        return reasons[(requested_backend, used_backend)]
    except KeyError:
        raise DiagnosticsCacheError(
            "impossible backend selection: "
            f"requested={requested_backend!r}, used={used_backend!r}"
        ) from None


def _require_sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise DiagnosticsCacheError(
            f"{field} must be a lowercase 64-character SHA-256 hex digest"
        )
    return value


def _require_nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise DiagnosticsCacheError(f"{field} must be a non-empty trimmed string")
    return value


def _require_shape(value: object, field: str) -> tuple[int, ...]:
    if (
        not isinstance(value, (list, tuple))
        or not value
        or any(
            isinstance(dimension, bool)
            or not isinstance(dimension, (int, np.integer))
            or int(dimension) <= 0
            for dimension in value
        )
    ):
        raise DiagnosticsCacheError(f"{field} must contain positive dimensions")
    return tuple(int(dimension) for dimension in value)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise DiagnosticsCacheError(
            f"cannot read cache artifact {path.name}: {error}"
        ) from error
    return digest.hexdigest()


def _canonical_uint16(array: npt.ArrayLike, *, role: str, channels: int) -> np.ndarray:
    value = np.asarray(array)
    if value.dtype.kind != "u" or value.dtype.itemsize != 2:
        raise DiagnosticsCacheError(f"{role} must be uint16")
    if value.ndim != 3 or value.shape[2] != channels:
        raise DiagnosticsCacheError(f"{role} must have shape HxWx{channels}")
    if value.shape[0] == 0 or value.shape[1] == 0:
        raise DiagnosticsCacheError(f"{role} cannot have an empty image dimension")
    return np.array(value, dtype=np.dtype("<u2"), order="C", copy=True)


def _canonical_float32(array: npt.ArrayLike, *, role: str) -> np.ndarray:
    value = np.asarray(array)
    if value.dtype.kind != "f" or value.dtype.itemsize != 4:
        raise DiagnosticsCacheError(f"{role} must be float32")
    if value.ndim != 2 or 0 in value.shape:
        raise DiagnosticsCacheError(f"{role} must be a non-empty HxW array")
    return np.array(value, dtype=np.dtype("<f4"), order="C", copy=True)


def _canonical_bool(array: npt.ArrayLike, *, role: str) -> np.ndarray:
    value = np.asarray(array)
    if value.dtype != np.dtype(np.bool_):
        raise DiagnosticsCacheError(f"{role} must be bool")
    if value.ndim != 2 or 0 in value.shape:
        raise DiagnosticsCacheError(f"{role} must be a non-empty HxW array")
    return np.array(value, dtype=np.dtype("|b1"), order="C", copy=True)


def _raw_sha256(array: np.ndarray) -> str:
    if not array.flags.c_contiguous:
        raise AssertionError("raw cache hashing requires a C-contiguous array")
    return _sha256_bytes(array.tobytes(order="C"))


def hash_rgbir16(array: npt.ArrayLike) -> str:
    """Hash RGBI input in the cache's little-endian uint16 C-order format."""

    return _raw_sha256(_canonical_uint16(array, role="RGBI input", channels=4))


def hash_rgb16(array: npt.ArrayLike) -> str:
    """Hash RGB output in the cache's little-endian uint16 C-order format."""

    return _raw_sha256(_canonical_uint16(array, role="RGB output", channels=3))


def canonical_json_bytes(document: object) -> bytes:
    """Encode one cache JSON document in the only accepted representation."""

    try:
        encoded = json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise DiagnosticsCacheError(
            f"manifest is not canonical-JSON encodable: {error}"
        ) from error
    return encoded.encode("utf-8") + b"\n"


def _current_core_project_root() -> Path:
    package_file = getattr(portable_digital_ice, "__file__", None)
    if package_file is None:
        raise DiagnosticsCacheError("cannot locate portable_digital_ice source")
    package_directory = Path(package_file).resolve().parent
    project_root = package_directory.parent.parent
    expected_package = project_root / "src" / "portable_digital_ice"
    if not (project_root / "pyproject.toml").is_file() or not expected_package.is_dir():
        raise DiagnosticsCacheError(
            "portable_digital_ice must expose its source tree and pyproject.toml "
            "for cache binding"
        )
    try:
        if not expected_package.samefile(package_directory):
            raise DiagnosticsCacheError(
                "portable_digital_ice package is not loaded from its bound source tree"
            )
    except OSError as error:
        raise DiagnosticsCacheError(
            f"cannot resolve core source tree: {error}"
        ) from error
    return project_root


def compute_core_source_manifest(project_root: str | Path | None = None) -> str:
    """Hash the core pyproject and Python sources using stable relative names.

    This deliberately matches the core CUDA binding gate: sorted
    ``src/portable_digital_ice/**/*.py`` records followed by the
    ``pyproject.toml`` record.  Each UTF-8 record has the form
    ``relative/path:sha256(file-bytes)\n``.
    """

    root = (
        _current_core_project_root()
        if project_root is None
        else Path(project_root).resolve()
    )
    pyproject = root / "pyproject.toml"
    package_directory = root / "src" / "portable_digital_ice"
    if not pyproject.is_file() or not package_directory.is_dir():
        raise DiagnosticsCacheError(
            "core source manifest requires pyproject.toml and src/portable_digital_ice"
        )
    files = sorted(
        package_directory.rglob("*.py"),
        key=lambda item: item.relative_to(root).as_posix(),
    )
    if not files:
        raise DiagnosticsCacheError("core source manifest found no Python sources")
    files.append(pyproject)
    records: list[bytes] = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        try:
            file_digest = _sha256_bytes(path.read_bytes())
        except OSError as error:
            raise DiagnosticsCacheError(
                f"cannot hash core source {relative}: {error}"
            ) from error
        records.append(f"{relative}:{file_digest}\n".encode("utf-8"))
    return _sha256_bytes(b"".join(records))


def _current_core_version() -> str:
    try:
        version = metadata.version("portable-digital-ice")
    except metadata.PackageNotFoundError as error:
        raise DiagnosticsCacheError(
            "portable-digital-ice distribution is unavailable"
        ) from error
    return _require_nonempty_string(version, "core_version")


@dataclass(frozen=True)
class DiagnosticsCacheBinding:
    """External facts a cache must match exactly before it may be loaded."""

    prepass_raw_sha256: str
    prepass_shape: tuple[int, int, int]
    main_raw_sha256: str
    main_shape: tuple[int, int, int]
    provenance_assertion_id: str
    provenance_assertion_sha256: str
    profile_id: str
    core_version: str
    core_source_manifest_sha256: str
    requested_backend: str
    used_backend: str
    backend_reason: str

    def __post_init__(self) -> None:
        _require_sha256(self.prepass_raw_sha256, "prepass_raw_sha256")
        _require_sha256(self.main_raw_sha256, "main_raw_sha256")
        _require_sha256(
            self.provenance_assertion_sha256,
            "provenance_assertion_sha256",
        )
        _require_sha256(
            self.core_source_manifest_sha256,
            "core_source_manifest_sha256",
        )
        if (
            _require_shape(self.prepass_shape, "prepass_shape")
            != tuple(self.prepass_shape)
            or len(self.prepass_shape) != 3
            or self.prepass_shape[2] != 4
        ):
            raise DiagnosticsCacheError("prepass_shape must be HxWx4")
        if (
            _require_shape(self.main_shape, "main_shape") != tuple(self.main_shape)
            or len(self.main_shape) != 3
            or self.main_shape[2] != 4
        ):
            raise DiagnosticsCacheError("main_shape must be HxWx4")
        _require_nonempty_string(
            self.provenance_assertion_id,
            "provenance_assertion_id",
        )
        _require_nonempty_string(self.core_version, "core_version")
        _require_nonempty_string(self.backend_reason, "backend_reason")
        if self.profile_id != DEFAULT_PROFILE.profile_id:
            raise DiagnosticsCacheError(
                f"profile_id must be the fixed profile {DEFAULT_PROFILE.profile_id!r}"
            )
        if self.requested_backend not in _BACKEND_REQUESTS:
            raise DiagnosticsCacheError(
                "requested_backend must be auto, cpu, cpu-fast, or cuda"
            )
        if self.used_backend not in _BACKEND_RESULTS:
            raise DiagnosticsCacheError("used_backend must be cpu, cpu-fast, or cuda")
        expected_reason = canonical_backend_reason(
            self.requested_backend,
            self.used_backend,
        )
        if self.backend_reason != expected_reason:
            raise DiagnosticsCacheError(
                "backend_reason does not match the canonical backend selection reason"
            )


def build_cache_binding(
    prepass_rgbir16: npt.ArrayLike,
    main_rgbir16: npt.ArrayLike,
    *,
    provenance_assertion_id: str,
    provenance_assertion_sha256: str,
    requested_backend: str,
    used_backend: str,
    backend_reason: str,
    core_source_manifest_sha256: str | None = None,
) -> DiagnosticsCacheBinding:
    """Build a binding from measured inputs and the currently imported core."""

    prepass = _canonical_uint16(
        prepass_rgbir16,
        role="prepass RGBI input",
        channels=4,
    )
    main = _canonical_uint16(main_rgbir16, role="main RGBI input", channels=4)
    current_source_manifest = compute_core_source_manifest()
    if core_source_manifest_sha256 is not None:
        _require_sha256(
            core_source_manifest_sha256,
            "core_source_manifest_sha256",
        )
        if core_source_manifest_sha256 != current_source_manifest:
            raise DiagnosticsCacheError(
                "core source changed after the run captured its manifest"
            )
    return DiagnosticsCacheBinding(
        prepass_raw_sha256=_raw_sha256(prepass),
        prepass_shape=prepass.shape,
        main_raw_sha256=_raw_sha256(main),
        main_shape=main.shape,
        provenance_assertion_id=provenance_assertion_id,
        provenance_assertion_sha256=provenance_assertion_sha256,
        profile_id=DEFAULT_PROFILE.profile_id,
        core_version=_current_core_version(),
        core_source_manifest_sha256=current_source_manifest,
        requested_backend=str(requested_backend),
        used_backend=str(used_backend),
        backend_reason=backend_reason,
    )


@dataclass(frozen=True)
class CachedDiagnostics:
    """Verified cache contents; all returned arrays are C-order and read-only."""

    output_rgb16: npt.NDArray[np.uint16]
    diagnostics: ProcessingDiagnostics
    binding: DiagnosticsCacheBinding
    output_sha256: str
    manifest_sha256: str


def _binding_document(binding: DiagnosticsCacheBinding) -> dict[str, object]:
    return {
        "backend_reason": binding.backend_reason,
        "core_source_manifest_sha256": binding.core_source_manifest_sha256,
        "core_version": binding.core_version,
        "main_raw_sha256": binding.main_raw_sha256,
        "main_shape": list(binding.main_shape),
        "prepass_raw_sha256": binding.prepass_raw_sha256,
        "prepass_shape": list(binding.prepass_shape),
        "profile_id": binding.profile_id,
        "provenance_assertion_id": binding.provenance_assertion_id,
        "provenance_assertion_sha256": binding.provenance_assertion_sha256,
        "requested_backend": binding.requested_backend,
        "used_backend": binding.used_backend,
    }


def _validate_binding_is_current(binding: DiagnosticsCacheBinding) -> None:
    current_version = _current_core_version()
    if binding.core_version != current_version:
        raise DiagnosticsCacheError(
            "cache binding core_version does not match the imported core: "
            f"expected {current_version!r}, got {binding.core_version!r}"
        )
    current_manifest = compute_core_source_manifest()
    if binding.core_source_manifest_sha256 != current_manifest:
        raise DiagnosticsCacheError(
            "cache binding core_source_manifest_sha256 does not match the imported core"
        )


def _write_exclusive(path: Path, payload: bytes) -> None:
    created = False
    completed = False
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        created = True
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        completed = True
    except OSError as error:
        raise DiagnosticsCacheError(
            f"cannot write cache artifact {path.name}: {error}"
        ) from error
    finally:
        if created and not completed:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


def _open_cache_directory_descriptor(directory: str | Path) -> int:
    """Open every supplied path component as a real directory, without links."""

    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise DiagnosticsCacheError(
            "this platform cannot enforce link-free cache path traversal"
        )
    source = Path(directory)
    if ".." in source.parts:
        raise DiagnosticsCacheError("cache path cannot contain parent traversal")
    absolute = source if source.is_absolute() else Path.cwd() / source
    anchor = absolute.anchor
    if not anchor:
        raise DiagnosticsCacheError("cache path must have an absolute anchor")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(anchor, flags)
        for component in absolute.parts[1:]:
            if component in ("", ".", ".."):
                raise OSError
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            measured = os.fstat(next_descriptor)
            if not stat.S_ISDIR(measured.st_mode):
                os.close(next_descriptor)
                raise OSError
            os.close(descriptor)
            descriptor = next_descriptor
        measured = os.fstat(descriptor)
        if not stat.S_ISDIR(measured.st_mode):
            raise OSError
        result = descriptor
        descriptor = None
        return result
    except OSError:
        raise DiagnosticsCacheError(
            "cache path must contain only available, real directories"
        ) from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _read_bounded_cache_file(
    directory: str | Path,
    filename: str,
    *,
    byte_limit: int,
    role: str,
) -> bytes:
    """Read one regular cache file through a single bounded no-follow handle."""

    directory_descriptor = _open_cache_directory_descriptor(directory)
    file_descriptor: int | None = None
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        file_descriptor = os.open(filename, flags, dir_fd=directory_descriptor)
        measured = os.fstat(file_descriptor)
        if not stat.S_ISREG(measured.st_mode):
            raise DiagnosticsCacheError(f"{role} must be a regular file")
        if measured.st_size > byte_limit:
            raise DiagnosticsCacheError(f"{role} exceeds its byte limit")
        with os.fdopen(file_descriptor, "rb") as handle:
            file_descriptor = None
            payload = bytearray()
            while True:
                block = handle.read(min(64 * 1024, byte_limit + 1 - len(payload)))
                if not block:
                    break
                payload.extend(block)
                if len(payload) > byte_limit:
                    raise DiagnosticsCacheError(f"{role} exceeds its byte limit")
            return bytes(payload)
    except DiagnosticsCacheError:
        raise
    except OSError:
        raise DiagnosticsCacheError(f"cannot securely read {role}") from None
    finally:
        os.close(directory_descriptor)
        if file_descriptor is not None:
            os.close(file_descriptor)


def _validated_cache_directory(directory: str | Path) -> Path:
    """Require one link-free directory containing only declared regular files."""

    source = Path(directory)
    descriptor = _open_cache_directory_descriptor(source)
    try:
        actual = set(os.listdir(descriptor))
        for name in actual:
            measured = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if not stat.S_ISREG(measured.st_mode):
                raise DiagnosticsCacheError(
                    f"cache entry {name!r} must be a regular file, not a link"
                )
    except DiagnosticsCacheError:
        raise
    except OSError:
        raise DiagnosticsCacheError("cannot inspect cache directory") from None
    finally:
        os.close(descriptor)
    if actual != _CACHE_FILENAMES:
        raise DiagnosticsCacheError(
            "cache directory entries mismatch: "
            f"expected {sorted(_CACHE_FILENAMES)}, got {sorted(actual)}"
        )
    return source


def _npy_bytes(array: np.ndarray) -> bytes:
    with tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024, mode="w+b") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.seek(0)
        return handle.read()


def _artifact_document(
    name: str, array: np.ndarray, npy_payload: bytes
) -> dict[str, object]:
    spec = _ARTIFACT_SPECS[name]
    return {
        "dtype": spec["dtype"],
        "file_sha256": _sha256_bytes(npy_payload),
        "filename": spec["filename"],
        "raw_sha256": _raw_sha256(array),
        "shape": list(array.shape),
        "source": spec["source"],
    }


def _score_floor_bits(value: object) -> int:
    floor = np.asarray(value)
    if floor.dtype != np.dtype(np.float32) or floor.ndim != 0:
        raise DiagnosticsCacheError("score floor must be a float32 scalar")
    canonical = np.asarray(floor, dtype=np.dtype("<f4"))
    return int(canonical.view(np.dtype("<u4"))[()])


def save_diagnostics_cache(
    directory: str | Path,
    *,
    processing_result: ProcessingResult,
    binding: DiagnosticsCacheBinding,
) -> CachedDiagnostics:
    """Write and then independently reload one diagnostics cache."""

    if not isinstance(processing_result, ProcessingResult):
        raise TypeError("processing_result must be a ProcessingResult")
    if not isinstance(binding, DiagnosticsCacheBinding):
        raise TypeError("binding must be a DiagnosticsCacheBinding")
    _validate_binding_is_current(binding)
    if processing_result.profile_id != binding.profile_id:
        raise DiagnosticsCacheError(
            "processing result profile does not match cache binding"
        )
    diagnostics = processing_result.diagnostics
    if diagnostics is None:
        raise DiagnosticsCacheError("processing result has no exported diagnostics")

    output = _canonical_uint16(
        processing_result.output_rgb16,
        role="pure Fauxce output",
        channels=3,
    )
    score = _canonical_float32(diagnostics.score_plane, role="score plane")
    at_floor = _canonical_bool(diagnostics.at_floor_mask, role="at-floor mask")
    changed = _canonical_bool(diagnostics.changed_mask, role="changed mask")
    expected_image_shape = output.shape[:2]
    if expected_image_shape != binding.main_shape[:2]:
        raise DiagnosticsCacheError(
            "pure output geometry does not match bound main input geometry"
        )
    if score.shape != expected_image_shape:
        raise DiagnosticsCacheError("score plane shape does not match pure output")
    if at_floor.shape != expected_image_shape or changed.shape != expected_image_shape:
        raise DiagnosticsCacheError("diagnostic mask shape does not match pure output")
    score_floor_bits = _score_floor_bits(diagnostics.score_floor)
    score_floor = np.array(score_floor_bits, dtype="<u4").view("<f4")[()]
    if not np.array_equal(at_floor, score == score_floor):
        raise DiagnosticsCacheError(
            "at-floor mask does not equal score_plane == score_floor"
        )

    output_sha256 = _raw_sha256(output)
    replay_hash = getattr(processing_result.replay, "output_sha256", None)
    if replay_hash != output_sha256:
        raise DiagnosticsCacheError(
            "processing replay output hash does not match pure output bytes"
        )

    destination = Path(directory)
    if os.path.lexists(destination):
        if destination.is_symlink() or not destination.is_dir():
            raise DiagnosticsCacheError("cache destination is not a directory")
        try:
            if any(destination.iterdir()):
                raise DiagnosticsCacheError(
                    f"refusing to overwrite non-empty cache path: {destination}"
                )
        except OSError as error:
            raise DiagnosticsCacheError(
                f"cannot inspect cache directory: {error}"
            ) from error
    else:
        try:
            destination.mkdir(mode=0o700, parents=True, exist_ok=False)
        except OSError as error:
            raise DiagnosticsCacheError(
                f"cannot create cache directory: {error}"
            ) from error
    if os.name != "nt":
        try:
            destination.chmod(0o700)
        except OSError as error:
            raise DiagnosticsCacheError(
                "cannot restrict diagnostics cache directory permissions"
            ) from error

    lock_path = destination / ".diagnostics-cache-write.lock"
    try:
        descriptor = os.open(
            lock_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        os.close(descriptor)
    except OSError as error:
        raise DiagnosticsCacheError(
            "cannot claim diagnostics cache destination exclusively"
        ) from error
    try:
        if {entry.name for entry in destination.iterdir()} != {lock_path.name}:
            raise DiagnosticsCacheError(
                "cache destination changed while it was being claimed"
            )
        arrays = {
            "pure_output_rgb16": output,
            "score_plane": score,
            "at_floor_mask": at_floor,
            "changed_mask": changed,
        }
        payloads = {name: _npy_bytes(array) for name, array in arrays.items()}
        artifacts = {
            name: _artifact_document(name, arrays[name], payloads[name])
            for name in sorted(arrays)
        }
        document = {
            "artifacts": artifacts,
            "binding": _binding_document(binding),
            "diagnostics": {"score_floor_u32_le_bits": score_floor_bits},
            "hash_canonicalization": _HASH_CANONICALIZATION,
            "pure_output": {
                "artifact": "pure_output_rgb16",
                "generative_modification": False,
                "output_sha256": output_sha256,
                "source": "portable_digital_ice.ProcessingResult.output_rgb16",
            },
            "schema": CACHE_SCHEMA,
        }
        manifest_payload = canonical_json_bytes(document)
        for name, payload in payloads.items():
            _write_exclusive(
                destination / str(_ARTIFACT_SPECS[name]["filename"]),
                payload,
            )
        _write_exclusive(
            destination / MANIFEST_FILENAME,
            manifest_payload,
        )
    finally:
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            pass
    return load_diagnostics_cache(
        destination,
        expected_binding=binding,
        expected_manifest_sha256=_sha256_bytes(manifest_payload),
    )


def _require_exact_keys(
    document: object,
    expected: set[str],
    role: str,
) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise DiagnosticsCacheError(f"{role} must be a JSON object")
    actual = set(document)
    if actual != expected:
        raise DiagnosticsCacheError(
            f"{role} keys mismatch: expected {sorted(expected)}, got {sorted(actual)}"
        )
    if any(not isinstance(key, str) for key in document):
        raise DiagnosticsCacheError(f"{role} keys must be strings")
    return document


def _compare_binding(document: object, expected: DiagnosticsCacheBinding) -> None:
    actual = _require_exact_keys(
        document,
        set(_binding_document(expected)),
        "cache binding",
    )
    expected_document = _binding_document(expected)
    for field in sorted(expected_document):
        if actual[field] != expected_document[field]:
            raise DiagnosticsCacheError(
                f"cache binding mismatch for {field}: "
                f"expected {expected_document[field]!r}, got {actual[field]!r}"
            )


def _read_cache_manifest(
    directory: str | Path,
    *,
    expected_manifest_sha256: str | None,
) -> tuple[Path, bytes, dict[str, Any]]:
    """Read one canonical manifest, checking any external trust anchor first."""

    source = _validated_cache_directory(directory)
    manifest_bytes = _read_bounded_cache_file(
        source,
        MANIFEST_FILENAME,
        byte_limit=_MAX_CACHE_MANIFEST_BYTES,
        role="cache manifest",
    )
    if expected_manifest_sha256 is not None:
        expected_digest = _require_sha256(
            expected_manifest_sha256,
            "expected_manifest_sha256",
        )
        actual_digest = _sha256_bytes(manifest_bytes)
        if actual_digest != expected_digest:
            raise DiagnosticsCacheError(
                "cache manifest SHA-256 mismatch: "
                f"expected {expected_digest}, got {actual_digest}"
            )
    try:
        document = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DiagnosticsCacheError(
            f"cache manifest is invalid JSON: {error}"
        ) from error
    if canonical_json_bytes(document) != manifest_bytes:
        raise DiagnosticsCacheError("cache manifest is not in canonical JSON form")
    if not isinstance(document, dict):
        raise DiagnosticsCacheError("cache manifest must be a JSON object")
    return source, manifest_bytes, document


def read_cache_binding(
    directory: str | Path,
    *,
    expected_manifest_sha256: str | None = None,
) -> DiagnosticsCacheBinding:
    """Read untrusted binding metadata without loading any array artifacts.

    This is only a discovery helper.  The returned values remain untrusted
    until the caller reconstructs an expected binding from independent inputs
    and passes that binding plus an independently obtained manifest digest to
    :func:`load_diagnostics_cache`.  When supplied here, the optional digest is
    checked before any binding fields are parsed.
    """

    _, _, document = _read_cache_manifest(
        directory,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    root = _require_exact_keys(
        document,
        {
            "artifacts",
            "binding",
            "diagnostics",
            "hash_canonicalization",
            "pure_output",
            "schema",
        },
        "cache manifest",
    )
    if root["schema"] != CACHE_SCHEMA:
        raise DiagnosticsCacheError(f"unsupported cache schema {root['schema']!r}")
    binding = _require_exact_keys(
        root["binding"],
        {
            "backend_reason",
            "core_source_manifest_sha256",
            "core_version",
            "main_raw_sha256",
            "main_shape",
            "prepass_raw_sha256",
            "prepass_shape",
            "profile_id",
            "provenance_assertion_id",
            "provenance_assertion_sha256",
            "requested_backend",
            "used_backend",
        },
        "cache binding",
    )
    try:
        return DiagnosticsCacheBinding(
            prepass_raw_sha256=binding["prepass_raw_sha256"],
            prepass_shape=_require_shape(binding["prepass_shape"], "prepass_shape"),
            main_raw_sha256=binding["main_raw_sha256"],
            main_shape=_require_shape(binding["main_shape"], "main_shape"),
            provenance_assertion_id=binding["provenance_assertion_id"],
            provenance_assertion_sha256=binding["provenance_assertion_sha256"],
            profile_id=binding["profile_id"],
            core_version=binding["core_version"],
            core_source_manifest_sha256=binding["core_source_manifest_sha256"],
            requested_backend=binding["requested_backend"],
            used_backend=binding["used_backend"],
            backend_reason=binding["backend_reason"],
        )
    except (KeyError, TypeError) as error:
        raise DiagnosticsCacheError(f"cache binding is malformed: {error}") from error


def _open_cache_artifact_descriptor(
    directory: Path,
    filename: str,
    *,
    name: str,
) -> tuple[int, os.stat_result]:
    """Open one bounded regular artifact relative to a link-free directory."""

    directory_descriptor = _open_cache_directory_descriptor(directory)
    artifact_descriptor: int | None = None
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        artifact_descriptor = os.open(
            filename,
            flags,
            dir_fd=directory_descriptor,
        )
        measured = os.fstat(artifact_descriptor)
        if not stat.S_ISREG(measured.st_mode):
            raise DiagnosticsCacheError(
                f"artifact {name} must be a regular file, not a link"
            )
        if measured.st_size > _MAX_NPY_ENCODED_BYTES:
            raise DiagnosticsCacheError(
                f"artifact {name} exceeds the encoded byte limit"
            )
        result = artifact_descriptor
        artifact_descriptor = None
        return result, measured
    except DiagnosticsCacheError:
        raise
    except OSError:
        raise DiagnosticsCacheError(f"cannot securely open artifact {name}") from None
    finally:
        os.close(directory_descriptor)
        if artifact_descriptor is not None:
            os.close(artifact_descriptor)


def _open_file_identity(measured: os.stat_result) -> tuple[int, int, int, int]:
    return (
        measured.st_dev,
        measured.st_ino,
        measured.st_size,
        measured.st_mtime_ns,
    )


def _sha256_open_artifact(handle: BinaryIO, *, name: str) -> str:
    digest = hashlib.sha256()
    total = 0
    try:
        handle.seek(0)
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            total += len(block)
            if total > _MAX_NPY_ENCODED_BYTES:
                raise DiagnosticsCacheError(
                    f"artifact {name} exceeds the encoded byte limit"
                )
            digest.update(block)
    except DiagnosticsCacheError:
        raise
    except OSError:
        raise DiagnosticsCacheError(f"cannot hash artifact {name}") from None
    return digest.hexdigest()


def _preflight_npy_header(
    handle: BinaryIO,
    *,
    name: str,
    encoded_size: int,
    expected_dtype: np.dtype[Any],
    expected_shape: tuple[int, ...],
) -> tuple[int, int]:
    """Validate the complete primitive C-order NPY contract before allocation."""

    try:
        handle.seek(0)
        prefix = handle.read(8)
        if len(prefix) != 8 or prefix[:6] != _NPY_MAGIC:
            raise DiagnosticsCacheError(f"artifact {name} has invalid NPY magic")
        version = (prefix[6], prefix[7])
        if version == (1, 0):
            length_size = 2
            reader = np.lib.format.read_array_header_1_0
        elif version == (2, 0):
            length_size = 4
            reader = np.lib.format.read_array_header_2_0
        else:
            raise DiagnosticsCacheError(
                f"artifact {name} has unsupported NPY format version"
            )
        encoded_header_length = handle.read(length_size)
        if len(encoded_header_length) != length_size:
            raise DiagnosticsCacheError(f"artifact {name} has a truncated NPY header")
        header_length = int.from_bytes(encoded_header_length, "little")
        if header_length > _MAX_NPY_HEADER_BYTES:
            raise DiagnosticsCacheError(
                f"artifact {name} exceeds the NPY header byte limit"
            )
        declared_data_offset = 8 + length_size + header_length
        if declared_data_offset > encoded_size:
            raise DiagnosticsCacheError(f"artifact {name} has a truncated NPY header")

        handle.seek(8)
        shape, fortran_order, dtype = reader(
            handle,
            max_header_size=_MAX_NPY_HEADER_BYTES,
        )
        data_offset = handle.tell()
    except DiagnosticsCacheError:
        raise
    except (EOFError, OSError, TypeError, ValueError):
        raise DiagnosticsCacheError(
            f"artifact {name} has an invalid NPY header"
        ) from None

    if data_offset != declared_data_offset:
        raise DiagnosticsCacheError(f"artifact {name} has an invalid NPY header length")
    if dtype.hasobject or dtype != expected_dtype:
        raise DiagnosticsCacheError(
            f"artifact {name} dtype mismatch: expected {expected_dtype.str}, "
            f"got {dtype.str}"
        )
    if fortran_order:
        raise DiagnosticsCacheError(f"artifact {name} must use C array order")
    if shape != expected_shape:
        raise DiagnosticsCacheError(
            f"artifact {name} shape mismatch: expected {expected_shape}, got {shape}"
        )
    decoded_size = math.prod(shape) * dtype.itemsize
    if decoded_size > _MAX_NPY_DECODED_BYTES:
        raise DiagnosticsCacheError(f"artifact {name} exceeds the decoded byte limit")
    if data_offset + decoded_size != encoded_size:
        raise DiagnosticsCacheError(
            f"artifact {name} NPY data length does not match its header"
        )
    return data_offset, decoded_size


def _decode_open_npy(
    handle: BinaryIO,
    *,
    name: str,
    data_offset: int,
    decoded_size: int,
    dtype: np.dtype[Any],
    shape: tuple[int, ...],
) -> np.ndarray:
    """Read a preflighted primitive payload directly into one bounded array."""

    try:
        value = np.empty(shape, dtype=dtype, order="C")
        target = memoryview(value).cast("B")
        handle.seek(data_offset)
        offset = 0
        while offset < decoded_size:
            read = handle.readinto(target[offset:])
            if read is None or read <= 0:
                raise DiagnosticsCacheError(
                    f"artifact {name} NPY data ended before its declared length"
                )
            offset += read
        if handle.read(1):
            raise DiagnosticsCacheError(
                f"artifact {name} NPY data exceeds its declared length"
            )
        return value
    except DiagnosticsCacheError:
        raise
    except (MemoryError, OSError, TypeError, ValueError):
        raise DiagnosticsCacheError(f"cannot decode artifact {name}") from None


def _load_artifact(
    directory: Path,
    name: str,
    document: object,
    *,
    required_shape: tuple[int, ...],
) -> np.ndarray:
    metadata_document = _require_exact_keys(
        document,
        {"dtype", "file_sha256", "filename", "raw_sha256", "shape", "source"},
        f"artifact {name}",
    )
    spec = _ARTIFACT_SPECS[name]
    for field in ("dtype", "filename", "source"):
        if metadata_document[field] != spec[field]:
            raise DiagnosticsCacheError(
                f"artifact {name} has unexpected {field}: {metadata_document[field]!r}"
            )
    expected_file_hash = _require_sha256(
        metadata_document["file_sha256"],
        f"artifact {name} file_sha256",
    )
    expected_raw_hash = _require_sha256(
        metadata_document["raw_sha256"],
        f"artifact {name} raw_sha256",
    )
    declared_shape = _require_shape(
        metadata_document["shape"], f"artifact {name} shape"
    )
    if declared_shape != required_shape:
        readable_name = name.replace("_", " ")
        raise DiagnosticsCacheError(
            f"artifact {readable_name} shape does not match bound main input geometry"
        )
    expected_shape = required_shape
    expected_dtype = np.dtype(str(spec["dtype"]))
    descriptor, opened_stat = _open_cache_artifact_descriptor(
        directory,
        str(spec["filename"]),
        name=name,
    )
    with os.fdopen(descriptor, "rb") as handle:
        actual_file_hash = _sha256_open_artifact(handle, name=name)
        hashed_stat = os.fstat(handle.fileno())
        if _open_file_identity(hashed_stat) != _open_file_identity(opened_stat):
            raise DiagnosticsCacheError(f"artifact {name} changed while hashing")
        if actual_file_hash != expected_file_hash:
            raise DiagnosticsCacheError(
                f"artifact {name} file SHA-256 mismatch: "
                f"expected {expected_file_hash}, got {actual_file_hash}"
            )
        data_offset, decoded_size = _preflight_npy_header(
            handle,
            name=name,
            encoded_size=hashed_stat.st_size,
            expected_dtype=expected_dtype,
            expected_shape=expected_shape,
        )
        value = _decode_open_npy(
            handle,
            name=name,
            data_offset=data_offset,
            decoded_size=decoded_size,
            dtype=expected_dtype,
            shape=expected_shape,
        )
        post_decode_hash = _sha256_open_artifact(handle, name=name)
        final_stat = os.fstat(handle.fileno())
        if post_decode_hash != actual_file_hash or _open_file_identity(
            final_stat
        ) != _open_file_identity(opened_stat):
            raise DiagnosticsCacheError(f"artifact {name} changed while decoding")
    actual_raw_hash = _raw_sha256(value)
    if actual_raw_hash != expected_raw_hash:
        raise DiagnosticsCacheError(
            f"artifact {name} raw SHA-256 mismatch: "
            f"expected {expected_raw_hash}, got {actual_raw_hash}"
        )
    value.flags.writeable = False
    return value


def load_diagnostics_cache(
    directory: str | Path,
    *,
    expected_binding: DiagnosticsCacheBinding,
    expected_manifest_sha256: str,
    _require_current_core: bool = True,
) -> CachedDiagnostics:
    """Load a cache only after its anchored manifest, binding, and artifacts verify."""

    if not isinstance(expected_binding, DiagnosticsCacheBinding):
        raise TypeError("expected_binding must be a DiagnosticsCacheBinding")
    if not isinstance(_require_current_core, bool):
        raise TypeError("_require_current_core must be a bool")
    if _require_current_core:
        _validate_binding_is_current(expected_binding)
    source, manifest_bytes, document = _read_cache_manifest(
        directory,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    root = _require_exact_keys(
        document,
        {
            "artifacts",
            "binding",
            "diagnostics",
            "hash_canonicalization",
            "pure_output",
            "schema",
        },
        "cache manifest",
    )
    if root["schema"] != CACHE_SCHEMA:
        raise DiagnosticsCacheError(f"unsupported cache schema {root['schema']!r}")
    if root["hash_canonicalization"] != _HASH_CANONICALIZATION:
        raise DiagnosticsCacheError("cache hash canonicalization contract mismatch")
    _compare_binding(root["binding"], expected_binding)

    artifact_documents = _require_exact_keys(
        root["artifacts"],
        set(_ARTIFACT_SPECS),
        "cache artifacts",
    )
    height, width = expected_binding.main_shape[:2]
    required_shapes = {
        "pure_output_rgb16": (height, width, 3),
        "score_plane": (height, width),
        "at_floor_mask": (height, width),
        "changed_mask": (height, width),
    }
    arrays = {
        name: _load_artifact(
            source,
            name,
            artifact_documents[name],
            required_shape=required_shapes[name],
        )
        for name in sorted(_ARTIFACT_SPECS)
    }
    output = arrays["pure_output_rgb16"]
    score = arrays["score_plane"]
    at_floor = arrays["at_floor_mask"]
    changed = arrays["changed_mask"]
    if output.ndim != 3 or output.shape[2] != 3:
        raise DiagnosticsCacheError("pure output artifact must have shape HxWx3")
    if output.shape[:2] != expected_binding.main_shape[:2]:
        raise DiagnosticsCacheError(
            "cached output geometry does not match bound main input geometry"
        )
    if score.ndim != 2 or score.shape != output.shape[:2]:
        raise DiagnosticsCacheError("score plane shape does not match pure output")
    if at_floor.shape != score.shape or changed.shape != score.shape:
        raise DiagnosticsCacheError("diagnostic mask shape does not match score plane")

    diagnostics_document = _require_exact_keys(
        root["diagnostics"],
        {"score_floor_u32_le_bits"},
        "cache diagnostics",
    )
    score_floor_bits = diagnostics_document["score_floor_u32_le_bits"]
    if (
        isinstance(score_floor_bits, bool)
        or not isinstance(score_floor_bits, int)
        or not 0 <= score_floor_bits <= 0xFFFFFFFF
    ):
        raise DiagnosticsCacheError("score_floor_u32_le_bits must be a uint32 integer")
    score_floor = np.array(score_floor_bits, dtype="<u4").view("<f4")[()]
    if not np.array_equal(at_floor, score == score_floor):
        raise DiagnosticsCacheError(
            "cached at-floor mask is inconsistent with score and floor"
        )

    output_sha256 = _raw_sha256(output)
    pure_output = _require_exact_keys(
        root["pure_output"],
        {"artifact", "generative_modification", "output_sha256", "source"},
        "pure output declaration",
    )
    expected_pure_output = {
        "artifact": "pure_output_rgb16",
        "generative_modification": False,
        "output_sha256": output_sha256,
        "source": "portable_digital_ice.ProcessingResult.output_rgb16",
    }
    if pure_output != expected_pure_output:
        raise DiagnosticsCacheError(
            "pure output declaration does not match the verified Portable Digital ICE output"
        )

    diagnostics = ProcessingDiagnostics(
        score_plane=score,
        score_floor=np.float32(score_floor),
        at_floor_mask=at_floor,
        changed_mask=changed,
    )
    return CachedDiagnostics(
        output_rgb16=output,
        diagnostics=diagnostics,
        binding=expected_binding,
        output_sha256=output_sha256,
        manifest_sha256=_sha256_bytes(manifest_bytes),
    )


def verify_diagnostics_cache_snapshot(directory: str | Path) -> CachedDiagnostics:
    """Verify a self-contained historical cache without trusting local source.

    The cache's own binding remains untrusted provenance until a caller
    cross-checks it against an independently authenticated envelope such as a
    hybrid receipt. Artifact hashes, canonical layout, internal accounting,
    and canonical backend semantics are nevertheless fully verified here.
    """

    manifest_sha256 = _sha256_bytes(
        _read_bounded_cache_file(
            directory,
            MANIFEST_FILENAME,
            byte_limit=_MAX_CACHE_MANIFEST_BYTES,
            role="cache manifest",
        )
    )
    binding = read_cache_binding(
        directory,
        expected_manifest_sha256=manifest_sha256,
    )
    return load_diagnostics_cache(
        directory,
        expected_binding=binding,
        expected_manifest_sha256=manifest_sha256,
        _require_current_core=False,
    )


__all__ = [
    "CACHE_SCHEMA",
    "MANIFEST_FILENAME",
    "CachedDiagnostics",
    "DiagnosticsCacheBinding",
    "DiagnosticsCacheError",
    "build_cache_binding",
    "canonical_backend_reason",
    "canonical_json_bytes",
    "compute_core_source_manifest",
    "hash_rgb16",
    "hash_rgbir16",
    "load_diagnostics_cache",
    "read_cache_binding",
    "save_diagnostics_cache",
    "verify_diagnostics_cache_snapshot",
]
