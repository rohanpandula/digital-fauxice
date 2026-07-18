"""Strict, self-verifying receipts for hybrid generative repair runs.

The receipt is deliberately narrower than a general provenance envelope.  It
binds one Portable Digital ICE output, one routing decision, one sequence of
IOPaint crop calls, and one mask-exact composite.  All filesystem references
are relative to the receipt directory; absolute model paths are never
accepted or serialized.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import stat
import struct
import tempfile
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from fractions import Fraction
from importlib import metadata
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, BinaryIO, Callable, Mapping, Sequence
from urllib.parse import urlsplit

import imageio.v3 as iio
import numpy as np
import numpy.typing as npt
from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import SchemaError
from portable_digital_ice import ProcessingDiagnostics

from .cache import (
    CachedDiagnostics,
    DiagnosticsCacheError,
    MANIFEST_FILENAME,
    canonical_backend_reason,
    verify_diagnostics_cache_snapshot,
)
from .composite import CompositeResult, composite_components
from .routing import (
    RoutingPolicy,
    RoutingResult,
    route_at_floor_mask,
    routing_json_document,
)


RECEIPT_SCHEMA = "fauxce-hybrid-receipt-v2"
RECEIPT_FILENAME = "hybrid-receipt.json"
SCHEMA_FILENAME = "fauxce-hybrid-receipt-v2.schema.json"

_HEX_DIGITS = frozenset("0123456789abcdef")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SAFE_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+_-]{0,127}$")
_SAFE_RUNTIME_TEXT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 .,+_/:()@-]{0,255}$")
_SAFE_DEVICE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 .,+_()@-]{0,255}$")
_SAFE_SPDX_EXPRESSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+() -]{0,127}$")
_CUDA_UUID_BODY = (
    r"(?:[0-9A-Fa-f]{1,32}|"
    r"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
    r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12})"
)
_CUDA_VISIBLE_TOKEN_RE = re.compile(
    rf"^(?:0|[1-9][0-9]*|GPU-{_CUDA_UUID_BODY}|MIG-{_CUDA_UUID_BODY}|"
    rf"MIG-GPU-{_CUDA_UUID_BODY}/(?:0|[1-9][0-9]*)/(?:0|[1-9][0-9]*))$"
)
_CUDA_VISIBLE_MAX_LENGTH = 4096
_CUDA_VISIBLE_MAX_TOKENS = 128
_MODEL_PATH_FLAGS = frozenset(
    ("--model", "--model-path", "--model-dir", "--weights", "--weights-path")
)
_REQUIRED_ROLES = frozenset(
    (
        "pure_output_rgb16",
        "hybrid_output_rgb16",
        "synthesis_mask_png",
        "diagnostics_cache_manifest",
        "routing_json",
        "run_metadata_json",
    )
)
_CROP_ROLES = frozenset(
    ("component_input_rgb8", "component_mask_png", "component_inpainted_rgb8")
)
_ALLOWED_ARTIFACT_ROLES = _REQUIRED_ROLES | _CROP_ROLES

# These are absolute verifier limits, not values supplied by a receipt.  They
# comfortably cover a native 5,782 x 3,946 RGB16 frame (~137 MiB on disk), its
# 22.8 million-pixel masks, and the current ~6 MiB routing/receipt documents.
_MAX_ARTIFACT_FILE_BYTES = 256 * 1024 * 1024
_MAX_OPAQUE_ARTIFACT_BYTES = 64 * 1024 * 1024
_MAX_RECEIPT_BYTES = 64 * 1024 * 1024
_MAX_DECODED_ARRAY_BYTES = 256 * 1024 * 1024
_MAX_AGGREGATE_DECODED_BYTES = 512 * 1024 * 1024
_MAX_ARRAY_ELEMENTS = 100_000_000
_MAX_IMAGE_PIXELS = 50_000_000
_MAX_ARRAY_DIMENSIONS = 4
_MAX_NPY_HEADER_BYTES = 64 * 1024
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_CHANNELS = {0: 1, 2: 3, 4: 2, 6: 4}


class ReceiptError(ValueError):
    """Raised when a hybrid receipt or one of its artifacts cannot be trusted."""


class RawEncoding(StrEnum):
    """How an artifact's decoded/raw SHA-256 is reproduced."""

    NPY_ARRAY = "npy_array_c_order"
    PNG_U8 = "png_decoded_uint8_c_order"
    OPAQUE_BYTES = "opaque_file_bytes"


def _require_sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise ReceiptError(
            f"{field} must be a lowercase 64-character SHA-256 hex digest"
        )
    return value


def _require_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ReceiptError(f"{field} must be a non-empty trimmed string")
    return value


def _require_safe_id(value: object, field: str) -> str:
    text = _require_text(value, field)
    if _SAFE_ID.fullmatch(text) is None:
        raise ReceiptError(
            f"{field} must contain only letters, digits, '.', '_', or '-'"
        )
    return text


def _require_version(value: object, field: str) -> str:
    text = _require_text(value, field)
    if _SAFE_VERSION.fullmatch(text) is None:
        raise ReceiptError(f"{field} must be a sanitized version string")
    return text


def _require_optional_version(value: object, field: str) -> str | None:
    if value is None:
        return None
    return _require_version(value, field)


def _require_runtime_text(value: object, field: str) -> str:
    text = _require_text(value, field)
    if _SAFE_RUNTIME_TEXT.fullmatch(text) is None:
        raise ReceiptError(f"{field} contains unsafe runtime metadata")
    return text


def _require_device_name(value: object, field: str) -> str:
    text = _require_text(value, field)
    if _SAFE_DEVICE_NAME.fullmatch(text) is None:
        raise ReceiptError(f"{field} contains unsafe device metadata")
    return text


def _require_cuda_visible_metadata(value: object) -> str:
    """Validate the exact CUDA visibility syntax accepted by the adapter."""

    if not isinstance(value, str):
        raise ReceiptError("cuda_visible_devices must be a string")
    if value in ("unset", "empty", "-1"):
        return value
    if len(value) > _CUDA_VISIBLE_MAX_LENGTH:
        raise ReceiptError("cuda_visible_devices exceeds the supported length")
    tokens = value.split(",")
    if not tokens or len(tokens) > _CUDA_VISIBLE_MAX_TOKENS:
        raise ReceiptError("cuda_visible_devices exceeds the supported token count")
    if any(_CUDA_VISIBLE_TOKEN_RE.fullmatch(token) is None for token in tokens):
        raise ReceiptError("cuda_visible_devices has unsupported token syntax")
    return value


def _canonical_iopaint_argv(device: str) -> tuple[str, ...]:
    """Return the sole receipt-safe argv emitted by the hardened adapter."""

    return (
        "python",
        "-I",
        "-m",
        "iopaint",
        "run",
        "--model",
        "lama",
        "--device",
        device,
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


def _require_spdx_expression(value: object, field: str) -> str:
    text = _require_text(value, field)
    if _SAFE_SPDX_EXPRESSION.fullmatch(text) is None:
        raise ReceiptError(f"{field} must be a sanitized SPDX expression")
    return text


def _require_safe_description(value: object, field: str) -> str:
    text = _require_text(value, field)
    if len(text) > 512 or any(
        ord(character) < 0x20 or ord(character) > 0x7E for character in text
    ):
        raise ReceiptError(
            f"{field} must contain at most 512 printable ASCII characters"
        )
    return text


def _require_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ReceiptError(f"{field} must be a boolean")
    return value


def _is_absolute_path_text(value: str) -> bool:
    return (
        value.startswith("~")
        or value.lower().startswith("file:")
        or PurePosixPath(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
    )


def _reject_absolute_model_argv(argv: Sequence[str]) -> None:
    for index, token in enumerate(argv):
        if token in _MODEL_PATH_FLAGS and index + 1 < len(argv):
            if _is_absolute_path_text(argv[index + 1]):
                raise ReceiptError("argv cannot disclose an absolute model path")
        for flag in _MODEL_PATH_FLAGS:
            prefix = f"{flag}="
            if token.startswith(prefix) and _is_absolute_path_text(
                token[len(prefix) :]
            ):
                raise ReceiptError("argv cannot disclose an absolute model path")


def _require_sanitized_external_reference(value: object) -> str:
    text = _require_text(value, "sanitized_external_reference")
    if _is_absolute_path_text(text) or "\\" in text:
        raise ReceiptError(
            "sanitized_external_reference cannot be an absolute filesystem path"
        )
    parsed = urlsplit(text)
    if parsed.scheme:
        if parsed.scheme not in ("https", "hf", "s3", "gs", "model"):
            raise ReceiptError(
                "sanitized_external_reference has an unsupported URI scheme"
            )
        if parsed.username is not None or parsed.password is not None:
            raise ReceiptError(
                "sanitized_external_reference cannot contain credentials"
            )
        if parsed.query or parsed.fragment:
            raise ReceiptError(
                "sanitized_external_reference cannot contain a query or fragment"
            )
        return text
    return _require_relative_path(text, "sanitized_external_reference")


def _require_relative_path(value: object, field: str) -> str:
    text = _require_text(value, field)
    if "\\" in text:
        raise ReceiptError(f"{field} must use portable '/' separators")
    path = PurePosixPath(text)
    if (
        _is_absolute_path_text(text)
        or any(part in ("", ".", "..") for part in path.parts)
        or text != path.as_posix()
    ):
        raise ReceiptError(f"{field} must be a normalized relative path")
    return path.as_posix()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ReceiptError(f"cannot hash artifact {path}: {error}") from error
    return digest.hexdigest()


def _raw_sha256(array: np.ndarray) -> str:
    return _sha256_bytes(np.ascontiguousarray(array).tobytes(order="C"))


def canonical_receipt_bytes(document: object) -> bytes:
    """Encode a receipt using its only accepted JSON representation."""

    try:
        encoded = json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ReceiptError(
            f"receipt is not canonical-JSON encodable: {error}"
        ) from error
    return encoded.encode("utf-8") + b"\n"


def _project_root() -> Path:
    root = Path(__file__).resolve().parents[2]
    if not (root / "pyproject.toml").is_file() or not (root / "src").is_dir():
        raise ReceiptError("cannot locate the fauxce-hybrid source project")
    return root


def compute_hybrid_source_manifest(project_root: str | Path | None = None) -> str:
    """Hash hybrid code, schema, and pyproject with stable relative names."""

    root = _project_root() if project_root is None else Path(project_root).resolve()
    pyproject = root / "pyproject.toml"
    package = root / "src" / "fauxce_hybrid"
    schema_directory = root / "schemas"
    if not pyproject.is_file() or not package.is_dir() or not schema_directory.is_dir():
        raise ReceiptError(
            "hybrid source manifest requires pyproject.toml, src/fauxce_hybrid, "
            "and schemas"
        )
    files = [pyproject, *package.rglob("*.py"), *schema_directory.glob("*.json")]
    if len(files) == 1:
        raise ReceiptError("hybrid source manifest found no Python sources")
    records: list[bytes] = []
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        try:
            digest = _sha256_bytes(path.read_bytes())
        except OSError as error:
            raise ReceiptError(
                f"cannot hash hybrid source {relative}: {error}"
            ) from error
        records.append(f"{relative}:{digest}\n".encode("utf-8"))
    return _sha256_bytes(b"".join(records))


def current_hybrid_version() -> str:
    """Return the installed hybrid distribution version."""

    try:
        version = metadata.version("fauxce-hybrid")
    except metadata.PackageNotFoundError as error:
        raise ReceiptError(
            "fauxce-hybrid distribution metadata is unavailable"
        ) from error
    return _require_text(version, "hybrid_version")


@dataclass(frozen=True)
class InputProvenance:
    """Caller-asserted binding between the prepass and main RGBI captures."""

    prepass_raw_sha256: str
    prepass_shape: tuple[int, int, int]
    main_raw_sha256: str
    main_shape: tuple[int, int, int]
    same_frame_assertion_id: str
    focus_exposure_assertion_id: str
    source_manifest_sha256: str | None = None

    def __post_init__(self) -> None:
        _require_sha256(self.prepass_raw_sha256, "prepass_raw_sha256")
        _require_sha256(self.main_raw_sha256, "main_raw_sha256")
        for name, shape in (
            ("prepass_shape", self.prepass_shape),
            ("main_shape", self.main_shape),
        ):
            if (
                len(shape) != 3
                or shape[2] != 4
                or any(
                    isinstance(value, bool) or not isinstance(value, int) or value <= 0
                    for value in shape
                )
            ):
                raise ReceiptError(f"{name} must be a positive HxWx4 shape")
        _require_text(self.same_frame_assertion_id, "same_frame_assertion_id")
        _require_text(
            self.focus_exposure_assertion_id,
            "focus_exposure_assertion_id",
        )
        if self.source_manifest_sha256 is not None:
            _require_sha256(
                self.source_manifest_sha256,
                "input source_manifest_sha256",
            )


@dataclass(frozen=True)
class CoreRunMetadata:
    """Exact core implementation and backend facts for the pure pass."""

    version: str
    source_manifest_sha256: str
    profile_id: str
    requested_backend: str
    used_backend: str
    backend_reason: str
    diagnostics_backend: str

    def __post_init__(self) -> None:
        _require_text(self.version, "core version")
        _require_sha256(self.source_manifest_sha256, "core source manifest")
        _require_text(self.profile_id, "profile_id")
        if self.requested_backend not in ("auto", "cpu", "cpu-fast", "cuda"):
            raise ReceiptError("requested_backend must be auto, cpu, cpu-fast, or cuda")
        if self.used_backend not in ("cpu", "cpu-fast", "cuda"):
            raise ReceiptError("used_backend must be cpu, cpu-fast, or cuda")
        if self.diagnostics_backend not in ("cpu", "cpu-fast", "cuda"):
            raise ReceiptError("diagnostics_backend must be cpu, cpu-fast, or cuda")
        _require_text(self.backend_reason, "backend_reason")
        try:
            expected_reason = canonical_backend_reason(
                self.requested_backend,
                self.used_backend,
            )
        except ValueError as error:
            raise ReceiptError(str(error)) from None
        if self.backend_reason != expected_reason:
            raise ReceiptError(
                "backend_reason does not match the canonical backend selection reason"
            )
        try:
            canonical_backend_reason(
                self.requested_backend,
                self.diagnostics_backend,
            )
        except ValueError as error:
            raise ReceiptError(
                "diagnostics_backend is impossible for the requested backend"
            ) from error
        if self.used_backend == "cpu" and self.diagnostics_backend != "cpu":
            raise ReceiptError(
                "CUDA diagnostics are impossible when the pure run used CPU"
            )
        if self.used_backend == "cpu-fast" and self.diagnostics_backend != "cpu-fast":
            raise ReceiptError(
                "diagnostics_backend must be cpu-fast when the pure run used cpu-fast"
            )


@dataclass(frozen=True)
class ModelWeightsAttestation:
    """External model weights identity without bundling or an absolute path."""

    sanitized_artifact_id: str
    sha256: str
    byte_size: int
    sanitized_external_reference: str | None = None

    def __post_init__(self) -> None:
        _require_safe_id(self.sanitized_artifact_id, "sanitized_artifact_id")
        _require_sha256(self.sha256, "model weights sha256")
        if (
            isinstance(self.byte_size, bool)
            or not isinstance(self.byte_size, int)
            or self.byte_size < 1
        ):
            raise ReceiptError("model weights byte_size must be a positive integer")
        if self.sanitized_external_reference is not None:
            _require_sanitized_external_reference(self.sanitized_external_reference)


ModelWeightsPayload = str | Path | bytes | bytearray | memoryview
ModelWeightsResolver = Callable[[ModelWeightsAttestation], ModelWeightsPayload]


@dataclass(frozen=True)
class InpaintingMetadata:
    """Receipt-safe IOPaint model and effective-environment attestation."""

    iopaint_version: str
    entrypoint: str
    tool_license_spdx: str
    iopaint_source_manifest_sha256: str
    iopaint_source_file_count: int
    effective_environment_sha256: str
    model_id: str
    model_version: str
    model_weights: ModelWeightsAttestation
    model_upstream_license_spdx: str
    model_artifact_license_status: str
    torch_version: str
    python_version: str
    python_implementation: str
    numpy_version: str
    pillow_version: str
    opencv_version: str
    pydantic_version: str
    typer_version: str
    platform_system: str
    platform_release: str
    platform_machine: str
    device: str
    threads: int
    seed: int
    seed_scope: str
    argv: tuple[str, ...]
    deterministic_algorithms_enabled: bool
    cudnn_benchmark: bool
    cuda_available: bool
    cuda_runtime_version: str | None
    cudnn_version: str | None
    cuda_device_names: tuple[str, ...]
    cuda_visible_devices: str
    hip_runtime_version: str | None
    mps_available: bool
    mps_device_name: str | None
    repeat_runs: int
    repeatability_observed: bool
    determinism_scope: str

    def __post_init__(self) -> None:
        _require_cuda_visible_metadata(self.cuda_visible_devices)
        if isinstance(self.model_id, str) and _is_absolute_path_text(self.model_id):
            raise ReceiptError("model_id cannot be an absolute filesystem path")
        for name, value in (
            ("iopaint_version", self.iopaint_version),
            ("torch_version", self.torch_version),
            ("python_version", self.python_version),
            ("numpy_version", self.numpy_version),
            ("pillow_version", self.pillow_version),
            ("opencv_version", self.opencv_version),
            ("pydantic_version", self.pydantic_version),
            ("typer_version", self.typer_version),
        ):
            _require_version(value, name)
        for name, value in (
            ("entrypoint", self.entrypoint),
            ("model_id", self.model_id),
            ("model_version", self.model_version),
            ("python_implementation", self.python_implementation),
            ("platform_system", self.platform_system),
            ("platform_release", self.platform_release),
            ("platform_machine", self.platform_machine),
        ):
            _require_runtime_text(value, name)
        _require_spdx_expression(self.tool_license_spdx, "tool_license_spdx")
        _require_spdx_expression(
            self.model_upstream_license_spdx,
            "model_upstream_license_spdx",
        )
        _require_safe_description(
            self.model_artifact_license_status,
            "model_artifact_license_status",
        )
        _require_safe_description(self.seed_scope, "seed_scope")
        _require_safe_description(self.determinism_scope, "determinism_scope")
        _require_sha256(
            self.iopaint_source_manifest_sha256,
            "iopaint_source_manifest_sha256",
        )
        _require_sha256(
            self.effective_environment_sha256,
            "effective_environment_sha256",
        )
        if (
            isinstance(self.iopaint_source_file_count, bool)
            or not isinstance(self.iopaint_source_file_count, int)
            or self.iopaint_source_file_count < 1
        ):
            raise ReceiptError("iopaint_source_file_count must be a positive integer")
        if not isinstance(self.model_weights, ModelWeightsAttestation):
            raise ReceiptError("model_weights must be a ModelWeightsAttestation")
        if self.device not in ("cpu", "cuda", "mps"):
            raise ReceiptError("device must be cpu, cuda, or mps")
        if (
            isinstance(self.threads, bool)
            or not isinstance(self.threads, int)
            or self.threads < 1
        ):
            raise ReceiptError("threads must be a positive integer")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
            or self.seed > 2**32 - 1
        ):
            raise ReceiptError("seed must be an integer in [0, 2**32 - 1]")
        if not self.argv or any(
            not isinstance(value, str)
            or not value
            or value.strip() != value
            or any(ord(character) < 0x20 for character in value)
            for value in self.argv
        ):
            raise ReceiptError("argv must contain safe, non-empty strings")
        _require_bool(
            self.deterministic_algorithms_enabled,
            "deterministic_algorithms_enabled",
        )
        _require_bool(self.cudnn_benchmark, "cudnn_benchmark")
        _require_bool(self.cuda_available, "cuda_available")
        _require_optional_version(self.cuda_runtime_version, "cuda_runtime_version")
        _require_optional_version(self.cudnn_version, "cudnn_version")
        _require_optional_version(self.hip_runtime_version, "hip_runtime_version")
        if not isinstance(self.cuda_device_names, tuple) or any(
            not isinstance(value, str) for value in self.cuda_device_names
        ):
            raise ReceiptError("cuda_device_names must be a tuple of strings")
        for index, value in enumerate(self.cuda_device_names):
            _require_device_name(value, f"cuda_device_names[{index}]")
        if not self.cuda_available and self.cuda_device_names:
            raise ReceiptError(
                "cuda_device_names must be empty when CUDA is unavailable"
            )
        _require_bool(self.mps_available, "mps_available")
        if self.mps_device_name is not None:
            _require_device_name(self.mps_device_name, "mps_device_name")
        if not self.mps_available and self.mps_device_name is not None:
            raise ReceiptError("mps_device_name must be null when MPS is unavailable")
        if self.device == "cuda":
            if not self.cuda_available or self.cuda_runtime_version is None:
                raise ReceiptError(
                    "CUDA device requires available NVIDIA CUDA runtime metadata"
                )
            if self.hip_runtime_version is not None:
                raise ReceiptError("CUDA device cannot be recorded as a HIP runtime")
        if self.device == "mps" and not self.mps_available:
            raise ReceiptError("MPS device requires available MPS runtime metadata")
        if (
            isinstance(self.repeat_runs, bool)
            or not isinstance(self.repeat_runs, int)
            or self.repeat_runs < 1
        ):
            raise ReceiptError("repeat_runs must be a positive integer")
        _require_bool(self.repeatability_observed, "repeatability_observed")
        expected_environment_sha256 = _effective_environment_sha256(self)
        if self.effective_environment_sha256 != expected_environment_sha256:
            raise ReceiptError(
                "effective_environment_sha256 does not match the recorded runtime"
            )
        _reject_absolute_model_argv(self.argv)
        if self.argv != _canonical_iopaint_argv(self.device):
            raise ReceiptError(
                "argv must equal the canonical IOPaint adapter invocation"
            )


def _effective_environment_sha256(metadata_value: InpaintingMetadata) -> str:
    """Reproduce the hardened adapter's effective-environment fingerprint."""

    environment_document = {
        "cudnn_benchmark": metadata_value.cudnn_benchmark,
        "cuda_available": metadata_value.cuda_available,
        "cuda_device_names": [*metadata_value.cuda_device_names],
        "cuda_runtime_version": metadata_value.cuda_runtime_version,
        "cuda_visible_devices": metadata_value.cuda_visible_devices,
        "cudnn_version": metadata_value.cudnn_version,
        "deterministic_algorithms": (metadata_value.deterministic_algorithms_enabled),
        "device": metadata_value.device,
        "iopaint_source_manifest_sha256": (
            metadata_value.iopaint_source_manifest_sha256
        ),
        "iopaint_version": metadata_value.iopaint_version,
        "hip_runtime_version": metadata_value.hip_runtime_version,
        "model_weights_sha256": metadata_value.model_weights.sha256,
        "mps_available": metadata_value.mps_available,
        "mps_device_name": metadata_value.mps_device_name,
        "numpy_version": metadata_value.numpy_version,
        "opencv_version": metadata_value.opencv_version,
        "pillow_version": metadata_value.pillow_version,
        "platform_machine": metadata_value.platform_machine,
        "platform_release": metadata_value.platform_release,
        "platform_system": metadata_value.platform_system,
        "pydantic_version": metadata_value.pydantic_version,
        "python_implementation": metadata_value.python_implementation,
        "python_version": metadata_value.python_version,
        "seed": metadata_value.seed,
        "thread_count": metadata_value.threads,
        "torch_version": metadata_value.torch_version,
        "typer_version": metadata_value.typer_version,
    }
    payload = json.dumps(
        environment_document,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return _sha256_bytes(payload)


@dataclass(frozen=True)
class ArtifactSource:
    """One file that is hashed and referenced relative to the receipt."""

    id: str
    role: str
    relative_path: str
    media_type: str
    raw_encoding: RawEncoding

    def __post_init__(self) -> None:
        _require_safe_id(self.id, "artifact id")
        _require_safe_id(self.role, "artifact role")
        _require_relative_path(self.relative_path, "artifact relative_path")
        _require_text(self.media_type, "artifact media_type")
        if not isinstance(self.raw_encoding, RawEncoding):
            raise ReceiptError("raw_encoding must be a RawEncoding")


@dataclass(frozen=True)
class ComponentArtifactBinding:
    """Artifact identifiers for one IOPaint component invocation."""

    component_id: int
    input_rgb8_artifact_id: str
    mask_artifact_id: str
    inpainted_rgb8_artifact_id: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.component_id, bool)
            or not isinstance(self.component_id, int)
            or self.component_id < 1
        ):
            raise ReceiptError("component_id must be a positive integer")
        _require_safe_id(self.input_rgb8_artifact_id, "input RGB8 artifact id")
        _require_safe_id(self.mask_artifact_id, "component mask artifact id")
        _require_safe_id(
            self.inpainted_rgb8_artifact_id,
            "inpainted RGB8 artifact id",
        )


@dataclass(frozen=True)
class VerifiedReceipt:
    """A successfully verified receipt and its decoded key artifacts."""

    document: Mapping[str, Any]
    receipt_sha256: str
    pure_output_rgb16: npt.NDArray[np.uint16]
    hybrid_output_rgb16: npt.NDArray[np.uint16]
    synthesis_mask: npt.NDArray[np.bool_]
    model_weights_rehashed: bool


@dataclass(frozen=True)
class _ArtifactMeasurement:
    file_sha256: str
    raw_sha256: str
    dtype: str
    shape: tuple[int, ...]
    array: np.ndarray | None
    opaque_bytes: bytes | None


@dataclass
class _DecodedArtifactBudget:
    limit: int
    used: int = 0

    def reserve(self, byte_count: int, *, label: str) -> None:
        if byte_count < 0 or byte_count > self.limit - self.used:
            raise ReceiptError(
                f"artifact {label} exceeds aggregate decoded artifact size limit"
            )
        self.used += byte_count


def _resolve_artifact(root: Path, relative_path: str) -> Path:
    normalized = _require_relative_path(relative_path, "artifact relative_path")
    absolute_root = Path(os.path.abspath(root))
    candidate = absolute_root.joinpath(*PurePosixPath(normalized).parts)
    current = absolute_root
    try:
        for part in PurePosixPath(normalized).parts:
            current = current / part
            metadata_value = current.lstat()
            if stat.S_ISLNK(metadata_value.st_mode):
                raise ReceiptError(
                    f"artifact path contains a symlink component: {normalized}"
                )
        if not stat.S_ISREG(metadata_value.st_mode):
            raise ReceiptError(f"artifact is not a regular file: {normalized}")
    except ReceiptError:
        raise
    except OSError as error:
        raise ReceiptError(
            f"artifact is missing or inaccessible: {normalized}: {error}"
        ) from error
    return candidate


def _file_open_flags(*, directory: bool = False) -> int:
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_CLOEXEC", 0)
    if directory:
        flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    return flags


def _open_artifact_once(root: Path, relative_path: str) -> tuple[Path, BinaryIO]:
    """Open one regular artifact without following receipt-controlled links."""

    normalized = _require_relative_path(relative_path, "artifact relative_path")
    parts = PurePosixPath(normalized).parts
    absolute_root = Path(os.path.abspath(root))
    display_path = absolute_root.joinpath(*parts)
    descriptor: int | None = None
    directory_descriptor: int | None = None
    supports_openat = os.open in os.supports_dir_fd
    supports_nofollow_stat = (
        os.stat in os.supports_dir_fd and os.stat in os.supports_follow_symlinks
    )
    try:
        if supports_openat and supports_nofollow_stat:
            directory_descriptor = os.open(
                absolute_root,
                _file_open_flags(directory=True),
            )
            for part in parts[:-1]:
                metadata_value = os.stat(
                    part,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
                if stat.S_ISLNK(metadata_value.st_mode):
                    raise ReceiptError(
                        f"artifact path contains a symlink component: {normalized}"
                    )
                if not stat.S_ISDIR(metadata_value.st_mode):
                    raise ReceiptError(
                        f"artifact path component is not a directory: {normalized}"
                    )
                next_descriptor = os.open(
                    part,
                    _file_open_flags(directory=True),
                    dir_fd=directory_descriptor,
                )
                os.close(directory_descriptor)
                directory_descriptor = next_descriptor
            metadata_value = os.stat(
                parts[-1],
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            if stat.S_ISLNK(metadata_value.st_mode):
                raise ReceiptError(
                    f"artifact path contains a symlink component: {normalized}"
                )
            if not stat.S_ISREG(metadata_value.st_mode):
                raise ReceiptError(f"artifact is not a regular file: {normalized}")
            descriptor = os.open(
                parts[-1],
                _file_open_flags(),
                dir_fd=directory_descriptor,
            )
        else:  # pragma: no cover - exercised on platforms without openat support
            _resolve_artifact(absolute_root, normalized)
            descriptor = os.open(display_path, _file_open_flags())
        opened_metadata = os.fstat(descriptor)
        if not stat.S_ISREG(opened_metadata.st_mode):
            raise ReceiptError(f"artifact is not a regular file: {normalized}")
        handle = os.fdopen(descriptor, "rb", closefd=True)
        descriptor = None
        return display_path, handle
    except ReceiptError:
        raise
    except OSError as error:
        raise ReceiptError(
            f"cannot securely open artifact {normalized}: {error}"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if directory_descriptor is not None:
            os.close(directory_descriptor)


def _hash_descriptor(handle: BinaryIO) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    handle.seek(0)
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
        byte_count += len(block)
    return digest.hexdigest(), byte_count


def _checked_element_count(shape: tuple[int, ...], *, label: str) -> int:
    if not shape or len(shape) > _MAX_ARRAY_DIMENSIONS:
        raise ReceiptError(
            f"{label} must have between 1 and {_MAX_ARRAY_DIMENSIONS} dimensions"
        )
    count = 1
    for dimension in shape:
        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension < 1
        ):
            raise ReceiptError(f"{label} dimensions must be positive integers")
        count *= dimension
        if count > _MAX_ARRAY_ELEMENTS:
            raise ReceiptError(f"{label} element count exceeds safe limit")
    return count


def _preflight_npy(
    handle: BinaryIO,
    *,
    file_size: int,
    path: Path,
    expected_dtype: str | None,
    expected_shape: Sequence[int] | None,
) -> tuple[np.dtype[Any], tuple[int, ...], int]:
    handle.seek(0)
    try:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(
                handle,
                max_header_size=_MAX_NPY_HEADER_BYTES,
            )
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(
                handle,
                max_header_size=_MAX_NPY_HEADER_BYTES,
            )
        else:
            raise ReceiptError(
                f"NumPy artifact {path} uses unsupported NPY version {version}"
            )
    except ReceiptError:
        raise
    except (EOFError, OSError, TypeError, ValueError) as error:
        raise ReceiptError(
            f"cannot preflight NumPy artifact {path}: {error}"
        ) from error
    shape_tuple = tuple(int(value) for value in shape)
    if fortran_order:
        raise ReceiptError(f"NumPy artifact {path} must use C order")
    dtype = np.dtype(dtype)
    if dtype.hasobject or dtype.itemsize < 1 or dtype.kind not in "buifc":
        raise ReceiptError(f"NumPy artifact {path} has an unsupported dtype")
    element_count = _checked_element_count(shape_tuple, label=f"NumPy artifact {path}")
    decoded_size = element_count * dtype.itemsize
    if decoded_size > _MAX_DECODED_ARRAY_BYTES:
        raise ReceiptError(f"NumPy artifact {path} decoded size exceeds safe limit")
    expected_file_size = handle.tell() + decoded_size
    if file_size != expected_file_size:
        raise ReceiptError(
            f"NumPy artifact {path} data length does not match its header"
        )
    if expected_dtype is not None and dtype.str != expected_dtype:
        raise ReceiptError(f"NumPy artifact {path} header dtype mismatch")
    if expected_shape is not None and [*shape_tuple] != [*expected_shape]:
        raise ReceiptError(f"NumPy artifact {path} header shape mismatch")
    return dtype, shape_tuple, decoded_size


def _preflight_png(
    handle: BinaryIO,
    *,
    path: Path,
    expected_dtype: str | None,
    expected_shape: Sequence[int] | None,
) -> tuple[tuple[int, ...], int]:
    handle.seek(0)
    header = handle.read(33)
    if len(header) != 33 or header[:8] != _PNG_SIGNATURE:
        raise ReceiptError(f"PNG artifact {path} has an invalid signature or IHDR")
    chunk_length = struct.unpack(">I", header[8:12])[0]
    chunk_type = header[12:16]
    if chunk_length != 13 or chunk_type != b"IHDR":
        raise ReceiptError(f"PNG artifact {path} must begin with a 13-byte IHDR")
    expected_crc = struct.unpack(">I", header[29:33])[0]
    actual_crc = zlib.crc32(header[12:29]) & 0xFFFFFFFF
    if actual_crc != expected_crc:
        raise ReceiptError(f"PNG artifact {path} has an invalid IHDR checksum")
    width, height, bit_depth, color_type, compression, filtering, interlace = (
        struct.unpack(">IIBBBBB", header[16:29])
    )
    if width < 1 or height < 1:
        raise ReceiptError(f"PNG artifact {path} dimensions must be positive")
    pixel_count = width * height
    if pixel_count > _MAX_IMAGE_PIXELS:
        raise ReceiptError(f"PNG artifact {path} pixel count exceeds safe limit")
    if bit_depth != 8 or color_type not in _PNG_CHANNELS:
        raise ReceiptError(
            f"PNG artifact {path} must use 8-bit grayscale, RGB, grayscale-alpha, "
            "or RGBA color"
        )
    if compression != 0 or filtering != 0 or interlace not in (0, 1):
        raise ReceiptError(f"PNG artifact {path} uses unsupported IHDR methods")
    channels = _PNG_CHANNELS[color_type]
    decoded_size = pixel_count * channels
    if decoded_size > _MAX_DECODED_ARRAY_BYTES:
        raise ReceiptError(f"PNG artifact {path} decoded size exceeds safe limit")
    shape = (height, width) if channels == 1 else (height, width, channels)
    if expected_dtype is not None and expected_dtype != np.dtype(np.uint8).str:
        raise ReceiptError(f"PNG artifact {path} declared dtype mismatch")
    if expected_shape is not None and [*shape] != [*expected_shape]:
        raise ReceiptError(f"PNG artifact {path} IHDR shape mismatch")
    return shape, decoded_size


def _read_descriptor_snapshot(
    handle: BinaryIO,
    *,
    byte_size: int,
    path: Path,
) -> bytes:
    handle.seek(0)
    payload = handle.read(byte_size + 1)
    if len(payload) != byte_size:
        raise ReceiptError(f"artifact {path} changed while being verified")
    return payload


def _measure_artifact(
    root: Path,
    relative_path: str,
    encoding: RawEncoding,
    *,
    expected_file_sha256: str | None = None,
    expected_dtype: str | None = None,
    expected_shape: Sequence[int] | None = None,
    artifact_id: str | None = None,
    decoded_budget: _DecodedArtifactBudget | None = None,
) -> _ArtifactMeasurement:
    path, handle = _open_artifact_once(root, relative_path)
    label = artifact_id or relative_path
    with handle:
        initial_metadata = os.fstat(handle.fileno())
        byte_size = initial_metadata.st_size
        size_limit = (
            _MAX_OPAQUE_ARTIFACT_BYTES
            if encoding is RawEncoding.OPAQUE_BYTES
            else _MAX_ARTIFACT_FILE_BYTES
        )
        if byte_size < 0 or byte_size > size_limit:
            raise ReceiptError(f"artifact {label} encoded size exceeds safe limit")
        file_hash, hashed_size = _hash_descriptor(handle)
        if hashed_size != byte_size:
            raise ReceiptError(f"artifact {label} changed while being verified")
        if expected_file_sha256 is not None and file_hash != expected_file_sha256:
            raise ReceiptError(f"artifact {label} file SHA-256 mismatch")

        opaque_bytes: bytes | None = None
        if encoding is RawEncoding.OPAQUE_BYTES:
            opaque_bytes = _read_descriptor_snapshot(
                handle,
                byte_size=byte_size,
                path=path,
            )
            raw_hash = _sha256_bytes(opaque_bytes)
            dtype = "bytes"
            shape: tuple[int, ...] = ()
            array: np.ndarray | None = None
        elif encoding is RawEncoding.NPY_ARRAY:
            header_dtype, header_shape, decoded_size = _preflight_npy(
                handle,
                file_size=byte_size,
                path=path,
                expected_dtype=expected_dtype,
                expected_shape=expected_shape,
            )
            if decoded_budget is not None:
                decoded_budget.reserve(decoded_size, label=label)
            try:
                handle.seek(0)
                decoded = np.load(handle, allow_pickle=False)
            except (EOFError, OSError, TypeError, ValueError) as error:
                raise ReceiptError(
                    f"cannot decode NumPy artifact {path}: {error}"
                ) from error
            if (
                not isinstance(decoded, np.ndarray)
                or decoded.dtype.hasobject
                or decoded.dtype != header_dtype
                or decoded.shape != header_shape
                or not decoded.flags.c_contiguous
            ):
                raise ReceiptError(
                    f"NumPy artifact {path} does not match its preflight header"
                )
            array = np.ascontiguousarray(decoded)
            raw_hash = _raw_sha256(array)
            dtype = array.dtype.str
            shape = tuple(int(value) for value in array.shape)
        elif encoding is RawEncoding.PNG_U8:
            header_shape, decoded_size = _preflight_png(
                handle,
                path=path,
                expected_dtype=expected_dtype,
                expected_shape=expected_shape,
            )
            if decoded_budget is not None:
                decoded_budget.reserve(decoded_size, label=label)
            payload = _read_descriptor_snapshot(
                handle,
                byte_size=byte_size,
                path=path,
            )
            try:
                decoded = np.asarray(
                    iio.imread(
                        io.BytesIO(payload),
                        extension=".png",
                        index=0,
                    )
                )
            except (OSError, ValueError, RuntimeError) as error:
                raise ReceiptError(
                    f"cannot decode PNG artifact {path}: {error}"
                ) from error
            if decoded.dtype != np.dtype(np.uint8) or decoded.shape != header_shape:
                raise ReceiptError(
                    f"PNG artifact {path} does not match its preflight IHDR"
                )
            array = np.ascontiguousarray(decoded)
            raw_hash = _raw_sha256(array)
            dtype = array.dtype.str
            shape = tuple(int(value) for value in array.shape)
        else:  # pragma: no cover - exhaustive guard for future enum additions
            raise ReceiptError(f"unsupported raw encoding: {encoding}")

        post_hash, post_size = _hash_descriptor(handle)
        final_metadata = os.fstat(handle.fileno())
        if (
            post_hash != file_hash
            or post_size != byte_size
            or final_metadata.st_size != byte_size
            or final_metadata.st_dev != initial_metadata.st_dev
            or final_metadata.st_ino != initial_metadata.st_ino
        ):
            raise ReceiptError(f"artifact {label} changed while being verified")
        return _ArtifactMeasurement(
            file_hash,
            raw_hash,
            dtype,
            shape,
            array,
            opaque_bytes,
        )


def _artifact_document(
    root: Path,
    source: ArtifactSource,
) -> tuple[dict[str, object], _ArtifactMeasurement]:
    measurement = _measure_artifact(
        root,
        source.relative_path,
        source.raw_encoding,
    )
    return (
        {
            "id": source.id,
            "role": source.role,
            "relative_path": source.relative_path,
            "media_type": source.media_type,
            "raw_encoding": source.raw_encoding.value,
            "file_sha256": measurement.file_sha256,
            "raw_sha256": measurement.raw_sha256,
            "dtype": measurement.dtype,
            "shape": [*measurement.shape],
        },
        measurement,
    )


def _utc_timestamp(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ReceiptError("generated_at_utc must be a timezone-aware datetime")
    utc = value.astimezone(timezone.utc)
    return utc.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _score_floor_bits(diagnostics: ProcessingDiagnostics) -> int:
    value = np.asarray(np.float32(diagnostics.score_floor), dtype="<f4")
    return int(value.view("<u4")[()])


def _composite_components_document(
    composite: CompositeResult,
    bindings: Sequence[ComponentArtifactBinding],
) -> list[dict[str, object]]:
    binding_by_id = {binding.component_id: binding for binding in bindings}
    if len(binding_by_id) != len(bindings):
        raise ReceiptError("component artifact bindings contain duplicate IDs")
    if set(binding_by_id) != {record.component_id for record in composite.components}:
        raise ReceiptError("component artifact bindings must exactly match composites")
    rows: list[dict[str, object]] = []
    for record in composite.components:
        binding = binding_by_id[record.component_id]
        rows.append(
            {
                "component_id": record.component_id,
                "component_bbox_yxyx_half_open": record.component_bbox.as_list(),
                "crop_bbox_yxyx_half_open": record.crop_bbox.as_list(),
                "pixel_count": record.pixel_count,
                "context_pixel_count": record.context_pixel_count,
                "channel_ranges": [
                    {
                        "channel": item.channel,
                        "lo": item.lo,
                        "hi": item.hi,
                        "degenerate": item.degenerate,
                    }
                    for item in record.channel_ranges
                ],
                "alpha_min": record.alpha_min,
                "alpha_max": record.alpha_max,
                "hashes": {
                    "input_rgb8_raw_sha256": record.input_rgb8_sha256,
                    "component_mask_u8_raw_sha256": record.component_mask_sha256,
                    "inpainted_rgb8_raw_sha256": record.inpainted_rgb8_sha256,
                    "decoded_rgb16_raw_sha256": record.decoded_rgb16_sha256,
                    "blended_component_rgb16_raw_sha256": (
                        record.blended_component_rgb16_sha256
                    ),
                },
                "artifacts": {
                    "input_rgb8": binding.input_rgb8_artifact_id,
                    "component_mask_png": binding.mask_artifact_id,
                    "inpainted_rgb8": binding.inpainted_rgb8_artifact_id,
                },
            }
        )
    return rows


def _inpainting_document(metadata_value: InpaintingMetadata) -> dict[str, object]:
    weights = metadata_value.model_weights
    return {
        "invoked": True,
        "tool": {
            "name": "IOPaint",
            "version": metadata_value.iopaint_version,
            "entrypoint": metadata_value.entrypoint,
            "tool_license_spdx": metadata_value.tool_license_spdx,
            "iopaint_source_manifest_sha256": (
                metadata_value.iopaint_source_manifest_sha256
            ),
            "iopaint_source_file_count": metadata_value.iopaint_source_file_count,
        },
        "model": {
            "id": metadata_value.model_id,
            "version": metadata_value.model_version,
            "sanitized_artifact_id": weights.sanitized_artifact_id,
            "weights_sha256": weights.sha256,
            "weights_byte_size": weights.byte_size,
            "sanitized_external_reference": weights.sanitized_external_reference,
            "model_upstream_license_spdx": (metadata_value.model_upstream_license_spdx),
            "model_artifact_license_status": (
                metadata_value.model_artifact_license_status
            ),
        },
        "runtime": {
            "effective_environment_sha256": (
                metadata_value.effective_environment_sha256
            ),
            "python_version": metadata_value.python_version,
            "python_implementation": metadata_value.python_implementation,
            "torch_version": metadata_value.torch_version,
            "numpy_version": metadata_value.numpy_version,
            "pillow_version": metadata_value.pillow_version,
            "opencv_version": metadata_value.opencv_version,
            "pydantic_version": metadata_value.pydantic_version,
            "typer_version": metadata_value.typer_version,
            "platform_system": metadata_value.platform_system,
            "platform_release": metadata_value.platform_release,
            "platform_machine": metadata_value.platform_machine,
            "device": metadata_value.device,
            "threads": metadata_value.threads,
            "seed": metadata_value.seed,
            "seed_scope": metadata_value.seed_scope,
            "argv": [*metadata_value.argv],
            "cuda_available": metadata_value.cuda_available,
            "cuda_runtime_version": metadata_value.cuda_runtime_version,
            "cudnn_version": metadata_value.cudnn_version,
            "cuda_device_names": [*metadata_value.cuda_device_names],
            "cuda_visible_devices": metadata_value.cuda_visible_devices,
            "hip_runtime_version": metadata_value.hip_runtime_version,
            "mps_available": metadata_value.mps_available,
            "mps_device_name": metadata_value.mps_device_name,
        },
        "determinism": {
            "scope": metadata_value.determinism_scope,
            "deterministic_algorithms_enabled": (
                metadata_value.deterministic_algorithms_enabled
            ),
            "cudnn_benchmark": metadata_value.cudnn_benchmark,
            "repeat_runs": metadata_value.repeat_runs,
            "repeatability_observed": metadata_value.repeatability_observed,
        },
    }


def _inpainting_metadata_from_document(
    document: Mapping[str, Any],
) -> InpaintingMetadata:
    tool = document["tool"]
    model = document["model"]
    runtime = document["runtime"]
    determinism = document["determinism"]
    return InpaintingMetadata(
        iopaint_version=tool["version"],
        entrypoint=tool["entrypoint"],
        tool_license_spdx=tool["tool_license_spdx"],
        iopaint_source_manifest_sha256=tool["iopaint_source_manifest_sha256"],
        iopaint_source_file_count=tool["iopaint_source_file_count"],
        effective_environment_sha256=runtime["effective_environment_sha256"],
        model_id=model["id"],
        model_version=model["version"],
        model_weights=ModelWeightsAttestation(
            sanitized_artifact_id=model["sanitized_artifact_id"],
            sha256=model["weights_sha256"],
            byte_size=model["weights_byte_size"],
            sanitized_external_reference=model["sanitized_external_reference"],
        ),
        model_upstream_license_spdx=model["model_upstream_license_spdx"],
        model_artifact_license_status=model["model_artifact_license_status"],
        torch_version=runtime["torch_version"],
        python_version=runtime["python_version"],
        python_implementation=runtime["python_implementation"],
        numpy_version=runtime["numpy_version"],
        pillow_version=runtime["pillow_version"],
        opencv_version=runtime["opencv_version"],
        pydantic_version=runtime["pydantic_version"],
        typer_version=runtime["typer_version"],
        platform_system=runtime["platform_system"],
        platform_release=runtime["platform_release"],
        platform_machine=runtime["platform_machine"],
        device=runtime["device"],
        threads=runtime["threads"],
        seed=runtime["seed"],
        seed_scope=runtime["seed_scope"],
        argv=tuple(runtime["argv"]),
        deterministic_algorithms_enabled=determinism[
            "deterministic_algorithms_enabled"
        ],
        cudnn_benchmark=determinism["cudnn_benchmark"],
        cuda_available=runtime["cuda_available"],
        cuda_runtime_version=runtime["cuda_runtime_version"],
        cudnn_version=runtime["cudnn_version"],
        cuda_device_names=tuple(runtime["cuda_device_names"]),
        cuda_visible_devices=runtime["cuda_visible_devices"],
        hip_runtime_version=runtime["hip_runtime_version"],
        mps_available=runtime["mps_available"],
        mps_device_name=runtime["mps_device_name"],
        repeat_runs=determinism["repeat_runs"],
        repeatability_observed=determinism["repeatability_observed"],
        determinism_scope=determinism["scope"],
    )


def build_receipt_document(
    *,
    artifact_root: str | Path,
    generated_at_utc: datetime,
    hybrid_version: str,
    hybrid_source_manifest_sha256: str,
    inputs: InputProvenance,
    core: CoreRunMetadata,
    diagnostics: ProcessingDiagnostics,
    routing: RoutingResult,
    composite: CompositeResult,
    changed_mask: npt.ArrayLike,
    crop_margin: int,
    inpainting: InpaintingMetadata | None,
    artifacts: Sequence[ArtifactSource],
    component_artifacts: Sequence[ComponentArtifactBinding],
) -> dict[str, object]:
    """Build one strict v2 receipt and hash all referenced artifacts."""

    if not isinstance(inputs, InputProvenance):
        raise TypeError("inputs must be InputProvenance")
    if not isinstance(core, CoreRunMetadata):
        raise TypeError("core must be CoreRunMetadata")
    if not isinstance(diagnostics, ProcessingDiagnostics):
        raise TypeError("diagnostics must be ProcessingDiagnostics")
    if not isinstance(routing, RoutingResult):
        raise TypeError("routing must be RoutingResult")
    if not isinstance(composite, CompositeResult):
        raise TypeError("composite must be CompositeResult")
    synthesis_is_empty = routing.synthesis_pixel_count == 0
    if synthesis_is_empty:
        if inpainting is not None:
            raise ReceiptError("empty synthesis must record IOPaint as not invoked")
    elif not isinstance(inpainting, InpaintingMetadata):
        raise ReceiptError("non-empty synthesis requires InpaintingMetadata")
    elif not routing.within_synthesis_budget:
        raise ReceiptError("IOPaint cannot be invoked above the synthesis budget")
    if (
        isinstance(crop_margin, bool)
        or not isinstance(crop_margin, int)
        or crop_margin < 0
    ):
        raise ReceiptError("crop_margin must be a non-negative integer")
    _require_text(hybrid_version, "hybrid_version")
    _require_sha256(hybrid_source_manifest_sha256, "hybrid source manifest")
    if hybrid_version != current_hybrid_version():
        raise ReceiptError("hybrid_version does not match the running distribution")
    if hybrid_source_manifest_sha256 != compute_hybrid_source_manifest():
        raise ReceiptError(
            "hybrid_source_manifest_sha256 does not match the running source tree"
        )

    score = np.asarray(diagnostics.score_plane)
    at_floor = np.asarray(diagnostics.at_floor_mask)
    changed = np.asarray(changed_mask)
    if score.dtype != np.dtype(np.float32) or score.shape != routing.image_shape:
        raise ReceiptError("diagnostic score plane must be float32 and match routing")
    if at_floor.dtype != np.dtype(np.bool_) or at_floor.shape != routing.image_shape:
        raise ReceiptError("diagnostic at-floor mask must be bool and match routing")
    if changed.dtype != np.dtype(np.bool_) or changed.shape != routing.image_shape:
        raise ReceiptError("changed_mask must be bool and match routing")
    if not np.array_equal(at_floor, routing.provisional_labels > 0):
        raise ReceiptError("routing input must exactly equal diagnostics at-floor mask")
    score_floor = np.array(
        _score_floor_bits(diagnostics),
        dtype="<u4",
    ).view("<f4")[()]
    if not np.array_equal(at_floor, score == score_floor):
        raise ReceiptError("diagnostics at-floor mask must equal score == score_floor")
    if not np.array_equal(changed, diagnostics.changed_mask):
        raise ReceiptError("changed_mask must exactly equal diagnostics.changed_mask")
    if inputs.main_shape[:2] != routing.image_shape:
        raise ReceiptError("main input geometry must match routing geometry")
    if composite.hybrid_rgb16.shape[:2] != routing.image_shape:
        raise ReceiptError("composite geometry must match routing geometry")
    root = Path(artifact_root)
    expected_artifact_count = len(_REQUIRED_ROLES) + 3 * len(composite.components)
    if len(artifacts) != expected_artifact_count:
        raise ReceiptError(
            "artifact count must equal six required artifacts plus three per "
            "composite component"
        )
    unsupported_roles = {
        source.role
        for source in artifacts
        if isinstance(source, ArtifactSource)
        and source.role not in _ALLOWED_ARTIFACT_ROLES
    }
    if unsupported_roles:
        raise ReceiptError(f"unsupported artifact roles: {sorted(unsupported_roles)}")
    artifact_rows: list[dict[str, object]] = []
    measurements: dict[str, _ArtifactMeasurement] = {}
    seen_paths: set[str] = set()
    for source in artifacts:
        if not isinstance(source, ArtifactSource):
            raise TypeError("artifacts must contain ArtifactSource values")
        if source.id in measurements:
            raise ReceiptError(f"duplicate artifact id: {source.id}")
        if source.relative_path in seen_paths:
            raise ReceiptError(f"duplicate artifact path: {source.relative_path}")
        row, measurement = _artifact_document(root, source)
        artifact_rows.append(row)
        measurements[source.id] = measurement
        seen_paths.add(source.relative_path)
    artifact_rows.sort(key=lambda row: str(row["id"]))

    role_to_ids: dict[str, list[str]] = {}
    for row in artifact_rows:
        role_to_ids.setdefault(str(row["role"]), []).append(str(row["id"]))
    missing_roles = _REQUIRED_ROLES - role_to_ids.keys()
    if missing_roles:
        raise ReceiptError(f"missing required artifact roles: {sorted(missing_roles)}")
    for role in _REQUIRED_ROLES:
        if len(role_to_ids[role]) != 1:
            raise ReceiptError(f"artifact role {role} must occur exactly once")

    pure_id = role_to_ids["pure_output_rgb16"][0]
    hybrid_id = role_to_ids["hybrid_output_rgb16"][0]
    mask_id = role_to_ids["synthesis_mask_png"][0]
    cache_id = role_to_ids["diagnostics_cache_manifest"][0]

    pure_measurement = measurements[pure_id]
    hybrid_measurement = measurements[hybrid_id]
    mask_measurement = measurements[mask_id]
    if pure_measurement.array is None or hybrid_measurement.array is None:
        raise ReceiptError("pure and hybrid outputs must be decoded array artifacts")
    if mask_measurement.array is None:
        raise ReceiptError("synthesis mask must be a decoded PNG artifact")
    _validate_output_array(pure_measurement.array, "pure output")
    _validate_output_array(hybrid_measurement.array, "hybrid output")
    mask_bool = _decode_mask_array(mask_measurement.array)
    if not np.array_equal(mask_bool, routing.synthesis_mask):
        raise ReceiptError("synthesis mask PNG does not equal routing synthesis mask")
    if pure_measurement.raw_sha256 != composite.pure_rgb16_sha256:
        raise ReceiptError("pure output artifact does not match composite source")
    if hybrid_measurement.raw_sha256 != composite.hybrid_rgb16_sha256:
        raise ReceiptError("hybrid output artifact does not match composite output")
    if _raw_sha256(mask_bool) != composite.synthesis_mask_sha256:
        raise ReceiptError("synthesis mask does not match composite mask hash")
    if not np.array_equal(
        hybrid_measurement.array[~mask_bool],
        pure_measurement.array[~mask_bool],
    ):
        raise ReceiptError("hybrid output differs from pure output outside the mask")

    routing_document = routing_json_document(routing)
    try:
        routing_document["scipy_version"] = metadata.version("scipy")
    except metadata.PackageNotFoundError as error:
        raise ReceiptError("scipy distribution metadata is unavailable") from error
    routing_document["id_rules"] = {
        "provisional": "scipy_ndimage_label_scan_order_8_connected",
        "final": "bbox_y0_x0_y1_x1_then_raw_id",
        "paste": "ascending_final_component_id",
    }
    excluded_region = routing_document["perimeter_excluded_region"]
    if excluded_region is not None:
        excluded_region["output_treatment"] = "left_pure"
    routing_document["counts"]["unchanged_synthesis_pixels"] = int(
        np.count_nonzero(routing.synthesis_mask & ~changed)
    )

    composite_rows = _composite_components_document(composite, component_artifacts)
    if inpainting is None:
        inpainting_document: dict[str, object] = {
            "invoked": False,
            "reason": "empty_synthesis_mask",
        }
    else:
        inpainting_document = _inpainting_document(inpainting)
    document: dict[str, object] = {
        "schema": RECEIPT_SCHEMA,
        "disclosure": {
            "claim": (
                "Portable Digital ICE output is byte-exact outside the synthesis "
                "mask only; pixels inside the mask include generative content."
            ),
            "exact_scope": "outside_synthesis_mask_only",
            "inside_mask": "generative_repair",
            "verification": "uint16_rgb_bytes_compared_outside_decoded_mask",
        },
        "generation": {
            "generated_at_utc": _utc_timestamp(generated_at_utc),
            "hybrid_version": hybrid_version,
            "hybrid_source_manifest_sha256": hybrid_source_manifest_sha256,
            "source_manifest_algorithm": (
                "sha256(sorted(relative_path:sha256(file_bytes)\\n)); "
                "pyproject.toml + src/fauxce_hybrid/**/*.py + schemas/*.json"
            ),
        },
        "inputs": {
            "prepass": {
                "raw_sha256": inputs.prepass_raw_sha256,
                "shape": [*inputs.prepass_shape],
                "canonical_encoding": "uint16_little_endian_c_order",
            },
            "main": {
                "raw_sha256": inputs.main_raw_sha256,
                "shape": [*inputs.main_shape],
                "canonical_encoding": "uint16_little_endian_c_order",
            },
            "provenance": {
                "basis": "caller_asserted",
                "same_frame_assertion_id": inputs.same_frame_assertion_id,
                "focus_exposure_assertion_id": inputs.focus_exposure_assertion_id,
                "source_manifest_sha256": inputs.source_manifest_sha256,
            },
            "geometry": {
                "output_shape": [*composite.hybrid_rgb16.shape],
                "mask_shape": [*routing.image_shape],
            },
        },
        "core": {
            "version": core.version,
            "source_manifest_sha256": core.source_manifest_sha256,
            "profile_id": core.profile_id,
            "backend": {
                "requested": core.requested_backend,
                "used": core.used_backend,
                "reason": core.backend_reason,
                "diagnostics": core.diagnostics_backend,
            },
            "diagnostics": {
                "score_floor_u32_le_bits": _score_floor_bits(diagnostics),
                "at_floor_pixel_count": int(np.count_nonzero(at_floor)),
                "changed_pixel_count": int(np.count_nonzero(changed)),
            },
        },
        "routing": routing_document,
        "composite": {
            "source": "immutable_pure_fauxce_rgb16",
            "crop_margin": crop_margin,
            "component_order": "ascending_final_component_id",
            "channel_range": "per_crop_outside_global_mask_min_max",
            "encode": "clip(round_half_up((x-lo)*255/(hi-lo)),0,255)",
            "decode": "clip(round_half_up(lo+x*(hi-lo)/255),0,65535)",
            "degenerate_range": "encode_zero_decode_lo",
            "alpha": "min(chessboard_distance_to_component_outside,3)/3",
            "alpha_border_semantics": "image_border_is_interior",
            "blend": "round_half_up(pure*(1-alpha)+generated*alpha)",
            "paste_scope": "component_pixels_only",
            "pure_rgb16_raw_sha256": composite.pure_rgb16_sha256,
            "synthesis_mask_bool_raw_sha256": composite.synthesis_mask_sha256,
            "hybrid_rgb16_raw_sha256": composite.hybrid_rgb16_sha256,
            "components": composite_rows,
        },
        "inpainting": inpainting_document,
        "artifacts": artifact_rows,
        "synthesis": {
            "pure_output_artifact_id": pure_id,
            "hybrid_output_artifact_id": hybrid_id,
            "mask_artifact_id": mask_id,
            "diagnostics_cache_artifact_id": cache_id,
            "pixel_count": routing.synthesis_pixel_count,
            "frame_pixel_count": routing.image_shape[0] * routing.image_shape[1],
            "fraction": routing.synthesis_fraction,
            "within_budget": routing.within_synthesis_budget,
            "maximum_fraction": routing.policy.max_synth_fraction,
        },
    }
    validate_receipt_document(document)
    _validate_document_semantics(document)
    return document


def _schema_path() -> Path:
    return _project_root() / "schemas" / SCHEMA_FILENAME


def load_receipt_schema() -> dict[str, Any]:
    """Load and meta-validate the checked-in Draft 2020-12 JSON Schema."""

    path = _schema_path()
    try:
        document = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReceiptError(f"cannot load receipt JSON Schema: {error}") from error
    try:
        Draft202012Validator.check_schema(document)
    except SchemaError as error:
        raise ReceiptError(
            f"invalid checked-in receipt JSON Schema: {error.message}"
        ) from error
    return document


def validate_receipt_document(document: object) -> None:
    """Validate only the JSON shape; filesystem checks happen in verification."""

    validator = Draft202012Validator(
        load_receipt_schema(),
        format_checker=FormatChecker(),
    )
    errors = sorted(validator.iter_errors(document), key=lambda item: list(item.path))
    if errors:
        error = errors[0]
        location = ".".join(str(part) for part in error.absolute_path) or "<root>"
        raise ReceiptError(
            f"receipt schema validation failed at {location}: {error.message}"
        )


def _artifacts_by_id(document: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    artifacts = document["artifacts"]
    by_id: dict[str, Mapping[str, Any]] = {}
    paths: set[str] = set()
    for artifact in artifacts:
        artifact_id = artifact["id"]
        if artifact_id in by_id:
            raise ReceiptError(f"duplicate artifact id: {artifact_id}")
        if artifact["relative_path"] in paths:
            raise ReceiptError(f"duplicate artifact path: {artifact['relative_path']}")
        _require_relative_path(artifact["relative_path"], "artifact relative_path")
        by_id[artifact_id] = artifact
        paths.add(artifact["relative_path"])
    return by_id


def _validate_document_semantics(document: Mapping[str, Any]) -> None:
    artifacts = _artifacts_by_id(document)
    roles: dict[str, list[str]] = {}
    for artifact_id, artifact in artifacts.items():
        roles.setdefault(artifact["role"], []).append(artifact_id)
    for role in _REQUIRED_ROLES:
        if len(roles.get(role, ())) != 1:
            raise ReceiptError(f"artifact role {role} must occur exactly once")
    if "model_weights" in roles:
        raise ReceiptError("model weights must not be copied into receipt artifacts")
    unsupported_roles = roles.keys() - _ALLOWED_ARTIFACT_ROLES
    if unsupported_roles:
        raise ReceiptError(f"unsupported artifact roles: {sorted(unsupported_roles)}")
    component_count = len(document["composite"]["components"])
    expected_artifact_count = len(_REQUIRED_ROLES) + 3 * component_count
    if len(artifacts) != expected_artifact_count:
        raise ReceiptError(
            "artifact count must equal six required artifacts plus three per "
            "composite component"
        )

    core = document["core"]
    backend = core["backend"]
    CoreRunMetadata(
        version=core["version"],
        source_manifest_sha256=core["source_manifest_sha256"],
        profile_id=core["profile_id"],
        requested_backend=backend["requested"],
        used_backend=backend["used"],
        backend_reason=backend["reason"],
        diagnostics_backend=backend["diagnostics"],
    )

    synthesis = document["synthesis"]
    role_bindings = {
        "pure_output_artifact_id": "pure_output_rgb16",
        "hybrid_output_artifact_id": "hybrid_output_rgb16",
        "mask_artifact_id": "synthesis_mask_png",
        "diagnostics_cache_artifact_id": "diagnostics_cache_manifest",
    }
    for field, role in role_bindings.items():
        artifact_id = synthesis[field]
        if artifact_id not in artifacts or artifacts[artifact_id]["role"] != role:
            raise ReceiptError(f"{field} does not reference the {role} artifact")

    inpainting = document["inpainting"]
    if synthesis["pixel_count"] == 0:
        if inpainting["invoked"]:
            raise ReceiptError("empty synthesis cannot invoke IOPaint")
    else:
        if not inpainting["invoked"]:
            raise ReceiptError("non-empty synthesis must invoke IOPaint")
        if not synthesis["within_budget"]:
            raise ReceiptError("IOPaint cannot be invoked above the synthesis budget")
        metadata_value = _inpainting_metadata_from_document(inpainting)
        if _inpainting_document(metadata_value) != inpainting:
            raise ReceiptError(
                "invoked inpainting metadata does not round-trip canonically"
            )

    expected_encodings = {
        "pure_output_rgb16": RawEncoding.NPY_ARRAY.value,
        "hybrid_output_rgb16": RawEncoding.NPY_ARRAY.value,
        "synthesis_mask_png": RawEncoding.PNG_U8.value,
        "diagnostics_cache_manifest": RawEncoding.OPAQUE_BYTES.value,
        "routing_json": RawEncoding.OPAQUE_BYTES.value,
        "run_metadata_json": RawEncoding.OPAQUE_BYTES.value,
        "component_input_rgb8": RawEncoding.PNG_U8.value,
        "component_mask_png": RawEncoding.PNG_U8.value,
        "component_inpainted_rgb8": RawEncoding.PNG_U8.value,
    }
    for artifact in artifacts.values():
        expected = expected_encodings.get(artifact["role"])
        if expected is not None and artifact["raw_encoding"] != expected:
            raise ReceiptError(
                f"artifact role {artifact['role']} must use raw encoding {expected}"
            )

    routing = document["routing"]
    counts = routing["counts"]
    if synthesis["pixel_count"] != counts["synthesis_pixels"]:
        raise ReceiptError("synthesis pixel counts disagree")
    if synthesis["frame_pixel_count"] != counts["frame_pixels"]:
        raise ReceiptError("frame pixel counts disagree")
    if synthesis["fraction"] != routing["synthesis_fraction"]:
        raise ReceiptError("synthesis fractions disagree")
    if synthesis["within_budget"] != routing["within_synthesis_budget"]:
        raise ReceiptError("synthesis budget decisions disagree")
    if synthesis["maximum_fraction"] != routing["policy"]["max_synth_fraction"]:
        raise ReceiptError("synthesis maximum fractions disagree")
    if (
        synthesis["fraction"]
        != synthesis["pixel_count"] / synthesis["frame_pixel_count"]
    ):
        raise ReceiptError("synthesis fraction is inconsistent with its counts")
    maximum = Fraction(str(synthesis["maximum_fraction"]))
    within_budget = (
        synthesis["pixel_count"] * maximum.denominator
        <= synthesis["frame_pixel_count"] * maximum.numerator
    )
    if synthesis["within_budget"] != within_budget:
        raise ReceiptError("synthesis budget decision is inconsistent with its counts")
    if (
        document["core"]["diagnostics"]["at_floor_pixel_count"]
        != counts["at_floor_pixels"]
    ):
        raise ReceiptError("at-floor pixel counts disagree")
    if (
        sum(region["area"] for region in routing["final_regions"])
        != synthesis["pixel_count"]
    ):
        raise ReceiptError("final component areas do not equal synthesis count")
    provisional = routing["provisional_regions"]
    final = routing["final_regions"]
    if counts["provisional_regions"] != len(provisional):
        raise ReceiptError("provisional region count disagrees with its table")
    if counts["final_regions"] != len(final):
        raise ReceiptError("final region count disagrees with its table")
    provisional_ids = [region["id"] for region in provisional]
    if provisional_ids != list(range(1, len(provisional_ids) + 1)):
        raise ReceiptError("provisional region IDs must be contiguous and ascending")
    if counts["at_floor_pixels"] != sum(region["area"] for region in provisional):
        raise ReceiptError("at-floor count disagrees with provisional region areas")
    if counts["directly_routed_pixels"] != sum(
        region["area"] for region in provisional if region["disposition"] == "routed"
    ):
        raise ReceiptError("directly routed count disagrees with dispositions")
    if counts["non_floor_synthesis_pixels"] != sum(
        region["non_floor_synthesis_pixels"] for region in final
    ):
        raise ReceiptError("non-floor synthesis count disagrees with final regions")
    absorbed_pixels = sum(
        region["absorbed_pixel_count"]
        for region in provisional
        if region["disposition"] == "absorbed"
    )
    if synthesis["pixel_count"] != (
        counts["directly_routed_pixels"]
        + absorbed_pixels
        + counts["non_floor_synthesis_pixels"]
    ):
        raise ReceiptError("synthesis pixel accounting is not conservative")
    if not 0 <= counts["unchanged_synthesis_pixels"] <= synthesis["pixel_count"]:
        raise ReceiptError("unchanged synthesis count is outside its valid range")

    excluded = routing["perimeter_excluded_region"]
    excluded_rows = [
        region
        for region in provisional
        if region["disposition"] == "perimeter_excluded"
    ]
    if len(excluded_rows) != counts["perimeter_excluded_regions"]:
        raise ReceiptError("perimeter exclusion disposition count disagrees")
    if counts["perimeter_excluded_regions"] == 0:
        if excluded is not None or counts["perimeter_excluded_pixels"] != 0:
            raise ReceiptError("perimeter exclusion object and counts disagree")
    else:
        if excluded is None:
            raise ReceiptError("perimeter exclusion counts require an explicit object")
        excluded_row = next(
            (region for region in provisional if region["id"] == excluded["id"]),
            None,
        )
        if (
            excluded_row is None
            or excluded_row["disposition"] != "perimeter_excluded"
            or excluded_row["area"] != excluded["area"]
            or excluded_row["bbox_yxyx_half_open"] != excluded["bbox_yxyx_half_open"]
            or excluded_row["final_component_ids"]
            or excluded_row["absorbed_pixel_count"] != 0
            or counts["perimeter_excluded_pixels"] != excluded["area"]
        ):
            raise ReceiptError("perimeter exclusion object disagrees with its region")

    final_ids = [region["id"] for region in routing["final_regions"]]
    if final_ids != list(range(1, len(final_ids) + 1)):
        raise ReceiptError("final component IDs must be contiguous and ascending")
    final_id_set = set(final_ids)
    provisional_by_id = {region["id"]: region for region in provisional}
    final_table_by_id = {region["id"]: region for region in final}
    for region in provisional:
        if region["final_component_ids"] != sorted(
            set(region["final_component_ids"])
        ) or not set(region["final_component_ids"]).issubset(final_id_set):
            raise ReceiptError(
                f"provisional region {region['id']} has invalid final bindings"
            )
    for region in final:
        if region["direct_region_ids"] != sorted(
            set(region["direct_region_ids"])
        ) or region["absorbed_region_ids"] != sorted(
            set(region["absorbed_region_ids"])
        ):
            raise ReceiptError(f"final region {region['id']} has duplicate bindings")
        for provisional_id in region["direct_region_ids"]:
            bound = provisional_by_id.get(provisional_id)
            if (
                bound is None
                or bound["disposition"] != "routed"
                or region["id"] not in bound["final_component_ids"]
            ):
                raise ReceiptError("direct routing bindings disagree")
        for provisional_id in region["absorbed_region_ids"]:
            bound = provisional_by_id.get(provisional_id)
            if (
                bound is None
                or bound["disposition"] != "absorbed"
                or region["id"] not in bound["final_component_ids"]
            ):
                raise ReceiptError("absorbed routing bindings disagree")
    for region in provisional:
        binding_field = (
            "direct_region_ids"
            if region["disposition"] == "routed"
            else "absorbed_region_ids"
            if region["disposition"] == "absorbed"
            else None
        )
        if binding_field is None and region["final_component_ids"]:
            raise ReceiptError(
                f"provisional region {region['id']} cannot bind final components"
            )
        if binding_field is not None:
            for final_id in region["final_component_ids"]:
                if region["id"] not in final_table_by_id[final_id][binding_field]:
                    raise ReceiptError("routing component bindings are not reciprocal")
    composite_rows = document["composite"]["components"]
    if [row["component_id"] for row in composite_rows] != final_ids:
        raise ReceiptError("composite order must equal ascending final component IDs")
    final_by_id = {region["id"]: region for region in routing["final_regions"]}
    referenced_crop_artifacts: set[str] = set()
    for row in composite_rows:
        component_id = row["component_id"]
        final = final_by_id[component_id]
        if (
            row["component_bbox_yxyx_half_open"] != final["bbox_yxyx_half_open"]
            or row["pixel_count"] != final["area"]
        ):
            raise ReceiptError(f"component {component_id} disagrees with routing")
        expected_roles = {
            "input_rgb8": "component_input_rgb8",
            "component_mask_png": "component_mask_png",
            "inpainted_rgb8": "component_inpainted_rgb8",
        }
        expected_hashes = {
            "input_rgb8": "input_rgb8_raw_sha256",
            "component_mask_png": "component_mask_u8_raw_sha256",
            "inpainted_rgb8": "inpainted_rgb8_raw_sha256",
        }
        for field, role in expected_roles.items():
            artifact_id = row["artifacts"][field]
            if artifact_id not in artifacts or artifacts[artifact_id]["role"] != role:
                raise ReceiptError(
                    f"component {component_id} {field} does not reference role {role}"
                )
            if (
                artifacts[artifact_id]["raw_sha256"]
                != row["hashes"][expected_hashes[field]]
            ):
                raise ReceiptError(f"component {component_id} {field} hash disagrees")
            referenced_crop_artifacts.add(artifact_id)
    declared_crop_artifacts = {
        artifact_id
        for artifact_id, artifact in artifacts.items()
        if artifact["role"] in _CROP_ROLES
    }
    if referenced_crop_artifacts != declared_crop_artifacts:
        raise ReceiptError("crop artifact table and component references differ")

    composite = document["composite"]
    pure = artifacts[synthesis["pure_output_artifact_id"]]
    hybrid = artifacts[synthesis["hybrid_output_artifact_id"]]
    if composite["pure_rgb16_raw_sha256"] != pure["raw_sha256"]:
        raise ReceiptError("pure output hashes disagree")
    if composite["hybrid_rgb16_raw_sha256"] != hybrid["raw_sha256"]:
        raise ReceiptError("hybrid output hashes disagree")


def _validate_output_array(array: np.ndarray, role: str) -> None:
    if array.dtype.str != "<u2":
        raise ReceiptError(f"{role} must decode as little-endian uint16")
    if array.ndim != 3 or array.shape[2] != 3 or 0 in array.shape:
        raise ReceiptError(f"{role} must have non-empty shape HxWx3")


def _decode_mask_array(array: np.ndarray) -> npt.NDArray[np.bool_]:
    if array.dtype != np.dtype(np.uint8) or array.ndim != 2 or 0 in array.shape:
        raise ReceiptError("synthesis mask PNG must decode as non-empty HxW uint8")
    unique = np.unique(array)
    if np.any((unique != 0) & (unique != 255)):
        raise ReceiptError("synthesis mask PNG must contain only 0 and 255")
    return np.ascontiguousarray(array == 255, dtype=np.bool_)


def _bbox_slices(bbox: Sequence[int]) -> tuple[slice, slice]:
    y0, x0, y1, x1 = (int(value) for value in bbox)
    return slice(y0, y1), slice(x0, x1)


def _verify_composite_replay(
    document: Mapping[str, Any],
    measurements: Mapping[str, _ArtifactMeasurement],
    pure: np.ndarray,
    hybrid: np.ndarray,
    mask: npt.NDArray[np.bool_],
    expected_final_labels: npt.NDArray[np.int32],
) -> None:
    rows = document["composite"]["components"]
    labels = np.zeros(mask.shape, dtype=np.int32)
    inpainted_by_id: dict[int, np.ndarray] = {}
    bindings: list[ComponentArtifactBinding] = []
    for row in rows:
        component_id = int(row["component_id"])
        crop_bbox = row["crop_bbox_yxyx_half_open"]
        component_bbox = row["component_bbox_yxyx_half_open"]
        crop_y0, crop_x0, crop_y1, crop_x1 = (int(value) for value in crop_bbox)
        component_y0, component_x0, component_y1, component_x1 = (
            int(value) for value in component_bbox
        )
        if not (
            0 <= crop_y0 <= component_y0 < component_y1 <= crop_y1 <= mask.shape[0]
            and 0 <= crop_x0 <= component_x0 < component_x1 <= crop_x1 <= mask.shape[1]
        ):
            raise ReceiptError(f"component {component_id} has invalid crop geometry")

        artifact_ids = row["artifacts"]
        input_rgb8 = measurements[artifact_ids["input_rgb8"]].array
        component_mask_u8 = measurements[artifact_ids["component_mask_png"]].array
        inpainted_rgb8 = measurements[artifact_ids["inpainted_rgb8"]].array
        if input_rgb8 is None or component_mask_u8 is None or inpainted_rgb8 is None:
            raise ReceiptError(
                f"component {component_id} artifacts must decode as arrays"
            )
        crop_shape = (crop_y1 - crop_y0, crop_x1 - crop_x0)
        if (
            input_rgb8.dtype != np.dtype(np.uint8)
            or input_rgb8.shape != (*crop_shape, 3)
            or inpainted_rgb8.dtype != np.dtype(np.uint8)
            or inpainted_rgb8.shape != input_rgb8.shape
        ):
            raise ReceiptError(
                f"component {component_id} RGB crop artifacts are invalid"
            )
        local_component = _decode_mask_array(component_mask_u8)
        if local_component.shape != crop_shape:
            raise ReceiptError(
                f"component {component_id} mask crop geometry is invalid"
            )
        if int(np.count_nonzero(local_component)) != row["pixel_count"]:
            raise ReceiptError(f"component {component_id} mask pixel count disagrees")

        locations = np.argwhere(local_component)
        if locations.size == 0:
            raise ReceiptError(f"component {component_id} mask is empty")
        local_bbox = [
            int(locations[:, 0].min()) + crop_y0,
            int(locations[:, 1].min()) + crop_x0,
            int(locations[:, 0].max()) + crop_y0 + 1,
            int(locations[:, 1].max()) + crop_x0 + 1,
        ]
        if local_bbox != component_bbox:
            raise ReceiptError(f"component {component_id} mask bounding box disagrees")
        label_view = labels[_bbox_slices(crop_bbox)]
        if np.any(label_view[local_component] != 0):
            raise ReceiptError("component masks overlap")
        label_view[local_component] = component_id
        inpainted_by_id[component_id] = inpainted_rgb8
        bindings.append(
            ComponentArtifactBinding(
                component_id=component_id,
                input_rgb8_artifact_id=artifact_ids["input_rgb8"],
                mask_artifact_id=artifact_ids["component_mask_png"],
                inpainted_rgb8_artifact_id=artifact_ids["inpainted_rgb8"],
            )
        )

    if not np.array_equal(labels > 0, mask):
        raise ReceiptError(
            "component mask artifacts do not exactly union to synthesis mask"
        )
    if not np.array_equal(labels, expected_final_labels):
        raise ReceiptError(
            "component label plane disagrees with deterministic routing replay"
        )
    call_index = 0

    def replay_inpainter(
        input_rgb8: npt.NDArray[np.uint8],
        component_mask_u8: npt.NDArray[np.uint8],
    ) -> npt.NDArray[np.uint8]:
        nonlocal call_index
        if call_index >= len(rows):
            raise ReceiptError("composite replay made an unexpected model call")
        row = rows[call_index]
        call_index += 1
        artifact_ids = row["artifacts"]
        expected_input = measurements[artifact_ids["input_rgb8"]].array
        expected_mask = measurements[artifact_ids["component_mask_png"]].array
        if not np.array_equal(input_rgb8, expected_input):
            raise ReceiptError(
                f"component {row['component_id']} encoded input does not replay"
            )
        if not np.array_equal(component_mask_u8, expected_mask):
            raise ReceiptError(
                f"component {row['component_id']} model mask does not replay"
            )
        return np.array(inpainted_by_id[row["component_id"]], copy=True)

    try:
        replayed = composite_components(
            pure,
            labels,
            mask,
            replay_inpainter,
            crop_margin=document["composite"]["crop_margin"],
        )
    except ReceiptError:
        raise
    except (TypeError, ValueError, RuntimeError) as error:
        raise ReceiptError(f"composite replay failed: {error}") from error
    if call_index != len(rows):
        raise ReceiptError("composite replay did not consume every component")
    if not np.array_equal(replayed.hybrid_rgb16, hybrid):
        raise ReceiptError(
            "hybrid output does not equal deterministic composite replay"
        )
    replayed_rows = _composite_components_document(replayed, bindings)
    if replayed_rows != rows:
        raise ReceiptError("composite records do not equal deterministic replay")


def write_synthesis_mask_png(path: str | Path, mask: npt.ArrayLike) -> str:
    """Write an exact 0/255 grayscale PNG with imageio and return its file hash."""

    array = np.asarray(mask)
    if array.dtype != np.dtype(np.bool_) or array.ndim != 2 or 0 in array.shape:
        raise ReceiptError("synthesis mask must be a non-empty HxW bool array")
    pixels = np.zeros(array.shape, dtype=np.uint8)
    pixels[array] = 255
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".png",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        iio.imwrite(temporary, pixels, extension=".png")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except (OSError, ValueError, RuntimeError) as error:
        raise ReceiptError(f"cannot write synthesis mask PNG: {error}") from error
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
    return _sha256_file(destination)


def write_receipt(path: str | Path, document: Mapping[str, Any]) -> str:
    """Validate and atomically write a canonical receipt; return its SHA-256."""

    validate_receipt_document(document)
    _validate_document_semantics(document)
    payload = canonical_receipt_bytes(document)
    destination = Path(path)
    if destination.name != RECEIPT_FILENAME:
        raise ReceiptError(f"receipt filename must be {RECEIPT_FILENAME}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except OSError as error:
        raise ReceiptError(f"cannot write receipt: {error}") from error
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
    return _sha256_bytes(payload)


def _model_attestation_from_document(
    document: Mapping[str, Any],
) -> ModelWeightsAttestation | None:
    inpainting = document["inpainting"]
    if not inpainting["invoked"]:
        return None
    model = inpainting["model"]
    return ModelWeightsAttestation(
        sanitized_artifact_id=model["sanitized_artifact_id"],
        sha256=model["weights_sha256"],
        byte_size=model["weights_byte_size"],
        sanitized_external_reference=model["sanitized_external_reference"],
    )


def _verify_resolved_model_weights(
    attestation: ModelWeightsAttestation,
    resolver: ModelWeightsResolver,
) -> None:
    try:
        payload = resolver(attestation)
    except Exception as error:
        raise ReceiptError(f"model weights resolver failed: {error}") from error
    if isinstance(payload, (bytes, bytearray, memoryview)):
        raw = bytes(payload)
        byte_size = len(raw)
        digest = _sha256_bytes(raw)
    elif isinstance(payload, (str, Path)):
        path = Path(payload)
        digest_object = hashlib.sha256()
        byte_size = 0
        try:
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest_object.update(block)
                    byte_size += len(block)
        except OSError as error:
            raise ReceiptError(
                f"cannot read resolved model weights: {error}"
            ) from error
        digest = digest_object.hexdigest()
    else:
        raise ReceiptError("model weights resolver must return bytes or a local path")
    if byte_size != attestation.byte_size:
        raise ReceiptError("resolved model weights byte size mismatch")
    if digest != attestation.sha256:
        raise ReceiptError("resolved model weights SHA-256 mismatch")


def _verify_embedded_diagnostics_cache(
    document: Mapping[str, Any],
    receipt_path: Path,
    artifacts: Mapping[str, Mapping[str, Any]],
    measurements: Mapping[str, _ArtifactMeasurement],
    pure_output: np.ndarray,
) -> CachedDiagnostics:
    cache_rows = [
        artifact
        for artifact in artifacts.values()
        if artifact["role"] == "diagnostics_cache_manifest"
    ]
    if len(cache_rows) != 1:
        raise ReceiptError("receipt must identify one diagnostics cache manifest")
    manifest_path = _resolve_artifact(
        receipt_path.parent,
        cache_rows[0]["relative_path"],
    )
    if manifest_path.name != MANIFEST_FILENAME:
        raise ReceiptError(
            f"diagnostics cache manifest must be named {MANIFEST_FILENAME}"
        )
    try:
        cached = verify_diagnostics_cache_snapshot(manifest_path.parent)
    except DiagnosticsCacheError as error:
        raise ReceiptError(
            f"embedded diagnostics cache failed verification: {error}"
        ) from None
    cache_artifact_id = cache_rows[0]["id"]
    if cached.manifest_sha256 != measurements[cache_artifact_id].file_sha256:
        raise ReceiptError(
            "embedded diagnostics cache manifest changed after artifact verification"
        )

    binding = cached.binding
    inputs = document["inputs"]
    core = document["core"]
    backend = core["backend"]
    expected_binding_fields = {
        "prepass_raw_sha256": inputs["prepass"]["raw_sha256"],
        "prepass_shape": tuple(inputs["prepass"]["shape"]),
        "main_raw_sha256": inputs["main"]["raw_sha256"],
        "main_shape": tuple(inputs["main"]["shape"]),
        "provenance_assertion_id": inputs["provenance"]["same_frame_assertion_id"],
        "profile_id": core["profile_id"],
        "core_version": core["version"],
        "core_source_manifest_sha256": core["source_manifest_sha256"],
        "requested_backend": backend["requested"],
        "used_backend": backend["diagnostics"],
    }
    for field, expected in expected_binding_fields.items():
        if getattr(binding, field) != expected:
            raise ReceiptError(
                f"embedded diagnostics cache binding disagrees with receipt {field}"
            )

    run_metadata_rows = [
        artifact
        for artifact in artifacts.values()
        if artifact["role"] == "run_metadata_json"
    ]
    if len(run_metadata_rows) != 1:
        raise ReceiptError("receipt must identify one run metadata artifact")
    run_metadata_artifact_id = run_metadata_rows[0]["id"]
    run_metadata_bytes = measurements[run_metadata_artifact_id].opaque_bytes
    if run_metadata_bytes is None:
        raise ReceiptError("run metadata artifact has no verified byte snapshot")
    run_metadata = _load_canonical_json_artifact(
        run_metadata_bytes,
        label="run metadata JSON",
    )
    try:
        assertion_sha256 = run_metadata["provenance"]["assertion_sha256"]
    except (KeyError, TypeError) as error:
        raise ReceiptError(
            "run metadata artifact does not expose provenance assertion_sha256"
        ) from error
    try:
        _require_sha256(
            assertion_sha256,
            "run metadata provenance assertion_sha256",
        )
    except ReceiptError as error:
        raise ReceiptError(
            "run metadata artifact has an invalid provenance assertion_sha256"
        ) from error
    if binding.provenance_assertion_sha256 != assertion_sha256:
        raise ReceiptError(
            "embedded diagnostics cache binding disagrees with run metadata "
            "provenance_assertion_sha256"
        )

    if not np.array_equal(cached.output_rgb16, pure_output):
        raise ReceiptError(
            "embedded diagnostics cache pure output disagrees with receipt artifact"
        )
    diagnostics = cached.diagnostics
    declared_diagnostics = core["diagnostics"]
    if (
        _score_floor_bits(diagnostics)
        != declared_diagnostics["score_floor_u32_le_bits"]
    ):
        raise ReceiptError("embedded diagnostics cache score floor disagrees")
    if (
        int(np.count_nonzero(diagnostics.at_floor_mask))
        != declared_diagnostics["at_floor_pixel_count"]
    ):
        raise ReceiptError("embedded diagnostics cache at-floor count disagrees")
    if (
        int(np.count_nonzero(diagnostics.changed_mask))
        != declared_diagnostics["changed_pixel_count"]
    ):
        raise ReceiptError("embedded diagnostics cache changed count disagrees")
    return cached


def _one_artifact_for_role(
    artifacts: Mapping[str, Mapping[str, Any]],
    role: str,
) -> Mapping[str, Any]:
    rows = [artifact for artifact in artifacts.values() if artifact["role"] == role]
    if len(rows) != 1:
        raise ReceiptError(f"receipt must identify one {role} artifact")
    return rows[0]


def _load_canonical_json_artifact(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReceiptError(f"cannot decode {label} artifact: {error}") from error
    if not isinstance(document, dict):
        raise ReceiptError(f"{label} artifact must contain a JSON object")
    if canonical_receipt_bytes(document) != payload:
        raise ReceiptError(f"{label} artifact is not in canonical JSON form")
    return document


def _operational_routing_from_receipt(
    receipt_routing: Mapping[str, Any],
) -> dict[str, Any]:
    """Remove only the evidence fields added while constructing the receipt."""

    # Round-tripping through canonical JSON gives us an unaliased JSON value and
    # also refuses any value that cannot be represented by the on-disk contract.
    routing = json.loads(canonical_receipt_bytes(receipt_routing))
    try:
        del routing["scipy_version"]
        del routing["id_rules"]
        del routing["counts"]["unchanged_synthesis_pixels"]
        excluded = routing["perimeter_excluded_region"]
        if excluded is not None:
            del excluded["output_treatment"]
    except (KeyError, TypeError) as error:  # schema normally catches this first
        raise ReceiptError(
            "receipt routing is missing its receipt-only evidence fields"
        ) from error
    return routing


def _verify_routing_replay(
    receipt_routing: Mapping[str, Any],
    cached: CachedDiagnostics,
    synthesis_mask: npt.NDArray[np.bool_],
) -> RoutingResult:
    """Recompute routing from verified diagnostics and bind every output plane."""

    try:
        policy_document = receipt_routing["policy"]
        policy = RoutingPolicy(
            min_area=policy_document["min_area"],
            min_radius=policy_document["min_radius"],
            margin=policy_document["margin"],
            max_synth_fraction=policy_document["max_synth_fraction"],
        )
        replayed = route_at_floor_mask(
            cached.diagnostics.at_floor_mask,
            policy,
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise ReceiptError(f"deterministic routing replay failed: {error}") from error

    expected_document = _operational_routing_from_receipt(receipt_routing)
    if routing_json_document(replayed) != expected_document:
        raise ReceiptError(
            "receipt routing document disagrees with deterministic routing replay"
        )
    if not np.array_equal(
        replayed.provisional_labels > 0,
        cached.diagnostics.at_floor_mask,
    ):
        raise ReceiptError(
            "provisional label plane disagrees with verified at-floor diagnostics"
        )
    if not np.array_equal(replayed.synthesis_mask, synthesis_mask):
        raise ReceiptError("synthesis mask disagrees with deterministic routing replay")
    if not np.array_equal(replayed.final_labels > 0, synthesis_mask):
        raise ReceiptError(
            "final label plane disagrees with deterministic routing replay"
        )
    return replayed


def _runtime_document_from_receipt(
    inpainting: Mapping[str, Any],
) -> dict[str, Any]:
    tool = inpainting["tool"]
    model = inpainting["model"]
    runtime = inpainting["runtime"]
    determinism = inpainting["determinism"]
    return {
        "tool_name": tool["name"],
        "tool_version": tool["version"],
        "tool_license_spdx": tool["tool_license_spdx"],
        "iopaint_source_manifest_sha256": (tool["iopaint_source_manifest_sha256"]),
        "iopaint_source_file_count": tool["iopaint_source_file_count"],
        "python_version": runtime["python_version"],
        "python_implementation": runtime["python_implementation"],
        "torch_version": runtime["torch_version"],
        "numpy_version": runtime["numpy_version"],
        "pillow_version": runtime["pillow_version"],
        "opencv_version": runtime["opencv_version"],
        "pydantic_version": runtime["pydantic_version"],
        "typer_version": runtime["typer_version"],
        "platform_system": runtime["platform_system"],
        "platform_release": runtime["platform_release"],
        "platform_machine": runtime["platform_machine"],
        "deterministic_algorithms": determinism["deterministic_algorithms_enabled"],
        "cudnn_benchmark": determinism["cudnn_benchmark"],
        "cuda_available": runtime["cuda_available"],
        "cuda_runtime_version": runtime["cuda_runtime_version"],
        "cudnn_version": runtime["cudnn_version"],
        "hip_runtime_version": runtime["hip_runtime_version"],
        "cuda_device_names": runtime["cuda_device_names"],
        "cuda_visible_devices": runtime["cuda_visible_devices"],
        "mps_available": runtime["mps_available"],
        "mps_device_name": runtime["mps_device_name"],
        "effective_environment_sha256": runtime["effective_environment_sha256"],
        "model_name": model["id"],
        "model_release": model["version"],
        "model_artifact_identifier": model["sanitized_artifact_id"],
        "model_weights_sha256": model["weights_sha256"],
        "model_upstream_license_spdx": model["model_upstream_license_spdx"],
        "model_artifact_license_status": model["model_artifact_license_status"],
        "device": runtime["device"],
        "thread_count": runtime["threads"],
        "seed": runtime["seed"],
        "seed_scope": runtime["seed_scope"],
        "determinism_scope": determinism["scope"],
    }


def _deterministic_environment_from_receipt(
    inpainting: Mapping[str, Any],
) -> dict[str, str]:
    runtime = inpainting["runtime"]
    thread_count = str(runtime["threads"])
    environment = {
        "DIFFUSERS_CACHE": "<private-model-cache>/huggingface/hub",
        "HF_HUB_OFFLINE": "1",
        "HF_HOME": "<private-model-cache>/huggingface",
        "HF_HUB_CACHE": "<private-model-cache>/huggingface/hub",
        "HUGGINGFACE_HUB_CACHE": "<private-model-cache>/huggingface/hub",
        "LAMA_MODEL_URL": "<private-model-artifact>",
        "MKL_DYNAMIC": "FALSE",
        "MKL_NUM_THREADS": thread_count,
        "NUMEXPR_NUM_THREADS": thread_count,
        "OMP_DYNAMIC": "FALSE",
        "OMP_NUM_THREADS": thread_count,
        "OPENBLAS_NUM_THREADS": thread_count,
        "TORCH_HOME": "<private-model-cache>/torch",
        "TRANSFORMERS_CACHE": "<private-model-cache>/huggingface/hub",
        "TRANSFORMERS_OFFLINE": "1",
        "VECLIB_MAXIMUM_THREADS": thread_count,
        "XDG_CACHE_HOME": "<private-model-cache>",
    }
    if runtime["device"] == "cuda":
        environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return environment


def _require_nonnegative_finite_number(value: object, *, field: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise ReceiptError(f"run metadata {field} must be finite and non-negative")


def _verify_operational_documents(
    document: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
    measurements: Mapping[str, _ArtifactMeasurement],
) -> None:
    """Bind the two operator-facing JSON files to the receipt's claims."""

    routing_artifact = _one_artifact_for_role(artifacts, "routing_json")
    routing_bytes = measurements[routing_artifact["id"]].opaque_bytes
    if routing_bytes is None:
        raise ReceiptError("routing JSON artifact has no verified byte snapshot")
    operational_routing = _load_canonical_json_artifact(
        routing_bytes,
        label="routing JSON",
    )
    expected_routing = _operational_routing_from_receipt(document["routing"])
    if operational_routing != expected_routing:
        raise ReceiptError("routing JSON artifact disagrees with receipt routing")

    metadata_artifact = _one_artifact_for_role(artifacts, "run_metadata_json")
    metadata_bytes = measurements[metadata_artifact["id"]].opaque_bytes
    if metadata_bytes is None:
        raise ReceiptError("run metadata JSON artifact has no verified byte snapshot")
    run_metadata = _load_canonical_json_artifact(
        metadata_bytes,
        label="run metadata JSON",
    )
    expected_root_keys = {
        "artifacts",
        "backend",
        "cache",
        "generative_model_loaded",
        "inpainting",
        "inputs",
        "mode",
        "profile",
        "provenance",
        "routing",
        "schema",
        "source_manifests",
        "timing",
    }
    if set(run_metadata) != expected_root_keys:
        raise ReceiptError("run metadata JSON has an unexpected document shape")
    if run_metadata["schema"] != "fauxce-hybrid-run-metadata-v1":
        raise ReceiptError("run metadata JSON has an unsupported schema")

    inputs = document["inputs"]
    expected_inputs = {
        "main": {
            "raw_sha256": inputs["main"]["raw_sha256"],
            "shape": inputs["main"]["shape"],
        },
        "prepass": {
            "raw_sha256": inputs["prepass"]["raw_sha256"],
            "shape": inputs["prepass"]["shape"],
        },
    }
    if run_metadata["inputs"] != expected_inputs:
        raise ReceiptError("run metadata inputs disagree with receipt inputs")

    core = document["core"]
    backend = core["backend"]
    expected_backend = {
        "reason": backend["reason"],
        "requested": backend["requested"],
        "used": backend["used"],
    }
    if run_metadata["backend"] != expected_backend:
        raise ReceiptError("run metadata backend disagrees with receipt core")
    expected_profile = {
        "bit_depth": 16,
        "id": core["profile_id"],
        "main_dpi": 4_000,
        "mode": "normal",
        "prepass_dpi": 285,
        "resolution_metric": 4_000,
        "scanner_model": "nikon-super-coolscan-5000-ed",
        "selector": 8,
    }
    if run_metadata["profile"] != expected_profile:
        raise ReceiptError("run metadata profile disagrees with receipt core")
    expected_source_manifests = {
        "core": core["source_manifest_sha256"],
        "hybrid": document["generation"]["hybrid_source_manifest_sha256"],
    }
    if run_metadata["source_manifests"] != expected_source_manifests:
        raise ReceiptError("run metadata source manifests disagree with receipt")

    source_manifest = inputs["provenance"]["source_manifest_sha256"]
    acquisition_manifest = (
        {"provided": False}
        if source_manifest is None
        else {
            "file_sha256": source_manifest,
            "provided": True,
            "validated_claims": [
                "same_frame_id",
                "focus_exposure_locked",
                "prepass_raw_sha256",
                "main_raw_sha256",
            ],
        }
    )
    run_provenance = run_metadata["provenance"]
    try:
        original_same_frame_id = run_provenance["assertion"]["assertions"][
            "same_frame_id"
        ]
    except (KeyError, TypeError) as error:
        raise ReceiptError(
            "run metadata provenance has an unexpected document shape"
        ) from error
    _require_text(original_same_frame_id, "run metadata same_frame_id")
    assertion = {
        "acquisition_manifest": acquisition_manifest,
        "assertions": {
            "focus_exposure_locked": True,
            "same_frame_id": original_same_frame_id,
        },
        "inputs": {
            "main": {"raw_sha256": inputs["main"]["raw_sha256"]},
            "prepass": {"raw_sha256": inputs["prepass"]["raw_sha256"]},
        },
        "provenance_class": "caller_asserted_bare_npy",
        "scanner_evidence": False,
        "schema": "fauxce-hybrid-caller-assertion-v1",
    }
    assertion_sha256 = _sha256_bytes(canonical_receipt_bytes(assertion))
    expected_assertion_id = f"caller-asserted-bare-npy:{assertion_sha256[:16]}"
    assertion_id = inputs["provenance"]["same_frame_assertion_id"]
    if assertion_id != expected_assertion_id:
        raise ReceiptError("receipt same-frame assertion ID is not canonical")
    if (
        inputs["provenance"]["focus_exposure_assertion_id"]
        != f"{assertion_id}:focus-exposure-locked"
    ):
        raise ReceiptError("receipt focus/exposure assertion ID is not canonical")
    expected_provenance = {
        "acquisition_manifest_sha256": source_manifest,
        "assertion": assertion,
        "assertion_id": assertion_id,
        "assertion_sha256": assertion_sha256,
        "classification": "caller_asserted_bare_npy",
    }
    if run_provenance != expected_provenance:
        raise ReceiptError("run metadata provenance disagrees with receipt inputs")

    synthesis = document["synthesis"]
    pure_artifact = artifacts[synthesis["pure_output_artifact_id"]]
    hybrid_artifact = artifacts[synthesis["hybrid_output_artifact_id"]]
    mask_artifact = artifacts[synthesis["mask_artifact_id"]]
    cache_artifact = artifacts[synthesis["diagnostics_cache_artifact_id"]]
    cache_path = PurePosixPath(cache_artifact["relative_path"])
    expected_artifacts = {
        "diagnostics_cache": {
            "directory": cache_path.parent.as_posix(),
            "manifest_filename": cache_path.name,
            "manifest_sha256": cache_artifact["file_sha256"],
        },
        "hybrid_output_rgb16": {
            "filename": hybrid_artifact["relative_path"],
            "raw_sha256": hybrid_artifact["raw_sha256"],
        },
        "output_rgb16": {
            "filename": pure_artifact["relative_path"],
            "raw_sha256": pure_artifact["raw_sha256"],
            "source": "portable_digital_ice.ProcessingResult.output_rgb16",
        },
        "routing": {
            "filename": routing_artifact["relative_path"],
            "sha256": routing_artifact["file_sha256"],
        },
        "synthesis_mask": {
            "filename": mask_artifact["relative_path"],
            "sha256": mask_artifact["file_sha256"],
        },
    }
    if run_metadata["artifacts"] != expected_artifacts:
        raise ReceiptError("run metadata artifact identities disagree with receipt")

    run_cache = run_metadata["cache"]
    if not isinstance(run_cache, dict) or set(run_cache) != {
        "embedded_manifest_sha256",
        "external_manifest_sha256",
        "external_mode",
    }:
        raise ReceiptError("run metadata cache has an unexpected document shape")
    if run_cache["embedded_manifest_sha256"] != cache_artifact["file_sha256"]:
        raise ReceiptError(
            "run metadata embedded cache identity disagrees with receipt"
        )
    external_mode = run_cache["external_mode"]
    if external_mode not in ("none", "saved", "loaded"):
        raise ReceiptError("run metadata external cache mode is invalid")
    expected_external_hash = (
        None if external_mode == "none" else cache_artifact["file_sha256"]
    )
    if run_cache["external_manifest_sha256"] != expected_external_hash:
        raise ReceiptError("run metadata external cache identity is inconsistent")

    routing_counts = document["routing"]["counts"]
    expected_routing_summary = {
        "at_floor_pixels": routing_counts["at_floor_pixels"],
        "final_regions": routing_counts["final_regions"],
        "synthesis_fraction": document["routing"]["synthesis_fraction"],
        "synthesis_pixels": routing_counts["synthesis_pixels"],
        "within_synthesis_budget": document["routing"]["within_synthesis_budget"],
    }
    if run_metadata["routing"] != expected_routing_summary:
        raise ReceiptError("run metadata routing summary disagrees with receipt")

    invoked = bool(document["inpainting"]["invoked"])
    if run_metadata["generative_model_loaded"] is not invoked:
        raise ReceiptError("run metadata model-loaded state disagrees with receipt")
    expected_mode = "hybrid_lama_inpaint" if invoked else "hybrid_empty_synthesis"
    if run_metadata["mode"] != expected_mode:
        raise ReceiptError("run metadata mode disagrees with receipt")

    timing = run_metadata["timing"]
    if not isinstance(timing, dict) or set(timing) != {
        "composite_and_inpainting_seconds",
        "iopaint_batch_seconds",
        "routing_seconds",
    }:
        raise ReceiptError("run metadata timing has an unexpected document shape")
    _require_nonnegative_finite_number(
        timing["composite_and_inpainting_seconds"],
        field="composite_and_inpainting_seconds",
    )
    _require_nonnegative_finite_number(
        timing["routing_seconds"],
        field="routing_seconds",
    )
    batch_seconds = timing["iopaint_batch_seconds"]
    if not isinstance(batch_seconds, list) or len(batch_seconds) != int(invoked):
        raise ReceiptError("run metadata batch timing count disagrees with receipt")
    for index, value in enumerate(batch_seconds):
        _require_nonnegative_finite_number(
            value,
            field=f"iopaint_batch_seconds[{index}]",
        )

    if not invoked:
        expected_inpainting = {
            "invoked": False,
            "reason": "empty_synthesis_mask",
        }
        if run_metadata["inpainting"] != expected_inpainting:
            raise ReceiptError("run metadata empty inpainting state disagrees")
        return

    inpainting = document["inpainting"]
    run_inpainting = run_metadata["inpainting"]
    if not isinstance(run_inpainting, dict) or set(run_inpainting) != {
        "invocation_count",
        "invocations",
        "invoked",
        "runtime",
    }:
        raise ReceiptError("run metadata inpainting has an unexpected document shape")
    if run_inpainting["invoked"] is not True:
        raise ReceiptError("run metadata inpainting invocation state disagrees")
    expected_runtime = _runtime_document_from_receipt(inpainting)
    if run_inpainting["runtime"] != expected_runtime:
        raise ReceiptError("run metadata inpainting runtime disagrees with receipt")

    components = document["composite"]["components"]
    invocations = run_inpainting["invocations"]
    if (
        isinstance(run_inpainting["invocation_count"], bool)
        or run_inpainting["invocation_count"] != len(components)
        or not isinstance(invocations, list)
        or len(invocations) != len(components)
    ):
        raise ReceiptError("run metadata invocation count disagrees with components")
    expected_environment = _deterministic_environment_from_receipt(inpainting)
    expected_config = {
        "hd_strategy": "Original",
        "sd_seed": inpainting["runtime"]["seed"],
    }
    expected_argv = inpainting["runtime"]["argv"]
    for component, invocation in zip(components, invocations, strict=True):
        hashes = component["hashes"]
        expected_invocation = {
            "config": expected_config,
            "input_rgb8_raw_sha256": hashes["input_rgb8_raw_sha256"],
            "mask_u8_raw_sha256": hashes["component_mask_u8_raw_sha256"],
            "output_rgb8_raw_sha256": hashes["inpainted_rgb8_raw_sha256"],
            "sanitized_argv": expected_argv,
        }
        if not isinstance(invocation, dict) or set(invocation) != {
            *expected_invocation,
            "deterministic_environment",
        }:
            raise ReceiptError(
                "run metadata invocation has an unexpected document shape"
            )
        actual_environment = invocation["deterministic_environment"]
        if actual_environment != expected_environment:
            raise ReceiptError(
                "run metadata deterministic environment disagrees with receipt"
            )
        actual_invocation = dict(invocation)
        del actual_invocation["deterministic_environment"]
        if actual_invocation != expected_invocation:
            raise ReceiptError(
                "run metadata invocation identity disagrees with receipt component "
                f"{component['component_id']}"
            )


def _read_receipt_payload(path: Path) -> bytes:
    descriptor: int | None = None
    try:
        descriptor = os.open(path, _file_open_flags())
        metadata_value = os.fstat(descriptor)
        if not stat.S_ISREG(metadata_value.st_mode):
            raise ReceiptError("receipt must be a regular file")
        if metadata_value.st_size < 0 or metadata_value.st_size > _MAX_RECEIPT_BYTES:
            raise ReceiptError("receipt encoded size exceeds safe limit")
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = None
            payload = handle.read(metadata_value.st_size + 1)
            if len(payload) != metadata_value.st_size:
                raise ReceiptError("receipt changed while being verified")
            post_hash, post_size = _hash_descriptor(handle)
            final_metadata = os.fstat(handle.fileno())
            if (
                post_hash != _sha256_bytes(payload)
                or post_size != metadata_value.st_size
                or final_metadata.st_size != metadata_value.st_size
                or final_metadata.st_dev != metadata_value.st_dev
                or final_metadata.st_ino != metadata_value.st_ino
            ):
                raise ReceiptError("receipt changed while being verified")
            return payload
    except ReceiptError:
        raise
    except OSError as error:
        raise ReceiptError(f"cannot securely read receipt: {error}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def verify_receipt(
    path: str | Path,
    *,
    model_weights_resolver: ModelWeightsResolver | None = None,
    require_model_weights: bool = False,
) -> VerifiedReceipt:
    """Verify schema, all artifact hashes, mask accounting, and exact scope."""

    if not isinstance(require_model_weights, bool):
        raise TypeError("require_model_weights must be a bool")
    if model_weights_resolver is not None and not callable(model_weights_resolver):
        raise TypeError("model_weights_resolver must be callable")
    receipt_path = Path(path)
    if receipt_path.name != RECEIPT_FILENAME:
        raise ReceiptError(f"receipt filename must be {RECEIPT_FILENAME}")
    try:
        payload = _read_receipt_payload(receipt_path)
        document = json.loads(payload)
    except ReceiptError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReceiptError(f"cannot read receipt: {error}") from error
    if canonical_receipt_bytes(document) != payload:
        raise ReceiptError("receipt is not in canonical JSON form")
    validate_receipt_document(document)
    if (
        document["generation"]["hybrid_source_manifest_sha256"]
        != compute_hybrid_source_manifest()
    ):
        raise ReceiptError(
            "receipt hybrid source manifest does not match the running source tree"
        )
    _validate_document_semantics(document)
    attestation = _model_attestation_from_document(document)
    model_weights_rehashed = False
    if attestation is not None:
        if model_weights_resolver is None:
            if require_model_weights:
                raise ReceiptError(
                    "model weights resolver is required for external rehash verification"
                )
        else:
            _verify_resolved_model_weights(attestation, model_weights_resolver)
            model_weights_rehashed = True

    artifacts = _artifacts_by_id(document)
    measurements: dict[str, _ArtifactMeasurement] = {}
    decoded_budget = _DecodedArtifactBudget(_MAX_AGGREGATE_DECODED_BYTES)
    for artifact_id, artifact in artifacts.items():
        try:
            encoding = RawEncoding(artifact["raw_encoding"])
        except (
            ValueError
        ) as error:  # schema should catch this; retain fail-closed guard
            raise ReceiptError(
                f"unsupported artifact encoding for {artifact_id}"
            ) from error
        measurement = _measure_artifact(
            receipt_path.parent,
            artifact["relative_path"],
            encoding,
            expected_file_sha256=artifact["file_sha256"],
            expected_dtype=artifact["dtype"],
            expected_shape=artifact["shape"],
            artifact_id=artifact_id,
            decoded_budget=decoded_budget,
        )
        if measurement.raw_sha256 != artifact["raw_sha256"]:
            raise ReceiptError(f"artifact {artifact_id} raw SHA-256 mismatch")
        if measurement.dtype != artifact["dtype"]:
            raise ReceiptError(f"artifact {artifact_id} dtype mismatch")
        if [*measurement.shape] != artifact["shape"]:
            raise ReceiptError(f"artifact {artifact_id} shape mismatch")
        measurements[artifact_id] = measurement

    synthesis = document["synthesis"]
    pure = measurements[synthesis["pure_output_artifact_id"]].array
    hybrid = measurements[synthesis["hybrid_output_artifact_id"]].array
    mask_u8 = measurements[synthesis["mask_artifact_id"]].array
    if pure is None or hybrid is None or mask_u8 is None:
        raise ReceiptError("key output and mask artifacts must decode as arrays")
    _validate_output_array(pure, "pure output")
    _validate_output_array(hybrid, "hybrid output")
    mask = _decode_mask_array(mask_u8)
    if pure.shape != hybrid.shape or pure.shape[:2] != mask.shape:
        raise ReceiptError("pure, hybrid, and synthesis mask geometry disagree")
    if [*pure.shape] != document["inputs"]["geometry"]["output_shape"]:
        raise ReceiptError("output artifact geometry disagrees with receipt")
    if [*mask.shape] != document["inputs"]["geometry"]["mask_shape"]:
        raise ReceiptError("mask artifact geometry disagrees with receipt")
    cached = _verify_embedded_diagnostics_cache(
        document,
        receipt_path,
        artifacts,
        measurements,
        pure,
    )
    pixel_count = int(np.count_nonzero(mask))
    frame_count = int(mask.size)
    fraction = pixel_count / frame_count
    if pixel_count != synthesis["pixel_count"]:
        raise ReceiptError("decoded synthesis mask pixel count mismatch")
    if frame_count != synthesis["frame_pixel_count"]:
        raise ReceiptError("decoded synthesis mask frame count mismatch")
    if fraction != synthesis["fraction"]:
        raise ReceiptError("decoded synthesis mask fraction mismatch")
    routing_counts = document["routing"]["counts"]
    unchanged_synthesis_pixels = int(
        np.count_nonzero(mask & ~cached.diagnostics.changed_mask)
    )
    if unchanged_synthesis_pixels != routing_counts["unchanged_synthesis_pixels"]:
        raise ReceiptError("unchanged synthesis pixel count disagrees with diagnostics")
    non_floor_synthesis_pixels = int(
        np.count_nonzero(mask & ~cached.diagnostics.at_floor_mask)
    )
    if non_floor_synthesis_pixels != routing_counts["non_floor_synthesis_pixels"]:
        raise ReceiptError("non-floor synthesis pixel count disagrees with diagnostics")
    routing_replay = _verify_routing_replay(
        document["routing"],
        cached,
        mask,
    )
    bool_hash = _raw_sha256(mask)
    if bool_hash != document["composite"]["synthesis_mask_bool_raw_sha256"]:
        raise ReceiptError("decoded synthesis mask bool SHA-256 mismatch")
    if not np.array_equal(hybrid[~mask], pure[~mask]):
        raise ReceiptError(
            "hybrid output is not byte-identical to pure output outside synthesis mask"
        )
    _verify_composite_replay(
        document,
        measurements,
        pure,
        hybrid,
        mask,
        routing_replay.final_labels,
    )
    _verify_operational_documents(
        document,
        artifacts,
        measurements,
    )

    for array in (pure, hybrid, mask):
        array.setflags(write=False)
    return VerifiedReceipt(
        document=document,
        receipt_sha256=_sha256_bytes(payload),
        pure_output_rgb16=pure,
        hybrid_output_rgb16=hybrid,
        synthesis_mask=mask,
        model_weights_rehashed=model_weights_rehashed,
    )


__all__ = [
    "ArtifactSource",
    "ComponentArtifactBinding",
    "CoreRunMetadata",
    "InpaintingMetadata",
    "InputProvenance",
    "ModelWeightsAttestation",
    "ModelWeightsPayload",
    "ModelWeightsResolver",
    "RawEncoding",
    "RECEIPT_FILENAME",
    "RECEIPT_SCHEMA",
    "ReceiptError",
    "VerifiedReceipt",
    "build_receipt_document",
    "canonical_receipt_bytes",
    "compute_hybrid_source_manifest",
    "current_hybrid_version",
    "load_receipt_schema",
    "validate_receipt_document",
    "verify_receipt",
    "write_receipt",
    "write_synthesis_mask_png",
]
