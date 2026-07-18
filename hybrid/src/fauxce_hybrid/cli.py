"""Fail-closed command line boundary for the Fauxce hybrid pipeline."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import imageio.v3 as iio
import numpy as np
import numpy.typing as npt
from portable_digital_ice import (
    AcquisitionEpoch,
    BackendProcessingResult,
    BackendSelection,
    ComputeBackend,
    DEFAULT_PROFILE,
    DualRGBIAcquisition,
    ProcessingDiagnostics,
    ProcessingJob,
    ProcessingMode,
    RGBI16Frame,
    ScannerModel,
    ProcessingResult,
    process as process_digital_ice,
)
from portable_digital_ice.cuda_backend.engine import CudaBackendUnavailable
from portable_digital_ice.fast_cpu import CpuFastUnavailable

from .cache import (
    CachedDiagnostics,
    DiagnosticsCacheBinding,
    DiagnosticsCacheError,
    MANIFEST_FILENAME,
    build_cache_binding,
    canonical_backend_reason,
    canonical_json_bytes,
    compute_core_source_manifest,
    hash_rgb16,
    hash_rgbir16,
    load_diagnostics_cache,
    read_cache_binding,
    save_diagnostics_cache,
)
from .composite import CompositeResult, NoContextPixelsError, composite_components
from .iopaint import (
    IOPaintError,
    IOPaintLaMaAdapter,
    IOPaintLaMaConfig,
    ModelArtifactAttestation,
    measure_model_artifact,
)
from .receipts import (
    ArtifactSource,
    ComponentArtifactBinding,
    CoreRunMetadata,
    InpaintingMetadata,
    InputProvenance,
    ModelWeightsAttestation,
    RawEncoding,
    RECEIPT_FILENAME,
    ReceiptError,
    build_receipt_document,
    compute_hybrid_source_manifest,
    current_hybrid_version,
    verify_receipt,
    write_receipt,
    write_synthesis_mask_png,
)
from .routing import (
    NoHealthyContextError,
    RoutingPolicy,
    RoutingResult,
    SynthesisBudgetExceeded,
    route_at_floor_mask,
    routing_json_document,
)


class HybridCLIError(ValueError):
    """A deterministic, user-actionable command line contract failure."""


@dataclass(frozen=True)
class _Provenance:
    assertion_id: str
    assertion_sha256: str
    document: dict[str, object]
    manifest_sha256: str | None


@dataclass(frozen=True)
class _RunProducts:
    output_rgb16: npt.NDArray[np.uint16]
    diagnostics: ProcessingDiagnostics
    requested_backend: str
    used_backend: str
    backend_reason: str
    cache_mode: str
    cache_manifest_sha256: str | None
    cache_binding: DiagnosticsCacheBinding


def _nonempty_argument(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("must be non-empty")
    if value != value.strip():
        raise argparse.ArgumentTypeError("must not have leading or trailing whitespace")
    return value


def _sha256_argument(value: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise argparse.ArgumentTypeError(
            "must be a lowercase 64-character SHA-256 hex digest"
        )
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fauxce-hybrid",
        description=(
            "Run exact Portable Digital ICE diagnostics and deterministic "
            "routing, with an optional receipt-bound LaMa fallback."
        ),
    )
    parser.add_argument("--prepass", required=True, type=Path)
    parser.add_argument("--main", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--same-frame-id",
        required=True,
        type=_nonempty_argument,
        help="caller-asserted identity shared by the prepass and main capture",
    )
    parser.add_argument(
        "--assert-focus-exposure-locked",
        required=True,
        action="store_true",
        help="assert that main capture retained prepass focus and exposure",
    )
    parser.add_argument(
        "--acquisition-manifest",
        type=Path,
        help=(
            "optional JSON assertion containing same_frame_id, "
            "focus_exposure_locked, prepass_raw_sha256, and main_raw_sha256"
        ),
    )
    parser.add_argument(
        "--backend",
        choices=tuple(backend.value for backend in ComputeBackend),
        default=ComputeBackend.AUTO.value,
    )
    parser.add_argument("--min-area", type=int, default=400)
    parser.add_argument("--min-radius", type=int, default=5)
    parser.add_argument("--margin", type=int, default=4)
    parser.add_argument("--max-synth-fraction", type=float, default=0.02)
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument("--save-diagnostics", type=Path)
    cache_group.add_argument("--from-diagnostics", type=Path)
    parser.add_argument(
        "--diagnostics-manifest-sha256",
        type=_sha256_argument,
        help=(
            "independently obtained SHA-256 of the canonical diagnostics-cache "
            "manifest; required with --from-diagnostics"
        ),
    )
    parser.add_argument(
        "--no-inpaint",
        action="store_true",
        help="stop after routing without probing or loading a model",
    )
    parser.add_argument("--iopaint-python", type=Path)
    parser.add_argument("--iopaint-executable", type=Path)
    parser.add_argument(
        "--iopaint-source-manifest-sha256",
        type=_sha256_argument,
    )
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--model-weights", type=Path)
    parser.add_argument("--model-weights-sha256")
    parser.add_argument(
        "--model-artifact-id",
        type=_nonempty_argument,
        default="Sanster-models-add_big_lama-big-lama.pt",
    )
    parser.add_argument(
        "--model-external-reference",
        type=_nonempty_argument,
        default="https://github.com/Sanster/models/releases/tag/add_big_lama",
    )
    parser.add_argument(
        "--inpaint-device",
        choices=("cpu", "cuda", "mps"),
        default="cpu",
    )
    parser.add_argument("--inpaint-threads", type=int, default=1)
    parser.add_argument("--inpaint-seed", type=int, default=0)
    parser.add_argument("--inpaint-timeout", type=float, default=600.0)
    return parser


def _load_rgbi16(path: Path, *, role: str) -> npt.NDArray[np.uint16]:
    try:
        with path.open("rb") as handle:
            loaded = np.load(handle, allow_pickle=False)
    except (OSError, ValueError, EOFError) as error:
        raise HybridCLIError(f"cannot load {role} array {path}: {error}") from error
    if not isinstance(loaded, np.ndarray):
        raise HybridCLIError(f"{role} input must contain one NumPy array")
    if loaded.dtype != np.dtype(np.uint16):
        raise HybridCLIError(f"{role} input must have dtype uint16")
    if loaded.ndim != 3 or loaded.shape[2] != 4:
        raise HybridCLIError(f"{role} input must have shape HxWx4")
    if loaded.shape[0] == 0 or loaded.shape[1] == 0:
        raise HybridCLIError(f"{role} input cannot have an empty image dimension")
    return np.array(loaded, dtype=np.uint16, order="C", copy=True)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    document: dict[str, object] = {}
    for key, value in pairs:
        if key in document:
            raise HybridCLIError(f"acquisition manifest repeats JSON key {key!r}")
        document[key] = value
    return document


def _reject_nonfinite_json(value: str) -> object:
    raise HybridCLIError(
        f"acquisition manifest contains non-finite JSON number {value!r}"
    )


def _read_acquisition_manifest(path: Path) -> tuple[dict[str, object], str]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise HybridCLIError(
            f"cannot read acquisition manifest {path}: {error}"
        ) from error
    try:
        document = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HybridCLIError(
            f"acquisition manifest is invalid JSON: {error}"
        ) from error
    if not isinstance(document, dict):
        raise HybridCLIError("acquisition manifest must be a JSON object")
    return document, _sha256(payload)


def _nested_manifest_value(
    document: dict[str, object],
    *,
    flat_key: str,
    section: str,
    nested_key: str,
) -> object:
    values: list[object] = []
    if flat_key in document:
        values.append(document[flat_key])
    section_value = document.get(section)
    if section_value is not None:
        if not isinstance(section_value, dict):
            raise HybridCLIError(
                f"acquisition manifest field {section!r} must be an object"
            )
        if nested_key in section_value:
            values.append(section_value[nested_key])
    if not values:
        raise HybridCLIError(
            f"acquisition manifest is missing required claim {flat_key!r}"
        )
    if any(value != values[0] for value in values[1:]):
        raise HybridCLIError(
            f"acquisition manifest has conflicting claims for {flat_key!r}"
        )
    return values[0]


def _manifest_raw_hash(
    document: dict[str, object],
    *,
    role: str,
) -> object:
    flat_key = f"{role}_raw_sha256"
    values: list[object] = []
    if flat_key in document:
        values.append(document[flat_key])
    inputs = document.get("inputs")
    if inputs is not None:
        if not isinstance(inputs, dict):
            raise HybridCLIError(
                "acquisition manifest field 'inputs' must be an object"
            )
        role_document = inputs.get(role)
        if role_document is not None:
            if not isinstance(role_document, dict):
                raise HybridCLIError(
                    f"acquisition manifest inputs.{role} must be an object"
                )
            if "raw_sha256" in role_document:
                values.append(role_document["raw_sha256"])
    if not values:
        raise HybridCLIError(
            f"acquisition manifest is missing required claim {flat_key!r}"
        )
    if any(value != values[0] for value in values[1:]):
        raise HybridCLIError(
            f"acquisition manifest has conflicting claims for {flat_key!r}"
        )
    return values[0]


def _validate_acquisition_manifest(
    document: dict[str, object],
    *,
    same_frame_id: str,
    prepass_raw_sha256: str,
    main_raw_sha256: str,
) -> None:
    manifest_frame_id = _nested_manifest_value(
        document,
        flat_key="same_frame_id",
        section="assertions",
        nested_key="same_frame_id",
    )
    if manifest_frame_id != same_frame_id:
        raise HybridCLIError(
            "acquisition manifest same_frame_id does not match --same-frame-id"
        )
    focus_locked = _nested_manifest_value(
        document,
        flat_key="focus_exposure_locked",
        section="assertions",
        nested_key="focus_exposure_locked",
    )
    if focus_locked is not True:
        raise HybridCLIError(
            "acquisition manifest must assert focus_exposure_locked=true"
        )
    for role, expected_hash in (
        ("prepass", prepass_raw_sha256),
        ("main", main_raw_sha256),
    ):
        claimed_hash = _manifest_raw_hash(document, role=role)
        if not isinstance(claimed_hash, str) or claimed_hash != expected_hash:
            raise HybridCLIError(
                f"acquisition manifest {role}_raw_sha256 does not match input bytes"
            )


def _build_provenance(
    *,
    same_frame_id: str,
    prepass_raw_sha256: str,
    main_raw_sha256: str,
    acquisition_manifest: Path | None,
) -> _Provenance:
    manifest_sha256: str | None = None
    if acquisition_manifest is not None:
        manifest, manifest_sha256 = _read_acquisition_manifest(acquisition_manifest)
        _validate_acquisition_manifest(
            manifest,
            same_frame_id=same_frame_id,
            prepass_raw_sha256=prepass_raw_sha256,
            main_raw_sha256=main_raw_sha256,
        )
    document: dict[str, object] = {
        "acquisition_manifest": (
            {"provided": False}
            if manifest_sha256 is None
            else {
                "file_sha256": manifest_sha256,
                "provided": True,
                "validated_claims": [
                    "same_frame_id",
                    "focus_exposure_locked",
                    "prepass_raw_sha256",
                    "main_raw_sha256",
                ],
            }
        ),
        "assertions": {
            "focus_exposure_locked": True,
            "same_frame_id": same_frame_id,
        },
        "inputs": {
            "main": {"raw_sha256": main_raw_sha256},
            "prepass": {"raw_sha256": prepass_raw_sha256},
        },
        "provenance_class": "caller_asserted_bare_npy",
        "scanner_evidence": False,
        "schema": "fauxce-hybrid-caller-assertion-v1",
    }
    assertion_sha256 = _sha256(canonical_json_bytes(document))
    return _Provenance(
        assertion_id=f"caller-asserted-bare-npy:{assertion_sha256[:16]}",
        assertion_sha256=assertion_sha256,
        document=document,
        manifest_sha256=manifest_sha256,
    )


def _build_job(
    prepass: npt.NDArray[np.uint16],
    main: npt.NDArray[np.uint16],
    *,
    same_frame_id: str,
    prepass_raw_sha256: str,
    main_raw_sha256: str,
) -> ProcessingJob:
    acquisition = DualRGBIAcquisition(
        prepass=RGBI16Frame(
            prepass,
            AcquisitionEpoch.PREPASS,
            285,
            f"caller-asserted-npy-sha256:{prepass_raw_sha256}",
        ),
        main=RGBI16Frame(
            main,
            AcquisitionEpoch.MAIN,
            4_000,
            f"caller-asserted-npy-sha256:{main_raw_sha256}",
        ),
        same_frame_id=same_frame_id,
    )
    return ProcessingJob(
        acquisition=acquisition,
        scanner_model=ScannerModel.NIKON_SUPER_COOLSCAN_5000_ED,
        mode=ProcessingMode.NORMAL,
        selector=8,
        resolution_metric=4_000,
        bit_depth=16,
        focus_exposure_locked=True,
    )


def _process_with_auto_full_run_fallback(
    job: ProcessingJob,
    *,
    requested_backend: str,
) -> BackendProcessingResult:
    try:
        processed = process_digital_ice(
            job,
            backend=requested_backend,
            export_diagnostics=True,
        )
    except CudaBackendUnavailable:
        if requested_backend != ComputeBackend.AUTO.value:
            raise
        fallback = process_digital_ice(
            job,
            backend=ComputeBackend.CPU,
            export_diagnostics=True,
        )
        return BackendProcessingResult(
            result=fallback.result,
            selection=BackendSelection(
                requested=ComputeBackend.AUTO,
                used=ComputeBackend.CPU,
                reason=canonical_backend_reason("auto", "cpu"),
            ),
        )
    selection = processed.selection
    if selection.requested.value != requested_backend:
        raise HybridCLIError("core returned a mismatched requested backend")
    return BackendProcessingResult(
        result=processed.result,
        selection=BackendSelection(
            requested=selection.requested,
            used=selection.used,
            reason=canonical_backend_reason(
                selection.requested.value,
                selection.used.value,
            ),
        ),
    )


def _require_new_output_directory(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise HybridCLIError(f"output directory already exists: {path}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise HybridCLIError(
            f"cannot create output parent {path.parent}: {error}"
        ) from error


def _atomic_publish_directory(source: Path, destination: Path) -> None:
    """Rename a staged directory only when the destination is still absent."""

    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform == "darwin":
        library = ctypes.CDLL(None, use_errno=True)
        rename_exclusive = library.renamex_np
        rename_exclusive.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename_exclusive.restype = ctypes.c_int
        result = rename_exclusive(source_bytes, destination_bytes, 0x00000004)
    elif sys.platform.startswith("linux"):
        library = ctypes.CDLL(None, use_errno=True)
        try:
            rename_exclusive = library.renameat2
        except AttributeError as error:
            raise HybridCLIError(
                "atomic no-replace directory publication is unavailable"
            ) from error
        rename_exclusive.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename_exclusive.restype = ctypes.c_int
        result = rename_exclusive(
            -100,
            source_bytes,
            -100,
            destination_bytes,
            0x00000001,
        )
    elif os.name == "nt":  # pragma: no cover - Windows os.rename is exclusive
        os.rename(source, destination)
        return
    else:  # pragma: no cover - fail closed on an unproven platform
        raise HybridCLIError(
            "atomic no-replace directory publication is unsupported on this platform"
        )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in (errno.EEXIST, errno.ENOTEMPTY):
            raise FileExistsError(
                error_number,
                "output directory appeared before atomic publication",
                os.fspath(destination),
            )
        raise OSError(
            error_number,
            os.strerror(error_number),
            os.fspath(destination),
        )


def _require_source_manifests_unchanged(
    *,
    core_source_manifest_sha256: str,
    hybrid_source_manifest_sha256: str,
) -> None:
    if compute_core_source_manifest() != core_source_manifest_sha256:
        raise HybridCLIError("core source changed during the run")
    if compute_hybrid_source_manifest() != hybrid_source_manifest_sha256:
        raise HybridCLIError("hybrid source changed during the run")


def _validate_cache_path_relationships(
    output: Path,
    cache: Path | None,
) -> None:
    if cache is None:
        return
    output_resolved = output.resolve(strict=False)
    cache_resolved = cache.resolve(strict=False)
    if (
        output_resolved == cache_resolved
        or output_resolved in cache_resolved.parents
        or cache_resolved in output_resolved.parents
    ):
        raise HybridCLIError("diagnostics cache and output directory must be disjoint")


def _obtain_run_products(
    *,
    args: argparse.Namespace,
    prepass: npt.NDArray[np.uint16],
    main: npt.NDArray[np.uint16],
    job: ProcessingJob,
    provenance: _Provenance,
    core_source_manifest_sha256: str,
) -> _RunProducts:
    if args.from_diagnostics is not None:
        assert args.diagnostics_manifest_sha256 is not None
        stored_binding = read_cache_binding(
            args.from_diagnostics,
            expected_manifest_sha256=args.diagnostics_manifest_sha256,
        )
        expected_binding = build_cache_binding(
            prepass,
            main,
            provenance_assertion_id=provenance.assertion_id,
            provenance_assertion_sha256=provenance.assertion_sha256,
            requested_backend=args.backend,
            used_backend=stored_binding.used_backend,
            backend_reason=canonical_backend_reason(
                args.backend,
                stored_binding.used_backend,
            ),
            core_source_manifest_sha256=core_source_manifest_sha256,
        )
        cached = load_diagnostics_cache(
            args.from_diagnostics,
            expected_binding=expected_binding,
            expected_manifest_sha256=args.diagnostics_manifest_sha256,
        )
        return _products_from_cache(
            cached,
            cache_mode="loaded",
        )

    processed = _process_with_auto_full_run_fallback(
        job,
        requested_backend=args.backend,
    )
    diagnostics = processed.result.diagnostics
    if diagnostics is None:
        raise HybridCLIError(
            "Portable Digital ICE did not return requested diagnostics"
        )
    selection = processed.selection
    binding = build_cache_binding(
        prepass,
        main,
        provenance_assertion_id=provenance.assertion_id,
        provenance_assertion_sha256=provenance.assertion_sha256,
        requested_backend=args.backend,
        used_backend=selection.used.value,
        backend_reason=selection.reason,
        core_source_manifest_sha256=core_source_manifest_sha256,
    )
    cache_manifest_sha256: str | None = None
    cache_mode = "none"
    if args.save_diagnostics is not None:
        if args.save_diagnostics.exists() or args.save_diagnostics.is_symlink():
            raise HybridCLIError(
                f"diagnostics cache directory already exists: {args.save_diagnostics}"
            )
        cached = save_diagnostics_cache(
            args.save_diagnostics,
            processing_result=processed.result,
            binding=binding,
        )
        cache_manifest_sha256 = cached.manifest_sha256
        cache_mode = "saved"
    return _RunProducts(
        output_rgb16=processed.result.output_rgb16,
        diagnostics=diagnostics,
        requested_backend=selection.requested.value,
        used_backend=selection.used.value,
        backend_reason=selection.reason,
        cache_mode=cache_mode,
        cache_manifest_sha256=cache_manifest_sha256,
        cache_binding=binding,
    )


def _products_from_cache(
    cached: CachedDiagnostics,
    *,
    cache_mode: str,
) -> _RunProducts:
    return _RunProducts(
        output_rgb16=cached.output_rgb16,
        diagnostics=cached.diagnostics,
        requested_backend=cached.binding.requested_backend,
        used_backend=cached.binding.used_backend,
        backend_reason=cached.binding.backend_reason,
        cache_mode=cache_mode,
        cache_manifest_sha256=cached.manifest_sha256,
        cache_binding=cached.binding,
    )


def _write_npy(path: Path, array: np.ndarray) -> None:
    with path.open("xb") as handle:
        np.save(handle, array, allow_pickle=False)


@dataclass(frozen=True)
class _RecordedCrop:
    input_rgb8: npt.NDArray[np.uint8]
    component_mask: npt.NDArray[np.uint8]
    inpainted_rgb8: npt.NDArray[np.uint8]


def _raw_array_sha256(array: np.ndarray) -> str:
    """Hash the exact C-order array bytes used by model provenance records."""

    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _require_model_invocation_bindings(
    *,
    adapter: IOPaintLaMaAdapter | None,
    crops: Sequence[_RecordedCrop],
    composite: CompositeResult,
) -> None:
    """Bind model metadata and archived arrays to each composite component."""

    if len(crops) != len(composite.components):
        raise HybridCLIError("model invocation count does not match final components")
    invocations = () if adapter is None else adapter.invocations
    if adapter is not None and len(invocations) != len(composite.components):
        raise HybridCLIError("IOPaint metadata count does not match final components")

    for crop, record in zip(crops, composite.components, strict=True):
        archived_hashes = (
            _raw_array_sha256(crop.input_rgb8),
            _raw_array_sha256(crop.component_mask),
            _raw_array_sha256(crop.inpainted_rgb8),
        )
        composite_hashes = (
            record.input_rgb8_sha256,
            record.component_mask_sha256,
            record.inpainted_rgb8_sha256,
        )
        if archived_hashes != composite_hashes:
            raise HybridCLIError(
                f"archived model arrays do not match composite component "
                f"{record.component_id}"
            )

    for invocation, record in zip(invocations, composite.components, strict=True):
        invocation_hashes = (
            invocation.input_rgb8_raw_sha256,
            invocation.mask_u8_raw_sha256,
            invocation.output_rgb8_raw_sha256,
        )
        composite_hashes = (
            record.input_rgb8_sha256,
            record.component_mask_sha256,
            record.inpainted_rgb8_sha256,
        )
        if invocation_hashes != composite_hashes:
            raise HybridCLIError(
                f"IOPaint metadata does not match composite component "
                f"{record.component_id}"
            )


class _ArchivingInpainter:
    """Capture exact callback arrays while delegating to the isolated model."""

    def __init__(self, adapter: IOPaintLaMaAdapter) -> None:
        self.adapter = adapter
        self.crops: list[_RecordedCrop] = []
        self.elapsed_seconds: list[float] = []

    def _archive(
        self,
        rgb_crop: npt.NDArray[np.uint8],
        component_mask: npt.NDArray[np.uint8],
        generated: npt.NDArray[np.uint8],
    ) -> None:
        self.crops.append(
            _RecordedCrop(
                input_rgb8=np.array(rgb_crop, dtype=np.uint8, order="C", copy=True),
                component_mask=np.array(
                    component_mask,
                    dtype=np.uint8,
                    order="C",
                    copy=True,
                ),
                inpainted_rgb8=np.array(
                    generated,
                    dtype=np.uint8,
                    order="C",
                    copy=True,
                ),
            )
        )

    def __call__(
        self,
        rgb_crop: npt.NDArray[np.uint8],
        component_mask: npt.NDArray[np.uint8],
    ) -> npt.NDArray[np.uint8]:
        started = time.monotonic()
        generated = self.adapter(rgb_crop, component_mask)
        self.elapsed_seconds.append(time.monotonic() - started)
        self._archive(rgb_crop, component_mask, generated)
        return generated

    def inpaint_batch(
        self,
        rgb_crops: Sequence[npt.NDArray[np.uint8]],
        component_masks: Sequence[npt.NDArray[np.uint8]],
    ) -> tuple[npt.NDArray[np.uint8], ...]:
        started = time.monotonic()
        batch_method = getattr(self.adapter, "inpaint_batch", None)
        if callable(batch_method):
            batch_values = batch_method(rgb_crops, component_masks)
            try:
                batch_iterator = iter(batch_values)
            except TypeError as error:
                raise HybridCLIError(
                    "batch model output must be an iterable"
                ) from error
            # Bound consumption at the model/archive boundary.  The
            # compositor performs the same guard independently, but it cannot
            # protect this wrapper from an infinite adapter result.
            outputs = tuple(islice(batch_iterator, len(rgb_crops) + 1))
        else:  # Compatibility for injected test doubles and simple adapters.
            outputs = tuple(
                self.adapter(rgb_crop, component_mask)
                for rgb_crop, component_mask in zip(
                    rgb_crops,
                    component_masks,
                    strict=True,
                )
            )
        self.elapsed_seconds.append(time.monotonic() - started)
        if len(outputs) != len(rgb_crops):
            raise HybridCLIError("batch model output count does not match crop count")
        for rgb_crop, component_mask, generated in zip(
            rgb_crops,
            component_masks,
            outputs,
            strict=True,
        ):
            self._archive(rgb_crop, component_mask, generated)
        return outputs


def _forbidden_empty_inpainter(
    _rgb_crop: npt.NDArray[np.uint8],
    _component_mask: npt.NDArray[np.uint8],
) -> npt.NDArray[np.uint8]:
    raise RuntimeError("empty synthesis unexpectedly invoked the model")


def _build_iopaint_adapter(args: argparse.Namespace) -> IOPaintLaMaAdapter:
    required = {
        "--iopaint-python": args.iopaint_python,
        "--iopaint-executable": args.iopaint_executable,
        "--iopaint-source-manifest-sha256": (args.iopaint_source_manifest_sha256),
        "--model-dir": args.model_dir,
        "--model-weights": args.model_weights,
        "--model-weights-sha256": args.model_weights_sha256,
    }
    missing = [flag for flag, value in required.items() if value is None]
    if missing:
        raise HybridCLIError(
            "non-empty synthesis requires model arguments: " + ", ".join(missing)
        )
    assert args.iopaint_python is not None
    assert args.iopaint_executable is not None
    assert args.iopaint_source_manifest_sha256 is not None
    assert args.model_dir is not None
    assert args.model_weights is not None
    assert args.model_weights_sha256 is not None
    ModelWeightsAttestation(
        sanitized_artifact_id=args.model_artifact_id,
        sha256=args.model_weights_sha256,
        byte_size=1,
        sanitized_external_reference=args.model_external_reference,
    )
    artifact = ModelArtifactAttestation(
        identifier=args.model_artifact_id,
        sha256=args.model_weights_sha256,
    )
    config = IOPaintLaMaConfig(
        iopaint_executable=args.iopaint_executable,
        python_executable=args.iopaint_python,
        model_dir=args.model_dir,
        weights_file=args.model_weights,
        artifact=artifact,
        expected_iopaint_source_manifest_sha256=(args.iopaint_source_manifest_sha256),
        device=args.inpaint_device,
        thread_count=args.inpaint_threads,
        seed=args.inpaint_seed,
        timeout_seconds=args.inpaint_timeout,
    )
    return IOPaintLaMaAdapter(config)


def _processing_result_for_cache(products: _RunProducts) -> ProcessingResult:
    output_sha256 = hash_rgb16(products.output_rgb16)
    return ProcessingResult(
        output_rgb16=products.output_rgb16,
        replay=SimpleNamespace(output_sha256=output_sha256),
        profile_id=DEFAULT_PROFILE.profile_id,
        diagnostics=products.diagnostics,
    )


def _materialize_embedded_cache(
    destination: Path,
    *,
    products: _RunProducts,
) -> CachedDiagnostics:
    cached = save_diagnostics_cache(
        destination,
        processing_result=_processing_result_for_cache(products),
        binding=products.cache_binding,
    )
    return load_diagnostics_cache(
        destination,
        expected_binding=cached.binding,
        expected_manifest_sha256=cached.manifest_sha256,
    )


def _write_component_png(path: Path, array: npt.NDArray[np.uint8]) -> None:
    try:
        iio.imwrite(
            path,
            array,
            extension=".png",
            plugin="pillow",
            compress_level=9,
        )
    except (OSError, ValueError) as error:
        raise HybridCLIError(f"cannot write component evidence PNG: {error}") from error


def _inpainting_metadata(
    args: argparse.Namespace,
    *,
    adapter: IOPaintLaMaAdapter,
) -> InpaintingMetadata:
    invocation = adapter.last_invocation
    if invocation is None:
        raise HybridCLIError("IOPaint produced no successful invocation metadata")
    runtime = invocation.runtime
    assert args.model_weights is not None
    measured = measure_model_artifact(
        args.model_weights,
        identifier=args.model_artifact_id,
    )
    if measured.sha256 != runtime.model_weights_sha256:
        raise HybridCLIError("model weights changed after the inpainting run")
    try:
        byte_size = args.model_weights.stat().st_size
    except OSError:
        raise HybridCLIError("cannot stat the attested model weights") from None
    return InpaintingMetadata(
        iopaint_version=runtime.tool_version,
        entrypoint="python -I -m iopaint run",
        tool_license_spdx=runtime.tool_license_spdx,
        iopaint_source_manifest_sha256=runtime.iopaint_source_manifest_sha256,
        iopaint_source_file_count=runtime.iopaint_source_file_count,
        effective_environment_sha256=runtime.effective_environment_sha256,
        model_id=runtime.model_name,
        model_version=runtime.model_release,
        model_weights=ModelWeightsAttestation(
            sanitized_artifact_id=runtime.model_artifact_identifier,
            sha256=runtime.model_weights_sha256,
            byte_size=byte_size,
            sanitized_external_reference=args.model_external_reference,
        ),
        model_upstream_license_spdx=runtime.model_upstream_license_spdx,
        model_artifact_license_status=runtime.model_artifact_license_status,
        torch_version=runtime.torch_version,
        python_version=runtime.python_version,
        python_implementation=runtime.python_implementation,
        numpy_version=runtime.numpy_version,
        pillow_version=runtime.pillow_version,
        opencv_version=runtime.opencv_version,
        pydantic_version=runtime.pydantic_version,
        typer_version=runtime.typer_version,
        platform_system=runtime.platform_system,
        platform_release=runtime.platform_release,
        platform_machine=runtime.platform_machine,
        device=runtime.device,
        threads=runtime.thread_count,
        seed=runtime.seed,
        seed_scope=runtime.seed_scope,
        argv=invocation.sanitized_argv,
        deterministic_algorithms_enabled=runtime.deterministic_algorithms,
        cudnn_benchmark=runtime.cudnn_benchmark,
        cuda_available=runtime.cuda_available,
        cuda_runtime_version=runtime.cuda_runtime_version,
        cudnn_version=runtime.cudnn_version,
        cuda_device_names=runtime.cuda_device_names,
        cuda_visible_devices=runtime.cuda_visible_devices,
        hip_runtime_version=runtime.hip_runtime_version,
        mps_available=runtime.mps_available,
        mps_device_name=runtime.mps_device_name,
        repeat_runs=1,
        repeatability_observed=False,
        determinism_scope=runtime.determinism_scope,
    )


def _component_artifacts(
    root: Path,
    *,
    composite: CompositeResult,
    crops: Sequence[_RecordedCrop],
) -> tuple[list[ArtifactSource], list[ComponentArtifactBinding]]:
    records = getattr(composite, "components", ())
    if len(records) != len(crops):
        raise HybridCLIError("component evidence count does not match composite")
    if not records:
        return [], []
    component_directory = root / "components"
    component_directory.mkdir(mode=0o700)
    sources: list[ArtifactSource] = []
    bindings: list[ComponentArtifactBinding] = []
    for record, crop in zip(records, crops, strict=True):
        component_id = int(record.component_id)
        prefix = f"component-{component_id:04d}"
        paths = {
            "input": component_directory / f"{component_id:04d}-input.png",
            "mask": component_directory / f"{component_id:04d}-mask.png",
            "inpainted": component_directory / f"{component_id:04d}-inpainted.png",
        }
        _write_component_png(paths["input"], crop.input_rgb8)
        _write_component_png(paths["mask"], crop.component_mask)
        _write_component_png(paths["inpainted"], crop.inpainted_rgb8)
        input_id = f"{prefix}-input"
        mask_id = f"{prefix}-mask"
        inpainted_id = f"{prefix}-inpainted"
        sources.extend(
            (
                ArtifactSource(
                    input_id,
                    "component_input_rgb8",
                    f"components/{component_id:04d}-input.png",
                    "image/png",
                    RawEncoding.PNG_U8,
                ),
                ArtifactSource(
                    mask_id,
                    "component_mask_png",
                    f"components/{component_id:04d}-mask.png",
                    "image/png",
                    RawEncoding.PNG_U8,
                ),
                ArtifactSource(
                    inpainted_id,
                    "component_inpainted_rgb8",
                    f"components/{component_id:04d}-inpainted.png",
                    "image/png",
                    RawEncoding.PNG_U8,
                ),
            )
        )
        bindings.append(
            ComponentArtifactBinding(
                component_id=component_id,
                input_rgb8_artifact_id=input_id,
                mask_artifact_id=mask_id,
                inpainted_rgb8_artifact_id=inpainted_id,
            )
        )
    return sources, bindings


def _hybrid_run_metadata(
    base: dict[str, object],
    *,
    embedded_cache: CachedDiagnostics,
    products: _RunProducts,
    composite: CompositeResult,
    mask_png_sha256: str,
    adapter: IOPaintLaMaAdapter | None,
    archiver: _ArchivingInpainter | None,
    routing_seconds: float,
    composite_seconds: float,
) -> dict[str, object]:
    metadata = dict(base)
    metadata["artifacts"] = {
        **dict(base["artifacts"]),
        "diagnostics_cache": {
            "directory": "diagnostics-cache",
            "manifest_filename": MANIFEST_FILENAME,
            "manifest_sha256": embedded_cache.manifest_sha256,
        },
        "hybrid_output_rgb16": {
            "filename": "output-hybrid.rgb16.npy",
            "raw_sha256": hash_rgb16(composite.hybrid_rgb16),
        },
        "synthesis_mask": {
            "filename": "synth-mask.png",
            "sha256": mask_png_sha256,
        },
    }
    metadata["cache"] = {
        "embedded_manifest_sha256": embedded_cache.manifest_sha256,
        "external_manifest_sha256": products.cache_manifest_sha256,
        "external_mode": products.cache_mode,
    }
    metadata["generative_model_loaded"] = adapter is not None
    metadata["mode"] = (
        "hybrid_empty_synthesis" if adapter is None else "hybrid_lama_inpaint"
    )
    metadata["timing"] = {
        "composite_and_inpainting_seconds": composite_seconds,
        "iopaint_batch_seconds": [] if archiver is None else archiver.elapsed_seconds,
        "routing_seconds": routing_seconds,
    }
    if adapter is None:
        metadata["inpainting"] = {
            "invoked": False,
            "reason": "empty_synthesis_mask",
        }
    else:
        metadata["inpainting"] = {
            "invocation_count": len(adapter.invocations),
            "invocations": [
                {
                    "config": dict(invocation.config_document),
                    "deterministic_environment": dict(
                        invocation.deterministic_environment
                    ),
                    "input_rgb8_raw_sha256": invocation.input_rgb8_raw_sha256,
                    "mask_u8_raw_sha256": invocation.mask_u8_raw_sha256,
                    "output_rgb8_raw_sha256": invocation.output_rgb8_raw_sha256,
                    "sanitized_argv": list(invocation.sanitized_argv),
                }
                for invocation in adapter.invocations
            ],
            "invoked": True,
            "runtime": asdict(adapter.invocations[0].runtime),
        }
    return metadata


def _create_staging_directory(destination: Path) -> Path:
    try:
        return Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.",
                dir=destination.parent,
            )
        )
    except OSError:
        raise HybridCLIError(
            "cannot create the private output staging directory"
        ) from None


def _write_output_directory(
    destination: Path,
    *,
    output_rgb16: npt.NDArray[np.uint16],
    routing_document: dict[str, object],
    metadata_document: dict[str, object],
    core_source_manifest_sha256: str,
    hybrid_source_manifest_sha256: str,
) -> None:
    _require_new_output_directory(destination)
    temporary = _create_staging_directory(destination)
    renamed = False
    try:
        _write_npy(temporary / "output.rgb16.npy", output_rgb16)
        (temporary / "routing.json").write_bytes(canonical_json_bytes(routing_document))
        (temporary / "run-metadata.json").write_bytes(
            canonical_json_bytes(metadata_document)
        )
        _require_source_manifests_unchanged(
            core_source_manifest_sha256=core_source_manifest_sha256,
            hybrid_source_manifest_sha256=hybrid_source_manifest_sha256,
        )
        _atomic_publish_directory(temporary, destination)
        renamed = True
    except OSError as error:
        raise HybridCLIError(
            f"cannot publish output directory {destination}: {error}"
        ) from error
    finally:
        if not renamed:
            shutil.rmtree(temporary, ignore_errors=True)


def _write_hybrid_output_directory(
    destination: Path,
    *,
    args: argparse.Namespace,
    prepass: npt.NDArray[np.uint16],
    main: npt.NDArray[np.uint16],
    prepass_raw_sha256: str,
    main_raw_sha256: str,
    provenance: _Provenance,
    products: _RunProducts,
    routing: RoutingResult,
    routing_document: dict[str, object],
    metadata_document: dict[str, object],
    routing_seconds: float,
    core_source_manifest_sha256: str,
    hybrid_source_manifest_sha256: str,
) -> None:
    routing.require_safe_for_inpainting()
    adapter: IOPaintLaMaAdapter | None = None
    archiver: _ArchivingInpainter | None = None
    if routing.synthesis_pixel_count == 0:
        inpainter = _forbidden_empty_inpainter
    else:
        adapter = _build_iopaint_adapter(args)
        archiver = _ArchivingInpainter(adapter)
        inpainter = archiver

    composite_started = time.monotonic()
    composite = composite_components(
        products.output_rgb16,
        routing.final_labels,
        routing.synthesis_mask,
        inpainter,
        crop_margin=128,
    )
    composite_seconds = time.monotonic() - composite_started
    if routing.synthesis_pixel_count != int(np.count_nonzero(routing.synthesis_mask)):
        raise HybridCLIError("routing synthesis count does not equal the final mask")
    if not np.array_equal(
        composite.hybrid_rgb16[~routing.synthesis_mask],
        products.output_rgb16[~routing.synthesis_mask],
    ):
        raise HybridCLIError("hybrid output changed a pixel outside the mask")

    crops: Sequence[_RecordedCrop] = () if archiver is None else archiver.crops
    _require_model_invocation_bindings(
        adapter=adapter,
        crops=crops,
        composite=composite,
    )

    _require_new_output_directory(destination)
    temporary = _create_staging_directory(destination)
    renamed = False
    try:
        embedded_cache = _materialize_embedded_cache(
            temporary / "diagnostics-cache",
            products=products,
        )
        embedded_pure = temporary / "diagnostics-cache" / "output.rgb16.npy"
        shutil.copy2(embedded_pure, temporary / "output.rgb16.npy")
        routing_bytes = canonical_json_bytes(routing_document)
        (temporary / "routing.json").write_bytes(routing_bytes)
        _write_npy(temporary / "output-hybrid.rgb16.npy", composite.hybrid_rgb16)
        mask_png_sha256 = write_synthesis_mask_png(
            temporary / "synth-mask.png",
            routing.synthesis_mask,
        )
        component_sources, component_bindings = _component_artifacts(
            temporary,
            composite=composite,
            crops=crops,
        )
        metadata = _hybrid_run_metadata(
            metadata_document,
            embedded_cache=embedded_cache,
            products=products,
            composite=composite,
            mask_png_sha256=mask_png_sha256,
            adapter=adapter,
            archiver=archiver,
            routing_seconds=routing_seconds,
            composite_seconds=composite_seconds,
        )
        (temporary / "run-metadata.json").write_bytes(canonical_json_bytes(metadata))

        artifact_sources = [
            ArtifactSource(
                "pure-output",
                "pure_output_rgb16",
                "output.rgb16.npy",
                "application/x-npy",
                RawEncoding.NPY_ARRAY,
            ),
            ArtifactSource(
                "hybrid-output",
                "hybrid_output_rgb16",
                "output-hybrid.rgb16.npy",
                "application/x-npy",
                RawEncoding.NPY_ARRAY,
            ),
            ArtifactSource(
                "synthesis-mask",
                "synthesis_mask_png",
                "synth-mask.png",
                "image/png",
                RawEncoding.PNG_U8,
            ),
            ArtifactSource(
                "diagnostics-cache",
                "diagnostics_cache_manifest",
                f"diagnostics-cache/{MANIFEST_FILENAME}",
                "application/json",
                RawEncoding.OPAQUE_BYTES,
            ),
            ArtifactSource(
                "routing",
                "routing_json",
                "routing.json",
                "application/json",
                RawEncoding.OPAQUE_BYTES,
            ),
            ArtifactSource(
                "run-metadata",
                "run_metadata_json",
                "run-metadata.json",
                "application/json",
                RawEncoding.OPAQUE_BYTES,
            ),
            *component_sources,
        ]
        inpainting = (
            None if adapter is None else _inpainting_metadata(args, adapter=adapter)
        )
        receipt_document = build_receipt_document(
            artifact_root=temporary,
            generated_at_utc=datetime.now(timezone.utc),
            hybrid_version=current_hybrid_version(),
            hybrid_source_manifest_sha256=hybrid_source_manifest_sha256,
            inputs=InputProvenance(
                prepass_raw_sha256=prepass_raw_sha256,
                prepass_shape=tuple(int(value) for value in prepass.shape),
                main_raw_sha256=main_raw_sha256,
                main_shape=tuple(int(value) for value in main.shape),
                same_frame_assertion_id=provenance.assertion_id,
                focus_exposure_assertion_id=(
                    f"{provenance.assertion_id}:focus-exposure-locked"
                ),
                source_manifest_sha256=provenance.manifest_sha256,
            ),
            core=CoreRunMetadata(
                version=products.cache_binding.core_version,
                source_manifest_sha256=(
                    products.cache_binding.core_source_manifest_sha256
                ),
                profile_id=products.cache_binding.profile_id,
                requested_backend=products.requested_backend,
                used_backend=products.used_backend,
                backend_reason=products.backend_reason,
                diagnostics_backend=products.used_backend,
            ),
            diagnostics=products.diagnostics,
            routing=routing,
            composite=composite,
            changed_mask=products.diagnostics.changed_mask,
            crop_margin=128,
            inpainting=inpainting,
            artifacts=artifact_sources,
            component_artifacts=component_bindings,
        )
        receipt_sha256 = write_receipt(
            temporary / RECEIPT_FILENAME,
            receipt_document,
        )
        resolver = None
        if adapter is not None:
            assert args.model_weights is not None

            def resolve_model_weights(
                _attestation: ModelWeightsAttestation,
            ) -> Path:
                assert args.model_weights is not None
                return args.model_weights

            resolver = resolve_model_weights
        verified = verify_receipt(
            temporary / RECEIPT_FILENAME,
            model_weights_resolver=resolver,
            require_model_weights=adapter is not None,
        )
        if verified.receipt_sha256 != receipt_sha256:
            raise HybridCLIError("receipt hash changed during verification")

        _require_source_manifests_unchanged(
            core_source_manifest_sha256=core_source_manifest_sha256,
            hybrid_source_manifest_sha256=hybrid_source_manifest_sha256,
        )
        _atomic_publish_directory(temporary, destination)
        renamed = True
    except OSError as error:
        raise HybridCLIError(
            f"cannot publish hybrid output directory {destination}: {error}"
        ) from error
    finally:
        if not renamed:
            shutil.rmtree(temporary, ignore_errors=True)


def _execute(args: argparse.Namespace) -> None:
    if args.from_diagnostics is not None and args.diagnostics_manifest_sha256 is None:
        raise HybridCLIError(
            "--from-diagnostics requires --diagnostics-manifest-sha256"
        )
    if args.from_diagnostics is None and args.diagnostics_manifest_sha256 is not None:
        raise HybridCLIError(
            "--diagnostics-manifest-sha256 requires --from-diagnostics"
        )
    _require_new_output_directory(args.out)
    core_source_manifest_sha256 = compute_core_source_manifest()
    hybrid_source_manifest_sha256 = compute_hybrid_source_manifest()
    cache_path = args.save_diagnostics or args.from_diagnostics
    _validate_cache_path_relationships(args.out, cache_path)
    if args.from_diagnostics is not None and not args.from_diagnostics.is_dir():
        raise HybridCLIError(
            f"diagnostics cache directory does not exist: {args.from_diagnostics}"
        )
    if args.save_diagnostics is not None and (
        args.save_diagnostics.exists() or args.save_diagnostics.is_symlink()
    ):
        raise HybridCLIError(
            f"diagnostics cache directory already exists: {args.save_diagnostics}"
        )
    policy = RoutingPolicy(
        min_area=args.min_area,
        min_radius=args.min_radius,
        margin=args.margin,
        max_synth_fraction=args.max_synth_fraction,
    )

    prepass = _load_rgbi16(args.prepass, role="prepass")
    main = _load_rgbi16(args.main, role="main")
    prepass_raw_sha256 = hash_rgbir16(prepass)
    main_raw_sha256 = hash_rgbir16(main)
    provenance = _build_provenance(
        same_frame_id=args.same_frame_id,
        prepass_raw_sha256=prepass_raw_sha256,
        main_raw_sha256=main_raw_sha256,
        acquisition_manifest=args.acquisition_manifest,
    )
    job = _build_job(
        prepass,
        main,
        same_frame_id=args.same_frame_id,
        prepass_raw_sha256=prepass_raw_sha256,
        main_raw_sha256=main_raw_sha256,
    )
    products = _obtain_run_products(
        args=args,
        prepass=prepass,
        main=main,
        job=job,
        provenance=provenance,
        core_source_manifest_sha256=core_source_manifest_sha256,
    )
    routing_started = time.monotonic()
    routing = route_at_floor_mask(products.diagnostics.at_floor_mask, policy)
    routing_seconds = time.monotonic() - routing_started
    routing_document = routing_json_document(routing)
    routing_bytes = canonical_json_bytes(routing_document)
    output_sha256 = hash_rgb16(products.output_rgb16)
    metadata_document: dict[str, object] = {
        "artifacts": {
            "output_rgb16": {
                "filename": "output.rgb16.npy",
                "raw_sha256": output_sha256,
                "source": "portable_digital_ice.ProcessingResult.output_rgb16",
            },
            "routing": {
                "filename": "routing.json",
                "sha256": _sha256(routing_bytes),
            },
        },
        "backend": {
            "reason": products.backend_reason,
            "requested": products.requested_backend,
            "used": products.used_backend,
        },
        "cache": {
            "manifest_sha256": products.cache_manifest_sha256,
            "mode": products.cache_mode,
        },
        "inputs": {
            "main": {
                "raw_sha256": main_raw_sha256,
                "shape": list(main.shape),
            },
            "prepass": {
                "raw_sha256": prepass_raw_sha256,
                "shape": list(prepass.shape),
            },
        },
        "generative_model_loaded": False,
        "mode": ("routing_only_no_inpaint" if args.no_inpaint else "hybrid_pending"),
        "profile": {
            "bit_depth": 16,
            "id": DEFAULT_PROFILE.profile_id,
            "main_dpi": 4_000,
            "mode": "normal",
            "prepass_dpi": 285,
            "resolution_metric": 4_000,
            "scanner_model": "nikon-super-coolscan-5000-ed",
            "selector": 8,
        },
        "provenance": {
            "acquisition_manifest_sha256": provenance.manifest_sha256,
            "assertion": provenance.document,
            "assertion_id": provenance.assertion_id,
            "assertion_sha256": provenance.assertion_sha256,
            "classification": "caller_asserted_bare_npy",
        },
        "routing": {
            "at_floor_pixels": routing.at_floor_pixel_count,
            "final_regions": len(routing.final_regions),
            "synthesis_fraction": routing.synthesis_fraction,
            "synthesis_pixels": routing.synthesis_pixel_count,
            "within_synthesis_budget": routing.within_synthesis_budget,
        },
        "schema": "fauxce-hybrid-run-metadata-v1",
        "source_manifests": {
            "core": core_source_manifest_sha256,
            "hybrid": hybrid_source_manifest_sha256,
        },
    }
    if args.no_inpaint:
        _write_output_directory(
            args.out,
            output_rgb16=products.output_rgb16,
            routing_document=routing_document,
            metadata_document=metadata_document,
            core_source_manifest_sha256=core_source_manifest_sha256,
            hybrid_source_manifest_sha256=hybrid_source_manifest_sha256,
        )
    else:
        _write_hybrid_output_directory(
            args.out,
            args=args,
            prepass=prepass,
            main=main,
            prepass_raw_sha256=prepass_raw_sha256,
            main_raw_sha256=main_raw_sha256,
            provenance=provenance,
            products=products,
            routing=routing,
            routing_document=routing_document,
            metadata_document=metadata_document,
            routing_seconds=routing_seconds,
            core_source_manifest_sha256=core_source_manifest_sha256,
            hybrid_source_manifest_sha256=hybrid_source_manifest_sha256,
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the routing-only or receipt-bound hybrid CLI."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        _execute(args)
    except (
        CpuFastUnavailable,
        CudaBackendUnavailable,
        DiagnosticsCacheError,
        HybridCLIError,
        IOPaintError,
        NoContextPixelsError,
        NoHealthyContextError,
        ReceiptError,
        SynthesisBudgetExceeded,
        TypeError,
        ValueError,
    ) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through entry point
    sys.exit(main())
