from __future__ import annotations

import hashlib
import json
import os
import zlib
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import imageio.v3 as iio
import numpy as np
import pytest
from portable_digital_ice import (
    DEFAULT_PROFILE,
    ProcessingDiagnostics,
    ProcessingResult,
)

from fauxce_hybrid import receipts as receipts_module
from fauxce_hybrid.cache import (
    build_cache_binding,
    canonical_json_bytes,
    hash_rgb16,
    hash_rgbir16,
    save_diagnostics_cache,
)
from fauxce_hybrid.composite import composite_components
from fauxce_hybrid.receipts import (
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
    canonical_receipt_bytes,
    compute_hybrid_source_manifest,
    current_hybrid_version,
    load_receipt_schema,
    validate_receipt_document,
    verify_receipt,
    write_receipt,
    write_synthesis_mask_png,
)
from fauxce_hybrid.routing import (
    RoutingPolicy,
    route_at_floor_mask,
    routing_json_document,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _raw_sha256(array: np.ndarray) -> str:
    return _sha256(np.ascontiguousarray(array).tobytes(order="C"))


@dataclass(frozen=True)
class _Sample:
    root: Path
    receipt_path: Path
    document: dict[str, object]
    artifact_paths: dict[str, Path]
    pure: np.ndarray
    hybrid: np.ndarray
    mask: np.ndarray


def _save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)


def _write_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, array, extension=".png")


def _rewrite_png_ihdr_dimensions(path: Path, *, width: int, height: int) -> None:
    payload = bytearray(path.read_bytes())
    assert payload[:8] == b"\x89PNG\r\n\x1a\n"
    assert payload[12:16] == b"IHDR"
    payload[16:20] = width.to_bytes(4, "big")
    payload[20:24] = height.to_bytes(4, "big")
    payload[29:33] = (zlib.crc32(payload[12:29]) & 0xFFFFFFFF).to_bytes(4, "big")
    path.write_bytes(payload)


def _rehash_artifact_row(
    receipt_document: dict[str, object],
    *,
    artifact_id: str,
    path: Path,
    raw_sha256: str | None = None,
) -> None:
    row = next(
        artifact
        for artifact in receipt_document["artifacts"]
        if artifact["id"] == artifact_id
    )
    file_sha256 = _sha256(path.read_bytes())
    row["file_sha256"] = file_sha256
    row["raw_sha256"] = file_sha256 if raw_sha256 is None else raw_sha256


def _environment_sha256(values: dict[str, object]) -> str:
    model_weights = values["model_weights"]
    assert isinstance(model_weights, ModelWeightsAttestation)
    document = {
        "cudnn_benchmark": values["cudnn_benchmark"],
        "cuda_available": values["cuda_available"],
        "cuda_device_names": list(values["cuda_device_names"]),
        "cuda_runtime_version": values["cuda_runtime_version"],
        "cuda_visible_devices": values["cuda_visible_devices"],
        "cudnn_version": values["cudnn_version"],
        "deterministic_algorithms": values["deterministic_algorithms_enabled"],
        "device": values["device"],
        "iopaint_source_manifest_sha256": values["iopaint_source_manifest_sha256"],
        "iopaint_version": values["iopaint_version"],
        "hip_runtime_version": values["hip_runtime_version"],
        "model_weights_sha256": model_weights.sha256,
        "mps_available": values["mps_available"],
        "mps_device_name": values["mps_device_name"],
        "numpy_version": values["numpy_version"],
        "opencv_version": values["opencv_version"],
        "pillow_version": values["pillow_version"],
        "platform_machine": values["platform_machine"],
        "platform_release": values["platform_release"],
        "platform_system": values["platform_system"],
        "pydantic_version": values["pydantic_version"],
        "python_implementation": values["python_implementation"],
        "python_version": values["python_version"],
        "seed": values["seed"],
        "thread_count": values["threads"],
        "torch_version": values["torch_version"],
        "typer_version": values["typer_version"],
    }
    return _sha256(
        json.dumps(
            document,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    )


def _inpainting_metadata(**overrides: object) -> InpaintingMetadata:
    weights = b"synthetic deterministic weights\x00\x01"
    values: dict[str, object] = {
        "iopaint_version": "1.6.0",
        "entrypoint": "python -I -m iopaint run",
        "tool_license_spdx": "Apache-2.0",
        "iopaint_source_manifest_sha256": "b" * 64,
        "iopaint_source_file_count": 123,
        "model_id": "lama",
        "model_version": "Sanster/models:add_big_lama",
        "model_weights": ModelWeightsAttestation(
            sanitized_artifact_id="lama-weights",
            sha256=_sha256(weights),
            byte_size=len(weights),
            sanitized_external_reference="hf://models/big-lama",
        ),
        "model_upstream_license_spdx": "Apache-2.0",
        "model_artifact_license_status": (
            "upstream LaMa repository is Apache-2.0; the exact converted IOPaint "
            "weight release does not state a separate artifact license"
        ),
        "torch_version": "2.9.0+cu130",
        "python_version": "3.11.15",
        "python_implementation": "CPython",
        "numpy_version": "1.26.4",
        "pillow_version": "9.5.0",
        "opencv_version": "4.11.0.86",
        "pydantic_version": "2.13.4",
        "typer_version": "0.27.0",
        "platform_system": "Linux",
        "platform_release": "6.12.0",
        "platform_machine": "x86_64",
        "device": "cuda",
        "threads": 1,
        "seed": 20260716,
        "seed_scope": (
            "fixed IOPaint request; LaMa is feed-forward and does not expose a "
            "sampling seed"
        ),
        "argv": (
            "python",
            "-I",
            "-m",
            "iopaint",
            "run",
            "--model",
            "lama",
            "--device",
            "cuda",
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
        ),
        "deterministic_algorithms_enabled": True,
        "cudnn_benchmark": False,
        "cuda_available": True,
        "cuda_runtime_version": "13.0",
        "cudnn_version": "91002",
        "cuda_device_names": ("NVIDIA RTX 5090",),
        "cuda_visible_devices": "0",
        "hip_runtime_version": None,
        "mps_available": False,
        "mps_device_name": None,
        "repeat_runs": 2,
        "repeatability_observed": True,
        "determinism_scope": (
            "repeatability claim is limited to the recorded IOPaint, torch, Python, "
            "device, thread count, weights, deterministic-algorithm and cuDNN "
            "benchmark states, and host environment"
        ),
    }
    values.update(overrides)
    if "effective_environment_sha256" not in overrides:
        values["effective_environment_sha256"] = _environment_sha256(values)
    return InpaintingMetadata(**values)  # type: ignore[arg-type]


def _operational_runtime(metadata_value: InpaintingMetadata) -> dict[str, object]:
    weights = metadata_value.model_weights
    return {
        "tool_name": "IOPaint",
        "tool_version": metadata_value.iopaint_version,
        "tool_license_spdx": metadata_value.tool_license_spdx,
        "iopaint_source_manifest_sha256": (
            metadata_value.iopaint_source_manifest_sha256
        ),
        "iopaint_source_file_count": metadata_value.iopaint_source_file_count,
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
        "deterministic_algorithms": (metadata_value.deterministic_algorithms_enabled),
        "cudnn_benchmark": metadata_value.cudnn_benchmark,
        "cuda_available": metadata_value.cuda_available,
        "cuda_runtime_version": metadata_value.cuda_runtime_version,
        "cudnn_version": metadata_value.cudnn_version,
        "hip_runtime_version": metadata_value.hip_runtime_version,
        "cuda_device_names": list(metadata_value.cuda_device_names),
        "cuda_visible_devices": metadata_value.cuda_visible_devices,
        "mps_available": metadata_value.mps_available,
        "mps_device_name": metadata_value.mps_device_name,
        "effective_environment_sha256": (metadata_value.effective_environment_sha256),
        "model_name": metadata_value.model_id,
        "model_release": metadata_value.model_version,
        "model_artifact_identifier": weights.sanitized_artifact_id,
        "model_weights_sha256": weights.sha256,
        "model_upstream_license_spdx": metadata_value.model_upstream_license_spdx,
        "model_artifact_license_status": (metadata_value.model_artifact_license_status),
        "device": metadata_value.device,
        "thread_count": metadata_value.threads,
        "seed": metadata_value.seed,
        "seed_scope": metadata_value.seed_scope,
        "determinism_scope": metadata_value.determinism_scope,
    }


def _operational_environment(
    metadata_value: InpaintingMetadata,
) -> dict[str, str]:
    threads = str(metadata_value.threads)
    environment = {
        "DIFFUSERS_CACHE": "<private-model-cache>/huggingface/hub",
        "HF_HUB_OFFLINE": "1",
        "HF_HOME": "<private-model-cache>/huggingface",
        "HF_HUB_CACHE": "<private-model-cache>/huggingface/hub",
        "HUGGINGFACE_HUB_CACHE": "<private-model-cache>/huggingface/hub",
        "LAMA_MODEL_URL": "<private-model-artifact>",
        "MKL_DYNAMIC": "FALSE",
        "MKL_NUM_THREADS": threads,
        "NUMEXPR_NUM_THREADS": threads,
        "OMP_DYNAMIC": "FALSE",
        "OMP_NUM_THREADS": threads,
        "OPENBLAS_NUM_THREADS": threads,
        "TORCH_HOME": "<private-model-cache>/torch",
        "TRANSFORMERS_CACHE": "<private-model-cache>/huggingface/hub",
        "TRANSFORMERS_OFFLINE": "1",
        "VECLIB_MAXIMUM_THREADS": threads,
        "XDG_CACHE_HOME": "<private-model-cache>",
    }
    if metadata_value.device == "cuda":
        environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return environment


def _build_sample(
    tmp_path: Path,
    *,
    include_perimeter_exclusion: bool = False,
    empty_synthesis: bool = False,
    cuda_pure_cpu_diagnostics: bool = False,
) -> _Sample:
    assert not (include_perimeter_exclusion and empty_synthesis)
    root = tmp_path / "run"
    root.mkdir()
    height, width = 24, 28
    rows, columns = np.indices((height, width))
    pure = np.stack(
        (
            rows * 900 + columns * 17 + 300,
            rows * 500 + columns * 31 + 1_000,
            rows * 250 + columns * 47 + 2_000,
        ),
        axis=2,
    ).astype(np.uint16)

    at_floor = np.zeros((height, width), dtype=bool)
    if empty_synthesis:
        pass
    elif include_perimeter_exclusion:
        at_floor[0, :] = True
        at_floor[-1, :] = True
        at_floor[:, 0] = True
        at_floor[:, -1] = True
        at_floor[2:7, 10:16] = True
    else:
        at_floor[8:13, 10:16] = True
    routing = route_at_floor_mask(
        at_floor,
        RoutingPolicy(
            min_area=1,
            min_radius=100,
            margin=2 if include_perimeter_exclusion else 0,
            max_synth_fraction=1.0,
        ),
    )
    captured: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def deterministic_inpainter(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        generated = np.ascontiguousarray(255 - rgb, dtype=np.uint8)
        captured.append((rgb.copy(), mask.copy(), generated.copy()))
        return generated

    composite = composite_components(
        pure,
        routing.final_labels,
        routing.synthesis_mask,
        deterministic_inpainter,
        crop_margin=4,
    )
    expected_components = 0 if empty_synthesis else 1
    assert len(captured) == len(composite.components) == expected_components

    floor = np.float32(0.02)
    score = np.full((height, width), np.float32(0.08), dtype=np.float32)
    score[at_floor] = floor
    changed = np.zeros((height, width), dtype=bool)
    if not empty_synthesis:
        changed[9:12, 11:15] = True
    diagnostics = ProcessingDiagnostics(
        score_plane=score,
        score_floor=floor,
        at_floor_mask=at_floor,
        changed_mask=changed,
    )
    prepass = np.arange(4 * 5 * 4, dtype=np.uint16).reshape(4, 5, 4)
    main = np.arange(height * width * 4, dtype=np.uint16).reshape(
        height,
        width,
        4,
    )
    source_manifest_sha256 = _sha256(b"input manifest")
    assertion_document = {
        "acquisition_manifest": {
            "file_sha256": source_manifest_sha256,
            "provided": True,
            "validated_claims": [
                "same_frame_id",
                "focus_exposure_locked",
                "prepass_raw_sha256",
                "main_raw_sha256",
            ],
        },
        "assertions": {
            "focus_exposure_locked": True,
            "same_frame_id": "scan-0007",
        },
        "inputs": {
            "main": {"raw_sha256": hash_rgbir16(main)},
            "prepass": {"raw_sha256": hash_rgbir16(prepass)},
        },
        "provenance_class": "caller_asserted_bare_npy",
        "scanner_evidence": False,
        "schema": "fauxce-hybrid-caller-assertion-v1",
    }
    assertion_sha256 = _sha256(canonical_json_bytes(assertion_document))
    assertion_id = f"caller-asserted-bare-npy:{assertion_sha256[:16]}"
    core_used_backend = "cuda" if cuda_pure_cpu_diagnostics else "cpu"
    core_backend_reason = (
        "startup self-test passed byte parity"
        if cuda_pure_cpu_diagnostics
        else "CUDA unavailable; complete job ran on exact CPU reference"
    )
    binding = build_cache_binding(
        prepass,
        main,
        provenance_assertion_id=assertion_id,
        provenance_assertion_sha256=assertion_sha256,
        requested_backend="auto",
        used_backend="cpu",
        backend_reason="CUDA unavailable; complete job ran on exact CPU reference",
    )

    artifact_paths = {
        "pure-output": root / "output.rgb16.npy",
        "hybrid-output": root / "output-hybrid.rgb16.npy",
        "synthesis-mask": root / "synth-mask.png",
        "diagnostics-cache": (root / "diagnostics-cache" / "diagnostics-cache.json"),
        "routing": root / "routing.json",
        "run-metadata": root / "run-metadata.json",
    }
    if not empty_synthesis:
        artifact_paths.update(
            {
                "component-1-input": root / "components" / "0001-input.png",
                "component-1-mask": root / "components" / "0001-mask.png",
                "component-1-inpainted": (root / "components" / "0001-inpainted.png"),
            }
        )
    _save_npy(artifact_paths["pure-output"], pure)
    _save_npy(artifact_paths["hybrid-output"], composite.hybrid_rgb16)
    mask_png_sha256 = write_synthesis_mask_png(
        artifact_paths["synthesis-mask"],
        routing.synthesis_mask,
    )
    cached = save_diagnostics_cache(
        artifact_paths["diagnostics-cache"].parent,
        processing_result=ProcessingResult(
            output_rgb16=pure,
            replay=SimpleNamespace(output_sha256=hash_rgb16(pure)),
            profile_id=DEFAULT_PROFILE.profile_id,
            diagnostics=diagnostics,
        ),
        binding=binding,
    )
    routing_document = routing_json_document(routing)
    routing_bytes = canonical_json_bytes(routing_document)
    artifact_paths["routing"].write_bytes(routing_bytes)
    if not empty_synthesis:
        input_rgb8, component_mask, inpainted_rgb8 = captured[0]
        _write_png(artifact_paths["component-1-input"], input_rgb8)
        _write_png(artifact_paths["component-1-mask"], component_mask)
        _write_png(artifact_paths["component-1-inpainted"], inpainted_rgb8)

    hybrid_source_manifest_sha256 = compute_hybrid_source_manifest()
    inpainting_metadata = None if empty_synthesis else _inpainting_metadata()
    run_metadata: dict[str, object] = {
        "artifacts": {
            "diagnostics_cache": {
                "directory": "diagnostics-cache",
                "manifest_filename": "diagnostics-cache.json",
                "manifest_sha256": cached.manifest_sha256,
            },
            "hybrid_output_rgb16": {
                "filename": "output-hybrid.rgb16.npy",
                "raw_sha256": hash_rgb16(composite.hybrid_rgb16),
            },
            "output_rgb16": {
                "filename": "output.rgb16.npy",
                "raw_sha256": hash_rgb16(pure),
                "source": "portable_digital_ice.ProcessingResult.output_rgb16",
            },
            "routing": {
                "filename": "routing.json",
                "sha256": _sha256(routing_bytes),
            },
            "synthesis_mask": {
                "filename": "synth-mask.png",
                "sha256": mask_png_sha256,
            },
        },
        "backend": {
            "reason": core_backend_reason,
            "requested": "auto",
            "used": core_used_backend,
        },
        "cache": {
            "embedded_manifest_sha256": cached.manifest_sha256,
            "external_manifest_sha256": None,
            "external_mode": "none",
        },
        "generative_model_loaded": not empty_synthesis,
        "inputs": {
            "main": {
                "raw_sha256": hash_rgbir16(main),
                "shape": list(main.shape),
            },
            "prepass": {
                "raw_sha256": hash_rgbir16(prepass),
                "shape": list(prepass.shape),
            },
        },
        "mode": (
            "hybrid_empty_synthesis" if empty_synthesis else "hybrid_lama_inpaint"
        ),
        "profile": {
            "bit_depth": 16,
            "id": binding.profile_id,
            "main_dpi": 4_000,
            "mode": "normal",
            "prepass_dpi": 285,
            "resolution_metric": 4_000,
            "scanner_model": "nikon-super-coolscan-5000-ed",
            "selector": 8,
        },
        "provenance": {
            "acquisition_manifest_sha256": source_manifest_sha256,
            "assertion": assertion_document,
            "assertion_id": assertion_id,
            "assertion_sha256": assertion_sha256,
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
            "core": binding.core_source_manifest_sha256,
            "hybrid": hybrid_source_manifest_sha256,
        },
        "timing": {
            "composite_and_inpainting_seconds": 0.25,
            "iopaint_batch_seconds": [] if empty_synthesis else [0.2],
            "routing_seconds": 0.01,
        },
    }
    if inpainting_metadata is None:
        run_metadata["inpainting"] = {
            "invoked": False,
            "reason": "empty_synthesis_mask",
        }
    else:
        run_metadata["inpainting"] = {
            "invocation_count": len(composite.components),
            "invocations": [
                {
                    "config": {
                        "hd_strategy": "Original",
                        "sd_seed": inpainting_metadata.seed,
                    },
                    "deterministic_environment": _operational_environment(
                        inpainting_metadata
                    ),
                    "input_rgb8_raw_sha256": record.input_rgb8_sha256,
                    "mask_u8_raw_sha256": record.component_mask_sha256,
                    "output_rgb8_raw_sha256": record.inpainted_rgb8_sha256,
                    "sanitized_argv": list(inpainting_metadata.argv),
                }
                for record in composite.components
            ],
            "invoked": True,
            "runtime": _operational_runtime(inpainting_metadata),
        }
    artifact_paths["run-metadata"].write_bytes(canonical_json_bytes(run_metadata))

    sources = [
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
            "diagnostics-cache/diagnostics-cache.json",
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
    ]
    if not empty_synthesis:
        sources.extend(
            (
                ArtifactSource(
                    "component-1-input",
                    "component_input_rgb8",
                    "components/0001-input.png",
                    "image/png",
                    RawEncoding.PNG_U8,
                ),
                ArtifactSource(
                    "component-1-mask",
                    "component_mask_png",
                    "components/0001-mask.png",
                    "image/png",
                    RawEncoding.PNG_U8,
                ),
                ArtifactSource(
                    "component-1-inpainted",
                    "component_inpainted_rgb8",
                    "components/0001-inpainted.png",
                    "image/png",
                    RawEncoding.PNG_U8,
                ),
            )
        )
    document = build_receipt_document(
        artifact_root=root,
        generated_at_utc=datetime(2026, 7, 16, 18, 4, 3, 123456, tzinfo=timezone.utc),
        hybrid_version=current_hybrid_version(),
        hybrid_source_manifest_sha256=hybrid_source_manifest_sha256,
        inputs=InputProvenance(
            prepass_raw_sha256=hash_rgbir16(prepass),
            prepass_shape=prepass.shape,
            main_raw_sha256=hash_rgbir16(main),
            main_shape=main.shape,
            same_frame_assertion_id=assertion_id,
            focus_exposure_assertion_id=(f"{assertion_id}:focus-exposure-locked"),
            source_manifest_sha256=source_manifest_sha256,
        ),
        core=CoreRunMetadata(
            version=binding.core_version,
            source_manifest_sha256=binding.core_source_manifest_sha256,
            profile_id=binding.profile_id,
            requested_backend="auto",
            used_backend=core_used_backend,
            backend_reason=core_backend_reason,
            diagnostics_backend="cpu",
        ),
        diagnostics=diagnostics,
        routing=routing,
        composite=composite,
        changed_mask=changed,
        crop_margin=4,
        inpainting=inpainting_metadata,
        artifacts=sources,
        component_artifacts=(
            ()
            if empty_synthesis
            else (
                ComponentArtifactBinding(
                    component_id=1,
                    input_rgb8_artifact_id="component-1-input",
                    mask_artifact_id="component-1-mask",
                    inpainted_rgb8_artifact_id="component-1-inpainted",
                ),
            )
        ),
    )
    receipt_path = root / RECEIPT_FILENAME
    return _Sample(
        root=root,
        receipt_path=receipt_path,
        document=document,
        artifact_paths=artifact_paths,
        pure=pure,
        hybrid=composite.hybrid_rgb16,
        mask=routing.synthesis_mask,
    )


@pytest.mark.parametrize(
    ("requested", "used", "reason", "diagnostics", "message"),
    (
        (
            "cpu",
            "cuda",
            "explicit CPU request",
            "cuda",
            "impossible backend selection",
        ),
        (
            "cuda",
            "cpu",
            "explicit CUDA request; self-test passed",
            "cpu",
            "impossible backend selection",
        ),
        (
            "auto",
            "cpu",
            "CUDA unavailable; complete job ran on exact CPU reference",
            "cuda",
            "CUDA diagnostics are impossible",
        ),
        (
            "auto",
            "cuda",
            "CUDA unavailable; complete job ran on exact CPU reference",
            "cpu",
            "canonical backend selection reason",
        ),
        (
            "cpu-fast",
            "cpu",
            "explicit cpu-fast request; self-test passed byte parity",
            "cpu",
            "impossible backend selection",
        ),
        (
            "cuda",
            "cpu-fast",
            "explicit CUDA request; self-test passed",
            "cuda",
            "impossible backend selection",
        ),
        (
            "auto",
            "cpu-fast",
            "CUDA unavailable; cpu-fast startup self-test passed byte parity",
            "cpu",
            "must be cpu-fast when the pure run used cpu-fast",
        ),
        (
            "auto",
            "cpu-fast",
            "CUDA unavailable; cpu-fast startup self-test passed byte parity",
            "cuda",
            "must be cpu-fast when the pure run used cpu-fast",
        ),
    ),
)
def test_core_run_metadata_rejects_impossible_or_noncanonical_provenance(
    requested: str,
    used: str,
    reason: str,
    diagnostics: str,
    message: str,
) -> None:
    with pytest.raises(ReceiptError, match=message):
        CoreRunMetadata(
            version="1.0.0",
            source_manifest_sha256="a" * 64,
            profile_id="test-profile",
            requested_backend=requested,
            used_backend=used,
            backend_reason=reason,
            diagnostics_backend=diagnostics,
        )


@pytest.mark.parametrize(
    ("requested", "reason"),
    (
        ("cpu-fast", "explicit cpu-fast request; self-test passed byte parity"),
        (
            "auto",
            "CUDA unavailable; cpu-fast startup self-test passed byte parity",
        ),
    ),
)
def test_core_run_metadata_accepts_canonical_cpu_fast_provenance(
    requested: str,
    reason: str,
) -> None:
    metadata = CoreRunMetadata(
        version="1.0.0",
        source_manifest_sha256="a" * 64,
        profile_id="test-profile",
        requested_backend=requested,
        used_backend="cpu-fast",
        backend_reason=reason,
        diagnostics_backend="cpu-fast",
    )
    assert metadata.used_backend == "cpu-fast"
    assert metadata.diagnostics_backend == "cpu-fast"


@pytest.mark.parametrize(
    "relative_path",
    (
        "outputs//pure.rgb16.npy",
        "outputs/./pure.rgb16.npy",
        "outputs/pure.rgb16.npy/",
    ),
)
def test_artifact_source_rejects_noncanonical_relative_path_aliases(
    relative_path: str,
) -> None:
    with pytest.raises(ReceiptError, match="normalized relative path"):
        ArtifactSource(
            "pure-output",
            "pure_output_rgb16",
            relative_path,
            "application/x-npy",
            RawEncoding.NPY_ARRAY,
        )


def test_round_trip_is_canonical_strict_bound_and_read_only(tmp_path: Path) -> None:
    sample = _build_sample(tmp_path)
    receipt_hash = write_receipt(sample.receipt_path, sample.document)
    verified = verify_receipt(sample.receipt_path)

    payload = sample.receipt_path.read_bytes()
    assert payload == canonical_receipt_bytes(sample.document)
    assert receipt_hash == _sha256(payload) == verified.receipt_sha256
    assert load_receipt_schema()["title"] == "Fauxce hybrid receipt v2"
    assert verified.document["schema"] == "fauxce-hybrid-receipt-v2"
    assert verified.document["disclosure"]["exact_scope"] == (
        "outside_synthesis_mask_only"
    )
    assert verified.document["generation"]["generated_at_utc"].endswith("Z")
    assert verified.document["routing"]["counts"]["unchanged_synthesis_pixels"] > 0
    assert verified.document["routing"]["scipy_version"]
    assert (
        verified.document["inpainting"]["model"]["sanitized_artifact_id"]
        == "lama-weights"
    )
    assert verified.document["inpainting"]["invoked"] is True
    tool = verified.document["inpainting"]["tool"]
    model = verified.document["inpainting"]["model"]
    runtime = verified.document["inpainting"]["runtime"]
    determinism = verified.document["inpainting"]["determinism"]
    assert tool["tool_license_spdx"] == "Apache-2.0"
    assert tool["iopaint_source_manifest_sha256"] == "b" * 64
    assert tool["iopaint_source_file_count"] == 123
    assert model["model_upstream_license_spdx"] == "Apache-2.0"
    assert "separate artifact license" in model["model_artifact_license_status"]
    assert runtime["effective_environment_sha256"] == (
        _inpainting_metadata().effective_environment_sha256
    )
    assert runtime["python_version"] == "3.11.15"
    assert runtime["python_implementation"] == "CPython"
    assert {
        runtime[field]
        for field in (
            "torch_version",
            "numpy_version",
            "pillow_version",
            "opencv_version",
            "pydantic_version",
            "typer_version",
        )
    } == {
        "2.9.0+cu130",
        "1.26.4",
        "9.5.0",
        "4.11.0.86",
        "2.13.4",
        "0.27.0",
    }
    assert runtime["platform_system"] == "Linux"
    assert runtime["platform_release"] == "6.12.0"
    assert runtime["platform_machine"] == "x86_64"
    assert runtime["cuda_available"] is True
    assert runtime["cuda_runtime_version"] == "13.0"
    assert runtime["cudnn_version"] == "91002"
    assert runtime["cuda_device_names"] == ["NVIDIA RTX 5090"]
    assert runtime["cuda_visible_devices"] == "0"
    assert runtime["hip_runtime_version"] is None
    assert runtime["mps_available"] is False
    assert runtime["mps_device_name"] is None
    assert "LaMa is feed-forward" in runtime["seed_scope"]
    assert determinism["deterministic_algorithms_enabled"] is True
    assert determinism["cudnn_benchmark"] is False
    assert "host environment" in determinism["scope"]
    assert not verified.model_weights_rehashed
    assert all(
        artifact["role"] != "model_weights"
        for artifact in verified.document["artifacts"]
    )
    assert sample.receipt_path.name == "hybrid-receipt.json"
    assert [
        row["component_id"] for row in verified.document["composite"]["components"]
    ] == [1]

    np.testing.assert_array_equal(verified.pure_output_rgb16, sample.pure)
    np.testing.assert_array_equal(verified.hybrid_output_rgb16, sample.hybrid)
    np.testing.assert_array_equal(verified.synthesis_mask, sample.mask)
    np.testing.assert_array_equal(
        verified.hybrid_output_rgb16[~verified.synthesis_mask],
        verified.pure_output_rgb16[~verified.synthesis_mask],
    )
    for array in (
        verified.pure_output_rgb16,
        verified.hybrid_output_rgb16,
        verified.synthesis_mask,
    ):
        assert array.flags.c_contiguous
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flat[0] = array.flat[0]


def test_external_model_weights_can_be_rehashed_without_path_disclosure(
    tmp_path: Path,
) -> None:
    sample = _build_sample(tmp_path)
    write_receipt(sample.receipt_path, sample.document)
    weights = b"synthetic deterministic weights\x00\x01"
    external_path = tmp_path / "private-model-store" / "lama.safetensors"
    external_path.parent.mkdir()
    external_path.write_bytes(weights)

    verified = verify_receipt(
        sample.receipt_path,
        model_weights_resolver=lambda attestation: external_path,
        require_model_weights=True,
    )

    assert verified.model_weights_rehashed
    assert str(external_path) not in sample.receipt_path.read_text()
    assert verified.document["inpainting"]["model"]["weights_sha256"] == _sha256(
        weights
    )
    with pytest.raises(ReceiptError, match="resolver is required"):
        verify_receipt(sample.receipt_path, require_model_weights=True)
    with pytest.raises(ReceiptError, match="byte size mismatch"):
        verify_receipt(
            sample.receipt_path,
            model_weights_resolver=lambda _attestation: b"wrong",
        )
    with pytest.raises(ReceiptError, match="SHA-256 mismatch"):
        verify_receipt(
            sample.receipt_path,
            model_weights_resolver=lambda _attestation: b"x" * len(weights),
        )


def test_empty_synthesis_is_valid_and_never_invokes_or_requires_model(
    tmp_path: Path,
) -> None:
    sample = _build_sample(tmp_path, empty_synthesis=True)
    write_receipt(sample.receipt_path, sample.document)

    def forbidden(_attestation: ModelWeightsAttestation) -> bytes:
        raise AssertionError("empty synthesis must not resolve model weights")

    verified = verify_receipt(
        sample.receipt_path,
        model_weights_resolver=forbidden,
        require_model_weights=True,
    )

    assert verified.document["inpainting"] == {
        "invoked": False,
        "reason": "empty_synthesis_mask",
    }
    assert verified.document["composite"]["components"] == []
    assert verified.document["synthesis"]["pixel_count"] == 0
    assert len(verified.document["artifacts"]) == 6
    assert not verified.model_weights_rehashed
    np.testing.assert_array_equal(
        verified.hybrid_output_rgb16,
        verified.pure_output_rgb16,
    )


def test_cuda_pure_run_can_bind_cpu_diagnostics_after_parity_pin(
    tmp_path: Path,
) -> None:
    sample = _build_sample(tmp_path, cuda_pure_cpu_diagnostics=True)
    write_receipt(sample.receipt_path, sample.document)
    backend = verify_receipt(sample.receipt_path).document["core"]["backend"]

    assert backend["used"] == "cuda"
    assert backend["diagnostics"] == "cpu"
    assert backend["reason"] == "startup self-test passed byte parity"


def test_timestamp_is_the_only_intentionally_variable_builder_field(
    tmp_path: Path,
) -> None:
    sample = _build_sample(tmp_path)
    second = deepcopy(sample.document)
    second["generation"]["generated_at_utc"] = "2026-07-16T18:04:04.123456Z"
    validate_receipt_document(second)

    assert canonical_receipt_bytes(sample.document) != canonical_receipt_bytes(second)
    first_without_time = deepcopy(sample.document)
    second_without_time = deepcopy(second)
    del first_without_time["generation"]["generated_at_utc"]
    del second_without_time["generation"]["generated_at_utc"]
    assert first_without_time == second_without_time


def test_receipt_filename_is_fixed_by_contract(tmp_path: Path) -> None:
    sample = _build_sample(tmp_path)

    with pytest.raises(ReceiptError, match="hybrid-receipt.json"):
        write_receipt(sample.root / "wrong-name.json", sample.document)


def test_perimeter_exclusion_is_explicit_unverified_and_left_pure(
    tmp_path: Path,
) -> None:
    sample = _build_sample(tmp_path, include_perimeter_exclusion=True)
    write_receipt(sample.receipt_path, sample.document)
    document = verify_receipt(sample.receipt_path).document

    excluded = document["routing"]["perimeter_excluded_region"]
    assert excluded is not None
    assert excluded["content_interpretation"] == "unverified"
    assert excluded["output_treatment"] == "left_pure"
    assert document["routing"]["counts"]["perimeter_excluded_regions"] == 1
    assert (
        document["routing"]["counts"]["perimeter_excluded_pixels"] == excluded["area"]
    )
    assert document["routing"]["counts"]["perimeter_suppressed_halo_pixels"] > 0


@pytest.mark.parametrize(
    "artifact_id",
    (
        "pure-output",
        "hybrid-output",
        "synthesis-mask",
        "diagnostics-cache",
        "routing",
        "run-metadata",
        "component-1-input",
        "component-1-mask",
        "component-1-inpainted",
    ),
)
def test_one_byte_artifact_tamper_fails_closed(
    tmp_path: Path,
    artifact_id: str,
) -> None:
    sample = _build_sample(tmp_path)
    write_receipt(sample.receipt_path, sample.document)
    artifact = sample.artifact_paths[artifact_id]
    payload = bytearray(artifact.read_bytes())
    payload[-1] ^= 0x01
    artifact.write_bytes(payload)

    with pytest.raises(
        ReceiptError, match=f"artifact {artifact_id} file SHA-256 mismatch"
    ):
        verify_receipt(sample.receipt_path)


def test_file_digest_mismatch_fails_before_numpy_decoder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = _build_sample(tmp_path)
    write_receipt(sample.receipt_path, sample.document)
    pure_path = sample.artifact_paths["pure-output"]
    payload = bytearray(pure_path.read_bytes())
    payload[-1] ^= 0x01
    pure_path.write_bytes(payload)
    target_inode = pure_path.stat().st_ino
    original_load = receipts_module.np.load

    def guarded_decoder(handle: object, *args: object, **kwargs: object) -> object:
        if os.fstat(handle.fileno()).st_ino == target_inode:
            raise AssertionError("digest mismatch must fail before NumPy decode")
        return original_load(handle, *args, **kwargs)

    monkeypatch.setattr(receipts_module.np, "load", guarded_decoder)
    with pytest.raises(ReceiptError, match="pure-output file SHA-256 mismatch"):
        verify_receipt(sample.receipt_path)


def test_oversized_sparse_npy_fails_before_decoder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = _build_sample(tmp_path)
    write_receipt(sample.receipt_path, sample.document)
    pure_path = sample.artifact_paths["pure-output"]
    with pure_path.open("r+b") as handle:
        handle.truncate(receipts_module._MAX_ARTIFACT_FILE_BYTES + 1)
    target_inode = pure_path.stat().st_ino
    original_load = receipts_module.np.load

    def guarded_decoder(handle: object, *args: object, **kwargs: object) -> object:
        if os.fstat(handle.fileno()).st_ino == target_inode:
            raise AssertionError("oversized NPY must fail before NumPy decode")
        return original_load(handle, *args, **kwargs)

    monkeypatch.setattr(receipts_module.np, "load", guarded_decoder)
    with pytest.raises(ReceiptError, match="encoded size exceeds safe limit"):
        verify_receipt(sample.receipt_path)


def test_oversized_png_ihdr_fails_before_decoder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = _build_sample(tmp_path)
    mask_path = sample.artifact_paths["synthesis-mask"]
    _rewrite_png_ihdr_dimensions(
        mask_path,
        width=receipts_module._MAX_IMAGE_PIXELS + 1,
        height=1,
    )
    document = deepcopy(sample.document)
    row = next(
        artifact
        for artifact in document["artifacts"]
        if artifact["id"] == "synthesis-mask"
    )
    row["file_sha256"] = _sha256(mask_path.read_bytes())
    write_receipt(sample.receipt_path, document)
    target_sha256 = row["file_sha256"]
    original_imread = receipts_module.iio.imread

    def guarded_decoder(source: object, *args: object, **kwargs: object) -> object:
        if _sha256(source.getvalue()) == target_sha256:
            raise AssertionError("oversized PNG must fail before image decode")
        return original_imread(source, *args, **kwargs)

    monkeypatch.setattr(receipts_module.iio, "imread", guarded_decoder)
    with pytest.raises(ReceiptError, match="pixel count exceeds safe limit"):
        verify_receipt(sample.receipt_path)


def test_aggregate_decoded_budget_fails_before_next_decoder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = _build_sample(tmp_path)
    document = deepcopy(sample.document)
    artifact_rows = document["artifacts"]
    by_id = {row["id"]: row for row in artifact_rows}
    document["artifacts"] = [
        by_id["pure-output"],
        by_id["hybrid-output"],
        *(
            row
            for row in artifact_rows
            if row["id"] not in {"pure-output", "hybrid-output"}
        ),
    ]
    write_receipt(sample.receipt_path, document)
    monkeypatch.setattr(
        receipts_module,
        "_MAX_AGGREGATE_DECODED_BYTES",
        sample.pure.nbytes + sample.hybrid.nbytes - 1,
    )
    hybrid_inode = sample.artifact_paths["hybrid-output"].stat().st_ino
    original_load = receipts_module.np.load

    def guarded_decoder(handle: object, *args: object, **kwargs: object) -> object:
        if os.fstat(handle.fileno()).st_ino == hybrid_inode:
            raise AssertionError("aggregate cap must fail before next NumPy decode")
        return original_load(handle, *args, **kwargs)

    monkeypatch.setattr(receipts_module.np, "load", guarded_decoder)
    with pytest.raises(ReceiptError, match="aggregate decoded artifact size limit"):
        verify_receipt(sample.receipt_path)


@pytest.mark.parametrize(
    ("extra_role", "message"),
    (
        ("audit_blob", "unsupported artifact roles"),
        ("component_input_rgb8", "artifact count must equal"),
    ),
)
def test_unreferenced_artifacts_are_rejected_before_measurement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra_role: str,
    message: str,
) -> None:
    sample = _build_sample(tmp_path)
    document = deepcopy(sample.document)
    document["artifacts"].append(
        {
            "id": "unreferenced-extra",
            "role": extra_role,
            "relative_path": "unreferenced-extra.bin",
            "media_type": "application/octet-stream",
            "raw_encoding": "opaque_file_bytes",
            "file_sha256": "0" * 64,
            "raw_sha256": "0" * 64,
            "dtype": "bytes",
            "shape": [],
        }
    )
    sample.receipt_path.write_bytes(canonical_receipt_bytes(document))

    def forbidden_measurement(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("semantic artifact census must fail before measurement")

    monkeypatch.setattr(
        receipts_module,
        "_measure_artifact",
        forbidden_measurement,
    )
    with pytest.raises(ReceiptError, match=message):
        verify_receipt(sample.receipt_path)


@pytest.mark.parametrize("symlink_kind", ("leaf", "directory"))
def test_verifier_rejects_symlink_artifact_path_components(
    tmp_path: Path,
    symlink_kind: str,
) -> None:
    sample = _build_sample(tmp_path)
    write_receipt(sample.receipt_path, sample.document)
    if symlink_kind == "leaf":
        artifact_path = sample.artifact_paths["pure-output"]
        target = sample.root / "real-output.rgb16.npy"
        artifact_path.replace(target)
        artifact_path.symlink_to(target.name)
    else:
        component_directory = sample.root / "components"
        target = sample.root / "real-components"
        component_directory.replace(target)
        component_directory.symlink_to(target.name, target_is_directory=True)

    with pytest.raises(ReceiptError, match="symlink component"):
        verify_receipt(sample.receipt_path)


def test_path_swap_after_open_uses_original_artifact_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = _build_sample(tmp_path)
    write_receipt(sample.receipt_path, sample.document)
    pure_path = sample.artifact_paths["pure-output"]
    replacement_path = sample.root / "replacement-output.npy"
    replacement = np.bitwise_xor(sample.pure, np.uint16(1))
    _save_npy(replacement_path, replacement)
    backup_path = sample.root / "opened-original-output.npy"
    original_load = receipts_module.np.load
    target_inode = pure_path.stat().st_ino
    swapped = False

    def swapping_load(handle: object, *args: object, **kwargs: object) -> object:
        nonlocal swapped
        if not swapped and os.fstat(handle.fileno()).st_ino == target_inode:
            swapped = True
            pure_path.replace(backup_path)
            replacement_path.replace(pure_path)
        return original_load(handle, *args, **kwargs)

    monkeypatch.setattr(receipts_module.np, "load", swapping_load)
    verified = verify_receipt(sample.receipt_path)

    assert swapped
    np.testing.assert_array_equal(verified.pure_output_rgb16, sample.pure)
    assert not np.array_equal(np.load(pure_path, allow_pickle=False), sample.pure)


def test_same_inode_mutation_during_decode_fails_post_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = _build_sample(tmp_path)
    write_receipt(sample.receipt_path, sample.document)
    pure_path = sample.artifact_paths["pure-output"]
    original_load = receipts_module.np.load
    target_inode = pure_path.stat().st_ino
    mutated = False

    def mutating_load(handle: object, *args: object, **kwargs: object) -> object:
        nonlocal mutated
        decoded = original_load(handle, *args, **kwargs)
        if not mutated and os.fstat(handle.fileno()).st_ino == target_inode:
            mutated = True
            with pure_path.open("r+b") as handle:
                handle.seek(-1, 2)
                original = handle.read(1)
                handle.seek(-1, 2)
                handle.write(bytes((original[0] ^ 0x01,)))
        return decoded

    monkeypatch.setattr(receipts_module.np, "load", mutating_load)
    with pytest.raises(ReceiptError, match="changed while being verified"):
        verify_receipt(sample.receipt_path)


def test_operational_json_uses_verified_opaque_snapshots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = _build_sample(tmp_path)
    write_receipt(sample.receipt_path, sample.document)
    original_verify_cache = receipts_module._verify_embedded_diagnostics_cache
    swapped = False

    def swapping_verify_cache(*args: object, **kwargs: object) -> object:
        nonlocal swapped
        assert not swapped
        swapped = True
        for artifact_id in ("routing", "run-metadata"):
            path = sample.artifact_paths[artifact_id]
            path.replace(sample.root / f"verified-{path.name}")
            path.write_bytes(b"{}\n")
        return original_verify_cache(*args, **kwargs)

    monkeypatch.setattr(
        receipts_module,
        "_verify_embedded_diagnostics_cache",
        swapping_verify_cache,
    )
    verified = verify_receipt(sample.receipt_path)

    assert swapped
    np.testing.assert_array_equal(verified.pure_output_rgb16, sample.pure)


def test_one_byte_receipt_tamper_fails_schema_validation(tmp_path: Path) -> None:
    sample = _build_sample(tmp_path)
    write_receipt(sample.receipt_path, sample.document)
    payload = sample.receipt_path.read_bytes()
    assert b"fauxce-hybrid-receipt-v2" in payload
    sample.receipt_path.write_bytes(
        payload.replace(b"fauxce-hybrid-receipt-v2", b"gauxce-hybrid-receipt-v2", 1)
    )

    with pytest.raises(ReceiptError, match="schema validation failed"):
        verify_receipt(sample.receipt_path)


@pytest.mark.parametrize(
    "filename",
    (
        "output.rgb16.npy",
        "score-plane.npy",
        "at-floor-mask.npy",
        "changed-mask.npy",
    ),
)
def test_unlisted_embedded_cache_artifact_tamper_fails_closed(
    tmp_path: Path,
    filename: str,
) -> None:
    sample = _build_sample(tmp_path)
    write_receipt(sample.receipt_path, sample.document)
    artifact = sample.root / "diagnostics-cache" / filename
    payload = bytearray(artifact.read_bytes())
    payload[-1] ^= 0x01
    artifact.write_bytes(payload)

    with pytest.raises(
        ReceiptError,
        match="embedded diagnostics cache failed verification",
    ):
        verify_receipt(sample.receipt_path)


@pytest.mark.parametrize("role", ("routing_json", "run_metadata_json"))
def test_required_operational_artifact_role_cannot_be_omitted(
    tmp_path: Path,
    role: str,
) -> None:
    sample = _build_sample(tmp_path)
    document = deepcopy(sample.document)
    document["artifacts"] = [
        artifact for artifact in document["artifacts"] if artifact["role"] != role
    ]
    sample.receipt_path.write_bytes(canonical_receipt_bytes(document))

    with pytest.raises(ReceiptError, match=f"artifact role {role} must occur exactly"):
        verify_receipt(sample.receipt_path)


def test_jointly_rehashed_routing_document_must_equal_receipt_routing(
    tmp_path: Path,
) -> None:
    sample = _build_sample(tmp_path)
    receipt = deepcopy(sample.document)
    routing_path = sample.artifact_paths["routing"]
    routing = json.loads(routing_path.read_bytes())
    routing["policy"]["min_area"] += 1
    routing_path.write_bytes(canonical_json_bytes(routing))
    _rehash_artifact_row(
        receipt,
        artifact_id="routing",
        path=routing_path,
    )

    metadata_path = sample.artifact_paths["run-metadata"]
    run_metadata = json.loads(metadata_path.read_bytes())
    run_metadata["artifacts"]["routing"]["sha256"] = _sha256(routing_path.read_bytes())
    metadata_path.write_bytes(canonical_json_bytes(run_metadata))
    _rehash_artifact_row(
        receipt,
        artifact_id="run-metadata",
        path=metadata_path,
    )
    sample.receipt_path.write_bytes(canonical_receipt_bytes(receipt))

    with pytest.raises(ReceiptError, match="routing JSON artifact disagrees"):
        verify_receipt(sample.receipt_path)


def test_jointly_rehashed_invoked_inpainting_above_budget_fails_closed(
    tmp_path: Path,
) -> None:
    sample = _build_sample(tmp_path)
    receipt = deepcopy(sample.document)
    receipt["routing"]["policy"]["max_synth_fraction"] = 0.0
    receipt["routing"]["within_synthesis_budget"] = False
    receipt["synthesis"]["maximum_fraction"] = 0.0
    receipt["synthesis"]["within_budget"] = False

    routing_path = sample.artifact_paths["routing"]
    routing = json.loads(routing_path.read_bytes())
    routing["policy"]["max_synth_fraction"] = 0.0
    routing["within_synthesis_budget"] = False
    routing_path.write_bytes(canonical_json_bytes(routing))
    _rehash_artifact_row(
        receipt,
        artifact_id="routing",
        path=routing_path,
    )

    metadata_path = sample.artifact_paths["run-metadata"]
    run_metadata = json.loads(metadata_path.read_bytes())
    run_metadata["artifacts"]["routing"]["sha256"] = _sha256(routing_path.read_bytes())
    run_metadata["routing"]["within_synthesis_budget"] = False
    metadata_path.write_bytes(canonical_json_bytes(run_metadata))
    _rehash_artifact_row(
        receipt,
        artifact_id="run-metadata",
        path=metadata_path,
    )
    sample.receipt_path.write_bytes(canonical_receipt_bytes(receipt))

    with pytest.raises(
        ReceiptError,
        match="IOPaint cannot be invoked above the synthesis budget",
    ):
        verify_receipt(sample.receipt_path)


def test_jointly_rehashed_routing_policy_must_replay_cached_diagnostics(
    tmp_path: Path,
) -> None:
    sample = _build_sample(tmp_path)
    receipt = deepcopy(sample.document)
    receipt["routing"]["policy"]["min_area"] = 1_000

    routing_path = sample.artifact_paths["routing"]
    routing = json.loads(routing_path.read_bytes())
    routing["policy"]["min_area"] = 1_000
    routing_path.write_bytes(canonical_json_bytes(routing))
    _rehash_artifact_row(
        receipt,
        artifact_id="routing",
        path=routing_path,
    )

    metadata_path = sample.artifact_paths["run-metadata"]
    run_metadata = json.loads(metadata_path.read_bytes())
    run_metadata["artifacts"]["routing"]["sha256"] = _sha256(routing_path.read_bytes())
    metadata_path.write_bytes(canonical_json_bytes(run_metadata))
    _rehash_artifact_row(
        receipt,
        artifact_id="run-metadata",
        path=metadata_path,
    )
    sample.receipt_path.write_bytes(canonical_receipt_bytes(receipt))

    with pytest.raises(ReceiptError, match="deterministic routing replay"):
        verify_receipt(sample.receipt_path)


def test_jointly_rehashed_cache_mask_must_replay_routing_without_count_drift(
    tmp_path: Path,
) -> None:
    sample = _build_sample(tmp_path, include_perimeter_exclusion=True)
    receipt = deepcopy(sample.document)
    cache_directory = sample.artifact_paths["diagnostics-cache"].parent
    manifest_path = sample.artifact_paths["diagnostics-cache"]
    manifest = json.loads(manifest_path.read_bytes())

    at_floor_path = cache_directory / "at-floor-mask.npy"
    score_path = cache_directory / "score-plane.npy"
    at_floor = np.load(at_floor_path, allow_pickle=False)
    score = np.load(score_path, allow_pickle=False)
    removed = (6, 15)
    added = (1, 10)
    assert at_floor[removed] and not at_floor[added]
    assert sample.mask[removed] and sample.mask[added]
    floor = np.array(
        manifest["diagnostics"]["score_floor_u32_le_bits"],
        dtype="<u4",
    ).view("<f4")[()]
    at_floor[removed] = False
    at_floor[added] = True
    score[removed] = np.float32(0.08)
    score[added] = floor
    _save_npy(at_floor_path, at_floor)
    _save_npy(score_path, score)

    for artifact_name, path, array in (
        ("at_floor_mask", at_floor_path, at_floor),
        ("score_plane", score_path, score),
    ):
        row = manifest["artifacts"][artifact_name]
        row["file_sha256"] = _sha256(path.read_bytes())
        row["raw_sha256"] = _raw_sha256(array)
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    _rehash_artifact_row(
        receipt,
        artifact_id="diagnostics-cache",
        path=manifest_path,
    )

    manifest_sha256 = _sha256(manifest_path.read_bytes())
    metadata_path = sample.artifact_paths["run-metadata"]
    run_metadata = json.loads(metadata_path.read_bytes())
    run_metadata["artifacts"]["diagnostics_cache"]["manifest_sha256"] = manifest_sha256
    run_metadata["cache"]["embedded_manifest_sha256"] = manifest_sha256
    metadata_path.write_bytes(canonical_json_bytes(run_metadata))
    _rehash_artifact_row(
        receipt,
        artifact_id="run-metadata",
        path=metadata_path,
    )
    sample.receipt_path.write_bytes(canonical_receipt_bytes(receipt))

    with pytest.raises(ReceiptError, match="deterministic routing replay"):
        verify_receipt(sample.receipt_path)


@pytest.mark.parametrize("mutation", ("missing", "extra"))
def test_jointly_rehashed_run_metadata_environment_must_be_exact(
    tmp_path: Path,
    mutation: str,
) -> None:
    sample = _build_sample(tmp_path)
    receipt = deepcopy(sample.document)
    metadata_path = sample.artifact_paths["run-metadata"]
    run_metadata = json.loads(metadata_path.read_bytes())
    environment = run_metadata["inpainting"]["invocations"][0][
        "deterministic_environment"
    ]
    if mutation == "missing":
        del environment["OMP_NUM_THREADS"]
    else:
        environment["UNDECLARED_VARIABLE"] = "1"
    metadata_path.write_bytes(canonical_json_bytes(run_metadata))
    _rehash_artifact_row(
        receipt,
        artifact_id="run-metadata",
        path=metadata_path,
    )
    sample.receipt_path.write_bytes(canonical_receipt_bytes(receipt))

    with pytest.raises(
        ReceiptError,
        match="deterministic environment disagrees with receipt",
    ):
        verify_receipt(sample.receipt_path)


def test_jointly_rehashed_unsupported_cuda_visibility_cannot_claim_adapter_origin(
    tmp_path: Path,
) -> None:
    sample = _build_sample(tmp_path)
    receipt = deepcopy(sample.document)
    metadata_value = receipts_module._inpainting_metadata_from_document(
        receipt["inpainting"]
    )
    environment_values = dict(vars(metadata_value))
    environment_values["cuda_visible_devices"] = "garbage"
    environment_sha256 = _environment_sha256(environment_values)
    receipt["inpainting"]["runtime"]["cuda_visible_devices"] = "garbage"
    receipt["inpainting"]["runtime"]["effective_environment_sha256"] = (
        environment_sha256
    )

    metadata_path = sample.artifact_paths["run-metadata"]
    run_metadata = json.loads(metadata_path.read_bytes())
    run_metadata["inpainting"]["runtime"]["cuda_visible_devices"] = "garbage"
    run_metadata["inpainting"]["runtime"]["effective_environment_sha256"] = (
        environment_sha256
    )
    metadata_path.write_bytes(canonical_json_bytes(run_metadata))
    _rehash_artifact_row(
        receipt,
        artifact_id="run-metadata",
        path=metadata_path,
    )
    sample.receipt_path.write_bytes(canonical_receipt_bytes(receipt))

    with pytest.raises(ReceiptError, match="unsupported token syntax"):
        verify_receipt(sample.receipt_path)


def test_jointly_rehashed_noncanonical_argv_cannot_claim_adapter_origin(
    tmp_path: Path,
) -> None:
    sample = _build_sample(tmp_path)
    receipt = deepcopy(sample.document)
    mutated_argv = [*receipt["inpainting"]["runtime"]["argv"], "--unexpected"]
    receipt["inpainting"]["runtime"]["argv"] = mutated_argv

    metadata_path = sample.artifact_paths["run-metadata"]
    run_metadata = json.loads(metadata_path.read_bytes())
    for invocation in run_metadata["inpainting"]["invocations"]:
        invocation["sanitized_argv"] = mutated_argv
    metadata_path.write_bytes(canonical_json_bytes(run_metadata))
    _rehash_artifact_row(
        receipt,
        artifact_id="run-metadata",
        path=metadata_path,
    )
    sample.receipt_path.write_bytes(canonical_receipt_bytes(receipt))

    with pytest.raises(ReceiptError, match="canonical IOPaint adapter invocation"):
        verify_receipt(sample.receipt_path)


@pytest.mark.parametrize(
    "mutation",
    (
        "inputs",
        "backend",
        "profile",
        "source_manifest",
        "routing_summary",
        "cache_manifest",
        "pure_output",
        "hybrid_output",
        "synthesis_mask",
        "invocation",
        "inpainting_runtime",
    ),
)
def test_jointly_rehashed_run_metadata_must_match_receipt_semantics(
    tmp_path: Path,
    mutation: str,
) -> None:
    sample = _build_sample(tmp_path)
    receipt = deepcopy(sample.document)
    metadata_path = sample.artifact_paths["run-metadata"]
    run_metadata = json.loads(metadata_path.read_bytes())
    replacement = "c" * 64
    if mutation == "inputs":
        run_metadata["inputs"]["main"]["raw_sha256"] = replacement
    elif mutation == "backend":
        run_metadata["backend"]["used"] = "cuda"
    elif mutation == "profile":
        run_metadata["profile"]["id"] = "wrong-profile"
    elif mutation == "source_manifest":
        run_metadata["source_manifests"]["hybrid"] = replacement
    elif mutation == "routing_summary":
        run_metadata["routing"]["synthesis_pixels"] += 1
    elif mutation == "cache_manifest":
        run_metadata["cache"]["embedded_manifest_sha256"] = replacement
    elif mutation == "pure_output":
        run_metadata["artifacts"]["output_rgb16"]["raw_sha256"] = replacement
    elif mutation == "hybrid_output":
        run_metadata["artifacts"]["hybrid_output_rgb16"]["raw_sha256"] = replacement
    elif mutation == "synthesis_mask":
        run_metadata["artifacts"]["synthesis_mask"]["sha256"] = replacement
    elif mutation == "invocation":
        run_metadata["inpainting"]["invocations"][0]["input_rgb8_raw_sha256"] = (
            replacement
        )
    else:
        run_metadata["inpainting"]["runtime"]["model_weights_sha256"] = replacement
    metadata_path.write_bytes(canonical_json_bytes(run_metadata))
    _rehash_artifact_row(
        receipt,
        artifact_id="run-metadata",
        path=metadata_path,
    )
    sample.receipt_path.write_bytes(canonical_receipt_bytes(receipt))

    with pytest.raises(ReceiptError, match="run metadata"):
        verify_receipt(sample.receipt_path)


@pytest.mark.parametrize(
    ("diagnostic", "message"),
    (
        ("changed", "unchanged synthesis pixel count"),
        ("at_floor", "non-floor synthesis pixel count"),
    ),
)
def test_rehashed_cache_masks_must_match_synthesis_accounting(
    tmp_path: Path,
    diagnostic: str,
    message: str,
) -> None:
    sample = _build_sample(tmp_path)
    receipt = deepcopy(sample.document)
    cache_directory = sample.artifact_paths["diagnostics-cache"].parent
    manifest_path = sample.artifact_paths["diagnostics-cache"]
    manifest = json.loads(manifest_path.read_bytes())

    if diagnostic == "changed":
        artifact_name = "changed_mask"
        path = cache_directory / "changed-mask.npy"
        mask = np.load(path, allow_pickle=False)
        removed = tuple(np.argwhere(mask & sample.mask)[0])
        added = tuple(np.argwhere(~mask & ~sample.mask)[0])
        mask[removed] = False
        mask[added] = True
        _save_npy(path, mask)
        updated = {artifact_name: (path, mask)}
    else:
        at_floor_path = cache_directory / "at-floor-mask.npy"
        score_path = cache_directory / "score-plane.npy"
        at_floor = np.load(at_floor_path, allow_pickle=False)
        score = np.load(score_path, allow_pickle=False)
        removed = tuple(np.argwhere(at_floor & sample.mask)[0])
        added = tuple(np.argwhere(~at_floor & ~sample.mask)[0])
        floor = np.array(
            manifest["diagnostics"]["score_floor_u32_le_bits"],
            dtype="<u4",
        ).view("<f4")[()]
        at_floor[removed] = False
        at_floor[added] = True
        score[removed] = np.float32(0.08)
        score[added] = floor
        _save_npy(at_floor_path, at_floor)
        _save_npy(score_path, score)
        updated = {
            "at_floor_mask": (at_floor_path, at_floor),
            "score_plane": (score_path, score),
        }

    for artifact_name, (path, array) in updated.items():
        row = manifest["artifacts"][artifact_name]
        row["file_sha256"] = _sha256(path.read_bytes())
        row["raw_sha256"] = _raw_sha256(array)
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    _rehash_artifact_row(
        receipt,
        artifact_id="diagnostics-cache",
        path=manifest_path,
    )

    manifest_sha256 = _sha256(manifest_path.read_bytes())
    metadata_path = sample.artifact_paths["run-metadata"]
    run_metadata = json.loads(metadata_path.read_bytes())
    run_metadata["artifacts"]["diagnostics_cache"]["manifest_sha256"] = manifest_sha256
    run_metadata["cache"]["embedded_manifest_sha256"] = manifest_sha256
    metadata_path.write_bytes(canonical_json_bytes(run_metadata))
    _rehash_artifact_row(
        receipt,
        artifact_id="run-metadata",
        path=metadata_path,
    )
    sample.receipt_path.write_bytes(canonical_receipt_bytes(receipt))

    with pytest.raises(ReceiptError, match=message):
        verify_receipt(sample.receipt_path)


def test_verifier_rejects_noncanonical_artifact_path_alias(tmp_path: Path) -> None:
    sample = _build_sample(tmp_path)
    document = deepcopy(sample.document)
    row = next(
        artifact
        for artifact in document["artifacts"]
        if artifact["id"] == "hybrid-output"
    )
    row["relative_path"] = "outputs//hybrid.rgb16.npy"
    sample.receipt_path.write_bytes(canonical_receipt_bytes(document))

    with pytest.raises(ReceiptError, match="normalized relative path"):
        verify_receipt(sample.receipt_path)


def test_embedded_cache_provenance_sha_must_match_run_metadata(
    tmp_path: Path,
) -> None:
    sample = _build_sample(tmp_path)
    manifest_path = sample.artifact_paths["diagnostics-cache"]
    manifest = json.loads(manifest_path.read_bytes())
    manifest["binding"]["provenance_assertion_sha256"] = "c" * 64
    manifest_path.write_bytes(canonical_json_bytes(manifest))

    document = deepcopy(sample.document)
    row = next(
        artifact
        for artifact in document["artifacts"]
        if artifact["id"] == "diagnostics-cache"
    )
    manifest_sha256 = _sha256(manifest_path.read_bytes())
    row["file_sha256"] = manifest_sha256
    row["raw_sha256"] = manifest_sha256
    sample.receipt_path.write_bytes(canonical_receipt_bytes(document))

    with pytest.raises(
        ReceiptError,
        match="binding disagrees with run metadata provenance_assertion_sha256",
    ):
        verify_receipt(sample.receipt_path)


def test_verifier_rejects_impossible_backend_claim_in_receipt(tmp_path: Path) -> None:
    sample = _build_sample(tmp_path)
    document = deepcopy(sample.document)
    backend = document["core"]["backend"]
    backend["used"] = "cuda"
    backend["reason"] = "CUDA unavailable; complete job ran on exact CPU reference"
    sample.receipt_path.write_bytes(canonical_receipt_bytes(document))

    with pytest.raises(ReceiptError, match="canonical backend selection reason"):
        verify_receipt(sample.receipt_path)


def test_verifier_detects_outside_mask_change_even_with_rehashed_artifact(
    tmp_path: Path,
) -> None:
    sample = _build_sample(tmp_path)
    hybrid = np.array(sample.hybrid, copy=True)
    hybrid[0, 0, 0] ^= np.uint16(1)
    hybrid_path = sample.artifact_paths["hybrid-output"]
    _save_npy(hybrid_path, hybrid)

    document = deepcopy(sample.document)
    row = next(item for item in document["artifacts"] if item["id"] == "hybrid-output")
    row["file_sha256"] = _sha256(hybrid_path.read_bytes())
    row["raw_sha256"] = _raw_sha256(hybrid)
    document["composite"]["hybrid_rgb16_raw_sha256"] = row["raw_sha256"]
    sample.receipt_path.write_bytes(canonical_receipt_bytes(document))

    with pytest.raises(ReceiptError, match="outside synthesis mask"):
        verify_receipt(sample.receipt_path)


def test_verifier_replays_composite_after_rehashed_inside_mask_change(
    tmp_path: Path,
) -> None:
    sample = _build_sample(tmp_path)
    hybrid = np.array(sample.hybrid, copy=True)
    y, x = np.argwhere(sample.mask)[0]
    hybrid[y, x, 0] ^= np.uint16(1)
    hybrid_path = sample.artifact_paths["hybrid-output"]
    _save_npy(hybrid_path, hybrid)

    document = deepcopy(sample.document)
    row = next(item for item in document["artifacts"] if item["id"] == "hybrid-output")
    row["file_sha256"] = _sha256(hybrid_path.read_bytes())
    row["raw_sha256"] = _raw_sha256(hybrid)
    document["composite"]["hybrid_rgb16_raw_sha256"] = row["raw_sha256"]
    sample.receipt_path.write_bytes(canonical_receipt_bytes(document))

    with pytest.raises(ReceiptError, match="deterministic composite replay"):
        verify_receipt(sample.receipt_path)


@pytest.mark.parametrize("mutation", ("missing", "root_extra", "nested_extra"))
def test_schema_rejects_missing_and_extra_fields(
    tmp_path: Path,
    mutation: str,
) -> None:
    sample = _build_sample(tmp_path)
    document = deepcopy(sample.document)
    if mutation == "missing":
        del document["core"]["diagnostics"]["score_floor_u32_le_bits"]
    elif mutation == "root_extra":
        document["unexpected"] = True
    else:
        document["inpainting"]["model"]["absolute_path"] = "/secret/model.pt"

    with pytest.raises(ReceiptError, match="schema validation failed"):
        validate_receipt_document(document)


@pytest.mark.parametrize(
    ("section", "field"),
    (
        ("tool", "tool_license_spdx"),
        ("tool", "iopaint_source_manifest_sha256"),
        ("model", "model_artifact_license_status"),
        ("runtime", "effective_environment_sha256"),
        ("runtime", "python_version"),
        ("runtime", "cuda_available"),
        ("runtime", "mps_available"),
        ("determinism", "deterministic_algorithms_enabled"),
    ),
)
def test_schema_requires_effective_inference_environment_fields(
    tmp_path: Path,
    section: str,
    field: str,
) -> None:
    sample = _build_sample(tmp_path)
    document = deepcopy(sample.document)
    del document["inpainting"][section][field]

    with pytest.raises(ReceiptError, match="schema validation failed"):
        validate_receipt_document(document)


def test_semantic_verifier_rejects_inconsistent_accelerator_metadata(
    tmp_path: Path,
) -> None:
    sample = _build_sample(tmp_path)
    document = deepcopy(sample.document)
    document["inpainting"]["runtime"]["mps_device_name"] = "Apple M3 Max"
    validate_receipt_document(document)
    sample.receipt_path.write_bytes(canonical_receipt_bytes(document))

    with pytest.raises(ReceiptError, match="null when MPS is unavailable"):
        verify_receipt(sample.receipt_path)


def test_semantic_verifier_recomputes_effective_environment_hash(
    tmp_path: Path,
) -> None:
    sample = _build_sample(tmp_path)
    document = deepcopy(sample.document)
    document["inpainting"]["runtime"]["platform_machine"] = "aarch64"
    validate_receipt_document(document)
    sample.receipt_path.write_bytes(canonical_receipt_bytes(document))

    with pytest.raises(ReceiptError, match="does not match the recorded runtime"):
        verify_receipt(sample.receipt_path)


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"python_implementation": "CPython\nprivate"}, "unsafe runtime"),
        ({"iopaint_source_file_count": True}, "positive integer"),
        ({"deterministic_algorithms_enabled": 1}, "must be a boolean"),
        ({"cuda_device_names": ["NVIDIA RTX 5090"]}, "tuple of strings"),
        ({"mps_device_name": "Apple M3 Max"}, "null when MPS is unavailable"),
    ),
)
def test_inpainting_metadata_rejects_unsafe_or_mistyped_environment(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ReceiptError, match=message):
        _inpainting_metadata(**overrides)


@pytest.mark.parametrize(
    "value",
    (
        "unset",
        "empty",
        "-1",
        "0,2",
        "GPU-deadbeef",
        "MIG-deadbeef",
        "MIG-GPU-8932f937-d72c-4106-c12f-20bd9faed9f6/1/2",
    ),
)
def test_inpainting_metadata_accepts_adapter_cuda_visibility_syntax(
    value: str,
) -> None:
    assert (
        _inpainting_metadata(cuda_visible_devices=value).cuda_visible_devices == value
    )


@pytest.mark.parametrize(
    ("value", "message"),
    (
        ("garbage", "unsupported token syntax"),
        (",".join(["0"] * 129), "supported token count"),
        (",".join([f"GPU-{'a' * 32}"] * 128), "supported length"),
    ),
)
def test_inpainting_metadata_rejects_cuda_visibility_outside_adapter_contract(
    value: str,
    message: str,
) -> None:
    with pytest.raises(ReceiptError, match=message):
        _inpainting_metadata(cuda_visible_devices=value)


def test_absolute_model_paths_are_never_accepted() -> None:
    with pytest.raises(ReceiptError, match="absolute filesystem path"):
        _inpainting_metadata(model_id="/models/lama.pt")

    with pytest.raises(ReceiptError, match="argv cannot disclose"):
        _inpainting_metadata(argv=("iopaint", "--weights=/models/lama.pt"))

    with pytest.raises(ReceiptError, match="absolute filesystem path"):
        ModelWeightsAttestation(
            sanitized_artifact_id="lama-weights",
            sha256="0" * 64,
            byte_size=1,
            sanitized_external_reference="/models/lama.pt",
        )

    with pytest.raises(ReceiptError, match="query or fragment"):
        ModelWeightsAttestation(
            sanitized_artifact_id="lama-weights",
            sha256="0" * 64,
            byte_size=1,
            sanitized_external_reference="https://models.example/lama?token=secret",
        )

    with pytest.raises(ReceiptError, match="absolute filesystem path"):
        ModelWeightsAttestation(
            sanitized_artifact_id="lama-weights",
            sha256="0" * 64,
            byte_size=1,
            sanitized_external_reference="file:///models/lama.pt",
        )


def test_verifier_rejects_bundled_model_weights_artifact(tmp_path: Path) -> None:
    sample = _build_sample(tmp_path)
    document = deepcopy(sample.document)
    document["artifacts"].append(
        {
            "id": "copied-model",
            "role": "model_weights",
            "relative_path": "model/lama.safetensors",
            "media_type": "application/octet-stream",
            "raw_encoding": "opaque_file_bytes",
            "file_sha256": "0" * 64,
            "raw_sha256": "0" * 64,
            "dtype": "bytes",
            "shape": [],
        }
    )
    sample.receipt_path.write_bytes(canonical_receipt_bytes(document))

    with pytest.raises(ReceiptError, match="must not be copied"):
        verify_receipt(sample.receipt_path)


def test_verifier_rejects_absolute_model_path_in_argv(tmp_path: Path) -> None:
    sample = _build_sample(tmp_path)
    document = deepcopy(sample.document)
    document["inpainting"]["runtime"]["argv"] = [
        "iopaint",
        "--weights",
        "/secret/lama.pt",
    ]
    sample.receipt_path.write_bytes(canonical_receipt_bytes(document))

    with pytest.raises(ReceiptError, match="argv cannot disclose"):
        verify_receipt(sample.receipt_path)


def test_mask_writer_emits_exact_grayscale_zero_and_255(tmp_path: Path) -> None:
    mask = np.zeros((5, 7), dtype=bool)
    mask[1:4, 2:6] = True
    path = tmp_path / "mask.png"

    digest = write_synthesis_mask_png(path, mask)
    decoded = np.asarray(iio.imread(path, extension=".png"))

    assert digest == _sha256(path.read_bytes())
    assert decoded.dtype == np.uint8
    assert decoded.shape == mask.shape
    assert set(np.unique(decoded)) == {0, 255}
    np.testing.assert_array_equal(decoded == 255, mask)


def test_builder_records_current_hybrid_manifest(tmp_path: Path) -> None:
    sample = _build_sample(tmp_path)
    assert sample.document["generation"]["hybrid_source_manifest_sha256"] == (
        compute_hybrid_source_manifest()
    )
    assert sample.document["generation"]["hybrid_version"] == current_hybrid_version()


def test_verifier_rejects_jointly_rewritten_stale_hybrid_source_manifest(
    tmp_path: Path,
) -> None:
    sample = _build_sample(tmp_path)
    receipt = deepcopy(sample.document)
    stale_manifest = "c" * 64
    assert stale_manifest != compute_hybrid_source_manifest()
    receipt["generation"]["hybrid_source_manifest_sha256"] = stale_manifest

    metadata_path = sample.artifact_paths["run-metadata"]
    run_metadata = json.loads(metadata_path.read_bytes())
    run_metadata["source_manifests"]["hybrid"] = stale_manifest
    metadata_path.write_bytes(canonical_json_bytes(run_metadata))
    _rehash_artifact_row(
        receipt,
        artifact_id="run-metadata",
        path=metadata_path,
    )
    sample.receipt_path.write_bytes(canonical_receipt_bytes(receipt))

    with pytest.raises(ReceiptError, match="does not match the running source tree"):
        verify_receipt(sample.receipt_path)
