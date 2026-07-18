from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest
from portable_digital_ice import (
    AcquisitionEpoch,
    ComputeBackend,
    DualRGBIAcquisition,
    ProcessingJob,
    ProcessingMode,
    RGBI16Frame,
    ScannerModel,
    process,
)
from portable_digital_ice.cuda_backend.engine import CudaBackendUnavailable

from fauxce_hybrid import cli
from fauxce_hybrid.cache import canonical_json_bytes, hash_rgb16, hash_rgbir16
from fauxce_hybrid.iopaint import (
    IOPaintInvocationMetadata,
    IOPaintRuntimeMetadata,
)
from fauxce_hybrid.receipts import ReceiptError, verify_receipt


class _FakeIOPaintAdapter:
    def __init__(self, weights_sha256: str) -> None:
        self._weights_sha256 = weights_sha256
        self._invocations: list[IOPaintInvocationMetadata] = []

    @property
    def invocations(self) -> tuple[IOPaintInvocationMetadata, ...]:
        return tuple(self._invocations)

    @property
    def last_invocation(self) -> IOPaintInvocationMetadata | None:
        return None if not self._invocations else self._invocations[-1]

    def __call__(self, rgb_crop: np.ndarray, mask: np.ndarray) -> np.ndarray:
        output = np.ascontiguousarray(255 - rgb_crop, dtype=np.uint8)

        def raw_hash(value: np.ndarray) -> str:
            return hashlib.sha256(
                np.ascontiguousarray(value).tobytes(order="C")
            ).hexdigest()

        environment_document = {
            "cudnn_benchmark": False,
            "cuda_available": False,
            "cuda_device_names": [],
            "cuda_runtime_version": None,
            "cuda_visible_devices": "unset",
            "cudnn_version": None,
            "deterministic_algorithms": False,
            "device": "cpu",
            "hip_runtime_version": None,
            "iopaint_source_manifest_sha256": hashlib.sha256(b"iopaint").hexdigest(),
            "iopaint_version": "1.6.0",
            "model_weights_sha256": self._weights_sha256,
            "mps_available": True,
            "mps_device_name": "Apple M4",
            "numpy_version": "1.26.4",
            "opencv_version": "4.11.0.86",
            "pillow_version": "9.5.0",
            "platform_machine": "arm64",
            "platform_release": "25.5.0",
            "platform_system": "Darwin",
            "pydantic_version": "2.13.4",
            "python_implementation": "CPython",
            "python_version": "3.11.15",
            "seed": 0,
            "thread_count": 1,
            "torch_version": "2.13.0",
            "typer_version": "0.27.0",
        }
        environment_sha256 = hashlib.sha256(
            json.dumps(
                environment_document,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
        ).hexdigest()
        runtime = IOPaintRuntimeMetadata(
            tool_name="IOPaint",
            tool_version="1.6.0",
            tool_license_spdx="Apache-2.0",
            iopaint_source_manifest_sha256=hashlib.sha256(b"iopaint").hexdigest(),
            iopaint_source_file_count=1,
            python_version="3.11.15",
            python_implementation="CPython",
            torch_version="2.13.0",
            numpy_version="1.26.4",
            pillow_version="9.5.0",
            opencv_version="4.11.0.86",
            pydantic_version="2.13.4",
            typer_version="0.27.0",
            platform_system="Darwin",
            platform_release="25.5.0",
            platform_machine="arm64",
            deterministic_algorithms=False,
            cudnn_benchmark=False,
            cuda_available=False,
            cuda_runtime_version=None,
            cudnn_version=None,
            hip_runtime_version=None,
            cuda_device_names=(),
            cuda_visible_devices="unset",
            mps_available=True,
            mps_device_name="Apple M4",
            effective_environment_sha256=environment_sha256,
            model_name="lama",
            model_release="Sanster/models:add_big_lama",
            model_artifact_identifier="test-big-lama.pt",
            model_weights_sha256=self._weights_sha256,
            model_upstream_license_spdx="Apache-2.0",
            model_artifact_license_status="unconfirmed for converted artifact",
            device="cpu",
            thread_count=1,
            seed=0,
            seed_scope="fixed request; LaMa is feed-forward",
            determinism_scope="single recorded environment",
        )
        self._invocations.append(
            IOPaintInvocationMetadata(
                runtime=runtime,
                sanitized_argv=(
                    "python",
                    "-I",
                    "-m",
                    "iopaint",
                    "run",
                    "--model",
                    "lama",
                    "--device",
                    "cpu",
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
                deterministic_environment=tuple(
                    sorted(
                        {
                            "DIFFUSERS_CACHE": (
                                "<private-model-cache>/huggingface/hub"
                            ),
                            "HF_HOME": "<private-model-cache>/huggingface",
                            "HF_HUB_CACHE": ("<private-model-cache>/huggingface/hub"),
                            "HF_HUB_OFFLINE": "1",
                            "HUGGINGFACE_HUB_CACHE": (
                                "<private-model-cache>/huggingface/hub"
                            ),
                            "LAMA_MODEL_URL": "<private-model-artifact>",
                            "MKL_DYNAMIC": "FALSE",
                            "MKL_NUM_THREADS": "1",
                            "NUMEXPR_NUM_THREADS": "1",
                            "OMP_DYNAMIC": "FALSE",
                            "OMP_NUM_THREADS": "1",
                            "OPENBLAS_NUM_THREADS": "1",
                            "TORCH_HOME": "<private-model-cache>/torch",
                            "TRANSFORMERS_CACHE": (
                                "<private-model-cache>/huggingface/hub"
                            ),
                            "TRANSFORMERS_OFFLINE": "1",
                            "VECLIB_MAXIMUM_THREADS": "1",
                            "XDG_CACHE_HOME": "<private-model-cache>",
                        }.items()
                    )
                ),
                config_document=(("hd_strategy", "Original"), ("sd_seed", 0)),
                input_rgb8_raw_sha256=raw_hash(rgb_crop),
                mask_u8_raw_sha256=raw_hash(mask),
                output_rgb8_raw_sha256=raw_hash(output),
            )
        )
        return output


class _MismatchedMetadataAdapter(_FakeIOPaintAdapter):
    def __init__(self, weights_sha256: str, field: str) -> None:
        super().__init__(weights_sha256)
        self._field = field

    def __call__(self, rgb_crop: np.ndarray, mask: np.ndarray) -> np.ndarray:
        output = super().__call__(rgb_crop, mask)
        self._invocations[-1] = replace(
            self._invocations[-1],
            **{self._field: "0" * 64},
        )
        return output


class _OversizedBatchAdapter(_FakeIOPaintAdapter):
    def __init__(self, weights_sha256: str, kind: str) -> None:
        super().__init__(weights_sha256)
        self._kind = kind
        self.values_read = 0

    def inpaint_batch(
        self,
        rgb_crops: tuple[np.ndarray, ...],
        _component_masks: tuple[np.ndarray, ...],
    ) -> object:
        def generate() -> object:
            while self._kind == "infinite" or self.values_read < 100:
                self.values_read += 1
                yield np.zeros_like(rgb_crops[0])

        return generate()


def _synthetic_rgbi(height: int, width: int, *, offset: int) -> np.ndarray:
    y, x = np.indices((height, width), dtype=np.uint32)
    pixels = np.empty((height, width, 4), dtype=np.uint16)
    pixels[:, :, 0] = (12_000 + offset + 53 * y + 97 * x).astype(np.uint16)
    pixels[:, :, 1] = (16_000 + offset + 67 * y + 71 * x).astype(np.uint16)
    pixels[:, :, 2] = (20_000 + offset + 89 * y + 43 * x).astype(np.uint16)
    pixels[:, :, 3] = (44_000 + offset + 31 * y + 29 * x).astype(np.uint16)
    return pixels


@pytest.fixture
def acquisition(tmp_path: Path) -> tuple[np.ndarray, np.ndarray, Path, Path]:
    prepass = _synthetic_rgbi(8, 8, offset=0)
    main = _synthetic_rgbi(8, 8, offset=150)
    main[3:5, 3:5, 3] = np.uint16(0)
    prepass_path = tmp_path / "prepass.npy"
    main_path = tmp_path / "main.npy"
    np.save(prepass_path, prepass, allow_pickle=False)
    np.save(main_path, main, allow_pickle=False)
    return prepass, main, prepass_path, main_path


def _arguments(
    *,
    prepass: Path,
    main: Path,
    output: Path,
    backend: str = "cpu",
) -> list[str]:
    return [
        "--prepass",
        str(prepass),
        "--main",
        str(main),
        "--out",
        str(output),
        "--same-frame-id",
        "synthetic-frame-1",
        "--assert-focus-exposure-locked",
        "--backend",
        backend,
        "--no-inpaint",
    ]


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_arguments(tmp_path: Path) -> tuple[list[str], Path, str]:
    model_dir = tmp_path / "model-cache"
    weights = model_dir / "torch" / "hub" / "checkpoints" / "big-lama.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"synthetic model weights for CLI integration")
    weights_sha256 = hashlib.sha256(weights.read_bytes()).hexdigest()
    return (
        [
            "--iopaint-python",
            str(tmp_path / "iopaint-runtime" / "bin" / "python"),
            "--iopaint-executable",
            str(tmp_path / "iopaint-runtime" / "bin" / "iopaint"),
            "--iopaint-source-manifest-sha256",
            hashlib.sha256(b"synthetic iopaint source").hexdigest(),
            "--model-dir",
            str(model_dir),
            "--model-weights",
            str(weights),
            "--model-weights-sha256",
            weights_sha256,
            "--model-artifact-id",
            "test-big-lama.pt",
        ],
        weights,
        weights_sha256,
    )


def _direct_job(prepass: np.ndarray, main: np.ndarray) -> ProcessingJob:
    return ProcessingJob(
        acquisition=DualRGBIAcquisition(
            prepass=RGBI16Frame(
                prepass,
                AcquisitionEpoch.PREPASS,
                285,
                "direct-prepass",
            ),
            main=RGBI16Frame(
                main,
                AcquisitionEpoch.MAIN,
                4_000,
                "direct-main",
            ),
            same_frame_id="synthetic-frame-1",
        ),
        scanner_model=ScannerModel.NIKON_SUPER_COOLSCAN_5000_ED,
        mode=ProcessingMode.NORMAL,
        selector=8,
        resolution_metric=4_000,
        bit_depth=16,
        focus_exposure_locked=True,
    )


def test_no_inpaint_output_matches_direct_cpu_and_has_only_phase_b_artifacts(
    acquisition: tuple[np.ndarray, np.ndarray, Path, Path],
    tmp_path: Path,
) -> None:
    prepass, main, prepass_path, main_path = acquisition
    output = tmp_path / "output"

    assert (
        cli.main(_arguments(prepass=prepass_path, main=main_path, output=output)) == 0
    )

    emitted = np.load(output / "output.rgb16.npy", allow_pickle=False)
    direct = process(_direct_job(prepass, main), backend="cpu")
    assert hash_rgb16(emitted) == direct.result.replay.output_sha256
    np.testing.assert_array_equal(emitted, direct.result.output_rgb16)
    assert {path.name for path in output.iterdir()} == {
        "output.rgb16.npy",
        "routing.json",
        "run-metadata.json",
    }
    assert not (output / "output-hybrid.rgb16.npy").exists()
    assert not (output / "synth-mask.png").exists()
    assert not (output / "hybrid-receipt.json").exists()

    routing_bytes = (output / "routing.json").read_bytes()
    routing = json.loads(routing_bytes)
    assert routing_bytes == canonical_json_bytes(routing)
    metadata_bytes = (output / "run-metadata.json").read_bytes()
    metadata = json.loads(metadata_bytes)
    assert metadata_bytes == canonical_json_bytes(metadata)
    assert metadata["mode"] == "routing_only_no_inpaint"
    assert metadata["backend"] == {
        "reason": "explicit CPU request",
        "requested": "cpu",
        "used": "cpu",
    }
    assert metadata["provenance"]["classification"] == ("caller_asserted_bare_npy")
    assert metadata["provenance"]["assertion"]["scanner_evidence"] is False
    assert metadata["inputs"]["prepass"]["raw_sha256"] == hash_rgbir16(prepass)
    assert metadata["inputs"]["main"]["raw_sha256"] == hash_rgbir16(main)


def test_no_inpaint_cpu_fast_backend_matches_cpu_and_records_canonical_reason(
    acquisition: tuple[np.ndarray, np.ndarray, Path, Path],
    tmp_path: Path,
) -> None:
    pytest.importorskip("numba")
    prepass, main, prepass_path, main_path = acquisition
    output = tmp_path / "cpu-fast-output"

    assert (
        cli.main(
            _arguments(
                prepass=prepass_path,
                main=main_path,
                output=output,
                backend="cpu-fast",
            )
        )
        == 0
    )

    emitted = np.load(output / "output.rgb16.npy", allow_pickle=False)
    direct = process(_direct_job(prepass, main), backend="cpu")
    assert hash_rgb16(emitted) == direct.result.replay.output_sha256
    np.testing.assert_array_equal(emitted, direct.result.output_rgb16)
    metadata = json.loads((output / "run-metadata.json").read_bytes())
    assert metadata["backend"] == {
        "reason": "explicit cpu-fast request; self-test passed byte parity",
        "requested": "cpu-fast",
        "used": "cpu-fast",
    }


def test_diagnostics_cache_round_trip_is_output_and_routing_identical(
    acquisition: tuple[np.ndarray, np.ndarray, Path, Path],
    tmp_path: Path,
) -> None:
    _, _, prepass_path, main_path = acquisition
    cache = tmp_path / "diagnostics"
    fresh_output = tmp_path / "fresh"
    cached_output = tmp_path / "cached"

    fresh_arguments = _arguments(
        prepass=prepass_path,
        main=main_path,
        output=fresh_output,
    )
    assert cli.main([*fresh_arguments, "--save-diagnostics", str(cache)]) == 0
    cached_arguments = _arguments(
        prepass=prepass_path,
        main=main_path,
        output=cached_output,
    )
    assert (
        cli.main(
            [
                *cached_arguments,
                "--from-diagnostics",
                str(cache),
                "--diagnostics-manifest-sha256",
                _file_sha256(cache / "diagnostics-cache.json"),
            ]
        )
        == 0
    )

    fresh = np.load(fresh_output / "output.rgb16.npy", allow_pickle=False)
    cached = np.load(cached_output / "output.rgb16.npy", allow_pickle=False)
    np.testing.assert_array_equal(fresh, cached)
    assert (fresh_output / "routing.json").read_bytes() == (
        cached_output / "routing.json"
    ).read_bytes()
    fresh_metadata = json.loads((fresh_output / "run-metadata.json").read_bytes())
    cached_metadata = json.loads((cached_output / "run-metadata.json").read_bytes())
    assert fresh_metadata["cache"]["mode"] == "saved"
    assert cached_metadata["cache"]["mode"] == "loaded"
    assert (
        fresh_metadata["cache"]["manifest_sha256"]
        == (cached_metadata["cache"]["manifest_sha256"])
    )


def test_diagnostics_manifest_anchor_is_required_and_mode_bound(
    acquisition: tuple[np.ndarray, np.ndarray, Path, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, _, prepass_path, main_path = acquisition
    cache = tmp_path / "diagnostics"
    assert (
        cli.main(
            [
                *_arguments(
                    prepass=prepass_path,
                    main=main_path,
                    output=tmp_path / "producer",
                ),
                "--save-diagnostics",
                str(cache),
            ]
        )
        == 0
    )
    digest = _file_sha256(cache / "diagnostics-cache.json")

    missing_output = tmp_path / "missing-anchor"
    with pytest.raises(SystemExit) as raised:
        cli.main(
            [
                *_arguments(
                    prepass=prepass_path,
                    main=main_path,
                    output=missing_output,
                ),
                "--from-diagnostics",
                str(cache),
            ]
        )
    assert raised.value.code == 2
    assert "requires --diagnostics-manifest-sha256" in capsys.readouterr().err
    assert not missing_output.exists()

    orphan_output = tmp_path / "orphan-anchor"
    with pytest.raises(SystemExit) as raised:
        cli.main(
            [
                *_arguments(
                    prepass=prepass_path,
                    main=main_path,
                    output=orphan_output,
                ),
                "--diagnostics-manifest-sha256",
                digest,
            ]
        )
    assert raised.value.code == 2
    assert "requires --from-diagnostics" in capsys.readouterr().err
    assert not orphan_output.exists()

    invalid_output = tmp_path / "invalid-anchor"
    with pytest.raises(SystemExit) as raised:
        cli.main(
            [
                *_arguments(
                    prepass=prepass_path,
                    main=main_path,
                    output=invalid_output,
                ),
                "--from-diagnostics",
                str(cache),
                "--diagnostics-manifest-sha256",
                "A" * 64,
            ]
        )
    assert raised.value.code == 2
    assert "lowercase 64-character SHA-256" in capsys.readouterr().err
    assert not invalid_output.exists()


def test_hybrid_rebuilds_embedded_cache_as_independent_snapshot(
    acquisition: tuple[np.ndarray, np.ndarray, Path, Path],
    tmp_path: Path,
) -> None:
    _, _, prepass_path, main_path = acquisition
    cache = tmp_path / "source-cache"
    routing_only = tmp_path / "routing-only"
    assert (
        cli.main(
            [
                *_arguments(
                    prepass=prepass_path,
                    main=main_path,
                    output=routing_only,
                ),
                "--save-diagnostics",
                str(cache),
            ]
        )
        == 0
    )
    output = tmp_path / "hybrid-from-cache"
    arguments = _arguments(
        prepass=prepass_path,
        main=main_path,
        output=output,
    )
    arguments.remove("--no-inpaint")
    arguments.extend(
        [
            "--from-diagnostics",
            str(cache),
            "--diagnostics-manifest-sha256",
            _file_sha256(cache / "diagnostics-cache.json"),
        ]
    )
    assert cli.main(arguments) == 0

    embedded = output / "diagnostics-cache"
    assert {path.name for path in embedded.iterdir()} == {
        "at-floor-mask.npy",
        "changed-mask.npy",
        "diagnostics-cache.json",
        "output.rgb16.npy",
        "score-plane.npy",
    }
    for source in cache.iterdir():
        assert source.stat().st_ino != (embedded / source.name).stat().st_ino
    verify_receipt(output / "hybrid-receipt.json")


def test_empty_hybrid_emits_verified_zero_mask_without_model_access(
    acquisition: tuple[np.ndarray, np.ndarray, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, prepass_path, main_path = acquisition
    output = tmp_path / "empty-hybrid"
    arguments = _arguments(
        prepass=prepass_path,
        main=main_path,
        output=output,
    )
    arguments.remove("--no-inpaint")

    def forbidden_adapter(_args: object) -> None:
        raise AssertionError("empty routing must not construct a model adapter")

    monkeypatch.setattr(cli, "_build_iopaint_adapter", forbidden_adapter)
    assert cli.main(arguments) == 0

    assert {path.name for path in output.iterdir()} == {
        "diagnostics-cache",
        "hybrid-receipt.json",
        "output-hybrid.rgb16.npy",
        "output.rgb16.npy",
        "routing.json",
        "run-metadata.json",
        "synth-mask.png",
    }
    pure = np.load(output / "output.rgb16.npy", allow_pickle=False)
    hybrid = np.load(output / "output-hybrid.rgb16.npy", allow_pickle=False)
    np.testing.assert_array_equal(hybrid, pure)
    assert (output / "output.rgb16.npy").stat().st_ino != (
        output / "diagnostics-cache" / "output.rgb16.npy"
    ).stat().st_ino
    mask = iio.imread(output / "synth-mask.png", plugin="pillow")
    assert mask.dtype == np.uint8
    assert not np.any(mask)
    verified = verify_receipt(output / "hybrid-receipt.json")
    assert verified.model_weights_rehashed is False
    assert not np.any(verified.synthesis_mask)
    metadata = json.loads((output / "run-metadata.json").read_bytes())
    assert metadata["mode"] == "hybrid_empty_synthesis"
    assert metadata["generative_model_loaded"] is False
    assert metadata["inpainting"] == {
        "invoked": False,
        "reason": "empty_synthesis_mask",
    }


def test_nonempty_hybrid_is_mask_exact_and_receipt_replays(
    acquisition: tuple[np.ndarray, np.ndarray, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, prepass_path, main_path = acquisition
    output = tmp_path / "hybrid"
    model_arguments, weights, weights_sha256 = _model_arguments(tmp_path)
    fake = _FakeIOPaintAdapter(weights_sha256)
    monkeypatch.setattr(cli, "_build_iopaint_adapter", lambda _args: fake)
    arguments = _arguments(
        prepass=prepass_path,
        main=main_path,
        output=output,
    )
    arguments.remove("--no-inpaint")
    arguments.extend(
        [
            "--min-area",
            "1",
            "--min-radius",
            "100",
            "--margin",
            "0",
            "--max-synth-fraction",
            "1.0",
            *model_arguments,
        ]
    )

    assert cli.main(arguments) == 0
    assert len(fake.invocations) == 1
    pure = np.load(output / "output.rgb16.npy", allow_pickle=False)
    hybrid = np.load(output / "output-hybrid.rgb16.npy", allow_pickle=False)
    mask_u8 = iio.imread(output / "synth-mask.png", plugin="pillow")
    mask = mask_u8 == 255
    assert np.any(mask)
    np.testing.assert_array_equal(hybrid[~mask], pure[~mask])
    assert any(hybrid[mask].ravel() != pure[mask].ravel())
    assert {path.name for path in (output / "components").iterdir()} == {
        "0001-inpainted.png",
        "0001-input.png",
        "0001-mask.png",
    }
    verified = verify_receipt(
        output / "hybrid-receipt.json",
        model_weights_resolver=lambda _attestation: weights,
        require_model_weights=True,
    )
    assert verified.model_weights_rehashed is True
    np.testing.assert_array_equal(verified.hybrid_output_rgb16, hybrid)

    tampered = mask_u8.copy()
    tampered[0, 0] = np.uint8(255 if tampered[0, 0] == 0 else 0)
    iio.imwrite(output / "synth-mask.png", tampered, plugin="pillow")
    with pytest.raises(ReceiptError, match="SHA-256"):
        verify_receipt(
            output / "hybrid-receipt.json",
            model_weights_resolver=lambda _attestation: weights,
            require_model_weights=True,
        )


@pytest.mark.parametrize("kind", ("overlong", "infinite"))
def test_untrusted_batch_output_is_bounded_and_never_published(
    acquisition: tuple[np.ndarray, np.ndarray, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    kind: str,
) -> None:
    _, _, prepass_path, main_path = acquisition
    output = tmp_path / f"{kind}-batch-output"
    model_arguments, _, weights_sha256 = _model_arguments(tmp_path)
    fake = _OversizedBatchAdapter(weights_sha256, kind)
    monkeypatch.setattr(cli, "_build_iopaint_adapter", lambda _args: fake)
    arguments = _arguments(
        prepass=prepass_path,
        main=main_path,
        output=output,
    )
    arguments.remove("--no-inpaint")
    arguments.extend(
        [
            "--min-area",
            "1",
            "--min-radius",
            "100",
            "--margin",
            "0",
            "--max-synth-fraction",
            "1.0",
            *model_arguments,
        ]
    )

    with pytest.raises(SystemExit) as raised:
        cli.main(arguments)

    assert raised.value.code == 2
    assert "batch model output count does not match crop count" in (
        capsys.readouterr().err
    )
    # One routed crop permits reading exactly one result plus one sentinel.
    assert fake.values_read == 2
    assert not output.exists()


def test_repeated_hybrid_runs_have_identical_model_artifacts(
    acquisition: tuple[np.ndarray, np.ndarray, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, prepass_path, main_path = acquisition
    model_arguments, _, weights_sha256 = _model_arguments(tmp_path)
    outputs = (tmp_path / "repeat-a", tmp_path / "repeat-b")

    for output in outputs:
        fake = _FakeIOPaintAdapter(weights_sha256)
        monkeypatch.setattr(cli, "_build_iopaint_adapter", lambda _args, f=fake: f)
        arguments = _arguments(
            prepass=prepass_path,
            main=main_path,
            output=output,
        )
        arguments.remove("--no-inpaint")
        arguments.extend(
            [
                "--min-area",
                "1",
                "--min-radius",
                "100",
                "--margin",
                "0",
                "--max-synth-fraction",
                "1.0",
                *model_arguments,
            ]
        )
        assert cli.main(arguments) == 0

    repeated_files = (
        "output-hybrid.rgb16.npy",
        "synth-mask.png",
        "components/0001-input.png",
        "components/0001-mask.png",
        "components/0001-inpainted.png",
    )
    for relative_path in repeated_files:
        assert (outputs[0] / relative_path).read_bytes() == (
            outputs[1] / relative_path
        ).read_bytes()

    receipts = [
        json.loads((output / "hybrid-receipt.json").read_bytes()) for output in outputs
    ]
    assert receipts[0]["inpainting"]["model"] == receipts[1]["inpainting"]["model"]
    assert (
        receipts[0]["composite"]["components"]
        == (receipts[1]["composite"]["components"])
    )


@pytest.mark.parametrize(
    "field",
    (
        "input_rgb8_raw_sha256",
        "mask_u8_raw_sha256",
        "output_rgb8_raw_sha256",
    ),
)
def test_mismatched_model_metadata_fails_before_publish(
    acquisition: tuple[np.ndarray, np.ndarray, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    field: str,
) -> None:
    _, _, prepass_path, main_path = acquisition
    output = tmp_path / f"mismatched-{field}"
    model_arguments, _, weights_sha256 = _model_arguments(tmp_path)
    fake = _MismatchedMetadataAdapter(weights_sha256, field)
    monkeypatch.setattr(cli, "_build_iopaint_adapter", lambda _args: fake)
    arguments = _arguments(
        prepass=prepass_path,
        main=main_path,
        output=output,
    )
    arguments.remove("--no-inpaint")
    arguments.extend(
        [
            "--min-area",
            "1",
            "--min-radius",
            "100",
            "--margin",
            "0",
            "--max-synth-fraction",
            "1.0",
            *model_arguments,
        ]
    )

    with pytest.raises(SystemExit) as raised:
        cli.main(arguments)
    assert raised.value.code == 2
    assert "IOPaint metadata does not match composite component" in (
        capsys.readouterr().err
    )
    assert not output.exists()


def test_nonempty_hybrid_reports_missing_iopaint_without_publishing(
    acquisition: tuple[np.ndarray, np.ndarray, Path, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, _, prepass_path, main_path = acquisition
    output = tmp_path / "missing-iopaint"
    model_arguments, _, _ = _model_arguments(tmp_path)
    arguments = _arguments(
        prepass=prepass_path,
        main=main_path,
        output=output,
    )
    arguments.remove("--no-inpaint")
    arguments.extend(
        [
            "--min-area",
            "1",
            "--min-radius",
            "100",
            "--margin",
            "0",
            "--max-synth-fraction",
            "1.0",
            *model_arguments,
        ]
    )

    with pytest.raises(SystemExit) as raised:
        cli.main(arguments)
    assert raised.value.code == 2
    assert "IOPaint executable is unavailable" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize(
    ("flag", "value", "message"),
    (
        ("--model-artifact-id", "bad/artifact.pt", "sanitized_artifact_id"),
        (
            "--model-external-reference",
            "/private/model-cache/big-lama.pt",
            "cannot be an absolute filesystem path",
        ),
    ),
)
def test_receipt_invalid_model_identity_fails_before_adapter_construction(
    acquisition: tuple[np.ndarray, np.ndarray, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    flag: str,
    value: str,
    message: str,
) -> None:
    _, _, prepass_path, main_path = acquisition
    output = tmp_path / f"invalid-model-{flag.removeprefix('--')}"
    model_arguments, _, _ = _model_arguments(tmp_path)
    constructed = False

    def forbidden_adapter(_config: object) -> None:
        nonlocal constructed
        constructed = True
        raise AssertionError("invalid receipt metadata must fail first")

    monkeypatch.setattr(cli, "IOPaintLaMaAdapter", forbidden_adapter)
    arguments = _arguments(
        prepass=prepass_path,
        main=main_path,
        output=output,
    )
    arguments.remove("--no-inpaint")
    arguments.extend(
        [
            "--min-area",
            "1",
            "--min-radius",
            "100",
            "--margin",
            "0",
            "--max-synth-fraction",
            "1.0",
            *model_arguments,
            flag,
            value,
        ]
    )

    with pytest.raises(SystemExit) as raised:
        cli.main(arguments)
    assert raised.value.code == 2
    assert message in capsys.readouterr().err
    assert constructed is False
    assert not output.exists()


@pytest.mark.parametrize(
    ("kind", "message"),
    (
        (
            "missing",
            "non-empty synthesis requires model arguments: "
            "--iopaint-source-manifest-sha256",
        ),
        ("invalid", "must be a lowercase 64-character SHA-256 hex digest"),
    ),
)
def test_iopaint_source_manifest_anchor_fails_preflight(
    acquisition: tuple[np.ndarray, np.ndarray, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    kind: str,
    message: str,
) -> None:
    _, _, prepass_path, main_path = acquisition
    output = tmp_path / f"{kind}-iopaint-source-anchor"
    model_arguments, _, _ = _model_arguments(tmp_path)
    flag_index = model_arguments.index("--iopaint-source-manifest-sha256")
    if kind == "missing":
        del model_arguments[flag_index : flag_index + 2]
    else:
        model_arguments[flag_index + 1] = "A" * 64

    constructed = False

    def forbidden_adapter(_config: object) -> None:
        nonlocal constructed
        constructed = True
        raise AssertionError("invalid source anchor must fail before construction")

    monkeypatch.setattr(cli, "IOPaintLaMaAdapter", forbidden_adapter)
    arguments = _arguments(
        prepass=prepass_path,
        main=main_path,
        output=output,
    )
    arguments.remove("--no-inpaint")
    arguments.extend(
        [
            "--min-area",
            "1",
            "--min-radius",
            "100",
            "--margin",
            "0",
            "--max-synth-fraction",
            "1.0",
            *model_arguments,
        ]
    )

    with pytest.raises(SystemExit) as raised:
        cli.main(arguments)

    assert raised.value.code == 2
    assert message in capsys.readouterr().err
    assert constructed is False
    assert not output.exists()


def test_budget_and_context_fail_before_model_invocation(
    acquisition: tuple[np.ndarray, np.ndarray, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, _, prepass_path, main_path = acquisition
    fake = _FakeIOPaintAdapter(hashlib.sha256(b"unused").hexdigest())
    monkeypatch.setattr(cli, "_build_iopaint_adapter", lambda _args: fake)

    budget_output = tmp_path / "over-budget"
    budget_arguments = _arguments(
        prepass=prepass_path,
        main=main_path,
        output=budget_output,
    )
    budget_arguments.remove("--no-inpaint")
    budget_arguments.extend(
        [
            "--min-area",
            "1",
            "--min-radius",
            "100",
            "--margin",
            "0",
            "--max-synth-fraction",
            "0.01",
        ]
    )
    with pytest.raises(SystemExit) as raised:
        cli.main(budget_arguments)
    assert raised.value.code == 2
    assert "exceeds configured budget" in capsys.readouterr().err
    assert not fake.invocations
    assert not budget_output.exists()

    context_output = tmp_path / "no-context"
    context_arguments = _arguments(
        prepass=prepass_path,
        main=main_path,
        output=context_output,
    )
    context_arguments.remove("--no-inpaint")
    context_arguments.extend(
        [
            "--min-area",
            "1",
            "--min-radius",
            "100",
            "--margin",
            "8",
            "--max-synth-fraction",
            "1.0",
        ]
    )
    with pytest.raises(SystemExit) as raised:
        cli.main(context_arguments)
    assert raised.value.code == 2
    assert "no pixels outside the global synthesis mask" in capsys.readouterr().err
    assert not fake.invocations
    assert not context_output.exists()


def test_auto_full_run_cuda_unavailable_reruns_cpu_and_records_reason(
    acquisition: tuple[np.ndarray, np.ndarray, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, prepass_path, main_path = acquisition
    output = tmp_path / "auto-fallback"
    calls: list[ComputeBackend] = []
    direct_process = process

    def simulated_process(
        job: ProcessingJob,
        *,
        backend: ComputeBackend | str,
        export_diagnostics: bool,
    ):
        selected = ComputeBackend(backend)
        calls.append(selected)
        if selected is ComputeBackend.AUTO:
            raise CudaBackendUnavailable("simulated full-frame allocation failure")
        return direct_process(
            job,
            backend=selected,
            export_diagnostics=export_diagnostics,
        )

    monkeypatch.setattr(cli, "process_digital_ice", simulated_process)
    assert (
        cli.main(
            _arguments(
                prepass=prepass_path,
                main=main_path,
                output=output,
                backend="auto",
            )
        )
        == 0
    )

    assert calls == [ComputeBackend.AUTO, ComputeBackend.CPU]
    metadata = json.loads((output / "run-metadata.json").read_bytes())
    assert metadata["backend"]["requested"] == "auto"
    assert metadata["backend"]["used"] == "cpu"
    assert metadata["backend"]["reason"] == (
        "CUDA unavailable; complete job ran on exact CPU reference"
    )
    assert "simulated full-frame allocation failure" not in json.dumps(metadata)


def test_manifest_claims_are_hashed_and_must_match_inputs(
    acquisition: tuple[np.ndarray, np.ndarray, Path, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepass, main, prepass_path, main_path = acquisition
    good_manifest = {
        "focus_exposure_locked": True,
        "main_raw_sha256": hash_rgbir16(main),
        "prepass_raw_sha256": hash_rgbir16(prepass),
        "same_frame_id": "synthetic-frame-1",
    }
    manifest_path = tmp_path / "acquisition.json"
    payload = canonical_json_bytes(good_manifest)
    manifest_path.write_bytes(payload)
    output = tmp_path / "manifest-output"
    arguments = _arguments(
        prepass=prepass_path,
        main=main_path,
        output=output,
    )
    assert cli.main([*arguments, "--acquisition-manifest", str(manifest_path)]) == 0
    metadata = json.loads((output / "run-metadata.json").read_bytes())
    assert (
        metadata["provenance"]["acquisition_manifest_sha256"]
        == hashlib.sha256(payload).hexdigest()
    )

    bad_manifest = dict(good_manifest, main_raw_sha256="0" * 64)
    bad_path = tmp_path / "bad-acquisition.json"
    bad_path.write_bytes(canonical_json_bytes(bad_manifest))
    bad_output = tmp_path / "bad-manifest-output"
    with pytest.raises(SystemExit) as raised:
        cli.main(
            [
                *_arguments(
                    prepass=prepass_path,
                    main=main_path,
                    output=bad_output,
                ),
                "--acquisition-manifest",
                str(bad_path),
            ]
        )
    assert raised.value.code == 2
    assert "main_raw_sha256 does not match" in capsys.readouterr().err
    assert not bad_output.exists()


def test_bad_dtype_existing_output_and_missing_input_fail_before_publish(
    acquisition: tuple[np.ndarray, np.ndarray, Path, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, _, prepass_path, main_path = acquisition
    bad_prepass = tmp_path / "bad-prepass.npy"
    np.save(bad_prepass, np.zeros((8, 8, 4), dtype=np.uint8), allow_pickle=False)
    bad_output = tmp_path / "bad-dtype-output"
    with pytest.raises(SystemExit) as raised:
        cli.main(_arguments(prepass=bad_prepass, main=main_path, output=bad_output))
    assert raised.value.code == 2
    assert "prepass input must have dtype uint16" in capsys.readouterr().err
    assert not bad_output.exists()

    existing_output = tmp_path / "existing"
    existing_output.mkdir()
    with pytest.raises(SystemExit) as raised:
        cli.main(
            _arguments(
                prepass=prepass_path,
                main=main_path,
                output=existing_output,
            )
        )
    assert raised.value.code == 2
    assert "output directory already exists" in capsys.readouterr().err

    no_inpaint_output = tmp_path / "no-inpaint-missing"
    arguments = _arguments(
        prepass=tmp_path / "does-not-exist-prepass.npy",
        main=tmp_path / "does-not-exist-main.npy",
        output=no_inpaint_output,
    )
    arguments.remove("--no-inpaint")
    with pytest.raises(SystemExit) as raised:
        cli.main(arguments)
    assert raised.value.code == 2
    error = capsys.readouterr().err
    assert "cannot load prepass array" in error
    assert not no_inpaint_output.exists()


def test_atomic_publish_never_replaces_a_racing_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "racing-output"
    original_write_npy = cli._write_npy

    def write_then_create_racer(path: Path, array: np.ndarray) -> None:
        original_write_npy(path, array)
        destination.mkdir()
        (destination / "racer-owned.txt").write_text(
            "must survive",
            encoding="utf-8",
        )

    monkeypatch.setattr(cli, "_write_npy", write_then_create_racer)
    with pytest.raises(cli.HybridCLIError, match="cannot publish output directory"):
        cli._write_output_directory(
            destination,
            output_rgb16=np.zeros((2, 2, 3), dtype=np.uint16),
            routing_document={"schema": "test-routing"},
            metadata_document={"schema": "test-metadata"},
            core_source_manifest_sha256=cli.compute_core_source_manifest(),
            hybrid_source_manifest_sha256=cli.compute_hybrid_source_manifest(),
        )

    assert {path.name for path in destination.iterdir()} == {"racer-owned.txt"}
    assert (destination / "racer-owned.txt").read_text(encoding="utf-8") == (
        "must survive"
    )


def test_atomic_publish_refuses_an_existing_empty_directory(tmp_path: Path) -> None:
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "output.txt").write_text("staged", encoding="utf-8")
    destination = tmp_path / "existing-empty"
    destination.mkdir()
    destination_inode = destination.stat().st_ino

    with pytest.raises(FileExistsError):
        cli._atomic_publish_directory(staged, destination)

    assert destination.is_dir()
    assert destination.stat().st_ino == destination_inode
    assert not any(destination.iterdir())
    assert (staged / "output.txt").read_text(encoding="utf-8") == "staged"


def test_source_manifest_drift_fails_before_publication(
    acquisition: tuple[np.ndarray, np.ndarray, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, _, prepass_path, main_path = acquisition
    output = tmp_path / "source-drift"
    captured = cli.compute_hybrid_source_manifest()
    manifests = iter((captured, "f" * 64))
    monkeypatch.setattr(
        cli,
        "compute_hybrid_source_manifest",
        lambda: next(manifests),
    )

    with pytest.raises(SystemExit) as raised:
        cli.main(_arguments(prepass=prepass_path, main=main_path, output=output))
    assert raised.value.code == 2
    assert "hybrid source changed during the run" in capsys.readouterr().err
    assert not output.exists()


def test_staging_creation_failure_is_clean_and_fail_closed(
    acquisition: tuple[np.ndarray, np.ndarray, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, _, prepass_path, main_path = acquisition
    output = tmp_path / "unwritable-staging"

    def fail_staging(*_args: object, **_kwargs: object) -> str:
        raise PermissionError("private host path detail")

    monkeypatch.setattr(cli.tempfile, "mkdtemp", fail_staging)
    with pytest.raises(SystemExit) as raised:
        cli.main(_arguments(prepass=prepass_path, main=main_path, output=output))
    assert raised.value.code == 2
    error = capsys.readouterr().err
    assert "cannot create the private output staging directory" in error
    assert "private host path detail" not in error
    assert not output.exists()
