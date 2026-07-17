"""Level 2: adversarial synthetic tiles must match the CPU reference byte-for-byte.

Each case runs the complete job (or the streaming layer directly for the
metric-500 parameter shape) on CPU and CUDA and requires identical output
bytes, counters, RNG accounting, and repeated-run determinism.
"""

from __future__ import annotations

import numpy as np
import pytest

cupy = pytest.importorskip("cupy", reason="CUDA backend requires cupy")

from portable_digital_ice.contracts import (  # noqa: E402
    AcquisitionEpoch,
    DualRGBIAcquisition,
    RGBI16Frame,
)
from portable_digital_ice.cuda_backend.engine import (  # noqa: E402
    CudaBackendUnavailable,
    cuda_device_summary,
    run_streaming_replay_cuda,
)
from portable_digital_ice.dither import DitherBounds  # noqa: E402
from portable_digital_ice.engine import process_cpu  # noqa: E402
from portable_digital_ice.profile import (  # noqa: E402
    ProcessingJob,
    ProcessingMode,
    ScannerModel,
)
from portable_digital_ice.reconstruction import ReconstructionParameters  # noqa: E402
from portable_digital_ice.rng import LCG24  # noqa: E402
from portable_digital_ice.streaming import run_streaming_replay  # noqa: E402
from portable_digital_ice.x3a import (  # noqa: E402
    AuxiliaryParameters,
    DecisionParameters,
    ScoreParameters,
    SharedLookupInputResponse,
)


def _require_device() -> None:
    try:
        cuda_device_summary()
    except CudaBackendUnavailable as error:
        pytest.skip(f"CUDA device unavailable: {error}")


def _job(main: np.ndarray, prepass: np.ndarray, frame_id: str) -> ProcessingJob:
    return ProcessingJob(
        acquisition=DualRGBIAcquisition(
            prepass=RGBI16Frame(prepass, AcquisitionEpoch.PREPASS, 285, f"{frame_id}-pp"),
            main=RGBI16Frame(main, AcquisitionEpoch.MAIN, 4000, f"{frame_id}-main"),
            same_frame_id=frame_id,
        ),
        scanner_model=ScannerModel.NIKON_SUPER_COOLSCAN_5000_ED,
        mode=ProcessingMode.NORMAL,
        selector=8,
        resolution_metric=4000,
        bit_depth=16,
        focus_exposure_locked=True,
    )


def _assert_job_parity(main: np.ndarray, prepass: np.ndarray, frame_id: str) -> None:
    from portable_digital_ice.cuda_backend import process_cuda

    cpu = process_cpu(_job(main, prepass, frame_id))
    gpu = process_cuda(_job(main, prepass, frame_id))
    assert gpu.replay.output_sha256 == cpu.replay.output_sha256, frame_id
    assert np.array_equal(gpu.output_rgb16, cpu.output_rgb16), frame_id
    assert gpu.replay.attempted_pixels == cpu.replay.attempted_pixels, frame_id
    assert gpu.replay.written_pixels == cpu.replay.written_pixels, frame_id
    assert gpu.replay.public_rng_advances == cpu.replay.public_rng_advances, frame_id
    assert gpu.replay.final_rng_state == cpu.replay.final_rng_state, frame_id
    assert gpu.replay.changed_pixels == cpu.replay.changed_pixels, frame_id
    assert gpu.replay.startup == cpu.replay.startup, frame_id

    repeat = process_cuda(_job(main, prepass, frame_id))
    assert repeat.replay.output_sha256 == gpu.replay.output_sha256, frame_id
    assert repeat.replay.final_rng_state == gpu.replay.final_rng_state, frame_id


def _prepass(rng: np.random.Generator) -> np.ndarray:
    prepass = rng.integers(15000, 62000, size=(24, 32, 4), dtype=np.uint16)
    prepass[6:10, 8:13, 3] = 400
    return prepass


@pytest.mark.parametrize(
    "case",
    [
        "uniform",
        "impulse",
        "edge",
        "checkerboard",
        "high_dynamic_range",
        "corner_dust",
        "edge_band_dust",
        "epoch_boundary_dust",
        "threshold_straddle",
        "final_block_six_rows",
    ],
)
def test_adversarial_tiles_match_cpu(case: str) -> None:
    _require_device()
    rng = np.random.default_rng(abs(hash(case)) % (2**32))
    height, width = 41, 64  # 41 = 5 epochs + 1, final block has 1 valid row
    if case == "final_block_six_rows":
        height = 30  # 3 complete epochs + six-valid-row final block

    main = rng.integers(18000, 62000, size=(height, width, 4), dtype=np.uint16)
    if case == "uniform":
        main[:] = 43210
        main[:, :, 3] = 51234
    elif case == "impulse":
        main[:] = 40000
        main[height // 2, width // 2] = (65535, 0, 65535, 60000)
        main[height // 2, width // 2 + 1, 3] = 100
    elif case == "edge":
        main[:, : width // 2] = (15000, 16000, 17000, 52000)
        main[:, width // 2 :] = (60000, 59000, 58000, 52000)
        main[:, width // 2 - 1 : width // 2 + 1, 3] = 800
    elif case == "checkerboard":
        yy, xx = np.mgrid[0:height, 0:width]
        board = ((yy + xx) % 2).astype(np.uint16)
        for c in range(3):
            main[:, :, c] = 20000 + board * 30000
        main[:, :, 3] = 50000 + board * 10000
    elif case == "high_dynamic_range":
        main[:, :, :3] = rng.choice(
            np.array([0, 1, 2, 32768, 65533, 65534, 65535], dtype=np.uint16),
            size=(height, width, 3),
        )
        main[:, :, 3] = rng.choice(
            np.array([0, 8847, 8848, 8849, 65535], dtype=np.uint16),
            size=(height, width),
        )
    elif case == "corner_dust":
        for (y0, x0) in ((0, 0), (0, width - 3), (height - 3, 0), (height - 3, width - 3)):
            main[y0 : y0 + 3, x0 : x0 + 3, 3] = 500
    elif case == "edge_band_dust":
        main[0:4, :, 3] = rng.integers(100, 3000, size=(4, width), dtype=np.uint16)
        main[:, 0:4, 3] = rng.integers(100, 3000, size=(height, 4), dtype=np.uint16)
        main[:, width - 4 :, 3] = rng.integers(
            100, 3000, size=(height, 4), dtype=np.uint16
        )
        main[height - 4 :, :, 3] = rng.integers(
            100, 3000, size=(4, width), dtype=np.uint16
        )
    elif case == "epoch_boundary_dust":
        main[7:9, :, 3] = 600  # straddles the first eight-row producer epoch
        main[15:17, 24:40, 3] = 700
        main[23:25, :, 3] = rng.integers(100, 9000, size=(2, width), dtype=np.uint16)
    elif case == "threshold_straddle":
        main[:, :, 3] = rng.choice(
            np.array([8846, 8847, 8848, 8849, 8850], dtype=np.uint16),
            size=(height, width),
        )
    elif case == "final_block_six_rows":
        main[height - 6 :, :, 3] = rng.integers(
            100, 4000, size=(6, width), dtype=np.uint16
        )

    _assert_job_parity(main, _prepass(rng), f"tile-{case}")


def test_metric_500_center_only_streaming_parity() -> None:
    """Exercise the CENTER_ONLY feature-band mode shared with metric 500."""

    _require_device()
    rng = np.random.default_rng(500500)
    height, width = 26, 48
    pixels = rng.integers(15000, 64000, size=(height, width, 4), dtype=np.uint16)
    pixels[9:14, 20:29, 3] = 700
    response = SharedLookupInputResponse.nikon_logarithmic()
    auxiliary = AuxiliaryParameters(
        selected_visible_channel=0,
        alpha=float(np.float32(0.32)),
        calibration_offset=1.0,
        alpha_one_replacement=None,
    )
    score = ScoreParameters(
        base_primary=float(np.float32(52215.4)),
        base_addend=float(np.float32(-119.40625)),
        scale=float(np.float32(-0.0010412337)),
        offset=1.0,
        floor=float(np.float32(0.02)),
        resolution_metric=500,
        horizontal_minimum_resolution_cutoff=550,
    )
    decision = DecisionParameters(
        sample_threshold=float(np.float32(49383.234)),
        count_limit=8,
        perpendicular_radius=1,
    )
    reconstruction = ReconstructionParameters(
        resolution_metric=500,
        cross_neighbor_cutoff=1600,
        coarse_enabled=True,
        coarse_reference=float(np.float32(52215.4)),
        coarse_slopes=(1.1, 1.1, 1.1),
        band_enabled=(True, True, True),
        band_scales=(1.25, 1.25, 1.25),
        factors_a=((1.21, 1.23, 1.13), (1.17, 1.14, 1.08), (1.04, 0.93, 0.97)),
        factors_b=((1.09, 1.13, 1.04), (1.08, 1.05, 1.02), (0.96, 0.84, 0.89)),
        configured_strengths=(0.0, 0.0, 0.0),
        driver_gate_primary=True,
        driver_gate_secondary=True,
        row_reconstruction_gate=0,
        dither_scales=(0.015, 0.015, 0.025),
    )
    bounds = DitherBounds.from_lookup(response.table, maximum_index=65535)

    cpu_output = np.empty((height, width, 3), dtype=np.uint16)
    cpu_replay = run_streaming_replay(
        pixels,
        cpu_output,
        response=response,
        auxiliary_parameters=auxiliary,
        score_parameters=score,
        decision_parameters=decision,
        reconstruction_parameters=reconstruction,
        dither_bounds=bounds,
        generator=LCG24.from_nikon_pe_initial_state(),
    )
    gpu_output = np.empty((height, width, 3), dtype=np.uint16)
    gpu_replay = run_streaming_replay_cuda(
        pixels,
        gpu_output,
        response=response,
        auxiliary_parameters=auxiliary,
        score_parameters=score,
        decision_parameters=decision,
        reconstruction_parameters=reconstruction,
        dither_bounds=bounds,
        generator=LCG24.from_nikon_pe_initial_state(),
    )
    assert np.array_equal(gpu_output, cpu_output)
    assert gpu_replay == cpu_replay


def test_writer_gate_and_row_gate_branches() -> None:
    """Conditional-writer branch coverage: floor gates off, row gate on."""

    _require_device()
    rng = np.random.default_rng(99)
    height, width = 17, 40
    pixels = rng.integers(15000, 64000, size=(height, width, 4), dtype=np.uint16)
    pixels[5:9, 10:19, 3] = 900
    response = SharedLookupInputResponse.nikon_logarithmic()
    bounds = DitherBounds.from_lookup(response.table, maximum_index=65535)
    base = dict(
        response=response,
        auxiliary_parameters=AuxiliaryParameters(
            selected_visible_channel=0,
            alpha=float(np.float32(0.4)),
            calibration_offset=1.0,
            alpha_one_replacement=None,
        ),
        score_parameters=ScoreParameters(
            base_primary=float(np.float32(52215.4)),
            base_addend=float(np.float32(-119.40625)),
            scale=float(np.float32(-0.0010412337)),
            offset=1.0,
            floor=float(np.float32(0.02)),
            resolution_metric=4000,
            horizontal_minimum_resolution_cutoff=550,
        ),
        decision_parameters=DecisionParameters(
            sample_threshold=float(np.float32(49383.234)),
            count_limit=8,
            perpendicular_radius=4,
        ),
        dither_bounds=bounds,
    )
    for label, gates in (
        ("floor-both-off", dict(driver_gate_primary=False, driver_gate_secondary=False)),
        ("row-gate-on", dict(row_reconstruction_gate=3)),
    ):
        reconstruction = ReconstructionParameters(
            resolution_metric=4000,
            cross_neighbor_cutoff=1600,
            coarse_enabled=True,
            coarse_reference=float(np.float32(52215.4)),
            coarse_slopes=(1.1, 1.1, 1.1),
            band_enabled=(True, True, True),
            band_scales=(1.25, 1.25, 1.25),
            factors_a=((1.21, 1.23, 1.13), (1.17, 1.14, 1.08), (1.04, 0.93, 0.97)),
            factors_b=((1.09, 1.13, 1.04), (1.08, 1.05, 1.02), (0.96, 0.84, 0.89)),
            configured_strengths=(0.0, 0.0, 0.0),
            driver_gate_primary=True,
            driver_gate_secondary=True,
            row_reconstruction_gate=0,
            dither_scales=(0.015, 0.015, 0.025),
            **{},
        )
        from dataclasses import replace

        reconstruction = replace(reconstruction, **gates)
        cpu_output = np.empty((height, width, 3), dtype=np.uint16)
        cpu_replay = run_streaming_replay(
            pixels,
            cpu_output,
            reconstruction_parameters=reconstruction,
            generator=LCG24.from_nikon_pe_initial_state(),
            **base,
        )
        gpu_output = np.empty((height, width, 3), dtype=np.uint16)
        gpu_replay = run_streaming_replay_cuda(
            pixels,
            gpu_output,
            reconstruction_parameters=reconstruction,
            generator=LCG24.from_nikon_pe_initial_state(),
            **base,
        )
        assert np.array_equal(gpu_output, cpu_output), label
        assert gpu_replay == cpu_replay, label
