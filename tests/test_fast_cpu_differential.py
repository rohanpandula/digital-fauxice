"""cpu-fast vs the CPU reference: byte-exact differential gates (milestone M1/M2).

Every scenario here runs both `process_cpu`/`run_streaming_replay` (the exact
reference) and `process_cpu_fast`/`run_streaming_replay_fast` (the compiled
path) on the same input and asserts output bytes, every replay counter
(attempted/written/changed/public_rng_advances/final_rng_state), the startup
receipt, and (where exercised) diagnostics planes are identical.  No
tolerances anywhere -- byte equality is the only pass criterion.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("numba")

from portable_digital_ice.backend import _synthetic_self_test_job  # noqa: E402
from portable_digital_ice.contracts import (  # noqa: E402
    AcquisitionEpoch,
    DualRGBIAcquisition,
    RGBI16Frame,
)
from portable_digital_ice.dither import DitherBounds  # noqa: E402
from portable_digital_ice.engine import process_cpu  # noqa: E402
from portable_digital_ice.fast_cpu import process_cpu_fast  # noqa: E402
from portable_digital_ice.fast_cpu.engine import run_streaming_replay_fast  # noqa: E402
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


def _assert_job_parity(
    main: np.ndarray, prepass: np.ndarray, frame_id: str, *, export_diagnostics: bool = False
) -> None:
    cpu = process_cpu(_job(main, prepass, frame_id), export_diagnostics=export_diagnostics)
    fast = process_cpu_fast(
        _job(main, prepass, frame_id), export_diagnostics=export_diagnostics
    )
    assert fast.replay.output_sha256 == cpu.replay.output_sha256, frame_id
    assert np.array_equal(fast.output_rgb16, cpu.output_rgb16), frame_id
    assert fast.replay == cpu.replay, frame_id

    if export_diagnostics:
        assert cpu.diagnostics is not None and fast.diagnostics is not None, frame_id
        assert np.array_equal(
            cpu.diagnostics.score_plane.view(np.uint32),
            fast.diagnostics.score_plane.view(np.uint32),
        ), frame_id
        assert cpu.diagnostics.score_floor.view(np.uint32) == (
            fast.diagnostics.score_floor.view(np.uint32)
        ), frame_id
        assert np.array_equal(
            cpu.diagnostics.at_floor_mask, fast.diagnostics.at_floor_mask
        ), frame_id
        assert np.array_equal(
            cpu.diagnostics.changed_mask, fast.diagnostics.changed_mask
        ), frame_id

    # Determinism: repeated runs of the fast path agree with each other and
    # with the (already-verified) reference hash.
    repeat = process_cpu_fast(_job(main, prepass, frame_id))
    assert repeat.replay.output_sha256 == cpu.replay.output_sha256, frame_id
    assert repeat.replay.final_rng_state == cpu.replay.final_rng_state, frame_id


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
    """Mirrors the CUDA adversarial-tile suite, comparing fast vs CPU instead."""

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

    _assert_job_parity(main, _prepass(rng), f"tile-{case}", export_diagnostics=(case == "corner_dust"))


def test_synthetic_self_test_job_matches_cpu() -> None:
    """The exact job backend.py uses for its own self-test/warmup."""

    job = _synthetic_self_test_job()
    cpu = process_cpu(job)
    fast = process_cpu_fast(job)
    assert fast.replay == cpu.replay
    assert np.array_equal(fast.output_rgb16, cpu.output_rgb16)


def _direct_recon_kwargs(**overrides) -> dict:
    base = dict(
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
    base.update(overrides)
    return base


def _direct_parity(
    pixels: np.ndarray,
    *,
    auxiliary: AuxiliaryParameters | None = None,
    score: ScoreParameters | None = None,
    decision: DecisionParameters | None = None,
    reconstruction: ReconstructionParameters | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = pixels.shape[:2]
    response = SharedLookupInputResponse.nikon_logarithmic()
    auxiliary = auxiliary or AuxiliaryParameters(
        selected_visible_channel=0,
        alpha=float(np.float32(0.32)),
        calibration_offset=1.0,
        alpha_one_replacement=None,
    )
    score = score or ScoreParameters(
        base_primary=float(np.float32(52215.4)),
        base_addend=float(np.float32(-119.40625)),
        scale=float(np.float32(-0.0010412337)),
        offset=1.0,
        floor=float(np.float32(0.02)),
        resolution_metric=reconstruction.resolution_metric if reconstruction else 500,
        horizontal_minimum_resolution_cutoff=550,
    )
    decision = decision or DecisionParameters(
        sample_threshold=float(np.float32(49383.234)),
        count_limit=8,
        perpendicular_radius=1,
    )
    reconstruction = reconstruction or ReconstructionParameters(**_direct_recon_kwargs())
    bounds = DitherBounds.from_lookup(response.table, maximum_index=65535)

    ref_out = np.empty((height, width, 3), dtype=np.uint16)
    ref_replay = run_streaming_replay(
        pixels,
        ref_out,
        response=response,
        auxiliary_parameters=auxiliary,
        score_parameters=score,
        decision_parameters=decision,
        reconstruction_parameters=reconstruction,
        dither_bounds=bounds,
        generator=LCG24.from_nikon_pe_initial_state(),
    )
    fast_out = np.empty((height, width, 3), dtype=np.uint16)
    fast_replay = run_streaming_replay_fast(
        pixels,
        fast_out,
        response=response,
        auxiliary_parameters=auxiliary,
        score_parameters=score,
        decision_parameters=decision,
        reconstruction_parameters=reconstruction,
        dither_bounds=bounds,
        generator=LCG24.from_nikon_pe_initial_state(),
    )
    assert np.array_equal(ref_out, fast_out)
    assert ref_replay == fast_replay
    return ref_out, fast_out


def test_metric_500_center_only_streaming_parity() -> None:
    """Exercise the CENTER_ONLY feature-band mode shared with metric 500."""

    rng = np.random.default_rng(500500)
    height, width = 26, 48
    pixels = rng.integers(15000, 64000, size=(height, width, 4), dtype=np.uint16)
    pixels[9:14, 20:29, 3] = 700
    _direct_parity(pixels)


def test_cross_neighbor_mode_with_dust() -> None:
    """Same tile, but metric 4000 selects the CROSS_NEIGHBOR feature-band mode."""

    rng = np.random.default_rng(500501)
    height, width = 26, 48
    pixels = rng.integers(15000, 64000, size=(height, width, 4), dtype=np.uint16)
    pixels[9:14, 20:29, 3] = 700
    reconstruction = ReconstructionParameters(
        **_direct_recon_kwargs(resolution_metric=4000, cross_neighbor_cutoff=1600)
    )
    _direct_parity(pixels, reconstruction=reconstruction)


def test_writer_gate_and_row_gate_branches() -> None:
    """Conditional-writer branch coverage: floor gates off, row gate on."""

    rng = np.random.default_rng(99)
    height, width = 17, 40
    pixels = rng.integers(15000, 64000, size=(height, width, 4), dtype=np.uint16)
    pixels[5:9, 10:19, 3] = 900
    for gates in (
        dict(driver_gate_primary=False, driver_gate_secondary=False),
        dict(row_reconstruction_gate=3),
    ):
        reconstruction = ReconstructionParameters(
            **_direct_recon_kwargs(resolution_metric=4000, cross_neighbor_cutoff=1600, **gates)
        )
        _direct_parity(pixels, reconstruction=reconstruction)


def test_minimal_height_center_only() -> None:
    """height=4 is the reference's documented floor (CENTER_ONLY: no 5-row need)."""

    rng = np.random.default_rng(4004)
    pixels = rng.integers(15000, 64000, size=(4, 12, 4), dtype=np.uint16)
    pixels[1:3, 4:8, 3] = 600
    reconstruction = ReconstructionParameters(
        **_direct_recon_kwargs(resolution_metric=500, cross_neighbor_cutoff=1600)
    )
    _direct_parity(pixels, reconstruction=reconstruction)


def test_minimal_geometry_cross_neighbor() -> None:
    """CROSS_NEIGHBOR mode additionally requires 5 real rows for startup."""

    rng = np.random.default_rng(4005)
    pixels = rng.integers(15000, 64000, size=(5, 8, 4), dtype=np.uint16)
    pixels[1:3, 2:6, 3] = 600
    reconstruction = ReconstructionParameters(
        **_direct_recon_kwargs(resolution_metric=4000, cross_neighbor_cutoff=1600)
    )
    _direct_parity(pixels, reconstruction=reconstruction)


def test_minimal_width_one_center_only() -> None:
    """width=1: every cross-neighbor would be a horizontal boundary case."""

    rng = np.random.default_rng(1001)
    pixels = rng.integers(15000, 64000, size=(6, 1, 4), dtype=np.uint16)
    pixels[2:4, 0, 3] = 500
    reconstruction = ReconstructionParameters(
        **_direct_recon_kwargs(resolution_metric=500, cross_neighbor_cutoff=1600)
    )
    _direct_parity(pixels, reconstruction=reconstruction)


def test_all_fallback_row() -> None:
    """Every pixel in every row triggers the decision-fallback group gate."""

    rng = np.random.default_rng(2002)
    height, width = 10, 20
    pixels = rng.integers(15000, 64000, size=(height, width, 4), dtype=np.uint16)
    pixels[:, :, 3] = 500  # uniformly below sample_threshold everywhere
    reconstruction = ReconstructionParameters(
        **_direct_recon_kwargs(resolution_metric=4000, cross_neighbor_cutoff=1600)
    )
    ref_out, fast_out = _direct_parity(pixels, reconstruction=reconstruction)
    # Sanity: this scenario is meant to zero out reconstruction entirely.
    assert np.array_equal(ref_out, fast_out)


def test_row_reconstruction_gate_active_with_dust() -> None:
    """row_reconstruction_gate != 0 zeroes eligibility regardless of content."""

    rng = np.random.default_rng(3003)
    height, width = 12, 24
    pixels = rng.integers(15000, 64000, size=(height, width, 4), dtype=np.uint16)
    pixels[4:8, 8:16, 3] = 500  # would otherwise drive reconstruction
    reconstruction = ReconstructionParameters(
        **_direct_recon_kwargs(
            resolution_metric=4000, cross_neighbor_cutoff=1600, row_reconstruction_gate=7
        )
    )
    _direct_parity(pixels, reconstruction=reconstruction)


def test_alpha_one_replacement_path() -> None:
    """alpha == 1.0 takes the runtime-configured replacement branch."""

    rng = np.random.default_rng(7007)
    height, width = 14, 24
    pixels = rng.integers(15000, 64000, size=(height, width, 4), dtype=np.uint16)
    pixels[5:9, 10:18, 3] = 650
    auxiliary = AuxiliaryParameters(
        selected_visible_channel=0,
        alpha=1.0,
        calibration_offset=1.0,
        alpha_one_replacement=float(np.float32(41234.5)),
    )
    reconstruction = ReconstructionParameters(
        **_direct_recon_kwargs(resolution_metric=4000, cross_neighbor_cutoff=1600)
    )
    _direct_parity(pixels, auxiliary=auxiliary, reconstruction=reconstruction)


def test_diagnostics_planes_match_cpu() -> None:
    """export_diagnostics=True: score/at-floor/changed planes bitwise equal."""

    rng = np.random.default_rng(20260722)
    height, width = 30, 40
    main = rng.integers(18000, 62000, size=(height, width, 4), dtype=np.uint16)
    main[8:14, 12:22, 3] = 550
    main[0:2, 0:6, 3] = 500
    main[-2:, -6:, 3] = 700
    prepass = _prepass(rng)
    _assert_job_parity(main, prepass, "diagnostics-check", export_diagnostics=True)


def test_cancellation_leaves_output_untouched() -> None:
    """A cancelled fast run must not mutate the caller-owned output buffer."""

    from portable_digital_ice.engine import ProcessingCancelled

    job = _synthetic_self_test_job()
    output = np.full((job.acquisition.main.height, job.acquisition.main.width, 3), 12345, dtype=np.uint16)
    calls = {"n": 0}

    def cancel_after_start() -> bool:
        calls["n"] += 1
        return calls["n"] >= 5

    with pytest.raises(ProcessingCancelled):
        process_cpu_fast(job, output_rgb16=output, cancelled=cancel_after_start)
    assert np.array_equal(output, np.full_like(output, 12345))
