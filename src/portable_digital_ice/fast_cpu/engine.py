"""Compiled execution of the supported streaming profile (optional backend).

The reconstruction math ported here is byte-identical to the CPU reference in
``streaming.py`` and ``reconstruction.py``; only the per-attempted-pixel
scalar chain runs through compiled kernels instead of per-call Python.  Row
analysis (response lookup, auxiliary/score/weighted planes, decision
eligibility, history-window construction), stage-parameter resolution,
digesting, and callbacks stay in Python so behavior outside the compiled
chain is unchanged.

Importing this module never requires numba.  Only calling into the compiled
kernels does, and that failure is raised as :class:`CpuFastUnavailable` with a
specific reason instead of silently falling back to a different code path.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import replace

import numpy as np
import numpy.typing as npt

from ..dither import DitherBounds
from ..engine import (
    CancellationCallback,
    ProcessingDiagnostics,
    ProcessingPhase,
    ProcessingResult,
    _check_cancelled,
    _notify,
)
from ..output import emit_public_rgb16
from ..prepass import reduce_prepass_frame
from ..producer_parameters import (
    ContentDerivedStageParameterProvider,
    derive_producer_record_schedule,
)
from ..profile import DEFAULT_PROFILE, ProcessingJob
from ..reconstruction import FeatureBandExtremaMode, feature_band_extrema_mode
from ..rng import LCG24
from ..stage_parameters import (
    StageParameterProvider,
    validate_stage_calibration,
    writer_stage_for_row,
)
from ..streaming import (
    DiagnosticsRowCallback,
    StreamingReplayResult,
    _decision_fallback_row,
    _history_window,
    _RowCache,
    _startup_replay,
)
from ..x3a import (
    AuxiliaryParameters,
    DecisionParameters,
    ScoreParameters,
    SharedLookupInputResponse,
)


class CpuFastUnavailable(RuntimeError):
    """The compiled CPU backend was requested but cannot run exactly here."""


def _kernels():
    """Import the njit kernel module lazily so package import stays numba-free."""

    try:
        from . import kernels
    except Exception as error:  # pragma: no cover - import environment
        raise CpuFastUnavailable(f"numba is not importable: {error!r}") from error
    return kernels


RowProgressCallback = Callable[[int, int, int, int], None]
ProgressCallback = Callable[[object], None]


def run_streaming_replay_fast(
    main_rgbi: npt.ArrayLike,
    output_rgb16: npt.NDArray[np.uint16],
    *,
    response: SharedLookupInputResponse,
    auxiliary_parameters: AuxiliaryParameters,
    score_parameters: ScoreParameters,
    decision_parameters: DecisionParameters,
    reconstruction_parameters,
    dither_bounds: DitherBounds,
    generator: LCG24 | None = None,
    stage_parameter_provider: StageParameterProvider | None = None,
    progress: RowProgressCallback | None = None,
    diagnostics_row: DiagnosticsRowCallback | None = None,
) -> StreamingReplayResult:
    """Byte-exact compiled mirror of ``streaming.run_streaming_replay``.

    Row analysis (response lookup, auxiliary/score/weighted planes, decision
    eligibility, history-window construction) and the per-row emit/digest
    stay identical Python/NumPy; only the per-selected-pixel scalar chain
    (feature records, driver gate, candidates, combiner, writer) runs
    through the fused njit kernel in ``kernels.py``.
    """

    kernels = _kernels()

    pixels = np.asarray(main_rgbi)
    output = np.asarray(output_rgb16)
    if pixels.dtype != np.dtype(np.uint16) or pixels.ndim != 3 or pixels.shape[2] != 4:
        raise ValueError("streaming main input must be uint16 HxWx4")
    height, width, _ = pixels.shape
    if height < 4:
        raise ValueError("streaming startup requires at least four main rows")
    if output.dtype != np.dtype(np.uint16) or output.shape != (height, width, 3):
        raise ValueError("streaming output must be uint16 HxWx3")
    if not output.flags.writeable:
        raise ValueError("streaming output must be writable")

    cache = _RowCache(
        pixels,
        response=response,
        auxiliary_parameters=auxiliary_parameters,
        score_parameters=score_parameters,
        stage_parameter_provider=stage_parameter_provider,
    )
    active_generator = generator or LCG24.from_nikon_pe_initial_state()
    startup = _startup_replay(
        cache,
        width=width,
        score_parameters=score_parameters,
        decision_parameters=decision_parameters,
        reconstruction_parameters=reconstruction_parameters,
        dither_bounds=dither_bounds,
        generator=active_generator,
        stage_parameter_provider=stage_parameter_provider,
    )
    mode = feature_band_extrema_mode(
        resolution_metric=reconstruction_parameters.resolution_metric,
        cross_neighbor_cutoff=reconstruction_parameters.cross_neighbor_cutoff,
    )
    cross_neighbor = mode is FeatureBandExtremaMode.CROSS_NEIGHBOR
    record_count = 5 if cross_neighbor else 1
    score_floor = np.float32(score_parameters.floor)

    # Stable across the whole run: stage_parameter_provider only ever
    # replaces coarse_reference/driver_gate_secondary/row_reconstruction_gate
    # per row (see stage_parameters.py); everything else below is fixed.
    coarse_slopes = np.ascontiguousarray(
        reconstruction_parameters.coarse_slopes, dtype=np.float32
    )
    band_enabled = np.ascontiguousarray(
        reconstruction_parameters.band_enabled, dtype=np.uint8
    )
    band_scales = np.ascontiguousarray(
        reconstruction_parameters.band_scales, dtype=np.float32
    )
    factors_a = np.ascontiguousarray(reconstruction_parameters.factors_a, dtype=np.float32)
    factors_b = np.ascontiguousarray(reconstruction_parameters.factors_b, dtype=np.float32)
    configured_strengths = np.ascontiguousarray(
        reconstruction_parameters.configured_strengths, dtype=np.float32
    )
    dither_scales = np.ascontiguousarray(
        reconstruction_parameters.dither_scales, dtype=np.float32
    )
    coarse_enabled = bool(reconstruction_parameters.coarse_enabled)
    driver_gate_primary = bool(reconstruction_parameters.driver_gate_primary)

    low64 = float(np.float32(dither_bounds.low))
    high64 = float(np.float32(dither_bounds.high))
    low_lt_high = low64 < high64

    attempted = 0
    written = 0
    public_advances = 0
    changed = 0
    output_hash = hashlib.sha256()
    state = int(active_generator.state)

    # Reused scratch buffers: only ever read back through the neighbor_valid5
    # gate, so stale content from a prior pixel/row is never observed.
    score_patches = np.zeros((5, 9, 9), dtype=np.float32)
    waux_patches = np.zeros((5, 9, 9), dtype=np.float32)
    wrgb_center = np.zeros((9, 9, 3), dtype=np.float32)
    point_aux5 = np.zeros(5, dtype=np.float32)
    point_score5 = np.zeros(5, dtype=np.float32)
    neighbor_valid5 = np.zeros(5, dtype=np.uint8)
    neighbor_valid5[0] = 1
    if cross_neighbor:
        neighbor_valid5[3] = 1
        neighbor_valid5[4] = 1

    for y in range(height):
        active_reconstruction_parameters = reconstruction_parameters
        if stage_parameter_provider is not None:
            stage_hit = writer_stage_for_row(y)
            writer_calibration = stage_parameter_provider(stage_hit)
            validate_stage_calibration(writer_calibration, expected_stage_hit=stage_hit)
            active_reconstruction_parameters = replace(
                reconstruction_parameters,
                coarse_reference=writer_calibration.base_primary,
                driver_gate_secondary=writer_calibration.writer_gate_secondary,
                row_reconstruction_gate=writer_calibration.row_reconstruction_gate,
            )
        floor_enabled = (
            driver_gate_primary or active_reconstruction_parameters.driver_gate_secondary
        )
        fallback_value = np.float32(active_reconstruction_parameters.coarse_reference)
        row_reconstruction_gate = int(active_reconstruction_parameters.row_reconstruction_gate)

        current = cache.get(y)
        noop = emit_public_rgb16(current.working[np.newaxis, :, :3])[0]
        working_output = np.array(current.working[:, :3], dtype=np.float32, copy=True)
        decision_fallback = _decision_fallback_row(
            y,
            height=height,
            width=width,
            cache=cache,
            parameters=decision_parameters,
        )
        eligible = ~decision_fallback
        if active_reconstruction_parameters.row_reconstruction_gate != 0:
            eligible[:] = False
        elif floor_enabled:
            eligible &= current.score < np.float32(1.0)
        selected_x = np.flatnonzero(eligible)
        score_history, weighted_auxiliary_history, weighted_rgb_history = (
            _history_window(
                y,
                height=height,
                width=width,
                cache=cache,
                score_floor=score_floor,
            )
        )

        if cross_neighbor:
            point_aux_center = current.auxiliary
            point_score_center = current.score
            if y == 0:
                point_aux_up = cache.get(0).auxiliary
                point_score_up = np.full(width, score_floor, dtype=np.float32)
            else:
                up_row = cache.get(y - 1)
                point_aux_up = up_row.auxiliary
                point_score_up = up_row.score
            down_row = cache.get(height - 1 if y == height - 1 else y + 1)
            point_aux_down = down_row.auxiliary
            point_score_down = down_row.score

        for x_value in selected_x:
            x = int(x_value)
            score_patches[0] = score_history[1:10, x : x + 9]
            waux_patches[0] = weighted_auxiliary_history[1:10, x : x + 9]
            wrgb_center[:] = weighted_rgb_history[1:10, x : x + 9, :]
            point_aux5[0] = current.auxiliary[x]
            point_score5[0] = current.score[x]

            if cross_neighbor:
                if x > 0:
                    score_patches[1] = score_history[1:10, x - 1 : x - 1 + 9]
                    waux_patches[1] = weighted_auxiliary_history[1:10, x - 1 : x - 1 + 9]
                    point_aux5[1] = point_aux_center[x - 1]
                    point_score5[1] = point_score_center[x - 1]
                    neighbor_valid5[1] = 1
                else:
                    neighbor_valid5[1] = 0
                if x < width - 1:
                    score_patches[2] = score_history[1:10, x + 1 : x + 1 + 9]
                    waux_patches[2] = weighted_auxiliary_history[1:10, x + 1 : x + 1 + 9]
                    point_aux5[2] = point_aux_center[x + 1]
                    point_score5[2] = point_score_center[x + 1]
                    neighbor_valid5[2] = 1
                else:
                    neighbor_valid5[2] = 0
                score_patches[3] = score_history[0:9, x : x + 9]
                waux_patches[3] = weighted_auxiliary_history[0:9, x : x + 9]
                point_aux5[3] = point_aux_up[x]
                point_score5[3] = point_score_up[x]
                score_patches[4] = score_history[2:11, x : x + 9]
                waux_patches[4] = weighted_auxiliary_history[2:11, x : x + 9]
                point_aux5[4] = point_aux_down[x]
                point_score5[4] = point_score_down[x]

            original_rgb = np.ascontiguousarray(current.working[x, :3])

            attempted_flag, values, advances, state = kernels.process_selected_pixel(
                score_patches,
                waux_patches,
                wrgb_center,
                point_aux5,
                point_score5,
                neighbor_valid5,
                record_count,
                fallback_value,
                original_rgb,
                floor_enabled,
                row_reconstruction_gate,
                coarse_enabled,
                coarse_slopes,
                band_enabled,
                band_scales,
                factors_a,
                factors_b,
                configured_strengths,
                low64,
                high64,
                low_lt_high,
                dither_scales,
                state,
            )
            if not attempted_flag:
                continue
            attempted += 1
            working_output[x] = values
            public_advances += int(advances)
            written += int(np.any(values != original_rgb))

        rendered = emit_public_rgb16(working_output[np.newaxis, :, :])[0]
        output[y] = rendered
        output_hash.update(rendered.astype("<u2", copy=False).tobytes(order="C"))
        changed += int(np.count_nonzero(np.any(rendered != noop, axis=1)))
        if diagnostics_row is not None:
            diagnostics_row(y, current.score, score_floor, rendered, noop)
        if progress is not None:
            progress(y + 1, height, attempted, written)
        cache.discard_before(max(0, y - 5))

    active_generator.state = state
    return StreamingReplayResult(
        shape=(height, width, 3),
        startup=startup,
        attempted_pixels=attempted,
        written_pixels=written,
        public_rng_advances=public_advances,
        final_rng_state=active_generator.state,
        output_sha256=output_hash.hexdigest(),
        changed_pixels=changed,
    )


def process_cpu_fast(
    job: ProcessingJob,
    *,
    output_rgb16: npt.NDArray[np.uint16] | None = None,
    progress: ProgressCallback | None = None,
    cancelled: CancellationCallback | None = None,
    export_diagnostics: bool = False,
) -> ProcessingResult:
    """Process one validated LS-5000 selector-8 Normal acquisition on the
    compiled CPU backend.

    Byte-identical to :func:`portable_digital_ice.engine.process_cpu`; only
    the streaming row work is replaced by ``run_streaming_replay_fast``.
    """

    _notify(progress, ProcessingPhase.VALIDATING, 0, 1)
    DEFAULT_PROFILE.validate_job(job)
    main_pixels = job.acquisition.main.pixels
    expected_shape = (*main_pixels.shape[:2], 3)
    destination: npt.NDArray[np.uint16] | None = None
    if output_rgb16 is not None:
        destination = np.asarray(output_rgb16)
        if destination.dtype != np.dtype(np.uint16) or destination.shape != expected_shape:
            raise ValueError(f"output must be writable uint16 with shape {expected_shape}")
        if not destination.flags.writeable:
            raise ValueError("output must be writable")
    _check_cancelled(cancelled)
    _notify(progress, ProcessingPhase.VALIDATING, 1, 1)

    response = SharedLookupInputResponse.nikon_logarithmic()
    _notify(progress, ProcessingPhase.PREPASS, 0, 1)
    prepass = reduce_prepass_frame(
        job.acquisition.prepass,
        parameters=DEFAULT_PROFILE.prepass_parameters(),
        frame_index=0,
        evidence_id=f"portable-prepass:{job.acquisition.same_frame_id}",
        response=response,
    )
    _check_cancelled(cancelled)
    _notify(progress, ProcessingPhase.PREPASS, 1, 1)

    _notify(progress, ProcessingPhase.PRODUCER, 0, 1)
    schedule = derive_producer_record_schedule(main_pixels, response.table)
    provider = ContentDerivedStageParameterProvider(
        schedule,
        auxiliary_factor_b=DEFAULT_PROFILE.auxiliary_factor_b,
        hold_last_through_stage=job.acquisition.main.height + 6,
    )
    _check_cancelled(cancelled)
    _notify(progress, ProcessingPhase.PRODUCER, 1, 1)

    working_output = np.empty(expected_shape, dtype=np.uint16)
    score_plane: npt.NDArray[np.float32] | None = None
    diagnostic_score_floor: np.float32 | None = None
    at_floor_mask: npt.NDArray[np.bool_] | None = None
    changed_mask: npt.NDArray[np.bool_] | None = None
    diagnostics_row = None
    if export_diagnostics:
        diagnostic_shape = main_pixels.shape[:2]
        score_plane = np.empty(diagnostic_shape, dtype=np.float32)
        diagnostic_score_floor = np.float32(
            DEFAULT_PROFILE.score_parameters(prepass.record).floor
        )
        at_floor_mask = np.empty(diagnostic_shape, dtype=bool)
        changed_mask = np.empty(diagnostic_shape, dtype=bool)

        def capture_diagnostics_row(
            y: int,
            score: npt.NDArray[np.float32],
            score_floor: np.float32,
            rendered: npt.NDArray[np.uint16],
            noop: npt.NDArray[np.uint16],
        ) -> None:
            assert score_plane is not None
            assert diagnostic_score_floor is not None
            assert at_floor_mask is not None
            assert changed_mask is not None
            if score_floor.view(np.uint32) != diagnostic_score_floor.view(np.uint32):
                raise RuntimeError("diagnostic score floor disagrees with replay")
            score_plane[y] = score
            at_floor_mask[y] = score == score_floor
            changed_mask[y] = np.any(rendered != noop, axis=1)

        diagnostics_row = capture_diagnostics_row

    def row_progress(
        completed: int,
        total: int,
        attempted: int,
        written: int,
    ) -> None:
        _check_cancelled(cancelled)
        _notify(
            progress,
            ProcessingPhase.RECONSTRUCTION,
            completed,
            total,
            attempted,
            written,
        )

    _notify(
        progress,
        ProcessingPhase.RECONSTRUCTION,
        0,
        job.acquisition.main.height,
    )
    replay = run_streaming_replay_fast(
        main_pixels,
        working_output,
        response=response,
        auxiliary_parameters=DEFAULT_PROFILE.auxiliary_parameters(prepass.record),
        score_parameters=DEFAULT_PROFILE.score_parameters(prepass.record),
        decision_parameters=DEFAULT_PROFILE.decision_parameters(),
        reconstruction_parameters=DEFAULT_PROFILE.reconstruction_parameters(
            prepass.record
        ),
        dither_bounds=DEFAULT_PROFILE.dither_bounds(response.table),
        stage_parameter_provider=provider,
        progress=row_progress,
        diagnostics_row=diagnostics_row,
    )
    _check_cancelled(cancelled)
    diagnostics = None
    if export_diagnostics:
        assert score_plane is not None
        assert diagnostic_score_floor is not None
        assert at_floor_mask is not None
        assert changed_mask is not None
        score_plane.setflags(write=False)
        at_floor_mask.setflags(write=False)
        changed_mask.setflags(write=False)
        diagnostics = ProcessingDiagnostics(
            score_plane=score_plane,
            score_floor=diagnostic_score_floor,
            at_floor_mask=at_floor_mask,
            changed_mask=changed_mask,
        )
    if destination is None:
        output = working_output
    else:
        np.copyto(destination, working_output)
        output = destination
    _notify(progress, ProcessingPhase.COMPLETE, 1, 1)
    return ProcessingResult(
        output_rgb16=output,
        replay=replay,
        profile_id=DEFAULT_PROFILE.profile_id,
        diagnostics=diagnostics,
    )
