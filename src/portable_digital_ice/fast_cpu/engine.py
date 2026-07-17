"""Compiled execution of the supported streaming profile (optional backend).

The reconstruction math ported here is byte-identical to the CPU reference in
``streaming.py`` and ``reconstruction.py``.  Row cache products (response
lookup, auxiliary/score/weighted planes, per-row stage-parameter resolution)
stay in Python via the reference's own ``_RowCache``; decision eligibility,
history-window boundary handling, feature records, candidates, and the
combiner run as one parallel kernel call per band of rows, followed by the
strictly serial writer.  Emit, digest, and callbacks stay per-row in Python;
cooperative cancellation is honored per row while the analysis planes are
prepared and per band during writing.

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
from ..output import InverseResponseFactors, emit_public_rgb16
from ..prepass import reduce_prepass_frame
from ..producer_parameters import ContentDerivedStageParameterProvider
from ..profile import DEFAULT_PROFILE, ProcessingJob
from ..reconstruction import FeatureBandExtremaMode, feature_band_extrema_mode
from ..rng import LCG24
from ..stage_parameters import (
    StageParameterProvider,
    validate_stage_calibration,
    writer_stage_for_row,
)
from ..startup import (
    HIDDEN_CENTERS,
    StartupReplayResult,
    _startup_histories,
    _validate_planes,
)
from ..streaming import (
    DiagnosticsRowCallback,
    StreamingReplayResult,
    _RowCache,
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

# Rows analyzed per parallel band.  Bounds the candidate buffers (a band of
# 128 rows at width 3946 holds ~12 MB of float64 candidates) while giving the
# analysis prange enough rows to occupy every thread.
_BAND_ROWS = 128


def _startup_replay_fast(
    kernels,
    *,
    working_all: np.ndarray,
    auxiliary_all: np.ndarray,
    score_all: np.ndarray,
    width: int,
    score_parameters: ScoreParameters,
    decision_parameters: DecisionParameters,
    reconstruction_parameters,
    generator: LCG24,
    stage_parameter_provider: StageParameterProvider | None,
    cross_neighbor: bool,
    coarse_slopes: np.ndarray,
    band_enabled: np.ndarray,
    band_scales: np.ndarray,
    factors_a: np.ndarray,
    factors_b: np.ndarray,
    configured_strengths: np.ndarray,
    dither_scales: np.ndarray,
    low64: float,
    high64: float,
    low_lt_high: bool,
) -> StartupReplayResult:
    """Compiled mirror of ``startup.replay_hidden_startup_rows``.

    Histories come from the reference's own ``_startup_histories`` helper;
    per-stage calibration resolution replicates the reference's
    ``reconstruction_parameters_for_stage`` closure, including the distinct
    producer/neighbor stages that supply each feature record's fallback.
    """

    guard = 4
    first_row_count = 5 if cross_neighbor else 4
    available = min(first_row_count, working_all.shape[0])
    rgb = np.ascontiguousarray(working_all[:available, :, :3])
    auxiliary = np.ascontiguousarray(auxiliary_all[:available])
    score = np.ascontiguousarray(score_all[:available])
    rgb, auxiliary, score = _validate_planes(rgb, auxiliary, score)
    if cross_neighbor and rgb.shape[0] < 5:
        raise ValueError("cross-neighbor startup requires the first five real rows")
    floor = np.float32(score_parameters.floor)
    if not np.isfinite(floor) or floor <= np.float32(0.0):
        raise ValueError("startup score floor must be finite and positive")
    (
        score_history,
        weighted_auxiliary_history,
        weighted_rgb_history,
        raw_auxiliary_history,
        minimum_y,
    ) = _startup_histories(rgb, auxiliary, score, score_floor=floor, guard=guard)

    def resolve(stage_hit: int):
        if stage_parameter_provider is None:
            return reconstruction_parameters
        calibration = stage_parameter_provider(stage_hit)
        validate_stage_calibration(calibration, expected_stage_hit=stage_hit)
        return replace(
            reconstruction_parameters,
            coarse_reference=calibration.base_primary,
            driver_gate_secondary=calibration.writer_gate_secondary,
            row_reconstruction_gate=calibration.row_reconstruction_gate,
        )

    threshold = np.float32(decision_parameters.sample_threshold)
    radius = int(decision_parameters.perpendicular_radius)
    count_limit = int(decision_parameters.count_limit)
    record_count = 5 if cross_neighbor else 1
    coarse_enabled = bool(reconstruction_parameters.coarse_enabled)
    original_rgb_row = np.ascontiguousarray(rgb[0])
    attempted_per_stage: list[int] = []
    advances_per_stage: list[int] = []
    state = int(generator.state)

    for center_y in HIDDEN_CENTERS:
        stage_hit = center_y - HIDDEN_CENTERS[0] + 1
        active = resolve(stage_hit)
        producer_parameters = resolve(max(1, center_y + 6))
        up_parameters = resolve(max(1, center_y + 5))
        down_parameters = resolve(max(1, center_y + 7))
        writer_width = (
            width - 4 if cross_neighbor and stage_hit <= 2 else width
        )
        floor_enabled = bool(
            active.driver_gate_primary or active.driver_gate_secondary
        )
        attempted, advances, state = kernels.startup_stage(
            score_history,
            weighted_auxiliary_history,
            weighted_rgb_history,
            raw_auxiliary_history,
            center_y,
            minimum_y,
            width,
            writer_width,
            guard,
            HIDDEN_CENTERS[0],
            radius,
            threshold,
            count_limit,
            record_count,
            np.float32(producer_parameters.coarse_reference),
            np.float32(producer_parameters.coarse_reference),
            np.float32(producer_parameters.coarse_reference),
            np.float32(up_parameters.coarse_reference),
            np.float32(down_parameters.coarse_reference),
            np.float32(active.coarse_reference),
            floor_enabled,
            int(active.row_reconstruction_gate),
            original_rgb_row,
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
        attempted_per_stage.append(int(attempted))
        advances_per_stage.append(int(advances))

    generator.state = state
    return StartupReplayResult(
        attempted_per_stage=tuple(attempted_per_stage),  # type: ignore[arg-type]
        rng_advances_per_stage=tuple(advances_per_stage),  # type: ignore[arg-type]
        final_rng_state=state,
    )


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
    cancelled: CancellationCallback | None = None,
) -> StreamingReplayResult:
    """Byte-exact compiled mirror of ``streaming.run_streaming_replay``.

    Row cache products (response lookup, auxiliary/score/weighted planes via
    ``_RowCache.get``, which resolves per-row stage-parameter swaps exactly
    as the reference does) stay identical Python/NumPy and are materialized
    once into whole-image planes, with the cooperative ``cancelled``
    callback honored per row during that preparation.  Analysis then runs as
    one parallel kernel call per band of rows (``kernels.analyze_band``)
    followed by the strictly serial writer (``kernels.write_band``); emit,
    digest, and callbacks stay per-row in Python, so cancellation during
    writing is honored at band granularity (via the progress callback, as in
    the reference engine).  ``process_cpu_fast`` commits the caller-owned
    output only after successful completion, so a cancelled run leaves that
    buffer untouched.
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
    mode = feature_band_extrema_mode(
        resolution_metric=reconstruction_parameters.resolution_metric,
        cross_neighbor_cutoff=reconstruction_parameters.cross_neighbor_cutoff,
    )
    cross_neighbor = mode is FeatureBandExtremaMode.CROSS_NEIGHBOR
    record_count = 5 if cross_neighbor else 1
    score_floor = np.float32(score_parameters.floor)

    # Materialize the row cache once. _RowCache.get(row) already resolves
    # per-row stage-parameter swaps (source_stage_for_row/score_stage_for_row)
    # exactly as the reference streaming loop does; this loop only stacks
    # its results so the row kernel can index whole-image planes directly.
    working_all = np.empty((height, width, 4), dtype=np.float32)
    auxiliary_all = np.empty((height, width), dtype=np.float32)
    score_all = np.empty((height, width), dtype=np.float32)
    weighted_auxiliary_all = np.empty((height, width), dtype=np.float32)
    weighted_rgb_all = np.empty((height, width, 3), dtype=np.float32)
    for row in range(height):
        _check_cancelled(cancelled)
        analyzed = cache.get(row)
        working_all[row] = analyzed.working
        auxiliary_all[row] = analyzed.auxiliary
        score_all[row] = analyzed.score
        weighted_auxiliary_all[row] = analyzed.weighted_auxiliary
        weighted_rgb_all[row] = analyzed.weighted_rgb

    # The inverse-response factor tables are hash-checked on construction;
    # build them once per replay instead of once per emit call.
    factors = InverseResponseFactors.recovered_16bit()

    # Per-row resolved scalars (stage_parameter_provider only ever replaces
    # coarse_reference/driver_gate_secondary/row_reconstruction_gate; see
    # stage_parameters.py), resolved once up front the same way the
    # reference's row loop resolves them.
    fallback_values = np.empty(height, dtype=np.float32)
    floor_enabled_rows = np.zeros(height, dtype=np.uint8)
    row_reconstruction_gates = np.zeros(height, dtype=np.int64)
    driver_gate_primary = bool(reconstruction_parameters.driver_gate_primary)
    for row in range(height):
        active_reconstruction_parameters = reconstruction_parameters
        if stage_parameter_provider is not None:
            stage_hit = writer_stage_for_row(row)
            writer_calibration = stage_parameter_provider(stage_hit)
            validate_stage_calibration(writer_calibration, expected_stage_hit=stage_hit)
            active_reconstruction_parameters = replace(
                reconstruction_parameters,
                coarse_reference=writer_calibration.base_primary,
                driver_gate_secondary=writer_calibration.writer_gate_secondary,
                row_reconstruction_gate=writer_calibration.row_reconstruction_gate,
            )
        fallback_values[row] = np.float32(active_reconstruction_parameters.coarse_reference)
        floor_enabled_rows[row] = 1 if (
            driver_gate_primary or active_reconstruction_parameters.driver_gate_secondary
        ) else 0
        row_reconstruction_gates[row] = int(
            active_reconstruction_parameters.row_reconstruction_gate
        )

    # Stable across the whole run: everything else in ReconstructionParameters
    # is never touched by stage_parameter_provider's per-row replace() calls.
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

    decision_threshold = np.float32(decision_parameters.sample_threshold)
    decision_radius = int(decision_parameters.perpendicular_radius)
    decision_count_limit = int(decision_parameters.count_limit)

    low64 = float(np.float32(dither_bounds.low))
    high64 = float(np.float32(dither_bounds.high))
    low_lt_high = low64 < high64

    startup = _startup_replay_fast(
        kernels,
        working_all=working_all,
        auxiliary_all=auxiliary_all,
        score_all=score_all,
        width=width,
        score_parameters=score_parameters,
        decision_parameters=decision_parameters,
        reconstruction_parameters=reconstruction_parameters,
        generator=active_generator,
        stage_parameter_provider=stage_parameter_provider,
        cross_neighbor=cross_neighbor,
        coarse_slopes=coarse_slopes,
        band_enabled=band_enabled,
        band_scales=band_scales,
        factors_a=factors_a,
        factors_b=factors_b,
        configured_strengths=configured_strengths,
        dither_scales=dither_scales,
        low64=low64,
        high64=high64,
        low_lt_high=low_lt_high,
    )

    attempted = 0
    written = 0
    public_advances = 0
    changed = 0
    output_hash = hashlib.sha256()
    state = int(active_generator.state)

    # Band buffers, reused across bands.  The analysis phase runs one prange
    # over the band's rows (disjoint writes, zero reductions -- byte-equal
    # for every thread count); the writer then consumes the band strictly in
    # row-major order, threading the LCG state.
    band_rows_capacity = min(height, _BAND_ROWS)
    band_attempted = np.empty((band_rows_capacity, width), dtype=np.uint8)
    band_candidates = np.empty((band_rows_capacity, width, 3), dtype=np.float64)
    band_attempted_counts = np.empty(band_rows_capacity, dtype=np.int64)
    band_values = np.empty((band_rows_capacity, width, 3), dtype=np.float32)
    band_written = np.empty(band_rows_capacity, dtype=np.int64)
    band_advances = np.empty(band_rows_capacity, dtype=np.int64)

    for y0 in range(0, height, band_rows_capacity):
        band_rows = min(band_rows_capacity, height - y0)
        kernels.analyze_band(
            auxiliary_all,
            score_all,
            weighted_auxiliary_all,
            weighted_rgb_all,
            working_all,
            y0,
            band_rows,
            height,
            width,
            score_floor,
            decision_threshold,
            decision_radius,
            decision_count_limit,
            floor_enabled_rows,
            row_reconstruction_gates,
            fallback_values,
            record_count,
            coarse_enabled,
            coarse_slopes,
            band_enabled,
            band_scales,
            factors_a,
            factors_b,
            configured_strengths,
            band_attempted,
            band_candidates,
            band_attempted_counts,
        )
        state = kernels.write_band(
            band_attempted,
            band_candidates,
            working_all,
            y0,
            band_rows,
            width,
            floor_enabled_rows,
            low64,
            high64,
            low_lt_high,
            dither_scales,
            state,
            band_values,
            band_written,
            band_advances,
        )
        # Per-band emit: the digest consumes the same bytes in the same
        # row-major order as a per-row accumulation, so it is unchanged.
        band_slice = slice(y0, y0 + band_rows)
        noop_band = emit_public_rgb16(working_all[band_slice, :, :3], factors=factors)
        rendered_band = emit_public_rgb16(band_values[:band_rows], factors=factors)
        output[band_slice] = rendered_band
        output_hash.update(
            rendered_band.astype("<u2", copy=False).tobytes(order="C")
        )
        changed += int(
            np.count_nonzero(np.any(rendered_band != noop_band, axis=2))
        )
        for i in range(band_rows):
            y = y0 + i
            attempted += int(band_attempted_counts[i])
            written += int(band_written[i])
            public_advances += int(band_advances[i])
            if diagnostics_row is not None:
                diagnostics_row(
                    y, score_all[y], score_floor, rendered_band[i], noop_band[i]
                )
            if progress is not None:
                progress(y + 1, height, attempted, written)

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
    # Unsupported jobs fail identically to the reference, before the numba
    # availability probe; only then does the compiled path fail closed.
    DEFAULT_PROFILE.validate_job(job)
    _kernels()
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
    from .producer import derive_producer_record_schedule_fast

    schedule = derive_producer_record_schedule_fast(main_pixels, response.table)
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
        cancelled=cancelled,
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
