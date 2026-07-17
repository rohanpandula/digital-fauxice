"""Job-level CUDA processing mirroring :func:`..engine.process_cpu`."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ..engine import (
    CancellationCallback,
    ProcessingPhase,
    ProcessingResult,
    ProgressCallback,
    _check_cancelled,
    _notify,
)
from ..prepass import reduce_prepass_frame
from ..producer_parameters import ContentDerivedStageParameterProvider
from ..profile import DEFAULT_PROFILE, ProcessingJob
from ..x3a import SharedLookupInputResponse
from .engine import get_kernel_module, run_streaming_replay_cuda
from .producer import derive_producer_record_schedule_cuda


def process_cuda(
    job: ProcessingJob,
    *,
    output_rgb16: npt.NDArray[np.uint16] | None = None,
    progress: ProgressCallback | None = None,
    cancelled: CancellationCallback | None = None,
    stage_timings: dict | None = None,
) -> ProcessingResult:
    """Process one validated LS-5000 selector-8 Normal acquisition on CUDA.

    The result is only correct if it is byte-identical to ``process_cpu``;
    the synthetic and private parity gates enforce that.  This function never
    silently falls back to a different numeric path: any CUDA problem raises
    :class:`.engine.CudaBackendUnavailable`.
    """

    _notify(progress, ProcessingPhase.VALIDATING, 0, 1)
    DEFAULT_PROFILE.validate_job(job)
    _check_cancelled(cancelled)
    _notify(progress, ProcessingPhase.VALIDATING, 1, 1)

    response = SharedLookupInputResponse.nikon_logarithmic()
    import time as _time

    _notify(progress, ProcessingPhase.PREPASS, 0, 1)
    _prepass_started = _time.monotonic()
    prepass = reduce_prepass_frame(
        job.acquisition.prepass,
        parameters=DEFAULT_PROFILE.prepass_parameters(),
        frame_index=0,
        evidence_id=f"portable-prepass:{job.acquisition.same_frame_id}",
        response=response,
    )
    if stage_timings is not None:
        stage_timings["host.prepass"] = _time.monotonic() - _prepass_started
    _check_cancelled(cancelled)
    _notify(progress, ProcessingPhase.PREPASS, 1, 1)

    _notify(progress, ProcessingPhase.PRODUCER, 0, 1)
    main_pixels = job.acquisition.main.pixels
    module = get_kernel_module()
    _producer_started = _time.monotonic()
    schedule = derive_producer_record_schedule_cuda(
        module, main_pixels, response.table
    )
    if stage_timings is not None:
        stage_timings["producer.total"] = _time.monotonic() - _producer_started
    provider = ContentDerivedStageParameterProvider(
        schedule,
        auxiliary_factor_b=DEFAULT_PROFILE.auxiliary_factor_b,
        hold_last_through_stage=job.acquisition.main.height + 6,
    )
    _check_cancelled(cancelled)
    _notify(progress, ProcessingPhase.PRODUCER, 1, 1)

    expected_shape = (*main_pixels.shape[:2], 3)
    if output_rgb16 is None:
        output = np.empty(expected_shape, dtype=np.uint16)
    else:
        output = np.asarray(output_rgb16)
        if output.dtype != np.dtype(np.uint16) or output.shape != expected_shape:
            raise ValueError(
                f"output must be writable uint16 with shape {expected_shape}"
            )
        if not output.flags.writeable:
            raise ValueError("output must be writable")

    def row_progress(completed: int, total: int, attempted: int, written: int) -> None:
        _check_cancelled(cancelled)
        _notify(
            progress,
            ProcessingPhase.RECONSTRUCTION,
            completed,
            total,
            attempted,
            written,
        )

    _notify(progress, ProcessingPhase.RECONSTRUCTION, 0, job.acquisition.main.height)
    replay = run_streaming_replay_cuda(
        main_pixels,
        output,
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
        cancelled=cancelled,
        stage_timings=stage_timings,
    )
    _check_cancelled(cancelled)
    _notify(progress, ProcessingPhase.COMPLETE, 1, 1)
    return ProcessingResult(
        output_rgb16=output,
        replay=replay,
        profile_id=DEFAULT_PROFILE.profile_id,
    )


__all__ = ["process_cuda"]
