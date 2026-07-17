"""High-level CPU entry point for one supported dual-RGBI job."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt

from .prepass import reduce_prepass_frame
from .producer_parameters import (
    ContentDerivedStageParameterProvider,
    derive_producer_record_schedule,
)
from .profile import DEFAULT_PROFILE, ProcessingJob
from .streaming import StreamingReplayResult, run_streaming_replay
from .x3a import SharedLookupInputResponse


class ProcessingPhase(str, Enum):
    VALIDATING = "validating"
    PREPASS = "prepass"
    PRODUCER = "producer"
    RECONSTRUCTION = "reconstruction"
    COMPLETE = "complete"


@dataclass(frozen=True)
class ProcessingProgress:
    phase: ProcessingPhase
    completed: int
    total: int
    attempted_pixels: int = 0
    written_pixels: int = 0


@dataclass(frozen=True)
class ProcessingResult:
    output_rgb16: npt.NDArray[np.uint16]
    replay: StreamingReplayResult
    profile_id: str


class ProcessingCancelled(RuntimeError):
    """Raised when a caller requests cooperative cancellation."""


ProgressCallback = Callable[[ProcessingProgress], None]
CancellationCallback = Callable[[], bool]


def _notify(
    callback: ProgressCallback | None,
    phase: ProcessingPhase,
    completed: int,
    total: int,
    attempted_pixels: int = 0,
    written_pixels: int = 0,
) -> None:
    if callback is not None:
        callback(
            ProcessingProgress(
                phase=phase,
                completed=completed,
                total=total,
                attempted_pixels=attempted_pixels,
                written_pixels=written_pixels,
            )
        )


def _check_cancelled(cancelled: CancellationCallback | None) -> None:
    if cancelled is not None and cancelled():
        raise ProcessingCancelled("portable correction was cancelled")


def process_cpu(
    job: ProcessingJob,
    *,
    output_rgb16: npt.NDArray[np.uint16] | None = None,
    progress: ProgressCallback | None = None,
    cancelled: CancellationCallback | None = None,
) -> ProcessingResult:
    """Process one validated LS-5000 selector-8 Normal acquisition on CPU.

    A caller-owned output is committed only after successful completion.
    Validation errors and cancellation leave that buffer unchanged.
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
    replay = run_streaming_replay(
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
    )
    _check_cancelled(cancelled)
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
    )


__all__ = [
    "ProcessingCancelled",
    "ProcessingPhase",
    "ProcessingProgress",
    "ProcessingResult",
    "process_cpu",
]
