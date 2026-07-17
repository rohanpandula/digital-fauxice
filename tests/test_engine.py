from __future__ import annotations

import numpy as np
import pytest

from portable_digital_ice import (
    ProcessingCancelled,
    ProcessingJob,
    ProcessingPhase,
    process_cpu,
)


def test_cpu_entry_point_is_deterministic_and_reports_progress(
    supported_job: ProcessingJob,
) -> None:
    phases: list[ProcessingPhase] = []
    first = process_cpu(
        supported_job,
        progress=lambda event: phases.append(event.phase),
    )
    second = process_cpu(supported_job)

    assert first.output_rgb16.dtype == np.dtype(np.uint16)
    assert first.output_rgb16.shape == (8, 8, 3)
    np.testing.assert_array_equal(first.output_rgb16, second.output_rgb16)
    assert first.replay.output_sha256 == second.replay.output_sha256
    assert first.replay.output_sha256 == (
        "8d0992222a7523de293d69de38d560131ccf9bef2b90be9f7942406743a07941"
    )
    assert first.replay.attempted_pixels == 8
    assert first.replay.written_pixels == first.replay.changed_pixels == 2
    assert first.replay.public_rng_advances == 26
    assert first.profile_id == "nikon-ls5000-selector8-normal-metric4000"
    assert phases[0] is ProcessingPhase.VALIDATING
    assert ProcessingPhase.RECONSTRUCTION in phases
    assert phases[-1] is ProcessingPhase.COMPLETE


def test_cpu_entry_point_supports_caller_owned_output(
    supported_job: ProcessingJob,
) -> None:
    output = np.empty((8, 8, 3), dtype=np.uint16)
    result = process_cpu(supported_job, output_rgb16=output)
    assert result.output_rgb16 is output


def test_cancellation_fails_without_partial_success(
    supported_job: ProcessingJob,
) -> None:
    with pytest.raises(ProcessingCancelled):
        process_cpu(supported_job, cancelled=lambda: True)


def test_mid_reconstruction_cancellation_does_not_mutate_caller_output(
    supported_job: ProcessingJob,
) -> None:
    output = np.full((8, 8, 3), 12_345, dtype=np.uint16)
    calls = 0

    def cancel_after_processing_starts() -> bool:
        nonlocal calls
        calls += 1
        return calls >= 5

    with pytest.raises(ProcessingCancelled):
        process_cpu(
            supported_job,
            output_rgb16=output,
            cancelled=cancel_after_processing_starts,
        )

    np.testing.assert_array_equal(output, np.full_like(output, 12_345))
