"""Opt-in diagnostics preserve exact output and agree across backends."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from portable_digital_ice import (
    DualRGBIAcquisition,
    ComputeBackend,
    ProcessingDiagnostics,
    ProcessingJob,
    RGBI16Frame,
    process,
    process_cpu,
)
from portable_digital_ice.backend import _synthetic_self_test_job
from portable_digital_ice.output import emit_public_rgb16
from portable_digital_ice.prepass import reduce_prepass_frame
from portable_digital_ice.profile import DEFAULT_PROFILE
from portable_digital_ice.x3a import SharedLookupInputResponse


def _score_floor(job: ProcessingJob) -> np.float32:
    response = SharedLookupInputResponse.nikon_logarithmic()
    prepass = reduce_prepass_frame(
        job.acquisition.prepass,
        parameters=DEFAULT_PROFILE.prepass_parameters(),
        frame_index=0,
        evidence_id=f"portable-prepass:{job.acquisition.same_frame_id}",
        response=response,
    )
    return np.float32(DEFAULT_PROFILE.score_parameters(prepass.record).floor)


def _noop_rgb16(job: ProcessingJob) -> np.ndarray:
    response = SharedLookupInputResponse.nikon_logarithmic()
    working = response.convert(job.acquisition.main.pixels)
    return emit_public_rgb16(working[:, :, :3])


def _require_cuda_device() -> None:
    pytest.importorskip("cupy", reason="CUDA backend requires cupy")
    from portable_digital_ice.cuda_backend.engine import (
        CudaBackendUnavailable,
        cuda_device_summary,
    )

    try:
        cuda_device_summary()
    except CudaBackendUnavailable as error:
        pytest.skip(f"CUDA device unavailable: {error}")


def test_cpu_diagnostics_are_default_off(supported_job: ProcessingJob) -> None:
    assert process_cpu(supported_job).diagnostics is None
    assert process_cpu(supported_job, export_diagnostics=False).diagnostics is None


def test_cpu_diagnostics_are_typed_shaped_and_read_only(
    supported_job: ProcessingJob,
) -> None:
    result = process_cpu(supported_job, export_diagnostics=True)
    diagnostics = result.diagnostics

    assert isinstance(diagnostics, ProcessingDiagnostics)
    shape = supported_job.acquisition.main.pixels.shape[:2]
    assert diagnostics.score_plane.dtype == np.dtype(np.float32)
    assert np.asarray(diagnostics.score_floor).dtype == np.dtype(np.float32)
    assert np.asarray(diagnostics.score_floor).ndim == 0
    assert diagnostics.at_floor_mask.dtype == np.dtype(np.bool_)
    assert diagnostics.changed_mask.dtype == np.dtype(np.bool_)
    assert diagnostics.score_plane.shape == shape
    assert diagnostics.at_floor_mask.shape == shape
    assert diagnostics.changed_mask.shape == shape
    assert not diagnostics.score_plane.flags.writeable
    assert not diagnostics.at_floor_mask.flags.writeable
    assert not diagnostics.changed_mask.flags.writeable

    for array in (
        diagnostics.score_plane,
        diagnostics.at_floor_mask,
        diagnostics.changed_mask,
    ):
        with pytest.raises(ValueError):
            array.flat[0] = array.flat[0]


def test_cpu_diagnostic_masks_match_the_exact_run(
    supported_job: ProcessingJob,
) -> None:
    result = process_cpu(supported_job, export_diagnostics=True)
    diagnostics = result.diagnostics
    assert diagnostics is not None

    floor = diagnostics.score_floor
    assert floor.view(np.uint32) == _score_floor(supported_job).view(np.uint32)
    np.testing.assert_array_equal(
        diagnostics.at_floor_mask,
        diagnostics.score_plane == floor,
    )
    assert np.all(diagnostics.score_plane >= floor)

    expected_changed = np.any(result.output_rgb16 != _noop_rgb16(supported_job), axis=2)
    np.testing.assert_array_equal(diagnostics.changed_mask, expected_changed)
    assert int(np.count_nonzero(diagnostics.changed_mask)) == result.replay.changed_pixels


def test_cpu_at_floor_mask_is_non_vacuous_and_uses_horizontal_minimum(
    supported_job: ProcessingJob,
) -> None:
    acquisition = supported_job.acquisition
    main_pixels = acquisition.main.pixels.copy()
    main_pixels[3:5, 3:5, 3] = np.uint16(0)
    main = RGBI16Frame(
        main_pixels,
        acquisition.main.epoch,
        acquisition.main.resolution_dpi,
        "diagnostic-floor-main",
    )
    job = replace(
        supported_job,
        acquisition=DualRGBIAcquisition(
            prepass=acquisition.prepass,
            main=main,
            same_frame_id="diagnostic-floor-frame",
        ),
    )

    diagnostics = process_cpu(job, export_diagnostics=True).diagnostics
    assert diagnostics is not None
    expected_patch = np.zeros((8, 8), dtype=bool)
    expected_patch[3:5, 2:6] = True
    np.testing.assert_array_equal(diagnostics.at_floor_mask, expected_patch)


def test_cpu_diagnostics_flag_does_not_change_output(
    supported_job: ProcessingJob,
) -> None:
    without = process_cpu(supported_job, export_diagnostics=False)
    with_diagnostics = process_cpu(supported_job, export_diagnostics=True)

    np.testing.assert_array_equal(without.output_rgb16, with_diagnostics.output_rgb16)
    assert without.replay.output_sha256 == with_diagnostics.replay.output_sha256


def test_routed_cpu_propagates_diagnostics_flag(
    supported_job: ProcessingJob,
) -> None:
    routed = process(
        supported_job,
        backend=ComputeBackend.CPU,
        export_diagnostics=True,
    )

    assert routed.selection.used is ComputeBackend.CPU
    assert isinstance(routed.result.diagnostics, ProcessingDiagnostics)
    assert routed.result.diagnostics.changed_mask.shape == (8, 8)


def test_cuda_diagnostics_match_cpu_and_preserve_output() -> None:
    _require_cuda_device()
    from portable_digital_ice.cuda_backend import process_cuda

    job = _synthetic_self_test_job()
    cpu_without = process_cpu(job)
    cpu_with = process_cpu(job, export_diagnostics=True)
    cuda_without = process_cuda(job)
    cuda_with = process_cuda(job, export_diagnostics=True)

    assert cpu_without.diagnostics is None
    assert cuda_without.diagnostics is None
    assert cpu_with.diagnostics is not None
    assert cuda_with.diagnostics is not None

    np.testing.assert_array_equal(
        cpu_with.diagnostics.score_plane.view(np.uint32),
        cuda_with.diagnostics.score_plane.view(np.uint32),
    )
    assert (
        cpu_with.diagnostics.score_floor.view(np.uint32)
        == cuda_with.diagnostics.score_floor.view(np.uint32)
    )
    np.testing.assert_array_equal(
        cpu_with.diagnostics.at_floor_mask,
        cuda_with.diagnostics.at_floor_mask,
    )
    np.testing.assert_array_equal(
        cpu_with.diagnostics.changed_mask,
        cuda_with.diagnostics.changed_mask,
    )

    for without, with_diagnostics in (
        (cpu_without, cpu_with),
        (cuda_without, cuda_with),
    ):
        np.testing.assert_array_equal(
            without.output_rgb16,
            with_diagnostics.output_rgb16,
        )
        assert without.replay.output_sha256 == with_diagnostics.replay.output_sha256
        assert without.replay.changed_pixels == with_diagnostics.replay.changed_pixels
        assert (
            int(np.count_nonzero(with_diagnostics.diagnostics.changed_mask))
            == with_diagnostics.replay.changed_pixels
        )

    np.testing.assert_array_equal(cpu_with.output_rgb16, cuda_with.output_rgb16)
    assert cpu_with.replay.output_sha256 == cuda_with.replay.output_sha256
    assert cpu_with.replay.changed_pixels == cuda_with.replay.changed_pixels
