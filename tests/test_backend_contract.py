"""Backend-selection contract: cpu exact, cuda fail-closed, auto self-tested.

These tests run on every machine.  Where behavior depends on whether CUDA is
actually present, both legs are asserted explicitly.
"""

from __future__ import annotations

import numpy as np
import pytest

from portable_digital_ice import ComputeBackend, process, process_cpu
from portable_digital_ice.contracts import (
    AcquisitionEpoch,
    DualRGBIAcquisition,
    RGBI16Frame,
)
from portable_digital_ice.profile import (
    ProcessingJob,
    ProcessingMode,
    ScannerModel,
)


def _cuda_present() -> bool:
    try:
        from portable_digital_ice.cuda_backend.engine import cuda_device_summary

        cuda_device_summary()
        return True
    except Exception:
        return False


def _tiny_job(frame_id: str = "backend-contract") -> ProcessingJob:
    rng = np.random.default_rng(11)
    main = rng.integers(18000, 62000, size=(12, 24, 4), dtype=np.uint16)
    main[5:8, 9:14, 3] = 700
    prepass = rng.integers(18000, 62000, size=(8, 16, 4), dtype=np.uint16)
    return ProcessingJob(
        acquisition=DualRGBIAcquisition(
            prepass=RGBI16Frame(prepass, AcquisitionEpoch.PREPASS, 285, "pp"),
            main=RGBI16Frame(main, AcquisitionEpoch.MAIN, 4000, "main"),
            same_frame_id=frame_id,
        ),
        scanner_model=ScannerModel.NIKON_SUPER_COOLSCAN_5000_ED,
        mode=ProcessingMode.NORMAL,
        selector=8,
        resolution_metric=4000,
        bit_depth=16,
        focus_exposure_locked=True,
    )


def test_backend_enum_values():
    assert ComputeBackend.AUTO.value == "auto"
    assert ComputeBackend.CPU.value == "cpu"
    assert ComputeBackend.CUDA.value == "cuda"


def test_cpu_backend_is_reference():
    job = _tiny_job()
    direct = process_cpu(_tiny_job())
    routed = process(job, backend=ComputeBackend.CPU)
    assert routed.selection.used is ComputeBackend.CPU
    assert routed.result.replay.output_sha256 == direct.replay.output_sha256


def test_unsupported_job_fails_before_backend_probe():
    job = _tiny_job()
    bad = ProcessingJob(
        acquisition=job.acquisition,
        scanner_model=job.scanner_model,
        mode=job.mode,
        selector=9,
        resolution_metric=job.resolution_metric,
        bit_depth=job.bit_depth,
        focus_exposure_locked=True,
    )
    for backend in (ComputeBackend.CPU, ComputeBackend.CUDA, ComputeBackend.AUTO):
        with pytest.raises(ValueError):
            process(bad, backend=backend)


def test_cuda_request_fails_clearly_or_matches_cpu():
    from portable_digital_ice.cuda_backend.engine import CudaBackendUnavailable

    job = _tiny_job()
    if _cuda_present():
        routed = process(job, backend=ComputeBackend.CUDA)
        assert routed.selection.used is ComputeBackend.CUDA
        cpu = process_cpu(_tiny_job())
        assert routed.result.replay.output_sha256 == cpu.replay.output_sha256
    else:
        with pytest.raises(CudaBackendUnavailable):
            process(job, backend=ComputeBackend.CUDA)


def test_auto_reports_reason_and_never_fails_closed():
    job = _tiny_job()
    routed = process(job, backend=ComputeBackend.AUTO)
    if _cuda_present():
        assert routed.selection.used is ComputeBackend.CUDA
        assert "self-test" in routed.selection.reason
    else:
        assert routed.selection.used is ComputeBackend.CPU
        assert "CUDA unavailable" in routed.selection.reason
    cpu = process_cpu(_tiny_job())
    assert routed.result.replay.output_sha256 == cpu.replay.output_sha256


def test_cancellation_does_not_corrupt_output():
    from portable_digital_ice.engine import ProcessingCancelled

    job = _tiny_job()
    output = np.zeros((12, 24, 3), dtype=np.uint16)
    calls = {"n": 0}

    def cancel() -> bool:
        calls["n"] += 1
        return calls["n"] > 2

    with pytest.raises(ProcessingCancelled):
        process(job, backend=ComputeBackend.CPU, output_rgb16=output, cancelled=cancel)
