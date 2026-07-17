"""Backend-selection contract: cpu exact, cuda/cpu-fast fail-closed, auto
self-tested.

These tests run on every machine.  Where behavior depends on whether CUDA or
numba is actually present, every leg is asserted explicitly.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import pytest

from portable_digital_ice import ComputeBackend, process, process_cpu
from portable_digital_ice import backend as backend_module
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


def _numba_present() -> bool:
    try:
        import numba  # noqa: F401

        return True
    except Exception:
        return False


@contextmanager
def _self_test_cache_snapshot():
    """Isolate self-test cache mutations so tests stay order-independent."""

    saved = dict(backend_module._SELF_TEST_CACHE)
    try:
        yield backend_module._SELF_TEST_CACHE
    finally:
        backend_module._SELF_TEST_CACHE.clear()
        backend_module._SELF_TEST_CACHE.update(saved)


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
    assert ComputeBackend.CPU_FAST.value == "cpu-fast"
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
    elif _numba_present():
        assert routed.selection.used is ComputeBackend.CPU_FAST
        assert "CUDA unavailable" in routed.selection.reason
        assert "self-test passed byte parity" in routed.selection.reason
        import numba

        assert f"numba {numba.__version__}" in routed.selection.reason
    else:
        assert routed.selection.used is ComputeBackend.CPU
        assert "CUDA unavailable" in routed.selection.reason
        assert "cpu-fast unavailable" in routed.selection.reason
        assert "using exact CPU reference" in routed.selection.reason
    cpu = process_cpu(_tiny_job())
    assert routed.result.replay.output_sha256 == cpu.replay.output_sha256


def test_cpu_fast_request_matches_cpu_or_fails_clearly():
    from portable_digital_ice.fast_cpu import CpuFastUnavailable

    job = _tiny_job()
    if _numba_present():
        routed = process(job, backend=ComputeBackend.CPU_FAST)
        assert routed.selection.used is ComputeBackend.CPU_FAST
        assert "self-test passed byte parity" in routed.selection.reason
        import numba

        assert f"numba {numba.__version__}" in routed.selection.reason
        cpu = process_cpu(_tiny_job())
        assert routed.result.replay.output_sha256 == cpu.replay.output_sha256
        assert routed.result.replay == cpu.replay
    else:
        with pytest.raises(CpuFastUnavailable):
            process(job, backend=ComputeBackend.CPU_FAST)


def test_cpu_fast_string_request_is_accepted():
    if not _numba_present():
        pytest.skip("numba is unavailable")
    routed = process(_tiny_job(), backend="cpu-fast")
    assert routed.selection.used is ComputeBackend.CPU_FAST


def test_cpu_fast_self_test_outcome_is_cached(monkeypatch):
    if not _numba_present():
        pytest.skip("numba is unavailable")
    from portable_digital_ice import fast_cpu

    calls = {"n": 0}
    real_process_cpu_fast = fast_cpu.process_cpu_fast

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real_process_cpu_fast(*args, **kwargs)

    with _self_test_cache_snapshot() as cache:
        cache.pop("cpu-fast-outcome", None)
        monkeypatch.setattr(
            "portable_digital_ice.fast_cpu.process_cpu_fast", counting
        )
        backend_module.cpu_fast_self_test()
        assert calls["n"] == 1
        backend_module.cpu_fast_self_test()  # cached success: no rerun
        assert calls["n"] == 1


def test_cpu_fast_failure_is_cached_and_fails_closed(monkeypatch):
    from portable_digital_ice.fast_cpu import CpuFastUnavailable

    calls = {"n": 0}

    def broken(*args, **kwargs):
        calls["n"] += 1
        raise CpuFastUnavailable("numba is not importable: injected for test")

    with _self_test_cache_snapshot() as cache:
        cache.pop("cpu-fast-outcome", None)
        monkeypatch.setattr(
            "portable_digital_ice.fast_cpu.process_cpu_fast", broken
        )
        with pytest.raises(CpuFastUnavailable):
            process(_tiny_job(), backend=ComputeBackend.CPU_FAST)
        assert calls["n"] == 1
        # The failure outcome is cached: later calls raise without rerunning.
        with pytest.raises(CpuFastUnavailable) as second:
            process(_tiny_job(), backend=ComputeBackend.CPU_FAST)
        assert calls["n"] == 1
        assert "numba is not importable" in str(second.value)
        # AUTO records both fallthrough reasons and still returns exact output.
        routed = process(_tiny_job(), backend=ComputeBackend.AUTO)
        if _cuda_present():
            assert routed.selection.used is ComputeBackend.CUDA
        else:
            assert routed.selection.used is ComputeBackend.CPU
            assert "CUDA unavailable" in routed.selection.reason
            assert "cpu-fast unavailable" in routed.selection.reason
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
