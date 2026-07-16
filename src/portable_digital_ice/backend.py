"""Fail-closed compute-backend selection for portable Digital ICE.

``cpu`` always selects the exact validated reference.  ``cuda`` requires a
usable CUDA device and raises with a specific reason otherwise; it never
silently substitutes another implementation.  ``auto`` selects CUDA only after
a startup self-test reproduces the CPU reference byte-for-byte on a synthetic
acquisition, and otherwise falls back to CPU while reporting why.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt

from .contracts import AcquisitionEpoch, DualRGBIAcquisition, RGBI16Frame
from .engine import (
    CancellationCallback,
    ProcessingDiagnostics,
    ProcessingResult,
    ProgressCallback,
    process_cpu,
)
from .profile import ProcessingJob, ProcessingMode, ScannerModel


class ComputeBackend(StrEnum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


@dataclass(frozen=True)
class BackendSelection:
    """The backend that actually ran, plus why."""

    requested: ComputeBackend
    used: ComputeBackend
    reason: str


@dataclass(frozen=True)
class BackendProcessingResult:
    result: ProcessingResult
    selection: BackendSelection


_SELF_TEST_CACHE: dict[str, str | None] = {}


def _synthetic_self_test_job() -> ProcessingJob:
    """Deterministic small acquisition covering repair and dither paths."""

    rng = np.random.default_rng(20260716)
    height, width = 48, 64
    main = rng.integers(20000, 60000, size=(height, width, 4), dtype=np.uint16)
    # a dark infrared blob that triggers detection and reconstruction
    main[18:26, 30:39, 3] = rng.integers(200, 2000, size=(8, 9), dtype=np.uint16)
    # low-IR pixels near every edge so boundary carriers are exercised
    main[0:2, 0:6, 3] = 500
    main[-2:, -6:, 3] = 700
    main[3, width - 3, 3] = 900
    prepass = rng.integers(20000, 60000, size=(24, 32, 4), dtype=np.uint16)
    prepass[10:14, 12:17, 3] = 300
    acquisition = DualRGBIAcquisition(
        prepass=RGBI16Frame(prepass, AcquisitionEpoch.PREPASS, 285, "selftest-prepass"),
        main=RGBI16Frame(main, AcquisitionEpoch.MAIN, 4000, "selftest-main"),
        same_frame_id="cuda-backend-self-test",
    )
    return ProcessingJob(
        acquisition=acquisition,
        scanner_model=ScannerModel.NIKON_SUPER_COOLSCAN_5000_ED,
        mode=ProcessingMode.NORMAL,
        selector=8,
        resolution_metric=4000,
        bit_depth=16,
        focus_exposure_locked=True,
    )


def cuda_self_test() -> None:
    """Prove CUDA/CPU byte parity on the synthetic job or raise.

    The comparison covers output bytes, the RNG advance count, the final RNG
    state, and all writer counters.  The (successful or failed) outcome is
    cached per process; a failure reason is re-raised on later calls.
    """

    cached = _SELF_TEST_CACHE.get("outcome", "unset")
    if cached is None:
        return
    if cached != "unset":
        raise_from_reason(str(cached))

    try:
        from .cuda_backend import process_cuda

        job = _synthetic_self_test_job()
        cpu = process_cpu(job)
        gpu = process_cuda(job)
        failures: list[str] = []
        if cpu.replay.output_sha256 != gpu.replay.output_sha256:
            failures.append(
                "output hash mismatch "
                f"cpu={cpu.replay.output_sha256} cuda={gpu.replay.output_sha256}"
            )
        if not np.array_equal(cpu.output_rgb16, gpu.output_rgb16):
            failures.append("output sample mismatch")
        for field in (
            "attempted_pixels",
            "written_pixels",
            "public_rng_advances",
            "final_rng_state",
            "changed_pixels",
        ):
            cpu_value = getattr(cpu.replay, field)
            gpu_value = getattr(gpu.replay, field)
            if cpu_value != gpu_value:
                failures.append(f"{field} mismatch cpu={cpu_value} cuda={gpu_value}")
        if failures:
            reason = "CUDA self-test failed parity: " + "; ".join(failures)
            _SELF_TEST_CACHE["outcome"] = reason
            raise_from_reason(reason)
    except Exception as error:
        if _SELF_TEST_CACHE.get("outcome", "unset") == "unset":
            _SELF_TEST_CACHE["outcome"] = f"CUDA self-test raised: {error!r}"
        raise
    _SELF_TEST_CACHE["outcome"] = None


def raise_from_reason(reason: str) -> None:
    from .cuda_backend.engine import CudaBackendUnavailable

    raise CudaBackendUnavailable(reason)


def process(
    job: ProcessingJob,
    *,
    backend: ComputeBackend | str = ComputeBackend.AUTO,
    output_rgb16: npt.NDArray[np.uint16] | None = None,
    progress: ProgressCallback | None = None,
    cancelled: CancellationCallback | None = None,
    export_diagnostics: bool = False,
) -> BackendProcessingResult:
    """Process one job on the requested backend with fail-closed semantics."""

    from .profile import DEFAULT_PROFILE

    # Backend selection never changes the input contract: unsupported jobs
    # fail identically, before any backend probing or output mutation.
    DEFAULT_PROFILE.validate_job(job)
    requested = ComputeBackend(backend)
    if requested is ComputeBackend.CPU:
        result = process_cpu(
            job,
            output_rgb16=output_rgb16,
            progress=progress,
            cancelled=cancelled,
            export_diagnostics=export_diagnostics,
        )
        return BackendProcessingResult(
            result,
            BackendSelection(requested, ComputeBackend.CPU, "explicit CPU request"),
        )

    if requested is ComputeBackend.CUDA:
        cuda_self_test()
        from .cuda_backend import process_cuda

        result = process_cuda(
            job,
            output_rgb16=output_rgb16,
            progress=progress,
            cancelled=cancelled,
            export_diagnostics=export_diagnostics,
        )
        return BackendProcessingResult(
            result,
            BackendSelection(
                requested, ComputeBackend.CUDA, "explicit CUDA request; self-test passed"
            ),
        )

    # AUTO
    try:
        cuda_self_test()
    except Exception as error:
        result = process_cpu(
            job,
            output_rgb16=output_rgb16,
            progress=progress,
            cancelled=cancelled,
            export_diagnostics=export_diagnostics,
        )
        return BackendProcessingResult(
            result,
            BackendSelection(
                requested,
                ComputeBackend.CPU,
                f"CUDA unavailable, using exact CPU reference: {error}",
            ),
        )
    from .cuda_backend import process_cuda

    result = process_cuda(
        job,
        output_rgb16=output_rgb16,
        progress=progress,
        cancelled=cancelled,
        export_diagnostics=export_diagnostics,
    )
    return BackendProcessingResult(
        result,
        BackendSelection(
            requested, ComputeBackend.CUDA, "startup self-test passed byte parity"
        ),
    )


__all__ = [
    "BackendProcessingResult",
    "BackendSelection",
    "ComputeBackend",
    "ProcessingDiagnostics",
    "cuda_self_test",
    "process",
]
