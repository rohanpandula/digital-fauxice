"""Optional deterministic CUDA backend for the supported LS-5000 profile.

Importing this package does not require CUDA; every entry point performs its
own fail-closed availability check.  See ``docs/cuda-backend.md``.
"""

from .engine import (
    CudaBackendUnavailable,
    cuda_device_summary,
    get_kernel_module,
    run_streaming_replay_cuda,
)
from .process import process_cuda

__all__ = [
    "CudaBackendUnavailable",
    "cuda_device_summary",
    "get_kernel_module",
    "process_cuda",
    "run_streaming_replay_cuda",
]
