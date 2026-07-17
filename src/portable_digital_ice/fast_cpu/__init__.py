"""Optional compiled CPU backend for the supported LS-5000 profile.

Importing this package does not require numba; every entry point performs its
own fail-closed availability check, mirroring ``cuda_backend``'s discipline.
"""

from .engine import CpuFastUnavailable, process_cpu_fast

__all__ = [
    "CpuFastUnavailable",
    "process_cpu_fast",
]
