"""Optional deterministic Metal backend for the supported LS-5000 profile.

Importing this package does not require Metal; every entry point performs its
own fail-closed availability check.  See ``docs/metal-backend.md``.
"""

from .engine import (
    MetalBackendUnavailable,
    get_kernel_library,
    metal_device_summary,
    run_streaming_replay_metal,
)
from .process import process_metal

__all__ = [
    "MetalBackendUnavailable",
    "get_kernel_library",
    "metal_device_summary",
    "process_metal",
    "run_streaming_replay_metal",
]
