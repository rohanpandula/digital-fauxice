"""Portable infrared correction for one fail-closed LS-5000 profile."""

from .backend import (
    BackendProcessingResult,
    BackendSelection,
    ComputeBackend,
    process,
)
from .contracts import (
    AcquisitionEpoch,
    DualRGBIAcquisition,
    RGBI16Frame,
)
from .engine import (
    ProcessingCancelled,
    ProcessingDiagnostics,
    ProcessingPhase,
    ProcessingProgress,
    ProcessingResult,
    process_cpu,
)
from .profile import (
    DEFAULT_PROFILE,
    LS5000Selector8NormalProfile,
    ProcessingJob,
    ProcessingMode,
    ScannerModel,
)

__all__ = [
    "AcquisitionEpoch",
    "BackendProcessingResult",
    "BackendSelection",
    "ComputeBackend",
    "DEFAULT_PROFILE",
    "process",
    "DualRGBIAcquisition",
    "LS5000Selector8NormalProfile",
    "ProcessingCancelled",
    "ProcessingDiagnostics",
    "ProcessingJob",
    "ProcessingMode",
    "ProcessingPhase",
    "ProcessingProgress",
    "ProcessingResult",
    "RGBI16Frame",
    "ScannerModel",
    "process_cpu",
]
