from __future__ import annotations

import numpy as np
import pytest

from portable_digital_ice import (
    AcquisitionEpoch,
    DualRGBIAcquisition,
    ProcessingJob,
    ProcessingMode,
    RGBI16Frame,
    ScannerModel,
)


def synthetic_rgbi(height: int, width: int, *, offset: int) -> np.ndarray:
    """Build a deterministic synthetic RGBI target with no external assets."""

    y, x = np.indices((height, width), dtype=np.uint32)
    pixels = np.empty((height, width, 4), dtype=np.uint16)
    pixels[:, :, 0] = (12_000 + offset + 53 * y + 97 * x).astype(np.uint16)
    pixels[:, :, 1] = (16_000 + offset + 67 * y + 71 * x).astype(np.uint16)
    pixels[:, :, 2] = (20_000 + offset + 89 * y + 43 * x).astype(np.uint16)
    pixels[:, :, 3] = (44_000 + offset + 31 * y + 29 * x).astype(np.uint16)
    return pixels


@pytest.fixture
def supported_job() -> ProcessingJob:
    prepass = RGBI16Frame(
        synthetic_rgbi(8, 8, offset=0),
        AcquisitionEpoch.PREPASS,
        285,
        "synthetic-prepass",
    )
    main_pixels = synthetic_rgbi(8, 8, offset=150)
    main_pixels[3:5, 3:5, 3] = np.uint16(9_000)
    main = RGBI16Frame(
        main_pixels,
        AcquisitionEpoch.MAIN,
        4_000,
        "synthetic-main",
    )
    acquisition = DualRGBIAcquisition(prepass, main, "synthetic-frame-1")
    return ProcessingJob(
        acquisition=acquisition,
        scanner_model=ScannerModel.NIKON_SUPER_COOLSCAN_5000_ED,
        mode=ProcessingMode.NORMAL,
        selector=8,
        resolution_metric=4_000,
        bit_depth=16,
        focus_exposure_locked=True,
    )
