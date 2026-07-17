from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from portable_digital_ice import (
    AcquisitionEpoch,
    DEFAULT_PROFILE,
    DualRGBIAcquisition,
    ProcessingJob,
    RGBI16Frame,
)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("scanner_model", "other-scanner", "scanner model"),
        ("mode", "fine", "mode"),
        ("selector", 7, "selector"),
        ("resolution_metric", 500, "resolution metric"),
        ("bit_depth", 8, "bit depth"),
        ("focus_exposure_locked", False, "focus and exposure"),
        ("focus_exposure_locked", 1, "focus and exposure"),
    ),
)
def test_profile_fails_closed_for_unsupported_job_fields(
    supported_job: ProcessingJob,
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        DEFAULT_PROFILE.validate_job(replace(supported_job, **{field: value}))


def test_profile_rejects_wrong_capture_resolutions(
    supported_job: ProcessingJob,
) -> None:
    acquisition = supported_job.acquisition
    wrong_prepass = RGBI16Frame(
        acquisition.prepass.pixels,
        AcquisitionEpoch.PREPASS,
        500,
        "wrong-prepass",
    )
    with pytest.raises(ValueError, match="285 dpi"):
        DEFAULT_PROFILE.validate_job(
            replace(
                supported_job,
                acquisition=DualRGBIAcquisition(
                    wrong_prepass,
                    acquisition.main,
                    acquisition.same_frame_id,
                ),
            )
        )


def test_frame_contract_requires_uint16_rgbi() -> None:
    with pytest.raises(TypeError, match="dtype uint16"):
        RGBI16Frame(
            np.zeros((8, 8, 4), dtype=np.float32),
            AcquisitionEpoch.MAIN,
            4_000,
            "wrong-dtype",
        )
    with pytest.raises(ValueError, match="HxWx4"):
        RGBI16Frame(
            np.zeros((8, 8, 3), dtype=np.uint16),
            AcquisitionEpoch.MAIN,
            4_000,
            "wrong-lanes",
        )


def test_supported_profile_cannot_be_reconfigured() -> None:
    profile_type = type(DEFAULT_PROFILE)
    with pytest.raises(TypeError):
        profile_type(selector=7)
