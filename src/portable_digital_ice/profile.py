"""Fail-closed runtime profile for the supported LS-5000 acquisition path."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt

from .contracts import DualRGBIAcquisition, PrepassFrameRecord
from .dither import DitherBounds
from .prepass import PrepassParameters
from .reconstruction import ReconstructionParameters
from .x3a import AuxiliaryParameters, DecisionParameters, ScoreParameters


class ScannerModel(str, Enum):
    """Scanner identities with a validated runtime profile."""

    NIKON_SUPER_COOLSCAN_5000_ED = "nikon-super-coolscan-5000-ed"


class ProcessingMode(str, Enum):
    """Validated correction modes."""

    NORMAL = "normal"


@dataclass(frozen=True)
class ProcessingJob:
    """One explicitly identified same-frame dual RGBI acquisition."""

    acquisition: DualRGBIAcquisition
    scanner_model: ScannerModel
    mode: ProcessingMode
    selector: int
    resolution_metric: int
    bit_depth: int
    focus_exposure_locked: bool


def _f32_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def _base_primary(record: PrepassFrameRecord) -> np.float32:
    alpha = np.float32(record.observed_main_alpha)
    mean_r = np.float32(record.observed_calibration_mean_r)
    mean_ir = np.float32(record.observed_calibration_mean_ir)
    denominator = 1.0 - float(alpha)
    if denominator == 0.0:
        raise ValueError("prepass alpha cannot equal one")
    return np.float32((float(mean_ir) - float(alpha) * float(mean_r)) / denominator)


@dataclass(frozen=True, init=False)
class LS5000Selector8NormalProfile:
    """The single scanner, mode, and resolution contract shipped here."""

    scanner_model: ScannerModel = ScannerModel.NIKON_SUPER_COOLSCAN_5000_ED
    mode: ProcessingMode = ProcessingMode.NORMAL
    selector: int = 8
    resolution_metric: int = 4_000
    prepass_dpi: int = 285
    main_dpi: int = 4_000
    bit_depth: int = 16
    auxiliary_factor_b: float = 1.0

    @property
    def profile_id(self) -> str:
        return "nikon-ls5000-selector8-normal-metric4000"

    def validate_job(self, job: ProcessingJob) -> None:
        """Reject every scanner, mode, and acquisition outside this profile."""

        if not isinstance(job, ProcessingJob):
            raise TypeError("processing requires a ProcessingJob")
        expected = (
            ("scanner model", job.scanner_model, self.scanner_model),
            ("mode", job.mode, self.mode),
            ("selector", job.selector, self.selector),
            ("resolution metric", job.resolution_metric, self.resolution_metric),
            ("bit depth", job.bit_depth, self.bit_depth),
        )
        for label, actual, supported in expected:
            if actual != supported:
                raise ValueError(
                    f"unsupported {label} {actual!r}; this profile requires "
                    f"{supported!r}"
                )
        if job.focus_exposure_locked is not True:
            raise ValueError("main capture must retain prepass focus and exposure")
        acquisition = job.acquisition
        if not isinstance(acquisition, DualRGBIAcquisition):
            raise TypeError("processing requires a dual RGBI acquisition")
        if acquisition.prepass.resolution_dpi != self.prepass_dpi:
            raise ValueError(f"prepass must be captured at {self.prepass_dpi} dpi")
        if acquisition.main.resolution_dpi != self.main_dpi:
            raise ValueError(f"main frame must be captured at {self.main_dpi} dpi")
        main = acquisition.main.pixels
        if main.shape[0] < 5 or main.shape[1] < 8:
            raise ValueError("main RGBI frame must be at least 5 rows by 8 columns")
        if (main.shape[1] // 8) * 8 == 0:
            raise ValueError("main RGBI frame has no complete producer cell")

    def prepass_parameters(self) -> PrepassParameters:
        return PrepassParameters.captured_ls5000_selector8()

    def auxiliary_parameters(
        self,
        record: PrepassFrameRecord,
    ) -> AuxiliaryParameters:
        base = _base_primary(record)
        alpha = np.float32(
            np.float32(record.observed_main_alpha)
            * np.float32(self.auxiliary_factor_b)
        )
        return AuxiliaryParameters(
            selected_visible_channel=0,
            alpha=float(alpha),
            calibration_offset=float(np.float32(1.0)),
            alpha_one_replacement=float(base),
        )

    def score_parameters(self, record: PrepassFrameRecord) -> ScoreParameters:
        return ScoreParameters(
            base_primary=float(_base_primary(record)),
            base_addend=_f32_from_bits(0xC2EED000),
            scale=_f32_from_bits(0xBA887952),
            offset=_f32_from_bits(0x3F800000),
            floor=_f32_from_bits(0x3CA3D70A),
            resolution_metric=self.resolution_metric,
            horizontal_minimum_resolution_cutoff=550,
        )

    def decision_parameters(self) -> DecisionParameters:
        return DecisionParameters(
            sample_threshold=_f32_from_bits(0x4740E73C),
            count_limit=8,
            perpendicular_radius=4,
        )

    def reconstruction_parameters(
        self,
        record: PrepassFrameRecord,
    ) -> ReconstructionParameters:
        f = _f32_from_bits
        return ReconstructionParameters(
            resolution_metric=self.resolution_metric,
            cross_neighbor_cutoff=1_600,
            coarse_enabled=True,
            coarse_reference=float(_base_primary(record)),
            coarse_slopes=(f(0x3F8CCCCD), f(0x3F8CCCCD), f(0x3F8CCCCD)),
            band_enabled=(True, True, True),
            band_scales=(f(0x3FA00000), f(0x3FA00000), f(0x3FA00000)),
            factors_a=(
                (f(0x3F9AE148), f(0x3F9D70A4), f(0x3F90A3D7)),
                (f(0x3F95C28F), f(0x3F91EB85), f(0x3F8A3D71)),
                (f(0x3F851EB8), f(0x3F6E147B), f(0x3F7851EC)),
            ),
            factors_b=(
                (f(0x3F8B851F), f(0x3F90A3D7), f(0x3F851EB8)),
                (f(0x3F8A3D71), f(0x3F866666), f(0x3F828F5C)),
                (f(0x3F75C28F), f(0x3F570A3D), f(0x3F63D70A)),
            ),
            configured_strengths=(0.0, 0.0, 0.0),
            driver_gate_primary=True,
            driver_gate_secondary=True,
            row_reconstruction_gate=0,
            dither_scales=(f(0x3C75C28F), f(0x3C75C28F), f(0x3CCCCCCD)),
        )

    def dither_bounds(self, lookup: npt.ArrayLike) -> DitherBounds:
        return DitherBounds.from_lookup(lookup, maximum_index=65_535)


DEFAULT_PROFILE = LS5000Selector8NormalProfile()


__all__ = [
    "DEFAULT_PROFILE",
    "LS5000Selector8NormalProfile",
    "ProcessingJob",
    "ProcessingMode",
    "ScannerModel",
]
