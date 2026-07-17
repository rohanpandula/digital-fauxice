"""Stage-varying X3A parameters recovered from Nikon's 0x40-byte record.

The X3A scheduler does not keep its analysis calibration fixed for a whole
frame.  Before each scheduler stage Nikon passes four float32 values from the
active 0x40-byte record to a virtual setter.  The setter updates the analysis
coefficient, score/coarse reference, and two writer gates.  This module keeps
that dynamic state separate from the stable reconstruction tuning.

Only parameters derived from the current input pair are accepted. Recorded
trace loaders and external evidence files are intentionally outside the
runtime package.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


SETTER_SOURCE_RECORD_BYTES = 0x40
SETTER_WRITER_GATE_THRESHOLD = np.float32(0.075)


def _f32(value: float, label: str) -> np.float32:
    result = np.float32(value)
    if not np.isfinite(result):
        raise ValueError(f"{label} must be finite float32")
    return result


@dataclass(frozen=True)
class X3ASetterInputs:
    """The four source-record values passed to Nikon's virtual setter.

    Static caller recovery binds the fields as follows: ``p1=F+0x08``,
    ``p2=F+0x04``, ``p3=F+0x28``, and ``p4=F+0x2c``.
    """

    scaled_input: float
    input_scale: float
    calibration_mean_visible: float
    calibration_mean_infrared: float
    source_record_sha256: str

    def __post_init__(self) -> None:
        for label in (
            "scaled_input",
            "input_scale",
            "calibration_mean_visible",
            "calibration_mean_infrared",
        ):
            object.__setattr__(self, label, float(_f32(getattr(self, label), label)))
        if len(self.source_record_sha256) != 64:
            raise ValueError("source record SHA-256 must contain 64 hex characters")
        try:
            bytes.fromhex(self.source_record_sha256)
        except ValueError as error:
            raise ValueError("source record SHA-256 is not hexadecimal") from error

    @classmethod
    def from_record_payload(cls, payload: bytes) -> "X3ASetterInputs":
        """Decode the exact four caller fields from one 0x40-byte record."""

        raw = bytes(payload)
        if len(raw) != SETTER_SOURCE_RECORD_BYTES:
            raise ValueError("X3A setter source record must contain exactly 0x40 bytes")
        return cls(
            scaled_input=struct.unpack_from("<f", raw, 0x08)[0],
            input_scale=struct.unpack_from("<f", raw, 0x04)[0],
            calibration_mean_visible=struct.unpack_from("<f", raw, 0x28)[0],
            calibration_mean_infrared=struct.unpack_from("<f", raw, 0x2C)[0],
            source_record_sha256=hashlib.sha256(raw).hexdigest(),
        )


@dataclass(frozen=True)
class X3ASetterDerivedFields:
    """Fields whose setter equations are closed by static analysis."""

    scaled_input: float
    scaled_input_reciprocal: float
    input_scale: float
    input_scale_reciprocal: float
    base_primary: float
    writer_gate_secondary: bool


def derive_x3a_setter_fields(inputs: X3ASetterInputs) -> X3ASetterDerivedFields:
    """Replay Nikon's closed setter equations with float32 source operands.

    The installed x87 control word uses 53-bit intermediates.  Source values
    are therefore quantized to float32, widened for each expression, and
    narrowed once at the destination field.  Nikon flips ``p1`` and ``p2``
    together when ``p2`` is negative.
    """

    p1 = _f32(inputs.scaled_input, "scaled input")
    p2 = _f32(inputs.input_scale, "input scale")
    p3 = _f32(inputs.calibration_mean_visible, "visible calibration mean")
    p4 = _f32(inputs.calibration_mean_infrared, "infrared calibration mean")
    if p2 < np.float32(0.0):
        p1 = np.float32(-p1)
        p2 = np.float32(-p2)
    denominator_p1 = 1.0 - float(p1)
    denominator_p2 = 1.0 - float(p2)
    if denominator_p1 == 0.0 or denominator_p2 == 0.0:
        raise ValueError("X3A setter scale cannot equal one")
    reciprocal_p1 = np.float32(1.0 / denominator_p1)
    reciprocal_p2 = np.float32(1.0 / denominator_p2)
    base_primary = np.float32(
        (float(p4) - float(p3) * float(p2)) / denominator_p2
    )
    gate_secondary = not (
        p3 != np.float32(0.0) and p2 <= SETTER_WRITER_GATE_THRESHOLD
    )
    return X3ASetterDerivedFields(
        scaled_input=float(p1),
        scaled_input_reciprocal=float(reciprocal_p1),
        input_scale=float(p2),
        input_scale_reciprocal=float(reciprocal_p2),
        base_primary=float(base_primary),
        writer_gate_secondary=gate_secondary,
    )


@dataclass(frozen=True)
class StageCalibration:
    """Dynamic fields consumed while analyzing or writing one X3A stage."""

    stage_hit: int
    auxiliary_alpha: float
    calibration_offset: float
    base_primary: float
    row_reconstruction_gate: int
    writer_gate_secondary: bool
    setter_inputs: X3ASetterInputs

    def __post_init__(self) -> None:
        if self.stage_hit <= 0:
            raise ValueError("stage hit must be positive")
        for label in ("auxiliary_alpha", "calibration_offset", "base_primary"):
            object.__setattr__(self, label, float(_f32(getattr(self, label), label)))
        if self.row_reconstruction_gate < 0:
            raise ValueError("row reconstruction gate cannot be negative")

    @classmethod
    def from_setter_inputs(
        cls,
        *,
        stage_hit: int,
        setter_inputs: X3ASetterInputs,
        auxiliary_factor_b: float,
        calibration_offset: float,
        row_reconstruction_gate: int,
        observed_base_primary: float | None = None,
        observed_writer_gate_secondary: bool | None = None,
    ) -> "StageCalibration":
        """Build and optionally cross-check one runtime-observed stage.

        ``calibration_offset`` (object ``+0xf78``) and the row gate (``+0xf8c``)
        remain explicit because Nikon's two homologous setter classes assign
        different residual defaults and the row gate also consumes other
        stable context/LUT fields.
        """

        derived = derive_x3a_setter_fields(setter_inputs)
        factor_b = _f32(auxiliary_factor_b, "auxiliary factor B")
        alpha = np.float32(np.float32(derived.input_scale) * factor_b)
        if observed_base_primary is not None:
            observed = _f32(observed_base_primary, "observed base primary")
            if observed.view(np.uint32) != np.float32(derived.base_primary).view(
                np.uint32
            ):
                raise ValueError(
                    "observed +0xf2c disagrees with the recovered setter equation"
                )
        if (
            observed_writer_gate_secondary is not None
            and observed_writer_gate_secondary != derived.writer_gate_secondary
        ):
            raise ValueError(
                "observed +0xf90 disagrees with the recovered setter branch"
            )
        return cls(
            stage_hit=stage_hit,
            auxiliary_alpha=float(alpha),
            calibration_offset=calibration_offset,
            base_primary=derived.base_primary,
            row_reconstruction_gate=row_reconstruction_gate,
            writer_gate_secondary=derived.writer_gate_secondary,
            setter_inputs=setter_inputs,
        )


@runtime_checkable
class StageParameterProvider(Protocol):
    """Return the exact dynamic calibration active at a scheduler stage."""

    def __call__(self, stage_hit: int) -> StageCalibration: ...


def source_stage_for_row(row: int) -> int:
    """Map a zero-based input row to the stage that analyzes it."""

    if row < 0:
        raise ValueError("source row cannot be negative")
    return row + 1


def score_stage_for_row(row: int) -> int:
    """Return the scheduler stage whose calibration scores source ``row``.

    X3A derives the row's auxiliary plane when the source record enters the
    scheduler, then produces the continuous-score plane one stage later.
    """

    if row < 0:
        raise ValueError("source row cannot be negative")
    return row + 2


def writer_stage_for_row(row: int) -> int:
    """Map a zero-based output row to Nikon's six-stage-delayed writer."""

    if row < 0:
        raise ValueError("output row cannot be negative")
    return row + 7


def validate_stage_calibration(
    calibration: StageCalibration,
    *,
    expected_stage_hit: int | None = None,
) -> None:
    """Defensive public validator for custom provider implementations."""

    if not isinstance(calibration, StageCalibration):
        raise TypeError("stage parameter provider must return StageCalibration")
    if (
        expected_stage_hit is not None
        and calibration.stage_hit != expected_stage_hit
    ):
        raise ValueError(
            "stage parameter provider returned calibration for "
            f"stage {calibration.stage_hit}, expected {expected_stage_hit}"
        )
    numeric = (
        calibration.auxiliary_alpha,
        calibration.calibration_offset,
        calibration.base_primary,
    )
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("stage calibration contains a non-finite value")
