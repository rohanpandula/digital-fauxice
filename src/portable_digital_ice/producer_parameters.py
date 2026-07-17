"""Portable replay of Nikon's content-derived X3A producer calibration.

The live LS-5000 path builds the four float32 values later consumed by the
X3A setter in two independent pieces.  This module implements the closed
per-row piece: the cumulative visible/infrared calibration means stored at
record offsets ``+0x28`` and ``+0x2c``.

The equations below are bound to the active 0x10006d70/0x10006ed0 producer
variant and were checked against all 676 records in the authoritative 4000
dpi trace.  Both the per-row means and the separate eight-row
``+0x04``/``+0x08`` scale accumulator are closed; complete 0x40-byte records
reproduce all 16 trace words exactly.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.float32]
Float64Array = npt.NDArray[np.float64]

LIVE_SAMPLE_ACCUMULATOR_RVA = 0x6D70
LIVE_ROW_FINALIZER_RVA = 0x6ED0
LIVE_SCALE_ACCUMULATOR_RVA = 0x72D0

# Init RVA 0x58f0 computes object+0x20 as ``caller_width >> 3``.  Every cell
# is eight columns, so trailing transport columns outside a whole cell do not
# enter this producer (443 -> 440; 3946 -> 3944).
PRODUCER_CELL_WIDTH = 8
R4000_IR_ACCEPTANCE_THRESHOLD = np.float32(8847.2255859375)
R4000_VISIBLE_REFERENCE_CHANNEL = 0
R4000_INFRARED_CHANNEL = 3
PRODUCER_BLOCK_SIZE = 8
SCALE_RATIO_LOW = np.float32(-0.125)
SCALE_RATIO_FULL_LOW = np.float32(0.0)
SCALE_RATIO_FULL_HIGH = np.float32(0.3)
SCALE_RATIO_HIGH = np.float32(0.425)
SCALE_OUTER_WEIGHT = np.float32(0.70710677)
R4000_INITIAL_SCALED_INPUT = np.float32(0.15)
R4000_INITIAL_INPUT_SCALE = np.float32(0.15)
R4000_INITIAL_VISIBLE_MEAN = np.float32(46618.78515625)
R4000_INITIAL_INFRARED_MEAN = np.float32(56239.65625)
R4000_SCALE_MULTIPLIER = np.float32(1.5)


@dataclass(frozen=True)
class ProducerMeanSchedule:
    """Cumulative producer state and the two emitted float32 means per row."""

    weighted_visible_sum: Float64Array
    weighted_infrared_sum: Float64Array
    weight_sum: Float64Array
    calibration_mean_visible: FloatArray
    calibration_mean_infrared: FloatArray
    accepted_samples: npt.NDArray[np.uint32]
    active_width: int
    infrared_threshold: float

    def __post_init__(self) -> None:
        row_count = self.calibration_mean_visible.shape[0]
        expected = (row_count,)
        arrays = (
            (self.weighted_visible_sum, np.dtype(np.float64)),
            (self.weighted_infrared_sum, np.dtype(np.float64)),
            (self.weight_sum, np.dtype(np.float64)),
            (self.calibration_mean_visible, np.dtype(np.float32)),
            (self.calibration_mean_infrared, np.dtype(np.float32)),
            (self.accepted_samples, np.dtype(np.uint32)),
        )
        for values, dtype in arrays:
            if values.shape != expected or values.dtype != dtype:
                raise ValueError("producer schedule arrays disagree in shape/dtype")
            values.flags.writeable = False
        if self.active_width <= 0:
            raise ValueError("producer active width must be positive")


@dataclass(frozen=True)
class ProducerRecordSchedule:
    """Complete active 0x40-byte producer record after every source row."""

    mean_schedule: ProducerMeanSchedule
    scaled_input: FloatArray
    input_scale: FloatArray
    scale_denominator: FloatArray
    scale_numerator: FloatArray
    scale_add_denominator: FloatArray
    scale_add_numerator: FloatArray
    record_payloads: tuple[bytes, ...]
    scale_multiplier: float
    source_rgbi_sha256: str
    response_lut_sha256: str

    def __post_init__(self) -> None:
        row_count = self.mean_schedule.calibration_mean_visible.shape[0]
        expected = (row_count,)
        for values in (
            self.scaled_input,
            self.input_scale,
            self.scale_denominator,
            self.scale_numerator,
            self.scale_add_denominator,
            self.scale_add_numerator,
        ):
            if values.shape != expected or values.dtype != np.dtype(np.float32):
                raise ValueError("producer record schedule arrays disagree")
            values.flags.writeable = False
        if len(self.record_payloads) != row_count or any(
            len(payload) != 0x40 for payload in self.record_payloads
        ):
            raise ValueError("producer record payloads must be one 0x40 record per row")
        for digest in (self.source_rgbi_sha256, self.response_lut_sha256):
            if len(digest) != 64:
                raise ValueError("producer source hashes must be SHA-256 hex")

    @property
    def row_count(self) -> int:
        return len(self.record_payloads)

    def record_payload(self, row: int) -> bytes:
        if not 0 <= row < self.row_count:
            raise IndexError(row)
        return self.record_payloads[row]


def _f32(value: float) -> np.float32:
    return np.float32(value)


def _quadrant_means(block: FloatArray) -> tuple[np.float32, ...]:
    """Replay 0x100073b0's four stored float32 4x4 means."""

    if block.shape != (PRODUCER_BLOCK_SIZE, PRODUCER_BLOCK_SIZE):
        raise ValueError("producer response block must be 8x8")
    means: list[np.float32] = []
    for row_start, column_start in ((0, 0), (0, 4), (4, 0), (4, 4)):
        values = block[
            row_start : row_start + 4,
            column_start : column_start + 4,
        ]
        # 0x73b0's x87 tree is intentionally asymmetric: the first six
        # values are sequential, the remaining pairs are reduced before the
        # final adds.  A row-major reduction can differ in low binary64 bits.
        first = float(values[0, 0])
        first += float(values[0, 1])
        first += float(values[0, 2])
        first += float(values[0, 3])
        first += float(values[1, 0])
        first += float(values[1, 1])
        second_tail = float(values[1, 2]) + float(values[1, 3])
        first += second_tail
        third = float(values[2, 0]) + float(values[2, 1])
        third_tail = float(values[2, 2]) + float(values[2, 3])
        third += third_tail
        fourth = float(values[3, 0]) + float(values[3, 1])
        fourth_tail = float(values[3, 2]) + float(values[3, 3])
        fourth += fourth_tail
        third += fourth
        total = first + third
        means.append(_f32(total * 0.0625))
    return tuple(means)


def _center_quadrants(values: tuple[np.float32, ...]) -> tuple[np.float32, ...]:
    """Replay 0x10007520/0x10007540/0x10007560 store boundaries."""

    if len(values) != 4:
        raise ValueError("producer quadrant vector must contain four values")
    total = float(values[0])
    total += float(values[1])
    total += float(values[2])
    total += float(values[3])
    mean = _f32(total * 0.25)
    return tuple(_f32(float(value) - float(mean)) for value in values)


def _scale_epoch_additions(
    pixels: npt.NDArray[np.uint16],
    table: FloatArray,
    *,
    active_width: int,
    threshold: np.float32,
) -> tuple[np.float32, np.float32]:
    """Replay one eight-row 0x6ed0 scale finalization.

    The producer stores one 8x8 visible-response block, one infrared-response
    block, and a float32 running sum of raw infrared.  A threshold failure
    clears the block-valid word, so the entire block is omitted here.
    """

    if pixels.shape != (PRODUCER_BLOCK_SIZE, pixels.shape[1], 4):
        raise ValueError("producer scale epoch must contain eight RGBI rows")
    if active_width % PRODUCER_BLOCK_SIZE:
        raise ValueError("producer active width must contain whole 8x8 blocks")
    denominator = 0.0
    numerator = 0.0
    low = float(SCALE_RATIO_LOW)
    full_low = float(SCALE_RATIO_FULL_LOW)
    full_high = float(SCALE_RATIO_FULL_HIGH)
    high = float(SCALE_RATIO_HIGH)
    outer_weight = float(SCALE_OUTER_WEIGHT)
    threshold_value = float(threshold)
    for column in range(0, active_width, PRODUCER_BLOCK_SIZE):
        raw_infrared = pixels[
            :,
            column : column + PRODUCER_BLOCK_SIZE,
            R4000_INFRARED_CHANNEL,
        ]
        if np.any(raw_infrared <= threshold_value):
            continue
        visible_response = np.ascontiguousarray(
            table[
                pixels[
                    :,
                    column : column + PRODUCER_BLOCK_SIZE,
                    R4000_VISIBLE_REFERENCE_CHANNEL,
                ]
            ],
            dtype=np.float32,
        )
        infrared_response = np.ascontiguousarray(
            table[raw_infrared],
            dtype=np.float32,
        )
        visible_deviations = _center_quadrants(
            _quadrant_means(visible_response)
        )
        infrared_deviations = _center_quadrants(
            _quadrant_means(infrared_response)
        )
        raw_sum = np.float32(0.0)
        for raw_value in raw_infrared.reshape(-1):
            raw_sum = _f32(float(raw_sum) + int(raw_value))
        raw_sum_wide = float(raw_sum)
        for visible, infrared in zip(
            visible_deviations,
            infrared_deviations,
            strict=True,
        ):
            visible_wide = float(visible)
            if visible_wide == 0.0:
                ratio = 0.0
                weight = 0.0
            else:
                ratio = float(infrared) / visible_wide
                if ratio < low or ratio > high:
                    weight = 0.0
                elif ratio < full_low or ratio > full_high:
                    weight = visible_wide * outer_weight
                else:
                    weight = visible_wide
            term = weight * weight
            term *= raw_sum_wide
            term *= raw_sum_wide
            denominator += term
            numerator += ratio * term
    return _f32(denominator), _f32(numerator)


def derive_producer_mean_schedule(
    rgbi16: npt.ArrayLike,
    response_lut: npt.ArrayLike,
    *,
    active_width: int | None = None,
    infrared_threshold: float = float(R4000_IR_ACCEPTANCE_THRESHOLD),
    initial_visible_mean: float = 0.0,
    initial_infrared_mean: float = 0.0,
) -> ProducerMeanSchedule:
    """Replay the active producer's cumulative ``p3``/``p4`` schedule.

    For every accepted pixel Nikon uses ``w = raw_IR ** 2`` and accumulates
    ``w * response(raw_R)``, ``w * response(raw_IR)``, and ``w``.  It first
    sums one row with binary64 arithmetic, then adds that row to the three
    cumulative binary64 fields.  The emitted means are narrowed once to
    float32.  Preserving that row boundary is necessary for byte identity.

    The active mode accepts infrared samples strictly greater than the lower
    producer threshold.  If the cumulative weight remains zero, Nikon leaves
    the previous means untouched.
    """

    pixels = np.asarray(rgbi16)
    if pixels.dtype != np.dtype(np.uint16):
        raise TypeError("producer RGBI input must have dtype uint16")
    if pixels.ndim != 3 or pixels.shape[2] != 4:
        raise ValueError("producer RGBI input must have shape HxWx4")
    table = np.asarray(response_lut)
    if (
        table.dtype != np.dtype(np.float32)
        or table.shape != (65536,)
        or not np.all(np.isfinite(table))
    ):
        raise ValueError("producer response LUT must be finite float32[65536]")
    if active_width is None:
        active_width = (
            pixels.shape[1] // PRODUCER_CELL_WIDTH
        ) * PRODUCER_CELL_WIDTH
    if not 0 < active_width <= pixels.shape[1]:
        raise ValueError("producer active width is outside the RGBI row")
    threshold = np.float32(infrared_threshold)
    if not np.isfinite(threshold):
        raise ValueError("producer infrared threshold must be finite float32")

    row_count = pixels.shape[0]
    cumulative_visible = np.empty(row_count, dtype=np.float64)
    cumulative_infrared = np.empty(row_count, dtype=np.float64)
    cumulative_weight = np.empty(row_count, dtype=np.float64)
    visible_means = np.empty(row_count, dtype=np.float32)
    infrared_means = np.empty(row_count, dtype=np.float32)
    accepted_counts = np.empty(row_count, dtype=np.uint32)

    total_visible = 0.0
    total_infrared = 0.0
    total_weight = 0.0
    visible_mean = np.float32(initial_visible_mean)
    infrared_mean = np.float32(initial_infrared_mean)
    if not np.isfinite(visible_mean) or not np.isfinite(infrared_mean):
        raise ValueError("initial producer means must be finite float32")

    cell_count = (active_width + PRODUCER_CELL_WIDTH - 1) // PRODUCER_CELL_WIDTH
    valid_cells = np.ones(cell_count, dtype=np.bool_)

    # Deliberately scalar: Nikon's x87 path rounds after each binary64 add.
    # NumPy reductions use a different tree and drift by several kilobytes at
    # this magnitude even though the final float32 means often still agree.
    for row in range(row_count):
        if row % PRODUCER_BLOCK_SIZE == 0:
            valid_cells.fill(True)
        row_visible = 0.0
        row_infrared = 0.0
        row_weight = 0.0
        accepted = 0
        for column in range(active_width):
            cell = column // PRODUCER_CELL_WIDTH
            if not valid_cells[cell]:
                continue
            raw_infrared = int(pixels[row, column, R4000_INFRARED_CHANNEL])
            if not float(threshold) < raw_infrared:
                valid_cells[cell] = False
                continue
            weight = raw_infrared * raw_infrared
            raw_visible = int(
                pixels[row, column, R4000_VISIBLE_REFERENCE_CHANNEL]
            )
            row_visible += float(weight) * float(table[raw_visible])
            row_infrared += float(weight) * float(table[raw_infrared])
            row_weight += float(weight)
            accepted += 1
        total_visible += row_visible
        total_infrared += row_infrared
        total_weight += row_weight
        if total_weight != 0.0:
            visible_mean = np.float32(total_visible / total_weight)
            infrared_mean = np.float32(total_infrared / total_weight)
        cumulative_visible[row] = total_visible
        cumulative_infrared[row] = total_infrared
        cumulative_weight[row] = total_weight
        visible_means[row] = visible_mean
        infrared_means[row] = infrared_mean
        accepted_counts[row] = accepted

    return ProducerMeanSchedule(
        weighted_visible_sum=cumulative_visible,
        weighted_infrared_sum=cumulative_infrared,
        weight_sum=cumulative_weight,
        calibration_mean_visible=visible_means,
        calibration_mean_infrared=infrared_means,
        accepted_samples=accepted_counts,
        active_width=active_width,
        infrared_threshold=float(threshold),
    )


def derive_producer_record_schedule(
    rgbi16: npt.ArrayLike,
    response_lut: npt.ArrayLike,
    *,
    active_width: int | None = None,
    infrared_threshold: float = float(R4000_IR_ACCEPTANCE_THRESHOLD),
    initial_scaled_input: float = float(R4000_INITIAL_SCALED_INPUT),
    initial_input_scale: float = float(R4000_INITIAL_INPUT_SCALE),
    initial_visible_mean: float = float(R4000_INITIAL_VISIBLE_MEAN),
    initial_infrared_mean: float = float(R4000_INITIAL_INFRARED_MEAN),
    scale_multiplier: float = float(R4000_SCALE_MULTIPLIER),
) -> ProducerRecordSchedule:
    """Derive every active 0x40-byte producer record from RGBI content.

    This combines the per-row cumulative means with the independent eight-row
    scale estimator at 0x6ed0.  The active run has both optional rolling
    windows disabled.  At 0x72d0 the current float32 additions are widened and
    added to the previous float32 accumulators; the stored accumulators narrow
    to float32, while the still-live unrounded binary64 sums feed ``p2`` and
    ``p1``.  Reloading the stored fields would be one ULP wrong on this trace.
    """

    pixels = np.asarray(rgbi16)
    table = np.asarray(response_lut)
    means = derive_producer_mean_schedule(
        pixels,
        table,
        active_width=active_width,
        infrared_threshold=infrared_threshold,
        initial_visible_mean=initial_visible_mean,
        initial_infrared_mean=initial_infrared_mean,
    )
    active_width = means.active_width
    if active_width % PRODUCER_BLOCK_SIZE:
        raise ValueError("complete producer records require whole 8x8 cells")
    row_count = pixels.shape[0]
    if row_count <= 0:
        raise ValueError("producer record schedule cannot be empty")
    multiplier = _f32(scale_multiplier)
    current_p1 = _f32(initial_scaled_input)
    current_p2 = _f32(initial_input_scale)
    denominator = np.float32(0.0)
    numerator = np.float32(0.0)
    if not all(
        np.isfinite(value)
        for value in (multiplier, current_p1, current_p2)
    ):
        raise ValueError("producer scale inputs must be finite float32")

    p1_values = np.empty(row_count, dtype=np.float32)
    p2_values = np.empty(row_count, dtype=np.float32)
    denominators = np.empty(row_count, dtype=np.float32)
    numerators = np.empty(row_count, dtype=np.float32)
    add_denominators = np.zeros(row_count, dtype=np.float32)
    add_numerators = np.zeros(row_count, dtype=np.float32)
    threshold = _f32(means.infrared_threshold)

    for row in range(row_count):
        if row % PRODUCER_BLOCK_SIZE == PRODUCER_BLOCK_SIZE - 1:
            start = row + 1 - PRODUCER_BLOCK_SIZE
            add_denominator, add_numerator = _scale_epoch_additions(
                pixels[start : row + 1],
                table,
                active_width=active_width,
                threshold=threshold,
            )
            add_denominators[row] = add_denominator
            add_numerators[row] = add_numerator
            # x87 CW 0x023f: 53-bit precision, round-to-nearest-even.
            denominator_wide = float(denominator) + float(add_denominator)
            numerator_wide = float(numerator) + float(add_numerator)
            denominator = _f32(denominator_wide)
            numerator = _f32(numerator_wide)
            if denominator_wide != 0.0:
                ratio = numerator_wide / denominator_wide
                current_p2 = _f32(ratio)
                current_p1 = _f32(float(multiplier) * ratio)
        p1_values[row] = current_p1
        p2_values[row] = current_p2
        denominators[row] = denominator
        numerators[row] = numerator

    payloads: list[bytes] = []
    for row in range(row_count):
        payload = bytearray(0x40)
        struct.pack_into("<f", payload, 0x04, p2_values[row])
        struct.pack_into("<f", payload, 0x08, p1_values[row])
        struct.pack_into(
            "<ddd",
            payload,
            0x10,
            means.weighted_visible_sum[row],
            means.weighted_infrared_sum[row],
            means.weight_sum[row],
        )
        struct.pack_into("<f", payload, 0x28, means.calibration_mean_visible[row])
        struct.pack_into("<f", payload, 0x2C, means.calibration_mean_infrared[row])
        struct.pack_into("<f", payload, 0x30, denominators[row])
        struct.pack_into("<f", payload, 0x34, numerators[row])
        payloads.append(bytes(payload))

    canonical_pixels = np.ascontiguousarray(pixels, dtype="<u2")
    canonical_table = np.ascontiguousarray(table, dtype="<f4")
    return ProducerRecordSchedule(
        mean_schedule=means,
        scaled_input=p1_values,
        input_scale=p2_values,
        scale_denominator=denominators,
        scale_numerator=numerators,
        scale_add_denominator=add_denominators,
        scale_add_numerator=add_numerators,
        record_payloads=tuple(payloads),
        scale_multiplier=float(multiplier),
        source_rgbi_sha256=hashlib.sha256(canonical_pixels.tobytes()).hexdigest(),
        response_lut_sha256=hashlib.sha256(canonical_table.tobytes()).hexdigest(),
    )


class ContentDerivedStageParameterProvider:
    """Build X3A stage calibration from portable producer records.

    Producer row zero is installed before scheduler stage one.  After the last
    source row Nikon holds the final record through the six drain stages.
    """

    def __init__(
        self,
        schedule: ProducerRecordSchedule,
        *,
        auxiliary_factor_b: float,
        hold_last_through_stage: int | None = None,
    ) -> None:
        from .stage_parameters import StageCalibration, X3ASetterInputs

        self.schedule = schedule
        self.auxiliary_factor_b = float(_f32(auxiliary_factor_b))
        if hold_last_through_stage is None:
            hold_last_through_stage = schedule.row_count + 6
        if hold_last_through_stage < schedule.row_count:
            raise ValueError("producer hold cannot end before its source rows")
        self.hold_last_through_stage = hold_last_through_stage
        calibrations: list[StageCalibration] = []
        for stage_hit in range(1, hold_last_through_stage + 1):
            row = min(stage_hit - 1, schedule.row_count - 1)
            inputs = X3ASetterInputs.from_record_payload(
                schedule.record_payload(row)
            )
            calibrations.append(
                StageCalibration.from_setter_inputs(
                    stage_hit=stage_hit,
                    setter_inputs=inputs,
                    auxiliary_factor_b=self.auxiliary_factor_b,
                    calibration_offset=1.0,
                    row_reconstruction_gate=0,
                )
            )
        self._calibrations = tuple(calibrations)

    def __call__(self, stage_hit: int):
        if not 1 <= stage_hit <= self.hold_last_through_stage:
            raise KeyError(f"content-derived stage is unbound: {stage_hit}")
        return self._calibrations[stage_hit - 1]

    def require_range(self, first_stage: int, last_stage: int) -> None:
        if (
            first_stage <= 0
            or last_stage < first_stage
            or last_stage > self.hold_last_through_stage
        ):
            raise KeyError(
                f"content-derived stages are outside 1..{self.hold_last_through_stage}"
            )

    @property
    def stage_count(self) -> int:
        return self.schedule.row_count

    @property
    def provided_stage_count(self) -> int:
        return self.hold_last_through_stage
