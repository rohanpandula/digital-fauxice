"""Compiled derivation of the content-derived producer schedule.

`producer_parameters.derive_producer_record_schedule` is deliberately scalar
Python: its docstrings pin a strictly ordered binary64 accumulation per row, a
data-dependent cell-validity walk, and exact x87-style reduction trees inside
each eight-row scale epoch.  None of that order can be vectorized safely, so
this module ports the two inner hot loops line-by-line into `@njit` kernels
and keeps everything else -- validation, the cross-epoch float32 chaining,
record payload packing, and hashing -- in Python, replicating the reference
expressions exactly.  The result is the same `ProducerRecordSchedule` the
reference builds, byte-for-byte (see ``tests/test_fast_cpu_producer_parity``).

Importing this module requires numba.
"""

from __future__ import annotations

import hashlib
import struct

import numpy as np
import numpy.typing as npt
from numba import njit

from ..producer_parameters import (
    PRODUCER_BLOCK_SIZE,
    PRODUCER_CELL_WIDTH,
    ProducerMeanSchedule,
    ProducerRecordSchedule,
    R4000_INFRARED_CHANNEL,
    R4000_INITIAL_INFRARED_MEAN,
    R4000_INITIAL_INPUT_SCALE,
    R4000_INITIAL_SCALED_INPUT,
    R4000_INITIAL_VISIBLE_MEAN,
    R4000_IR_ACCEPTANCE_THRESHOLD,
    R4000_SCALE_MULTIPLIER,
    R4000_VISIBLE_REFERENCE_CHANNEL,
    SCALE_OUTER_WEIGHT,
    SCALE_RATIO_FULL_HIGH,
    SCALE_RATIO_FULL_LOW,
    SCALE_RATIO_HIGH,
    SCALE_RATIO_LOW,
)

FloatArray = npt.NDArray[np.float32]


def _f32(value: float) -> np.float32:
    return np.float32(value)


@njit(cache=True)
def _mean_schedule_kernel(
    pixels,
    table,
    active_width,
    threshold64,
    visible_channel,
    infrared_channel,
    initial_visible_mean,
    initial_infrared_mean,
    out_cumulative_visible,
    out_cumulative_infrared,
    out_cumulative_weight,
    out_visible_means,
    out_infrared_means,
    out_accepted,
):
    """derive_producer_mean_schedule's strictly ordered accumulation walk.

    Column-order binary64 row sums, a row-boundary add into the cumulative
    binary64 fields, one float32 narrowing per emitted mean, and the
    cell-validity latch that survives until the next eight-row block -- all
    exactly as the reference writes them.
    """

    row_count = pixels.shape[0]
    cell_count = (active_width + PRODUCER_CELL_WIDTH - 1) // PRODUCER_CELL_WIDTH
    valid_cells = np.ones(cell_count, dtype=np.uint8)

    total_visible = 0.0
    total_infrared = 0.0
    total_weight = 0.0
    visible_mean = initial_visible_mean
    infrared_mean = initial_infrared_mean

    for row in range(row_count):
        if row % PRODUCER_BLOCK_SIZE == 0:
            for cell in range(cell_count):
                valid_cells[cell] = 1
        row_visible = 0.0
        row_infrared = 0.0
        row_weight = 0.0
        accepted = 0
        for column in range(active_width):
            cell = column // PRODUCER_CELL_WIDTH
            if valid_cells[cell] == 0:
                continue
            raw_infrared = np.int64(pixels[row, column, infrared_channel])
            if not threshold64 < np.float64(raw_infrared):
                valid_cells[cell] = 0
                continue
            weight = raw_infrared * raw_infrared
            raw_visible = np.int64(pixels[row, column, visible_channel])
            row_visible += np.float64(weight) * np.float64(table[raw_visible])
            row_infrared += np.float64(weight) * np.float64(table[raw_infrared])
            row_weight += np.float64(weight)
            accepted += 1
        total_visible += row_visible
        total_infrared += row_infrared
        total_weight += row_weight
        if total_weight != 0.0:
            visible_mean = np.float32(total_visible / total_weight)
            infrared_mean = np.float32(total_infrared / total_weight)
        out_cumulative_visible[row] = total_visible
        out_cumulative_infrared[row] = total_infrared
        out_cumulative_weight[row] = total_weight
        out_visible_means[row] = visible_mean
        out_infrared_means[row] = infrared_mean
        out_accepted[row] = accepted


@njit(cache=True)
def _scale_epoch_kernel(
    pixels,
    table,
    active_width,
    threshold64,
    visible_channel,
    infrared_channel,
    low,
    full_low,
    full_high,
    high,
    outer_weight,
    out_add_denominator,
    out_add_numerator,
):
    """_scale_epoch_additions for every complete eight-row epoch.

    Preserves the asymmetric quadrant-mean reduction trees, the float32
    running raw-infrared sum in row-major order, and the quadrant-order
    denominator/numerator accumulation.
    """

    epoch_count = out_add_denominator.shape[0]
    means = np.empty(4, dtype=np.float32)
    visible_deviations = np.empty(4, dtype=np.float32)
    infrared_deviations = np.empty(4, dtype=np.float32)
    for epoch in range(epoch_count):
        row0 = epoch * PRODUCER_BLOCK_SIZE
        denominator = 0.0
        numerator = 0.0
        for column in range(0, active_width, PRODUCER_BLOCK_SIZE):
            block_ok = True
            for r in range(PRODUCER_BLOCK_SIZE):
                for c in range(PRODUCER_BLOCK_SIZE):
                    raw = pixels[row0 + r, column + c, infrared_channel]
                    if np.float64(raw) <= threshold64:
                        block_ok = False
                        break
                if not block_ok:
                    break
            if not block_ok:
                continue

            for lane in range(2):
                channel = visible_channel if lane == 0 else infrared_channel
                quadrant = 0
                for row_start in (0, 4):
                    for column_start in (0, 4):
                        # 0x73b0's intentionally asymmetric x87 tree: six
                        # sequential adds, then pair-reduced tails.
                        block = np.empty((4, 4), dtype=np.float64)
                        for r in range(4):
                            for c in range(4):
                                block[r, c] = np.float64(
                                    table[
                                        pixels[
                                            row0 + row_start + r,
                                            column + column_start + c,
                                            channel,
                                        ]
                                    ]
                                )
                        first = block[0, 0]
                        first += block[0, 1]
                        first += block[0, 2]
                        first += block[0, 3]
                        first += block[1, 0]
                        first += block[1, 1]
                        second_tail = block[1, 2] + block[1, 3]
                        first += second_tail
                        third = block[2, 0] + block[2, 1]
                        third_tail = block[2, 2] + block[2, 3]
                        third += third_tail
                        fourth = block[3, 0] + block[3, 1]
                        fourth_tail = block[3, 2] + block[3, 3]
                        fourth += fourth_tail
                        third += fourth
                        total = first + third
                        means[quadrant] = np.float32(total * 0.0625)
                        quadrant += 1
                center_total = np.float64(means[0])
                center_total += np.float64(means[1])
                center_total += np.float64(means[2])
                center_total += np.float64(means[3])
                center_mean = np.float32(center_total * 0.25)
                for q in range(4):
                    deviation = np.float32(
                        np.float64(means[q]) - np.float64(center_mean)
                    )
                    if lane == 0:
                        visible_deviations[q] = deviation
                    else:
                        infrared_deviations[q] = deviation

            raw_sum = np.float32(0.0)
            for r in range(PRODUCER_BLOCK_SIZE):
                for c in range(PRODUCER_BLOCK_SIZE):
                    raw_sum = np.float32(
                        np.float64(raw_sum)
                        + np.float64(pixels[row0 + r, column + c, infrared_channel])
                    )
            raw_sum_wide = np.float64(raw_sum)

            for q in range(4):
                visible_wide = np.float64(visible_deviations[q])
                if visible_wide == 0.0:
                    ratio = 0.0
                    weight = 0.0
                else:
                    ratio = np.float64(infrared_deviations[q]) / visible_wide
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
        out_add_denominator[epoch] = np.float32(denominator)
        out_add_numerator[epoch] = np.float32(numerator)


def derive_producer_mean_schedule_fast(
    rgbi16: npt.ArrayLike,
    response_lut: npt.ArrayLike,
    *,
    active_width: int | None = None,
    infrared_threshold: float = float(R4000_IR_ACCEPTANCE_THRESHOLD),
    initial_visible_mean: float = 0.0,
    initial_infrared_mean: float = 0.0,
) -> ProducerMeanSchedule:
    """Compiled mirror of ``derive_producer_mean_schedule``."""

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
        active_width = (pixels.shape[1] // PRODUCER_CELL_WIDTH) * PRODUCER_CELL_WIDTH
    if not 0 < active_width <= pixels.shape[1]:
        raise ValueError("producer active width is outside the RGBI row")
    threshold = np.float32(infrared_threshold)
    if not np.isfinite(threshold):
        raise ValueError("producer infrared threshold must be finite float32")
    visible_mean = np.float32(initial_visible_mean)
    infrared_mean = np.float32(initial_infrared_mean)
    if not np.isfinite(visible_mean) or not np.isfinite(infrared_mean):
        raise ValueError("initial producer means must be finite float32")

    row_count = pixels.shape[0]
    cumulative_visible = np.empty(row_count, dtype=np.float64)
    cumulative_infrared = np.empty(row_count, dtype=np.float64)
    cumulative_weight = np.empty(row_count, dtype=np.float64)
    visible_means = np.empty(row_count, dtype=np.float32)
    infrared_means = np.empty(row_count, dtype=np.float32)
    accepted_counts = np.empty(row_count, dtype=np.uint32)

    _mean_schedule_kernel(
        np.ascontiguousarray(pixels),
        table,
        active_width,
        float(threshold),
        R4000_VISIBLE_REFERENCE_CHANNEL,
        R4000_INFRARED_CHANNEL,
        visible_mean,
        infrared_mean,
        cumulative_visible,
        cumulative_infrared,
        cumulative_weight,
        visible_means,
        infrared_means,
        accepted_counts,
    )

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


def derive_producer_record_schedule_fast(
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
    """Compiled mirror of ``derive_producer_record_schedule``.

    The cross-epoch float32 chaining below reproduces the reference lines
    exactly: current float32 accumulators widened, added to the epoch's
    float32 additions, narrowed back to float32 stores, with the still-live
    binary64 sums feeding ``p1``/``p2``.
    """

    pixels = np.asarray(rgbi16)
    table = np.asarray(response_lut)
    means = derive_producer_mean_schedule_fast(
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
    if not all(np.isfinite(value) for value in (multiplier, current_p1, current_p2)):
        raise ValueError("producer scale inputs must be finite float32")

    threshold = _f32(means.infrared_threshold)
    epoch_count = row_count // PRODUCER_BLOCK_SIZE
    epoch_add_denominators = np.zeros(max(epoch_count, 1), dtype=np.float32)
    epoch_add_numerators = np.zeros(max(epoch_count, 1), dtype=np.float32)
    if epoch_count:
        _scale_epoch_kernel(
            np.ascontiguousarray(pixels),
            table,
            active_width,
            float(threshold),
            R4000_VISIBLE_REFERENCE_CHANNEL,
            R4000_INFRARED_CHANNEL,
            float(SCALE_RATIO_LOW),
            float(SCALE_RATIO_FULL_LOW),
            float(SCALE_RATIO_FULL_HIGH),
            float(SCALE_RATIO_HIGH),
            float(SCALE_OUTER_WEIGHT),
            epoch_add_denominators[:epoch_count],
            epoch_add_numerators[:epoch_count],
        )

    p1_values = np.empty(row_count, dtype=np.float32)
    p2_values = np.empty(row_count, dtype=np.float32)
    denominators = np.empty(row_count, dtype=np.float32)
    numerators = np.empty(row_count, dtype=np.float32)
    add_denominators = np.zeros(row_count, dtype=np.float32)
    add_numerators = np.zeros(row_count, dtype=np.float32)

    for row in range(row_count):
        if row % PRODUCER_BLOCK_SIZE == PRODUCER_BLOCK_SIZE - 1:
            epoch = row // PRODUCER_BLOCK_SIZE
            add_denominator = epoch_add_denominators[epoch]
            add_numerator = epoch_add_numerators[epoch]
            add_denominators[row] = add_denominator
            add_numerators[row] = add_numerator
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
