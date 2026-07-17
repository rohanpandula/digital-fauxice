"""CUDA-assisted content-derived producer schedule.

The reference producer (:mod:`..producer_parameters`) is a strictly ordered
scalar accumulation.  The row-internal sums are independent per row, so the
GPU computes them one thread per row in the exact column order; the cross-row
cumulative state and the eight-row scale accumulator round after every add and
therefore stay on the host, reproducing the reference loop verbatim.

Cell validity is a prefix property: within one eight-row block a cell stops
accepting samples at its first below-threshold infrared sample, scanned in
(row, column) order.  ``k_producer_failpos`` finds that first failure per
(block, cell) so row threads can evaluate acceptance without shared state.
"""

from __future__ import annotations

import hashlib
import struct

import numpy as np
import numpy.typing as npt

from ..producer_parameters import (
    PRODUCER_BLOCK_SIZE,
    PRODUCER_CELL_WIDTH,
    ProducerMeanSchedule,
    ProducerRecordSchedule,
    R4000_INITIAL_INFRARED_MEAN,
    R4000_INITIAL_INPUT_SCALE,
    R4000_INITIAL_SCALED_INPUT,
    R4000_INITIAL_VISIBLE_MEAN,
    R4000_IR_ACCEPTANCE_THRESHOLD,
    R4000_SCALE_MULTIPLIER,
    R4000_VISIBLE_REFERENCE_CHANNEL,
)


def derive_producer_record_schedule_cuda(
    module,
    rgbi16: npt.NDArray[np.uint16],
    response_lut: npt.NDArray[np.float32],
    *,
    device_pixels=None,
    device_lut=None,
) -> ProducerRecordSchedule:
    """Reproduce ``derive_producer_record_schedule`` with GPU row/epoch sums."""

    import cupy as cp

    pixels = np.asarray(rgbi16)
    if pixels.dtype != np.dtype(np.uint16) or pixels.ndim != 3 or pixels.shape[2] != 4:
        raise ValueError("producer RGBI input must be uint16 HxWx4")
    table = np.asarray(response_lut)
    if table.dtype != np.dtype(np.float32) or table.shape != (65536,):
        raise ValueError("producer response LUT must be float32[65536]")

    height, width, _ = pixels.shape
    active_width = (width // PRODUCER_CELL_WIDTH) * PRODUCER_CELL_WIDTH
    if active_width % PRODUCER_BLOCK_SIZE:
        raise ValueError("complete producer records require whole 8x8 cells")
    if active_width <= 0:
        raise ValueError("producer active width is outside the RGBI row")
    threshold = np.float32(R4000_IR_ACCEPTANCE_THRESHOLD)
    cell_count = active_width // PRODUCER_CELL_WIDTH
    block_count = (height + PRODUCER_BLOCK_SIZE - 1) // PRODUCER_BLOCK_SIZE
    epoch_count = height // PRODUCER_BLOCK_SIZE

    gpu_pixels = (
        device_pixels
        if device_pixels is not None
        else cp.asarray(np.ascontiguousarray(pixels))
    )
    gpu_lut = device_lut if device_lut is not None else cp.asarray(table)

    failpos = cp.empty((block_count, cell_count), dtype=cp.int32)
    grid = (
        (cell_count + 31) // 32,
        (block_count + 7) // 8,
    )
    module.get_function("k_producer_failpos")(
        grid,
        (32, 8),
        (
            gpu_pixels,
            np.int32(height),
            np.int32(width),
            np.int32(active_width),
            threshold,
            np.int32(block_count),
            np.int32(cell_count),
            failpos,
        ),
    )

    row_visible = cp.empty(height, dtype=cp.float64)
    row_infrared = cp.empty(height, dtype=cp.float64)
    row_weight = cp.empty(height, dtype=cp.float64)
    row_accepted = cp.empty(height, dtype=cp.uint32)
    module.get_function("k_producer_row_sums")(
        ((height + 127) // 128,),
        (128,),
        (
            gpu_pixels,
            gpu_lut,
            np.int32(height),
            np.int32(width),
            np.int32(active_width),
            np.int32(R4000_VISIBLE_REFERENCE_CHANNEL),
            np.int32(cell_count),
            failpos,
            row_visible,
            row_infrared,
            row_weight,
            row_accepted,
        ),
    )

    add_denominator = cp.zeros(max(epoch_count, 1), dtype=cp.float32)
    add_numerator = cp.zeros(max(epoch_count, 1), dtype=cp.float32)
    if epoch_count:
        module.get_function("k_producer_scale_epochs")(
            ((epoch_count + 63) // 64,),
            (64,),
            (
                gpu_pixels,
                gpu_lut,
                np.int32(height),
                np.int32(width),
                np.int32(active_width),
                np.int32(R4000_VISIBLE_REFERENCE_CHANNEL),
                threshold,
                np.int32(epoch_count),
                add_denominator,
                add_numerator,
            ),
        )

    host_row_visible = cp.asnumpy(row_visible)
    host_row_infrared = cp.asnumpy(row_infrared)
    host_row_weight = cp.asnumpy(row_weight)
    host_row_accepted = cp.asnumpy(row_accepted)
    host_add_denominator = cp.asnumpy(add_denominator)
    host_add_numerator = cp.asnumpy(add_numerator)

    # Host finalization: the reference rounds after every cross-row add.
    cumulative_visible = np.empty(height, dtype=np.float64)
    cumulative_infrared = np.empty(height, dtype=np.float64)
    cumulative_weight = np.empty(height, dtype=np.float64)
    visible_means = np.empty(height, dtype=np.float32)
    infrared_means = np.empty(height, dtype=np.float32)

    total_visible = 0.0
    total_infrared = 0.0
    total_weight = 0.0
    visible_mean = np.float32(R4000_INITIAL_VISIBLE_MEAN)
    infrared_mean = np.float32(R4000_INITIAL_INFRARED_MEAN)
    for row in range(height):
        total_visible += float(host_row_visible[row])
        total_infrared += float(host_row_infrared[row])
        total_weight += float(host_row_weight[row])
        if total_weight != 0.0:
            visible_mean = np.float32(total_visible / total_weight)
            infrared_mean = np.float32(total_infrared / total_weight)
        cumulative_visible[row] = total_visible
        cumulative_infrared[row] = total_infrared
        cumulative_weight[row] = total_weight
        visible_means[row] = visible_mean
        infrared_means[row] = infrared_mean

    mean_schedule = ProducerMeanSchedule(
        weighted_visible_sum=cumulative_visible,
        weighted_infrared_sum=cumulative_infrared,
        weight_sum=cumulative_weight,
        calibration_mean_visible=visible_means,
        calibration_mean_infrared=infrared_means,
        accepted_samples=host_row_accepted.astype(np.uint32),
        active_width=active_width,
        infrared_threshold=float(threshold),
    )

    multiplier = np.float32(R4000_SCALE_MULTIPLIER)
    current_p1 = np.float32(R4000_INITIAL_SCALED_INPUT)
    current_p2 = np.float32(R4000_INITIAL_INPUT_SCALE)
    denominator = np.float32(0.0)
    numerator = np.float32(0.0)
    p1_values = np.empty(height, dtype=np.float32)
    p2_values = np.empty(height, dtype=np.float32)
    denominators = np.empty(height, dtype=np.float32)
    numerators = np.empty(height, dtype=np.float32)
    add_denominators = np.zeros(height, dtype=np.float32)
    add_numerators = np.zeros(height, dtype=np.float32)

    for row in range(height):
        if row % PRODUCER_BLOCK_SIZE == PRODUCER_BLOCK_SIZE - 1:
            epoch = row // PRODUCER_BLOCK_SIZE
            add_den = np.float32(host_add_denominator[epoch])
            add_num = np.float32(host_add_numerator[epoch])
            add_denominators[row] = add_den
            add_numerators[row] = add_num
            denominator_wide = float(denominator) + float(add_den)
            numerator_wide = float(numerator) + float(add_num)
            denominator = np.float32(denominator_wide)
            numerator = np.float32(numerator_wide)
            if denominator_wide != 0.0:
                ratio = numerator_wide / denominator_wide
                current_p2 = np.float32(ratio)
                current_p1 = np.float32(float(multiplier) * ratio)
        p1_values[row] = current_p1
        p2_values[row] = current_p2
        denominators[row] = denominator
        numerators[row] = numerator

    payloads: list[bytes] = []
    for row in range(height):
        payload = bytearray(0x40)
        struct.pack_into("<f", payload, 0x04, p2_values[row])
        struct.pack_into("<f", payload, 0x08, p1_values[row])
        struct.pack_into(
            "<ddd",
            payload,
            0x10,
            cumulative_visible[row],
            cumulative_infrared[row],
            cumulative_weight[row],
        )
        struct.pack_into("<f", payload, 0x28, visible_means[row])
        struct.pack_into("<f", payload, 0x2C, infrared_means[row])
        struct.pack_into("<f", payload, 0x30, denominators[row])
        struct.pack_into("<f", payload, 0x34, numerators[row])
        payloads.append(bytes(payload))

    canonical_pixels = np.ascontiguousarray(pixels, dtype="<u2")
    canonical_table = np.ascontiguousarray(table, dtype="<f4")
    return ProducerRecordSchedule(
        mean_schedule=mean_schedule,
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
