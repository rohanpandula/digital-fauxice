"""Portable selector-8 prepass reducer for Nikon's 0x40-byte frame record."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import numpy as np

from .contracts import AcquisitionEpoch, PrepassFrameRecord, RGBI16Frame
from .x3a import SharedLookupInputResponse


@dataclass(frozen=True)
class PrepassParameters:
    """Runtime values required by the recovered selector-8 prepass path."""

    valid_ir_threshold: float
    alpha_scale: float
    ratio_minimum: float
    ratio_maximum: float
    zero_denominator_alpha: float
    completion_marker: int

    def __post_init__(self) -> None:
        for label in (
            "valid_ir_threshold",
            "alpha_scale",
            "ratio_minimum",
            "ratio_maximum",
            "zero_denominator_alpha",
        ):
            value = float(getattr(self, label))
            if not math.isfinite(value):
                raise ValueError(f"{label} must be finite")
        if self.ratio_minimum > self.ratio_maximum:
            raise ValueError("prepass ratio minimum exceeds maximum")
        if self.completion_marker < 0 or self.completion_marker > 0xFFFFFFFF:
            raise ValueError("prepass completion marker must fit uint32")

    @classmethod
    def captured_ls5000_selector8(cls) -> "PrepassParameters":
        """Return the hash-bound normal-color LS-5000 runtime values."""

        return cls(
            valid_ir_threshold=float(np.float32(8847.2255859375)),
            alpha_scale=float(np.float32(1.5)),
            ratio_minimum=float(np.float32(-0.2)),
            ratio_maximum=float(np.float32(0.2)),
            zero_denominator_alpha=float(np.float32(0.17)),
            completion_marker=16,
        )


@dataclass(frozen=True)
class PrepassReduction:
    """Portable prepass result plus scheduler evidence."""

    record: PrepassFrameRecord
    logical_shape: tuple[int, int]
    padded_shape: tuple[int, int]
    valid_pixels: int
    fully_valid_tiles: int


def _quadrant_means(values: np.ndarray) -> np.ndarray:
    """Reproduce 0x10003d10's four float32 4x4 means."""

    if values.dtype != np.dtype(np.float32) or values.shape != (8, 8):
        raise ValueError("prepass quadrant source must be float32 8x8")
    result = np.empty(4, dtype=np.float32)
    output_index = 0
    coefficient = float(0.0625)
    for first_y in (0, 4):
        for first_x in (0, 4):
            total = 0.0
            for y in range(first_y, first_y + 4):
                for x in range(first_x, first_x + 4):
                    total += float(values[y, x])
            result[output_index] = np.float32(total * coefficient)
            output_index += 1
    return result


def _global_mean(values: np.ndarray) -> np.float32:
    """Reproduce 0x10003e80's widened sum and final float32 store."""

    total = float(values[0])
    total += float(values[1])
    total += float(values[2])
    total += float(values[3])
    return np.float32(total * 0.25)


def reduce_prepass_frame(
    frame: RGBI16Frame,
    *,
    parameters: PrepassParameters,
    frame_index: int,
    evidence_id: str,
    response: SharedLookupInputResponse | None = None,
    prior_record: PrepassFrameRecord | None = None,
) -> PrepassReduction:
    """Reduce one logical prepass image to Nikon's complete frame record.

    The scheduler zero-pads to a multiple of eight and visits 8x8 tiles in
    row-major order.  Calibration sums include every individually valid pixel;
    the local regression is applied only when all 64 pixels in a tile pass the
    raw-IR threshold.
    """

    if frame.epoch is not AcquisitionEpoch.PREPASS:
        raise ValueError("prepass reducer requires a prepass RGBI frame")
    if prior_record is not None and prior_record.frame_index != frame_index:
        raise ValueError("prior prepass record belongs to a different frame index")
    prior_payload = (
        bytes(0x40) if prior_record is None else bytes(prior_record.payload)
    )

    pixels = frame.pixels
    height, width = pixels.shape[:2]
    padded_height = (height + 7) & ~7
    padded_width = (width + 7) & ~7
    padded = np.zeros((padded_height, padded_width, 4), dtype=np.uint16)
    padded[:height, :width] = pixels
    lookup = (
        SharedLookupInputResponse.nikon_logarithmic() if response is None else response
    ).table

    threshold = np.float32(parameters.valid_ir_threshold)
    ratio_minimum = float(np.float32(parameters.ratio_minimum))
    ratio_maximum = float(np.float32(parameters.ratio_maximum))
    alpha_scale = float(np.float32(parameters.alpha_scale))
    calibration_r = struct.unpack_from("<d", prior_payload, 0x10)[0]
    calibration_ir = struct.unpack_from("<d", prior_payload, 0x18)[0]
    calibration_square = struct.unpack_from("<d", prior_payload, 0x20)[0]
    regression_denominator = np.float32(
        struct.unpack_from("<f", prior_payload, 0x30)[0]
    )
    regression_numerator = np.float32(
        struct.unpack_from("<f", prior_payload, 0x34)[0]
    )
    valid_pixels = 0
    fully_valid_tiles = 0

    for first_y in range(0, padded_height, 8):
        for first_x in range(0, padded_width, 8):
            raw = padded[first_y : first_y + 8, first_x : first_x + 8]
            response_r = np.empty((8, 8), dtype=np.float32)
            response_ir = np.empty((8, 8), dtype=np.float32)
            sum_raw_ir = np.float32(0.0)
            tile_valid = True
            for y in range(8):
                for x in range(8):
                    raw_ir = int(raw[y, x, 3])
                    if raw_ir <= threshold:
                        tile_valid = False
                        continue
                    valid_pixels += 1
                    converted_r = lookup[int(raw[y, x, 0])]
                    converted_ir = lookup[raw_ir]
                    response_r[y, x] = converted_r
                    response_ir[y, x] = converted_ir
                    raw_ir_square = float(raw_ir * raw_ir)
                    calibration_r += raw_ir_square * float(converted_r)
                    calibration_ir += raw_ir_square * float(converted_ir)
                    calibration_square += raw_ir_square
                    sum_raw_ir = np.float32(float(sum_raw_ir) + float(raw_ir))
            if not tile_valid:
                continue
            fully_valid_tiles += 1
            r_means = _quadrant_means(response_r)
            ir_means = _quadrant_means(response_ir)
            r_global = _global_mean(r_means)
            ir_global = _global_mean(ir_means)
            r_deviations = np.asarray(
                [np.float32(float(value) - float(r_global)) for value in r_means],
                dtype=np.float32,
            )
            ir_deviations = np.asarray(
                [np.float32(float(value) - float(ir_global)) for value in ir_means],
                dtype=np.float32,
            )
            denominator = float(regression_denominator)
            numerator = float(regression_numerator)
            scalar = float(sum_raw_ir)
            for r_deviation, ir_deviation in zip(
                r_deviations, ir_deviations, strict=True
            ):
                if r_deviation == np.float32(0.0):
                    continue
                ratio = float(ir_deviation) / float(r_deviation)
                if ratio < ratio_minimum or ratio > ratio_maximum:
                    continue
                weight = float(r_deviation) * float(r_deviation) * scalar * scalar
                denominator += weight
                numerator += ratio * weight
            regression_denominator = np.float32(denominator)
            regression_numerator = np.float32(numerator)

    if regression_denominator == np.float32(0.0):
        unrounded_alpha = float(np.float32(parameters.zero_denominator_alpha))
    else:
        unrounded_alpha = float(regression_numerator) / float(regression_denominator)
    alpha = np.float32(unrounded_alpha)
    scaled_alpha = np.float32(unrounded_alpha * alpha_scale)

    payload = bytearray(prior_payload)
    struct.pack_into("<I", payload, 0x00, 1)
    struct.pack_into("<f", payload, 0x04, alpha)
    struct.pack_into("<f", payload, 0x08, scaled_alpha)
    struct.pack_into("<d", payload, 0x10, calibration_r)
    struct.pack_into("<d", payload, 0x18, calibration_ir)
    struct.pack_into("<d", payload, 0x20, calibration_square)
    # 0x1000325b..0x10003281 skips both mean stores when F+0x20 is
    # zero.  Nikon therefore retains the prior F+0x28/+0x2c bytes; a
    # freshly allocated frame record supplies the all-zero default above.
    if calibration_square != 0.0:
        struct.pack_into(
            "<f", payload, 0x28, np.float32(calibration_r / calibration_square)
        )
        struct.pack_into(
            "<f", payload, 0x2C, np.float32(calibration_ir / calibration_square)
        )
    struct.pack_into("<f", payload, 0x30, regression_denominator)
    struct.pack_into("<f", payload, 0x34, regression_numerator)
    struct.pack_into("<I", payload, 0x38, parameters.completion_marker)
    record = PrepassFrameRecord(bytes(payload), frame_index, evidence_id)
    return PrepassReduction(
        record=record,
        logical_shape=(height, width),
        padded_shape=(padded_height, padded_width),
        valid_pixels=valid_pixels,
        fully_valid_tiles=fully_valid_tiles,
    )
