"""Typed acquisition and transport contracts for portable Digital ICE."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import numpy as np
import numpy.typing as npt


UInt16Array = npt.NDArray[np.uint16]


class AcquisitionEpoch(str, Enum):
    """The two physically separate RGBI acquisitions consumed by X3A."""

    PREPASS = "prepass"
    MAIN = "main"


def _frozen_uint16_image(
    pixels: npt.ArrayLike,
    *,
    lanes: int,
    label: str,
) -> UInt16Array:
    array = np.asarray(pixels)
    if array.dtype != np.dtype(np.uint16):
        raise TypeError(f"{label} must have dtype uint16")
    if array.ndim != 3 or array.shape[2] != lanes:
        raise ValueError(f"{label} must have shape HxWx{lanes}")
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"{label} cannot have an empty dimension")
    owned = np.array(array, dtype=np.uint16, order="C", copy=True)
    owned.flags.writeable = False
    return owned


@dataclass(frozen=True)
class RGBI16Frame:
    """One immutable interleaved 16-bit RGBI acquisition."""

    pixels: UInt16Array
    epoch: AcquisitionEpoch
    resolution_dpi: int
    evidence_id: str

    def __post_init__(self) -> None:
        if self.resolution_dpi <= 0:
            raise ValueError("resolution_dpi must be positive")
        if not self.evidence_id.strip():
            raise ValueError("evidence_id must be non-empty")
        object.__setattr__(
            self,
            "pixels",
            _frozen_uint16_image(self.pixels, lanes=4, label="RGBI frame"),
        )

    @property
    def height(self) -> int:
        return int(self.pixels.shape[0])

    @property
    def width(self) -> int:
        return int(self.pixels.shape[1])

    @property
    def sha256(self) -> str:
        little_endian = self.pixels.astype("<u2", copy=False)
        return hashlib.sha256(little_endian.tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class RGB16Image:
    """One immutable interleaved 16-bit RGB output image."""

    pixels: UInt16Array
    evidence_id: str

    def __post_init__(self) -> None:
        if not self.evidence_id.strip():
            raise ValueError("evidence_id must be non-empty")
        object.__setattr__(
            self,
            "pixels",
            _frozen_uint16_image(self.pixels, lanes=3, label="RGB image"),
        )

    @property
    def height(self) -> int:
        return int(self.pixels.shape[0])

    @property
    def width(self) -> int:
        return int(self.pixels.shape[1])

    @property
    def sha256(self) -> str:
        little_endian = self.pixels.astype("<u2", copy=False)
        return hashlib.sha256(little_endian.tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class RGBI16Block:
    """A valid-row-only transport block with an absolute row origin."""

    pixels: UInt16Array
    start_row: int

    def __post_init__(self) -> None:
        if self.start_row < 0:
            raise ValueError("start_row cannot be negative")
        object.__setattr__(
            self,
            "pixels",
            _frozen_uint16_image(self.pixels, lanes=4, label="RGBI block"),
        )

    @property
    def valid_rows(self) -> int:
        return int(self.pixels.shape[0])

    @property
    def width(self) -> int:
        return int(self.pixels.shape[1])


@dataclass(frozen=True)
class BlockSchedule:
    """Allocated block geometry plus the exact valid-row boundary."""

    width: int
    height: int
    rows_per_block: int
    block_count: int
    final_valid_rows: int

    def __post_init__(self) -> None:
        if min(self.width, self.height, self.rows_per_block, self.block_count) <= 0:
            raise ValueError("schedule dimensions must be positive")
        if not 1 <= self.final_valid_rows <= self.rows_per_block:
            raise ValueError("final_valid_rows is outside the allocated block")
        covered = (self.block_count - 1) * self.rows_per_block + self.final_valid_rows
        if covered != self.height:
            raise ValueError("block schedule does not cover exactly height rows")

    def valid_rows_for(self, block_index: int) -> int:
        if not 0 <= block_index < self.block_count:
            raise IndexError(block_index)
        return (
            self.final_valid_rows
            if block_index == self.block_count - 1
            else self.rows_per_block
        )


@dataclass(frozen=True)
class DualRGBIAcquisition:
    """The required same-frame prepass/main input pair."""

    prepass: RGBI16Frame
    main: RGBI16Frame
    same_frame_id: str

    def __post_init__(self) -> None:
        if self.prepass.epoch is not AcquisitionEpoch.PREPASS:
            raise ValueError("prepass frame has the wrong acquisition epoch")
        if self.main.epoch is not AcquisitionEpoch.MAIN:
            raise ValueError("main frame has the wrong acquisition epoch")
        if not self.same_frame_id.strip():
            raise ValueError("same_frame_id must be non-empty")


@dataclass(frozen=True)
class PrepassFrameRecord:
    """Opaque 0x40-byte state carrier proven sufficient for the main pass.

    Nikon stores one record per frame at ``table + (frame_index << 6)``.  Full
    record swaps made baseline/IR/red perturbation outputs swap byte-exact in
    both directions, so the portable boundary deliberately preserves all 64
    bytes.  Offset ``+0x04`` is the observed main auxiliary alpha, but callers
    must not treat that one interpreted field as the whole prepass effect.
    """

    payload: bytes
    frame_index: int
    evidence_id: str

    def __post_init__(self) -> None:
        payload = bytes(self.payload)
        if len(payload) != 0x40:
            raise ValueError("prepass frame record must contain exactly 0x40 bytes")
        if self.frame_index < 0:
            raise ValueError("prepass frame index cannot be negative")
        if not self.evidence_id.strip():
            raise ValueError("prepass frame record evidence_id must be non-empty")
        object.__setattr__(self, "payload", payload)

    @property
    def table_byte_offset(self) -> int:
        return self.frame_index << 6

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.payload).hexdigest()

    @property
    def observed_main_alpha(self) -> np.float32:
        return np.float32(struct.unpack_from("<f", self.payload, 0x04)[0])

    @property
    def observed_calibration_mean_r(self) -> np.float32:
        return np.float32(struct.unpack_from("<f", self.payload, 0x28)[0])

    @property
    def observed_calibration_mean_ir(self) -> np.float32:
        return np.float32(struct.unpack_from("<f", self.payload, 0x2C)[0])


def split_rgbi16_frame(
    frame: RGBI16Frame,
    rows_per_block: int,
) -> tuple[RGBI16Block, ...]:
    """Split valid rows without adding transport padding."""

    if rows_per_block <= 0:
        raise ValueError("rows_per_block must be positive")
    return tuple(
        RGBI16Block(frame.pixels[start : start + rows_per_block], start)
        for start in range(0, frame.height, rows_per_block)
    )


def assemble_rgbi16_blocks(
    blocks: Iterable[RGBI16Block],
    *,
    epoch: AcquisitionEpoch,
    resolution_dpi: int,
    evidence_id: str,
) -> RGBI16Frame:
    """Validate a gap-free row stream and reconstruct its logical image."""

    materialized = tuple(blocks)
    if not materialized:
        raise ValueError("at least one RGBI block is required")
    expected_row = 0
    width = materialized[0].width
    for block in materialized:
        if block.start_row != expected_row:
            raise ValueError(
                "RGBI blocks must be ordered, gap-free, and non-overlapping"
            )
        if block.width != width:
            raise ValueError("all RGBI blocks must have the same width")
        expected_row += block.valid_rows
    pixels = np.concatenate([block.pixels for block in materialized], axis=0)
    return RGBI16Frame(pixels, epoch, resolution_dpi, evidence_id)
