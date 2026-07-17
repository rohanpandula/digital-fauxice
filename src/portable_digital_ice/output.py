"""Exact active X3A float-work-plane to public RGB16 conversion.

This is a clean-room implementation of the selector-8, 16-bit output path.
It contains no Nikon code or tables: the two factor tables are regenerated
from their recovered equations and hash-checked before use.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .x3a import UnboundBehaviorError


H_FACTOR_SHA256 = "fdc378493f13916e36d3dbd9b69b714e888d7993680509df4550486db6fcf328"
L_FACTOR_SHA256 = "b7f3b1b5c0ea816ddb20196e1cf7614d364ea4e164295e0dcb78119c88ef404d"


@dataclass(frozen=True)
class InverseResponseFactors:
    """The recovered high- and low-byte inverse-response factors."""

    high: npt.NDArray[np.uint32]
    low: npt.NDArray[np.uint32]

    def __post_init__(self) -> None:
        for label, values in (("high", self.high), ("low", self.low)):
            array = np.asarray(values)
            if array.dtype != np.dtype(np.uint32) or array.shape != (256,):
                raise ValueError(f"{label} factors must be uint32[256]")
            owned = np.array(array, dtype=np.uint32, order="C", copy=True)
            owned.flags.writeable = False
            object.__setattr__(self, label, owned)

    @classmethod
    def recovered_16bit(cls) -> "InverseResponseFactors":
        """Regenerate and pin Nikon's active 16-bit factorized inverse."""

        domain = np.arange(256, dtype=np.float64)
        scale = 16.0 * np.log10(np.float64(2.0)) / 65535.0
        high = np.floor(16.0 * np.power(10.0, 256.0 * domain * scale) + 0.5).astype(
            np.uint32
        )
        low = np.floor(65536.0 * np.power(10.0, domain * scale) + 0.5).astype(np.uint32)
        hashes = (
            hashlib.sha256(high.astype("<u4", copy=False).tobytes()).hexdigest(),
            hashlib.sha256(low.astype("<u4", copy=False).tobytes()).hexdigest(),
        )
        if hashes != (H_FACTOR_SHA256, L_FACTOR_SHA256):
            raise UnboundBehaviorError(
                "platform power function did not reproduce the recovered "
                "inverse-response factor tables"
            )
        return cls(high=high, low=low)


def work_value_indices(work_rgb: npt.ArrayLike) -> npt.NDArray[np.uint16]:
    """Apply Nikon's finite-domain half-up index conversion and clamp."""

    work = np.asarray(work_rgb)
    if work.dtype != np.dtype(np.float32):
        raise TypeError("work RGB must have dtype float32")
    if work.ndim != 3 or work.shape[2] != 3:
        raise ValueError("work RGB must have shape HxWx3")
    if not np.all(np.isfinite(work)):
        raise ValueError("work RGB must be finite")
    widened = work.astype(np.float64) + 0.5
    clamped = np.clip(widened, 0.0, 65535.0)
    return np.ascontiguousarray(np.trunc(clamped).astype(np.uint16))


def emit_public_rgb16(
    work_rgb: npt.ArrayLike,
    *,
    factors: InverseResponseFactors | None = None,
) -> npt.NDArray[np.uint16]:
    """Convert float32 work RGB to Nikon's public uint16 sample values.

    The returned array is HxWx3 in RGB order.  Use
    :func:`public_rgb16_bytes` when the exact little-endian public byte layout
    is needed.
    """

    active = factors or InverseResponseFactors.recovered_16bit()
    indices = work_value_indices(work_rgb)
    wide_indices = indices.astype(np.uint32)
    high = active.high[wide_indices >> 8].astype(np.uint64)
    low = active.low[wide_indices & 0xFF].astype(np.uint64)
    values = ((high * low) >> np.uint64(20)) - np.uint64(1)
    return np.ascontiguousarray(values.astype(np.uint16))


def public_rgb16_bytes(
    work_rgb: npt.ArrayLike,
    *,
    factors: InverseResponseFactors | None = None,
) -> bytes:
    """Return little-endian, pixel-interleaved ``RGBRGB...`` public bytes."""

    values = emit_public_rgb16(work_rgb, factors=factors)
    return values.astype("<u2", copy=False).tobytes(order="C")
