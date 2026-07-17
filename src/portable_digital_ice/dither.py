"""Recovered conditional-dither algebra with an explicit RNG owner."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .rng import LCG24


LOW_FRACTION = np.float32(0.01)
HIGH_FRACTION = np.float32(0.99)


def percentile_bound_indices(maximum_index: int) -> tuple[int, int]:
    """Return Nikon's truncated 1%/99% indices, each clamped to ``N``.

    ``maximum_index`` is Nikon's runtime ``N`` and the lookup must therefore
    contain at least ``N + 1`` entries.  The x87 conversion was in truncate
    mode; this is deliberately not nearest rounding.
    """

    if not 0 <= maximum_index <= 0xFFFF:
        raise ValueError("maximum_index must fit the recovered unsigned-16 field")
    n = np.float32(maximum_index)
    low = min(math.trunc(float(np.multiply(n, LOW_FRACTION))), maximum_index)
    high = min(math.trunc(float(np.multiply(n, HIGH_FRACTION))), maximum_index)
    return low, high


@dataclass(frozen=True)
class DitherBounds:
    low: np.float32
    high: np.float32
    low_index: int
    high_index: int

    @classmethod
    def from_lookup(
        cls,
        lookup: npt.ArrayLike,
        *,
        maximum_index: int,
    ) -> "DitherBounds":
        table = np.asarray(lookup)
        if table.dtype != np.dtype(np.float32) or table.ndim != 1:
            raise TypeError("dither lookup must be a float32 vector")
        if len(table) <= maximum_index or not np.all(np.isfinite(table)):
            raise ValueError("dither lookup does not contain every index through N")
        low_index, high_index = percentile_bound_indices(maximum_index)
        low = np.float32(table[low_index])
        high = np.float32(table[high_index])
        return cls(low, high, low_index, high_index)


def conditional_dither_delta(
    value: float,
    *,
    bounds: DitherBounds,
    scale: float,
    generator: LCG24,
) -> float:
    """Return Nikon's accepted binary64-valued dither delta, or zero.

    The generator is called exactly once only when ``low < value < high``.
    A generated delta that would leave the same strict interval is rejected,
    but its RNG call remains consumed.  Nikon narrows the argument and random
    amplitude to float32, while the taper, random product, returned delta, and
    bounds retry are evaluated on the widened x87 path.
    """

    candidate32 = np.float32(value)
    scale32 = np.float32(scale)
    low32 = np.float32(bounds.low)
    high32 = np.float32(bounds.high)
    if not all(
        math.isfinite(float(item)) for item in (candidate32, scale32, low32, high32)
    ):
        raise ValueError("dither inputs must be finite")
    candidate = float(candidate32)
    low = float(low32)
    high = float(high32)
    if not low < high:
        return 0.0
    if not (low < candidate < high):
        return 0.0

    width = high - low
    # Keep Nikon's two explicit binary64 temporaries and multiplication order.
    # Reassociating this as ``4 * (z-lo) * (hi-z) / width**2`` changes low
    # bits for real trace values even though the expressions are algebraically
    # equivalent.
    coefficient = 4.0 / (width * width)
    envelope = ((high - candidate) * (candidate - low)) * coefficient
    random_span = np.float32(float(scale32) * candidate)
    random_value = generator.sample(0.0, float(random_span))
    delta = envelope * random_value
    changed = candidate + delta
    if low < changed < high:
        return delta
    return 0.0


def apply_conditional_dither(
    value: float,
    *,
    bounds: DitherBounds,
    scale: float,
    generator: LCG24,
) -> np.float32:
    candidate = np.float32(value)
    delta = conditional_dither_delta(
        float(candidate),
        bounds=bounds,
        scale=scale,
        generator=generator,
    )
    return np.float32(float(candidate) + delta)
