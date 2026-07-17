"""Recovered reconstruction and writer primitives used by portable streaming."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import cast

import numpy as np
import numpy.typing as npt

from .dither import DitherBounds, apply_conditional_dither, conditional_dither_delta
from .rng import LCG24


def _centered_row_mask(widths: tuple[int, ...], size: int) -> np.ndarray:
    if len(widths) != size or size % 2 != 1:
        raise ValueError("centered mask dimensions must be odd and match row widths")
    mask = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    for row, width in enumerate(widths):
        if width <= 0 or width > size or width % 2 != 1:
            raise ValueError("every centered mask row width must be positive and odd")
        radius = width // 2
        mask[row, center - radius : center + radius + 1] = np.float32(1.0)
    mask.flags.writeable = False
    return mask


ROUNDED_9X9_69 = _centered_row_mask((5, 7, 9, 9, 9, 9, 9, 7, 5), 9)
ROUNDED_5X5_21 = _centered_row_mask((3, 5, 5, 5, 3), 5)
BINOMIAL_3X3_16 = np.asarray(
    [[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32
) / np.float32(16.0)
BINOMIAL_3X3_16.flags.writeable = False


def pad_reconstruction_history(
    source: npt.ArrayLike,
    *,
    guard: int = 4,
    pseudo_top_row: npt.ArrayLike | None = None,
) -> npt.NDArray[np.float32]:
    """Build the active X3A score/weighted-history boundary carrier.

    The captured scheduler is asymmetric. Its horizontal guards remain zero.
    Three rows above the first input remain zero. The immediately preceding
    pseudo-row is caller-supplied: it is the score floor for score history and
    ``float32(first_real * floor)`` for weighted auxiliary/RGB history. During
    final flushing, the last row is repeated bit-for-bit.

    This helper applies to the score, score-weighted auxiliary, and
    score-weighted RGB histories consumed by the W/A and candidate producers.
    Raw decision auxiliary uses an all-edge-replicated carrier instead.
    """

    values = np.asarray(source)
    if values.dtype != np.dtype(np.float32) or values.ndim not in (2, 3):
        raise ValueError("reconstruction history must be float32 HxW or HxWxL")
    if not np.all(np.isfinite(values)):
        raise ValueError("reconstruction history must be finite")
    if guard < 0:
        raise ValueError("history guard cannot be negative")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("reconstruction history cannot be empty")
    top = None if pseudo_top_row is None else np.asarray(pseudo_top_row)
    if top is not None and (
        top.dtype != np.dtype(np.float32) or top.shape != values.shape[1:]
    ):
        raise ValueError("pseudo top row must be float32 and match one source row")
    if top is not None and not np.all(np.isfinite(top)):
        raise ValueError("pseudo top row must be finite")
    height, width = values.shape[:2]
    padded_shape = (height + 2 * guard, width + 2 * guard, *values.shape[2:])
    padded = np.zeros(padded_shape, dtype=np.float32)
    padded[guard : guard + height, guard : guard + width] = values
    if guard:
        if top is not None:
            padded[guard - 1, guard : guard + width] = top
        padded[guard + height :, guard : guard + width] = values[-1]
    return np.ascontiguousarray(padded)


class NonpositiveNormalizerPolicy(str, Enum):
    """Explicit behavior for a nonpositive score/support denominator."""

    UNNORMALIZED_SUM = "unnormalized_sum"
    INVALID = "invalid"


@dataclass(frozen=True)
class Candidate:
    rgb: npt.NDArray[np.float32]
    normalizer: np.float32
    valid: bool

    def __post_init__(self) -> None:
        value = np.asarray(self.rgb)
        if value.dtype != np.dtype(np.float32) or value.shape != (3,):
            raise ValueError("candidate RGB must be a float32 triplet")


def score_normalized_candidate(
    weighted_rgb_patch: npt.ArrayLike,
    score_patch: npt.ArrayLike,
    kernel: npt.ArrayLike,
    *,
    nonpositive_policy: NonpositiveNormalizerPolicy,
) -> Candidate:
    """Compute one multiscale score-normalized RGB fill.

    Accumulation is deliberately scalar row-major float32 so results are
    deterministic across NumPy/BLAS builds.  A future byte-parity binding can
    replace this order if the x87 grouping proves observably different.
    """

    weighted = np.asarray(weighted_rgb_patch)
    score = np.asarray(score_patch)
    weights = np.asarray(kernel)
    if weighted.dtype != np.dtype(np.float32) or weighted.ndim != 3:
        raise TypeError("weighted RGB patch must be float32 HxWx3")
    if weighted.shape[2] != 3:
        raise ValueError("weighted RGB patch must have three lanes")
    if score.dtype != np.dtype(np.float32) or score.shape != weighted.shape[:2]:
        raise ValueError("score patch must be float32 and match RGB geometry")
    if weights.dtype != np.dtype(np.float32) or weights.shape != score.shape:
        raise ValueError("kernel must be float32 and match patch geometry")
    if not (
        np.all(np.isfinite(weighted))
        and np.all(np.isfinite(score))
        and np.all(np.isfinite(weights))
    ):
        raise ValueError("candidate inputs must be finite")
    numerator = np.zeros(3, dtype=np.float32)
    denominator = np.float32(0.0)
    for y in range(score.shape[0]):
        for x in range(score.shape[1]):
            weight = np.float32(weights[y, x])
            denominator = np.float32(denominator + np.float32(score[y, x] * weight))
            for channel in range(3):
                numerator[channel] = np.float32(
                    numerator[channel] + np.float32(weighted[y, x, channel] * weight)
                )
    if denominator > np.float32(0.0):
        rgb = np.asarray(
            [np.float32(value / denominator) for value in numerator],
            dtype=np.float32,
        )
        return Candidate(rgb, denominator, bool(np.all(rgb > 0.0)))
    if nonpositive_policy is NonpositiveNormalizerPolicy.UNNORMALIZED_SUM:
        return Candidate(numerator, denominator, bool(np.all(numerator > 0.0)))
    return Candidate(np.zeros(3, dtype=np.float32), denominator, False)


def recovered_multiscale_candidates(
    weighted_rgb_9x9: npt.ArrayLike,
    score_9x9: npt.ArrayLike,
    *,
    nonpositive_policy: NonpositiveNormalizerPolicy,
) -> tuple[Candidate, Candidate, Candidate]:
    """Return Nikon's C69, C21, and C16 candidates.

    This convenience entry point derives a score-normalizer record from the
    supplied patch.  The production pipeline should pass Nikon's explicit W
    record to :func:`recovered_rgb_candidates`; the W/A producer is a separate
    stage boundary and its trace binding remains open.
    """

    weighted = np.asarray(weighted_rgb_9x9)
    score = np.asarray(score_9x9)
    if (
        weighted.dtype != np.dtype(np.float32)
        or score.dtype != np.dtype(np.float32)
        or weighted.shape != (9, 9, 3)
        or score.shape != (9, 9)
    ):
        raise ValueError("multiscale reconstruction requires one centered 9x9 patch")
    normalizers = recovered_spatial_averages(score)
    return recovered_rgb_candidates(
        weighted,
        normalizers,
        nonpositive_policy=nonpositive_policy,
    )


def _vertical_scratch(source: np.ndarray) -> tuple[np.ndarray, ...]:
    """Build stored S3/S5/S7/S9 values in Nikon's recovered add order."""

    if source.dtype != np.dtype(np.float32) or source.shape[:2] != (9, 9):
        raise ValueError("vertical scratch source must be float32 with shape 9x9xL")
    lanes = 1 if source.ndim == 2 else source.shape[2]
    if source.ndim not in (2, 3):
        raise ValueError("vertical scratch source must have one optional lane axis")
    values = source[:, :, np.newaxis] if source.ndim == 2 else source
    scratch = [np.empty((9, lanes), dtype=np.float32) for _ in range(4)]
    for x in range(9):
        for lane in range(lanes):
            accumulator = float(values[3, x, lane]) + float(values[4, x, lane])
            accumulator += float(values[5, x, lane])
            scratch[0][x, lane] = np.float32(accumulator)
            accumulator += float(values[2, x, lane])
            accumulator += float(values[6, x, lane])
            scratch[1][x, lane] = np.float32(accumulator)
            accumulator += float(values[1, x, lane])
            accumulator += float(values[7, x, lane])
            scratch[2][x, lane] = np.float32(accumulator)
            accumulator += float(values[0, x, lane])
            accumulator += float(values[8, x, lane])
            scratch[3][x, lane] = np.float32(accumulator)
    return tuple(scratch)


def _vertical_binomial_16(source: np.ndarray) -> np.ndarray:
    """Build Nikon's stored vertical half of the 3x3 binomial filter.

    Nikon scales the three-row ``[1, 2, 1]`` sum by 1/16 and narrows that
    intermediate to float32 before its horizontal ``[1, 2, 1]`` pass.  The
    unusual split (rather than 1/4 at each pass) is observable in last bits.
    """

    if source.dtype != np.dtype(np.float32) or source.shape[:2] != (9, 9):
        raise ValueError("vertical binomial source must be float32 with shape 9x9xL")
    lanes = 1 if source.ndim == 2 else source.shape[2]
    if source.ndim not in (2, 3):
        raise ValueError("vertical binomial source must have one optional lane axis")
    values = source[:, :, np.newaxis] if source.ndim == 2 else source
    scratch = np.empty((9, lanes), dtype=np.float32)
    coefficient = float(np.float32(1.0 / 16.0))
    for x in range(9):
        for lane in range(lanes):
            center = float(values[4, x, lane])
            accumulator = float(values[3, x, lane]) + center
            accumulator += float(values[5, x, lane])
            accumulator += center
            scratch[x, lane] = np.float32(accumulator * coefficient)
    return scratch


def _recovered_unscaled_averages(source: np.ndarray) -> np.ndarray:
    """Return Q69/Q21/Q16 in widened precision before candidate narrowing."""

    s3, s5, s7, s9 = _vertical_scratch(source)
    binomial_16 = _vertical_binomial_16(source)
    lanes = s3.shape[1]
    result = np.empty((3, lanes), dtype=np.float64)
    coefficient_69 = float(np.float32(1.0 / 69.0))
    coefficient_21 = float(np.float32(1.0 / 21.0))
    for lane in range(lanes):
        total_21 = float(s3[6, lane]) + float(s3[2, lane])
        total_21 += float(s5[3, lane])
        total_21 += float(s5[4, lane])
        total_21 += float(s5[5, lane])

        total_69 = float(s5[8, lane]) + float(s5[0, lane])
        total_69 += float(s7[1, lane])
        total_69 += float(s7[7, lane])
        for x in range(2, 7):
            total_69 += float(s9[x, lane])

        total_16 = float(binomial_16[4, lane]) + float(binomial_16[3, lane])
        total_16 += float(binomial_16[5, lane])
        total_16 += float(binomial_16[4, lane])

        result[0, lane] = total_69 * coefficient_69
        result[1, lane] = total_21 * coefficient_21
        result[2, lane] = total_16
    return result


def _recovered_rgb_unscaled_averages(source: np.ndarray) -> np.ndarray:
    """Return Nikon's RGB-candidate Q69/Q21/Q16 widened accumulators.

    The paired W/A producer and the RGB-candidate producer share the 69- and
    21-sample accumulators, but their 3x3 binomial paths have different
    narrowing schedules.  W/A stores a vertically filtered float32
    intermediate.  The RGB path instead builds three horizontal ``[1,2,1]``
    rows in x87 precision, combines them vertically, and multiplies by the
    float32 ``1/16`` coefficient without an intervening float32 store.  That
    distinction is observable in the last bit and must not be folded into the
    already byte-exact W/A helper above.
    """

    result = _recovered_unscaled_averages(source)
    values = source[:, :, np.newaxis] if source.ndim == 2 else source
    coefficient = float(np.float32(1.0 / 16.0))
    for lane in range(result.shape[1]):
        rows: list[float] = []
        for y in (3, 4, 5):
            center = float(values[y, 4, lane])
            horizontal = float(values[y, 3, lane]) + float(values[y, 5, lane])
            horizontal += center
            horizontal += center
            rows.append(horizontal)
        total = rows[0] + rows[1]
        total += rows[1]
        total += rows[2]
        result[2, lane] = total * coefficient
    return result


def recovered_spatial_averages(source_9x9: npt.ArrayLike) -> npt.NDArray[np.float32]:
    """Compute stored Q69/Q21/Q16 floats for a scalar 9x9 patch."""

    source = np.asarray(source_9x9)
    if source.dtype != np.dtype(np.float32) or source.shape != (9, 9):
        raise ValueError("spatial-average source must be float32 9x9")
    return np.ascontiguousarray(
        _recovered_unscaled_averages(source)[:, 0].astype(np.float32)
    )


def recovered_weight_records_for_interior_row(
    score: npt.ArrayLike,
    *,
    center_row: int,
    first_x: int = 4,
    stop_x: int | None = None,
) -> npt.NDArray[np.float32]:
    """Produce Nikon's ``[W69, W21, W16, W1]`` records for one interior row.

    This closes the pixel-local producer math while keeping the surrounding
    ring-buffer scheduler explicit.  Callers must supply the source row and
    horizontal interval selected by that scheduler; no edge behavior is
    invented here.
    """

    values = np.asarray(score)
    if values.dtype != np.dtype(np.float32) or values.ndim != 2:
        raise ValueError("score source must be a float32 HxW plane")
    if not np.all(np.isfinite(values)):
        raise ValueError("score source must be finite")
    height, width = values.shape
    if center_row < 4 or center_row >= height - 4:
        raise ValueError("center row must have four source rows on either side")
    resolved_stop = width - 4 if stop_x is None else stop_x
    if first_x < 4 or resolved_stop > width - 4 or first_x > resolved_stop:
        raise ValueError("horizontal interval must remain inside the 9x9 support")
    records = np.empty((resolved_stop - first_x, 4), dtype=np.float32)
    row_patch = values[center_row - 4 : center_row + 5]
    for output_index, x in enumerate(range(first_x, resolved_stop)):
        records[output_index, :3] = recovered_spatial_averages(
            row_patch[:, x - 4 : x + 5]
        )
        records[output_index, 3] = values[center_row, x]
    return np.ascontiguousarray(records)


def recovered_weight_and_auxiliary_records_for_interior_row(
    score: npt.ArrayLike,
    auxiliary: npt.ArrayLike,
    *,
    center_row: int,
    fallback_value: float,
    first_x: int = 4,
    stop_x: int | None = None,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Produce paired ``W`` and normalized-auxiliary ``A`` records.

    The returned arrays are content-aligned by source center.  Nikon stores an
    A record at feature index ``x + 1`` while storing the matching W record at
    weight index ``x``; that ring/index shift belongs to the row scheduler and
    is intentionally not hidden in these arrays.

    The paired W/A producer keeps each W sum in the x87 register stack after
    writing its float32 W record.  The corresponding A numerator is divided by
    that *unrounded* in-register W value, not by the float32 value just stored.
    Preserving that detail closes every A lane byte-exact on the authoritative
    ``+0x10b0`` trace; dividing by ``weight_records`` instead is off by one ULP
    on a substantial minority of the samples.
    """

    weights = np.asarray(score)
    features = np.asarray(auxiliary)
    if weights.dtype != np.dtype(np.float32) or weights.ndim != 2:
        raise ValueError("score source must be a float32 HxW plane")
    if features.dtype != np.dtype(np.float32) or features.shape != weights.shape:
        raise ValueError("auxiliary source must be float32 and match score geometry")
    if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(features)):
        raise ValueError("score and auxiliary sources must be finite")
    fallback = np.float32(fallback_value)
    if not np.isfinite(fallback):
        raise ValueError("auxiliary fallback must be finite")
    height, width = weights.shape
    if center_row < 4 or center_row >= height - 4:
        raise ValueError("center row must have four source rows on either side")
    resolved_stop = width - 4 if stop_x is None else stop_x
    if first_x < 4 or resolved_stop > width - 4 or first_x > resolved_stop:
        raise ValueError("horizontal interval must remain inside the 9x9 support")
    weight_records = recovered_weight_records_for_interior_row(
        weights,
        center_row=center_row,
        first_x=first_x,
        stop_x=resolved_stop,
    )
    auxiliary_records = np.empty_like(weight_records)
    weighted_auxiliary = np.multiply(weights, features, dtype=np.float32)
    weight_row_patch = weights[center_row - 4 : center_row + 5]
    weighted_row_patch = weighted_auxiliary[center_row - 4 : center_row + 5]
    for output_index, x in enumerate(range(first_x, resolved_stop)):
        unrounded_weights = _recovered_unscaled_averages(
            weight_row_patch[:, x - 4 : x + 5]
        )[:, 0]
        numerator = _recovered_unscaled_averages(weighted_row_patch[:, x - 4 : x + 5])[
            :, 0
        ]
        for lane in range(3):
            denominator = unrounded_weights[lane]
            if denominator > 0.0:
                auxiliary_records[output_index, lane] = np.float32(
                    numerator[lane] / denominator
                )
            else:
                auxiliary_records[output_index, lane] = fallback
        if weight_records[output_index, 3] > np.float32(0.0):
            auxiliary_records[output_index, 3] = features[center_row, x]
        else:
            auxiliary_records[output_index, 3] = fallback
    return (
        np.ascontiguousarray(weight_records),
        np.ascontiguousarray(auxiliary_records),
    )


def recovered_rgb_candidates(
    weighted_rgb_9x9: npt.ArrayLike,
    normalizers_69_21_16: npt.ArrayLike,
    *,
    nonpositive_policy: NonpositiveNormalizerPolicy = (
        NonpositiveNormalizerPolicy.UNNORMALIZED_SUM
    ),
) -> tuple[Candidate, Candidate, Candidate]:
    """Build C69/C21/C16 from Nikon's explicit float32 W normalizers."""

    weighted = np.asarray(weighted_rgb_9x9)
    normalizers = np.asarray(normalizers_69_21_16)
    if weighted.dtype != np.dtype(np.float32) or weighted.shape != (9, 9, 3):
        raise ValueError("weighted RGB source must be float32 9x9x3")
    if normalizers.dtype != np.dtype(np.float32) or normalizers.shape != (3,):
        raise ValueError("candidate normalizers must be a float32 W69/W21/W16 triplet")
    if not np.all(np.isfinite(weighted)) or not np.all(np.isfinite(normalizers)):
        raise ValueError("candidate source and normalizers must be finite")
    averages = _recovered_rgb_unscaled_averages(weighted)
    candidates: list[Candidate] = []
    for index in range(3):
        denominator = np.float32(normalizers[index])
        if denominator > np.float32(0.0):
            reciprocal = 1.0 / float(denominator)
            rgb = np.asarray(
                [np.float32(value * reciprocal) for value in averages[index]],
                dtype=np.float32,
            )
        elif nonpositive_policy is NonpositiveNormalizerPolicy.UNNORMALIZED_SUM:
            rgb = averages[index].astype(np.float32)
        else:
            rgb = np.zeros(3, dtype=np.float32)
        candidates.append(
            Candidate(
                rgb, denominator, bool(np.all(np.isfinite(rgb)) and np.all(rgb > 0))
            )
        )
    return cast(tuple[Candidate, Candidate, Candidate], tuple(candidates))


def dead_zone_residual(
    value: npt.ArrayLike,
    minimum: npt.ArrayLike,
    maximum: npt.ArrayLike,
) -> npt.NDArray[np.float32]:
    """Remove an auxiliary-feature-band interval and retain only excess RGB change."""

    delta = np.asarray(value, dtype=np.float32)
    lower = np.asarray(minimum, dtype=np.float32)
    upper = np.asarray(maximum, dtype=np.float32)
    if delta.shape != lower.shape or delta.shape != upper.shape:
        raise ValueError("dead-zone value/minimum/maximum shapes must match")
    if not (
        np.all(np.isfinite(delta))
        and np.all(np.isfinite(lower))
        and np.all(np.isfinite(upper))
    ):
        raise ValueError("dead-zone inputs must be finite")
    if np.any(lower > upper):
        raise ValueError("dead-zone minimum exceeds maximum")
    residual = np.where(
        delta < lower,
        delta - lower,
        np.where(delta > upper, delta - upper, np.float32(0.0)),
    )
    return np.ascontiguousarray(residual, dtype=np.float32)


@dataclass(frozen=True)
class BandLimits:
    minimum: npt.NDArray[np.float32]
    maximum: npt.NDArray[np.float32]

    def __post_init__(self) -> None:
        minimum = np.asarray(self.minimum)
        maximum = np.asarray(self.maximum)
        if (
            minimum.dtype != np.dtype(np.float32)
            or maximum.dtype != np.dtype(np.float32)
            or minimum.shape != (3,)
            or maximum.shape != (3,)
        ):
            raise ValueError("band limits must be float32 RGB triplets")
        if np.any(minimum > maximum):
            raise ValueError("band minimum exceeds maximum")


def combine_multiscale_bands(
    candidate_a: npt.ArrayLike,
    candidate_b: npt.ArrayLike,
    candidate_c: npt.ArrayLike,
    original: npt.ArrayLike,
    *,
    limits_ab: BandLimits,
    limits_bc: BandLimits,
    limits_original_c: BandLimits,
    weights: tuple[float, float, float],
) -> npt.NDArray[np.float32]:
    """Apply A + gated(B-A) + gated(C-B) + gated(original-C).

    All three profile-supplied band weights are required; they are never
    replaced by unit weights implicitly.
    """

    vectors = [
        np.asarray(value) for value in (candidate_a, candidate_b, candidate_c, original)
    ]
    if any(
        value.dtype != np.dtype(np.float32) or value.shape != (3,) for value in vectors
    ):
        raise ValueError("multiscale candidates and original must be float32 RGB")
    if len(weights) != 3 or not all(math.isfinite(value) for value in weights):
        raise ValueError("three finite multiscale band weights are required")
    a, b, c, source = vectors
    band_ab = dead_zone_residual(b - a, limits_ab.minimum, limits_ab.maximum)
    band_bc = dead_zone_residual(c - b, limits_bc.minimum, limits_bc.maximum)
    band_original = dead_zone_residual(
        source - c,
        limits_original_c.minimum,
        limits_original_c.maximum,
    )
    result = np.array(a, dtype=np.float32, copy=True)
    for band, weight in zip((band_ab, band_bc, band_original), weights):
        result = np.add(
            result,
            np.multiply(band, np.float32(weight), dtype=np.float32),
            dtype=np.float32,
        )
    return np.ascontiguousarray(result, dtype=np.float32)


class FeatureBandExtremaMode(str, Enum):
    CENTER_ONLY = "center_only"
    CROSS_NEIGHBOR = "cross_neighbor"


def feature_band_extrema_mode(
    *,
    resolution_metric: int,
    cross_neighbor_cutoff: int,
) -> FeatureBandExtremaMode:
    if resolution_metric < 0 or cross_neighbor_cutoff < 0:
        raise ValueError("resolution metric and cutoff cannot be negative")
    return (
        FeatureBandExtremaMode.CROSS_NEIGHBOR
        if resolution_metric > cross_neighbor_cutoff
        else FeatureBandExtremaMode.CENTER_ONLY
    )


def select_feature_band_extrema(
    center_minimum: npt.ArrayLike,
    center_maximum: npt.ArrayLike,
    cross_minima: npt.ArrayLike,
    cross_maxima: npt.ArrayLike,
    *,
    mode: FeatureBandExtremaMode,
) -> BandLimits:
    center_min = np.asarray(center_minimum, dtype=np.float32)
    center_max = np.asarray(center_maximum, dtype=np.float32)
    minima = np.asarray(cross_minima, dtype=np.float32)
    maxima = np.asarray(cross_maxima, dtype=np.float32)
    if center_min.shape != (3,) or center_max.shape != (3,):
        raise ValueError("center extrema must be RGB triplets")
    if minima.ndim != 2 or maxima.shape != minima.shape or minima.shape[1] != 3:
        raise ValueError("cross extrema must be matching Nx3 arrays")
    if mode is FeatureBandExtremaMode.CENTER_ONLY:
        return BandLimits(center_min, center_max)
    return BandLimits(
        np.minimum(center_min, np.min(minima, axis=0)).astype(np.float32),
        np.maximum(center_max, np.max(maxima, axis=0)).astype(np.float32),
    )


@dataclass(frozen=True)
class FeatureBandRange:
    """One scalar auxiliary-pyramid transition range."""

    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.minimum) or not math.isfinite(self.maximum):
            raise ValueError("feature band range must be finite")
        if self.minimum > self.maximum:
            raise ValueError("feature band minimum exceeds maximum")


@dataclass(frozen=True)
class ReconstructionParameters:
    """Runtime fields consumed by the recovered 500/4000 dpi writer math.

    Band tuples are ordered ``(C21-C69, C16-C21, original-C16)``.  Factor
    matrices are ordered ``[band][R/G/B]``.
    """

    resolution_metric: int
    cross_neighbor_cutoff: int
    coarse_enabled: bool
    coarse_reference: float
    coarse_slopes: tuple[float, float, float]
    band_enabled: tuple[bool, bool, bool]
    band_scales: tuple[float, float, float]
    factors_a: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    factors_b: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    configured_strengths: tuple[float, float, float]
    driver_gate_primary: bool
    driver_gate_secondary: bool
    row_reconstruction_gate: int
    dither_scales: tuple[float, float, float]

    def __post_init__(self) -> None:
        if self.resolution_metric < 0 or self.cross_neighbor_cutoff < 0:
            raise ValueError("reconstruction resolution fields cannot be negative")
        if self.row_reconstruction_gate < 0:
            raise ValueError("row reconstruction gate cannot be negative")
        if len(self.coarse_slopes) != 3:
            raise ValueError("coarse correction requires three channel slopes")
        if len(self.band_enabled) != 3 or len(self.band_scales) != 3:
            raise ValueError("reconstruction requires exactly three bands")
        if len(self.factors_a) != 3 or len(self.factors_b) != 3:
            raise ValueError("reconstruction requires three factor rows")
        if any(len(row) != 3 for row in (*self.factors_a, *self.factors_b)):
            raise ValueError("each reconstruction factor row requires R/G/B values")
        if len(self.configured_strengths) != 3 or len(self.dither_scales) != 3:
            raise ValueError("strength and dither tuples require three values")
        numeric = (
            self.coarse_reference,
            *self.coarse_slopes,
            *self.band_scales,
            *(value for row in self.factors_a for value in row),
            *(value for row in self.factors_b for value in row),
            *self.configured_strengths,
            *self.dither_scales,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("reconstruction parameters must be finite")


def feature_band_ranges(
    feature_records: npt.ArrayLike,
    *,
    mode: FeatureBandExtremaMode,
) -> tuple[FeatureBandRange, FeatureBandRange, FeatureBandRange]:
    """Build A21-A69, A16-A21, and A1-A16 scalar ranges.

    ``feature_records[0]`` is the current location.  Cross-neighbor mode
    requires the current, left, right, and two adjacent-row records in any
    order after the current record because only their minimum/maximum matter.
    """

    records = np.asarray(feature_records)
    if records.dtype != np.dtype(np.float32) or records.ndim != 2:
        raise TypeError("feature records must be a float32 Nx4 array")
    if records.shape[1] != 4 or not np.all(np.isfinite(records)):
        raise ValueError("feature records must be finite A69/A21/A16/A1 records")
    if mode is FeatureBandExtremaMode.CENTER_ONLY:
        selected = records[:1]
    else:
        if records.shape[0] != 5:
            raise ValueError("cross-neighbor mode requires exactly five A records")
        selected = records
    ranges: list[FeatureBandRange] = []
    widened = selected.astype(np.float64)
    for transition in range(3):
        differences = widened[:, transition + 1] - widened[:, transition]
        minimum = np.float32(np.min(differences))
        maximum = np.float32(np.max(differences))
        ranges.append(FeatureBandRange(float(minimum), float(maximum)))
    return cast(
        tuple[FeatureBandRange, FeatureBandRange, FeatureBandRange],
        tuple(ranges),
    )


def asymmetric_dead_zone_residual(
    value: npt.ArrayLike,
    *,
    feature_range: FeatureBandRange,
    factors_a: tuple[float, float, float],
    factors_b: tuple[float, float, float],
) -> npt.NDArray[np.float64]:
    """Apply Nikon's sign-aware asymmetric dead zone in widened precision."""

    delta = np.asarray(value)
    if delta.shape != (3,) or not np.issubdtype(delta.dtype, np.floating):
        raise ValueError("asymmetric dead-zone value must be an RGB triplet")
    if not np.all(np.isfinite(delta)):
        raise ValueError("asymmetric dead-zone value must be finite")
    factor_a = np.asarray(factors_a, dtype=np.float32).astype(np.float64)
    factor_b = np.asarray(factors_b, dtype=np.float32).astype(np.float64)
    if factor_a.shape != (3,) or factor_b.shape != (3,):
        raise ValueError("asymmetric dead-zone factors must be RGB triplets")
    if not np.all(np.isfinite(factor_a)) or not np.all(np.isfinite(factor_b)):
        raise ValueError("asymmetric dead-zone factors must be finite")
    edge_min = float(feature_range.minimum)
    edge_max = float(feature_range.maximum)
    if edge_min < 0.0 and edge_max < 0.0:
        upper = factor_b * edge_max
        lower = factor_a * edge_min
    else:
        upper = factor_a * edge_max
        lower = factor_b * edge_min
    values = delta.astype(np.float64)
    residual = np.where(
        values > upper,
        values - upper,
        np.where(values < lower, values - lower, 0.0),
    )
    return np.ascontiguousarray(residual, dtype=np.float64)


def _automatic_strengths(
    weights: np.ndarray,
    configured: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    mutated = np.array(weights, dtype=np.float32, order="C", copy=True)
    strengths = np.empty(3, dtype=np.float64)
    configured32 = np.asarray(configured, dtype=np.float32)

    if configured32[0] == np.float32(0.0):
        mutated[1] = np.float32(2.0 * float(mutated[1]))
        mutated[1] = np.float32(min(1.0, max(0.0, float(mutated[1]))))
        strengths[0] = float(mutated[1])
    else:
        strengths[0] = float(configured32[0])

    if configured32[1] == np.float32(0.0):
        if mutated[2] < np.float32(0.0):
            mutated[2] = np.float32(0.0)
        strengths[1] = float(mutated[2])
    else:
        strengths[1] = float(configured32[1])

    if configured32[2] == np.float32(0.0):
        strengths[2] = float(mutated[3]) * float(mutated[3])
    else:
        strengths[2] = float(configured32[2])
    return mutated, strengths


@dataclass(frozen=True)
class CombinerResult:
    """Intermediate receipt for one recovered reconstruction candidate."""

    candidate: npt.NDArray[np.float64]
    weights_after_automatic_strengths: npt.NDArray[np.float32]
    strengths: npt.NDArray[np.float64]
    residuals: npt.NDArray[np.float64]
    candidates_after_stages: npt.NDArray[np.float64]
    valid: bool

    def __post_init__(self) -> None:
        if self.candidate.dtype != np.dtype(np.float64) or self.candidate.shape != (3,):
            raise ValueError("combined candidate must be a float64 RGB triplet")
        if self.weights_after_automatic_strengths.dtype != np.dtype(
            np.float32
        ) or self.weights_after_automatic_strengths.shape != (4,):
            raise ValueError("combined candidate must preserve one float32 W record")
        if self.strengths.dtype != np.dtype(np.float64) or self.strengths.shape != (3,):
            raise ValueError("combined candidate must expose three strengths")
        if self.residuals.dtype != np.dtype(np.float64) or self.residuals.shape != (
            3,
            3,
        ):
            raise ValueError("combined candidate must expose three RGB residuals")
        if self.candidates_after_stages.dtype != np.dtype(
            np.float64
        ) or self.candidates_after_stages.shape != (5, 3):
            raise ValueError(
                "combined candidate must expose initial/coarse/three-band stages"
            )


def combine_recovered_candidate(
    candidate_69: npt.ArrayLike,
    candidate_21: npt.ArrayLike,
    candidate_16: npt.ArrayLike,
    original: npt.ArrayLike,
    *,
    weight_record: npt.ArrayLike,
    feature_records: npt.ArrayLike,
    parameters: ReconstructionParameters,
) -> CombinerResult:
    """Apply Nikon's coarse correction and exact three-band writer algebra."""

    vectors = [
        np.asarray(value)
        for value in (candidate_69, candidate_21, candidate_16, original)
    ]
    if any(
        value.dtype != np.dtype(np.float32) or value.shape != (3,) for value in vectors
    ):
        raise ValueError("candidate inputs must be float32 RGB triplets")
    weights = np.asarray(weight_record)
    features = np.asarray(feature_records)
    if weights.dtype != np.dtype(np.float32) or weights.shape != (4,):
        raise ValueError("weight record must be float32 W69/W21/W16/W1")
    if (
        features.dtype != np.dtype(np.float32)
        or features.ndim != 2
        or features.shape[1] != 4
    ):
        raise ValueError("feature records must be float32 Nx4")
    if not all(np.all(np.isfinite(value)) for value in (*vectors, weights, features)):
        raise ValueError("combiner inputs must be finite")

    mode = feature_band_extrema_mode(
        resolution_metric=parameters.resolution_metric,
        cross_neighbor_cutoff=parameters.cross_neighbor_cutoff,
    )
    ranges = feature_band_ranges(features, mode=mode)
    mutated_weights, strengths = _automatic_strengths(
        weights, parameters.configured_strengths
    )
    c69, c21, c16, source = vectors
    candidate = c69.astype(np.float64)
    stages = np.empty((5, 3), dtype=np.float64)
    stages[0] = candidate
    if parameters.coarse_enabled:
        coarse_delta = float(np.float32(parameters.coarse_reference)) - float(
            features[0, 0]
        )
        slopes = np.asarray(parameters.coarse_slopes, dtype=np.float32).astype(
            np.float64
        )
        candidate = candidate + slopes * coarse_delta
    stages[1] = candidate

    residuals = np.zeros((3, 3), dtype=np.float64)
    pairs = ((c69, c21), (c21, c16), (c16, source))
    for band, (coarse, fine) in enumerate(pairs):
        if parameters.band_enabled[band]:
            scale = float(np.float32(parameters.band_scales[band]))
            difference = scale * (fine.astype(np.float64) - coarse.astype(np.float64))
            residuals[band] = asymmetric_dead_zone_residual(
                difference,
                feature_range=ranges[band],
                factors_a=parameters.factors_a[band],
                factors_b=parameters.factors_b[band],
            )
            candidate = candidate + strengths[band] * residuals[band]
        stages[band + 2] = candidate

    valid = bool(np.all(np.isfinite(candidate)) and np.all(candidate > 0.0))
    return CombinerResult(
        candidate=np.ascontiguousarray(candidate, dtype=np.float64),
        weights_after_automatic_strengths=np.ascontiguousarray(
            mutated_weights, dtype=np.float32
        ),
        strengths=np.ascontiguousarray(strengths, dtype=np.float64),
        residuals=np.ascontiguousarray(residuals, dtype=np.float64),
        candidates_after_stages=np.ascontiguousarray(stages, dtype=np.float64),
        valid=valid,
    )


def driver_forces_fallback(
    weight_record: npt.ArrayLike,
    parameters: ReconstructionParameters,
) -> bool:
    """Evaluate the recovered row gate before reconstruction is entered."""

    weights = np.asarray(weight_record)
    if weights.dtype != np.dtype(np.float32) or weights.shape != (4,):
        raise ValueError("driver gate requires one float32 W record")
    if not np.all(np.isfinite(weights)):
        raise ValueError("driver gate weight record must be finite")
    if parameters.row_reconstruction_gate != 0:
        return True
    floor_enabled = parameters.driver_gate_primary or parameters.driver_gate_secondary
    return bool(floor_enabled and weights[3] >= np.float32(1.0))


@dataclass(frozen=True)
class WriterResult:
    values: npt.NDArray[np.float32]
    copied_original: npt.NDArray[np.bool_]
    dither_invocations: tuple[int, int, int]
    rng_advances: tuple[int, int, int]
    valid_candidate: bool


def write_reconstructed_pixel(
    candidate: npt.ArrayLike,
    original: npt.ArrayLike,
    *,
    parameters: ReconstructionParameters,
    dither_bounds: DitherBounds,
    generator: LCG24,
) -> WriterResult:
    """Apply final validity, original-floor, redraw, and float32-store rules."""

    rebuilt = np.asarray(candidate)
    source = np.asarray(original)
    if rebuilt.dtype != np.dtype(np.float64) or rebuilt.shape != (3,):
        raise ValueError("writer candidate must be a float64 RGB triplet")
    if source.dtype != np.dtype(np.float32) or source.shape != (3,):
        raise ValueError("writer original must be a float32 RGB triplet")
    valid = bool(np.all(np.isfinite(rebuilt)) and np.all(rebuilt > 0.0))
    if not valid:
        return WriterResult(
            values=np.array(source, dtype=np.float32, copy=True),
            copied_original=np.ones(3, dtype=bool),
            dither_invocations=(0, 0, 0),
            rng_advances=(0, 0, 0),
            valid_candidate=False,
        )

    output = np.empty(3, dtype=np.float32)
    copied = np.zeros(3, dtype=bool)
    calls: list[int] = []
    advances: list[int] = []
    floor_enabled = parameters.driver_gate_primary or parameters.driver_gate_secondary
    for channel in range(3):
        initial_state = generator.state
        scale = parameters.dither_scales[channel]
        first_delta = conditional_dither_delta(
            rebuilt[channel],
            bounds=dither_bounds,
            scale=scale,
            generator=generator,
        )
        call_count = 1
        advance_count = int(generator.state != initial_state)
        if floor_enabled and rebuilt[channel] + first_delta <= float(source[channel]):
            output[channel] = source[channel]
            copied[channel] = True
        else:
            delta = first_delta
            if floor_enabled:
                state_before_redraw = generator.state
                delta = conditional_dither_delta(
                    rebuilt[channel],
                    bounds=dither_bounds,
                    scale=scale,
                    generator=generator,
                )
                call_count += 1
                advance_count += int(generator.state != state_before_redraw)
            output[channel] = np.float32(rebuilt[channel] + delta)
        calls.append(call_count)
        advances.append(advance_count)
    return WriterResult(
        values=output,
        copied_original=copied,
        dither_invocations=cast(tuple[int, int, int], tuple(calls)),
        rng_advances=cast(tuple[int, int, int], tuple(advances)),
        valid_candidate=True,
    )


@dataclass(frozen=True)
class ReconstructionDitherDraws:
    comparison_value: np.float32 | None
    stored_reconstruction_value: np.float32 | None
    lcg_advances: int


def reconstruction_dither_draws(
    value: float,
    *,
    dither_active: bool,
    reconstruction_wins: bool,
    bounds: DitherBounds,
    scale: float,
    generator: LCG24,
) -> ReconstructionDitherDraws:
    """Expose the recovered compare-then-redraw writer call schedule."""

    if not dither_active:
        return ReconstructionDitherDraws(None, None, 0)
    initial_state = generator.state
    comparison = apply_conditional_dither(
        value, bounds=bounds, scale=scale, generator=generator
    )
    after_comparison = generator.state
    stored: np.float32 | None = None
    if reconstruction_wins:
        stored = apply_conditional_dither(
            value, bounds=bounds, scale=scale, generator=generator
        )
    advances = int(after_comparison != initial_state) + int(
        reconstruction_wins and generator.state != after_comparison
    )
    return ReconstructionDitherDraws(comparison, stored, advances)
