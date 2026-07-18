"""Permanent numba bit-exactness harness for the optional cpu-fast backend.

These tests prove numba's arithmetic model matches the assumptions the
compiled kernels depend on: float64/float32 widening and narrowing behave
exactly like NumPy's, accumulation order is preserved (no reassociation), the
24-bit LCG state trajectory is reproduced exactly, and default ``@njit``
never contracts ``a * b + c`` into a fused multiply-add.  Comparisons are
always bitwise (``.view(np.uint*)``), never decimal tolerances.

The njit twins compare against the real reference implementations in
``reconstruction``, ``dither``, ``rng``, and ``x3a`` (not copies of their
math), so any future change to the reference is exercised here for free.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

numba = pytest.importorskip("numba")

from numba import njit  # noqa: E402

from portable_digital_ice.dither import DitherBounds, conditional_dither_delta  # noqa: E402
from portable_digital_ice.fast_cpu import CpuFastUnavailable  # noqa: E402
from portable_digital_ice.reconstruction import (  # noqa: E402
    _recovered_rgb_unscaled_averages,
    _recovered_unscaled_averages,
)
from portable_digital_ice.rng import LCG24, LCG24_MASK, NIKON_NORMALIZATION  # noqa: E402
from portable_digital_ice.x3a import ScoreParameters, continuous_score  # noqa: E402


def test_fast_cpu_package_exposes_unavailable_guard() -> None:
    """The package skeleton's fail-closed guard must exist independent of numba."""

    assert issubclass(CpuFastUnavailable, RuntimeError)


# --------------------------------------------------------------------------
# njit twins: scalar/loop ports of the reference's per-pixel building blocks.
# These mirror the eventual kernels.py port; they exist here so the harness
# does not depend on kernels.py's internal structure evolving in later
# milestones.
# --------------------------------------------------------------------------


@njit(cache=False)
def _jit_unscaled_averages(values: np.ndarray, out_result: np.ndarray) -> None:
    """Twin of reconstruction._recovered_unscaled_averages (W/A schedule)."""

    lanes = values.shape[2]
    s3 = np.empty((9, lanes), dtype=np.float32)
    s5 = np.empty((9, lanes), dtype=np.float32)
    s7 = np.empty((9, lanes), dtype=np.float32)
    s9 = np.empty((9, lanes), dtype=np.float32)
    binomial = np.empty((9, lanes), dtype=np.float32)
    bin_coeff = np.float64(np.float32(1.0 / 16.0))
    for x in range(9):
        for lane in range(lanes):
            accumulator = np.float64(values[3, x, lane]) + np.float64(values[4, x, lane])
            accumulator += np.float64(values[5, x, lane])
            s3[x, lane] = np.float32(accumulator)
            accumulator += np.float64(values[2, x, lane])
            accumulator += np.float64(values[6, x, lane])
            s5[x, lane] = np.float32(accumulator)
            accumulator += np.float64(values[1, x, lane])
            accumulator += np.float64(values[7, x, lane])
            s7[x, lane] = np.float32(accumulator)
            accumulator += np.float64(values[0, x, lane])
            accumulator += np.float64(values[8, x, lane])
            s9[x, lane] = np.float32(accumulator)

            center = np.float64(values[4, x, lane])
            bacc = np.float64(values[3, x, lane]) + center
            bacc += np.float64(values[5, x, lane])
            bacc += center
            binomial[x, lane] = np.float32(bacc * bin_coeff)

    coefficient_69 = np.float64(np.float32(1.0 / 69.0))
    coefficient_21 = np.float64(np.float32(1.0 / 21.0))
    for lane in range(lanes):
        total_21 = np.float64(s3[6, lane]) + np.float64(s3[2, lane])
        total_21 += np.float64(s5[3, lane])
        total_21 += np.float64(s5[4, lane])
        total_21 += np.float64(s5[5, lane])

        total_69 = np.float64(s5[8, lane]) + np.float64(s5[0, lane])
        total_69 += np.float64(s7[1, lane])
        total_69 += np.float64(s7[7, lane])
        for x in range(2, 7):
            total_69 += np.float64(s9[x, lane])

        total_16 = np.float64(binomial[4, lane]) + np.float64(binomial[3, lane])
        total_16 += np.float64(binomial[5, lane])
        total_16 += np.float64(binomial[4, lane])

        out_result[0, lane] = total_69 * coefficient_69
        out_result[1, lane] = total_21 * coefficient_21
        out_result[2, lane] = total_16


@njit(cache=False)
def _jit_rgb_unscaled_averages(values: np.ndarray, out_result: np.ndarray) -> None:
    """Twin of reconstruction._recovered_rgb_unscaled_averages (distinct 1/16 path)."""

    _jit_unscaled_averages(values, out_result)
    coefficient = np.float64(np.float32(1.0 / 16.0))
    for lane in range(out_result.shape[1]):
        row0 = 0.0
        row1 = 0.0
        row2 = 0.0
        for index, y in enumerate((3, 4, 5)):
            center = np.float64(values[y, 4, lane])
            horizontal = np.float64(values[y, 3, lane]) + np.float64(values[y, 5, lane])
            horizontal += center
            horizontal += center
            if index == 0:
                row0 = horizontal
            elif index == 1:
                row1 = horizontal
            else:
                row2 = horizontal
        total = row0 + row1
        total += row1
        total += row2
        out_result[2, lane] = total * coefficient


@njit(cache=False)
def _jit_dither_delta(
    value32: np.float32,
    low32: np.float32,
    high32: np.float32,
    scale32: np.float32,
    state: int,
) -> tuple[float, int]:
    """Twin of dither.conditional_dither_delta, threading the LCG24 state explicitly."""

    candidate = np.float64(value32)
    low = np.float64(low32)
    high = np.float64(high32)
    if not low < high:
        return 0.0, state
    if not (low < candidate < high):
        return 0.0, state
    width = high - low
    coefficient = 4.0 / (width * width)
    envelope = ((high - candidate) * (candidate - low)) * coefficient
    random_span = np.float32(np.float64(scale32) * candidate)
    state = (125 * state + 1) & LCG24_MASK
    centered = (state + 1) * NIKON_NORMALIZATION - 0.5
    random_value = 0.0 + centered * np.float64(random_span)
    delta = envelope * random_value
    changed = candidate + delta
    if low < changed < high:
        return delta, state
    return 0.0, state


@njit(cache=False)
def _jit_score(
    sample64: float, primary: float, addend: float, scale: float, offset: float, lower: float
) -> np.float32:
    """Twin of x3a.continuous_score's per-element affine expression."""

    score64 = ((primary + addend) - sample64) * scale + offset
    score64 = min(score64, 1.0)
    score64 = max(score64, lower)
    return np.float32(score64)


@njit(cache=False)
def _jit_mul_add(a: float, b: float, c: float) -> float:
    return a * b + c


# --------------------------------------------------------------------------
# 1/2: averages chains, 1-lane and 3-lane, four magnitude regimes.
# --------------------------------------------------------------------------

_MAGNITUDES = (1.0, 1e-3, 1e4, 6e4)
_LANE_COUNTS = (1, 3)
_TRIALS_PER_MAGNITUDE = 1000  # x4 magnitudes x2 lane-counts = 8000 comparisons/function


def test_recovered_unscaled_averages_bitexact() -> None:
    """W/A vertical-scratch + binomial-16 schedule: njit twin vs the reference."""

    rng = np.random.default_rng(20260717)
    checked = 0
    for magnitude in _MAGNITUDES:
        for trial in range(_TRIALS_PER_MAGNITUDE):
            for lanes in _LANE_COUNTS:
                patch = (rng.random((9, 9, lanes), dtype=np.float32) * magnitude).astype(
                    np.float32
                )
                expected = _recovered_unscaled_averages(patch)
                actual = np.empty((3, lanes), dtype=np.float64)
                _jit_unscaled_averages(patch, actual)
                assert (
                    expected.view(np.uint64).tobytes() == actual.view(np.uint64).tobytes()
                ), f"magnitude={magnitude} lanes={lanes} trial={trial}"
                checked += 1
    assert checked == len(_MAGNITUDES) * _TRIALS_PER_MAGNITUDE * len(_LANE_COUNTS)


def test_recovered_rgb_unscaled_averages_bitexact() -> None:
    """RGB-candidate schedule (distinct binomial narrowing): njit twin vs the reference."""

    rng = np.random.default_rng(20260718)
    checked = 0
    for magnitude in _MAGNITUDES:
        for trial in range(_TRIALS_PER_MAGNITUDE):
            for lanes in _LANE_COUNTS:
                patch = (rng.random((9, 9, lanes), dtype=np.float32) * magnitude).astype(
                    np.float32
                )
                expected = _recovered_rgb_unscaled_averages(patch)
                actual = np.empty((3, lanes), dtype=np.float64)
                _jit_rgb_unscaled_averages(patch, actual)
                assert (
                    expected.view(np.uint64).tobytes() == actual.view(np.uint64).tobytes()
                ), f"magnitude={magnitude} lanes={lanes} trial={trial}"
                checked += 1
    assert checked == len(_MAGNITUDES) * _TRIALS_PER_MAGNITUDE * len(_LANE_COUNTS)


# --------------------------------------------------------------------------
# 3: 500k sequential conditional-dither draws sharing one LCG, asserting the
# accepted/rejected delta AND the full state trajectory at every step.
# --------------------------------------------------------------------------


def test_dither_lcg_walk_bitexact_with_state_trajectory() -> None:
    """500k sequential draws: njit twin matches the reference delta and RNG state."""

    rng = np.random.default_rng(20260719)
    low32 = np.float32(1200.5)
    high32 = np.float32(58000.25)
    bounds = DitherBounds(low=low32, high=high32, low_index=0, high_index=0)
    generator = LCG24(0x3045)
    state = 0x3045
    draws = 500_000
    for index in range(draws):
        value = np.float32(rng.random(dtype=np.float32) * 65535.0)
        scale = np.float32((index % 3 + 1) * 0.001)
        expected_delta = conditional_dither_delta(
            float(value), bounds=bounds, scale=float(scale), generator=generator
        )
        actual_delta, state = _jit_dither_delta(value, low32, high32, scale, state)
        assert np.float64(expected_delta).view(np.uint64) == np.float64(actual_delta).view(
            np.uint64
        ), f"delta mismatch at draw {index}"
        assert state == generator.state, f"state mismatch at draw {index}"
    assert state == generator.state
    assert draws == 500_000


# --------------------------------------------------------------------------
# 4: the continuous_score float64 expression with final float32 narrowing.
# --------------------------------------------------------------------------


def test_score_expression_bitexact() -> None:
    """continuous_score's affine equation: scalar njit vs the vectorized reference."""

    rng = np.random.default_rng(20260720)
    primary = float(np.float32(46618.785))
    addend = float(np.float32(1.5))
    scale = float(np.float32(-2.147e-05))
    offset = float(np.float32(1.9))
    lower = float(np.float32(0.35))
    parameters = ScoreParameters(
        base_primary=primary,
        base_addend=addend,
        scale=scale,
        offset=offset,
        floor=lower,
        resolution_metric=4000,
        # cutoff == metric keeps `horizontal_minimum` False regardless of width,
        # isolating the affine equation itself from the horizontal min-filter.
        horizontal_minimum_resolution_cutoff=4000,
    )
    samples = (rng.random(200_000, dtype=np.float32) * 65535.0).astype(np.float32)
    expected = continuous_score(samples.reshape(1, -1), parameters)[0]
    wide = samples.astype(np.float64)
    checked = 0
    for index in range(0, samples.size, 997):
        jit_value = _jit_score(wide[index], primary, addend, scale, offset, lower)
        assert np.float32(jit_value).view(np.uint32) == expected[index].view(
            np.uint32
        ), index
        checked += 1
    assert checked > 0


# --------------------------------------------------------------------------
# 5: automatic strengths must match the reference clamp semantics exactly,
# including the NaN-collapsing min(1.0, max(0.0, x)) literal-first order.
# --------------------------------------------------------------------------


def test_automatic_strengths_matches_reference_including_nan() -> None:
    """kernels.automatic_strengths_scalar vs reconstruction._automatic_strengths.

    NaN weights are unreachable through the validated pipeline (the combiner
    rejects non-finite inputs first), but the helper itself must still be a
    bit-exact twin: the reference's ``min(1.0, max(0.0, x))`` collapses a NaN
    doubled W21 to 0.0, while the other two lanes propagate NaN.
    """

    from portable_digital_ice.fast_cpu import kernels
    from portable_digital_ice.reconstruction import _automatic_strengths

    rng = np.random.default_rng(20260724)
    cases = [
        np.array([0.5, 0.25, -0.125, 0.75], dtype=np.float32),
        np.array([0.5, 0.75, 0.5, 1.25], dtype=np.float32),
        np.array([0.5, -0.25, 0.0, 0.0], dtype=np.float32),
    ]
    for _ in range(50):
        cases.append((rng.random(4, dtype=np.float32) * 2.0 - 0.5).astype(np.float32))
    for lane in (1, 2, 3):
        poisoned = np.array([0.5, 0.25, -0.125, 0.75], dtype=np.float32)
        poisoned[lane] = np.float32(np.nan)
        cases.append(poisoned)

    zero = np.float32(0.0)
    for index, weights in enumerate(cases):
        _, expected = _automatic_strengths(weights, (0.0, 0.0, 0.0))
        actual = kernels.automatic_strengths_scalar(weights, zero, zero, zero)
        assert np.array_equal(
            np.asarray(expected).view(np.uint64),
            np.asarray(actual).view(np.uint64),
        ), f"case {index}: expected={expected!r} actual={actual!r}"


# --------------------------------------------------------------------------
# 6: contraction canary. Find triples where FMA changes a*b+c and prove njit
# does not silently fuse under default @njit (no fastmath).
# --------------------------------------------------------------------------


def test_fma_contraction_canary_zero_deviations() -> None:
    """njit's a*b+c must not fuse into FMA on discriminating triples.

    ``math.fma`` exists only on Python 3.13+, so older interpreters run the
    zero-deviation comparison over every triple and leave the
    discriminating-power proof to the 3.13 lanes.
    """

    fused_reference = getattr(math, "fma", None)
    rng = np.random.default_rng(20260721)
    canary_diff = 0
    canary_failures = 0
    trials = 200_000
    for _ in range(trials):
        a = rng.random() * 2.0 - 1.0
        b = rng.random() * 2.0 - 1.0
        c = (rng.random() * 2.0 - 1.0) * 1e-8
        unfused = a * b + c
        if _jit_mul_add(a, b, c) != unfused:
            canary_failures += 1
        if fused_reference is not None and fused_reference(a, b, c) != unfused:
            canary_diff += 1
    if fused_reference is not None:
        assert canary_diff > 0, (
            "canary produced no discriminating triples; test is vacuous"
        )
    assert canary_failures == 0, (
        f"{canary_failures} njit a*b+c calls deviated from unfused evaluation"
    )
