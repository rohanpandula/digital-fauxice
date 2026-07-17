"""Compiled per-pixel kernels for the optional cpu-fast backend.

Every function here is a line-by-line port of the audited CPU reference in
``reconstruction.py`` and ``dither.py``: the same float64 widening, float32
narrowing, and accumulation order, with no fastmath, no parallel reductions,
and no reassociation.  Callers gather the small per-pixel patches (already a
cheap NumPy slice of the row-window arrays streaming.py builds) and pass them
in; every scalar reduction, division, and store boundary below matches its
reference counterpart's expression order exactly.

Importing this module requires numba.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from ..rng import LCG24_MASK, NIKON_NORMALIZATION

# Reference coefficients are quantized to float32 once, then widened for every
# use (reconstruction._recovered_unscaled_averages / _vertical_binomial_16).
_COEFF_69 = float(np.float32(1.0 / 69.0))
_COEFF_21 = float(np.float32(1.0 / 21.0))
_COEFF_16 = float(np.float32(1.0 / 16.0))


@njit(cache=True)
def unscaled_averages_scalar(patch):
    """reconstruction._recovered_unscaled_averages for one scalar 9x9 patch.

    Ports _vertical_scratch (S3/S5/S7/S9, float32-narrowed at each store) and
    _vertical_binomial_16 (the vertically-filtered float32 intermediate),
    then the final W69/W21/W16 float64 combination.  Every add order and
    float32 narrowing point matches the reference exactly.
    """

    s3 = np.empty(9, dtype=np.float32)
    s5 = np.empty(9, dtype=np.float32)
    s7 = np.empty(9, dtype=np.float32)
    s9 = np.empty(9, dtype=np.float32)
    binomial = np.empty(9, dtype=np.float32)
    for x in range(9):
        accumulator = np.float64(patch[3, x]) + np.float64(patch[4, x])
        accumulator += np.float64(patch[5, x])
        s3[x] = np.float32(accumulator)
        accumulator += np.float64(patch[2, x])
        accumulator += np.float64(patch[6, x])
        s5[x] = np.float32(accumulator)
        accumulator += np.float64(patch[1, x])
        accumulator += np.float64(patch[7, x])
        s7[x] = np.float32(accumulator)
        accumulator += np.float64(patch[0, x])
        accumulator += np.float64(patch[8, x])
        s9[x] = np.float32(accumulator)

        center = np.float64(patch[4, x])
        bacc = np.float64(patch[3, x]) + center
        bacc += np.float64(patch[5, x])
        bacc += center
        binomial[x] = np.float32(bacc * _COEFF_16)

    total_21 = np.float64(s3[6]) + np.float64(s3[2])
    total_21 += np.float64(s5[3])
    total_21 += np.float64(s5[4])
    total_21 += np.float64(s5[5])

    total_69 = np.float64(s5[8]) + np.float64(s5[0])
    total_69 += np.float64(s7[1])
    total_69 += np.float64(s7[7])
    for x in range(2, 7):
        total_69 += np.float64(s9[x])

    total_16 = np.float64(binomial[4]) + np.float64(binomial[3])
    total_16 += np.float64(binomial[5])
    total_16 += np.float64(binomial[4])

    result = np.empty(3, dtype=np.float64)
    result[0] = total_69 * _COEFF_69
    result[1] = total_21 * _COEFF_21
    result[2] = total_16
    return result


@njit(cache=True)
def rgb_unscaled_averages_scalar(patch3):
    """reconstruction._recovered_rgb_unscaled_averages for one 9x9x3 patch.

    Shares Q69/Q21 with the single-lane helper above, but the 3x3 binomial
    path keeps x87-equivalent (float64) precision through three horizontal
    [1, 2, 1] rows and one vertical combine before a single float32
    coefficient multiply -- no intervening float32 store.  That distinction
    from the W/A schedule is observable in the last bit.
    """

    result = np.empty((3, 3), dtype=np.float64)
    for channel in range(3):
        lane = np.empty((9, 9), dtype=np.float32)
        for y in range(9):
            for x in range(9):
                lane[y, x] = patch3[y, x, channel]
        q = unscaled_averages_scalar(lane)
        result[0, channel] = q[0]
        result[1, channel] = q[1]

        row0 = 0.0
        row1 = 0.0
        row2 = 0.0
        for index in range(3):
            y = 3 + index
            center = np.float64(lane[y, 4])
            horizontal = np.float64(lane[y, 3]) + np.float64(lane[y, 5])
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
        result[2, channel] = total * _COEFF_16
    return result


@njit(cache=True)
def feature_weights_and_feature_scalar(score_patch, waux_patch, point_aux, point_score, fallback):
    """streaming._feature_record for the center location.

    A-lane numerators divide by the UNROUNDED float64 W value, never the
    float32-narrowed weight just stored -- one ULP wrong on a substantial
    minority of samples otherwise.
    """

    unrounded_weights = unscaled_averages_scalar(score_patch)
    weights = np.empty(4, dtype=np.float32)
    weights[0] = np.float32(unrounded_weights[0])
    weights[1] = np.float32(unrounded_weights[1])
    weights[2] = np.float32(unrounded_weights[2])
    weights[3] = point_score
    numerator = unscaled_averages_scalar(waux_patch)
    feature = np.empty(4, dtype=np.float32)
    for lane in range(3):
        if unrounded_weights[lane] > 0.0:
            feature[lane] = np.float32(numerator[lane] / unrounded_weights[lane])
        else:
            feature[lane] = fallback
    if weights[3] > np.float32(0.0):
        feature[3] = point_aux
    else:
        feature[3] = fallback
    return weights, feature


@njit(cache=True)
def neighbor_feature_scalar(score_patch, waux_patch, point_aux, point_score, fallback, is_valid):
    """streaming._cross_neighbor_feature_record: only the feature is kept.

    The scheduler's horizontal guard columns stay zero (not fallback) when a
    neighbor falls outside the row; the caller's weights are never used for
    neighbors, only their feature record.
    """

    if not is_valid:
        return np.zeros(4, dtype=np.float32)
    _, feature = feature_weights_and_feature_scalar(
        score_patch, waux_patch, point_aux, point_score, fallback
    )
    return feature


@njit(cache=True)
def driver_forces_fallback_scalar(w3, row_reconstruction_gate, floor_enabled):
    """reconstruction.driver_forces_fallback."""

    if row_reconstruction_gate != 0:
        return True
    return floor_enabled and w3 >= np.float32(1.0)


@njit(cache=True)
def rgb_candidates_scalar(wrgb_patch, w3):
    """reconstruction.recovered_rgb_candidates (UNNORMALIZED_SUM policy)."""

    averages = rgb_unscaled_averages_scalar(wrgb_patch)
    candidates = np.empty((3, 3), dtype=np.float32)
    for scale_index in range(3):
        denominator = w3[scale_index]
        if denominator > np.float32(0.0):
            reciprocal = 1.0 / np.float64(denominator)
            for channel in range(3):
                candidates[scale_index, channel] = np.float32(
                    averages[scale_index, channel] * reciprocal
                )
        else:
            for channel in range(3):
                candidates[scale_index, channel] = np.float32(averages[scale_index, channel])
    return candidates


@njit(cache=True)
def feature_band_ranges_scalar(features, record_count):
    """reconstruction.feature_band_ranges.

    ``record_count`` is 1 for CENTER_ONLY (minimum == maximum, the center's
    own transition) or 5 for CROSS_NEIGHBOR.  Min/max over a set is exact and
    order-independent, so an incremental running extremum matches NumPy's
    batch reduction bit-for-bit.
    """

    range_min = np.empty(3, dtype=np.float64)
    range_max = np.empty(3, dtype=np.float64)
    for transition in range(3):
        d0 = np.float64(features[0, transition + 1]) - np.float64(features[0, transition])
        minimum = d0
        maximum = d0
        for r in range(1, record_count):
            d = np.float64(features[r, transition + 1]) - np.float64(features[r, transition])
            if d < minimum:
                minimum = d
            if d > maximum:
                maximum = d
        range_min[transition] = np.float64(np.float32(minimum))
        range_max[transition] = np.float64(np.float32(maximum))
    return range_min, range_max


@njit(cache=True)
def automatic_strengths_scalar(weights, cfg0, cfg1, cfg2):
    """reconstruction._automatic_strengths.

    Operates on a fresh copy semantics: the mutation here never feeds back
    into the RGB-candidate normalizers, which already consumed the original
    weights earlier in the chain.
    """

    strengths = np.empty(3, dtype=np.float64)

    if cfg0 == np.float32(0.0):
        doubled = np.float32(2.0 * np.float64(weights[1]))
        clamped = np.float64(doubled)
        if clamped < 0.0:
            clamped = 0.0
        if clamped > 1.0:
            clamped = 1.0
        strengths[0] = np.float64(np.float32(clamped))
    else:
        strengths[0] = np.float64(cfg0)

    if cfg1 == np.float32(0.0):
        m2 = weights[2]
        if m2 < np.float32(0.0):
            m2 = np.float32(0.0)
        strengths[1] = np.float64(m2)
    else:
        strengths[1] = np.float64(cfg1)

    if cfg2 == np.float32(0.0):
        strengths[2] = np.float64(weights[3]) * np.float64(weights[3])
    else:
        strengths[2] = np.float64(cfg2)

    return strengths


@njit(cache=True)
def combine_candidate_scalar(
    c69,
    c21,
    c16,
    source,
    weights,
    features,
    record_count,
    coarse_enabled,
    coarse_reference,
    coarse_slopes,
    band_enabled,
    band_scales,
    factors_a,
    factors_b,
    configured_strengths,
):
    """reconstruction.combine_recovered_candidate (final candidate only).

    Diagnostic-only fields (mutated weights, strengths, per-band residuals,
    per-stage candidates) are not part of the public byte-exact path -- only
    ``combined.candidate`` reaches write_reconstructed_pixel -- so they are
    not materialized here.
    """

    range_min, range_max = feature_band_ranges_scalar(features, record_count)
    strengths = automatic_strengths_scalar(
        weights, configured_strengths[0], configured_strengths[1], configured_strengths[2]
    )
    candidate = np.empty(3, dtype=np.float64)
    for channel in range(3):
        candidate[channel] = np.float64(c69[channel])
    if coarse_enabled:
        coarse_delta = np.float64(coarse_reference) - np.float64(features[0, 0])
        for channel in range(3):
            candidate[channel] = candidate[channel] + np.float64(coarse_slopes[channel]) * coarse_delta

    for band in range(3):
        if band_enabled[band] == 0:
            continue
        scale = np.float64(band_scales[band])
        edge_min = range_min[band]
        edge_max = range_max[band]
        negative_band = edge_min < 0.0 and edge_max < 0.0
        for channel in range(3):
            if band == 0:
                coarse_value = np.float64(c69[channel])
                fine_value = np.float64(c21[channel])
            elif band == 1:
                coarse_value = np.float64(c21[channel])
                fine_value = np.float64(c16[channel])
            else:
                coarse_value = np.float64(c16[channel])
                fine_value = np.float64(source[channel])
            difference = scale * (fine_value - coarse_value)
            if negative_band:
                upper = np.float64(factors_b[band, channel]) * edge_max
                lower = np.float64(factors_a[band, channel]) * edge_min
            else:
                upper = np.float64(factors_a[band, channel]) * edge_max
                lower = np.float64(factors_b[band, channel]) * edge_min
            if difference > upper:
                residual = difference - upper
            elif difference < lower:
                residual = difference - lower
            else:
                residual = 0.0
            candidate[channel] = candidate[channel] + strengths[band] * residual
    return candidate


@njit(cache=True)
def dither_delta_scalar(value64, low64, high64, low_lt_high, scale32, state):
    """dither.conditional_dither_delta, threading the LCG24 state explicitly.

    The generator advances exactly once per call that reaches the widened
    comparison; a candidate outside the strict (low, high) interval returns
    zero without consuming a draw.
    """

    candidate32 = np.float32(value64)
    candidate = np.float64(candidate32)
    if not low_lt_high:
        return 0.0, state
    if not (low64 < candidate < high64):
        return 0.0, state
    width = high64 - low64
    coefficient = 4.0 / (width * width)
    envelope = ((high64 - candidate) * (candidate - low64)) * coefficient
    random_span = np.float32(np.float64(scale32) * candidate)
    state = (125 * state + 1) & LCG24_MASK
    centered = (state + 1) * NIKON_NORMALIZATION - 0.5
    random_value = 0.0 + centered * np.float64(random_span)
    delta = envelope * random_value
    changed = candidate + delta
    if low64 < changed < high64:
        return delta, state
    return 0.0, state


@njit(cache=True)
def write_pixel_scalar(candidate, original, floor_enabled, low64, high64, low_lt_high, dither_scales, state):
    """reconstruction.write_reconstructed_pixel.

    RNG advance counting replicates the reference's ``state != previous``
    comparisons rather than assuming every reachable draw advances the LCG.
    """

    valid = (
        math.isfinite(candidate[0])
        and math.isfinite(candidate[1])
        and math.isfinite(candidate[2])
        and candidate[0] > 0.0
        and candidate[1] > 0.0
        and candidate[2] > 0.0
    )
    values = np.empty(3, dtype=np.float32)
    if not valid:
        values[0] = original[0]
        values[1] = original[1]
        values[2] = original[2]
        return values, np.int64(0), state

    advances = 0
    for channel in range(3):
        scale32 = dither_scales[channel]
        state_before = state
        first_delta, state = dither_delta_scalar(
            candidate[channel], low64, high64, low_lt_high, scale32, state
        )
        if state != state_before:
            advances += 1
        source_channel = np.float64(original[channel])
        if floor_enabled and (candidate[channel] + first_delta <= source_channel):
            values[channel] = original[channel]
        else:
            delta = first_delta
            if floor_enabled:
                state_before_redraw = state
                delta, state = dither_delta_scalar(
                    candidate[channel], low64, high64, low_lt_high, scale32, state
                )
                if state != state_before_redraw:
                    advances += 1
            values[channel] = np.float32(candidate[channel] + delta)
    return values, np.int64(advances), state


@njit(cache=True)
def process_selected_pixel(
    score_patches,
    waux_patches,
    wrgb_center,
    point_aux5,
    point_score5,
    neighbor_valid5,
    record_count,
    fallback_value,
    original_rgb,
    floor_enabled,
    row_reconstruction_gate,
    coarse_enabled,
    coarse_slopes,
    band_enabled,
    band_scales,
    factors_a,
    factors_b,
    configured_strengths,
    low64,
    high64,
    low_lt_high,
    dither_scales,
    state,
):
    """The full per-attempted-pixel chain: feature records -> driver gate ->
    candidates -> combiner -> writer, fused into one compiled call.

    ``score_patches``/``waux_patches`` hold five 9x9 patches in the order
    [center, left, right, up, down] (only index 0 is read when
    ``record_count`` is 1).  ``point_aux5``/``point_score5``/
    ``neighbor_valid5`` are the matching point values and the
    "in-bounds" flag streaming._cross_neighbor_feature_record applies to the
    two horizontal neighbors.  Returns
    ``(attempted, values, rng_advances, new_state)``; when not attempted, the
    caller must leave its output unchanged (``values`` is a no-op echo of
    ``original_rgb``).
    """

    weights, center_feature = feature_weights_and_feature_scalar(
        score_patches[0], waux_patches[0], point_aux5[0], point_score5[0], fallback_value
    )
    if driver_forces_fallback_scalar(weights[3], row_reconstruction_gate, floor_enabled):
        return False, original_rgb, np.int64(0), state

    features = np.zeros((5, 4), dtype=np.float32)
    for lane in range(4):
        features[0, lane] = center_feature[lane]
    if record_count == 5:
        for i in range(1, 5):
            feature_i = neighbor_feature_scalar(
                score_patches[i],
                waux_patches[i],
                point_aux5[i],
                point_score5[i],
                fallback_value,
                neighbor_valid5[i] != 0,
            )
            for lane in range(4):
                features[i, lane] = feature_i[lane]

    w3 = np.empty(3, dtype=np.float32)
    w3[0] = weights[0]
    w3[1] = weights[1]
    w3[2] = weights[2]
    candidates = rgb_candidates_scalar(wrgb_center, w3)

    combined = combine_candidate_scalar(
        candidates[0],
        candidates[1],
        candidates[2],
        original_rgb,
        weights,
        features,
        record_count,
        coarse_enabled,
        fallback_value,
        coarse_slopes,
        band_enabled,
        band_scales,
        factors_a,
        factors_b,
        configured_strengths,
    )
    values, advances, new_state = write_pixel_scalar(
        combined, original_rgb, floor_enabled, low64, high64, low_lt_high, dither_scales, state
    )
    return True, values, advances, new_state
