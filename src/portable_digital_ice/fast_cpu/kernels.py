"""Compiled per-pixel and per-row kernels for the optional cpu-fast backend.

Every function here is a line-by-line port of the audited CPU reference in
``streaming.py``, ``reconstruction.py``, and ``dither.py``: the same float64
widening, float32 narrowing, and accumulation order, with no fastmath, no
parallel reductions, and no reassociation.  The entry point is
``process_row``, which fuses one whole output row -- decision eligibility,
history-window boundary handling, feature records, candidates, combiner, and
writer -- into a single compiled call; its helpers gather 9x9 patches
directly from whole-image planes the caller builds once per run.

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


# ---------------------------------------------------------------------------
# Hidden startup replay (startup.replay_hidden_startup_rows).  The histories
# are built by the reference's own _startup_histories helper (padded width,
# pseudo-row and guard rules already materialized); this kernel walks one
# hidden stage: the four decision bars on the raw auxiliary history, the
# zero-record rule for logical rows at or below the first hidden center and
# for out-of-row neighbors, the per-record fallback values (each neighbor's
# producer stage differs), and the same candidate/combiner/writer chain as
# the public rows against row zero's working RGB.
# ---------------------------------------------------------------------------


@njit(cache=True)
def startup_feature_record(
    score_history,
    waux_history,
    raw_aux_history,
    logical_y,
    x,
    width,
    guard,
    minimum_y,
    hidden_floor,
    fallback,
):
    """startup._startup_feature_record: W/A records for one hidden center.

    Ring slots for rows at or below the first hidden center retain allocator
    zero, as do horizontal guards -- both weights and feature stay zero, not
    fallback.
    """

    weights = np.zeros(4, dtype=np.float32)
    feature = np.zeros(4, dtype=np.float32)
    if logical_y <= hidden_floor or x < 0 or x >= width:
        return weights, feature
    center_row = logical_y - minimum_y
    padded_x = x + guard
    score_patch = score_history[
        center_row - 4 : center_row + 5, padded_x - 4 : padded_x + 5
    ]
    waux_patch = waux_history[
        center_row - 4 : center_row + 5, padded_x - 4 : padded_x + 5
    ]
    unrounded_weights = unscaled_averages_scalar(score_patch)
    weights[0] = np.float32(unrounded_weights[0])
    weights[1] = np.float32(unrounded_weights[1])
    weights[2] = np.float32(unrounded_weights[2])
    weights[3] = score_history[center_row, padded_x]
    numerator = unscaled_averages_scalar(waux_patch)
    for lane in range(3):
        if unrounded_weights[lane] > 0.0:
            feature[lane] = np.float32(numerator[lane] / unrounded_weights[lane])
        else:
            feature[lane] = fallback
    if weights[3] > np.float32(0.0):
        feature[3] = raw_aux_history[center_row, padded_x]
    else:
        feature[3] = fallback
    return weights, feature


@njit(cache=True)
def startup_stage(
    score_history,
    waux_history,
    wrgb_history,
    raw_aux_history,
    center_y,
    minimum_y,
    width,
    writer_width,
    guard,
    hidden_floor,
    radius,
    threshold,
    count_limit,
    record_count,
    fallback_center,
    fallback_left,
    fallback_right,
    fallback_up,
    fallback_down,
    combiner_coarse_reference,
    floor_enabled,
    row_reconstruction_gate,
    original_rgb_row,
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
    """One hidden startup stage; returns (attempted, rng_advances, state)."""

    center_row = center_y - minimum_y
    attempted = 0
    advances = np.int64(0)
    features = np.zeros((5, 4), dtype=np.float32)
    w3 = np.empty(3, dtype=np.float32)
    for x in range(writer_width):
        padded_x = x + guard
        # The recovered four nine-sample decision bars on raw auxiliary
        # history (x3a.DecisionOffsets.captured_normal_ls5000 layout).
        decision_fallback = False
        for group in range(4):
            count = 0
            for sample in range(9):
                axis = sample - 4
                if group == 0:
                    dy = axis
                    dx = -radius
                elif group == 1:
                    dy = -radius
                    dx = axis
                elif group == 2:
                    dy = axis
                    dx = radius
                else:
                    dy = radius
                    dx = axis
                if raw_aux_history[center_row + dy, padded_x + dx] < threshold:
                    count += 1
            if count > count_limit:
                decision_fallback = True
                break

        weights, center_feature = startup_feature_record(
            score_history,
            waux_history,
            raw_aux_history,
            center_y,
            x,
            width,
            guard,
            minimum_y,
            hidden_floor,
            fallback_center,
        )
        if decision_fallback or driver_forces_fallback_scalar(
            weights[3], row_reconstruction_gate, floor_enabled
        ):
            continue
        attempted += 1
        for lane in range(4):
            features[0, lane] = center_feature[lane]
        if record_count == 5:
            _, feature_n = startup_feature_record(
                score_history,
                waux_history,
                raw_aux_history,
                center_y,
                x - 1,
                width,
                guard,
                minimum_y,
                hidden_floor,
                fallback_left,
            )
            for lane in range(4):
                features[1, lane] = feature_n[lane]
            _, feature_n = startup_feature_record(
                score_history,
                waux_history,
                raw_aux_history,
                center_y,
                x + 1,
                width,
                guard,
                minimum_y,
                hidden_floor,
                fallback_right,
            )
            for lane in range(4):
                features[2, lane] = feature_n[lane]
            _, feature_n = startup_feature_record(
                score_history,
                waux_history,
                raw_aux_history,
                center_y - 1,
                x,
                width,
                guard,
                minimum_y,
                hidden_floor,
                fallback_up,
            )
            for lane in range(4):
                features[3, lane] = feature_n[lane]
            _, feature_n = startup_feature_record(
                score_history,
                waux_history,
                raw_aux_history,
                center_y + 1,
                x,
                width,
                guard,
                minimum_y,
                hidden_floor,
                fallback_down,
            )
            for lane in range(4):
                features[4, lane] = feature_n[lane]

        wrgb_patch = wrgb_history[center_row - 4 : center_row + 5, x : x + 9]
        w3[0] = weights[0]
        w3[1] = weights[1]
        w3[2] = weights[2]
        candidates = rgb_candidates_scalar(wrgb_patch, w3)
        combined = combine_candidate_scalar(
            candidates[0],
            candidates[1],
            candidates[2],
            original_rgb_row[x],
            weights,
            features,
            record_count,
            coarse_enabled,
            combiner_coarse_reference,
            coarse_slopes,
            band_enabled,
            band_scales,
            factors_a,
            factors_b,
            configured_strengths,
        )
        _values, pixel_advances, state = write_pixel_scalar(
            combined,
            original_rgb_row[x],
            floor_enabled,
            low64,
            high64,
            low_lt_high,
            dither_scales,
            state,
        )
        advances += pixel_advances
    return attempted, advances, state


# ---------------------------------------------------------------------------
# M2: whole-row fusion.  The scalar helpers above are the byte-exact per-
# pixel chain (feature records, driver gate, candidates, combiner, writer).
# The functions below replace the *gather* step -- instead of Python slicing
# streaming._history_window's per-row padded strip once per selected pixel,
# these index directly into whole-image auxiliary/score/weighted-auxiliary/
# weighted-rgb/working planes with the exact same boundary rules
# pad_reconstruction_history/_history_window encode:
# horizontal guards stay zero, logical row -1 is the pseudo-row (score
# floor; weighted = first-real-row * floor), rows >= height repeat the last
# real row, and rows below -1 stay zero.  This mirrors the CUDA backend's
# load_score_hist/load_waux_hist/load_wrgb_hist loaders, which are already
# proven byte-exact against this same reference.
# ---------------------------------------------------------------------------


@njit(cache=True)
def load_score_hist(score_all, height, width, score_floor, y, x):
    """streaming._history_window's score-history boundary rule, one sample."""

    if x < 0 or x >= width:
        return np.float32(0.0)
    if y < -1:
        return np.float32(0.0)
    if y == -1:
        return score_floor
    if y >= height:
        y = height - 1
    return score_all[y, x]


@njit(cache=True)
def load_waux_hist(waux_all, aux_all, height, width, score_floor, y, x):
    """streaming._history_window's weighted-auxiliary boundary rule.

    The y == -1 pseudo-row multiplies the FIRST real row's raw auxiliary by
    the score floor -- not that row's own weighted-auxiliary product.
    """

    if x < 0 or x >= width:
        return np.float32(0.0)
    if y < -1:
        return np.float32(0.0)
    if y == -1:
        return np.float32(aux_all[0, x] * score_floor)
    if y >= height:
        y = height - 1
    return waux_all[y, x]


@njit(cache=True)
def load_wrgb_hist_channel(wrgb_all, working_all, height, width, score_floor, y, x, channel):
    """streaming._history_window's weighted-RGB boundary rule, one channel."""

    if x < 0 or x >= width:
        return np.float32(0.0)
    if y < -1:
        return np.float32(0.0)
    if y == -1:
        return np.float32(working_all[0, x, channel] * score_floor)
    if y >= height:
        y = height - 1
    return wrgb_all[y, x, channel]


@njit(cache=True)
def gather_score_patch(score_all, height, width, score_floor, cy, cx, out):
    for dy in range(-4, 5):
        for dx in range(-4, 5):
            out[dy + 4, dx + 4] = load_score_hist(
                score_all, height, width, score_floor, cy + dy, cx + dx
            )


@njit(cache=True)
def gather_waux_patch(waux_all, aux_all, height, width, score_floor, cy, cx, out):
    for dy in range(-4, 5):
        for dx in range(-4, 5):
            out[dy + 4, dx + 4] = load_waux_hist(
                waux_all, aux_all, height, width, score_floor, cy + dy, cx + dx
            )


@njit(cache=True)
def gather_wrgb_patch(wrgb_all, working_all, height, width, score_floor, cy, cx, out):
    for dy in range(-4, 5):
        for dx in range(-4, 5):
            for channel in range(3):
                out[dy + 4, dx + 4, channel] = load_wrgb_hist_channel(
                    wrgb_all, working_all, height, width, score_floor, cy + dy, cx + dx, channel
                )


@njit(cache=True)
def compute_row_eligibility(
    auxiliary_all,
    score_all,
    height,
    width,
    y,
    threshold,
    radius,
    count_limit,
    row_reconstruction_gate,
    floor_enabled,
    out_eligible,
):
    """streaming._decision_fallback_row + the row's eligibility gating.

    Decision fallback uses simple edge-replicated row/column clamps (not the
    zero/pseudo-row/repeat-last-row history boundary above) -- a distinct
    windowing scheme on the raw auxiliary plane, matching
    streaming._decision_fallback_row and _horizontal_low_count exactly.
    """

    if row_reconstruction_gate != 0:
        for x in range(width):
            out_eligible[x] = 0
        return

    for x in range(width):
        lx = x - radius
        if lx < 0:
            lx = 0
        elif lx >= width:
            lx = width - 1
        rx = x + radius
        if rx < 0:
            rx = 0
        elif rx >= width:
            rx = width - 1
        vertical_left = 0
        vertical_right = 0
        for dy in range(-4, 5):
            yy = y + dy
            if yy < 0:
                yy = 0
            elif yy >= height:
                yy = height - 1
            if auxiliary_all[yy, lx] < threshold:
                vertical_left += 1
            if auxiliary_all[yy, rx] < threshold:
                vertical_right += 1
        ay = y - radius
        if ay < 0:
            ay = 0
        by = y + radius
        if by >= height:
            by = height - 1
        horizontal_above = 0
        horizontal_below = 0
        for dx in range(-4, 5):
            xx = x + dx
            if xx < 0:
                xx = 0
            elif xx >= width:
                xx = width - 1
            if auxiliary_all[ay, xx] < threshold:
                horizontal_above += 1
            if auxiliary_all[by, xx] < threshold:
                horizontal_below += 1
        decision_fallback = (
            vertical_left > count_limit
            or horizontal_above > count_limit
            or vertical_right > count_limit
            or horizontal_below > count_limit
        )
        if decision_fallback:
            out_eligible[x] = 0
        elif floor_enabled and score_all[y, x] >= np.float32(1.0):
            out_eligible[x] = 0
        else:
            out_eligible[x] = 1


@njit(cache=True)
def process_row(
    auxiliary_all,
    score_all,
    weighted_auxiliary_all,
    weighted_rgb_all,
    working_all,
    y,
    height,
    width,
    score_floor,
    decision_threshold,
    decision_radius,
    decision_count_limit,
    floor_enabled,
    row_reconstruction_gate,
    fallback_value,
    record_count,
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
    """One entire output row: eligibility, then the fused per-pixel chain.

    ``record_count`` is 1 (CENTER_ONLY) or 5 (CROSS_NEIGHBOR, gathering the
    four adjacent feature records too).  Returns
    ``(working_output_row, attempted, written, rng_advances, new_state)``.
    """

    working_output = np.empty((width, 3), dtype=np.float32)
    for x in range(width):
        working_output[x, 0] = working_all[y, x, 0]
        working_output[x, 1] = working_all[y, x, 1]
        working_output[x, 2] = working_all[y, x, 2]

    eligible = np.empty(width, dtype=np.uint8)
    compute_row_eligibility(
        auxiliary_all,
        score_all,
        height,
        width,
        y,
        decision_threshold,
        decision_radius,
        decision_count_limit,
        row_reconstruction_gate,
        floor_enabled,
        eligible,
    )

    attempted = 0
    written = 0
    advances_total = np.int64(0)

    score_patch = np.empty((9, 9), dtype=np.float32)
    waux_patch = np.empty((9, 9), dtype=np.float32)
    wrgb_patch = np.empty((9, 9, 3), dtype=np.float32)
    features = np.zeros((5, 4), dtype=np.float32)
    w3 = np.empty(3, dtype=np.float32)
    original_rgb = np.empty(3, dtype=np.float32)

    for x in range(width):
        if eligible[x] == 0:
            continue

        gather_score_patch(score_all, height, width, score_floor, y, x, score_patch)
        gather_waux_patch(
            weighted_auxiliary_all, auxiliary_all, height, width, score_floor, y, x, waux_patch
        )
        weights, center_feature = feature_weights_and_feature_scalar(
            score_patch, waux_patch, auxiliary_all[y, x], score_all[y, x], fallback_value
        )
        if driver_forces_fallback_scalar(weights[3], row_reconstruction_gate, floor_enabled):
            continue
        attempted += 1

        for lane in range(4):
            features[0, lane] = center_feature[lane]

        if record_count == 5:
            if x > 0:
                gather_score_patch(score_all, height, width, score_floor, y, x - 1, score_patch)
                gather_waux_patch(
                    weighted_auxiliary_all,
                    auxiliary_all,
                    height,
                    width,
                    score_floor,
                    y,
                    x - 1,
                    waux_patch,
                )
                feature_n = neighbor_feature_scalar(
                    score_patch,
                    waux_patch,
                    auxiliary_all[y, x - 1],
                    score_all[y, x - 1],
                    fallback_value,
                    True,
                )
            else:
                feature_n = np.zeros(4, dtype=np.float32)
            for lane in range(4):
                features[1, lane] = feature_n[lane]

            if x < width - 1:
                gather_score_patch(score_all, height, width, score_floor, y, x + 1, score_patch)
                gather_waux_patch(
                    weighted_auxiliary_all,
                    auxiliary_all,
                    height,
                    width,
                    score_floor,
                    y,
                    x + 1,
                    waux_patch,
                )
                feature_n = neighbor_feature_scalar(
                    score_patch,
                    waux_patch,
                    auxiliary_all[y, x + 1],
                    score_all[y, x + 1],
                    fallback_value,
                    True,
                )
            else:
                feature_n = np.zeros(4, dtype=np.float32)
            for lane in range(4):
                features[2, lane] = feature_n[lane]

            up_row = y - 1
            gather_score_patch(score_all, height, width, score_floor, up_row, x, score_patch)
            gather_waux_patch(
                weighted_auxiliary_all,
                auxiliary_all,
                height,
                width,
                score_floor,
                up_row,
                x,
                waux_patch,
            )
            if up_row < 0:
                point_aux_u = auxiliary_all[0, x]
                point_score_u = score_floor
            else:
                point_aux_u = auxiliary_all[up_row, x]
                point_score_u = score_all[up_row, x]
            feature_n = neighbor_feature_scalar(
                score_patch, waux_patch, point_aux_u, point_score_u, fallback_value, True
            )
            for lane in range(4):
                features[3, lane] = feature_n[lane]

            down_row = y + 1
            gather_score_patch(score_all, height, width, score_floor, down_row, x, score_patch)
            gather_waux_patch(
                weighted_auxiliary_all,
                auxiliary_all,
                height,
                width,
                score_floor,
                down_row,
                x,
                waux_patch,
            )
            clipped_down = down_row if down_row < height else height - 1
            point_aux_d = auxiliary_all[clipped_down, x]
            point_score_d = score_all[clipped_down, x]
            feature_n = neighbor_feature_scalar(
                score_patch, waux_patch, point_aux_d, point_score_d, fallback_value, True
            )
            for lane in range(4):
                features[4, lane] = feature_n[lane]

        gather_wrgb_patch(weighted_rgb_all, working_all, height, width, score_floor, y, x, wrgb_patch)
        w3[0] = weights[0]
        w3[1] = weights[1]
        w3[2] = weights[2]
        candidates = rgb_candidates_scalar(wrgb_patch, w3)

        original_rgb[0] = working_all[y, x, 0]
        original_rgb[1] = working_all[y, x, 1]
        original_rgb[2] = working_all[y, x, 2]

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
        values, advances, state = write_pixel_scalar(
            combined, original_rgb, floor_enabled, low64, high64, low_lt_high, dither_scales, state
        )

        working_output[x, 0] = values[0]
        working_output[x, 1] = values[1]
        working_output[x, 2] = values[2]
        advances_total += advances
        if (
            values[0] != original_rgb[0]
            or values[1] != original_rgb[1]
            or values[2] != original_rgb[2]
        ):
            written += 1

    return working_output, attempted, written, advances_total, state
