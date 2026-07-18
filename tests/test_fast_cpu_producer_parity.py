"""Compiled producer schedule vs the reference: byte-exact on every field.

The reference `derive_producer_record_schedule` pins a strictly ordered
binary64 accumulation, a data-dependent cell-validity latch, and exact
reduction trees per scale epoch.  The compiled port must reproduce every
schedule array, every 0x40-byte record payload, and both binding hashes
bit-for-bit; comparisons use raw byte views, never tolerances.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("numba")

from portable_digital_ice.producer_parameters import (  # noqa: E402
    derive_producer_record_schedule,
)
from portable_digital_ice.fast_cpu.producer import (  # noqa: E402
    derive_producer_record_schedule_fast,
)
from portable_digital_ice.x3a import SharedLookupInputResponse  # noqa: E402


def _assert_schedules_equal(reference, fast) -> None:
    ref_means = reference.mean_schedule
    fast_means = fast.mean_schedule
    for field in ("weighted_visible_sum", "weighted_infrared_sum", "weight_sum"):
        assert np.array_equal(
            getattr(ref_means, field).view(np.uint64),
            getattr(fast_means, field).view(np.uint64),
        ), field
    for field in ("calibration_mean_visible", "calibration_mean_infrared"):
        assert np.array_equal(
            getattr(ref_means, field).view(np.uint32),
            getattr(fast_means, field).view(np.uint32),
        ), field
    assert np.array_equal(ref_means.accepted_samples, fast_means.accepted_samples)
    assert ref_means.active_width == fast_means.active_width
    assert ref_means.infrared_threshold == fast_means.infrared_threshold

    for field in (
        "scaled_input",
        "input_scale",
        "scale_denominator",
        "scale_numerator",
        "scale_add_denominator",
        "scale_add_numerator",
    ):
        assert np.array_equal(
            getattr(reference, field).view(np.uint32),
            getattr(fast, field).view(np.uint32),
        ), field
    assert reference.record_payloads == fast.record_payloads
    assert reference.scale_multiplier == fast.scale_multiplier
    assert reference.source_rgbi_sha256 == fast.source_rgbi_sha256
    assert reference.response_lut_sha256 == fast.response_lut_sha256


@pytest.mark.parametrize(
    ("height", "width", "seed", "description"),
    [
        (48, 64, 20260716, "self-test geometry with dust blob"),
        (41, 64, 11, "five epochs plus one remainder row"),
        (30, 40, 22, "final block has six valid rows"),
        (8, 8, 33, "exactly one epoch, one cell"),
        (7, 16, 44, "no complete epoch at all"),
        (9, 17, 55, "one epoch plus a row; trailing column outside cells"),
        (12, 24, 66, "backend-contract geometry"),
        (25, 33, 77, "cells straddling the trailing partial column"),
    ],
)
def test_producer_records_byte_exact(height, width, seed, description):
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 65536, size=(height, width, 4), dtype=np.uint16)
    # Force threshold-straddling infrared content: the acceptance threshold is
    # 8847.2255859375, so 8847 rejects and 8848 accepts.
    straddle = rng.random(size=(height, width)) < 0.25
    pixels[:, :, 3][straddle] = rng.choice(
        np.array([8846, 8847, 8848, 8849], dtype=np.uint16),
        size=int(straddle.sum()),
    )
    # A dark blob so entire cells invalidate mid-block.
    pixels[height // 3 : height // 3 + 3, : min(width, 12), 3] = 500

    table = SharedLookupInputResponse.nikon_logarithmic().table
    reference = derive_producer_record_schedule(pixels, table)
    fast = derive_producer_record_schedule_fast(pixels, table)
    _assert_schedules_equal(reference, fast)


def test_producer_all_below_threshold_keeps_initial_means():
    """Zero accepted weight leaves the initial means untouched, exactly."""

    pixels = np.full((16, 16, 4), 30000, dtype=np.uint16)
    pixels[:, :, 3] = 100  # everything below the acceptance threshold
    table = SharedLookupInputResponse.nikon_logarithmic().table
    reference = derive_producer_record_schedule(pixels, table)
    fast = derive_producer_record_schedule_fast(pixels, table)
    _assert_schedules_equal(reference, fast)
    assert np.all(fast.mean_schedule.accepted_samples == 0)


def test_producer_uniform_high_infrared():
    """Every sample accepted; epoch ratios exercise the full-weight branch."""

    rng = np.random.default_rng(88)
    pixels = rng.integers(40000, 65535, size=(24, 32, 4), dtype=np.uint16)
    pixels[:, :, 3] = rng.integers(50000, 65535, size=(24, 32), dtype=np.uint16)
    table = SharedLookupInputResponse.nikon_logarithmic().table
    reference = derive_producer_record_schedule(pixels, table)
    fast = derive_producer_record_schedule_fast(pixels, table)
    _assert_schedules_equal(reference, fast)


def test_producer_custom_active_width():
    """A caller-supplied active width (whole cells) must match the reference."""

    rng = np.random.default_rng(99)
    pixels = rng.integers(15000, 64000, size=(17, 40, 4), dtype=np.uint16)
    pixels[5:9, 10:19, 3] = 900
    table = SharedLookupInputResponse.nikon_logarithmic().table
    reference = derive_producer_record_schedule(pixels, table, active_width=24)
    fast = derive_producer_record_schedule_fast(pixels, table, active_width=24)
    _assert_schedules_equal(reference, fast)
