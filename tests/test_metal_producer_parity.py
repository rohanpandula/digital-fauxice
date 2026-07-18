"""Metal producer schedule must reproduce every 0x40-byte record exactly."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("Metal", reason="Metal backend requires pyobjc-framework-Metal")

from portable_digital_ice.metal_backend.engine import (  # noqa: E402
    MetalBackendUnavailable,
    metal_device_summary,
)
from portable_digital_ice.metal_backend.producer import (  # noqa: E402
    derive_producer_record_schedule_metal,
)
from portable_digital_ice.producer_parameters import (  # noqa: E402
    derive_producer_record_schedule,
)
from portable_digital_ice.x3a import SharedLookupInputResponse  # noqa: E402


def _require_device() -> None:
    try:
        metal_device_summary()
    except MetalBackendUnavailable as error:
        pytest.skip(f"Metal device unavailable: {error}")


@pytest.mark.parametrize(
    "height,width,seed,description",
    [
        (16, 32, 1, "two complete epochs"),
        (41, 64, 2, "partial final block, epoch boundaries"),
        (30, 48, 3, "six-valid-row final block"),
        (9, 24, 4, "one epoch plus one row"),
        (26, 41, 5, "trailing transport column outside whole cell"),
    ],
)
def test_producer_records_byte_exact(height, width, seed, description):
    _require_device()
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 65536, size=(height, width, 4), dtype=np.uint16)
    # dense threshold straddles: cell invalidation and epoch-block rejection
    pixels[:, :, 3] = rng.choice(
        np.array([500, 8846, 8847, 8848, 8849, 30000, 65535], dtype=np.uint16),
        size=(height, width),
    )
    table = SharedLookupInputResponse.nikon_logarithmic().table

    reference = derive_producer_record_schedule(pixels, table)
    metal = derive_producer_record_schedule_metal(pixels, table)

    assert metal.mean_schedule.active_width == reference.mean_schedule.active_width
    assert np.array_equal(
        metal.mean_schedule.accepted_samples,
        reference.mean_schedule.accepted_samples,
    ), description
    assert np.array_equal(
        metal.mean_schedule.weighted_visible_sum.view(np.uint64),
        reference.mean_schedule.weighted_visible_sum.view(np.uint64),
    ), description
    assert np.array_equal(
        metal.mean_schedule.weighted_infrared_sum.view(np.uint64),
        reference.mean_schedule.weighted_infrared_sum.view(np.uint64),
    ), description
    assert np.array_equal(
        metal.mean_schedule.weight_sum.view(np.uint64),
        reference.mean_schedule.weight_sum.view(np.uint64),
    ), description
    assert np.array_equal(
        metal.scale_add_denominator.view(np.uint32),
        reference.scale_add_denominator.view(np.uint32),
    ), description
    assert np.array_equal(
        metal.scale_add_numerator.view(np.uint32),
        reference.scale_add_numerator.view(np.uint32),
    ), description
    assert metal.record_payloads == reference.record_payloads, description
