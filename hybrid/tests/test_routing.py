from __future__ import annotations

import json

import numpy as np
import pytest
from scipy import ndimage

import fauxce_hybrid.routing as routing_module
from fauxce_hybrid.routing import (
    BoundingBox,
    NoHealthyContextError,
    RegionDisposition,
    RoutedReason,
    RoutingPolicy,
    SynthesisBudgetExceeded,
    route_at_floor_mask,
    routing_json_document,
    serialize_routing_json,
)


def _rectangle(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height + 4, width + 4), dtype=bool)
    mask[2 : 2 + height, 2 : 2 + width] = True
    return mask


@pytest.mark.parametrize(
    ("height", "width", "area", "radius", "reason"),
    (
        (3, 3, 9, 2, RoutedReason.NOT_ROUTED),
        (1, 400, 400, 1, RoutedReason.AREA),
        (9, 9, 81, 5, RoutedReason.CHESSBOARD_RADIUS),
        (30, 30, 900, 15, RoutedReason.AREA_AND_CHESSBOARD_RADIUS),
        (500, 12, 6_000, 6, RoutedReason.AREA_AND_CHESSBOARD_RADIUS),
    ),
)
def test_reason_attribution_for_reviewed_geometries(
    height: int,
    width: int,
    area: int,
    radius: int,
    reason: RoutedReason,
) -> None:
    result = route_at_floor_mask(
        _rectangle(height, width),
        RoutingPolicy(margin=0, max_synth_fraction=1.0),
    )

    assert len(result.provisional_regions) == 1
    region = result.provisional_regions[0]
    assert region.area == area
    assert region.max_chessboard_radius == radius
    assert region.routed_reason is reason
    if reason is RoutedReason.NOT_ROUTED:
        assert region.disposition is RegionDisposition.PRISTINE
        assert result.synthesis_pixel_count == 0
    else:
        assert region.disposition is RegionDisposition.ROUTED
        assert result.synthesis_pixel_count == area


def test_radius_boundaries_use_chessboard_distance() -> None:
    eight_wide = route_at_floor_mask(
        _rectangle(12, 8),
        RoutingPolicy(margin=0, max_synth_fraction=1.0),
    )
    ten_wide = route_at_floor_mask(
        _rectangle(12, 10),
        RoutingPolicy(margin=0, max_synth_fraction=1.0),
    )

    assert eight_wide.provisional_regions[0].max_chessboard_radius == 4
    assert eight_wide.provisional_regions[0].routed_reason is RoutedReason.NOT_ROUTED
    assert ten_wide.provisional_regions[0].max_chessboard_radius == 5
    assert (
        ten_wide.provisional_regions[0].routed_reason is RoutedReason.CHESSBOARD_RADIUS
    )


def test_diagonal_connectivity_does_not_inflate_radius_from_bbox() -> None:
    mask = np.zeros((34, 34), dtype=bool)
    indices = np.arange(2, 32)
    mask[indices, indices] = True

    result = route_at_floor_mask(
        mask,
        RoutingPolicy(margin=0, max_synth_fraction=1.0),
    )

    assert len(result.provisional_regions) == 1
    region = result.provisional_regions[0]
    assert region.bbox == BoundingBox(2, 2, 32, 32)
    assert region.area == 30
    assert region.max_chessboard_radius == 1
    assert region.routed_reason is RoutedReason.NOT_ROUTED


def test_one_pixel_square_dilation_and_margin_zero_branch() -> None:
    mask = np.zeros((21, 21), dtype=bool)
    mask[10, 10] = True
    policy = RoutingPolicy(
        min_area=1,
        min_radius=100,
        margin=4,
        max_synth_fraction=1.0,
    )

    dilated = route_at_floor_mask(mask, policy)
    expected = np.zeros_like(mask)
    expected[6:15, 6:15] = True
    np.testing.assert_array_equal(dilated.synthesis_mask, expected)
    assert dilated.final_regions[0].bbox == BoundingBox(6, 6, 15, 15)

    undilated = route_at_floor_mask(
        mask,
        RoutingPolicy(
            min_area=1,
            min_radius=100,
            margin=0,
            max_synth_fraction=1.0,
        ),
    )
    np.testing.assert_array_equal(undilated.synthesis_mask, mask)


def test_huge_margin_is_frame_clipped_without_allocating_a_footprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mask = np.zeros((2, 2), dtype=bool)
    mask[0, 0] = True
    requested_margin = 10**100

    def reject_footprint_allocation(*args: object, **kwargs: object) -> None:
        raise AssertionError("routing must not allocate a dilation footprint")

    monkeypatch.setattr(routing_module.np, "ones", reject_footprint_allocation)
    result = route_at_floor_mask(
        mask,
        RoutingPolicy(
            min_area=1,
            min_radius=100,
            margin=requested_margin,
            max_synth_fraction=1.0,
        ),
    )

    np.testing.assert_array_equal(result.synthesis_mask, np.full((2, 2), True))
    assert result.policy.margin == requested_margin
    assert routing_json_document(result)["policy"]["margin"] == requested_margin


def test_overlapping_halos_merge_and_partially_absorb_unrouted_region() -> None:
    mask = np.zeros((50, 70), dtype=bool)
    mask[10:19, 10:19] = True
    mask[10:19, 25:34] = True
    mask[22:25, 14:17] = True

    result = route_at_floor_mask(
        mask,
        RoutingPolicy(max_synth_fraction=1.0),
    )

    assert len(result.provisional_regions) == 3
    assert len(result.final_regions) == 1
    routed = tuple(
        region
        for region in result.provisional_regions
        if region.disposition is RegionDisposition.ROUTED
    )
    absorbed = tuple(
        region
        for region in result.provisional_regions
        if region.disposition is RegionDisposition.ABSORBED
    )
    assert len(routed) == 2
    assert len(absorbed) == 1
    assert all(region.directly_routed for region in routed)
    assert not absorbed[0].directly_routed
    assert absorbed[0].absorbed_pixel_count == 3
    assert absorbed[0].final_component_ids == (1,)
    assert result.final_regions[0].direct_region_ids == tuple(
        region.id for region in routed
    )
    assert result.final_regions[0].absorbed_region_ids == (absorbed[0].id,)

    routed_union = np.isin(
        result.provisional_labels,
        [region.id for region in routed],
    )
    expected = ndimage.binary_dilation(
        routed_union,
        structure=np.ones((9, 9), dtype=bool),
        iterations=1,
        border_value=0,
    )
    np.testing.assert_array_equal(result.synthesis_mask, expected)
    assert result.non_floor_synth_pixel_count == int(np.count_nonzero(expected & ~mask))
    assert result.final_regions[0].non_floor_synth_pixel_count == (
        result.non_floor_synth_pixel_count
    )
    assert result.synthesis_pixel_count == (
        result.directly_routed_pixel_count
        + absorbed[0].absorbed_pixel_count
        + result.non_floor_synth_pixel_count
    )


def test_absorbed_region_can_bind_two_disjoint_final_components() -> None:
    mask = np.zeros((40, 70), dtype=bool)
    mask[10:19, 10:19] = True
    mask[10:19, 40:49] = True
    mask[22, 22:37] = True

    result = route_at_floor_mask(
        mask,
        RoutingPolicy(max_synth_fraction=1.0),
    )

    assert len(result.final_regions) == 2
    bridge = next(
        region
        for region in result.provisional_regions
        if region.routed_reason is RoutedReason.NOT_ROUTED
    )
    assert bridge.disposition is RegionDisposition.ABSORBED
    assert bridge.absorbed_pixel_count == 2
    assert bridge.final_component_ids == (1, 2)
    assert all(bridge.id in final.absorbed_region_ids for final in result.final_regions)
    assert np.array_equal(result.final_labels > 0, result.synthesis_mask)
    assert sum(region.area for region in result.final_regions) == (
        result.synthesis_pixel_count
    )


def test_edge_touching_dilation_is_clipped_to_frame() -> None:
    mask = np.zeros((30, 30), dtype=bool)
    mask[:9, :9] = True

    result = route_at_floor_mask(
        mask,
        RoutingPolicy(max_synth_fraction=1.0),
    )

    expected = np.zeros_like(mask)
    expected[:13, :13] = True
    np.testing.assert_array_equal(result.synthesis_mask, expected)
    assert result.final_regions[0].bbox == BoundingBox(0, 0, 13, 13)
    assert result.provisional_regions[0].disposition is RegionDisposition.ROUTED
    assert result.perimeter_excluded_region_count == 0
    # Unpadded CDT does not pretend that pixels outside the image are healthy.
    assert result.provisional_regions[0].max_chessboard_radius == 9


def test_full_perimeter_component_is_measured_but_never_synthesized() -> None:
    mask = np.zeros((30, 40), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True

    result = route_at_floor_mask(
        mask,
        RoutingPolicy(
            min_area=1,
            min_radius=100,
            max_synth_fraction=1.0,
        ),
    )

    assert len(result.provisional_regions) == 1
    perimeter = result.provisional_regions[0]
    assert perimeter.routed_reason is RoutedReason.AREA
    assert perimeter.disposition is RegionDisposition.PERIMETER_EXCLUDED
    assert not perimeter.directly_routed
    assert perimeter.final_component_ids == ()
    assert result.perimeter_excluded_region_count == 1
    assert result.perimeter_excluded_pixel_count == int(np.count_nonzero(mask))
    assert result.perimeter_suppressed_halo_pixel_count == 0
    assert result.directly_routed_pixel_count == 0
    assert result.synthesis_pixel_count == 0
    assert result.final_regions == ()
    assert not np.any(result.final_labels)

    document = routing_json_document(result)
    assert document["counts"]["perimeter_excluded_regions"] == 1
    assert document["counts"]["perimeter_excluded_pixels"] == int(
        np.count_nonzero(mask)
    )
    assert document["perimeter_excluded_region"] == {
        "id": perimeter.id,
        "bbox_yxyx_half_open": [0, 0, 30, 40],
        "area": int(np.count_nonzero(mask)),
        "max_chessboard_radius": perimeter.max_chessboard_radius,
        "routed_reason_by_thresholds": "area",
        "perimeter_pixel_count": 136,
        "predicate": "contains_every_frame_perimeter_pixel",
        "content_interpretation": "unverified",
    }
    assert serialize_routing_json(result) == serialize_routing_json(
        route_at_floor_mask(
            mask.copy(),
            RoutingPolicy(
                min_area=1,
                min_radius=100,
                max_synth_fraction=1.0,
            ),
        )
    )


def test_missing_perimeter_pixel_restores_normal_routing() -> None:
    mask = np.zeros((30, 40), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    mask[0, 20] = False

    result = route_at_floor_mask(
        mask,
        RoutingPolicy(
            min_area=1,
            min_radius=100,
            margin=0,
            max_synth_fraction=1.0,
        ),
    )

    assert len(result.provisional_regions) == 1
    assert result.provisional_regions[0].disposition is RegionDisposition.ROUTED
    assert result.perimeter_excluded_region_count == 0
    assert result.perimeter_excluded_pixel_count == 0
    np.testing.assert_array_equal(result.synthesis_mask, mask)


def test_routed_halo_never_synthesizes_perimeter_excluded_pixels() -> None:
    mask = np.zeros((30, 40), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    mask[2:11, 15:24] = True

    result = route_at_floor_mask(
        mask,
        RoutingPolicy(max_synth_fraction=1.0),
    )

    excluded = next(
        region
        for region in result.provisional_regions
        if region.disposition is RegionDisposition.PERIMETER_EXCLUDED
    )
    routed = next(
        region
        for region in result.provisional_regions
        if region.disposition is RegionDisposition.ROUTED
    )
    routed_union = result.provisional_labels == routed.id
    raw_dilation = ndimage.binary_dilation(
        routed_union,
        structure=np.ones((9, 9), dtype=bool),
        iterations=1,
        border_value=0,
    )
    excluded_pixels = result.provisional_labels == excluded.id
    expected_suppressed = int(np.count_nonzero(raw_dilation & excluded_pixels))
    assert expected_suppressed > 0
    assert result.perimeter_suppressed_halo_pixel_count == expected_suppressed
    np.testing.assert_array_equal(
        result.synthesis_mask,
        raw_dilation & ~excluded_pixels,
    )
    assert not np.any(result.synthesis_mask & excluded_pixels)
    assert excluded.final_component_ids == ()
    assert routed.final_component_ids == (1,)
    assert result.final_regions[0].direct_region_ids == (routed.id,)
    assert (
        routing_json_document(result)["counts"]["perimeter_suppressed_halo_pixels"]
        == expected_suppressed
    )


def test_region_connected_to_full_perimeter_is_conservatively_excluded() -> None:
    mask = np.zeros((30, 40), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    mask[1:10, 15:24] = True

    result = route_at_floor_mask(
        mask,
        RoutingPolicy(max_synth_fraction=1.0),
    )

    assert len(result.provisional_regions) == 1
    assert (
        result.provisional_regions[0].disposition
        is RegionDisposition.PERIMETER_EXCLUDED
    )
    assert result.synthesis_pixel_count == 0
    assert result.within_synthesis_budget


def test_near_all_true_mask_is_reported_as_excluded_not_synthesized() -> None:
    mask = np.ones((11, 13), dtype=bool)
    mask[5, 6] = False

    result = route_at_floor_mask(mask)

    assert len(result.provisional_regions) == 1
    assert (
        result.provisional_regions[0].disposition
        is RegionDisposition.PERIMETER_EXCLUDED
    )
    assert result.perimeter_excluded_pixel_count == mask.size - 1
    assert result.synthesis_pixel_count == 0
    assert result.within_synthesis_budget


def test_final_ids_are_normalized_in_bbox_order() -> None:
    mask = np.zeros((50, 50), dtype=bool)
    mask[30:39, 35:44] = True
    mask[4:13, 5:14] = True

    result = route_at_floor_mask(
        mask,
        RoutingPolicy(margin=0, max_synth_fraction=1.0),
    )

    assert [region.id for region in result.final_regions] == [1, 2]
    assert [region.bbox for region in result.final_regions] == [
        BoundingBox(4, 5, 13, 14),
        BoundingBox(30, 35, 39, 44),
    ]
    assert int(result.final_labels[4, 5]) == 1
    assert int(result.final_labels[30, 35]) == 2


@pytest.mark.parametrize("shape", ((1, 1), (1, 9), (6, 9)))
def test_all_true_mask_fails_closed_with_finite_radius_sentinel(
    shape: tuple[int, int],
) -> None:
    mask = np.ones(shape, dtype=bool)

    with pytest.raises(NoHealthyContextError, match="no healthy in-frame") as raised:
        route_at_floor_mask(mask)
    assert raised.value.image_shape == mask.shape
    assert raised.value.finite_radius_sentinel == max(shape)


def test_budget_boundary_empty_routing_and_guard() -> None:
    exactly_two_percent = np.zeros((10, 10), dtype=bool)
    exactly_two_percent[1, 1] = True
    exactly_two_percent[8, 8] = True
    within = route_at_floor_mask(
        exactly_two_percent,
        RoutingPolicy(
            min_area=1,
            min_radius=100,
            margin=0,
        ),
    )
    assert within.synthesis_fraction == 0.02
    assert within.within_synthesis_budget
    within.require_safe_for_inpainting()

    over = exactly_two_percent.copy()
    over[5, 5] = True
    exceeded = route_at_floor_mask(
        over,
        RoutingPolicy(
            min_area=1,
            min_radius=100,
            margin=0,
        ),
    )
    assert not exceeded.within_synthesis_budget
    with pytest.raises(SynthesisBudgetExceeded, match="3/100 pixels"):
        exceeded.require_safe_for_inpainting()

    empty = route_at_floor_mask(np.zeros((10, 10), dtype=bool))
    assert empty.provisional_regions == ()
    assert empty.final_regions == ()
    assert empty.synthesis_pixel_count == 0
    assert empty.within_synthesis_budget
    empty.require_safe_for_inpainting()

    zero_budget_empty = route_at_floor_mask(
        np.zeros((10, 10), dtype=bool),
        RoutingPolicy(max_synth_fraction=0.0),
    )
    assert zero_budget_empty.within_synthesis_budget
    zero_budget_nonempty = route_at_floor_mask(
        exactly_two_percent,
        RoutingPolicy(
            min_area=1,
            min_radius=100,
            margin=0,
            max_synth_fraction=0.0,
        ),
    )
    assert not zero_budget_nonempty.within_synthesis_budget

    one_of_forty_nine = np.zeros((7, 7), dtype=bool)
    one_of_forty_nine[3, 3] = True
    rationally_over = route_at_floor_mask(
        one_of_forty_nine,
        RoutingPolicy(
            min_area=1,
            min_radius=100,
            margin=0,
        ),
    )
    assert rationally_over.synthesis_fraction == 1 / 49
    assert not rationally_over.within_synthesis_budget


def test_arrays_and_serialization_are_read_only_and_deterministic() -> None:
    mask = np.zeros((40, 40), dtype=bool)
    mask[4:13, 6:15] = True
    mask[27:30, 28:31] = True

    first = route_at_floor_mask(
        mask,
        RoutingPolicy(max_synth_fraction=1.0),
    )
    second = route_at_floor_mask(
        mask.copy(),
        RoutingPolicy(max_synth_fraction=1.0),
    )

    assert first.provisional_regions == second.provisional_regions
    assert first.final_regions == second.final_regions
    for first_array, second_array, dtype in (
        (first.provisional_labels, second.provisional_labels, np.dtype(np.int32)),
        (first.synthesis_mask, second.synthesis_mask, np.dtype(np.bool_)),
        (first.final_labels, second.final_labels, np.dtype(np.int32)),
    ):
        np.testing.assert_array_equal(first_array, second_array)
        assert first_array.dtype == dtype
        assert first_array.flags.c_contiguous
        assert not first_array.flags.writeable
        with pytest.raises(ValueError):
            first_array.flat[0] = first_array.flat[0]

    first_bytes = serialize_routing_json(first)
    assert first_bytes == serialize_routing_json(second)
    document = json.loads(first_bytes)
    assert document == routing_json_document(first)
    assert document["schema"] == "fauxce-hybrid-routing-v1"
    assert document["policy"]["connectivity"] == 8
    assert document["policy"]["distance_border_semantics"] == (
        "unpadded_in_frame_background"
    )


@pytest.mark.parametrize(
    "invalid_mask",
    (
        np.zeros((4, 4), dtype=np.uint8),
        np.zeros(4, dtype=bool),
        np.zeros((2, 2, 1), dtype=bool),
        np.zeros((0, 4), dtype=bool),
        np.zeros((4, 0), dtype=bool),
    ),
)
def test_invalid_mask_contract_is_rejected(invalid_mask: np.ndarray) -> None:
    expected = TypeError if invalid_mask.dtype != np.dtype(np.bool_) else ValueError
    with pytest.raises(expected):
        route_at_floor_mask(invalid_mask)


@pytest.mark.parametrize(
    "arguments",
    (
        {"min_area": 0},
        {"min_area": True},
        {"min_radius": 0},
        {"margin": -1},
        {"max_synth_fraction": -0.01},
        {"max_synth_fraction": 1.01},
        {"max_synth_fraction": float("nan")},
        {"max_synth_fraction": True},
    ),
)
def test_invalid_policy_is_rejected(arguments: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        RoutingPolicy(**arguments)


def test_non_policy_object_is_rejected() -> None:
    with pytest.raises(TypeError, match="RoutingPolicy"):
        route_at_floor_mask(np.zeros((4, 4), dtype=bool), policy=object())
