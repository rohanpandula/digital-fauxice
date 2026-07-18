"""Deterministic routing of at-floor evidence to synthesis components."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction

import numpy as np
import numpy.typing as npt
from scipy import ndimage


BoolImage = npt.NDArray[np.bool_]
LabelImage = npt.NDArray[np.int32]
_CONNECTIVITY_8 = np.ones((3, 3), dtype=np.bool_)


class RoutedReason(StrEnum):
    """The threshold predicate or predicates that routed a region."""

    NOT_ROUTED = "not_routed"
    AREA = "area"
    CHESSBOARD_RADIUS = "chessboard_radius"
    AREA_AND_CHESSBOARD_RADIUS = "area+chessboard_radius"


class RegionDisposition(StrEnum):
    """How one provisional at-floor region relates to synthesis."""

    ROUTED = "routed"
    ABSORBED = "absorbed"
    PRISTINE = "pristine"
    PERIMETER_EXCLUDED = "perimeter_excluded"


class SynthesisBudgetExceeded(RuntimeError):
    """Raised before inpainting when the routed mask exceeds its budget."""

    def __init__(
        self,
        *,
        synthesis_pixel_count: int,
        frame_pixel_count: int,
        synthesis_fraction: float,
        maximum_fraction: float,
    ) -> None:
        self.synthesis_pixel_count = synthesis_pixel_count
        self.frame_pixel_count = frame_pixel_count
        self.synthesis_fraction = synthesis_fraction
        self.maximum_fraction = maximum_fraction
        super().__init__(
            "synthesis mask exceeds configured budget: "
            f"{synthesis_pixel_count}/{frame_pixel_count} pixels "
            f"({synthesis_fraction:.6%}) > {maximum_fraction:.6%}"
        )


class NoHealthyContextError(RuntimeError):
    """Raised when an all-true frame provides no routing context."""

    def __init__(self, image_shape: tuple[int, int]) -> None:
        self.image_shape = image_shape
        self.finite_radius_sentinel = max(image_shape)
        super().__init__(
            "at_floor_mask covers the entire frame; no healthy in-frame "
            "context exists for bounded synthesis"
        )


def _validate_int(value: object, *, name: str, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if minimum == 1 else f">= {minimum}"
        raise ValueError(f"{name} must be a {qualifier} integer")


@dataclass(frozen=True)
class RoutingPolicy:
    """Provisional routing thresholds and fail-closed synthesis budget."""

    min_area: int = 400
    min_radius: int = 5
    margin: int = 4
    max_synth_fraction: float = 0.02

    def __post_init__(self) -> None:
        _validate_int(self.min_area, name="min_area", minimum=1)
        _validate_int(self.min_radius, name="min_radius", minimum=1)
        _validate_int(self.margin, name="margin", minimum=0)
        if (
            isinstance(self.max_synth_fraction, bool)
            or not isinstance(self.max_synth_fraction, (int, float))
            or not math.isfinite(float(self.max_synth_fraction))
            or not 0.0 <= float(self.max_synth_fraction) <= 1.0
        ):
            raise ValueError("max_synth_fraction must be finite and in [0, 1]")
        object.__setattr__(
            self,
            "max_synth_fraction",
            float(self.max_synth_fraction),
        )


@dataclass(frozen=True, order=True)
class BoundingBox:
    """A half-open ``[y0:y1, x0:x1]`` image bounding box."""

    y0: int
    x0: int
    y1: int
    x1: int

    def __post_init__(self) -> None:
        for name, value in (
            ("y0", self.y0),
            ("x0", self.x0),
            ("y1", self.y1),
            ("x1", self.x1),
        ):
            _validate_int(value, name=name, minimum=0)
        if self.y1 <= self.y0 or self.x1 <= self.x0:
            raise ValueError("bounding box must have positive half-open geometry")

    def as_list(self) -> list[int]:
        return [self.y0, self.x0, self.y1, self.x1]


@dataclass(frozen=True)
class ProvisionalRegion:
    """One 8-connected component from the input at-floor mask."""

    id: int
    bbox: BoundingBox
    area: int
    max_chessboard_radius: int
    routed_reason: RoutedReason
    disposition: RegionDisposition
    final_component_ids: tuple[int, ...]
    absorbed_pixel_count: int

    @property
    def directly_routed(self) -> bool:
        return self.disposition is RegionDisposition.ROUTED


@dataclass(frozen=True)
class FinalSynthesisRegion:
    """One disjoint component of the dilated routed union."""

    id: int
    bbox: BoundingBox
    area: int
    direct_region_ids: tuple[int, ...]
    absorbed_region_ids: tuple[int, ...]
    non_floor_synth_pixel_count: int


def _validate_read_only_c_array(
    array: np.ndarray,
    *,
    name: str,
    dtype: np.dtype,
    shape: tuple[int, int],
) -> None:
    if array.dtype != dtype or array.shape != shape:
        raise ValueError(f"{name} must have dtype {dtype} and shape {shape}")
    if not array.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if array.flags.writeable:
        raise ValueError(f"{name} must be read-only")


@dataclass(frozen=True)
class RoutingResult:
    """Immutable region tables and planes from one routing decision."""

    image_shape: tuple[int, int]
    policy: RoutingPolicy
    provisional_regions: tuple[ProvisionalRegion, ...]
    final_regions: tuple[FinalSynthesisRegion, ...]
    provisional_labels: LabelImage
    synthesis_mask: BoolImage
    final_labels: LabelImage
    at_floor_pixel_count: int
    directly_routed_pixel_count: int
    perimeter_excluded_region_count: int
    perimeter_excluded_pixel_count: int
    perimeter_suppressed_halo_pixel_count: int
    synthesis_pixel_count: int
    non_floor_synth_pixel_count: int
    synthesis_fraction: float
    within_synthesis_budget: bool

    def __post_init__(self) -> None:
        _validate_read_only_c_array(
            self.provisional_labels,
            name="provisional_labels",
            dtype=np.dtype(np.int32),
            shape=self.image_shape,
        )
        _validate_read_only_c_array(
            self.synthesis_mask,
            name="synthesis_mask",
            dtype=np.dtype(np.bool_),
            shape=self.image_shape,
        )
        _validate_read_only_c_array(
            self.final_labels,
            name="final_labels",
            dtype=np.dtype(np.int32),
            shape=self.image_shape,
        )
        if not np.array_equal(self.final_labels > 0, self.synthesis_mask):
            raise ValueError("positive final labels must equal synthesis_mask")

    def require_safe_for_inpainting(self) -> None:
        """Fail before any model load when the synthesis budget is exceeded."""

        if not self.within_synthesis_budget:
            raise SynthesisBudgetExceeded(
                synthesis_pixel_count=self.synthesis_pixel_count,
                frame_pixel_count=self.image_shape[0] * self.image_shape[1],
                synthesis_fraction=self.synthesis_fraction,
                maximum_fraction=self.policy.max_synth_fraction,
            )


def _bbox_from_slice(region_slice: tuple[slice, slice] | None) -> BoundingBox:
    if region_slice is None:
        raise RuntimeError("labeled component has no bounding box")
    y_slice, x_slice = region_slice
    if (
        y_slice.start is None
        or y_slice.stop is None
        or x_slice.start is None
        or x_slice.stop is None
    ):
        raise RuntimeError("component bounding box is not finite")
    return BoundingBox(y_slice.start, x_slice.start, y_slice.stop, x_slice.stop)


def _component_radii(
    mask: BoolImage,
    labels: LabelImage,
    component_count: int,
) -> npt.NDArray[np.int64]:
    if component_count == 0:
        return np.empty(0, dtype=np.int64)
    distances = ndimage.distance_transform_cdt(mask, metric="chessboard")
    label_ids = np.arange(1, component_count + 1, dtype=np.int32)
    radii = ndimage.maximum(distances, labels=labels, index=label_ids)
    return np.asarray(radii, dtype=np.int64).reshape(component_count)


def _routed_reason(
    *,
    area: int,
    radius: int,
    policy: RoutingPolicy,
) -> RoutedReason:
    area_hit = area >= policy.min_area
    radius_hit = radius >= policy.min_radius
    if area_hit and radius_hit:
        return RoutedReason.AREA_AND_CHESSBOARD_RADIUS
    if area_hit:
        return RoutedReason.AREA
    if radius_hit:
        return RoutedReason.CHESSBOARD_RADIUS
    return RoutedReason.NOT_ROUTED


def _full_perimeter_component_id(labels: LabelImage) -> int | None:
    """Return the component containing every frame-perimeter pixel, if any."""

    candidate = int(labels[0, 0])
    if candidate == 0:
        return None
    if (
        np.all(labels[0, :] == candidate)
        and np.all(labels[-1, :] == candidate)
        and np.all(labels[:, 0] == candidate)
        and np.all(labels[:, -1] == candidate)
    ):
        return candidate
    return None


def _within_synthesis_budget(
    synthesis_pixel_count: int,
    frame_pixel_count: int,
    maximum_fraction: float,
) -> bool:
    limit = Fraction(str(maximum_fraction))
    return (
        synthesis_pixel_count * limit.denominator <= frame_pixel_count * limit.numerator
    )


def _square_dilation_clipped_to_frame(
    mask: BoolImage,
    *,
    margin: int,
) -> BoolImage:
    """Dilate by a square radius without allocating a quadratic footprint.

    A square Chebyshev dilation is separable into vertical and horizontal
    maximum filters.  Radius beyond an axis' last in-frame pixel is
    geometrically redundant, so clipping each filter radius preserves the
    requested policy's exact in-frame result while bounding filter sizes by
    the image geometry.
    """

    height, width = mask.shape
    vertical_radius = min(margin, height - 1)
    horizontal_radius = min(margin, width - 1)
    vertically_dilated = ndimage.maximum_filter1d(
        mask,
        size=2 * vertical_radius + 1,
        axis=0,
        mode="constant",
        cval=0,
    )
    return np.ascontiguousarray(
        ndimage.maximum_filter1d(
            vertically_dilated,
            size=2 * horizontal_radius + 1,
            axis=1,
            mode="constant",
            cval=0,
        ),
        dtype=np.bool_,
    )


def _normalized_final_labels(
    synthesis_mask: BoolImage,
) -> tuple[LabelImage, tuple[BoundingBox, ...]]:
    raw_labels = np.empty(synthesis_mask.shape, dtype=np.int32)
    component_count = int(
        ndimage.label(
            synthesis_mask,
            structure=_CONNECTIVITY_8,
            output=raw_labels,
        )
    )
    if component_count == 0:
        return raw_labels, ()

    raw_boxes = tuple(
        _bbox_from_slice(region_slice)
        for region_slice in ndimage.find_objects(
            raw_labels,
            max_label=component_count,
        )
    )
    ordered_raw_ids = sorted(
        range(1, component_count + 1),
        key=lambda raw_id: (*raw_boxes[raw_id - 1].as_list(), raw_id),
    )
    old_to_new = np.zeros(component_count + 1, dtype=np.int32)
    for new_id, raw_id in enumerate(ordered_raw_ids, start=1):
        old_to_new[raw_id] = new_id
    normalized = np.ascontiguousarray(old_to_new[raw_labels], dtype=np.int32)
    normalized_boxes = tuple(raw_boxes[raw_id - 1] for raw_id in ordered_raw_ids)
    return normalized, normalized_boxes


def route_at_floor_mask(
    at_floor_mask: npt.ArrayLike,
    policy: RoutingPolicy = RoutingPolicy(),
) -> RoutingResult:
    """Measure and route an at-floor mask without invoking an inpainter."""

    if not isinstance(policy, RoutingPolicy):
        raise TypeError("policy must be a RoutingPolicy")
    mask = np.asarray(at_floor_mask)
    if mask.dtype != np.dtype(np.bool_):
        raise TypeError("at_floor_mask must have dtype bool")
    if mask.ndim != 2:
        raise ValueError("at_floor_mask must have shape HxW")
    if mask.shape[0] == 0 or mask.shape[1] == 0:
        raise ValueError("at_floor_mask cannot have an empty dimension")
    mask = np.ascontiguousarray(mask, dtype=np.bool_)
    height, width = (int(mask.shape[0]), int(mask.shape[1]))
    if bool(np.all(mask)):
        raise NoHealthyContextError((height, width))

    provisional_labels = np.empty(mask.shape, dtype=np.int32)
    component_count = int(
        ndimage.label(
            mask,
            structure=_CONNECTIVITY_8,
            output=provisional_labels,
        )
    )
    counts = np.bincount(
        provisional_labels.reshape(-1),
        minlength=component_count + 1,
    )
    areas = np.asarray(counts[1:], dtype=np.int64)
    radii = _component_radii(mask, provisional_labels, component_count)
    provisional_boxes = tuple(
        _bbox_from_slice(region_slice)
        for region_slice in ndimage.find_objects(
            provisional_labels,
            max_label=component_count,
        )
    )
    reasons = tuple(
        _routed_reason(area=int(area), radius=int(radius), policy=policy)
        for area, radius in zip(areas, radii, strict=True)
    )
    perimeter_excluded_id = _full_perimeter_component_id(provisional_labels)
    directly_routed = tuple(
        reason is not RoutedReason.NOT_ROUTED and region_id != perimeter_excluded_id
        for region_id, reason in enumerate(reasons, start=1)
    )

    routed_lookup = np.zeros(component_count + 1, dtype=np.bool_)
    if component_count:
        routed_lookup[1:] = np.fromiter(
            directly_routed,
            dtype=np.bool_,
            count=component_count,
        )
    routed_union = routed_lookup[provisional_labels]
    if policy.margin == 0:
        synthesis_mask = np.array(
            routed_union,
            dtype=np.bool_,
            order="C",
            copy=True,
        )
    else:
        synthesis_mask = _square_dilation_clipped_to_frame(
            routed_union,
            margin=policy.margin,
        )
    perimeter_suppressed_halo_pixel_count = 0
    if perimeter_excluded_id is not None:
        perimeter_pixels = provisional_labels == perimeter_excluded_id
        perimeter_suppressed_halo_pixel_count = int(
            np.count_nonzero(synthesis_mask & perimeter_pixels)
        )
        synthesis_mask[perimeter_pixels] = False

    final_labels, final_boxes = _normalized_final_labels(synthesis_mask)
    final_count = len(final_boxes)

    # Build provisional/final overlap accounting in one frame scan.  A
    # per-component ``provisional_labels == id`` scan is quadratic in practice
    # on real frames with tens of thousands of components.
    synthesized_pixels = final_labels > 0
    overlap_provisional = provisional_labels[synthesized_pixels]
    overlap_final = final_labels[synthesized_pixels]
    at_floor_overlap = overlap_provisional > 0
    overlap_counts = np.bincount(
        overlap_provisional[at_floor_overlap],
        minlength=component_count + 1,
    )
    final_at_floor_counts = np.bincount(
        overlap_final[at_floor_overlap],
        minlength=final_count + 1,
    )
    final_ids_by_provisional: list[list[int]] = [[] for _ in range(component_count)]
    direct_ids: list[list[int]] = [[] for _ in range(final_count)]
    absorbed_ids: list[list[int]] = [[] for _ in range(final_count)]
    if np.any(at_floor_overlap):
        pair_base = final_count + 1
        pair_codes = np.unique(
            overlap_provisional[at_floor_overlap].astype(np.int64) * pair_base
            + overlap_final[at_floor_overlap].astype(np.int64)
        )
        for pair_code in pair_codes:
            provisional_id, final_id = divmod(int(pair_code), pair_base)
            if provisional_id == perimeter_excluded_id:
                raise RuntimeError("perimeter-excluded region entered synthesis mask")
            final_ids_by_provisional[provisional_id - 1].append(final_id)
            if directly_routed[provisional_id - 1]:
                direct_ids[final_id - 1].append(provisional_id)
            else:
                absorbed_ids[final_id - 1].append(provisional_id)

    region_rows: list[ProvisionalRegion] = []
    for index, (bbox, area, radius, reason, is_directly_routed) in enumerate(
        zip(
            provisional_boxes,
            areas,
            radii,
            reasons,
            directly_routed,
            strict=True,
        ),
        start=1,
    ):
        final_component_ids = tuple(final_ids_by_provisional[index - 1])
        synthesized_pixel_count = int(overlap_counts[index])
        if index == perimeter_excluded_id:
            if synthesized_pixel_count:
                raise RuntimeError("perimeter-excluded region entered synthesis mask")
            disposition = RegionDisposition.PERIMETER_EXCLUDED
            absorbed_pixel_count = 0
        elif is_directly_routed:
            if len(final_component_ids) != 1:
                raise RuntimeError("directly routed region lost its final binding")
            disposition = RegionDisposition.ROUTED
            absorbed_pixel_count = 0
        elif synthesized_pixel_count:
            disposition = RegionDisposition.ABSORBED
            absorbed_pixel_count = synthesized_pixel_count
        else:
            disposition = RegionDisposition.PRISTINE
            absorbed_pixel_count = 0
        region_rows.append(
            ProvisionalRegion(
                id=index,
                bbox=bbox,
                area=int(area),
                max_chessboard_radius=int(radius),
                routed_reason=reason,
                disposition=disposition,
                final_component_ids=final_component_ids,
                absorbed_pixel_count=absorbed_pixel_count,
            )
        )

    final_counts = np.bincount(
        final_labels.reshape(-1),
        minlength=final_count + 1,
    )
    non_floor_counts = final_counts - final_at_floor_counts
    final_rows = tuple(
        FinalSynthesisRegion(
            id=final_id,
            bbox=final_boxes[final_id - 1],
            area=int(final_counts[final_id]),
            direct_region_ids=tuple(direct_ids[final_id - 1]),
            absorbed_region_ids=tuple(absorbed_ids[final_id - 1]),
            non_floor_synth_pixel_count=int(non_floor_counts[final_id]),
        )
        for final_id in range(1, final_count + 1)
    )

    at_floor_pixel_count = int(np.count_nonzero(mask))
    directly_routed_pixel_count = int(
        sum(
            region.area
            for region in region_rows
            if region.disposition is RegionDisposition.ROUTED
        )
    )
    perimeter_excluded_region_count = int(perimeter_excluded_id is not None)
    perimeter_excluded_pixel_count = (
        int(areas[perimeter_excluded_id - 1])
        if perimeter_excluded_id is not None
        else 0
    )
    synthesis_pixel_count = int(np.count_nonzero(synthesis_mask))
    non_floor_synth_pixel_count = int(np.count_nonzero(synthesis_mask & ~mask))
    synthesis_fraction = synthesis_pixel_count / (height * width)
    within_synthesis_budget = _within_synthesis_budget(
        synthesis_pixel_count,
        height * width,
        policy.max_synth_fraction,
    )
    absorbed_pixel_count = sum(
        region.absorbed_pixel_count
        for region in region_rows
        if region.disposition is RegionDisposition.ABSORBED
    )
    if synthesis_pixel_count != (
        directly_routed_pixel_count + absorbed_pixel_count + non_floor_synth_pixel_count
    ):
        raise RuntimeError("synthesis pixel accounting is not conservative")
    if sum(region.area for region in final_rows) != synthesis_pixel_count:
        raise RuntimeError("final region areas do not equal synthesis pixel count")

    provisional_labels = np.ascontiguousarray(provisional_labels, dtype=np.int32)
    final_labels = np.ascontiguousarray(final_labels, dtype=np.int32)
    synthesis_mask = np.ascontiguousarray(synthesis_mask, dtype=np.bool_)
    provisional_labels.setflags(write=False)
    synthesis_mask.setflags(write=False)
    final_labels.setflags(write=False)

    return RoutingResult(
        image_shape=(height, width),
        policy=policy,
        provisional_regions=tuple(region_rows),
        final_regions=final_rows,
        provisional_labels=provisional_labels,
        synthesis_mask=synthesis_mask,
        final_labels=final_labels,
        at_floor_pixel_count=at_floor_pixel_count,
        directly_routed_pixel_count=directly_routed_pixel_count,
        perimeter_excluded_region_count=perimeter_excluded_region_count,
        perimeter_excluded_pixel_count=perimeter_excluded_pixel_count,
        perimeter_suppressed_halo_pixel_count=(perimeter_suppressed_halo_pixel_count),
        synthesis_pixel_count=synthesis_pixel_count,
        non_floor_synth_pixel_count=non_floor_synth_pixel_count,
        synthesis_fraction=synthesis_fraction,
        within_synthesis_budget=within_synthesis_budget,
    )


def routing_json_document(result: RoutingResult) -> dict[str, object]:
    """Build the versioned JSON-safe routing census document."""

    if not isinstance(result, RoutingResult):
        raise TypeError("result must be a RoutingResult")
    perimeter_excluded = next(
        (
            region
            for region in result.provisional_regions
            if region.disposition is RegionDisposition.PERIMETER_EXCLUDED
        ),
        None,
    )
    height, width = result.image_shape
    perimeter_pixel_count = (
        width if height == 1 else height if width == 1 else 2 * (height + width) - 4
    )
    return {
        "schema": "fauxce-hybrid-routing-v1",
        "image_shape": [*result.image_shape],
        "policy": {
            "connectivity": 8,
            "distance_metric": "chessboard",
            "distance_border_semantics": "unpadded_in_frame_background",
            "min_area": result.policy.min_area,
            "min_radius": result.policy.min_radius,
            "margin": result.policy.margin,
            "dilation": "single_square_chebyshev_footprint",
            "max_synth_fraction": result.policy.max_synth_fraction,
            "thresholds_provisional": True,
            "perimeter_rule": "exclude_component_covering_full_perimeter",
            "final_id_sort_key": ["y0", "x0", "y1", "x1", "raw_id"],
        },
        "counts": {
            "frame_pixels": result.image_shape[0] * result.image_shape[1],
            "at_floor_pixels": result.at_floor_pixel_count,
            "directly_routed_pixels": result.directly_routed_pixel_count,
            "perimeter_excluded_regions": result.perimeter_excluded_region_count,
            "perimeter_excluded_pixels": result.perimeter_excluded_pixel_count,
            "perimeter_suppressed_halo_pixels": (
                result.perimeter_suppressed_halo_pixel_count
            ),
            "provisional_regions": len(result.provisional_regions),
            "final_regions": len(result.final_regions),
            "synthesis_pixels": result.synthesis_pixel_count,
            "non_floor_synthesis_pixels": result.non_floor_synth_pixel_count,
        },
        "synthesis_fraction": result.synthesis_fraction,
        "within_synthesis_budget": result.within_synthesis_budget,
        "perimeter_excluded_region": (
            None
            if perimeter_excluded is None
            else {
                "id": perimeter_excluded.id,
                "bbox_yxyx_half_open": perimeter_excluded.bbox.as_list(),
                "area": perimeter_excluded.area,
                "max_chessboard_radius": (perimeter_excluded.max_chessboard_radius),
                "routed_reason_by_thresholds": (perimeter_excluded.routed_reason.value),
                "perimeter_pixel_count": perimeter_pixel_count,
                "predicate": "contains_every_frame_perimeter_pixel",
                "content_interpretation": "unverified",
            }
        ),
        "provisional_regions": [
            {
                "id": region.id,
                "bbox_yxyx_half_open": region.bbox.as_list(),
                "area": region.area,
                "max_chessboard_radius": region.max_chessboard_radius,
                "routed_reason": region.routed_reason.value,
                "disposition": region.disposition.value,
                "final_component_ids": [*region.final_component_ids],
                "absorbed_pixel_count": region.absorbed_pixel_count,
            }
            for region in result.provisional_regions
        ],
        "final_regions": [
            {
                "id": region.id,
                "bbox_yxyx_half_open": region.bbox.as_list(),
                "area": region.area,
                "direct_region_ids": [*region.direct_region_ids],
                "absorbed_region_ids": [*region.absorbed_region_ids],
                "non_floor_synthesis_pixels": (region.non_floor_synth_pixel_count),
            }
            for region in result.final_regions
        ],
    }


def serialize_routing_json(result: RoutingResult) -> bytes:
    """Serialize a routing census to canonical deterministic UTF-8 JSON."""

    payload = json.dumps(
        routing_json_document(result),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return payload.encode("utf-8") + b"\n"


__all__ = [
    "BoundingBox",
    "FinalSynthesisRegion",
    "NoHealthyContextError",
    "ProvisionalRegion",
    "RegionDisposition",
    "RoutedReason",
    "RoutingPolicy",
    "RoutingResult",
    "SynthesisBudgetExceeded",
    "route_at_floor_mask",
    "routing_json_document",
    "serialize_routing_json",
]
