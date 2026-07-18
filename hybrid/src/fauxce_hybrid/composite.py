"""Deterministic, mask-exact compositing for generative repair crops.

The functions in this module deliberately keep model execution behind a small
callback.  Everything before and after that callback is exact, reviewable
array logic: context-only range normalization, component-only feathering, and
structural preservation of every pixel outside the synthesis mask.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import islice
from typing import Final, Protocol

import numpy as np
import numpy.typing as npt
from scipy import ndimage

from .routing import BoundingBox


UInt8RgbImage = npt.NDArray[np.uint8]
UInt16RgbImage = npt.NDArray[np.uint16]
UInt8Mask = npt.NDArray[np.uint8]
BoolImage = npt.NDArray[np.bool_]
LabelImage = npt.NDArray[np.int32]

_CONNECTIVITY_8 = np.ones((3, 3), dtype=np.bool_)
_FEATHER_RADIUS = 3
MAX_COMPONENTS_PER_RUN: Final = 4_096
MAX_TOTAL_CROP_PIXELS: Final = 16_777_216


class Inpainter(Protocol):
    """A crop-local RGB inpainter used by :func:`composite_components`."""

    def __call__(
        self,
        rgb_crop: UInt8RgbImage,
        component_mask: UInt8Mask,
    ) -> npt.ArrayLike:
        """Return an RGB uint8 crop with the same shape as ``rgb_crop``."""


class BatchInpainter(Protocol):
    """An optional ordered batch interface for crop-local RGB inpainting."""

    def inpaint_batch(
        self,
        rgb_crops: Sequence[UInt8RgbImage],
        component_masks: Sequence[UInt8Mask],
    ) -> Sequence[npt.ArrayLike]:
        """Return one RGB uint8 crop for every ordered input crop."""


class NoContextPixelsError(RuntimeError):
    """Raised before model execution when a crop has no healthy context."""

    def __init__(self, component_id: int, crop_bbox: BoundingBox) -> None:
        self.component_id = component_id
        self.crop_bbox = crop_bbox
        super().__init__(
            f"component {component_id} crop has no pixels outside the global "
            "synthesis mask"
        )


class CompositeResourceLimitError(ValueError):
    """Raised before crop preparation when bounded work limits are exceeded."""


@dataclass(frozen=True)
class ChannelRange:
    """One channel's context-derived uint16 normalization interval."""

    channel: int
    lo: int
    hi: int
    degenerate: bool


@dataclass(frozen=True)
class ComponentCompositeRecord:
    """Review and provenance fields for one inpainter invocation."""

    component_id: int
    component_bbox: BoundingBox
    crop_bbox: BoundingBox
    pixel_count: int
    context_pixel_count: int
    channel_ranges: tuple[ChannelRange, ChannelRange, ChannelRange]
    alpha_min: float
    alpha_max: float
    input_rgb8_sha256: str
    component_mask_sha256: str
    inpainted_rgb8_sha256: str
    decoded_rgb16_sha256: str
    blended_component_rgb16_sha256: str


@dataclass(frozen=True)
class CompositeResult:
    """A read-only hybrid image and deterministic component records."""

    hybrid_rgb16: UInt16RgbImage
    components: tuple[ComponentCompositeRecord, ...]
    pure_rgb16_sha256: str
    synthesis_mask_sha256: str
    hybrid_rgb16_sha256: str

    def __post_init__(self) -> None:
        image = self.hybrid_rgb16
        if image.dtype != np.dtype(np.uint16) or image.ndim != 3:
            raise ValueError("hybrid_rgb16 must be an HxWx3 uint16 array")
        if image.shape[2] != 3 or not image.flags.c_contiguous:
            raise ValueError("hybrid_rgb16 must be a C-contiguous HxWx3 array")
        if image.flags.writeable:
            raise ValueError("hybrid_rgb16 must be read-only")


@dataclass(frozen=True)
class _ComponentPlan:
    component_id: int
    component_bbox: BoundingBox
    crop_bbox: BoundingBox
    pixel_count: int
    context_pixel_count: int
    lows: tuple[int, int, int]
    highs: tuple[int, int, int]


@dataclass(frozen=True)
class _ComponentGeometry:
    component_id: int
    component_bbox: BoundingBox
    crop_bbox: BoundingBox
    pixel_count: int


@dataclass(frozen=True)
class _PreparedComponent:
    plan: _ComponentPlan
    input_rgb8: UInt8RgbImage
    component_mask: UInt8Mask
    alpha: npt.NDArray[np.float64]
    bbox_component: BoolImage
    input_rgb8_sha256: str
    component_mask_sha256: str


def _sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def _validate_rgb_image(
    image: npt.ArrayLike,
    *,
    dtype: np.dtype,
    name: str,
) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"{name} must have shape HxWx3")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} cannot have an empty dimension")
    return array


def _validate_ranges(
    lows: tuple[int, int, int],
    highs: tuple[int, int, int],
) -> None:
    if len(lows) != 3 or len(highs) != 3:
        raise ValueError("lows and highs must contain exactly three values")
    for channel, (lo, hi) in enumerate(zip(lows, highs, strict=True)):
        if isinstance(lo, bool) or not isinstance(lo, (int, np.integer)):
            raise TypeError(f"lows[{channel}] must be an integer")
        if isinstance(hi, bool) or not isinstance(hi, (int, np.integer)):
            raise TypeError(f"highs[{channel}] must be an integer")
        if not 0 <= int(lo) <= int(hi) <= 65_535:
            raise ValueError(
                f"channel {channel} range must satisfy 0 <= lo <= hi <= 65535"
            )


def affine_encode_uint8(
    image_rgb16: npt.ArrayLike,
    lows: tuple[int, int, int],
    highs: tuple[int, int, int],
) -> UInt8RgbImage:
    """Encode uint16 RGB with explicit context-derived channel intervals.

    Rounding is round-half-up.  Values outside each interval are clipped, and
    a degenerate interval encodes to zero exactly.
    """

    image = _validate_rgb_image(
        image_rgb16,
        dtype=np.dtype(np.uint16),
        name="image_rgb16",
    )
    _validate_ranges(lows, highs)
    encoded = np.empty(image.shape, dtype=np.uint8)
    values = image.astype(np.float64, copy=False)
    for channel, (lo_value, hi_value) in enumerate(zip(lows, highs, strict=True)):
        lo = int(lo_value)
        hi = int(hi_value)
        if lo == hi:
            encoded[..., channel] = 0
            continue
        scaled = (values[..., channel] - lo) * 255.0 / (hi - lo)
        rounded = np.floor(scaled + 0.5)
        encoded[..., channel] = np.clip(rounded, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(encoded)


def affine_decode_uint16(
    image_rgb8: npt.ArrayLike,
    lows: tuple[int, int, int],
    highs: tuple[int, int, int],
) -> UInt16RgbImage:
    """Decode uint8 RGB to uint16 using exact, monotonic endpoints."""

    image = _validate_rgb_image(
        image_rgb8,
        dtype=np.dtype(np.uint8),
        name="image_rgb8",
    )
    _validate_ranges(lows, highs)
    decoded = np.empty(image.shape, dtype=np.uint16)
    values = image.astype(np.float64, copy=False)
    for channel, (lo_value, hi_value) in enumerate(zip(lows, highs, strict=True)):
        lo = int(lo_value)
        hi = int(hi_value)
        restored = lo + values[..., channel] * (hi - lo) / 255.0
        rounded = np.floor(restored + 0.5)
        decoded[..., channel] = np.clip(rounded, 0.0, 65_535.0).astype(np.uint16)
    return np.ascontiguousarray(decoded)


def _bbox_from_slice(component_slice: tuple[slice, slice] | None) -> BoundingBox:
    if component_slice is None:
        raise RuntimeError("component label has no pixels")
    rows, columns = component_slice
    if (
        rows.start is None
        or rows.stop is None
        or columns.start is None
        or columns.stop is None
    ):
        raise RuntimeError("component label has no finite bounding box")
    return BoundingBox(
        int(rows.start),
        int(columns.start),
        int(rows.stop),
        int(columns.stop),
    )


def _expanded_bbox(
    bbox: BoundingBox,
    *,
    margin: int,
    image_shape: tuple[int, int],
) -> BoundingBox:
    height, width = image_shape
    return BoundingBox(
        max(0, bbox.y0 - margin),
        max(0, bbox.x0 - margin),
        min(height, bbox.y1 + margin),
        min(width, bbox.x1 + margin),
    )


def _slices(bbox: BoundingBox) -> tuple[slice, slice]:
    return slice(bbox.y0, bbox.y1), slice(bbox.x0, bbox.x1)


def _component_alpha(
    labels: LabelImage,
    *,
    component_id: int,
    component_bbox: BoundingBox,
) -> tuple[npt.NDArray[np.float64], BoolImage]:
    # A three-pixel halo is sufficient because the distance is capped at three.
    # Clipping the halo at the image boundary preserves the required unpadded
    # semantics: outside-frame coordinates are not invented as background.
    alpha_bbox = _expanded_bbox(
        component_bbox,
        margin=_FEATHER_RADIUS,
        image_shape=labels.shape,
    )
    alpha_slices = _slices(alpha_bbox)
    local_component = np.ascontiguousarray(
        labels[alpha_slices] == component_id,
        dtype=np.bool_,
    )
    distances = ndimage.distance_transform_cdt(
        local_component,
        metric="chessboard",
    )
    component_slices = _slices(component_bbox)
    y0 = component_bbox.y0 - alpha_bbox.y0
    x0 = component_bbox.x0 - alpha_bbox.x0
    y1 = y0 + (component_bbox.y1 - component_bbox.y0)
    x1 = x0 + (component_bbox.x1 - component_bbox.x0)
    bbox_component = labels[component_slices] == component_id
    bbox_distances = distances[y0:y1, x0:x1]
    component_distances = np.asarray(
        bbox_distances[bbox_component],
        dtype=np.float64,
    )
    if component_distances.size == 0 or np.any(component_distances < 1):
        raise RuntimeError("component distance transform violated mask geometry")
    alpha = np.minimum(component_distances, _FEATHER_RADIUS) / float(_FEATHER_RADIUS)
    return alpha, np.ascontiguousarray(bbox_component, dtype=np.bool_)


def _validate_and_snapshot_inputs(
    pure_fauxce_rgb16: npt.ArrayLike,
    final_labels: npt.ArrayLike,
    synthesis_mask: npt.ArrayLike,
) -> tuple[UInt16RgbImage, LabelImage, BoolImage, tuple[int, ...]]:
    pure = _validate_rgb_image(
        pure_fauxce_rgb16,
        dtype=np.dtype(np.uint16),
        name="pure_fauxce_rgb16",
    )
    labels = np.asarray(final_labels)
    mask = np.asarray(synthesis_mask)
    image_shape = (int(pure.shape[0]), int(pure.shape[1]))
    if labels.dtype != np.dtype(np.int32):
        raise TypeError("final_labels must have dtype int32")
    if labels.shape != image_shape:
        raise ValueError("final_labels must have shape HxW matching the image")
    if mask.dtype != np.dtype(np.bool_):
        raise TypeError("synthesis_mask must have dtype bool")
    if mask.shape != image_shape:
        raise ValueError("synthesis_mask must have shape HxW matching the image")
    if np.any(labels < 0):
        raise ValueError("final_labels cannot contain negative values")
    if not np.array_equal(labels != 0, mask):
        raise ValueError(
            "synthesis_mask must be exactly equivalent to final_labels != 0"
        )

    source = np.array(pure, dtype=np.uint16, order="C", copy=True)
    label_snapshot = np.array(labels, dtype=np.int32, order="C", copy=True)
    mask_snapshot = np.array(mask, dtype=np.bool_, order="C", copy=True)
    source.setflags(write=False)
    label_snapshot.setflags(write=False)
    mask_snapshot.setflags(write=False)
    component_ids = tuple(
        int(value) for value in np.unique(label_snapshot[label_snapshot > 0])
    )
    return source, label_snapshot, mask_snapshot, component_ids


def _build_plans(
    source: UInt16RgbImage,
    labels: LabelImage,
    synthesis_mask: BoolImage,
    *,
    component_ids: tuple[int, ...],
    crop_margin: int,
) -> tuple[_ComponentPlan, ...]:
    if len(component_ids) > MAX_COMPONENTS_PER_RUN:
        raise CompositeResourceLimitError(
            "component count exceeds the bounded composite limit: "
            f"{len(component_ids)} > {MAX_COMPONENTS_PER_RUN}"
        )
    plans: list[_ComponentPlan] = []
    dense_labels = np.zeros(labels.shape, dtype=np.int32)
    if component_ids:
        positive = labels > 0
        sorted_ids = np.asarray(component_ids, dtype=np.int32)
        dense_labels[positive] = (
            np.searchsorted(sorted_ids, labels[positive]).astype(np.int32) + 1
        )
    component_slices = ndimage.find_objects(
        dense_labels,
        max_label=len(component_ids),
    )
    geometries: list[_ComponentGeometry] = []
    total_crop_pixels = 0
    for dense_id, component_id in enumerate(component_ids, start=1):
        if dense_id > len(component_slices):
            raise RuntimeError(f"component label {component_id} has no bounding box")
        component_bbox = _bbox_from_slice(component_slices[dense_id - 1])
        bbox_slices = _slices(component_bbox)
        local_component = labels[bbox_slices] == component_id
        connected_count = int(
            ndimage.label(local_component, structure=_CONNECTIVITY_8)[1]
        )
        if connected_count != 1:
            raise ValueError(
                f"final label {component_id} must identify one 8-connected component"
            )
        crop_bbox = _expanded_bbox(
            component_bbox,
            margin=crop_margin,
            image_shape=labels.shape,
        )
        crop_pixels = (crop_bbox.y1 - crop_bbox.y0) * (crop_bbox.x1 - crop_bbox.x0)
        if crop_pixels > MAX_TOTAL_CROP_PIXELS - total_crop_pixels:
            raise CompositeResourceLimitError(
                "total planned crop pixels exceed the bounded composite limit: "
                f"> {MAX_TOTAL_CROP_PIXELS}"
            )
        total_crop_pixels += crop_pixels
        geometries.append(
            _ComponentGeometry(
                component_id=component_id,
                component_bbox=component_bbox,
                crop_bbox=crop_bbox,
                pixel_count=int(np.count_nonzero(local_component)),
            )
        )

    for geometry in geometries:
        component_id = geometry.component_id
        component_bbox = geometry.component_bbox
        crop_bbox = geometry.crop_bbox
        crop_slices = _slices(crop_bbox)
        context = ~synthesis_mask[crop_slices]
        context_pixel_count = int(np.count_nonzero(context))
        if context_pixel_count == 0:
            raise NoContextPixelsError(component_id, crop_bbox)
        context_values = source[crop_slices][context]
        lows_array = np.min(context_values, axis=0)
        highs_array = np.max(context_values, axis=0)
        lows = tuple(int(value) for value in lows_array)
        highs = tuple(int(value) for value in highs_array)
        plans.append(
            _ComponentPlan(
                component_id=component_id,
                component_bbox=component_bbox,
                crop_bbox=crop_bbox,
                pixel_count=geometry.pixel_count,
                context_pixel_count=context_pixel_count,
                lows=(lows[0], lows[1], lows[2]),
                highs=(highs[0], highs[1], highs[2]),
            )
        )
    return tuple(plans)


def _prepare_components(
    source: UInt16RgbImage,
    labels: LabelImage,
    plans: tuple[_ComponentPlan, ...],
) -> tuple[_PreparedComponent, ...]:
    """Snapshot every model input and deterministic blend input up front."""

    prepared: list[_PreparedComponent] = []
    for plan in plans:
        crop_slices = _slices(plan.crop_bbox)
        input_rgb8 = affine_encode_uint8(
            source[crop_slices],
            plan.lows,
            plan.highs,
        )
        local_component = labels[crop_slices] == plan.component_id
        component_mask = np.zeros(local_component.shape, dtype=np.uint8)
        component_mask[local_component] = 255
        expected_crop_shape = (
            plan.crop_bbox.y1 - plan.crop_bbox.y0,
            plan.crop_bbox.x1 - plan.crop_bbox.x0,
        )
        if input_rgb8.shape != (*expected_crop_shape, 3):
            raise RuntimeError(
                f"component {plan.component_id} encoded crop geometry is invalid"
            )
        if component_mask.shape != expected_crop_shape:
            raise RuntimeError(
                f"component {plan.component_id} model mask geometry is invalid"
            )
        if int(np.count_nonzero(component_mask)) != plan.pixel_count:
            raise RuntimeError(
                f"component {plan.component_id} model mask pixel count is invalid"
            )
        if np.any((component_mask != 0) & (component_mask != 255)):
            raise RuntimeError(
                f"component {plan.component_id} model mask is not binary"
            )
        alpha, bbox_component = _component_alpha(
            labels,
            component_id=plan.component_id,
            component_bbox=plan.component_bbox,
        )
        input_rgb8.setflags(write=False)
        component_mask.setflags(write=False)
        alpha.setflags(write=False)
        bbox_component.setflags(write=False)
        prepared.append(
            _PreparedComponent(
                plan=plan,
                input_rgb8=input_rgb8,
                component_mask=component_mask,
                alpha=alpha,
                bbox_component=bbox_component,
                input_rgb8_sha256=_sha256(input_rgb8),
                component_mask_sha256=_sha256(component_mask),
            )
        )
    return tuple(prepared)


def _snapshot_inpainted_output(
    value: npt.ArrayLike,
    item: _PreparedComponent,
) -> UInt8RgbImage:
    plan = item.plan
    output = np.asarray(value)
    if output.dtype != np.dtype(np.uint8):
        raise TypeError(
            f"inpainter result for component {plan.component_id} must have dtype uint8"
        )
    expected_shape = (
        plan.crop_bbox.y1 - plan.crop_bbox.y0,
        plan.crop_bbox.x1 - plan.crop_bbox.x0,
        3,
    )
    if output.shape != expected_shape:
        raise ValueError(
            f"inpainter result for component {plan.component_id} must "
            f"have shape {expected_shape}"
        )
    return np.array(output, dtype=np.uint8, order="C", copy=True)


def _snapshot_batch_outputs(
    values: Sequence[npt.ArrayLike],
    prepared: tuple[_PreparedComponent, ...],
) -> tuple[UInt8RgbImage, ...]:
    if len(values) != len(prepared):
        raise ValueError(
            f"batch inpainter returned {len(values)} results for "
            f"{len(prepared)} components"
        )
    return tuple(
        _snapshot_inpainted_output(value, item)
        for value, item in zip(values, prepared, strict=True)
    )


def _invoke_inpainter(
    inpainter: Inpainter | BatchInpainter,
    prepared: tuple[_PreparedComponent, ...],
    *,
    batch_method: object,
) -> tuple[UInt8RgbImage, ...]:
    if not prepared:
        return ()
    # The canonical preflight snapshots remain immutable provenance inputs.
    # Writable working copies preserve the legacy callback contract without
    # allowing a mutating model to falsify the recorded input or mask hashes.
    rgb_inputs = tuple(
        np.array(item.input_rgb8, dtype=np.uint8, order="C", copy=True)
        for item in prepared
    )
    mask_inputs = tuple(
        np.array(item.component_mask, dtype=np.uint8, order="C", copy=True)
        for item in prepared
    )
    if callable(batch_method):
        batch_values = batch_method(rgb_inputs, mask_inputs)
        try:
            batch_iterator = iter(batch_values)
        except TypeError as error:
            raise TypeError("batch inpainter result must be an iterable") from error
        # A model boundary is untrusted: an oversized or non-terminating
        # iterator must not make the compositor exhaust arbitrary output.  One
        # item beyond the expected cardinality is sufficient to reject it.
        values = tuple(islice(batch_iterator, len(prepared) + 1))
        return _snapshot_batch_outputs(values, prepared)

    if not callable(inpainter):
        raise TypeError("inpainter must be callable or expose callable inpaint_batch")
    outputs: list[UInt8RgbImage] = []
    for item, rgb_input, mask_input in zip(
        prepared,
        rgb_inputs,
        mask_inputs,
        strict=True,
    ):
        value = inpainter(rgb_input, mask_input)
        outputs.append(_snapshot_inpainted_output(value, item))
    return tuple(outputs)


def composite_components(
    pure_fauxce_rgb16: npt.ArrayLike,
    final_labels: npt.ArrayLike,
    synthesis_mask: npt.ArrayLike,
    inpainter: Inpainter | BatchInpainter,
    *,
    crop_margin: int = 128,
) -> CompositeResult:
    """Inpaint and feather disjoint labeled components into a pure ICE image.

    All component plans, encoded crops, masks, and blend geometry are validated
    before the first model call.  If ``inpainter`` exposes a callable
    ``inpaint_batch`` method, every non-empty component is sent through that
    method in numeric component order with one call.  Otherwise the legacy
    scalar callback is invoked once per component in the same order.
    """

    if isinstance(crop_margin, bool) or not isinstance(crop_margin, int):
        raise TypeError("crop_margin must be an integer")
    if crop_margin < 0:
        raise ValueError("crop_margin must be >= 0")
    batch_method = getattr(inpainter, "inpaint_batch", None)
    if not callable(inpainter) and not callable(batch_method):
        raise TypeError("inpainter must be callable or expose callable inpaint_batch")

    source, labels, mask, component_ids = _validate_and_snapshot_inputs(
        pure_fauxce_rgb16,
        final_labels,
        synthesis_mask,
    )
    plans = _build_plans(
        source,
        labels,
        mask,
        component_ids=component_ids,
        crop_margin=crop_margin,
    )
    prepared = _prepare_components(source, labels, plans)
    inpainted_outputs = _invoke_inpainter(
        inpainter,
        prepared,
        batch_method=batch_method,
    )
    hybrid = np.array(source, dtype=np.uint16, order="C", copy=True)
    records: list[ComponentCompositeRecord] = []

    for item, inpainted in zip(prepared, inpainted_outputs, strict=True):
        plan = item.plan
        decoded = affine_decode_uint16(
            inpainted,
            plan.lows,
            plan.highs,
        )

        alpha = item.alpha
        bbox_component = item.bbox_component
        component_slices = _slices(plan.component_bbox)
        source_values = source[component_slices][bbox_component]
        crop_y0 = plan.component_bbox.y0 - plan.crop_bbox.y0
        crop_x0 = plan.component_bbox.x0 - plan.crop_bbox.x0
        crop_y1 = crop_y0 + (plan.component_bbox.y1 - plan.component_bbox.y0)
        crop_x1 = crop_x0 + (plan.component_bbox.x1 - plan.component_bbox.x0)
        generated_values = decoded[crop_y0:crop_y1, crop_x0:crop_x1][bbox_component]
        blended_float = (
            source_values.astype(np.float64) * (1.0 - alpha[:, None])
            + generated_values.astype(np.float64) * alpha[:, None]
        )
        blended_values = np.clip(
            np.floor(blended_float + 0.5),
            0.0,
            65_535.0,
        ).astype(np.uint16)
        hybrid_view = hybrid[component_slices]
        hybrid_view[bbox_component] = blended_values

        ranges = tuple(
            ChannelRange(
                channel=channel,
                lo=lo,
                hi=hi,
                degenerate=lo == hi,
            )
            for channel, (lo, hi) in enumerate(zip(plan.lows, plan.highs, strict=True))
        )
        records.append(
            ComponentCompositeRecord(
                component_id=plan.component_id,
                component_bbox=plan.component_bbox,
                crop_bbox=plan.crop_bbox,
                pixel_count=plan.pixel_count,
                context_pixel_count=plan.context_pixel_count,
                channel_ranges=(ranges[0], ranges[1], ranges[2]),
                alpha_min=float(np.min(alpha)),
                alpha_max=float(np.max(alpha)),
                input_rgb8_sha256=item.input_rgb8_sha256,
                component_mask_sha256=item.component_mask_sha256,
                inpainted_rgb8_sha256=_sha256(inpainted),
                decoded_rgb16_sha256=_sha256(decoded),
                blended_component_rgb16_sha256=_sha256(blended_values),
            )
        )

    # Structural indexing, followed by an assertion, prevents any crop or model
    # behavior from altering bytes outside the global synthesis mask.
    if not np.array_equal(hybrid[~mask], source[~mask]):
        raise RuntimeError("pixels outside synthesis_mask were modified")
    hybrid = np.ascontiguousarray(hybrid, dtype=np.uint16)
    hybrid.setflags(write=False)
    return CompositeResult(
        hybrid_rgb16=hybrid,
        components=tuple(records),
        pure_rgb16_sha256=_sha256(source),
        synthesis_mask_sha256=_sha256(mask),
        hybrid_rgb16_sha256=_sha256(hybrid),
    )


__all__ = [
    "BatchInpainter",
    "ChannelRange",
    "CompositeResourceLimitError",
    "ComponentCompositeRecord",
    "CompositeResult",
    "Inpainter",
    "MAX_COMPONENTS_PER_RUN",
    "MAX_TOTAL_CROP_PIXELS",
    "NoContextPixelsError",
    "affine_decode_uint16",
    "affine_encode_uint8",
    "composite_components",
]
