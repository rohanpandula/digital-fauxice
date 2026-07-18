from __future__ import annotations

import hashlib

import numpy as np
import pytest

import fauxce_hybrid.composite as composite_module
from fauxce_hybrid.composite import (
    MAX_COMPONENTS_PER_RUN,
    CompositeResourceLimitError,
    NoContextPixelsError,
    affine_decode_uint16,
    affine_encode_uint8,
    composite_components,
)
from fauxce_hybrid.routing import BoundingBox


def _constant_rgb(
    shape: tuple[int, int],
    value: int = 0,
) -> np.ndarray:
    return np.full((*shape, 3), value, dtype=np.uint16)


def _raw_hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _multiple_component_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, columns = np.indices((14, 18))
    source = np.stack(
        (
            rows * 500 + columns * 13 + 1_000,
            rows * 300 + columns * 29 + 2_000,
            rows * 100 + columns * 47 + 3_000,
        ),
        axis=2,
    ).astype(np.uint16)
    labels = np.zeros((14, 18), dtype=np.int32)
    # Numeric order intentionally opposes spatial order.  Different component
    # sizes make the order visible in the batch masks as six then eight pixels.
    labels[8:11, 2:4] = 2
    labels[2:4, 11:15] = 9
    return source, labels, labels != 0


def test_affine_rounding_clipping_degenerate_range_and_inverse_monotonicity() -> None:
    values = np.array(
        [
            [[0, 0, 65_535], [1_000, 2_000, 1_234]],
            [[31_000, 30_000, 1_234], [61_000, 65_535, 0]],
        ],
        dtype=np.uint16,
    )
    lows = (1_000, 2_000, 1_234)
    highs = (61_000, 62_000, 1_234)

    encoded = affine_encode_uint8(values, lows, highs)

    expected_red = np.clip(
        np.floor((values[..., 0].astype(np.float64) - 1_000) * 255 / 60_000 + 0.5),
        0,
        255,
    ).astype(np.uint8)
    np.testing.assert_array_equal(encoded[..., 0], expected_red)
    assert encoded[0, 0, 0] == 0
    assert encoded[1, 1, 0] == 255
    assert np.all(encoded[..., 2] == 0)

    ramp = np.arange(256, dtype=np.uint8)[None, :, None]
    ramp = np.repeat(ramp, 3, axis=2)
    decoded = affine_decode_uint16(ramp, lows, highs)
    assert tuple(int(value) for value in decoded[0, 0]) == lows
    assert tuple(int(value) for value in decoded[0, -1]) == highs
    assert np.all(np.diff(decoded[..., 0], axis=1) >= 0)
    assert np.all(np.diff(decoded[..., 1], axis=1) >= 0)
    assert np.all(decoded[..., 2] == 1_234)


def test_empty_mask_is_exact_read_only_copy_and_never_calls_inpainter() -> None:
    source = np.arange(7 * 9 * 3, dtype=np.uint16).reshape(7, 9, 3)
    labels = np.zeros(source.shape[:2], dtype=np.int32)
    mask = np.zeros(source.shape[:2], dtype=bool)

    def forbidden(_rgb: np.ndarray, _mask: np.ndarray) -> np.ndarray:
        raise AssertionError("empty synthesis mask must not invoke inpainter")

    result = composite_components(source, labels, mask, forbidden)

    np.testing.assert_array_equal(result.hybrid_rgb16, source)
    assert not np.shares_memory(result.hybrid_rgb16, source)
    assert result.components == ()
    assert result.pure_rgb16_sha256 == _raw_hash(source)
    assert result.hybrid_rgb16_sha256 == _raw_hash(source)
    assert result.synthesis_mask_sha256 == _raw_hash(mask)
    assert result.hybrid_rgb16.flags.c_contiguous
    assert not result.hybrid_rgb16.flags.writeable
    with pytest.raises(ValueError):
        result.hybrid_rgb16[0, 0, 0] = 1


def test_empty_mask_never_calls_batch_only_inpainter() -> None:
    source = _constant_rgb((5, 7), 12_000)
    labels = np.zeros(source.shape[:2], dtype=np.int32)
    mask = labels != 0

    class BatchOnly:
        calls = 0

        def inpaint_batch(
            self,
            _rgb_crops: object,
            _component_masks: object,
        ) -> tuple[np.ndarray, ...]:
            self.calls += 1
            raise AssertionError("empty routing must not invoke a batch")

    inpainter = BatchOnly()
    result = composite_components(source, labels, mask, inpainter)

    assert inpainter.calls == 0
    assert result.components == ()
    np.testing.assert_array_equal(result.hybrid_rgb16, source)


def test_poisoned_crop_changes_only_component_and_records_context_ranges() -> None:
    rows, columns = np.indices((12, 14))
    source = np.stack(
        (
            rows * 1_000 + columns * 10,
            rows * 700 + columns * 30 + 2_000,
            rows * 300 + columns * 50 + 4_000,
        ),
        axis=2,
    ).astype(np.uint16)
    original = source.copy()
    labels = np.zeros((12, 14), dtype=np.int32)
    labels[4:8, 5:10] = 1
    mask = labels != 0

    def poison(rgb: np.ndarray, component_mask: np.ndarray) -> np.ndarray:
        assert rgb.dtype == np.uint8
        assert component_mask.dtype == np.uint8
        assert set(np.unique(component_mask)) == {0, 255}
        assert np.count_nonzero(component_mask == 255) == 20
        return np.full_like(rgb, 255)

    result = composite_components(
        source,
        labels,
        mask,
        poison,
        crop_margin=3,
    )

    np.testing.assert_array_equal(source, original)
    np.testing.assert_array_equal(result.hybrid_rgb16[~mask], original[~mask])
    assert np.any(result.hybrid_rgb16[mask] != original[mask])
    record = result.components[0]
    assert record.component_id == 1
    assert record.component_bbox == BoundingBox(4, 5, 8, 10)
    assert record.crop_bbox == BoundingBox(1, 2, 11, 13)
    assert record.pixel_count == 20
    crop = original[1:11, 2:13]
    context = ~mask[1:11, 2:13]
    expected_lo = crop[context].min(axis=0)
    expected_hi = crop[context].max(axis=0)
    assert tuple(item.lo for item in record.channel_ranges) == tuple(expected_lo)
    assert tuple(item.hi for item in record.channel_ranges) == tuple(expected_hi)
    assert all(not item.degenerate for item in record.channel_ranges)
    for digest in (
        record.input_rgb8_sha256,
        record.component_mask_sha256,
        record.inpainted_rgb8_sha256,
        record.decoded_rgb16_sha256,
        record.blended_component_rgb16_sha256,
    ):
        assert len(digest) == 64


def test_three_pixel_chessboard_feather_uses_half_up_blending() -> None:
    source = _constant_rgb((10, 10))
    source[0, 0] = 300
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[2:7, 2:7] = 1
    mask = labels != 0

    result = composite_components(
        source,
        labels,
        mask,
        lambda rgb, _mask: np.full_like(rgb, 255),
        crop_margin=2,
    )

    expected = np.array(
        [
            [100, 100, 100, 100, 100],
            [100, 200, 200, 200, 100],
            [100, 200, 300, 200, 100],
            [100, 200, 200, 200, 100],
            [100, 100, 100, 100, 100],
        ],
        dtype=np.uint16,
    )
    for channel in range(3):
        np.testing.assert_array_equal(
            result.hybrid_rgb16[2:7, 2:7, channel],
            expected,
        )
    assert result.components[0].alpha_min == pytest.approx(1 / 3)
    assert result.components[0].alpha_max == 1.0


def test_frame_edge_is_interior_for_alpha_and_has_no_artificial_ramp() -> None:
    source = _constant_rgb((8, 8))
    source[7, 7] = 300
    labels = np.zeros((8, 8), dtype=np.int32)
    labels[:5, :5] = 1
    mask = labels != 0

    result = composite_components(
        source,
        labels,
        mask,
        lambda rgb, _mask: np.full_like(rgb, 255),
    )

    output = result.hybrid_rgb16[..., 0]
    assert output[0, 0] == 300
    assert output[2, 2] == 300
    assert output[0, 4] == 100
    assert output[4, 4] == 100
    assert result.components[0].alpha_max == 1.0


def test_all_components_are_preflighted_before_first_callback() -> None:
    source = _constant_rgb((9, 12), 10_000)
    labels = np.zeros((9, 12), dtype=np.int32)
    # This L-shape has healthy context inside its bbox at margin zero.
    labels[1, 1:4] = 1
    labels[2:4, 1] = 1
    # This solid rectangle has no context inside its bbox at margin zero.
    labels[5:8, 7:10] = 2
    mask = labels != 0
    calls = 0

    def count_calls(rgb: np.ndarray, _mask: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return rgb

    with pytest.raises(NoContextPixelsError, match="component 2") as raised:
        composite_components(
            source,
            labels,
            mask,
            count_calls,
            crop_margin=0,
        )
    assert raised.value.component_id == 2
    assert calls == 0


def test_overlapping_crops_always_encode_from_immutable_pure_source() -> None:
    rows, columns = np.indices((15, 15))
    source = np.stack(
        (
            rows * 500 + columns * 11,
            rows * 333 + columns * 21 + 1_000,
            rows * 123 + columns * 41 + 2_000,
        ),
        axis=2,
    ).astype(np.uint16)
    original = source.copy()
    labels = np.zeros((15, 15), dtype=np.int32)
    labels[6, 9] = 2
    labels[6, 5] = 7
    mask = labels != 0
    expected_inputs: list[np.ndarray] = []
    expected_masks: list[np.ndarray] = []
    for component_id in (2, 7):
        row, column = np.argwhere(labels == component_id)[0]
        y0, y1 = max(0, row - 5), min(15, row + 6)
        x0, x1 = max(0, column - 5), min(15, column + 6)
        crop = original[y0:y1, x0:x1]
        context = ~mask[y0:y1, x0:x1]
        lows = tuple(int(value) for value in crop[context].min(axis=0))
        highs = tuple(int(value) for value in crop[context].max(axis=0))
        expected_inputs.append(affine_encode_uint8(crop, lows, highs))
        local_component = labels[y0:y1, x0:x1] == component_id
        expected_masks.append(local_component.astype(np.uint8) * 255)

    call_index = 0

    def destructive(rgb: np.ndarray, local_mask: np.ndarray) -> np.ndarray:
        nonlocal call_index
        np.testing.assert_array_equal(rgb, expected_inputs[call_index])
        np.testing.assert_array_equal(local_mask, expected_masks[call_index])
        call_index += 1
        rgb[...] = 255
        local_mask[...] = 0
        return rgb

    result = composite_components(
        source,
        labels,
        mask,
        destructive,
        crop_margin=5,
    )

    assert call_index == 2
    assert [item.component_id for item in result.components] == [2, 7]
    assert [item.input_rgb8_sha256 for item in result.components] == [
        _raw_hash(value) for value in expected_inputs
    ]
    assert [item.component_mask_sha256 for item in result.components] == [
        _raw_hash(value) for value in expected_masks
    ]
    np.testing.assert_array_equal(source, original)
    np.testing.assert_array_equal(result.hybrid_rgb16[~mask], original[~mask])


def test_multiple_components_use_one_ordered_batch_and_no_scalar_calls() -> None:
    source, labels, mask = _multiple_component_inputs()

    class RecordingBatchInpainter:
        def __init__(self) -> None:
            self.batch_calls = 0
            self.scalar_calls = 0
            self.input_hashes: list[tuple[str, ...]] = []
            self.mask_pixel_counts: list[tuple[int, ...]] = []

        def __call__(self, _rgb: np.ndarray, _mask: np.ndarray) -> np.ndarray:
            self.scalar_calls += 1
            raise AssertionError("batch-capable inpainter must not use scalar calls")

        def inpaint_batch(
            self,
            rgb_crops: tuple[np.ndarray, ...],
            component_masks: tuple[np.ndarray, ...],
        ) -> tuple[np.ndarray, ...]:
            self.batch_calls += 1
            self.input_hashes.append(tuple(_raw_hash(rgb) for rgb in rgb_crops))
            self.mask_pixel_counts.append(
                tuple(int(np.count_nonzero(mask)) for mask in component_masks)
            )
            return tuple(np.zeros_like(rgb) for rgb in rgb_crops)

    first_inpainter = RecordingBatchInpainter()
    first = composite_components(
        source,
        labels,
        mask,
        first_inpainter,
        crop_margin=1,
    )
    second_inpainter = RecordingBatchInpainter()
    second = composite_components(
        source,
        labels,
        mask,
        second_inpainter,
        crop_margin=1,
    )
    scalar = composite_components(
        source,
        labels,
        mask,
        lambda rgb, _local_mask: np.zeros_like(rgb),
        crop_margin=1,
    )

    assert first_inpainter.batch_calls == second_inpainter.batch_calls == 1
    assert first_inpainter.scalar_calls == second_inpainter.scalar_calls == 0
    assert first_inpainter.mask_pixel_counts == second_inpainter.mask_pixel_counts
    assert first_inpainter.mask_pixel_counts == [(6, 8)]
    assert first_inpainter.input_hashes == second_inpainter.input_hashes
    assert [record.component_id for record in first.components] == [2, 9]
    assert first.components == second.components
    assert first.components == scalar.components
    np.testing.assert_array_equal(first.hybrid_rgb16, second.hybrid_rgb16)
    np.testing.assert_array_equal(first.hybrid_rgb16, scalar.hybrid_rgb16)
    np.testing.assert_array_equal(first.hybrid_rgb16[~mask], source[~mask])


def test_mutating_batch_cannot_corrupt_canonical_input_provenance() -> None:
    source, labels, mask = _multiple_component_inputs()

    class MutatingBatch:
        def __init__(self) -> None:
            self.input_hashes: tuple[str, ...] = ()
            self.mask_hashes: tuple[str, ...] = ()

        def inpaint_batch(
            self,
            rgb_crops: tuple[np.ndarray, ...],
            component_masks: tuple[np.ndarray, ...],
        ) -> tuple[np.ndarray, ...]:
            assert all(value.flags.writeable for value in rgb_crops)
            assert all(value.flags.writeable for value in component_masks)
            self.input_hashes = tuple(_raw_hash(value) for value in rgb_crops)
            self.mask_hashes = tuple(_raw_hash(value) for value in component_masks)
            for rgb, component_mask in zip(
                rgb_crops,
                component_masks,
                strict=True,
            ):
                rgb[...] = 255
                component_mask[...] = 0
            return rgb_crops

    inpainter = MutatingBatch()
    result = composite_components(
        source,
        labels,
        mask,
        inpainter,
        crop_margin=1,
    )

    assert tuple(item.input_rgb8_sha256 for item in result.components) == (
        inpainter.input_hashes
    )
    assert tuple(item.component_mask_sha256 for item in result.components) == (
        inpainter.mask_hashes
    )
    assert all(
        item.inpainted_rgb8_sha256 != item.input_rgb8_sha256
        for item in result.components
    )
    np.testing.assert_array_equal(result.hybrid_rgb16[~mask], source[~mask])


def test_batch_output_count_is_validated_before_compositing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, labels, mask = _multiple_component_inputs()
    decode_calls = 0
    original_decode = composite_module.affine_decode_uint16

    def track_decode(*args: object, **kwargs: object) -> np.ndarray:
        nonlocal decode_calls
        decode_calls += 1
        return original_decode(*args, **kwargs)

    monkeypatch.setattr(composite_module, "affine_decode_uint16", track_decode)

    class ShortBatch:
        scalar_calls = 0
        batch_calls = 0

        def __call__(self, _rgb: np.ndarray, _mask: np.ndarray) -> np.ndarray:
            self.scalar_calls += 1
            raise AssertionError("scalar path must not be used")

        def inpaint_batch(
            self,
            rgb_crops: tuple[np.ndarray, ...],
            _component_masks: tuple[np.ndarray, ...],
        ) -> tuple[np.ndarray, ...]:
            self.batch_calls += 1
            return (np.zeros_like(rgb_crops[0]),)

    inpainter = ShortBatch()
    with pytest.raises(
        ValueError,
        match="batch inpainter returned 1 results for 2 components",
    ):
        composite_components(
            source,
            labels,
            mask,
            inpainter,
            crop_margin=1,
        )

    assert inpainter.batch_calls == 1
    assert inpainter.scalar_calls == 0
    assert decode_calls == 0


@pytest.mark.parametrize("kind", ("overlong", "infinite"))
def test_oversized_batch_iterator_consumption_is_bounded_before_compositing(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    source, labels, mask = _multiple_component_inputs()
    decode_calls = 0
    original_decode = composite_module.affine_decode_uint16

    def track_decode(*args: object, **kwargs: object) -> np.ndarray:
        nonlocal decode_calls
        decode_calls += 1
        return original_decode(*args, **kwargs)

    monkeypatch.setattr(composite_module, "affine_decode_uint16", track_decode)

    class OversizedBatch:
        values_read = 0

        def inpaint_batch(
            self,
            rgb_crops: tuple[np.ndarray, ...],
            _component_masks: tuple[np.ndarray, ...],
        ) -> object:
            def generate() -> object:
                while kind == "infinite" or self.values_read < 100:
                    self.values_read += 1
                    yield np.zeros_like(rgb_crops[0])

            return generate()

    inpainter = OversizedBatch()
    with pytest.raises(
        ValueError,
        match="batch inpainter returned 3 results for 2 components",
    ):
        composite_components(
            source,
            labels,
            mask,
            inpainter,
            crop_margin=1,
        )

    assert inpainter.values_read == 3
    assert decode_calls == 0


@pytest.mark.parametrize(
    ("invalid_kind", "exception", "message"),
    (
        ("dtype", TypeError, "component 9.*dtype uint8"),
        ("shape", ValueError, "component 9.*shape"),
    ),
)
def test_later_invalid_batch_output_is_rejected_before_any_compositing(
    monkeypatch: pytest.MonkeyPatch,
    invalid_kind: str,
    exception: type[Exception],
    message: str,
) -> None:
    source, labels, mask = _multiple_component_inputs()
    decode_calls = 0
    original_decode = composite_module.affine_decode_uint16

    def track_decode(*args: object, **kwargs: object) -> np.ndarray:
        nonlocal decode_calls
        decode_calls += 1
        return original_decode(*args, **kwargs)

    monkeypatch.setattr(composite_module, "affine_decode_uint16", track_decode)

    class InvalidLaterBatch:
        scalar_calls = 0
        batch_calls = 0

        def __call__(self, _rgb: np.ndarray, _mask: np.ndarray) -> np.ndarray:
            self.scalar_calls += 1
            raise AssertionError("scalar path must not be used")

        def inpaint_batch(
            self,
            rgb_crops: tuple[np.ndarray, ...],
            _component_masks: tuple[np.ndarray, ...],
        ) -> tuple[np.ndarray, ...]:
            self.batch_calls += 1
            valid = np.zeros_like(rgb_crops[0])
            if invalid_kind == "dtype":
                invalid = np.zeros(rgb_crops[1].shape, dtype=np.uint16)
            else:
                invalid = np.zeros(rgb_crops[1].shape[:2], dtype=np.uint8)
            return valid, invalid

    inpainter = InvalidLaterBatch()
    with pytest.raises(exception, match=message):
        composite_components(
            source,
            labels,
            mask,
            inpainter,
            crop_margin=1,
        )

    assert inpainter.batch_calls == 1
    assert inpainter.scalar_calls == 0
    assert decode_calls == 0


def test_one_connected_merged_label_causes_exactly_one_invocation() -> None:
    source = _constant_rgb((14, 18))
    source[0, 0] = 1_000
    labels = np.zeros((14, 18), dtype=np.int32)
    labels[4:8, 3:7] = 5
    labels[4:8, 11:15] = 5
    labels[5:7, 7:11] = 5
    mask = labels != 0
    calls = 0

    def identity(rgb: np.ndarray, component_mask: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        assert np.count_nonzero(component_mask) == np.count_nonzero(mask)
        return rgb

    result = composite_components(source, labels, mask, identity)

    assert calls == 1
    assert len(result.components) == 1
    assert result.components[0].component_id == 5


def test_numeric_component_order_and_result_are_deterministic() -> None:
    source = _constant_rgb((12, 12))
    source[0, 0] = (100, 200, 300)
    labels = np.zeros((12, 12), dtype=np.int32)
    labels[2:4, 7:9] = 9
    labels[7:9, 2:4] = 2
    mask = labels != 0

    def run() -> tuple[list[int], object]:
        seen: list[int] = []

        def zero(rgb: np.ndarray, local_mask: np.ndarray) -> np.ndarray:
            # The local mask position identifies which final label was called.
            if local_mask.shape == (12, 12):
                ys, xs = np.nonzero(local_mask)
                seen.append(int(labels[ys[0], xs[0]]))
            return np.zeros_like(rgb)

        return seen, composite_components(source, labels, mask, zero)

    first_seen, first = run()
    second_seen, second = run()

    assert first_seen == second_seen == [2, 9]
    assert [record.component_id for record in first.components] == [2, 9]
    np.testing.assert_array_equal(first.hybrid_rgb16, second.hybrid_rgb16)
    assert first.components == second.components
    assert first.hybrid_rgb16_sha256 == second.hybrid_rgb16_sha256


def test_permuted_internal_component_iteration_cannot_change_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _constant_rgb((12, 12))
    source[0, 0] = (100, 200, 300)
    labels = np.zeros((12, 12), dtype=np.int32)
    labels[2:4, 7:9] = 9
    labels[7:9, 2:4] = 2
    mask = labels != 0
    baseline = composite_components(
        source,
        labels,
        mask,
        lambda rgb, _local_mask: np.zeros_like(rgb),
    )
    original_build_plans = composite_module._build_plans

    def reversed_build_plans(*args: object, **kwargs: object) -> tuple[object, ...]:
        return tuple(reversed(original_build_plans(*args, **kwargs)))

    monkeypatch.setattr(composite_module, "_build_plans", reversed_build_plans)
    permuted = composite_components(
        source,
        labels,
        mask,
        lambda rgb, _local_mask: np.zeros_like(rgb),
    )

    assert [record.component_id for record in baseline.components] == [2, 9]
    assert [record.component_id for record in permuted.components] == [9, 2]
    np.testing.assert_array_equal(permuted.hybrid_rgb16, baseline.hybrid_rgb16)
    assert permuted.hybrid_rgb16_sha256 == baseline.hybrid_rgb16_sha256


def test_invalid_or_disconnected_label_contract_is_rejected() -> None:
    source = _constant_rgb((6, 6))
    labels = np.zeros((6, 6), dtype=np.int32)
    labels[1, 1] = 1
    labels[4, 4] = 1
    mask = labels != 0

    with pytest.raises(ValueError, match="one 8-connected"):
        composite_components(source, labels, mask, lambda rgb, _mask: rgb)

    wrong_mask = mask.copy()
    wrong_mask[0, 0] = True
    with pytest.raises(ValueError, match="exactly equivalent"):
        composite_components(source, labels, wrong_mask, lambda rgb, _mask: rgb)


def test_max_int_sparse_label_is_densely_remapped_before_find_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _constant_rgb((2, 2))
    labels = np.zeros((2, 2), dtype=np.int32)
    labels[0, 0] = np.iinfo(np.int32).max
    mask = labels != 0
    observed_find_objects: list[tuple[int, int | None]] = []
    original_find_objects = composite_module.ndimage.find_objects

    def bounded_find_objects(
        input_labels: np.ndarray,
        max_label: int | None = None,
    ) -> object:
        observed_find_objects.append((int(input_labels.max()), max_label))
        return original_find_objects(input_labels, max_label=max_label)

    monkeypatch.setattr(composite_module.ndimage, "find_objects", bounded_find_objects)
    result = composite_components(source, labels, mask, lambda rgb, _mask: rgb)

    assert observed_find_objects == [(1, 1)]
    assert len(result.components) == 1
    assert result.components[0].component_id == np.iinfo(np.int32).max


@pytest.mark.parametrize(
    ("returned", "exception"),
    (
        (np.zeros((6, 6, 3), dtype=np.uint16), TypeError),
        (np.zeros((6, 6), dtype=np.uint8), ValueError),
    ),
)
def test_inpainter_result_contract_is_enforced(
    returned: np.ndarray,
    exception: type[Exception],
) -> None:
    source = _constant_rgb((6, 6))
    labels = np.zeros((6, 6), dtype=np.int32)
    labels[2:4, 2:4] = 1
    mask = labels != 0

    with pytest.raises(exception):
        composite_components(
            source,
            labels,
            mask,
            lambda _rgb, _mask: returned,
        )


def test_sparse_mask_component_count_fails_before_model_execution() -> None:
    shape = (1_000, 1_000)
    source = _constant_rgb(shape, 12_000)
    labels = np.zeros(shape, dtype=np.int32)
    row_grid, column_grid = np.meshgrid(
        np.arange(0, shape[0], 10),
        np.arange(0, shape[1], 10),
        indexing="ij",
    )
    positions = np.column_stack((row_grid.ravel(), column_grid.ravel()))[
        : MAX_COMPONENTS_PER_RUN + 1
    ]
    labels[positions[:, 0], positions[:, 1]] = np.arange(
        1,
        len(positions) + 1,
        dtype=np.int32,
    )
    mask = labels != 0
    called = False

    def forbidden(_rgb: np.ndarray, _mask: np.ndarray) -> np.ndarray:
        nonlocal called
        called = True
        raise AssertionError("resource preflight must run before the model")

    assert np.count_nonzero(mask) / mask.size < 0.02
    with pytest.raises(CompositeResourceLimitError, match="component count"):
        composite_components(source, labels, mask, forbidden)
    assert not called


def test_sparse_mask_total_crop_pixels_fail_before_model_execution() -> None:
    shape = (1_024, 1_024)
    source = _constant_rgb(shape, 12_000)
    labels = np.zeros(shape, dtype=np.int32)
    rows, columns = np.meshgrid(
        np.arange(8, shape[0], 32),
        np.arange(8, shape[1], 32),
        indexing="ij",
    )
    positions = np.column_stack((rows.ravel(), columns.ravel()))
    labels[positions[:, 0], positions[:, 1]] = np.arange(
        1,
        len(positions) + 1,
        dtype=np.int32,
    )
    mask = labels != 0
    called = False

    def forbidden(_rgb: np.ndarray, _mask: np.ndarray) -> np.ndarray:
        nonlocal called
        called = True
        raise AssertionError("resource preflight must run before the model")

    assert len(positions) < MAX_COMPONENTS_PER_RUN
    assert np.count_nonzero(mask) / mask.size < 0.02
    with pytest.raises(CompositeResourceLimitError, match="crop pixels"):
        composite_components(
            source,
            labels,
            mask,
            forbidden,
            crop_margin=128,
        )
    assert not called
