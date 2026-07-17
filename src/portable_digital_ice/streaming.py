"""Memory-bounded row-stream execution for the supported CPU profile.

The implementation keeps an eleven-row analysis window, evaluates the gates
with NumPy, and enters the scalar writer only for selected pixels. Selected
coordinates remain strictly row-major so conditional random draws stay
deterministic.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt

from .dither import DitherBounds
from .output import emit_public_rgb16
from .reconstruction import (
    ReconstructionParameters,
    _recovered_unscaled_averages,
    combine_recovered_candidate,
    driver_forces_fallback,
    feature_band_extrema_mode,
    FeatureBandExtremaMode,
    recovered_rgb_candidates,
    write_reconstructed_pixel,
)
from .rng import LCG24
from .startup import StartupReplayResult, replay_hidden_startup_rows
from .stage_parameters import (
    score_stage_for_row,
    StageParameterProvider,
    source_stage_for_row,
    validate_stage_calibration,
    writer_stage_for_row,
)
from .x3a import (
    AuxiliaryParameters,
    DecisionParameters,
    ScoreParameters,
    SharedLookupInputResponse,
    continuous_score,
    derive_auxiliary,
)


UInt16Image = npt.NDArray[np.uint16]
ProgressCallback = Callable[[int, int, int, int], None]


@dataclass(frozen=True)
class StreamingReplayResult:
    """Deterministic counters from one streamed execution."""

    shape: tuple[int, int, int]
    startup: StartupReplayResult | None
    attempted_pixels: int
    written_pixels: int
    public_rng_advances: int
    final_rng_state: int
    output_sha256: str
    changed_pixels: int


@dataclass(frozen=True)
class _AnalyzedRow:
    working: npt.NDArray[np.float32]
    auxiliary: npt.NDArray[np.float32]
    score: npt.NDArray[np.float32]
    weighted_auxiliary: npt.NDArray[np.float32]
    weighted_rgb: npt.NDArray[np.float32]


class _RowCache:
    def __init__(
        self,
        pixels: npt.NDArray[np.uint16],
        *,
        response: SharedLookupInputResponse,
        auxiliary_parameters: AuxiliaryParameters,
        score_parameters: ScoreParameters,
        stage_parameter_provider: StageParameterProvider | None,
    ) -> None:
        self._pixels = pixels
        self._response = response
        self._auxiliary_parameters = auxiliary_parameters
        self._score_parameters = score_parameters
        self._stage_parameter_provider = stage_parameter_provider
        self._rows: dict[int, _AnalyzedRow] = {}

    def get(self, row: int) -> _AnalyzedRow:
        if not 0 <= row < self._pixels.shape[0]:
            raise IndexError(row)
        cached = self._rows.get(row)
        if cached is not None:
            return cached
        auxiliary_parameters = self._auxiliary_parameters
        score_parameters = self._score_parameters
        if self._stage_parameter_provider is not None:
            auxiliary_stage_hit = source_stage_for_row(row)
            calibration = self._stage_parameter_provider(auxiliary_stage_hit)
            validate_stage_calibration(
                calibration,
                expected_stage_hit=auxiliary_stage_hit,
            )
            auxiliary_parameters = replace(
                auxiliary_parameters,
                alpha=calibration.auxiliary_alpha,
                calibration_offset=calibration.calibration_offset,
                alpha_one_replacement=calibration.base_primary,
            )
            score_stage_hit = score_stage_for_row(row)
            score_calibration = self._stage_parameter_provider(score_stage_hit)
            validate_stage_calibration(
                score_calibration,
                expected_stage_hit=score_stage_hit,
            )
            score_parameters = replace(
                score_parameters,
                base_primary=score_calibration.base_primary,
            )
        working = self._response.convert(self._pixels[row : row + 1])[0]
        auxiliary = derive_auxiliary(
            working[np.newaxis, :, :], auxiliary_parameters
        )[0]
        score = continuous_score(auxiliary[np.newaxis, :], score_parameters)[0]
        cached = _AnalyzedRow(
            working=np.ascontiguousarray(working, dtype=np.float32),
            auxiliary=np.ascontiguousarray(auxiliary, dtype=np.float32),
            score=np.ascontiguousarray(score, dtype=np.float32),
            weighted_auxiliary=np.multiply(score, auxiliary, dtype=np.float32),
            weighted_rgb=np.multiply(
                score[:, np.newaxis], working[:, :3], dtype=np.float32
            ),
        )
        self._rows[row] = cached
        return cached

    def discard_before(self, row: int) -> None:
        for key in tuple(self._rows):
            if key < row:
                del self._rows[key]


def _horizontal_low_count(row: np.ndarray, threshold: np.float32) -> np.ndarray:
    low = np.asarray(row < threshold, dtype=np.uint8)
    padded = np.pad(low, (4, 4), mode="edge")
    count = np.zeros(row.shape, dtype=np.uint8)
    for offset in range(9):
        count += padded[offset : offset + row.size]
    return count


def _decision_fallback_row(
    y: int,
    *,
    height: int,
    width: int,
    cache: _RowCache,
    parameters: DecisionParameters,
) -> npt.NDArray[np.bool_]:
    """Evaluate Nikon's four replicated groups without 36 image planes."""

    threshold = np.float32(parameters.sample_threshold)
    radius = parameters.perpendicular_radius
    x = np.arange(width, dtype=np.int32)
    left = np.clip(x - radius, 0, width - 1)
    right = np.clip(x + radius, 0, width - 1)
    vertical_left = np.zeros(width, dtype=np.uint8)
    vertical_right = np.zeros(width, dtype=np.uint8)
    for dy in range(-4, 5):
        auxiliary = cache.get(min(height - 1, max(0, y + dy))).auxiliary
        vertical_left += auxiliary[left] < threshold
        vertical_right += auxiliary[right] < threshold
    above = cache.get(max(0, y - radius)).auxiliary
    below = cache.get(min(height - 1, y + radius)).auxiliary
    horizontal_above = _horizontal_low_count(above, threshold)
    horizontal_below = _horizontal_low_count(below, threshold)
    limit = parameters.count_limit
    return np.ascontiguousarray(
        (vertical_left > limit)
        | (horizontal_above > limit)
        | (vertical_right > limit)
        | (horizontal_below > limit),
        dtype=bool,
    )


def _history_window(
    y: int,
    *,
    height: int,
    width: int,
    cache: _RowCache,
    score_floor: np.float32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build logical rows ``y-5..y+5`` with Nikon's asymmetric guards."""

    score = np.zeros((11, width + 8), dtype=np.float32)
    weighted_auxiliary = np.zeros_like(score)
    weighted_rgb = np.zeros((11, width + 8, 3), dtype=np.float32)
    active = slice(4, 4 + width)
    first = cache.get(0)
    last = cache.get(height - 1)
    for destination, logical_y in enumerate(range(y - 5, y + 6)):
        if logical_y < -1:
            continue
        if logical_y == -1:
            score[destination, active] = score_floor
            weighted_auxiliary[destination, active] = np.multiply(
                first.auxiliary, score_floor, dtype=np.float32
            )
            weighted_rgb[destination, active] = np.multiply(
                first.working[:, :3], score_floor, dtype=np.float32
            )
            continue
        source = last if logical_y >= height else cache.get(logical_y)
        score[destination, active] = source.score
        weighted_auxiliary[destination, active] = source.weighted_auxiliary
        weighted_rgb[destination, active] = source.weighted_rgb
    return score, weighted_auxiliary, weighted_rgb


def _feature_record(
    score_patch: np.ndarray,
    weighted_auxiliary_patch: np.ndarray,
    point_auxiliary: np.float32,
    point_score: np.float32,
    fallback: np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    unrounded_weights = _recovered_unscaled_averages(score_patch)[:, 0]
    weights = np.empty(4, dtype=np.float32)
    weights[:3] = unrounded_weights.astype(np.float32)
    weights[3] = point_score
    numerator = _recovered_unscaled_averages(weighted_auxiliary_patch)[:, 0]
    feature = np.empty(4, dtype=np.float32)
    for lane in range(3):
        feature[lane] = (
            np.float32(numerator[lane] / unrounded_weights[lane])
            if unrounded_weights[lane] > 0.0
            else fallback
        )
    feature[3] = point_auxiliary if weights[3] > np.float32(0.0) else fallback
    return weights, feature


def _cross_neighbor_feature_record(
    *,
    output_y: int,
    neighbor_y: int,
    neighbor_x: int,
    height: int,
    width: int,
    cache: _RowCache,
    score_history: np.ndarray,
    weighted_auxiliary_history: np.ndarray,
    score_floor: np.float32,
    fallback: np.float32,
) -> np.ndarray:
    """Recover one of the four neighboring 4000 dpi feature records.

    The scheduler allocates one horizontal guard record on each side of the A
    ring, and those records remain zero.  Vertically it instead materializes a
    pseudo row above the image and repeats the final real row below it.  The
    history planes passed here already contain the corresponding weighted
    9x9 neighborhoods; only the point sample needs the same boundary rule.
    """

    if neighbor_x < 0 or neighbor_x >= width:
        return np.zeros(4, dtype=np.float32)
    row_offset = neighbor_y - output_y
    if row_offset not in (-1, 0, 1):
        raise ValueError("cross-neighbor row must be adjacent to output row")
    if neighbor_y < 0:
        point = cache.get(0)
        point_auxiliary = point.auxiliary[neighbor_x]
        point_score = score_floor
    elif neighbor_y >= height:
        point = cache.get(height - 1)
        point_auxiliary = point.auxiliary[neighbor_x]
        point_score = point.score[neighbor_x]
    else:
        point = cache.get(neighbor_y)
        point_auxiliary = point.auxiliary[neighbor_x]
        point_score = point.score[neighbor_x]
    row_start = 1 + row_offset
    _, feature = _feature_record(
        score_history[row_start : row_start + 9, neighbor_x : neighbor_x + 9],
        weighted_auxiliary_history[
            row_start : row_start + 9, neighbor_x : neighbor_x + 9
        ],
        point_auxiliary,
        point_score,
        fallback,
    )
    return feature


def _startup_replay(
    cache: _RowCache,
    *,
    width: int,
    score_parameters: ScoreParameters,
    decision_parameters: DecisionParameters,
    reconstruction_parameters: ReconstructionParameters,
    dither_bounds: DitherBounds,
    generator: LCG24,
    stage_parameter_provider: StageParameterProvider | None,
) -> StartupReplayResult:
    mode = feature_band_extrema_mode(
        resolution_metric=reconstruction_parameters.resolution_metric,
        cross_neighbor_cutoff=reconstruction_parameters.cross_neighbor_cutoff,
    )
    first_row_count = (
        5 if mode is FeatureBandExtremaMode.CROSS_NEIGHBOR else 4
    )
    first_rows = [cache.get(row) for row in range(first_row_count)]

    def reconstruction_parameters_for_stage(
        stage_hit: int,
    ) -> ReconstructionParameters:
        if stage_parameter_provider is None:
            return reconstruction_parameters
        calibration = stage_parameter_provider(stage_hit)
        validate_stage_calibration(
            calibration,
            expected_stage_hit=stage_hit,
        )
        return replace(
            reconstruction_parameters,
            coarse_reference=calibration.base_primary,
            driver_gate_secondary=calibration.writer_gate_secondary,
            row_reconstruction_gate=calibration.row_reconstruction_gate,
        )

    return replay_hidden_startup_rows(
        np.stack([row.working[:, :3] for row in first_rows]),
        np.stack([row.auxiliary for row in first_rows]),
        np.stack([row.score for row in first_rows]),
        score_floor=score_parameters.floor,
        decision_parameters=decision_parameters,
        reconstruction_parameters=reconstruction_parameters,
        dither_bounds=dither_bounds,
        generator=generator,
        reconstruction_parameters_for_stage=(
            reconstruction_parameters_for_stage
            if stage_parameter_provider is not None
            else None
        ),
    )


def run_streaming_replay(
    main_rgbi: npt.ArrayLike,
    output_rgb16: UInt16Image,
    *,
    response: SharedLookupInputResponse,
    auxiliary_parameters: AuxiliaryParameters,
    score_parameters: ScoreParameters,
    decision_parameters: DecisionParameters,
    reconstruction_parameters: ReconstructionParameters,
    dither_bounds: DitherBounds,
    generator: LCG24 | None = None,
    stage_parameter_provider: StageParameterProvider | None = None,
    progress: ProgressCallback | None = None,
) -> StreamingReplayResult:
    """Execute the CPU writer with memory bounded by image width.

    ``output_rgb16`` may be a normal array or a writable ``numpy.memmap``.
    It contains logical rows only, never transport padding.
    """

    pixels = np.asarray(main_rgbi)
    output = np.asarray(output_rgb16)
    if pixels.dtype != np.dtype(np.uint16) or pixels.ndim != 3 or pixels.shape[2] != 4:
        raise ValueError("streaming main input must be uint16 HxWx4")
    height, width, _ = pixels.shape
    if height < 4:
        raise ValueError("streaming startup requires at least four main rows")
    if output.dtype != np.dtype(np.uint16) or output.shape != (height, width, 3):
        raise ValueError("streaming output must be uint16 HxWx3")
    if not output.flags.writeable:
        raise ValueError("streaming output must be writable")

    cache = _RowCache(
        pixels,
        response=response,
        auxiliary_parameters=auxiliary_parameters,
        score_parameters=score_parameters,
        stage_parameter_provider=stage_parameter_provider,
    )
    active_generator = generator or LCG24.from_nikon_pe_initial_state()
    startup = _startup_replay(
        cache,
        width=width,
        score_parameters=score_parameters,
        decision_parameters=decision_parameters,
        reconstruction_parameters=reconstruction_parameters,
        dither_bounds=dither_bounds,
        generator=active_generator,
        stage_parameter_provider=stage_parameter_provider,
    )
    mode = feature_band_extrema_mode(
        resolution_metric=reconstruction_parameters.resolution_metric,
        cross_neighbor_cutoff=reconstruction_parameters.cross_neighbor_cutoff,
    )
    score_floor = np.float32(score_parameters.floor)
    attempted = 0
    written = 0
    public_advances = 0
    changed = 0
    output_hash = hashlib.sha256()

    for y in range(height):
        active_reconstruction_parameters = reconstruction_parameters
        if stage_parameter_provider is not None:
            stage_hit = writer_stage_for_row(y)
            writer_calibration = stage_parameter_provider(stage_hit)
            validate_stage_calibration(
                writer_calibration,
                expected_stage_hit=stage_hit,
            )
            active_reconstruction_parameters = replace(
                reconstruction_parameters,
                coarse_reference=writer_calibration.base_primary,
                driver_gate_secondary=writer_calibration.writer_gate_secondary,
                row_reconstruction_gate=(
                    writer_calibration.row_reconstruction_gate
                ),
            )
        floor_enabled = (
            active_reconstruction_parameters.driver_gate_primary
            or active_reconstruction_parameters.driver_gate_secondary
        )
        fallback_value = np.float32(
            active_reconstruction_parameters.coarse_reference
        )
        current = cache.get(y)
        noop = emit_public_rgb16(current.working[np.newaxis, :, :3])[0]
        working_output = np.array(current.working[:, :3], dtype=np.float32, copy=True)
        decision_fallback = _decision_fallback_row(
            y,
            height=height,
            width=width,
            cache=cache,
            parameters=decision_parameters,
        )
        eligible = ~decision_fallback
        if active_reconstruction_parameters.row_reconstruction_gate != 0:
            eligible[:] = False
        elif floor_enabled:
            eligible &= current.score < np.float32(1.0)
        selected_x = np.flatnonzero(eligible)
        score_history, weighted_auxiliary_history, weighted_rgb_history = (
            _history_window(
                y,
                height=height,
                width=width,
                cache=cache,
                score_floor=score_floor,
            )
        )
        for x_value in selected_x:
            x = int(x_value)
            score_patch = score_history[1:10, x : x + 9]
            weighted_auxiliary_patch = weighted_auxiliary_history[1:10, x : x + 9]
            weights, center_feature = _feature_record(
                score_patch,
                weighted_auxiliary_patch,
                current.auxiliary[x],
                current.score[x],
                fallback_value,
            )
            if driver_forces_fallback(weights, active_reconstruction_parameters):
                continue
            attempted += 1
            features = center_feature[np.newaxis, :]
            if mode is FeatureBandExtremaMode.CROSS_NEIGHBOR:
                neighbors: list[np.ndarray] = [center_feature]
                for dy, dx in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                    neighbor_y = y + dy
                    feature = _cross_neighbor_feature_record(
                        output_y=y,
                        neighbor_y=neighbor_y,
                        neighbor_x=x + dx,
                        height=height,
                        width=width,
                        cache=cache,
                        score_history=score_history,
                        weighted_auxiliary_history=weighted_auxiliary_history,
                        score_floor=score_floor,
                        fallback=fallback_value,
                    )
                    neighbors.append(feature)
                features = np.stack(neighbors)
            candidates = recovered_rgb_candidates(
                weighted_rgb_history[1:10, x : x + 9],
                weights[:3],
            )
            combined = combine_recovered_candidate(
                candidates[0].rgb,
                candidates[1].rgb,
                candidates[2].rgb,
                current.working[x, :3],
                weight_record=weights,
                feature_records=features,
                parameters=active_reconstruction_parameters,
            )
            result = write_reconstructed_pixel(
                combined.candidate,
                current.working[x, :3],
                parameters=active_reconstruction_parameters,
                dither_bounds=dither_bounds,
                generator=active_generator,
            )
            working_output[x] = result.values
            public_advances += sum(result.rng_advances)
            written += int(np.any(result.values != current.working[x, :3]))

        rendered = emit_public_rgb16(working_output[np.newaxis, :, :])[0]
        output[y] = rendered
        output_hash.update(rendered.astype("<u2", copy=False).tobytes(order="C"))
        changed += int(np.count_nonzero(np.any(rendered != noop, axis=1)))
        if progress is not None:
            progress(y + 1, height, attempted, written)
        cache.discard_before(max(0, y - 5))

    return StreamingReplayResult(
        shape=(height, width, 3),
        startup=startup,
        attempted_pixels=attempted,
        written_pixels=written,
        public_rng_advances=public_advances,
        final_rng_state=active_generator.state,
        output_sha256=output_hash.hexdigest(),
        changed_pixels=changed,
    )
