"""Recovered six-row X3A startup scheduler.

Nikon runs the ordinary decision/reconstruction writer for six hidden centers
before emitting public row zero.  Those rows do not escape the pipeline, but
their conditional dither calls advance the job RNG.  Replaying them is
therefore required for byte-exact public output; substituting a captured state
would only fit one fixture.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt

from .dither import DitherBounds
from .reconstruction import (
    FeatureBandExtremaMode,
    ReconstructionParameters,
    _recovered_unscaled_averages,
    combine_recovered_candidate,
    driver_forces_fallback,
    feature_band_extrema_mode,
    recovered_rgb_candidates,
    write_reconstructed_pixel,
)
from .rng import LCG24
from .x3a import DecisionOffsets, DecisionParameters, normal_group_fallback


HIDDEN_CENTERS = (-6, -5, -4, -3, -2, -1)


@dataclass(frozen=True)
class StartupReplayResult:
    """Observable receipt from the six hidden writer stages."""

    attempted_per_stage: tuple[int, int, int, int, int, int]
    rng_advances_per_stage: tuple[int, int, int, int, int, int]
    final_rng_state: int


def _validate_planes(
    working_rgb: npt.ArrayLike,
    auxiliary: npt.ArrayLike,
    score: npt.ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = np.asarray(working_rgb)
    aux = np.asarray(auxiliary)
    score_plane = np.asarray(score)
    if rgb.dtype != np.dtype(np.float32) or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("startup working RGB must be float32 HxWx3")
    if (
        aux.dtype != np.dtype(np.float32)
        or score_plane.dtype != np.dtype(np.float32)
        or aux.shape != rgb.shape[:2]
        or score_plane.shape != rgb.shape[:2]
    ):
        raise ValueError("startup auxiliary and score must match RGB geometry")
    if rgb.shape[0] < 4 or rgb.shape[1] == 0:
        raise ValueError("startup replay requires at least four real rows")
    if not (
        np.all(np.isfinite(rgb))
        and np.all(np.isfinite(aux))
        and np.all(np.isfinite(score_plane))
    ):
        raise ValueError("startup planes must be finite")
    return rgb, aux, score_plane


def _startup_histories(
    rgb: np.ndarray,
    auxiliary: np.ndarray,
    score: np.ndarray,
    *,
    score_floor: np.float32,
    guard: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Build exact logical rows y=-10..+3 for all startup consumers."""

    minimum_y = -10
    maximum_y = rgb.shape[0] - 1
    row_count = maximum_y - minimum_y + 1
    width = rgb.shape[1]
    padded_width = width + 2 * guard
    active = slice(guard, guard + width)
    scores = np.zeros((row_count, padded_width), dtype=np.float32)
    weighted_auxiliary = np.zeros_like(scores)
    weighted_rgb = np.zeros((row_count, padded_width, 3), dtype=np.float32)
    raw_auxiliary = np.zeros_like(scores)

    def row_index(y: int) -> int:
        return y - minimum_y

    first_auxiliary = auxiliary[0]
    first_rgb = rgb[0]
    for y in range(minimum_y, maximum_y + 1):
        index = row_index(y)
        if y == -1:
            scores[index, active] = score_floor
            weighted_auxiliary[index, active] = np.multiply(
                first_auxiliary, score_floor, dtype=np.float32
            )
            weighted_rgb[index, active] = np.multiply(
                first_rgb, score_floor, dtype=np.float32
            )
        elif y >= 0:
            scores[index, active] = score[y]
            weighted_auxiliary[index, active] = np.multiply(
                score[y], auxiliary[y], dtype=np.float32
            )
            weighted_rgb[index, active] = np.multiply(
                score[y, :, None], rgb[y], dtype=np.float32
            )

        # Raw decision history is initialized from the first real auxiliary
        # row. Negative rows retain the allocator-zero right guard; real rows
        # use ordinary two-sided endpoint replication.
        source = first_auxiliary if y < 0 else auxiliary[y]
        raw_auxiliary[index, :guard] = source[0]
        raw_auxiliary[index, active] = source
        if y >= 0:
            raw_auxiliary[index, guard + width :] = source[-1]

    return scores, weighted_auxiliary, weighted_rgb, raw_auxiliary, minimum_y


def _startup_feature_record(
    *,
    logical_y: int,
    x: int,
    width: int,
    guard: int,
    minimum_y: int,
    score_history: np.ndarray,
    weighted_auxiliary_history: np.ndarray,
    raw_auxiliary_history: np.ndarray,
    fallback: np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the W/A records produced for one hidden logical center.

    No W/A row centered at ``y <= -6`` has been produced when the hidden
    writer can first observe it, so those ring slots retain allocator zero.
    Horizontal A guards likewise remain zero instead of clamping to the first
    or final active record.
    """

    if logical_y <= HIDDEN_CENTERS[0] or x < 0 or x >= width:
        zero = np.zeros(4, dtype=np.float32)
        return zero, np.array(zero, copy=True)
    center_row = logical_y - minimum_y
    padded_x = x + guard
    score_patch = score_history[
        center_row - 4 : center_row + 5,
        padded_x - 4 : padded_x + 5,
    ]
    weighted_patch = weighted_auxiliary_history[
        center_row - 4 : center_row + 5,
        padded_x - 4 : padded_x + 5,
    ]
    if score_patch.shape != (9, 9) or weighted_patch.shape != (9, 9):
        raise ValueError("startup feature support is incomplete")
    unrounded_weights = _recovered_unscaled_averages(score_patch)[:, 0]
    weights = np.empty(4, dtype=np.float32)
    weights[:3] = unrounded_weights.astype(np.float32)
    weights[3] = score_history[center_row, padded_x]
    numerator = _recovered_unscaled_averages(weighted_patch)[:, 0]
    feature = np.empty(4, dtype=np.float32)
    for lane in range(3):
        feature[lane] = (
            np.float32(numerator[lane] / unrounded_weights[lane])
            if unrounded_weights[lane] > 0.0
            else fallback
        )
    feature[3] = (
        raw_auxiliary_history[center_row, padded_x]
        if weights[3] > np.float32(0.0)
        else fallback
    )
    return weights, feature


def replay_hidden_startup_rows(
    working_rgb: npt.ArrayLike,
    auxiliary: npt.ArrayLike,
    score: npt.ArrayLike,
    *,
    score_floor: float,
    decision_parameters: DecisionParameters,
    reconstruction_parameters: ReconstructionParameters,
    dither_bounds: DitherBounds,
    generator: LCG24,
    reconstruction_parameters_for_stage: (
        Callable[[int], ReconstructionParameters] | None
    ) = None,
    guard: int = 4,
) -> StartupReplayResult:
    """Execute hidden centers y=-6..-1 and advance ``generator`` exactly."""

    rgb, aux, score_plane = _validate_planes(working_rgb, auxiliary, score)
    if guard != 4:
        raise ValueError("captured X3A startup requires a four-sample guard")
    mode = feature_band_extrema_mode(
        resolution_metric=reconstruction_parameters.resolution_metric,
        cross_neighbor_cutoff=reconstruction_parameters.cross_neighbor_cutoff,
    )
    if mode is FeatureBandExtremaMode.CROSS_NEIGHBOR and rgb.shape[0] < 5:
        raise ValueError(
            "cross-neighbor startup requires the first five real rows"
        )
    floor = np.float32(score_floor)
    if not np.isfinite(floor) or floor <= np.float32(0.0):
        raise ValueError("startup score floor must be finite and positive")
    (
        score_history,
        weighted_auxiliary_history,
        weighted_rgb_history,
        raw_auxiliary_history,
        minimum_y,
    ) = _startup_histories(rgb, aux, score_plane, score_floor=floor, guard=guard)
    width = rgb.shape[1]
    offsets = DecisionOffsets.captured_normal_ls5000(
        decision_parameters.perpendicular_radius
    ).values
    attempted_per_stage: list[int] = []
    advances_per_stage: list[int] = []

    for center_y in HIDDEN_CENTERS:
        stage_hit = center_y - HIDDEN_CENTERS[0] + 1
        active_reconstruction_parameters = (
            reconstruction_parameters
            if reconstruction_parameters_for_stage is None
            else reconstruction_parameters_for_stage(stage_hit)
        )
        if not isinstance(
            active_reconstruction_parameters, ReconstructionParameters
        ):
            raise TypeError(
                "startup stage callback must return ReconstructionParameters"
            )
        # The first two scheduler calls have not filled the four-sample
        # horizontal lookahead yet. Nikon therefore invokes the hidden writer
        # only for x=0..width-5; calls three through six cover the full row.
        writer_width = (
            width - 4
            if mode is FeatureBandExtremaMode.CROSS_NEIGHBOR and stage_hit <= 2
            else width
        )
        center_row = center_y - minimum_y
        groups = np.empty((writer_width, 4, 9), dtype=np.float32)
        for x in range(writer_width):
            padded_x = x + guard
            for group in range(4):
                for sample, (dy, dx) in enumerate(offsets[group]):
                    groups[x, group, sample] = raw_auxiliary_history[
                        center_row + int(dy), padded_x + int(dx)
                    ]
        decision_fallback = normal_group_fallback(groups, decision_parameters)
        stage_attempted = 0
        stage_advances = 0
        for x in range(writer_width):
            producer_stage = center_y + 6
            producer_parameters = (
                reconstruction_parameters
                if reconstruction_parameters_for_stage is None
                else reconstruction_parameters_for_stage(max(1, producer_stage))
            )
            weights, feature = _startup_feature_record(
                logical_y=center_y,
                x=x,
                width=width,
                guard=guard,
                minimum_y=minimum_y,
                score_history=score_history,
                weighted_auxiliary_history=weighted_auxiliary_history,
                raw_auxiliary_history=raw_auxiliary_history,
                fallback=np.float32(producer_parameters.coarse_reference),
            )
            if decision_fallback[x] or driver_forces_fallback(
                weights, active_reconstruction_parameters
            ):
                continue
            stage_attempted += 1
            candidates = recovered_rgb_candidates(
                weighted_rgb_history[center_row - 4 : center_row + 5, x : x + 9],
                weights[:3],
            )
            features = feature[np.newaxis, :]
            if mode is FeatureBandExtremaMode.CROSS_NEIGHBOR:
                cross_features = [feature]
                for neighbor_y, neighbor_x in (
                    (center_y, x - 1),
                    (center_y, x + 1),
                    (center_y - 1, x),
                    (center_y + 1, x),
                ):
                    neighbor_stage = neighbor_y + 6
                    neighbor_parameters = (
                        reconstruction_parameters
                        if reconstruction_parameters_for_stage is None
                        else reconstruction_parameters_for_stage(
                            max(1, neighbor_stage)
                        )
                    )
                    _, neighbor_feature = _startup_feature_record(
                        logical_y=neighbor_y,
                        x=neighbor_x,
                        width=width,
                        guard=guard,
                        minimum_y=minimum_y,
                        score_history=score_history,
                        weighted_auxiliary_history=weighted_auxiliary_history,
                        raw_auxiliary_history=raw_auxiliary_history,
                        fallback=np.float32(
                            neighbor_parameters.coarse_reference
                        ),
                    )
                    cross_features.append(neighbor_feature)
                features = np.stack(cross_features)
            combined = combine_recovered_candidate(
                candidates[0].rgb,
                candidates[1].rgb,
                candidates[2].rgb,
                rgb[0, x],
                weight_record=weights,
                feature_records=features,
                parameters=active_reconstruction_parameters,
            )
            result = write_reconstructed_pixel(
                combined.candidate,
                rgb[0, x],
                parameters=active_reconstruction_parameters,
                dither_bounds=dither_bounds,
                generator=generator,
            )
            stage_advances += sum(result.rng_advances)
        attempted_per_stage.append(stage_attempted)
        advances_per_stage.append(stage_advances)

    return StartupReplayResult(
        attempted_per_stage=cast(
            tuple[int, int, int, int, int, int],
            tuple(attempted_per_stage),
        ),
        rng_advances_per_stage=cast(
            tuple[int, int, int, int, int, int],
            tuple(advances_per_stage),
        ),
        final_rng_state=generator.state,
    )
