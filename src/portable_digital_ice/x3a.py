"""Low-level clean-room X3A primitives and generic batch interfaces.

The exact product-reference path lives in :mod:`portable_dice.streaming` and
uses audited profiles plus frame-derived calibration.  This lower-level module
also keeps test models and caller-injected policies; those generic entry points
raise ``UnboundBehaviorError`` when a required value or implementation is not
supplied.  That fail-closed API behavior is not an open algorithm boundary.
"""

from __future__ import annotations

import math
import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from .contracts import RGBI16Frame


FloatArray = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]
NIKON_RESPONSE_LUT_SHA256 = (
    "5fd225e25719a544df9475315e3017f9b8027d593ac43e392245de59f49f3fcb"
)
NIKON_X87_CONTROL_WORD = 0x023F
NIKON_X87_PRECISION_BITS = 53


class UnboundBehaviorError(RuntimeError):
    """Raised instead of silently substituting an unproven algorithm value."""


def _finite(value: float, label: str) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{label} must be finite")
    return converted


@runtime_checkable
class InputResponse(Protocol):
    """Convert integer RGBI samples into Nikon's four float working lanes."""

    def convert(self, rgbi16: npt.NDArray[np.uint16]) -> FloatArray: ...


@dataclass(frozen=True)
class LinearInputResponse:
    """An explicit affine response model for tests and measured bindings.

    This is an explicit test model, not Nikon's closed logarithmic response.
    A caller must provide all four gains and offsets deliberately.
    """

    gains: tuple[float, float, float, float]
    offsets: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if len(self.gains) != 4 or len(self.offsets) != 4:
            raise ValueError("linear response requires four gains and four offsets")
        for index, value in enumerate((*self.gains, *self.offsets)):
            _finite(value, f"linear response value {index}")

    def convert(self, rgbi16: npt.NDArray[np.uint16]) -> FloatArray:
        pixels = np.asarray(rgbi16)
        if pixels.dtype != np.dtype(np.uint16):
            raise TypeError("RGBI response input must have dtype uint16")
        if pixels.ndim != 3 or pixels.shape[2] != 4:
            raise ValueError("RGBI response input must have shape HxWx4")
        gains = np.asarray(self.gains, dtype=np.float32)
        offsets = np.asarray(self.offsets, dtype=np.float32)
        working = pixels.astype(np.float32)
        working = np.multiply(working, gains, dtype=np.float32)
        return np.ascontiguousarray(
            np.add(working, offsets, dtype=np.float32), dtype=np.float32
        )


@dataclass(frozen=True)
class LookupInputResponse:
    """Four explicit 65,536-entry float response tables."""

    tables: FloatArray

    def __post_init__(self) -> None:
        tables = np.asarray(self.tables)
        if tables.dtype != np.dtype(np.float32):
            raise TypeError("response tables must have dtype float32")
        if tables.shape != (4, 65536) or not np.all(np.isfinite(tables)):
            raise ValueError("response tables must be finite with shape 4x65536")
        owned = np.array(tables, dtype=np.float32, order="C", copy=True)
        owned.flags.writeable = False
        object.__setattr__(self, "tables", owned)

    def convert(self, rgbi16: npt.NDArray[np.uint16]) -> FloatArray:
        pixels = np.asarray(rgbi16)
        if pixels.dtype != np.dtype(np.uint16):
            raise TypeError("RGBI response input must have dtype uint16")
        if pixels.ndim != 3 or pixels.shape[2] != 4:
            raise ValueError("RGBI response input must have shape HxWx4")
        result = np.empty(pixels.shape, dtype=np.float32)
        for channel in range(4):
            result[:, :, channel] = self.tables[channel, pixels[:, :, channel]]
        return np.ascontiguousarray(result)


def generate_nikon_response_lut() -> FloatArray:
    """Generate Nikon's complete hash-bound 16-bit logarithmic response LUT.

    The binary64 equation plus one final float32 conversion reproduces every
    one of the 65,536 captured entries on Apple Silicon.  The hash check makes
    any platform/libm last-bit difference fail closed instead of silently
    entering the image pipeline.
    """

    domain = np.arange(65536, dtype=np.float64)
    table = np.asarray(
        65535.0 * np.log2(domain + np.float64(1.0)) / 16.0,
        dtype=np.float32,
        order="C",
    )
    actual_hash = hashlib.sha256(table.astype("<f4", copy=False).tobytes()).hexdigest()
    if actual_hash != NIKON_RESPONSE_LUT_SHA256:
        raise UnboundBehaviorError(
            "platform logarithm does not reproduce Nikon's response LUT bit-exact"
        )
    table.flags.writeable = False
    return table


@dataclass(frozen=True)
class SharedLookupInputResponse:
    """One response table shared by all four active RGBI lanes."""

    table: FloatArray

    def __post_init__(self) -> None:
        table = np.asarray(self.table)
        if (
            table.dtype != np.dtype(np.float32)
            or table.shape != (65536,)
            or not np.all(np.isfinite(table))
        ):
            raise ValueError("shared response table must be finite float32[65536]")
        owned = np.array(table, dtype=np.float32, order="C", copy=True)
        owned.flags.writeable = False
        object.__setattr__(self, "table", owned)

    @classmethod
    def nikon_logarithmic(cls) -> "SharedLookupInputResponse":
        return cls(generate_nikon_response_lut())

    def convert(self, rgbi16: npt.NDArray[np.uint16]) -> FloatArray:
        pixels = np.asarray(rgbi16)
        if pixels.dtype != np.dtype(np.uint16):
            raise TypeError("RGBI response input must have dtype uint16")
        if pixels.ndim != 3 or pixels.shape[2] != 4:
            raise ValueError("RGBI response input must have shape HxWx4")
        return np.ascontiguousarray(self.table[pixels], dtype=np.float32)


@dataclass(frozen=True)
class AuxiliaryParameters:
    selected_visible_channel: int
    alpha: float
    calibration_offset: float
    alpha_one_replacement: float | None

    def __post_init__(self) -> None:
        if self.selected_visible_channel not in (0, 1, 2):
            raise ValueError("selected_visible_channel must be 0, 1, or 2")
        _finite(self.alpha, "alpha")
        _finite(self.calibration_offset, "calibration_offset")
        if self.alpha_one_replacement is not None:
            _finite(self.alpha_one_replacement, "alpha_one_replacement")


def _validate_working_rgbi(working_rgbi: npt.ArrayLike) -> FloatArray:
    array = np.asarray(working_rgbi)
    if array.dtype != np.dtype(np.float32):
        raise TypeError("working RGBI must have dtype float32")
    if array.ndim != 3 or array.shape[2] != 4:
        raise ValueError("working RGBI must have shape HxWx4")
    if not np.all(np.isfinite(array)):
        raise ValueError("working RGBI must be finite")
    return array


def derive_auxiliary(
    working_rgbi: npt.ArrayLike,
    parameters: AuxiliaryParameters,
) -> FloatArray:
    """Apply the recovered four-lane auxiliary equation with one final store.

    Context factors and response samples are float32 operands, but the active
    x87 control word uses 53-bit intermediates.  Widening the operands to
    binary64 and narrowing only the result reproduces all 443 observable
    auxiliary values in the final-flush trace byte-exact.
    """

    working = _validate_working_rgbi(working_rgbi)
    alpha32 = np.float32(parameters.alpha)
    offset32 = np.float32(parameters.calibration_offset)
    if alpha32 == np.float32(1.0):
        if parameters.alpha_one_replacement is None:
            raise UnboundBehaviorError(
                "alpha==1 requires the runtime-configured auxiliary replacement"
            )
        return np.full(
            working.shape[:2], parameters.alpha_one_replacement, dtype=np.float32
        )
    alpha = float(alpha32)
    offset = float(offset32)
    visible = working[:, :, parameters.selected_visible_channel].astype(np.float64)
    infrared = working[:, :, 3].astype(np.float64)
    auxiliary = (infrared - alpha * visible) / (1.0 - alpha) - offset
    return np.ascontiguousarray(auxiliary.astype(np.float32))


@dataclass(frozen=True)
class ScoreParameters:
    base_primary: float
    base_addend: float
    scale: float
    offset: float
    floor: float
    resolution_metric: int
    horizontal_minimum_resolution_cutoff: int

    def __post_init__(self) -> None:
        for label in ("base_primary", "base_addend", "scale", "offset", "floor"):
            _finite(getattr(self, label), label)
        if self.floor > 1.0:
            raise ValueError("score floor cannot exceed the recovered ceiling 1.0")
        if self.resolution_metric < 0:
            raise ValueError("resolution_metric cannot be negative")
        if self.horizontal_minimum_resolution_cutoff < 0:
            raise ValueError("horizontal minimum resolution cutoff cannot be negative")


def continuous_score(
    auxiliary: npt.ArrayLike,
    parameters: ScoreParameters,
) -> FloatArray:
    """Apply the recovered score equation and pointwise boundary rule.

    Nikon's 32-bit implementation loads float32 context/sample values into the
    x87 path, evaluates the affine expression at widened precision, and stores
    one float32 result.  Binary64 intermediates reproduce all 256 observable
    point-score values in the hash-pinned final-flush trace exactly.  Keeping
    that single final narrowing is important: rounding after each operation in
    float32 differs by as many as three ULPs on the same trace.
    """

    values = np.asarray(auxiliary)
    if values.dtype != np.dtype(np.float32):
        raise TypeError("auxiliary score input must have dtype float32")
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("auxiliary score input must be a finite HxW plane")
    _, width = values.shape
    sample = np.array(values, dtype=np.float32, order="C", copy=True)
    horizontal_minimum = (
        parameters.horizontal_minimum_resolution_cutoff < parameters.resolution_metric
    )
    if horizontal_minimum and width > 2:
        sample[:, 1:-1] = np.minimum(
            np.minimum(values[:, :-2], values[:, 1:-1]),
            values[:, 2:],
        )
    # Every source operand is a float32 field/sample.  Quantize the caller's
    # Python values first, then widen them, so the portable implementation does
    # not accidentally admit precision Nikon's context never contained.
    primary = np.float64(np.float32(parameters.base_primary))
    addend = np.float64(np.float32(parameters.base_addend))
    scale = np.float64(np.float32(parameters.scale))
    offset = np.float64(np.float32(parameters.offset))
    lower = np.float64(np.float32(parameters.floor))
    widened_sample = sample.astype(np.float64)
    score64 = ((primary + addend) - widened_sample) * scale + offset
    score64 = np.minimum(score64, np.float64(1.0))
    score64 = np.maximum(score64, lower)
    return np.ascontiguousarray(score64.astype(np.float32))


@dataclass(frozen=True)
class X3AAnalysis:
    """Deterministic float intermediates through score-weighted RGB history."""

    working_rgbi: FloatArray
    auxiliary: FloatArray
    score: FloatArray
    weighted_rgb: FloatArray
    source_evidence_id: str

    def __post_init__(self) -> None:
        height, width = self.auxiliary.shape
        expected_rgb = (height, width, 3)
        if self.working_rgbi.shape != (height, width, 4):
            raise ValueError("working RGBI geometry disagrees with auxiliary")
        if self.score.shape != (height, width):
            raise ValueError("score geometry disagrees with auxiliary")
        if self.weighted_rgb.shape != expected_rgb:
            raise ValueError("weighted RGB geometry disagrees with auxiliary")


def analyze_main(
    frame: RGBI16Frame,
    *,
    response: InputResponse,
    auxiliary_parameters: AuxiliaryParameters,
    score_parameters: ScoreParameters,
) -> X3AAnalysis:
    """Run the currently bound main-pass float stages.

    This stage-level entry point deliberately does not run the separate
    prepass. Callers supply the frame calibration through the typed parameters.
    """

    working = response.convert(frame.pixels)
    working = _validate_working_rgbi(working)
    auxiliary = derive_auxiliary(working, auxiliary_parameters)
    score = continuous_score(auxiliary, score_parameters)
    weighted = np.multiply(working[:, :, :3], score[:, :, np.newaxis], dtype=np.float32)
    return X3AAnalysis(
        working_rgbi=np.ascontiguousarray(working),
        auxiliary=auxiliary,
        score=score,
        weighted_rgb=np.ascontiguousarray(weighted),
        source_evidence_id=frame.evidence_id,
    )


@dataclass(frozen=True)
class DecisionParameters:
    sample_threshold: float
    count_limit: int
    perpendicular_radius: int

    def __post_init__(self) -> None:
        _finite(self.sample_threshold, "sample_threshold")
        if not 0 <= self.count_limit <= 9:
            raise ValueError("count_limit must be in [0, 9]")
        if not 1 <= self.perpendicular_radius <= 4:
            raise ValueError("perpendicular_radius must be in [1, 4]")


@dataclass(frozen=True)
class DecisionOffsets:
    """Explicit 4x9 (dy, dx) mapping; runtime geometry is not guessed."""

    values: npt.NDArray[np.int32]

    def __post_init__(self) -> None:
        offsets = np.asarray(self.values)
        if not np.issubdtype(offsets.dtype, np.integer) or offsets.shape != (4, 9, 2):
            raise ValueError("decision offsets must be an integer 4x9x2 array")
        owned = np.array(offsets, dtype=np.int32, order="C", copy=True)
        owned.flags.writeable = False
        object.__setattr__(self, "values", owned)

    @classmethod
    def captured_normal_ls5000(
        cls,
        perpendicular_radius: int,
    ) -> "DecisionOffsets":
        """Return the recovered four nine-sample bars around a center."""

        if not 1 <= perpendicular_radius <= 4:
            raise ValueError("perpendicular_radius must be in [1, 4]")

        offsets = np.empty((4, 9, 2), dtype=np.int32)
        axis = np.arange(-4, 5, dtype=np.int32)
        offsets[0, :, 0] = axis
        offsets[0, :, 1] = -perpendicular_radius
        offsets[1, :, 0] = -perpendicular_radius
        offsets[1, :, 1] = axis
        offsets[2, :, 0] = axis
        offsets[2, :, 1] = perpendicular_radius
        offsets[3, :, 0] = perpendicular_radius
        offsets[3, :, 1] = axis
        return cls(offsets)


def normal_group_fallback(
    groups: npt.ArrayLike,
    parameters: DecisionParameters,
) -> BoolArray:
    """Return fallback when any history group exceeds the below-threshold limit.

    This threshold is on Nikon's auxiliary/raw-intensity history scale, not on
    the separate bounded continuous score plane.
    """

    values = np.asarray(groups)
    if values.shape[-2:] != (4, 9) or not np.issubdtype(values.dtype, np.floating):
        raise ValueError("decision groups must end with shape 4x9 and be floating")
    if not np.all(np.isfinite(values)):
        raise ValueError("decision groups must be finite")
    low_counts = np.count_nonzero(values < parameters.sample_threshold, axis=-1)
    return np.ascontiguousarray(
        np.any(low_counts > parameters.count_limit, axis=-1), dtype=bool
    )


class BoundaryPolicy(str, Enum):
    FALLBACK = "fallback"
    REPLICATE = "replicate"
    ERROR = "error"


@dataclass(frozen=True)
class DecisionPlane:
    group_fallback: BoolArray
    boundary_unavailable: BoolArray

    def __post_init__(self) -> None:
        if self.group_fallback.dtype != np.dtype(bool):
            raise TypeError("group_fallback must be bool")
        if self.boundary_unavailable.dtype != np.dtype(bool):
            raise TypeError("boundary_unavailable must be bool")
        if self.group_fallback.shape != self.boundary_unavailable.shape:
            raise ValueError("decision planes must have matching shapes")


def _shifted_overlap(
    height: int,
    width: int,
    dy: int,
    dx: int,
) -> tuple[tuple[slice, slice], tuple[slice, slice]] | None:
    destination_y0 = max(0, -dy)
    destination_y1 = min(height, height - dy)
    destination_x0 = max(0, -dx)
    destination_x1 = min(width, width - dx)
    if destination_y0 >= destination_y1 or destination_x0 >= destination_x1:
        return None
    destination = (
        slice(destination_y0, destination_y1),
        slice(destination_x0, destination_x1),
    )
    source = (
        slice(destination_y0 + dy, destination_y1 + dy),
        slice(destination_x0 + dx, destination_x1 + dx),
    )
    return destination, source


def evaluate_normal_decision(
    decision_history: npt.ArrayLike,
    *,
    offsets: DecisionOffsets,
    parameters: DecisionParameters,
    boundary_policy: BoundaryPolicy,
) -> DecisionPlane:
    """Evaluate the normal history count gate without materializing 36 planes."""

    values = np.asarray(decision_history)
    if values.dtype != np.dtype(np.float32):
        raise TypeError("decision history must have dtype float32")
    if values.ndim != 3 or values.shape[2] != 4 or not np.all(np.isfinite(values)):
        raise ValueError("decision history must be a finite HxWx4 array")
    height, width, _ = values.shape
    counts = np.zeros((height, width, 4), dtype=np.uint8)
    valid_counts = np.zeros((height, width, 4), dtype=np.uint8)
    for group_index in range(4):
        for dy, dx in offsets.values[group_index]:
            overlap = _shifted_overlap(height, width, int(dy), int(dx))
            if overlap is None:
                if boundary_policy is BoundaryPolicy.REPLICATE:
                    source_y = np.clip(
                        np.arange(height, dtype=np.int32) + int(dy), 0, height - 1
                    )
                    source_x = np.clip(
                        np.arange(width, dtype=np.int32) + int(dx), 0, width - 1
                    )
                    counts[:, :, group_index] += (
                        values[
                            source_y[:, np.newaxis],
                            source_x[np.newaxis, :],
                            group_index,
                        ]
                        < parameters.sample_threshold
                    )
                continue
            destination, source = overlap
            if boundary_policy is BoundaryPolicy.REPLICATE:
                source_y = np.clip(
                    np.arange(height, dtype=np.int32) + int(dy), 0, height - 1
                )
                source_x = np.clip(
                    np.arange(width, dtype=np.int32) + int(dx), 0, width - 1
                )
                counts[:, :, group_index] += (
                    values[
                        source_y[:, np.newaxis],
                        source_x[np.newaxis, :],
                        group_index,
                    ]
                    < parameters.sample_threshold
                )
            else:
                counts[destination + (group_index,)] += (
                    values[source + (group_index,)] < parameters.sample_threshold
                )
            valid_counts[destination + (group_index,)] += 1
    boundary = np.any(valid_counts != 9, axis=2)
    if boundary_policy is BoundaryPolicy.ERROR and np.any(boundary):
        raise UnboundBehaviorError(
            "decision groups cross the image boundary and no fallback policy was allowed"
        )
    group_fallback = np.any(counts > parameters.count_limit, axis=2)
    if boundary_policy is BoundaryPolicy.FALLBACK:
        group_fallback = np.logical_or(group_fallback, boundary)
    return DecisionPlane(
        group_fallback=np.ascontiguousarray(group_fallback, dtype=bool),
        boundary_unavailable=np.ascontiguousarray(boundary, dtype=bool),
    )


def evaluate_normal_auxiliary_decision(
    auxiliary: npt.ArrayLike,
    *,
    parameters: DecisionParameters,
    boundary_policy: BoundaryPolicy,
) -> DecisionPlane:
    """Evaluate the recovered four bars directly on the auxiliary plane."""

    values = np.asarray(auxiliary)
    if values.dtype != np.dtype(np.float32) or values.ndim != 2:
        raise ValueError("decision auxiliary source must be a float32 HxW plane")
    if not np.all(np.isfinite(values)):
        raise ValueError("decision auxiliary source must be finite")
    offsets = DecisionOffsets.captured_normal_ls5000(
        parameters.perpendicular_radius
    ).values
    height, width = values.shape
    counts = np.zeros((height, width, 4), dtype=np.uint8)
    valid_counts = np.zeros((height, width, 4), dtype=np.uint8)
    for group_index in range(4):
        for dy, dx in offsets[group_index]:
            overlap = _shifted_overlap(height, width, int(dy), int(dx))
            if overlap is None:
                if boundary_policy is BoundaryPolicy.REPLICATE:
                    source_y = np.clip(
                        np.arange(height, dtype=np.int32) + int(dy), 0, height - 1
                    )
                    source_x = np.clip(
                        np.arange(width, dtype=np.int32) + int(dx), 0, width - 1
                    )
                    counts[:, :, group_index] += (
                        values[source_y[:, np.newaxis], source_x[np.newaxis, :]]
                        < parameters.sample_threshold
                    )
                continue
            destination, source = overlap
            if boundary_policy is BoundaryPolicy.REPLICATE:
                source_y = np.clip(
                    np.arange(height, dtype=np.int32) + int(dy), 0, height - 1
                )
                source_x = np.clip(
                    np.arange(width, dtype=np.int32) + int(dx), 0, width - 1
                )
                counts[:, :, group_index] += (
                    values[source_y[:, np.newaxis], source_x[np.newaxis, :]]
                    < parameters.sample_threshold
                )
            else:
                counts[destination + (group_index,)] += (
                    values[source] < parameters.sample_threshold
                )
            valid_counts[destination + (group_index,)] += 1
    boundary = np.any(valid_counts != 9, axis=2)
    if boundary_policy is BoundaryPolicy.ERROR and np.any(boundary):
        raise UnboundBehaviorError(
            "decision groups cross the image boundary and no fallback policy was allowed"
        )
    group_fallback = np.any(counts > parameters.count_limit, axis=2)
    if boundary_policy is BoundaryPolicy.FALLBACK:
        group_fallback = np.logical_or(group_fallback, boundary)
    return DecisionPlane(
        group_fallback=np.ascontiguousarray(group_fallback, dtype=bool),
        boundary_unavailable=np.ascontiguousarray(boundary, dtype=bool),
    )


class FallbackMode(str, Enum):
    ORIGINAL = "original"
    CONFIGURED = "configured"


@dataclass(frozen=True)
class FallbackParameters:
    mode: FallbackMode
    configured_rgb: tuple[float, float, float] | None

    def __post_init__(self) -> None:
        if self.mode is FallbackMode.ORIGINAL:
            if self.configured_rgb is not None:
                raise ValueError("original fallback cannot also configure RGB values")
        elif self.configured_rgb is None:
            raise UnboundBehaviorError("configured fallback requires three RGB values")
        else:
            if len(self.configured_rgb) != 3:
                raise ValueError("configured fallback requires three RGB values")
            for index, value in enumerate(self.configured_rgb):
                _finite(value, f"configured fallback channel {index}")


@dataclass(frozen=True)
class ReconstructionResult:
    values: FloatArray
    valid: BoolArray


@runtime_checkable
class Reconstructor(Protocol):
    """Inject reconstruction into the generic batch-rendering interface.

    The exact streaming path uses the recovered implementation directly.
    """

    def reconstruct(
        self,
        analysis: X3AAnalysis,
        requested: BoolArray,
    ) -> ReconstructionResult: ...


def _fallback_canvas(
    analysis: X3AAnalysis,
    parameters: FallbackParameters,
) -> FloatArray:
    if parameters.mode is FallbackMode.ORIGINAL:
        return np.array(
            analysis.working_rgbi[:, :, :3], dtype=np.float32, order="C", copy=True
        )
    assert parameters.configured_rgb is not None
    canvas = np.empty(analysis.weighted_rgb.shape, dtype=np.float32)
    canvas[:] = np.asarray(parameters.configured_rgb, dtype=np.float32)
    return canvas


def render_working_rgb(
    analysis: X3AAnalysis,
    *,
    decision: DecisionPlane,
    driver_force_fallback: npt.ArrayLike,
    fallback: FallbackParameters,
    reconstructor: Reconstructor | None,
) -> FloatArray:
    """Route pixels through explicit fallback or a supplied reconstructor.

    This generic API accepts the row-driver decision as an explicit image-sized
    plane. The exact streaming path derives it from recovered history and the
    selected profile. A missing reconstructor is an error whenever any pixel
    requests repair.
    """

    driver = np.asarray(driver_force_fallback)
    expected_shape = analysis.score.shape
    if driver.dtype != np.dtype(bool) or driver.shape != expected_shape:
        raise ValueError("driver_force_fallback must be a bool HxW plane")
    if decision.group_fallback.shape != expected_shape:
        raise ValueError("decision geometry disagrees with analysis")
    force_fallback = np.logical_or(driver, decision.group_fallback)
    requested = np.ascontiguousarray(~force_fallback, dtype=bool)
    output = _fallback_canvas(analysis, fallback)
    if not np.any(requested):
        return output
    if reconstructor is None:
        raise UnboundBehaviorError(
            "at least one pixel requests the still-unbound reconstruction stage"
        )
    result = reconstructor.reconstruct(analysis, requested)
    values = np.asarray(result.values)
    valid = np.asarray(result.valid)
    if values.dtype != np.dtype(np.float32) or values.shape != output.shape:
        raise ValueError("reconstructor values must be float32 HxWx3")
    if valid.dtype != np.dtype(bool) or valid.shape != expected_shape:
        raise ValueError("reconstructor validity must be bool HxW")
    if not np.all(np.isfinite(values[valid])):
        raise ValueError("valid reconstructed values must be finite")
    accepted = np.logical_and(requested, valid)
    output[accepted] = values[accepted]
    return np.ascontiguousarray(output, dtype=np.float32)
