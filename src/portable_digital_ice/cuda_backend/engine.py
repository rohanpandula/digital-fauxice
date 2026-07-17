"""Deterministic CUDA execution of the supported streaming profile.

Hybrid boundary (measured, see docs/cuda-backend.md):

- GPU: response LUT indexing, auxiliary/score/weighted planes, decision
  neighborhoods, per-pixel feature records, multiscale candidates, the
  combiner, the sequential conditional-dither writer chain (one device thread
  carrying the 24-bit LCG), output conversion, and the producer's per-row /
  per-epoch sums.
- CPU: input validation, prepass reduction, producer cross-row finalization,
  stage-parameter resolution, the recovered six-stage hidden startup replay,
  final hashing, and receipt assembly.

The kernels never approximate: every operation keeps the reference's widening,
rounding, and store schedule, and the backend refuses to run rather than
substitute a different numeric path.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from contextlib import contextmanager

import numpy as np
import numpy.typing as npt

from ..dither import DitherBounds
from ..reconstruction import (
    FeatureBandExtremaMode,
    ReconstructionParameters,
    feature_band_extrema_mode,
)
from ..rng import LCG24
from ..stage_parameters import StageParameterProvider
from ..streaming import (
    StreamingReplayResult,
    _RowCache,
    _startup_replay,
)
from ..output import InverseResponseFactors
from ..x3a import (
    AuxiliaryParameters,
    DecisionParameters,
    ScoreParameters,
    SharedLookupInputResponse,
)
from .kernels import NVRTC_OPTIONS, render_kernel_source
from .rowparams import derive_row_parameter_table


class CudaBackendUnavailable(RuntimeError):
    """CUDA execution was requested but cannot run exactly here."""


ProgressCallback = Callable[[int, int, int, int], None]

_MODULE_CACHE: dict[int, object] = {}


def _cupy():
    try:
        import cupy
    except Exception as error:  # pragma: no cover - import environment
        raise CudaBackendUnavailable(
            f"cupy is not importable: {error!r}"
        ) from error
    return cupy


def cuda_device_summary() -> dict:
    """Describe the active CUDA device, failing closed when absent."""

    cp = _cupy()
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except Exception as error:
        raise CudaBackendUnavailable(
            f"CUDA runtime reports no usable device: {error!r}"
        ) from error
    if count < 1:
        raise CudaBackendUnavailable("no CUDA device is visible")
    properties = cp.cuda.runtime.getDeviceProperties(0)
    free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
    return {
        "name": properties["name"].decode(),
        "compute_capability": f"{properties['major']}.{properties['minor']}",
        "total_vram_bytes": int(properties["totalGlobalMem"]),
        "free_vram_bytes": int(free_bytes),
        "driver_version": int(cp.cuda.runtime.driverGetVersion()),
        "runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        "cupy_version": cp.__version__,
        "nvrtc_options": list(NVRTC_OPTIONS),
    }


def get_kernel_module():
    """Compile (once per device) and return the RawModule."""

    cp = _cupy()
    device = cp.cuda.Device()
    key = int(device.id)
    module = _MODULE_CACHE.get(key)
    if module is None:
        module = cp.RawModule(
            code=render_kernel_source(), options=NVRTC_OPTIONS
        )
        module.compile()
        _MODULE_CACHE[key] = module
    return module


class _StageTimer:
    """Optional per-stage wall/device timing; never alters computation."""

    def __init__(self, sink: dict | None) -> None:
        self._sink = sink
        self._gpu_events: list[tuple[str, object, object]] = []

    @contextmanager
    def cpu(self, label: str):
        if self._sink is None:
            yield
            return
        started = time.monotonic()
        try:
            yield
        finally:
            self._sink[label] = self._sink.get(label, 0.0) + (
                time.monotonic() - started
            )

    @contextmanager
    def gpu(self, label: str):
        if self._sink is None:
            yield
            return
        import cupy as cp

        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        try:
            yield
        finally:
            end.record()
            self._gpu_events.append((label, start, end))

    def finalize(self) -> None:
        if self._sink is None or not self._gpu_events:
            return
        import cupy as cp

        cp.cuda.Device().synchronize()
        for label, start, end in self._gpu_events:
            elapsed = cp.cuda.get_elapsed_time(start, end) / 1000.0
            key = f"{label}.device"
            self._sink[key] = self._sink.get(key, 0.0) + elapsed
        self._gpu_events.clear()


def _estimate_vram_bytes(height: int, width: int) -> int:
    pixels = height * width
    planes = (
        pixels * 4 * 2  # rgbi uint16
        + pixels * 4 * 4  # working float32 x4
        + pixels * 4 * 3  # auxiliary + score + weighted auxiliary
        + pixels * 4 * 3  # weighted rgb
        + pixels * 4 * 3  # working output
        + pixels * 1  # eligibility
        + pixels * 2 * 3  # rgb16 output
    )
    per_site = 8 + 1 + 24 + 12 + 12 + 1  # index, attempted, cand, orig, values, written
    return planes + pixels * per_site // 2 + (64 << 20)


def run_streaming_replay_cuda(
    main_rgbi: npt.ArrayLike,
    output_rgb16: npt.NDArray[np.uint16],
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
    cancelled: Callable[[], bool] | None = None,
    stage_timings: dict | None = None,
) -> StreamingReplayResult:
    """Byte-exact CUDA mirror of :func:`..streaming.run_streaming_replay`."""

    cp = _cupy()
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

    summary = cuda_device_summary()
    needed = _estimate_vram_bytes(height, width)
    if summary["free_vram_bytes"] < needed:
        raise CudaBackendUnavailable(
            f"insufficient free VRAM: need ~{needed} bytes, "
            f"free {summary['free_vram_bytes']}"
        )

    def _check_cancelled() -> None:
        if cancelled is not None and cancelled():
            from ..engine import ProcessingCancelled

            raise ProcessingCancelled("portable correction was cancelled")

    module = get_kernel_module()
    timer = _StageTimer(stage_timings)

    # --- host-side exact preludes -----------------------------------------
    with timer.cpu("host.row-parameter-table"):
        table = derive_row_parameter_table(
            height,
            auxiliary_parameters=auxiliary_parameters,
            score_parameters=score_parameters,
            reconstruction_parameters=reconstruction_parameters,
            stage_parameter_provider=stage_parameter_provider,
        )
    cache = _RowCache(
        pixels,
        response=response,
        auxiliary_parameters=auxiliary_parameters,
        score_parameters=score_parameters,
        stage_parameter_provider=stage_parameter_provider,
    )
    active_generator = generator or LCG24.from_nikon_pe_initial_state()
    with timer.cpu("host.startup-replay"):
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
    _check_cancelled()
    if progress is not None:
        progress(0, height, 0, 0)

    mode = feature_band_extrema_mode(
        resolution_metric=reconstruction_parameters.resolution_metric,
        cross_neighbor_cutoff=reconstruction_parameters.cross_neighbor_cutoff,
    )
    factors = InverseResponseFactors.recovered_16bit()

    total = height * width
    threads = 256
    grid_pixels = ((total + threads - 1) // threads,)

    # --- device buffers ----------------------------------------------------
    with timer.gpu("upload.inputs"):
        gpu_rgbi = cp.asarray(np.ascontiguousarray(pixels))
        gpu_lut = cp.asarray(response.table)
    gpu_working = cp.empty((height, width, 4), dtype=cp.float32)
    gpu_aux = cp.empty((height, width), dtype=cp.float32)
    gpu_score = cp.empty((height, width), dtype=cp.float32)
    gpu_waux = cp.empty((height, width), dtype=cp.float32)
    gpu_wrgb = cp.empty((height, width, 3), dtype=cp.float32)
    gpu_eligible = cp.empty(total, dtype=cp.uint8)
    gpu_error_flags = cp.zeros(1, dtype=cp.uint32)

    gpu_aux_alpha = cp.asarray(table.aux_alpha)
    gpu_aux_is_one = cp.asarray(table.aux_alpha_is_one)
    gpu_aux_one_repl = cp.asarray(table.aux_alpha_one_replacement)
    gpu_aux_offset = cp.asarray(table.aux_offset)
    gpu_score_base = cp.asarray(table.score_base_primary)
    gpu_writer_reference = cp.asarray(table.writer_coarse_reference)
    gpu_floor_enabled = cp.asarray(table.writer_floor_enabled)
    gpu_row_gate = cp.asarray(table.writer_row_gate)

    with timer.gpu("kernel.convert-auxiliary"):
        module.get_function("k_convert_and_auxiliary")(
            grid_pixels,
            (threads,),
            (
                gpu_rgbi,
                gpu_lut,
                gpu_aux_alpha,
                gpu_aux_is_one,
                gpu_aux_one_repl,
                gpu_aux_offset,
                np.int32(auxiliary_parameters.selected_visible_channel),
                np.int32(height),
                np.int32(width),
                gpu_working,
                gpu_aux,
            ),
        )
    horizontal_minimum = int(
        score_parameters.horizontal_minimum_resolution_cutoff
        < score_parameters.resolution_metric
    )
    with timer.gpu("kernel.score-weighted"):
        module.get_function("k_score_and_weighted")(
        grid_pixels,
        (threads,),
        (
            gpu_aux,
            gpu_working,
            gpu_score_base,
            np.float64(np.float32(score_parameters.base_addend)),
            np.float64(np.float32(score_parameters.scale)),
            np.float64(np.float32(score_parameters.offset)),
            np.float64(np.float32(score_parameters.floor)),
            np.int32(horizontal_minimum),
            np.int32(height),
            np.int32(width),
            gpu_score,
            gpu_waux,
            gpu_wrgb,
            gpu_error_flags,
        ),
    )
    with timer.gpu("kernel.decision-eligibility"):
        module.get_function("k_decision_eligibility")(
        grid_pixels,
        (threads,),
        (
            gpu_aux,
            gpu_score,
            np.float32(decision_parameters.sample_threshold),
            np.int32(decision_parameters.count_limit),
            np.int32(decision_parameters.perpendicular_radius),
            gpu_row_gate,
            gpu_floor_enabled,
            np.int32(height),
            np.int32(width),
            gpu_eligible,
        ),
    )
    _check_cancelled()

    with timer.gpu("kernel.compact-selected"):
        selected = cp.flatnonzero(gpu_eligible).astype(cp.int64)
        selected_count = int(selected.size)

    gpu_attempted = cp.zeros(max(selected_count, 1), dtype=cp.uint8)
    gpu_candidate = cp.zeros(max(selected_count, 1) * 3, dtype=cp.float64)
    gpu_original = cp.zeros(max(selected_count, 1) * 3, dtype=cp.float32)
    with timer.gpu("kernel.features-combine"):
      if selected_count:
        module.get_function("k_features_and_combine")(
            ((selected_count + threads - 1) // threads,),
            (threads,),
            (
                selected,
                np.int64(selected_count),
                gpu_score,
                gpu_waux,
                gpu_wrgb,
                gpu_aux,
                gpu_working,
                np.int32(height),
                np.int32(width),
                np.float32(score_parameters.floor),
                gpu_writer_reference,
                gpu_floor_enabled,
                gpu_row_gate,
                np.int32(mode is FeatureBandExtremaMode.CROSS_NEIGHBOR),
                np.int32(bool(reconstruction_parameters.coarse_enabled)),
                cp.asarray(
                    np.asarray(
                        reconstruction_parameters.coarse_slopes, dtype=np.float32
                    )
                ),
                cp.asarray(
                    np.asarray(
                        reconstruction_parameters.band_enabled, dtype=np.uint8
                    )
                ),
                cp.asarray(
                    np.asarray(
                        reconstruction_parameters.band_scales, dtype=np.float32
                    )
                ),
                cp.asarray(
                    np.asarray(
                        reconstruction_parameters.factors_a, dtype=np.float32
                    ).reshape(-1)
                ),
                cp.asarray(
                    np.asarray(
                        reconstruction_parameters.factors_b, dtype=np.float32
                    ).reshape(-1)
                ),
                cp.asarray(
                    np.asarray(
                        reconstruction_parameters.configured_strengths,
                        dtype=np.float32,
                    )
                ),
                gpu_attempted,
                gpu_candidate,
                gpu_original,
            ),
        )
    _check_cancelled()

    gpu_values = cp.zeros(max(selected_count, 1) * 3, dtype=cp.float32)
    gpu_written = cp.zeros(max(selected_count, 1), dtype=cp.uint8)
    gpu_advances = cp.zeros(1, dtype=cp.uint64)
    gpu_state_out = cp.zeros(1, dtype=cp.uint32)
    scales = [np.float32(value) for value in reconstruction_parameters.dither_scales]
    with timer.gpu("kernel.writer-chain"):
        module.get_function("k_writer_chain")(
        (1,),
        (1,),
        (
            selected,
            np.int64(selected_count),
            gpu_attempted,
            gpu_candidate,
            gpu_original,
            gpu_floor_enabled,
            np.int32(width),
            np.uint32(active_generator.state),
            np.float32(dither_bounds.low),
            np.float32(dither_bounds.high),
            scales[0],
            scales[1],
            scales[2],
            gpu_values,
            gpu_written,
            gpu_advances,
            gpu_state_out,
        ),
    )
    _check_cancelled()

    gpu_work_output = cp.empty((height, width, 3), dtype=cp.float32)
    with timer.gpu("kernel.assemble-emit"):
        module.get_function("k_copy_visible")(
        grid_pixels,
        (threads,),
        (gpu_working, np.int64(total), gpu_work_output),
    )
    if selected_count:
        module.get_function("k_scatter_values")(
            ((selected_count + threads - 1) // threads,),
            (threads,),
            (
                selected,
                np.int64(selected_count),
                gpu_values,
                gpu_work_output,
                gpu_error_flags,
            ),
        )

    gpu_factor_high = cp.asarray(factors.high)
    gpu_factor_low = cp.asarray(factors.low)
    gpu_out = cp.empty((height, width, 3), dtype=cp.uint16)
    module.get_function("k_emit_rgb16")(
        (((total * 3) + threads - 1) // threads,),
        (threads,),
        (
            gpu_work_output,
            gpu_factor_high,
            gpu_factor_low,
            np.int64(total * 3),
            gpu_out,
        ),
    )
    gpu_counters = cp.zeros(3, dtype=cp.uint64)
    if selected_count:
        module.get_function("k_site_counters")(
            ((selected_count + threads - 1) // threads,),
            (threads,),
            (
                gpu_attempted,
                gpu_values,
                gpu_original,
                gpu_written,
                np.int64(selected_count),
                gpu_factor_high,
                gpu_factor_low,
                gpu_counters,
            ),
        )
    _check_cancelled()

    timer.finalize()
    error_flags = int(cp.asnumpy(gpu_error_flags)[0])
    if error_flags & 1:
        raise ValueError("auxiliary score input must be a finite HxW plane")
    if error_flags & 2:
        raise ValueError("work RGB must be finite")
    with timer.cpu("download.output"):
        host_out = cp.asnumpy(gpu_out)
    counters = cp.asnumpy(gpu_counters)
    advances = int(cp.asnumpy(gpu_advances)[0])
    final_state = int(cp.asnumpy(gpu_state_out)[0])

    output[:] = host_out
    with timer.cpu("host.sha256"):
        output_hash = hashlib.sha256(
            host_out.astype("<u2", copy=False).tobytes(order="C")
        ).hexdigest()
    active_generator.state = final_state
    if progress is not None:
        progress(height, height, int(counters[0]), int(counters[1]))

    return StreamingReplayResult(
        shape=(height, width, 3),
        startup=startup,
        attempted_pixels=int(counters[0]),
        written_pixels=int(counters[1]),
        public_rng_advances=advances,
        final_rng_state=final_state,
        output_sha256=output_hash,
        changed_pixels=int(counters[2]),
    )
