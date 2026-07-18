"""Deterministic Metal execution of the supported streaming profile.

Hybrid boundary (mirroring the CUDA backend, see docs/metal-backend.md):

- GPU: response LUT indexing, auxiliary/score/weighted planes, decision
  neighborhoods, per-pixel feature records, multiscale candidates, the
  combiner, output conversion, and the producer's per-row / per-epoch sums.
  Apple GPUs have no binary64 hardware, so every binary64 operation runs
  through the software IEEE-754 implementation in :mod:`.kernels`, which is
  integer arithmetic and therefore immune to contraction, reassociation,
  and flush-to-zero by construction.
- CPU: input validation, prepass reduction, producer cross-row finalization,
  stage-parameter resolution, the recovered six-stage hidden startup replay,
  the sequential conditional-dither writer chain (the compiled ``fast_cpu``
  writer on one host core, reused through
  :mod:`..cuda_backend.host_writer`), final hashing, and receipt assembly.

The kernels never approximate: every operation keeps the reference's
widening, rounding, and store schedule, and the backend refuses to run
rather than substitute a different numeric path.  The Metal library is
compiled with fast math disabled; that setting only ever governs the
comparison predicates and raw loads, because every value-producing float
operation goes through the softfloat path.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import numpy as np
import numpy.typing as npt

from ..dither import DitherBounds
from ..output import InverseResponseFactors, emit_public_rgb16
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
from ..x3a import (
    AuxiliaryParameters,
    DecisionParameters,
    ScoreParameters,
    SharedLookupInputResponse,
)
from ..cuda_backend.host_writer import run_writer_chain
from ..cuda_backend.rowparams import derive_row_parameter_table
from .kernels import KERNEL_NAMES, render_kernel_source


class MetalBackendUnavailable(RuntimeError):
    """Metal execution was requested but cannot run exactly here."""


ProgressCallback = Callable[[int, int, int, int], None]

_DEVICE_CACHE: dict[str, Any] = {}
_LIBRARY_CACHE: dict[int, Any] = {}
_PIPELINE_CACHE: dict[tuple[int, str], Any] = {}

_COMMAND_BUFFER_COMPLETED = 4  # MTLCommandBufferStatusCompleted


def _metal():
    try:
        import Metal
    except Exception as error:  # pragma: no cover - import environment
        raise MetalBackendUnavailable(
            f"pyobjc Metal binding is not importable: {error!r}"
        ) from error
    return Metal


def _device():
    metal = _metal()
    device = _DEVICE_CACHE.get("device")
    if device is None:
        device = metal.MTLCreateSystemDefaultDevice()
        if device is None:
            raise MetalBackendUnavailable("no Metal device is visible")
        _DEVICE_CACHE["device"] = device
    return metal, device


def ensure_host_writer_available() -> Any:
    """Return the compiled fast_cpu kernel module or fail closed.

    The writer chain runs on the host through the same compiled path as the
    ``cpu-fast`` backend, so Metal availability requires numba (and that
    module's baked-RNG-constant canary) in addition to a Metal device.
    """

    from ..fast_cpu.engine import CpuFastUnavailable, _kernels

    try:
        return _kernels()
    except CpuFastUnavailable as error:
        raise MetalBackendUnavailable(
            f"Metal writer chain requires the compiled host writer: {error}"
        ) from error


def metal_device_summary() -> dict:
    """Describe the default Metal device, failing closed when absent."""

    import objc

    metal, device = _device()
    options_probe = metal.MTLCompileOptions.alloc().init()
    math_mode = "fastMathEnabled=False"
    if options_probe.respondsToSelector_("setMathMode:"):
        math_mode = "fastMathEnabled=False, mathMode=safe"
    return {
        "name": str(device.name()),
        "has_unified_memory": bool(device.hasUnifiedMemory()),
        "recommended_max_working_set_bytes": int(
            device.recommendedMaxWorkingSetSize()
        ),
        "current_allocated_bytes": int(device.currentAllocatedSize()),
        "max_buffer_bytes": int(device.maxBufferLength()),
        "compile_options": math_mode,
        "pyobjc_version": objc.__version__,
        "binary64_execution": "software IEEE-754 (integer arithmetic)",
    }


def _compile_options(metal):
    options = metal.MTLCompileOptions.alloc().init()
    options.setFastMathEnabled_(False)
    if options.respondsToSelector_("setMathMode:"):
        # MTLMathModeSafe: no contraction, no algebraic rewrites.  The
        # softfloat path is integer arithmetic either way; this governs the
        # remaining float comparisons and loads.
        options.setMathMode_(0)
    return options


def compile_library(source: str):
    """Compile one MSL source string with the backend's exactness options."""

    metal, device = _device()
    library, error = device.newLibraryWithSource_options_error_(
        source, _compile_options(metal), None
    )
    if library is None:
        raise MetalBackendUnavailable(
            f"Metal kernel compilation failed: {error}"
        )
    return library


def get_kernel_library():
    """Compile (once per device) and return the pipeline kernel library."""

    _, device = _device()
    key = int(device.registryID())
    library = _LIBRARY_CACHE.get(key)
    if library is None:
        library = compile_library(render_kernel_source())
        for name in KERNEL_NAMES:
            _pipeline_for(library, name, cache_key=(key, name))
        _LIBRARY_CACHE[key] = library
    return library


def _pipeline_for(library, name: str, *, cache_key=None):
    _, device = _device()
    if cache_key is None:
        cache_key = (int(device.registryID()), name)
    pipeline = _PIPELINE_CACHE.get(cache_key)
    if pipeline is None:
        function = library.newFunctionWithName_(name)
        if function is None:
            raise MetalBackendUnavailable(f"Metal kernel {name} is missing")
        pipeline, error = device.newComputePipelineStateWithFunction_error_(
            function, None
        )
        if pipeline is None:
            raise MetalBackendUnavailable(
                f"Metal pipeline for {name} failed: {error}"
            )
        _PIPELINE_CACHE[cache_key] = pipeline
    return pipeline


def get_pipeline(name: str):
    library = get_kernel_library()
    return _pipeline_for(library, name)


class _Session:
    """One queue plus shared-buffer bookkeeping for a replay run."""

    def __init__(self) -> None:
        self.metal, self.device = _device()
        self.queue = self.device.newCommandQueue()
        if self.queue is None:
            raise MetalBackendUnavailable("Metal command queue creation failed")

    def buffer(self, nbytes: int):
        buf = self.device.newBufferWithLength_options_(
            max(int(nbytes), 16), self.metal.MTLResourceStorageModeShared
        )
        if buf is None:
            raise MetalBackendUnavailable(
                f"Metal buffer allocation failed ({nbytes} bytes)"
            )
        return buf

    def view(self, buf, dtype, shape) -> np.ndarray:
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        raw = buf.contents().as_buffer(nbytes)
        return np.frombuffer(raw, dtype=dtype).reshape(shape)

    def upload(self, array: np.ndarray):
        array = np.ascontiguousarray(array)
        buf = self.buffer(array.nbytes)
        view = self.view(buf, array.dtype, array.shape)
        view[...] = array
        return buf, view

    def alloc(self, dtype, shape, *, zero: bool = False):
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        buf = self.buffer(nbytes)
        view = self.view(buf, dtype, shape)
        if zero:
            view[...] = 0
        return buf, view

    def run(self, launches) -> None:
        """Encode one dispatch per encoder; encoder boundaries order hazards."""

        command_buffer = self.queue.commandBuffer()
        for pipeline, buffers, grid in launches:
            encoder = command_buffer.computeCommandEncoder()
            encoder.setComputePipelineState_(pipeline)
            for index, buf in enumerate(buffers):
                encoder.setBuffer_offset_atIndex_(buf, 0, index)
            width = min(256, int(pipeline.maxTotalThreadsPerThreadgroup()))
            if len(grid) == 1:
                threads = self.metal.MTLSizeMake(grid[0], 1, 1)
                group = self.metal.MTLSizeMake(width, 1, 1)
            else:
                threads = self.metal.MTLSizeMake(grid[0], grid[1], 1)
                group = self.metal.MTLSizeMake(32, max(1, width // 32), 1)
            encoder.dispatchThreads_threadsPerThreadgroup_(threads, group)
            encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()
        if command_buffer.status() != _COMMAND_BUFFER_COMPLETED:
            raise MetalBackendUnavailable(
                f"Metal command buffer failed: {command_buffer.error()}"
            )


class _StageTimer:
    """Optional per-stage wall timing; never alters computation."""

    def __init__(self, sink: dict | None) -> None:
        self._sink = sink

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


def _estimate_memory_bytes(height: int, width: int) -> int:
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


def run_streaming_replay_metal(
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
    _diagnostic_score_plane: npt.NDArray[np.float32] | None = None,
    _diagnostic_at_floor_mask: npt.NDArray[np.bool_] | None = None,
    _diagnostic_changed_mask: npt.NDArray[np.bool_] | None = None,
) -> StreamingReplayResult:
    """Byte-exact Metal mirror of :func:`..streaming.run_streaming_replay`."""

    fast_kernels = ensure_host_writer_available()
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

    summary = metal_device_summary()
    needed = _estimate_memory_bytes(height, width)
    available = (
        summary["recommended_max_working_set_bytes"]
        - summary["current_allocated_bytes"]
    )
    if available < needed:
        raise MetalBackendUnavailable(
            f"insufficient Metal working-set headroom: need ~{needed} bytes, "
            f"available {available}"
        )

    def _check_cancelled() -> None:
        if cancelled is not None and cancelled():
            from ..engine import ProcessingCancelled

            raise ProcessingCancelled("portable correction was cancelled")

    get_kernel_library()
    session = _Session()
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

    # --- device buffers ----------------------------------------------------
    with timer.cpu("upload.inputs"):
        buf_rgbi, _ = session.upload(pixels)
        buf_lut, _ = session.upload(np.ascontiguousarray(response.table))
        buf_aux_alpha, _ = session.upload(table.aux_alpha)
        buf_aux_is_one, _ = session.upload(table.aux_alpha_is_one)
        buf_aux_one_repl, _ = session.upload(table.aux_alpha_one_replacement)
        buf_aux_offset, _ = session.upload(table.aux_offset)
        buf_score_base, _ = session.upload(table.score_base_primary)
        buf_writer_reference, _ = session.upload(table.writer_coarse_reference)
        buf_floor_enabled, _ = session.upload(table.writer_floor_enabled)
        buf_row_gate, _ = session.upload(table.writer_row_gate)
    buf_working, working_view = session.alloc(np.float32, (height, width, 4))
    buf_aux, _aux_view = session.alloc(np.float32, (height, width))
    buf_score, score_view = session.alloc(np.float32, (height, width))
    buf_waux, _waux_view = session.alloc(np.float32, (height, width))
    buf_wrgb, _wrgb_view = session.alloc(np.float32, (height, width, 3))
    buf_eligible, eligible_view = session.alloc(np.uint8, (total,))
    buf_error_flags, error_flags_view = session.alloc(np.uint32, (1,), zero=True)

    horizontal_minimum = int(
        score_parameters.horizontal_minimum_resolution_cutoff
        < score_parameters.resolution_metric
    )

    def _i32(*values: int):
        return session.upload(np.asarray(values, dtype=np.int32))[0]

    def _f32(*values: float):
        return session.upload(np.asarray(values, dtype=np.float32))[0]

    def _q64(*values: float):
        bits = np.asarray(values, dtype=np.float64).view(np.uint64)
        return session.upload(bits)[0]

    with timer.cpu("device.analysis-planes"):
        session.run(
            [
                (
                    get_pipeline("k_convert_and_auxiliary"),
                    [
                        buf_rgbi,
                        buf_lut,
                        buf_aux_alpha,
                        buf_aux_is_one,
                        buf_aux_one_repl,
                        buf_aux_offset,
                        buf_working,
                        buf_aux,
                        _i32(
                            auxiliary_parameters.selected_visible_channel,
                            height,
                            width,
                        ),
                    ],
                    (total,),
                ),
                (
                    get_pipeline("k_score_and_weighted"),
                    [
                        buf_aux,
                        buf_working,
                        buf_score_base,
                        buf_score,
                        buf_waux,
                        buf_wrgb,
                        buf_error_flags,
                        _q64(
                            float(np.float32(score_parameters.base_addend)),
                            float(np.float32(score_parameters.scale)),
                            float(np.float32(score_parameters.offset)),
                            float(np.float32(score_parameters.floor)),
                        ),
                        _i32(horizontal_minimum, height, width),
                    ],
                    (total,),
                ),
                (
                    get_pipeline("k_decision_eligibility"),
                    [
                        buf_aux,
                        buf_score,
                        buf_row_gate,
                        buf_floor_enabled,
                        buf_eligible,
                        _f32(np.float32(decision_parameters.sample_threshold)),
                        _i32(
                            decision_parameters.count_limit,
                            decision_parameters.perpendicular_radius,
                            height,
                            width,
                        ),
                    ],
                    (total,),
                ),
            ]
        )
    _check_cancelled()

    with timer.cpu("host.compact-selected"):
        selected = np.flatnonzero(eligible_view).astype(np.int64)
        selected_count = int(selected.size)
        buf_selected, _ = session.upload(
            selected if selected_count else np.zeros(1, dtype=np.int64)
        )

    buf_attempted, attempted_view = session.alloc(
        np.uint8, (max(selected_count, 1),), zero=True
    )
    buf_candidate, candidate_view = session.alloc(
        np.float64, (max(selected_count, 1) * 3,), zero=True
    )
    buf_original, _original_view = session.alloc(
        np.float32, (max(selected_count, 1) * 3,), zero=True
    )
    with timer.cpu("device.features-combine"):
        if selected_count:
            session.run(
                [
                    (
                        get_pipeline("k_features_and_combine"),
                        [
                            buf_selected,
                            buf_score,
                            buf_waux,
                            buf_wrgb,
                            buf_aux,
                            buf_working,
                            buf_writer_reference,
                            buf_floor_enabled,
                            buf_row_gate,
                            session.upload(
                                np.asarray(
                                    reconstruction_parameters.coarse_slopes,
                                    dtype=np.float32,
                                )
                            )[0],
                            session.upload(
                                np.asarray(
                                    reconstruction_parameters.band_enabled,
                                    dtype=np.uint8,
                                )
                            )[0],
                            session.upload(
                                np.asarray(
                                    reconstruction_parameters.band_scales,
                                    dtype=np.float32,
                                )
                            )[0],
                            session.upload(
                                np.asarray(
                                    reconstruction_parameters.factors_a,
                                    dtype=np.float32,
                                ).reshape(-1)
                            )[0],
                            session.upload(
                                np.asarray(
                                    reconstruction_parameters.factors_b,
                                    dtype=np.float32,
                                ).reshape(-1)
                            )[0],
                            session.upload(
                                np.asarray(
                                    reconstruction_parameters.configured_strengths,
                                    dtype=np.float32,
                                )
                            )[0],
                            buf_attempted,
                            buf_candidate,
                            buf_original,
                            _i32(
                                selected_count,
                                height,
                                width,
                                int(mode is FeatureBandExtremaMode.CROSS_NEIGHBOR),
                                int(bool(reconstruction_parameters.coarse_enabled)),
                            ),
                            _f32(np.float32(score_parameters.floor)),
                        ],
                        (selected_count,),
                    )
                ]
            )
    _check_cancelled()

    # The writer chain runs on one host CPU core via the compiled fast_cpu
    # path, exactly as the CUDA backend does: the per-selected-site
    # attempted/candidate arrays feed the same write_band already proven
    # byte-exact against this reference.  Unified memory makes the transfer
    # a view, not a copy.
    low64 = float(np.float32(dither_bounds.low))
    high64 = float(np.float32(dither_bounds.high))
    low_lt_high = low64 < high64
    dither_scales_array = np.asarray(
        reconstruction_parameters.dither_scales, dtype=np.float32
    )
    state_in = int(active_generator.state)
    with timer.cpu("host.writer-chain"):
        (
            values_at_selected,
            written_at_selected,
            advances,
            final_state,
        ) = run_writer_chain(
            fast_kernels,
            selected=selected,
            attempted=attempted_view[:selected_count],
            candidate=candidate_view[: selected_count * 3].reshape(
                selected_count, 3
            ),
            working_all=working_view,
            floor_enabled_rows=table.writer_floor_enabled,
            width=width,
            state_in=state_in,
            low64=low64,
            high64=high64,
            low_lt_high=low_lt_high,
            dither_scales=dither_scales_array,
        )
        buf_values, _ = session.upload(
            np.ascontiguousarray(values_at_selected.reshape(-1), dtype=np.float32)
            if selected_count
            else np.zeros(3, dtype=np.float32)
        )
        buf_written, _ = session.upload(
            written_at_selected
            if selected_count
            else np.zeros(1, dtype=np.uint8)
        )
    _check_cancelled()

    buf_work_output, _work_output_view = session.alloc(
        np.float32, (height, width, 3)
    )
    buf_out, out_view = session.alloc(np.uint16, (height, width, 3))
    buf_factor_high, _ = session.upload(factors.high)
    buf_factor_low, _ = session.upload(factors.low)
    buf_counters, counters_view = session.alloc(np.uint32, (3,), zero=True)
    with timer.cpu("device.assemble-emit"):
        launches = [
            (
                get_pipeline("k_copy_visible"),
                [buf_working, buf_work_output, _i32(total)],
                (total,),
            )
        ]
        if selected_count:
            launches.append(
                (
                    get_pipeline("k_scatter_values"),
                    [
                        buf_selected,
                        buf_values,
                        buf_work_output,
                        buf_error_flags,
                        _i32(selected_count),
                    ],
                    (selected_count,),
                )
            )
        launches.append(
            (
                get_pipeline("k_emit_rgb16"),
                [
                    buf_work_output,
                    buf_factor_high,
                    buf_factor_low,
                    buf_out,
                    _i32(total * 3),
                ],
                (total * 3,),
            )
        )
        if selected_count:
            launches.append(
                (
                    get_pipeline("k_site_counters"),
                    [
                        buf_attempted,
                        buf_values,
                        buf_original,
                        buf_written,
                        buf_factor_high,
                        buf_factor_low,
                        buf_counters,
                        _i32(selected_count),
                    ],
                    (selected_count,),
                )
            )
        session.run(launches)
    _check_cancelled()

    error_flags = int(error_flags_view[0])
    if error_flags & 1:
        raise ValueError("auxiliary score input must be a finite HxW plane")
    if error_flags & 2:
        raise ValueError("work RGB must be finite")
    with timer.cpu("download.output"):
        host_out = np.array(out_view)
    counters = np.array(counters_view)

    if _diagnostic_score_plane is not None:
        if _diagnostic_at_floor_mask is None or _diagnostic_changed_mask is None:
            raise ValueError("all Metal diagnostic buffers must be provided")
        with timer.cpu("download.diagnostics"):
            _diagnostic_score_plane[...] = score_view
            np.equal(
                _diagnostic_score_plane,
                np.float32(score_parameters.floor),
                out=_diagnostic_at_floor_mask,
            )
            bytes_per_working_row = width * 4 * np.dtype(np.float32).itemsize
            rows_per_chunk = max(
                1,
                min(height, (8 << 20) // bytes_per_working_row),
            )
            for row_start in range(0, height, rows_per_chunk):
                row_stop = min(height, row_start + rows_per_chunk)
                working_chunk = np.array(working_view[row_start:row_stop])
                noop_chunk = emit_public_rgb16(
                    working_chunk[:, :, :3],
                    factors=factors,
                )
                np.any(
                    host_out[row_start:row_stop] != noop_chunk,
                    axis=2,
                    out=_diagnostic_changed_mask[row_start:row_stop],
                )
                _check_cancelled()

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
