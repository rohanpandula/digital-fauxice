"""Level 1: CUDA primitive parity, compared as raw bit patterns.

Every test skips with a clear reason when CUDA is unavailable.  Comparisons
use integer/IEEE-754 bit views, never decimal tolerances.
"""

from __future__ import annotations

import numpy as np
import pytest

cupy = pytest.importorskip("cupy", reason="CUDA backend requires cupy")

from portable_digital_ice.cuda_backend.engine import (  # noqa: E402
    CudaBackendUnavailable,
    cuda_device_summary,
)
from portable_digital_ice.cuda_backend.kernels import (  # noqa: E402
    NVRTC_OPTIONS,
    render_kernel_source,
)
from portable_digital_ice.dither import DitherBounds, conditional_dither_delta  # noqa: E402
from portable_digital_ice.output import emit_public_rgb16  # noqa: E402
from portable_digital_ice.reconstruction import (  # noqa: E402
    _recovered_rgb_unscaled_averages,
    _recovered_unscaled_averages,
)
from portable_digital_ice.rng import LCG24  # noqa: E402
from portable_digital_ice.x3a import (  # noqa: E402
    AuxiliaryParameters,
    ScoreParameters,
    SharedLookupInputResponse,
    continuous_score,
    derive_auxiliary,
)


def _require_device() -> None:
    try:
        cuda_device_summary()
    except CudaBackendUnavailable as error:
        pytest.skip(f"CUDA device unavailable: {error}")


TEST_WRAPPERS = r"""
extern "C" __global__ void t_unscaled_averages(
    const float* patches, i64 count, double* out) {
  i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  float p[9][9];
  for (int y = 0; y < 9; ++y)
    for (int x = 0; x < 9; ++x) p[y][x] = patches[(i * 81) + y * 9 + x];
  double q[3];
  unscaled_averages_scalar(p, q);
  out[i * 3 + 0] = q[0];
  out[i * 3 + 1] = q[1];
  out[i * 3 + 2] = q[2];
}

extern "C" __global__ void t_rgb_unscaled_averages(
    const float* patches, i64 count, double* out) {
  i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  float p[9][9][3];
  for (int y = 0; y < 9; ++y)
    for (int x = 0; x < 9; ++x)
      for (int c = 0; c < 3; ++c)
        p[y][x][c] = patches[(((i * 9) + y) * 9 + x) * 3 + c];
  double q[3][3];
  rgb_unscaled_averages(p, q);
  for (int s = 0; s < 3; ++s)
    for (int c = 0; c < 3; ++c) out[(i * 3 + s) * 3 + c] = q[s][c];
}

extern "C" __global__ void t_dither_chain(
    const double* values, const float* scales, i64 count, u32 state_in,
    float low32, float high32, double* deltas, u32* state_out,
    unsigned long long* advances_out) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  u32 state = state_in;
  u64 advances = 0;
  double low = (double)low32, high = (double)high32;
  bool low_lt_high = low < high;
  for (i64 i = 0; i < count; ++i) {
    deltas[i] = dither_delta(values[i], low, high, low_lt_high, scales[i],
                             &state, &advances);
  }
  *state_out = state;
  *advances_out = advances;
}

extern "C" __global__ void t_emit(const float* values, i64 count, u16* out,
                                  const u32* factor_high,
                                  const u32* factor_low) {
  i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  out[i] = emit_one(values[i], factor_high, factor_low);
}
"""


@pytest.fixture(scope="module")
def module():
    _require_device()
    raw = cupy.RawModule(
        code=render_kernel_source() + TEST_WRAPPERS, options=NVRTC_OPTIONS
    )
    raw.compile()
    return raw


def test_device_summary_reports_capability():
    _require_device()
    summary = cuda_device_summary()
    assert summary["total_vram_bytes"] > 0
    assert "--fmad=false" in summary["nvrtc_options"]


def test_response_lut_indexing_and_auxiliary_bits(module):
    rng = np.random.default_rng(1)
    height, width = 37, 53
    pixels = rng.integers(0, 65536, size=(height, width, 4), dtype=np.uint16)
    response = SharedLookupInputResponse.nikon_logarithmic()
    parameters = AuxiliaryParameters(
        selected_visible_channel=0,
        alpha=float(np.float32(0.4375)),
        calibration_offset=1.0,
        alpha_one_replacement=None,
    )
    working_cpu = response.convert(pixels)
    aux_cpu = derive_auxiliary(working_cpu, parameters)

    gpu_working = cupy.empty((height, width, 4), dtype=cupy.float32)
    gpu_aux = cupy.empty((height, width), dtype=cupy.float32)
    alpha_rows = cupy.asarray(np.full(height, np.float32(0.4375), dtype=np.float32))
    is_one = cupy.zeros(height, dtype=cupy.uint8)
    replacement = cupy.zeros(height, dtype=cupy.float32)
    offset_rows = cupy.asarray(np.full(height, np.float32(1.0), dtype=np.float32))
    total = height * width
    module.get_function("k_convert_and_auxiliary")(
        ((total + 255) // 256,),
        (256,),
        (
            cupy.asarray(pixels),
            cupy.asarray(response.table),
            alpha_rows,
            is_one,
            replacement,
            offset_rows,
            np.int32(0),
            np.int32(height),
            np.int32(width),
            gpu_working,
            gpu_aux,
        ),
    )
    assert np.array_equal(
        cupy.asnumpy(gpu_working).view(np.uint32), working_cpu.view(np.uint32)
    )
    assert np.array_equal(
        cupy.asnumpy(gpu_aux).view(np.uint32), aux_cpu.view(np.uint32)
    )


def test_score_equation_bits(module):
    rng = np.random.default_rng(2)
    height, width = 23, 61
    aux = rng.uniform(-5000.0, 66000.0, size=(height, width)).astype(np.float32)
    working = rng.uniform(0.0, 65535.0, size=(height, width, 4)).astype(np.float32)
    parameters = ScoreParameters(
        base_primary=float(np.float32(52011.4)),
        base_addend=float(np.frombuffer(np.uint32(0xC2EED000).tobytes(), dtype=np.float32)[0]),
        scale=float(np.frombuffer(np.uint32(0xBA887952).tobytes(), dtype=np.float32)[0]),
        offset=1.0,
        floor=float(np.frombuffer(np.uint32(0x3CA3D70A).tobytes(), dtype=np.float32)[0]),
        resolution_metric=4000,
        horizontal_minimum_resolution_cutoff=550,
    )
    score_cpu = continuous_score(aux, parameters)

    gpu_score = cupy.empty((height, width), dtype=cupy.float32)
    gpu_waux = cupy.empty((height, width), dtype=cupy.float32)
    gpu_wrgb = cupy.empty((height, width, 3), dtype=cupy.float32)
    error_flags = cupy.zeros(1, dtype=cupy.uint32)
    base_rows = cupy.asarray(
        np.full(height, np.float32(parameters.base_primary), dtype=np.float32)
    )
    total = height * width
    module.get_function("k_score_and_weighted")(
        ((total + 255) // 256,),
        (256,),
        (
            cupy.asarray(aux),
            cupy.asarray(working),
            base_rows,
            np.float64(np.float32(parameters.base_addend)),
            np.float64(np.float32(parameters.scale)),
            np.float64(np.float32(parameters.offset)),
            np.float64(np.float32(parameters.floor)),
            np.int32(1),
            np.int32(height),
            np.int32(width),
            gpu_score,
            gpu_waux,
            gpu_wrgb,
            error_flags,
        ),
    )
    assert int(error_flags.get()[0]) == 0
    assert np.array_equal(
        cupy.asnumpy(gpu_score).view(np.uint32), score_cpu.view(np.uint32)
    )
    waux_cpu = np.multiply(score_cpu, aux, dtype=np.float32)
    wrgb_cpu = np.multiply(score_cpu[:, :, None], working[:, :, :3], dtype=np.float32)
    assert np.array_equal(
        cupy.asnumpy(gpu_waux).view(np.uint32), waux_cpu.view(np.uint32)
    )
    assert np.array_equal(
        cupy.asnumpy(gpu_wrgb).view(np.uint32), wrgb_cpu.view(np.uint32)
    )


def test_unscaled_averages_bits(module):
    rng = np.random.default_rng(3)
    count = 4096
    patches = rng.uniform(0.0, 66000.0, size=(count, 9, 9)).astype(np.float32)
    patches[0] = 0.0
    patches[1] = 65535.0
    patches[2, :, :] = rng.uniform(0.0, 1.0, size=(9, 9)).astype(np.float32)
    expected = np.stack(
        [_recovered_unscaled_averages(patch)[:, 0] for patch in patches]
    )
    out = cupy.empty((count, 3), dtype=cupy.float64)
    module.get_function("t_unscaled_averages")(
        ((count + 127) // 128,),
        (128,),
        (cupy.asarray(patches.reshape(count, -1)), np.int64(count), out),
    )
    assert np.array_equal(
        cupy.asnumpy(out).view(np.uint64), expected.view(np.uint64)
    )


def test_rgb_candidate_averages_bits(module):
    rng = np.random.default_rng(4)
    count = 2048
    patches = rng.uniform(0.0, 66000.0, size=(count, 9, 9, 3)).astype(np.float32)
    expected = np.stack([_recovered_rgb_unscaled_averages(patch) for patch in patches])
    out = cupy.empty((count, 3, 3), dtype=cupy.float64)
    module.get_function("t_rgb_unscaled_averages")(
        ((count + 127) // 128,),
        (128,),
        (cupy.asarray(patches.reshape(count, -1)), np.int64(count), out),
    )
    assert np.array_equal(
        cupy.asnumpy(out).view(np.uint64), expected.view(np.uint64)
    )


def test_conditional_dither_chain_bits(module):
    response = SharedLookupInputResponse.nikon_logarithmic()
    bounds = DitherBounds.from_lookup(response.table, maximum_index=65535)
    rng = np.random.default_rng(5)
    count = 50000
    low = float(np.float32(bounds.low))
    high = float(np.float32(bounds.high))
    values = np.concatenate(
        [
            rng.uniform(low - 100.0, high + 100.0, size=count - 6),
            np.array(
                [low, high, np.nextafter(low, high), np.nextafter(high, low), 0.0, 65535.0]
            ),
        ]
    ).astype(np.float64)
    scales = rng.choice(
        np.array(
            [np.float32(0.015), np.float32(0.025), np.float32(0.0)], dtype=np.float32
        ),
        size=count,
    ).astype(np.float32)

    generator = LCG24.from_nikon_pe_initial_state()
    expected = np.empty(count, dtype=np.float64)
    for index in range(count):
        expected[index] = conditional_dither_delta(
            values[index],
            bounds=bounds,
            scale=float(scales[index]),
            generator=generator,
        )
    deltas = cupy.empty(count, dtype=cupy.float64)
    state_out = cupy.zeros(1, dtype=cupy.uint32)
    advances = cupy.zeros(1, dtype=cupy.uint64)
    module.get_function("t_dither_chain")(
        (1,),
        (1,),
        (
            cupy.asarray(values),
            cupy.asarray(scales),
            np.int64(count),
            np.uint32(0x3045),
            np.float32(bounds.low),
            np.float32(bounds.high),
            deltas,
            state_out,
            advances,
        ),
    )
    assert np.array_equal(
        cupy.asnumpy(deltas).view(np.uint64), expected.view(np.uint64)
    )
    assert int(state_out.get()[0]) == generator.state


def test_output_conversion_bits(module):
    from portable_digital_ice.output import InverseResponseFactors

    rng = np.random.default_rng(6)
    values = np.concatenate(
        [
            rng.uniform(-10.0, 65545.0, size=100000),
            np.array([-0.5, -0.4999999, 0.0, 0.49999997, 0.5, 65534.5, 65535.0, 65540.0]),
        ]
    ).astype(np.float32)
    expected = emit_public_rgb16(values.reshape(1, -1, 1).repeat(3, axis=2))[0, :, 0]
    factors = InverseResponseFactors.recovered_16bit()
    out = cupy.empty(values.size, dtype=cupy.uint16)
    module.get_function("t_emit")(
        ((values.size + 255) // 256,),
        (256,),
        (
            cupy.asarray(values),
            np.int64(values.size),
            out,
            cupy.asarray(factors.high),
            cupy.asarray(factors.low),
        ),
    )
    assert np.array_equal(cupy.asnumpy(out), expected)


def test_nonfinite_auxiliary_fails_closed_like_cpu(module):
    """A NaN auxiliary plane raises on CPU; the kernel must flag, not launder.

    Unreachable through the public API (uint16 inputs, hash-pinned finite
    LUT, finiteness-validated calibration), but the C ternary clamps would
    otherwise silently convert NaN into a bound where NumPy propagates it
    into the reference's fail-closed validation.
    """

    height, width = 5, 16
    aux = np.full((height, width), 1000.0, dtype=np.float32)
    aux[2, 7] = np.nan
    working = np.full((height, width, 4), 1000.0, dtype=np.float32)
    parameters = ScoreParameters(
        base_primary=50000.0,
        base_addend=-119.40625,
        scale=-0.001041,
        offset=1.0,
        floor=0.02,
        resolution_metric=4000,
        horizontal_minimum_resolution_cutoff=550,
    )
    with pytest.raises(ValueError):
        continuous_score(aux, parameters)

    error_flags = cupy.zeros(1, dtype=cupy.uint32)
    module.get_function("k_score_and_weighted")(
        ((height * width + 255) // 256,),
        (256,),
        (
            cupy.asarray(aux),
            cupy.asarray(working),
            cupy.asarray(np.full(height, np.float32(50000.0), dtype=np.float32)),
            np.float64(np.float32(parameters.base_addend)),
            np.float64(np.float32(parameters.scale)),
            np.float64(np.float32(parameters.offset)),
            np.float64(np.float32(parameters.floor)),
            np.int32(1),
            np.int32(height),
            np.int32(width),
            cupy.empty((height, width), dtype=cupy.float32),
            cupy.empty((height, width), dtype=cupy.float32),
            cupy.empty((height, width, 3), dtype=cupy.float32),
            error_flags,
        ),
    )
    assert int(error_flags.get()[0]) & 1


def test_decision_kernel_all_edges(module):
    from portable_digital_ice.streaming import _RowCache, _decision_fallback_row
    from portable_digital_ice.x3a import DecisionParameters

    rng = np.random.default_rng(7)
    height, width = 21, 33
    pixels = rng.integers(0, 65536, size=(height, width, 4), dtype=np.uint16)
    response = SharedLookupInputResponse.nikon_logarithmic()
    aux_params = AuxiliaryParameters(
        selected_visible_channel=0,
        alpha=float(np.float32(0.5)),
        calibration_offset=1.0,
        alpha_one_replacement=None,
    )
    score_params = ScoreParameters(
        base_primary=50000.0,
        base_addend=-119.40625,
        scale=-0.001041,
        offset=1.0,
        floor=0.02,
        resolution_metric=4000,
        horizontal_minimum_resolution_cutoff=550,
    )
    cache = _RowCache(
        pixels,
        response=response,
        auxiliary_parameters=aux_params,
        score_parameters=score_params,
        stage_parameter_provider=None,
    )
    aux_plane = np.stack([cache.get(y).auxiliary for y in range(height)])
    score_plane = np.stack([cache.get(y).score for y in range(height)])

    for radius in (1, 2, 3, 4):
        parameters = DecisionParameters(
            sample_threshold=float(np.float32(49383.234)),
            count_limit=8,
            perpendicular_radius=radius,
        )
        expected = np.stack(
            [
                _decision_fallback_row(
                    y,
                    height=height,
                    width=width,
                    cache=cache,
                    parameters=parameters,
                )
                for y in range(height)
            ]
        )
        eligible = cupy.empty(height * width, dtype=cupy.uint8)
        module.get_function("k_decision_eligibility")(
            ((height * width + 255) // 256,),
            (256,),
            (
                cupy.asarray(aux_plane),
                cupy.asarray(score_plane),
                np.float32(parameters.sample_threshold),
                np.int32(parameters.count_limit),
                np.int32(parameters.perpendicular_radius),
                cupy.zeros(height, dtype=cupy.int32),
                cupy.zeros(height, dtype=cupy.uint8),  # floor disabled
                np.int32(height),
                np.int32(width),
                eligible,
            ),
        )
        got_fallback = cupy.asnumpy(eligible).reshape(height, width) == 0
        assert np.array_equal(got_fallback, expected), f"radius {radius}"
