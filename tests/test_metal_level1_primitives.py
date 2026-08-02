"""Level 1: Metal primitive parity, compared as raw bit patterns.

Mirrors the CUDA Level 1 suite: each device building block runs against the
real reference implementation on the same inputs.  Every test skips with a
clear reason when Metal is unavailable.  Comparisons use integer/IEEE-754
bit views, never decimal tolerances.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("Metal", reason="Metal backend requires pyobjc-framework-Metal")

from portable_digital_ice.dither import (  # noqa: E402
    DitherBounds,
    conditional_dither_delta,
)
from portable_digital_ice.metal_backend.engine import (  # noqa: E402
    MetalBackendUnavailable,
    metal_device_summary,
)
from portable_digital_ice.metal_backend.kernels import render_kernel_source  # noqa: E402
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

from metal_harness import MetalHarness  # noqa: E402


def _require_device() -> None:
    try:
        metal_device_summary()
    except MetalBackendUnavailable as error:
        pytest.skip(f"Metal device unavailable: {error}")


TEST_WRAPPERS = r"""
kernel void t_unscaled_averages(
    device const float* patches [[buffer(0)]],
    device u64* out [[buffer(1)]],
    constant int* iparams [[buffer(2)]],
    uint idx [[thread_position_in_grid]]) {
  long count = (long)iparams[0];
  long i = (long)idx;
  if (i >= count) return;
  float p[9][9];
  for (int y = 0; y < 9; ++y)
    for (int x = 0; x < 9; ++x) p[y][x] = patches[(i * 81) + y * 9 + x];
  u64 q[3];
  unscaled_averages_scalar(p, q);
  out[i * 3 + 0] = q[0];
  out[i * 3 + 1] = q[1];
  out[i * 3 + 2] = q[2];
}

kernel void t_rgb_unscaled_averages(
    device const float* patches [[buffer(0)]],
    device u64* out [[buffer(1)]],
    constant int* iparams [[buffer(2)]],
    uint idx [[thread_position_in_grid]]) {
  long count = (long)iparams[0];
  long i = (long)idx;
  if (i >= count) return;
  float p[3][9][9];
  for (int y = 0; y < 9; ++y)
    for (int x = 0; x < 9; ++x)
      for (int c = 0; c < 3; ++c)
        p[c][y][x] = patches[(((i * 9) + y) * 9 + x) * 3 + c];
  u64 q[3][3];
  rgb_unscaled_averages(p, q);
  for (int s = 0; s < 3; ++s)
    for (int c = 0; c < 3; ++c) out[(i * 3 + s) * 3 + c] = q[s][c];
}

kernel void t_rgb_history_averages(
    device const float* wrgb [[buffer(0)]],
    device const float* working [[buffer(1)]],
    device const int* centers [[buffer(2)]],
    device u64* patch_out [[buffer(3)]],
    device u64* direct_out [[buffer(4)]],
    constant float* fparams [[buffer(5)]],
    constant int* iparams [[buffer(6)]],
    uint idx [[thread_position_in_grid]]) {
  long count = (long)iparams[0];
  long i = (long)idx;
  if (i >= count) return;
  int H = iparams[1];
  int W = iparams[2];
  int cy = centers[i * 2 + 0];
  int cx = centers[i * 2 + 1];
  float score_floor = fparams[0];
  float p[3][9][9];
  for (int dy = -4; dy <= 4; ++dy) {
    int ady = dy < 0 ? -dy : dy;
    int x_radius = ady == 4 ? 2 : (ady == 3 ? 3 : 4);
    for (int dx = -x_radius; dx <= x_radius; ++dx)
      for (int c = 0; c < 3; ++c)
        p[c][dy + 4][dx + 4] = load_wrgb_hist(
            wrgb, working, H, W, score_floor, cy + dy, cx + dx, c);
  }
  u64 patch_q[3][3], direct_q[3][3];
  rgb_unscaled_averages(p, patch_q);
  rgb_unscaled_averages_at(wrgb, working, H, W, score_floor, cy, cx,
                           direct_q);
  for (int s = 0; s < 3; ++s) {
    for (int c = 0; c < 3; ++c) {
      long out_index = (i * 3 + s) * 3 + c;
      patch_out[out_index] = patch_q[s][c];
      direct_out[out_index] = direct_q[s][c];
    }
  }
}

kernel void t_feature_range_reducers(
    device const float* records [[buffer(0)]],
    device const u8* record_counts [[buffer(1)]],
    device u64* materialized_out [[buffer(2)]],
    device u64* streamed_out [[buffer(3)]],
    constant int* iparams [[buffer(4)]],
    uint idx [[thread_position_in_grid]]) {
  long count = (long)iparams[0];
  long i = (long)idx;
  if (i >= count) return;
  int record_count = (int)record_counts[i];
  float features[5][4];
  for (int r = 0; r < 5; ++r)
    for (int lane = 0; lane < 4; ++lane)
      features[r][lane] = records[(i * 5 + r) * 4 + lane];

  u64 old_min[3], old_max[3];
  for (int t = 0; t < 3; ++t) {
    u64 mn = f64_sub(f64_from_f32(features[0][t + 1]),
                     f64_from_f32(features[0][t]));
    u64 mx = mn;
    for (int r = 1; r < record_count; ++r) {
      u64 d = f64_sub(f64_from_f32(features[r][t + 1]),
                      f64_from_f32(features[r][t]));
      mn = f64_lt(d, mn) ? d : mn;
      mx = f64_lt(mx, d) ? d : mx;
    }
    old_min[t] = f64_from_f32(f32_from_f64(mn));
    old_max[t] = f64_from_f32(f32_from_f64(mx));
  }

  u64 new_min[3], new_max[3];
  for (int t = 0; t < 3; ++t) {
    u64 d = f64_sub(f64_from_f32(features[0][t + 1]),
                    f64_from_f32(features[0][t]));
    new_min[t] = d;
    new_max[t] = d;
  }
  for (int r = 1; r < record_count; ++r) {
    for (int t = 0; t < 3; ++t) {
      u64 d = f64_sub(f64_from_f32(features[r][t + 1]),
                      f64_from_f32(features[r][t]));
      new_min[t] = f64_lt(d, new_min[t]) ? d : new_min[t];
      new_max[t] = f64_lt(new_max[t], d) ? d : new_max[t];
    }
  }
  for (int t = 0; t < 3; ++t) {
    new_min[t] = f64_from_f32(f32_from_f64(new_min[t]));
    new_max[t] = f64_from_f32(f32_from_f64(new_max[t]));
    materialized_out[i * 6 + t] = old_min[t];
    materialized_out[i * 6 + 3 + t] = old_max[t];
    streamed_out[i * 6 + t] = new_min[t];
    streamed_out[i * 6 + 3 + t] = new_max[t];
  }
}

kernel void t_dither_chain(
    device const u64* values [[buffer(0)]],
    device const float* scales [[buffer(1)]],
    device u64* deltas [[buffer(2)]],
    device u32* state_out [[buffer(3)]],
    device u64* advances_out [[buffer(4)]],
    constant u64* qparams [[buffer(5)]],
    constant int* iparams [[buffer(6)]],
    uint idx [[thread_position_in_grid]]) {
  if (idx != 0) return;
  long count = (long)iparams[0];
  u32 state = (u32)iparams[1];
  u64 low = qparams[0];
  u64 high = qparams[1];
  bool low_lt_high = f64_lt(low, high);
  u64 advances = 0ul;
  for (long i = 0; i < count; ++i) {
    deltas[i] = dither_delta(values[i], low, high, low_lt_high, scales[i],
                             &state, &advances);
  }
  state_out[0] = state;
  advances_out[0] = advances;
}

kernel void t_emit(
    device const float* values [[buffer(0)]],
    device u16* out [[buffer(1)]],
    device const u32* factor_high [[buffer(2)]],
    device const u32* factor_low [[buffer(3)]],
    constant int* iparams [[buffer(4)]],
    uint idx [[thread_position_in_grid]]) {
  long count = (long)iparams[0];
  long i = (long)idx;
  if (i >= count) return;
  out[i] = emit_one(values[i], factor_high, factor_low);
}
"""


@pytest.fixture(scope="module")
def harness() -> MetalHarness:
    _require_device()
    return MetalHarness(render_kernel_source() + TEST_WRAPPERS)


def _i32(harness: MetalHarness, *values: int):
    return harness.buffer_from(np.asarray(values, dtype=np.int32))


def _q64(harness: MetalHarness, *values: float):
    return harness.buffer_from(
        np.asarray(values, dtype=np.float64).view(np.uint64)
    )


def test_device_summary_reports_exactness_configuration():
    _require_device()
    summary = metal_device_summary()
    assert summary["recommended_max_working_set_bytes"] > 0
    assert "fastMathEnabled=False" in summary["compile_options"]
    assert "software IEEE-754" in summary["binary64_execution"]


def test_response_lut_indexing_and_auxiliary_bits(harness: MetalHarness):
    rng = np.random.default_rng(1)
    height, width = 37, 53  # odd dimensions on purpose
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

    total = height * width
    buf_working = harness.buffer_out(np.float32, total * 4)
    buf_aux = harness.buffer_out(np.float32, total)
    harness.run(
        "k_convert_and_auxiliary",
        [
            harness.buffer_from(pixels),
            harness.buffer_from(response.table),
            harness.buffer_from(np.full(height, np.float32(0.4375), dtype=np.float32)),
            harness.buffer_from(np.zeros(height, dtype=np.uint8)),
            harness.buffer_from(np.zeros(height, dtype=np.float32)),
            harness.buffer_from(np.full(height, np.float32(1.0), dtype=np.float32)),
            buf_working,
            buf_aux,
            _i32(harness, 0, height, width),
        ],
        total,
    )
    got_working = harness.read(buf_working, np.float32, total * 4).reshape(
        height, width, 4
    )
    got_aux = harness.read(buf_aux, np.float32, total).reshape(height, width)
    assert np.array_equal(got_working.view(np.uint32), working_cpu.view(np.uint32))
    assert np.array_equal(got_aux.view(np.uint32), aux_cpu.view(np.uint32))


def test_score_equation_bits(harness: MetalHarness):
    rng = np.random.default_rng(2)
    height, width = 23, 61
    aux = rng.uniform(-5000.0, 66000.0, size=(height, width)).astype(np.float32)
    working = rng.uniform(0.0, 65535.0, size=(height, width, 4)).astype(np.float32)
    parameters = ScoreParameters(
        base_primary=float(np.float32(52011.4)),
        base_addend=float(
            np.frombuffer(np.uint32(0xC2EED000).tobytes(), dtype=np.float32)[0]
        ),
        scale=float(
            np.frombuffer(np.uint32(0xBA887952).tobytes(), dtype=np.float32)[0]
        ),
        offset=1.0,
        floor=float(
            np.frombuffer(np.uint32(0x3CA3D70A).tobytes(), dtype=np.float32)[0]
        ),
        resolution_metric=4000,
        horizontal_minimum_resolution_cutoff=550,
    )
    score_cpu = continuous_score(aux, parameters)

    total = height * width
    buf_score = harness.buffer_out(np.float32, total)
    buf_waux = harness.buffer_out(np.float32, total)
    buf_wrgb = harness.buffer_out(np.float32, total * 3)
    buf_flags = harness.buffer_from(np.zeros(1, dtype=np.uint32))
    harness.run(
        "k_score_and_weighted",
        [
            harness.buffer_from(aux),
            harness.buffer_from(working),
            harness.buffer_from(
                np.full(height, np.float32(parameters.base_primary), dtype=np.float32)
            ),
            buf_score,
            buf_waux,
            buf_wrgb,
            buf_flags,
            _q64(
                harness,
                float(np.float32(parameters.base_addend)),
                float(np.float32(parameters.scale)),
                float(np.float32(parameters.offset)),
                float(np.float32(parameters.floor)),
            ),
            _i32(harness, 1, height, width),
        ],
        total,
    )
    assert int(harness.read(buf_flags, np.uint32, 1)[0]) == 0
    got_score = harness.read(buf_score, np.float32, total).reshape(height, width)
    assert np.array_equal(got_score.view(np.uint32), score_cpu.view(np.uint32))
    waux_cpu = np.multiply(score_cpu, aux, dtype=np.float32)
    wrgb_cpu = np.multiply(
        score_cpu[:, :, None], working[:, :, :3], dtype=np.float32
    )
    got_waux = harness.read(buf_waux, np.float32, total).reshape(height, width)
    got_wrgb = harness.read(buf_wrgb, np.float32, total * 3).reshape(
        height, width, 3
    )
    assert np.array_equal(got_waux.view(np.uint32), waux_cpu.view(np.uint32))
    assert np.array_equal(got_wrgb.view(np.uint32), wrgb_cpu.view(np.uint32))


def test_unscaled_averages_bits(harness: MetalHarness):
    rng = np.random.default_rng(3)
    count = 4096
    patches = rng.uniform(0.0, 66000.0, size=(count, 9, 9)).astype(np.float32)
    patches[0] = 0.0
    patches[1] = 65535.0
    patches[2, :, :] = rng.uniform(0.0, 1.0, size=(9, 9)).astype(np.float32)
    expected = np.stack(
        [_recovered_unscaled_averages(patch)[:, 0] for patch in patches]
    )
    out = harness.buffer_out(np.uint64, count * 3)
    harness.run(
        "t_unscaled_averages",
        [
            harness.buffer_from(patches.reshape(count, -1)),
            out,
            _i32(harness, count),
        ],
        count,
    )
    got = harness.read(out, np.uint64, count * 3).reshape(count, 3)
    assert np.array_equal(got, expected.view(np.uint64))


def test_rgb_candidate_averages_bits(harness: MetalHarness):
    rng = np.random.default_rng(4)
    count = 2048
    patches = rng.uniform(0.0, 66000.0, size=(count, 9, 9, 3)).astype(np.float32)
    expected = np.stack(
        [_recovered_rgb_unscaled_averages(patch) for patch in patches]
    )
    out = harness.buffer_out(np.uint64, count * 9)
    harness.run(
        "t_rgb_unscaled_averages",
        [
            harness.buffer_from(patches.reshape(count, -1)),
            out,
            _i32(harness, count),
        ],
        count,
    )
    got = harness.read(out, np.uint64, count * 9).reshape(count, 3, 3)
    assert np.array_equal(got, expected.view(np.uint64))


@pytest.mark.parametrize("score_floor", [0.0, 0.02, 1.0])
def test_rgb_direct_history_matches_patch_bits(
    harness: MetalHarness, score_floor: float
):
    rng = np.random.default_rng(4000 + int(score_floor * 100))
    height, width = 11, 13
    wrgb = rng.uniform(-100.0, 66000.0, size=(height, width, 3)).astype(
        np.float32
    )
    working = rng.uniform(-100.0, 66000.0, size=(height, width, 4)).astype(
        np.float32
    )
    special_values = np.asarray(
        [
            0x00000000,
            0x80000000,
            0x00000001,
            0x007FFFFF,
            0x00800000,
            0x7F7FFFFF,
            0xFF7FFFFF,
            0x7F800000,
            0xFF800000,
            0x7FC00000,
            0xFFC00001,
        ],
        dtype=np.uint32,
    ).view(np.float32)
    for index, value in enumerate(special_values):
        y = index % height
        x = (index * 5) % width
        c = index % 3
        wrgb[y, x, c] = value
        working[y, x, c] = value
    centers = np.asarray(
        [
            (y, x)
            for y in (0, 1, height // 2, height - 2, height - 1)
            for x in (0, 1, width // 2, width - 2, width - 1)
        ],
        dtype=np.int32,
    )
    count = centers.shape[0]
    patch_out = harness.buffer_out(np.uint64, count * 9)
    direct_out = harness.buffer_out(np.uint64, count * 9)
    harness.run(
        "t_rgb_history_averages",
        [
            harness.buffer_from(wrgb),
            harness.buffer_from(working),
            harness.buffer_from(centers),
            patch_out,
            direct_out,
            harness.buffer_from(np.asarray([score_floor], dtype=np.float32)),
            _i32(harness, count, height, width),
        ],
        count,
    )
    patch_bits = harness.read(patch_out, np.uint64, count * 9)
    direct_bits = harness.read(direct_out, np.uint64, count * 9)
    np.testing.assert_array_equal(direct_bits, patch_bits)


def test_streamed_feature_extrema_match_materialized_bits(
    harness: MetalHarness,
) -> None:
    rng = np.random.default_rng(0xE871)
    count = 4096
    records = rng.standard_normal((count, 5, 4), dtype=np.float32)
    adversarial_bits = np.asarray(
        [
            0x00000000,
            0x80000000,
            0x00000001,
            0x007FFFFF,
            0x00800000,
            0x3F800000,
            0xBF800000,
            0x7F7FFFFF,
            0xFF7FFFFF,
            0x7F800000,
            0xFF800000,
            0x7FC00000,
            0xFFC00001,
        ],
        dtype=np.uint32,
    ).view(np.float32)
    records[: adversarial_bits.size] = adversarial_bits[:, None, None]
    for i, value in enumerate(adversarial_bits):
        records[i, i % 5, i % 4] = value
        records[i, (i + 1) % 5, (i + 1) % 4] = np.float32(-value)
    record_counts = np.where(np.arange(count) % 2 == 0, 1, 5).astype(np.uint8)
    materialized = harness.buffer_out(np.uint64, count * 6)
    streamed = harness.buffer_out(np.uint64, count * 6)
    harness.run(
        "t_feature_range_reducers",
        [
            harness.buffer_from(records),
            harness.buffer_from(record_counts),
            materialized,
            streamed,
            _i32(harness, count),
        ],
        count,
    )
    np.testing.assert_array_equal(
        harness.read(streamed, np.uint64, count * 6),
        harness.read(materialized, np.uint64, count * 6),
    )


def test_conditional_dither_chain_bits(harness: MetalHarness):
    """The unreachable device dither primitive is a standing drift guard."""

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
                [
                    low,
                    high,
                    np.nextafter(low, high),
                    np.nextafter(high, low),
                    0.0,
                    65535.0,
                ]
            ),
        ]
    ).astype(np.float64)
    scales = rng.choice(
        np.array(
            [np.float32(0.015), np.float32(0.025), np.float32(0.0)],
            dtype=np.float32,
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
    buf_deltas = harness.buffer_out(np.uint64, count)
    buf_state = harness.buffer_out(np.uint32, 1)
    buf_advances = harness.buffer_out(np.uint64, 1)
    harness.run(
        "t_dither_chain",
        [
            harness.buffer_from(values.view(np.uint64)),
            harness.buffer_from(scales),
            buf_deltas,
            buf_state,
            buf_advances,
            _q64(harness, low, high),
            _i32(harness, count, 0x3045),
        ],
        1,
    )
    got = harness.read(buf_deltas, np.uint64, count)
    assert np.array_equal(got, expected.view(np.uint64))
    assert int(harness.read(buf_state, np.uint32, 1)[0]) == generator.state


def test_output_conversion_bits(harness: MetalHarness):
    from portable_digital_ice.output import InverseResponseFactors

    rng = np.random.default_rng(6)
    values = np.concatenate(
        [
            rng.uniform(-10.0, 65545.0, size=100000),
            np.array(
                [-0.5, -0.4999999, 0.0, 0.49999997, 0.5, 65534.5, 65535.0, 65540.0]
            ),
        ]
    ).astype(np.float32)
    expected = emit_public_rgb16(values.reshape(1, -1, 1).repeat(3, axis=2))[0, :, 0]
    factors = InverseResponseFactors.recovered_16bit()
    out = harness.buffer_out(np.uint16, values.size)
    harness.run(
        "t_emit",
        [
            harness.buffer_from(values),
            out,
            harness.buffer_from(factors.high),
            harness.buffer_from(factors.low),
            _i32(harness, values.size),
        ],
        values.size,
    )
    assert np.array_equal(harness.read(out, np.uint16, values.size), expected)


def test_nonfinite_auxiliary_fails_closed_like_cpu(harness: MetalHarness):
    """A NaN auxiliary plane raises on CPU; the kernel must flag, not launder.

    Unreachable through the public API (uint16 inputs, hash-pinned finite
    LUT, finiteness-validated calibration), but the comparison-based clamps
    would otherwise silently convert NaN into a bound where NumPy propagates
    it into the reference's fail-closed validation.
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

    total = height * width
    buf_flags = harness.buffer_from(np.zeros(1, dtype=np.uint32))
    harness.run(
        "k_score_and_weighted",
        [
            harness.buffer_from(aux),
            harness.buffer_from(working),
            harness.buffer_from(
                np.full(height, np.float32(50000.0), dtype=np.float32)
            ),
            harness.buffer_out(np.float32, total),
            harness.buffer_out(np.float32, total),
            harness.buffer_out(np.float32, total * 3),
            buf_flags,
            _q64(
                harness,
                float(np.float32(parameters.base_addend)),
                float(np.float32(parameters.scale)),
                float(np.float32(parameters.offset)),
                float(np.float32(parameters.floor)),
            ),
            _i32(harness, 1, height, width),
        ],
        total,
    )
    assert int(harness.read(buf_flags, np.uint32, 1)[0]) & 1


def test_decision_kernel_all_edges(harness: MetalHarness):
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
        buf_eligible = harness.buffer_out(np.uint8, height * width)
        harness.run(
            "k_decision_eligibility",
            [
                harness.buffer_from(aux_plane),
                harness.buffer_from(score_plane),
                harness.buffer_from(np.zeros(height, dtype=np.int32)),
                harness.buffer_from(np.zeros(height, dtype=np.uint8)),
                buf_eligible,
                harness.buffer_from(
                    np.asarray(
                        [parameters.sample_threshold], dtype=np.float32
                    )
                ),
                _i32(
                    harness,
                    parameters.count_limit,
                    parameters.perpendicular_radius,
                    height,
                    width,
                ),
            ],
            height * width,
        )
        got_fallback = (
            harness.read(buf_eligible, np.uint8, height * width).reshape(
                height, width
            )
            == 0
        )
        assert np.array_equal(got_fallback, expected), f"radius {radius}"
