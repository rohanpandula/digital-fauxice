"""CUDA C sources for the deterministic LS-5000 selector-8 backend.

Every device function is a line-by-line translation of the audited CPU
reference in this package.  The reference widens float32 operands to binary64,
evaluates with one rounding per written operation, and narrows once at each
recorded store boundary.  The kernels reproduce that schedule literally:

- compiled with ``--fmad=false`` so multiply/add pairs never contract;
- no fast-math, no reciprocal approximations, no reordered reductions;
- float64 -> float32 narrowing uses the default round-to-nearest-even cast;
- the 24-bit LCG and conditional-dither writer are sequential by necessity
  (the number of draws a site consumes depends on the drawn values) and run
  on one host CPU core via the compiled ``fast_cpu`` path
  (``cuda_backend/host_writer.py``) instead of one device thread; the
  ``dither_delta`` device function stays here as a validated primitive only,
  no longer reachable from any kernel this module compiles.

Numeric constants that the CPU reference derives by quantizing Python floats
to float32 (for example ``float32(1/69)``) are injected from host Python as
exact hexadecimal binary64 literals so the kernel text cannot drift from the
reference quantization.
"""

from __future__ import annotations

import numpy as np


def _hex64(value: float) -> str:
    return float(value).hex()


# The reference multiplies by float32-quantized reciprocals widened to
# binary64 (reconstruction._recovered_unscaled_averages).
COEFFICIENT_69 = _hex64(float(np.float32(1.0 / 69.0)))
COEFFICIENT_21 = _hex64(float(np.float32(1.0 / 21.0)))
COEFFICIENT_16 = _hex64(float(np.float32(1.0 / 16.0)))
NIKON_NORMALIZATION_HEX = "0x1.fffffep-25"

NVRTC_OPTIONS = ("--fmad=false", "--std=c++17")

KERNEL_SOURCE = r"""
typedef unsigned int u32;
typedef unsigned short u16;
typedef unsigned char u8;
typedef long long i64;
typedef unsigned long long u64;

#define COEFF69 __COEFF69__
#define COEFF21 __COEFF21__
#define COEFF16 __COEFF16__
#define NIKON_NORM __NIKON_NORM__

// ---------------------------------------------------------------------------
// history loaders: X3A score / weighted-history boundary carrier
//   horizontal guards (x outside [0, W)) stay zero
//   logical row -1 is the pseudo row (first real row scaled by the floor)
//   rows below -1 stay zero, rows >= H repeat the final real row
// ---------------------------------------------------------------------------

__device__ __forceinline__ float load_score_hist(
    const float* score, int H, int W, float score_floor, int y, int x) {
  if (x < 0 || x >= W) return 0.0f;
  if (y < -1) return 0.0f;
  if (y == -1) return score_floor;
  if (y >= H) y = H - 1;
  return score[(i64)y * W + x];
}

__device__ __forceinline__ float load_waux_hist(
    const float* waux, const float* aux, int H, int W, float score_floor,
    int y, int x) {
  if (x < 0 || x >= W) return 0.0f;
  if (y < -1) return 0.0f;
  if (y == -1) return aux[x] * score_floor;  // float32 multiply, no fma
  if (y >= H) y = H - 1;
  return waux[(i64)y * W + x];
}

__device__ __forceinline__ float load_wrgb_hist(
    const float* wrgb, const float* working, int H, int W, float score_floor,
    int y, int x, int c) {
  if (x < 0 || x >= W) return 0.0f;
  if (y < -1) return 0.0f;
  if (y == -1) return working[(i64)x * 4 + c] * score_floor;  // first row
  if (y >= H) y = H - 1;
  return wrgb[((i64)y * W + x) * 3 + c];
}

// ---------------------------------------------------------------------------
// reconstruction._vertical_scratch + _vertical_binomial_16 +
// _recovered_unscaled_averages for one scalar 9x9 patch.
// The accumulation order and float32 store boundaries are load-bearing.
// ---------------------------------------------------------------------------

__device__ void unscaled_averages_scalar(const float p[9][9], double q[3]) {
  float s3[9], s5[9], s7[9], s9[9], b16[9];
  for (int x = 0; x < 9; ++x) {
    double acc = (double)p[3][x] + (double)p[4][x];
    acc += (double)p[5][x];
    s3[x] = (float)acc;
    acc += (double)p[2][x];
    acc += (double)p[6][x];
    s5[x] = (float)acc;
    acc += (double)p[1][x];
    acc += (double)p[7][x];
    s7[x] = (float)acc;
    acc += (double)p[0][x];
    acc += (double)p[8][x];
    s9[x] = (float)acc;
    double center = (double)p[4][x];
    double bacc = (double)p[3][x] + center;
    bacc += (double)p[5][x];
    bacc += center;
    b16[x] = (float)(bacc * COEFF16);
  }
  double t21 = (double)s3[6] + (double)s3[2];
  t21 += (double)s5[3];
  t21 += (double)s5[4];
  t21 += (double)s5[5];
  double t69 = (double)s5[8] + (double)s5[0];
  t69 += (double)s7[1];
  t69 += (double)s7[7];
  for (int x = 2; x < 7; ++x) t69 += (double)s9[x];
  double t16 = (double)b16[4] + (double)b16[3];
  t16 += (double)b16[5];
  t16 += (double)b16[4];
  q[0] = t69 * COEFF69;
  q[1] = t21 * COEFF21;
  q[2] = t16;
}

// reconstruction._recovered_rgb_unscaled_averages: shares Q69/Q21 with the
// scalar helper, but the 3x3 binomial path keeps x87 precision until one
// final float64 multiply (no intermediate float32 store).
__device__ void rgb_unscaled_averages(
    const float p[9][9][3], double q[3][3]) {
  for (int c = 0; c < 3; ++c) {
    float lane[9][9];
    for (int y = 0; y < 9; ++y)
      for (int x = 0; x < 9; ++x) lane[y][x] = p[y][x][c];
    double ql[3];
    unscaled_averages_scalar(lane, ql);
    double rows[3];
    for (int r = 0; r < 3; ++r) {
      int y = 3 + r;
      double center = (double)lane[y][4];
      double horizontal = (double)lane[y][3] + (double)lane[y][5];
      horizontal += center;
      horizontal += center;
      rows[r] = horizontal;
    }
    double total = rows[0] + rows[1];
    total += rows[1];
    total += rows[2];
    q[0][c] = ql[0];
    q[1][c] = ql[1];
    q[2][c] = total * COEFF16;
  }
}

// streaming._feature_record on gathered history patches.  ``uw`` keeps the
// unrounded binary64 weights: features divide by the in-register value, not
// the float32 W record.
__device__ void feature_record_at(
    const float* score, const float* waux, const float* aux, int H, int W,
    float score_floor, int cy, int cx, float point_score, float point_aux,
    float fallback, float w_out[4], float f_out[4]) {
  float score_patch[9][9], waux_patch[9][9];
  for (int dy = -4; dy <= 4; ++dy) {
    for (int dx = -4; dx <= 4; ++dx) {
      score_patch[dy + 4][dx + 4] =
          load_score_hist(score, H, W, score_floor, cy + dy, cx + dx);
      waux_patch[dy + 4][dx + 4] =
          load_waux_hist(waux, aux, H, W, score_floor, cy + dy, cx + dx);
    }
  }
  double uw[3], num[3];
  unscaled_averages_scalar(score_patch, uw);
  unscaled_averages_scalar(waux_patch, num);
  w_out[0] = (float)uw[0];
  w_out[1] = (float)uw[1];
  w_out[2] = (float)uw[2];
  w_out[3] = point_score;
  for (int lane = 0; lane < 3; ++lane) {
    f_out[lane] = (uw[lane] > 0.0) ? (float)(num[lane] / uw[lane]) : fallback;
  }
  f_out[3] = (w_out[3] > 0.0f) ? point_aux : fallback;
}

// streaming._cross_neighbor_feature_record point-sample boundary rules.
__device__ void cross_feature_at(
    const float* score, const float* waux, const float* aux, int H, int W,
    float score_floor, int ny, int nx, float fallback, float f_out[4]) {
  if (nx < 0 || nx >= W) {
    f_out[0] = f_out[1] = f_out[2] = f_out[3] = 0.0f;
    return;
  }
  float point_aux, point_score;
  if (ny < 0) {
    point_aux = aux[nx];
    point_score = score_floor;
  } else if (ny >= H) {
    point_aux = aux[(i64)(H - 1) * W + nx];
    point_score = score[(i64)(H - 1) * W + nx];
  } else {
    point_aux = aux[(i64)ny * W + nx];
    point_score = score[(i64)ny * W + nx];
  }
  float w_unused[4];
  feature_record_at(score, waux, aux, H, W, score_floor, ny, nx, point_score,
                    point_aux, fallback, w_unused, f_out);
}

// ---------------------------------------------------------------------------
// stage 1: response LUT + auxiliary plane (x3a.derive_auxiliary)
// ---------------------------------------------------------------------------

extern "C" __global__ void k_convert_and_auxiliary(
    const u16* __restrict__ rgbi, const float* __restrict__ lut,
    const float* __restrict__ aux_alpha, const u8* __restrict__ alpha_is_one,
    const float* __restrict__ alpha_one_replacement,
    const float* __restrict__ aux_offset, int visible_channel, int H, int W,
    float* __restrict__ working, float* __restrict__ aux) {
  i64 idx = (i64)blockIdx.x * blockDim.x + threadIdx.x;
  i64 total = (i64)H * W;
  if (idx >= total) return;
  int y = (int)(idx / W);
  float lanes[4];
  for (int c = 0; c < 4; ++c) {
    lanes[c] = lut[rgbi[idx * 4 + c]];
    working[idx * 4 + c] = lanes[c];
  }
  float value;
  if (alpha_is_one[y]) {
    value = alpha_one_replacement[y];
  } else {
    double alpha = (double)aux_alpha[y];
    double visible = (double)lanes[visible_channel];
    double infrared = (double)lanes[3];
    value = (float)((infrared - alpha * visible) / (1.0 - alpha) -
                    (double)aux_offset[y]);
  }
  aux[idx] = value;
}

// ---------------------------------------------------------------------------
// stage 2: continuous score + weighted planes (x3a.continuous_score)
// ---------------------------------------------------------------------------

extern "C" __global__ void k_score_and_weighted(
    const float* __restrict__ aux, const float* __restrict__ working,
    const float* __restrict__ score_base_primary, double base_addend,
    double scale, double offset, double floor_value, int horizontal_minimum,
    int H, int W, float* __restrict__ score, float* __restrict__ waux,
    float* __restrict__ wrgb, u32* __restrict__ error_flags) {
  i64 idx = (i64)blockIdx.x * blockDim.x + threadIdx.x;
  i64 total = (i64)H * W;
  if (idx >= total) return;
  int y = (int)(idx / W);
  int x = (int)(idx - (i64)y * W);
  float value = aux[idx];
  // continuous_score fails closed on a nonfinite auxiliary plane; the C
  // ternary clamps below would otherwise launder NaN instead of raising.
  if (!isfinite(value)) atomicOr(&error_flags[0], 1u);
  float sample = value;
  if (horizontal_minimum && W > 2 && x >= 1 && x <= W - 2) {
    float left = aux[idx - 1];
    float right = aux[idx + 1];
    float m = left < value ? left : value;
    sample = m < right ? m : right;
  }
  double primary = (double)score_base_primary[y];
  double s64 = ((primary + base_addend) - (double)sample) * scale + offset;
  s64 = s64 < 1.0 ? s64 : 1.0;
  s64 = s64 > floor_value ? s64 : floor_value;
  float s = (float)s64;
  score[idx] = s;
  waux[idx] = s * value;
  for (int c = 0; c < 3; ++c) wrgb[idx * 3 + c] = s * working[idx * 4 + c];
}

// ---------------------------------------------------------------------------
// stage 3: decision fallback + row eligibility (streaming rules)
// raw decision history replicates every edge.
// ---------------------------------------------------------------------------

extern "C" __global__ void k_decision_eligibility(
    const float* __restrict__ aux, const float* __restrict__ score,
    float threshold, int count_limit, int radius,
    const int* __restrict__ row_gate, const u8* __restrict__ floor_enabled,
    int H, int W, u8* __restrict__ eligible) {
  i64 idx = (i64)blockIdx.x * blockDim.x + threadIdx.x;
  i64 total = (i64)H * W;
  if (idx >= total) return;
  int y = (int)(idx / W);
  int x = (int)(idx - (i64)y * W);
  int lx = x - radius;
  lx = lx < 0 ? 0 : (lx >= W ? W - 1 : lx);
  int rx = x + radius;
  rx = rx < 0 ? 0 : (rx >= W ? W - 1 : rx);
  int vertical_left = 0, vertical_right = 0;
  for (int dy = -4; dy <= 4; ++dy) {
    int yy = y + dy;
    yy = yy < 0 ? 0 : (yy >= H ? H - 1 : yy);
    const float* row = aux + (i64)yy * W;
    vertical_left += row[lx] < threshold;
    vertical_right += row[rx] < threshold;
  }
  int ay = y - radius < 0 ? 0 : y - radius;
  int by = y + radius >= H ? H - 1 : y + radius;
  const float* above = aux + (i64)ay * W;
  const float* below = aux + (i64)by * W;
  int horizontal_above = 0, horizontal_below = 0;
  for (int dx = -4; dx <= 4; ++dx) {
    int xx = x + dx;
    xx = xx < 0 ? 0 : (xx >= W ? W - 1 : xx);
    horizontal_above += above[xx] < threshold;
    horizontal_below += below[xx] < threshold;
  }
  bool fallback = (vertical_left > count_limit) |
                  (horizontal_above > count_limit) |
                  (vertical_right > count_limit) |
                  (horizontal_below > count_limit);
  bool ok = !fallback;
  if (row_gate[y] != 0) {
    ok = false;
  } else if (floor_enabled[y]) {
    ok = ok && (score[idx] < 1.0f);
  }
  eligible[idx] = ok ? 1 : 0;
}

// ---------------------------------------------------------------------------
// stage 4: per-selected-pixel feature records, candidates, combiner
// (streaming inner loop + reconstruction.combine_recovered_candidate)
// ---------------------------------------------------------------------------

extern "C" __global__ void k_features_and_combine(
    const i64* __restrict__ selected, i64 selected_count,
    const float* __restrict__ score, const float* __restrict__ waux,
    const float* __restrict__ wrgb, const float* __restrict__ aux,
    const float* __restrict__ working, int H, int W, float score_floor,
    const float* __restrict__ writer_coarse_reference,
    const u8* __restrict__ floor_enabled, const int* __restrict__ row_gate,
    int cross_neighbor_mode, int coarse_enabled,
    const float* __restrict__ coarse_slopes,
    const u8* __restrict__ band_enabled,
    const float* __restrict__ band_scales,
    const float* __restrict__ factors_a,   // [band][channel] row-major 3x3
    const float* __restrict__ factors_b,
    const float* __restrict__ configured_strengths,
    u8* __restrict__ attempted, double* __restrict__ candidate,
    float* __restrict__ original) {
  i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= selected_count) return;
  i64 pixel = selected[i];
  int y = (int)(pixel / W);
  int x = (int)(pixel - (i64)y * W);

  float source_rgb[3];
  for (int c = 0; c < 3; ++c) source_rgb[c] = working[pixel * 4 + c];
  original[i * 3 + 0] = source_rgb[0];
  original[i * 3 + 1] = source_rgb[1];
  original[i * 3 + 2] = source_rgb[2];
  candidate[i * 3 + 0] = 0.0;
  candidate[i * 3 + 1] = 0.0;
  candidate[i * 3 + 2] = 0.0;
  attempted[i] = 0;

  float fallback = writer_coarse_reference[y];
  float weights[4], features[5][4];
  feature_record_at(score, waux, aux, H, W, score_floor, y, x, score[pixel],
                    aux[pixel], fallback, weights, features[0]);

  // reconstruction.driver_forces_fallback
  bool floor_on = floor_enabled[y] != 0;
  if (row_gate[y] != 0) return;
  if (floor_on && weights[3] >= 1.0f) return;
  attempted[i] = 1;

  int record_count = 1;
  if (cross_neighbor_mode) {
    record_count = 5;
    const int offsets[4][2] = {{0, -1}, {0, 1}, {-1, 0}, {1, 0}};
    for (int n = 0; n < 4; ++n) {
      cross_feature_at(score, waux, aux, H, W, score_floor,
                       y + offsets[n][0], x + offsets[n][1], fallback,
                       features[n + 1]);
    }
  }

  // reconstruction.recovered_rgb_candidates with UNNORMALIZED_SUM policy
  float wrgb_patch[9][9][3];
  for (int dy = -4; dy <= 4; ++dy)
    for (int dx = -4; dx <= 4; ++dx)
      for (int c = 0; c < 3; ++c)
        wrgb_patch[dy + 4][dx + 4][c] = load_wrgb_hist(
            wrgb, working, H, W, score_floor, y + dy, x + dx, c);
  double averages[3][3];
  rgb_unscaled_averages(wrgb_patch, averages);
  float cands[3][3];
  for (int scale_index = 0; scale_index < 3; ++scale_index) {
    float denominator = weights[scale_index];
    if (denominator > 0.0f) {
      double reciprocal = 1.0 / (double)denominator;
      for (int c = 0; c < 3; ++c)
        cands[scale_index][c] =
            (float)(averages[scale_index][c] * reciprocal);
    } else {
      for (int c = 0; c < 3; ++c)
        cands[scale_index][c] = (float)averages[scale_index][c];
    }
  }

  // reconstruction.feature_band_ranges (float64 differences, float32 extrema)
  double range_min[3], range_max[3];
  for (int t = 0; t < 3; ++t) {
    double mn = (double)features[0][t + 1] - (double)features[0][t];
    double mx = mn;
    for (int r = 1; r < record_count; ++r) {
      double d = (double)features[r][t + 1] - (double)features[r][t];
      mn = d < mn ? d : mn;
      mx = d > mx ? d : mx;
    }
    range_min[t] = (double)(float)mn;
    range_max[t] = (double)(float)mx;
  }

  // reconstruction._automatic_strengths
  double strengths[3];
  {
    float m1 = weights[1];
    if (configured_strengths[0] == 0.0f) {
      m1 = (float)(2.0 * (double)m1);
      double clamped = (double)m1;
      clamped = clamped > 0.0 ? clamped : 0.0;
      clamped = clamped < 1.0 ? clamped : 1.0;
      m1 = (float)clamped;
      strengths[0] = (double)m1;
    } else {
      strengths[0] = (double)configured_strengths[0];
    }
    float m2 = weights[2];
    if (configured_strengths[1] == 0.0f) {
      if (m2 < 0.0f) m2 = 0.0f;
      strengths[1] = (double)m2;
    } else {
      strengths[1] = (double)configured_strengths[1];
    }
    if (configured_strengths[2] == 0.0f) {
      strengths[2] = (double)weights[3] * (double)weights[3];
    } else {
      strengths[2] = (double)configured_strengths[2];
    }
  }

  // reconstruction.combine_recovered_candidate
  double cand[3];
  for (int c = 0; c < 3; ++c) cand[c] = (double)cands[0][c];
  if (coarse_enabled) {
    double coarse_delta =
        (double)writer_coarse_reference[y] - (double)features[0][0];
    for (int c = 0; c < 3; ++c)
      cand[c] = cand[c] + (double)coarse_slopes[c] * coarse_delta;
  }
  for (int band = 0; band < 3; ++band) {
    if (!band_enabled[band]) continue;
    double band_scale = (double)band_scales[band];
    bool negative_band = (range_min[band] < 0.0) && (range_max[band] < 0.0);
    for (int c = 0; c < 3; ++c) {
      double coarse_value, fine_value;
      if (band == 0) {
        coarse_value = (double)cands[0][c];
        fine_value = (double)cands[1][c];
      } else if (band == 1) {
        coarse_value = (double)cands[1][c];
        fine_value = (double)cands[2][c];
      } else {
        coarse_value = (double)cands[2][c];
        fine_value = (double)source_rgb[c];
      }
      double difference = band_scale * (fine_value - coarse_value);
      double factor_a = (double)factors_a[band * 3 + c];
      double factor_b = (double)factors_b[band * 3 + c];
      double upper, lower;
      if (negative_band) {
        upper = factor_b * range_max[band];
        lower = factor_a * range_min[band];
      } else {
        upper = factor_a * range_max[band];
        lower = factor_b * range_min[band];
      }
      double residual = difference > upper
                            ? difference - upper
                            : (difference < lower ? difference - lower : 0.0);
      cand[c] = cand[c] + strengths[band] * residual;
    }
  }
  candidate[i * 3 + 0] = cand[0];
  candidate[i * 3 + 1] = cand[1];
  candidate[i * 3 + 2] = cand[2];
}

// ---------------------------------------------------------------------------
// stage 5 (host): the sequential conditional-dither writer chain now runs on
// one host CPU core through the compiled fast_cpu.kernels.write_band path
// (cuda_backend/host_writer.py) -- it was sequential by necessity (draw
// consumption depends on drawn values) and a single device thread paid an
// order of magnitude more wall time for the same schedule than one host
// core.  dither_delta stays here, unused by any remaining kernel, as the
// validated primitive tests/test_cuda_level1_primitives.py still checks
// bit-for-bit against dither.conditional_dither_delta -- a standing guard
// against the device and host math ever drifting apart.
// ---------------------------------------------------------------------------

__device__ __forceinline__ double dither_delta(
    double value, double low, double high, bool low_lt_high, float scale32,
    u32* state, u64* advances) {
  float candidate32 = (float)value;
  double candidate = (double)candidate32;
  if (!low_lt_high) return 0.0;
  if (!(low < candidate && candidate < high)) return 0.0;
  double width = high - low;
  double coefficient = 4.0 / (width * width);
  double envelope = ((high - candidate) * (candidate - low)) * coefficient;
  float random_span = (float)((double)scale32 * candidate);
  *state = (125u * (*state) + 1u) & 0x00FFFFFFu;
  *advances += 1;
  double centered = ((double)(*state + 1u)) * NIKON_NORM - 0.5;
  double random_value = 0.0 + centered * (double)random_span;
  double delta = envelope * random_value;
  double changed = candidate + delta;
  if (low < changed && changed < high) return delta;
  return 0.0;
}

// ---------------------------------------------------------------------------
// stage 6: output assembly (output.emit_public_rgb16)
// ---------------------------------------------------------------------------

__device__ __forceinline__ u16 emit_one(float value, const u32* factor_high,
                                        const u32* factor_low) {
  double widened = (double)value + 0.5;
  widened = widened < 0.0 ? 0.0 : (widened > 65535.0 ? 65535.0 : widened);
  u32 index = (u32)trunc(widened);
  u64 product = ((u64)factor_high[index >> 8] * (u64)factor_low[index & 0xFF]);
  return (u16)((product >> 20) - 1ULL);
}

extern "C" __global__ void k_copy_visible(
    const float* __restrict__ working, i64 total,
    float* __restrict__ work_output) {
  i64 idx = (i64)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  work_output[idx * 3 + 0] = working[idx * 4 + 0];
  work_output[idx * 3 + 1] = working[idx * 4 + 1];
  work_output[idx * 3 + 2] = working[idx * 4 + 2];
}

extern "C" __global__ void k_scatter_values(
    const i64* __restrict__ selected, i64 selected_count,
    const float* __restrict__ values, float* __restrict__ work_output,
    u32* __restrict__ error_flags) {
  i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= selected_count) return;
  i64 pixel = selected[i];
  for (int c = 0; c < 3; ++c) {
    float value = values[i * 3 + c];
    // work_value_indices fails closed on nonfinite work values; emit_one's
    // clamp would otherwise launder them into 0 or 65535.
    if (!isfinite(value)) atomicOr(&error_flags[0], 2u);
    work_output[pixel * 3 + c] = value;
  }
}

extern "C" __global__ void k_emit_rgb16(
    const float* __restrict__ work_output, const u32* __restrict__ factor_high,
    const u32* __restrict__ factor_low, i64 total, u16* __restrict__ out) {
  i64 idx = (i64)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  out[idx] = emit_one(work_output[idx], factor_high, factor_low);
}

// per-attempted-site changed-pixel accounting against the no-op emit
extern "C" __global__ void k_site_counters(
    const u8* __restrict__ attempted, const float* __restrict__ values,
    const float* __restrict__ original, const u8* __restrict__ written,
    i64 selected_count, const u32* __restrict__ factor_high,
    const u32* __restrict__ factor_low,
    unsigned long long* __restrict__ counters) {
  // counters: [0] attempted, [1] written, [2] changed
  i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= selected_count) return;
  if (!attempted[i]) return;
  atomicAdd(&counters[0], 1ULL);
  if (written[i]) atomicAdd(&counters[1], 1ULL);
  bool changed = false;
  for (int c = 0; c < 3; ++c) {
    u16 rendered = emit_one(values[i * 3 + c], factor_high, factor_low);
    u16 noop = emit_one(original[i * 3 + c], factor_high, factor_low);
    changed |= rendered != noop;
  }
  if (changed) atomicAdd(&counters[2], 1ULL);
}

// ---------------------------------------------------------------------------
// content-derived producer (producer_parameters) row/epoch primitives
// ---------------------------------------------------------------------------

// first failing (row, column) index inside each 8x8 producer cell-block,
// in the row-major scan order of derive_producer_mean_schedule.
extern "C" __global__ void k_producer_failpos(
    const u16* __restrict__ rgbi, int H, int W, int active_width,
    float threshold, int block_count, int cell_count,
    int* __restrict__ failpos) {
  int block = blockIdx.y * blockDim.y + threadIdx.y;
  int cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (block >= block_count || cell >= cell_count) return;
  int row0 = block * 8;
  int rows = H - row0 < 8 ? H - row0 : 8;
  int col0 = cell * 8;
  double thr = (double)threshold;
  int fail = rows * 8;  // no failure inside this block
  for (int r = 0; r < rows && fail == rows * 8; ++r) {
    for (int c = 0; c < 8; ++c) {
      u16 raw = rgbi[(((i64)(row0 + r) * W) + col0 + c) * 4 + 3];
      if (!(thr < (double)raw)) {
        fail = r * 8 + c;
        break;
      }
    }
  }
  failpos[(i64)block * cell_count + cell] = fail;
}

// per-row binary64 sums in exact column order (one thread per row)
extern "C" __global__ void k_producer_row_sums(
    const u16* __restrict__ rgbi, const float* __restrict__ lut, int H, int W,
    int active_width, int visible_channel, int cell_count,
    const int* __restrict__ failpos, double* __restrict__ row_visible,
    double* __restrict__ row_infrared, double* __restrict__ row_weight,
    u32* __restrict__ row_accepted) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= H) return;
  int block = row / 8;
  int row_in_block = row - block * 8;
  double visible_sum = 0.0, infrared_sum = 0.0, weight_sum = 0.0;
  u32 accepted = 0;
  const int* block_fail = failpos + (i64)block * cell_count;
  for (int column = 0; column < active_width; ++column) {
    int cell = column >> 3;
    int col_in_cell = column & 7;
    int index = row_in_block * 8 + col_in_cell;
    if (index >= block_fail[cell]) continue;
    u16 raw_ir = rgbi[(((i64)row * W) + column) * 4 + 3];
    u16 raw_vis = rgbi[(((i64)row * W) + column) * 4 + visible_channel];
    double weight = (double)((u32)raw_ir * (u32)raw_ir);
    visible_sum += weight * (double)lut[raw_vis];
    infrared_sum += weight * (double)lut[raw_ir];
    weight_sum += weight;
    accepted += 1;
  }
  row_visible[row] = visible_sum;
  row_infrared[row] = infrared_sum;
  row_weight[row] = weight_sum;
  row_accepted[row] = accepted;
}

// one complete eight-row scale epoch (producer_parameters._scale_epoch_additions)
extern "C" __global__ void k_producer_scale_epochs(
    const u16* __restrict__ rgbi, const float* __restrict__ lut, int H, int W,
    int active_width, int visible_channel, float threshold, int epoch_count,
    float* __restrict__ add_denominator, float* __restrict__ add_numerator) {
  int epoch = blockIdx.x * blockDim.x + threadIdx.x;
  if (epoch >= epoch_count) return;
  int row0 = epoch * 8;
  double thr = (double)threshold;
  const double low = (double)(-0.125f);
  const double full_low = 0.0;
  const double full_high = (double)(0.3f);
  const double high = (double)(0.425f);
  const double outer_weight = (double)(0.70710677f);
  double denominator = 0.0, numerator = 0.0;
  for (int column = 0; column < active_width; column += 8) {
    bool block_ok = true;
    for (int r = 0; r < 8 && block_ok; ++r)
      for (int c = 0; c < 8; ++c) {
        u16 raw = rgbi[(((i64)(row0 + r) * W) + column + c) * 4 + 3];
        if ((double)raw <= thr) {
          block_ok = false;
          break;
        }
      }
    if (!block_ok) continue;
    // stored float32 4x4 quadrant means over response values, exact x87 tree
    float visible_means[4], infrared_means[4];
    for (int lane = 0; lane < 2; ++lane) {
      float* means = lane == 0 ? visible_means : infrared_means;
      int channel = lane == 0 ? visible_channel : 3;
      int quadrant = 0;
      for (int rs = 0; rs < 8; rs += 4)
        for (int cs = 0; cs < 8; cs += 4) {
          float v[4][4];
          for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
              v[r][c] = lut[rgbi[(((i64)(row0 + rs + r) * W) + column + cs +
                                  c) *
                                     4 +
                                 channel]];
          double first = (double)v[0][0];
          first += (double)v[0][1];
          first += (double)v[0][2];
          first += (double)v[0][3];
          first += (double)v[1][0];
          first += (double)v[1][1];
          double second_tail = (double)v[1][2] + (double)v[1][3];
          first += second_tail;
          double third = (double)v[2][0] + (double)v[2][1];
          double third_tail = (double)v[2][2] + (double)v[2][3];
          third += third_tail;
          double fourth = (double)v[3][0] + (double)v[3][1];
          double fourth_tail = (double)v[3][2] + (double)v[3][3];
          fourth += fourth_tail;
          third += fourth;
          double total = first + third;
          means[quadrant++] = (float)(total * 0.0625);
        }
    }
    // _center_quadrants
    float visible_dev[4], infrared_dev[4];
    for (int lane = 0; lane < 2; ++lane) {
      const float* means = lane == 0 ? visible_means : infrared_means;
      float* dev = lane == 0 ? visible_dev : infrared_dev;
      double total = (double)means[0];
      total += (double)means[1];
      total += (double)means[2];
      total += (double)means[3];
      float mean = (float)(total * 0.25);
      for (int q = 0; q < 4; ++q) dev[q] = (float)((double)means[q] - (double)mean);
    }
    // float32 running sum of raw infrared in row-major order
    float raw_sum = 0.0f;
    for (int r = 0; r < 8; ++r)
      for (int c = 0; c < 8; ++c) {
        u16 raw = rgbi[(((i64)(row0 + r) * W) + column + c) * 4 + 3];
        raw_sum = (float)((double)raw_sum + (double)raw);
      }
    double raw_sum_wide = (double)raw_sum;
    for (int q = 0; q < 4; ++q) {
      double visible_wide = (double)visible_dev[q];
      double ratio, weight;
      if (visible_wide == 0.0) {
        ratio = 0.0;
        weight = 0.0;
      } else {
        ratio = (double)infrared_dev[q] / visible_wide;
        if (ratio < low || ratio > high) {
          weight = 0.0;
        } else if (ratio < full_low || ratio > full_high) {
          weight = visible_wide * outer_weight;
        } else {
          weight = visible_wide;
        }
      }
      double term = weight * weight;
      term *= raw_sum_wide;
      term *= raw_sum_wide;
      denominator += term;
      numerator += ratio * term;
    }
  }
  add_denominator[epoch] = (float)denominator;
  add_numerator[epoch] = (float)numerator;
}
"""


def render_kernel_source() -> str:
    """Inject exact host-quantized constants into the CUDA source."""

    return (
        KERNEL_SOURCE.replace("__COEFF69__", COEFFICIENT_69)
        .replace("__COEFF21__", COEFFICIENT_21)
        .replace("__COEFF16__", COEFFICIENT_16)
        .replace("__NIKON_NORM__", NIKON_NORMALIZATION_HEX)
    )
