"""Metal Shading Language sources for the deterministic LS-5000 backend.

Apple GPUs have no binary64 hardware and MSL has no ``double`` type, so the
kernels carry binary64 values as ``ulong`` bit patterns and perform every
binary64 operation in software: a faithful integer-arithmetic implementation
of IEEE-754 round-to-nearest-even following the classic SoftFloat
construction.  This is stricter than a compiler flag.  The CUDA backend needs
``--fmad=false`` to stop the compiler contracting multiply/add pairs; here
the binary64 schedule is integer arithmetic end to end, which no compiler
mode can contract, reassociate, or flush.  The float32 multiplies the
reference performs at store boundaries are composed from the same softfloat
path (a float32 product is exact in binary64, so one narrowing reproduces
the correctly rounded float32 multiply, subnormals included), leaving native
GPU float behavior in charge of nothing but comparisons.

Every kernel is a line-by-line translation of the audited CPU reference,
matching the CUDA port operation for operation:

- binary64 arithmetic with one rounding per written operation, and float32
  narrowing only at the reference's recorded store boundaries;
- the 24-bit LCG and conditional-dither writer are sequential by necessity
  and run on one host CPU core via the compiled ``fast_cpu`` path (the same
  host writer the CUDA backend uses); the ``dither_delta`` device function
  stays here as a validated primitive only, unreachable from any pipeline
  kernel, kept as a standing drift guard between device and host math;
- numeric constants that the reference derives by quantizing Python floats
  to float32 (for example ``float32(1/69)``) are injected from host Python
  as exact binary64 bit patterns so the kernel text cannot drift from the
  reference quantization.  MSL has no binary64 literals, so every binary64
  constant is injected this way.
"""

from __future__ import annotations

import numpy as np


def _f64_bits(value: float) -> str:
    return f"0x{int(np.float64(value).view(np.uint64)):016X}ul"


_CONSTANTS = {
    "F64C_HALF": 0.5,
    "F64C_ONE": 1.0,
    "F64C_TWO": 2.0,
    "F64C_FOUR": 4.0,
    "F64C_65535": 65535.0,
    "F64C_SIXTEENTH_EXACT": 0.0625,
    "F64C_QUARTER": 0.25,
    # The reference multiplies by float32-quantized reciprocals widened to
    # binary64 (reconstruction._recovered_unscaled_averages).
    "F64C_COEFF69": float(np.float32(1.0 / 69.0)),
    "F64C_COEFF21": float(np.float32(1.0 / 21.0)),
    "F64C_COEFF16": float(np.float32(1.0 / 16.0)),
    # ICEDLL.dll's biased normalization constant, the float32 value just
    # below exact 2**-24, expressed exactly (rng.NIKON_NORMALIZATION).
    "F64C_NIKON_NORM": float.fromhex("0x1.fffffep-25"),
    # producer_parameters._scale_epoch_additions ratio-window constants,
    # float32-quantized by the reference before widening.
    "F64C_PROD_LOW": float(np.float32(-0.125)),
    "F64C_PROD_FULL_HIGH": float(np.float32(0.3)),
    "F64C_PROD_HIGH": float(np.float32(0.425)),
    "F64C_PROD_OUTER": float(np.float32(0.70710677)),
}


KERNEL_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

typedef ulong u64;
typedef uint u32;
typedef ushort u16;
typedef uchar u8;

__CONSTANT_DEFINES__

// ===========================================================================
// software IEEE-754 binary64
//
// Bit patterns travel in ulong.  Working significands follow the SoftFloat
// convention: normal range [2^62, 2^63) with ten trailing rounding bits, and
// an exponent that pairs with that range so that packing merges the
// significand's integer bit into the exponent field.
// ===========================================================================

constant u64 F64_SIGN = 0x8000000000000000ul;
constant u64 F64_EXP_MASK = 0x7FF0000000000000ul;
constant u64 F64_FRAC_MASK = 0x000FFFFFFFFFFFFFul;
constant u64 F64_QNAN = 0x7FF8000000000000ul;
constant u64 F64_ZERO = 0x0000000000000000ul;

inline uint clz64(u64 x) {
  u32 hi = (u32)(x >> 32);
  u32 lo = (u32)(x & 0xFFFFFFFFul);
  return hi != 0u ? clz(hi) : 32u + clz(lo);
}

inline u64 sf_shr_jam(u64 x, uint n) {
  if (n == 0u) return x;
  if (n >= 64u) return x != 0ul ? 1ul : 0ul;
  return (x >> n) | (((x << (64u - n)) != 0ul) ? 1ul : 0ul);
}

inline u32 sf_shr_jam32(u32 x, uint n) {
  if (n == 0u) return x;
  if (n >= 32u) return x != 0u ? 1u : 0u;
  return (x >> n) | (((x << (32u - n)) != 0u) ? 1u : 0u);
}

inline void sf_mul_64_128(u64 x, u64 y, thread u64* hi, thread u64* lo) {
  u64 a0 = x & 0xFFFFFFFFul, a1 = x >> 32;
  u64 b0 = y & 0xFFFFFFFFul, b1 = y >> 32;
  u64 t = a0 * b0;
  u64 w0 = t & 0xFFFFFFFFul;
  u64 k = t >> 32;
  t = a1 * b0 + k;
  u64 w1 = t & 0xFFFFFFFFul;
  u64 w2 = t >> 32;
  t = a0 * b1 + w1;
  k = t >> 32;
  *hi = a1 * b1 + w2 + k;
  *lo = (t << 32) | w0;
}

inline u64 sf_round_pack(bool sign, int exp, u64 sig) {
  if ((uint)exp >= 0x7FDu) {
    if (exp < 0) {
      sig = sf_shr_jam(sig, (uint)(-exp));
      exp = 0;
    } else if (exp > 0x7FD || (long)(sig + 0x200ul) < 0l) {
      return (sign ? F64_SIGN : 0ul) + F64_EXP_MASK;
    }
  }
  u32 round_bits = (u32)(sig & 0x3FFul);
  sig = (sig + 0x200ul) >> 10;
  if (round_bits == 0x200u) sig &= ~1ul;
  if (sig == 0ul) exp = 0;
  return (sign ? F64_SIGN : 0ul) + ((u64)(u32)exp << 52) + sig;
}

inline u64 sf_norm_round_pack(bool sign, int exp, u64 sig) {
  int shift = (int)clz64(sig) - 1;
  exp -= shift;
  if (shift >= 10 && (uint)exp < 0x7FDu) {
    u64 packed_sig = sig != 0ul ? (sig << (uint)(shift - 10)) : 0ul;
    int packed_exp = sig != 0ul ? exp : 0;
    return (sign ? F64_SIGN : 0ul) + ((u64)(u32)packed_exp << 52) + packed_sig;
  }
  return sf_round_pack(sign, exp, sig << (uint)shift);
}

inline u64 sf_add_mags(u64 a, u64 b, bool sign) {
  int exp_a = (int)((a >> 52) & 0x7FFul);
  u64 sig_a = a & F64_FRAC_MASK;
  int exp_b = (int)((b >> 52) & 0x7FFul);
  u64 sig_b = b & F64_FRAC_MASK;
  int exp_diff = exp_a - exp_b;
  if (exp_diff == 0) {
    if (exp_a == 0) {
      // both subnormal or zero: integer addition of encodings is exact
      return (sign ? F64_SIGN : 0ul) + (a & ~F64_SIGN) + sig_b;
    }
    if (exp_a == 0x7FF) {
      if ((sig_a | sig_b) != 0ul) return F64_QNAN;
      return (sign ? F64_SIGN : 0ul) | F64_EXP_MASK;
    }
    u64 sig = 0x0020000000000000ul + sig_a + sig_b;
    return sf_round_pack(sign, exp_a, sig << 9);
  }
  sig_a <<= 9;
  sig_b <<= 9;
  int exp;
  if (exp_diff < 0) {
    if (exp_b == 0x7FF) {
      if (sig_b != 0ul) return F64_QNAN;
      return (sign ? F64_SIGN : 0ul) | F64_EXP_MASK;
    }
    exp = exp_b;
    sig_a += (exp_a != 0) ? 0x2000000000000000ul : sig_a;
    sig_a = sf_shr_jam(sig_a, (uint)(-exp_diff));
  } else {
    if (exp_a == 0x7FF) {
      if (sig_a != 0ul) return F64_QNAN;
      return (sign ? F64_SIGN : 0ul) | F64_EXP_MASK;
    }
    exp = exp_a;
    sig_b += (exp_b != 0) ? 0x2000000000000000ul : sig_b;
    sig_b = sf_shr_jam(sig_b, (uint)exp_diff);
  }
  u64 sig = 0x2000000000000000ul + sig_a + sig_b;
  if (sig < 0x4000000000000000ul) {
    --exp;
    sig <<= 1;
  }
  return sf_round_pack(sign, exp, sig);
}

inline u64 sf_sub_mags(u64 a, u64 b, bool sign) {
  int exp_a = (int)((a >> 52) & 0x7FFul);
  u64 sig_a = a & F64_FRAC_MASK;
  int exp_b = (int)((b >> 52) & 0x7FFul);
  u64 sig_b = b & F64_FRAC_MASK;
  int exp_diff = exp_a - exp_b;
  if (exp_diff == 0) {
    if (exp_a == 0x7FF) {
      return F64_QNAN;  // NaN handled by the caller; here inf - inf
    }
    long sig_diff = (long)sig_a - (long)sig_b;
    if (sig_diff == 0l) return 0ul;  // exact cancellation gives +0 in RNE
    if (sig_diff < 0l) {
      sign = !sign;
      sig_diff = -sig_diff;
    }
    u64 mag = (u64)sig_diff;
    if (exp_a == 0) {
      return (sign ? F64_SIGN : 0ul) | mag;  // both subnormal: exact
    }
    int shift = (int)clz64(mag) - 11;
    int exp_z = exp_a - shift - 1;
    if (exp_z < 0) {
      return (sign ? F64_SIGN : 0ul) | (mag << (uint)(exp_a - 1));
    }
    return (sign ? F64_SIGN : 0ul) + ((u64)(u32)exp_z << 52) +
           (mag << (uint)shift);
  }
  sig_a <<= 10;
  sig_b <<= 10;
  int exp;
  u64 sig;
  if (exp_diff < 0) {
    if (exp_b == 0x7FF) {
      if (sig_b != 0ul) return F64_QNAN;
      return (!sign ? F64_SIGN : 0ul) | F64_EXP_MASK;  // finite minus inf
    }
    sign = !sign;
    sig_a += (exp_a != 0) ? 0x4000000000000000ul : sig_a;
    sig_a = sf_shr_jam(sig_a, (uint)(-exp_diff));
    sig_b |= 0x4000000000000000ul;
    exp = exp_b;
    sig = sig_b - sig_a;
  } else {
    if (exp_a == 0x7FF) {
      if (sig_a != 0ul) return F64_QNAN;
      return (sign ? F64_SIGN : 0ul) | F64_EXP_MASK;  // inf minus finite
    }
    sig_b += (exp_b != 0) ? 0x4000000000000000ul : sig_b;
    sig_b = sf_shr_jam(sig_b, (uint)exp_diff);
    sig_a |= 0x4000000000000000ul;
    exp = exp_a;
    sig = sig_a - sig_b;
  }
  return sf_norm_round_pack(sign, exp - 1, sig);
}

inline bool f64_is_nan(u64 a) {
  return (a & F64_EXP_MASK) == F64_EXP_MASK && (a & F64_FRAC_MASK) != 0ul;
}

inline u64 f64_add(u64 a, u64 b) {
  bool sign_a = (a >> 63) != 0ul;
  bool sign_b = (b >> 63) != 0ul;
  if (f64_is_nan(a) || f64_is_nan(b)) return F64_QNAN;
  if (sign_a == sign_b) return sf_add_mags(a, b, sign_a);
  if ((a & ~F64_SIGN) == 0ul && (b & ~F64_SIGN) == 0ul) {
    return 0ul;  // opposite-signed zeros sum to +0 in RNE
  }
  return sf_sub_mags(a, b, sign_a);
}

inline u64 f64_sub(u64 a, u64 b) { return f64_add(a, b ^ F64_SIGN); }

inline u64 f64_mul(u64 a, u64 b) {
  bool sign = ((a ^ b) >> 63) != 0ul;
  int exp_a = (int)((a >> 52) & 0x7FFul);
  u64 sig_a = a & F64_FRAC_MASK;
  int exp_b = (int)((b >> 52) & 0x7FFul);
  u64 sig_b = b & F64_FRAC_MASK;
  if (exp_a == 0x7FF || exp_b == 0x7FF) {
    if ((exp_a == 0x7FF && sig_a != 0ul) || (exp_b == 0x7FF && sig_b != 0ul)) {
      return F64_QNAN;
    }
    if ((exp_a == 0x7FF && exp_b == 0 && sig_b == 0ul) ||
        (exp_b == 0x7FF && exp_a == 0 && sig_a == 0ul)) {
      return F64_QNAN;  // inf times zero
    }
    return (sign ? F64_SIGN : 0ul) | F64_EXP_MASK;
  }
  if (exp_a == 0) {
    if (sig_a == 0ul) return sign ? F64_SIGN : 0ul;
    int shift = (int)clz64(sig_a) - 11;
    sig_a <<= (uint)shift;
    exp_a = 1 - shift;
  }
  if (exp_b == 0) {
    if (sig_b == 0ul) return sign ? F64_SIGN : 0ul;
    int shift = (int)clz64(sig_b) - 11;
    sig_b <<= (uint)shift;
    exp_b = 1 - shift;
  }
  int exp = exp_a + exp_b - 0x3FF;
  sig_a = (sig_a | 0x0010000000000000ul) << 10;
  sig_b = (sig_b | 0x0010000000000000ul) << 11;
  u64 hi, lo;
  sf_mul_64_128(sig_a, sig_b, &hi, &lo);
  u64 sig = hi | (lo != 0ul ? 1ul : 0ul);
  if (sig < 0x4000000000000000ul) {
    --exp;
    sig <<= 1;
  }
  return sf_round_pack(sign, exp, sig);
}

inline u64 f64_div(u64 a, u64 b) {
  bool sign = ((a ^ b) >> 63) != 0ul;
  int exp_a = (int)((a >> 52) & 0x7FFul);
  u64 sig_a = a & F64_FRAC_MASK;
  int exp_b = (int)((b >> 52) & 0x7FFul);
  u64 sig_b = b & F64_FRAC_MASK;
  if (exp_a == 0x7FF || exp_b == 0x7FF) {
    if ((exp_a == 0x7FF && sig_a != 0ul) || (exp_b == 0x7FF && sig_b != 0ul)) {
      return F64_QNAN;
    }
    if (exp_a == 0x7FF && exp_b == 0x7FF) return F64_QNAN;
    if (exp_a == 0x7FF) return (sign ? F64_SIGN : 0ul) | F64_EXP_MASK;
    return sign ? F64_SIGN : 0ul;
  }
  if (exp_b == 0 && sig_b == 0ul) {
    if (exp_a == 0 && sig_a == 0ul) return F64_QNAN;
    return (sign ? F64_SIGN : 0ul) | F64_EXP_MASK;
  }
  if (exp_a == 0 && sig_a == 0ul) return sign ? F64_SIGN : 0ul;
  if (exp_a == 0) {
    int shift = (int)clz64(sig_a) - 11;
    sig_a <<= (uint)shift;
    exp_a = 1 - shift;
  }
  if (exp_b == 0) {
    int shift = (int)clz64(sig_b) - 11;
    sig_b <<= (uint)shift;
    exp_b = 1 - shift;
  }
  int exp = exp_a - exp_b + 0x3FE;
  sig_a |= 0x0010000000000000ul;
  sig_b |= 0x0010000000000000ul;
  u64 rem = sig_a;
  if (rem < sig_b) {
    rem <<= 1;
    --exp;
  }
  u64 q = 0ul;
  for (int i = 0; i < 62; ++i) {
    q <<= 1;
    if (rem >= sig_b) {
      rem -= sig_b;
      q |= 1ul;
    }
    rem <<= 1;
  }
  u64 sig = (q << 1) | (rem != 0ul ? 1ul : 0ul);
  return sf_round_pack(sign, exp, sig);
}

inline u64 f64_from_f32(float f) {
  u32 u = as_type<u32>(f);
  u64 sign = ((u64)(u >> 31)) << 63;
  u32 exp = (u >> 23) & 0xFFu;
  u32 frac = u & 0x7FFFFFu;
  if (exp == 0xFFu) {
    return frac != 0u ? F64_QNAN : (sign | F64_EXP_MASK);
  }
  if (exp == 0u) {
    if (frac == 0u) return sign;
    uint shift = clz(frac) - 8u;
    frac <<= shift;
    u32 e64 = 0x381u - shift;
    return sign | ((u64)e64 << 52) | (((u64)frac & 0x7FFFFFul) << 29);
  }
  return sign | ((u64)(exp + 0x380u) << 52) | ((u64)frac << 29);
}

inline float f32_from_f64(u64 u) {
  bool sign = (u >> 63) != 0ul;
  int exp = (int)((u >> 52) & 0x7FFul);
  u64 frac = u & F64_FRAC_MASK;
  u32 s32 = sign ? 0x80000000u : 0u;
  if (exp == 0x7FF) {
    return as_type<float>(frac != 0ul ? (s32 | 0x7FC00000u)
                                      : (s32 | 0x7F800000u));
  }
  if (exp == 0) {
    // binary64 subnormals sit far below half the smallest binary32
    // subnormal, so round-to-nearest-even collapses them to signed zero.
    return as_type<float>(s32);
  }
  u32 sig = (u32)sf_shr_jam(frac, 22u) | 0x40000000u;
  int exp32 = exp - 0x381;
  if ((uint)exp32 >= 0xFDu) {
    if (exp32 < 0) {
      sig = sf_shr_jam32(sig, (uint)(-exp32));
      exp32 = 0;
    } else if (exp32 > 0xFD || (int)(sig + 0x40u) < 0) {
      return as_type<float>(s32 | 0x7F800000u);
    }
  }
  u32 round_bits = sig & 0x7Fu;
  sig = (sig + 0x40u) >> 7;
  if (round_bits == 0x40u) sig &= ~1u;
  if (sig == 0u) exp32 = 0;
  return as_type<float>(s32 + ((u32)exp32 << 23) + sig);
}

inline u64 f64_from_u32(u32 v) {
  if (v == 0u) return 0ul;
  uint shift = clz(v);
  u64 sig = ((u64)v) << (shift + 21u);
  u64 e = (u64)(1023u + 31u - shift);
  return (e << 52) | (sig & F64_FRAC_MASK);
}

// Truncate a nonnegative binary64 to u32.  The pipeline calls this only
// after clamping to [0, 65535], so the shift is always in range.
inline u32 f64_trunc_u32(u64 u) {
  int exp = (int)((u >> 52) & 0x7FFul);
  if (exp < 0x3FF) return 0u;
  u64 sig = 0x0010000000000000ul | (u & F64_FRAC_MASK);
  int shift = 1075 - exp;
  if (shift <= 0) return (u32)(sig << (uint)(-shift));
  if (shift >= 64) return 0u;
  return (u32)(sig >> (uint)shift);
}

inline bool f64_lt(u64 a, u64 b) {
  if (f64_is_nan(a) || f64_is_nan(b)) return false;
  bool sign_a = (a >> 63) != 0ul;
  bool sign_b = (b >> 63) != 0ul;
  u64 mag_a = a & ~F64_SIGN;
  u64 mag_b = b & ~F64_SIGN;
  if (sign_a != sign_b) return sign_a && ((mag_a | mag_b) != 0ul);
  if (mag_a == mag_b) return false;
  return sign_a ? (mag_a > mag_b) : (mag_a < mag_b);
}

inline bool f64_le(u64 a, u64 b) {
  if (f64_is_nan(a) || f64_is_nan(b)) return false;
  bool sign_a = (a >> 63) != 0ul;
  bool sign_b = (b >> 63) != 0ul;
  u64 mag_a = a & ~F64_SIGN;
  u64 mag_b = b & ~F64_SIGN;
  if (sign_a != sign_b) return sign_a || ((mag_a | mag_b) == 0ul);
  if (mag_a == mag_b) return true;
  return sign_a ? (mag_a > mag_b) : (mag_a < mag_b);
}

inline bool f64_eq(u64 a, u64 b) {
  if (f64_is_nan(a) || f64_is_nan(b)) return false;
  if (a == b) return true;
  return ((a | b) & ~F64_SIGN) == 0ul;
}

// Correctly rounded float32 multiply composed from the binary64 path: a
// float32 product is exact in binary64 (24 + 24 significand bits), so one
// narrowing performs the single rounding, subnormal outputs included,
// independent of GPU float behavior.
inline float f32_mul(float a, float b) {
  return f32_from_f64(f64_mul(f64_from_f32(a), f64_from_f32(b)));
}

// ===========================================================================
// history loaders: X3A score / weighted-history boundary carrier
//   horizontal guards (x outside [0, W)) stay zero
//   logical row -1 is the pseudo row (first real row scaled by the floor)
//   rows below -1 stay zero, rows >= H repeat the final real row
// ===========================================================================

inline float load_score_hist(device const float* score, int H, int W,
                             float score_floor, int y, int x) {
  if (x < 0 || x >= W) return 0.0f;
  if (y < -1) return 0.0f;
  if (y == -1) return score_floor;
  if (y >= H) y = H - 1;
  return score[(long)y * W + x];
}

inline float load_waux_hist(device const float* waux, device const float* aux,
                            int H, int W, float score_floor, int y, int x) {
  if (x < 0 || x >= W) return 0.0f;
  if (y < -1) return 0.0f;
  if (y == -1) return f32_mul(aux[x], score_floor);  // float32 multiply
  if (y >= H) y = H - 1;
  return waux[(long)y * W + x];
}

inline float load_wrgb_hist(device const float* wrgb,
                            device const float* working, int H, int W,
                            float score_floor, int y, int x, int c) {
  if (x < 0 || x >= W) return 0.0f;
  if (y < -1) return 0.0f;
  if (y == -1) return f32_mul(working[(long)x * 4 + c], score_floor);
  if (y >= H) y = H - 1;
  return wrgb[((long)y * W + x) * 3 + c];
}

// ===========================================================================
// reconstruction._vertical_scratch + _vertical_binomial_16 +
// _recovered_unscaled_averages for one scalar 9x9 patch.
// The accumulation order and float32 store boundaries are load-bearing.
// ===========================================================================

inline void unscaled_averages_scalar(thread const float (*p)[9],
                                     thread u64* q) {
  float s3[9], s5[9], s7[9], s9[9], b16[9];
  for (int x = 0; x < 9; ++x) {
    u64 acc = f64_add(f64_from_f32(p[3][x]), f64_from_f32(p[4][x]));
    acc = f64_add(acc, f64_from_f32(p[5][x]));
    s3[x] = f32_from_f64(acc);
    acc = f64_add(acc, f64_from_f32(p[2][x]));
    acc = f64_add(acc, f64_from_f32(p[6][x]));
    s5[x] = f32_from_f64(acc);
    acc = f64_add(acc, f64_from_f32(p[1][x]));
    acc = f64_add(acc, f64_from_f32(p[7][x]));
    s7[x] = f32_from_f64(acc);
    acc = f64_add(acc, f64_from_f32(p[0][x]));
    acc = f64_add(acc, f64_from_f32(p[8][x]));
    s9[x] = f32_from_f64(acc);
    u64 center = f64_from_f32(p[4][x]);
    u64 bacc = f64_add(f64_from_f32(p[3][x]), center);
    bacc = f64_add(bacc, f64_from_f32(p[5][x]));
    bacc = f64_add(bacc, center);
    b16[x] = f32_from_f64(f64_mul(bacc, F64C_COEFF16));
  }
  u64 t21 = f64_add(f64_from_f32(s3[6]), f64_from_f32(s3[2]));
  t21 = f64_add(t21, f64_from_f32(s5[3]));
  t21 = f64_add(t21, f64_from_f32(s5[4]));
  t21 = f64_add(t21, f64_from_f32(s5[5]));
  u64 t69 = f64_add(f64_from_f32(s5[8]), f64_from_f32(s5[0]));
  t69 = f64_add(t69, f64_from_f32(s7[1]));
  t69 = f64_add(t69, f64_from_f32(s7[7]));
  for (int x = 2; x < 7; ++x) t69 = f64_add(t69, f64_from_f32(s9[x]));
  u64 t16 = f64_add(f64_from_f32(b16[4]), f64_from_f32(b16[3]));
  t16 = f64_add(t16, f64_from_f32(b16[5]));
  t16 = f64_add(t16, f64_from_f32(b16[4]));
  q[0] = f64_mul(t69, F64C_COEFF69);
  q[1] = f64_mul(t21, F64C_COEFF21);
  q[2] = t16;
}

// reconstruction._recovered_rgb_unscaled_averages: shares Q69/Q21 with the
// scalar helper, but the 3x3 binomial path keeps binary64 precision until
// one final multiply (no intermediate float32 store).
inline void rgb_unscaled_averages(thread const float (*p)[9][3],
                                  thread u64 (*q)[3]) {
  for (int c = 0; c < 3; ++c) {
    float lane[9][9];
    for (int y = 0; y < 9; ++y)
      for (int x = 0; x < 9; ++x) lane[y][x] = p[y][x][c];
    u64 ql[3];
    unscaled_averages_scalar(lane, ql);
    u64 rows[3];
    for (int r = 0; r < 3; ++r) {
      int y = 3 + r;
      u64 center = f64_from_f32(lane[y][4]);
      u64 horizontal = f64_add(f64_from_f32(lane[y][3]),
                               f64_from_f32(lane[y][5]));
      horizontal = f64_add(horizontal, center);
      horizontal = f64_add(horizontal, center);
      rows[r] = horizontal;
    }
    u64 total = f64_add(rows[0], rows[1]);
    total = f64_add(total, rows[1]);
    total = f64_add(total, rows[2]);
    q[0][c] = ql[0];
    q[1][c] = ql[1];
    q[2][c] = f64_mul(total, F64C_COEFF16);
  }
}

// streaming._feature_record on gathered history patches.  ``uw`` keeps the
// unrounded binary64 weights: features divide by the in-register value, not
// the float32 W record.
inline void feature_record_at(device const float* score,
                              device const float* waux,
                              device const float* aux, int H, int W,
                              float score_floor, int cy, int cx,
                              float point_score, float point_aux,
                              float fallback, thread float* w_out,
                              thread float* f_out) {
  float score_patch[9][9], waux_patch[9][9];
  for (int dy = -4; dy <= 4; ++dy) {
    for (int dx = -4; dx <= 4; ++dx) {
      score_patch[dy + 4][dx + 4] =
          load_score_hist(score, H, W, score_floor, cy + dy, cx + dx);
      waux_patch[dy + 4][dx + 4] =
          load_waux_hist(waux, aux, H, W, score_floor, cy + dy, cx + dx);
    }
  }
  u64 uw[3], num[3];
  unscaled_averages_scalar(score_patch, uw);
  unscaled_averages_scalar(waux_patch, num);
  w_out[0] = f32_from_f64(uw[0]);
  w_out[1] = f32_from_f64(uw[1]);
  w_out[2] = f32_from_f64(uw[2]);
  w_out[3] = point_score;
  for (int lane = 0; lane < 3; ++lane) {
    f_out[lane] = f64_lt(F64_ZERO, uw[lane])
                      ? f32_from_f64(f64_div(num[lane], uw[lane]))
                      : fallback;
  }
  f_out[3] = (w_out[3] > 0.0f) ? point_aux : fallback;
}

// streaming._cross_neighbor_feature_record point-sample boundary rules.
inline void cross_feature_at(device const float* score,
                             device const float* waux,
                             device const float* aux, int H, int W,
                             float score_floor, int ny, int nx, float fallback,
                             thread float* f_out) {
  if (nx < 0 || nx >= W) {
    f_out[0] = f_out[1] = f_out[2] = f_out[3] = 0.0f;
    return;
  }
  float point_aux, point_score;
  if (ny < 0) {
    point_aux = aux[nx];
    point_score = score_floor;
  } else if (ny >= H) {
    point_aux = aux[(long)(H - 1) * W + nx];
    point_score = score[(long)(H - 1) * W + nx];
  } else {
    point_aux = aux[(long)ny * W + nx];
    point_score = score[(long)ny * W + nx];
  }
  float w_unused[4];
  feature_record_at(score, waux, aux, H, W, score_floor, ny, nx, point_score,
                    point_aux, fallback, w_unused, f_out);
}

// ===========================================================================
// stage 1: response LUT + auxiliary plane (x3a.derive_auxiliary)
// iparams: 0 visible_channel, 1 H, 2 W
// ===========================================================================

kernel void k_convert_and_auxiliary(
    device const u16* rgbi [[buffer(0)]],
    device const float* lut [[buffer(1)]],
    device const float* aux_alpha [[buffer(2)]],
    device const u8* alpha_is_one [[buffer(3)]],
    device const float* alpha_one_replacement [[buffer(4)]],
    device const float* aux_offset [[buffer(5)]],
    device float* working [[buffer(6)]],
    device float* aux [[buffer(7)]],
    constant int* iparams [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {
  int visible_channel = iparams[0];
  int H = iparams[1];
  int W = iparams[2];
  long idx = (long)gid;
  long total = (long)H * W;
  if (idx >= total) return;
  int y = (int)(idx / W);
  float lanes[4];
  for (int c = 0; c < 4; ++c) {
    lanes[c] = lut[rgbi[idx * 4 + c]];
    working[idx * 4 + c] = lanes[c];
  }
  float value;
  if (alpha_is_one[y] != 0) {
    value = alpha_one_replacement[y];
  } else {
    u64 alpha = f64_from_f32(aux_alpha[y]);
    u64 visible = f64_from_f32(lanes[visible_channel]);
    u64 infrared = f64_from_f32(lanes[3]);
    u64 numerator = f64_sub(infrared, f64_mul(alpha, visible));
    u64 denominator = f64_sub(F64C_ONE, alpha);
    value = f32_from_f64(
        f64_sub(f64_div(numerator, denominator), f64_from_f32(aux_offset[y])));
  }
  aux[idx] = value;
}

// ===========================================================================
// stage 2: continuous score + weighted planes (x3a.continuous_score)
// qparams: 0 base_addend, 1 scale, 2 offset, 3 floor (binary64 bits)
// iparams: 0 horizontal_minimum, 1 H, 2 W
// ===========================================================================

kernel void k_score_and_weighted(
    device const float* aux [[buffer(0)]],
    device const float* working [[buffer(1)]],
    device const float* score_base_primary [[buffer(2)]],
    device float* score [[buffer(3)]],
    device float* waux [[buffer(4)]],
    device float* wrgb [[buffer(5)]],
    device atomic_uint* error_flags [[buffer(6)]],
    constant u64* qparams [[buffer(7)]],
    constant int* iparams [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {
  int horizontal_minimum = iparams[0];
  int H = iparams[1];
  int W = iparams[2];
  long idx = (long)gid;
  long total = (long)H * W;
  if (idx >= total) return;
  int y = (int)(idx / W);
  int x = (int)(idx - (long)y * W);
  float value = aux[idx];
  // continuous_score fails closed on a nonfinite auxiliary plane; the
  // comparison-based clamps below would otherwise launder NaN.
  if (!isfinite(value)) {
    atomic_fetch_or_explicit(&error_flags[0], 1u, memory_order_relaxed);
  }
  float sample = value;
  if (horizontal_minimum != 0 && W > 2 && x >= 1 && x <= W - 2) {
    float left = aux[idx - 1];
    float right = aux[idx + 1];
    float m = left < value ? left : value;
    sample = m < right ? m : right;
  }
  u64 primary = f64_from_f32(score_base_primary[y]);
  u64 s64 = f64_add(
      f64_mul(f64_sub(f64_add(primary, qparams[0]), f64_from_f32(sample)),
              qparams[1]),
      qparams[2]);
  s64 = f64_lt(s64, F64C_ONE) ? s64 : F64C_ONE;
  s64 = f64_lt(qparams[3], s64) ? s64 : qparams[3];
  float s = f32_from_f64(s64);
  score[idx] = s;
  waux[idx] = f32_mul(s, value);
  for (int c = 0; c < 3; ++c) {
    wrgb[idx * 3 + c] = f32_mul(s, working[idx * 4 + c]);
  }
}

// ===========================================================================
// stage 3: decision fallback + row eligibility (streaming rules)
// raw decision history replicates every edge.  Comparisons only; exact.
// fparams: 0 threshold; iparams: 0 count_limit, 1 radius, 2 H, 3 W
// ===========================================================================

kernel void k_decision_eligibility(
    device const float* aux [[buffer(0)]],
    device const float* score [[buffer(1)]],
    device const int* row_gate [[buffer(2)]],
    device const u8* floor_enabled [[buffer(3)]],
    device u8* eligible [[buffer(4)]],
    constant float* fparams [[buffer(5)]],
    constant int* iparams [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
  float threshold = fparams[0];
  int count_limit = iparams[0];
  int radius = iparams[1];
  int H = iparams[2];
  int W = iparams[3];
  long idx = (long)gid;
  long total = (long)H * W;
  if (idx >= total) return;
  int y = (int)(idx / W);
  int x = (int)(idx - (long)y * W);
  int lx = x - radius;
  lx = lx < 0 ? 0 : (lx >= W ? W - 1 : lx);
  int rx = x + radius;
  rx = rx < 0 ? 0 : (rx >= W ? W - 1 : rx);
  int vertical_left = 0, vertical_right = 0;
  for (int dy = -4; dy <= 4; ++dy) {
    int yy = y + dy;
    yy = yy < 0 ? 0 : (yy >= H ? H - 1 : yy);
    device const float* row = aux + (long)yy * W;
    vertical_left += row[lx] < threshold ? 1 : 0;
    vertical_right += row[rx] < threshold ? 1 : 0;
  }
  int ay = y - radius < 0 ? 0 : y - radius;
  int by = y + radius >= H ? H - 1 : y + radius;
  device const float* above = aux + (long)ay * W;
  device const float* below = aux + (long)by * W;
  int horizontal_above = 0, horizontal_below = 0;
  for (int dx = -4; dx <= 4; ++dx) {
    int xx = x + dx;
    xx = xx < 0 ? 0 : (xx >= W ? W - 1 : xx);
    horizontal_above += above[xx] < threshold ? 1 : 0;
    horizontal_below += below[xx] < threshold ? 1 : 0;
  }
  bool fallback = (vertical_left > count_limit) |
                  (horizontal_above > count_limit) |
                  (vertical_right > count_limit) |
                  (horizontal_below > count_limit);
  bool ok = !fallback;
  if (row_gate[y] != 0) {
    ok = false;
  } else if (floor_enabled[y] != 0) {
    ok = ok && (score[idx] < 1.0f);
  }
  eligible[idx] = ok ? 1 : 0;
}

// ===========================================================================
// stage 4: per-selected-pixel feature records, candidates, combiner
// (streaming inner loop + reconstruction.combine_recovered_candidate)
// iparams: 0 selected_count, 1 H, 2 W, 3 cross_neighbor_mode,
//          4 coarse_enabled
// fparams: 0 score_floor
// ===========================================================================

kernel void k_features_and_combine(
    device const long* selected [[buffer(0)]],
    device const float* score [[buffer(1)]],
    device const float* waux [[buffer(2)]],
    device const float* wrgb [[buffer(3)]],
    device const float* aux [[buffer(4)]],
    device const float* working [[buffer(5)]],
    device const float* writer_coarse_reference [[buffer(6)]],
    device const u8* floor_enabled [[buffer(7)]],
    device const int* row_gate [[buffer(8)]],
    device const float* coarse_slopes [[buffer(9)]],
    device const u8* band_enabled [[buffer(10)]],
    device const float* band_scales [[buffer(11)]],
    device const float* factors_a [[buffer(12)]],
    device const float* factors_b [[buffer(13)]],
    device const float* configured_strengths [[buffer(14)]],
    device u8* attempted [[buffer(15)]],
    device u64* candidate [[buffer(16)]],
    device float* original [[buffer(17)]],
    constant int* iparams [[buffer(18)]],
    constant float* fparams [[buffer(19)]],
    uint gid [[thread_position_in_grid]]) {
  long selected_count = (long)iparams[0];
  int H = iparams[1];
  int W = iparams[2];
  int cross_neighbor_mode = iparams[3];
  int coarse_enabled = iparams[4];
  float score_floor = fparams[0];
  long i = (long)gid;
  if (i >= selected_count) return;
  long pixel = selected[i];
  int y = (int)(pixel / W);
  int x = (int)(pixel - (long)y * W);

  float source_rgb[3];
  for (int c = 0; c < 3; ++c) source_rgb[c] = working[pixel * 4 + c];
  original[i * 3 + 0] = source_rgb[0];
  original[i * 3 + 1] = source_rgb[1];
  original[i * 3 + 2] = source_rgb[2];
  candidate[i * 3 + 0] = F64_ZERO;
  candidate[i * 3 + 1] = F64_ZERO;
  candidate[i * 3 + 2] = F64_ZERO;
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
  if (cross_neighbor_mode != 0) {
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
  u64 averages[3][3];
  rgb_unscaled_averages(wrgb_patch, averages);
  float cands[3][3];
  for (int scale_index = 0; scale_index < 3; ++scale_index) {
    float denominator = weights[scale_index];
    if (denominator > 0.0f) {
      u64 reciprocal = f64_div(F64C_ONE, f64_from_f32(denominator));
      for (int c = 0; c < 3; ++c)
        cands[scale_index][c] =
            f32_from_f64(f64_mul(averages[scale_index][c], reciprocal));
    } else {
      for (int c = 0; c < 3; ++c)
        cands[scale_index][c] = f32_from_f64(averages[scale_index][c]);
    }
  }

  // reconstruction.feature_band_ranges (binary64 differences, f32 extrema)
  u64 range_min[3], range_max[3];
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
    range_min[t] = f64_from_f32(f32_from_f64(mn));
    range_max[t] = f64_from_f32(f32_from_f64(mx));
  }

  // reconstruction._automatic_strengths
  u64 strengths[3];
  {
    float m1 = weights[1];
    if (configured_strengths[0] == 0.0f) {
      m1 = f32_from_f64(f64_mul(F64C_TWO, f64_from_f32(m1)));
      u64 clamped = f64_from_f32(m1);
      clamped = f64_lt(F64_ZERO, clamped) ? clamped : F64_ZERO;
      clamped = f64_lt(clamped, F64C_ONE) ? clamped : F64C_ONE;
      m1 = f32_from_f64(clamped);
      strengths[0] = f64_from_f32(m1);
    } else {
      strengths[0] = f64_from_f32(configured_strengths[0]);
    }
    float m2 = weights[2];
    if (configured_strengths[1] == 0.0f) {
      if (m2 < 0.0f) m2 = 0.0f;
      strengths[1] = f64_from_f32(m2);
    } else {
      strengths[1] = f64_from_f32(configured_strengths[1]);
    }
    if (configured_strengths[2] == 0.0f) {
      strengths[2] = f64_mul(f64_from_f32(weights[3]),
                             f64_from_f32(weights[3]));
    } else {
      strengths[2] = f64_from_f32(configured_strengths[2]);
    }
  }

  // reconstruction.combine_recovered_candidate
  u64 cand[3];
  for (int c = 0; c < 3; ++c) cand[c] = f64_from_f32(cands[0][c]);
  if (coarse_enabled != 0) {
    u64 coarse_delta = f64_sub(f64_from_f32(writer_coarse_reference[y]),
                               f64_from_f32(features[0][0]));
    for (int c = 0; c < 3; ++c)
      cand[c] = f64_add(cand[c],
                        f64_mul(f64_from_f32(coarse_slopes[c]), coarse_delta));
  }
  for (int band = 0; band < 3; ++band) {
    if (band_enabled[band] == 0) continue;
    u64 band_scale = f64_from_f32(band_scales[band]);
    bool negative_band = f64_lt(range_min[band], F64_ZERO) &&
                         f64_lt(range_max[band], F64_ZERO);
    for (int c = 0; c < 3; ++c) {
      u64 coarse_value, fine_value;
      if (band == 0) {
        coarse_value = f64_from_f32(cands[0][c]);
        fine_value = f64_from_f32(cands[1][c]);
      } else if (band == 1) {
        coarse_value = f64_from_f32(cands[1][c]);
        fine_value = f64_from_f32(cands[2][c]);
      } else {
        coarse_value = f64_from_f32(cands[2][c]);
        fine_value = f64_from_f32(source_rgb[c]);
      }
      u64 difference = f64_mul(band_scale, f64_sub(fine_value, coarse_value));
      u64 factor_a = f64_from_f32(factors_a[band * 3 + c]);
      u64 factor_b = f64_from_f32(factors_b[band * 3 + c]);
      u64 upper, lower;
      if (negative_band) {
        upper = f64_mul(factor_b, range_max[band]);
        lower = f64_mul(factor_a, range_min[band]);
      } else {
        upper = f64_mul(factor_a, range_max[band]);
        lower = f64_mul(factor_b, range_min[band]);
      }
      u64 residual =
          f64_lt(upper, difference)
              ? f64_sub(difference, upper)
              : (f64_lt(difference, lower) ? f64_sub(difference, lower)
                                           : F64_ZERO);
      cand[c] = f64_add(cand[c], f64_mul(strengths[band], residual));
    }
  }
  candidate[i * 3 + 0] = cand[0];
  candidate[i * 3 + 1] = cand[1];
  candidate[i * 3 + 2] = cand[2];
}

// ===========================================================================
// stage 5 (host): the sequential conditional-dither writer chain runs on one
// host CPU core through the compiled fast_cpu.kernels.write_band path, the
// same host writer the CUDA backend uses (cuda_backend/host_writer.py maps
// the per-selected-site arrays onto it).  dither_delta stays here, unused by
// any pipeline kernel, as the validated primitive the Level 1 tests still
// check bit-for-bit against dither.conditional_dither_delta, a standing
// guard against the device and host math ever drifting apart.
// ===========================================================================

inline u64 dither_delta(u64 value, u64 low, u64 high, bool low_lt_high,
                        float scale32, thread u32* state,
                        thread u64* advances) {
  float candidate32 = f32_from_f64(value);
  u64 candidate = f64_from_f32(candidate32);
  if (!low_lt_high) return F64_ZERO;
  if (!(f64_lt(low, candidate) && f64_lt(candidate, high))) return F64_ZERO;
  u64 width = f64_sub(high, low);
  u64 coefficient = f64_div(F64C_FOUR, f64_mul(width, width));
  u64 envelope = f64_mul(
      f64_mul(f64_sub(high, candidate), f64_sub(candidate, low)), coefficient);
  float random_span = f32_from_f64(f64_mul(f64_from_f32(scale32), candidate));
  *state = (125u * (*state) + 1u) & 0x00FFFFFFu;
  *advances += 1ul;
  u64 centered = f64_sub(f64_mul(f64_from_u32(*state + 1u), F64C_NIKON_NORM),
                         F64C_HALF);
  u64 random_value =
      f64_add(F64_ZERO, f64_mul(centered, f64_from_f32(random_span)));
  u64 delta = f64_mul(envelope, random_value);
  u64 changed = f64_add(candidate, delta);
  if (f64_lt(low, changed) && f64_lt(changed, high)) return delta;
  return F64_ZERO;
}

// ===========================================================================
// stage 6: output assembly (output.emit_public_rgb16)
// ===========================================================================

inline u16 emit_one(float value, device const u32* factor_high,
                    device const u32* factor_low) {
  u64 widened = f64_add(f64_from_f32(value), F64C_HALF);
  widened = f64_lt(widened, F64_ZERO)
                ? F64_ZERO
                : (f64_lt(F64C_65535, widened) ? F64C_65535 : widened);
  u32 index = f64_trunc_u32(widened);
  u64 product = (u64)factor_high[index >> 8] * (u64)factor_low[index & 0xFF];
  return (u16)((product >> 20) - 1ul);
}

// iparams: 0 total
kernel void k_copy_visible(
    device const float* working [[buffer(0)]],
    device float* work_output [[buffer(1)]],
    constant int* iparams [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  long total = (long)iparams[0];
  long idx = (long)gid;
  if (idx >= total) return;
  work_output[idx * 3 + 0] = working[idx * 4 + 0];
  work_output[idx * 3 + 1] = working[idx * 4 + 1];
  work_output[idx * 3 + 2] = working[idx * 4 + 2];
}

// iparams: 0 selected_count
kernel void k_scatter_values(
    device const long* selected [[buffer(0)]],
    device const float* values [[buffer(1)]],
    device float* work_output [[buffer(2)]],
    device atomic_uint* error_flags [[buffer(3)]],
    constant int* iparams [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  long selected_count = (long)iparams[0];
  long i = (long)gid;
  if (i >= selected_count) return;
  long pixel = selected[i];
  for (int c = 0; c < 3; ++c) {
    float value = values[i * 3 + c];
    // work_value_indices fails closed on nonfinite work values; emit_one's
    // clamp would otherwise launder them into 0 or 65535.
    if (!isfinite(value)) {
      atomic_fetch_or_explicit(&error_flags[0], 2u, memory_order_relaxed);
    }
    work_output[pixel * 3 + c] = value;
  }
}

// iparams: 0 total3 (H * W * 3)
kernel void k_emit_rgb16(
    device const float* work_output [[buffer(0)]],
    device const u32* factor_high [[buffer(1)]],
    device const u32* factor_low [[buffer(2)]],
    device u16* out [[buffer(3)]],
    constant int* iparams [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  long total = (long)iparams[0];
  long idx = (long)gid;
  if (idx >= total) return;
  out[idx] = emit_one(work_output[idx], factor_high, factor_low);
}

// per-attempted-site changed-pixel accounting against the no-op emit
// counters: [0] attempted, [1] written, [2] changed (uint; the frame pixel
// count is far below 2^32).  iparams: 0 selected_count
kernel void k_site_counters(
    device const u8* attempted [[buffer(0)]],
    device const float* values [[buffer(1)]],
    device const float* original [[buffer(2)]],
    device const u8* written [[buffer(3)]],
    device const u32* factor_high [[buffer(4)]],
    device const u32* factor_low [[buffer(5)]],
    device atomic_uint* counters [[buffer(6)]],
    constant int* iparams [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
  long selected_count = (long)iparams[0];
  long i = (long)gid;
  if (i >= selected_count) return;
  if (attempted[i] == 0) return;
  atomic_fetch_add_explicit(&counters[0], 1u, memory_order_relaxed);
  if (written[i] != 0) {
    atomic_fetch_add_explicit(&counters[1], 1u, memory_order_relaxed);
  }
  bool changed = false;
  for (int c = 0; c < 3; ++c) {
    u16 rendered = emit_one(values[i * 3 + c], factor_high, factor_low);
    u16 noop = emit_one(original[i * 3 + c], factor_high, factor_low);
    changed = changed || (rendered != noop);
  }
  if (changed) {
    atomic_fetch_add_explicit(&counters[2], 1u, memory_order_relaxed);
  }
}

// ===========================================================================
// content-derived producer (producer_parameters) row/epoch primitives
// ===========================================================================

// first failing (row, column) index inside each 8x8 producer cell-block,
// in the row-major scan order of derive_producer_mean_schedule.
// qparams: 0 threshold (binary64 bits)
// iparams: 0 H, 1 W, 2 active_width, 3 block_count, 4 cell_count
kernel void k_producer_failpos(
    device const u16* rgbi [[buffer(0)]],
    device int* failpos [[buffer(1)]],
    constant u64* qparams [[buffer(2)]],
    constant int* iparams [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]) {
  int H = iparams[0];
  int W = iparams[1];
  int block_count = iparams[3];
  int cell_count = iparams[4];
  int cell = (int)gid.x;
  int block = (int)gid.y;
  if (block >= block_count || cell >= cell_count) return;
  int row0 = block * 8;
  int rows = H - row0 < 8 ? H - row0 : 8;
  int col0 = cell * 8;
  u64 thr = qparams[0];
  int fail = rows * 8;  // no failure inside this block
  for (int r = 0; r < rows && fail == rows * 8; ++r) {
    for (int c = 0; c < 8; ++c) {
      u16 raw = rgbi[(((long)(row0 + r) * W) + col0 + c) * 4 + 3];
      if (!f64_lt(thr, f64_from_u32((u32)raw))) {
        fail = r * 8 + c;
        break;
      }
    }
  }
  failpos[(long)block * cell_count + cell] = fail;
}

// per-row binary64 sums in exact column order (one thread per row)
// iparams: 0 H, 1 W, 2 active_width, 3 visible_channel, 4 cell_count
kernel void k_producer_row_sums(
    device const u16* rgbi [[buffer(0)]],
    device const float* lut [[buffer(1)]],
    device const int* failpos [[buffer(2)]],
    device u64* row_visible [[buffer(3)]],
    device u64* row_infrared [[buffer(4)]],
    device u64* row_weight [[buffer(5)]],
    device u32* row_accepted [[buffer(6)]],
    constant int* iparams [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
  int H = iparams[0];
  int W = iparams[1];
  int active_width = iparams[2];
  int visible_channel = iparams[3];
  int cell_count = iparams[4];
  int row = (int)gid;
  if (row >= H) return;
  int block = row / 8;
  int row_in_block = row - block * 8;
  u64 visible_sum = F64_ZERO, infrared_sum = F64_ZERO, weight_sum = F64_ZERO;
  u32 accepted = 0u;
  device const int* block_fail = failpos + (long)block * cell_count;
  for (int column = 0; column < active_width; ++column) {
    int cell = column >> 3;
    int col_in_cell = column & 7;
    int index = row_in_block * 8 + col_in_cell;
    if (index >= block_fail[cell]) continue;
    u16 raw_ir = rgbi[(((long)row * W) + column) * 4 + 3];
    u16 raw_vis = rgbi[(((long)row * W) + column) * 4 + visible_channel];
    u64 weight = f64_from_u32((u32)raw_ir * (u32)raw_ir);
    visible_sum =
        f64_add(visible_sum, f64_mul(weight, f64_from_f32(lut[raw_vis])));
    infrared_sum =
        f64_add(infrared_sum, f64_mul(weight, f64_from_f32(lut[raw_ir])));
    weight_sum = f64_add(weight_sum, weight);
    accepted += 1u;
  }
  row_visible[row] = visible_sum;
  row_infrared[row] = infrared_sum;
  row_weight[row] = weight_sum;
  row_accepted[row] = accepted;
}

// one complete eight-row scale epoch
// (producer_parameters._scale_epoch_additions)
// qparams: 0 threshold (binary64 bits)
// iparams: 0 H, 1 W, 2 active_width, 3 visible_channel, 4 epoch_count
kernel void k_producer_scale_epochs(
    device const u16* rgbi [[buffer(0)]],
    device const float* lut [[buffer(1)]],
    device float* add_denominator [[buffer(2)]],
    device float* add_numerator [[buffer(3)]],
    constant u64* qparams [[buffer(4)]],
    constant int* iparams [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  int W = iparams[1];
  int active_width = iparams[2];
  int visible_channel = iparams[3];
  int epoch_count = iparams[4];
  int epoch = (int)gid;
  if (epoch >= epoch_count) return;
  int row0 = epoch * 8;
  u64 thr = qparams[0];
  u64 denominator = F64_ZERO, numerator = F64_ZERO;
  for (int column = 0; column < active_width; column += 8) {
    bool block_ok = true;
    for (int r = 0; r < 8 && block_ok; ++r)
      for (int c = 0; c < 8; ++c) {
        u16 raw = rgbi[(((long)(row0 + r) * W) + column + c) * 4 + 3];
        if (f64_le(f64_from_u32((u32)raw), thr)) {
          block_ok = false;
          break;
        }
      }
    if (!block_ok) continue;
    // stored float32 4x4 quadrant means over response values, in the exact
    // reference accumulation tree
    float visible_means[4], infrared_means[4];
    for (int lane = 0; lane < 2; ++lane) {
      thread float* means = lane == 0 ? visible_means : infrared_means;
      int channel = lane == 0 ? visible_channel : 3;
      int quadrant = 0;
      for (int rs = 0; rs < 8; rs += 4)
        for (int cs = 0; cs < 8; cs += 4) {
          float v[4][4];
          for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
              v[r][c] = lut[rgbi[(((long)(row0 + rs + r) * W) + column + cs +
                                  c) *
                                     4 +
                                 channel]];
          u64 first = f64_from_f32(v[0][0]);
          first = f64_add(first, f64_from_f32(v[0][1]));
          first = f64_add(first, f64_from_f32(v[0][2]));
          first = f64_add(first, f64_from_f32(v[0][3]));
          first = f64_add(first, f64_from_f32(v[1][0]));
          first = f64_add(first, f64_from_f32(v[1][1]));
          u64 second_tail =
              f64_add(f64_from_f32(v[1][2]), f64_from_f32(v[1][3]));
          first = f64_add(first, second_tail);
          u64 third = f64_add(f64_from_f32(v[2][0]), f64_from_f32(v[2][1]));
          u64 third_tail =
              f64_add(f64_from_f32(v[2][2]), f64_from_f32(v[2][3]));
          third = f64_add(third, third_tail);
          u64 fourth = f64_add(f64_from_f32(v[3][0]), f64_from_f32(v[3][1]));
          u64 fourth_tail =
              f64_add(f64_from_f32(v[3][2]), f64_from_f32(v[3][3]));
          fourth = f64_add(fourth, fourth_tail);
          third = f64_add(third, fourth);
          u64 total = f64_add(first, third);
          means[quadrant++] = f32_from_f64(f64_mul(total, F64C_SIXTEENTH_EXACT));
        }
    }
    // _center_quadrants
    float visible_dev[4], infrared_dev[4];
    for (int lane = 0; lane < 2; ++lane) {
      thread const float* means = lane == 0 ? visible_means : infrared_means;
      thread float* dev = lane == 0 ? visible_dev : infrared_dev;
      u64 total = f64_from_f32(means[0]);
      total = f64_add(total, f64_from_f32(means[1]));
      total = f64_add(total, f64_from_f32(means[2]));
      total = f64_add(total, f64_from_f32(means[3]));
      float mean = f32_from_f64(f64_mul(total, F64C_QUARTER));
      for (int q = 0; q < 4; ++q)
        dev[q] = f32_from_f64(
            f64_sub(f64_from_f32(means[q]), f64_from_f32(mean)));
    }
    // float32 running sum of raw infrared in row-major order
    float raw_sum = 0.0f;
    for (int r = 0; r < 8; ++r)
      for (int c = 0; c < 8; ++c) {
        u16 raw = rgbi[(((long)(row0 + r) * W) + column + c) * 4 + 3];
        raw_sum = f32_from_f64(
            f64_add(f64_from_f32(raw_sum), f64_from_u32((u32)raw)));
      }
    u64 raw_sum_wide = f64_from_f32(raw_sum);
    for (int q = 0; q < 4; ++q) {
      u64 visible_wide = f64_from_f32(visible_dev[q]);
      u64 ratio, weight;
      if (f64_eq(visible_wide, F64_ZERO)) {
        ratio = F64_ZERO;
        weight = F64_ZERO;
      } else {
        ratio = f64_div(f64_from_f32(infrared_dev[q]), visible_wide);
        if (f64_lt(ratio, F64C_PROD_LOW) || f64_lt(F64C_PROD_HIGH, ratio)) {
          weight = F64_ZERO;
        } else if (f64_lt(ratio, F64_ZERO) ||
                   f64_lt(F64C_PROD_FULL_HIGH, ratio)) {
          weight = f64_mul(visible_wide, F64C_PROD_OUTER);
        } else {
          weight = visible_wide;
        }
      }
      u64 term = f64_mul(weight, weight);
      term = f64_mul(term, raw_sum_wide);
      term = f64_mul(term, raw_sum_wide);
      denominator = f64_add(denominator, term);
      numerator = f64_add(numerator, f64_mul(ratio, term));
    }
  }
  add_denominator[epoch] = f32_from_f64(denominator);
  add_numerator[epoch] = f32_from_f64(numerator);
}
"""


def render_kernel_source() -> str:
    """Inject exact host-quantized binary64 bit patterns into the MSL source."""

    defines = "\n".join(
        f"#define {name} {_f64_bits(value)}" for name, value in _CONSTANTS.items()
    )
    return KERNEL_SOURCE.replace("__CONSTANT_DEFINES__", defines)


KERNEL_NAMES = (
    "k_convert_and_auxiliary",
    "k_score_and_weighted",
    "k_decision_eligibility",
    "k_features_and_combine",
    "k_copy_visible",
    "k_scatter_values",
    "k_emit_rgb16",
    "k_site_counters",
    "k_producer_failpos",
    "k_producer_row_sums",
    "k_producer_scale_epochs",
)
