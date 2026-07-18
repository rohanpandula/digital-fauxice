"""Level 0: the software binary64 layer, compared as raw bit patterns.

Apple GPUs have no binary64 hardware, so every binary64 operation in the
Metal kernels runs through the integer softfloat implementation in
``metal_backend/kernels.py``.  These tests are the foundation the whole
backend stands on: each operation must reproduce numpy's IEEE-754
round-to-nearest-even result bit for bit across the full finite spectrum,
including subnormals, signed zeros, exact cancellation, ties, and overflow
to infinity.  Comparisons use integer bit views, never decimal tolerances.

Every test skips with a clear reason when Metal is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("Metal", reason="Metal backend requires pyobjc-framework-Metal")

from portable_digital_ice.metal_backend.engine import (  # noqa: E402
    MetalBackendUnavailable,
    metal_device_summary,
)
from portable_digital_ice.metal_backend.kernels import render_kernel_source  # noqa: E402

from metal_harness import MetalHarness  # noqa: E402


def _require_device() -> None:
    try:
        metal_device_summary()
    except MetalBackendUnavailable as error:
        pytest.skip(f"Metal device unavailable: {error}")


TEST_WRAPPERS = r"""
kernel void t_f64_ops(
    device const u64* a [[buffer(0)]],
    device const u64* b [[buffer(1)]],
    device u64* out_add [[buffer(2)]],
    device u64* out_sub [[buffer(3)]],
    device u64* out_mul [[buffer(4)]],
    device u64* out_div [[buffer(5)]],
    uint idx [[thread_position_in_grid]]) {
  u64 x = a[idx];
  u64 y = b[idx];
  out_add[idx] = f64_add(x, y);
  out_sub[idx] = f64_sub(x, y);
  out_mul[idx] = f64_mul(x, y);
  out_div[idx] = f64_div(x, y);
}

kernel void t_f64_convert(
    device const u64* a [[buffer(0)]],
    device const u32* f [[buffer(1)]],
    device u32* out_narrow [[buffer(2)]],
    device u64* out_widen [[buffer(3)]],
    device u32* out_trunc [[buffer(4)]],
    device u64* out_from_u32 [[buffer(5)]],
    uint idx [[thread_position_in_grid]]) {
  out_narrow[idx] = as_type<u32>(f32_from_f64(a[idx]));
  out_widen[idx] = f64_from_f32(as_type<float>(f[idx]));
  out_trunc[idx] = f64_trunc_u32(a[idx]);
  out_from_u32[idx] = f64_from_u32(f[idx]);
}

kernel void t_f64_compare(
    device const u64* a [[buffer(0)]],
    device const u64* b [[buffer(1)]],
    device u32* out [[buffer(2)]],
    uint idx [[thread_position_in_grid]]) {
  u64 x = a[idx];
  u64 y = b[idx];
  u32 bits = 0u;
  if (f64_lt(x, y)) bits |= 1u;
  if (f64_le(x, y)) bits |= 2u;
  if (f64_eq(x, y)) bits |= 4u;
  out[idx] = bits;
}

kernel void t_f32_mul(
    device const u32* a [[buffer(0)]],
    device const u32* b [[buffer(1)]],
    device u32* out [[buffer(2)]],
    uint idx [[thread_position_in_grid]]) {
  out[idx] = as_type<u32>(
      f32_mul(as_type<float>(a[idx]), as_type<float>(b[idx])));
}
"""


@pytest.fixture(scope="module")
def harness() -> MetalHarness:
    _require_device()
    return MetalHarness(render_kernel_source() + TEST_WRAPPERS)


def _random_finite_f64(rng: np.random.Generator, n: int) -> np.ndarray:
    """Finite bit patterns weighted toward pipeline magnitudes, with tails."""

    sign = rng.integers(0, 2, n, dtype=np.uint64) << np.uint64(63)
    kind = rng.integers(0, 100, n)
    exp = np.where(
        kind < 80,
        rng.integers(896, 1151, n),  # everyday pipeline magnitudes
        rng.integers(1, 2046, n),  # full normal range
    ).astype(np.uint64)
    exp = np.where(kind >= 97, np.uint64(0), exp)  # subnormals and zeros
    frac = rng.integers(0, 1 << 52, n, dtype=np.uint64)
    return sign | (exp << np.uint64(52)) | frac


def _operand_pairs(n: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260718)
    a = _random_finite_f64(rng, n).view(np.float64).copy()
    b = _random_finite_f64(rng, n).view(np.float64).copy()
    # adversarial structure: near-equal operands (catastrophic cancellation),
    # exact equality, exact negation, powers of two, zeros, infinities,
    # extreme subnormals and the smallest normal
    a[:2000] = b[:2000] * (
        1.0 + np.ldexp(rng.integers(-4, 5, 2000).astype(np.float64), -52)
    )
    a[2000:2500] = b[2000:2500]
    b[2500:3000] = -a[2500:3000]
    a[3000:3500] = np.ldexp(1.0, rng.integers(-1022, 1023, 500))
    b[3500:4000] = 0.0
    a[4000:4100] = np.inf
    b[4100:4200] = -np.inf
    a[4200:4300] = 5e-324
    b[4300:4400] = 2.2250738585072014e-308
    a[4400:4500] = -0.0
    return a, b


def test_f64_arithmetic_bitexact(harness: MetalHarness) -> None:
    n = 400_000
    a, b = _operand_pairs(n)
    buf_a = harness.buffer_from(a.view(np.uint64))
    buf_b = harness.buffer_from(b.view(np.uint64))
    outs = [harness.buffer_out(np.uint64, n) for _ in range(4)]
    harness.run("t_f64_ops", [buf_a, buf_b, *outs], n)
    with np.errstate(all="ignore"):
        expected = [
            (a + b).view(np.uint64),
            (a - b).view(np.uint64),
            (a * b).view(np.uint64),
            (a / b).view(np.uint64),
        ]
    for name, buf, want in zip(("add", "sub", "mul", "div"), outs, expected):
        got = harness.read(buf, np.uint64, n)
        # NaN encodings compare as a class: the softfloat path canonicalizes
        # payloads, and no NaN is reachable through the validated pipeline.
        both_nan = np.isnan(got.view(np.float64)) & np.isnan(want.view(np.float64))
        mismatches = int(np.count_nonzero((got != want) & ~both_nan))
        assert mismatches == 0, f"{name}: {mismatches} bit mismatches"


def test_f64_conversions_bitexact(harness: MetalHarness) -> None:
    n = 400_000
    rng = np.random.default_rng(20260719)
    f32_bits = rng.integers(0, 1 << 32, n, dtype=np.uint32)
    conv = _random_finite_f64(rng, n).view(np.float64).copy()
    # trunc domain: the pipeline truncates only after clamping to [0, 65535]
    trunc_domain = np.concatenate(
        [
            rng.uniform(0.0, 65535.0, n - 4000),
            rng.uniform(0.0, 1.0, 2000),
            np.array([0.0, 0.5, 0.9999999999999999, 1.0]),
            rng.uniform(65534.0, 65535.0, 1996),
        ]
    )
    conv[: trunc_domain.size] = trunc_domain
    buf_a = harness.buffer_from(conv.view(np.uint64))
    buf_f = harness.buffer_from(f32_bits)
    out_narrow = harness.buffer_out(np.uint32, n)
    out_widen = harness.buffer_out(np.uint64, n)
    out_trunc = harness.buffer_out(np.uint32, n)
    out_from = harness.buffer_out(np.uint64, n)
    harness.run(
        "t_f64_convert",
        [buf_a, buf_f, out_narrow, out_widen, out_trunc, out_from],
        n,
    )

    got_narrow = harness.read(out_narrow, np.uint32, n)
    with np.errstate(all="ignore"):
        want_narrow = conv.astype(np.float32).view(np.uint32)
    valid = ~np.isnan(conv)
    assert int(np.count_nonzero((got_narrow != want_narrow) & valid)) == 0

    got_widen = harness.read(out_widen, np.uint64, n)
    f32_vals = f32_bits.view(np.float32)
    with np.errstate(all="ignore"):
        want_widen = f32_vals.astype(np.float64).view(np.uint64)
    valid = ~np.isnan(f32_vals)
    assert int(np.count_nonzero((got_widen != want_widen) & valid)) == 0

    got_trunc = harness.read(out_trunc, np.uint32, n)[: trunc_domain.size]
    want_trunc = np.trunc(trunc_domain).astype(np.uint32)
    assert np.array_equal(got_trunc, want_trunc)

    got_from = harness.read(out_from, np.uint64, n)
    want_from = f32_bits.astype(np.float64).view(np.uint64)
    assert np.array_equal(got_from, want_from)


def test_f64_comparisons_match_numpy(harness: MetalHarness) -> None:
    n = 400_000
    a, b = _operand_pairs(n)
    buf_a = harness.buffer_from(a.view(np.uint64))
    buf_b = harness.buffer_from(b.view(np.uint64))
    out = harness.buffer_out(np.uint32, n)
    harness.run("t_f64_compare", [buf_a, buf_b, out], n)
    got = harness.read(out, np.uint32, n)
    with np.errstate(all="ignore"):
        want = (
            (a < b).astype(np.uint32)
            | ((a <= b).astype(np.uint32) << 1)
            | ((a == b).astype(np.uint32) << 2)
        )
    assert np.array_equal(got, want)


def test_f32_multiply_composed_bitexact(harness: MetalHarness) -> None:
    """The composed float32 multiply must equal numpy's, subnormals included."""

    n = 400_000
    rng = np.random.default_rng(20260720)
    fa = rng.integers(0, 1 << 32, n, dtype=np.uint32)
    fb = rng.integers(0, 1 << 32, n, dtype=np.uint32)
    # products that land in the float32 subnormal range
    fa[:4000] = rng.integers(0, 1 << 23, 4000, dtype=np.uint32) | np.uint32(
        0x0B000000
    )
    fb[:4000] = rng.integers(0, 1 << 23, 4000, dtype=np.uint32) | np.uint32(
        0x0B800000
    )
    buf_a = harness.buffer_from(fa)
    buf_b = harness.buffer_from(fb)
    out = harness.buffer_out(np.uint32, n)
    harness.run("t_f32_mul", [buf_a, buf_b, out], n)
    got = harness.read(out, np.uint32, n)
    va = fa.view(np.float32)
    vb = fb.view(np.float32)
    with np.errstate(all="ignore"):
        product = va * vb
    want = product.view(np.uint32)
    valid = ~(np.isnan(va) | np.isnan(vb) | np.isnan(product))
    assert int(np.count_nonzero((got != want) & valid)) == 0
