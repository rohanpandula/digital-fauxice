# Decision record: CUDA implementation stack and hybrid boundary

Date: 2026-07-16. Status: accepted, validated by the full parity ladder.

## Requirement

A CUDA backend may only be labeled exact if its complete output matches the
CPU reference byte for byte. That demands explicit control over FMA
contraction, division and square-root precision, denormal handling, rounding
of float64 -> float32 stores, and reduction order, plus a packaging story
small enough for an optional dependency.

## Options considered

| Option | Verdict | Notes |
|---|---|---|
| CuPy `RawModule` (NVRTC) | **Chosen** | One pip wheel (`cupy-cuda12x`); kernels are plain CUDA C compiled at runtime with explicit options; no host toolchain or build step; Python 3.11+ supported. |
| C++/CUDA extension (CMake + pybind11) | Viable fallback | Same numeric control via NVCC, but adds a compiled wheel matrix and a build system to a package that otherwise installs from source trivially. Revisit if NVRTC control ever proves insufficient. |
| `cuda-python` + NVRTC directly | Workable, more code | Same compiler, lower-level launch plumbing than CuPy without adding numeric control. |
| Numba CUDA | Rejected | Generated PTX and math rewriting are not specified to the operation level; proving FMA/rounding behavior per release is more work than writing CUDA C. |
| PyTorch | Rejected | Heavy dependency; kernel fusion and reduction order are explicitly not contract-stable. |

## Verified compiler and arithmetic controls

Probes run inside the validation container on the RTX A4000 before any
porting (and re-run by `tests/test_cuda_level1_primitives.py`):

- `--fmad=false` compiles `a * b + c` to separate multiply and add: the probe
  input `(1 + 2^-29)(1 - 2^-29) - 1` returns `0x0.0p+0`, while `--fmad=true`
  returns `-0x1p-58`. The backend compiles everything with `--fmad=false`.
- float64 multiply-then-add, float64 division, and float64 -> float32 casts
  matched NumPy bit-for-bit on 100,000 random samples each.
- The 24-bit LCG recurrence and Nikon's biased normalization constant
  (`0x1.fffffep-25`) reproduce the reference draw sequence exactly.
- No `--use_fast_math`, no `__fdividef`, no approximate intrinsics anywhere;
  float32 work never passes through a flush-to-zero path (NVRTC default
  `--ftz=false`).
- Host-quantized constants (for example `float32(1/69)` widened to binary64)
  are injected into the kernel text as exact hexadecimal literals so compiler
  literal parsing cannot drift from the reference quantization.

## Hybrid boundary (chosen by measurement)

The Phase 1 profile of the CPU reference on a 300-row native-width crop
attributed essentially all streaming time to per-selected-pixel scalar math:
feature records 74%, multiscale candidates and combiner 17%, dither/writer
5%, and every whole-row NumPy stage under 1%. The boundary follows:

- **GPU**: response-LUT indexing, auxiliary/score/weighted planes, decision
  neighborhoods, feature records, candidates, combiner, output conversion,
  changed-pixel accounting, the producer's row-internal and epoch-internal
  sums, and the sequential writer chain (one device thread owning the LCG).
- **CPU**: prepass reduction, producer cross-row accumulation (rounds after
  every add), stage-parameter resolution, the hidden six-stage startup
  replay, output hashing, receipt assembly.

The writer chain is sequential by necessity: whether a channel consumes one
or two draws depends on the first drawn value, so any parallel schedule would
have to speculate on the 24-bit RNG state. A single device thread executes
the exact recovered order at about 18.5 s per native frame; keeping it on the
GPU avoids a mid-pipeline device-host round trip of per-site records.

Measured consequence (A4000, complete native frame, warm): total 21.2 s, of
which writer chain 18.5 s, feature/candidate kernel 0.18 s, producer 0.2 s,
startup replay (CPU Python) 1.7 s, all remaining kernels and transfers under
0.4 s combined.

## Ranked next optimizations (measured, not yet implemented)

1. ~~Writer chain (18.5 s, 87%)~~ -- **landed 2026-07-17**, see below.
2. Startup replay (1.7 s): same sequential math, currently reference Python;
   the same compiled-loop approach applies.
3. Overlap producer/upload with prepass (about 0.3 s combined).

Any change to these stages re-enters the ladder at Level 1 and must repass
both complete-frame gates before the exact label applies again.

## Update 2026-07-17: writer chain moved to the host CPU

Ranked optimization 1 landed: `k_writer_chain` (the device kernel and its
launch) is deleted, and the same sequential draw/redraw/floor schedule now
runs on one host CPU core through the compiled `fast_cpu.kernels.write_band`
path (`cuda_backend/host_writer.py` maps the GPU pipeline's per-selected-site
attempted/candidate arrays onto it; it recomputes no reconstruction math).
Validated byte-exact on both complete frames, 12/12 binding gates each,
twice per frame for determinism, on the same A4000 lane.

Measured consequence (A4000, complete native frame, warm):

| | Before | After |
|---|---:|---:|
| Writer chain | 18.5 s | 2.7 - 2.9 s |
| Total wall time | 21.2 - 21.5 s | 5.3 - 5.8 s |

About 6.7x on the writer-chain stage itself, about 3.7-4.0x on the whole
frame. Device-to-host transfer of the working plane and per-site arrays
(about 1.1-1.2 s) is now the largest single piece of the writer-chain stage,
ahead of the host computation itself (about 1.5-1.7 s); the upload of
results back to the device is negligible (about 0.01 s). Peak device memory
pool dropped slightly (2.08 GB -> 1.86 GB, a few small device buffers
removed); peak host RSS rose (1.37 GB -> about 2.75 GB) because the host
writer now materializes the full working plane and per-site candidate
arrays once per frame. Ranked optimization 2 (startup replay) remains open.

## Independent translation review

An adversarial equation-by-equation review of the kernel translation against
the reference modules confirmed the mainstream paths and every strict
comparison, literal quantization, boundary loader, index-arithmetic edge, and
stage-substitution rule (nine categories verified, no divergence). It found
one latent asymmetry in unreachable territory: NumPy's
`minimum`/`maximum`/`min`/`max` propagate NaN, while C ternary clamps launder
it to a bound. The reference's observable behavior on a nonfinite plane is to
fail closed at its validation boundaries, so the kernels now detect nonfinite
auxiliary and writer values with a device-side error flag and the engine
raises the matching `ValueError` instead of laundering. This is covered by
`test_nonfinite_auxiliary_fails_closed_like_cpu`. The condition remains
unreachable through the public API (uint16 inputs, hash-pinned finite LUT,
finiteness-validated calibration scalars).

## Suitability for a later NegPy backend

The backend is a library API (`ComputeBackend` + `process()`) with no UI
coupling, fail-closed availability, and a per-process parity self-test, so an
application can expose an "auto" toggle without inheriting exactness risk.
`cupy-cuda12x` remains a single optional wheel on the application side.
