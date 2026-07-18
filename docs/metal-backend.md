# Metal backend

The package ships an optional deterministic Metal backend for the same single
supported profile as the CPU reference: Nikon LS-5000 ED, selector 8, Digital
ICE Normal, internal metric 4000. The exact label rests on receipts, not on
availability or speed: complete output, including every RGB16 byte, the
changed-pixel accounting, the number of public RNG advances, the final RNG
state, the per-stage hidden-startup RNG advances, and all three diagnostics
planes, compared bitwise on both complete validation frames, 26 binding
checks per frame. The receipts are checked in under
[`evidence/`](../evidence/) (`metal-frame-*-parity.json`) and bind this
tree's current source manifest.

The full-frame comparison baseline is the compiled `cpu-fast` backend run in
the same process, chained to the CPU reference by pinned hashes: the Metal
output hash equals the CPU-reference output hash recorded by the checked-in
cuda and cpu-fast receipts for the same fixture bytes, and the output plus
all three diagnostics planes also match the archived full-frame export of
the completed CUDA gate bitwise. The synthetic startup self-test additionally
proves direct byte parity against `process_cpu` on every machine, every
process, before the backend accepts any real frame.

## Install

```sh
python -m pip install -e '.[metal]'
```

The `metal` extra installs `pyobjc-framework-Metal` (the Metal binding) and
numba (the host writer chain, shared with the `cpu-fast` and `cuda`
backends). It only resolves on macOS; elsewhere the backend fails closed
with a specific reason. Kernels compile at first use through the Metal
runtime compiler (about 2.7 s on the validation machine); the startup
self-test doubles as the warmup, and recorded per-frame times exclude those
first-call costs.

## Selecting a backend

```python
from portable_digital_ice import ComputeBackend, process

routed = process(job, backend=ComputeBackend.AUTO)
routed.result.output_rgb16   # uint16 HxWx3
routed.selection.used        # CUDA, METAL, CPU_FAST, or CPU
routed.selection.reason      # why that backend ran
```

- `ComputeBackend.CPU` always runs the exact validated reference.
- `ComputeBackend.METAL` requires a Metal device plus the compiled host
  writer and raises `MetalBackendUnavailable` with a specific reason
  otherwise: pyobjc missing, no device, insufficient working-set headroom,
  numba missing, or a failed parity self-test. It never silently falls back.
- `ComputeBackend.AUTO` tries CUDA first, then Metal, then cpu-fast, then
  the CPU reference. Each candidate must first pass a startup self-test
  that processes a small synthetic acquisition on both that backend and the
  reference and requires byte equality across output, counters, RNG
  accounting, and diagnostics planes. Every fallthrough reason is recorded
  in `selection.reason`. Self-test outcomes are cached per process.

Backend selection never changes the input contract: unsupported jobs raise
the same validation errors before any backend probing or output mutation.
`process_cpu` remains available unchanged for callers that want the
reference directly.

## Exactness design: software binary64

Apple GPUs have no double-precision hardware and the Metal Shading Language
has no `double` type. The reference pipeline, however, is specified in
binary64: it widens float32 operands, evaluates with one rounding per
written operation, and narrows once at each recorded store boundary. A
backend that substituted float32 for those steps would be a different
algorithm.

The Metal kernels therefore carry binary64 values as `ulong` bit patterns
and perform every binary64 operation in software: an integer-arithmetic
IEEE-754 round-to-nearest-even implementation in the SoftFloat style,
covering add, subtract, multiply, divide, conversions in both directions,
comparisons, and truncation, with full subnormal, signed-zero, and
overflow-to-infinity handling. This is a stronger position than a compiler
flag. The CUDA backend needs `--fmad=false` to stop the compiler
contracting multiply/add pairs; the Metal binary64 schedule is integer
arithmetic end to end, which no compiler mode can contract, reassociate,
or flush. Division is restoring long division, correct to the final bit by
construction rather than by a hardware convention.

The rest of the design mirrors the CUDA backend:

- the float32 multiplies at the reference's store boundaries are composed
  from the same softfloat path: a float32 product is exact in binary64
  (24 + 24 significand bits), so one narrowing reproduces the correctly
  rounded float32 multiply, subnormals included. Native GPU float behavior
  is left in charge of nothing but comparison predicates, which do not
  round;
- host-quantized constants (for example `float32(1/69)` widened to
  binary64) are injected into the kernel text as exact bit patterns. MSL
  has no binary64 literals, so every binary64 constant enters this way and
  the kernel text cannot drift from the reference quantization;
- the logarithmic response LUT and the output factor tables are generated
  on the CPU (hash-checked, fail-closed) and uploaded, never recomputed
  with GPU transcendentals;
- the 24-bit LCG and conditional-dither writer run in strict row-major
  site order on one host CPU core, through the same compiled path
  validated on the `cpu-fast` backend (`cuda_backend/host_writer.py` maps
  the per-site arrays onto it), because the number of draws a site
  consumes depends on the drawn values themselves. A softfloat
  `dither_delta` stays in the kernel source, unreachable from any pipeline
  kernel, as a standing drift guard checked bit-for-bit by the Level 1
  tests;
- the content-derived producer keeps its strictly ordered cross-row
  accumulation on the host; the GPU computes only the row-internal and
  epoch-internal sums, in the exact reference order, returning binary64
  bit patterns;
- the six-stage hidden startup replay stays on the CPU reference code and
  seeds the RNG chain;
- the library is compiled with fast math disabled (and `MTLMathModeSafe`
  where the OS provides it), governing the comparison predicates and raw
  loads that remain native.

The softfloat layer is proven bit-for-bit against numpy in
`tests/test_metal_softfloat.py`, 400,000 samples per operation, including
subnormals, exact cancellation, ties, and overflow, and the per-machine
startup self-test re-proves whole-pipeline byte parity on the local device
and driver in every process.

Counters (`attempted_pixels`, `written_pixels`, `changed_pixels`) are
integer reductions and are deterministic; no floating-point atomics exist
anywhere.

## Performance

Measured on an Apple M4 (10-core GPU, 16 GB unified memory, macOS 26.5,
Python 3.13.5, numpy 2.4.6, numba 0.66.0, pyobjc 12.2.1), complete
5,782 x 3,946 native frame, warm process, diagnostics export enabled, while
other sessions loaded the machine (1-minute load average about 11; byte
equality is load-independent, wall time is not):

| Metric | Frame 1 | Frame 2 |
|---|---:|---:|
| Full frame wall time (2 runs each) | 8.2 - 9.9 s | 8.6 - 8.7 s |
| cpu-fast on the same frames, same session | 14.0 s | 14.8 s |
| Reference CPU wall time, same frame | 3,545.7 s | 4,183.4 s |
| Speedup vs reference | about 400x | about 480x |
| Repeated-run output hash | identical (deterministic) | identical (deterministic) |

Stage breakdown of a representative frame-1 run: feature records,
candidates, and combiner on device 1.9 s; host writer chain 1.5 s; hidden
startup replay (reference Python) 2.6 - 3.9 s; diagnostics export 1.3 s;
analysis planes on device 0.2 - 0.4 s; producer 0.3 s; everything else
under 0.3 s combined. The softfloat features kernel carries roughly two
thousand software binary64 operations per selected site and still clears
the frame's roughly seven million sites in about two seconds; the
next-largest costs are host-side and shared with the other backends.
One warm frame run with diagnostics measured a peak memory footprint of
about 4.3 GB (peak resident set 2.8 GB); unified memory makes the device
planes and the host writer views the same bytes, so nothing crosses a bus
twice, and a free-headroom check against the device's recommended working
set fails closed before allocation instead of degrading.

Progress callbacks fire at coarser boundaries than the CPU path (per stage
rather than per row) with exact totals at completion; cancellation is
honored between stages and can never corrupt partial output because the
output buffer is written only once, after all device work completes.

## Validation status

- Level 0 (softfloat vs numpy, bitwise): passed on the validation machine.
- Level 1 (device primitives and plane kernels vs the reference, bitwise):
  passed.
- Level 2 (ten adversarial tiles, metric-500 shape, writer-gate branches;
  complete jobs vs `process_cpu`, bitwise, with repeat-run determinism):
  passed.
- Backend contract (fail-closed legs, self-test caching, tampered-result
  detection, AUTO reasons): passed.
- Complete-frame gates (both validation frames vs `cpu-fast` in-process,
  bound to the pinned CPU-reference hashes and the archived CUDA-gate
  export): 26/26 checks per frame, receipts under `evidence/`.

The synthetic suites run wherever a Metal device is present and skip with a
reason elsewhere; continuous integration does not currently exercise a
Metal device, so the receipts above are the arm64 validation host's.

## Files

| Path | Contents |
|---|---|
| `src/portable_digital_ice/backend.py` | `ComputeBackend`, self-tests, `process()` |
| `src/portable_digital_ice/metal_backend/kernels.py` | MSL sources incl. software binary64 |
| `src/portable_digital_ice/metal_backend/engine.py` | streaming-replay mirror |
| `src/portable_digital_ice/metal_backend/producer.py` | producer schedule |
| `evidence/metal-frame-*.json` | full-frame parity receipts |
