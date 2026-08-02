# Compiled CPU backend (cpu-fast)

The package ships an optional compiled CPU backend for the same single
supported profile as the CPU reference: Nikon LS-5000 ED, selector 8, Digital
ICE Normal, internal metric 4000. The exact label rests on receipts, not on
availability or speed: complete output, including every RGB16 byte, the
changed-pixel accounting, the number of public RNG advances, the final RNG
state, the per-stage hidden-startup RNG advances, and all three diagnostics
planes, compared sample by sample against this package's CPU reference on
both complete validation frames. The receipts are checked in under
[`evidence/`](../evidence/) (`cpu-fast-frame-*-parity.json`) and bind the
source manifest of the tree they were minted on. The pipelined-writer tree
reproduced both frames' receipt output hashes and counters exactly, but its
own receipt mint is still pending, so the checked-in receipts do not yet
name this tree's manifest.

## Install

```sh
python -m pip install -e '.[fast]'
```

The `fast` extra installs numba, and accepts 0.65 or 0.66. The kernels use
only `@njit`, `prange`, and plain numpy calls, so both versions compile the
same code; the startup self-test proves byte parity against the reference
backend on whichever one is installed. The floor sits at 0.65 so that an
application already pinned there can use this backend without moving its own
pin. numba currently caps numpy below 2.5, so the effective combined
requirement with this package is numpy 2.4.x; an incompatible pre-installed
numpy surfaces as `CpuFastUnavailable` carrying the underlying import error
in its message. The first cpu-fast run in a
process pays a one-time JIT compile or compile-cache load; the kernels are
cached on disk (`cache=True`), and the startup self-test doubles as the
warmup. Recorded per-frame times exclude that first-call cost.

## Selecting a backend

```python
from portable_digital_ice import ComputeBackend, process

routed = process(job, backend=ComputeBackend.AUTO)
routed.result.output_rgb16   # uint16 HxWx3
routed.selection.used        # CUDA, CPU_FAST, or CPU
routed.selection.reason      # why that backend ran
```

- `ComputeBackend.CPU` always runs the exact validated reference.
- `ComputeBackend.CPU_FAST` requires numba and raises `CpuFastUnavailable`
  with a specific reason otherwise: numba missing or broken, or a failed
  parity self-test. It never silently falls back.
- `ComputeBackend.AUTO` tries CUDA first (unchanged), then cpu-fast, then
  the CPU reference. Each candidate must first pass a startup self-test that
  processes a small synthetic acquisition on both that backend and the
  reference and requires byte equality across output, counters, RNG
  accounting, and diagnostics planes. Every fallthrough reason is recorded
  in `selection.reason`, including the numba version when cpu-fast is used.
  Self-test outcomes are cached per process.

Backend selection never changes the input contract: unsupported jobs raise
the same validation errors before any backend probing or output mutation.
`process_cpu` remains available unchanged for callers that want the
reference directly.

## Exactness design

The kernels are a line-by-line translation of the reference modules with the
same widening, rounding, and store schedule:

- `@njit` with `nogil=True` and nothing else: no fastmath, no contraction
  of multiply/add pairs into FMA (verified by a permanent contraction
  canary in the test suite), no reassociated reductions -- releasing the
  GIL changes thread scheduling only, never code generation;
- binary64 arithmetic with one rounding per written operation, and float32
  narrowing only at the reference's recorded store boundaries;
- the logarithmic response LUT and the output factor tables come from the
  reference's own hash-checked, fail-closed generators;
- the 24-bit LCG and conditional-dither writer run strictly serially in
  row-major site order, because the number of draws a site consumes depends
  on the drawn values themselves;
- the analysis phase (eligibility, feature records, candidates, combiner)
  parallelizes over rows with disjoint writes and zero reductions, so
  results are byte-identical for every thread count: outputs, digests, and
  counters were verified equal under `NUMBA_NUM_THREADS=1`, `3`, and the
  machine default, and across repeated runs;
- the next band's analysis kernel runs on one worker thread while the main
  thread writes, emits, and digests the current band. The overlap cannot
  change a byte: the writer never mutates the planes the analysis reads,
  analysis outputs are double-buffered per band and consumed only behind
  the worker's completion barrier, and the LCG state still threads through
  `write_band` calls in strict band order on one thread. Both full
  validation frames reproduced their receipt output hashes, counters, and
  RNG accounting exactly under the overlapped schedule, again at every
  swept thread count;
- the strictly ordered producer-schedule accumulation and the six-stage
  hidden startup replay are compiled ports of the same reference order,
  each covered by dedicated byte-parity tests.

Counters are integer accumulations in deterministic order; no floating-point
reductions exist anywhere in the compiled path.

## Performance

Measured on the arm64 validation host (Apple M4, 4 P-cores + 6 E-cores,
16 GB, macOS; Python 3.13.5, numba 0.66.0, llvmlite 0.48.0, numpy 2.4.6),
complete 5,782 x 3,946 native frame, warm process:

| Metric | Value |
|---|---:|
| Full frame wall time (default threads) | 9.2 - 9.5 s |
| Full frame wall time (single thread) | 21.5 - 23.0 s |
| Reference CPU wall time, same frames | 3,545.7 / 4,183.4 s |
| Speedup vs reference (default threads) | about 400x |
| CUDA backend, same frames, for context | 6.3 / 5.5 s |
| Repeated-run output hash | identical (deterministic) |

The CUDA context row quotes the checked-in CUDA receipts minted after the
writer chain moved to a host CPU core (`cuda-frame-*-parity.json`,
`cuda_runtime.elapsed_seconds` 6.314 and 5.462).

The wall-time rows above predate the analysis/writer overlap and measure
the sequential schedule on a quiet host. The overlap hides the serial
writer and emit behind the parallel analysis; the stage split measured on
validation frame 1 under heavy host load (1-minute load average about 14
on 10 cores) was 8.9 s of analysis on the worker thread against 1.2 s of
writing and 1.7 s of emit on the main thread, so the expected quiet-host
saving is roughly the writer-plus-emit time per frame. A quiet-host
wall-time re-measurement has not been captured yet, and no smaller number
is claimed until it is.

The compiled path materializes whole-image analysis planes instead of the
reference's eleven-row streaming window, so peak host memory scales with
height x width -- about 2 GB at the supported full-frame geometry -- while
the reference stays memory-bounded by image width alone. Progress callbacks
still fire per row with exact cumulative
totals; cooperative cancellation is honored per row while the analysis
planes are prepared and per band (128 rows) during writing, and can never
corrupt partial output because the caller's buffer is written only once,
after all work completes.

The parity evidence is from the arm64 host above. The response-LUT and
factor-table hash checks fail closed on any platform whose libm produces
different tables, so an unverified platform refuses to run rather than
producing near-correct output.

## Files

| Path | Contents |
|---|---|
| `src/portable_digital_ice/backend.py` | `ComputeBackend`, self-tests, `process()` |
| `src/portable_digital_ice/fast_cpu/kernels.py` | compiled per-pixel/per-row kernels |
| `src/portable_digital_ice/fast_cpu/engine.py` | streaming-replay mirror |
| `src/portable_digital_ice/fast_cpu/producer.py` | compiled producer schedule |
| `evidence/cpu-fast-frame-*.json` | full-frame parity receipts |
