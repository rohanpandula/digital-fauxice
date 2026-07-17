# CUDA backend

The package ships an optional deterministic CUDA backend for the same single
supported profile as the CPU reference: Nikon LS-5000 ED, selector 8, Digital
ICE Normal, internal metric 4000. The exact label rests on receipts, not on
availability or speed: complete output, including every RGB16 byte, the
changed-pixel accounting, the number of public RNG advances, and the final RNG state,
compared sample by sample against this package's CPU reference on both
complete validation frames, plus a 300-row crop receipt with
cross-architecture agreement. All receipts are checked in under
[`evidence/`](../evidence/) and bind the same source manifest of this tree.

## Install

```sh
python -m pip install -e '.[cuda]'
```

The `cuda` extra installs CuPy built for CUDA 12.x (`cupy-cuda12x`). The host
needs an NVIDIA driver new enough for the CUDA 12 runtime; no toolkit install
is required because kernels compile at first use through NVRTC. The first
CUDA run in a process pays a one-time compile cost (about 1.5 s on the
validation machine).

## Selecting a backend

```python
from portable_digital_ice import ComputeBackend, process

routed = process(job, backend=ComputeBackend.AUTO)
routed.result.output_rgb16   # uint16 HxWx3
routed.selection.used        # ComputeBackend.CUDA or ComputeBackend.CPU
routed.selection.reason      # why that backend ran
```

- `ComputeBackend.CPU` always runs the exact validated reference.
- `ComputeBackend.CUDA` requires a working device and raises
  `CudaBackendUnavailable` with a specific reason otherwise: missing cupy, no
  device, insufficient free VRAM, or a failed parity self-test. It never
  silently falls back.
- `ComputeBackend.AUTO` first runs a startup self-test that processes a small
  synthetic acquisition on both backends and requires byte equality plus
  identical RNG accounting. Only a passing self-test selects CUDA; any failure
  falls back to CPU and reports the reason in `selection.reason`. The
  self-test outcome is cached per process.

Backend selection never changes the input contract: unsupported jobs raise the
same validation errors before any backend probing or output mutation.
`process_cpu` remains available unchanged for callers that want the reference
directly.

## Exactness design

The kernels are a line-by-line translation of the reference modules with the
same widening, rounding, and store schedule:

- compiled with `--fmad=false` so multiply/add pairs never contract into FMA;
- binary64 arithmetic with one rounding per written operation, and float32
  narrowing only at the reference's recorded store boundaries;
- the logarithmic response LUT and the output factor tables are generated on
  the CPU (hash-checked, fail-closed) and uploaded, never recomputed with GPU
  transcendentals;
- the 24-bit LCG and conditional-dither writer run as one sequential device
  thread in strict row-major site order, because the number of draws a site
  consumes depends on the drawn values themselves;
- the content-derived producer keeps its strictly ordered cross-row
  accumulation on the host; the GPU computes only the row-internal and
  epoch-internal sums, in the exact reference order;
- the six-stage hidden startup replay stays on the CPU reference code and
  seeds the device RNG chain.

Counters (`attempted_pixels`, `written_pixels`, `changed_pixels`) are integer
reductions and are deterministic; no floating-point atomics exist anywhere.

## Performance

Measured on an NVIDIA RTX A4000 (16 GB, CC 8.6, driver 610.43.02, CUDA 12
runtime, CuPy 14.1.1), complete 5,782 x 3,946 native frame, warm process:

| Metric | Value |
|---|---:|
| Full frame wall time (3 warm runs) | 21.2 - 21.5 s |
| Reference CPU wall time, same frame | 4,313.8 s |
| Speedup | about 200x |
| Peak device memory pool | 2.08 GB |
| Peak host RSS | 1.37 GB |
| Peak GPU power / utilization | 66 W / 100% |
| Repeated-run output hash | identical (deterministic) |

About 87% of the remaining wall time is the intentionally sequential
conditional-dither writer chain. The full frame fits device memory with a
wide margin at the supported geometry, so no tiling is used; a free-VRAM check
fails closed before allocation instead of degrading precision or seams.

Progress callbacks fire at coarser boundaries than the CPU path (per stage
rather than per row) with exact totals at completion; cancellation is honored
between stages and can never corrupt partial output because the output buffer
is written only once, after all device work completes.

## Files

| Path | Contents |
|---|---|
| `src/portable_digital_ice/backend.py` | `ComputeBackend`, self-test, `process()` |
| `src/portable_digital_ice/cuda_backend/kernels.py` | CUDA C sources |
| `src/portable_digital_ice/cuda_backend/engine.py` | streaming-replay mirror |
| `src/portable_digital_ice/cuda_backend/producer.py` | producer schedule |
| `src/portable_digital_ice/cuda_backend/rowparams.py` | per-row calibration |
| `docs/cuda-decision-record.md` | stack decision and probes |
| `evidence/cuda-*.json` | parity and performance receipts |
