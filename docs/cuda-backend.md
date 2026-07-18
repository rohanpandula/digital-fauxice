# CUDA backend

The package ships an optional deterministic CUDA backend for the same single
supported profile as the CPU reference: Nikon LS-5000 ED, selector 8, Digital
ICE Normal, internal metric 4000. The exact label rests on receipts, not on
availability or speed: complete output, including every RGB16 byte, the
changed-pixel accounting, the number of public RNG advances, and the final RNG state,
compared sample by sample against this package's CPU reference on both
complete validation frames, plus a 300-row crop receipt with
cross-architecture agreement. The receipts are checked in under
[`evidence/`](../evidence/). The two complete-frame CUDA receipts bind this
tree's current source manifest; `DERIVATION.md` records the ancestry and scope
of the older public CPU and crop receipts.

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
- the 24-bit LCG and conditional-dither writer run in strict row-major site
  order on one host CPU core, through the same compiled path validated on
  the `cpu-fast` backend (`cuda_backend/host_writer.py` maps the GPU
  pipeline's per-site arrays onto it), because the number of draws a site
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
runtime, CuPy 14.1.1), complete 5,782 x 3,946 native frame, warm process,
writer chain on the host CPU (i9-12900K, one core):

| Metric | Frame 1 | Frame 2 |
|---|---:|---:|
| Full frame wall time (3 warm runs) | 5.30 - 5.81 s | 5.29 - 5.47 s |
| Writer-chain stage (download + compute + upload) | 2.65 - 2.86 s | 2.68 - 2.89 s |
| Reference CPU wall time, same frame | 3,545.7 s | 4,183.4 s |
| Speedup | about 640x | about 780x |
| Peak device memory pool | 1.86 GB | 1.86 GB |
| Peak host RSS | 2.74 GB | 2.76 GB |
| Repeated-run output hash | identical (deterministic) | identical (deterministic) |

Before this change, the sequential conditional-dither writer chain was one
device thread and took 18.5 s -- 87% of a 21.2-21.5 s total. It now runs on
one host CPU core through the compiled `cpu-fast` writer path (`write_band`)
and takes about 2.7-2.9 s -- roughly half of a 5.3-5.8 s total, with the
device-to-host transfer of the working plane and per-site arrays (about
1.1-1.2 s) the largest single piece of that stage. The reconstruction math
is unchanged; only the writer chain's execution location moved. The full
frame fits device memory with a wide margin at the supported geometry, so
no tiling is used; a free-VRAM check fails closed before allocation instead
of degrading precision or seams.

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
