# CUDA validation environment

The parity receipts and benchmarks under [`evidence/`](../evidence/) were
produced in this reproducible container environment. Nothing in it is
specific to one machine beyond the GPU itself.

## Host

- NVIDIA RTX A4000 16 GB (GA104GL, compute capability 8.6)
- Host NVIDIA driver 610.43.02 (CUDA compatibility 13.3)
- Docker with the NVIDIA container runtime (`--runtime=nvidia`)
- Host OS during validation: Unraid 7.3.1 (Linux 6.18), Intel i9-12900K

## Container

```Dockerfile
FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        "cupy-cuda12x" \
        "numba" \
        "pytest" \
        "pytest-xdist"

WORKDIR /work
CMD ["/bin/bash"]
```

`numba` is not optional here. Since the writer chain moved to the host
(`cuda_backend/host_writer.py`), CUDA availability depends on the compiled
`fast_cpu` kernels, and the backend fails closed with a specific reason
without them. Installing it resolves numpy to 2.4.6 rather than the 2.5.1
this file previously pinned, which is why the receipts record 2.4.6; byte
equality is unaffected, and the package's own tests cover both.

Validated component versions: Python 3.12.3, NumPy 2.4.6, numba 0.66.0,
CuPy 14.1.1 (`cupy-cuda12x`), CUDA runtime 12.9 (bundled by CuPy), NVRTC
options `--fmad=false --std=c++17`.

## Reproducing the public suite

```sh
docker build -t dice-cuda:dev .
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all \
    -v "$PWD":/work/portable-digital-ice dice-cuda:dev \
    bash -c "cd /work/portable-digital-ice && pip install -e . -q && pytest -q"
```

The private full-frame gates additionally mount the hash-pinned fixture pair
read-only and compare against a CPU-reference output produced by this same
package; their sanitized results are the `evidence/cuda-*-parity.json`
receipts. `tools/mint_parity_receipt.py` runs those gates and re-verifies
the receipts — see
[`validation.md`](validation.md#regenerating-a-receipt):

```sh
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all \
    -v "$PWD":/work/pdi -v /path/to/fixtures:/fixtures:ro dice-cuda:dev \
    bash -c "cd /work/pdi && pip install -e . -q && \
      python tools/mint_parity_receipt.py --case frame1 --backend cuda \
        --fixtures /work/pdi/manifest.json"
```

## Benchmark method

Warm timings are wall-clock over `process_cuda` after one discarded cold run
(NVRTC compile); three repetitions per shape; determinism asserted by
comparing output SHA-256 across repetitions. Device-stage times come from
CUDA events (`stage_timings` parameter); utilization/power/VRAM from 1 Hz
`nvidia-smi` sampling on the host. Full data: `evidence/cuda-stage-profile.json`.
