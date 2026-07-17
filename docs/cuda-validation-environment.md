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
        "numpy==2.5.1" \
        "cupy-cuda12x" \
        "pytest" \
        "pytest-xdist"

WORKDIR /work
CMD ["/bin/bash"]
```

Validated component versions: Python 3.12, NumPy 2.5.1, CuPy 14.1.1
(`cupy-cuda12x`), CUDA runtime 12.9 (bundled by CuPy), NVRTC options
`--fmad=false --std=c++17`.

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
receipts.

## Benchmark method

Warm timings are wall-clock over `process_cuda` after one discarded cold run
(NVRTC compile); three repetitions per shape; determinism asserted by
comparing output SHA-256 across repetitions. Device-stage times come from
CUDA events (`stage_timings` parameter); utilization/power/VRAM from 1 Hz
`nvidia-smi` sampling on the host. Full data: `evidence/cuda-stage-profile.json`.
