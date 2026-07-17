# Source derivation boundary

The code under `src/portable_digital_ice/` is a namespace-renamed,
product-focused extraction from the earlier frozen closure implementation. It
removes evidence readers, fixture parsers, oracle comparison hooks, diagnostic
commands, and private-file path bindings.

The public full-frame receipts `evidence/frame-1-public-receipt.json` and
`evidence/frame-2-public-receipt.json` bind a source-manifest hash for the
original frozen closure tree. They do not bind the files in this extracted
runtime, and they must not be described as receipts produced by this package.
They document validation of the extraction's algorithmic ancestor only.

The CUDA parity receipts (`evidence/cuda-frame-1-parity.json` and
`evidence/cuda-frame-2-parity.json`) are different: they were produced by
this extracted package, bind one fresh source manifest over
`src/portable_digital_ice/` and `pyproject.toml` (manifest method: SHA-256
over sorted `path:sha256` lines), and compare fresh CPU-reference runs of
this package against its CUDA backend on both complete frames. In those runs
the extracted CPU reference reproduced the ancestor receipts' logical output
hashes exactly, so the extraction now has complete-frame evidence of its own
for both validation frames. The 25-check schema-v2 gate runner remains
private to the research tree; the CUDA receipts enumerate their own check
list. No private inputs or comparison outputs belong in this public tree.
