# Validation and claim boundary

This document defines the CPU-reference closure gates. The CUDA backend has
its own parity receipts (`evidence/cuda-frame-1-parity.json` and
`evidence/cuda-frame-2-parity.json`, both PASS 12/12) whose acceptance rule
is byte equality with this package's CPU reference on the complete frame,
plus identical counters, RNG accounting, and startup receipts. See
[`cuda-backend.md`](cuda-backend.md).

## Acceptance rule

A complete-frame result passes only when all valid RGB16 samples equal Nikon's
logical output and every receipt check is true. Visual review, average error,
mask overlap alone, and prefix equality do not qualify.

Each complete gate checks:

- pinned hashes for the main RGBI input and prepass input;
- the content-derived prepass record;
- exact shape and sample count;
- zero sample and pixel mismatches;
- zero maximum and summed absolute delta;
- identical logical output hashes;
- identical changed-pixel masks;
- startup and public RNG behavior;
- the final partial block and all edge paths;
- the absence of dynamic traces or captured producer schedules at runtime;
- immutable inputs and distinct file roles; and
- an unchanged source manifest before and after execution.

The independent verifier imports neither the portable package nor the gate
runner. It rehashes the persisted artifacts and checks the RNG arithmetic from
the receipt.

## Complete native frame 1

- Geometry: 5,782 x 3,946 x 3 RGB16
- Samples compared: 68,447,316
- Mismatched samples and pixels: 0 and 0
- Maximum and summed absolute delta: 0 and 0
- Changed pixels: 6,426,156, identical on both sides
- Public RNG advances: 34,596,507
- Final RNG state: 8,880,392
- Unbound edge fallbacks: 0
- Gate checks: 25 of 25
- Logical output SHA-256:
  `c3ee49f49f71cbc544da595901522d945f549464f0f3b9a543e98171e56f6ad7`
- Private canonical receipt SHA-256:
  `763e5d2748a0d56121c9adbcc61b29a67544a535598451d477bc06e70fbc636e`
- Runtime: 4,313.758 seconds

## Independent native frame 2

- Geometry: 5,782 x 3,946 x 3 RGB16
- Samples compared: 68,447,316
- Mismatched samples and pixels: 0 and 0
- Maximum and summed absolute delta: 0 and 0
- Changed pixels: 6,718,151, identical on both sides
- Public RNG advances: 36,383,248
- Final RNG state: 16,418,997
- Unbound edge fallbacks: 0
- Gate checks: 25 of 25
- Logical output SHA-256:
  `f2e9b84ddc6bc49e2e34b9dd86cac992ba12add6e0f4a14720e2fde74719958a`
- Private canonical receipt SHA-256:
  `0aa6afe0f169253e92e27c000f6a38606c259b3c1c4aa92c5aff9ccf1de00749`
- Runtime: 2,934.204 seconds

The frozen source manifest for both complete gates was:
`089d6496065685f3791cd1ad0ccd140e284aa2d0e0d1672727e30de288a26f8c`.
That manifest names the original frozen closure source. It does not name or
validate the namespace-renamed extraction under `src/portable_digital_ice/`.
See [`DERIVATION.md`](../DERIVATION.md) for the revalidation requirement.

## Independent-content check

Registered high-pass correlation between frame 1 and frame 2 was 0.003436. A
known repeat capture of frame 1 measured 0.617244. This check rejects a renamed
or lightly shifted duplicate as the independent validation frame.

## What the receipts do not prove

Two complete frames do not establish every film stock, defect shape, scanner
model, or Digital ICE mode. The evidence supports the LS-5000 selector-8 Normal
path at the observed metrics. The runtime rejects unsupported profiles rather
than guessing.

The original 25-check receipts do not directly bind the extracted runtime in
this repository. The two later CUDA parity receipts do: each compares a fresh
CPU run from this package with its CUDA output on a complete frame, and the CPU
output hashes also match the original Nikon oracle results exactly. See
[`DERIVATION.md`](../DERIVATION.md) for the two receipt lineages.

The public JSON files omit private paths, raw scanner data, proprietary oracle
buffers, and binary-analysis artifacts. Their `canonical_private_receipt_sha256`
fields bind them to the full internal receipts without publishing those files.
