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

## Regenerating a receipt

[`tools/mint_parity_receipt.py`](../tools/mint_parity_receipt.py) mints the
complete-frame parity receipts and re-verifies a checked-in one. It produces
the `evidence/` files in the format they are checked in, and reproduces the
existing Metal ones exactly, so a receipt can be regenerated and audited
instead of taken on trust.

The private captures are not redistributable and are not in this repository.
Point the script at them with a fixture manifest, either `--fixtures` or the
`PORTABLE_DICE_FIXTURE_MANIFEST` environment variable;
[`tools/fixtures.example.json`](../tools/fixtures.example.json) documents
every field. Each case names two files:

| Role | File | Contents |
|---|---|---|
| main | `<case>/main.dicein1` | the main-pass RGBI16 acquisition |
| prepass | `<case>/prepass.dicepp1` | the prepass acquisition and its variant table |

Paths in the manifest resolve against its `workspace_root` (or
`--workspace-root`). The Nikon oracle is not needed to mint these receipts:
parity is established against the in-process baseline backend and bound to
`expected_logical_output_sha256`, the oracle-matched CPU-reference hash the
original gates recorded.

Re-verify the checked-in Metal receipt for frame 1:

```sh
python tools/mint_parity_receipt.py --case frame1 --backend metal \
    --fixtures /path/to/manifest.json
```

The script runs the baseline backend and the candidate backend in one process
over identical input bytes, compares them with the package's own
`_parity_failures` comparator plus a full-frame delta sweep, re-runs the
candidate to prove consecutive-run determinism, and prints the minted
receipt. Without `--out` it writes nothing, so a checked-in receipt is never
overwritten by accident. Without `--verify` it compares against the
`evidence/` file for that backend and case and exits non-zero on any
difference in a binding field.

It fails closed with exit 2, before running anything, when the manifest is
missing or unreadable, its schema is unknown, the case is unknown, a fixture
is absent, a fixture's SHA-256 does not match its pin, or two fixture roles
turn out to be the same file. It never writes a partial or empty receipt.

The same gate runs under pytest:

```sh
PORTABLE_DICE_FIXTURE_MANIFEST=/path/to/manifest.json \
    pytest tests/test_full_frame_receipts.py
```

Without that variable the complete-frame tests skip and say why. Everything
else in that file — the DICEIN1/DICEPP1 reader against synthetic fixtures
built in the test, and every fail-closed leg of the minting script — runs in
ordinary continuous integration with no private data.

### What is reproduced, and what is not

`--verify` compares the fields that carry the claim: status, geometry, sample
count, the four mismatch and delta counters, `diagnostics_planes_equal`, the
output hash, `attempted_pixels`, `written_pixels`, `changed_pixels`,
`public_rng_advances`, `final_rng_state`, `startup_rng_advances_per_stage`,
both fixture hashes, and both raw RGBI16 input hashes. Prose, wall times,
host details, and the gate-check tally are excluded: they vary by machine and
do not carry the claim. Fields an older receipt never recorded are reported
as not compared rather than counted as agreement, and a verify pass that
compared nothing is a failure, not a success.

The output-hash field is named after the two backends that agreed, so
re-minting against a different baseline renames it. It is compared by value
across the two names rather than skipped: it is the same frame's output hash
either way, and what binds it to the CPU reference is the
`candidate_matches_pinned_cpu_reference_output` gate check, not the key
name.

Verified 2026-07-26 on the arm64 validation host (Apple M4, macOS 26.5.2,
Python 3.13.5, numpy 2.4.6, numba 0.66.0, pyobjc 12.2.1): `--backend metal`
re-mints both `metal-frame1-parity.json` and `metal-frame2-parity.json` with
all 19 comparable binding fields identical and 28 of 28 named gate checks
passing, and its `source_manifest_sha256` equals the 31-file value those
receipts record in their re-verification block.

The same day, `--backend cpu-fast --baseline cpu` re-minted
`cpu-fast-frame-1-parity.json` against the exact CPU reference, 28 of 28
checks, with all 15 comparable binding fields identical; the four fields that
receipt predates were reported as not compared rather than counted as
agreement. That run matters beyond the receipt: the Metal claim is
hash-transitive through the CPU-reference output hash, and this is the CPU
reference itself re-deriving `c3ee49f4…6ad7` on the same host, so the chain
no longer rests only on the earlier receipts. It took 2,537.8 seconds
against the compiled backend's 10.0.

`--backend cuda` re-minted `cuda-frame-1-parity.json` and
`cuda-frame-2-parity.json` the same day on the recorded validation host
(RTX A4000, driver 610.43.02, the `nvidia/cuda:12.6.3-devel-ubuntu24.04`
container, Python 3.12.3, numpy 2.4.6, CuPy 14.1.1), 28 of 28 checks each,
with all 16 comparable binding fields identical. Warm CUDA wall time was
6.305 s on frame 1 against the 6.314 s that receipt records. Those two
receipts predate the diagnostics-plane and raw-input fields, so those were
reported as not compared. Both were re-minted against the compiled
`cpu-fast` baseline rather than the hour-long reference; the binding to the
CPU reference is the `candidate_matches_pinned_cpu_reference_output` check,
which passed in every case.

Every complete-frame receipt under `evidence/` has now been regenerated from
its fixtures and matched against its checked-in values.

### Why this script exists

The scripts that minted the original receipts were never checked in — for the
CUDA and cpu-fast receipts as well as the Metal ones, and including the
re-verification after the Metal session-leak fix. The 32-file source manifest
the Metal receipts pin included the minting script itself, which is how its
absence became visible: the 31-file re-verification manifest is the same
scope with the script removed. This script closes that gap. It lives in the
repository, is covered by tests, and records its own SHA-256 in every receipt
it mints under `minted_by`.

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
