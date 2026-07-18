# Hybrid repair

`fauxce-hybrid` is an optional, provenance-first fallback around Portable
Digital ICE. It leaves the exact Portable Digital ICE result untouched except
inside a separately emitted synthesis mask. Pixels inside that mask contain
generative LaMa output. They are not scanner measurements, recovered source
samples, or an exact reconstruction.

The receipt's exactness claim is intentionally narrow:

> The hybrid output is byte-identical to the pure Portable Digital ICE output
> outside the synthesis mask. Every pixel inside the mask is disclosed as
> generative repair.

The core `portable-digital-ice` package has no ML dependency. IOPaint runs in a
separate pinned Python environment only after routing and all fail-closed
checks pass.

## What gets routed

The router consumes the core's exact `at_floor_mask`. This is an operational
signal: it means the float32 defect score saturated at the run's exported
floor. It does **not** prove that source information is absent.

The default policy is intentionally conservative and remains a judgment knob:

- label provisional regions with 8-connectivity;
- route a region when its area is at least 400 pixels **or** its maximum
  unpadded in-frame chessboard radius is at least 5 pixels;
- square-dilate the routed union by a Chebyshev margin of 4 pixels;
- relabel the dilated union so overlapping halos become one model call; and
- fail closed before model access if the final mask exceeds 2% of the frame.

"Unpadded" matters at the image edge: coordinates outside the frame are not
invented as healthy context. An all-true mask therefore fails for lack of any
healthy in-frame context.

One additional operational exclusion handles the two measured validation
frames. If one connected candidate contains **every** pixel of the complete
frame perimeter, that candidate is reported as `perimeter_excluded` and left
pure. This name makes no physical claim about film, holder, or background.
Ordinary edge-touching regions remain eligible. The final mask is exactly:

```text
square_dilate(routed_union, margin) AND NOT perimeter_excluded_component
```

The receipt records any routed halo pixels suppressed by that subtraction.

## Command-line boundary

Bare `.npy` inputs must be `uint16` RGBI arrays with shape `H x W x 4`. The
same-frame identity and locked focus/exposure values are explicit caller
assertions, not scanner evidence.

Routing-only QA does not load a model:

```shell
fauxce-hybrid \
  --prepass prepass.rgbir16.npy \
  --main main.rgbir16.npy \
  --out routing-only/ \
  --same-frame-id scan-0007 \
  --assert-focus-exposure-locked \
  --backend auto \
  --no-inpaint
```

An optional acquisition manifest can bind the two raw-array hashes. An
optional diagnostics cache stores non-pickle NumPy artifacts and a canonical
manifest bound to both inputs, the caller assertions, profile, core version and
source manifest, selected backend, exact floor bits, diagnostic arrays, and
pure output. Loading with `--from-diagnostics` also requires
`--diagnostics-manifest-sha256` with the independently retained SHA-256 of the
canonical cache manifest; the cache is rejected before its own claims are
trusted if that digest differs. Cache loading fails on any binding or artifact
mismatch.

The generative CLI additionally requires an isolated IOPaint interpreter and
an already downloaded, measured `big-lama.pt`. Run `fauxce-hybrid --help` for
the installed help text. The weights file must be exactly
`MODEL_DIR/torch/hub/checkpoints/big-lama.pt`; the caller supplies its measured
SHA-256 rather than asking the tool to trust a filename. Install IOPaint
1.6.0 into its own virtualenv
(`python -m venv ~/iopaint-venv && ~/iopaint-venv/bin/pip install iopaint==1.6.0`)
and download `big-lama.pt` from the upstream Sanster release, verifying its
SHA-256 before use.

```shell
fauxce-hybrid \
  --prepass prepass.rgbir16.npy \
  --main main.rgbir16.npy \
  --out hybrid-run/ \
  --same-frame-id scan-0007 \
  --assert-focus-exposure-locked \
  --backend auto \
  --iopaint-python ~/iopaint-venv/bin/python \
  --iopaint-executable ~/iopaint-venv/bin/iopaint \
  --iopaint-source-manifest-sha256 "$IOPAINT_SOURCE_MANIFEST_SHA256" \
  --model-dir ~/lama-models \
  --model-weights ~/lama-models/torch/hub/checkpoints/big-lama.pt \
  --model-weights-sha256 "$BIG_LAMA_SHA256" \
  --model-artifact-id Sanster-models-add_big_lama-big-lama.pt \
  --inpaint-device cpu \
  --inpaint-threads 1 \
  --inpaint-seed 0
```

To reuse a previously verified diagnostics cache, add both arguments together:

```shell
  --from-diagnostics diagnostics-cache/ \
  --diagnostics-manifest-sha256 "$DIAGNOSTICS_MANIFEST_SHA256"
```

The manifest digest must come from the cache-creation boundary or another
trusted channel, not from the mutable cache being loaded.

The runtime paths are execution inputs and are deliberately omitted from the
receipt. The expected IOPaint source-manifest digest must be retained from the
trusted runtime-install boundary; a digest read from the runtime immediately
before the same run is observation, not independent supply-chain
authentication. The weights remain external; neither the run directory nor
the examples copy them.

## 16-bit composite contract

Before allocating model crops, compositing rejects more than 4,096 final
components or more than 16,777,216 aggregate crop pixels. These fixed safety
limits are evaluated after deterministic routing and before any model call;
an over-limit run fails closed instead of retaining an unbounded number of
overlapping context windows.

Each final synthesis component is prepared, bound to a batch filename, and
composited in ascending stable ID order. IOPaint may enumerate its independent
batch files internally, so no claim is made about its private inference order;
exact filenames bind every output back to its component. Each crop is the
component bounding box expanded by 128 pixels and clamped to the frame. Every
crop comes from one immutable snapshot of the pure Portable Digital ICE
output, never a partially composited image.

For each RGB channel, the tool measures `(lo, hi)` over crop pixels outside the
global synthesis mask. It maps the 16-bit crop to 8-bit model space with
explicit float64 round-half-up affine math. A degenerate channel (`lo == hi`)
encodes to zero and decodes to `lo`. The model result is inverse-mapped with the
same recorded interval. This conversion preserves endpoints and monotonicity;
it does not claim a lossless 16-to-8-to-16 round trip.

Only component pixels are pasted. The blend weight is
`min(chessboard_distance_to_component_outside, 3) / 3`, with the frame border
treated as interior. Thus every mask pixel receives a positive generative
contribution, and touching the frame does not create a false edge ramp. The
pipeline then asserts byte identity at every pixel outside the mask.

## IOPaint and model provenance

The supported external runtime is
[IOPaint 1.6.0](https://pypi.org/project/IOPaint/), invoked as
`python -I -m iopaint run`. The project documents directory-based
[batch processing](https://github.com/Sanster/IOPaint/blob/main/iopaint/cli.py)
and its built-in
[LaMa loader](https://github.com/Sanster/IOPaint/blob/main/iopaint/model/lama.py).
The IOPaint repository was archived in August 2025, so the runtime is pinned
rather than treated as a maintained dependency.

Before each frame batch process, the adapter re-hashes the external weights,
copies them into a private minimal cache, verifies the copy, and points IOPaint
and Torch only at that cache. It constructs a minimal subprocess environment;
credentials, proxy settings, dynamic-loader injection, Python import-injection,
and shared model-cache variables are not inherited. Temporary inputs, outputs,
configuration, and the weight snapshot are removed on every exit. The receipt
records the exact weights SHA-256 and byte size, sanitized model identity,
runtime/source fingerprint, device, dependency versions, thread count, seed
controls, and a path-free invocation.

The recorded environment fingerprint is kept narrow: it covers the
enumerated runtime, dependency, platform, device, thread, weights, and Torch
determinism fields in the receipt. It is not a hash of every inherited
environment variable or the complete driver stack, so a single run records
`repeatability_observed=false`. Repeat evidence is reported separately only
after complete outputs and model artifacts are re-hashed across runs.

IOPaint's general device enum exposes Apple `mps`, but version 1.6.0 marks LaMa
as unsupported there and silently switches that model to CPU. This adapter
rejects a LaMa/MPS request instead of writing a falsely labeled MPS receipt;
use an explicitly recorded CPU or CUDA runtime. A direct probe that loads the
attested weights and runs a forward pass on `mps` and `cpu` shows the model
itself executes correctly on MPS, matching CPU output within ordinary
cross-device floating-point noise, so this is an IOPaint pinning decision
rather than a hardware limitation.

IOPaint and the
[upstream LaMa repository](https://github.com/advimman/lama/blob/main/LICENSE)
use Apache-2.0. The exact converted `big-lama.pt`
[release](https://github.com/Sanster/models/releases/tag/add_big_lama) does
not state a separate artifact license. This package therefore records that
uncertainty and never redistributes the weights.

## Output and verification

A routing-only run writes the pure RGB16 output, routing table, and run
metadata. A hybrid run additionally writes the hybrid RGB16 output, an 8-bit
mask PNG (`255` means synthesized), per-component crop evidence, a complete
diagnostics cache, and `hybrid-receipt.json` using the checked-in
`fauxce-hybrid-receipt-v2` JSON Schema.

Receipt verification re-hashes every ordinary run artifact, reconstructs the
component composite from the archived input/mask/inpainted crops, checks
routing reciprocity and synthesis counts, and re-proves outside-mask byte
identity. Before artifact or model access, it also requires the receipt to
match the running hybrid source manifest and validates the exact canonical
IOPaint command and bounded `CUDA_VISIBLE_DEVICES` form. Artifact paths are
opened without following symlinks, hashed before decoding from the same file
descriptor, and checked against per-file and aggregate decoded-size limits.
Model weights are not a run artifact: an explicit resolver can provide their
bytes or a local path for independent re-hashing without serializing that
path. Empty synthesis emits a byte-identical hybrid and a zero-count receipt
without probing IOPaint or requiring weights.

## Measured routing census

These are measured defaults on the two complete 5,782 x 3,946 validation
frames. They are routing evidence, not a claim of visual acceptance.

| Measurement | Frame 1 | Frame 2 |
|---|---:|---:|
| Frame pixels | 22,815,772 | 22,815,772 |
| At-floor pixels | 3,359,978 | 3,602,693 |
| Provisional 8-connected regions | 22,149 | 29,544 |
| Direct routed regions / pixels | 13 / 6,161 | 76 / 115,861 |
| Absorbed regions / overlap pixels | 35 / 221 | 280 / 1,661 |
| Final synthesis components | 13 | 75 |
| Perimeter-excluded pixels | 3,091,685 | 3,146,962 |
| Suppressed routed-halo pixels | 53 | 0 |
| Synthesis pixels | 16,137 | 309,360 |
| Non-floor pixels inside mask | 9,755 | 191,838 |
| Unchanged pixels inside mask | 3,551 | 136,222 |
| Synthesis fraction | 0.070727% | 1.355904% |
| 2% budget | PASS | PASS |

The mask and tables replayed deterministically. Final-source routing plus
census took 1.772205 seconds for frame 1 and 1.368509 seconds for frame 2 in
the recorded validation environment. The four-thread CPU IOPaint frame
batches took 19.369004 and 174.717674 seconds; composite plus inpainting took
20.706760 and 177.914118 seconds. Those timings are environment-specific.

## Honest limits

- The area/radius thresholds decide where generative content is permitted;
  they are not learned truth and should not be raised merely to make an output
  look pleasing.
- Grain inside synthesized regions is generated by the model and is not
  correlated with the original film grain.
- The perimeter rule is evidence-driven operational handling, not semantic
  background detection.
- Bare NumPy provenance remains caller asserted.
- Black-and-white film is out of scope.
- IOPaint/LaMa visual quality is not an exactness property.

Human acceptance review of the two measured validation frames was completed
on 2026-07-17 against dated contact sheets. Human review is required before
anyone treats the hybrid path, or these provisional thresholds, as a
default.
