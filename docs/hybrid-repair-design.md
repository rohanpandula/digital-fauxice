# Hybrid repair design: exact ICE core, bounded inpainting fallback

The hybrid mode routes severe, spatially broad defect candidates to a
generative fallback while leaving the exact ICE result everywhere else. The
routing signal is conservative operational evidence, not proof that visible
signal is physically absent. A provenance sidecar records exactly which
pixels received any generative contribution. This document fixes the
architecture decisions, the phased plan, the test gates, and the definition
of done for that mode.

## Background

The relevant context, in reading order:

1. `README.md` and `docs/input-contract.md` — what the package is, the single
   supported profile (Nikon LS-5000, selector 8, ICE Normal, metric 4000,
   RGBI uint16 in, RGB16 out), and the fail-closed philosophy.
2. `docs/validation.md` and `DERIVATION.md` — what "exact" means here: byte
   parity receipts and source-manifest binding. The core's value is that its
   output is bit-for-bit the scanner vendor's, proven on complete frames.
3. `docs/cuda-backend.md` and `docs/cuda-decision-record.md` — the CUDA
   backend and the CPU/GPU boundary.
4. `src/portable_digital_ice/x3a.py` (`derive_auxiliary`,
   `continuous_score`), `src/portable_digital_ice/streaming.py` (the row
   engine), `src/portable_digital_ice/backend.py` (`ComputeBackend`,
   `process()`).

The physics that motivates the feature: dust and scratches mostly *dim* the
image rather than erase it. The scanner's infrared channel helps measure that
attenuation. The exact path often recovers attenuated visible data and uses a
small local reconstruction aperture where its gates permit repair. A
generative inpainter can be useful only for unusually broad losses, and it
must never replace recoverable pixels casually.

The routing signal already exists inside the engine. The score plane
(`continuous_score`) is a float32 H×W operational severity map computed from
an auxiliary that mixes infrared, one visible channel, and run-specific
prepass calibration. It uses a three-sample horizontal minimum away from the
first and last columns, then clamps to a run-constant floor. Therefore
`at_floor_mask` means only "the score used by this run saturated at its
float32 floor." It is prepass-relative, its geometry includes that horizontal
minimum, and it makes no claim that RGB signal is absent. Connected at-floor
regions are candidate evidence to measure and route conservatively.

## Ground rules

- **The exact core stays exact.** The core package receives exactly one
  additive change (Phase A) and nothing else. No edits to any existing
  computation line, in either backend. Any core change is followed by the
  full re-gate in Phase A before later phases proceed.
- **No new required dependencies for the core package.** scipy, torch, and
  the inpainting runtime live only in the separate hybrid package.
- **The hybrid tool never claims exactness.** Its receipts must state that
  synthesized regions are generative. Pixels outside the synthesis mask must
  be byte-identical to the pure exact output; that equality is the hybrid's
  own testable contract.
- Traditional silver B&W film is out of scope: it has no usable IR channel.
- A gate that fails twice in a row is evidence the design needs revision,
  not permission to weaken the gate.

## Preconditions

1. `evidence/cuda-frame-1-parity.json` and `evidence/cuda-frame-2-parity.json`
   both exist with `"status": "PASS"`. Both validation lanes must be closed
   before Phase A starts.
2. The full public test suite passes.
3. The current `source_manifest_sha256` from the receipts is noted; Phase A
   supersedes it and must mint new receipts.

## Phase A — diagnostics export from the exact core

Goal: the engine optionally exposes the three planes the router needs,
without changing a single output byte.

Both `process_cpu` and `process_cuda` gain an opt-in diagnostics output
(keyword `export_diagnostics: bool = False`; when true, a small dataclass is
attached to the returned `ProcessingResult`):

- `score_plane`: float32 H×W — the score values the run actually used.
- `score_floor`: float32 scalar — the exact run floor, retained even when no
  pixel equals it so receipts never infer the floor from a possibly empty mask.
- `at_floor_mask`: bool H×W — score exactly equal to the run's float32 floor
  value, computed from the same buffer the run used, not re-derived.
- `changed_mask`: bool H×W — final RGB16 differs from the no-op conversion
  of the input. Both backends already count these pixels
  (`replay.changed_pixels`); the mask is materialized behind the flag.

Rules: additive only; default-off; zero allocations or kernel changes on the
default path (CUDA may download existing device buffers when the flag is on;
the score plane already lives on-device). No computation is reordered.

Self-checking tests (public, synthetic):

1. `changed_mask.sum() == replay.changed_pixels` on a synthetic acquisition,
   both backends.
2. CPU and CUDA diagnostics are bit/bool-identical on the synthetic parity
   self-test acquisition.
3. A run with `export_diagnostics=False` returns a result with no
   diagnostics and its output hash equals the flag-on run's hash.
4. A deliberately constructed low-IR patch produces a non-empty at-floor mask
   with the exact expected horizontal widening; the tests must not pass
   vacuously on two all-false masks. The exported floor bits equal the run
   parameters on both backends, including when `at_floor_mask` is empty.

**Phase A gate (in this order):**

1. The full public suite passes on CPU, and on a CUDA host for the CUDA lane.
2. The binding parity gate re-runs for BOTH validation frames on the new
   tree. Because Phase A must not change output bytes, the existing
   CPU-reference outputs remain valid comparison targets; the gate re-binds
   the new source manifest. All checks must pass 12/12 per frame.
3. The fresh receipts replace `evidence/cuda-frame-1-parity.json` /
   `cuda-frame-2-parity.json`.
4. Any mismatch means the "additive" change altered the pipeline: revert and
   rethink. No phase proceeds on a red gate.

## Phase B — hybrid package skeleton and routing (no ML yet)

The hybrid tool is a sibling package (never nested inside the core):

```
hybrid-repair/
  pyproject.toml        # package fauxce_hybrid; deps: portable-digital-ice, numpy, scipy, imageio
  src/fauxce_hybrid/
    __init__.py
    routing.py          # region extraction + triage
    composite.py        # 16-bit crop/stretch/paste/feather (Phase C)
    receipts.py         # sidecar mask + JSON receipt
    cli.py              # fauxce-hybrid entry point
  tests/
```

`routing.py`:

- Input: `at_floor_mask` (bool H×W). Validate dtype/rank/non-empty geometry.
  Label with explicit 8-connectivity:
  `scipy.ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))`.
- Compute the chessboard distance transform inside the mask without treating
  the image boundary as healthy context. Record each region's maximum
  in-region chessboard radius. An all-true frame has no healthy in-frame model
  context and fails closed before routing; it is not assigned an invented
  infinite radius or sent to a model.
- Route a region iff `area >= min_area` (provisional default **400** px) OR
  `max_chessboard_radius >= min_radius` (provisional default **5** px). Radius
  5 means at least one point has no non-candidate pixel in the core's 9×9
  local aperture. Record `routed_reason` as `area`, `chessboard_radius`, both,
  or `not_routed`. These defaults remain provisional until the real-frame
  census gate below; they are not validated physical thresholds.
- Independently of its threshold reason, a component containing every pixel
  of the complete frame perimeter is classified `perimeter_excluded` and
  excluded from synthesis. This is an operational geometry classification,
  not a claim that its visible content is physically background. The narrow
  rule is evidence-driven: both validation frames contain a dominant component
  with exactly that geometry. A component merely touching one or more edges
  remains eligible. If image support is not represented by this exact
  geometry on another input, a caller-provided, hash-bound valid-image mask
  is required rather than broadening the heuristic. A candidate 8-connected
  to the excluded component is conservatively excluded with it; this
  intentional under-repair is recorded.
- Form the routed union, then apply one clipped square/Chebyshev dilation with
  `margin` (default **4** px; a `(2*margin+1)²` footprint). Relabel the dilated
  union with 8-connectivity. Each resulting disjoint component is processed
  exactly once; overlapping halos can never be pasted in input-region order.
- The final mask equation is
  `square_dilate(routed_union, margin) AND NOT perimeter_excluded_component`.
  The exact number of dilated halo pixels suppressed by the exclusion is
  recorded; they are never silently carved.
- Non-routed regions absorbed by a routed halo are recorded with the final
  component IDs they touch, the exact absorbed-pixel count, and the number of
  synthesis-mask pixels that were not at floor. Those pixels are not called
  defects or holes.
- Output is deterministic, read-only label/mask arrays plus sorted
  provisional and final component tables. Bboxes are half-open. Serialization
  uses a versioned schema and stable key/region order. Provisional
  dispositions are exactly `routed`, `absorbed`, `perimeter_excluded`, or
  `pristine`. Provisional IDs are scipy's deterministic row-major label IDs;
  final IDs are normalized by the stable key `(y0,x0,y1,x1,raw_label_id)`,
  and both rules are recorded.
- Synthesis fraction is computed and checked against `max_synth_fraction`
  (provisional default **0.02**). Routing-only QA may still emit its census
  when over budget, but no model may load or run until the budget check
  passes. The CLI permits an explicit numeric limit override, never an
  unrecorded force bypass.

`cli.py` (v1 surface, minimal):

```
fauxce-hybrid --prepass prepass.npy --main main.npy --out outdir/ \
    --same-frame-id ID --assert-focus-exposure-locked \
    [--backend auto|cpu|cuda] [--min-area 400] [--min-radius 5] \
    [--margin 4] [--max-synth-fraction 0.02] [--no-inpaint] \
    [--acquisition-manifest provenance.json] \
    [--save-diagnostics DIR | --from-diagnostics DIR]
```

- Loads the two RGBI uint16 arrays with `allow_pickle=False`, validates exact
  H×W×4 uint16 layout, hashes both arrays, builds the fixed-profile
  `ProcessingJob`, and runs `process()` with diagnostics on. For bare `.npy`
  input, `--same-frame-id` and `--assert-focus-exposure-locked` are explicit
  caller assertions and are labeled that way in every cache and receipt. An
  optional acquisition manifest is itself hashed and may assert the expected
  raw-array hashes, but does not turn bare `.npy` files into scanner evidence.
- Writes: `output.rgb16.npy` (pure exact output) and the region table JSON.
  After Phase C it also writes `output-hybrid.rgb16.npy`, `synth-mask.png`,
  and `hybrid-receipt.json`. `--no-inpaint` stops after routing (useful for
  QA).
- An optional diagnostics cache stores separate non-pickle `.npy` artifacts
  for the pure output and all diagnostic planes plus a JSON manifest binding
  every artifact file hash and canonical raw-array hash, both input hashes,
  provenance assertions/manifest hash, profile, core version/source manifest,
  output hash, exact floor bits, and selected backend. Canonical raw bytes
  are little-endian uint16 for RGB arrays, float32 bit patterns in C order
  for scores, and one byte per bool in C order for masks. Canonical JSON is
  UTF-8, sorted keys, no NaN, compact separators, one trailing newline.
  Loading a cache identifies the pure output artifact explicitly and fails
  closed on any hash/type/shape/input/provenance/core mismatch.
- For `auto`, a full-run `CudaBackendUnavailable` may fall back to CPU with
  the reason recorded. An explicit `cuda` request remains fail-closed. Until
  the full-frame diagnostics-parity gate below passes, real hybrid routing
  uses CPU diagnostics even when the pure output came from CUDA. If the gate
  runs and any diagnostic hash disagrees, CUDA diagnostics are disqualified:
  routing pins to CPU and the mismatch is recorded rather than treated as an
  unavailable gate.

Phase B tests (synthetic, no ML):

1. Reason attribution: a 3×3 speck does not route; a 1×400 strip routes by
   area only; a 9×9 blob routes by radius only; 30×30 and 500×12 regions
   route by both; a one-pixel diagonal with a large bbox does not route by
   radius.
2. Radius boundary cases: an 8-pixel-wide band has maximum radius 4 and does
   not route by radius; a 10-pixel-wide band has radius 5 and does.
3. Diagonal-touching pixels form one provisional region. A one-pixel input
   dilated by margin 4 is exactly a clipped 9×9 square.
4. Two routed regions whose halos overlap become one final component and one
   paste target. A nearby unrouted speck is reported as absorbed exactly.
5. Edge-touching dilation has the expected clipped bbox and mask. A component
   covering the complete perimeter is reported as `perimeter_excluded` and
   excluded, while an ordinary edge-touching component remains eligible. A
   routed region whose halo overlaps the excluded component records the exact
   suppressed count and never synthesizes an excluded pixel. An all-true mask
   fails closed for lack of healthy context before model load.
6. The synthesis mask equals the square-dilated routed union exactly; two
   runs produce identical tables, labels, masks, and serialized bytes.
7. Over-budget routing remains reportable under `--no-inpaint` but the model
   guard raises before any inpainter import or invocation.

**Phase B gate:** tests pass; the CLI's `--no-inpaint` mode runs end-to-end
on a synthetic acquisition and its `output.rgb16.npy` hash equals a direct
`process()` call's output hash. No hybrid output, mask PNG, or hybrid receipt
exists in routing-only mode. A cache round-trip passes; tampering with any
artifact or input binding fails closed.

## Mandatory evidence gate before Phase C

Phase C may not start until both items are satisfied:

1. On both complete validation frames, CPU and CUDA run with diagnostics
   enabled and byte hashes are recorded for the score plane, at-floor mask,
   and changed mask plus the score-floor bits. All three CPU/CUDA plane-hash
   pairs and the scalar bits per frame must agree, and changed-mask counts
   must equal replay counters. If this full-frame gate is not available, real
   hybrid routing pins to CPU diagnostics and states that limit.
2. Both real-frame at-floor masks are censused before defaults are
   confirmed: pixel count/fraction, 8-connected region count, area/radius
   distributions, would-route reasons, dilated synthesis fraction,
   absorbed-region count, and budget outcome. Provisional defaults are
   revised from evidence or the 2% fail-closed guard is kept; thresholds are
   never tuned merely until a visually pleasing output appears.

## Phase C — inpainting integration and 16-bit composite

Inpainter: **LaMa via IOPaint 1.6.0** (`pip install IOPaint==1.6.0`, model
`lama`; weights download on first use). IOPaint's repository was archived in
August 2025, so this is an explicitly pinned external runtime rather than a
maintained dependency. The IOPaint and upstream LaMa code licenses are
confirmed and recorded at install. The converted weights artifact's license
is treated as unconfirmed unless the exact download supplies one; it is never
redistributed. The adapter probes the actual CLI of the pinned version and
invokes it as a subprocess on a directory of crops (image + white-on-black
mask pairs). One device and thread count are pinned per run and recorded;
determinism is claimed only within that recorded environment. torch and
IOPaint are never added to the core package. The adapter executes help,
metadata probes, and inference through one isolated interpreter, scrubs
Python import-injection variables, fingerprints the effective
dependency/device environment, and copies the attested weights into a
private minimal model cache before inference. It verifies the snapshot,
points all IOPaint/Torch cache variables at that private directory, never
exposes the provenance file to IOPaint, and removes the snapshot afterward.

The composite starts as a bit-copy of the pure exact output. All component
crops come from that immutable pure output, never from a partially
composited image. Final components are processed in ascending deterministic
ID order even though their masks are disjoint.

Per final synthesis component, in `composite.py`:

1. Crop the component bbox expanded by **128 px** per side, clamped to frame.
2. For each channel, compute `(lo, hi)` over crop pixels outside the union of
   all synthesis components. At least one such context pixel is required or
   the component fails closed before model load. For `hi > lo`, map a uint16
   source sample `v` to
   `clip(floor((float64(v)-lo)*255/(hi-lo) + 0.5), 0, 255)` and then cast to
   uint8. If `lo == hi`, encode that channel as zero and record it as
   degenerate. The 8-bit crop is model context; no uint16 round-trip
   exactness is claimed.
3. Inpaint the 8-bit crop using only this component's white-on-black mask.
4. Inverse-map model values with
   `clip(floor(lo + v8*(hi-lo)/255 + 0.5), 0, 65535)` in float64. Endpoints
   must map exactly and the conversion must be monotonic.
5. Define `D(p)` as chessboard distance from an in-component pixel to the
   nearest outside-component pixel, treating the image border as interior so
   no feather is introduced merely by touching the frame edge. With `F=3`,
   `alpha(p)=min(D(p),F)/F`. Every mask pixel therefore has alpha at least
   1/3; no mask pixel falsely claims synthesis while receiving zero weight.
6. Paste only through component-mask indexing. Blend in public RGB16 space:
   `floor(alpha*synth + (1-alpha)*exact + 0.5)`, float64 then clipped uint16.
   No assignment may target an outside-mask pixel. A poisoned full-frame
   synthetic buffer test must prove this is structural, not accidental.

After each run: `hybrid[~mask] == exact[~mask]` byte-for-byte,
`synth_pixel_count == mask.sum()`, and `mask[p]` iff that pixel received a
strictly positive generative contribution. Empty routing emits a
byte-identical hybrid output and a zero-count receipt without invoking the
inpainter.

`receipts.py` — always emitted with a hybrid output:

- `synth-mask.png`: 8-bit PNG, 255 = synthesized pixel, 0 otherwise.
- `hybrid-receipt.json`, schema `fauxce-hybrid-receipt-v2`, validated against
  the checked-in package resource
  `hybrid/src/fauxce_hybrid/schemas/fauxce-hybrid-receipt-v2.schema.json`.
  Required
  groups:
  - disclosure: generative regions true; exactness claim limited to outside
    the synthesis mask;
  - generation time, hybrid tool version, and deterministic hybrid source
    manifest hash;
  - input raw-array hashes, caller-asserted same-frame provenance, geometry;
  - core package version, core source-manifest hash, profile, requested/used
    backend, selection reason, diagnostics-source backend, floor bits, and
    at-floor count;
  - routing policy/connectivity/distance/dilation, provisional-default flag,
    scipy version, all provisional/final components, absorbed regions,
    non-floor and unchanged pixels synthesized, budget and limit;
  - per-component crop bbox, per-channel `(lo,hi,degenerate)`, rounding rule,
    feather metric/width/border rule, and paste order;
  - inpainter/model name and version, an external attestation for the exact
    model weights (measured SHA-256, byte size, and sanitized artifact
    identifier, never an absolute user path), torch version, device, thread
    count, seed, invocation argv, and single-environment determinism scope;
    model weights are never copied into the run directory or examples; and
  - SHA-256 for the pure output, hybrid output, synthesis-mask PNG, and every
    referenced artifact, plus synth count/fraction.

A weights hash is never claimed unless it was measured from the exact file
loaded. Receipt verification re-hashes every ordinary run artifact and fails
on any mismatch. For a non-empty run it also accepts an explicit external
model-weights resolver, re-hashes the returned bytes or local file, and
fails closed when weights verification is required but the resolver is
missing or does not match the attestation. No private resolver path is
serialized.

Phase C tests:

1. **The invariant test (most important):** on a synthetic acquisition with
   an injected large defect, run the full hybrid path; every pixel where
   `synth-mask == 0` is byte-identical to the pure exact output, and
   `synth_pixel_count == (mask != 0).sum()`.
2. With a mock inpainter returning arbitrary values, poison every pixel in
   its returned crop and prove non-mask output remains byte-identical. Test
   inverse endpoint exactness and monotonicity for ranges 0, 1, 255, 256,
   and 65535.
3. Unit-test alpha: positive on every mask pixel, exactly 1 at distance ≥3,
   no frame-edge ramp, and hand-computed RGB16 blends at distances 1, 2, 3.
4. Adjacent input regions whose halos merge invoke the model once. Repeated
   runs in the same environment produce identical output, mask, crop, and
   model-artifact hashes (receipt bytes may differ by generation timestamp),
   and permuting a test-only component iteration order cannot change output.
5. Receipt schema completeness: every field present; hashes verify against
   inputs, outputs, mask, cache, and externally resolved weights. Flipping
   one mask byte must fail verification. Empty synthesis requires no weights
   resolver and records that the inpainter was not invoked.
6. If the inpainter is not installed, the CLI fails closed with a clear
   message (and `--no-inpaint` still works).

**Phase C gate:** all tests pass; one synthetic end-to-end run archived
(inputs, outputs, mask, receipt) under the hybrid package's `examples/`.

## Phase D — real-frame validation and docs

The hybrid CLI runs on both validation frames and, where available, frames
from a full roll. The following are measured and reported, not estimated:

- at-floor pixel count and fraction of changed pixels, per frame;
- region area/radius distribution; how many route for each reason; absorbed
  regions; non-floor and unchanged pixels inside the dilated mask;
  synthesized fraction and 2% budget outcome (actual, never an expectation);
- wall time of routing and of inpainting;
- a contact-sheet PNG per frame: exact output with routed regions outlined,
  plus before/after crops of the 3 largest regions — the artifact a human
  reviewer actually judges.

`docs/hybrid-repair.md` in the hybrid package then documents: what routes
where and why, threshold semantics, the generative-content disclosure, the
receipt schema, and the honest limits (the threshold is a judgment knob;
grain in synthesized regions is uncorrelated until the stretch goal lands;
B&W is out of scope). The core README gains one paragraph and a link in its
backends section, clearly labeling the hybrid tool as optional and
non-exact.

**Phase D gate:** contact sheets exist for both validation frames; the
measured report is written; every run stayed within its recorded synthesis
budget or failed closed; and a human review is explicitly requested before
anyone treats hybrid output as a default. A budget failure is evidence that
the routing premise or defaults need revision, not permission to raise the
limit until output looks attractive.

## Stretch goals (only after Phase D acceptance)

- Grain matching: measure high-frequency noise statistics in a ring of real
  pixels around each synthesized region; add matched noise inside the region.
- Diffusion-model fallback for very large losses.
- TIFF16 output in the CLI.

## Definition of done

Same standard as the CUDA lane: a checklist claim is only done when its gate
ran and passed, and every number in the report was measured, not estimated.
The hybrid's exactness contract — outside-mask bytes identical to the pure
exact output — is enforced by a test, not by intention. When in doubt
between a nice result and an honest one, pick honest.
