# Hybrid repair roadmap

Digital Fauxice has an exact mode with a narrow promise: for the validated
Nikon LS-5000 profile, it reproduces the recovered Digital ICE Normal result.
That mode stays fixed. Hybrid repair will be a separate, opt-in mode for cases
where a larger scratch, damaged edge, or unusual texture needs a different
fill.

The goal is not to replace infrared detection with RGB guesswork. Infrared is
the reason the system can distinguish surface damage from film detail. The
hybrid path should preserve that advantage while giving difficult regions a
second repair option.

## Proposed pipeline

1. Validate and align the 285 dpi prepass and 4000 dpi main RGBI capture using
   the same fail-closed input contract as exact mode.
2. Run the exact detector and reconstruction first.
3. Score each repaired region using infrared support, neighborhood agreement,
   edge continuity, and texture consistency.
4. Keep the exact result in high-confidence regions.
5. Send only low-confidence regions, with a small amount of surrounding
   context, to a bounded content-aware repair method.
6. Return cleaned RGB together with a provenance mask that identifies untouched
   pixels, exact repairs, and hybrid repairs.

The fallback may use deterministic patch matching or a learned fill, but it
must not edit pixels outside the infrared-supported defect region. The first
implementation should test both approaches instead of assuming that the more
complex one is better.

## Rules that do not change

- Exact mode remains available and keeps its existing byte-parity receipts.
- Hybrid mode never inherits the word "exact" from the detector it uses.
- Raw RGBI inputs remain unchanged and available for reprocessing.
- Unsupported scanner modes, image geometry, and infrared alignment fail
  before output is written.
- Every output records the mode, engine version, backend, thresholds, and
  repair counts.
- A requested accelerator cannot silently fall back to another backend.

## Validation before release

The comparison set should include small dust, dense clusters, hair, long
scratches, defects crossing strong edges, fine grain, foliage, faces, sky, and
clean textured regions that should not change. It should contain multiple
physical frames and at least one complete roll acquired without retuning.

Each candidate should be compared with four baselines: no repair, generic
RGB-only inpainting, exact Digital Fauxice, and the hybrid result. The gate
should report error inside known defect regions, collateral changes outside
the mask, edge continuity, repeatability, and runtime. Full-frame crops should
also be reviewed without labels so a visually convincing fill cannot hide
damage to clean film detail.

Hybrid mode is ready only if it improves the low-confidence cases while staying
inside the permitted mask. A faster or more attractive result is not enough if
it changes clean pixels or cannot be reproduced.

## Implementation order

1. Expose confidence and provenance from the exact engine without changing its
   output.
2. Build a deterministic patch-based baseline and run it against the validation
   set.
3. Add a learned candidate only if it beats that baseline on the same frozen
   gates.
4. Batch low-confidence regions so CUDA and a future Metal backend avoid
   per-region launch overhead.
5. Add the opt-in API only after the output schema and receipts are stable.

This is a roadmap, not a claim about v0.1.0. The released version contains the
exact CPU and CUDA paths only.
