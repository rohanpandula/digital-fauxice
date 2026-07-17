# Reverse-engineering notes

## The target

The goal was narrow: reproduce the LS-5000 Digital ICE Normal output without
shipping or loading Nikon's implementation. The first useful boundary was not
"remove dust well." It was "given the same prepass and main RGBI inputs, produce
the same valid RGB16 samples."

That distinction mattered. Modern inpainting can make an image look clean while
making different decisions about grain, texture, and defect edges. This project
needed an executable description of Nikon's behavior, including the odd details.

## Evidence discipline

The research copies of the Nikon modules were hash pinned and kept outside the
public package. Their hashes identified the exact build under study. Static
analysis supplied candidate functions, constants, data layouts, and call
relationships.

Runtime observation used small pass-through recorders at selected boundaries.
They recorded arguments, state, and buffers, then called the original function
unchanged. The useful captures were deliberately boring: one boundary, one data
shape, one hash, and enough surrounding state to repeat the observation.

The independent implementation never reads those binaries, static-analysis
files, captured schedules, or Nikon output. The comparison harness maps Nikon's
RGB output read-only after all portable inputs and source hashes are pinned.

## From observations to rules

Broad captures located the main stages. Small perturbations identified the
rules inside them.

Changing a single RGBI lane separated visible-image behavior from infrared
behavior. Single-pixel and single-row changes exposed neighborhood size,
producer epochs, and boundary conditions. State-field substitutions established
which values came from the prepass and which remained fixed for the scanner
profile. Long-run comparisons exposed the RNG schedule and conditional writer
calls.

Several candidate implementations could match a crop. Only one survived a
complete frame.

## The radius-4 failure

An early 4000-metric implementation used a perpendicular decision radius of 1.
It matched enough leading content to look convincing, then diverged. The runtime
state actually carried four equal radius fields set to 4 for this path.

Changing the model to radius 4 made rows 0 through 400 byte exact, then rows 0
through 1,193 byte exact across the full 3,946-pixel width. The complete-frame
run later matched all 68,447,316 valid RGB16 samples.

The point was not that radius 4 was a lucky constant. The package now treats the
resolution-dependent radius as audited profile state and rejects metrics for
which the required shape cannot be represented.

## Startup state

The writer can consume random-number state before ordinary image rows, but the
number of startup writes depends on content. One smoke fixture exercised a
nonzero startup regime. Both complete native frames correctly used zero startup
writes and began the public path at state 12,357.

Zero was initially suspicious because a broken path can also do nothing. The
output hash, explicit startup counters, and complete RNG chain settled it. The
implementation now asserts the startup regime instead of treating either zero
or nonzero activity as proof by itself.

## Long-run RNG and dither

The writer uses a 24-bit linear congruential generator. Conditional dither means
the generator advances only when the writer reaches particular decisions.
Missing one advance shifts every later draw.

The first full frame checked 34,596,507 public advances. The independent frame
checked 36,383,248. Both final states matched, and an independent verifier
recomputed the arithmetic without importing the runtime package.

## The bottom edge

Native input arrives in allocated eight-row blocks. A 5,782-row image ends with
six valid rows, not eight. Repeating rows, reading padding, or dropping the
partial block can all produce a clean-looking image with a wrong tail.

The complete gates compare the true final six rows and require zero unbound edge
fallbacks. All 723 blocks are represented in the oracle and persisted output.

## Promotion to independent-frame closure

The first complete frame proved a full execution, but it could not prove that a
frame-specific schedule or accidental fit had slipped into the code. The final
promotion rule required another physical frame with the same frozen source and
profile.

The second frame had a different scene and dust field. Its registered high-pass
correlation with the first was 0.003436, while a known repeat of the first frame
measured 0.617244. No parameter was retuned. It matched all 68,447,316 samples
and all 25 receipt checks.

## What remains product work

The closed result is an algorithmic reference for a specific scanner path. The
library now exposes progress reporting and transactional cancellation, but
scanner acquisition, TIFF writing, and application UI still need product
integration. The scalar Python reference is intentionally conservative. Its
two complete runs took roughly 49 and 72 minutes.

The CUDA backend passed its own full parity gates on both independent frames.
It produced the same 68,447,316 RGB16 samples per frame, changed-pixel counts,
RNG accounting, and startup receipts as the CPU reference. A Metal backend
still needs the same proof. Speed alone is not parity evidence: a backend is
called exact only after its complete output matches the reference byte for
byte.
