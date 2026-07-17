# Portable Digital ICE

By Rohan Pandula

[![tests](https://github.com/rohanpandula/portable-digital-ice/actions/workflows/tests.yml/badge.svg)](https://github.com/rohanpandula/portable-digital-ice/actions/workflows/tests.yml)

Portable Digital ICE is an independent, from-scratch reimplementation of the
Nikon LS-5000 selector-8 Digital ICE Normal processing path. It reproduced
Nikon's logical 16-bit RGB output exactly on two different 3,946 x 5,782
physical frames: 68,447,316 samples per frame with zero mismatches. The second
frame used the frozen implementation and profile without retuning.

This repository contains the portable processing core, synthetic tests, and
sanitized validation receipts. It does not contain Nikon software, firmware,
scanner binaries, personal scans, or color-inversion work.

## What Digital ICE does

A compatible film scanner records four channels: red, green, blue, and
infrared. The dyes in ordinary color negative film transmit much of the
infrared light. Dust and scratches do not, so the infrared channel gives the
software a second view of surface damage that is partly independent of the
photograph.

The hard part is deciding what is really a defect, then repairing it without
flattening grain or copying an edge into the wrong place. Nikon's implementation
does much more than threshold the infrared plane and call an inpainting tool.
It uses a low-resolution prepass, content-derived calibration, several local
reconstruction scales, deterministic conditional dither, and a block scheduler
with unusual edge behavior.

Portable Digital ICE recreates that processing without loading Nikon code at
runtime. The Nikon result appears only in the validation campaign as a read-only
comparison oracle.

## The result

| Gate | Frame 1 | Independent frame 2 |
|---|---:|---:|
| Image shape | 5,782 x 3,946 x 3 | 5,782 x 3,946 x 3 |
| RGB16 samples compared | 68,447,316 | 68,447,316 |
| Mismatched samples | 0 | 0 |
| Mismatched pixels | 0 | 0 |
| Maximum absolute delta | 0 | 0 |
| Changed-pixel mask agreement | 6,426,156 / 6,426,156 | 6,718,151 / 6,718,151 |
| RNG advances checked | 34,596,507 | 36,383,248 |
| Unbound edge fallbacks | 0 | 0 |
| Receipt checks | 25 / 25 | 25 / 25 |
| Frozen code used without retuning | baseline | yes |

The second frame was not a shifted or renamed copy. Registered high-pass
correlation with frame 1 was 0.003436. A known repeat scan of frame 1 measured
0.617244 under the same check.

The current Apple Silicon Python and NumPy reference took 71 minutes 54 seconds
for the first full-frame gate and 48 minutes 54 seconds for the second. Those
times describe an intentionally conservative research implementation, not a
finished CPU backend.

The public receipts live in [`evidence/`](evidence/). They bind the original
frozen closure source manifest, not the later namespace-renamed runtime under
`src/portable_digital_ice/`. See [`DERIVATION.md`](DERIVATION.md) before using
them as validation evidence. The validation method and claim boundary are
documented in [`docs/validation.md`](docs/validation.md).

## What had to be recovered

The final implementation includes:

- the relationship between the 285 dpi prepass and the 4000 dpi main RGBI scan;
- the 16-bit logarithmic input response and four-lane working representation;
- the infrared auxiliary signal and content-derived frame calibration;
- the score, decision history, and resolution-dependent radius;
- the multiscale local reconstruction candidates and their exact accumulation
  order;
- Nikon's 24-bit linear congruential random number generator and the conditional
  dither schedule;
- the output response tables, integer conversion, and little-endian RGB16
  scatter; and
- the streaming scheduler, seams, startup state, and partial final block.

A one-pixel change can disturb the RNG chain millions of calls later. A wrong
neighborhood radius can look plausible for hundreds of rows. The last block in
the native frame has only six valid rows. These details are why visual
similarity was never accepted as the gate.

## How the work was done

The research used hash-pinned binaries as evidence, but keeps those files
private. Static analysis established the broad structure, constants, object
state, and call boundaries. Narrow runtime recorders then captured inputs and
outputs at specific functions without changing the data flowing through them.

Controlled perturbations did the rest. Individual channels, rows, pixels, and
state fields were changed one at a time to determine which output behavior
moved with them. Candidate rules were implemented independently and compared
against held-out output. Every ambiguous case stayed open until a test could
separate the competing explanations.

The complete-frame gate also checks source hashes before and after execution,
input immutability, file-role separation, output persistence, RNG arithmetic,
the changed-pixel mask, and bottom-edge handling. An independent verifier does
not import the portable package or the gate runner.

[`docs/reverse-engineering.md`](docs/reverse-engineering.md) has the longer
account, including the radius-4 failure, startup-state correction, and the
independent-frame promotion gate.

## Supported boundary

The reverse-engineering evidence covers:

- Nikon Super Coolscan 5000 ED;
- Nikon's internal selector-8 X3A path;
- Digital ICE Normal;
- the observed internal resolution metrics 500 and 4000;
- complete native 4000 dpi processing on two different mounted C-41 frames; and
- exact CPU execution with Python and NumPy; and
- exact CUDA execution on an NVIDIA RTX A4000.

The packaged runtime currently enables only the metric-4000 profile used by the
two complete native-frame gates. It fails closed outside that profile. It does
not claim support for other scanners, other Digital ICE modes, unobserved resolution
metrics, or every possible film and defect distribution. Traditional
silver-based black-and-white film and some Kodachrome material can be unsuitable
for infrared repair because the image itself blocks infrared light.

This code repairs RGB pixels from an infrared defect signal. It does not invert
negatives, reproduce Nikon's color rendering, or include any of the separate
color research.

## Install and test

The reference package requires Python 3.11 or newer and NumPy.

```sh
git clone https://github.com/rohanpandula/portable-digital-ice.git
cd portable-digital-ice
python -m pip install -e '.[dev]'
pytest
```

The runtime accepts two immutable 16-bit RGBI arrays representing the same
physical frame: a 285 dpi prepass and a 4000 dpi main capture. Focus, exposure,
frame position, and crop geometry must remain fixed between acquisitions.
Unsupported scanner, mode, depth, geometry, or profile combinations raise an
error before output is written.

The scanner acquisition adapter and end-user TIFF workflow are still product
work. See [`docs/input-contract.md`](docs/input-contract.md) for the library
boundary.

## Performance and backends

The checked-in CPU path is the exact reference. An optional deterministic
CUDA backend ships behind `ComputeBackend.AUTO | CPU | CUDA`, validated by
binding receipts on both complete native 4000 dpi frames: identical RGB16
output to this package's CPU reference, compared sample by sample
(68,447,316 values per frame, zero mismatches), with identical changed-pixel
accounting, RNG advance counts, final RNG states, startup receipts, and the
full synthetic adversarial suite. Both receipts bind the same fresh source
manifest of this tree. On an NVIDIA RTX A4000 a complete frame takes about
21 seconds versus roughly an hour for the reference run, with deterministic
repeated-run output. `cuda` fails closed with a specific reason when
unusable; `auto` selects CUDA only after a startup self-test passes byte
parity. See [`docs/cuda-backend.md`](docs/cuda-backend.md) and the receipts
under [`evidence/`](evidence/).

A Metal backend for Apple Silicon remains planned under the same rule:
availability and speed are not parity evidence, and no backend is labeled
exact before its complete output matches the reference byte for byte.

The NegPy integration is being developed separately so this repository remains
a small, scanner-focused engine rather than an application fork.

## Repository map

| Path | Contents |
|---|---|
| `src/portable_digital_ice/` | Independent runtime and fail-closed LS-5000 profile |
| `tests/` | Redistributable synthetic and contract tests |
| `evidence/` | Sanitized complete-frame receipts |
| `docs/reverse-engineering.md` | Research method and the hard parts of the recovery |
| `docs/validation.md` | Exact gates, receipt semantics, and limits |
| `docs/input-contract.md` | Dual-RGBI acquisition and API requirements |

## License and names

The original code in this repository is available under the MIT
License. See [`LICENSE`](LICENSE).

Nikon, Nikon Scan, COOLSCAN, and Digital ICE belong to their respective owners.
This is an independent interoperability project. It is not affiliated with,
endorsed by, or supported by Nikon or the owners of Digital ICE.
