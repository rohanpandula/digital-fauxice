# Dust heal comparison

This page backs the README section on grain-preserving repair. It fixes one
infrared defect mask and runs several repair methods over it, so the only thing
being measured is what each method does to the pixels next to a speck. Removing
the dust is not the hard part; leaving the surrounding grain alone is.

## The line worth drawing

On film, the dust and the grain are the same scale, so a repair that guesses at
a patch from its surroundings tends to average the grain away. The reader sees
it as a smooth ring around every cleaned spot, the plasticky tell of an
inpainter. A repair that copies a real clean neighbour from the side facing away
from the defect, and writes only where the mask says there is a defect, leaves
the rest of the frame exactly as it was. The exact engine in this package sits
on the copy side of that line by construction, and the hybrid sits on it too
because it changes nothing outside its disclosed mask. I wanted to see how big
the gap actually is when an inpainter is pointed at the same mask.

## Synthetic scan, shared mask

The seeded synthetic RGBI scan carries 30 round dust spots and two hair
scratches with a known ground truth, so both recall and collateral are
measurable. Each row below starts from the same IR-derived mask and differs
only in the heal step. The reflection-copy and luminance rows are NegPy's
guarded reflection-copy heal, with and without its luminance detector. The
inpaint row is OpenCV's Navier-Stokes fill standing in for a generative tier.

| Method | Clean-pixel PSNR | Inside-defect MAE |
|---|---:|---:|
| Unhealed | (baseline) | 0.2095 |
| Reflection copy | 99.0 dB | 0.2095 |
| Navier-Stokes inpaint | 52.9 dB | 0.2077 |
| NegPy luminance plus IR | 99.0 dB | 0.2095 |

Clean-pixel PSNR is taken over everything outside a three-pixel dilation of the
mask, so a perfect 99 dB means the output is bit-identical to the input there:
zero halo by construction. The inpainter matches the copier on defect recall
yet sits at 52.9 dB because its padded solve diffuses into the clean rim. That
is the whole contest on a synthetic plate with Gaussian stand-in grain.

![Four-panel dust heal comparison on the seeded synthetic scan: raw, reflection-copy, Navier-Stokes inpaint, and NegPy luminance plus IR, with zoomed defect spots and the per-method collateral table](media/dust-heal-synthetic-comparison.png)

## Real crop, scores only

The same hold-the-mask test on an interior 1600-pixel crop of a real LS-5000
frame, clean pixels only:

| Method | Clean-pixel PSNR | Clean residual RMS | Heal time |
|---|---:|---:|---:|
| Reflection copy | 99.0 dB | 0.0000 | 0.72 s |
| Navier-Stokes inpaint | 52.9 dB | 0.0023 | 0.07 s |
| NegPy luminance plus IR | 99.0 dB | 0.0000 | 0.61 s |

The pixels in that crop are a personal scan, so per this repository's rule they
do not ship here; only the aggregate numbers do, exactly as the per-frame
validation stats are quoted without their source frames. The synthetic montage
above has no personal data, which is why it can be published.

## What it means for this package

Two tiers, not one cleverer algorithm. The grain-preserving pass handles the
ordinary dust and hair that make up nearly the whole mask; both an exact
reconstruction and a reflection copy qualify, and neither writes outside its
mask. A generative fill is right only on the saturated tears where there is no
clean neighbour to copy, which is the case the hybrid already routes, dilates by
four pixels, and refuses past 2% of the frame. The instinct to reach for
inpainting on the easy 99% of the mask is the part the numbers push against. The
luminance z-score detector adds nothing over an IR mask on a scanner frame that
already has an IR channel; it is the play for infrared-free camera scans.

## Reproducing it

The synthetic generator is a short standalone script kept in the author's
research tree and shared on request. It cannot live inside this repository for a
licensing reason: the reflection-copy and luminance paths call NegPy at runtime,
and NegPy is GPL-3.0, so embedding the kernel here would sit oddly against this
package's MIT "original code" statement. Reproducing the figure needs NumPy,
SciPy, OpenCV, matplotlib, and tifffile for the scoring and the montage, plus a
checkout of NegPy on the Python path for the two copy-based rows. The synthetic
scan is seeded (RNG 41), so the defected RGBI plate and its ground-truth mask
regenerate bit-for-bit.

## Credit

The guarded reflection-copy heal measured here is from
[NegPy](https://github.com/marcinz606/NegPy) by Marcin Zawalski, GPL-3.0. It is
the cleanest copy-instead-of-fill implementation I found, and it shaped both the
comparison and the framing of this page. NegPy is a comparator and a reference
technique, not a runtime dependency; no NegPy code is bundled in this package.

## Limits

This is one synthetic scan and one real crop scored on clean pixels, not a
blind perceptual study or a multi-roll sweep, and the inserts are small to
medium dust. A heavily scratched frame with wide gouges is exactly where a
generative model's wider context can genuinely help, and that case is not
benchmarked here because it is the hybrid's job, with the failure closed and the
invented pixels disclosed. Hardware IR detection also still fails on IR-opaque
emulsions such as traditional silver black-and-white and some Kodachrome.
