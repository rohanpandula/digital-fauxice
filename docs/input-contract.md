# Input contract

Portable Digital ICE consumes two RGBI acquisitions of the same physical film
frame.

## Required captures

1. A 285 dpi, 16-bit RGBI prepass.
2. A 4000 dpi, 16-bit RGBI main scan.

Both arrays use interleaved `uint16` samples in red, green, blue, infrared order.
The scanner must keep frame position, focus, per-channel exposure, crop, and
orientation fixed between the two captures.

The current exact profile accepts the Nikon LS-5000 selector-8 Digital ICE
Normal path. The library rejects unknown scanner models, modes, metrics, sample
depths, lane counts, and incompatible geometry before it writes output.

## Why one RGBI file is not enough

The prepass supplies frame-dependent calibration consumed by the main path. It
is not a downsampled convenience image. Reconstructing or guessing it from the
main scan would move outside the byte-exact claim.

This is why the NegPy integration belongs in scanner acquisition rather than in
its existing post-scan infrared repair tool. A file-only workflow can continue
using NegPy's current IR correction, but it cannot claim this exact path without
the paired acquisition.

## Output

The reference returns an interleaved `uint16` RGB array with the main scan's
logical height and width. Color inversion and color management happen outside
this library.

## Library example

The scanner adapter is responsible for supplying `prepass_rgbi` and
`main_rgbi` as NumPy `uint16` arrays with shape `height x width x 4`.

```python
from portable_digital_ice import (
    AcquisitionEpoch,
    ComputeBackend,
    DualRGBIAcquisition,
    ProcessingJob,
    ProcessingMode,
    RGBI16Frame,
    ScannerModel,
    process,
)

prepass = RGBI16Frame(
    prepass_rgbi,
    AcquisitionEpoch.PREPASS,
    285,
    "roll-12-frame-08-prepass",
)
main = RGBI16Frame(
    main_rgbi,
    AcquisitionEpoch.MAIN,
    4000,
    "roll-12-frame-08-main",
)
job = ProcessingJob(
    acquisition=DualRGBIAcquisition(prepass, main, "roll-12-frame-08"),
    scanner_model=ScannerModel.NIKON_SUPER_COOLSCAN_5000_ED,
    mode=ProcessingMode.NORMAL,
    selector=8,
    resolution_metric=4000,
    bit_depth=16,
    focus_exposure_locked=True,
)

routed = process(job, backend=ComputeBackend.CPU)
clean_rgb16 = routed.result.output_rgb16
```

The frame wrappers copy and freeze their input arrays. A caller-owned output
array is optional. It is only updated after the complete operation succeeds.

## Safety behavior

The API validates the complete job before processing. Unsupported inputs raise
an exception. The CPU reference and optional CUDA backend are exact for the
supported profile. An explicit CUDA request fails closed if the device or
runtime is unavailable, while `AUTO` selects CUDA only after a startup parity
self-test. No Metal backend is currently implemented.
