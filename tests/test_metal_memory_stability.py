"""Metal working-set headroom must not shrink frame over frame.

Regression test for a leak where every buffer :class:`.metal_backend.engine._Session`
allocated (``newBufferWithLength:options:``) - and the per-session command
queue (``newCommandQueue``) - stayed permanently retained: pyobjc was not
balancing the ownership transfer for Metal's ``new*``-family selectors, so
Python garbage collection alone never freed them.  ``MTLDevice.currentAllocatedSize()``
grew by roughly one frame's worth of buffers on every call to
:func:`process_metal` (dominated by the RGBI input buffer at full frame
size), eating into ``recommendedMaxWorkingSetSize()`` headroom until later
frames in the same process failed closed with ``MetalBackendUnavailable:
insufficient Metal working-set headroom`` even though each frame's peak
usage on its own fit comfortably.  The fix makes ``_Session`` track and
explicitly ``.release()`` every object it creates once a job is done with
it (success, cancellation, or error alike); see ``_Session.release`` for
the mechanism and the probe that isolated it.

This suite runs the small deterministic self-test job - the same fixture
:func:`portable_digital_ice.backend._synthetic_self_test_job` uses for the
AUTO backend's own byte-exact self-test - repeatedly in one process and
asserts headroom does not trend downward.  It intentionally does not
process full-size frames: the leak is per-buffer-object, not size-dependent,
so a tiny job reproduces it in milliseconds instead of the ~8s/frame a real
5,782x3,946 frame costs, and it exercises both leak sites in one call
(``process_metal`` drives both the replay session in
``metal_backend/engine.py`` and the separate producer-schedule session in
``metal_backend/producer.py``).
"""

from __future__ import annotations

import pytest

pytest.importorskip("Metal", reason="Metal backend requires pyobjc-framework-Metal")

from portable_digital_ice.backend import _synthetic_self_test_job  # noqa: E402
from portable_digital_ice.metal_backend.engine import (  # noqa: E402
    MetalBackendUnavailable,
    metal_device_summary,
)
from portable_digital_ice.metal_backend.process import process_metal  # noqa: E402

FRAME_COUNT = 12  # matches the live batch size that first surfaced the leak
# One frame's worth of small-job buffers rounds to well under this; the
# original bug leaked roughly a fixed ~1 MiB per frame on this fixture, so
# this threshold is comfortably below a real regression and comfortably
# above allocator/rounding noise.
MAX_TOLERATED_DRIFT_BYTES = 256 * 1024


def _require_device() -> None:
    try:
        metal_device_summary()
    except MetalBackendUnavailable as error:
        pytest.skip(f"Metal device unavailable: {error}")


def test_repeated_frames_do_not_shrink_working_set_headroom() -> None:
    _require_device()
    job = _synthetic_self_test_job()

    # Warm up: first call compiles kernels and populates the process-lifetime
    # library/pipeline caches, which are one-time costs and must not be
    # mistaken for a per-frame leak.
    process_metal(job, export_diagnostics=True)

    def headroom() -> int:
        summary = metal_device_summary()
        return (
            summary["recommended_max_working_set_bytes"]
            - summary["current_allocated_bytes"]
        )

    baseline = headroom()
    samples = [baseline]
    for _ in range(FRAME_COUNT):
        process_metal(job, export_diagnostics=True)
        samples.append(headroom())

    drift = baseline - samples[-1]
    assert drift <= MAX_TOLERATED_DRIFT_BYTES, (
        f"Metal working-set headroom fell by {drift} bytes over "
        f"{FRAME_COUNT} frames in one process (samples={samples}); a "
        "per-frame Metal buffer or command-queue leak has reappeared - "
        "see _Session.release in metal_backend/engine.py"
    )


def test_repeated_frames_stay_byte_identical() -> None:
    """The release-every-frame fix must not disturb the pipeline's output.

    Same job, same deterministic RNG seed every call (no generator is
    threaded across calls), so every run must reproduce the same output
    hash.  This would catch a broken fix that released a buffer while a
    downstream numpy view still aliased its memory.
    """

    _require_device()
    job = _synthetic_self_test_job()

    first = process_metal(job, export_diagnostics=True)
    for _ in range(4):
        again = process_metal(job, export_diagnostics=True)
        assert again.replay.output_sha256 == first.replay.output_sha256
