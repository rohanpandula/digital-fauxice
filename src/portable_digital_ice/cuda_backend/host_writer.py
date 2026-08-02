"""Host CPU adapter for the sequential conditional-dither writer chain.

The writer chain is sequential by necessity (whether a channel consumes one
or two draws depends on the first drawn value), so it cannot be parallelized
across the device.  A single host CPU core running the compiled reference
port (:mod:`..fast_cpu.kernels`) executes the exact recovered draw/redraw/
floor schedule about an order of magnitude faster than one GPU thread.

This module is a thin layout adapter, not a second implementation: it passes
the CUDA pipeline's compacted per-selected-site arrays (``attempted`` /
``candidate``, produced by ``k_features_and_combine`` and downloaded here as
raw bytes -- bit preservation is guaranteed by memcpy) to
``fast_cpu.kernels.write_selected``.  That serial kernel walks the same
strictly ascending pixel order as ``write_band`` without materializing dense
frame-sized attempted, candidate, or output arrays.  The writer arithmetic
itself -- every draw, redraw, and floor decision -- remains the proven
``write_pixel_scalar`` path already validated byte-exact against the same
reference this backend targets.

Importing this module never requires numba; only :func:`ensure_available`
and :func:`run_writer_chain` touch it, and both fail closed with a specific
reason when it cannot be imported.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


def ensure_available() -> Any:
    """Return the compiled fast_cpu kernel module or fail closed.

    The CUDA backend's writer chain now runs on the host through the same
    compiled path as the ``cpu-fast`` backend, so CUDA availability gains a
    dependency on numba (and on that module's own baked-RNG-constant canary)
    in addition to cupy and a visible device.
    """

    from ..fast_cpu.engine import CpuFastUnavailable, _kernels

    try:
        return _kernels()
    except CpuFastUnavailable as error:
        from .engine import CudaBackendUnavailable

        raise CudaBackendUnavailable(
            "CUDA writer chain requires the compiled host writer: "
            f"{error}"
        ) from error


def run_writer_chain(
    kernels: Any,
    *,
    selected: npt.NDArray[np.int64],
    attempted: npt.NDArray[np.uint8],
    candidate: npt.NDArray[np.float64],
    working_all: npt.NDArray[np.float32],
    floor_enabled_rows: npt.NDArray[np.uint8],
    width: int,
    state_in: int,
    low64: float,
    high64: float,
    low_lt_high: bool,
    dither_scales: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8], int, int]:
    """Run the proven serial writer over the CUDA pipeline's selected sites.

    ``selected`` holds the pixel indices (row-major ``y * width + x``) of
    every eligible site in strictly ascending order -- the same enumeration
    the deleted ``k_writer_chain`` walked as ``selected[i]``.  ``attempted``
    and ``candidate`` are :func:`k_features_and_combine`'s per-site outputs,
    unchanged.  ``working_all`` is the full ``(height, width, 4)`` converted
    plane; ``write_selected`` derives each site's "original" RGB directly from
    it, exactly as the device writer read ``original[i] == working[pixel]``
    (``working`` is never mutated between ``k_convert_and_auxiliary`` and the
    writer chain, on device or here).

    Sites that are selected-but-not-attempted (row-gated or floor-forced
    fallback) retain their default ``working_all`` value and consume no draw.
    Ineligible sites are absent from ``selected`` and never needed by the
    later device scatter.  Because ``selected`` is strictly ascending, the
    attempted subsequence reaches ``write_pixel_scalar`` in the same row-major
    order as the dense reference writer and reproduces its RNG stream exactly.

    Returns ``(values_at_selected, written_at_selected, advances,
    final_state)`` sized and typed to drop straight into the unchanged
    ``k_scatter_values`` / ``k_site_counters`` device launches.
    """

    image_width = working_all.shape[1]
    selected_count = int(selected.shape[0])
    if width != image_width:
        raise ValueError("writer width must equal the working plane width")

    values_at_selected = np.empty((selected_count, 3), dtype=np.float32)
    written_at_selected = np.empty(selected_count, dtype=np.uint8)
    total_advances, final_state = kernels.write_selected(
        selected,
        attempted,
        candidate,
        working_all,
        width,
        floor_enabled_rows,
        low64,
        high64,
        low_lt_high,
        dither_scales,
        int(state_in),
        values_at_selected,
        written_at_selected,
    )

    return (
        values_at_selected,
        written_at_selected,
        int(total_advances),
        int(final_state),
    )


__all__ = ["ensure_available", "run_writer_chain"]
