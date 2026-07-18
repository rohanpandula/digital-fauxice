"""Host CPU adapter for the sequential conditional-dither writer chain.

The writer chain is sequential by necessity (whether a channel consumes one
or two draws depends on the first drawn value), so it cannot be parallelized
across the device.  A single host CPU core running the compiled reference
port (:mod:`..fast_cpu.kernels`) executes the exact recovered draw/redraw/
floor schedule about an order of magnitude faster than one GPU thread.

This module is a thin layout adapter, not a second implementation: it maps
the CUDA pipeline's compacted per-selected-site arrays (``attempted`` /
``candidate``, produced by ``k_features_and_combine`` and downloaded here as
raw bytes -- bit preservation is guaranteed by memcpy) onto the dense
per-row layout ``fast_cpu.kernels.write_band`` expects, then compacts its
dense output back to the per-site layout the unchanged device-side scatter
and counter kernels already consume.  The writer arithmetic itself -- every
draw, redraw, and floor decision -- is exactly the proven ``write_band`` /
``write_pixel_scalar`` path, already validated byte-exact against the same
reference this backend targets.  Any divergence here indicts the layout
mapping below, not the arithmetic in ``fast_cpu.kernels``.

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
    """Run the proven ``write_band`` writer over the CUDA pipeline's sites.

    ``selected`` holds the pixel indices (row-major ``y * width + x``) of
    every eligible site in strictly ascending order -- the same enumeration
    the deleted ``k_writer_chain`` walked as ``selected[i]``.  ``attempted``
    and ``candidate`` are :func:`k_features_and_combine`'s per-site outputs,
    unchanged.  ``working_all`` is the full ``(height, width, 4)`` converted
    plane; ``write_band`` derives each site's "original" RGB directly from
    it, exactly as the device writer read ``original[i] == working[pixel]``
    (``working`` is never mutated between ``k_convert_and_auxiliary`` and the
    writer chain, on device or here).

    Sites that are not selected (ineligible) or selected-but-not-attempted
    (row-gated or floor-forced fallback) both fall through to their default
    ``working_all`` value in the dense output -- the same outcome the
    original kernel produced by never touching them.  Feeding
    ``write_band`` a dense attempted mask that is zero everywhere except at
    ``selected`` reproduces the RNG draw order and count bit-for-bit,
    because both the ineligible sites (never visited by the old compacted
    loop) and the eligible-but-unattempted sites (visited but skipped
    without a draw) contribute zero draws either way; only the visited,
    attempted sites draw, in the same strict row-major order.

    Returns ``(values_at_selected, written_at_selected, advances,
    final_state)`` sized and typed to drop straight into the unchanged
    ``k_scatter_values`` / ``k_site_counters`` device launches.
    """

    height, image_width = working_all.shape[0], working_all.shape[1]
    total_sites = height * image_width
    selected_count = int(selected.shape[0])

    dense_attempted = np.zeros(total_sites, dtype=np.uint8)
    dense_candidate = np.zeros((total_sites, 3), dtype=np.float64)
    if selected_count:
        dense_attempted[selected] = attempted
        dense_candidate[selected] = candidate
    dense_attempted = dense_attempted.reshape(height, image_width)
    dense_candidate = dense_candidate.reshape(height, image_width, 3)

    out_values = np.empty((height, image_width, 3), dtype=np.float32)
    out_written = np.empty(height, dtype=np.int64)
    out_advances = np.empty(height, dtype=np.int64)

    final_state = kernels.write_band(
        dense_attempted,
        dense_candidate,
        working_all,
        0,
        height,
        width,
        floor_enabled_rows,
        low64,
        high64,
        low_lt_high,
        dither_scales,
        int(state_in),
        out_values,
        out_written,
        out_advances,
    )

    if selected_count:
        flat_values = out_values.reshape(total_sites, 3)
        flat_original = working_all.reshape(total_sites, 4)[:, :3]
        values_at_selected = flat_values[selected]
        written_at_selected = np.any(
            values_at_selected != flat_original[selected], axis=1
        ).astype(np.uint8)
    else:
        values_at_selected = np.empty((0, 3), dtype=np.float32)
        written_at_selected = np.empty((0,), dtype=np.uint8)

    total_advances = int(out_advances.sum())
    return values_at_selected, written_at_selected, total_advances, int(final_state)


__all__ = ["ensure_available", "run_writer_chain"]
