"""Direct differential coverage for the compact host writer adapter."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("numba", reason="compiled host writer requires numba")

from portable_digital_ice.cuda_backend.host_writer import (  # noqa: E402
    ensure_available,
    run_writer_chain,
)


def _dense_reference(
    kernels,
    *,
    selected: np.ndarray,
    attempted: np.ndarray,
    candidate: np.ndarray,
    working_all: np.ndarray,
    floor_enabled_rows: np.ndarray,
    state_in: int,
    low64: float,
    high64: float,
    dither_scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    height, width = working_all.shape[:2]
    total_sites = height * width
    dense_attempted = np.zeros(total_sites, dtype=np.uint8)
    dense_candidate = np.zeros((total_sites, 3), dtype=np.float64)
    if selected.size:
        dense_attempted[selected] = attempted
        dense_candidate[selected] = candidate

    dense_values = np.empty((height, width, 3), dtype=np.float32)
    row_written = np.empty(height, dtype=np.int64)
    row_advances = np.empty(height, dtype=np.int64)
    final_state = kernels.write_band(
        dense_attempted.reshape(height, width),
        dense_candidate.reshape(height, width, 3),
        working_all,
        0,
        height,
        width,
        floor_enabled_rows,
        low64,
        high64,
        low64 < high64,
        dither_scales,
        state_in,
        dense_values,
        row_written,
        row_advances,
    )

    flat_original = working_all.reshape(total_sites, 4)[:, :3]
    values = dense_values.reshape(total_sites, 3)[selected]
    written = np.any(values != flat_original[selected], axis=1).astype(np.uint8)
    return values, written, int(row_advances.sum()), int(final_state)


@pytest.mark.parametrize(
    "case",
    [
        "empty",
        "none-attempted",
        "mixed-rows",
        "invalid-candidates",
        "all-sites",
    ],
)
def test_compact_writer_matches_dense_reference(case: str) -> None:
    kernels = ensure_available()
    rng = np.random.default_rng(0xD1CE + len(case))
    height, width = 6, 8
    total_sites = height * width
    working_all = rng.uniform(0.15, 0.95, size=(height, width, 4)).astype(
        np.float32
    )
    floor_enabled_rows = np.asarray([0, 1, 0, 1, 1, 0], dtype=np.uint8)

    if case == "empty":
        selected = np.empty(0, dtype=np.int64)
        attempted = np.empty(0, dtype=np.uint8)
    elif case == "none-attempted":
        selected = np.arange(total_sites, dtype=np.int64)
        attempted = np.zeros(total_sites, dtype=np.uint8)
    elif case == "mixed-rows":
        selected = np.asarray([0, 2, 7, 8, 15, 16, 22, 31, 40, 47], dtype=np.int64)
        attempted = np.asarray([1, 0, 1, 1, 0, 1, 1, 0, 1, 1], dtype=np.uint8)
    elif case == "invalid-candidates":
        selected = np.asarray([1, 9, 17, 25, 33, 41], dtype=np.int64)
        attempted = np.ones(selected.size, dtype=np.uint8)
    else:
        selected = np.arange(total_sites, dtype=np.int64)
        attempted = (np.arange(total_sites) % 3 != 0).astype(np.uint8)

    candidate = rng.uniform(0.2, 0.9, size=(selected.size, 3)).astype(np.float64)
    if case == "invalid-candidates":
        candidate[0, 0] = np.nan
        candidate[1, 1] = np.inf
        candidate[2, 2] = -np.inf
        candidate[3, 0] = -1.0
        flat_working = working_all.reshape(total_sites, 4)
        flat_working[selected[0], 0] = np.nan
        flat_working[selected[1], 1] = np.inf
        flat_working[selected[2], 2] = -np.inf
        flat_working[selected[3], 0] = np.float32(-0.0)

    arguments = dict(
        selected=selected,
        attempted=attempted,
        candidate=candidate,
        working_all=working_all,
        floor_enabled_rows=floor_enabled_rows,
        state_in=0x4A3B2C,
        low64=0.1,
        high64=1.0,
        dither_scales=np.asarray([0.015, 0.015, 0.025], dtype=np.float32),
    )
    dense = _dense_reference(kernels, **arguments)
    compact = run_writer_chain(
        kernels,
        width=width,
        low_lt_high=True,
        **arguments,
    )

    np.testing.assert_array_equal(
        compact[0].view(np.uint32), dense[0].view(np.uint32)
    )
    np.testing.assert_array_equal(compact[1], dense[1])
    assert compact[2:] == dense[2:]


def test_compact_writer_rejects_mismatched_width() -> None:
    kernels = ensure_available()
    with pytest.raises(ValueError, match="writer width"):
        run_writer_chain(
            kernels,
            selected=np.empty(0, dtype=np.int64),
            attempted=np.empty(0, dtype=np.uint8),
            candidate=np.empty((0, 3), dtype=np.float64),
            working_all=np.zeros((4, 8, 4), dtype=np.float32),
            floor_enabled_rows=np.zeros(4, dtype=np.uint8),
            width=7,
            state_in=0,
            low64=0.1,
            high64=1.0,
            low_lt_high=True,
            dither_scales=np.zeros(3, dtype=np.float32),
        )
