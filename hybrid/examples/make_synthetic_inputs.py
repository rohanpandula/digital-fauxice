#!/usr/bin/env python3
"""Create the deterministic, redistributable hybrid-repair example inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fauxce_hybrid.cache import canonical_json_bytes, hash_rgbir16


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser


def _frame(*, offset: int) -> np.ndarray:
    y, x = np.indices((256, 256), dtype=np.uint32)
    frame = np.empty((256, 256, 4), dtype=np.uint16)
    frame[:, :, 0] = (12_000 + offset + 31 * y + 47 * x).astype(np.uint16)
    frame[:, :, 1] = (16_000 + offset + 43 * y + 29 * x).astype(np.uint16)
    frame[:, :, 2] = (20_000 + offset + 37 * y + 41 * x).astype(np.uint16)
    frame[:, :, 3] = (45_000 + offset + 17 * y + 13 * x).astype(np.uint16)
    return frame


def main() -> int:
    args = _parser().parse_args()
    destination = args.out
    if destination.exists():
        expected = {
            "acquisition.json",
            "main.rgbir16.npy",
            "prepass.rgbir16.npy",
        }
        if not args.force or {path.name for path in destination.iterdir()} != expected:
            raise RuntimeError("refusing to overwrite a non-example directory")
    else:
        destination.mkdir(parents=True, exist_ok=False)
    prepass = _frame(offset=0)
    main = _frame(offset=120)
    main[110:130, 118:138, 3] = np.uint16(0)
    prepass_path = destination / "prepass.rgbir16.npy"
    main_path = destination / "main.rgbir16.npy"
    with prepass_path.open("wb") as handle:
        np.save(handle, prepass, allow_pickle=False)
    with main_path.open("wb") as handle:
        np.save(handle, main, allow_pickle=False)
    manifest = {
        "focus_exposure_locked": True,
        "main_raw_sha256": hash_rgbir16(main),
        "prepass_raw_sha256": hash_rgbir16(prepass),
        "same_frame_id": "synthetic-lama-example-v1",
    }
    (destination / "acquisition.json").write_bytes(canonical_json_bytes(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
