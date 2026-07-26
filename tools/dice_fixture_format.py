"""Strict readers for the private DICEIN1 + DICEPP1 validation fixture pair.

The complete-frame parity receipts under ``evidence/`` are minted from two
private scanner captures.  Those captures are not redistributable and never
enter this repository (see ``.gitignore``), but the *container* they are
stored in is this project's own format, so its reader lives here: without it
nobody holding the fixtures could regenerate or audit a receipt from the
repository alone.

Two files describe one validation frame:

``<case>.dicein1``
    The main pass.  A 256-byte header, an Init record and its result, then
    the allocated RGBI16 payload laid out as ``block_count`` blocks of
    ``rows_per_block`` rows.  Only the first ``height`` rows are logical;
    the final block is padded.  These files exceed 170 MiB at the scanner's
    native 4000 dpi, so the payload is exposed through a read-only memory
    map and the file is hashed by bounded streaming.

``<case>.dicepp1``
    The prepass.  Same header discipline plus a variant table; each variant
    is one complete alternative prepass reduction.  The complete-frame gates
    use the ``native-exact`` variant.  Small enough to parse in memory.

Every structural claim a header makes is checked against the bytes that
follow it, and both files carry payload hashes that are verified on load.
A malformed, truncated, or substituted fixture raises ``ValueError`` here
rather than producing a partial frame downstream.

This module deliberately depends on nothing but the standard library and
numpy so that it can be read and audited on its own.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1
HEADER_BYTES = 256
MAX_FIXTURE_BYTES = 16 * 1024 * 1024
MAX_MAIN_FIXTURE_BYTES = 1024 * 1024 * 1024
MAX_MAIN_BLOCKS = 4096
MAX_PREPASS_BLOCKS = 16
MAX_VARIANTS = 16
HASH_CHUNK_BYTES = 4 * 1024 * 1024
MAIN_HEADER = struct.Struct("<8sIIIi" + "I" * 19 + "32s32s32s60s")
PREPASS_HEADER = struct.Struct("<8sIIIi" + "I" * 18 + "32s32s32s64s")
VARIANT_ENTRY = struct.Struct("<32sIIIIII32s8s")
assert MAIN_HEADER.size == HEADER_BYTES
assert PREPASS_HEADER.size == HEADER_BYTES
assert VARIANT_ENTRY.size == 96


def sha256_file(path: Path) -> str:
    """Hash a file of any size by bounded streaming."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_rgbi16(pixels: np.ndarray) -> str:
    """Hash logical RGBI16 pixels as little-endian C-order bytes, row by row.

    Streaming per row keeps a 170 MiB memory-mapped main frame off the heap
    and produces exactly the same digest as hashing the whole buffer.
    """

    digest = hashlib.sha256()
    for row in pixels:
        digest.update(row.astype("<u2", copy=False).tobytes(order="C"))
    return digest.hexdigest()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_bounded(path: Path) -> bytes:
    size = path.stat().st_size
    if not HEADER_BYTES <= size <= MAX_FIXTURE_BYTES:
        raise ValueError(f"{path} is outside the bounded fixture size")
    data = path.read_bytes()
    if len(data) != size:
        raise ValueError(f"{path} changed while being read")
    return data


def _stream_main_hashes(path: Path) -> tuple[bytes, bytes, int]:
    """Hash a large main fixture and its post-header payload in one pass."""

    before = path.stat()
    if not HEADER_BYTES <= before.st_size <= MAX_MAIN_FIXTURE_BYTES:
        raise ValueError(f"{path} is outside the bounded main-fixture size")
    whole = hashlib.sha256()
    payload = hashlib.sha256()
    position = 0
    with path.open("rb") as stream:
        while chunk := stream.read(HASH_CHUNK_BYTES):
            whole.update(chunk)
            chunk_end = position + len(chunk)
            if chunk_end > HEADER_BYTES:
                start = max(0, HEADER_BYTES - position)
                payload.update(chunk[start:])
            position = chunk_end
    after = path.stat()
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity or position != before.st_size:
        raise ValueError(f"{path} changed while being read")
    return whole.digest(), payload.digest(), position


def _read_exact_region(path: Path, offset: int, size: int) -> bytes:
    with path.open("rb") as stream:
        stream.seek(offset)
        data = stream.read(size)
    if len(data) != size:
        raise ValueError(f"{path} ended inside a declared region")
    return data


def _validate_region(
    name: str,
    offset: int,
    size: int,
    *,
    total: int,
    previous_end: int,
) -> int:
    if offset < previous_end or offset % 16 or size <= 0 or offset + size > total:
        raise ValueError(f"{name} region is overlapping, unaligned, or out of bounds")
    return offset + size


def _logical_interleaved(
    payload: bytes,
    *,
    block_count: int,
    rows_per_block: int,
    final_valid_rows: int,
    height: int,
    width: int,
    lanes: int,
) -> np.ndarray:
    allocated = np.frombuffer(payload, dtype="<u2").reshape(
        block_count, rows_per_block, width, lanes
    )
    valid = np.concatenate(
        [
            allocated[
                index,
                : final_valid_rows if index == block_count - 1 else rows_per_block,
            ]
            for index in range(block_count)
        ],
        axis=0,
    )
    if valid.shape != (height, width, lanes):
        raise AssertionError("validated fixture geometry produced an invalid image")
    return np.ascontiguousarray(valid, dtype=np.uint16)


@dataclass(frozen=True)
class MainGeneratorInput:
    """One validated DICEIN1 main pass."""

    path: Path
    file_sha256: str
    selector: int
    block_count: int
    width: int
    height: int
    rows_per_block: int
    final_valid_rows: int
    begin_extent_0: int
    begin_extent_1: int
    input_offset: int
    input_bytes: int
    strict_report_sha256: str
    evidence_set_sha256: str
    payload_sha256: str

    def allocated_pixels_mmap(self) -> np.memmap:
        """Return the allocated block payload as a read-only RGBI16 mmap."""

        pixels = np.memmap(
            self.path,
            dtype="<u2",
            mode="r",
            offset=self.input_offset,
            shape=(self.block_count, self.rows_per_block, self.width, 4),
            order="C",
        )
        pixels.flags.writeable = False
        return pixels

    def logical_pixels_mmap(self) -> np.ndarray:
        """Return valid logical rows without copying final-block padding."""

        allocated = self.allocated_pixels_mmap()
        logical = allocated.reshape(
            self.block_count * self.rows_per_block, self.width, 4
        )[: self.height]
        logical.flags.writeable = False
        return logical


def parse_main_generator_input(path: Path) -> MainGeneratorInput:
    """Validate a DICEIN1 main fixture end to end, or raise ``ValueError``."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"main fixture does not exist: {path}")
    file_hash, suffix_hash, file_size = _stream_main_hashes(path)
    header = _read_exact_region(path, 0, HEADER_BYTES)
    (
        magic,
        version,
        header_bytes,
        total_bytes,
        selector,
        block_count,
        width,
        height,
        rows_per_block,
        final_valid_rows,
        extent_0,
        extent_1,
        input_row_bytes,
        input_block_bytes,
        output_row_bytes,
        output_block_bytes,
        init_offset,
        init_bytes,
        result_offset,
        result_bytes,
        input_offset,
        input_bytes,
        oracle_offset,
        oracle_bytes,
        strict_hash,
        evidence_hash,
        payload_hash,
        reserved,
    ) = MAIN_HEADER.unpack_from(header)
    if magic != b"DICEIN1\0" or version != FORMAT_VERSION:
        raise ValueError("main input is not DICEIN1 version 1")
    if header_bytes != HEADER_BYTES or total_bytes != file_size:
        raise ValueError("DICEIN1 self-reported size disagrees")
    if reserved != bytes(len(reserved)):
        raise ValueError("DICEIN1 reserved bytes are nonzero")
    if selector not in (8, 9) or not 1 <= block_count <= MAX_MAIN_BLOCKS:
        raise ValueError("DICEIN1 selector/block count is invalid")
    if min(width, height, rows_per_block) <= 0:
        raise ValueError("DICEIN1 geometry contains zero")
    if not 1 <= final_valid_rows <= rows_per_block:
        raise ValueError("DICEIN1 final valid rows are invalid")
    if (block_count - 1) * rows_per_block + final_valid_rows != height:
        raise ValueError("DICEIN1 block schedule does not cover its height")
    if input_row_bytes != width * 8 or output_row_bytes != width * 6:
        raise ValueError("DICEIN1 row sizes are not RGBI16/RGB16")
    if input_block_bytes != input_row_bytes * rows_per_block:
        raise ValueError("DICEIN1 input block size disagrees")
    if output_block_bytes != output_row_bytes * rows_per_block:
        raise ValueError("DICEIN1 output block size disagrees")
    if input_bytes != input_block_bytes * block_count:
        raise ValueError("DICEIN1 input payload size disagrees")
    if init_bytes != 128 or result_bytes != 24:
        raise ValueError("DICEIN1 Init record is not 128/24 bytes")
    if oracle_offset != 0 or oracle_bytes != 0:
        raise ValueError("DICEIN1 must not embed a Nikon output oracle")
    if not any(strict_hash) or not any(evidence_hash):
        raise ValueError("DICEIN1 source hashes are missing")
    previous_end = HEADER_BYTES
    for name, offset, size in (
        ("Init", init_offset, init_bytes),
        ("Init result", result_offset, result_bytes),
        ("RGBI input", input_offset, input_bytes),
    ):
        end = _validate_region(
            name, offset, size, total=file_size, previous_end=previous_end
        )
        if any(_read_exact_region(path, previous_end, offset - previous_end)):
            raise ValueError(f"DICEIN1 padding before {name} is nonzero")
        previous_end = end
    if previous_end != file_size:
        raise ValueError("DICEIN1 has trailing data")
    if suffix_hash != payload_hash:
        raise ValueError("DICEIN1 payload hash mismatch")
    return MainGeneratorInput(
        path=path.resolve(),
        file_sha256=file_hash.hex(),
        selector=selector,
        block_count=block_count,
        width=width,
        height=height,
        rows_per_block=rows_per_block,
        final_valid_rows=final_valid_rows,
        begin_extent_0=extent_0,
        begin_extent_1=extent_1,
        input_offset=input_offset,
        input_bytes=input_bytes,
        strict_report_sha256=strict_hash.hex(),
        evidence_set_sha256=evidence_hash.hex(),
        payload_sha256=payload_hash.hex(),
    )


@dataclass(frozen=True)
class PrepassVariant:
    name: str
    method: int
    padding_policy: int
    payload: bytes
    payload_sha256: str


@dataclass(frozen=True)
class PrepassGeneratorInput:
    """One validated DICEPP1 prepass and its variant table."""

    path: Path
    file_sha256: str
    selector: int
    block_count: int
    width: int
    height: int
    rows_per_block: int
    final_valid_rows: int
    begin_extent_0: int
    begin_extent_1: int
    source_main_sha256: str
    source_init_sha256: str
    payload_sha256: str
    variants: tuple[PrepassVariant, ...]

    def variant(self, name: str) -> PrepassVariant:
        matches = [variant for variant in self.variants if variant.name == name]
        if len(matches) != 1:
            available = ", ".join(sorted(v.name for v in self.variants)) or "none"
            raise ValueError(
                f"prepass variant is not uniquely present: {name!r} "
                f"(available: {available})"
            )
        return matches[0]

    def logical_pixels(self, variant_name: str) -> np.ndarray:
        """Return one variant's logical RGBI16 rows, padding removed."""

        return _logical_interleaved(
            self.variant(variant_name).payload,
            block_count=self.block_count,
            rows_per_block=self.rows_per_block,
            final_valid_rows=self.final_valid_rows,
            height=self.height,
            width=self.width,
            lanes=4,
        )


def parse_prepass_generator_input(path: Path) -> PrepassGeneratorInput:
    """Validate a DICEPP1 prepass fixture end to end, or raise ``ValueError``."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"prepass fixture does not exist: {path}")
    data = _read_bounded(path)
    (
        magic,
        version,
        header_bytes,
        total_bytes,
        selector,
        variant_count,
        block_count,
        width,
        height,
        rows_per_block,
        final_valid_rows,
        extent_0,
        extent_1,
        input_row_bytes,
        input_block_bytes,
        init_offset,
        init_bytes,
        result_offset,
        result_bytes,
        table_offset,
        table_bytes,
        payload_offset,
        payload_bytes,
        main_hash,
        init_hash,
        aggregate_hash,
        reserved,
    ) = PREPASS_HEADER.unpack_from(data)
    if magic != b"DICEPP1\0" or version != FORMAT_VERSION:
        raise ValueError("prepass input is not DICEPP1 version 1")
    if header_bytes != HEADER_BYTES or total_bytes != len(data):
        raise ValueError("DICEPP1 self-reported size disagrees")
    if reserved != bytes(len(reserved)):
        raise ValueError("DICEPP1 reserved bytes are nonzero")
    if selector not in (8, 9):
        raise ValueError("DICEPP1 selector is invalid")
    if not 1 <= variant_count <= MAX_VARIANTS:
        raise ValueError("DICEPP1 variant count is invalid")
    if not 1 <= block_count <= MAX_PREPASS_BLOCKS:
        raise ValueError("DICEPP1 block count is invalid")
    if min(width, height, rows_per_block) <= 0:
        raise ValueError("DICEPP1 geometry contains zero")
    if not 1 <= final_valid_rows <= rows_per_block:
        raise ValueError("DICEPP1 final valid rows are invalid")
    if (block_count - 1) * rows_per_block + final_valid_rows != height:
        raise ValueError("DICEPP1 block schedule does not cover its height")
    if input_row_bytes != width * 8:
        raise ValueError("DICEPP1 row size is not RGBI16")
    if input_block_bytes != input_row_bytes * rows_per_block:
        raise ValueError("DICEPP1 block size disagrees")
    if init_bytes != 128 or result_bytes != 24:
        raise ValueError("DICEPP1 Init record is not 128/24 bytes")
    if table_bytes != variant_count * VARIANT_ENTRY.size:
        raise ValueError("DICEPP1 variant table size disagrees")
    one_variant_bytes = block_count * input_block_bytes
    if payload_bytes != variant_count * one_variant_bytes:
        raise ValueError("DICEPP1 variant payload size disagrees")
    if not any(main_hash) or not any(init_hash):
        raise ValueError("DICEPP1 source hashes are missing")
    previous_end = HEADER_BYTES
    for name, offset, size in (
        ("Init", init_offset, init_bytes),
        ("Init result", result_offset, result_bytes),
        ("variant table", table_offset, table_bytes),
        ("variant payload", payload_offset, payload_bytes),
    ):
        end = _validate_region(
            name, offset, size, total=len(data), previous_end=previous_end
        )
        if any(data[previous_end:offset]):
            raise ValueError(f"DICEPP1 padding before {name} is nonzero")
        previous_end = end
    if previous_end != len(data):
        raise ValueError("DICEPP1 has trailing data")
    if hashlib.sha256(data[HEADER_BYTES:]).digest() != aggregate_hash:
        raise ValueError("DICEPP1 aggregate payload hash mismatch")
    variants: list[PrepassVariant] = []
    seen: set[str] = set()
    for index in range(variant_count):
        entry_offset = table_offset + index * VARIANT_ENTRY.size
        (
            raw_name,
            method,
            crop_top,
            crop_bottom,
            padding_policy,
            variant_offset,
            variant_bytes,
            variant_hash,
            entry_reserved,
        ) = VARIANT_ENTRY.unpack_from(data, entry_offset)
        name_bytes = raw_name.split(b"\0", 1)[0]
        try:
            name = name_bytes.decode("ascii")
        except UnicodeDecodeError as error:
            raise ValueError("DICEPP1 variant name is not ASCII") from error
        if not name or name in seen or len(name_bytes) >= len(raw_name):
            raise ValueError("DICEPP1 variant name is invalid")
        if method not in (1, 2, 3) or padding_policy not in (1, 2):
            raise ValueError("DICEPP1 variant method or padding policy is unknown")
        if crop_top + crop_bottom >= 676:
            raise ValueError("DICEPP1 variant crop is invalid")
        expected_offset = payload_offset + index * one_variant_bytes
        if variant_offset != expected_offset or variant_bytes != one_variant_bytes:
            raise ValueError("DICEPP1 variant payload is non-canonical")
        payload = data[variant_offset : variant_offset + variant_bytes]
        if hashlib.sha256(payload).digest() != variant_hash:
            raise ValueError("DICEPP1 variant hash mismatch")
        if entry_reserved != bytes(len(entry_reserved)):
            raise ValueError("DICEPP1 variant reserved bytes are nonzero")
        variants.append(
            PrepassVariant(
                name=name,
                method=method,
                padding_policy=padding_policy,
                payload=payload,
                payload_sha256=variant_hash.hex(),
            )
        )
        seen.add(name)
    return PrepassGeneratorInput(
        path=path.resolve(),
        file_sha256=_sha256(data),
        selector=selector,
        block_count=block_count,
        width=width,
        height=height,
        rows_per_block=rows_per_block,
        final_valid_rows=final_valid_rows,
        begin_extent_0=extent_0,
        begin_extent_1=extent_1,
        source_main_sha256=main_hash.hex(),
        source_init_sha256=init_hash.hex(),
        payload_sha256=aggregate_hash.hex(),
        variants=tuple(variants),
    )


__all__ = [
    "MainGeneratorInput",
    "PrepassGeneratorInput",
    "PrepassVariant",
    "parse_main_generator_input",
    "parse_prepass_generator_input",
    "sha256_file",
    "sha256_rgbi16",
]
