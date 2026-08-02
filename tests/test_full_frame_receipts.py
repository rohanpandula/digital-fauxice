"""Tests for the complete-frame receipt minting harness.

Three layers, only the last of which needs private data:

* the DICEIN1/DICEPP1 reader is exercised against synthetic fixtures built
  in this file, so ordinary CI proves it parses a well-formed pair and
  rejects a malformed one;
* the minting script's fail-closed legs are exercised without any fixture
  at all, so CI proves it refuses to mint rather than minting something
  empty or partial; and
* the complete-frame gate itself re-mints a receipt and compares it against
  the checked-in one, which needs the private captures and skips with a
  reason when they are absent.

Point the last layer at fixtures with ``PORTABLE_DICE_FIXTURE_MANIFEST``
(see ``tools/fixtures.example.json`` and ``docs/validation.md``)::

    PORTABLE_DICE_FIXTURE_MANIFEST=/path/to/manifest.json pytest \
        tests/test_full_frame_receipts.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TOOLS = REPOSITORY_ROOT / "tools"
EVIDENCE = REPOSITORY_ROOT / "evidence"
MANIFEST_ENVIRONMENT_VARIABLE = "PORTABLE_DICE_FIXTURE_MANIFEST"


def _load(name: str):
    """Import a tools/ script by path; tools/ is deliberately not a package."""

    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fixture_format = _load("dice_fixture_format")
mint = _load("mint_parity_receipt")


def test_receipt_import_replaces_a_stale_installed_package(tmp_path: Path) -> None:
    """The gate must execute the same checkout whose source manifest it binds."""

    fake_root = tmp_path / "fake-install"
    fake_package = fake_root / "portable_digital_ice"
    fake_package.mkdir(parents=True)
    (fake_package / "__init__.py").write_text(
        "SOURCE = 'stale-installed-copy'\n", encoding="utf-8"
    )
    script = f"""
import importlib.util
import pathlib
import sys

sys.path.insert(0, {str(fake_root)!r})
import portable_digital_ice
assert portable_digital_ice.SOURCE == 'stale-installed-copy'

spec = importlib.util.spec_from_file_location(
    'receipt_mint_isolated', {str(TOOLS / 'mint_parity_receipt.py')!r}
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
package = module._import_package()
print(pathlib.Path(package.__file__).resolve())
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert Path(completed.stdout.strip()) == (
        REPOSITORY_ROOT / "src" / "portable_digital_ice" / "__init__.py"
    ).resolve()


# --------------------------------------------------------------------------
# synthetic fixture builders (no private data)
# --------------------------------------------------------------------------

_ALIGN = 16


def _align(value: int) -> int:
    return (value + _ALIGN - 1) // _ALIGN * _ALIGN


def build_main_fixture(
    *,
    width: int = 4,
    rows_per_block: int = 3,
    block_count: int = 2,
    final_valid_rows: int = 2,
    selector: int = 8,
    seed: int = 7,
) -> tuple[bytes, np.ndarray]:
    """Build a well-formed DICEIN1 file and the logical pixels it encodes."""

    height = (block_count - 1) * rows_per_block + final_valid_rows
    input_row_bytes = width * 8
    input_block_bytes = input_row_bytes * rows_per_block
    input_bytes = input_block_bytes * block_count

    rng = np.random.default_rng(seed)
    allocated = rng.integers(
        0, 65536, size=(block_count, rows_per_block, width, 4), dtype=np.uint16
    )
    logical = allocated.reshape(block_count * rows_per_block, width, 4)[:height]

    init_offset, init_bytes = _align(fixture_format.HEADER_BYTES), 128
    result_offset = _align(init_offset + init_bytes)
    result_bytes = 24
    input_offset = _align(result_offset + result_bytes)
    total = input_offset + input_bytes

    body = bytearray(total - fixture_format.HEADER_BYTES)

    def place(offset: int, payload: bytes) -> None:
        start = offset - fixture_format.HEADER_BYTES
        body[start : start + len(payload)] = payload

    place(init_offset, bytes(range(128)))
    place(result_offset, bytes(range(24)))
    place(input_offset, allocated.astype("<u2").tobytes(order="C"))

    payload_hash = hashlib.sha256(bytes(body)).digest()
    header = fixture_format.MAIN_HEADER.pack(
        b"DICEIN1\0",
        fixture_format.FORMAT_VERSION,
        fixture_format.HEADER_BYTES,
        total,
        selector,
        block_count,
        width,
        height,
        rows_per_block,
        final_valid_rows,
        0,
        0,
        input_row_bytes,
        input_block_bytes,
        width * 6,
        width * 6 * rows_per_block,
        init_offset,
        init_bytes,
        result_offset,
        result_bytes,
        input_offset,
        input_bytes,
        0,
        0,
        b"\x11" * 32,
        b"\x22" * 32,
        payload_hash,
        bytes(60),
    )
    return bytes(header) + bytes(body), np.ascontiguousarray(logical)


def build_prepass_fixture(
    *,
    main_sha256_hex: str,
    width: int = 4,
    rows_per_block: int = 2,
    block_count: int = 2,
    final_valid_rows: int = 1,
    selector: int = 8,
    variant_name: str = "native-exact",
    seed: int = 11,
) -> tuple[bytes, np.ndarray]:
    """Build a well-formed DICEPP1 file bound to a given main fixture."""

    height = (block_count - 1) * rows_per_block + final_valid_rows
    input_row_bytes = width * 8
    input_block_bytes = input_row_bytes * rows_per_block
    one_variant_bytes = block_count * input_block_bytes

    rng = np.random.default_rng(seed)
    allocated = rng.integers(
        0, 65536, size=(block_count, rows_per_block, width, 4), dtype=np.uint16
    )
    logical = allocated.reshape(block_count * rows_per_block, width, 4)[:height]
    variant_payload = allocated.astype("<u2").tobytes(order="C")
    assert len(variant_payload) == one_variant_bytes

    init_offset, init_bytes = _align(fixture_format.HEADER_BYTES), 128
    result_offset, result_bytes = _align(init_offset + init_bytes), 24
    table_offset = _align(result_offset + result_bytes)
    table_bytes = fixture_format.VARIANT_ENTRY.size
    payload_offset = _align(table_offset + table_bytes)
    total = payload_offset + one_variant_bytes

    body = bytearray(total - fixture_format.HEADER_BYTES)

    def place(offset: int, payload: bytes) -> None:
        start = offset - fixture_format.HEADER_BYTES
        body[start : start + len(payload)] = payload

    entry = fixture_format.VARIANT_ENTRY.pack(
        variant_name.encode("ascii").ljust(32, b"\0"),
        1,
        0,
        0,
        1,
        payload_offset,
        one_variant_bytes,
        hashlib.sha256(variant_payload).digest(),
        bytes(8),
    )
    place(init_offset, bytes(range(128)))
    place(result_offset, bytes(range(24)))
    place(table_offset, entry)
    place(payload_offset, variant_payload)

    aggregate = hashlib.sha256(bytes(body)).digest()
    header = fixture_format.PREPASS_HEADER.pack(
        b"DICEPP1\0",
        fixture_format.FORMAT_VERSION,
        fixture_format.HEADER_BYTES,
        total,
        selector,
        1,
        block_count,
        width,
        height,
        rows_per_block,
        final_valid_rows,
        0,
        0,
        input_row_bytes,
        input_block_bytes,
        init_offset,
        init_bytes,
        result_offset,
        result_bytes,
        table_offset,
        table_bytes,
        payload_offset,
        one_variant_bytes,
        bytes.fromhex(main_sha256_hex),
        b"\x33" * 32,
        aggregate,
        bytes(64),
    )
    return bytes(header) + bytes(body), np.ascontiguousarray(logical)


@pytest.fixture
def synthetic_pair(tmp_path: Path) -> tuple[Path, Path, np.ndarray, np.ndarray]:
    main_bytes, main_pixels = build_main_fixture()
    main_path = tmp_path / "synthetic.dicein1"
    main_path.write_bytes(main_bytes)
    prepass_bytes, prepass_pixels = build_prepass_fixture(
        main_sha256_hex=hashlib.sha256(main_bytes).hexdigest()
    )
    prepass_path = tmp_path / "synthetic.dicepp1"
    prepass_path.write_bytes(prepass_bytes)
    return main_path, prepass_path, main_pixels, prepass_pixels


# --------------------------------------------------------------------------
# the reader parses what it should and rejects what it should
# --------------------------------------------------------------------------


def test_reader_round_trips_a_well_formed_pair(synthetic_pair) -> None:
    main_path, prepass_path, main_pixels, prepass_pixels = synthetic_pair

    main = fixture_format.parse_main_generator_input(main_path)
    assert (main.height, main.width) == main_pixels.shape[:2]
    assert main.file_sha256 == fixture_format.sha256_file(main_path)
    assert np.array_equal(main.logical_pixels_mmap(), main_pixels)

    prepass = fixture_format.parse_prepass_generator_input(prepass_path)
    assert prepass.source_main_sha256 == main.file_sha256
    assert [variant.name for variant in prepass.variants] == ["native-exact"]
    assert np.array_equal(prepass.logical_pixels("native-exact"), prepass_pixels)


def test_reader_hash_matches_whole_buffer_hash(synthetic_pair) -> None:
    """Row-streaming must digest exactly what a single tobytes() would."""

    _, _, main_pixels, _ = synthetic_pair
    whole = hashlib.sha256(
        main_pixels.astype("<u2", copy=False).tobytes(order="C")
    ).hexdigest()
    assert fixture_format.sha256_rgbi16(main_pixels) == whole


def test_reader_reports_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        fixture_format.parse_main_generator_input(tmp_path / "absent.dicein1")
    with pytest.raises(FileNotFoundError):
        fixture_format.parse_prepass_generator_input(tmp_path / "absent.dicepp1")


@pytest.mark.parametrize(
    ("description", "mutate"),
    [
        ("wrong magic", lambda data: b"NOTDICE\0" + data[8:]),
        ("trailing data", lambda data: data + b"\0" * 16),
        (
            "corrupted payload",
            lambda data: data[:-2] + bytes([data[-2] ^ 0xFF, data[-1]]),
        ),
        ("nonzero reserved", lambda data: data[:250] + b"\x01" + data[251:]),
    ],
)
def test_reader_rejects_a_damaged_main_fixture(
    tmp_path: Path, description: str, mutate
) -> None:
    main_bytes, _ = build_main_fixture()
    damaged = tmp_path / "damaged.dicein1"
    damaged.write_bytes(mutate(main_bytes))
    with pytest.raises(ValueError):
        fixture_format.parse_main_generator_input(damaged)


def test_reader_rejects_a_damaged_prepass_fixture(tmp_path: Path) -> None:
    main_bytes, _ = build_main_fixture()
    prepass_bytes, _ = build_prepass_fixture(
        main_sha256_hex=hashlib.sha256(main_bytes).hexdigest()
    )
    damaged = tmp_path / "damaged.dicepp1"
    damaged.write_bytes(prepass_bytes[:-2] + bytes([prepass_bytes[-2] ^ 0xFF, 0]))
    with pytest.raises(ValueError):
        fixture_format.parse_prepass_generator_input(damaged)


def test_reader_rejects_an_unknown_variant(synthetic_pair) -> None:
    _, prepass_path, _, _ = synthetic_pair
    prepass = fixture_format.parse_prepass_generator_input(prepass_path)
    with pytest.raises(ValueError, match="not uniquely present"):
        prepass.logical_pixels("no-such-variant")


# --------------------------------------------------------------------------
# the minting script fails closed instead of minting something empty
# --------------------------------------------------------------------------


def _manifest_document(main: Path, prepass: Path) -> dict:
    return {
        "schema": mint.MANIFEST_SCHEMA,
        "workspace_root": str(main.parent),
        "profile": {
            "scanner_model": "nikon-super-coolscan-5000-ed",
            "mode": "normal",
            "selector": 8,
            "resolution_metric": 4000,
            "bit_depth": 16,
            "prepass_dpi": 285,
            "main_dpi": 4000,
            "prepass_variant": "native-exact",
        },
        "cases": {
            "synthetic": {
                "main": str(main),
                "main_sha256": fixture_format.sha256_file(main),
                "prepass": str(prepass),
                "prepass_sha256": fixture_format.sha256_file(prepass),
                "expected_logical_output_sha256": "0" * 64,
                "expected_shape_hwc": [5, 4, 3],
                "expected_compared_rgb16_samples": 60,
            }
        },
    }


def _run_gate(manifest: Path, case: str = "synthetic", backend: str = "metal"):
    return mint.run_gate(
        manifest_path=manifest,
        case_name=case,
        backend=backend,
        baseline="cpu-fast",
        runs=1,
        workspace_root=None,
    )


def test_absent_manifest_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(mint.GateFailure, match="does not exist"):
        _run_gate(tmp_path / "absent.json")


def test_unsupported_manifest_schema_fails_loudly(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": "something-else", "cases": {}}))
    with pytest.raises(mint.GateFailure, match="schema is unsupported"):
        _run_gate(manifest)


def test_malformed_manifest_fails_loudly(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{not json")
    with pytest.raises(mint.GateFailure, match="not valid JSON"):
        _run_gate(manifest)


def test_unknown_case_fails_loudly(tmp_path: Path, synthetic_pair) -> None:
    main_path, prepass_path, _, _ = synthetic_pair
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_manifest_document(main_path, prepass_path)))
    with pytest.raises(mint.GateFailure, match="unknown validation case"):
        _run_gate(manifest, case="frame404")


def test_absent_fixture_fails_loudly(tmp_path: Path, synthetic_pair) -> None:
    main_path, prepass_path, _, _ = synthetic_pair
    document = _manifest_document(main_path, prepass_path)
    document["cases"]["synthetic"]["main"] = str(tmp_path / "gone.dicein1")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(document))
    with pytest.raises(mint.GateFailure, match="fixture does not exist"):
        _run_gate(manifest)


def test_pin_mismatch_fails_loudly(tmp_path: Path, synthetic_pair) -> None:
    main_path, prepass_path, _, _ = synthetic_pair
    document = _manifest_document(main_path, prepass_path)
    document["cases"]["synthetic"]["main_sha256"] = "0" * 64
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(document))
    with pytest.raises(mint.GateFailure, match="pin mismatch"):
        _run_gate(manifest)


def test_aliased_fixture_roles_fail_loudly(tmp_path: Path, synthetic_pair) -> None:
    main_path, prepass_path, _, _ = synthetic_pair
    document = _manifest_document(main_path, prepass_path)
    document["cases"]["synthetic"]["prepass"] = str(main_path)
    document["cases"]["synthetic"]["prepass_sha256"] = fixture_format.sha256_file(
        main_path
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(document))
    with pytest.raises(mint.GateFailure, match="distinct paths"):
        _run_gate(manifest)


# --------------------------------------------------------------------------
# the checked-in receipts stay comparable
# --------------------------------------------------------------------------


@pytest.mark.parametrize("case", ["frame1", "frame2"])
def test_checked_in_metal_receipt_carries_every_binding_field(case: str) -> None:
    """A verify pass that compared nothing would report success."""

    receipt = json.loads(
        (EVIDENCE / f"metal-{case}-parity.json").read_text(encoding="utf-8")
    )
    missing = [field for field in mint.BINDING_FIELDS if field not in receipt]
    assert not missing, f"metal-{case}-parity.json is missing {missing}"
    assert "cpu_fast_and_metal_output_sha256" in receipt


def test_verify_compares_the_output_hash_across_baselines() -> None:
    """A differently named output hash must never go silently uncompared.

    The key is named after the two backends that agreed, so re-minting
    against a different baseline renames it. It is still the same frame's
    output hash, so it must be compared by value -- collecting the key only
    from the minted receipt would drop the most important field on the floor
    and still report success.
    """

    reference = EVIDENCE / "cuda-frame-1-parity.json"
    minted = json.loads(reference.read_text(encoding="utf-8"))
    minted["cpu_fast_and_cuda_output_sha256"] = minted.pop(
        "cpu_and_cuda_output_sha256"
    )
    differences, compared, _ = mint.verify_against(minted, reference)
    assert not differences
    label = next(item for item in compared if item.startswith("output_sha256 ("))
    assert "cpu_fast_and_cuda_output_sha256" in label
    assert "cpu_and_cuda_output_sha256" in label


def test_verify_catches_a_changed_hash_under_a_renamed_key() -> None:
    """Renaming the key must not become a way to smuggle a changed hash."""

    reference = EVIDENCE / "cuda-frame-1-parity.json"
    minted = json.loads(reference.read_text(encoding="utf-8"))
    minted.pop("cpu_and_cuda_output_sha256")
    minted["cpu_fast_and_cuda_output_sha256"] = "0" * 64
    differences, _, _ = mint.verify_against(minted, reference)
    assert any("output_sha256" in difference for difference in differences)


def test_verify_compares_a_matching_output_hash_key() -> None:
    reference = EVIDENCE / "cuda-frame-1-parity.json"
    minted = json.loads(reference.read_text(encoding="utf-8"))
    differences, compared, _ = mint.verify_against(minted, reference)
    assert not differences
    assert "cpu_and_cuda_output_sha256" in compared


def test_verify_reports_a_changed_value_as_a_difference() -> None:
    reference = EVIDENCE / "metal-frame1-parity.json"
    minted = json.loads(reference.read_text(encoding="utf-8"))
    minted["final_rng_state"] = minted["final_rng_state"] + 1
    differences, _, _ = mint.verify_against(minted, reference)
    assert any(difference.startswith("final_rng_state:") for difference in differences)


def test_source_manifest_recipe_matches_the_pinned_scope() -> None:
    """The receipts pin a source manifest; the tool must build the same one."""

    manifest = mint._source_manifest(REPOSITORY_ROOT)
    assert manifest, "source manifest must not be empty"
    assert all(path.startswith("src/portable_digital_ice/") for path in manifest)
    assert all(path.endswith(".py") for path in manifest)

    pinned = json.loads(
        (EVIDENCE / "metal-frame1-parity.json").read_text(encoding="utf-8")
    )
    if mint._manifest_hash(manifest) != pinned["source_manifest_sha256"]:
        pytest.skip(
            "package sources have changed since the receipts were last "
            "minted, so the pinned source manifest no longer describes this "
            "tree; re-mint the receipts to re-bind them"
        )


# --------------------------------------------------------------------------
# the complete-frame gate itself (needs the private captures)
# --------------------------------------------------------------------------


def _manifest_from_environment() -> Path:
    value = os.environ.get(MANIFEST_ENVIRONMENT_VARIABLE)
    if not value:
        pytest.skip(
            f"{MANIFEST_ENVIRONMENT_VARIABLE} is not set; the complete-frame "
            "gates need the two private scanner captures. See "
            "docs/validation.md and tools/fixtures.example.json."
        )
    path = Path(value)
    if not path.is_file():
        pytest.fail(
            f"{MANIFEST_ENVIRONMENT_VARIABLE} points at a file that does not "
            f"exist: {path}"
        )
    return path


def _require_metal() -> None:
    try:
        from portable_digital_ice.backend import metal_self_test

        metal_self_test()
    except Exception as error:  # pragma: no cover - host dependent
        pytest.skip(f"Metal backend unavailable here: {error}")


@pytest.mark.slow
@pytest.mark.parametrize("case", ["frame1", "frame2"])
def test_metal_full_frame_receipt_regenerates(case: str) -> None:
    """Re-mint a complete-frame receipt and require the checked-in values."""

    manifest = _manifest_from_environment()
    _require_metal()
    receipt = mint.run_gate(
        manifest_path=manifest,
        case_name=case,
        backend="metal",
        baseline="cpu-fast",
        runs=2,
        workspace_root=None,
    )
    failed = [name for name, value in receipt["gate_checks"].items() if not value]
    assert not failed, f"gate checks failed: {failed}"
    assert receipt["status"] == "PASS"

    reference = EVIDENCE / f"metal-{case}-parity.json"
    differences, compared, uncomparable = mint.verify_against(receipt, reference)
    assert not differences, (
        f"minted receipt differs from {reference.name}: {differences}"
    )
    # The Metal receipts record every binding field, so a verify pass here
    # that compared less than all of them would be hiding something.
    assert not uncomparable, f"{reference.name} lacks {uncomparable}"
    assert len(compared) >= len(mint.BINDING_FIELDS)
