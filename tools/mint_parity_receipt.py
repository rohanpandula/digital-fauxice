#!/usr/bin/env python3
"""Mint and re-verify the complete-frame parity receipts under ``evidence/``.

The package's acceptance rule is byte equality on a complete native frame
(``docs/validation.md``).  The receipts that record it are checked in; this
is the script that produces them, so a receipt can be regenerated and
independently audited by anyone holding the two private fixture files.

What it does, for one validation case and one candidate backend:

1. resolves the fixture pair from a manifest and refuses to continue unless
   every file exists and matches its pinned SHA-256;
2. parses both fixtures with the strict readers in ``dice_fixture_format``
   and re-derives the logical RGBI16 input hashes;
3. runs the *baseline* backend and the *candidate* backend in one process
   over the identical input bytes, both with diagnostics exported;
4. compares them with the package's own ``_parity_failures`` comparator --
   the one the AUTO self-test uses -- plus a full-frame delta sweep;
5. re-runs the candidate to prove consecutive-run determinism;
6. binds the result to the pinned CPU-reference output hash; and
7. emits the public receipt JSON, and/or verifies it against a receipt
   already checked in.

Nothing here is optional or best-effort.  A missing fixture, a hash that
does not match its pin, an unavailable backend, or any failed check exits
non-zero; the script never writes a partial or empty receipt.

Fixture manifest
----------------

Fixtures live outside this repository.  Point the script at a manifest with
``--fixtures`` or the ``PORTABLE_DICE_FIXTURE_MANIFEST`` environment
variable.  The manifest is the ``portable-digital-ice-private-validation-
manifest-v1`` document; ``tools/fixtures.example.json`` documents every
field, and ``docs/validation.md`` explains where the real one lives.

Examples
--------

Re-verify the checked-in Metal receipt for frame 1::

    python tools/mint_parity_receipt.py --case frame1 --backend metal \
        --fixtures /path/to/private-validation-manifest.json

Mint a fresh receipt to a scratch path (checked-in receipts are never
overwritten unless ``--out`` names them explicitly)::

    python tools/mint_parity_receipt.py --case frame2 --backend metal \
        --fixtures /path/to/manifest.json --out /tmp/metal-frame2.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

TOOLS_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = TOOLS_DIRECTORY.parent
if str(TOOLS_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIRECTORY))

from dice_fixture_format import (  # noqa: E402
    parse_main_generator_input,
    parse_prepass_generator_input,
    sha256_file,
    sha256_rgbi16,
)

MANIFEST_SCHEMA = "portable-digital-ice-private-validation-manifest-v1"
MANIFEST_ENVIRONMENT_VARIABLE = "PORTABLE_DICE_FIXTURE_MANIFEST"

# Backend key -> (ComputeBackend value, receipt name fragment, receipt schema).
BACKENDS = {
    "cpu": ("cpu", "cpu", None),
    "cpu-fast": ("cpu-fast", "cpu_fast", "cpu-fast"),
    "cuda": ("cuda", "cuda", "cuda"),
    "metal": ("metal", "metal", "metal"),
}

# The evidence file a given (backend, case) pair is checked in as.
EVIDENCE_NAMES = {
    ("metal", "frame1"): "metal-frame1-parity.json",
    ("metal", "frame2"): "metal-frame2-parity.json",
    ("cuda", "frame1"): "cuda-frame-1-parity.json",
    ("cuda", "frame2"): "cuda-frame-2-parity.json",
    ("cpu-fast", "frame1"): "cpu-fast-frame-1-parity.json",
    ("cpu-fast", "frame2"): "cpu-fast-frame-2-parity.json",
}

# Fields whose values bind the claim.  ``--verify`` compares exactly these
# against a checked-in receipt; prose, timings, host details, and the
# bookkeeping check counts are deliberately excluded because they are not
# reproducible across machines and do not carry the parity claim.
BINDING_FIELDS = (
    "status",
    "geometry_hwc",
    "compared_rgb16_samples",
    "mismatch_samples",
    "mismatch_pixels",
    "maximum_absolute_delta",
    "absolute_delta_sum",
    "diagnostics_planes_equal",
    "attempted_pixels",
    "written_pixels",
    "changed_pixels",
    "public_rng_advances",
    "final_rng_state",
    "startup_rng_advances_per_stage",
    "main_fixture_sha256",
    "prepass_fixture_sha256",
    "main_rgbi16_raw_byte_sha256",
    "prepass_rgbi16_raw_byte_sha256",
)


class GateFailure(RuntimeError):
    """A gate check failed, or an input could not be trusted."""


# --------------------------------------------------------------------------
# manifest and fixtures
# --------------------------------------------------------------------------


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise GateFailure(
            f"fixture manifest does not exist: {path}\n"
            "Pass --fixtures <manifest.json> or set "
            f"{MANIFEST_ENVIRONMENT_VARIABLE}. See tools/fixtures.example.json "
            "for the schema and docs/validation.md for what the fixtures are."
        )
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise GateFailure(f"fixture manifest is not valid JSON: {path}: {error}")
    if document.get("schema") != MANIFEST_SCHEMA:
        raise GateFailure(
            f"fixture manifest schema is unsupported: {document.get('schema')!r} "
            f"(expected {MANIFEST_SCHEMA!r})"
        )
    if not isinstance(document.get("cases"), dict) or not document["cases"]:
        raise GateFailure("fixture manifest declares no cases")
    if not isinstance(document.get("profile"), dict):
        raise GateFailure("fixture manifest declares no profile")
    return document


def _resolve(workspace: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else workspace / path


def _require_file(role: str, path: Path) -> Path:
    if not path.is_file():
        raise GateFailure(
            f"{role} fixture does not exist: {path}\n"
            "The complete-frame gates need the two private scanner captures; "
            "they are not redistributable and are not in this repository. "
            "Check the manifest's workspace_root and paths."
        )
    return path


def _assert_distinct(paths: dict[str, Path]) -> None:
    """Fixture roles must be different files, not aliases of one file."""

    resolved = [path.resolve(strict=False) for path in paths.values()]
    if len(resolved) != len(set(resolved)):
        raise GateFailure("fixture roles must use distinct paths")
    inodes = [(path.stat().st_dev, path.stat().st_ino) for path in paths.values()]
    if len(inodes) != len(set(inodes)):
        raise GateFailure("fixture roles cannot be hard-link aliases")


# --------------------------------------------------------------------------
# source manifest
# --------------------------------------------------------------------------


def _source_manifest(root: Path) -> dict[str, str]:
    """Hash every runtime source file, the scope the receipts pin.

    ``src/portable_digital_ice/**/*.py`` -- 31 files at the time of writing,
    reproducing the ``source_manifest_sha256`` recorded in the checked-in
    Metal receipts' re-verification block.
    """

    sources = sorted((root / "src" / "portable_digital_ice").rglob("*.py"))
    if not sources:
        raise GateFailure(f"no package sources found under {root}")
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sources
        if path.is_file()
    }


def _manifest_hash(manifest: dict[str, str]) -> str:
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _tool_manifest() -> dict[str, str]:
    """Hash this script and its reader, so the mint path is bound too."""

    return {
        f"tools/{path.name}": sha256_file(path)
        for path in sorted(TOOLS_DIRECTORY.glob("*.py"))
    }


# --------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameComparison:
    shape_matches: bool
    mismatch_samples: int
    mismatch_pixels: int
    maximum_absolute_delta: int
    absolute_delta_sum: int
    first_mismatch: dict[str, Any] | None


def _compare_exact(actual: np.ndarray, expected: np.ndarray) -> FrameComparison:
    """Full-frame delta sweep, one row at a time to bound memory."""

    if actual.dtype != np.dtype(np.uint16) or expected.dtype != np.dtype(np.uint16):
        raise GateFailure("comparison arrays must be uint16")
    if actual.shape != expected.shape:
        return FrameComparison(False, -1, -1, -1, -1, None)
    mismatch_samples = 0
    mismatch_pixels = 0
    maximum_delta = 0
    delta_sum = 0
    first: dict[str, Any] | None = None
    for y in range(actual.shape[0]):
        left = actual[y]
        right = expected[y]
        mismatch = left != right
        sample_count = int(np.count_nonzero(mismatch))
        if sample_count:
            mismatch_samples += sample_count
            mismatch_pixels += int(np.count_nonzero(np.any(mismatch, axis=1)))
            delta = np.abs(left.astype(np.int32) - right.astype(np.int32))
            maximum_delta = max(maximum_delta, int(delta.max()))
            delta_sum += int(delta.sum(dtype=np.int64))
            if first is None:
                x, channel = np.argwhere(mismatch)[0]
                first = {
                    "y": y,
                    "x": int(x),
                    "channel": int(channel),
                    "actual": int(left[x, channel]),
                    "expected": int(right[x, channel]),
                }
    return FrameComparison(
        True, mismatch_samples, mismatch_pixels, maximum_delta, delta_sum, first
    )


def _diagnostics_fingerprint(diagnostics: Any) -> dict[str, str]:
    """Hash the three diagnostics planes so runs compare without retention."""

    if diagnostics is None:
        raise GateFailure("a gate run returned no diagnostics planes")
    return {
        "score_plane": hashlib.sha256(
            np.ascontiguousarray(diagnostics.score_plane).view(np.uint32).tobytes()
        ).hexdigest(),
        "score_floor": hashlib.sha256(
            np.asarray(diagnostics.score_floor).view(np.uint32).tobytes()
        ).hexdigest(),
        "at_floor_mask": hashlib.sha256(
            np.ascontiguousarray(diagnostics.at_floor_mask).tobytes()
        ).hexdigest(),
        "changed_mask": hashlib.sha256(
            np.ascontiguousarray(diagnostics.changed_mask).tobytes()
        ).hexdigest(),
    }


def _counters(result: Any) -> dict[str, Any]:
    replay = result.replay
    startup = replay.startup
    return {
        "output_sha256": replay.output_sha256,
        "attempted_pixels": int(replay.attempted_pixels),
        "written_pixels": int(replay.written_pixels),
        "changed_pixels": int(replay.changed_pixels),
        "public_rng_advances": int(replay.public_rng_advances),
        "final_rng_state": int(replay.final_rng_state),
        "startup_rng_advances_per_stage": (
            [int(value) for value in startup.rng_advances_per_stage]
            if startup is not None
            else None
        ),
    }


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------


def _import_package() -> Any:
    """Import the package under test, preferring an installed copy."""

    try:
        import portable_digital_ice  # noqa: F401
    except ImportError:
        source = REPOSITORY_ROOT / "src"
        if not source.is_dir():
            raise GateFailure(
                "portable_digital_ice is not importable and "
                f"{source} does not exist; install the package first"
            )
        sys.path.insert(0, str(source))
    import portable_digital_ice as package

    return package


def _run(package: Any, job: Any, backend: str, label: str) -> Any:
    """Run one backend, letting its own fail-closed reason surface."""

    started = time.monotonic()
    try:
        routed = package.process(
            job, backend=package.ComputeBackend(backend), export_diagnostics=True
        )
    except Exception as error:
        raise GateFailure(
            f"{label} backend {backend!r} could not run this gate: "
            f"{type(error).__name__}: {error}"
        ) from error
    elapsed = time.monotonic() - started
    return routed.result, routed.selection, elapsed


def run_gate(
    *,
    manifest_path: Path,
    case_name: str,
    backend: str,
    baseline: str,
    runs: int,
    workspace_root: Path | None,
) -> dict[str, Any]:
    document = _load_manifest(manifest_path)
    cases = document["cases"]
    if case_name not in cases:
        raise GateFailure(
            f"unknown validation case {case_name!r}; "
            f"manifest declares {sorted(cases)}"
        )
    case = cases[case_name]
    profile = document["profile"]
    workspace = (
        workspace_root
        if workspace_root is not None
        else Path(document.get("workspace_root", manifest_path.parent))
    )

    paths = {
        "main": _require_file("main", _resolve(workspace, case["main"])),
        "prepass": _require_file("prepass", _resolve(workspace, case["prepass"])),
    }
    _assert_distinct(paths)

    # Preflight: the fixtures must be the pinned bytes before anything runs.
    fixture_before = {role: sha256_file(path) for role, path in paths.items()}
    pins = {"main": case["main_sha256"], "prepass": case["prepass_sha256"]}
    checks: dict[str, bool] = {
        f"{role}_fixture_sha256_matches_pin": fixture_before[role] == pins[role]
        for role in pins
    }
    if not all(checks.values()):
        raise GateFailure(
            "private fixture pin mismatch -- the fixture files are not the "
            f"bytes this receipt is about: {checks}"
        )

    main_input = parse_main_generator_input(paths["main"])
    prepass_input = parse_prepass_generator_input(paths["prepass"])
    variant_name = str(profile["prepass_variant"])

    checks["main_selector_matches_profile"] = main_input.selector == int(
        profile["selector"]
    )
    checks["prepass_selector_matches_main"] = (
        prepass_input.selector == main_input.selector
    )
    checks["prepass_is_hash_bound_to_main"] = (
        prepass_input.source_main_sha256 == main_input.file_sha256
    )
    if not all(checks.values()):
        raise GateFailure(f"fixture pair is not self-consistent: {checks}")

    package = _import_package()
    main_pixels = main_input.logical_pixels_mmap()
    prepass_pixels = prepass_input.logical_pixels(variant_name)
    main_raw_sha256 = sha256_rgbi16(main_pixels)
    prepass_raw_sha256 = sha256_rgbi16(prepass_pixels)

    acquisition = package.DualRGBIAcquisition(
        package.RGBI16Frame(
            prepass_pixels,
            package.AcquisitionEpoch.PREPASS,
            int(profile["prepass_dpi"]),
            f"DICEPP1:{prepass_input.file_sha256}:{variant_name}",
        ),
        package.RGBI16Frame(
            main_pixels,
            package.AcquisitionEpoch.MAIN,
            int(profile["main_dpi"]),
            f"DICEIN1:{main_input.file_sha256}",
        ),
        case_name,
    )
    job = package.ProcessingJob(
        acquisition=acquisition,
        scanner_model=package.ScannerModel(profile["scanner_model"]),
        mode=package.ProcessingMode(profile["mode"]),
        selector=int(profile["selector"]),
        resolution_metric=int(profile["resolution_metric"]),
        bit_depth=int(profile["bit_depth"]),
        focus_exposure_locked=True,
    )

    source_before = _source_manifest(REPOSITORY_ROOT)

    print(f"[gate] {case_name}: baseline {baseline} ...", flush=True)
    baseline_result, baseline_selection, baseline_elapsed = _run(
        package, job, baseline, "baseline"
    )
    baseline_counters = _counters(baseline_result)
    baseline_planes = _diagnostics_fingerprint(baseline_result.diagnostics)
    print(f"[gate] baseline done in {baseline_elapsed:.3f}s", flush=True)

    candidate_elapsed: list[float] = []
    candidate_counters: dict[str, Any] | None = None
    candidate_planes: dict[str, str] | None = None
    comparison: FrameComparison | None = None
    parity_failures: list[str] = []
    determinism_identical = True
    candidate_selection = None

    for index in range(runs):
        print(f"[gate] {case_name}: {backend} run {index + 1}/{runs} ...", flush=True)
        result, selection, elapsed = _run(package, job, backend, "candidate")
        candidate_elapsed.append(round(elapsed, 3))
        counters = _counters(result)
        planes = _diagnostics_fingerprint(result.diagnostics)
        if index == 0:
            candidate_selection = selection
            candidate_counters = counters
            candidate_planes = planes
            # The package's own comparator, the one the AUTO self-test uses.
            from portable_digital_ice.backend import _parity_failures

            parity_failures = _parity_failures(baseline_result, result, backend)
            comparison = _compare_exact(
                result.output_rgb16, baseline_result.output_rgb16
            )
        else:
            if counters != candidate_counters or planes != candidate_planes:
                determinism_identical = False
        del result
        print(f"[gate] {backend} run {index + 1} done in {elapsed:.3f}s", flush=True)

    assert candidate_counters is not None and candidate_planes is not None
    assert comparison is not None

    source_after = _source_manifest(REPOSITORY_ROOT)
    fixture_after = {role: sha256_file(path) for role, path in paths.items()}
    pinned_output = case["expected_logical_output_sha256"]
    expected_shape = tuple(int(value) for value in case["expected_shape_hwc"])

    checks.update(
        {
            "geometry_matches_expected": tuple(baseline_result.output_rgb16.shape)
            == expected_shape,
            "sample_count_matches_expected": int(np.prod(expected_shape))
            == int(case["expected_compared_rgb16_samples"]),
            "shape_matches": comparison.shape_matches,
            "mismatch_samples_zero": comparison.mismatch_samples == 0,
            "mismatch_pixels_zero": comparison.mismatch_pixels == 0,
            "maximum_absolute_delta_zero": comparison.maximum_absolute_delta == 0,
            "absolute_delta_sum_zero": comparison.absolute_delta_sum == 0,
            "package_parity_comparator_clean": not parity_failures,
            "output_hash_equal": baseline_counters["output_sha256"]
            == candidate_counters["output_sha256"],
            "attempted_pixels_equal": baseline_counters["attempted_pixels"]
            == candidate_counters["attempted_pixels"],
            "written_pixels_equal": baseline_counters["written_pixels"]
            == candidate_counters["written_pixels"],
            "changed_pixels_equal": baseline_counters["changed_pixels"]
            == candidate_counters["changed_pixels"],
            "public_rng_advances_equal": baseline_counters["public_rng_advances"]
            == candidate_counters["public_rng_advances"],
            "final_rng_state_equal": baseline_counters["final_rng_state"]
            == candidate_counters["final_rng_state"],
            "startup_rng_advances_equal": baseline_counters[
                "startup_rng_advances_per_stage"
            ]
            == candidate_counters["startup_rng_advances_per_stage"],
            "diagnostics_planes_equal": baseline_planes == candidate_planes,
            "baseline_matches_pinned_cpu_reference_output": baseline_counters[
                "output_sha256"
            ]
            == pinned_output,
            "candidate_matches_pinned_cpu_reference_output": candidate_counters[
                "output_sha256"
            ]
            == pinned_output,
            "consecutive_runs_identical": determinism_identical,
            "fixtures_immutable_across_run": fixture_before == fixture_after,
            "source_manifest_stable_across_run": source_before == source_after,
        }
    )
    # Manifests may pin the logical RGBI16 input bytes as well as the file
    # bytes.  Only assert what the manifest actually declares -- a check that
    # is unconditionally true asserts nothing.
    for role, actual in (
        ("main", main_raw_sha256),
        ("prepass", prepass_raw_sha256),
    ):
        pin = case.get(f"expected_{role}_rgbi16_raw_byte_sha256")
        if pin is not None:
            checks[f"{role}_rgbi16_raw_byte_sha256_matches_pin"] = actual == pin

    _, candidate_fragment, schema_name = BACKENDS[backend]
    _, baseline_fragment, _ = BACKENDS[baseline]
    output_key = f"{baseline_fragment}_and_{candidate_fragment}_output_sha256"

    receipt: dict[str, Any] = {
        "schema": f"portable-digital-ice-{schema_name}-parity-receipt-v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "claim": (
            f"{backend} backend byte parity with the CPU reference of this "
            f"package, complete native {profile['resolution_metric']} frame "
            f"(validation {case_name}), established in-process against the "
            f"{baseline} backend and bound to the pinned CPU-reference output "
            "of the checked-in receipts"
        ),
        "parity_baseline": (
            f"compared bitwise in-process against ComputeBackend.{baseline.upper().replace('-', '_')} "
            "(output bytes, all counters, RNG accounting, startup receipt, and "
            "the three diagnostics planes) using the package's own "
            "_parity_failures comparator; the output hash equals the "
            "CPU-reference hash pinned for this fixture, so the chain to the "
            "CPU reference is hash-transitive on identical input bytes"
        ),
        "case": case_name,
        "geometry_hwc": list(expected_shape),
        "compared_rgb16_samples": int(np.prod(expected_shape)),
        "mismatch_samples": comparison.mismatch_samples,
        "mismatch_pixels": comparison.mismatch_pixels,
        "maximum_absolute_delta": comparison.maximum_absolute_delta,
        "absolute_delta_sum": comparison.absolute_delta_sum,
        "diagnostics_planes_equal": baseline_planes == candidate_planes,
        output_key: candidate_counters["output_sha256"],
        "attempted_pixels": candidate_counters["attempted_pixels"],
        "written_pixels": candidate_counters["written_pixels"],
        "changed_pixels": candidate_counters["changed_pixels"],
        "public_rng_advances": candidate_counters["public_rng_advances"],
        "final_rng_state": candidate_counters["final_rng_state"],
        "startup_rng_advances_per_stage": candidate_counters[
            "startup_rng_advances_per_stage"
        ],
        "main_fixture_sha256": fixture_before["main"],
        "prepass_fixture_sha256": fixture_before["prepass"],
        "main_rgbi16_raw_byte_sha256": main_raw_sha256,
        "prepass_rgbi16_raw_byte_sha256": prepass_raw_sha256,
        "gate_checks": dict(sorted(checks.items())),
        "gate_checks_passed": sum(1 for value in checks.values() if value),
        "gate_checks_total": len(checks),
        "determinism": {
            "runs": runs,
            "consecutive_runs_identical": determinism_identical,
        },
        "source_manifest_sha256": _manifest_hash(source_before),
        "source_manifest_file_count": len(source_before),
        "source_manifest_scope": "src/portable_digital_ice/**/*.py",
        "minted_by": {
            "tool": "tools/mint_parity_receipt.py",
            "files": _tool_manifest(),
            "baseline_backend": baseline,
            "baseline_selection_reason": baseline_selection.reason,
            "candidate_selection_reason": (
                candidate_selection.reason if candidate_selection else None
            ),
        },
        "runtime": {
            "machine": platform.machine(),
            "os": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            f"{candidate_fragment}_elapsed_seconds_per_run": candidate_elapsed,
            f"{baseline_fragment}_elapsed_seconds": round(baseline_elapsed, 3),
        },
    }
    if parity_failures:
        receipt["parity_failures"] = parity_failures
    if comparison.first_mismatch is not None:
        receipt["first_mismatch"] = comparison.first_mismatch
    return receipt


# --------------------------------------------------------------------------
# verification against a checked-in receipt
# --------------------------------------------------------------------------


def verify_against(
    receipt: dict[str, Any], reference_path: Path
) -> tuple[list[str], list[str], list[str]]:
    """Compare a minted receipt against a checked-in one.

    Returns the binding fields that *differ*, the ones actually compared,
    and the ones that could not be compared because the older receipt never
    recorded them.  A caller can then report coverage instead of assuming
    it: a verify pass that compared nothing must not look like success.

    An uncomparable field is not a difference.  The earlier receipts predate
    the raw-input hashes, and a newer receipt carrying more evidence than an
    older one does not contradict it.

    The output-hash field is named after the two backends that agreed, so a
    receipt minted against a different baseline carries a differently named
    key.  It is still the same frame's output hash, so it is compared by
    value across the two names rather than skipped -- taking the key only
    from the minted receipt would let the single most important field go
    silently uncompared whenever the baselines differ.  What binds that hash
    to the CPU reference is not the key name but the
    ``candidate_matches_pinned_cpu_reference_output`` gate check.
    """

    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    differences: list[str] = []
    compared: list[str] = []
    uncomparable: list[str] = []

    for field in BINDING_FIELDS:
        if field not in receipt or field not in reference:
            if field in receipt or field in reference:
                uncomparable.append(field)
            continue
        compared.append(field)
        if receipt[field] != reference[field]:
            differences.append(
                f"{field}: minted={receipt[field]!r} checked-in={reference[field]!r}"
            )

    minted_keys = sorted(key for key in receipt if key.endswith("_output_sha256"))
    reference_keys = sorted(key for key in reference if key.endswith("_output_sha256"))
    if len(minted_keys) == 1 and len(reference_keys) == 1:
        minted_key, reference_key = minted_keys[0], reference_keys[0]
        label = (
            minted_key
            if minted_key == reference_key
            else f"output_sha256 ({minted_key} vs {reference_key})"
        )
        compared.append(label)
        if receipt[minted_key] != reference[reference_key]:
            differences.append(
                f"{label}: minted={receipt[minted_key]!r} "
                f"checked-in={reference[reference_key]!r}"
            )
    else:
        uncomparable.extend(sorted(set(minted_keys) ^ set(reference_keys)))

    return differences, compared, uncomparable


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=os.environ.get(MANIFEST_ENVIRONMENT_VARIABLE),
        help=(
            "private validation manifest JSON; defaults to "
            f"${MANIFEST_ENVIRONMENT_VARIABLE}"
        ),
    )
    parser.add_argument("--case", required=True, help="validation case, e.g. frame1")
    parser.add_argument(
        "--backend",
        required=True,
        choices=("metal", "cuda", "cpu-fast"),
        help="the backend this receipt is about",
    )
    parser.add_argument(
        "--baseline",
        choices=("cpu-fast", "cpu"),
        help=(
            "in-process comparison baseline; defaults to cpu-fast for "
            "metal/cuda and to the exact cpu reference for cpu-fast"
        ),
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=2,
        help="candidate runs; 2 or more proves consecutive-run determinism",
    )
    parser.add_argument("--workspace-root", type=Path)
    parser.add_argument(
        "--out", type=Path, help="write the minted receipt here (never implicit)"
    )
    parser.add_argument(
        "--verify",
        type=Path,
        help=(
            "compare the minted receipt's binding fields against this one; "
            "defaults to the checked-in evidence file for this backend/case"
        ),
    )
    parser.add_argument(
        "--no-verify", action="store_true", help="skip the checked-in comparison"
    )
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    if arguments.fixtures is None:
        raise GateFailure(
            "no fixture manifest given. Pass --fixtures <manifest.json> or set "
            f"{MANIFEST_ENVIRONMENT_VARIABLE}. The complete-frame gates cannot "
            "run without the two private scanner captures; see "
            "docs/validation.md and tools/fixtures.example.json."
        )
    if arguments.runs < 1:
        raise GateFailure("--runs must be at least 1")
    baseline = arguments.baseline or (
        "cpu" if arguments.backend == "cpu-fast" else "cpu-fast"
    )
    if baseline == arguments.backend:
        raise GateFailure("the baseline and the candidate backend must differ")

    receipt = run_gate(
        manifest_path=Path(arguments.fixtures),
        case_name=arguments.case,
        backend=arguments.backend,
        baseline=baseline,
        runs=arguments.runs,
        workspace_root=arguments.workspace_root,
    )

    failed = [name for name, value in receipt["gate_checks"].items() if not value]
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)

    if arguments.out is not None:
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        arguments.out.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"[gate] wrote {arguments.out}", flush=True)

    status = 0
    if failed:
        print(f"[gate] FAILED checks: {failed}", file=sys.stderr, flush=True)
        status = 1

    if not arguments.no_verify:
        reference = arguments.verify
        if reference is None:
            name = EVIDENCE_NAMES.get((arguments.backend, arguments.case))
            candidate = REPOSITORY_ROOT / "evidence" / name if name else None
            reference = candidate if candidate and candidate.is_file() else None
        if reference is None:
            print(
                "[gate] no checked-in receipt to verify against "
                "(pass --verify <file> or --no-verify)",
                flush=True,
            )
        else:
            differences, compared, uncomparable = verify_against(receipt, reference)
            if uncomparable:
                print(
                    f"[gate] not recorded by {reference.name}, so not "
                    f"compared: {', '.join(uncomparable)}",
                    flush=True,
                )
            if differences:
                print(
                    f"[gate] MINTED RECEIPT DIFFERS FROM {reference.name}:",
                    file=sys.stderr,
                    flush=True,
                )
                for difference in differences:
                    print(f"  - {difference}", file=sys.stderr, flush=True)
                status = 1
            elif not compared:
                print(
                    f"[gate] nothing in {reference.name} was comparable; "
                    "this is not a successful verification",
                    file=sys.stderr,
                    flush=True,
                )
                status = 1
            else:
                print(
                    f"[gate] reproduces {reference.name} exactly on all "
                    f"{len(compared)} comparable binding fields: "
                    f"{', '.join(compared)}",
                    flush=True,
                )
    return status


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateFailure as failure:
        print(f"gate failure: {failure}", file=sys.stderr, flush=True)
        raise SystemExit(2)
