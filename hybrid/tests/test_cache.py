from __future__ import annotations

import hashlib
import io
import json
import os
import stat
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from portable_digital_ice import (
    DEFAULT_PROFILE,
    ProcessingDiagnostics,
    ProcessingResult,
)

from fauxce_hybrid import cache as cache_module
from fauxce_hybrid.cache import (
    CACHE_SCHEMA,
    MANIFEST_FILENAME,
    DiagnosticsCacheBinding,
    DiagnosticsCacheError,
    build_cache_binding,
    canonical_json_bytes,
    compute_core_source_manifest,
    hash_rgb16,
    hash_rgbir16,
    load_diagnostics_cache,
    read_cache_binding,
    save_diagnostics_cache,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sample() -> tuple[
    np.ndarray, np.ndarray, ProcessingResult, DiagnosticsCacheBinding
]:
    prepass = np.arange(3 * 4 * 4, dtype=np.uint16).reshape(3, 4, 4)
    main = (np.arange(5 * 7 * 4, dtype=np.uint16).reshape(5, 7, 4) * 17) + 31
    output = np.ascontiguousarray(main[:, :, :3] + np.uint16(9))

    floor = np.float32(0.02)
    score = np.linspace(0.01, 0.08, 35, dtype=np.float32).reshape(5, 7)
    score[1, 2] = floor
    score[3, 5] = floor
    at_floor = score == floor
    changed = np.zeros((5, 7), dtype=bool)
    changed[1:4, 2:6] = True
    diagnostics = ProcessingDiagnostics(
        score_plane=score,
        score_floor=floor,
        at_floor_mask=at_floor,
        changed_mask=changed,
    )
    result = ProcessingResult(
        output_rgb16=output,
        replay=SimpleNamespace(output_sha256=hash_rgb16(output)),
        profile_id=DEFAULT_PROFILE.profile_id,
        diagnostics=diagnostics,
    )
    assertion_hash = _sha256(b"caller asserts same frame and locked exposure")
    binding = build_cache_binding(
        prepass,
        main,
        provenance_assertion_id="caller-asserted:scan-0007",
        provenance_assertion_sha256=assertion_hash,
        requested_backend="auto",
        used_backend="cpu",
        backend_reason="CUDA unavailable; complete job ran on exact CPU reference",
    )
    return prepass, main, result, binding


def _write_manifest(directory: Path, document: object) -> None:
    (directory / MANIFEST_FILENAME).write_bytes(canonical_json_bytes(document))


def _manifest_sha256(directory: Path) -> str:
    return _sha256((directory / MANIFEST_FILENAME).read_bytes())


def _replace_npy_and_manifest_metadata(
    directory: Path,
    artifact_name: str,
    array: np.ndarray,
) -> dict[str, object]:
    document = json.loads((directory / MANIFEST_FILENAME).read_bytes())
    metadata = document["artifacts"][artifact_name]
    path = directory / metadata["filename"]
    with path.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    payload = path.read_bytes()
    metadata["file_sha256"] = _sha256(payload)
    metadata["raw_sha256"] = _sha256(np.ascontiguousarray(array).tobytes())
    metadata["dtype"] = array.dtype.str
    metadata["shape"] = list(array.shape)
    _write_manifest(directory, document)
    return document


def test_round_trip_is_deterministic_bound_and_read_only(tmp_path: Path) -> None:
    prepass, main, result, binding = _sample()
    first = tmp_path / "first"
    second = tmp_path / "second"

    saved = save_diagnostics_cache(
        first,
        processing_result=result,
        binding=binding,
    )
    save_diagnostics_cache(second, processing_result=result, binding=binding)
    loaded = load_diagnostics_cache(
        first,
        expected_binding=binding,
        expected_manifest_sha256=saved.manifest_sha256,
    )

    assert (first / MANIFEST_FILENAME).read_bytes() == (
        second / MANIFEST_FILENAME
    ).read_bytes()
    assert saved.manifest_sha256 == loaded.manifest_sha256
    assert saved.output_sha256 == hash_rgb16(result.output_rgb16)
    assert loaded.binding == binding
    assert (
        read_cache_binding(
            first,
            expected_manifest_sha256=saved.manifest_sha256,
        )
        == binding
    )
    assert hash_rgbir16(prepass) == binding.prepass_raw_sha256
    assert hash_rgbir16(main) == binding.main_raw_sha256

    expected_files = {
        "at-floor-mask.npy",
        "changed-mask.npy",
        "diagnostics-cache.json",
        "output.rgb16.npy",
        "score-plane.npy",
    }
    assert {path.name for path in first.iterdir()} == expected_files
    np.testing.assert_array_equal(loaded.output_rgb16, result.output_rgb16)
    np.testing.assert_array_equal(
        loaded.diagnostics.score_plane,
        result.diagnostics.score_plane,
    )
    np.testing.assert_array_equal(
        loaded.diagnostics.at_floor_mask,
        result.diagnostics.at_floor_mask,
    )
    np.testing.assert_array_equal(
        loaded.diagnostics.changed_mask,
        result.diagnostics.changed_mask,
    )
    assert loaded.diagnostics.score_floor.view(
        np.uint32
    ) == result.diagnostics.score_floor.view(np.uint32)
    for array in (
        loaded.output_rgb16,
        loaded.diagnostics.score_plane,
        loaded.diagnostics.at_floor_mask,
        loaded.diagnostics.changed_mask,
    ):
        assert array.flags.c_contiguous
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flat[0] = array.flat[0]

    manifest_bytes = (first / MANIFEST_FILENAME).read_bytes()
    document = json.loads(manifest_bytes)
    assert manifest_bytes == canonical_json_bytes(document)
    assert document["schema"] == CACHE_SCHEMA
    assert document["pure_output"] == {
        "artifact": "pure_output_rgb16",
        "generative_modification": False,
        "output_sha256": hash_rgb16(result.output_rgb16),
        "source": "portable_digital_ice.ProcessingResult.output_rgb16",
    }
    assert document["hash_canonicalization"]["uint16"] == (
        "little-endian unsigned 16-bit bytes"
    )
    for metadata in document["artifacts"].values():
        artifact_path = first / metadata["filename"]
        assert metadata["file_sha256"] == _sha256(artifact_path.read_bytes())
        with artifact_path.open("rb") as handle:
            assert not np.load(handle, allow_pickle=False).dtype.hasobject


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits only")
def test_cache_permissions_are_private_even_with_open_umask(tmp_path: Path) -> None:
    _, _, result, binding = _sample()
    destination = tmp_path / "private-cache"
    previous_umask = os.umask(0)
    try:
        save_diagnostics_cache(
            destination,
            processing_result=result,
            binding=binding,
        )
    finally:
        os.umask(previous_umask)

    assert stat.S_IMODE(destination.stat().st_mode) == 0o700
    for path in destination.iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    existing = tmp_path / "existing-wide-cache"
    existing.mkdir(mode=0o777)
    existing.chmod(0o777)
    save_diagnostics_cache(existing, processing_result=result, binding=binding)
    assert stat.S_IMODE(existing.stat().st_mode) == 0o700
    for path in existing.iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.parametrize(
    "filename",
    (
        "output.rgb16.npy",
        "score-plane.npy",
        "at-floor-mask.npy",
        "changed-mask.npy",
    ),
)
def test_single_byte_artifact_tamper_fails_closed(
    tmp_path: Path,
    filename: str,
) -> None:
    _, _, result, binding = _sample()
    saved = save_diagnostics_cache(
        tmp_path,
        processing_result=result,
        binding=binding,
    )
    path = tmp_path / filename
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 0x01
    path.write_bytes(payload)

    with pytest.raises(DiagnosticsCacheError, match="file SHA-256 mismatch"):
        load_diagnostics_cache(
            tmp_path,
            expected_binding=binding,
            expected_manifest_sha256=saved.manifest_sha256,
        )


def test_single_byte_manifest_tamper_fails_closed(tmp_path: Path) -> None:
    _, _, result, binding = _sample()
    saved = save_diagnostics_cache(
        tmp_path,
        processing_result=result,
        binding=binding,
    )
    path = tmp_path / MANIFEST_FILENAME
    payload = path.read_bytes()
    assert CACHE_SCHEMA.encode() in payload
    path.write_bytes(
        payload.replace(CACHE_SCHEMA.encode(), b"g" + CACHE_SCHEMA[1:].encode())
    )

    with pytest.raises(DiagnosticsCacheError, match="manifest SHA-256 mismatch"):
        load_diagnostics_cache(
            tmp_path,
            expected_binding=binding,
            expected_manifest_sha256=saved.manifest_sha256,
        )


def test_oversized_sparse_manifest_fails_before_json_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, result, binding = _sample()
    save_diagnostics_cache(tmp_path, processing_result=result, binding=binding)
    manifest = tmp_path / MANIFEST_FILENAME
    with manifest.open("r+b") as handle:
        handle.truncate(cache_module._MAX_CACHE_MANIFEST_BYTES + 1)

    decoded = False

    def forbidden_json_decode(*_args: object, **_kwargs: object) -> object:
        nonlocal decoded
        decoded = True
        raise AssertionError("oversized manifests must fail before JSON decoding")

    monkeypatch.setattr(cache_module.json, "loads", forbidden_json_decode)

    with pytest.raises(DiagnosticsCacheError, match="manifest exceeds its byte limit"):
        read_cache_binding(tmp_path)

    assert not decoded


def test_oversized_sparse_artifact_fails_before_hash_or_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, result, binding = _sample()
    saved = save_diagnostics_cache(tmp_path, processing_result=result, binding=binding)
    artifact = tmp_path / "at-floor-mask.npy"
    with artifact.open("r+b") as handle:
        handle.truncate(cache_module._MAX_NPY_ENCODED_BYTES + 1)
    monkeypatch.setattr(
        cache_module,
        "_sha256_open_artifact",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("oversized sparse artifacts must fail before hashing")
        ),
    )

    with pytest.raises(DiagnosticsCacheError, match="encoded byte limit"):
        load_diagnostics_cache(
            tmp_path,
            expected_binding=binding,
            expected_manifest_sha256=saved.manifest_sha256,
        )


def test_jointly_rehashed_header_cannot_claim_oversized_decoded_array(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, result, binding = _sample()
    save_diagnostics_cache(tmp_path, processing_result=result, binding=binding)
    huge_shape = (cache_module._MAX_NPY_DECODED_BYTES + 1, 1)
    encoded = io.BytesIO()
    np.lib.format.write_array_header_2_0(
        encoded,
        {
            "descr": "|b1",
            "fortran_order": False,
            "shape": huge_shape,
        },
    )
    payload = encoded.getvalue()
    artifact = tmp_path / "at-floor-mask.npy"
    artifact.write_bytes(payload)
    document = json.loads((tmp_path / MANIFEST_FILENAME).read_bytes())
    metadata = document["artifacts"]["at_floor_mask"]
    metadata["file_sha256"] = _sha256(payload)
    metadata["shape"] = list(huge_shape)
    huge_binding = replace(
        binding,
        main_shape=(huge_shape[0], huge_shape[1], 4),
    )
    document["binding"]["main_shape"] = list(huge_binding.main_shape)
    _write_manifest(tmp_path, document)
    monkeypatch.setattr(
        cache_module,
        "_decode_open_npy",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("oversized declared arrays must fail before decoding")
        ),
    )

    with pytest.raises(DiagnosticsCacheError, match="decoded byte limit"):
        load_diagnostics_cache(
            tmp_path,
            expected_binding=huge_binding,
            expected_manifest_sha256=_manifest_sha256(tmp_path),
        )


def test_rehashed_fortran_order_and_trailing_data_fail_header_preflight(
    tmp_path: Path,
) -> None:
    _, _, result, binding = _sample()
    fortran_cache = tmp_path / "fortran"
    save_diagnostics_cache(
        fortran_cache,
        processing_result=result,
        binding=binding,
    )
    _replace_npy_and_manifest_metadata(
        fortran_cache,
        "score_plane",
        np.asfortranarray(result.diagnostics.score_plane),
    )
    with pytest.raises(DiagnosticsCacheError, match="C array order"):
        load_diagnostics_cache(
            fortran_cache,
            expected_binding=binding,
            expected_manifest_sha256=_manifest_sha256(fortran_cache),
        )

    trailing_cache = tmp_path / "trailing"
    save_diagnostics_cache(
        trailing_cache,
        processing_result=result,
        binding=binding,
    )
    artifact = trailing_cache / "at-floor-mask.npy"
    payload = artifact.read_bytes() + b"trailing"
    artifact.write_bytes(payload)
    document = json.loads((trailing_cache / MANIFEST_FILENAME).read_bytes())
    document["artifacts"]["at_floor_mask"]["file_sha256"] = _sha256(payload)
    _write_manifest(trailing_cache, document)
    with pytest.raises(DiagnosticsCacheError, match="data length"):
        load_diagnostics_cache(
            trailing_cache,
            expected_binding=binding,
            expected_manifest_sha256=_manifest_sha256(trailing_cache),
        )


def test_path_swap_after_open_cannot_change_descriptor_used_for_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, result, binding = _sample()
    saved = save_diagnostics_cache(tmp_path, processing_result=result, binding=binding)
    target = tmp_path / "at-floor-mask.npy"
    detached = tmp_path.parent / f"{tmp_path.name}-detached-at-floor.npy"
    forged = tmp_path.parent / f"{tmp_path.name}-forged-at-floor.npy"
    with forged.open("wb") as handle:
        np.save(
            handle,
            np.logical_not(result.diagnostics.at_floor_mask),
            allow_pickle=False,
        )

    original_open = cache_module._open_cache_artifact_descriptor
    original_hash = cache_module._sha256_open_artifact
    swapped = False
    descriptor_identities: list[tuple[int, int]] = []

    def open_then_swap(
        directory: Path,
        filename: str,
        *,
        name: str,
    ) -> tuple[int, os.stat_result]:
        nonlocal swapped
        descriptor, measured = original_open(directory, filename, name=name)
        if name == "at_floor_mask":
            target.replace(detached)
            forged.replace(target)
            swapped = True
        return descriptor, measured

    def record_descriptor(handle: object, *, name: str) -> str:
        if name == "at_floor_mask":
            descriptor = handle.fileno()
            descriptor_identities.append((descriptor, os.fstat(descriptor).st_ino))
        return original_hash(handle, name=name)

    monkeypatch.setattr(
        cache_module,
        "_open_cache_artifact_descriptor",
        open_then_swap,
    )
    monkeypatch.setattr(cache_module, "_sha256_open_artifact", record_descriptor)

    loaded = load_diagnostics_cache(
        tmp_path,
        expected_binding=binding,
        expected_manifest_sha256=saved.manifest_sha256,
    )

    assert swapped
    assert len(descriptor_identities) == 2
    assert descriptor_identities[0] == descriptor_identities[1]
    np.testing.assert_array_equal(
        loaded.diagnostics.at_floor_mask,
        result.diagnostics.at_floor_mask,
    )
    assert not np.array_equal(
        loaded.diagnostics.at_floor_mask,
        np.logical_not(result.diagnostics.at_floor_mask),
    )


def test_coherent_artifact_and_manifest_rewrite_fails_external_anchor(
    tmp_path: Path,
) -> None:
    _, _, result, binding = _sample()
    saved = save_diagnostics_cache(
        tmp_path,
        processing_result=result,
        binding=binding,
    )
    forged_output = np.full_like(result.output_rgb16, np.uint16(61_337))
    document = _replace_npy_and_manifest_metadata(
        tmp_path,
        "pure_output_rgb16",
        forged_output,
    )
    document["pure_output"]["output_sha256"] = hash_rgb16(forged_output)
    _write_manifest(tmp_path, document)

    with pytest.raises(DiagnosticsCacheError, match="manifest SHA-256 mismatch"):
        read_cache_binding(
            tmp_path,
            expected_manifest_sha256=saved.manifest_sha256,
        )
    with pytest.raises(DiagnosticsCacheError, match="manifest SHA-256 mismatch"):
        load_diagnostics_cache(
            tmp_path,
            expected_binding=binding,
            expected_manifest_sha256=saved.manifest_sha256,
        )


def test_cache_layout_rejects_extra_entries_and_links(tmp_path: Path) -> None:
    _, _, result, binding = _sample()
    cache = tmp_path / "cache"
    saved = save_diagnostics_cache(cache, processing_result=result, binding=binding)
    (cache / "extra.txt").write_text("not declared", encoding="utf-8")
    with pytest.raises(DiagnosticsCacheError, match="entries mismatch"):
        read_cache_binding(cache)
    with pytest.raises(DiagnosticsCacheError, match="entries mismatch"):
        load_diagnostics_cache(
            cache,
            expected_binding=binding,
            expected_manifest_sha256=saved.manifest_sha256,
        )

    (cache / "extra.txt").unlink()
    artifact = cache / "score-plane.npy"
    real_artifact = tmp_path / "score-plane-real.npy"
    artifact.replace(real_artifact)
    artifact.symlink_to(real_artifact)
    with pytest.raises(DiagnosticsCacheError, match="regular file, not a link"):
        load_diagnostics_cache(
            cache,
            expected_binding=binding,
            expected_manifest_sha256=saved.manifest_sha256,
        )


def test_save_rejects_symlinked_destination(tmp_path: Path) -> None:
    _, _, result, binding = _sample()
    real_directory = tmp_path / "real"
    real_directory.mkdir()
    link = tmp_path / "cache-link"
    link.symlink_to(real_directory, target_is_directory=True)
    with pytest.raises(DiagnosticsCacheError, match="not a directory"):
        save_diagnostics_cache(link, processing_result=result, binding=binding)


def test_load_rejects_symlinked_cache_path_component(tmp_path: Path) -> None:
    _, _, result, binding = _sample()
    real_parent = tmp_path / "real-parent"
    cache = real_parent / "cache"
    saved = save_diagnostics_cache(cache, processing_result=result, binding=binding)
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(DiagnosticsCacheError, match="real directories"):
        load_diagnostics_cache(
            linked_parent / "cache",
            expected_binding=binding,
            expected_manifest_sha256=saved.manifest_sha256,
        )


def test_save_never_overwrites_a_racing_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, result, binding = _sample()
    destination = tmp_path / "cache"
    original_write = cache_module._write_exclusive
    racer_payload = b"racer-owned-content"
    raced_path: Path | None = None

    def race_first_artifact(path: Path, payload: bytes) -> None:
        nonlocal raced_path
        if raced_path is None:
            path.write_bytes(racer_payload)
            raced_path = path
        original_write(path, payload)

    monkeypatch.setattr(cache_module, "_write_exclusive", race_first_artifact)
    with pytest.raises(DiagnosticsCacheError, match="cannot write cache artifact"):
        save_diagnostics_cache(
            destination,
            processing_result=result,
            binding=binding,
        )
    assert raced_path is not None
    assert raced_path.read_bytes() == racer_payload
    assert not (destination / ".diagnostics-cache-write.lock").exists()


def test_input_and_provenance_binding_mismatches_fail_closed(tmp_path: Path) -> None:
    prepass, main, result, binding = _sample()
    saved = save_diagnostics_cache(
        tmp_path,
        processing_result=result,
        binding=binding,
    )
    changed_main = main.copy()
    changed_main[0, 0, 0] ^= np.uint16(1)
    wrong_input = build_cache_binding(
        prepass,
        changed_main,
        provenance_assertion_id=binding.provenance_assertion_id,
        provenance_assertion_sha256=binding.provenance_assertion_sha256,
        requested_backend=binding.requested_backend,
        used_backend=binding.used_backend,
        backend_reason=binding.backend_reason,
    )

    with pytest.raises(DiagnosticsCacheError, match="main_raw_sha256"):
        load_diagnostics_cache(
            tmp_path,
            expected_binding=wrong_input,
            expected_manifest_sha256=saved.manifest_sha256,
        )

    wrong_provenance = replace(
        binding,
        provenance_assertion_sha256="f" * 64,
    )
    with pytest.raises(DiagnosticsCacheError, match="provenance_assertion_sha256"):
        load_diagnostics_cache(
            tmp_path,
            expected_binding=wrong_provenance,
            expected_manifest_sha256=saved.manifest_sha256,
        )


def test_backend_selection_binding_mismatch_fails_closed(tmp_path: Path) -> None:
    _, _, result, binding = _sample()
    saved = save_diagnostics_cache(
        tmp_path,
        processing_result=result,
        binding=binding,
    )

    with pytest.raises(DiagnosticsCacheError, match="cache binding mismatch"):
        load_diagnostics_cache(
            tmp_path,
            expected_binding=replace(
                binding,
                requested_backend="cpu",
                backend_reason="explicit CPU request",
            ),
            expected_manifest_sha256=saved.manifest_sha256,
        )


def test_manifest_core_binding_tamper_fails_closed(tmp_path: Path) -> None:
    _, _, result, binding = _sample()
    save_diagnostics_cache(tmp_path, processing_result=result, binding=binding)
    document = json.loads((tmp_path / MANIFEST_FILENAME).read_bytes())
    document["binding"]["core_source_manifest_sha256"] = "0" * 64
    _write_manifest(tmp_path, document)

    with pytest.raises(DiagnosticsCacheError, match="core_source_manifest_sha256"):
        load_diagnostics_cache(
            tmp_path,
            expected_binding=binding,
            expected_manifest_sha256=_manifest_sha256(tmp_path),
        )


def test_core_source_manifest_is_order_stable_and_content_sensitive(
    tmp_path: Path,
) -> None:
    package = tmp_path / "src" / "portable_digital_ice"
    package.mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text("[project]\nname='core'\n")
    (package / "z.py").write_text("Z = 1\n")
    (package / "a.py").write_text("A = 1\n")

    first = compute_core_source_manifest(tmp_path)
    second = compute_core_source_manifest(tmp_path)
    assert first == second

    (package / "a.py").write_text("A = 2\n")
    assert compute_core_source_manifest(tmp_path) != first


def test_rehashed_wrong_dtype_still_fails_closed(tmp_path: Path) -> None:
    _, _, result, binding = _sample()
    save_diagnostics_cache(tmp_path, processing_result=result, binding=binding)
    wrong = result.output_rgb16.astype(np.uint8)
    _replace_npy_and_manifest_metadata(tmp_path, "pure_output_rgb16", wrong)

    with pytest.raises(DiagnosticsCacheError, match="unexpected dtype"):
        load_diagnostics_cache(
            tmp_path,
            expected_binding=binding,
            expected_manifest_sha256=_manifest_sha256(tmp_path),
        )


def test_rehashed_wrong_shape_still_fails_closed(tmp_path: Path) -> None:
    _, _, result, binding = _sample()
    save_diagnostics_cache(tmp_path, processing_result=result, binding=binding)
    wrong = result.diagnostics.score_plane[:-1].copy()
    _replace_npy_and_manifest_metadata(tmp_path, "score_plane", wrong)

    with pytest.raises(DiagnosticsCacheError, match="score plane shape"):
        load_diagnostics_cache(
            tmp_path,
            expected_binding=binding,
            expected_manifest_sha256=_manifest_sha256(tmp_path),
        )


def test_save_rejects_output_geometry_outside_main_input_binding(
    tmp_path: Path,
) -> None:
    _, _, result, binding = _sample()
    wrong_binding = replace(binding, main_shape=(6, 7, 4))

    with pytest.raises(DiagnosticsCacheError, match="bound main input geometry"):
        save_diagnostics_cache(
            tmp_path,
            processing_result=result,
            binding=wrong_binding,
        )


def test_jointly_rehashed_cache_geometry_fails_before_any_array_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, result, binding = _sample()
    save_diagnostics_cache(tmp_path, processing_result=result, binding=binding)

    _replace_npy_and_manifest_metadata(
        tmp_path,
        "pure_output_rgb16",
        result.output_rgb16[:-1].copy(),
    )
    _replace_npy_and_manifest_metadata(
        tmp_path,
        "score_plane",
        result.diagnostics.score_plane[:-1].copy(),
    )
    _replace_npy_and_manifest_metadata(
        tmp_path,
        "at_floor_mask",
        result.diagnostics.at_floor_mask[:-1].copy(),
    )
    document = _replace_npy_and_manifest_metadata(
        tmp_path,
        "changed_mask",
        result.diagnostics.changed_mask[:-1].copy(),
    )
    document["pure_output"]["output_sha256"] = document["artifacts"][
        "pure_output_rgb16"
    ]["raw_sha256"]
    _write_manifest(tmp_path, document)
    decoded = False

    def forbidden_decode(*_args: object, **_kwargs: object) -> np.ndarray:
        nonlocal decoded
        decoded = True
        raise AssertionError("wrong cache geometry must fail before array decoding")

    monkeypatch.setattr(cache_module, "_decode_open_npy", forbidden_decode)

    with pytest.raises(DiagnosticsCacheError, match="bound main input geometry"):
        load_diagnostics_cache(
            tmp_path,
            expected_binding=binding,
            expected_manifest_sha256=_manifest_sha256(tmp_path),
        )
    assert not decoded


def test_rehashed_inconsistent_floor_mask_still_fails_closed(tmp_path: Path) -> None:
    _, _, result, binding = _sample()
    save_diagnostics_cache(tmp_path, processing_result=result, binding=binding)
    wrong = result.diagnostics.at_floor_mask.copy()
    wrong[0, 0] = ~wrong[0, 0]
    document = _replace_npy_and_manifest_metadata(tmp_path, "at_floor_mask", wrong)
    document["artifacts"]["at_floor_mask"]["dtype"] = "|b1"
    _write_manifest(tmp_path, document)

    with pytest.raises(
        DiagnosticsCacheError, match="inconsistent with score and floor"
    ):
        load_diagnostics_cache(
            tmp_path,
            expected_binding=binding,
            expected_manifest_sha256=_manifest_sha256(tmp_path),
        )


def test_save_rejects_missing_diagnostics_and_replay_hash_mismatch(
    tmp_path: Path,
) -> None:
    _, _, result, binding = _sample()
    without_diagnostics = replace(result, diagnostics=None)
    with pytest.raises(DiagnosticsCacheError, match="no exported diagnostics"):
        save_diagnostics_cache(
            tmp_path / "missing",
            processing_result=without_diagnostics,
            binding=binding,
        )

    bad_replay = replace(result, replay=SimpleNamespace(output_sha256="0" * 64))
    with pytest.raises(DiagnosticsCacheError, match="replay output hash"):
        save_diagnostics_cache(
            tmp_path / "bad-hash",
            processing_result=bad_replay,
            binding=binding,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("provenance_assertion_sha256", "not-a-hash"),
        ("profile_id", "some-other-profile"),
        ("requested_backend", "magic"),
        ("used_backend", "auto"),
        ("backend_reason", ""),
        ("backend_reason", "caller supplied but impossible reason"),
    ),
)
def test_binding_rejects_invalid_contract_fields(
    field: str,
    value: object,
) -> None:
    _, _, _, binding = _sample()
    with pytest.raises(DiagnosticsCacheError):
        replace(binding, **{field: value})


@pytest.mark.parametrize(
    ("requested_backend", "used_backend"),
    (
        ("cpu", "cuda"),
        ("cuda", "cpu"),
        ("cpu", "cpu-fast"),
        ("cpu-fast", "cpu"),
        ("cpu-fast", "cuda"),
        ("cuda", "cpu-fast"),
    ),
)
def test_binding_rejects_impossible_backend_selection(
    requested_backend: str,
    used_backend: str,
) -> None:
    _, _, _, binding = _sample()
    with pytest.raises(DiagnosticsCacheError, match="impossible backend selection"):
        replace(
            binding,
            requested_backend=requested_backend,
            used_backend=used_backend,
            backend_reason="not reachable",
        )


@pytest.mark.parametrize(
    ("requested_backend", "used_backend", "backend_reason"),
    (
        (
            "cpu-fast",
            "cpu-fast",
            "explicit cpu-fast request; self-test passed byte parity",
        ),
        (
            "auto",
            "cpu-fast",
            "CUDA unavailable; cpu-fast startup self-test passed byte parity",
        ),
    ),
)
def test_cpu_fast_backend_selection_round_trips(
    tmp_path: Path,
    requested_backend: str,
    used_backend: str,
    backend_reason: str,
) -> None:
    _, _, result, binding = _sample()
    fast_binding = replace(
        binding,
        requested_backend=requested_backend,
        used_backend=used_backend,
        backend_reason=backend_reason,
    )
    saved = save_diagnostics_cache(
        tmp_path,
        processing_result=result,
        binding=fast_binding,
    )
    loaded = load_diagnostics_cache(
        tmp_path,
        expected_binding=fast_binding,
        expected_manifest_sha256=saved.manifest_sha256,
    )
    assert loaded.binding == fast_binding
    assert (
        read_cache_binding(
            tmp_path,
            expected_manifest_sha256=saved.manifest_sha256,
        )
        == fast_binding
    )


def test_cpu_fast_binding_rejects_noncanonical_reason() -> None:
    _, _, _, binding = _sample()
    with pytest.raises(
        DiagnosticsCacheError, match="canonical backend selection reason"
    ):
        replace(
            binding,
            requested_backend="cpu-fast",
            used_backend="cpu-fast",
            backend_reason="explicit cpu-fast request",
        )
