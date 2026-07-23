"""Regression coverage for the ordinary installed-wheel runtime boundary."""

from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path


HYBRID_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = HYBRID_ROOT.parent


def _run(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, (
        f"command failed ({completed.returncode}): {' '.join(command)}\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    return completed


def test_bound_pyproject_copies_match_build_inputs() -> None:
    assert (CORE_ROOT / "pyproject.toml").read_bytes() == (
        CORE_ROOT / "src" / "portable_digital_ice" / "_source_pyproject.toml"
    ).read_bytes()
    assert (HYBRID_ROOT / "pyproject.toml").read_bytes() == (
        HYBRID_ROOT / "src" / "fauxce_hybrid" / "_source_pyproject.toml"
    ).read_bytes()


def test_wheels_compute_manifests_and_load_schema_without_source_checkout(
    tmp_path: Path,
) -> None:
    core_dist = tmp_path / "core-dist"
    hybrid_dist = tmp_path / "hybrid-dist"
    _run(["uv", "build", "--wheel", "--out-dir", str(core_dist)], cwd=CORE_ROOT)
    _run(
        ["uv", "build", "--wheel", "--out-dir", str(hybrid_dist)],
        cwd=HYBRID_ROOT,
    )
    core_wheel = next(core_dist.glob("*.whl"))
    hybrid_wheel = next(hybrid_dist.glob("*.whl"))

    with zipfile.ZipFile(core_wheel) as archive:
        assert "portable_digital_ice/_source_pyproject.toml" in archive.namelist()
    with zipfile.ZipFile(hybrid_wheel) as archive:
        names = set(archive.namelist())
        assert "fauxce_hybrid/_source_pyproject.toml" in names
        assert "fauxce_hybrid/schemas/fauxce-hybrid-receipt-v2.schema.json" in names

    venv = tmp_path / "venv"
    _run(
        [
            "uv",
            "venv",
            "--python",
            sys.executable,
            "--system-site-packages",
            str(venv),
        ],
        cwd=tmp_path,
    )
    venv_python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    _run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(venv_python),
            "--no-deps",
            str(core_wheel),
            str(hybrid_wheel),
        ],
        cwd=tmp_path,
    )
    smoke = """
from pathlib import Path
import fauxce_hybrid
import portable_digital_ice
from fauxce_hybrid.cache import compute_core_source_manifest
from fauxce_hybrid.receipts import compute_hybrid_source_manifest, load_receipt_schema

site = Path(__import__('sys').argv[1]).resolve()
assert Path(fauxce_hybrid.__file__).resolve().is_relative_to(site)
assert Path(portable_digital_ice.__file__).resolve().is_relative_to(site)
assert len(compute_core_source_manifest()) == 64
assert len(compute_hybrid_source_manifest()) == 64
assert load_receipt_schema()['$schema'] == 'https://json-schema.org/draft/2020-12/schema'
"""
    site_packages = _run(
        [
            str(venv_python),
            "-c",
            "import site; print(site.getsitepackages()[0])",
        ],
        cwd=tmp_path,
    ).stdout.strip()
    dependency_site = next(
        Path(entry)
        for entry in sys.path
        if (Path(entry) / "numpy").is_dir() and (Path(entry) / "jsonschema").is_dir()
    )
    (Path(site_packages) / "fauxce-wheel-test-dependencies.pth").write_text(
        str(dependency_site) + "\n",
        encoding="utf-8",
    )
    _run(
        [str(venv_python), "-I", "-c", smoke, site_packages],
        cwd=tmp_path,
    )
