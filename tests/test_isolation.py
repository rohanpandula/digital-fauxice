from __future__ import annotations

from pathlib import Path


def test_runtime_has_no_research_or_vendor_file_dependencies() -> None:
    source = Path(__file__).resolve().parents[1] / "src" / "portable_digital_ice"
    documents = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(source.glob("*.py"))
    )
    forbidden = (
        "from .fixtures",
        "from .bindings",
        "oracle_rgb16",
        "map_allocated_rgb16_oracle",
        "reverse_engineering/",
        ".work-",
        "vendor-pinned",
    )
    assert all(token not in documents for token in forbidden)


def test_distribution_contains_no_binary_assets() -> None:
    root = Path(__file__).resolve().parents[1]
    binary_suffixes = {
        ".bin",
        ".dll",
        ".dng",
        ".ds",
        ".dylib",
        ".exe",
        ".npy",
        ".npz",
        ".pyc",
        ".raw",
        ".so",
        ".tif",
        ".tiff",
    }
    release_paths = (root / "src", root / "tests", root / "docs", root / "evidence")
    assert not [
        path
        for release_path in release_paths
        for path in release_path.rglob("*")
        if (
            path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix.lower() in binary_suffixes
        )
    ]
