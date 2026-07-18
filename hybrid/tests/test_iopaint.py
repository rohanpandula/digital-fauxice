from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import traceback
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable

import imageio.v3 as iio
import numpy as np
import pytest

import fauxce_hybrid.iopaint as iopaint_module
from fauxce_hybrid.iopaint import (
    IOPaintArtifactError,
    IOPaintExecutionError,
    IOPaintLaMaAdapter,
    IOPaintLaMaConfig,
    IOPaintOutputError,
    IOPaintUnavailableError,
    ModelArtifactAttestation,
    measure_model_artifact,
)


_HELP = " ".join(
    (
        "--model",
        "--device",
        "--image",
        "--mask",
        "--output",
        "--config",
        "--model-dir",
        "--no-concat",
    )
)
_FAKE_SOURCE_RELATIVE_NAME = "iopaint/__init__.py"
_FAKE_SOURCE_PAYLOAD = b'"""Synthetic IOPaint package used by adapter tests."""\n'
_FAKE_SOURCE_MANIFEST = hashlib.sha256(
    _FAKE_SOURCE_RELATIVE_NAME.encode("utf-8")
    + b"\0"
    + hashlib.sha256(_FAKE_SOURCE_PAYLOAD).digest()
).hexdigest()


def _make_executable(path: Path, *, interpreter: Path | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    shebang = "/bin/sh" if interpreter is None else os.fspath(interpreter)
    path.write_text(f"#!{shebang}\nexit 99\n", encoding="utf-8")
    path.chmod(0o700)
    return path


def _sample_arrays() -> tuple[np.ndarray, np.ndarray]:
    rows, columns = np.indices((7, 9))
    rgb = np.stack(
        (
            rows * 17 + columns,
            rows * 5 + columns * 11,
            rows * 3 + columns * 7 + 19,
        ),
        axis=2,
    ).astype(np.uint8)
    mask = np.zeros((7, 9), dtype=np.uint8)
    mask[2:6, 3:8] = 255
    return rgb, mask


def _sample_batch(
    count: int = 3,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    rgb, mask = _sample_arrays()
    rgbs = tuple(
        ((rgb.astype(np.uint16) + index * 29) % 256).astype(np.uint8)
        for index in range(count)
    )
    masks = tuple(np.roll(mask, shift=index, axis=1) for index in range(count))
    return rgbs, masks


def _raw_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _make_config(
    tmp_path: Path, *, weights_payload: bytes = b"fake-weights"
) -> IOPaintLaMaConfig:
    python = _make_executable(tmp_path / "runtime" / "bin" / "python")
    iopaint = _make_executable(
        tmp_path / "runtime" / "bin" / "iopaint",
        interpreter=python,
    )
    source = (
        tmp_path
        / "runtime"
        / "lib"
        / "python3.11"
        / "site-packages"
        / "iopaint"
        / "__init__.py"
    )
    source.parent.mkdir(parents=True)
    source.write_bytes(_FAKE_SOURCE_PAYLOAD)
    model_dir = tmp_path / "private-model-cache"
    weights = model_dir / "torch" / "hub" / "checkpoints" / "big-lama.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(weights_payload)
    artifact = measure_model_artifact(
        weights,
        identifier="iopaint/lama/add_big_lama/big-lama.pt",
    )
    temp_parent = tmp_path / "private-runs"
    temp_parent.mkdir(mode=0o700)
    return IOPaintLaMaConfig(
        iopaint_executable=iopaint,
        python_executable=python,
        model_dir=model_dir,
        weights_file=weights,
        artifact=artifact,
        expected_iopaint_source_manifest_sha256=_FAKE_SOURCE_MANIFEST,
        device="cpu",
        thread_count=2,
        seed=17,
        timeout_seconds=30,
        temp_parent=temp_parent,
    )


def _fake_source_path(config: IOPaintLaMaConfig) -> Path:
    return (
        config.python_executable.parent.parent
        / "lib"
        / "python3.11"
        / "site-packages"
        / "iopaint"
        / "__init__.py"
    )


def _runtime_probe_document(**overrides: object) -> dict[str, object]:
    document: dict[str, object] = {
        "cudnn_benchmark": False,
        "cuda_available": False,
        "cuda_device_names": [],
        "cuda_runtime_version": None,
        "cuda_visible_devices": "unset",
        "cudnn_version": None,
        "deterministic_algorithms": False,
        "hip_runtime_version": None,
        "iopaint_source_file_count": 1,
        "iopaint_source_manifest_sha256": _FAKE_SOURCE_MANIFEST,
        "iopaint_version": "1.6.0",
        "mps_available": False,
        "mps_device_name": None,
        "numpy_version": "1.26.4",
        "opencv_version": "4.11.0.86",
        "pillow_version": "9.5.0",
        "platform_machine": "arm64",
        "platform_release": "25.5.0",
        "platform_system": "Darwin",
        "pydantic_version": "2.13.4",
        "python_implementation": "CPython",
        "python_version": "3.11.15",
        "torch_version": "2.13.0",
        "typer_version": "0.27.0",
    }
    document.update(overrides)
    return document


def _runtime_probe_result(
    argv: list[str],
    **overrides: object,
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        argv,
        0,
        stdout=(
            json.dumps(
                _runtime_probe_document(**overrides),
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ),
        stderr="",
    )


def _successful_runner(
    config: IOPaintLaMaConfig,
    *,
    observed: dict[str, object] | None = None,
    output_factory: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    runtime_overrides: dict[str, object] | None = None,
) -> Callable[..., subprocess.CompletedProcess[str]]:
    if observed is None:
        observed = {}
    if runtime_overrides is None:
        runtime_overrides = {}
    runtime_overrides = {
        "iopaint_source_manifest_sha256": (
            config.expected_iopaint_source_manifest_sha256
        ),
        **runtime_overrides,
    }

    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls = observed.setdefault("calls", [])
        assert isinstance(calls, list)
        calls.append((tuple(argv), kwargs))
        assert kwargs["shell"] is False
        assert kwargs["check"] is False
        is_help = argv[0] == os.fspath(config.python_executable) and argv[1:] == [
            "-I",
            "-m",
            "iopaint",
            "run",
            "--help",
        ]
        is_runtime_probe = argv[0] == os.fspath(config.python_executable) and argv[
            1:3
        ] == ["-I", "-c"]
        if is_help or is_runtime_probe:
            assert hasattr(kwargs["stdout"], "write")
            assert kwargs["stderr"] is subprocess.DEVNULL
        else:
            assert kwargs["stdout"] is subprocess.DEVNULL
            assert kwargs["stderr"] is subprocess.DEVNULL

        if is_help:
            return subprocess.CompletedProcess(argv, 0, stdout=_HELP, stderr="")
        if is_runtime_probe:
            assert argv[1:3] == ["-I", "-c"]
            return _runtime_probe_result(argv, **runtime_overrides)

        assert argv[:10] == [
            os.fspath(config.python_executable),
            "-I",
            "-m",
            "iopaint",
            "run",
            "--model",
            "lama",
            "--device",
            config.device,
            "--image",
        ]
        assert argv[-1] == "--no-concat"
        image_dir = Path(argv[argv.index("--image") + 1])
        mask_dir = Path(argv[argv.index("--mask") + 1])
        output_dir = Path(argv[argv.index("--output") + 1])
        config_path = Path(argv[argv.index("--config") + 1])
        model_dir = Path(argv[argv.index("--model-dir") + 1])
        cwd = Path(kwargs["cwd"])
        environment = kwargs["env"]
        assert isinstance(environment, dict)

        input_paths = tuple(sorted(image_dir.iterdir(), key=lambda path: path.name))
        mask_paths = tuple(sorted(mask_dir.iterdir(), key=lambda path: path.name))
        input_names = tuple(path.name for path in input_paths)
        mask_names = tuple(path.name for path in mask_paths)
        assert input_names == mask_names
        assert input_names
        assert all(path.is_file() for path in (*input_paths, *mask_paths))
        assert all(path.stat().st_mode & 0o777 == 0o600 for path in input_paths)
        assert all(path.stat().st_mode & 0o777 == 0o600 for path in mask_paths)
        request = json.loads(config_path.read_bytes())
        observed.setdefault("input_names", []).append(input_names)
        observed.setdefault("requests", []).append(request)
        observed.setdefault("run_argv", []).append(tuple(argv))
        observed.setdefault("run_cwds", []).append(cwd)
        observed.setdefault("run_envs", []).append(dict(environment))
        observed.setdefault("file_modes", []).append(
            {
                "root": cwd.stat().st_mode & 0o777,
                "image_dir": image_dir.stat().st_mode & 0o777,
                "mask_dir": mask_dir.stat().st_mode & 0o777,
                "output_dir": output_dir.stat().st_mode & 0o777,
                "image": input_paths[0].stat().st_mode & 0o777,
                "mask": mask_paths[0].stat().st_mode & 0o777,
                "config": config_path.stat().st_mode & 0o777,
            }
        )
        assert model_dir == cwd / "model-cache"
        runtime_weights = model_dir / "torch" / "hub" / "checkpoints" / "big-lama.pt"
        assert runtime_weights.read_bytes() == config.weights_file.read_bytes()
        assert Path(environment["LAMA_MODEL_URL"]) == runtime_weights
        assert Path(environment["XDG_CACHE_HOME"]) == model_dir
        for key in (
            "DIFFUSERS_CACHE",
            "HF_HOME",
            "HF_HUB_CACHE",
            "HUGGINGFACE_HUB_CACHE",
            "TORCH_HOME",
            "TRANSFORMERS_CACHE",
        ):
            assert Path(environment[key]).is_relative_to(model_dir)
        assert runtime_weights != config.weights_file.resolve()
        assert model_dir != config.model_dir.resolve()
        assert sorted(
            path.relative_to(model_dir).as_posix()
            for path in model_dir.rglob("*")
            if path.is_file()
        ) == ["torch/hub/checkpoints/big-lama.pt"]
        for name, image_path, mask_path in zip(
            input_names, input_paths, mask_paths, strict=True
        ):
            input_rgb = iio.imread(image_path, plugin="pillow")
            input_mask = iio.imread(mask_path, plugin="pillow")
            observed.setdefault("input_rgbs", []).append(input_rgb.copy())
            observed.setdefault("input_masks", []).append(input_mask.copy())
            output = (
                input_rgb.copy()
                if output_factory is None
                else output_factory(input_rgb, input_mask)
            )
            iio.imwrite(
                output_dir / name,
                output,
                extension=".png",
                plugin="pillow",
            )
        return subprocess.CompletedProcess(argv, 0, stdout="done", stderr="")

    return run


def test_success_writes_exact_private_pngs_and_records_safe_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path)
    rgb, mask = _sample_arrays()
    original_weights = config.weights_file.read_bytes()
    unrelated_cache_file = config.model_dir / "stable_diffusion" / "unrelated.ckpt"
    unrelated_cache_file.parent.mkdir()
    unrelated_cache_file.write_bytes(b"must-never-be-scanned")
    secret_python_path = tmp_path / "must-not-be-imported"
    monkeypatch.setenv("PYTHONPATH", os.fspath(secret_python_path))
    monkeypatch.setenv("PYTHONHOME", os.fspath(tmp_path / "poisoned-home"))
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "must-not-leak-to-cpu")
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        subprocess,
        "run",
        _successful_runner(
            config,
            observed=observed,
            output_factory=lambda image, _mask: 255 - image,
        ),
    )

    adapter = IOPaintLaMaAdapter(config)
    output = adapter(rgb, mask)

    np.testing.assert_array_equal(output, 255 - rgb)
    np.testing.assert_array_equal(observed["input_rgbs"][0], rgb)
    np.testing.assert_array_equal(observed["input_masks"][0], mask)
    assert set(np.unique(observed["input_masks"][0])) == {0, 255}
    assert observed["requests"] == [{"hd_strategy": "Original", "sd_seed": 17}]
    assert observed["file_modes"] == [
        {
            "root": 0o700,
            "image_dir": 0o700,
            "mask_dir": 0o700,
            "output_dir": 0o700,
            "image": 0o600,
            "mask": 0o600,
            "config": 0o600,
        }
    ]
    assert output.flags.c_contiguous
    assert not output.flags.writeable
    with pytest.raises(ValueError):
        output[0, 0, 0] = 0

    metadata = adapter.last_invocation
    assert metadata is not None
    assert metadata.runtime.tool_version == "1.6.0"
    assert metadata.runtime.iopaint_source_manifest_sha256 == _FAKE_SOURCE_MANIFEST
    assert metadata.runtime.iopaint_source_file_count == 1
    assert metadata.runtime.torch_version == "2.13.0"
    assert metadata.runtime.numpy_version == "1.26.4"
    assert metadata.runtime.pillow_version == "9.5.0"
    assert metadata.runtime.opencv_version == "4.11.0.86"
    assert metadata.runtime.pydantic_version == "2.13.4"
    assert metadata.runtime.typer_version == "0.27.0"
    assert metadata.runtime.python_version == "3.11.15"
    assert metadata.runtime.python_implementation == "CPython"
    assert metadata.runtime.platform_system == "Darwin"
    assert metadata.runtime.platform_machine == "arm64"
    assert metadata.runtime.deterministic_algorithms is False
    assert metadata.runtime.cudnn_benchmark is False
    assert metadata.runtime.cuda_available is False
    assert metadata.runtime.cuda_runtime_version is None
    assert metadata.runtime.cudnn_version is None
    assert metadata.runtime.hip_runtime_version is None
    assert metadata.runtime.cuda_device_names == ()
    assert metadata.runtime.cuda_visible_devices == "unset"
    assert metadata.runtime.mps_available is False
    assert metadata.runtime.mps_device_name is None
    assert len(metadata.runtime.effective_environment_sha256) == 64
    assert metadata.runtime.device == "cpu"
    assert metadata.runtime.thread_count == 2
    assert metadata.runtime.seed == 17
    assert metadata.runtime.model_weights_sha256 == config.artifact.sha256
    assert metadata.sanitized_argv == (
        "python",
        "-I",
        "-m",
        "iopaint",
        "run",
        "--model",
        "lama",
        "--device",
        "cpu",
        "--image",
        "<private-input-dir>",
        "--mask",
        "<private-mask-dir>",
        "--output",
        "<private-output-dir>",
        "--config",
        "<private-config-json>",
        "--model-dir",
        "<model-cache-dir>",
        "--no-concat",
    )
    environment = dict(metadata.deterministic_environment)
    assert environment["OMP_NUM_THREADS"] == "2"
    assert environment["MKL_NUM_THREADS"] == "2"
    assert "PYTHONHASHSEED" not in environment
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert "CUBLAS_WORKSPACE_CONFIG" not in environment
    inference_environment = observed["run_envs"][0]
    assert "PYTHONPATH" not in inference_environment
    assert "PYTHONHOME" not in inference_environment
    assert "CUBLAS_WORKSPACE_CONFIG" not in inference_environment
    assert config.weights_file.read_bytes() == original_weights
    assert unrelated_cache_file.read_bytes() == b"must-never-be-scanned"

    serialized = json.dumps(asdict(metadata), sort_keys=True)
    assert os.fspath(tmp_path) not in serialized
    assert os.fspath(config.iopaint_executable) not in serialized
    assert os.fspath(config.weights_file) not in serialized
    assert list(config.temp_parent.iterdir()) == []


def test_subprocess_environment_is_minimal_private_and_secret_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path)
    rgb, mask = _sample_arrays()
    sensitive_keys = {
        "ALL_PROXY",
        "AWS_SECRET_ACCESS_KEY",
        "AZURE_CLIENT_SECRET",
        "DYLD_INSERT_LIBRARIES",
        "DYLD_LIBRARY_PATH",
        "GITHUB_TOKEN",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "LD_PRELOAD",
        "NO_PROXY",
        "SSH_AUTH_SOCK",
    }
    for key in sensitive_keys:
        monkeypatch.setenv(key, f"sentinel-{key.lower()}")
    monkeypatch.setenv("HOME", "sentinel-host-home")
    monkeypatch.setenv("USERPROFILE", "sentinel-user-profile")
    monkeypatch.setenv("PATH", "sentinel-host-path")
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        subprocess,
        "run",
        _successful_runner(config, observed=observed),
    )

    IOPaintLaMaAdapter(config)(rgb, mask)

    calls = observed["calls"]
    assert isinstance(calls, list)
    assert len(calls) == 3
    for _argv, kwargs in calls:
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        assert sensitive_keys.isdisjoint(environment)
        assert environment["PATH"] == os.defpath
        cwd = Path(kwargs["cwd"])
        assert Path(environment["HOME"]).is_relative_to(cwd)
        assert Path(environment["USERPROFILE"]).is_relative_to(cwd)
        for key in ("TEMP", "TMP", "TMPDIR"):
            assert Path(environment[key]).is_relative_to(cwd)
        assert "sentinel" not in json.dumps(environment, sort_keys=True)
    assert list(config.temp_parent.iterdir()) == []


@pytest.mark.parametrize(
    "visibility",
    (
        "-1",
        "0,2",
        "GPU-deadbeef",
        "MIG-deadbeef",
        "MIG-c7384736-a75d-5afc-978f-d2f1294409fd",
        "MIG-GPU-8932f937-d72c-4106-c12f-20bd9faed9f6/1/2",
    ),
)
def test_cuda_visibility_accepts_only_documented_token_shapes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    visibility: str,
) -> None:
    config = _make_config(tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visibility)
    monkeypatch.setattr(
        subprocess,
        "run",
        _successful_runner(
            config,
            runtime_overrides={"cuda_visible_devices": visibility},
        ),
    )

    runtime = IOPaintLaMaAdapter(config).probe()

    assert runtime.cuda_visible_devices == visibility


@pytest.mark.parametrize(
    "visibility",
    (
        "Users/alice/private-model-cache",
        "0, 1",
        "GPU-deadbeef/private",
        "not-a-cuda-device-token",
    ),
)
def test_unsafe_cuda_visibility_fails_before_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    visibility: str,
) -> None:
    config = _make_config(tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visibility)
    called = False

    def forbidden(*_args: object, **_kwargs: object) -> None:
        nonlocal called
        called = True
        raise AssertionError("unsafe CUDA visibility must fail before subprocess")

    monkeypatch.setattr(subprocess, "run", forbidden)

    with pytest.raises(IOPaintUnavailableError, match="unsupported token syntax"):
        IOPaintLaMaAdapter(config).probe()
    assert not called
    assert list(config.temp_parent.iterdir()) == []


def test_runtime_cannot_misreport_cuda_visibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        subprocess,
        "run",
        _successful_runner(
            config,
            runtime_overrides={"cuda_visible_devices": "1"},
        ),
    )

    with pytest.raises(IOPaintUnavailableError, match="mismatched CUDA visibility"):
        IOPaintLaMaAdapter(config).probe()


def test_probe_output_is_file_backed_and_byte_limited(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path)

    def oversized_runtime_probe(
        argv: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        output_handle = kwargs["stdout"]
        assert hasattr(output_handle, "write")
        assert callable(kwargs["preexec_fn"])
        if argv[-1] == "--help":
            output_handle.write(_HELP.encode("utf-8"))
        else:
            output_handle.write(b"x" * (iopaint_module._PROBE_OUTPUT_MAX_BYTES + 1))
        output_handle.flush()
        return subprocess.CompletedProcess(argv, 0, stdout=None, stderr=None)

    monkeypatch.setattr(subprocess, "run", oversized_runtime_probe)

    with pytest.raises(IOPaintUnavailableError, match="exceeded byte limit"):
        IOPaintLaMaAdapter(config).probe()
    assert list(config.temp_parent.iterdir()) == []


@pytest.mark.skipif(os.name != "posix", reason="RLIMIT_FSIZE is a POSIX primitive")
def test_probe_output_cap_is_enforced_inside_the_real_child(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    adapter = IOPaintLaMaAdapter(config)
    writer = (
        "import os\n"
        f"remaining = {iopaint_module._PROBE_OUTPUT_MAX_BYTES + 4096}\n"
        "chunk = b'x' * 65536\n"
        "while remaining:\n"
        "    written = os.write(1, chunk[:remaining])\n"
        "    remaining -= written\n"
    )

    with pytest.raises(IOPaintUnavailableError, match="exceeded byte limit"):
        adapter._run_probe_command(
            [sys.executable, "-I", "-c", writer],
            cuda_visible_devices="unset",
            failure_message="synthetic oversized probe failed",
        )

    assert list(config.temp_parent.iterdir()) == []


def test_probe_fails_closed_when_hard_output_limit_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path)
    called = False

    def forbidden(*_args: object, **_kwargs: object) -> None:
        nonlocal called
        called = True
        raise AssertionError("subprocess must not run without a hard output cap")

    monkeypatch.setattr(iopaint_module, "_resource", None)
    monkeypatch.setattr(subprocess, "run", forbidden)

    with pytest.raises(IOPaintUnavailableError, match="hard IOPaint probe-output"):
        IOPaintLaMaAdapter(config).probe()

    assert not called
    assert list(config.temp_parent.iterdir()) == []


def test_external_source_anchor_mismatch_fails_before_help_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(
        _make_config(tmp_path),
        expected_iopaint_source_manifest_sha256="c" * 64,
    )
    called = False

    def forbidden(*_args: object, **_kwargs: object) -> None:
        nonlocal called
        called = True
        raise AssertionError("mismatched source must fail before subprocess")

    monkeypatch.setattr(subprocess, "run", forbidden)

    with pytest.raises(IOPaintUnavailableError, match="required trust anchor"):
        IOPaintLaMaAdapter(config).probe()

    assert not called
    assert list(config.temp_parent.iterdir()) == []


def test_runtime_source_report_is_only_a_cross_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path)
    monkeypatch.setattr(
        subprocess,
        "run",
        _successful_runner(
            config,
            runtime_overrides={"iopaint_source_manifest_sha256": "c" * 64},
        ),
    )

    with pytest.raises(IOPaintUnavailableError, match="runtime source report"):
        IOPaintLaMaAdapter(config).probe()


def test_source_mutation_before_inference_fails_before_model_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path)
    rgb, mask = _sample_arrays()
    observed: dict[str, object] = {}
    real_write = iopaint_module._write_private_png
    mutated = False

    def mutate_after_first_input(path: Path, array: np.ndarray) -> None:
        nonlocal mutated
        real_write(path, array)
        if not mutated:
            _fake_source_path(config).write_bytes(b"changed before inference\n")
            mutated = True

    monkeypatch.setattr(iopaint_module, "_write_private_png", mutate_after_first_input)
    monkeypatch.setattr(
        subprocess,
        "run",
        _successful_runner(config, observed=observed),
    )
    adapter = IOPaintLaMaAdapter(config)

    with pytest.raises(IOPaintUnavailableError, match="before IOPaint execution"):
        adapter(rgb, mask)

    assert len(observed["calls"]) == 2
    assert adapter.invocations == ()
    assert list(config.temp_parent.iterdir()) == []


def test_source_set_mutation_during_inference_invalidates_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path)
    rgb, mask = _sample_arrays()
    base_runner = _successful_runner(config)

    def mutate_after_inference(
        argv: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        result = base_runner(argv, **kwargs)
        if argv[-1] == "--no-concat":
            _fake_source_path(config).with_name("injected.py").write_bytes(
                b"# changed during inference\n"
            )
        return result

    monkeypatch.setattr(subprocess, "run", mutate_after_inference)
    adapter = IOPaintLaMaAdapter(config)

    with pytest.raises(IOPaintUnavailableError, match="during IOPaint execution"):
        adapter(rgb, mask)

    assert adapter.invocations == ()
    assert list(config.temp_parent.iterdir()) == []


def test_batch_runs_once_preserves_order_and_records_each_crop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path)
    rgbs, masks = _sample_batch()
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        subprocess,
        "run",
        _successful_runner(
            config,
            observed=observed,
            output_factory=lambda image, _mask: 255 - image,
        ),
    )
    adapter = IOPaintLaMaAdapter(config)

    outputs = adapter.inpaint_batch(rgbs, masks)

    assert isinstance(outputs, tuple)
    assert len(outputs) == len(rgbs)
    for output, rgb in zip(outputs, rgbs, strict=True):
        np.testing.assert_array_equal(output, 255 - rgb)
        assert output.flags.c_contiguous
        assert not output.flags.writeable
    assert observed["input_names"] == [
        (
            "crop-00000000.png",
            "crop-00000001.png",
            "crop-00000002.png",
        )
    ]
    for observed_rgb, expected_rgb in zip(observed["input_rgbs"], rgbs, strict=True):
        np.testing.assert_array_equal(observed_rgb, expected_rgb)
    for observed_mask, expected_mask in zip(
        observed["input_masks"], masks, strict=True
    ):
        np.testing.assert_array_equal(observed_mask, expected_mask)
    assert len(observed["calls"]) == 3
    assert len(observed["run_argv"]) == 1
    assert observed["calls"][-1][1]["timeout"] == 90.0
    assert observed["requests"] == [{"hd_strategy": "Original", "sd_seed": 17}]

    metadata = adapter.invocations
    assert len(metadata) == len(rgbs)
    assert adapter.last_invocation is metadata[-1]
    assert all(item.runtime is metadata[0].runtime for item in metadata)
    assert all(item.sanitized_argv is metadata[0].sanitized_argv for item in metadata)
    assert all(
        item.deterministic_environment is metadata[0].deterministic_environment
        for item in metadata
    )
    assert all(item.config_document is metadata[0].config_document for item in metadata)
    for item, rgb, mask, output in zip(metadata, rgbs, masks, outputs, strict=True):
        assert item.input_rgb8_raw_sha256 == _raw_sha256(rgb)
        assert item.mask_u8_raw_sha256 == _raw_sha256(mask)
        assert item.output_rgb8_raw_sha256 == _raw_sha256(output)
    assert list(config.temp_parent.iterdir()) == []


def test_batch_validates_later_item_before_model_or_subprocess_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path)
    rgbs, masks = _sample_batch()
    config.weights_file.unlink()
    called = False

    def forbidden(*_args: object, **_kwargs: object) -> None:
        nonlocal called
        called = True
        raise AssertionError("subprocess must not run")

    monkeypatch.setattr(subprocess, "run", forbidden)
    invalid_masks = (masks[0], masks[1].astype(bool), masks[2])
    adapter = IOPaintLaMaAdapter(config)

    with pytest.raises(TypeError, match="batch item 1: component_mask"):
        adapter.inpaint_batch(rgbs, invalid_masks)

    assert not called
    assert adapter.invocations == ()
    assert list(config.temp_parent.iterdir()) == []


def test_batch_rejects_empty_or_mismatched_sequences_before_model_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path)
    rgb, mask = _sample_arrays()
    config.weights_file.unlink()
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("subprocess must not run")
        ),
    )
    adapter = IOPaintLaMaAdapter(config)

    with pytest.raises(ValueError, match="at least one"):
        adapter.inpaint_batch((), ())
    with pytest.raises(ValueError, match="equal length"):
        adapter.inpaint_batch((rgb,), (mask, mask))

    assert adapter.invocations == ()


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing", "1 missing, 0 unexpected"),
        ("extra", "0 missing, 1 unexpected"),
        ("renamed", "1 missing, 1 unexpected"),
    ),
)
def test_batch_rejects_missing_extra_or_renamed_outputs_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    config = _make_config(tmp_path)
    rgbs, masks = _sample_batch()
    base_runner = _successful_runner(config)

    def mutate_output_set(
        argv: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        result = base_runner(argv, **kwargs)
        if argv[-1] == "--no-concat":
            output_dir = Path(argv[argv.index("--output") + 1])
            if mutation == "missing":
                (output_dir / "crop-00000001.png").unlink()
            elif mutation == "extra":
                (output_dir / "unexpected-output.txt").write_bytes(b"unexpected")
            else:
                (output_dir / "crop-00000001.png").rename(
                    output_dir / "unexpected-output.png"
                )
        return result

    monkeypatch.setattr(subprocess, "run", mutate_output_set)
    adapter = IOPaintLaMaAdapter(config)

    with pytest.raises(IOPaintOutputError, match=message) as raised:
        adapter.inpaint_batch(rgbs, masks)

    rendered = "".join(
        traceback.format_exception(
            type(raised.value), raised.value, raised.value.__traceback__
        )
    )
    assert os.fspath(tmp_path) not in rendered
    assert adapter.invocations == ()
    assert list(config.temp_parent.iterdir()) == []


def test_batch_rejects_later_output_shape_mismatch_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path)
    rgbs, masks = _sample_batch()
    output_index = 0

    def malformed_later_output(image: np.ndarray, _mask: np.ndarray) -> np.ndarray:
        nonlocal output_index
        current = output_index
        output_index += 1
        return image[..., 0] if current == 1 else image

    monkeypatch.setattr(
        subprocess,
        "run",
        _successful_runner(config, output_factory=malformed_later_output),
    )
    adapter = IOPaintLaMaAdapter(config)

    with pytest.raises(IOPaintOutputError, match="item 1 must have shape"):
        adapter.inpaint_batch(rgbs, masks)

    assert adapter.invocations == ()
    assert list(config.temp_parent.iterdir()) == []


def test_single_crop_call_delegates_to_batch_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path)
    rgb, mask = _sample_arrays()
    expected = np.full_like(rgb, 73)
    adapter = IOPaintLaMaAdapter(config)

    def fake_batch(
        rgb_crops: tuple[np.ndarray, ...],
        component_masks: tuple[np.ndarray, ...],
    ) -> tuple[np.ndarray, ...]:
        assert len(rgb_crops) == 1 and rgb_crops[0] is rgb
        assert len(component_masks) == 1 and component_masks[0] is mask
        return (expected,)

    monkeypatch.setattr(adapter, "inpaint_batch", fake_batch)

    assert adapter(rgb, mask) is expected


def test_actual_argv_order_environment_and_sanitized_order_are_repeatable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path)
    rgb, mask = _sample_arrays()
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        subprocess,
        "run",
        _successful_runner(config, observed=observed),
    )
    adapter = IOPaintLaMaAdapter(config)

    first = adapter(rgb, mask)
    second = adapter(rgb, mask)

    np.testing.assert_array_equal(first, second)
    run_argv = observed["run_argv"]
    assert len(run_argv) == 2
    for argv in run_argv:
        assert argv[:5] == (
            os.fspath(config.python_executable),
            "-I",
            "-m",
            "iopaint",
            "run",
        )
        assert argv[5::2] == (
            "--model",
            "--device",
            "--image",
            "--mask",
            "--output",
            "--config",
            "--model-dir",
            "--no-concat",
        )
    assert (
        adapter.invocations[0].sanitized_argv == adapter.invocations[1].sanitized_argv
    )
    # Each fresh process is re-probed: two help, metadata, and inference calls.
    assert len(observed["calls"]) == 6
    assert list(config.temp_parent.iterdir()) == []


def test_cuda_device_runtime_and_visibility_are_fingerprinted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(_make_config(tmp_path), device="cuda")
    rgb, mask = _sample_arrays()
    observed: dict[str, object] = {}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-deadbeef")
    monkeypatch.setattr(
        subprocess,
        "run",
        _successful_runner(
            config,
            observed=observed,
            runtime_overrides={
                "cudnn_benchmark": True,
                "cuda_available": True,
                "cuda_device_names": ["NVIDIA RTX 4090"],
                "cuda_runtime_version": "12.8",
                "cuda_visible_devices": "GPU-deadbeef",
                "cudnn_version": 90100,
                "deterministic_algorithms": True,
            },
        ),
    )

    adapter = IOPaintLaMaAdapter(config)
    adapter(rgb, mask)

    runtime = adapter.invocations[0].runtime
    assert runtime.device == "cuda"
    assert runtime.cuda_available is True
    assert runtime.cuda_runtime_version == "12.8"
    assert runtime.cudnn_version == "90100"
    assert runtime.hip_runtime_version is None
    assert runtime.cuda_device_names == ("NVIDIA RTX 4090",)
    assert runtime.cuda_visible_devices == "GPU-deadbeef"
    assert runtime.deterministic_algorithms is True
    assert runtime.cudnn_benchmark is True
    assert runtime.mps_available is False
    metadata = adapter.last_invocation
    assert metadata is not None
    assert (
        dict(metadata.deterministic_environment)["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    )
    probe_calls = observed["calls"][:2]
    for _argv, kwargs in probe_calls:
        assert "CUBLAS_WORKSPACE_CONFIG" not in kwargs["env"]
    assert observed["run_envs"][0]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_rocm_runtime_cannot_be_mislabeled_as_cuda(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(_make_config(tmp_path), device="cuda")
    rgb, mask = _sample_arrays()
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        subprocess,
        "run",
        _successful_runner(
            config,
            observed=observed,
            runtime_overrides={
                "cuda_available": True,
                "cuda_device_names": ["AMD Radeon PRO W7900"],
                "cuda_runtime_version": None,
                "hip_runtime_version": "6.4.43483",
            },
        ),
    )
    adapter = IOPaintLaMaAdapter(config)

    with pytest.raises(IOPaintUnavailableError, match="ROCm/HIP"):
        adapter(rgb, mask)

    assert len(observed["calls"]) == 2
    for _argv, kwargs in observed["calls"]:
        assert "CUBLAS_WORKSPACE_CONFIG" not in kwargs["env"]
    assert adapter.invocations == ()


def test_mps_capability_and_name_are_fingerprinted_without_claiming_mps_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path)
    rgb, mask = _sample_arrays()
    observed: dict[str, object] = {}
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "must-not-leak-to-mps")
    monkeypatch.setattr(
        subprocess,
        "run",
        _successful_runner(
            config,
            observed=observed,
            runtime_overrides={
                "mps_available": True,
                "mps_device_name": "Apple M4 Max",
            },
        ),
    )

    adapter = IOPaintLaMaAdapter(config)
    adapter(rgb, mask)

    metadata = adapter.last_invocation
    assert metadata is not None
    runtime = metadata.runtime
    assert runtime.device == "cpu"
    assert runtime.mps_available is True
    assert runtime.mps_device_name == "Apple M4 Max"
    assert runtime.deterministic_algorithms is False
    assert runtime.cudnn_benchmark is False
    assert "CUBLAS_WORKSPACE_CONFIG" not in dict(metadata.deterministic_environment)
    for _argv, kwargs in observed["calls"]:
        assert "CUBLAS_WORKSPACE_CONFIG" not in kwargs["env"]


def test_mps_request_rejects_iopaint_lama_silent_cpu_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(_make_config(tmp_path), device="mps")
    rgb, mask = _sample_arrays()
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        subprocess,
        "run",
        _successful_runner(
            config,
            observed=observed,
            runtime_overrides={
                "mps_available": True,
                "mps_device_name": "Apple M4 Max",
            },
        ),
    )
    adapter = IOPaintLaMaAdapter(config)

    with pytest.raises(IOPaintUnavailableError, match="routes LaMa MPS.*CPU"):
        adapter(rgb, mask)

    # Help and runtime capability probes ran; inference never did.
    assert len(observed["calls"]) == 2
    assert adapter.invocations == ()
    assert list(config.temp_parent.iterdir()) == []


def test_mps_request_fails_closed_when_runtime_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(_make_config(tmp_path), device="mps")
    rgb, mask = _sample_arrays()
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        subprocess,
        "run",
        _successful_runner(config, observed=observed),
    )
    adapter = IOPaintLaMaAdapter(config)

    with pytest.raises(IOPaintUnavailableError, match="MPS was requested"):
        adapter(rgb, mask)

    assert len(observed["calls"]) == 2
    assert adapter.invocations == ()
    assert list(config.temp_parent.iterdir()) == []


@pytest.mark.parametrize(
    "runtime_override",
    (
        {"mps_available": True},
        {"deterministic_algorithms": True},
        {"cudnn_benchmark": True},
        {"hip_runtime_version": "6.4.43483"},
    ),
)
def test_mps_and_determinism_runtime_facts_affect_environment_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime_override: dict[str, object],
) -> None:
    config = _make_config(tmp_path)
    monkeypatch.setattr(subprocess, "run", _successful_runner(config))
    baseline = IOPaintLaMaAdapter(config).probe()
    monkeypatch.setattr(
        subprocess,
        "run",
        _successful_runner(config, runtime_overrides=runtime_override),
    )
    changed = IOPaintLaMaAdapter(config).probe()

    assert changed.effective_environment_sha256 != (
        baseline.effective_environment_sha256
    )


@pytest.mark.parametrize(
    ("runtime_override", "message"),
    (
        ({"mps_available": 1}, "invalid MPS state"),
        ({"mps_available": True, "mps_device_name": "/private/gpu"}, "unsafe MPS"),
        ({"mps_device_name": "Apple M4"}, "inconsistent MPS"),
        ({"deterministic_algorithms": 1}, "deterministic-algorithms"),
        ({"cudnn_benchmark": 0}, "cuDNN benchmark"),
        ({"hip_runtime_version": 64}, "invalid HIP version"),
    ),
)
def test_mps_and_determinism_runtime_facts_fail_closed_when_malformed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime_override: dict[str, object],
    message: str,
) -> None:
    config = _make_config(tmp_path)
    monkeypatch.setattr(
        subprocess,
        "run",
        _successful_runner(config, runtime_overrides=runtime_override),
    )

    with pytest.raises(IOPaintUnavailableError, match=message):
        IOPaintLaMaAdapter(config).probe()


def test_config_accepts_only_official_iopaint_devices(tmp_path: Path) -> None:
    config = _make_config(tmp_path)

    for device in ("cpu", "cuda", "mps"):
        assert replace(config, device=device).device == device
    with pytest.raises(ValueError, match="'cpu', 'cuda', or 'mps'"):
        replace(config, device="metal")


def test_missing_iopaint_fails_closed_before_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path)
    missing = tmp_path / "does-not-exist" / "iopaint"
    config = IOPaintLaMaConfig(
        **{
            **config.__dict__,
            "iopaint_executable": missing,
        }
    )
    called = False

    def forbidden(*_args: object, **_kwargs: object) -> None:
        nonlocal called
        called = True
        raise AssertionError("subprocess must not run")

    monkeypatch.setattr(subprocess, "run", forbidden)
    rgb, mask = _sample_arrays()

    with pytest.raises(IOPaintUnavailableError, match="IOPaint executable"):
        IOPaintLaMaAdapter(config)(rgb, mask)
    assert not called
    assert list(config.temp_parent.iterdir()) == []


def test_construction_is_inert_for_empty_routing() -> None:
    config = IOPaintLaMaConfig(
        iopaint_executable=Path("/missing/iopaint"),
        python_executable=Path("/missing/python"),
        model_dir=Path("/missing/model-cache"),
        weights_file=Path("/missing/model-cache/torch/hub/checkpoints/big-lama.pt"),
        artifact=ModelArtifactAttestation(
            identifier="iopaint/lama/big-lama.pt",
            sha256="a" * 64,
        ),
        expected_iopaint_source_manifest_sha256="b" * 64,
    )

    adapter = IOPaintLaMaAdapter(config)

    assert adapter.invocations == ()
    assert adapter.last_invocation is None


def test_cli_and_version_probe_must_use_the_same_python_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path)
    other_python = _make_executable(tmp_path / "other-runtime" / "bin" / "python")
    mismatched = IOPaintLaMaConfig(
        **{
            **config.__dict__,
            "python_executable": other_python,
        }
    )
    called = False

    def forbidden(*_args: object, **_kwargs: object) -> None:
        nonlocal called
        called = True
        raise AssertionError("subprocess must not run")

    monkeypatch.setattr(subprocess, "run", forbidden)
    rgb, mask = _sample_arrays()

    with pytest.raises(IOPaintUnavailableError, match="Python runtimes differ"):
        IOPaintLaMaAdapter(mismatched)(rgb, mask)
    assert not called


def test_nonzero_exit_is_sanitized_and_private_tree_is_cleaned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path)

    def run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        if argv[-1] == "--help":
            return subprocess.CompletedProcess(argv, 0, stdout=_HELP, stderr="")
        if argv[1:3] == ["-I", "-c"]:
            return _runtime_probe_result(argv)
        cwd = Path(kwargs["cwd"])
        return subprocess.CompletedProcess(
            argv,
            23,
            stdout="",
            stderr=(
                f"failure under {cwd}/input and {config.model_dir}/torch\n"
                f"Traceback: {config.python_executable.parent.parent}/lib/"
                "python3.11/site-packages/iopaint/model/lama.py"
            ),
        )

    monkeypatch.setattr(subprocess, "run", run)
    rgb, mask = _sample_arrays()

    with pytest.raises(IOPaintExecutionError, match="status 23") as raised:
        IOPaintLaMaAdapter(config)(rgb, mask)
    message = str(raised.value)
    rendered_traceback = "".join(
        traceback.format_exception(
            type(raised.value), raised.value, raised.value.__traceback__
        )
    )
    assert os.fspath(tmp_path) not in message
    assert os.fspath(tmp_path) not in rendered_traceback
    assert message == (
        "IOPaint exited with status 23; subprocess output was withheld to protect "
        "private paths"
    )
    assert "site-packages" not in rendered_traceback
    assert list(config.temp_parent.iterdir()) == []


def test_timeout_traceback_suppresses_private_argv_and_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path)

    def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if argv[-1] == "--help":
            return subprocess.CompletedProcess(argv, 0, stdout=_HELP, stderr="")
        if argv[1:3] == ["-I", "-c"]:
            return _runtime_probe_result(argv)
        raise subprocess.TimeoutExpired(argv, timeout=30, output=str(argv))

    monkeypatch.setattr(subprocess, "run", run)
    rgb, mask = _sample_arrays()

    with pytest.raises(IOPaintExecutionError, match="timed out") as raised:
        IOPaintLaMaAdapter(config)(rgb, mask)
    rendered = "".join(
        traceback.format_exception(
            type(raised.value), raised.value, raised.value.__traceback__
        )
    )
    assert os.fspath(tmp_path) not in rendered
    assert list(config.temp_parent.iterdir()) == []


def test_missing_output_fails_closed_and_cleans_private_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path)

    def run(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if argv[-1] == "--help":
            return subprocess.CompletedProcess(argv, 0, stdout=_HELP, stderr="")
        if argv[1:3] == ["-I", "-c"]:
            return _runtime_probe_result(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    rgb, mask = _sample_arrays()

    with pytest.raises(IOPaintOutputError, match="1 missing, 0 unexpected"):
        IOPaintLaMaAdapter(config)(rgb, mask)
    assert list(config.temp_parent.iterdir()) == []


@pytest.mark.parametrize("malformation", ("shape", "dtype"))
def test_malformed_output_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    malformation: str,
) -> None:
    config = _make_config(tmp_path)
    rgb, mask = _sample_arrays()

    def malformed(image: np.ndarray, _mask: np.ndarray) -> np.ndarray:
        if malformation == "shape":
            return image[..., 0]
        return image[..., 0].astype(np.uint16) * np.uint16(257)

    monkeypatch.setattr(
        subprocess,
        "run",
        _successful_runner(config, output_factory=malformed),
    )

    expected = "shape" if malformation == "shape" else "dtype uint8"
    with pytest.raises(IOPaintOutputError, match=expected):
        IOPaintLaMaAdapter(config)(rgb, mask)
    assert list(config.temp_parent.iterdir()) == []


def test_help_contract_and_runtime_version_are_probed_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path)
    rgb, mask = _sample_arrays()

    def missing_option(
        argv: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=_HELP.replace("--config", ""),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", missing_option)
    with pytest.raises(IOPaintUnavailableError, match="--config"):
        IOPaintLaMaAdapter(config)(rgb, mask)

    calls = 0

    def wrong_version(
        argv: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return subprocess.CompletedProcess(argv, 0, stdout=_HELP, stderr="")
        return _runtime_probe_result(
            argv,
            iopaint_version="1.5.4",
            python_version="3.10.9",
            torch_version="2.1.2",
        )

    monkeypatch.setattr(subprocess, "run", wrong_version)
    with pytest.raises(IOPaintUnavailableError, match="required 1.6.0, got 1.5.4"):
        IOPaintLaMaAdapter(config)(rgb, mask)
    assert list(config.temp_parent.iterdir()) == []


def test_effective_environment_is_reprobed_and_cannot_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path)
    rgb, mask = _sample_arrays()
    base_runner = _successful_runner(config)
    metadata_probes = 0

    def drifting_runtime(
        argv: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        nonlocal metadata_probes
        if argv[1:3] == ["-I", "-c"]:
            metadata_probes += 1
            version = "1.26.4" if metadata_probes == 1 else "1.26.5"
            return _runtime_probe_result(argv, numpy_version=version)
        return base_runner(argv, **kwargs)

    monkeypatch.setattr(subprocess, "run", drifting_runtime)
    adapter = IOPaintLaMaAdapter(config)

    adapter(rgb, mask)
    with pytest.raises(IOPaintUnavailableError, match="environment changed"):
        adapter(rgb, mask)
    assert len(adapter.invocations) == 1
    assert list(config.temp_parent.iterdir()) == []


def test_weights_must_be_exact_cache_file_and_match_attested_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path)
    rgb, mask = _sample_arrays()
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("subprocess must not run")
        ),
    )

    wrong_location = tmp_path / "copied-big-lama.pt"
    wrong_location.write_bytes(config.weights_file.read_bytes())
    wrong_path_config = IOPaintLaMaConfig(
        **{
            **config.__dict__,
            "weights_file": wrong_location,
        }
    )
    with pytest.raises(IOPaintArtifactError, match="exact IOPaint LaMa"):
        IOPaintLaMaAdapter(wrong_path_config)(rgb, mask)

    bad_hash_config = IOPaintLaMaConfig(
        **{
            **config.__dict__,
            "artifact": ModelArtifactAttestation(
                identifier=config.artifact.identifier,
                sha256="0" * 64,
            ),
        }
    )
    with pytest.raises(IOPaintArtifactError, match="does not match"):
        IOPaintLaMaAdapter(bad_hash_config)(rgb, mask)


def test_weights_changed_during_execution_invalidates_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path)
    rgb, mask = _sample_arrays()
    base_runner = _successful_runner(config)

    def mutate_weights(
        argv: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        result = base_runner(argv, **kwargs)
        if argv[-1] == "--no-concat":
            config.weights_file.write_bytes(b"changed-during-run")
        return result

    monkeypatch.setattr(subprocess, "run", mutate_weights)
    adapter = IOPaintLaMaAdapter(config)

    with pytest.raises(IOPaintArtifactError, match="changed during"):
        adapter(rgb, mask)
    assert adapter.invocations == ()
    assert list(config.temp_parent.iterdir()) == []


def test_iopaint_can_delete_only_private_snapshot_not_provenance_weights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path)
    rgb, mask = _sample_arrays()
    original = config.weights_file.read_bytes()
    base_runner = _successful_runner(config)

    def delete_loaded_snapshot(
        argv: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        result = base_runner(argv, **kwargs)
        if argv[-1] == "--no-concat":
            environment = kwargs["env"]
            assert isinstance(environment, dict)
            Path(environment["LAMA_MODEL_URL"]).unlink()
        return result

    monkeypatch.setattr(subprocess, "run", delete_loaded_snapshot)

    with pytest.raises(IOPaintArtifactError, match="cannot read"):
        IOPaintLaMaAdapter(config)(rgb, mask)
    assert config.weights_file.read_bytes() == original
    assert list(config.temp_parent.iterdir()) == []


@pytest.mark.parametrize(
    ("rgb_transform", "mask_transform", "error", "message"),
    (
        (
            lambda value: value.astype(np.uint16),
            lambda value: value,
            TypeError,
            "rgb_crop",
        ),
        (lambda value: value[..., 0], lambda value: value, ValueError, "HxWx3"),
        (
            lambda value: value,
            lambda value: value.astype(bool),
            TypeError,
            "component_mask",
        ),
        (lambda value: value, lambda value: value[:, :-1], ValueError, "matching"),
        (
            lambda value: value,
            lambda value: np.where(value != 0, 127, 0).astype(np.uint8),
            ValueError,
            "only 0 and 255",
        ),
        (
            lambda value: value,
            lambda value: np.zeros_like(value),
            ValueError,
            "at least one",
        ),
    ),
)
def test_input_contract_fails_before_tool_or_model_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rgb_transform: Callable[[np.ndarray], np.ndarray],
    mask_transform: Callable[[np.ndarray], np.ndarray],
    error: type[Exception],
    message: str,
) -> None:
    config = _make_config(tmp_path)
    rgb, mask = _sample_arrays()
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("subprocess must not run")
        ),
    )

    with pytest.raises(error, match=message):
        IOPaintLaMaAdapter(config)(rgb_transform(rgb), mask_transform(mask))


@pytest.mark.parametrize(
    "identifier",
    (
        "/Users/private/big-lama.pt",
        "../big-lama.pt",
        "~/.cache/big-lama.pt",
        r"C:\private\big-lama.pt",
    ),
)
def test_artifact_identifier_rejects_path_leaks(identifier: str) -> None:
    with pytest.raises(ValueError, match="sanitized relative"):
        ModelArtifactAttestation(identifier=identifier, sha256="a" * 64)


@pytest.mark.parametrize("timeout", (float("nan"), float("inf"), float("-inf"), 0.0))
def test_config_rejects_nonfinite_or_nonpositive_timeout(
    tmp_path: Path,
    timeout: float,
) -> None:
    with pytest.raises(ValueError, match="finite and > 0"):
        replace(_make_config(tmp_path), timeout_seconds=timeout)


def test_config_enforces_subprocess_safe_timeout_boundary(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    maximum = iopaint_module._SUBPROCESS_TIMEOUT_MAX_SECONDS

    assert replace(config, timeout_seconds=maximum).timeout_seconds == maximum
    for timeout in (math.nextafter(float(maximum), math.inf), 1e308, 10**1000):
        with pytest.raises(ValueError, match="subprocess-safe limit"):
            replace(config, timeout_seconds=timeout)


def test_multi_item_timeout_product_fails_before_any_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    maximum = iopaint_module._SUBPROCESS_TIMEOUT_MAX_SECONDS
    config = replace(_make_config(tmp_path), timeout_seconds=maximum / 2 + 1)
    rgbs, masks = _sample_batch(2)
    called = False

    def forbidden(*_args: object, **_kwargs: object) -> None:
        nonlocal called
        called = True
        raise AssertionError("invalid batch timeout must fail before subprocess")

    monkeypatch.setattr(subprocess, "run", forbidden)
    adapter = IOPaintLaMaAdapter(config)

    with pytest.raises(IOPaintExecutionError, match="batch timeout"):
        adapter.inpaint_batch(rgbs, masks)

    assert not called
    assert adapter.invocations == ()
    assert list(config.temp_parent.iterdir()) == []


def test_platform_timeout_conversion_overflow_is_sanitized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _make_config(tmp_path)
    rgb, mask = _sample_arrays()
    base_runner = _successful_runner(config)

    def overflow_at_inference(
        argv: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        if argv[-1] == "--no-concat":
            raise OverflowError("private platform conversion detail")
        return base_runner(argv, **kwargs)

    monkeypatch.setattr(subprocess, "run", overflow_at_inference)
    adapter = IOPaintLaMaAdapter(config)

    with pytest.raises(IOPaintExecutionError, match="unsupported by the platform"):
        adapter(rgb, mask)

    assert adapter.invocations == ()
    assert list(config.temp_parent.iterdir()) == []


def test_config_rejects_invalid_iopaint_source_manifest_anchor(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="source manifest SHA-256"):
        replace(
            _make_config(tmp_path),
            expected_iopaint_source_manifest_sha256="B" * 64,
        )


@pytest.mark.skipif(
    os.environ.get("FAUXCE_RUN_IOPAINT_MODEL_SMOKE") != "1",
    reason=(
        "real IOPaint/LaMa smoke requires an explicitly supplied runtime and "
        "external weights artifact"
    ),
)
def test_real_iopaint_lama_smoke_with_explicit_runtime(tmp_path: Path) -> None:
    required = {
        name: os.environ.get(name)
        for name in (
            "FAUXCE_IOPAINT_EXECUTABLE",
            "FAUXCE_IOPAINT_MODEL_DIR",
            "FAUXCE_IOPAINT_PYTHON",
            "FAUXCE_IOPAINT_SOURCE_MANIFEST_SHA256",
            "FAUXCE_IOPAINT_WEIGHTS",
        )
    }
    missing = sorted(name for name, value in required.items() if not value)
    if missing:
        pytest.fail("missing gated smoke configuration: " + ", ".join(missing))
    device = os.environ.get("FAUXCE_IOPAINT_DEVICE", "cpu")
    if device not in ("cpu", "cuda", "mps"):
        pytest.fail("FAUXCE_IOPAINT_DEVICE must be cpu, cuda, or mps")
    if device == "mps":
        pytest.fail(
            "IOPaint 1.6.0 silently routes LaMa mps requests to CPU; use cpu "
            "locally or cuda on a CUDA host"
        )
    weights = Path(required["FAUXCE_IOPAINT_WEIGHTS"])
    artifact = measure_model_artifact(
        weights,
        identifier=os.environ.get(
            "FAUXCE_IOPAINT_ARTIFACT_ID",
            "iopaint/lama/add_big_lama/big-lama.pt",
        ),
    )
    expected_sha256 = os.environ.get("FAUXCE_IOPAINT_WEIGHTS_SHA256")
    if expected_sha256 is not None:
        assert artifact.sha256 == expected_sha256
    config = IOPaintLaMaConfig(
        iopaint_executable=Path(required["FAUXCE_IOPAINT_EXECUTABLE"]),
        python_executable=Path(required["FAUXCE_IOPAINT_PYTHON"]),
        model_dir=Path(required["FAUXCE_IOPAINT_MODEL_DIR"]),
        weights_file=weights,
        artifact=artifact,
        expected_iopaint_source_manifest_sha256=required[
            "FAUXCE_IOPAINT_SOURCE_MANIFEST_SHA256"
        ],
        device=device,
        thread_count=1,
        seed=0,
        temp_parent=tmp_path,
    )
    rows, columns = np.indices((32, 32))
    rgb = np.stack(
        (
            rows * 3 + columns,
            rows + columns * 5,
            rows * 7 + columns * 2,
        ),
        axis=2,
    ).astype(np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[12:20, 12:20] = 255
    second_rgb = np.flip(rgb, axis=1).copy()
    second_mask = np.flip(mask, axis=1).copy()

    adapter = IOPaintLaMaAdapter(config)
    outputs = adapter.inpaint_batch((rgb, second_rgb), (mask, second_mask))

    assert len(outputs) == 2
    assert all(output.shape == rgb.shape for output in outputs)
    assert all(output.dtype == np.uint8 for output in outputs)
    assert len(adapter.invocations) == 2
    assert adapter.last_invocation is not None
