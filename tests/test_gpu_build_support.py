"""Unit tests exercising the production CUDA build-support functions."""

import hashlib
import importlib
import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.cpu, pytest.mark.unit]


@pytest.fixture
def support():
    source = Path(__file__).resolve().parents[1] / "build_support"
    if not source.exists():
        pytest.skip("Build-support sources are not part of the runtime wheel")
    return (
        importlib.import_module("localize_psf_build_support.cmake"),
        importlib.import_module("localize_psf_build_support.provision"),
    )


def test_download_checks_real_file_integrity(support, tmp_path):
    _, provision = support
    source = tmp_path / "source.bin"
    source.write_bytes(b"CUDA build dependency")
    destination = tmp_path / "download.bin"
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    provision._download(source.as_uri(), destination, expected)
    assert destination.read_bytes() == source.read_bytes()


def test_download_rejects_corruption_without_replacing_existing_file(support, tmp_path):
    _, provision = support
    source = tmp_path / "source.bin"
    source.write_bytes(b"corrupted dependency")
    destination = tmp_path / "download.bin"
    destination.write_bytes(b"previous verified dependency")
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        provision._download(source.as_uri(), destination, "0" * 64)
    assert destination.read_bytes() == b"previous verified dependency"
    assert not destination.with_suffix(".partial").exists()


def test_installation_lock_can_be_reacquired(support, tmp_path):
    _, provision = support
    filename = tmp_path / "install.lock"
    with provision._installation_lock(filename):
        assert filename.is_file()
    with provision._installation_lock(filename):
        assert filename.is_file()


@pytest.mark.parametrize("windows", [False, True])
def test_build_configuration_always_selects_package_sdk(support, tmp_path, windows):
    wrapper, _ = support
    system = tmp_path / "system CUDA"
    (system / "bin").mkdir(parents=True)
    (system / "bin/nvcc.exe").touch()
    package = tmp_path / "package CUDA"
    environment = {
        "CUDA_PATH": str(system),
        "CUDA_HOME": str(system),
        "CUDAToolkit_ROOT": str(system),
        "CUDACXX": str(system / "bin/nvcc.exe"),
        "CUDAHOSTCXX": "/system/compiler",
        "NVCC_PREPEND_FLAGS": "-ccbin=/system/compiler",
        "CMAKE_GENERATOR": "Visual Studio 17 2022",
        "PATH": "/existing/bin",
    }
    original = dict(environment)
    arguments = ["-S", "/source with spaces", "-B", "/build"]
    configured, selected = wrapper._build_configuration(
        arguments,
        environment,
        package,
        windows=windows,
        compiler="/msvc/cl.exe" if windows else None,
    )
    assert environment == original
    assert configured[: len(arguments)] == arguments
    assert selected["CUDA_PATH"] == selected["CUDAToolkit_ROOT"] == str(package)
    assert selected["CUDA_HOME"] == str(package)
    assert selected["CUDACXX"] == str(package / ("bin/nvcc.exe" if windows else "bin/nvcc"))
    assert selected["CUDAHOSTCXX"] == selected["CXX"]
    assert "NVCC_PREPEND_FLAGS" not in selected
    assert selected["PATH"].split(os.pathsep)[0] == str(package / "bin")
    definitions = dict(value[2:].split("=", 1) for value in configured[len(arguments) :])
    assert definitions["CMAKE_CUDA_COMPILER"] == selected["CUDACXX"]
    assert definitions["CUDAToolkit_ROOT"] == str(package)
    if windows:
        assert selected["CXX"] == "/msvc/cl.exe"
        assert selected["CMAKE_GENERATOR"] == "Ninja"
        assert "CMAKE_PROJECT_INCLUDE" not in definitions
    else:
        assert selected["CXX"] == str(package / "bin/x86_64-conda-linux-gnu-c++")
        assert Path(definitions["CMAKE_PROJECT_INCLUDE"]).is_file()


def test_windows_configuration_requires_an_initialized_compiler(support, tmp_path):
    wrapper, _ = support
    with pytest.raises(RuntimeError, match="MSVC developer environment"):
        wrapper._build_configuration(["-S", "/source"], {}, tmp_path, windows=True)
