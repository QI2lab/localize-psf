"""Unit tests of the compiled GPUfit ABI and its required runtime artifacts."""

import importlib
import os
import sys
from importlib.util import find_spec
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def native_gpufit():
    if os.environ.get("LOCALIZE_PSF_REQUIRE_NATIVE_ABI") == "1":
        return importlib.import_module("pygpufit.gpufit")
    return pytest.importorskip("pygpufit.gpufit")


def test_native_abi_matches_the_pinned_build(native_gpufit):
    capabilities = native_gpufit.get_capabilities()
    package = find_spec("localize_psf")
    assert package is not None and package.origin is not None
    revision = (Path(package.origin).parent / "gpufit_revision.txt").read_text().strip()
    assert capabilities["source_revision"] == revision
    assert capabilities["source_dirty"] is False
    assert capabilities["abi_version"] == 3
    assert capabilities["precision"] == "float32"
    assert capabilities["use_cublas"] is True
    assert capabilities["cuda_runtime_linkage"] == "static"
    assert capabilities["cublas_linkage"] == "shared"
    if sys.platform == "win32":
        assert capabilities["cuda_toolkit"].startswith("12.8.")
        assert capabilities["compiler"].startswith("MSVC 19.38.")
    else:
        assert capabilities["cuda_toolkit"].startswith("12.9.")
        assert capabilities["compiler"].startswith("GNU 13.")


def test_native_library_includes_its_cublas_dependencies(native_gpufit):
    folder = Path(native_gpufit.__file__).parent
    if sys.platform == "win32":
        libraries = sorted(p.name.lower() for p in folder.glob("*.dll"))
        assert libraries == ["cublas64_12.dll", "cublaslt64_12.dll", "gpufit.dll"], libraries
    else:
        libraries = sorted(p.name for p in folder.glob("*.so*"))
        assert libraries == ["libGpufit.so", "libcublas.so.12", "libcublasLt.so.12"], libraries
