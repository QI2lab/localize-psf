"""CPU import without optional CUDA packages."""

import subprocess
import sys

import pytest

from localize_psf import _cuda


@pytest.mark.cpu
@pytest.mark.unit
def test_preload_without_optional_cuda_packages():
    code = f"from runpy import run_path; run_path({_cuda.__file__!r})['preload_cuda_libraries']()"
    subprocess.run([sys.executable, "-I", "-S", "-W", "error", "-c", code], check=True)
