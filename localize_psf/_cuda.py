"""Preload CUDA component wheels before importing GPU libraries."""

import sys
import warnings
from pathlib import Path


def preload_cuda_libraries() -> None:
    """Use NVIDIA's loader and handle CuPy's CUDA 12 component-layout warning."""
    if not sys.platform.startswith(("linux", "win32")):
        return
    try:
        from cuda.pathfinder import DynamicLibNotFoundError, load_nvidia_dynamic_lib
    except ImportError:
        return
    try:
        load_nvidia_dynamic_lib("cudart")
        nvrtc = load_nvidia_dynamic_lib("nvrtc")
    except DynamicLibNotFoundError:
        return
    if (
        sys.platform == "win32"
        and nvrtc.abs_path
        and Path(nvrtc.abs_path).parent.parent.name == "cuda_nvrtc"
    ):
        # CUDA 12 wheels have separate runtime/compiler roots; both loaded above.
        warnings.filterwarnings(
            "ignore",
            message=(
                r"CUDA path could not be detected\. Set CUDA_PATH environment "
                r"variable if CuPy fails to load\.$"
            ),
            category=UserWarning,
            module=r"cupy\._environment$",
        )
    for name in ("cublas", "cufft", "curand", "cusparse", "cusolver", "nvJitLink"):
        try:
            load_nvidia_dynamic_lib(name)
        except DynamicLibNotFoundError:
            continue
