"""Run CMake with the CUDA compiler declared by the build-support project."""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def _build_configuration(arguments, environment, prefix, *, windows=False, compiler=None):
    """Select the declared SDK without changing the caller's environment."""
    arguments = list(arguments)
    environment = dict(environment)
    prefix = Path(prefix)
    if windows:
        if compiler is None:
            raise RuntimeError("Initialize an x64 MSVC developer environment before building.")
        cc = cxx = compiler
        nvcc = prefix / "bin/nvcc.exe"
        environment["CMAKE_GENERATOR"] = "Ninja"
    else:
        cc = str(prefix / "bin/x86_64-conda-linux-gnu-cc")
        cxx = str(prefix / "bin/x86_64-conda-linux-gnu-c++")
        nvcc = prefix / "bin/nvcc"
    environment.update(
        CC=cc,
        CXX=cxx,
        CUDAHOSTCXX=cxx,
        CUDACXX=str(nvcc),
        CUDAToolkit_ROOT=str(prefix),
        CUDA_HOME=str(prefix),
        CUDA_PATH=str(prefix),
        PATH=str(prefix / "bin") + os.pathsep + environment.get("PATH", ""),
    )
    for name in ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "NVCC_CCBIN"):
        environment.pop(name, None)
    if any(argument.startswith("-S") for argument in arguments):
        arguments.extend(
            [
                f"-DCMAKE_CXX_COMPILER={environment['CXX']}",
                f"-DCMAKE_CUDA_COMPILER={environment['CUDACXX']}",
                f"-DCMAKE_CUDA_HOST_COMPILER={environment['CUDAHOSTCXX']}",
                f"-DCUDAToolkit_ROOT={prefix}",
            ]
        )
        if not windows:
            include = Path(__file__).parent / "cmake/LocalizePsfCuda.cmake"
            arguments.append(f"-DCMAKE_PROJECT_INCLUDE={include}")
    return arguments, environment


def main() -> int:
    """Run the real CMake with the production build configuration."""
    from cmake import CMAKE_BIN_DIR

    arguments = sys.argv[1:]
    environment = os.environ.copy()
    probe = arguments in (["--version"], ["--help"], ["-E", "capabilities"])
    if sys.platform.startswith(("linux", "win32")) and not probe:
        from .provision import toolchain_prefix

        windows = sys.platform == "win32"
        compiler = shutil.which("cl.exe") if windows else None
        if windows and compiler is None:
            raise RuntimeError("Initialize an x64 MSVC developer environment before building.")
        arguments, environment = _build_configuration(
            arguments, environment, toolchain_prefix(), windows=windows, compiler=compiler
        )
    executable = Path(CMAKE_BIN_DIR) / ("cmake.exe" if sys.platform == "win32" else "cmake")
    return subprocess.call([str(executable), *arguments], env=environment)
