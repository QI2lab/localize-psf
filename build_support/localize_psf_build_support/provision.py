"""Install the declared CUDA build dependencies inside the isolated build environment."""

import errno
import hashlib
import os
import platform
import subprocess
import sys
import tarfile
import time
import tomllib
import urllib.request
from contextlib import contextmanager
from pathlib import Path


def _download(url: str, destination: Path, expected: str) -> None:
    temporary = destination.with_suffix(".partial")
    try:
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
            while data := response.read(1024 * 1024):
                output.write(data)
        with temporary.open("rb") as downloaded:
            actual = hashlib.file_digest(downloaded, "sha256").hexdigest()
        if actual != expected:
            raise RuntimeError(f"SHA256 mismatch downloading {url}: {actual}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _installation_lock(path: Path):
    """Serialize bootstrap calls on both supported build platforms."""
    with path.open("a+b") as lock:
        if sys.platform == "win32":
            import msvcrt

            if lock.tell() == 0:
                lock.write(b"\0")
                lock.flush()
            while True:
                lock.seek(0)
                try:
                    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as error:
                    if error.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                        raise
                    time.sleep(0.1)
            try:
                yield
            finally:
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock, fcntl.LOCK_EX)
            yield


def toolchain_prefix() -> Path:
    """Return a complete, hash-verified toolchain; never use a system CUDA SDK."""
    package = Path(__file__).resolve().parent
    configuration = package / "pyproject.toml"
    if not configuration.exists():
        configuration = package.parent / "pyproject.toml"
    settings = tomllib.loads(configuration.read_text())["tool"]["localize-psf-build-support"]
    windows = sys.platform == "win32"
    if platform.machine().lower() not in ("x86_64", "amd64"):
        raise RuntimeError("The declared CUDA build toolchain currently supports x86_64.")
    if windows:
        settings = settings["windows"]

    state = package / "_build_tools"
    state.mkdir(exist_ok=True)
    prefix = state / "cuda"
    sdk = prefix / "Library" if windows else prefix
    manifest = "@EXPLICIT\n" + "\n".join(settings["packages"]) + "\n"
    identity = hashlib.sha256(manifest.encode()).hexdigest()
    marker = prefix / ".localize-psf-toolchain"
    required_files = (
        (
            "bin/nvcc.exe",
            "include/cuda_runtime.h",
            "include/cublas_v2.h",
            "lib/cudart_static.lib",
            "lib/cublas.lib",
            "bin/cublas64_12.dll",
            "bin/cublasLt64_12.dll",
        )
        if windows
        else (
            "bin/nvcc",
            "bin/x86_64-conda-linux-gnu-cc",
            "bin/x86_64-conda-linux-gnu-c++",
            "targets/x86_64-linux/include/cuda_runtime.h",
            "targets/x86_64-linux/include/cublas_v2.h",
            "targets/x86_64-linux/lib/libcudart_static.a",
            "targets/x86_64-linux/lib/libcublas.so.12",
            "targets/x86_64-linux/lib/libcublasLt.so.12",
        )
    )
    with _installation_lock(state / "install.lock"):
        if (
            marker.exists()
            and marker.read_text() == identity
            and all((sdk / relative).is_file() for relative in required_files)
        ):
            return sdk

        description = "CUDA 12.8" if windows else "CUDA 12.9 / GCC 13"
        print(f"Installing the pyproject {description} build dependencies...", file=sys.stderr)
        executable = state / ("micromamba.exe" if windows else "micromamba")
        if not executable.exists():
            archive = state / "micromamba.tar.bz2"
            _download(settings["micromamba-url"], archive, settings["micromamba-sha256"])
            temporary_executable = executable.with_suffix(".partial")
            try:
                with tarfile.open(archive) as source:
                    member_name = "Library/bin/micromamba.exe" if windows else "bin/micromamba"
                    member = source.extractfile(member_name)
                    if member is None:
                        raise RuntimeError("The micromamba archive contains no executable.")
                    with member:
                        temporary_executable.write_bytes(member.read())
                temporary_executable.chmod(0o755)
                temporary_executable.replace(executable)
            finally:
                temporary_executable.unlink(missing_ok=True)
            archive.unlink()

        specification = state / "toolchain.lock"
        specification.write_text(manifest)
        # Do not read an existing Conda installation or write its configuration/cache.
        environment = {
            name: value
            for name, value in os.environ.items()
            if not name.startswith(("CONDA_", "MAMBA_", "_CONDA_"))
            and name not in {"_CE_M", "_CE_CONDA"}
        }
        environment["MAMBA_ROOT_PREFIX"] = str(state / "mamba")
        environment["XDG_CACHE_HOME"] = str(state / "cache")
        action = "install" if (prefix / "conda-meta" / "history").exists() else "create"
        subprocess.run(
            [
                str(executable),
                "--no-rc",
                action,
                "--yes",
                "--prefix",
                str(prefix),
                "--file",
                str(specification),
            ],
            env=environment,
            stdout=sys.stderr,
            check=True,
        )
        for relative in required_files:
            if not (sdk / relative).is_file():
                raise RuntimeError(f"Incomplete CUDA build dependency: {relative}")
        marker.write_text(identity)
    return sdk
