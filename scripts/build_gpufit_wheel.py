"""Build the locked GPUfit source as a universal-Python platform wheel."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path


def _locked_revision(lock: Path) -> str:
    """Return the immutable GPUfit source revision from a uv lock."""
    packages = tomllib.loads(lock.read_text(encoding="utf-8"))["package"]
    dependency = next(package for package in packages if package["name"] == "pygpufit")
    return dependency["source"]["git"].rsplit("#", 1)[1]


def main() -> None:
    """Build one wheel after validating its source and platform-independent tag."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    repository = Path(__file__).resolve().parents[1]
    source = arguments.source.resolve(strict=True)
    output = arguments.output.resolve()
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    expected_revision = _locked_revision(repository / "uv.lock")
    if revision != expected_revision:
        raise RuntimeError(f"GPUfit source {revision} does not match lock {expected_revision}.")
    status = subprocess.run(
        ["git", "-C", str(source), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("The GPUfit source checkout is dirty.")

    scripts = Path(sys.executable).parent
    cmake = scripts / (
        "localize-gpu-cmake.exe" if sys.platform == "win32" else "localize-gpu-cmake"
    )
    if not cmake.is_file():
        discovered = shutil.which("localize-gpu-cmake")
        cmake = Path(discovered) if discovered is not None else cmake
    if not cmake.is_file():
        raise RuntimeError("The localize-psf build support is not installed.")
    environment = os.environ.copy()
    environment.update(
        CMAKE_EXECUTABLE=str(cmake),
        CMAKE_GENERATOR="Ninja",
        PATH=str(scripts) + os.pathsep + environment.get("PATH", ""),
    )
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(output),
            str(source),
        ],
        check=True,
        env=environment,
    )

    wheels = list(output.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"Expected one GPUfit wheel, found {len(wheels)}.")
    expected_tags = {"linux": "linux_x86_64", "win32": "win_amd64"}
    try:
        platform_tag = expected_tags[sys.platform]
    except KeyError as error:
        raise RuntimeError(f"Unsupported wheel platform: {sys.platform}") from error
    pattern = rf"pygpufit-1\.2\.0-py3-none-{re.escape(platform_tag)}\.whl"
    if re.fullmatch(pattern, wheels[0].name) is None:
        raise RuntimeError(f"Unexpected GPUfit wheel tag: {wheels[0].name}")


if __name__ == "__main__":
    main()
