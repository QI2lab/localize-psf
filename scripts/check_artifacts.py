"""Build and validate localization artifacts in independent locked uv projects."""

import json
import os
import shutil
import subprocess
import tempfile
import tomllib
import zipfile
from pathlib import Path
from tarfile import open as open_tar

ROOT = Path(__file__).resolve().parents[1]
UV = shutil.which("uv") or str(Path.home() / ".local/bin/uv.exe")


def checked(*arguments, cwd):
    subprocess.run([UV, *arguments], cwd=cwd, check=True)


def verify(wheel, gpu):
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert "localize_psf/uncertainty.py" in names
        assert "localize_psf/array_backend.py" in names
        assert any(name.endswith("licenses/LICENSE") for name in names)
        assert not any(".venv/" in name or ".uv-cache/" in name for name in names)
    with tempfile.TemporaryDirectory(prefix="localize-artifact-") as directory:
        target = Path(directory).resolve()
        assert target.is_relative_to(Path(tempfile.gettempdir()).resolve())
        dependencies = ["localize_psf[gpu]==0.2.0"] if gpu else ["localize_psf==0.2.0"]
        sources = f"localize-psf={{path={json.dumps(wheel.as_posix())}}}\n"
        (target / "pyproject.toml").write_text(
            '[project]\nname="localize-artifact-check"\nversion="0.0.0"\n'
            'requires-python=">=3.12"\ndependencies='
            + json.dumps([*dependencies, "pytest==9.1.1"])
            + "\n[tool.uv.sources]\n"
            + sources,
            encoding="utf-8",
        )
        shutil.copytree(
            ROOT / "tests", target / "tests", ignore=shutil.ignore_patterns("__pycache__")
        )
        pytest_options = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))["tool"][
            "pytest"
        ]["ini_options"]
        (target / "pytest.ini").write_text(
            "[pytest]\npython_files = "
            + " ".join(pytest_options["python_files"])
            + "\nmarkers =\n"
            + "\n".join("    " + marker for marker in pytest_options["markers"])
            + "\n",
            encoding="utf-8",
        )
        # Production-profile tests use CUDA; CPU calculations are independent references.
        checked("lock", cwd=target)
        checked("sync", "--locked", cwd=target)
        checked("sync", "--locked", cwd=target)
        checked(
            "run",
            "--locked",
            "--no-sync",
            "pytest",
            "-q",
            "-m",
            "not example" if gpu else "not gpu and not example",
            cwd=target,
        )
        script = (
            "import importlib.util, pathlib, localize_psf; "
            "p=pathlib.Path(localize_psf.__file__).resolve(); print(p); "
            f"assert not p.is_relative_to(pathlib.Path({str(ROOT)!r})); "
        )
        if not gpu:
            script += "assert importlib.util.find_spec('cupy') is None; assert importlib.util.find_spec('pygpufit') is None"
        else:
            script += "import pygpufit.gpufit as gf; assert gf.get_capabilities()['abi_version']==3"
        checked("run", "--locked", "--no-sync", "python", "-c", script, cwd=target)


def main():
    """Validate the wheel and a wheel reconstructed from the source archive."""
    checked("build", "--no-sources", cwd=ROOT)
    wheel = next((ROOT / "dist").glob("*.whl"))
    verify(wheel, gpu=False)
    verify(wheel, gpu=True)
    with open_tar(next((ROOT / "dist").glob("*.tar.gz"))) as archive:
        assert not any(".venv/" in name or ".uv-cache/" in name for name in archive.getnames())
        with tempfile.TemporaryDirectory(prefix="localize-sdist-") as directory:
            target = Path(directory).resolve()
            assert target.is_relative_to(Path(tempfile.gettempdir()).resolve())
            archive.extractall(target, filter="data")
            source = next(target.iterdir())
            checked("build", "--wheel", "--no-sources", cwd=source)
            verify(next((source / "dist").glob("*.whl")), gpu=True)


if __name__ == "__main__":
    # Keep CUDA SDK selection and cache placement with the caller's GPU runner.
    os.environ.setdefault("CUPY_CACHE_DIR", str(ROOT / "build/cupy-cache"))
    main()
