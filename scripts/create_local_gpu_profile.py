"""Create the ignored, independently locked sibling-checkout GPU profile."""

from pathlib import Path


def main() -> None:
    """Generate relative uv sources without altering portable project metadata."""
    root = Path(__file__).resolve().parents[1]
    gpufit = root.parent / "Gpufit"
    if not (gpufit / "pyproject.toml").is_file():
        raise FileNotFoundError("Place the tested QI2lab Gpufit checkout beside localize-psf.")
    target = root / "build" / "local-gpu"
    target.mkdir(parents=True, exist_ok=True)
    (target / "pyproject.toml").write_text('''[project]
name = "localize-psf-local-gpu"
version = "0.0.0"
requires-python = ">=3.12"
dependencies = ["localize_psf[gpu]", "pyGpufit==1.1.0"]

[dependency-groups]
dev = ["pytest==9.1.1", "ruff==0.16.6"]

[tool.uv.sources]
localize-psf = {path = "../..", editable = true}
pygpufit = {path = "../../../Gpufit", editable = true}
''', encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
