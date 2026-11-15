"""Create the ignored, independently locked GPU profile."""

import json
import tomllib
from pathlib import Path


def main() -> None:
    """Generate relative uv sources without altering portable project metadata."""
    root = Path(__file__).resolve().parents[1]
    target = root / "build" / "local-gpu"
    target.mkdir(parents=True, exist_ok=True)
    uv = tomllib.loads((root / "pyproject.toml").read_text())["tool"]["uv"]
    build_dependencies = json.dumps(uv["extra-build-dependencies"]["pygpufit"])
    build_variables = ", ".join(
        f"{name} = {json.dumps(value)}"
        for name, value in uv["extra-build-variables"]["pygpufit"].items()
    )
    (target / "pyproject.toml").write_text(
        """[project]
name = "localize-psf-local-gpu"
version = "0.0.0"
requires-python = ">=3.12"
dependencies = ["localize_psf[gpu]"]

[dependency-groups]
dev = ["pytest==9.1.1", "ruff==0.16.6"]

[tool.uv.sources]
localize-psf = {path = "../..", editable = true}
localize-psf-build-support = {path = "../../build_support"}

"""
        + "[tool.uv.extra-build-dependencies]\npygpufit = "
        + build_dependencies
        + "\n\n[tool.uv.extra-build-variables]\npygpufit = {"
        + build_variables
        + "}\n"
        + """

[[tool.uv.index]]
name = "nvidia"
url = "https://pypi.nvidia.com"
""",
        encoding="utf-8",
    )
    print(target)


if __name__ == "__main__":
    main()
