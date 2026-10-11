# Locked development environments

`pyproject.toml` owns distribution metadata (`localize_psf`, import `localize_psf`),
runtime dependencies, optional features and build requirements. The tested CPU
environment uses Python 3.12.10, uv 0.11.30 and Hatchling 1.27.0. Python >=3.12
is required. GPL-3.0-only licensing is preserved.

```powershell
uv lock
uv sync --locked
uv run --locked pytest
uv build
```

The default profile uses no CUDA toolkit or GPUfit. It installs the package
editable into `.venv`; wheels and sdists go into ignored `dist`. The development
lock includes optional resolutions but does not install them unless selected.
Build isolation pins Hatchling independently of the development lock.

GPU localization requires the QI2lab native fork with the public constrained API,
cuBLAS, float32 ABI and microscopy model IDs 13--16. Until its tested revisions
are published, use the separate local integration profile:

```powershell
uv run --locked python scripts/create_local_gpu_profile.py
uv lock --project build/local-gpu
powershell -File ../Gpufit/scripts/build_windows.ps1 -Action sync -Architectures 86 -ProjectPath build/local-gpu
uv run --locked --project build/local-gpu --no-sync python -c "import pygpufit.gpufit as gf; print(gf.get_capabilities())"
```

Keep sibling checkouts named `Gpufit` and `localize-psf`. The generated project
and its independent lock use relative editable sources and live under ignored
`build/local-gpu`. Its dependencies explicitly include `pyGpufit==1.1.0` and
`localize_psf[gpu]`; repeating locked sync retains these sources. The root `gpu`
extra supplies CuPy only while the fork has no published artifact. Installing
that extra alone is insufficient for fitting. Do not manually replace its
dependency with a stock GPUfit wheel. The native capability check rejects model
collisions before fitting.

After native edits, use the same command with `-Action rebuild`. On Linux,
initialize an approved CUDA/compiler environment and run `uv sync --locked
--project build/local-gpu --config-settings-package=pygpufit:cmake.define.USE_CUBLAS=ON
--config-settings-package=pygpufit:cmake.define.DOUBLE_PRECISION=OFF`.
Linux execution and remote clean installation remain pending. Before release,
add an exact reachable QI2lab GPUfit revision/artifact to the consumer profile,
regenerate the portable lock and verify it outside the local checkouts.

Existing CPU reference tests pass (28 passed, one pre-existing PSF test skipped).
Packaging and new fitting interfaces have separate acceptance checks.
