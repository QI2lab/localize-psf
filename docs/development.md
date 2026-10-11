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

The modified fitting path uses existing SciPy/model fits serially on CPU and the
public QI2lab native APIs on GPU. Dask and zarr-python are optional only for the
retained legacy PSF calibration plotting/results feature (`legacy-calibration`).
They are absent from default and local GPU localization installations.

`localize.fit_rois` accepts numeric `bounds=(lower,upper)`, independently broadcast
from `(nparams,)` or given per fit. Explicit bounds bypass guesses; guessed bounds
exclude padding. Initial/fixed values must be feasible before and after float32
conversion. Parameters retain optimizer conventions, preserving fixed values and
constraints rather than normalizing widths/angles after optimization. `weights`
are nonnegative LSE objective multipliers; Poisson MLE is explicit and GPU-only.
The high-level `localize_beads_generic` forwards these controls and accepts a
physical ZYX `origin`, applied once when building coordinates. Its image-shaped
weights use the same ROI packing as the data. Empty outputs retain array shapes
and candidate IDs. Optimizer failures and acceptance-filter decisions are distinct.
# GPU model and uncertainty execution

The four microscopy models reuse their existing model and derivative code with
CuPy arrays, including rotated models and batched ROI coordinates. Device arrays
stay on CUDA through model evaluation; CPU calculations remain available as test
references. Windows CuPy requires the selected CUDA SDK headers as well as DLLs.
The local runner sets those paths for its child process and keeps its kernel cache
inside the ignored build directory:

```powershell
./scripts/run_gpu.ps1 -CommandArguments @('pytest', '-q', 'tests/test_gpu_psf_derivatives.py')
```

The runner defaults to the tested CUDA 12.8 installation. Supply `-CudaPath` to
select another compatible SDK. It uses the existing locked `build/local-gpu`
profile and never modifies persistent environment variables or installs packages.
# Localization uncertainty

Acceptance on the RTX 3090/CUDA 12.8 profile: all four native microscopy models
passed noisy LSE and Poisson MLE comparisons against independent CPU references.
Eight held-out Gaussian/Poisson coverage cases use 240 noisy CUDA-generated images
each, two brightnesses, two depths, and anisotropic ZYX sampling. Their predeclared
binomial coverage and whitened-residual/bias checks passed. Per-case reports are
written under ignored `build/covariance-validation`; experimental controls remain
pending because none were supplied.

```powershell
./scripts/run_gpu.ps1 -CommandArguments @('pytest', '-q', 'tests/test_gpu_uncertainty_validation.py')
```

`fit_rois(..., use_gpu=True, return_covariance=True)` evaluates the existing PSF
derivatives and undamped free-parameter information on CUDA in bounded batches
(default 32 ROIs), using float64 CuPy linear algebra. The full nuisance-parameter
covariance is solved before extracting `center_covariance_zyx`; off-diagonal terms
are retained. The native fitting ABI still takes host float32 NumPy buffers;
the uncertainty postpass has explicit transfers and does not imply zero-copy fitting.

Choose the noise convention explicitly:

- `estimated_gaussian`: weighted residual variance scales inverse information.
  Weights specify relative independent variances under this model.
- `calibrated_gaussian`: supply positive `variances` separately from LSE objective
  `weights`. Inverse-variance weights give inverse information; other weights,
  including unweighted heteroscedastic fitting, give the estimator's sandwich
  covariance. Known variance is never scaled by the observed residual.
- `poisson`: requires native `estimator='MLE'` and positive fitted means. It uses
  Poisson Fisher information. This convention does not describe camera read noise
  or correlated statistics after deskewing/deconvolution.

The result includes full `covariance`, `center_covariance_zyx`, method/backend,
information condition number, active bounds, validity, and named quality flags.
Failed fits, ill-conditioned information, fixed centers, active bounds, saturation,
truncation, unresolved overlap, or model mismatch invalidate center covariance.
Callers supply observed quality flags; their absence is not a calibration claim.
Invalid center covariance is NaN and must be excluded from covariance-based linking.
No numerical repair or arbitrary fallback covariance is supplied. Fixing nuisance
parameters makes uncertainty conditional on those values and excludes uncertainty
in their independent calibration.

`localize_beads_generic` packs an image-shaped `variances` map through the existing
ROI preparation and forwards covariance controls. CPU fitting is retained for
independent references: its already-computed covariance is propagated without a
second fit, and `absolute_sigma=True` disables residual scaling in the lower fitter.
Production PTV uses CUDA; CPU fitting is a test reference only.

```powershell
./scripts/run_gpu.ps1 -CommandArguments @('pytest', '-q', 'tests/test_localization_covariance.py')
```
