# Locked development environments

`pyproject.toml` owns distribution metadata (`localize_psf`, import `localize_psf`),
runtime dependencies, optional features and build requirements. The tested CPU
environment uses Python 3.12.10, uv 0.11.30 and Hatchling 1.27.0. Python >=3.12
is required. GPL-3.0-only licensing is preserved.

```powershell
uv lock
uv sync --locked --group test-cpu
uv run --locked --group test-cpu pytest -m "not gpu"
uv build
```

The default profile uses no CUDA toolkit or GPUfit. It installs the package
editable into `.venv`; wheels and sdists go into ignored `dist`. The development
lock includes optional resolutions but does not install them unless selected.
Build isolation pins Hatchling independently of the development lock.

GPU localization requires the QI2lab `opm-ptv` native fork, `pyGpufit==1.2.0`,
with the public constrained API, cuBLAS and float32 capability ABI 3. The
[upstream merge](https://github.com/QI2lab/Gpufit/commit/30185b039346ab5ae43b08d2bc6032abaa9741a9)
preserves upstream model IDs 0--14 and moves the microscopy models to 15--18:
`GAUSS_3D_ARB`, `GAUSS_LOR_3D_ARB`, `GAUSS_3D_ROT_ARB` and `GAUSS_3D_ASYM_ARB`.
Localization uses these named constants; GPUfit validates its native manifest
when fitting. ABI 2 installations must be rebuilt and reinstalled.

The `gpu` extra installs `cupy-cuda12x[ctk]` and builds GPUfit directly from
`QI2lab/Gpufit` on `opm-ptv`. CuPy's `ctk` extra supplies the runtime libraries;
the CUDA 12 pip compiler component does not contain `nvcc` on Linux.
No cuCIM dependency is included.

On Linux x86_64, run the following directly from this checkout. Windows builds
require an initialized x64 MSVC compiler environment:

```powershell
uv sync --locked --extra gpu
```

Add `--extra example` to install CUDA PyTorch and the U-FISH workflow. Linux
builds need no system CUDA SDK, Conda installation, GCC version selection or
manual activation. `tool.uv.extra-build-dependencies` installs the local
`build_support` package into GPUfit's isolated build environment.
`tool.uv.extra-build-variables` selects its CMake wrapper. The helper's
`pyproject.toml` declares exact CUDA 12.9, GCC/G++ 13, binutils and Linux sysroot
artifacts, including their SHA256 hashes. A pinned, verified micromamba binary
installs those artifacts inside the build environment without reading existing
Conda configuration. The wrapper selects only these compilers and SDK for the
native build, and fixes GPUfit's Linux cuBLAS runtime installation targets.
The resulting GPUfit wheel includes cuBLAS and cuBLASLt with relative library
lookup paths.

The first Linux GPU build downloads approximately 800 MB of build dependencies
in addition to Python dependencies. Subsequent syncs reuse the native wheel
cached by uv. CPU-only syncs do not install this build toolchain. The NVIDIA
driver remains a host prerequisite for GPU execution; compiling and importing
GPUfit do not require a visible GPU. Other Linux architectures are not yet
covered by the pinned compiler artifacts.

On Windows x86_64, the build helper always provisions the pinned CUDA 12.8
Update 1 SDK from the Windows artifact declarations in
`build_support/pyproject.toml`. Existing CUDA installations and inherited CUDA
path variables are overridden on both platforms. MSVC must
already be initialized; the tested toolset is MSVC 14.38. CI initializes that
toolset automatically. The Windows SDK bootstrap stays inside the isolated
build environment and does not install a system CUDA toolkit.

Use the same root project for local GPU integration:

```powershell
uv sync --locked --extra gpu --extra example --group test-cpu
uv run --locked --no-sync python -c "import pygpufit.gpufit as gf; print(gf.get_capabilities())"
```

The root lock and build settings select the same package-managed CUDA toolchain
on both platforms. To replace an old native binary, use
`uv sync --locked --extra gpu --reinstall-package pygpufit`.
GPUfit's build defaults select cuBLAS and
float32. Linux source builds, 64 core GPU unit and integration tests (including
Monte Carlo simulations), and the U-FISH integration test have been validated
on Ubuntu 26.04 x86_64 with a TITAN RTX, CUDA 12.9 and GCC 13.4.

The [U-FISH localization example](../examples/localize_ufish_prior.md) covers
3D localization, optional PSF averaging and microscope calibration, time/channel
batching and command line usage. Install its dependencies with the `example`
extra. The optional uv environments select CUDA PyTorch. Unit and integration
tests are separate for CPU and GPU. The `test-cpu` group supplies TensorStore
and Typer for hosted CPU tests.

Tests are unit tests or integration tests and execute production code directly.
Simulations supply input images, noise and physical parameters. Package modules,
functions and models are never replaced by mocks, spies or duplicated algorithms.
The compiled ABI remains a required unit test in the CUDA build jobs.

GitHub Actions runs the existing CPU/distribution checks on Windows and Linux.
The [CUDA build workflow](../.github/workflows/gpu.yml) also runs on every
push and pull request with Ubuntu 24.04 and Windows Server 2022, Python 3.12.10
and uv 0.12.3. Both jobs start with an empty uv cache, run
`uv sync --locked --extra gpu`, and run production build-support and compiled ABI
unit tests. The ABI unit tests check the locked native source, precision,
toolchain and required cuBLAS runtime artifacts. They use the `pyproject.toml`
toolchains without a system CUDA SDK or GPU. Windows uses MSVC 14.38 with the
pinned CUDA 12.8 SDK; the workflow installs the MSVC component if the runner
image no longer includes it. Linux uses CUDA 12.9 and GCC 13. Both jobs verify
their native library and both cuBLAS runtime libraries (`.dll` on Windows, `.so`
on Linux). These hosted jobs test the `gpu` extra. CPU example checks remain part of the existing CPU
workflow.

To automate numerical CUDA tests, register a Linux x86_64 self-hosted runner
with the `gpu` label and a working NVIDIA driver. In Actions, select **CUDA
builds**, choose **Run workflow**, and enable **run_gpu_tests**. That
optional job requires a visible CUDA device and runs the core GPU tests;
it excludes the U-FISH example and Monte Carlo suite. Pushes and pull requests
do not schedule jobs on that runner.

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
MLE batches are grouped by valid ROI length as well as constraint pattern and
trimmed before dispatch: zero-valued padding is invalid in GPUfit's Poisson
gradient and Hessian. Results retain their original candidate order.
The high-level `localize_beads_generic` forwards these controls and accepts a
physical ZYX `origin`, applied once when building coordinates. Its image-shaped
weights use the same ROI packing as the data. Empty outputs retain array shapes
and candidate IDs. Optimizer failures and acceptance-filter decisions are distinct.
# GPU model and uncertainty execution

## Batch settings

`localize_psf.config.LocalizationConfig` owns the version 1 JSON contract in
[localization.schema.json](localization.schema.json). The supplied
[configuration](../examples/localization.json) uses the same detector and
parameter-policy adapters as an interactive caller. Version 1 rejects unknown
settings, unsupported versions, nonfinite numbers and implicit type coercion.
Legacy optical metadata remains supported by the example and packaged runner.

```bash
python -m localize_psf.batch INPUT --metadata CONFIG.json --output-dir OUTPUT
```

Use the existing uv environment with the `gpu` and `example` extras installed.
The runner requires a real CUDA GPU and uses the package-managed GPUfit toolchain.
It reads OME-Zarr through yaozarrs/TensorStore and OME-TIFF through tifffile,
preserves stored axes and transforms, and honors explicitly exported calibration.
yaozarrs owns OME-Zarr metadata conversion and transform composition. Opening
and selecting an image keeps a lazy TensorStore view. Inference reads only the
current volume or XY plane; fitting gathers ROI samples without rereading a
whole volume. Batch detection reuses its loaded model without retaining input
or enhancement images, and fitting reuses CUDA allocations between batches.
`preview` describes interactive reproduction scope; batch execution reads the
full spatial image. `batch_axes=true` processes all times and channels; otherwise
the explicit `timepoint` and `channel` select them. In 2D, each XY plane is
processed separately, using the production 3D Gaussian with fixed physical Z
and `sz=1 µm`.

Centers use µm offsets from each prior. Explicit bounds intersect the ROI;
null starts retain production estimates, and infeasible starts are rejected
without clipping. All candidate rows and fitting/uncertainty evidence are kept.
Centroid checkpoints include source geometry, weights, dimensional mode, blend,
threshold, edge exclusion, inference batch size and preprocessing policy.
Changing detection settings requires another output directory; fit-only changes
reuse centroids.

Interactive callers save current JSON with `LocalizationConfig.save` and
copy a quoted terminal command using their own installed interpreter. They do
not launch or manage a batch process. The example script retains its CLI and
exports through the packaged workflow. Upstream tests call the production
adapters directly; plugin UI tests stay in the plugin repository.

## Model execution

With `use_gpu_filter=True`, the high-level path keeps images, detected peaks,
duplicate filtering, coordinate conversion, ROI gathering, variance/weight maps,
and initial Gaussian parameter estimates on CUDA. `prepare_rois` selects its
backend from the input image and preserves the existing C-order/NaN-padding
contract. Only packed arrays are transferred when the public native GPUfit API
needs its host buffers. Returned filtered images remain CuPy arrays.

The four microscopy models reuse their existing model and derivative code with
CuPy arrays, including rotated models and batched ROI coordinates. Device arrays
stay on CUDA through model evaluation; CPU calculations remain available as test
references. Package startup uses NVIDIA's `cuda.pathfinder` to preload CUDA
component libraries before importing CuPy. On Windows, CUDA 12 component wheels
have no single CUDA root; the corresponding CuPy path warning is filtered only
after the runtime and NVRTC load and NVRTC is located in that wheel layout.
Other warnings remain visible. CPU installations import without CUDA components.

The local test runner uses the package CUDA components and keeps its kernel
cache inside the ignored build directory:

```powershell
./scripts/run_gpu.ps1 -CommandArguments @('pytest', '-q', 'tests/test_gpu_psf_derivatives.py')
```

The runner clears inherited CUDA path variables for the command and restores
them afterward. It uses the root project's locked environment and never
modifies persistent environment variables or installs packages.
# Localization uncertainty

All four native microscopy models are tested with noisy LSE and Poisson MLE
simulations. LSE fits are compared with the production CPU fitter; Poisson MLE
fits are checked against simulated centers and the production model's numerical
Jacobian.
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

Artifact acceptance also runs in fresh, locked uv projects outside the source
checkout. It validates the portable wheel, reconstructs a wheel from the sdist,
checks package/license contents, and runs production unit and integration tests,
including GPUfit ABI 3 validation. GPU artifact checks install the fork through
the `gpu` extra with the same package-managed build toolchain.

```powershell
./scripts/run_gpu.ps1 -CommandArguments @('python', 'scripts/check_artifacts.py')
```

```powershell
./scripts/run_gpu.ps1 -CommandArguments @('pytest', '-q', 'tests/test_localization_covariance.py')
```
