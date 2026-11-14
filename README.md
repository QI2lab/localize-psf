# localize-psf
This repository contains code for localizing diffraction limited fluorescent objects, fitting spots to various
different point-spread function (PSF) models, and other related image analysis tasks. The tools
contained here are general purpose, and they have been split into a separate package
so that both the [mcSIM code](https://github.com/QI2lab/mcSIM) and the 
[OPM code](https://github.com/QI2lab/OPM) can use them.

For a locked editable development environment (Python >=3.12),
```powershell
git clone https://github.com/QI2lab/localize-psf.git
cd localize-psf
uv sync --locked
uv run --locked pytest
uv build
```
See [development instructions](docs/development.md) for independently locked CPU
and local GPU profiles, build prerequisites and publication checks. GPU fitting
requires the [QI2lab GPUfit `opm-ptv` fork](https://github.com/QI2lab/Gpufit/tree/opm-ptv),
`pyGpufit==1.2.0` with capability ABI 3. Rebuild and reinstall older GPU profiles
using the development instructions.

# Examples
Scripts illustrating example usage of localize-psf are available in [examples](examples).
Associated data necessary to run some of these examples can be downloaded from [Zenodo](https://doi.org/10.5281/zenodo.10022862).
These scripts assume this data has been placed in a directory called `examples/data`

## [psf_calibration.py](examples/psf_calibration.py)
Estimate the 3D PSF of a microscope from an image of many diffraction limited beads on a coverslip


# Contents

## [localize.py](localize_psf/localize.py)
Tools for localizing diffraction limited spots from image data. This is primarily intended as
support for FISH spot finding and PSF fitting. This code is also the basis of
FISH spotfinding on the OPM, although the tilted/skewed geometry functions are contained in
[OPM repository](https://github.com/QI2lab/OPM). The localization routines can be run in parallel, either
with the existing CPU reference fitter or on the GPU using a modified version of
[GPUfit](https://github.com/QI2lab/Gpufit)

The module contains a high-level function for localizing diffraction limited objects in a 3D image,
`localize_beads_generic()`. This function is designed to work with extensible fit `model` and `filter` objects.

`roi_batch_size` (default 32) bounds ROI packing, parameter guesses, native fitting
and the associated covariance work. Candidate IDs refer to the complete detected
set and remain stable across batches, including failed or flagged fits. Weights,
known variances, per-candidate bounds and quality flags retain their alignment.
Only compact fit results accumulate; full-volume filtering and detection still
require the input image in memory. GPUfit's public binding transfers packed ROI
buffers to the host before native CUDA fitting.

`localize_psf.optical_bounds.gaussian_width_bounds` derives Gaussian width
priors from NA, wavelength in micrometres, immersion refractive index, and stored
ZYX voxel sizes. It reuses `na2sxy` for the lateral Airy/Gaussian convention and
the existing model's axial approximation. The caller explicitly supplies a
relative sigma range and sampling floor in pixels. Production calculations use
CuPy; `use_gpu=False` provides independent CPU references. These are optical
priors, not measured deconvolution resolution or covariance calibration.

Pass the nominal widths through `localize_beads_generic`'s
`initial_parameter_values={"sxy": ..., "sz": ...}` to initialize them explicitly.
Other parameters use the existing estimates. Infeasible starts still fail;
they are never clipped. `active_bounds` is returned even without covariance.

It also contains a high-level function for determining an experimental point-spread function and fitting
an image containing many 3D diffraction limited spots to a PSF model, `autofit_psfs()`.

## [fit_psf.py](localize_psf/fit_psf.py)
Tools for working with point-spread functions (PSF's), optical transfer functions (OTF's) and etc.
The most realistic functions rely on the 
[psfmodels](https://pypi.org/project/psfmodels/) package.

## [affine.py](localize_psf/affine.py)
Tools for working with affine transformations. These include manipulating transformation matrices
and fitting affine transformations from coordinate data or raw image data). The fitting
options include a simple implementation of the RANSAC algorithm

## [fit.py](localize_psf/fit.py)
Tools for non-linear least squares fitting. The most important function is `fit_least_squares()`
which is a wrapper aroung `scipy.optimize.least_squares()` with extra support for fixing
parameters. The most commonly used function is `fit_model()` which is used for fitting nD data
arrays  to various models.

## [camera.py](localize_psf/camera.py)
Tools for simulating the action of a camera and imaging models more generally. This includes binning, simulating
shot noise and readout noise, etc.

## [rois.py](localize_psf/rois.py)
Tools for dealing with regions of interest (ROI's)

# Docs
Documentation is generated from function docstrings and built with Sphinx. e.g. navigate to `docs` and run
```
make html
```
Then open `docs/_build/html/index.html` in your browser
