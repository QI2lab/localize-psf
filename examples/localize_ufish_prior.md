# 3D localization with U-FISH priors

[localize_ufish_prior.py](localize_ufish_prior.py) detects candidate coordinates
with CUDA U-FISH and its simfish checkpoint, then fits the original counts using
`gaussian3d_psf_model` and GPUfit Poisson MLE. Static bead data can also produce
an average PSF, focus gradient and field curvature.

Input is a NumPy array or OME-Zarr path/URL. Values must already be finite,
nonnegative photon/electron counts. OME-Zarr input uses yaozarrs' `open_group`,
`ome_metadata` and `to_tensorstore` directly, always selecting the highest
resolution. TensorStore slices remain lazy until a ZYX volume is needed.

# Installation

In an initialized CUDA/C++ compiler environment:

```powershell
uv sync --locked --extra gpu --extra example
```

The `gpu` extra installs CUDA libraries through `cupy-cuda12x[ctk]` and builds
QI2lab/Gpufit's `opm-ptv` branch. The `example` extra adds CUDA PyTorch, U-FISH,
yaozarrs from upstream `main` with TensorStore, Typer, PyArrow and tifffile.
See the [development instructions](../docs/development.md) for Windows build
prerequisites.

# Python usage

The function accepts data and one metadata mapping. Supply your experiment's
objective NA, emission wavelength, immersion refractive index and relative
lower/upper sigma factors. These numbers illustrate the required inputs:

```python
from examples.localize_ufish_prior import localize_ufish_prior

metadata = {
    "numerical_aperture": 1.3,
    "wavelength_um": 0.515,
    "refractive_index": 1.51,
    "relative_sigma_range": (0.5, 2.0),
    "fit_roi_size_um": (1.75, 0.65, 0.65),
}
result = localize_ufish_prior("spots.ome.zarr", metadata)

accepted = result.fit_results["success"]
parameters = result.fit_results["fit_params"][accepted]
centers_zyx_um = parameters[:, (3, 2, 1)]
```

Parameter order is `(amp, cx, cy, cz, sxy, sz, bg)`. Positions and widths are in
micrometres; center covariance uses ZYX order and squared micrometres. Optical
width bounds use the production `gaussian_width_bounds` helper with no sampling
floor. Amplitude/background are nonnegative and centers stay within their ROI.
All candidates keep their result rows. Convergence, active bounds, truncation
and Poisson covariance are reported by production `fit_rois`.

# Metadata and batching

OME-Zarr supplies axes, spatial units and scale/translation metadata. Origins
and fitted positions may be negative. Stored image geometry is used directly;
pass the path to an image group or a bioformats2raw collection containing one
image. For collections with multiple images, pass the image-group path.
NA and wavelength are read from
image-group attributes named `numerical_aperture` and `wavelength_um` when not
provided in metadata. A channel-to-wavelength mapping is supported, including
string channel keys in JSON.

For arrays, provide `voxel_size_zyx_um`; the origin defaults to zero. Axis order
defaults to ZYX for 3D and TCZYX for 5D. Set `axes="tzyx"` or `axes="czyx"` for
4D arrays. Other axis permutations can be specified explicitly.

`batch_axes=True` processes every time and channel index. With `batch_axes=False`,
provide a `timepoint` and/or `channel` for every corresponding input axis,
including singleton axes.

U-FISH infers all selected volumes before fitting starts, using its own inference
defaults. The detector and unused PyTorch CUDA cache are released before fitting.
ROIs and initial estimates are prepared and fitted one volume at a time, using
production `prepare_rois`, `fit_rois` and the library's result accumulator. Only
compact fit results accumulate across volumes. GPUfit handles native batching
within each volume, and the production covariance routine handles its own batches.

# Optional PSF diagnostics

Set `psf_diagnostics=True` for static z-stacks of isolated beads. This enables
PSF averaging and microscope calibration together; the default is `False`.
Time-lapse input is rejected for calibration, including when one timepoint is
selected. Static channels are calibrated independently.

```python
metadata.update(psf_diagnostics=True, psf_roi_size_px=(11, 21, 21))
result = localize_ufish_prior("beads.ome.zarr", metadata)
volume = result.volume_results[(0, 0)]
average_psf = volume["average_psf"]
focus = volume["focus"]
```

The original algorithm fits a plane, then fits movable-center isotropic curvature
to the plane residual with `fit_model`. PSF extraction uses `average_exp_psfs`.
`fit_roi_size_um` is the physical ZYX fitting size; `psf_roi_size_px` is the
ZYX pixel size for averaging. Their defaults are `(1.75, 0.65, 0.65)` and
`(11, 21, 21)` respectively. Both use production `get_roi_size` to round up to
odd pixel dimensions, keeping the center on a voxel. The averaging dimensions
actually used are returned in metadata.

# Command line usage

The CLI takes the data path, metadata JSON and output directory. JSON uses the
same keys as the Python mapping, including optical settings, ROIs, batching,
T/C selection and `psf_diagnostics`.

[localize_ufish_prior.json](localize_ufish_prior.json) sets NA to 1.35 and
immersion refractive index to 1.4, with emission wavelength 0.515 um, the example
sigma factors `[0.5, 2.0]` and default batching, diagnostics and ROI settings.
Image geometry is read from OME-Zarr.

```powershell
uv run --locked --extra gpu --extra example python examples/localize_ufish_prior.py spots.ome.zarr --metadata examples/localize_ufish_prior.json --output-dir localizations
uv run --locked --extra gpu --extra example python examples/localize_ufish_prior.py --help
```

# Results and tests

`LocalizationResult` contains metadata, parameter names, candidate coordinates
in pixels/micrometres, source time/channel indices, ROI bounds and the production
`fit_rois` result. Poisson deviance is `fit_results["chi_sqrs"]`; truncation is
`fit_results["covariance_flags"]["truncated"]`.
`volume_results[(timepoint, channel)]` holds candidate IDs and any enabled PSF
diagnostics, including entries for empty volumes.

The function returns data without writing files. The CLI and `save_localizations`
write `localizations.parquet`, including vector/matrix columns and embedded
metadata. Enabled calibration also writes the original ImageJ `psf.tif` and
`focus_gradient_and_field_curvature.png` exports, prefixed by channel when needed.

Tests use production readers, fitting and exports with simulated counts.
CPU and GPU tests remain separate; GitHub Actions runs the CPU tests on Windows
and Ubuntu for pushes and pull requests:

```powershell
uv run --locked --group test-cpu pytest -m "not gpu"
uv run --locked --extra gpu --extra example pytest -m gpu
```
