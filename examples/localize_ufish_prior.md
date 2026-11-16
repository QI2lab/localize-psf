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

This script is a compatibility caller of `localize_psf.workflow`. The same
workflow is installed as `python -m localize_psf.batch`. Both accept legacy
metadata below and strict version 1 configurations exported by an interactive
caller. Version 1 settings, including explicit fitting policies, are documented
in [batch settings](../docs/development.md#batch-settings).

On Linux x86_64, run this directly from the checkout. On Windows, initialize
the x64 MSVC compiler environment first:

```powershell
uv sync --locked --extra gpu --extra example
```

The `gpu` extra installs CUDA libraries through `cupy-cuda12x[ctk]` and builds
QI2lab/Gpufit's `opm-ptv` branch. The `example` extra adds CUDA PyTorch, U-FISH,
yaozarrs from upstream `main` with TensorStore, Typer, PyArrow and tifffile.
On Linux, `pyproject.toml` supplies the CUDA SDK and compatible C++ compiler
automatically as isolated GPUfit build dependencies. A system CUDA installation
is unnecessary. See the [development instructions](../docs/development.md) for
details and Windows build prerequisites.

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
result = localize_ufish_prior("spots.ome.zarr", metadata, output_dir="localizations")

accepted = result["fit_results"]["success"]
parameters = result["fit_results"]["fit_params"][accepted]
centers_zyx_um = parameters[:, (3, 2, 1)]
```

Parameter order is `(amp, cx, cy, cz, sxy, sz, bg)`. Positions and widths are in
micrometres; center covariance uses ZYX order and squared micrometres. Optical
width bounds use the production `gaussian_width_bounds` helper with no sampling
floor. Amplitude/background are nonnegative and centers stay within their ROI.
All candidates keep their result rows. Convergence, active bounds, truncation
and Poisson covariance are reported by production `fit_rois`.

Interactive callers use the same settings and fit policy. Given a source array
`data` and a `LocalizationConfig` named `config`, preview preparation returns
the image and its physical crop origin directly:

```python
from localize_psf.processing import Detector, fit_spots, prepare_volume

image, origin = prepare_volume(data, config)
detector = Detector()
points, enhanced = detector.detect(image, config)
detector.release()
fits = fit_spots(image, points, config, origin_zyx_um=origin)
```

Keep the detector alive while tuning thresholds to reuse enhancement, then
release it before fitting. Spacing comes from `config`; a crop's returned origin
is passed explicitly. Batch and preview results are ordinary dictionaries.

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
ROIs from multiple timepoints and channels are packed into each fitting batch
using production `prepare_rois` and fitted through `fit_rois`. Per-channel optical
bounds and candidate order are preserved. Batch size is estimated from half the
free GPU memory, allowing for float64 covariance derivatives and matrix workspaces.
The same size is passed to the production covariance routine. Unused CuPy memory
is released between batches so GPUfit can use it. Only the current batch's packed
samples and compact accumulated fit results are retained; GPUfit handles its
native workspace batching.
`tqdm` counts volumes for U-FISH inference and ROI batches for Gaussian fitting.
Routine U-FISH logging is suppressed.

`intensity_threshold` controls U-FISH's local-maxima spot caller. The default
`0.0` keeps every positive local maximum of its Laplacian response. Set the same
key in JSON or Python metadata to raise the cutoff. `predict` forwards this
argument to the spot caller as `intensity_threshold`.

Set `exclude_edge_centroids=true` to discard centroids in an edge band along
Z, Y and X before fitting. `edge_margin_px` sets the band width at each face
and defaults to two voxels; for 2D, only Y and X are filtered. Edge exclusion
defaults to `false`. Interactive callers can change the margin without
recomputing enhancement. Changing an enabled filter requires a new centroid
checkpoint; the margin is ignored when exclusion is disabled.

# Centroid checkpoint

The CLI updates one `centroids.parquet` file in the output directory after each
completed volume. Columns are `timepoint`, `channel`, `z_px`, `y_px` and `x_px`.
File metadata records the input URI, shape, axes, U-FISH weights, intensity
cutoff and completed volumes, including volumes with no detections. Each update
replaces the file atomically, preserving the previous checkpoint if an update
is interrupted.

Rerun the same command to infer the remaining volumes, then fit all selected
centroids. Once inference is complete, reruns skip U-FISH. Change the fitting
parameters in the JSON to refit the saved centroids. The Python function enables
the same behavior with `output_dir`; leaving it unset keeps results in memory.

Keep one output directory per unchanged input dataset. Input URI, shape and axes
must match the checkpoint; array inputs have no URI, so reuse their directory
only with the same data. Remove `centroids.parquet` to repeat detection.
Changing `intensity_threshold` requires a new centroid checkpoint; fitting
settings can change while reusing the same centroids.

# Optional PSF diagnostics

Set `psf_diagnostics=True` for static z-stacks of isolated beads. This enables
PSF averaging and microscope calibration together; the default is `False`.
Time-lapse input is rejected for calibration, including when one timepoint is
selected. Static channels are calibrated independently.

```python
metadata.update(psf_diagnostics=True, psf_roi_size_px=(11, 21, 21))
result = localize_ufish_prior("beads.ome.zarr", metadata)
volume = result["volume_results"][(0, 0)]
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
uv run --locked --extra gpu --extra example python examples/localize_ufish_prior.py spots.ome.zarr --metadata examples/localize_ufish_prior.json --output-dir localizations --export-trackpy
uv run --locked --extra gpu --extra example python examples/localize_ufish_prior.py --help
```

# Trackpy export

`--export-trackpy` writes an additional `trackpy.parquet` containing all converged
fits. It retains `covariance_valid`; unavailable position uncertainties remain
NaN. Positions, widths and amplitude come from the Gaussian fit.

| Columns | Values |
| --- | --- |
| `x`, `y`, `z` | Fitted pixel coordinates, relative to the image origin |
| `xum`, `yum`, `zum` | Fitted positions in micrometres, including the image origin |
| `frame`, `channel` | Original integer timepoint and channel indices |
| `signal` | Gaussian amplitude |
| `size_x`, `size_y`, `size_z` | Gaussian sigma in pixels along each axis |
| `ep_x`, `ep_y`, `ep_z` | Position standard errors in pixels along each axis |
| `covariance_valid` | Whether position covariance passed the production checks |

Mass and raw-intensity columns are omitted. For 3D linking, use
`pos_columns=['xum', 'yum', 'zum']` as in the
[Trackpy tutorial](https://soft-matter.github.io/trackpy/dev/tutorial/tracking-3d.html),
and process channels independently. Trackpy itself is only needed for linking.

Existing fit files can be exported without refitting:

```python
from examples.localize_ufish_prior import save_trackpy

save_trackpy("localizations/localizations.parquet")
```

`save_trackpy` streams the required columns and writes the additional file
atomically. `to_trackpy_dataframe` converts an Arrow table to a pandas DataFrame.
For programmatic exports, set `export_trackpy=True` in `save_localizations`.

# Results and tests

The returned dictionary (`LocalizationResult` is a type alias) contains metadata,
parameter names, candidate coordinates
in pixels/micrometres, source time/channel indices, ROI bounds and the production
`fit_rois` result. Poisson deviance is `fit_results["chi_sqrs"]`; truncation is
`fit_results["covariance_flags"]["truncated"]`.
`volume_results[(timepoint, channel)]` holds candidate IDs and any enabled PSF
diagnostics, including entries for empty volumes.

The function returns fit data and updates the centroid checkpoint when
`output_dir` is supplied. The CLI and `save_localizations` write
`localizations.parquet`, including vector/matrix columns and embedded metadata.
Enabled calibration also writes the original ImageJ `psf.tif` and
`focus_gradient_and_field_curvature.png` exports, prefixed by channel when needed.

Tests use production readers, fitting and exports with simulated counts.
CPU and GPU tests remain separate; GitHub Actions runs the CPU tests on Windows
and Ubuntu for pushes and pull requests:

```powershell
uv run --locked --group test-cpu pytest -m "not gpu"
uv run --locked --extra gpu --extra example pytest -m gpu
```
