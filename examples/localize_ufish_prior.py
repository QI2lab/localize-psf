"""
Localize 3D spots with CUDA U-FISH and optically bounded Gaussian Poisson MLE.
Optional static-bead diagnostics retain the PSF averaging and field fits from
psf_calibration.py. See localize_ufish_prior.md for metadata and CLI usage.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
from os import PathLike, fsync
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import typer

from localize_psf.batch_fitting import _accumulate_fit_batch
from localize_psf.fit import fit_model
from localize_psf.fit_psf import average_exp_psfs, gaussian3d_psf_model, get_psf_coords
from localize_psf.localize import (
    fit_rois,
    get_coords,
    plot_bead_locations,
    prepare_rois,
)
from localize_psf.optical_bounds import gaussian_width_bounds
from localize_psf.rois import get_centered_rois, get_roi_size


@dataclass
class LocalizationResult:
    """Aligned candidate/fit arrays and optional diagnostics for each (time, channel)."""

    metadata: dict
    parameter_names: tuple[str, ...]
    candidate_centroids_zyx_px: np.ndarray
    candidate_centroids_zyx_um: np.ndarray
    candidate_timepoints: np.ndarray
    candidate_channels: np.ndarray
    roi_bounds_zyx: np.ndarray
    fit_results: dict
    volume_results: dict


def _spatial_metadata(multiscale):
    """Convert stored scale/translation metadata to the library's physical ZYX coordinates."""
    axes = multiscale.axes
    names = [axis.name for axis in axes]
    scale, origin = np.ones(len(axes)), np.zeros(len(axes))
    for transform in (multiscale.datasets[0].coordinateTransformations +
                      (multiscale.coordinateTransformations or [])):
        if transform.type == "scale":
            scale *= transform.scale
            origin *= transform.scale
        else:
            origin += transform.translation
    units = {"meter": 1e6, "centimeter": 1e4, "millimeter": 1e3,
             "micrometer": 1, "nanometer": 1e-3, "picometer": 1e-6}
    indices = [names.index(name) for name in "zyx"]
    factors = [units[axes[index].unit] for index in indices]
    return dict(axes="".join(names), voxel_size_zyx_um=scale[indices] * factors,
                origin_zyx_um=origin[indices] * factors)


def _prepare_data(data, metadata):
    """Select T/C indices and create a TCZYX view; pixel data stays lazy."""
    axes = metadata.get("axes") or {3: "zyx", 5: "tczyx"}.get(data.ndim)
    if axes is None:
        raise ValueError("4D array input requires axes='tzyx' or 'czyx'")
    axes = axes.lower()
    if (len(axes) != data.ndim or len(set(axes)) != len(axes)
        or not set("zyx") <= set(axes) or not set(axes) <= set("tczyx")):
        raise ValueError("axes must describe unique t/c/z/y/x dimensions including ZYX")
    metadata["axes"] = axes
    selections = []
    for axis, label in zip("tc", ("timepoint", "channel"), strict=True):
        selected = metadata.get(label)
        size = data.shape[axes.index(axis)] if axis in axes else 1
        if axis == "t" and size > 1 and metadata["psf_diagnostics"]:
            raise ValueError("PSF calibration is not supported for time-lapse data")
        if axis not in axes:
            if selected not in (None, 0):
                raise ValueError(f"A nonzero {label} requires that axis in the input")
            selections.append((0,))
        elif metadata["batch_axes"]:
            if selected is not None:
                raise ValueError(f"Set batch_axes=False to select a {label}")
            selections.append(range(size))
        else:
            if selected is None:
                raise ValueError(f"batch_axes=False requires an explicit {label}")
            selections.append((selected,))
    for axis in "ct":
        if axis not in axes:
            data, axes = data[None], axis + axes
    data = data.transpose(tuple(axes.index(axis) for axis in "tczyx"))
    return data, tuple(product(*selections))


def _psf_diagnostics(data, result, coords):
    result.metadata["psf_roi_size_px"] = get_roi_size(result.metadata["psf_roi_size_px"], (1, 1, 1))
    size, spacing = result.metadata["psf_roi_size_px"], result.metadata["voxel_size_zyx_um"]
    for key, volume in result.volume_results.items():
        ids = volume["candidate_ids"]
        ids = ids[result.fit_results["success"][ids]]
        centers = result.fit_results["fit_params"][ids][:, (3, 2, 1)]
        focus = _focus_diagnostics(centers)
        image = np.asarray(data[key])
        volume.update(focus=focus, image_xy=np.max(image, axis=0))
        if len(ids):
            volume["average_psf"] = average_exp_psfs(image, coords, centers, size,
                                                    backgrounds=result.fit_results["fit_params"][ids, 6])
            volume["average_psf_coords_zyx_um"] = tuple(get_psf_coords(size, spacing))


def _write_parquet(table, path):
    """Publish a closed, flushed Parquet file without exposing partial writes."""
    import pyarrow.parquet as pq

    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("wb") as stream:
        pq.write_table(table, stream)
        stream.flush()
        fsync(stream.fileno())
    temporary.replace(path)


def localize_ufish_prior(data, metadata: Mapping | None = None,
                         output_dir: str | PathLike | None = None) -> LocalizationResult:
    """
    Localize a NumPy array or highest-resolution OME-Zarr image using CUDA.

    Metadata holds optical parameters, array axes/spacing, T/C selection, ROI
    sizes and optional static PSF diagnostics. See localize_ufish_prior.md.
    All inference completes before production fit_rois dispatches Poisson MLE
    to GPUfit. Results preserve the candidate order, including failed fits.
    With output_dir, completed centroid volumes are saved and reused on reruns.
    """
    metadata = {"batch_axes": True, "psf_diagnostics": False,
                "fit_roi_size_um": (1.75, 0.65, 0.65), "psf_roi_size_px": (11, 21, 21),
                **(metadata or {})}
    source = None
    if isinstance(data, (str, PathLike)):
        from yaozarrs import bioformats2raw_series, convert_version, open_group, v05

        group = open_group(str(data))
        image = convert_version(group.ome_metadata(), "0.5")
        if isinstance(image, v05.Bf2Raw):
            paths = bioformats2raw_series(group)
            if len(paths) != 1:
                raise ValueError(f"Collection has {len(paths)} images; pass an image group path")
            group = group[paths[0]]
            image = convert_version(group.ome_metadata(), "0.5")
        multiscale = image.multiscales[0]
        source = group.store_path
        metadata.update(_spatial_metadata(multiscale))
        for name in ("numerical_aperture", "wavelength_um"):
            if name in group.attrs:
                metadata.setdefault(name, group.attrs[name])
        data = group[multiscale.datasets[0].path].to_tensorstore()
    data, keys = _prepare_data(data, metadata)
    metadata["voxel_size_zyx_um"] = np.asarray(metadata["voxel_size_zyx_um"], dtype=float)
    metadata["origin_zyx_um"] = np.asarray(metadata.get("origin_zyx_um", (0, 0, 0)), dtype=float)
    # Optical bounds and physical coordinates use production library helpers.
    for name in ("numerical_aperture", "wavelength_um", "refractive_index", "relative_sigma_range"):
        if metadata.get(name) is None:
            raise ValueError(f"Provide {name} in metadata")
    widths = {}
    for channel in {c for _, c in keys}:
        wavelength = metadata["wavelength_um"]
        if isinstance(wavelength, Mapping):
            wavelength = wavelength.get(channel, wavelength.get(str(channel)))
        if wavelength is None:
            raise ValueError(f"Provide wavelength_um for channel {channel}")
        bounds = gaussian_width_bounds(metadata["numerical_aperture"], wavelength,
                                       metadata["refractive_index"], metadata["voxel_size_zyx_um"],
                                       relative_sigma_range=metadata["relative_sigma_range"],
                                       sampling_floor_pixels=0)
        widths[channel] = {name: value.get() for name, value in bounds.items()}
    spacing, origin = metadata["voxel_size_zyx_um"], metadata["origin_zyx_um"]
    coords = get_coords(data.shape[-3:], spacing, origin=origin)
    roi_size = get_roi_size(metadata["fit_roi_size_um"], spacing, ensure_odd=True)
    weights = "finetune_models/v1.0.1-simfish_model.pth"
    checkpoint = None
    completed = []
    if output_dir is not None:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        checkpoint = Path(output_dir) / "centroids.parquet"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        identity = dict(source=source, shape=list(data.shape), axes=metadata["axes"],
                        ufish_weights=weights)
        schema = pa.schema([("timepoint", pa.int64()), ("channel", pa.int64())] +
                           [(axis + "_px", pa.float64()) for axis in "zyx"])
        if checkpoint.exists():
            saved = pq.read_table(checkpoint)
            previous = json.loads((saved.schema.metadata or {}).get(b"localize_psf", b"{}"))
            completed = [tuple(key) for key in previous.pop("completed_volumes", [])]
            if previous != identity:
                raise ValueError("Centroid checkpoint belongs to a different input; "
                                 "use a different output directory")
        else:
            saved = schema.empty_table()
    import torch
    from tqdm.auto import tqdm
    from ufish.api import UFish

    # Infer all selected volumes before fitting. Slices are read only when needed.
    detector = None
    candidates, times, channels = [], [], []
    for key in tqdm(keys, desc="U-FISH inference", unit="volume"):
        if key in completed:
            volume = saved.filter((pc.field("timepoint") == key[0]) & (pc.field("channel") == key[1]))
            points = np.column_stack([volume[axis + "_px"].to_numpy() for axis in "zyx"])
        else:
            if detector is None:
                detector = UFish(device="cuda")
                detector.load_weights(weights_file=weights)
            spots = detector.predict(np.asarray(data[key]), axes="zyx", batch_size=256, threshold=0.3)[0]
            points = spots[["axis-0", "axis-1", "axis-2"]].to_numpy(dtype=float)
            if checkpoint is not None:
                columns = dict(timepoint=np.full(len(points), key[0], int),
                               channel=np.full(len(points), key[1], int),
                               **{axis + "_px": points[:, i] for i, axis in enumerate("zyx")})
                saved = pa.concat_tables([saved, pa.table(columns, schema=schema)])
                completed.append(key)
                status = json.dumps(dict(identity, completed_volumes=completed)).encode("utf-8")
                saved = saved.replace_schema_metadata({b"localize_psf": status})
                _write_parquet(saved, checkpoint)
        candidates.append(points)
        times.append(np.full(len(points), key[0], int))
        channels.append(np.full(len(points), key[1], int))
    del detector
    if checkpoint is not None:
        del saved
    torch.cuda.empty_cache()
    offsets = np.cumsum([0] + [len(points) for points in candidates])
    candidates, times, channels = map(np.concatenate, (candidates, times, channels))
    # Pack and fit one volume at a time; retain only compact fit results.
    model = gaussian3d_psf_model()
    centers = origin + candidates * spacing
    rois = get_centered_rois(np.rint(candidates).astype(int), roi_size, [0, 0, 0], data.shape[-3:])
    fits, volumes = None, {}
    for key, start, stop in tqdm(zip(keys, offsets[:-1], offsets[1:], strict=True),
                                 total=len(keys), desc="Gaussian fitting", unit="volume"):
        volumes[key] = dict(candidate_ids=np.arange(start, stop),
                            focus=dict(available=False), average_psf=None)
        if start == stop:
            continue
        selected = slice(start, stop)
        values, coordinates, counts = prepare_rois(np.asarray(data[key]), coords, rois[selected])
        with np.errstate(divide="ignore", invalid="ignore"):
            initial = model.estimate_parameters(values, coordinates, num_preserved_dims=1)
        limits = np.asarray([model.estimate_bounds(tuple(c[i, :count] for c in coordinates))
                             for i, count in enumerate(counts)])
        lower, upper = limits[:, 0], limits[:, 1]
        lower[:, (0, 6)] = 0
        bounds = widths[key[1]]
        initial[:, 1:4] = centers[selected, ::-1]
        initial[:, 4:6] = bounds["nominal_sigma_zyx_um"][[1, 0]]
        lower[:, 4:6] = bounds["lower_sigma_zyx_um"][[1, 0]]
        upper[:, 4:6] = bounds["upper_sigma_zyx_um"][[1, 0]]
        truncated = np.any(rois[selected, 1::2] - rois[selected, ::2] != roi_size, axis=1)
        batch = fit_rois(values, coordinates, counts, initial, model=model,
                         use_gpu=True, estimator="MLE", noise_model="poisson", bounds=(lower, upper),
                         return_covariance=True, quality_flags={"truncated": truncated})
        fits = _accumulate_fit_batch(fits, batch, int(start), len(candidates))
        del values, coordinates, counts, initial, limits, lower, upper, batch
    if fits is None:
        empty = np.empty((0, 0))
        fits = fit_rois(empty, (empty,) * 3, np.empty(0, int), np.empty((0, model.nparams)),
                        model=model, use_gpu=True, estimator="MLE", noise_model="poisson",
                        return_covariance=True, quality_flags={"truncated": np.empty(0, bool)})
    result = LocalizationResult(metadata=metadata,
                                parameter_names=tuple(model.parameter_names),
                                candidate_centroids_zyx_px=candidates,
                                candidate_centroids_zyx_um=centers,
                                candidate_timepoints=times,
                                candidate_channels=channels,
                                roi_bounds_zyx=rois,
                                fit_results=fits,
                                volume_results=volumes)
    if metadata["psf_diagnostics"]:
        _psf_diagnostics(data, result, coords)
    return result


def _focus_diagnostics(centers):
    output = dict(available=False, reason="At least three noncollinear accepted centers are required")
    if len(centers) < 3:
        return output
    z, y, x = centers.T
    design = np.column_stack((np.ones(len(z)), x, y))
    plane, _, rank, _ = np.linalg.lstsq(design, z, rcond=None)
    if rank < 3:
        return output
    residual = z - design @ plane

    def curvature_model_fn(p):
        return p[2] * ((p[0] - x)**2 + (p[1] - y)**2)

    curvature = fit_model(residual, curvature_model_fn, np.array([np.mean(x), np.mean(y), 0]))
    parameters = curvature["fit_params"]
    return dict(available=True,
                plane_parameters=plane,
                gradient_magnitude=float(np.hypot(*plane[1:])),
                gradient_angle_rad=float(np.arctan2(plane[2], plane[1])),
                residual_after_plane_um=residual,
                curvature_parameters=parameters,
                curvature_per_um=float(parameters[2]),
                curvature_success=bool(curvature["success"]),
                residual_after_curvature_um=residual - curvature_model_fn(parameters))


def plot_calibration(result: LocalizationResult):
    """Plot the original focus, plane residual and curvature residual diagnostics."""
    figures = {}
    for (_, channel), volume in result.volume_results.items():
        if not volume["focus"]["available"]:
            continue
        ids = volume["candidate_ids"][result.fit_results["success"][volume["candidate_ids"]]]
        centers = result.fit_results["fit_params"][ids][:, (3, 2, 1)]
        focus = volume["focus"]
        prefix = "" if len(result.volume_results) == 1 else f"c{channel}_"
        width_ratios = [1, 0.05, 0.05] * 3
        figure, axes = plt.subplots(1, len(width_ratios), figsize=(30, 10),
                                     gridspec_kw={"width_ratios": width_ratios,
                                                  "wspace": 0.2 / np.mean(width_ratios)})
        figure.suptitle("Focus gradient and field curvature")
        coords = get_coords(volume["image_xy"].shape,
                            result.metadata["voxel_size_zyx_um"][1:],
                            origin=result.metadata["origin_zyx_um"][1:])
        curve = focus["curvature_parameters"]
        values = [centers[:, 0], focus["residual_after_plane_um"], focus["residual_after_curvature_um"]]
        titles = [
            f"grad={focus['gradient_magnitude'] * 1e2:.3g}um/100um, "
            f"angle={np.rad2deg(focus['gradient_angle_rad']):.1f}deg",
            f"field curvature after gradient removal: {curve[2]:.3g} 1/um, "
            f"center=({curve[0]:.1f}, {curve[1]:.1f}) um",
            "residual shift after removing gradient and curvature",
        ]
        for i, (weights, title) in enumerate(zip(values, titles, strict=True)):
            limits = [[np.percentile(weights, 1), np.percentile(weights, 99)]] if i < 2 else None
            plot_bead_locations(volume["image_xy"],
                                centers,
                                coords=coords,
                                title=title,
                                color_lists=["hsv"],
                                weights=weights,
                                cbar_labels=[r"$c_z$ (um)"],
                                color_limits=limits,
                                gamma=0.5,
                                axes=axes[3*i:3*(i+1)])
        figures[prefix + "focus_gradient_and_field_curvature"] = figure
    return figures


def save_localizations(result: LocalizationResult, output_dir: str | PathLike):
    """Write Parquet localizations and original ImageJ TIFF/PNG calibration exports."""
    import pyarrow as pa
    import tifffile

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    arrays = {name: value for name, value in result.fit_results.items() if isinstance(value, np.ndarray)}
    arrays.update(candidate_centroids_zyx_px=result.candidate_centroids_zyx_px,
                  candidate_centroids_zyx_um=result.candidate_centroids_zyx_um,
                  candidate_timepoints=result.candidate_timepoints,
                  candidate_channels=result.candidate_channels,
                  roi_bounds_zyx=result.roi_bounds_zyx,
                  bounds_lower=result.fit_results["bounds"][0],
                  bounds_upper=result.fit_results["bounds"][1])
    for name, flags in result.fit_results.get("covariance_flags", {}).items():
        arrays[f"covariance_flag_{name}"] = flags
    columns = {}
    for name, values in arrays.items():
        column = pa.array(values.reshape(-1))
        for size in reversed(values.shape[1:]):
            column = pa.FixedSizeListArray.from_arrays(column, int(size))
        columns[name] = column
    summary = dict(metadata=result.metadata,
                   parameter_names=result.parameter_names,
                   fit_states_key=result.fit_results["fit_states_key"],
                   volumes={f"t{t}_c{c}": dict(focus=v["focus"])
                            for (t, c), v in result.volume_results.items()})
    encoded = json.dumps(summary, default=lambda v: v.tolist() if isinstance(v, np.ndarray) else v.item())
    table = pa.table(columns).replace_schema_metadata({b"localize_psf": encoded.encode("utf-8")})
    _write_parquet(table, output / "localizations.parquet")
    dz, dy, dx = result.metadata["voxel_size_zyx_um"]
    for (_, channel), volume in result.volume_results.items():
        if volume["average_psf"] is None:
            continue
        prefix = "" if len(result.volume_results) == 1 else f"c{channel}_"
        wavelength = result.metadata["wavelength_um"]
        if isinstance(wavelength, Mapping):
            wavelength = wavelength.get(channel, wavelength.get(str(channel)))
        tifffile.imwrite(output / f"{prefix}psf.tif",
                         tifffile.transpose_axes(volume["average_psf"].astype(np.float32),
                                                 "ZYX", asaxes="TZCYXS"),
                         imagej=True,
                         resolution=(1/dx, 1/dy),
                         metadata={"Info": f"PSF, wavelength={wavelength:.03f} um",
                                   "unit": "um", "min": 0, "max": 1, "spacing": dz})
    for name, figure in plot_calibration(result).items():
        figure.savefig(output / f"{name}.png")
        plt.close(figure)


app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def cli(data: str = typer.Argument(..., help="OME-Zarr store or URL"),
        output_dir: Path = typer.Option(..., "--output-dir"),
        metadata_path: Path = typer.Option(..., "--metadata", exists=True, dir_okay=False)):
    """Localize 3D spots with optional static PSF calibration."""
    metadata = json.loads(metadata_path.read_text("utf-8"))
    try:
        result = localize_ufish_prior(data, metadata, output_dir=output_dir)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    save_localizations(result, output_dir)
    typer.echo(f"Converged {result.fit_results['success'].sum()} / "
               f"{len(result.candidate_centroids_zyx_px)} candidates; saved {output_dir}")


if __name__ == "__main__":
    app()
