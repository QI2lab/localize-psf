"""Localize 3D spots with CUDA U-FISH and optically bounded Gaussian Poisson MLE.

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

_TRACKPY_SOURCE_COLUMNS = (
    "fit_params",
    "center_covariance_zyx",
    "success",
    "covariance_valid",
    "candidate_timepoints",
    "candidate_channels",
)


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
    """Convert OME-Zarr spatial transforms to micrometers.

    Parameters
    ----------
    multiscale : yaozarrs.v05.Multiscale
        Image axes and transforms, with the highest-resolution dataset first.

    Returns
    -------
    dict
        Stored axis order and ZYX voxel sizes and origins in micrometers, under
        ``axes``, ``voxel_size_zyx_um`` and ``origin_zyx_um``.
    """
    axes = multiscale.axes
    names = [axis.name for axis in axes]
    scale, origin = np.ones(len(axes)), np.zeros(len(axes))

    # Compose the dataset and image coordinate transforms.
    for transform in multiscale.datasets[0].coordinateTransformations + (
        multiscale.coordinateTransformations or []
    ):
        if transform.type == "scale":
            scale *= transform.scale
            origin *= transform.scale
        else:
            origin += transform.translation

    # Express the spatial coordinates in micrometres.
    units = {
        "meter": 1e6,
        "centimeter": 1e4,
        "millimeter": 1e3,
        "micrometer": 1,
        "nanometer": 1e-3,
        "picometer": 1e-6,
    }
    indices = [names.index(name) for name in "zyx"]
    factors = [units[axes[index].unit] for index in indices]
    return dict(
        axes="".join(names),
        voxel_size_zyx_um=scale[indices] * factors,
        origin_zyx_um=origin[indices] * factors,
    )


def _prepare_data(data, metadata):
    """Create a TCZYX view and select the volume indices to process.

    Parameters
    ----------
    data : numpy.ndarray or tensorstore.TensorStore
        Image with ZYX spatial axes and optional time and channel axes.
    metadata : dict
        Axis order, batching and PSF diagnostic flags, and optional timepoint
        and channel selectors. Updated in place with the resolved input axes.

    Returns
    -------
    data : numpy.ndarray or tensorstore.TensorStore
        TCZYX view without reading image values.
    keys : tuple of tuple of int
        Selected ``(timepoint, channel)`` indices in processing order.

    Raises
    ------
    ValueError
        If axes or selectors are incompatible, or PSF diagnostics are requested
        for time-lapse input.
    """
    axes = metadata.get("axes") or {3: "zyx", 5: "tczyx"}.get(data.ndim)
    if axes is None:
        raise ValueError("4D array input requires axes='tzyx' or 'czyx'")
    axes = axes.lower()
    if (
        len(axes) != data.ndim
        or len(set(axes)) != len(axes)
        or not set("zyx") <= set(axes)
        or not set(axes) <= set("tczyx")
    ):
        raise ValueError("axes must describe unique t/c/z/y/x dimensions including ZYX")
    metadata["axes"] = axes

    # Select the timepoints and channels to process.
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

    # Create a lazy TCZYX view.
    for axis in "ct":
        if axis not in axes:
            data, axes = data[None], axis + axes
    data = data.transpose(tuple(axes.index(axis) for axis in "tczyx"))
    return data, tuple(product(*selections))


def _psf_diagnostics(data, result, coords):
    """Add static PSF averages and focus diagnostics to localization results.

    Parameters
    ----------
    data : numpy.ndarray or tensorstore.TensorStore
        TCZYX image containing the localized static bead volumes.
    result : LocalizationResult
        Localization results to update in place using successful fits.
    coords : tuple of numpy.ndarray
        Sparse ZYX coordinate arrays in micrometers, broadcastable to a volume.

    Returns
    -------
    None
        Updates per-volume diagnostics and the odd PSF ROI size in ``result``.
    """
    result.metadata["psf_roi_size_px"] = get_roi_size(
        result.metadata["psf_roi_size_px"], (1, 1, 1)
    )
    size, spacing = (
        result.metadata["psf_roi_size_px"],
        result.metadata["voxel_size_zyx_um"],
    )
    for key, volume in result.volume_results.items():
        ids = volume["candidate_ids"]
        ids = ids[result.fit_results["success"][ids]]
        centers = result.fit_results["fit_params"][ids][:, (3, 2, 1)]

        # Fit the focus surface and retain the image projection.
        focus = _focus_diagnostics(centers)
        image = np.asarray(data[key])
        volume.update(focus=focus, image_xy=np.max(image, axis=0))

        # Average the PSFs from successful fits.
        if len(ids):
            volume["average_psf"] = average_exp_psfs(
                image,
                coords,
                centers,
                size,
                backgrounds=result.fit_results["fit_params"][ids, 6],
            )
            volume["average_psf_coords_zyx_um"] = tuple(get_psf_coords(size, spacing))


def _write_parquet(reader, path):
    """Write a complete Parquet file before atomically replacing the target.

    Parameters
    ----------
    reader : pyarrow.RecordBatchReader
        Record batches and schema to write. Batches are consumed in order.
    path : pathlib.Path
        Destination file, replaced after the temporary file is flushed.

    Returns
    -------
    None
        Writes the Parquet file at ``path``.
    """
    import pyarrow.parquet as pq

    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("wb") as stream:
        with pq.ParquetWriter(stream, reader.schema) as writer:
            for batch in reader:
                writer.write_batch(batch)
        stream.flush()
        fsync(stream.fileno())

    # Publish the completed file atomically.
    temporary.replace(path)


def _prepare_fit_rois(data, coords, rois, times, channels):
    """Pack an ROI batch spanning volumes with production ``prepare_rois``.

    Parameters
    ----------
    data : numpy.ndarray or tensorstore.TensorStore
        TCZYX image from which source volumes are read as needed.
    coords : tuple of numpy.ndarray
        Sparse ZYX coordinate arrays in micrometers, broadcastable to a volume.
    rois : numpy.ndarray, shape (nrois, 6)
        Start-inclusive, stop-exclusive bounds in ZYX order for a nonempty batch.
    times : numpy.ndarray, shape (nrois,)
        Source timepoint index for each ROI.
    channels : numpy.ndarray, shape (nrois,)
        Source channel index for each ROI.

    Returns
    -------
    values : numpy.ndarray, shape (nrois, max_samples)
        ROI intensities padded with NaN to the largest ROI in the batch.
    coordinates : tuple of numpy.ndarray
        Corresponding ZYX coordinate samples in micrometers, with NaN padding.
    counts : numpy.ndarray, shape (nrois,)
        Number of valid samples in each ROI.
    """
    npoints = int(np.max(np.prod(rois[:, 1::2] - rois[:, ::2], axis=1)))
    values = np.full((len(rois), npoints), np.nan)
    coordinates = tuple(np.full_like(values, np.nan) for _ in range(3))
    counts = np.empty(len(rois), int)

    # Pack each source volume into the shared ROI batch.
    for key in dict.fromkeys(zip(times, channels, strict=True)):
        selected = (times == key[0]) & (channels == key[1])
        packed, packed_coords, sizes = prepare_rois(
            np.asarray(data[key]), coords, rois[selected]
        )
        values[selected, : packed.shape[1]] = packed
        counts[selected] = sizes
        for target, source in zip(coordinates, packed_coords, strict=True):
            target[selected, : packed.shape[1]] = source

    return values, coordinates, counts


def localize_ufish_prior(
    data, metadata: Mapping | None = None, output_dir: str | PathLike | None = None
) -> LocalizationResult:
    """Localize 3D spots with CUDA U-FISH and Gaussian Poisson MLE.

    All selected volumes are inferred before production ``fit_rois`` dispatches
    fitting batches to GPUfit. See localize_ufish_prior.md for metadata and CLI
    usage.

    Parameters
    ----------
    data : numpy.ndarray or str or os.PathLike
        Calibrated photon/electron counts with ZYX spatial axes and optional
        time and channel axes, or an OME-Zarr store path or URL. OME-Zarr data
        is read lazily at the highest resolution.
    metadata : Mapping, optional
        Optical parameters, array geometry, axis selection, ROI sizes and PSF
        diagnostic settings. Array input requires ``voxel_size_zyx_um``. Optical
        bounds require ``numerical_aperture``, ``wavelength_um``,
        ``refractive_index`` and ``relative_sigma_range``; stored OME-Zarr NA
        and wavelength attributes are used when available.
    output_dir : str or os.PathLike, optional
        Directory for the recoverable ``centroids.parquet`` checkpoint.
        Completed volumes are reused on reruns with matching input and detector
        settings. If omitted, centroids are kept only in memory.

    Returns
    -------
    LocalizationResult
        Aligned candidate coordinates, volume indices, ROI bounds, fit results
        and covariance, including failed fits, with optional static PSF
        diagnostics. Positions and widths use micrometers.

    Raises
    ------
    ValueError
        If required metadata is missing, axes or selections are incompatible,
        checkpoint settings differ, or PSF diagnostics are requested for
        time-lapse input.
    """
    # Set defaults for localization and optional PSF diagnostics.
    metadata = {
        "batch_axes": True,
        "psf_diagnostics": False,
        "intensity_threshold": 0.0,
        "fit_roi_size_um": (1.75, 0.65, 0.65),
        "psf_roi_size_px": (11, 21, 21),
        **(metadata or {}),
    }
    source = None

    # Open the highest-resolution image with lazy TensorStore access.
    if isinstance(data, (str, PathLike)):
        from yaozarrs import bioformats2raw_series, convert_version, open_group, v05

        group = open_group(str(data))
        image = convert_version(group.ome_metadata(), "0.5")
        if isinstance(image, v05.Bf2Raw):
            paths = bioformats2raw_series(group)
            if len(paths) != 1:
                raise ValueError(
                    f"Collection has {len(paths)} images; pass an image group path"
                )
            group = group[paths[0]]
            image = convert_version(group.ome_metadata(), "0.5")

        # Read the image geometry and stored optical parameters.
        multiscale = image.multiscales[0]
        source = group.store_path
        metadata.update(_spatial_metadata(multiscale))
        for name in ("numerical_aperture", "wavelength_um"):
            if name in group.attrs:
                metadata.setdefault(name, group.attrs[name])
        data = group[multiscale.datasets[0].path].to_tensorstore()

    # Select volumes and establish physical coordinates.
    data, keys = _prepare_data(data, metadata)
    metadata["voxel_size_zyx_um"] = np.asarray(
        metadata["voxel_size_zyx_um"], dtype=float
    )
    metadata["origin_zyx_um"] = np.asarray(
        metadata.get("origin_zyx_um", (0, 0, 0)), dtype=float
    )

    # Require the optical parameters used for the fit bounds.
    for name in (
        "numerical_aperture",
        "wavelength_um",
        "refractive_index",
        "relative_sigma_range",
    ):
        if metadata.get(name) is None:
            raise ValueError(f"Provide {name} in metadata")

    # Compute optical width bounds for each channel.
    widths = {}
    for channel in {c for _, c in keys}:
        wavelength = metadata["wavelength_um"]
        if isinstance(wavelength, Mapping):
            wavelength = wavelength.get(channel, wavelength.get(str(channel)))
        if wavelength is None:
            raise ValueError(f"Provide wavelength_um for channel {channel}")
        bounds = gaussian_width_bounds(
            metadata["numerical_aperture"],
            wavelength,
            metadata["refractive_index"],
            metadata["voxel_size_zyx_um"],
            relative_sigma_range=metadata["relative_sigma_range"],
            sampling_floor_pixels=0,
        )
        widths[channel] = {name: value.get() for name, value in bounds.items()}

    # Set the fitting grid and centered ROI dimensions.
    spacing, origin = metadata["voxel_size_zyx_um"], metadata["origin_zyx_um"]
    coords = get_coords(data.shape[-3:], spacing, origin=origin)
    roi_size = get_roi_size(metadata["fit_roi_size_um"], spacing, ensure_odd=True)

    # Load checkpoint state for recovery and refitting.
    weights = "finetune_models/v1.0.1-simfish_model.pth"
    checkpoint = None
    completed = []
    if output_dir is not None:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        checkpoint = Path(output_dir) / "centroids.parquet"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        identity = dict(
            source=source,
            shape=list(data.shape),
            axes=metadata["axes"],
            ufish_weights=weights,
            intensity_threshold=metadata["intensity_threshold"],
        )
        schema = pa.schema(
            [("timepoint", pa.int64()), ("channel", pa.int64())]
            + [(axis + "_px", pa.float64()) for axis in "zyx"]
        )
        if checkpoint.exists():
            saved = pq.read_table(checkpoint)
            previous = json.loads(
                (saved.schema.metadata or {}).get(b"localize_psf", b"{}")
            )
            completed = [tuple(key) for key in previous.pop("completed_volumes", [])]
            if previous != identity:
                raise ValueError(
                    "Centroid checkpoint has different input or U-FISH settings; "
                    "use a different output directory"
                )
        else:
            saved = schema.empty_table()

    # Configure CUDA inference and suppress routine U-FISH logging.
    import cupy as cp
    import torch
    from tqdm.auto import tqdm
    from ufish.api import UFish
    from ufish.utils.log import logger as ufish_logger

    ufish_logger.disable("ufish")

    # Infer all selected volumes before fitting. Slices are read only when needed.
    detector = None
    candidates, times, channels = [], [], []
    for key in tqdm(keys, desc="U-FISH inference", unit="volume"):
        if key in completed:
            volume = saved.filter(
                (pc.field("timepoint") == key[0]) & (pc.field("channel") == key[1])
            )
            points = np.column_stack(
                [volume[axis + "_px"].to_numpy() for axis in "zyx"]
            )
        else:
            if detector is None:
                detector = UFish(device="cuda")
                detector.load_weights(weights_file=weights)

            # Predict the candidate centroids for this volume.
            spots = detector.predict(
                np.asarray(data[key]),
                axes="zyx",
                batch_size=256,
                intensity_threshold=metadata["intensity_threshold"],
            )[0]
            points = spots[["axis-0", "axis-1", "axis-2"]].to_numpy(dtype=float)

            # Save the completed volume before proceeding.
            if checkpoint is not None:
                columns = dict(
                    timepoint=np.full(len(points), key[0], int),
                    channel=np.full(len(points), key[1], int),
                    **{axis + "_px": points[:, i] for i, axis in enumerate("zyx")},
                )
                saved = pa.concat_tables([saved, pa.table(columns, schema=schema)])
                completed.append(key)
                status = json.dumps(dict(identity, completed_volumes=completed)).encode(
                    "utf-8"
                )
                saved = saved.replace_schema_metadata({b"localize_psf": status})
                _write_parquet(saved.to_reader(), checkpoint)

        # Preserve candidate order across volumes.
        candidates.append(points)
        times.append(np.full(len(points), key[0], int))
        channels.append(np.full(len(points), key[1], int))

    # Release inference memory before preparing fit batches.
    del detector
    if checkpoint is not None:
        del saved
    torch.cuda.empty_cache()

    # Align candidate arrays and construct the fitting ROIs.
    offsets = np.cumsum([0] + [len(points) for points in candidates])
    candidates, times, channels = map(np.concatenate, (candidates, times, channels))
    model = gaussian3d_psf_model()
    centers = origin + candidates * spacing
    rois = get_centered_rois(
        np.rint(candidates).astype(int), roi_size, [0, 0, 0], data.shape[-3:]
    )
    volumes = {
        key: dict(
            candidate_ids=np.arange(start, stop),
            focus=dict(available=False),
            average_psf=None,
        )
        for key, start, stop in zip(keys, offsets[:-1], offsets[1:], strict=True)
    }

    # Size the fitting and covariance batches from free GPU memory.
    cp.get_default_memory_pool().free_all_blocks()
    # Budget half of free VRAM for four derivative buffers, twelve sample buffers
    # and eight small parameter matrices, all float64 in the covariance routine.
    nparams, npoints = model.nparams, int(np.prod(roi_size))
    workspace = np.dtype(float).itemsize * (
        npoints * (4 * nparams + 12) + 8 * nparams**2
    )
    batch_size = max(1, int(cp.cuda.runtime.memGetInfo()[0] // (2 * workspace)))

    # Pack and fit bounded batches while retaining only compact results.
    fits = None
    for start in tqdm(
        range(0, len(candidates), batch_size), desc="Gaussian fitting", unit="batch"
    ):
        stop = min(start + batch_size, len(candidates))
        selected = slice(start, stop)
        values, coordinates, counts = _prepare_fit_rois(
            data, coords, rois[selected], times[selected], channels[selected]
        )

        # Estimate initial parameters and coordinate bounds.
        with np.errstate(divide="ignore", invalid="ignore"):
            initial = model.estimate_parameters(
                values, coordinates, num_preserved_dims=1
            )
        limits = np.asarray(
            [
                model.estimate_bounds(tuple(c[i, :count] for c in coordinates))
                for i, count in enumerate(counts)
            ]
        )
        lower, upper = limits[:, 0], limits[:, 1]
        lower[:, (0, 6)] = 0
        initial[:, 1:4] = centers[selected, ::-1]

        # Apply the optical width bounds for each channel.
        for channel, bounds in widths.items():
            same = channels[selected] == channel
            initial[same, 4:6] = bounds["nominal_sigma_zyx_um"][[1, 0]]
            lower[same, 4:6] = bounds["lower_sigma_zyx_um"][[1, 0]]
            upper[same, 4:6] = bounds["upper_sigma_zyx_um"][[1, 0]]

        # Fit with Poisson MLE and compute position uncertainty.
        truncated = np.any(
            rois[selected, 1::2] - rois[selected, ::2] != roi_size, axis=1
        )
        batch = fit_rois(
            values,
            coordinates,
            counts,
            initial,
            model=model,
            use_gpu=True,
            estimator="MLE",
            noise_model="poisson",
            bounds=(lower, upper),
            return_covariance=True,
            quality_flags={"truncated": truncated},
            covariance_batch_size=batch_size,
        )

        # Accumulate the results and release batch workspaces.
        cp.get_default_memory_pool().free_all_blocks()
        fits = _accumulate_fit_batch(fits, batch, int(start), len(candidates))
        del values, coordinates, counts, initial, limits, lower, upper, batch

    # Preserve the result shapes when no candidates were detected.
    if fits is None:
        empty = np.empty((0, 0))
        fits = fit_rois(
            empty,
            (empty,) * 3,
            np.empty(0, int),
            np.empty((0, model.nparams)),
            model=model,
            use_gpu=True,
            estimator="MLE",
            noise_model="poisson",
            return_covariance=True,
            quality_flags={"truncated": np.empty(0, bool)},
        )

    # Assemble the fit evidence and optional static PSF diagnostics.
    result = LocalizationResult(
        metadata=metadata,
        parameter_names=tuple(model.parameter_names),
        candidate_centroids_zyx_px=candidates,
        candidate_centroids_zyx_um=centers,
        candidate_timepoints=times,
        candidate_channels=channels,
        roi_bounds_zyx=rois,
        fit_results=fits,
        volume_results=volumes,
    )
    if metadata["psf_diagnostics"]:
        _psf_diagnostics(data, result, coords)
    return result


def _focus_diagnostics(centers):
    """Fit a focus plane followed by curvature in the plane residual.

    Parameters
    ----------
    centers : numpy.ndarray, shape (nspots, 3)
        Accepted fitted centers in ZYX order, in micrometers.

    Returns
    -------
    dict
        Availability flag and failure reason for insufficient or collinear
        centers, otherwise plane and curvature parameters, gradient magnitude
        and angle, and per-spot residuals. Coordinates and residuals use
        micrometers, angles use radians and curvature uses inverse micrometers.
    """
    output = dict(
        available=False,
        reason="At least three noncollinear accepted centers are required",
    )
    if len(centers) < 3:
        return output

    # Fit and remove the focus gradient.
    z, y, x = centers.T
    design = np.column_stack((np.ones(len(z)), x, y))
    plane, _, rank, _ = np.linalg.lstsq(design, z, rcond=None)
    if rank < 3:
        return output
    residual = z - design @ plane

    # Fit a movable curvature center to the plane residual.
    def curvature_model_fn(p):
        """Evaluate isotropic curvature at the fitted spot coordinates.

        Parameters
        ----------
        p : array_like, shape (3,)
            X and Y center coordinates in micrometers, followed by curvature
            in inverse micrometers.

        Returns
        -------
        numpy.ndarray, shape (nspots,)
            Predicted axial displacement in micrometers.
        """
        return p[2] * ((p[0] - x) ** 2 + (p[1] - y) ** 2)

    curvature = fit_model(
        residual, curvature_model_fn, np.array([np.mean(x), np.mean(y), 0])
    )
    parameters = curvature["fit_params"]
    return dict(
        available=True,
        plane_parameters=plane,
        gradient_magnitude=float(np.hypot(*plane[1:])),
        gradient_angle_rad=float(np.arctan2(plane[2], plane[1])),
        residual_after_plane_um=residual,
        curvature_parameters=parameters,
        curvature_per_um=float(parameters[2]),
        curvature_success=bool(curvature["success"]),
        residual_after_curvature_um=residual - curvature_model_fn(parameters),
    )


def plot_calibration(result: LocalizationResult):
    """Plot focus positions, plane residuals and curvature residuals.

    Parameters
    ----------
    result : LocalizationResult
        Localization results containing static PSF diagnostic data.

    Returns
    -------
    dict of str to matplotlib.figure.Figure
        Diagnostic figures keyed by export name, with a channel prefix when
        needed. Empty if no focus fit is available.
    """
    figures = {}
    for (_, channel), volume in result.volume_results.items():
        if not volume["focus"]["available"]:
            continue
        ids = volume["candidate_ids"][
            result.fit_results["success"][volume["candidate_ids"]]
        ]
        centers = result.fit_results["fit_params"][ids][:, (3, 2, 1)]
        focus = volume["focus"]
        prefix = "" if len(result.volume_results) == 1 else f"c{channel}_"

        # Set up the focus and residual panels with their colorbars.
        width_ratios = [1, 0.05, 0.05] * 3
        figure, axes = plt.subplots(
            1,
            len(width_ratios),
            figsize=(30, 10),
            gridspec_kw={
                "width_ratios": width_ratios,
                "wspace": 0.2 / np.mean(width_ratios),
            },
        )
        figure.suptitle("Focus gradient and field curvature")

        # Prepare physical coordinates and labels for each surface.
        coords = get_coords(
            volume["image_xy"].shape,
            result.metadata["voxel_size_zyx_um"][1:],
            origin=result.metadata["origin_zyx_um"][1:],
        )
        curve = focus["curvature_parameters"]
        values = [
            centers[:, 0],
            focus["residual_after_plane_um"],
            focus["residual_after_curvature_um"],
        ]
        titles = [
            f"grad={focus['gradient_magnitude'] * 1e2:.3g}um/100um, "
            f"angle={np.rad2deg(focus['gradient_angle_rad']):.1f}deg",
            f"field curvature after gradient removal: {curve[2]:.3g} 1/um, "
            f"center=({curve[0]:.1f}, {curve[1]:.1f}) um",
            "residual shift after removing gradient and curvature",
        ]

        # Plot the fitted focus surface and the two residuals.
        for i, (weights, title) in enumerate(zip(values, titles, strict=True)):
            limits = (
                [[np.percentile(weights, 1), np.percentile(weights, 99)]]
                if i < 2
                else None
            )
            plot_bead_locations(
                volume["image_xy"],
                centers,
                coords=coords,
                title=title,
                color_lists=["hsv"],
                weights=weights,
                cbar_labels=[r"$c_z$ (um)"],
                color_limits=limits,
                gamma=0.5,
                axes=axes[3 * i : 3 * (i + 1)],
            )
        figures[prefix + "focus_gradient_and_field_curvature"] = figure

    return figures


def to_trackpy_dataframe(table):
    """Convert converged Gaussian fits to a Trackpy-compatible dataframe.

    Parameters
    ----------
    table : pyarrow.Table
        Localization table containing fit parameters, center covariance,
        success and covariance flags, timepoint/channel indices, and the
        ``localize_psf`` schema metadata with image geometry.

    Returns
    -------
    pandas.DataFrame
        Converged fits with ``frame``, ``channel``, Gaussian amplitude as
        ``signal`` and ``covariance_valid``. Positions ``x/y/z``, widths
        ``size_*`` and errors ``ep_*`` use pixels; ``xum/yum/zum`` use world
        coordinates in micrometers. Invalid covariance retains NaN errors.
    """
    import pandas as pd

    table = table.select(_TRACKPY_SOURCE_COLUMNS).filter(table["success"])
    metadata = json.loads(table.schema.metadata[b"localize_psf"])["metadata"]
    spacing = np.asarray(metadata["voxel_size_zyx_um"])[::-1]
    origin = np.asarray(metadata["origin_zyx_um"])[::-1]

    # Extract fitted parameters and position errors from the Arrow lists.
    parameters = (
        table["fit_params"].combine_chunks().flatten().to_numpy().reshape(-1, 7)
    )
    covariance = (
        table["center_covariance_zyx"].combine_chunks().flatten().flatten().to_numpy()
    )
    covariance = covariance.reshape(-1, 3, 3)
    errors = np.sqrt(np.diagonal(covariance, axis1=1, axis2=2)[:, ::-1]) / spacing

    # Build Trackpy positions, signals, widths and position errors.
    columns = dict(
        frame=np.asarray(table["candidate_timepoints"]),
        channel=np.asarray(table["candidate_channels"]),
        signal=parameters[:, 0],
        covariance_valid=np.asarray(table["covariance_valid"]),
    )
    for index, axis in enumerate("xyz"):
        columns[axis] = (parameters[:, index + 1] - origin[index]) / spacing[index]
        columns[axis + "um"] = parameters[:, index + 1]
        columns["size_" + axis] = parameters[:, 4 if index < 2 else 5] / spacing[index]
        columns["ep_" + axis] = errors[:, index]

    return pd.DataFrame(columns)


def save_trackpy(localizations_path: str | PathLike) -> Path:
    """Stream saved Gaussian localizations to an additional Trackpy Parquet file.

    Parameters
    ----------
    localizations_path : str or os.PathLike
        Localization Parquet file written by ``save_localizations``.

    Returns
    -------
    pathlib.Path
        Path to the atomically written ``trackpy.parquet`` beside the input
        file, containing all converged fits.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    output = Path(localizations_path).with_name("trackpy.parquet")
    with pq.ParquetFile(localizations_path) as source:
        empty = to_trackpy_dataframe(source.schema_arrow.empty_table())
        schema = pa.Schema.from_pandas(empty, preserve_index=False)

        # Convert and write one Parquet batch at a time.
        batches = (
            pa.RecordBatch.from_pandas(
                to_trackpy_dataframe(pa.Table.from_batches([batch])),
                schema=schema,
                preserve_index=False,
            )
            for batch in source.iter_batches(columns=_TRACKPY_SOURCE_COLUMNS)
        )
        _write_parquet(pa.RecordBatchReader.from_batches(schema, batches), output)

    return output


def save_localizations(
    result: LocalizationResult, output_dir: str | PathLike, export_trackpy: bool = False
):
    """Save localizations and optional Trackpy and static PSF exports.

    Parameters
    ----------
    result : LocalizationResult
        Candidate and fit arrays, metadata and any static PSF diagnostics.
    output_dir : str or os.PathLike
        Destination directory, created if needed. Writes
        ``localizations.parquet`` and available PSF TIFFs and diagnostic PNGs.
    export_trackpy : bool, optional
        Also write ``trackpy.parquet`` containing all converged fits.

    Returns
    -------
    None
        Writes the requested files and closes the exported diagnostic figures.
    """
    import pyarrow as pa
    import tifffile

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Collect candidate arrays, fit values and covariance flags.
    arrays = {
        name: value
        for name, value in result.fit_results.items()
        if isinstance(value, np.ndarray)
    }
    arrays.update(
        candidate_centroids_zyx_px=result.candidate_centroids_zyx_px,
        candidate_centroids_zyx_um=result.candidate_centroids_zyx_um,
        candidate_timepoints=result.candidate_timepoints,
        candidate_channels=result.candidate_channels,
        roi_bounds_zyx=result.roi_bounds_zyx,
        bounds_lower=result.fit_results["bounds"][0],
        bounds_upper=result.fit_results["bounds"][1],
    )
    for name, flags in result.fit_results.get("covariance_flags", {}).items():
        arrays[f"covariance_flag_{name}"] = flags

    # Preserve vector and matrix columns as Arrow lists.
    columns = {}
    for name, values in arrays.items():
        column = pa.array(values.reshape(-1))
        for size in reversed(values.shape[1:]):
            column = pa.FixedSizeListArray.from_arrays(column, int(size))
        columns[name] = column

    # Save the fit table with its geometry and diagnostic metadata.
    summary = dict(
        metadata=result.metadata,
        parameter_names=result.parameter_names,
        fit_states_key=result.fit_results["fit_states_key"],
        volumes={
            f"t{t}_c{c}": dict(focus=v["focus"])
            for (t, c), v in result.volume_results.items()
        },
    )
    encoded = json.dumps(
        summary, default=lambda v: v.tolist() if isinstance(v, np.ndarray) else v.item()
    )
    table = pa.table(columns).replace_schema_metadata(
        {b"localize_psf": encoded.encode("utf-8")}
    )
    _write_parquet(table.to_reader(), output / "localizations.parquet")
    if export_trackpy:
        save_trackpy(output / "localizations.parquet")

    # Export the averaged PSFs in the original ImageJ TIFF format.
    dz, dy, dx = result.metadata["voxel_size_zyx_um"]
    for (_, channel), volume in result.volume_results.items():
        if volume["average_psf"] is None:
            continue
        prefix = "" if len(result.volume_results) == 1 else f"c{channel}_"
        wavelength = result.metadata["wavelength_um"]
        if isinstance(wavelength, Mapping):
            wavelength = wavelength.get(channel, wavelength.get(str(channel)))
        tifffile.imwrite(
            output / f"{prefix}psf.tif",
            tifffile.transpose_axes(
                volume["average_psf"].astype(np.float32), "ZYX", asaxes="TZCYXS"
            ),
            imagej=True,
            resolution=(1 / dx, 1 / dy),
            metadata={
                "Info": f"PSF, wavelength={wavelength:.03f} um",
                "unit": "um",
                "min": 0,
                "max": 1,
                "spacing": dz,
            },
        )

    # Save the focus-gradient and field-curvature plots.
    for name, figure in plot_calibration(result).items():
        figure.savefig(output / f"{name}.png")
        plt.close(figure)


app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def cli(
    data: str = typer.Argument(..., help="OME-Zarr store or URL"),
    output_dir: Path = typer.Option(..., "--output-dir"),
    metadata_path: Path = typer.Option(..., "--metadata", exists=True, dir_okay=False),
    export_trackpy: bool = typer.Option(
        False, "--export-trackpy", help="Also write converged fits to trackpy.parquet"
    ),
):
    """Localize an OME-Zarr image with optional static PSF calibration.

    Parameters
    ----------
    data : str
        OME-Zarr store path or URL.
    output_dir : pathlib.Path
        Directory for centroid checkpoints, localizations and diagnostic files.
    metadata_path : pathlib.Path
        JSON file containing optical parameters and localization settings.
    export_trackpy : bool, optional
        Also export converged fits to ``trackpy.parquet``.

    Returns
    -------
    None
        Saves the results and prints the number of converged fits.

    Raises
    ------
    typer.BadParameter
        If localization rejects the metadata or checkpoint settings.
    """
    # Read the settings and run localization.
    metadata = json.loads(metadata_path.read_text("utf-8"))
    try:
        result = localize_ufish_prior(data, metadata, output_dir=output_dir)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error

    # Write the requested tables and optional calibration diagnostics.
    save_localizations(result, output_dir, export_trackpy=export_trackpy)
    typer.echo(
        f"Converged {result.fit_results['success'].sum()} / "
        f"{len(result.candidate_centroids_zyx_px)} candidates; saved {output_dir}"
    )


if __name__ == "__main__":
    app()
