"""Localize 3D spots with CUDA U-FISH and optically bounded Gaussian Poisson MLE.

Optional static-bead diagnostics retain the PSF averaging and field fits from
psf_calibration.py. See localize_ufish_prior.md for metadata and CLI usage.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from itertools import product
from os import PathLike, fsync
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import typer

from localize_psf.batch_fitting import _accumulate_fit_batch
from localize_psf.config import DEFAULT_PSF_ROI_SIZE_PX, LocalizationConfig, Parameter
from localize_psf.fit import fit_model
from localize_psf.fit_psf import average_exp_psfs, get_psf_coords
from localize_psf.fit_psf import gaussian3d_psf_model as gaussian3d_psf_model
from localize_psf.localize import (
    get_coords,
    plot_bead_locations,
    prepare_rois,
)
from localize_psf.optical_bounds import gaussian_width_bounds
from localize_psf.processing import (
    Detector,
    calibrate_tilt,
    data_footprint,
    fit_filter_mask,
    fit_packed_rois,
    gaussian_model,
    optical_parameters,
)
from localize_psf.rois import get_centered_rois, get_roi_size

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from pandas import DataFrame
    from pyarrow import RecordBatchReader, Table
    from tensorstore import TensorStore
    from yaozarrs.v05 import Multiscale

_TRACKPY_SOURCE_COLUMNS = (
    "fit_params",
    "center_covariance_zyx",
    "success",
    "filter_pass",
    "covariance_valid",
    "roi_truncated",
    "candidate_timepoints",
    "candidate_channels",
)

_MICROMETERS_PER_UNIT = {
    "meter": 1e6,
    "m": 1e6,
    "centimeter": 1e4,
    "cm": 1e4,
    "millimeter": 1e3,
    "mm": 1e3,
    "micrometer": 1,
    "um": 1,
    "µm": 1,
    "nanometer": 1e-3,
    "nm": 1e-3,
    "picometer": 1e-6,
    "pm": 1e-6,
}


# Batch results follow the production fitter's dictionary convention.
type LocalizationResult = dict[str, Any]


def _spatial_metadata(multiscale: Multiscale, calibration: Mapping | None = None) -> dict[str, Any]:
    """Convert OME-Zarr spatial transforms to micrometers.

    Parameters
    ----------
    multiscale : yaozarrs.v05.Multiscale
        Image axes and transforms, with the highest-resolution dataset first.
    calibration : Mapping, optional
        Explicit ZYX voxel sizes and first-voxel origins in µm. Both values
        override stored calibration; they are required if source units are absent.

    Returns
    -------
    dict
        Stored axis order and ZYX voxel sizes and origins in micrometers, under
        ``axes``, ``voxel_size_zyx_um`` and ``origin_zyx_um``.

    Notes
    -----
    yaozarrs normalizes dataset/image transform composition. This adapter only
    selects spatial axes and converts their units to the fitter's micrometers.
    """
    names = [axis.name for axis in multiscale.axes]
    if calibration and all(name in calibration for name in ("voxel_size_zyx_um", "origin_zyx_um")):
        return dict(
            axes="".join(names),
            voxel_size_zyx_um=calibration["voxel_size_zyx_um"],
            origin_zyx_um=calibration["origin_zyx_um"],
        )
    indices = [names.index(name) if name in names else None for name in "zyx"]
    if any(multiscale.axes[i].unit not in _MICROMETERS_PER_UNIT for i in indices if i is not None):
        raise ValueError(
            "Source spatial units require explicit voxel sizes and origins in micrometers"
        )
    from yaozarrs import convert_version

    # yaozarrs folds image-level transforms into its native dataset models.
    dataset = convert_version(convert_version(multiscale, "0.6"), "0.5").datasets[0]
    scale = dataset.scale_transform.scale
    translation = dataset.translation_transform
    origin = translation.translation if translation is not None else np.zeros(len(names))
    factors = [
        _MICROMETERS_PER_UNIT[multiscale.axes[i].unit] if i is not None else 1 for i in indices
    ]
    return dict(
        axes="".join(names),
        voxel_size_zyx_um=np.array([scale[i] if i is not None else 1 for i in indices]) * factors,
        origin_zyx_um=np.array([origin[i] if i is not None else 0 for i in indices]) * factors,
    )


def ome_zarr_image_paths(source: str) -> list[str]:
    """Resolve an OME-Zarr root to image groups in stored series order.

    A standalone image returns its own path; a Bio-Formats/qi2lab collection
    returns its image-group paths, using yaozarrs metadata discovery without
    requiring OME-XML. Paths are image groups, never resolution arrays: readers
    select the first (highest-resolution) dataset declared by OME multiscales.
    Only metadata is read. Local paths and remote URIs retain their source root.
    """
    from yaozarrs import bioformats2raw_series, convert_version, open_group, v05

    group = open_group(source)
    image = group.ome_metadata()
    if image is not None:
        image = convert_version(image, "0.5")
        if isinstance(image, v05.Bf2Raw):
            return [
                f"{source.rstrip('/')}/{path}" if "://" in source else str(Path(source) / path)
                for path in bioformats2raw_series(group)
            ]
        if isinstance(image, v05.Image) and len(image.multiscales) == 1:
            return [source]
    raise ValueError("Input must identify an OME-Zarr image or bioformats2raw collection")


def open_image(
    source: str, calibration: Mapping | None = None
) -> tuple[np.ndarray | TensorStore, dict[str, Any]]:
    """Open the highest-resolution image while preserving its stored axis order.

    Parameters
    ----------
    source : str
        OME-Zarr store root/image group path or URI, or local OME-TIFF filename.
        Single-image collections are resolved automatically; resolution paths
        are selected from OME multiscales metadata.
    calibration : Mapping, optional
        Explicit ``voxel_size_zyx_um`` and ``origin_zyx_um`` in micrometers.

    Returns
    -------
    data : tensorstore.TensorStore or numpy.memmap
        OME-Zarr remains a lazy yaozarrs/TensorStore view with no pixel reads.
        tifffile decodes OME-TIFF into a memory map.
    metadata : dict
        Axes, ZYX calibration in µm, source metadata and stored channel names.

    Raises
    ------
    ValueError
        If the input is ambiguous, not an OME image, or its calibration cannot
        be represented by the fitter's axis-aligned coordinate system.
    """
    if not source.lower().endswith((".tif", ".tiff")):
        from yaozarrs import convert_version, open_group, v05

        paths = ome_zarr_image_paths(source)
        if len(paths) != 1:
            raise ValueError(f"Collection has {len(paths)} images; pass an image group path")
        group = open_group(paths[0])
        image = convert_version(group.ome_metadata(), "0.5")
        if not isinstance(image, v05.Image) or len(image.multiscales) != 1:
            raise ValueError("Input must identify a single OME-Zarr image group")
        multiscale = image.multiscales[0]
        metadata = _spatial_metadata(multiscale, calibration)
        metadata["source_metadata"] = image.model_dump(mode="json", by_alias=True)
        metadata["channel_names"] = [c.label for c in image.omero.channels] if image.omero else []
        for name in ("numerical_aperture", "wavelength_um"):
            if name in group.attrs:
                metadata[name] = group.attrs[name]
        return group[multiscale.datasets[0].path].to_tensorstore(), metadata

    import tifffile

    # tifffile owns OME XML parsing, series axes, pyramid levels and decoding.
    with tifffile.TiffFile(source) as stored:
        if not stored.is_ome or len(stored.series) != 1:
            raise ValueError("Input must contain a single OME-TIFF image")
        ome = tifffile.xml2dict(stored.ome_metadata)["OME"]
        image = ome["Image"]
        if isinstance(image, list):
            if len(image) != 1:
                raise ValueError("Input must contain a single OME-TIFF image")
            image = image[0]
        pixels = image["Pixels"]
        series = stored.series[0].levels[0]
        axes = series.axes.lower()
        spacing = np.array(
            [
                float(pixels.get("PhysicalSize" + a, 1))
                * _MICROMETERS_PER_UNIT[pixels.get("PhysicalSize" + a + "Unit", "µm")]
                for a in "ZYX"
            ]
        )
        origin = np.zeros(3)
        planes = pixels.get("Plane", [])
        if isinstance(planes, dict):
            planes = [planes]
        for i, axis in enumerate("ZYX"):
            translations = [
                float(p["Position" + axis])
                * _MICROMETERS_PER_UNIT[p.get("Position" + axis + "Unit", "µm")]
                - (int(p.get("TheZ", 0)) * spacing[0] if axis == "Z" else 0)
                for p in planes
                if "Position" + axis in p
            ]
            if translations:
                if not np.allclose(translations, translations[0]):
                    raise ValueError("Changing OME-TIFF plane translations require resampling")
                origin[i] = translations[0]
        channel_metadata = pixels.get("Channel", [])
        if isinstance(channel_metadata, dict):
            channel_metadata = [channel_metadata]
        metadata = dict(
            axes=axes,
            voxel_size_zyx_um=spacing,
            origin_zyx_um=origin,
            source_metadata=ome,
            channel_names=[c.get("Name", c.get("ID")) for c in channel_metadata],
        )
        return series.asarray(out="memmap"), metadata


def _prepare_data(
    data: np.ndarray | TensorStore, metadata: dict[str, Any]
) -> tuple[np.ndarray | TensorStore, tuple[tuple[int, int], ...]]:
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
    axes = metadata.get("axes") or {2: "yx", 3: "zyx", 5: "tczyx"}.get(data.ndim)
    if axes is None:
        raise ValueError("4D array input requires axes='tzyx' or 'czyx'")
    axes = axes.lower()
    if (
        len(axes) != data.ndim
        or len(set(axes)) != len(axes)
        or not set("yx") <= set(axes)
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
            if selected is not None and metadata.get("schema_version") != 1:
                raise ValueError(f"Set batch_axes=False to select a {label}")
            selections.append(range(size))
        else:
            if selected is None:
                raise ValueError(f"batch_axes=False requires an explicit {label}")
            if not isinstance(selected, int) or selected < 0 or selected >= size:
                raise ValueError(f"Selected {label} is outside the input")
            selections.append((selected,))

    # Create a lazy TCZYX view.
    for axis in "zct":
        if axis not in axes:
            data, axes = data[None], axis + axes
    data = data.transpose(tuple(axes.index(axis) for axis in "tczyx"))
    return data, tuple(product(*selections))


def _psf_diagnostics(
    data: np.ndarray | TensorStore, result: LocalizationResult, coords: tuple[np.ndarray, ...]
) -> None:
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
    metadata, fits = result["metadata"], result["fit_results"]
    metadata["psf_roi_size_px"] = get_roi_size(metadata["psf_roi_size_px"], (1, 1, 1))
    size, spacing = (
        metadata["psf_roi_size_px"],
        metadata["voxel_size_zyx_um"],
    )
    for key, volume in result["volume_results"].items():
        ids = volume["candidate_ids"]
        ids = ids[fits["success"][ids]]
        if "filter_pass" in fits:
            ids = ids[fits["filter_pass"][ids]]
        centers = fits["fit_params"][ids][:, (3, 2, 1)]

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
                backgrounds=fits["fit_params"][ids, 6 if fits["fit_params"].shape[1] == 7 else 7],
            )
            volume["average_psf_coords_zyx_um"] = tuple(get_psf_coords(size, spacing))


def analyze_psf(
    image: np.ndarray,
    fit_results: dict[str, Any],
    config: LocalizationConfig,
    *,
    roi_size_zyx_px: tuple[int, int, int] = DEFAULT_PSF_ROI_SIZE_PX,
    origin_zyx_um: tuple[float, float, float] | None = None,
) -> LocalizationResult:
    """Average PSFs and fit the focus surface of one already-localized 3D volume.

    ``image`` is original ZYX counts. Fit parameters use the Gaussian parameter
    order and physical µm coordinates; success and optional filter_pass masks
    select accepted beads. Cropped previews supply their first-voxel origin in
    µm. ROI sizes use voxels and the existing ROI utility makes them odd.
    No detection, localization or batch processing runs here. Returned metadata,
    fit_results and volume_results are compatible with plot_calibration; the
    sole volume has local key (0, 0). Insufficient focus evidence is reported
    separately, so a PSF average can still be returned for fewer than three beads.
    """
    if config.fit_model != "3d" or image.ndim != 3 or min(image.shape) < 2:
        raise ValueError("PSF averaging and field curvature require a 3D fit volume")
    parameters = np.asarray(fit_results["fit_params"])
    accepted = np.asarray(fit_results["success"]) & fit_results.get("filter_pass", True)
    if (
        parameters.ndim != 2
        or parameters.shape[1] not in (7, 11)
        or accepted.shape != (len(parameters),)
    ):
        raise ValueError(
            "Provide native Gaussian (N,7) or tilted (N,11) parameters and matching fit masks"
        )
    if not np.any(accepted):
        raise ValueError("No successful fits pass the current fit filters")
    if not np.isfinite(parameters[accepted]).all():
        raise ValueError("Accepted fit parameters must be finite")
    origin = config.origin_zyx_um if origin_zyx_um is None else origin_zyx_um
    result = dict(
        metadata=dict(
            config.model_dump(mode="json"), origin_zyx_um=origin, psf_roi_size_px=roi_size_zyx_px
        ),
        fit_results=fit_results,
        volume_results={(0, 0): dict(candidate_ids=np.arange(len(parameters)), average_psf=None)},
    )
    coords = get_coords(image.shape, config.voxel_size_zyx_um, origin=origin)
    _psf_diagnostics(image[None, None], result, coords)
    return result


def _write_parquet(reader: RecordBatchReader, path: Path) -> None:
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


def localize_ufish_prior(
    data: np.ndarray | TensorStore | str | PathLike,
    metadata: Mapping | LocalizationConfig | None = None,
    output_dir: str | PathLike | None = None,
) -> LocalizationResult:
    """Run shared CUDA detection and bounded fits on full spatial input.

    Parameters
    ----------
    data : numpy.ndarray, tensorstore.TensorStore, str or os.PathLike
        Original counts or a single OME-Zarr image group/OME-TIFF. File readers
        supply axis order and physical geometry; array input requires calibration.
    metadata : Mapping or LocalizationConfig, optional
        Strict version 1 settings or legacy optical metadata. ``batch_axes``
        selects all T/C indices or explicit selectors. Preview crops do not
        restrict batch scope; 2D processes each XY plane independently.
    output_dir : str or os.PathLike, optional
        Centroid checkpoint directory. Geometry/detector changes invalidate
        existing checkpoints; fitting-only changes reuse their centroids.

    Returns
    -------
    dict
        Metadata, parameter names, pixel/µm candidate centroids, T/C identities,
        ROI bounds, fit_results and volume_results keyed by (time, channel).
        Every candidate is retained. Fitted positions/widths use µm; covariance
        is optional. Array keys match the localization Parquet column names.

    Raises
    ------
    ValueError
        If settings, geometry, selections, checkpoints or starts are incompatible.
    RuntimeError
        If no real CUDA device is available in the installed environment.

    Notes
    -----
    yaozarrs supplies OME-Zarr metadata and lazy TensorStore views. Only the
    current inference volume (3D) or plane (2D) is materialized for U-FISH;
    subsequent fit batches read ROI samples. Original counts are fitted.
    """
    raw = (
        metadata.model_dump(mode="json")
        if isinstance(metadata, LocalizationConfig)
        else dict(metadata or {})
    )
    versioned = "schema_version" in raw
    if versioned:
        config = LocalizationConfig.model_validate_json(json.dumps(raw))
        metadata = dict(config.model_dump(mode="json"), psf_diagnostics=False)
    else:
        allowed = set(LocalizationConfig.model_fields) | {
            "axes",
            "psf_diagnostics",
            "psf_roi_size_px",
        }
        if unknown := set(raw) - allowed:
            raise ValueError(f"Unsupported localization settings: {sorted(unknown)}")
        metadata = {
            "batch_axes": True,
            "psf_diagnostics": False,
            "intensity_threshold": LocalizationConfig().intensity_threshold,
            "fit_roi_size_um": (1.75, 0.65, 0.65),
            "psf_roi_size_px": DEFAULT_PSF_ROI_SIZE_PX,
            **raw,
        }
    source = None
    if isinstance(data, (str, PathLike)):
        source = str(data) if "://" in str(data) else str(Path(data).resolve())
        data, stored = open_image(source, raw)
        # Stored axis order always describes the actual input. Explicit exported
        # calibration overrides storage geometry; crop offsets are never reapplied.
        metadata.update(
            {name: value for name, value in stored.items() if name == "axes" or name not in raw}
        )
    elif versioned:
        metadata["axes"] = config.preview.axes
    data, keys = _prepare_data(data, metadata)
    spacing = np.asarray(metadata["voxel_size_zyx_um"], dtype=float)
    origin = np.asarray(metadata.get("origin_zyx_um", (0, 0, 0)), dtype=float)
    metadata.update(voxel_size_zyx_um=spacing, origin_zyx_um=origin)
    if (
        spacing.shape != (3,)
        or origin.shape != (3,)
        or not np.isfinite([spacing, origin]).all()
        or np.any(spacing <= 0)
    ):
        raise ValueError("Input calibration requires finite ZYX origins and positive voxel sizes")
    mode = metadata.get("fit_model", "3d")
    if mode == "3d" and data.shape[2] < 2:
        raise ValueError("3D fitting requires a Z stack; select 2D for an XY image")
    if mode == "2d" and metadata["psf_diagnostics"]:
        raise ValueError("PSF diagnostics require 3D volumes")

    configs = {}
    for channel in {c for _, c in keys}:
        if versioned:
            configs[channel] = config
            continue
        for name in (
            "numerical_aperture",
            "wavelength_um",
            "refractive_index",
            "relative_sigma_range",
        ):
            if metadata.get(name) is None:
                raise ValueError(f"Provide {name} in metadata")
        wavelength = metadata["wavelength_um"]
        if isinstance(wavelength, Mapping):
            wavelength = wavelength.get(channel, wavelength.get(str(channel)))
        if wavelength is None:
            raise ValueError(f"Provide wavelength_um for channel {channel}")
        widths = gaussian_width_bounds(
            metadata["numerical_aperture"],
            wavelength,
            metadata["refractive_index"],
            spacing,
            relative_sigma_range=metadata["relative_sigma_range"],
            sampling_floor_pixels=0,
            use_gpu=False,
        )
        parameters = dict(LocalizationConfig().fit_parameters)
        for name, axis in (("sxy", 1), ("sz", 0)):
            parameters[name] = Parameter(
                lower=float(widths["lower_sigma_zyx_um"][axis]),
                upper=float(widths["upper_sigma_zyx_um"][axis]),
                initial=float(widths["nominal_sigma_zyx_um"][axis]),
            )
        options = {
            name: metadata[name] for name in LocalizationConfig.model_fields if name in metadata
        }
        options.update(
            wavelength_um=float(wavelength),
            fit_parameters=metadata.get("fit_parameters", parameters),
            return_covariance=metadata.get("return_covariance", True),
            voxel_size_zyx_um=tuple(spacing),
            origin_zyx_um=tuple(origin),
        )
        configs[channel] = LocalizationConfig.model_validate_json(
            json.dumps(
                options,
                default=lambda v: v.model_dump() if isinstance(v, Parameter) else v.tolist(),
            )
        )

    import cupy as cp
    import torch
    from tqdm.auto import tqdm
    from ufish.utils.log import logger as ufish_logger

    if not torch.cuda.is_available() or cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("Localization requires a real CUDA GPU in the installed environment")
    ufish_logger.disable("ufish")
    representative = next(iter(configs.values()))
    weights = Path(representative.ufish_weights)
    weights_hash = None
    if weights.is_file():
        with weights.open("rb") as stream:
            weights_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    identity = dict(
        source=source,
        shape=list(data.shape),
        dtype=str(data.dtype),
        axes=metadata["axes"],
        voxel_size_zyx_um=spacing.tolist(),
        origin_zyx_um=origin.tolist(),
        ufish_weights=representative.ufish_weights,
        weights_sha256=weights_hash,
        fit_model=mode,
        blend_3d=representative.blend_3d,
        intensity_threshold=representative.intensity_threshold,
        collapse_close_priors=representative.collapse_close_priors,
        prior_merge_optics={
            str(channel): [cfg.numerical_aperture, cfg.wavelength_um, cfg.refractive_index]
            for channel, cfg in configs.items()
        }
        if representative.collapse_close_priors
        else None,
        exclude_edge_centroids=representative.exclude_edge_centroids,
        edge_margin_px=representative.edge_margin_px
        if representative.exclude_edge_centroids
        else 0,
        inference_batch_size=representative.inference_batch_size,
        preprocessing_policy="ufish.predict-default-v1;fit-original-counts"
        + (";median-gpu-size3-reflect-v1" if representative.ufish_median_filter else "")
        + (";collapse-optical-fwhm-radial-v2" if representative.collapse_close_priors else "")
        + (";edge-projected-footprint-v1" if representative.exclude_edge_centroids else ""),
    )
    inference_keys = (
        tuple((t, c, z) for t, c in keys for z in range(data.shape[2])) if mode == "2d" else keys
    )
    checkpoint, completed = None, set()
    if output_dir is not None:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        checkpoint = Path(output_dir) / "centroids.parquet"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        schema = pa.schema(
            [("timepoint", pa.int64()), ("channel", pa.int64())]
            + [(axis + "_px", pa.float64()) for axis in "zyx"]
        )
        saved = schema.empty_table()
        if checkpoint.exists():
            saved = pq.read_table(checkpoint)
            previous = json.loads((saved.schema.metadata or {}).get(b"localize_psf", b"{}"))
            completed = {tuple(key) for key in previous.pop("completed_volumes", [])}
            previous.setdefault("exclude_edge_centroids", False)
            previous.setdefault("collapse_close_priors", False)
            previous.setdefault("prior_merge_optics", None)
            previous.setdefault("edge_margin_px", 1 if previous["exclude_edge_centroids"] else 0)
            if (
                previous["prior_merge_optics"] is not None
                and identity["prior_merge_optics"] is not None
            ):
                # New channels have no saved priors; retain and validate optics of existing ones.
                identity["prior_merge_optics"] = (
                    previous["prior_merge_optics"] | identity["prior_merge_optics"]
                )
                for channel, optics in identity["prior_merge_optics"].items():
                    previous["prior_merge_optics"].setdefault(channel, optics)
            if previous != identity:
                raise ValueError(
                    "Centroid checkpoint has different input or U-FISH settings; use a different output directory"
                )

    detector = Detector()
    candidates, offsets = [], [0]
    try:
        for key in tqdm(inference_keys, desc="U-FISH inference", unit="volume"):
            t, c = key[:2]
            if key in completed:
                mask = (pc.field("timepoint") == t) & (pc.field("channel") == c)
                if mode == "2d":
                    mask = mask & (pc.field("z_px") == key[2])
                table = saved.filter(mask)
                points = np.column_stack([table[axis + "_px"].to_numpy() for axis in "zyx"])
            else:
                image = np.asarray(data[t, c, key[2] : key[2] + 1] if mode == "2d" else data[t, c])
                if not np.isfinite(image).all() or (
                    representative.estimator == "MLE" and np.any(image < 0)
                ):
                    raise ValueError(
                        "Localization requires finite counts; Poisson MLE requires nonnegative counts"
                    )
                points = detector.detect(image, configs[c], cache=False)[0]
                del image
                if mode == "2d":
                    points[:, 0] += key[2]
                if checkpoint is not None:
                    columns = dict(
                        timepoint=np.full(len(points), t, int),
                        channel=np.full(len(points), c, int),
                        **{axis + "_px": points[:, i] for i, axis in enumerate("zyx")},
                    )
                    saved = pa.concat_tables([saved, pa.table(columns, schema=schema)])
                    completed.add(key)
                    status = json.dumps(dict(identity, completed_volumes=sorted(completed))).encode(
                        "utf-8"
                    )
                    saved = saved.replace_schema_metadata({b"localize_psf": status})
                    _write_parquet(saved.to_reader(), checkpoint)
            candidates.append(points)
            offsets.append(offsets[-1] + len(points))
    finally:
        detector.release()
    candidates = np.concatenate(candidates)
    times, channels = np.repeat(np.asarray(inference_keys)[:, :2].T, np.diff(offsets), axis=1)
    centers = origin + candidates * spacing
    coords = get_coords(data.shape[-3:], spacing, origin=origin)
    roi_size = np.asarray(get_roi_size(representative.fit_roi_size_um, spacing, ensure_odd=True))
    if mode == "2d":
        roi_size[0] = 1
    rois = get_centered_rois(np.rint(candidates).astype(int), roi_size, [0, 0, 0], data.shape[-3:])
    fits = None
    for channel, config in list(configs.items()):
        if config.psf_model != "tilted_gaussian" or config.psf_tilt_yz_rad is not None:
            continue
        for key, first, last in zip(inference_keys, offsets[:-1], offsets[1:], strict=True):
            if key[1] != channel or last - first < 3:
                continue
            angle = calibrate_tilt(np.asarray(data[key[:2]]), candidates[first:last], config)
            if angle is not None:
                configs[channel] = config.model_copy(update={"psf_tilt_yz_rad": angle})
                break
        else:
            if np.any(channels == channel):
                raise ValueError(
                    "Automatic tilt calibration found no volume with three isolated, fully supported spots; export a calibrated preview"
                )
    if representative.psf_model == "tilted_gaussian":
        metadata["psf_tilt_yz_by_channel_rad"] = {
            str(c): cfg.psf_tilt_yz_rad for c, cfg in configs.items()
        }
        if len(configs) == 1:
            metadata["psf_tilt_yz_rad"] = next(iter(configs.values())).psf_tilt_yz_rad
    model = gaussian_model(representative)
    for key, first, last in tqdm(
        zip(inference_keys, offsets[:-1], offsets[1:], strict=True),
        total=len(inference_keys),
        desc="Gaussian fitting",
        unit="volume",
    ):
        config = configs[key[1]]
        if first == last:
            continue
        # Infer support once per volume/plane, including resumed centroid runs.
        image = data[key[:2]]
        footprint = data_footprint(
            np.asarray(image[key[2] : key[2] + 1] if mode == "2d" else image)
        )
        footprint = np.broadcast_to(footprint, image.shape)
        for start in range(first, last, config.roi_batch_size):
            stop = min(start + config.roi_batch_size, last)
            selected = slice(start, stop)
            # Each inference key owns one source volume; pack its lazy view directly.
            values, coordinates, counts = prepare_rois(
                image, coords, rois[selected], mask=footprint
            )
            truncated = counts != np.prod(roi_size)
            batch = fit_packed_rois(
                values, coordinates, counts, centers[selected], config, truncated, model=model
            )
            fits = _accumulate_fit_batch(fits, batch, start, len(candidates))
        del footprint, image
        if fits is not None and last > first:
            # Finish the entire volume/plane before suppressing converged duplicates.
            selected = slice(first, last)
            distances = (
                optical_parameters(config)["collapse_distance_zyx_um"]
                if config.collapse_close_priors
                else None
            )
            fits["filter_pass"][selected] = fit_filter_mask(
                fits["fit_params"][selected],
                fits["success"][selected],
                fits["covariance_valid"][selected] if "covariance_valid" in fits else None,
                config.fit_filters,
                collapse_distance_zyx_um=distances,
            )
    if fits is None:
        empty = np.empty((0, 0))
        fits = fit_packed_rois(
            empty,
            (empty,) * 3,
            np.empty(0, int),
            np.empty((0, 3)),
            representative,
            np.empty(0, bool),
            model=model,
        )
    volume_offsets = offsets[:: data.shape[2] if mode == "2d" else 1]
    volumes = {
        key: dict(
            candidate_ids=np.arange(first, last),
            focus=dict(available=False),
            average_psf=None,
        )
        for key, first, last in zip(keys, volume_offsets[:-1], volume_offsets[1:], strict=True)
    }
    result = dict(
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


def _focus_diagnostics(centers: np.ndarray) -> dict[str, Any]:
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
    def curvature_model_fn(p: np.ndarray) -> np.ndarray:
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

    curvature = fit_model(residual, curvature_model_fn, np.array([np.mean(x), np.mean(y), 0]))
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


def plot_calibration(result: LocalizationResult) -> dict[str, Figure]:
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
    for (_, channel), volume in result["volume_results"].items():
        if not volume["focus"]["available"]:
            continue
        ids = volume["candidate_ids"][result["fit_results"]["success"][volume["candidate_ids"]]]
        if "filter_pass" in result["fit_results"]:
            ids = ids[result["fit_results"]["filter_pass"][ids]]
        centers = result["fit_results"]["fit_params"][ids][:, (3, 2, 1)]
        focus = volume["focus"]
        prefix = "" if len(result["volume_results"]) == 1 else f"c{channel}_"

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
            result["metadata"]["voxel_size_zyx_um"][1:],
            origin=result["metadata"]["origin_zyx_um"][1:],
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
            limits = [[np.percentile(weights, 1), np.percentile(weights, 99)]] if i < 2 else None
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


def to_trackpy_dataframe(table: Table, *, converged_only: bool = True) -> DataFrame:
    """Convert Gaussian fits to a Trackpy-compatible dataframe.

    Parameters
    ----------
    table : pyarrow.Table
        Localization table containing fit parameters, optional center covariance,
        success and covariance flags, timepoint/channel indices, and the
        ``localize_psf`` schema metadata with image geometry.
    converged_only : bool, optional
        Keep converged fits by default. False retains accepted nonconverged
        records for inspection and includes their ``success`` flags.

    Returns
    -------
    pandas.DataFrame
        Converged fits accepted by any ``filter_pass`` mask, with ``frame``,
        ``channel``, Gaussian amplitude as
        ``signal`` and ``covariance_valid``. Positions ``x/y/z``, widths
        ``size_*`` and errors ``ep_*`` use pixels; ``xum/yum/zum`` use world
        coordinates in micrometers. Invalid covariance retains NaN errors.
    """
    import pandas as pd

    table = table.select([name for name in _TRACKPY_SOURCE_COLUMNS if name in table.column_names])
    if converged_only:
        table = table.filter(table["success"])
    if "filter_pass" in table.column_names:
        table = table.filter(table["filter_pass"])
    metadata = json.loads(table.schema.metadata[b"localize_psf"])["metadata"]
    spacing = np.asarray(metadata["voxel_size_zyx_um"])[::-1]
    origin = np.asarray(metadata["origin_zyx_um"])[::-1]

    # Extract fitted parameters and position errors from the Arrow lists.
    summary = json.loads(table.schema.metadata[b"localize_psf"])
    names = summary.get("parameter_names", ("amp", "cx", "cy", "cz", "sxy", "sz", "bg"))
    parameters = table["fit_params"].combine_chunks().flatten().to_numpy().reshape(-1, len(names))
    if "sxy" in names:
        widths = parameters[:, [4, 4, 5]]
    else:
        from localize_psf.rotation import euler_mat

        rotation = euler_mat(*parameters[:, 8:].T)
        widths = np.sqrt(np.einsum("ijn,nj->ni", rotation**2, parameters[:, 4:7] ** 2))
    if "center_covariance_zyx" in table.column_names:
        covariance = table["center_covariance_zyx"].combine_chunks().flatten().flatten().to_numpy()
        covariance = covariance.reshape(-1, 3, 3)
        errors = np.sqrt(np.diagonal(covariance, axis1=1, axis2=2)[:, ::-1]) / spacing
    else:
        errors = np.full((len(parameters), 3), np.nan)

    # Build Trackpy positions, signals, widths and position errors.
    columns = dict(
        frame=np.asarray(table["candidate_timepoints"]),
        channel=np.asarray(table["candidate_channels"]),
        signal=parameters[:, 0],
        covariance_valid=np.asarray(table["covariance_valid"])
        if "covariance_valid" in table.column_names
        else np.zeros(len(parameters), bool),
    )
    if "roi_truncated" in table.column_names:
        columns["roi_truncated"] = np.asarray(table["roi_truncated"])
    if not converged_only:
        columns["success"] = np.asarray(table["success"])
    for index, axis in enumerate("xyz"):
        columns[axis] = (parameters[:, index + 1] - origin[index]) / spacing[index]
        columns[axis + "um"] = parameters[:, index + 1]
        columns["size_" + axis] = widths[:, index] / spacing[index]
        columns["ep_" + axis] = errors[:, index]

    return pd.DataFrame(columns)


def read_localizations(
    path: str | PathLike,
    *,
    timepoint: int | None = None,
    channel: int | None = None,
) -> tuple[DataFrame, dict[str, Any]]:
    """Read saved batch fits and geometry without loading images or running CUDA.

    ``path`` is a batch output directory or localization Parquet file. Optional
    zero-based T/C selectors are pushed down to the Parquet reader. Return all
    accepted records, including nonconverged fits, and their saved metadata.
    The dataframe uses the existing Trackpy convention: ``frame``/``channel``
    indices, ``x/y/z`` in full-image pixels, ``xum/yum/zum`` in micrometers,
    amplitude ``signal`` in original counts, widths/errors in pixels, and
    ``success``/``covariance_valid`` flags. Empty selections retain the columns.
    """
    import pyarrow.parquet as pq

    source = Path(path).expanduser()
    if source.is_dir():
        source = source / "localizations.parquet"
    schema = pq.read_schema(source)
    required = {"fit_params", "success", "candidate_timepoints", "candidate_channels"}
    if not required <= set(schema.names) or b"localize_psf" not in (schema.metadata or {}):
        raise ValueError("Select a localize-psf localizations.parquet file")
    metadata = json.loads(schema.metadata[b"localize_psf"])["metadata"]
    # Reuse the production geometry validation before converting stored centers.
    LocalizationConfig.model_validate_json(
        json.dumps(
            {
                "voxel_size_zyx_um": metadata["voxel_size_zyx_um"],
                "origin_zyx_um": metadata["origin_zyx_um"],
            }
        )
    )
    filters = [
        (name, "=", value)
        for name, value in (("candidate_timepoints", timepoint), ("candidate_channels", channel))
        if value is not None
    ]
    table = pq.read_table(
        source,
        columns=[name for name in _TRACKPY_SOURCE_COLUMNS if name in schema.names],
        filters=filters or None,
    )
    return to_trackpy_dataframe(table, converged_only=False), metadata


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
            for batch in source.iter_batches(
                columns=[
                    name for name in _TRACKPY_SOURCE_COLUMNS if name in source.schema_arrow.names
                ]
            )
        )
        _write_parquet(pa.RecordBatchReader.from_batches(schema, batches), output)

    return output


def save_localizations(
    result: LocalizationResult, output_dir: str | PathLike, export_trackpy: bool = False
) -> None:
    """Save localizations and optional Trackpy and static PSF exports.

    Parameters
    ----------
    result : LocalizationResult
        Candidate and fit arrays, metadata and any static PSF diagnostics.
    output_dir : str or os.PathLike
        Destination directory, created if needed. Writes
        ``localizations.parquet`` with the fits accepted by ``fit_filters``, and
        available PSF TIFFs and diagnostic PNGs. All candidate evidence remains
        in the input result; original candidate IDs are preserved in the export.
    export_trackpy : bool, optional
        Also write ``trackpy.parquet`` containing accepted converged fits.

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
        for name, value in result["fit_results"].items()
        if isinstance(value, np.ndarray)
    }
    arrays.update(
        candidate_centroids_zyx_px=result["candidate_centroids_zyx_px"],
        candidate_centroids_zyx_um=result["candidate_centroids_zyx_um"],
        candidate_timepoints=result["candidate_timepoints"],
        candidate_channels=result["candidate_channels"],
        roi_bounds_zyx=result["roi_bounds_zyx"],
        bounds_lower=result["fit_results"]["bounds"][0],
        bounds_upper=result["fit_results"]["bounds"][1],
    )
    for name, flags in result["fit_results"].get("covariance_flags", {}).items():
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
        metadata=result["metadata"],
        parameter_names=result["parameter_names"],
        fit_states_key=result["fit_results"]["fit_states_key"],
        volumes={
            f"t{t}_c{c}": dict(focus=v["focus"]) for (t, c), v in result["volume_results"].items()
        },
    )
    encoded = json.dumps(
        summary, default=lambda v: v.tolist() if isinstance(v, np.ndarray) else v.item()
    )
    table = pa.table(columns).replace_schema_metadata({b"localize_psf": encoded.encode("utf-8")})
    if "filter_pass" in table.column_names:
        table = table.filter(table["filter_pass"])
    _write_parquet(table.to_reader(), output / "localizations.parquet")
    if export_trackpy:
        save_trackpy(output / "localizations.parquet")

    # Export the averaged PSFs in the original ImageJ TIFF format.
    dz, dy, dx = result["metadata"]["voxel_size_zyx_um"]
    for (_, channel), volume in result["volume_results"].items():
        if volume["average_psf"] is None:
            continue
        prefix = "" if len(result["volume_results"]) == 1 else f"c{channel}_"
        wavelength = result["metadata"]["wavelength_um"]
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
    data: str = typer.Argument(..., help="Single OME-Zarr image group or OME-TIFF"),
    output_dir: Path = typer.Option(..., "--output-dir"),
    metadata_path: Path = typer.Option(..., "--metadata", exists=True, dir_okay=False),
    export_trackpy: bool = typer.Option(
        False, "--export-trackpy", help="Also write converged fits to trackpy.parquet"
    ),
) -> None:
    """Localize an OME-Zarr or OME-TIFF image with optional static PSF calibration.

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
    try:
        metadata = json.loads(metadata_path.read_text("utf-8"))
        result = localize_ufish_prior(data, metadata, output_dir=output_dir)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error

    # Write the requested tables and optional calibration diagnostics.
    save_localizations(result, output_dir, export_trackpy=export_trackpy)
    typer.echo(
        f"Converged {result['fit_results']['success'].sum()} / "
        f"{len(result['candidate_centroids_zyx_px'])} candidates; saved {output_dir}"
    )


if __name__ == "__main__":
    app()
