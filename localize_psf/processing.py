"""Shared detector and parameter policy for interactive and batch localization."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import CancelledError
from itertools import combinations
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from .config import FitFilters, LocalizationConfig

if TYPE_CHECKING:
    from cupy import ndarray as CudaArray
    from tensorstore import TensorStore
    from ufish.api import UFish

    from localize_psf.fit_psf import gaussian3d_psf_model


# Preserve the caller's public type name for the production fitter's result dictionary.
type FitResults = dict[str, Any]


def fit_filter_mask(
    parameters: NDArray[np.float64],
    success: NDArray[np.bool_],
    covariance_valid: NDArray[np.bool_] | None,
    filters: FitFilters,
) -> NDArray[np.bool_]:
    """Return post-fit acceptance without removing candidates or changing evidence.

    ``parameters`` has shape (N,7) in the production Gaussian's parameter order;
    column zero is amplitude in original image counts. ``success`` and optional
    ``covariance_valid`` are upstream boolean (N,) flags. All selected conditions
    must pass. Missing covariance raises when either covariance selector is used.
    Empty inputs return an empty mask; default filters accept every candidate.
    """
    from localize_psf.localize import range_filter

    keep = np.ones(len(parameters), dtype=bool)
    if filters.amplitude_min is not None or filters.amplitude_max is not None:
        keep &= (
            range_filter(
                -np.inf if filters.amplitude_min is None else filters.amplitude_min,
                np.inf if filters.amplitude_max is None else filters.amplitude_max,
                0,
                "amplitude",
            )
            .filter(parameters)
            .all(axis=1)
        )
    if filters.convergence != "all":
        keep &= success == (filters.convergence == "pass")
    if filters.covariance != "all":
        if covariance_valid is None:
            raise ValueError(
                "Covariance diagnostics were not computed. Enable covariance and rerun GPU fits."
            )
        keep &= covariance_valid == (filters.covariance == "pass")
    return keep


def prepare_volume(
    data: np.ndarray | TensorStore, config: LocalizationConfig
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Read the configured preview crop as original float32 ZYX counts.

    ``data`` is a NumPy array or lazy TensorStore in ``config.preview.axes``
    order. Preview settings select T/C, crop and, for 2D, one XY plane.
    Only that selection is materialized; no projection is performed.

    Return the image and its crop-first-voxel ZYX origin in µm, applying the
    crop offset once. Spacing remains in config. Incompatible selections,
    geometry or counts raise ValueError; Poisson MLE requires nonnegative counts.
    """
    settings = config.preview
    if len(settings.axes) != data.ndim:
        raise ValueError("Select an image with an explicit, matching axis order")
    crop = np.asarray(settings.crop_zyx_px).reshape(3, 2).copy()
    if config.fit_model == "2d":
        crop[0] = (settings.z_plane, settings.z_plane + 1) if "z" in settings.axes else (0, 1)
    elif "z" not in settings.axes or data.shape[settings.axes.index("z")] < 2:
        raise ValueError("3D fitting requires a Z stack; select 2D for an XY image")
    indices: list[int | slice] = []
    for axis, size in zip(settings.axes, data.shape, strict=True):
        if axis in "tc":
            index = settings.timepoint if axis == "t" else settings.channel
            if index >= size:
                raise ValueError(f"Selected {axis} index is outside the image")
            indices.append(index)
        else:
            start, stop = crop["zyx".index(axis)]
            if stop > size:
                raise ValueError(f"Preview crop exceeds the {axis} dimension ({size})")
            indices.append(slice(int(start), int(stop)))
    selected = data[tuple(indices)]
    spatial_axes = "".join(a for a in settings.axes if a not in "tc")
    selected = selected.transpose(tuple(spatial_axes.index(a) for a in "zyx" if a in spatial_axes))
    image = np.asarray(selected, dtype=np.float32)
    if "z" not in spatial_axes:
        image = image[None]
    if not np.isfinite(image).all():
        raise ValueError("Preview data must contain only finite counts")
    if config.estimator == "MLE" and np.any(image < 0):
        raise ValueError("Poisson MLE requires nonnegative photon/electron counts")
    if min(image.shape[1:]) < 3:
        raise ValueError("Preview must contain at least three pixels along X and Y")
    origin = np.asarray(config.origin_zyx_um) + crop[:, 0] * config.voxel_size_zyx_um
    return image, tuple(origin)


class Detector:
    """Cache one production enhancement so threshold tuning only recalls spots."""

    def __init__(self) -> None:
        self._detector: UFish | None = None
        self._weights: str | None = None
        self._key: tuple[int, bool, int, str, str] | None = None
        self._image: np.ndarray | None = None
        self._enhanced: NDArray[np.float32] | None = None
        self._edge_distances: list[np.ndarray] | None = None

    def detect(
        self, image: np.ndarray, config: LocalizationConfig, *, cache: bool = True
    ) -> tuple[NDArray[np.float64], NDArray[np.float32]]:
        """Run CUDA U-FISH with the exported weights and inference settings.

        ``image`` supplies original ZYX counts (singleton Z in 2D). ``config``
        supplies weights, dimensional mode, blending, threshold and batch size.
        Return (crop-local ZYX pixel priors, enhancement). Priors have shape
        (N,3), with zero Z in 2D; enhancement is float32 ZYX or YX respectively.

        ``exclude_edge_centroids`` removes a voxel-grid distance band at the
        array faces and the nonzero data footprint in each axis projection.
        This follows sloping zero-padded boundaries without treating interior
        zero counts as holes. 2D uses only YX. Enhancement and footprint distances
        are reused when changing the margin; the same priors drive batch fitting.

        With ``cache=True``, retain the image identity and enhancement so threshold
        changes only recall spots. A new image, weights, dimensional mode, blend
        or batch size invalidates enhancement. ``cache=False`` streams batch
        volumes without retaining images; loaded weights are reused in both modes.
        """
        from ufish.api import UFish

        axes = "yx" if config.fit_model == "2d" else "zyx"
        if image.ndim != 3 or (axes == "yx" and image.shape[0] != 1):
            raise ValueError("Detection requires ZYX counts, with a single XY plane in 2D")
        key = (
            id(image),
            config.blend_3d,
            config.inference_batch_size,
            config.ufish_weights,
            axes,
        )
        if not cache or self._key != key:
            self._enhanced = self._image = self._key = None
            self._edge_distances = None
            if self._detector is None or self._weights != config.ufish_weights:
                self._weights = None
                self._detector = UFish(device="cuda")
                weights = config.ufish_weights
                if Path(weights).is_file():
                    self._detector.load_weights(weights_path=weights)
                else:
                    self._detector.load_weights(weights_file=weights)
                self._weights = weights
            spots, enhanced = self._detector.predict(
                image[0] if axes == "yx" else image,
                axes=axes,
                blend_3d=config.blend_3d,
                batch_size=config.inference_batch_size,
                intensity_threshold=config.intensity_threshold,
            )
            if cache:
                self._enhanced = enhanced
                self._image, self._key = image, key
        else:
            assert self._detector is not None and self._enhanced is not None
            enhanced = self._enhanced
            spots = self._detector.call_spots(
                self._enhanced, intensity_threshold=config.intensity_threshold
            )
        points = spots[[f"axis-{i}" for i in range(len(axes))]].to_numpy(dtype=float)
        if axes == "yx":
            points = np.column_stack((np.zeros(len(points)), points))
        if config.exclude_edge_centroids and len(points):
            from scipy.ndimage import binary_fill_holes, distance_transform_edt

            pairs = tuple(combinations(range(1 if axes == "yx" else 0, 3), 2))
            distances = self._edge_distances if cache else None
            if distances is None:
                support = (image[0] if axes == "yx" else image) != 0
                distances = []
                for pair in pairs:
                    projected = support if axes == "yx" else np.any(support, axis=3 - sum(pair))
                    footprint = binary_fill_holes(projected)
                    distances.append(
                        distance_transform_edt(np.pad(footprint, 1))[1:-1, 1:-1].astype(np.float32)
                    )
                if cache:
                    self._edge_distances = distances
            keep = np.ones(len(points), dtype=bool)
            for pair, distance in zip(pairs, distances, strict=True):
                keep &= distance[tuple(points[:, pair].astype(np.intp).T)] > config.edge_margin_px
            points = points[keep]
        return points, enhanced

    def release(self) -> None:
        """Release the inference model before GPUfit; retain no stale cache key."""
        loaded = self._detector is not None
        self._detector = self._enhanced = self._key = self._image = None
        self._edge_distances = None
        self._weights = None
        if loaded:
            import torch

            torch.cuda.empty_cache()


def optical_parameters(config: LocalizationConfig) -> dict[str, tuple[float, float, float]]:
    """Return production nominal/lower/upper optical sigmas as ZYX µm triples.

    Uses CUDA ``gaussian_width_bounds`` with the configuration's NA, wavelength,
    refractive index, voxel spacing and relative sigma range. Result keys are
    ``nominal_sigma_zyx_um``, ``lower_sigma_zyx_um`` and ``upper_sigma_zyx_um``.
    """
    from localize_psf.optical_bounds import gaussian_width_bounds

    result = gaussian_width_bounds(
        config.numerical_aperture,
        config.wavelength_um,
        config.refractive_index,
        config.voxel_size_zyx_um,
        relative_sigma_range=config.relative_sigma_range,
        sampling_floor_pixels=0,
        use_gpu=True,
    )
    return {name: tuple(value.get().tolist()) for name, value in result.items()}


def fit_spots(
    image: np.ndarray,
    points: NDArray[np.float64],
    config: LocalizationConfig,
    cancelled: Event | None = None,
    progress: Callable[[str], None] | None = None,
    *,
    origin_zyx_um: tuple[float, float, float] | None = None,
) -> FitResults:
    """Fit crop-local ZYX pixel priors on original image counts with GPUfit.

    ``image`` contains ZYX counts; ``points`` is a finite (N,3) pixel array
    inside it. ``config`` supplies spacing in µm, ROI dimensions, parameter
    policies, estimator, solver/batch controls
    and covariance selection. Center policies are µm offsets from each prior,
    intersected with ROI limits; infeasible starts raise ValueError without clipping.

    Return all candidates in input order with IDs, statuses, active bounds and
    requested uncertainty evidence, plus the post-fit ``filter_pass`` mask.
    Fitted positions and widths use µm. 2D
    requires one XY plane and fixes the native 3D Gaussian's cz to that plane's
    physical Z and sz to 1 µm. CUDA workspaces are reused between fit batches.

    ``cancelled`` is an optional Event checked between batches; cancellation
    raises CancelledError. ``progress`` optionally receives each batch's status
    string. Invalid priors or geometry raise ValueError.

    ``origin_zyx_um`` is the input image's first-voxel coordinate in µm,
    defaulting to config's source origin. For a crop, pass prepare_volume's
    returned origin so its offset is applied exactly once.
    """
    import cupy as cp

    from localize_psf.batch_fitting import _accumulate_fit_batch
    from localize_psf.fit_psf import gaussian3d_psf_model
    from localize_psf.localize import get_coords, prepare_rois
    from localize_psf.rois import get_centered_rois, get_roi_size

    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Candidate coordinates must be finite (N,3) ZYX pixels")
    if np.any(points < 0) or np.any(points >= np.asarray(image.shape)):
        raise ValueError("Candidate coordinates must lie inside the preview")
    model = gaussian3d_psf_model()
    if not len(points):
        empty = np.empty((0, 0))
        return fit_packed_rois(
            empty,
            (empty,) * 3,
            np.empty(0, int),
            np.empty((0, 3)),
            config,
            np.empty(0, bool),
            model=model,
        )
    spacing = config.voxel_size_zyx_um
    origin = config.origin_zyx_um if origin_zyx_um is None else origin_zyx_um
    roi_size = np.asarray(get_roi_size(config.fit_roi_size_um, spacing, ensure_odd=True))
    if config.fit_model == "2d":
        if image.shape[0] != 1:
            raise ValueError("2D fitting requires a single XY plane")
        roi_size[0] = 1
    image = cp.asarray(image)
    coords = get_coords(image.shape, spacing, origin=origin, use_gpu=True)
    centers = cp.asarray(origin) + cp.asarray(points) * cp.asarray(spacing)
    rois = get_centered_rois(
        cp.rint(cp.asarray(points)).astype(int), roi_size, [0, 0, 0], image.shape
    )
    complete = None
    try:
        for start in range(0, len(points), config.roi_batch_size):
            if cancelled is not None and cancelled.is_set():
                raise CancelledError("Localization cancelled")
            stop = min(start + config.roi_batch_size, len(points))
            if progress is not None:
                progress(f"Fitting candidates {start + 1}–{stop} of {len(points)}")
            selected = slice(start, stop)
            values, coordinates, sizes = prepare_rois(image, coords, rois[selected])
            truncated = cp.any(
                rois[selected, 1::2] - rois[selected, ::2] != cp.asarray(roi_size), axis=1
            )
            batch = fit_packed_rois(
                values, coordinates, sizes, centers[selected], config, truncated, model=model
            )
            complete = _accumulate_fit_batch(complete, batch, start, len(points))
            del values, coordinates, batch
    finally:
        del image, coords
        cp.get_default_memory_pool().free_all_blocks()
    assert complete is not None
    return complete


def fit_packed_rois(
    values: np.ndarray | CudaArray,
    coordinates: tuple[np.ndarray | CudaArray, ...],
    sizes: np.ndarray | CudaArray,
    centers_zyx_um: np.ndarray | CudaArray,
    config: LocalizationConfig,
    truncated: np.ndarray | CudaArray,
    *,
    model: gaussian3d_psf_model | None = None,
) -> FitResults:
    """Apply exported parameter policy and GPUfit to packed original counts.

    ``values`` and the ZYX ``coordinates`` share a NumPy/CuPy backend and shape
    (N,P), with NaN padding after each ROI's valid ``sizes`` samples. Physical
    priors ``centers_zyx_um`` have shape (N,3); ``truncated`` is a boolean (N,)
    boundary-clipping flag. An optional production ``model`` is reused across batches.

    ``config`` controls parameters, estimator, solver and covariance. Center
    bounds/starts are prior-relative µm offsets; bounds intersect ROI limits.
    Null starts retain model estimates, and infeasible starts raise ValueError
    without clipping. 2D fixes cz at the plane and sz at 1 µm. Return FitResults
    for every candidate, including statuses, active bounds, uncertainty flags and
    a ``filter_pass`` mask evaluated using ``config.fit_filters``.

    Estimates use the input backend. Bounds/policies reuse the fitter's required
    host conversion, avoiding per-ROI CUDA reductions and extra buffer transfers.
    """
    from localize_psf.batch_fitting import _host_array, fit_rois
    from localize_psf.fit_psf import gaussian3d_psf_model

    model = gaussian3d_psf_model() if model is None else model
    centers = _host_array(centers_zyx_um)
    fixed = np.zeros(model.nparams, dtype=bool)
    if len(values):
        initial = _host_array(
            model.estimate_parameters(values, coordinates, num_preserved_dims=1), dtype=float
        )
        coordinates = tuple(_host_array(c, dtype=float) for c in coordinates)
        sizes = _host_array(sizes)
        limits = [
            model.estimate_bounds(tuple(c[i, : int(count)] for c in coordinates))
            for i, count in enumerate(sizes)
        ]
        limits = np.asarray(limits)
        lower, upper = limits[:, 0], limits[:, 1]
        for i, name in enumerate(model.parameter_names):
            policy = config.fit_parameters[name]
            base = centers[:, 3 - i] if name in {"cx", "cy", "cz"} else 0
            if policy.lower is not None:
                lower[:, i] = np.maximum(lower[:, i], base + policy.lower)
            if policy.upper is not None:
                upper[:, i] = np.minimum(upper[:, i], base + policy.upper)
            if policy.initial is not None:
                initial[:, i] = base + policy.initial
            fixed[i] = policy.fixed
        if config.fit_model == "2d":
            initial[:, 3], initial[:, 5] = centers[:, 0], 1
            lower[:, 3] = upper[:, 3] = initial[:, 3]
            lower[:, 5] = upper[:, 5] = 1
            fixed[[3, 5]] = True
    else:
        initial = lower = upper = np.empty((0, model.nparams))
    result = fit_rois(
        values,
        coordinates,
        sizes,
        initial,
        model=model,
        use_gpu=True,
        bounds=(lower, upper),
        fixed_params=fixed,
        estimator=config.estimator,
        noise_model="poisson" if config.estimator == "MLE" else "estimated_gaussian",
        max_number_iterations=config.max_number_iterations,
        tolerance=config.tolerance,
        return_covariance=config.return_covariance,
        quality_flags={"truncated": _host_array(truncated)},
        covariance_batch_size=config.roi_batch_size,
    )
    result["filter_pass"] = fit_filter_mask(
        result["fit_params"], result["success"], result.get("covariance_valid"), config.fit_filters
    )
    return result
