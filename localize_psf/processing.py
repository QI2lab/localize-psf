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

from .config import FitFilters, LocalizationConfig, Parameter

if TYPE_CHECKING:
    from cupy import ndarray as CudaArray
    from tensorstore import TensorStore
    from ufish.api import UFish

    from localize_psf.fit_psf import gaussian3d_asymmetric_rotated_pixelated, gaussian3d_psf_model


# Preserve the caller's public type name for the production fitter's result dictionary.
type FitResults = dict[str, Any]


def fit_filter_mask(
    parameters: NDArray[np.float64],
    success: NDArray[np.bool_],
    covariance_valid: NDArray[np.bool_] | None,
    filters: FitFilters,
    *,
    collapse_distance_zyx_um: tuple[float, float, float] | None = None,
) -> NDArray[np.bool_]:
    """Return post-fit acceptance without removing candidates or changing evidence.

    ``parameters`` has shape (N,7) or (N,11) in the native Gaussian parameter order;
    column zero is amplitude in original image counts. ``success`` and optional
    ``covariance_valid`` are upstream boolean (N,) flags. All selected conditions
    must pass. Missing covariance raises when either covariance selector is used.
    Empty inputs return an empty mask. Optional ``collapse_distance_zyx_um``
    supplies the optical Z limit and radial XY limit to the existing
    ``filter_nearby_peaks`` keep-one policy, after amplitude/status filters.
    The Y and X entries are the same lateral limit. Converged fits have priority,
    followed by valid covariance, then amplitude; exact ties keep the original candidate.
    This function operates on one volume or 2D plane, never mixed T/C groups.
    Distances and fitted centers use µm. No GPU work or refitting is performed.
    """
    from localize_psf.localize import filter_nearby_peaks, range_filter

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
    if collapse_distance_zyx_um is not None:
        keep &= np.isfinite(parameters[:, (3, 2, 1)]).all(axis=1) & np.isfinite(parameters[:, 0])
        candidates = np.flatnonzero(keep)
        if len(candidates) > 1:
            covariance = (
                covariance_valid[candidates].astype(int)
                if covariance_valid is not None
                else np.zeros(len(candidates), dtype=int)
            )
            order = np.lexsort(
                (-parameters[candidates, 0], -covariance, -success[candidates].astype(int))
            )
            ranked = candidates[order]
            _, retained = filter_nearby_peaks(
                parameters[ranked][:, (3, 2, 1)],
                collapse_distance_zyx_um[1],
                collapse_distance_zyx_um[0],
            )
            keep[candidates] = False
            keep[ranked[retained]] = True
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


def data_footprint(image: np.ndarray) -> NDArray[np.bool_]:
    """Infer supported ZYX voxels from hole-filled nonzero axis projections.

    Reuse the detector's footprint convention, independently per T/C volume.
    Intersect YZ, ZX and YX footprints; a single-Z image uses YX only. Enclosed
    zero-count background is retained. This estimates support from image counts,
    rather than treating each zero-valued voxel as missing data.
    """
    from scipy.ndimage import binary_fill_holes

    support = image != 0
    footprint = np.ones(image.shape, dtype=bool)
    for pair in combinations(range(1 if image.shape[0] == 1 else 0, 3), 2):
        axis = 3 - sum(pair)
        projected = binary_fill_holes(np.any(support, axis=axis))
        footprint &= np.expand_dims(projected, axis)
    return footprint


class Detector:
    """Cache one production enhancement so threshold tuning only recalls spots."""

    def __init__(self) -> None:
        self._detector: UFish | None = None
        self._weights: str | None = None
        self._key: tuple[int, bool, bool, int, str, str] | None = None
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

        ``ufish_median_filter`` applies CuPy's size-3 GPU median filter with
        reflected borders before prediction: 3x3x3 in ZYX or 3x3 in YX for 2D.
        T/C selection happens before this interface; neither axis is filtered.
        The original image is never modified and remains the fitting input and
        edge-footprint reference. There is no CPU preprocessing fallback.

        ``exclude_edge_centroids`` removes a voxel-grid distance band at the
        array faces and the nonzero data footprint in each axis projection.
        This follows sloping zero-padded boundaries without treating interior
        zero counts as holes. 2D uses only YX. Enhancement and footprint distances
        are reused when changing the margin; the same priors drive batch fitting.

        ``collapse_close_priors`` prioritizes the strongest enhancement peak within
        one nominal optical Gaussian FWHM axially and radially in XY
        (XY only in 2D), after edge exclusion. Physical distances use stored
        voxel spacing. Suppression uses the existing nearby-peak keep-one policy.

        With ``cache=True``, retain the image identity and enhancement so threshold
        changes only recall spots. A new image, weights, dimensional mode, blend,
        median setting or batch size invalidates enhancement. ``cache=False`` streams batch
        volumes without retaining images; loaded weights are reused in both modes.
        """
        from ufish.api import UFish

        axes = "yx" if config.fit_model == "2d" else "zyx"
        if image.ndim != 3 or (axes == "yx" and image.shape[0] != 1):
            raise ValueError("Detection requires ZYX counts, with a single XY plane in 2D")
        key = (
            id(image),
            config.blend_3d,
            config.ufish_median_filter,
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
            inference_image = image[0] if axes == "yx" else image
            if config.ufish_median_filter:
                import cupy as cp
                from cupyx.scipy.ndimage import median_filter

                inference_image = cp.asnumpy(
                    median_filter(cp.asarray(inference_image), size=3, mode="reflect")
                )
            spots, enhanced = self._detector.predict(
                inference_image,
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
                footprint = data_footprint(image)
                distances = []
                for pair in pairs:
                    projected = binary_fill_holes(np.any(footprint, axis=3 - sum(pair)))
                    distances.append(
                        distance_transform_edt(np.pad(projected, 1))[1:-1, 1:-1].astype(np.float32)
                    )
                if cache:
                    self._edge_distances = distances
            keep = np.ones(len(points), dtype=bool)
            for pair, distance in zip(pairs, distances, strict=True):
                keep &= distance[tuple(points[:, pair].astype(np.intp).T)] > config.edge_margin_px
            points = points[keep]
        if config.collapse_close_priors and len(points) > 1:
            from localize_psf.localize import filter_nearby_peaks

            distances = optical_parameters(config)["collapse_distance_zyx_um"]
            pixels = points.astype(np.intp)[:, 1:] if axes == "yx" else points.astype(np.intp)
            strengths = enhanced[tuple(pixels.T)]
            order = np.argsort(-strengths, kind="stable")
            _, indices = filter_nearby_peaks(
                points[order] * config.voxel_size_zyx_um,
                distances[1],
                distances[0],
            )
            points = points[np.sort(order[indices])]
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
    """Return nominal optical FWHM and nominal/lower/upper sigmas as ZYX µm triples.

    Uses CUDA ``gaussian_width_bounds`` with the configuration's NA, wavelength,
    refractive index, voxel spacing and relative sigma range. Result keys are
    ``nominal_sigma_zyx_um``, ``lower_sigma_zyx_um``, ``upper_sigma_zyx_um`` and
    ``nominal_fwhm_zyx_um``. FWHM is 2*sqrt(2*ln(2)) times the nominal sigma;
    ``collapse_distance_zyx_um`` is one nominal FWHM for both priors and fits,
    interpreted as an axial Z limit and a radial XY limit by filter_nearby_peaks.
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
    result["nominal_fwhm_zyx_um"] = 2 * np.sqrt(2 * np.log(2)) * result["nominal_sigma_zyx_um"]
    result["collapse_distance_zyx_um"] = result["nominal_fwhm_zyx_um"]
    return {name: tuple(value.get().tolist()) for name, value in result.items()}


def gaussian_model(
    config: LocalizationConfig,
) -> gaussian3d_psf_model | gaussian3d_asymmetric_rotated_pixelated:
    """Select the existing native model; tilted fits use fixed XYZ Euler angles."""
    from localize_psf.fit_psf import (
        gaussian3d_asymmetric_rotated_pixelated,
        gaussian3d_psf_model,
    )

    return (
        gaussian3d_asymmetric_rotated_pixelated()
        if config.psf_model == "tilted_gaussian"
        else gaussian3d_psf_model()
    )


def calibrate_tilt(
    image: np.ndarray,
    points: NDArray[np.float64],
    config: LocalizationConfig,
    cancelled: Event | None = None,
    progress: Callable[[str], None] | None = None,
    *,
    origin_zyx_um: tuple[float, float, float] | None = None,
) -> float | None:
    """Estimate one YZ tilt from up to 64 isolated, fully observed 3D PSFs.

    Inputs use crop-local ZYX pixels, original counts and physical µm geometry.
    Reuse Gaussian localization, nearby-peak removal and experimental PSF
    averaging. Fit the normalized average by native GPUfit LSE with phi=pi/2,
    psi=-pi/2 and one free theta, trying 0 and +/-30 degrees. No likelihood or
    uncertainty for raw counts is assigned to the normalized average. Return
    radians, or None when fewer than three calibration spots or an identifiable
    average fit are available. A supplied calibration is reused by callers.
    """
    from localize_psf.batch_fitting import fit_rois
    from localize_psf.fit_psf import average_exp_psfs, gaussian3d_psf_model, get_psf_coords
    from localize_psf.localize import filter_nearby_peaks, get_coords, prepare_rois
    from localize_psf.rois import get_centered_rois, get_roi_size

    if len(points) < 3:
        return None
    reference = config.model_copy(
        update={
            "psf_model": "gaussian",
            "psf_tilt_yz_rad": None,
            "return_covariance": True,
            "fit_filters": FitFilters(
                amplitude_min=config.fit_filters.amplitude_min,
                amplitude_max=config.fit_filters.amplitude_max,
            ),
        }
    )
    fits = fit_spots(image, points, reference, cancelled, progress, origin_zyx_um=origin_zyx_um)
    parameters = fits["fit_params"]
    centers = parameters[:, (3, 2, 1)]
    _, isolated = filter_nearby_peaks(
        centers,
        max(config.fit_roi_size_um[1:]) / 2,
        config.fit_roi_size_um[0] / 2,
        mode="remove",
    )
    origin = config.origin_zyx_um if origin_zyx_um is None else origin_zyx_um
    spacing = config.voxel_size_zyx_um
    size = get_roi_size(config.fit_roi_size_um, spacing, ensure_odd=True)
    coords = get_coords(image.shape, spacing, origin=origin)
    rois = get_centered_rois(
        np.rint((centers - origin) / spacing).astype(int),
        size,
        (0, 0, 0),
        image.shape,
    )
    _, _, counts = prepare_rois(image, coords, rois, mask=data_footprint(image))
    keep = (
        fits["success"]
        & fits["filter_pass"]
        & (parameters[:, 0] > 0)
        & ~fits["active_bounds"][:, :4].any(axis=1)
        & ~fits["covariance_flags"]["invalid_information"]
        & (counts == np.prod(size))
    )
    ids = isolated[keep[isolated]]
    ids = ids[np.argsort(-parameters[ids, 0], kind="stable")[:64]]
    if len(ids) < 3:
        return None
    if cancelled is not None and cancelled.is_set():
        raise CancelledError("Tilt calibration cancelled")
    average = average_exp_psfs(image, coords, centers[ids], size, backgrounds=parameters[ids, 6])
    psf_coords = tuple(get_psf_coords(size, spacing))
    values, coordinates, counts = prepare_rois(
        average,
        psf_coords,
        np.array([[0, size[0], 0, size[1], 0, size[2]]]),
        mask=np.isfinite(average),
    )
    model = gaussian_model(config.model_copy(update={"psf_model": "tilted_gaussian"}))
    lo, hi = (np.asarray(side, dtype=float) for side in model.estimate_bounds(psf_coords))
    for index, name in ((4, "sxy"), (5, "sxy"), (6, "sz")):
        policy = config.fit_parameters[name]
        lo[index] = policy.lower
        hi[index] = np.inf if policy.upper is None else policy.upper
    lo[7], hi[7] = -np.inf, np.inf  # Background-subtracted PSF averages can be negative.
    lo[8:] = (np.pi / 2, -np.pi / 2, -np.pi / 2)
    hi[8:] = (np.pi / 2, np.pi / 2, -np.pi / 2)
    fixed = np.zeros(11, dtype=bool)
    fixed[[8, 10]] = True
    for index, name in ((4, "sxy"), (5, "sxy"), (6, "sz")):
        fixed[index] = config.fit_parameters[name].fixed
    lo[0] = 0
    seed = gaussian3d_psf_model().estimate_parameters(
        values[0, : counts[0]], tuple(c[0, : counts[0]] for c in coordinates)
    )
    starts = np.r_[seed[:4], seed[4], seed[4], seed[5:], np.zeros(3)]
    starts[1:4], starts[8], starts[10] = 0, np.pi / 2, -np.pi / 2
    for index, name in ((4, "sxy"), (5, "sxy"), (6, "sz")):
        if config.fit_parameters[name].initial is not None:
            starts[index] = config.fit_parameters[name].initial
    best = None
    for angle in (0.0, -np.pi / 6, np.pi / 6):
        if cancelled is not None and cancelled.is_set():
            raise CancelledError("Tilt calibration cancelled")
        starts[9] = angle
        fitted = fit_rois(
            values,
            coordinates,
            counts,
            starts[None],
            model=model,
            bounds=(lo, hi),
            fixed_params=fixed,
            estimator="LSE",
            use_gpu=True,
            return_covariance=True,
            max_number_iterations=config.max_number_iterations,
            tolerance=config.tolerance,
        )
        if fitted["success"][0] and fitted["covariance_valid"][0]:
            if best is None or fitted["chi_sqrs"][0] < best["chi_sqrs"][0]:
                best = fitted
    if best is None:
        return None
    angle = float(best["fit_params"][0, 9])
    if progress is not None:
        progress(f"Calibrated YZ PSF tilt {np.degrees(angle):.2f} degrees from {len(ids)} spots")
    return angle


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

    Fit only samples inside the shared data footprint, retaining supported zero
    counts. Array clipping or footprint masking sets the informational roi_truncated flag.

    Return all candidates in input order with IDs, statuses, active bounds and
    requested uncertainty evidence, plus the post-fit ``filter_pass`` mask.
    Final fitted positions are collapsed across all ROI batches when
    ``collapse_close_priors`` is enabled, using the same optical distance as
    prior suppression. ``collapse_distance_zyx_um`` records that distance for
    interactive re-filtering; null disables collapse. Raw fit rows are retained.
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
    from localize_psf.localize import get_coords, prepare_rois
    from localize_psf.rois import get_centered_rois, get_roi_size

    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Candidate coordinates must be finite (N,3) ZYX pixels")
    if np.any(points < 0) or np.any(points >= np.asarray(image.shape)):
        raise ValueError("Candidate coordinates must lie inside the preview")
    if len(points) and config.psf_model == "tilted_gaussian" and config.psf_tilt_yz_rad is None:
        angle = calibrate_tilt(
            image, points, config, cancelled, progress, origin_zyx_um=origin_zyx_um
        )
        if angle is None:
            raise ValueError(
                "Tilt calibration needs at least three isolated, fully supported 3D spots with a regular PSF fit"
            )
        config = config.model_copy(update={"psf_tilt_yz_rad": angle})
    model = gaussian_model(config)
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
    footprint = cp.asarray(data_footprint(image))
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
            values, coordinates, sizes = prepare_rois(image, coords, rois[selected], mask=footprint)
            truncated = sizes != np.prod(roi_size)
            batch = fit_packed_rois(
                values, coordinates, sizes, centers[selected], config, truncated, model=model
            )
            complete = _accumulate_fit_batch(complete, batch, start, len(points))
            del values, coordinates, batch
    finally:
        del image, coords, footprint
        cp.get_default_memory_pool().free_all_blocks()
    assert complete is not None
    distances = (
        optical_parameters(config)["collapse_distance_zyx_um"]
        if config.collapse_close_priors
        else None
    )
    complete["collapse_distance_zyx_um"] = distances
    complete["psf_tilt_yz_rad"] = config.psf_tilt_yz_rad
    complete["filter_pass"] = fit_filter_mask(
        complete["fit_params"],
        complete["success"],
        complete.get("covariance_valid"),
        config.fit_filters,
        collapse_distance_zyx_um=distances,
    )
    return complete


def fit_packed_rois(
    values: np.ndarray | CudaArray,
    coordinates: tuple[np.ndarray | CudaArray, ...],
    sizes: np.ndarray | CudaArray,
    centers_zyx_um: np.ndarray | CudaArray,
    config: LocalizationConfig,
    truncated: np.ndarray | CudaArray,
    *,
    model: gaussian3d_psf_model | gaussian3d_asymmetric_rotated_pixelated | None = None,
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
    Packed batches apply amplitude/status filters; the preview and workflow
    apply final-center collapse after collecting every ROI batch for a volume.

    Estimates use the input backend. Bounds/policies reuse the fitter's required
    host conversion, avoiding per-ROI CUDA reductions and extra buffer transfers.
    """
    from localize_psf.batch_fitting import _host_array, fit_rois
    from localize_psf.fit_psf import gaussian3d_psf_model

    model = gaussian_model(config) if model is None else model
    centers = _host_array(centers_zyx_um)
    fixed = np.zeros(model.nparams, dtype=bool)
    if len(values):
        initial = _host_array(
            gaussian3d_psf_model().estimate_parameters(values, coordinates, num_preserved_dims=1),
            dtype=float,
        )
        if config.psf_model == "tilted_gaussian":
            initial = np.column_stack(
                (
                    initial[:, :4],
                    initial[:, 4],
                    initial[:, 4],
                    initial[:, 5:],
                    np.zeros((len(values), 3)),
                )
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
            if name in {"phi", "theta", "psi"}:
                if config.psf_tilt_yz_rad is None:
                    raise ValueError("Calibrate the tilted Gaussian before fitting packed ROIs")
                angle = {"phi": np.pi / 2, "theta": config.psf_tilt_yz_rad, "psi": -np.pi / 2}[name]
                policy = Parameter(lower=angle, upper=angle, initial=angle, fixed=True)
            else:
                policy = config.fit_parameters[
                    {"A": "amp", "sx": "sxy", "sy": "sxy"}.get(name, name)
                ]
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
        covariance_batch_size=config.roi_batch_size,
    )
    # Missing samples are already excluded by sizes. Coverage is separate from
    # the numerical covariance validity of this correctly masked likelihood.
    result["roi_truncated"] = _host_array(truncated)
    result["parameter_names"] = tuple(model.parameter_names)
    result["filter_pass"] = fit_filter_mask(
        result["fit_params"], result["success"], result.get("covariance_valid"), config.fit_filters
    )
    return result
