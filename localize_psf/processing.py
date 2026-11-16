"""Shared detector and parameter policy for interactive and batch localization."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import CancelledError
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING, NotRequired, TypedDict, cast

import numpy as np
from numpy.typing import NDArray

from .config import LocalizationConfig

if TYPE_CHECKING:
    from cupy import ndarray as CudaArray
    from tensorstore import TensorStore
    from ufish.api import UFish

    from localize_psf.fit_psf import gaussian3d_psf_model


class FitResults(TypedDict):
    """Native fitting evidence shared by preview and batch callers."""

    fit_params: NDArray[np.floating]
    init_params: NDArray[np.floating]
    success: NDArray[np.bool_]
    candidate_ids: NDArray[np.integer]
    niters: NDArray[np.integer]
    chi_sqrs: NDArray[np.floating]
    bounds: tuple[NDArray[np.floating], NDArray[np.floating]]
    active_bounds: NDArray[np.bool_]
    covariance_valid: NotRequired[NDArray[np.bool_]]
    covariance_backend: NotRequired[str]


@dataclass(frozen=True)
class Volume:
    """Original ZYX counts and physical calibration for a volume or XY plane.

    ``image`` has shape (Z,Y,X), without enhancement or rescaling. ``spacing``
    and ``origin`` are ZYX voxel sizes and first-voxel coordinates in µm.
    """

    image: np.ndarray
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]


def prepare_volume(data: np.ndarray | TensorStore, config: LocalizationConfig) -> Volume:
    """Read the configured preview crop as original float32 ZYX counts.

    ``data`` is a NumPy array or lazy TensorStore in ``config.preview.axes``
    order. Preview settings select T/C, crop and, for 2D, one XY plane.
    Only that selection is materialized; no projection is performed.

    Return a Volume with the exported ZYX spacing and the crop-first-voxel
    origin in µm, applying the crop offset once. Incompatible selections,
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
    return Volume(image, config.voxel_size_zyx_um, tuple(origin))


@dataclass(frozen=True)
class Detection:
    """U-FISH enhancement and crop-local ZYX pixel priors.

    ``enhanced`` is float32 ZYX for 3D or YX for 2D. ``points`` is a finite
    float64 (N,3) array; 2D Z coordinates are zero.
    """

    enhanced: NDArray[np.float32]
    points: NDArray[np.float64]


class Detector:
    """Cache one production enhancement so threshold tuning only recalls spots."""

    def __init__(self) -> None:
        self._detector: UFish | None = None
        self._weights: str | None = None
        self._key: tuple[int, bool, int, str, str] | None = None
        self._preview: Volume | None = None
        self._enhanced: NDArray[np.float32] | None = None

    def detect(
        self, preview: Volume, config: LocalizationConfig, *, cache: bool = True
    ) -> Detection:
        """Run CUDA U-FISH with the exported weights and inference settings.

        ``preview`` supplies original ZYX counts (singleton Z in 2D). ``config``
        supplies weights, dimensional mode, blending, threshold and batch size.
        Return the enhancement and crop-local ZYX pixel priors as a Detection.

        With ``cache=True``, retain the image identity and enhancement so threshold
        changes only recall spots. A new image, weights, dimensional mode, blend
        or batch size invalidates enhancement. ``cache=False`` streams batch
        volumes without retaining images; loaded weights are reused in both modes.
        """
        from ufish.api import UFish

        axes = "yx" if config.fit_model == "2d" else "zyx"
        key = (
            id(preview),
            config.blend_3d,
            config.inference_batch_size,
            config.ufish_weights,
            axes,
        )
        if not cache or self._key != key:
            self._enhanced = self._preview = self._key = None
            if self._detector is None or self._weights != config.ufish_weights:
                self._weights = None
                self._detector = UFish(device="cuda")
                weights = config.ufish_weights
                if Path(weights).is_file():
                    self._detector.load_weights(weights_path=weights)
                else:
                    self._detector.load_weights(weights_file=weights)
                self._weights = weights
            image = preview.image[0] if axes == "yx" else preview.image
            spots, enhanced = self._detector.predict(
                image,
                axes=axes,
                blend_3d=config.blend_3d,
                batch_size=config.inference_batch_size,
                intensity_threshold=config.intensity_threshold,
            )
            if cache:
                self._enhanced = enhanced
                self._preview, self._key = preview, key
        else:
            assert self._detector is not None and self._enhanced is not None
            enhanced = self._enhanced
            spots = self._detector.call_spots(
                self._enhanced, intensity_threshold=config.intensity_threshold
            )
        points = spots[[f"axis-{i}" for i in range(len(axes))]].to_numpy(dtype=float)
        if axes == "yx":
            points = np.column_stack((np.zeros(len(points)), points))
        return Detection(enhanced, points)

    def release(self) -> None:
        """Release the inference model before GPUfit; retain no stale cache key."""
        loaded = self._detector is not None
        self._detector = self._enhanced = self._key = self._preview = None
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
    preview: Volume,
    points: NDArray[np.float64],
    config: LocalizationConfig,
    cancelled: Event | None = None,
    progress: Callable[[str], None] | None = None,
) -> FitResults:
    """Fit crop-local ZYX pixel priors on a Volume's original counts with GPUfit.

    ``points`` is a finite (N,3) array inside ``preview.image``. ``config``
    supplies ROI dimensions, parameter policies, estimator, solver/batch controls
    and covariance selection. Center policies are µm offsets from each prior,
    intersected with ROI limits; infeasible starts raise ValueError without clipping.

    Return all candidates in input order with IDs, statuses, active bounds and
    requested uncertainty evidence. Fitted positions and widths use µm. 2D
    requires one XY plane and fixes the native 3D Gaussian's cz to that plane's
    physical Z and sz to 1 µm. CUDA workspaces are reused between fit batches.

    ``cancelled`` is an optional Event checked between batches; cancellation
    raises CancelledError. ``progress`` optionally receives each batch's status
    string. Invalid priors or geometry raise ValueError.
    """
    import cupy as cp

    from localize_psf.batch_fitting import _accumulate_fit_batch
    from localize_psf.fit_psf import gaussian3d_psf_model
    from localize_psf.localize import get_coords, prepare_rois
    from localize_psf.rois import get_centered_rois, get_roi_size

    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Candidate coordinates must be finite (N,3) ZYX pixels")
    if np.any(points < 0) or np.any(points >= np.asarray(preview.image.shape)):
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
    roi_size = np.asarray(get_roi_size(config.fit_roi_size_um, preview.spacing, ensure_odd=True))
    if config.fit_model == "2d":
        if preview.image.shape[0] != 1:
            raise ValueError("2D fitting requires a single XY plane")
        roi_size[0] = 1
    image = cp.asarray(preview.image)
    coords = get_coords(image.shape, preview.spacing, origin=preview.origin, use_gpu=True)
    centers = cp.asarray(preview.origin) + cp.asarray(points) * cp.asarray(preview.spacing)
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
    return cast(FitResults, complete)


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
    for every candidate, including statuses, active bounds and uncertainty flags.

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
    return cast(
        FitResults,
        fit_rois(
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
        ),
    )
