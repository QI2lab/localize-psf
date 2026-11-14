"""Bounded ROI orchestration around the existing CPU and QI2lab GPUfit solvers."""

from collections.abc import Sequence
from time import perf_counter

import numpy as np

from localize_psf import fit_psf as psf
from localize_psf.array_backend import get_array_module
from localize_psf.fitting_controls import validate_bounds, validate_fixed
from localize_psf.uncertainty import add_covariance, parameter_bound_activity, validate_uncertainty


def _host_array(value, dtype=None):
    backend = get_array_module(value)
    return np.asarray(backend.asnumpy(value) if backend is not np else value, dtype=dtype)


def _slice_fit_controls(bounds, quality_flags, selected, nfits, nparams):
    """Slice candidate controls without interpreting their numerical semantics."""
    selected_bounds = None
    if bounds is not None:
        if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
            raise ValueError("bounds must be a numeric (lower, upper) pair")
        sides = []
        for side in bounds:
            values = _host_array(side)
            if values.shape == (nparams,):
                sides.append(values)
            elif values.shape == (nfits, nparams):
                sides.append(values[selected])
            else:
                raise ValueError(
                    "each numeric bound side must have shape (nparams,) or (nfits,nparams)"
                )
        selected_bounds = tuple(sides)
    selected_flags = None
    if quality_flags is not None:
        if not isinstance(quality_flags, dict):
            raise ValueError("quality_flags must be a dictionary")
        selected_flags = {}
        for name, value in quality_flags.items():
            values = _host_array(value)
            if values.shape != (nfits,) or values.dtype.kind != "b":
                raise ValueError("each quality flag must be a boolean array shaped (nfits,)")
            selected_flags[name] = values[selected]
    return selected_bounds, selected_flags


def _accumulate_fit_batch(complete, batch, start, nfits):
    """Persist compact fit evidence; never retain packed samples across batches."""
    selected = slice(start, start + len(batch["fit_params"]))
    if complete is None:
        complete = {}
        for name, value in batch.items():
            if isinstance(value, np.ndarray):
                complete[name] = np.empty((nfits, *value.shape[1:]), dtype=value.dtype)
            elif name == "bounds":
                complete[name] = tuple(
                    np.empty((nfits, *v.shape[1:]), dtype=v.dtype) for v in value
                )
            elif name == "covariance_flags":
                complete[name] = {key: np.empty(nfits, dtype=v.dtype) for key, v in value.items()}
            else:
                complete[name] = 0.0 if name == "fit_time" else value
    for name, value in batch.items():
        if isinstance(value, np.ndarray):
            complete[name][selected] = value + start if name == "candidate_ids" else value
        elif name == "bounds":
            for target, side in zip(complete[name], value):
                target[selected] = side
        elif name == "covariance_flags":
            for key, flag in value.items():
                complete[name][key][selected] = flag
        elif name == "fit_time":
            complete[name] += value
        elif name == "covariance_method" and value != complete[name]:
            # Inverse-variance batches are also valid sandwich estimates. Mixed
            # objective weights require the general convention for the whole run.
            methods = {value, complete[name]}
            if methods <= {"calibrated_gaussian_information", "calibrated_gaussian_sandwich"}:
                complete[name] = "calibrated_gaussian_sandwich"
            else:
                raise RuntimeError("inconsistent covariance convention between ROI batches")
        elif value != complete[name]:
            raise RuntimeError(f"inconsistent {name} between ROI batches")
    return complete


def fit_rois(
    img_rois: np.ndarray,
    coords_rois: Sequence[np.ndarray],
    roi_sizes: np.ndarray,
    init_params: np.ndarray,
    max_number_iterations: int = 100,
    tolerance: float | None = None,
    estimator: str = "LSE",
    fixed_params: np.ndarray | None = None,
    guess_bounds: bool = False,
    use_gpu: bool = False,
    debug: bool = False,
    verbose: bool = False,
    model: psf.pixelated_psf_model | None = None,
    bounds: tuple | None = None,
    weights: np.ndarray | None = None,
    return_covariance: bool = False,
    noise_model: str = "estimated_gaussian",
    variances: np.ndarray | None = None,
    quality_flags: dict | None = None,
    covariance_batch_size: int = 32,
    max_covariance_condition: float = 1e12,
) -> dict:
    """Fit valid ROI samples with explicit bounds and inverse-variance LSE weights.

    Parameters
    ----------
    img_rois : ndarray
        Host data shaped (nfits,npoints), with valid samples first and arbitrary padding.
    coords_rois : sequence of ndarray
        Physical ZYX coordinates with the same shape as img_rois.
    roi_sizes : ndarray
        Positive valid sample counts; padding is excluded from guessing and fitting.
    init_params : ndarray
        Initial parameters shaped (nfits,model.nparams), in model parameter order.
    bounds : tuple or None
        Numeric lower/upper sides shaped (nparams,) or (nfits,nparams) independently.
        Explicit bounds override guess_bounds. Equal bounds require fixed parameters.
    weights : ndarray or None
        Nonnegative LSE objective multipliers shaped like data; zero excludes a sample.
    return_covariance : bool
        Request full parameter and ZYX center covariance and validity diagnostics.
    noise_model : {'estimated_gaussian', 'calibrated_gaussian', 'poisson'}
        Explicit independent-noise convention. Estimated Gaussian scales by residual
        variance; calibrated Gaussian uses known variances; Poisson requires MLE.
    variances : ndarray or None
        Known positive per-sample variances for calibrated_gaussian, separate from weights.
    quality_flags : dict or None
        Boolean (nfits,) arrays for saturated, truncated, unresolved_overlap,
        and model_mismatch. Each invalidates ordinary center covariance.
    covariance_batch_size : int
        Maximum ROIs per uncertainty batch. CUDA derivatives and solves use float64.
    max_covariance_condition : float
        Largest accepted undamped free-parameter information condition number.
    estimator : {'LSE', 'MLE'}
        MLE uses GPUfit's pure-Poisson objective and requires nonnegative counts.
        CPU reference fitting supports LSE only.
    use_gpu : bool
        Explicit GPU request. Missing/incompatible CUDA or fork fails clearly.
    guess_bounds : bool
        Reuse model.estimate_bounds on valid coordinates when bounds is None.
    fixed_params : ndarray or None
        Shared binary fixed mask of shape (nparams,).
    max_number_iterations : int
        Native iteration limit or CPU maximum function evaluations.
    tolerance : float or None
        Native objective tolerance or CPU ftol/xtol/gtol; defaults are backend defaults.
    model : pixelated_psf_model or None
        Existing PSF implementation; None selects Gaussian3D.
    debug, verbose : bool
        Retained calling controls. CPU execution is deterministic and serial.

    Returns
    -------
    dict
        Fit parameters, initial parameters, states, success, candidate IDs, objective,
        bounds, iterations and timing. Empty batches retain all array shapes.
        Parameters retain optimizer conventions so bounds/fixed values are preserved.
    """
    model = psf.gaussian3d_psf_model() if model is None else model
    # GPUfit's public wrapper requires host buffers. Transfer packed ROIs here,
    # after GPU detection, packing and parameter estimation, not whole images.
    data = _host_array(img_rois, dtype=float)
    coordinates = tuple(_host_array(c, dtype=float) for c in coords_rois)
    sizes = _host_array(roi_sizes)
    initial = _host_array(init_params, dtype=float)
    if variances is not None:
        variances = _host_array(variances, dtype=float)
    if data.ndim != 2 or len(coordinates) != 3 or any(c.shape != data.shape for c in coordinates):
        raise ValueError("data and three ZYX coordinate arrays must share shape (nfits,npoints)")
    nfits, npoints = data.shape
    if (
        sizes.shape != (nfits,)
        or sizes.dtype.kind not in "iu"
        or np.any((sizes <= 0) | (sizes > npoints))
    ):
        raise ValueError("roi_sizes must contain positive integer valid sample counts")
    if initial.shape != (nfits, model.nparams):
        raise ValueError("init_params must have shape (nfits,model.nparams)")
    mask = validate_fixed(fixed_params, model.nparams)
    valid = np.arange(npoints)[None, :] < sizes[:, None]
    if any(not np.all(np.isfinite(a[valid])) for a in (data, *coordinates)):
        raise ValueError("valid ROI data and coordinates must be finite")
    if estimator not in ("LSE", "MLE"):
        raise ValueError("estimator must be LSE or MLE")
    if estimator == "MLE" and (np.any(data[valid] < 0) or weights is not None):
        raise ValueError("Poisson MLE requires nonnegative counts and no LSE weights")
    if not isinstance(max_number_iterations, (int, np.integer)) or max_number_iterations <= 0:
        raise ValueError("max_number_iterations must be a positive integer")
    if tolerance is not None and (not np.isfinite(tolerance) or tolerance <= 0):
        raise ValueError("tolerance must be finite and positive")
    objective_weights = None
    if weights is not None:
        objective_weights = _host_array(weights, dtype=float)
        if (
            objective_weights.shape != data.shape
            or not np.all(np.isfinite(objective_weights[valid]))
            or np.any(objective_weights[valid] < 0)
        ):
            raise ValueError("weights must match data and be finite/nonnegative on valid samples")
        if np.any(np.sum(np.where(valid, objective_weights, 0), axis=1) <= 0):
            raise ValueError("each ROI needs positive total fitting weight")
    if bounds is None and guess_bounds and nfits:
        guessed = [
            model.estimate_bounds(tuple(c[i, : sizes[i]] for c in coordinates))
            for i in range(nfits)
        ]
        bounds = (np.array([b[0] for b in guessed]), np.array([b[1] for b in guessed]))
    lower, upper, initial = validate_bounds(
        bounds, initial, mask, np.float32 if use_gpu else np.float64
    )
    result = {
        "fit_params": initial.copy(),
        "init_params": initial.copy(),
        "fit_states": np.zeros(nfits, np.int32),
        "fit_states_key": {},
        "success": np.zeros(nfits, bool),
        "candidate_ids": np.arange(nfits),
        "chi_sqrs": np.zeros(nfits),
        "niters": np.zeros(nfits, np.int32),
        "fit_time": 0.0,
        "model": repr(model),
        "bounds": (lower, upper),
        "active_bounds": np.zeros((nfits, model.nparams), bool),
    }
    variance, flags = (None, None)
    if return_covariance:
        variance, flags = validate_uncertainty(
            data,
            valid,
            estimator,
            noise_model,
            variances,
            quality_flags,
            covariance_batch_size,
            max_covariance_condition,
        )
    cpu_covariance = None
    if not nfits:
        if return_covariance:
            add_covariance(
                result,
                data,
                coordinates,
                sizes,
                mask,
                objective_weights,
                model,
                noise_model,
                variance,
                flags,
                covariance_batch_size,
                max_covariance_condition,
                use_gpu,
            )
        return result
    started = perf_counter()
    if not use_gpu:
        if estimator != "LSE":
            raise NotImplementedError("CPU reference supports LSE; request GPUfit for Poisson MLE")
        results = []
        for i in range(nfits):
            selected = valid[i].copy()
            if objective_weights is not None:
                selected &= objective_weights[i] > 0
            sd = None if objective_weights is None else 1 / np.sqrt(objective_weights[i, selected])
            options = (
                {} if tolerance is None else dict(ftol=tolerance, xtol=tolerance, gtol=tolerance)
            )
            results.append(
                model.fit(
                    data[i, selected],
                    tuple(c[i, selected] for c in coordinates),
                    initial[i],
                    fixed_params=mask,
                    bounds=(lower[i], upper[i]),
                    sd=sd,
                    absolute_sigma=noise_model == "calibrated_gaussian" and return_covariance,
                    max_nfev=max_number_iterations,
                    **options,
                )
            )
        for output, key in [
            ("fit_params", "fit_params"),
            ("fit_states", "status"),
            ("chi_sqrs", "chi_squared"),
            ("niters", "nfev"),
            ("success", "success"),
        ]:
            result[output] = np.asarray([r[key] for r in results])
        result["fit_states_key"] = results[0]["status_codes"]
        if return_covariance:
            cpu_covariance = np.array([r["covariance"] for r in results])
    else:
        _fit_gpu(
            result,
            data,
            coordinates,
            sizes,
            valid,
            initial,
            lower,
            upper,
            mask,
            objective_weights,
            model,
            estimator,
            tolerance,
            max_number_iterations,
        )
    if return_covariance:
        add_covariance(
            result,
            data,
            coordinates,
            sizes,
            mask,
            objective_weights,
            model,
            noise_model,
            variance,
            flags,
            covariance_batch_size,
            max_covariance_condition,
            use_gpu,
            cpu_covariance,
        )
    else:
        if use_gpu:
            import cupy as xp
        else:
            xp = np
        activity = parameter_bound_activity(result["fit_params"], result["bounds"], mask, xp)
        result["active_bounds"][:] = xp.asnumpy(activity) if use_gpu else activity
    result["fit_time"] = perf_counter() - started
    return result


def _gpufit_model_id(gf, model):
    """Resolve the named microscopy model only after validating the native contract."""
    mapping = {
        psf.gaussian3d_psf_model: "GAUSS_3D_ARB",
        psf.gaussian_lorentzian_psf_model: "GAUSS_LOR_3D_ARB",
        psf.gaussian3d_asymmetric_rotated_pixelated: "GAUSS_3D_ROT_ARB",
        psf.gaussian3d_asymmetric_pixelated: "GAUSS_3D_ASYM_ARB",
    }
    if type(model) not in mapping:
        raise NotImplementedError(f"No native microscopy model for {type(model)}")
    requirement = (
        "GPU fitting requires QI2lab GPUfit 1.2.0 with capability ABI 3 and float32 "
        "microscopy models; rebuild or reinstall the opm-ptv fork"
    )
    name = mapping[type(model)]
    try:
        capabilities = gf.get_capabilities()
        model_id = getattr(gf.ModelID, name)
    except (AttributeError, RuntimeError) as error:
        raise RuntimeError(requirement) from error
    if not isinstance(capabilities, dict):
        raise RuntimeError(requirement)
    models = capabilities.get("models")
    if (
        capabilities.get("abi_version") != 3
        or capabilities.get("precision") != "float32"
        or capabilities.get("constraints_layout") != "fit-major-interleaved-lower-upper"
        or not isinstance(models, dict)
        or models.get(name) != {"id": model_id, "parameters": model.nparams}
    ):
        raise RuntimeError(requirement)
    return model_id


def _fit_gpu(
    result,
    data,
    coordinates,
    sizes,
    valid,
    initial,
    lower,
    upper,
    fixed,
    weights,
    model,
    estimator,
    tolerance,
    iterations,
):
    try:
        import pygpufit.gpufit as gf
    except (AttributeError, ImportError, OSError) as error:
        raise RuntimeError(
            "GPU fitting requires the built QI2lab GPUfit 1.2.0 local profile (capability ABI 3)"
        ) from error
    model_id = _gpufit_model_id(gf, model)
    if not gf.cuda_available():
        raise RuntimeError(f"CUDA fitting unavailable: {gf.get_last_error()}")
    if model.sf != 1:
        raise NotImplementedError("GPUfit supports sampling factor 1")
    types = np.isfinite(lower).astype(np.int32) + 2 * np.isfinite(upper).astype(np.int32)
    # Microscopy padding evaluates to zero. Poisson gradients/Hessians divide by
    # the model value, so MLE groups must contain only valid samples.
    group_keys = np.column_stack((types, sizes)) if estimator == "MLE" else types
    patterns, grouping = np.unique(group_keys, axis=0, return_inverse=True)
    for group, pattern in enumerate(patterns):
        indices = np.flatnonzero(grouping == group)
        pattern = pattern[: model.nparams]
        npoints = int(sizes[indices[0]]) if estimator == "MLE" else data.shape[1]
        group_valid = valid[indices, :npoints]
        packed_data = np.ascontiguousarray(
            np.where(group_valid, data[indices, :npoints], 0), dtype=np.float32
        )
        packed_coords = [np.where(group_valid, c[indices, :npoints], 0) for c in coordinates]
        info = np.concatenate(
            [
                np.stack(packed_coords[::-1], axis=1).ravel(),
                sizes[indices],
                [] if model_id == gf.ModelID.GAUSS_LOR_3D_ARB else model.minimum_sigmas,
            ]
        ).astype(np.float32)
        packed_weights = (
            None
            if weights is None
            else np.ascontiguousarray(
                np.where(group_valid, weights[indices, :npoints], 0), dtype=np.float32
            )
        )
        if (
            not np.all(np.isfinite(packed_data))
            or not np.all(np.isfinite(info))
            or (packed_weights is not None and not np.all(np.isfinite(packed_weights)))
        ):
            raise ValueError("float32 conversion overflow in ROI data, coordinates or weights")
        controls = dict(
            tolerance=tolerance,
            max_number_iterations=iterations,
            estimator_id=gf.EstimatorID.LSE if estimator == "LSE" else gf.EstimatorID.MLE,
            parameters_to_fit=(~fixed).astype(np.int32),
            user_info=info,
        )
        if np.any(pattern):
            # Unused endpoints remain IEEE infinities in the localization contract;
            # the native ABI ignores those endpoints, which are packed as zero.
            packed = np.stack(
                [
                    np.where(np.isfinite(lower[indices]), lower[indices], 0),
                    np.where(np.isfinite(upper[indices]), upper[indices], 0),
                ],
                axis=-1,
            )
            output = gf.fit_constrained(
                packed_data,
                packed_weights,
                model_id,
                initial[indices],
                np.ascontiguousarray(packed.reshape(len(indices), -1)),
                pattern.astype(np.int32),
                **controls,
            )
        else:
            output = gf.fit(packed_data, packed_weights, model_id, initial[indices], **controls)
        for key, values in zip(("fit_params", "fit_states", "chi_sqrs", "niters"), output[:4]):
            result[key][indices] = values
    result["success"] = result["fit_states"] == 0
    result["fit_states_key"] = {
        "converged": 0,
        "max_iteration": 1,
        "singular_hessian": 2,
        "neg_curvature_mle": 3,
        "gpu_not_ready": 4,
    }
