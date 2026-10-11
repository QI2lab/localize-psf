"""Optional undamped localization information using the existing PSF derivatives."""

import numpy as np

QUALITY_FLAGS = ("saturated", "truncated", "unresolved_overlap", "model_mismatch")


def validate_uncertainty(
    data, valid, estimator, noise_model, variances, quality_flags, batch_size, max_condition
):
    """Validate distinct noise, quality and memory controls before fitting."""
    if noise_model not in ("estimated_gaussian", "calibrated_gaussian", "poisson"):
        raise ValueError("noise_model must be estimated_gaussian, calibrated_gaussian or poisson")
    if (noise_model == "poisson") != (estimator == "MLE"):
        raise ValueError("Poisson uncertainty requires MLE; Gaussian uncertainty requires LSE")
    variance = None
    if noise_model == "calibrated_gaussian":
        variance = np.asarray(variances, dtype=float)
        if (
            variance.shape != data.shape
            or not np.all(np.isfinite(variance[valid]))
            or np.any(variance[valid] <= 0)
        ):
            raise ValueError(
                "calibrated variances must match data and be finite/positive on valid samples"
            )
    elif variances is not None:
        raise ValueError("variances are only accepted for calibrated_gaussian uncertainty")
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("covariance_batch_size must be a positive integer")
    if not np.isfinite(max_condition) or max_condition <= 1:
        raise ValueError("max_covariance_condition must be finite and greater than one")
    supplied = {} if quality_flags is None else quality_flags
    if not isinstance(supplied, dict) or set(supplied) - set(QUALITY_FLAGS):
        raise ValueError(f"quality_flags may only contain {QUALITY_FLAGS}")
    flags = {}
    for name in QUALITY_FLAGS:
        flag = np.asarray(supplied.get(name, np.zeros(len(data), dtype=bool)))
        if flag.shape != (len(data),) or flag.dtype.kind != "b":
            raise ValueError("each quality flag must be a boolean array shaped (nfits,)")
        flags[name] = flag.copy()
    return variance, flags


def add_covariance(
    result,
    data,
    coordinates,
    sizes,
    fixed,
    weights,
    model,
    noise_model,
    variances,
    flags,
    batch_size,
    max_condition,
    use_gpu,
    cpu_covariance=None,
):
    """Evaluate information in bounded CUDA batches; retain CPU reference covariance.

    All free nuisance parameters participate before the ZYX center block is
    extracted. Calibrated variances remain distinct from objective weights. For
    general LSE weights the sandwich describes the estimator actually optimized.
    Fixed nuisance parameters make the result conditional on their calibration.
    Invalid covariance never receives a zero or isotropic replacement.
    """
    nfits, npoints = data.shape
    nparams = model.nparams
    centers = [model.parameter_names.index(name) for name in ("cz", "cy", "cx")]
    free = np.flatnonzero(~fixed)
    if not free.size:
        raise ValueError("uncertainty requires at least one free parameter")
    calibrated_information = (
        noise_model == "calibrated_gaussian"
        and weights is not None
        and np.allclose(
            (weights * variances)[np.arange(npoints)[None, :] < sizes[:, None]],
            1,
            rtol=1e-8,
            atol=1e-10,
        )
    )
    method = (
        "poisson_information"
        if noise_model == "poisson"
        else "estimated_gaussian_information"
        if noise_model == "estimated_gaussian"
        else "calibrated_gaussian_information"
        if calibrated_information
        else "calibrated_gaussian_sandwich"
    )
    covariance = np.full((nfits, nparams, nparams), np.nan)
    center_covariance = np.full((nfits, 3, 3), np.nan)
    condition = np.full(nfits, np.inf)
    active = np.zeros((nfits, nparams), bool)
    flags = {
        **flags,
        "fit_failed": ~result["success"],
        "active_bound": np.zeros(nfits, bool),
        "fixed_center": np.full(nfits, np.any(fixed[centers])),
        "invalid_information": np.zeros(nfits, bool),
    }
    result.update(
        covariance=covariance,
        center_covariance_zyx=center_covariance,
        covariance_condition=condition,
        covariance_valid=np.zeros(nfits, bool),
        covariance_method=method,
        covariance_backend="cuda" if use_gpu else "cpu_reference",
        covariance_conditional=bool(np.any(fixed)),
        covariance_flags=flags,
        active_bounds=active,
    )
    if not nfits:
        return
    if use_gpu:
        import cupy as xp
    else:
        xp = np
    to_host = xp.asnumpy if use_gpu else np.asarray
    for start in range(0, nfits, batch_size):
        stop = min(start + batch_size, nfits)
        selected = slice(start, stop)
        valid = xp.arange(npoints)[None, :] < xp.asarray(sizes[selected])[:, None]
        coords = tuple(xp.where(valid, xp.asarray(c[selected]), 0) for c in coordinates)
        parameters = xp.asarray(result["fit_params"][selected])
        initial = xp.asarray(result["init_params"][selected])
        parameters = xp.where(xp.isfinite(parameters), parameters, initial)
        evaluated_parameters = parameters.T[:, :, None, None]
        jacobian = xp.stack(model.jacobian(coords, evaluated_parameters), axis=-1)
        jacobian = jacobian[..., free]
        predicted = model.model(coords, evaluated_parameters)
        observations = xp.where(valid, xp.asarray(data[selected]), 0)
        weight = xp.ones_like(observations) if weights is None else xp.asarray(weights[selected])
        weight = xp.where(valid, weight, 0)
        if noise_model == "poisson":
            invalid_mean = xp.any(valid & ((predicted <= 0) | ~xp.isfinite(predicted)), axis=1)
            weight = xp.where(valid, 1 / xp.where(predicted > 0, predicted, 1), 0)
        else:
            invalid_mean = xp.zeros(stop - start, dtype=bool)
        jacobian = xp.where(valid[..., None], jacobian, 0)
        information = jacobian.swapaxes(1, 2) @ (weight[..., None] * jacobian)
        finite = xp.all(xp.isfinite(information), axis=(1, 2)) & ~invalid_mean
        identity = xp.eye(len(free))
        # Placeholders keep a bad row from aborting the entire batched solver.
        # They never become returned covariance: every bad row remains flagged/NaN.
        eigenvalues = xp.linalg.eigvalsh(xp.where(finite[:, None, None], information, identity))
        positive = eigenvalues[:, 0] > 0
        cond = xp.where(
            positive & finite, eigenvalues[:, -1] / xp.where(positive, eigenvalues[:, 0], 1), xp.inf
        )
        solvable = finite & positive & (cond <= max_condition)
        inverse = xp.linalg.solve(
            xp.where(solvable[:, None, None], information, identity),
            xp.broadcast_to(identity, information.shape),
        )
        if noise_model == "calibrated_gaussian" and not calibrated_information:
            variance = xp.where(valid, xp.asarray(variances[selected]), 0)
            meat = jacobian.swapaxes(1, 2) @ ((weight**2 * variance)[..., None] * jacobian)
            free_covariance = inverse @ meat @ inverse
        elif noise_model == "estimated_gaussian":
            dof = xp.sum(valid & (weight > 0), axis=1) - len(free)
            scale = xp.sum(weight * (predicted - observations) ** 2, axis=1) / xp.where(
                dof > 0, dof, 1
            )
            solvable &= dof > 0
            free_covariance = inverse * scale[:, None, None]
        else:
            free_covariance = inverse
        free_covariance = xp.where(solvable[:, None, None], free_covariance, xp.nan)
        host_covariance = to_host(free_covariance)
        for i in range(stop - start):
            covariance[start + i][np.ix_(free, free)] = host_covariance[i]
        if cpu_covariance is not None and (
            noise_model == "estimated_gaussian" or calibrated_information
        ):
            covariance[selected] = cpu_covariance[selected]
        condition[selected] = to_host(cond)
        flags["invalid_information"][selected] = ~to_host(solvable)
        lower, upper = (xp.asarray(side[selected]) for side in result["bounds"])
        tolerance = (2e-5 if use_gpu else 1e-7) * (1 + xp.abs(parameters))
        active[selected] = to_host(
            (~xp.asarray(fixed))[None, :]
            & (
                (xp.isfinite(lower) & (xp.abs(parameters - lower) <= tolerance))
                | (xp.isfinite(upper) & (xp.abs(parameters - upper) <= tolerance))
            )
        )
    flags["active_bound"][:] = np.any(active, axis=1)
    center_covariance[:] = covariance[:, centers][:, :, centers]
    valid_covariance = ~np.logical_or.reduce(list(flags.values()))
    # Numerical checks are GPU operations for production; no repair is attempted.
    block = xp.asarray(center_covariance)
    finite = xp.all(xp.isfinite(block), axis=(1, 2))
    values = xp.linalg.eigvalsh(xp.where(finite[:, None, None], block, xp.eye(3)))
    valid_covariance &= to_host(finite & xp.all(values > 0, axis=1))
    result["covariance_valid"][:] = valid_covariance
    center_covariance[~valid_covariance] = np.nan
