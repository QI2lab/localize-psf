"""Production native fits and covariance for simulated noisy microscopy images."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import chi2

from localize_psf import fit_psf as psf
from localize_psf.batch_fitting import fit_rois

pytest_plugins = ["test_gpu_psf_derivatives"]
pytestmark = pytest.mark.integration


def controls(model, parameters):
    lower = np.full(model.nparams, -np.inf)
    upper = np.full(model.nparams, np.inf)
    lower[0], upper[0] = 1, 300
    lower[1:4], upper[1:4] = -0.5, 0.5
    background = model.parameter_names.index("bg")
    lower[4:background], upper[4:background] = 0.05, 1.5
    lower[background], upper[background] = 0.5, 20
    return lower, upper


@pytest.mark.gpu
@pytest.mark.parametrize("estimator", ["LSE", "MLE"])
def test_noisy_native_estimators_fit_simulated_images(microscopy_model, estimator):
    cp = pytest.importorskip("cupy")
    model, parameters, coords = microscopy_model
    mean = model.model(tuple(cp.asarray(c) for c in coords), cp.asarray(parameters))
    rng = cp.random.RandomState(14817)
    data = cp.asnumpy(
        mean + rng.normal(0, 3, mean.shape) if estimator == "LSE" else rng.poisson(mean)
    ).astype(float)
    lower, upper = controls(model, parameters)
    starts = parameters.copy()
    starts[1:4] += 0.025
    variance = np.full((1, data.size), 9.0) if estimator == "LSE" else None
    result = fit_rois(
        data[None],
        tuple(c[None] for c in coords),
        np.array([data.size]),
        starts[None],
        model=model,
        use_gpu=True,
        estimator=estimator,
        bounds=(lower, upper),
        weights=None if variance is None else 1 / variance,
        variances=variance,
        return_covariance=True,
        noise_model="calibrated_gaussian" if estimator == "LSE" else "poisson",
        max_number_iterations=300,
        tolerance=1e-6,
    )
    assert result["success"][0]
    fitted = result["fit_params"][0].astype(float)
    native_mean = model.model(coords, fitted)
    if estimator == "LSE":
        reference = model.fit(
            data,
            coords,
            starts,
            sd=np.full(data.size, 3),
            absolute_sigma=True,
            bounds=(lower, upper),
            max_nfev=300,
            ftol=1e-11,
            xtol=1e-11,
            gtol=1e-11,
        )
        assert reference["success"]
        reference_parameters = reference["fit_params"]
        native_objective = np.sum((native_mean - data) ** 2 / 9)
        reference_objective = np.sum((model.model(coords, reference_parameters) - data) ** 2 / 9)
        np.testing.assert_allclose(fitted[1:4], reference_parameters[1:4], atol=2e-4, rtol=0)
    else:
        # The input is a Poisson simulation; estimation uses the native production fitter.
        center_error = np.abs(fitted[1:4] - parameters[1:4])
        center_sigma = np.sqrt(np.diag(result["covariance"][0])[1:4])
        assert np.all(center_error <= 5 * center_sigma)
        assert np.isfinite(result["chi_sqrs"][0])
        jac = np.stack(model.test_jacobian(coords, fitted, dp=1e-5)[0], axis=-1)
        expected = np.linalg.inv(jac.T @ (jac / native_mean[:, None]))
        np.testing.assert_allclose(result["covariance"][0], expected, rtol=1e-4, atol=1e-8)
        assert result["covariance_method"] == "poisson_information"
    if estimator == "LSE":
        assert native_objective - reference_objective < 0.002


@pytest.mark.gpu
@pytest.mark.monte_carlo
@pytest.mark.parametrize("brightness,depth", [(40, -0.18), (40, 0.23), (110, -0.18), (110, 0.23)])
@pytest.mark.parametrize("noise_model", ["calibrated_gaussian", "poisson"])
def test_held_out_cuda_center_covariance_coverage(brightness, depth, noise_model):
    cp = pytest.importorskip("cupy")
    model = psf.gaussian3d_psf_model(minimum_sigmas=(0.04, 0.08))
    coords = cp.meshgrid(
        cp.arange(-5, 6) * 0.23, cp.arange(-5, 6) * 0.12, cp.arange(-5, 6) * 0.15, indexing="ij"
    )
    coords = tuple(c.ravel() for c in coords)
    parameters = cp.array([brightness, 0.073, -0.052, depth, 0.28, 0.5, 4.0])
    mean = model.model(coords, parameters)
    trials = 240
    rng = cp.random.RandomState(97261 + brightness + int((depth + 1) * 100))
    data = (
        mean[None] + rng.normal(0, 3, (trials, mean.size))
        if noise_model == "calibrated_gaussian"
        else rng.poisson(mean, (trials, mean.size))
    )
    truth = cp.asnumpy(parameters)
    starts = np.tile(truth, (trials, 1))
    starts[:, 1:4] += 0.015
    lower, upper = controls(model, truth)
    variance = np.full((trials, mean.size), 9) if noise_model == "calibrated_gaussian" else None
    result = fit_rois(
        cp.asnumpy(data),
        tuple(np.tile(cp.asnumpy(c), (trials, 1)) for c in coords),
        np.full(trials, mean.size),
        starts,
        model=model,
        use_gpu=True,
        estimator="LSE" if variance is not None else "MLE",
        bounds=(lower, upper),
        weights=None if variance is None else 1 / variance,
        variances=variance,
        noise_model=noise_model,
        return_covariance=True,
        max_number_iterations=200,
        tolerance=1e-6,
    )
    accepted = result["covariance_valid"]
    assert accepted.mean() >= 0.98
    residuals = cp.asarray(result["fit_params"][accepted][:, [3, 2, 1]]) - parameters[[3, 2, 1]]
    covariance = cp.asarray(result["center_covariance_zyx"][accepted])
    whitened_squared = cp.sum(
        residuals * cp.linalg.solve(covariance, residuals[..., None])[..., 0], axis=1
    )
    count = len(residuals)
    coverage = float(cp.mean(whitened_squared <= chi2.ppf(0.9, 3)))
    whitened_mean = float(cp.mean(whitened_squared))
    # Predeclared statistical tolerances, using independent held-out seeds.
    assert abs(coverage - 0.9) <= 3.5 * np.sqrt(0.9 * 0.1 / count)
    assert abs(whitened_mean - 3) <= 4.5 * np.sqrt(6 / count)
    standard_error = cp.sqrt(cp.mean(cp.diagonal(covariance, axis1=1, axis2=2), axis=0) / count)
    standardized_bias = cp.asnumpy(cp.mean(residuals, axis=0) / standard_error)
    assert np.all(np.abs(standardized_bias) <= 4.5)
    report = {
        "noise_model": noise_model,
        "brightness": brightness,
        "depth_um": depth,
        "spacing_zyx_um": [0.23, 0.12, 0.15],
        "trials": trials,
        "accepted": count,
        "coverage_90_percent": coverage,
        "mean_whitened_squared": whitened_mean,
        "bias_standard_errors_zyx": standardized_bias.tolist(),
        "backend": "cuda",
        "seed": 97261 + brightness + int((depth + 1) * 100),
    }
    directory = Path("build/covariance-validation")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{noise_model}-{brightness}-{depth}.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
