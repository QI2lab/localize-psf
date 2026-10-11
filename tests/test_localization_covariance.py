"""GPU localization uncertainty versus independent numerical reference information."""

import numpy as np
import pytest
from test_gpu_psf_derivatives import numerical_jacobian

from localize_psf import fit_psf as psf
from localize_psf.batch_fitting import fit_rois
from localize_psf.localize import localize_beads_generic

pytest_plugins = ["test_gpu_psf_derivatives"]


def packed(model, parameters, coords, count=1):
    truth = model.model(coords, parameters)
    return (
        np.tile(truth, (count, 1)),
        tuple(np.tile(c, (count, 1)) for c in coords),
        np.full(count, truth.size),
        np.tile(parameters, (count, 1)),
    )


@pytest.mark.gpu
def test_calibrated_cuda_covariance_full_nuisance_information(microscopy_model, monkeypatch):
    cp = pytest.importorskip("cupy")
    model, parameters, coords = microscopy_model
    data, coordinates, sizes, starts = packed(model, parameters, coords, count=3)
    variance = np.full_like(data, 9.0)
    calls = []
    original = model.jacobian

    def recorded(coordinates, parameters):
        calls.append(isinstance(coordinates[0], cp.ndarray))
        return original(coordinates, parameters)

    model.jacobian = recorded
    result = fit_rois(
        data,
        coordinates,
        sizes,
        starts,
        model=model,
        use_gpu=True,
        weights=1 / variance,
        variances=variance,
        noise_model="calibrated_gaussian",
        return_covariance=True,
        covariance_batch_size=2,
    )
    assert calls == [True, True]
    assert result["covariance_backend"] == "cuda"
    assert result["covariance_valid"].all()
    for index, fitted in enumerate(result["fit_params"]):
        jac = numerical_jacobian(model, coords, fitted)
        expected = np.linalg.inv(jac.T @ jac / 9)
        np.testing.assert_allclose(result["covariance"][index], expected, rtol=2e-4, atol=1e-9)
        np.testing.assert_allclose(
            result["center_covariance_zyx"][index],
            expected[np.ix_([3, 2, 1], [3, 2, 1])],
            rtol=2e-4,
            atol=1e-9,
        )
        assert np.all(np.diag(result["center_covariance_zyx"][index]) > 0)
    assert result["covariance_method"] == "calibrated_gaussian_information"


@pytest.mark.gpu
@pytest.mark.parametrize("weighted", [False, True])
def test_cuda_heteroscedastic_covariance_matches_estimator_sandwich(weighted):
    model = psf.gaussian3d_psf_model()
    parameters = np.array([70, 0.1, -0.12, 0.16, 0.3, 0.55, 4])
    coords = tuple(
        c.ravel()
        for c in np.meshgrid(
            np.arange(-3, 6) * 0.23, np.arange(-2, 7) * 0.12, np.arange(-4, 5) * 0.15, indexing="ij"
        )
    )
    data, coordinates, sizes, starts = packed(model, parameters, coords)
    variances = 3 + np.arange(data.size).reshape(data.shape) / data.size * 12
    weights = np.full_like(data, 0.4) if weighted else None
    result = fit_rois(
        data,
        coordinates,
        sizes,
        starts,
        model=model,
        use_gpu=True,
        weights=weights,
        variances=variances,
        noise_model="calibrated_gaussian",
        return_covariance=True,
    )
    jac = numerical_jacobian(model, coords, result["fit_params"][0])
    weight = np.ones(data.size) if weights is None else weights[0]
    inverse = np.linalg.inv(jac.T @ (weight[:, None] * jac))
    expected = inverse @ (jac.T @ ((weight**2 * variances[0])[:, None] * jac)) @ inverse
    np.testing.assert_allclose(result["covariance"][0], expected, rtol=1e-5, atol=1e-10)
    assert abs(expected[1, 2]) > 1e-9
    assert result["covariance_method"] == "calibrated_gaussian_sandwich"


@pytest.mark.gpu
def test_quality_flags_active_bounds_and_fixed_center_invalidate_covariance():
    model = psf.gaussian3d_psf_model()
    parameters = np.array([70, 0, 0, 0, 0.3, 0.55, 4])
    coords = tuple(
        c.ravel()
        for c in np.meshgrid(
            np.arange(-4, 5) * 0.23, np.arange(-4, 5) * 0.12, np.arange(-4, 5) * 0.15, indexing="ij"
        )
    )
    data, coordinates, sizes, starts = packed(model, parameters, coords, count=5)
    flags = {
        name: np.arange(5) == index
        for index, name in enumerate(
            ("saturated", "truncated", "unresolved_overlap", "model_mismatch")
        )
    }
    lower = np.full_like(starts, -np.inf)
    upper = np.full_like(starts, np.inf)
    lower[4, 1] = 0
    result = fit_rois(
        data,
        coordinates,
        sizes,
        starts,
        model=model,
        use_gpu=True,
        bounds=(lower, upper),
        return_covariance=True,
        noise_model="calibrated_gaussian",
        variances=np.ones_like(data),
        quality_flags=flags,
    )
    assert not result["covariance_valid"].any()
    assert np.isnan(result["center_covariance_zyx"]).all()
    assert result["active_bounds"][4, 1]
    fixed = np.array([False, True, False, False, False, False, False])
    fixed_result = fit_rois(
        data[:1],
        tuple(c[:1] for c in coordinates),
        sizes[:1],
        starts[:1],
        model=model,
        use_gpu=True,
        fixed_params=fixed,
        return_covariance=True,
        noise_model="calibrated_gaussian",
        variances=np.ones_like(data[:1]),
    )
    assert not fixed_result["covariance_valid"][0]
    assert fixed_result["covariance_flags"]["fixed_center"][0]


def test_cpu_reference_covariance_is_preserved_without_a_second_fit(monkeypatch):
    model = psf.gaussian3d_psf_model()
    coords = tuple(
        c.ravel()
        for c in np.meshgrid(
            np.arange(-4, 5) * 0.23, np.arange(-4, 5) * 0.12, np.arange(-4, 5) * 0.15, indexing="ij"
        )
    )
    parameters = np.array([70, 0.1, -0.12, 0.16, 0.3, 0.55, 4])
    data, coordinates, sizes, starts = packed(model, parameters, coords)
    data += np.random.default_rng(12).normal(0, 2, data.shape)
    original = model.fit
    computed = []

    def recorded(*args, **kwargs):
        result = original(*args, **kwargs)
        computed.append(result["covariance"].copy())
        return result

    monkeypatch.setattr(model, "fit", recorded)
    result = fit_rois(
        data, coordinates, sizes, starts, model=model, use_gpu=False, return_covariance=True
    )
    assert len(computed) == 1
    np.testing.assert_array_equal(result["covariance"][0], computed[0])


def test_known_cpu_reference_variance_does_not_scale_with_noisefree_residual():
    model = psf.gaussian3d_psf_model()
    coords = tuple(
        c.ravel()
        for c in np.meshgrid(
            np.arange(-4, 5) * 0.23, np.arange(-4, 5) * 0.12, np.arange(-4, 5) * 0.15, indexing="ij"
        )
    )
    parameters = np.array([70, 0.1, -0.12, 0.16, 0.3, 0.55, 4])
    result = model.fit(
        model.model(coords, parameters),
        coords,
        parameters,
        sd=np.full(coords[0].shape, 3),
        absolute_sigma=True,
    )
    jac = numerical_jacobian(model, coords, parameters)
    np.testing.assert_allclose(result["covariance"], np.linalg.inv(jac.T @ jac / 9), rtol=1e-5)


@pytest.mark.gpu
def test_high_level_cuda_localization_returns_covariance():
    model = psf.gaussian3d_psf_model()
    coords = np.meshgrid(
        np.arange(-6, 7) * 0.23, np.arange(-7, 8) * 0.12, np.arange(-7, 8) * 0.15, indexing="ij"
    )
    parameters = np.array([70, 0.1, -0.12, 0.16, 0.3, 0.55, 4])
    image = model.model(coords, parameters)
    _, result, _ = localize_beads_generic(
        image,
        (0.23, 0.12, 0.15),
        40,
        origin=(-6 * 0.23, -7 * 0.12, -7 * 0.15),
        roi_size=(2.3, 1.6, 1.8),
        min_spot_sep=(0.7, 0.7),
        use_gpu_fit=True,
        use_gpu_filter=True,
        variances=np.full(image.shape, 9),
        noise_model="calibrated_gaussian",
        return_covariance=True,
        verbose=False,
    )
    assert result["fit_params"].shape == (1, 7)
    assert result["covariance_valid"].all()
    assert result["covariance_backend"] == "cuda"


@pytest.mark.gpu
def test_empty_gpu_covariance_shapes_are_stable():
    result = fit_rois(
        np.empty((0, 0)),
        tuple(np.empty((0, 0)) for _ in range(3)),
        np.empty(0, dtype=int),
        np.empty((0, 7)),
        use_gpu=True,
        return_covariance=True,
    )
    assert result["covariance"].shape == (0, 7, 7)
    assert result["center_covariance_zyx"].shape == (0, 3, 3)
    assert result["covariance_valid"].shape == (0,)
