"""Integration tests of production bounds, coordinates and localization."""

import importlib.util

import numpy as np
import pytest

from localize_psf.fit_psf import gaussian3d_psf_model
from localize_psf.localize import fit_rois, localize_beads_generic

pytestmark = pytest.mark.integration


@pytest.fixture(params=[False, pytest.param(True, marks=pytest.mark.gpu)])
def backend(request):
    if request.param and importlib.util.find_spec("pygpufit") is None:
        pytest.skip("local GPU profile required")
    return request.param


@pytest.fixture
def batch():
    model = gaussian3d_psf_model()
    coords = np.meshgrid(
        np.linspace(-3, 1, 9), np.linspace(2, 4, 11), np.linspace(-2, 0, 11), indexing="ij"
    )
    truth = np.array(
        [[100, -1.13, 3.17, -0.73, 0.35, 0.7, 5], [120, -0.87, 2.83, -1.24, 0.4, 0.8, 7]]
    )
    data = np.array([model.model(coords, p).ravel() for p in truth])
    coordinates = tuple(np.tile(c.ravel(), (2, 1)) for c in coords)
    starts = truth.copy()
    starts[:, 1:4] += 0.04
    return model, data, coordinates, np.full(2, data.shape[1]), starts, truth


@pytest.mark.cpu
@pytest.mark.parametrize("value", [-1, np.nan])
def test_poisson_invalid_counts_are_rejected_before_dispatch(batch, value):
    model, data, coords, sizes, starts, _ = batch
    data[0, 0] = value
    with pytest.raises(ValueError, match="finite|nonnegative"):
        fit_rois(data, coords, sizes, starts, model=model, use_gpu=True, estimator="MLE")


def test_independent_broadcast_and_explicit_overrides_guess(batch, backend):
    model, data, coords, sizes, start, truth = batch
    lower = np.array([0, -2, 2, -3, 0.05, 0.05, 0])
    upper = np.array([[200, 0, 4, 1, 2, 2, 20], [300, 0, 4, 1, 2, 2, 20]])
    result = fit_rois(
        data,
        coords,
        sizes,
        start,
        model=model,
        use_gpu=backend,
        bounds=(lower, upper),
        guess_bounds=True,
    )
    np.testing.assert_allclose(result["fit_params"], truth, atol=2e-3, rtol=2e-3)
    np.testing.assert_array_equal(result["candidate_ids"], [0, 1])
    np.testing.assert_allclose(result["bounds"][0], np.broadcast_to(lower, start.shape), rtol=3e-8)
    np.testing.assert_allclose(result["bounds"][1], upper, rtol=3e-8)


def test_active_center_bound_and_fixed_background(batch, backend):
    model, data, coords, sizes, start, truth = batch
    lower = np.full(start.shape, -np.inf)
    upper = np.full(start.shape, np.inf)
    lower[:, 4:6] = 0.05
    lower[:, 6] = truth[:, 6]
    upper[:, 6] = truth[:, 6]
    lower[:, 1] = start[:, 1]
    mask = np.array([False] * 6 + [True])
    result = fit_rois(
        data,
        coords,
        sizes,
        start,
        model=model,
        use_gpu=backend,
        bounds=(lower, upper),
        fixed_params=mask,
    )
    assert np.all(result["fit_params"][:, 1] >= lower[:, 1] - 2e-6)
    np.testing.assert_array_equal(result["fit_params"][:, 6], truth[:, 6])


@pytest.mark.parametrize(
    "invalid",
    [
        "none",
        "nan",
        "shape",
        "reversed",
        "equal",
        "infinity",
        "start",
        "fixed",
        "overflow",
        "collapse",
    ],
)
def test_bounds_rejected_before_optimizer(batch, invalid):
    model, data, coords, sizes, start, _ = batch
    lower = np.full(start.shape, -np.inf)
    upper = np.full(start.shape, np.inf)
    fixed = np.zeros(model.nparams, bool)
    gpu = False
    if invalid == "none":
        lower = lower.astype(object)
        lower[0, 0] = None
    elif invalid == "nan":
        lower[0, 0] = np.nan
    elif invalid == "shape":
        lower = np.ones((1, model.nparams))
    elif invalid == "reversed":
        lower[0, 0], upper[0, 0] = 10, 0
    elif invalid == "equal":
        lower[0, 0] = upper[0, 0] = start[0, 0]
    elif invalid == "infinity":
        lower[0, 0] = np.inf
    elif invalid == "start":
        start[0, 0] = np.inf
    elif invalid == "fixed":
        fixed[0] = True
        upper[0, 0] = 50
    elif invalid == "overflow":
        upper[0, 0] = 1e40
        gpu = True
    else:
        lower[0, 0], upper[0, 0] = 100, 100 + 1e-7
        gpu = True
    with pytest.raises(ValueError):
        fit_rois(
            data,
            coords,
            sizes,
            start,
            bounds=(lower, upper),
            model=model,
            use_gpu=gpu,
            fixed_params=fixed,
        )


def test_padding_excluded_from_guessed_bounds(batch, backend):
    model, data, coords, sizes, start, truth = batch
    n = data.shape[1]
    padded = np.pad(data, ((0, 0), (0, 5)), constant_values=np.nan)
    padded_coords = tuple(np.pad(c, ((0, 0), (0, 5)), constant_values=np.nan) for c in coords)
    result = fit_rois(
        padded, padded_coords, sizes, start, model=model, use_gpu=backend, guess_bounds=True
    )
    np.testing.assert_allclose(result["fit_params"], truth, atol=2e-3, rtol=2e-3)
    assert n == sizes[0]


def test_empty_batch_and_empty_high_level():
    result = fit_rois(
        np.empty((0, 4)),
        tuple(np.empty((0, 4)) for _ in range(3)),
        np.empty(0, int),
        np.empty((0, 7)),
        use_gpu=False,
    )
    assert result["fit_params"].shape == (0, 7)
    _, result, _ = localize_beads_generic(
        np.zeros((5, 9, 9)),
        (0.4, 0.2, 0.2),
        100,
        min_spot_sep=(0.5, 0.3),
        use_gpu_fit=False,
        use_gpu_filter=False,
        verbose=False,
    )
    assert result["fit_params"].shape == (0, 7)
    assert result["candidate_ids"].shape == (0,)


def test_high_level_nonzero_origin_and_physical_bounds(backend):
    from localize_psf.localize import get_coords

    model = gaussian3d_psf_model()
    spacing = (0.4, 0.2, 0.15)
    origin = (-8.0, 3.0, -2.0)
    shape = (13, 17, 19)
    coordinates = get_coords(shape, spacing, broadcast=True, origin=origin)
    truth = np.array([100, -0.61, 4.63, -5.57, 0.35, 0.7, 5])
    image = model.model(coordinates, truth)
    lower = [0, -2, 3, -8, 0.05, 0.05, 0]
    upper = [200, 1, 7, -2, 2, 2, 20]
    actual_coords, result, _ = localize_beads_generic(
        image,
        spacing,
        50,
        roi_size=(4.0, 2.4, 2.4),
        min_spot_sep=(0.5, 0.3),
        origin=origin,
        bounds=(lower, upper),
        model=model,
        weights=np.ones(shape),
        estimator="LSE",
        tolerance=1e-6,
        use_gpu_fit=backend,
        use_gpu_filter=False,
        verbose=False,
    )
    np.testing.assert_allclose(result["fit_params"], truth[None], atol=0.005, rtol=0.005)
    np.testing.assert_allclose([a.ravel()[0] for a in actual_coords], origin)


@pytest.mark.gpu
def test_grouped_native_fits_preserve_candidate_ids(batch):
    if importlib.util.find_spec("pygpufit") is None:
        pytest.skip("local GPU profile required")
    model, data, coords, sizes, starts, truth = batch
    lower = np.full(starts.shape, -np.inf)
    upper = np.full(starts.shape, np.inf)
    lower[0, 0] = 0
    weights = np.full_like(data, 0.5)
    result = fit_rois(
        data,
        coords,
        sizes,
        starts,
        bounds=(lower, upper),
        model=model,
        weights=weights,
        tolerance=1e-6,
        max_number_iterations=90,
        use_gpu=True,
    )
    np.testing.assert_array_equal(result["candidate_ids"], [0, 1])
    np.testing.assert_allclose(result["fit_params"], truth, atol=0.002, rtol=0.002)
