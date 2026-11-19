"""GPU integration tests for the production microscopy models and padded ROIs."""

import numpy as np
import pytest

from localize_psf import batch_fitting

pytest_plugins = ["test_gpu_psf_derivatives"]
pytestmark = [pytest.mark.gpu, pytest.mark.integration]


@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("estimator", ["LSE", "MLE"])
def test_native_microscopy_models(microscopy_model, bounded, estimator):
    pytest.importorskip("pygpufit.gpufit")
    model, truth, coords = microscopy_model
    # Fit distinct physical coordinate grids and unequal valid lengths in one call.
    coordinates = tuple(np.stack([c, c + offset]) for c, offset in zip(coords, (0.1, -0.2, 0.3)))
    parameters = np.stack([truth, truth])
    parameters[1, 1:4] += [0.3, -0.2, 0.1]
    parameters[1, 0] += 10
    sizes = np.array([len(coords[0]), len(coords[0]) - 37])
    data = np.array(
        [model.model(tuple(c[i] for c in coordinates), p) for i, p in enumerate(parameters)]
    )
    data[1, sizes[1] :] = np.nan
    for c in coordinates:
        c[1, sizes[1] :] = np.nan
    starts = parameters.copy()
    starts[:, 1:4] += 0.02
    bounds = None
    if bounded:
        lower = np.full_like(parameters, -np.inf)
        upper = np.full_like(parameters, np.inf)
        lower[:, 0] = [1, 2]
        upper[:, 0] = [150, 160]
        lower[:, 4 : model.parameter_names.index("bg")] = 0.05
        bounds = lower, upper
    result = batch_fitting.fit_rois(
        data,
        coordinates,
        sizes,
        starts,
        model=model,
        use_gpu=True,
        bounds=bounds,
        estimator=estimator,
        tolerance=1e-7,
        max_number_iterations=200,
    )
    assert result["success"].all()
    np.testing.assert_allclose(result["fit_params"], parameters, atol=0.003, rtol=0.003)
    np.testing.assert_array_equal(result["candidate_ids"], [0, 1])
