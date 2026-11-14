"""GPU integration tests for the production microscopy models and padded ROIs."""

import os
import subprocess
import sys

import numpy as np
import pytest

from localize_psf import batch_fitting

pytest_plugins = ["test_gpu_psf_derivatives"]
pytestmark = [pytest.mark.gpu, pytest.mark.integration]


def test_cuda_component_wheels_without_sdk_environment():
    environment = os.environ.copy()
    environment.pop("CUDA_PATH", None)
    environment.pop("CUDA_HOME", None)
    code = """
import warnings
import localize_psf
import cupy as cp
import pygpufit.gpufit as gf

assert cp.arange(10).sum().item() == 45
assert cp.fft.fft(cp.ones(8))[0].item() == 8
assert cp.linalg.solve(cp.eye(2), cp.ones(2)).sum().item() == 2
assert gf.cuda_available()
try:
    warnings.warn_explicit('other warning', UserWarning, 'cuda', 1, module='cupy._environment')
except UserWarning:
    pass
else:
    raise AssertionError('unrelated warnings must remain visible')
"""
    subprocess.run(
        [sys.executable, "-I", "-W", "error::UserWarning", "-c", code],
        env=environment,
        check=True,
    )


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
