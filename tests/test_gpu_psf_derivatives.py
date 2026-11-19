"""Existing microscopy derivatives must execute on CUDA and match finite differences."""

import numpy as np
import pytest

from localize_psf import fit_psf as psf

pytestmark = pytest.mark.unit


@pytest.fixture(params=["gaussian", "lorentzian", "asymmetric", "rotated"])
def microscopy_model(request):
    models = {
        "gaussian": (
            psf.gaussian3d_psf_model(minimum_sigmas=(0.04, 0.08)),
            [80, 0.09, -0.06, 0.12, 0.28, 0.5, 4],
        ),
        "lorentzian": (psf.gaussian_lorentzian_psf_model(), [80, 0.09, -0.06, 0.12, 0.28, 0.5, 4]),
        "asymmetric": (
            psf.gaussian3d_asymmetric_pixelated(minimum_sigmas=(0.03, 0.04, 0.08)),
            [80, 0.09, -0.06, 0.12, 0.26, 0.35, 0.5, 4],
        ),
        "rotated": (
            psf.gaussian3d_asymmetric_rotated_pixelated(minimum_sigmas=(0.03, 0.04, 0.08)),
            [80, 0.09, -0.06, 0.12, 0.26, 0.35, 0.5, 4, 0.2, 0.4, -0.3],
        ),
    }
    model, parameters = models[request.param]
    coords = np.meshgrid(
        np.arange(-4, 5) * 0.23, np.arange(-4, 5) * 0.115, np.arange(-4, 5) * 0.14, indexing="ij"
    )
    return model, np.array(parameters), tuple(c.ravel() for c in coords)


@pytest.mark.gpu
def test_cuda_derivatives_match_independent_cpu_finite_differences(microscopy_model):
    cp = pytest.importorskip("cupy")
    model, parameters, coords = microscopy_model
    device_coords = tuple(cp.asarray(c) for c in coords)
    device_parameters = cp.asarray(parameters)
    derivatives = model.jacobian(device_coords, device_parameters)
    assert all(isinstance(d, cp.ndarray) for d in derivatives)
    actual = cp.asnumpy(cp.stack(derivatives, axis=-1))
    np.testing.assert_allclose(
        actual,
        np.stack(model.test_jacobian(coords, parameters, dp=1e-5)[0], axis=-1),
        rtol=2e-5,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        cp.asnumpy(model.model(device_coords, device_parameters)),
        model.model(coords, parameters),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.gpu
def test_cuda_model_derivatives_broadcast_over_roi_batch(microscopy_model):
    cp = pytest.importorskip("cupy")
    model, parameters, coords = microscopy_model
    batch = np.stack([parameters, parameters, parameters])
    batch[:, 1:4] += np.arange(3)[:, None] * 0.025
    device_parameters = cp.asarray(batch.T[:, :, None, None])
    device_coords = tuple(cp.broadcast_to(cp.asarray(c), (3, len(c))) for c in coords)
    derivatives = model.jacobian(device_coords, device_parameters)
    actual = cp.asnumpy(cp.stack(derivatives, axis=-1))
    expected = np.stack(
        [np.stack(model.test_jacobian(coords, p, dp=1e-5)[0], axis=-1) for p in batch]
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-7)
