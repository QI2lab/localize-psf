"""GPU localization must retain device arrays through packing and parameter guessing."""

import numpy as np
import pytest

from localize_psf.fit_psf import gaussian3d_psf_model
from localize_psf.localize import (
    filter_nearby_peaks,
    get_coords,
    localize_beads_generic,
    prepare_rois,
)


@pytest.mark.gpu
def test_existing_roi_packer_accepts_cuda_arrays_and_matches_cpu_reference():
    cp = pytest.importorskip("cupy")
    image = np.arange(7 * 8 * 9).reshape(7, 8, 9).astype(float)
    coords = get_coords(image.shape, (0.23, 0.12, 0.15), origin=(-1.2, 3.4, -0.8))
    rois = np.array([[0, 3, 1, 5, 2, 7], [2, 7, 0, 2, 1, 4]])
    reference = prepare_rois(image, coords, rois)
    actual = prepare_rois(cp.asarray(image), tuple(cp.asarray(c) for c in coords), cp.asarray(rois))
    assert isinstance(actual[0], cp.ndarray)
    assert all(isinstance(c, cp.ndarray) for c in actual[1])
    assert isinstance(actual[2], cp.ndarray)
    np.testing.assert_array_equal(cp.asnumpy(actual[0]), reference[0])
    for device, host in zip(actual[1], reference[1]):
        np.testing.assert_array_equal(cp.asnumpy(device), host)
    np.testing.assert_array_equal(cp.asnumpy(actual[2]), reference[2])


@pytest.mark.gpu
def test_high_level_parameter_guess_and_packing_execute_on_cuda(monkeypatch):
    cp = pytest.importorskip("cupy")
    model = gaussian3d_psf_model()
    coords = get_coords((13, 15, 15), (0.23, 0.12, 0.15), origin=(-1.38, -0.84, -1.05))
    truth = np.array([70, 0.1, -0.12, 0.16, 0.3, 0.55, 4])
    image = model.model(coords, truth)
    original = model.estimate_parameters
    calls = []

    def recorded(data, coordinates, **kwargs):
        calls.append(
            isinstance(data, cp.ndarray) and all(isinstance(c, cp.ndarray) for c in coordinates)
        )
        return original(data, coordinates, **kwargs)

    monkeypatch.setattr(model, "estimate_parameters", recorded)
    _, result, filtered = localize_beads_generic(
        image,
        (0.23, 0.12, 0.15),
        40,
        origin=(-1.38, -0.84, -1.05),
        roi_size=(2.3, 1.6, 1.8),
        min_spot_sep=(0.7, 0.7),
        model=model,
        use_gpu_fit=True,
        use_gpu_filter=True,
        verbose=False,
        return_filtered_images=True,
    )
    assert calls == [True]
    assert result["fit_params"].shape == (1, 7)
    np.testing.assert_allclose(result["fit_params"][0], truth, atol=0.005, rtol=0.005)
    assert isinstance(filtered, cp.ndarray)


@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["average", "keep-one", "remove"])
def test_existing_peak_duplicate_policy_runs_on_cuda(mode):
    cp = pytest.importorskip("cupy")
    centers = np.array([[0, 0, 0], [0.04, 0.04, 0.04], [0.8, 0.7, 0.9]])
    weights = np.array([2.0, 3.0, 4.0])
    reference = filter_nearby_peaks(centers, 0.15, 0.1, mode=mode, weights=weights)
    np.testing.assert_array_equal(weights, [2.0, 3.0, 4.0])
    actual = filter_nearby_peaks(
        cp.asarray(centers), 0.15, 0.1, mode=mode, weights=cp.asarray(weights)
    )
    assert all(isinstance(array, cp.ndarray) for array in actual)
    for device, host in zip(actual, reference):
        np.testing.assert_allclose(cp.asnumpy(device), host)
