"""Bounded high-level packing must preserve global fit and uncertainty evidence."""

import numpy as np
import pytest

from localize_psf.fit_psf import gaussian3d_psf_model
from localize_psf.localize import get_coords, localize_beads_generic

pytestmark = pytest.mark.integration


@pytest.mark.gpu
@pytest.mark.parametrize("batch_size", [1, 3])
def test_weighted_localization_batches_preserve_fits_and_covariance(batch_size):
    cp = pytest.importorskip("cupy")
    model = gaussian3d_psf_model()
    spacing = (0.3, 0.2, 0.2)
    origin = (-1.8, -1.6, -1.0)
    shape = (13, 17, 99)
    coords = get_coords(shape, spacing, origin=origin, use_gpu=True)
    image = cp.full(shape, 4.0)
    for index in range(7):
        z = -1.5 if index == 0 else 0.0  # Different valid sample counts/padding at the image edge.
        image += model.model(coords, cp.asarray([100, 1 + 2.4 * index, 0, z, 0.24, 0.42, 0]))
    variances = cp.broadcast_to(2 + 0.02 * cp.arange(shape[2])[None, None, :], shape)
    lower = np.tile([0, -2, -2, -2, 0.05, 0.1, 0], (7, 1))
    lower[:, 1] = np.arange(7) * 2.4
    upper = [200, 20, 2, 2, 0.8, 1.2, 20]
    mismatch = np.array([False, False, False, True, False, False, False])
    options = dict(
        roi_size=(2.1, 1.8, 1.8),
        min_spot_sep=(0.6, 0.6),
        origin=origin,
        model=model,
        bounds=(lower, upper),
        weights=1 / variances,
        variances=variances,
        return_covariance=True,
        noise_model="calibrated_gaussian",
        quality_flags={"model_mismatch": mismatch},
        use_gpu_filter=True,
        use_gpu_fit=True,
        covariance_batch_size=2,
        tolerance=1e-6,
        verbose=False,
    )
    _, reference, _ = localize_beads_generic(image, spacing, 40, **options)
    assert len(reference["fit_params"]) == 7
    _, result, _ = localize_beads_generic(image, spacing, 40, roi_batch_size=batch_size, **options)
    np.testing.assert_array_equal(result["candidate_ids"], np.arange(7))
    for name in ("fit_params", "init_params", "chi_sqrs", "covariance", "center_covariance_zyx"):
        np.testing.assert_allclose(
            result[name], reference[name], atol=1e-5, rtol=2e-4, equal_nan=True
        )
    for name in ("fit_states", "success", "covariance_valid", "active_bounds"):
        np.testing.assert_array_equal(result[name], reference[name])
    for actual, expected in zip(result["bounds"], reference["bounds"]):
        np.testing.assert_array_equal(actual, expected)
    for name, expected in reference["covariance_flags"].items():
        np.testing.assert_array_equal(result["covariance_flags"][name], expected)
    assert not result["covariance_valid"][3]
    assert np.all(result["covariance_valid"][[0, 1, 2, 4, 5, 6]])
    assert result["covariance_backend"] == "cuda"
    assert result["covariance_method"] == reference["covariance_method"]


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5, None])
def test_invalid_roi_batch_control_fails_even_without_candidates(batch_size):
    with pytest.raises(ValueError, match="roi_batch_size"):
        localize_beads_generic(
            np.zeros((5, 9, 9)),
            (0.3, 0.2, 0.2),
            100,
            min_spot_sep=(0.6, 0.6),
            roi_batch_size=batch_size,
            verbose=False,
        )
