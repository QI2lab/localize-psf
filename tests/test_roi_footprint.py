"""Unit and integration tests of supported-voxel ROI packing."""

import numpy as np
import pytest

from localize_psf.localize import get_coords, prepare_rois
from localize_psf.processing import data_footprint


@pytest.mark.cpu
@pytest.mark.unit
@pytest.mark.parametrize("nz", [1, 7])
def test_projected_footprint_retains_enclosed_zero_counts(nz):
    z, y, x = np.indices((nz, 9, 11))
    support = (y >= z + 1) & (x >= 2)
    image = np.where(support, 5.0, 0.0)
    image[nz // 2, 6, 6] = 0
    np.testing.assert_array_equal(data_footprint(image), support)


@pytest.mark.cpu
@pytest.mark.integration
@pytest.mark.parametrize("backend", ["numpy", "tensorstore"])
def test_masked_rois_keep_sample_coordinate_alignment(backend):
    image = np.arange(7 * 9 * 11, dtype=float).reshape(7, 9, 11)
    z, y, _ = np.indices(image.shape)
    mask = y >= z
    image[3, 5, 6] = 0  # Real zero inside support must remain a fitted sample.
    coords = get_coords(image.shape, (0.23, 0.12, 0.15), origin=(-1.2, 3.4, -0.8))
    rois = np.array([[1, 6, 1, 8, 2, 9], [2, 5, 4, 7, 5, 8]])
    source = image if backend == "numpy" else pytest.importorskip("tensorstore").array(image)
    values, coordinates, counts = prepare_rois(source, coords, rois, mask=mask)
    for i, roi in enumerate(rois):
        selected = tuple(slice(start, stop) for start, stop in roi.reshape(3, 2))
        keep = mask[selected]
        assert counts[i] == keep.sum()
        np.testing.assert_array_equal(values[i, : counts[i]], image[selected][keep])
        assert np.isnan(values[i, counts[i] :]).all()
        assert np.any(values[i, : counts[i]] == 0)
        for actual, field in zip(coordinates, coords, strict=True):
            expected = np.broadcast_to(field, image.shape)[selected][keep]
            np.testing.assert_array_equal(actual[i, : counts[i]], expected)
            assert np.isnan(actual[i, counts[i] :]).all()


@pytest.mark.cpu
@pytest.mark.unit
def test_masked_roi_rejects_missing_support_and_invalid_masks():
    image = np.ones((3, 3, 3))
    coords = get_coords(image.shape, (1, 1, 1))
    rois = np.array([[0, 3, 0, 3, 0, 3]])
    for mask in (np.ones((1, 3, 3), bool), np.ones(image.shape)):
        with pytest.raises(ValueError, match="boolean and match"):
            prepare_rois(image, coords, rois, mask=mask)
    with pytest.raises(ValueError, match="supported samples"):
        prepare_rois(image, coords, rois, mask=np.zeros(image.shape, bool))
