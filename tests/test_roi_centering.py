"""Unit tests of the production ROI centering and deprecated forwarding path."""

import numpy as np
import pytest

from localize_psf.rois import get_centered_roi, get_centered_rois

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("backend", ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
def test_centered_rois_preserve_half_pixel_rounding_and_edge_clipping(backend):
    xp = np if backend == "cpu" else pytest.importorskip("cupy")
    centers = xp.asarray([[0.5, 1.5], [2.5, 3.5], [-0.5, 7.5]])
    sizes = xp.asarray([[3, 4], [3, 4], [5, 3]])
    result = get_centered_rois(centers, sizes, [0, 0], [8, 8])
    actual = result if xp is np else xp.asnumpy(result)
    np.testing.assert_array_equal(actual, [[0, 3, 0, 4], [2, 5, 2, 6], [0, 3, 6, 8]])


@pytest.mark.parametrize("backend", ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("dtype", ["float32", "int32"])
def test_centered_rois_without_limits_preserve_numeric_input_support(backend, dtype):
    xp = np if backend == "cpu" else pytest.importorskip("cupy")
    centers = xp.asarray([2, 4], dtype=dtype)
    result = get_centered_rois(centers, [3, 4])
    actual = result if xp is np else xp.asnumpy(result)
    np.testing.assert_array_equal(actual, [[1, 4, 3, 7]])
    with pytest.warns(UserWarning, match="deprecated"):
        assert get_centered_roi(centers, [3, 4]) == [1, 4, 3, 7]


@pytest.mark.parametrize(
    "center,size,expected",
    [
        ([0.5, 1.5], [3, 4], [0, 3, 0, 4]),
        ([2.5, 3.5], [3, 4], [2, 5, 2, 6]),
        ([-0.5, 7.5], [5, 3], [0, 3, 6, 8]),
    ],
)
def test_deprecated_single_roi_uses_the_production_centering_policy(center, size, expected):
    with pytest.warns(UserWarning, match="deprecated"):
        assert get_centered_roi(center, size, [0, 0], [8, 8]) == expected
