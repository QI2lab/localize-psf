"""Unit tests of fit acceptance with the existing axial/radial peak policy."""

import numpy as np
import pytest

from localize_psf.config import FitFilters
from localize_psf.processing import fit_filter_mask

pytestmark = pytest.mark.unit


def test_fit_separation_uses_radial_xy_and_inclusive_axial_limits():
    centers = np.array([[0, 0, 0], [0, 0.8, 0.8], [0, 0, 0.5], [1.01, 0, 0], [1, 0, 0], [0, 0, 1]])
    parameters = np.zeros((len(centers), 7))
    parameters[:, (3, 2, 1)] = centers
    parameters[:, 0] = [10, 6, 5, 4, 3, 2]
    success = np.ones(len(centers), dtype=bool)
    keep = fit_filter_mask(
        parameters, success, None, FitFilters(), collapse_distance_zyx_um=(1, 1, 1)
    )
    # The diagonal XY point is outside a radius of one; the exact limits are inclusive.
    np.testing.assert_array_equal(np.flatnonzero(keep), [0, 1, 3])
    empty = fit_filter_mask(
        parameters[:0], success[:0], None, FitFilters(), collapse_distance_zyx_um=(1, 1, 1)
    )
    assert empty.shape == (0,) and empty.dtype == np.bool_


@pytest.mark.parametrize(
    "filters, expected",
    [
        (FitFilters(), 2),
        (FitFilters(amplitude_max=79), 4),
        (FitFilters(covariance="fail"), 1),
        (FitFilters(convergence="fail"), 0),
    ],
)
def test_fit_separation_preserves_quality_priority_and_candidate_order(filters, expected):
    parameters = np.zeros((5, 7))
    parameters[:, 0] = [100, 90, 80, 80, 79]
    success = np.array([False, True, True, True, True])
    covariance = np.array([True, False, True, True, True])
    keep = fit_filter_mask(
        parameters, success, covariance, filters, collapse_distance_zyx_um=(1, 1, 1)
    )
    np.testing.assert_array_equal(np.flatnonzero(keep), [expected])
