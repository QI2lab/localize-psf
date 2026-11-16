"""Unit tests of shared peak suppression using simulated physical coordinates."""

import numpy as np
import pytest

from localize_psf.localize import filter_nearby_peaks

pytestmark = pytest.mark.unit


def test_axis_aligned_limits_strengths_and_inclusive_boundaries():
    centers = np.array(
        [
            [0, 0, 0],
            [0, 0.8, 0.4],
            [2.01, 0, 0],
            [0, 1.01, 0],
            [0, 0, 0.51],
            [0, 0, 0],
            [2, 1, 0.5],
        ]
    )
    strengths = np.array([10, 5, 4, 3, 2, 10, 1], dtype=np.uint16)
    original = centers.copy()
    retained, indices = filter_nearby_peaks(
        centers,
        0,
        0,
        weights=strengths,
        min_distance_zyx=(2, 1, 0.5),
        nmax=1,
    )
    np.testing.assert_array_equal(indices, [0, 2, 3, 4])
    np.testing.assert_array_equal(retained, original[indices])
    np.testing.assert_array_equal(centers, original)
    np.testing.assert_array_equal(strengths, [10, 5, 4, 3, 2, 10, 1])


def test_strongest_peak_wins_without_transitive_chaining():
    centers = np.array([[0, 0, 0], [0.75, 0, 0], [1.5, 0, 0]])
    _, strongest = filter_nearby_peaks(
        centers,
        0,
        0,
        weights=np.array([1, 3, 2]),
        min_distance_zyx=(1, 1, 1),
    )
    np.testing.assert_array_equal(strongest, [1])
    _, separate = filter_nearby_peaks(
        centers,
        0,
        0,
        weights=np.array([3, 2, 1]),
        min_distance_zyx=(1, 1, 1),
    )
    np.testing.assert_array_equal(separate, [0, 2])


def test_axis_aligned_mode_and_empty_input():
    centers = np.array([[0, 0, 0], [0.5, 0.8, 0.8]])
    assert len(filter_nearby_peaks(centers, 1, 1)[0]) == 2
    assert len(filter_nearby_peaks(centers, 0, 0, min_distance_zyx=(1, 1, 1))[0]) == 1
    empty, indices = filter_nearby_peaks(np.empty((0, 3)), 0, 0, min_distance_zyx=(1, 1, 1))
    assert empty.shape == (0, 3) and indices.shape == (0,) and indices.dtype.kind == "i"


@pytest.mark.parametrize("distances", [(1, 0, 1), (1, np.inf, 1), (1, 1)])
def test_invalid_axis_limits(distances):
    with pytest.raises(ValueError, match="positive distances"):
        filter_nearby_peaks(np.zeros((1, 3)), 0, 0, min_distance_zyx=distances)


def test_invalid_strengths_and_mode():
    for strengths in (np.array([np.nan]), np.ones(2)):
        with pytest.raises(ValueError, match="strengths"):
            filter_nearby_peaks(
                np.zeros((1, 3)), 0, 0, weights=strengths, min_distance_zyx=(1, 1, 1)
            )
    with pytest.raises(ValueError, match="keep-one"):
        filter_nearby_peaks(np.zeros((1, 3)), 0, 0, mode="average", min_distance_zyx=(1, 1, 1))


@pytest.mark.gpu
def test_axis_aligned_cuda_inputs_keep_original_peak_indices():
    cp = pytest.importorskip("cupy")
    centers = cp.asarray([[0, 0, 0], [0.5, 0.5, 0.5], [3, 0, 0]], dtype=float)
    retained, indices = filter_nearby_peaks(
        centers, 0, 0, weights=cp.asarray([2, 5, 1]), min_distance_zyx=(1, 1, 1)
    )
    assert isinstance(retained, cp.ndarray) and isinstance(indices, cp.ndarray)
    np.testing.assert_array_equal(indices.get(), [1, 2])
    np.testing.assert_array_equal(retained.get(), centers.get()[[1, 2]])
