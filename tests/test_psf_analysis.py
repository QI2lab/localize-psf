"""Unit and integration coverage of post-localization PSF analysis adapters."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from localize_psf.batch_fitting import fit_rois
from localize_psf.config import FitFilters, LocalizationConfig
from localize_psf.fit_psf import gaussian3d_psf_model
from localize_psf.localize import get_coords, prepare_rois
from localize_psf.processing import fit_filter_mask
from localize_psf.rois import get_centered_rois
from localize_psf.workflow import analyze_psf, plot_calibration

pytestmark = [pytest.mark.cpu, pytest.mark.example]


@pytest.fixture(scope="module")
def bead_fits():
    """Localize simulated beads with the real CPU fitter; no backend is replaced."""
    shape, spacing, origin = (25, 71, 71), (0.2, 0.1, 0.1), (-1.6, 3.3, -3.6)
    pixels = np.array(
        [[12.3, 15, 15], [13, 15, 55], [11.7, 55, 15], [12.6, 55, 55], [12.2, 35, 35]]
    )
    centers = origin + pixels * spacing
    parameters = np.array(
        [[a, *c[::-1], 0.15, 0.3, 5] for a, c in zip((100, 110, 120, 130, 400), centers)]
    )
    coords, model = get_coords(shape, spacing, origin=origin), gaussian3d_psf_model()
    image = 5 + sum(model.model(coords, p) - p[6] for p in parameters)
    rois = get_centered_rois(pixels, (9, 15, 15), min_vals=(0, 0, 0), max_vals=shape)
    values, coordinates, sizes = prepare_rois(image, coords, rois)
    fits = fit_rois(values, coordinates, sizes, parameters, model=model, use_gpu=False)
    fits["filter_pass"] = fit_filter_mask(
        fits["fit_params"], fits["success"], None, FitFilters(amplitude_max=200)
    )
    config = LocalizationConfig(voxel_size_zyx_um=spacing, origin_zyx_um=(-2, 3, -4))
    return image, fits, config, origin


@pytest.mark.integration
def test_analyze_existing_fits_reuses_filters_and_crop_coordinates(bead_fits):
    image, fits, config, origin = bead_fits
    original = image.copy()
    result = analyze_psf(image, fits, config, roi_size_zyx_px=(8, 14, 14), origin_zyx_um=origin)
    assert result["fit_results"] is fits
    assert result["metadata"]["origin_zyx_um"] == origin
    volume = result["volume_results"][0, 0]
    assert volume["average_psf"].shape == (9, 15, 15)
    assert volume["focus"]["available"]
    assert len(volume["focus"]["residual_after_plane_um"]) == 4
    for axis, coordinate in enumerate(volume["average_psf_coords_zyx_um"]):
        assert coordinate.ravel()[len(coordinate.ravel()) // 2] == 0
        np.testing.assert_allclose(np.diff(coordinate.ravel()), config.voxel_size_zyx_um[axis])
    np.testing.assert_array_equal(image, original)
    figures = plot_calibration(result)
    try:
        assert set(figures) == {"focus_gradient_and_field_curvature"}
        for figure in figures.values():
            figure.canvas.draw()
    finally:
        for figure in figures.values():
            plt.close(figure)


@pytest.mark.integration
def test_one_accepted_bead_still_produces_an_average(bead_fits):
    image, fits, config, origin = bead_fits
    selected = dict(fits, filter_pass=np.arange(len(fits["success"])) == 0)
    result = analyze_psf(image, selected, config, origin_zyx_um=origin)
    volume = result["volume_results"][0, 0]
    assert volume["average_psf"].shape == (11, 21, 21)
    assert not volume["focus"]["available"]
    assert "noncollinear" in volume["focus"]["reason"]
    assert not plot_calibration(result)


@pytest.mark.unit
def test_analysis_rejects_empty_selection_and_2d_inputs(bead_fits):
    image, fits, config, origin = bead_fits
    empty = dict(fits, filter_pass=np.zeros(len(fits["success"]), dtype=bool))
    with pytest.raises(ValueError, match="No successful fits"):
        analyze_psf(image, empty, config)
    for data, settings in (
        (image, config.model_copy(update={"fit_model": "2d"})),
        (image[:1], config),
    ):
        with pytest.raises(ValueError, match="3D fit volume"):
            analyze_psf(data, fits, settings, origin_zyx_um=origin)
