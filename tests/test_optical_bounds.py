"""Optical bounds reuse the owner PSF convention and observed sampling."""

import numpy as np
import pytest


@pytest.mark.gpu
@pytest.mark.unit
@pytest.mark.parametrize("floor", [0, 0.5])
def test_cuda_optical_widths_and_anisotropic_sampling_reference(floor):
    cp = pytest.importorskip("cupy")
    from localize_psf.optical_bounds import gaussian_width_bounds

    result = gaussian_width_bounds(
        1.35,
        0.637,
        1.4,
        cp.asarray([0.23, 0.115, 0.15]),
        relative_sigma_range=(0.5, 2.0),
        sampling_floor_pixels=floor,
    )
    expected = np.array(
        [
            np.sqrt(6) / np.pi * 1.4 * 0.637 / 1.35**2,
            1.49686886 / (2 * np.pi) * 0.637 / 1.35,
            1.49686886 / (2 * np.pi) * 0.637 / 1.35,
        ]
    )
    assert all(isinstance(result[key], cp.ndarray) for key in result)
    np.testing.assert_allclose(result["nominal_sigma_zyx_um"].get(), expected, rtol=1e-13)
    np.testing.assert_allclose(
        result["lower_sigma_zyx_um"].get(),
        np.maximum(0.5 * expected, floor * np.array([0.23, 0.115, 0.15])),
    )
    np.testing.assert_allclose(result["upper_sigma_zyx_um"].get(), 2 * expected)


@pytest.mark.parametrize(
    "change",
    ["na", "wavelength", "index", "spacing", "range", "floor", "undersampled"],
)
def test_invalid_optics_cannot_produce_arbitrary_bounds(change):
    from localize_psf.optical_bounds import gaussian_width_bounds

    args = [1.35, 0.637, 1.4, np.array([0.23, 0.115, 0.115])]
    options = dict(relative_sigma_range=(0.5, 2.0), sampling_floor_pixels=0.5, use_gpu=False)
    if change == "na":
        args[0] = 1.5
    elif change == "wavelength":
        args[1] = np.nan
    elif change == "index":
        args[2] = 0
    elif change == "spacing":
        args[3] = np.ones(2)
    elif change == "range":
        options["relative_sigma_range"] = (1.1, 2.0)
    elif change == "floor":
        options["sampling_floor_pixels"] = -1
    elif change == "undersampled":
        args[3] = np.array([3.0, 2.0, 2.0])
    with pytest.raises(ValueError):
        gaussian_width_bounds(*args, **options)


@pytest.mark.gpu
def test_named_optical_initial_values_reach_actual_bounded_cuda_fit():
    cp = pytest.importorskip("cupy")
    from localize_psf.fit_psf import gaussian3d_psf_model
    from localize_psf.localize import get_coords, localize_beads_generic
    from localize_psf.optical_bounds import gaussian_width_bounds

    spacing = (0.23, 0.115, 0.115)
    origin = (-1.84, -1.38, -1.38)
    shape = (17, 25, 25)
    optics = gaussian_width_bounds(
        1.35,
        0.637,
        1.4,
        cp.asarray(spacing),
        relative_sigma_range=(0.5, 2.0),
        sampling_floor_pixels=0.5,
    )
    nominal = optics["nominal_sigma_zyx_um"]
    lower, upper = optics["lower_sigma_zyx_um"], optics["upper_sigma_zyx_um"]
    model = gaussian3d_psf_model()
    truth = cp.asarray([80, 0.05, -0.04, 0.02, float(nominal[1]), float(nominal[0]), 3.0])
    image = model.model(get_coords(shape, spacing, origin=origin, use_gpu=True), truth)
    _, result, _ = localize_beads_generic(
        image,
        spacing,
        20,
        min_spot_sep=(0.6, 0.6),
        origin=origin,
        roi_size=(2.3, 1.8, 1.8),
        model=model,
        use_gpu_fit=True,
        use_gpu_filter=True,
        bounds=(
            [0, -np.inf, -np.inf, -np.inf, float(cp.max(lower[1:])), float(lower[0]), -np.inf],
            [np.inf, np.inf, np.inf, np.inf, float(cp.min(upper[1:])), float(upper[0]), np.inf],
        ),
        initial_parameter_values={"sxy": nominal[1], "sz": nominal[0]},
        roi_batch_size=1,
        verbose=False,
    )
    np.testing.assert_allclose(result["fit_params"], truth.get()[None], rtol=0.005, atol=0.002)
    np.testing.assert_allclose(result["init_params"][0, [4, 5]], nominal.get()[[1, 0]], rtol=1e-7)


@pytest.mark.parametrize(
    "values", [{"unknown": 1}, {"sz": np.nan}, {"sz": [1, 2]}, {"sz": True}, [1]]
)
def test_invalid_named_initialization_is_rejected_even_for_empty_images(values):
    from localize_psf.localize import localize_beads_generic

    with pytest.raises(ValueError, match="initial"):
        localize_beads_generic(
            np.zeros((5, 9, 9)),
            (0.3, 0.2, 0.2),
            100,
            min_spot_sep=(0.6, 0.6),
            initial_parameter_values=values,
            verbose=False,
        )
