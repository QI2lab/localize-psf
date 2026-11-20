"""Unit and integration tests of shared, automatically calibrated PSF tilt."""

import json

import numpy as np
import pytest

from localize_psf.config import FitFilters, LocalizationConfig
from localize_psf.fit_psf import gaussian3d_asymmetric_rotated_pixelated
from localize_psf.localize import filter_nearby_peaks, get_coords
from localize_psf.processing import Detector, calibrate_tilt, fit_spots, gaussian_model


@pytest.mark.unit
@pytest.mark.cpu
def test_tilt_configuration_roundtrip_and_plane_validation(tmp_path):
    config = LocalizationConfig(psf_model="tilted_gaussian", psf_tilt_yz_rad=-0.2)
    path = tmp_path / "settings.json"
    config.save(path)
    assert LocalizationConfig.load(path) == config
    assert gaussian_model(config).nparams == 11
    assert gaussian_model(LocalizationConfig()).nparams == 7
    for options in ({"fit_model": "2d"}, {"psf_tilt_yz_rad": 2.0}):
        with pytest.raises(ValueError):
            LocalizationConfig(psf_model="tilted_gaussian", **options)
    assert calibrate_tilt(np.ones((3, 5, 5)), np.zeros((2, 3)), config) is None


@pytest.fixture
def tilted_scene():
    model = gaussian3d_asymmetric_rotated_pixelated()
    shape, spacing, origin = (31, 81, 81), (0.2, 0.1, 0.1), (-2.0, 3.0, -4.0)
    pixels = np.array(
        [
            [15.1, 18.2, 18.3],
            [15.4, 18.3, 60.2],
            [14.8, 60.1, 18.1],
            [15.3, 60.2, 60.1],
            [15.0, 39.8, 40.2],
        ]
    )
    centers = origin + pixels * spacing
    angle = -0.18
    truth = np.array(
        [
            [a, *c[::-1], 0.16, 0.18, 0.36, 5, np.pi / 2, angle, -np.pi / 2]
            for a, c in zip((1200, 1400, 1600, 1800, 2000), centers)
        ]
    )
    coords = get_coords(shape, spacing, origin=origin)
    image = 5 + sum(model.model(coords, p) - p[7] for p in truth)
    image = np.random.default_rng(19).poisson(image).astype(np.float32)
    config = LocalizationConfig(
        psf_model="tilted_gaussian",
        voxel_size_zyx_um=spacing,
        origin_zyx_um=origin,
        fit_roi_size_um=(2.3, 1.5, 1.5),
        return_covariance=True,
        max_number_iterations=500,
        ufish_median_filter=False,
    )
    return image, pixels, centers, config, angle


@pytest.mark.integration
@pytest.mark.gpu
def test_automatic_tilt_uses_one_calibration_for_real_gpu_fits(tilted_scene):
    image, points, centers, config, truth = tilted_scene
    result = fit_spots(image, points, config)
    assert len(result["parameter_names"]) == 11
    assert result["fit_params"].shape == (len(points), 11)
    assert result["psf_tilt_yz_rad"] == pytest.approx(truth, abs=0.04)
    assert result["success"].all()
    np.testing.assert_allclose(result["fit_params"][:, (3, 2, 1)], centers, atol=0.03)
    np.testing.assert_allclose(result["fit_params"][:, 9], result["psf_tilt_yz_rad"], atol=1e-6)
    assert not result["roi_truncated"].any()
    fixed = config.model_copy(update={"psf_tilt_yz_rad": result["psf_tilt_yz_rad"]})
    one = fit_spots(image, points[:1], fixed)
    assert one["fit_params"].shape == (1, 11)  # Reuse needs no three-spot calibration.
    assert one["psf_tilt_yz_rad"] == fixed.psf_tilt_yz_rad
    with pytest.raises(ValueError, match="three isolated"):
        fit_spots(image, points[:1], config)
    limited = config.model_copy(update={"fit_filters": FitFilters(amplitude_max=1)})
    with pytest.raises(ValueError, match="three isolated"):
        fit_spots(image, points, limited)
    empty = fit_spots(image, points[:0], config)
    assert empty["fit_params"].shape == (0, 11)
    assert len(empty["parameter_names"]) == 11


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.example
def test_tilted_preview_batch_export_and_reader_share_the_calibration(
    localization, tilted_scene, tmp_path
):
    import pyarrow.parquet as pq

    image, _, _, config, truth = tilted_scene
    detector = Detector()
    try:
        points, _ = detector.detect(image, config)
    finally:
        detector.release()
    assert len(points) >= 3
    preview = fit_spots(image, points, config)
    resolved = config.model_copy(update={"psf_tilt_yz_rad": preview["psf_tilt_yz_rad"]})
    batch = localization.localize_ufish_prior(image, resolved)
    np.testing.assert_allclose(
        batch["fit_results"]["fit_params"], preview["fit_params"], atol=1e-4, rtol=1e-4
    )
    assert batch["metadata"]["psf_tilt_yz_rad"] == resolved.psf_tilt_yz_rad
    auto = localization.localize_ufish_prior(image, config)
    assert auto["metadata"]["psf_tilt_yz_rad"] == pytest.approx(truth, abs=0.04)
    localization.save_localizations(batch, tmp_path, export_trackpy=True)
    saved = pq.read_table(tmp_path / "localizations.parquet")
    assert saved["fit_params"].type.list_size == 11
    assert saved["covariance"].type.list_size == 11
    assert len(json.loads(saved.schema.metadata[b"localize_psf"])["parameter_names"]) == 11
    rows, metadata = localization.read_localizations(tmp_path)
    assert len(rows) and metadata["psf_model"] == "tilted_gaussian"
    np.testing.assert_allclose(rows["zum"], np.asarray(saved["fit_params"].to_pylist())[:, 3])
    assert np.isfinite(rows[["size_x", "size_y", "size_z"]]).all().all()
    assert len(pq.read_table(tmp_path / "trackpy.parquet")) == int(rows.success.sum())
    analysis = localization.analyze_psf(image, batch["fit_results"], resolved)
    assert analysis["volume_results"][0, 0]["average_psf"].shape == (11, 21, 21)


@pytest.mark.unit
@pytest.mark.cpu
@pytest.mark.parametrize(
    "centers,expected",
    [
        ([[0, 0, 0], [0, 1, 0], [0, 2, 0]], [0, 1, 2]),
        ([[0, 0, 0], [0, 0.1, 0], [0, 2, 0]], [2]),
        ([[0, 0, 0], [0, 0.1, 0], [0, 0.2, 0]], []),
    ],
)
def test_remove_nearby_mode_keeps_only_isolated_centers(centers, expected):
    centers = np.asarray(centers, dtype=float)
    retained, indices = filter_nearby_peaks(centers, 0.15, 0.2, mode="remove", nmax=2)
    expected = np.asarray(expected, dtype=int)
    np.testing.assert_array_equal(indices, expected)
    np.testing.assert_array_equal(retained, centers[expected])


@pytest.mark.integration
@pytest.mark.cpu
@pytest.mark.example
def test_rotated_parameter_export_uses_world_axis_widths_without_cuda():
    import pyarrow as pa

    from localize_psf.batch_fitting import fit_rois
    from localize_psf.localize import prepare_rois
    from localize_psf.workflow import to_trackpy_dataframe

    spacing, origin = (0.2, 0.1, 0.1), (-1.6, 3.0, -2.0)
    shape = (17, 31, 25)
    pixels = np.array([8.1, 15.2, 12.3])
    center = origin + pixels * spacing
    model = gaussian3d_asymmetric_rotated_pixelated()
    parameters = np.array(
        [100, *center[::-1], 0.16, 0.17, 0.4, 5, np.pi / 2, np.pi / 2, -np.pi / 2]
    )
    coords = get_coords(shape, spacing, origin=origin)
    image = model.model(coords, parameters)
    values, coordinates, sizes = prepare_rois(image, coords, np.array([[0, 17, 0, 31, 0, 25]]))
    fixed = np.zeros(11, bool)
    fixed[8:] = True
    fits = fit_rois(values, coordinates, sizes, parameters[None], model=model, fixed_params=fixed)
    assert fits["success"].all()
    config = LocalizationConfig(voxel_size_zyx_um=spacing, origin_zyx_um=origin)
    table = pa.table(
        {
            "fit_params": fits["fit_params"].tolist(),
            "success": fits["success"],
            "candidate_timepoints": [0],
            "candidate_channels": [0],
        }
    )
    table = table.replace_schema_metadata(
        {
            b"localize_psf": json.dumps(
                {
                    "metadata": config.model_dump(mode="json"),
                    "parameter_names": model.parameter_names,
                }
            ).encode()
        }
    )
    converted = to_trackpy_dataframe(table)
    np.testing.assert_allclose(converted[["z", "y", "x"]].iloc[0], pixels, atol=1e-5)
    # A quarter turn in YZ swaps the world Y/Z widths while leaving X unchanged.
    np.testing.assert_allclose(
        converted[["size_z", "size_y", "size_x"]].iloc[0].to_numpy() * np.asarray(spacing),
        [0.17, 0.4, 0.16],
        atol=1e-5,
    )
    assert to_trackpy_dataframe(table.slice(0, 0)).empty
