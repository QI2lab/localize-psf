"""Real CUDA integration of shared preview policy, readers, checkpoints and CLI."""

import json
import subprocess
import sys
from concurrent.futures import CancelledError
from threading import Event

import numpy as np
import pytest

from localize_psf.config import FitFilters, LocalizationConfig, Parameter, PreviewSettings
from localize_psf.fit_psf import gaussian3d_psf_model
from localize_psf.localize import get_coords
from localize_psf.processing import (
    Detector,
    fit_filter_mask,
    fit_spots,
    optical_parameters,
    prepare_volume,
)

pytestmark = [pytest.mark.integration, pytest.mark.gpu, pytest.mark.example]


@pytest.fixture(params=["2d", "3d"])
def scene(request):
    shape = (1 if request.param == "2d" else 19, 35, 37)
    spacing, origin = (0.2, 0.1, 0.11), (-2.0, 3.0, -4.0)
    center = np.array(origin) + np.array((0 if shape[0] == 1 else 9.1, 17.2, 18.1)) * spacing
    truth = np.array([5000, *center[::-1], 0.16, 1 if shape[0] == 1 else 0.35, 5])
    image = (
        gaussian3d_psf_model()
        .model(get_coords(shape, spacing, origin=origin), truth)
        .astype(np.float32)
    )
    config = LocalizationConfig(
        fit_model=request.param,
        voxel_size_zyx_um=spacing,
        origin_zyx_um=origin,
        blend_3d=False,
        return_covariance=True,
        roi_batch_size=1,
        fit_roi_size_um=(1.4, 1.2, 1.2),
        preview=PreviewSettings(crop_zyx_px=(0, 1, 0, 3, 0, 3)),
    )
    parameters = dict(
        config.fit_parameters,
        bg=Parameter(lower=5, upper=5, initial=5, fixed=True),
        cx=Parameter(lower=-0.3, upper=0.3),
        cy=Parameter(lower=-0.3, upper=0.3),
    )
    return (
        image,
        config.model_copy(update={"fit_parameters": parameters}),
        truth,
    )


def test_gpu_median_preprocessing_is_spatial_and_reuses_enhancement(scene):
    import cupy as cp
    from cupyx.scipy.ndimage import median_filter

    image, config, _ = scene
    # Distinct simulated positions across T/C expose accidental cross-volume mixing.
    data = np.stack(
        [
            np.stack([np.roll(image, t * 3, axis=-1), np.roll(image, t * 3 + 2, axis=-2)])
            for t in range(3)
        ]
    )
    data[:, :, 0, 5, 5] += 20000
    original = data.copy()
    detector = Detector()
    try:
        for timepoint, channel in ((0, 1), (1, 0)):
            selected_config = config.model_copy(
                update={
                    "ufish_median_filter": True,
                    "preview": PreviewSettings(
                        axes="tczyx",
                        timepoint=timepoint,
                        channel=channel,
                        crop_zyx_px=(0, image.shape[0], 0, image.shape[1], 0, image.shape[2]),
                    ),
                }
            )
            selected, _ = prepare_volume(data, selected_config)
            reference_config = selected_config.model_copy(update={"ufish_median_filter": False})
            _, raw = detector.detect(selected, reference_config)
            points, enhanced = detector.detect(selected, selected_config)
            assert enhanced is not raw
            _, cached = detector.detect(
                selected, selected_config.model_copy(update={"intensity_threshold": 0.8})
            )
            assert cached is enhanced
            # Exercise the real U-FISH path with an independently prefiltered input.
            filtered = cp.asnumpy(median_filter(cp.asarray(selected), size=3, mode="reflect"))
            expected_points, expected = detector.detect(filtered, reference_config)
            np.testing.assert_allclose(enhanced, expected)
            np.testing.assert_array_equal(points, expected_points)
            _, streamed = detector.detect(selected, selected_config, cache=False)
            np.testing.assert_allclose(streamed, expected)
            _, unfiltered = detector.detect(selected, reference_config)
            assert unfiltered is not enhanced
        np.testing.assert_array_equal(data, original)
    finally:
        detector.release()


def test_preview_batch_parity_and_original_counts(localization, scene):
    image, config, truth = scene
    config = config.model_copy(update={"exclude_edge_centroids": True, "ufish_median_filter": True})
    detector = Detector()
    try:
        points, enhanced = detector.detect(image, config)
        assert len(points)
        _, recalled = detector.detect(image, config.model_copy(update={"intensity_threshold": 0.8}))
        assert recalled is enhanced
        _, smaller_batch = detector.detect(
            image, config.model_copy(update={"inference_batch_size": 7})
        )
        assert smaller_batch is not enhanced
        if config.fit_model == "3d":
            _, blended = detector.detect(image, config.model_copy(update={"blend_3d": True}))
            assert blended.shape == image.shape
        weights = str(detector._detector.weight_path)
        _, local = detector.detect(image, config.model_copy(update={"ufish_weights": weights}))
        np.testing.assert_allclose(local, enhanced)
    finally:
        detector.release()
    preview = fit_spots(image, points, config)
    batch = localization.localize_ufish_prior(image, config)
    np.testing.assert_allclose(batch["candidate_centroids_zyx_px"], points, atol=1e-7)
    assert np.any(
        batch["candidate_centroids_zyx_px"][:, 1:] > 3
    )  # Ignore the exported preview crop.
    np.testing.assert_array_equal(batch["fit_results"]["candidate_ids"], np.arange(len(points)))
    for name in ("fit_params", "init_params", "covariance", "center_covariance_zyx"):
        np.testing.assert_allclose(
            batch["fit_results"][name], preview[name], rtol=1e-3, atol=1e-5, equal_nan=True
        )
    for name in ("success", "fit_states", "active_bounds", "covariance_valid"):
        np.testing.assert_array_equal(batch["fit_results"][name], preview[name])
    for name, expected in preview["covariance_flags"].items():
        np.testing.assert_array_equal(batch["fit_results"]["covariance_flags"][name], expected)
    nearest = np.argmin(
        np.linalg.norm(batch["fit_results"]["fit_params"][:, 1:4] - truth[1:4], axis=1)
    )
    np.testing.assert_allclose(
        batch["fit_results"]["fit_params"][nearest], truth, atol=0.02, rtol=1e-3
    )
    np.testing.assert_array_equal(batch["fit_results"]["fit_params"][:, 6], 5)
    if config.fit_model == "2d":
        np.testing.assert_array_equal(
            batch["fit_results"]["fit_params"][:, 3], config.origin_zyx_um[0]
        )
        np.testing.assert_array_equal(batch["fit_results"]["fit_params"][:, 5], 1)


def test_fit_filters_preview_batch_and_export(localization, scene, tmp_path):
    import pyarrow.parquet as pq

    image, config, truth = scene
    filters = FitFilters(
        amplitude_min=float(truth[0] / 2), amplitude_max=float(truth[0] * 2), convergence="pass"
    )
    config = config.model_copy(update={"fit_filters": filters})
    batch = localization.localize_ufish_prior(image, config)
    fits = batch["fit_results"]
    eligible = (
        fits["success"]
        & (fits["fit_params"][:, 0] >= filters.amplitude_min)
        & (fits["fit_params"][:, 0] <= filters.amplitude_max)
    )
    expected = fit_filter_mask(
        fits["fit_params"],
        fits["success"],
        fits["covariance_valid"],
        filters,
        collapse_distance_zyx_um=optical_parameters(config)["collapse_distance_zyx_um"],
    )
    assert expected.any()
    assert np.all(eligible[expected])
    np.testing.assert_array_equal(fits["filter_pass"], expected)
    preview = fit_spots(image, batch["candidate_centroids_zyx_px"], config)
    np.testing.assert_array_equal(preview["filter_pass"], expected)
    localization.save_localizations(batch, tmp_path, export_trackpy=True)
    saved = pq.read_table(tmp_path / "localizations.parquet")
    assert len(saved) == expected.sum()
    assert len(pq.read_table(tmp_path / "trackpy.parquet")) == expected.sum()
    np.testing.assert_array_equal(
        np.asarray(saved["candidate_ids"]), fits["candidate_ids"][expected]
    )
    np.testing.assert_allclose(
        np.array(saved["fit_params"].to_pylist()), fits["fit_params"][expected]
    )
    with pytest.raises(ValueError, match="not computed"):
        fit_filter_mask(fits["fit_params"], fits["success"], None, FitFilters(covariance="fail"))
    empty = fit_filter_mask(
        fits["fit_params"][:0], fits["success"][:0], None, FitFilters(amplitude_min=1)
    )
    assert empty.shape == (0,) and empty.dtype == np.bool_
    for selector in ("pass", "fail"):
        settings = FitFilters(covariance=selector, convergence=selector)
        mask = fit_filter_mask(
            fits["fit_params"], fits["success"], fits["covariance_valid"], settings
        )
        np.testing.assert_array_equal(
            mask,
            (fits["success"] == (selector == "pass"))
            & (fits["covariance_valid"] == (selector == "pass")),
        )
    fits["filter_pass"] = fit_filter_mask(
        fits["fit_params"], fits["success"], fits["covariance_valid"], FitFilters(amplitude_max=0)
    )
    assert not fits["filter_pass"].any()
    localization.save_localizations(batch, tmp_path, export_trackpy=True)
    assert pq.read_table(tmp_path / "localizations.parquet").schema == saved.schema
    assert len(pq.read_table(tmp_path / "localizations.parquet")) == 0
    assert len(pq.read_table(tmp_path / "trackpy.parquet")) == 0


@pytest.mark.parametrize("all_axes", [False, True])
def test_full_time_channel_scope_and_separate_xy_planes(localization, scene, all_axes):
    image, config, _ = scene
    image = image if config.fit_model == "3d" else np.repeat(image, 2, axis=0)
    counts = np.broadcast_to(image, (2, 2, *image.shape)).copy()
    config = config.model_copy(
        update={
            "batch_axes": all_axes,
            "timepoint": 1,
            "channel": 1,
            "preview": PreviewSettings(axes="tczyx"),
        }
    )
    result = localization.localize_ufish_prior(counts, config)
    assert result["candidate_timepoints"].flags.c_contiguous
    assert result["candidate_channels"].flags.c_contiguous
    pairs = set(zip(result["candidate_timepoints"], result["candidate_channels"], strict=True))
    assert pairs == ({(0, 0), (0, 1), (1, 0), (1, 1)} if all_axes else {(1, 1)})
    np.testing.assert_array_equal(
        result["fit_results"]["candidate_ids"], np.arange(len(result["candidate_channels"]))
    )
    for (timepoint, channel), record in result["volume_results"].items():
        expected = np.flatnonzero(
            (result["candidate_timepoints"] == timepoint)
            & (result["candidate_channels"] == channel)
        )
        np.testing.assert_array_equal(record["candidate_ids"], expected)
    if config.fit_model == "2d":
        assert set(result["candidate_centroids_zyx_px"][:, 0]) == {0, 1}
        np.testing.assert_allclose(
            result["fit_results"]["fit_params"][:, 3],
            config.origin_zyx_um[0]
            + result["candidate_centroids_zyx_px"][:, 0] * config.voxel_size_zyx_um[0],
        )
        np.testing.assert_array_equal(result["fit_results"]["fit_params"][:, 5], 1)


def test_checkpoint_identity_and_fit_only_reuse(localization, scene, tmp_path):
    image, config, _ = scene
    result = localization.localize_ufish_prior(image, config, tmp_path)
    checkpoint = tmp_path / "centroids.parquet"
    before = checkpoint.read_bytes()
    refit_config = config.model_copy(
        update={
            "roi_batch_size": 3,
            "fit_roi_size_um": (1.8, 1.4, 1.4),
            "return_covariance": False,
            "tolerance": 2e-6,
            "edge_margin_px": 3,
        }
    )
    refit = localization.localize_ufish_prior(image, refit_config, tmp_path)
    assert checkpoint.read_bytes() == before
    np.testing.assert_array_equal(
        refit["candidate_centroids_zyx_px"], result["candidate_centroids_zyx_px"]
    )
    assert "covariance" not in refit["fit_results"]
    localization.save_localizations(refit, tmp_path, export_trackpy=True)
    for update in (
        {"blend_3d": True},
        {"ufish_median_filter": True},
        {"intensity_threshold": 0.6},
        {"collapse_close_priors": False},
        {"wavelength_um": 0.7},
        {"numerical_aperture": 1.25},
        {"refractive_index": 1.5},
        {"exclude_edge_centroids": True},
        {"inference_batch_size": 7},
        {"ufish_weights": "other_weights.pth"},
        {"origin_zyx_um": (-1.0, 3.0, -4.0)},
        {"voxel_size_zyx_um": (0.3, 0.1, 0.11)},
    ):
        with pytest.raises(ValueError, match="different input or U-FISH settings"):
            localization.localize_ufish_prior(image, config.model_copy(update=update), tmp_path)
    if config.fit_model == "3d":
        with pytest.raises(ValueError, match="U-FISH settings"):
            localization.localize_ufish_prior(
                image, config.model_copy(update={"fit_model": "2d"}), tmp_path
            )
    with pytest.raises(ValueError, match="different input"):
        localization.localize_ufish_prior(image[..., :-1], config, tmp_path)
    import pyarrow.parquet as pq

    identity = json.loads(pq.read_schema(checkpoint).metadata[b"localize_psf"])
    assert identity["preprocessing_policy"].endswith("collapse-optical-fwhm-radial-v2")
    assert identity["collapse_close_priors"]
    assert identity["prior_merge_optics"]["0"] == [
        config.numerical_aperture,
        config.wavelength_um,
        config.refractive_index,
    ]
    assert identity["edge_margin_px"] == 0

    table = pq.read_table(checkpoint)
    for former_policy in ("collapse-optical-fwhm-box-v1", "collapse-optical-fwhm-radial-v1"):
        previous = dict(identity)
        previous["preprocessing_policy"] = identity["preprocessing_policy"].replace(
            "collapse-optical-fwhm-radial-v2", former_policy
        )
        pq.write_table(
            table.replace_schema_metadata({b"localize_psf": json.dumps(previous).encode()}),
            checkpoint,
        )
        with pytest.raises(ValueError, match="different input or U-FISH settings"):
            localization.localize_ufish_prior(image, config, tmp_path)

    enabled = config.model_copy(update={"exclude_edge_centroids": True})
    filtered_output = tmp_path / "filtered"
    localization.localize_ufish_prior(image, enabled, filtered_output)
    filtered_checkpoint = filtered_output / "centroids.parquet"
    table = pq.read_table(filtered_checkpoint)
    previous = json.loads(table.schema.metadata[b"localize_psf"])
    assert previous["preprocessing_policy"].endswith("edge-projected-footprint-v1")
    # Simulate a checkpoint produced by the former rectangular-edge policy.
    previous["preprocessing_policy"] = identity["preprocessing_policy"]
    pq.write_table(
        table.replace_schema_metadata({b"localize_psf": json.dumps(previous).encode()}),
        filtered_checkpoint,
    )
    with pytest.raises(ValueError, match="different input or U-FISH settings"):
        localization.localize_ufish_prior(image, enabled, filtered_output)


def test_empty_cancelled_and_infeasible_shared_fits(localization, scene, tmp_path):
    image, config, truth = scene
    with pytest.raises(ValueError, match="ZYX"):
        Detector().detect(image[0], config)
    empty = localization.localize_ufish_prior(np.zeros_like(image), config, tmp_path)
    assert empty["fit_results"]["fit_params"].shape == (0, 7)
    assert empty["fit_results"]["covariance"].shape == (0, 7, 7)
    localization.save_localizations(empty, tmp_path)
    points = ((truth[[3, 2, 1]] - config.origin_zyx_um) / config.voxel_size_zyx_um)[None]
    assert fit_spots(image, np.empty((0, 3)), config)["covariance"].shape == (0, 7, 7)
    cancelled = Event()
    cancelled.set()
    with pytest.raises(CancelledError):
        fit_spots(image, points, config, cancelled)
    parameters = dict(config.fit_parameters, cx=Parameter(lower=0.2, upper=0.4))
    with pytest.raises(ValueError, match="feasible"):
        fit_spots(image, points, config.model_copy(update={"fit_parameters": parameters}))
    messages = []
    fits = fit_spots(image, np.repeat(points, 3, axis=0), config, progress=messages.append)
    np.testing.assert_array_equal(fits["candidate_ids"], [0, 1, 2])
    assert fits["success"].all() and len(messages) == 3
    for invalid in (np.zeros((1, 2)), np.array([[np.nan, 0, 0]]), np.array([[-1, 0, 0]])):
        with pytest.raises(ValueError, match="Candidate coordinates"):
            fit_spots(image, invalid, config)


def test_exported_configuration_runs_batch_with_explicit_calibration(scene, tmp_path):
    import pyarrow.parquet as pq
    import tifffile

    image, config, truth = scene
    path = tmp_path / "input's counts.ome.tif"
    tifffile.imwrite(
        path,
        image,
        ome=True,
        photometric="minisblack",
        metadata={"axes": "ZYX", "PhysicalSizeX": 9, "PhysicalSizeY": 8, "PhysicalSizeZ": 7},
    )
    config = config.model_copy(update={"input_uri": str(path)})
    output = tmp_path / "output's fits"
    settings = tmp_path / "settings.json"
    config.save(settings)
    command = [
        sys.executable,
        "-m",
        "localize_psf.batch",
        str(path),
        "--metadata",
        str(settings),
        "--output-dir",
        str(output),
    ]
    completed = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stderr + completed.stdout
    table = pq.read_table(output / "localizations.parquet")
    metadata = json.loads(table.schema.metadata[b"localize_psf"])["metadata"]
    np.testing.assert_allclose(metadata["voxel_size_zyx_um"], config.voxel_size_zyx_um)
    np.testing.assert_allclose(metadata["origin_zyx_um"], config.origin_zyx_um)
    fitted = np.asarray(table["fit_params"].to_pylist())
    assert len(fitted) > 0
    assert np.min(np.linalg.norm(fitted[:, 1:4] - truth[1:4], axis=1)) < 0.01


def test_null_center_starts_and_explicit_offset_bounds(scene):
    image, config, truth = scene
    point = (truth[[3, 2, 1]] - config.origin_zyx_um) / config.voxel_size_zyx_um
    shifted = point + (0, 0.7, 0.8)
    estimated = fit_spots(image, shifted[None], config)
    prior_xyz = (np.array(config.origin_zyx_um) + shifted * config.voxel_size_zyx_um)[::-1]
    assert np.linalg.norm(estimated["init_params"][0, 1:3] - prior_xyz[:2]) > 0.02
    np.testing.assert_allclose(estimated["init_params"][0, 1:3], truth[1:3], atol=0.02)
    parameters = dict(config.fit_parameters, cx=Parameter(lower=0.05, upper=0.3, initial=0.1))
    bounded = fit_spots(
        image, point[None], config.model_copy(update={"fit_parameters": parameters})
    )
    np.testing.assert_allclose(bounded["fit_params"][0, 1], truth[1] + 0.05, atol=5e-4)
    assert bounded["active_bounds"][0, 1]
    assert bounded["covariance_flags"]["active_bound"][0]


def test_iteration_limit_retains_failed_candidate_rows(scene):
    image, config, truth = scene
    point = (truth[[3, 2, 1]] - config.origin_zyx_um) / config.voxel_size_zyx_um
    parameters = dict(config.fit_parameters, sxy=Parameter(lower=0.03, upper=1, initial=0.8))
    limited = config.model_copy(
        update={
            "estimator": "LSE",
            "max_number_iterations": 1,
            "tolerance": 1e-12,
            "fit_parameters": parameters,
        }
    )
    result = fit_spots(image, np.tile(point, (3, 1)), limited)
    assert result["fit_params"].shape == (3, 7)
    np.testing.assert_array_equal(result["candidate_ids"], [0, 1, 2])
    assert np.any(~result["success"])
    assert np.all(result["niters"] <= 1)
    assert result["active_bounds"].shape == (3, 7)
    assert all(flags.shape == (3,) for flags in result["covariance_flags"].values())


def test_cropped_caller_applies_physical_origin_once(scene):
    image, config, truth = scene
    point = (truth[[3, 2, 1]] - config.origin_zyx_um) / config.voxel_size_zyx_um
    first = np.array([0 if config.fit_model == "2d" else 3, 4, 5])
    stops = np.array(image.shape) - (0 if config.fit_model == "2d" else 2, 3, 3)
    preview = PreviewSettings(
        crop_zyx_px=tuple(int(v) for pair in zip(first, stops, strict=True) for v in pair)
    )
    cropped, crop_origin = prepare_volume(image, config.model_copy(update={"preview": preview}))
    full = fit_spots(image, point[None], config)
    selected = fit_spots(cropped, (point - first)[None], config, origin_zyx_um=crop_origin)
    np.testing.assert_allclose(selected["fit_params"], full["fit_params"], rtol=1e-5, atol=1e-5)
    for actual, expected in zip(selected["bounds"], full["bounds"], strict=True):
        np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(
        selected["center_covariance_zyx"],
        full["center_covariance_zyx"],
        rtol=1e-4,
        atol=1e-8,
        equal_nan=True,
    )


def test_final_fit_collapse_across_roi_batches(scene):
    image, config, truth = scene
    # These distinct priors exceed the initial lateral separation limit, but
    # overlapping ROIs can converge on the same physical emitter.
    center = (truth[[3, 2, 1]] - config.origin_zyx_um) / config.voxel_size_zyx_um
    priors = np.repeat(center[None], 2, axis=0)
    priors[:, 2] += [-2.5, 2.5]
    fits = fit_spots(image, priors, config)
    distances = fits["collapse_distance_zyx_um"]
    assert np.any(np.abs((priors[0] - priors[1]) * config.voxel_size_zyx_um) > distances)
    assert fits["success"].all()
    centers = fits["fit_params"][:, (3, 2, 1)]
    assert np.all(np.abs(centers[0] - centers[1]) <= distances)
    assert fits["filter_pass"].sum() == 1
    assert config.roi_batch_size == 1
    # Live quality filters can select an alternative fit before collapsing.
    for selector in ("all", "pass", "fail"):
        mask = fit_filter_mask(
            fits["fit_params"],
            fits["success"],
            fits.get("covariance_valid"),
            FitFilters(covariance=selector),
            collapse_distance_zyx_um=distances,
        )
        eligible = (
            np.ones(2, dtype=bool)
            if selector == "all"
            else fits["covariance_valid"] == (selector == "pass")
        )
        assert mask.sum() == int(eligible.any())
        assert np.all(eligible[mask])
    weaker = float(fits["fit_params"][:, 0].min())
    limited = fit_filter_mask(
        fits["fit_params"],
        fits["success"],
        fits.get("covariance_valid"),
        FitFilters(amplitude_max=weaker),
        collapse_distance_zyx_um=distances,
    )
    assert limited.sum() == 1 and fits["fit_params"][limited, 0][0] <= weaker
    unrestricted = fit_spots(
        image, priors, config.model_copy(update={"collapse_close_priors": False})
    )
    assert unrestricted["collapse_distance_zyx_um"] is None
    assert unrestricted["filter_pass"].all()


def test_final_batch_collapse_respects_time_channel_and_plane(localization, scene):
    image, config, _ = scene
    # A small optical distance leaves voxel-grid priors distinct; fitted centers
    # can nevertheless coincide within that distance. Repeated volumes must
    # remain independent when selecting accepted localizations.
    config = config.model_copy(
        update={
            "wavelength_um": 0.05 if config.fit_model == "3d" else config.wavelength_um,
            "intensity_threshold": 0.05,
            "batch_axes": True,
            "fit_filters": FitFilters(amplitude_min=100, convergence="pass"),
            "preview": PreviewSettings(axes="tczyx"),
        }
    )
    image = image if config.fit_model == "3d" else np.repeat(image, 2, axis=0)
    data = np.broadcast_to(image, (2, 2, *image.shape)).copy()
    result = localization.localize_ufish_prior(data, config)
    fits = result["fit_results"]
    keys = np.column_stack((result["candidate_timepoints"], result["candidate_channels"]))
    if config.fit_model == "2d":
        keys = np.column_stack((keys, result["candidate_centroids_zyx_px"][:, 0]))
    distance = optical_parameters(config)["collapse_distance_zyx_um"]
    for key in np.unique(keys, axis=0):
        selected = np.all(keys == key, axis=1)
        expected = fit_filter_mask(
            fits["fit_params"][selected],
            fits["success"][selected],
            fits["covariance_valid"][selected],
            config.fit_filters,
            collapse_distance_zyx_um=distance,
        )
        assert expected.any()
        np.testing.assert_array_equal(fits["filter_pass"][selected], expected)
    if config.fit_model == "3d":
        quality_only = fit_filter_mask(
            fits["fit_params"], fits["success"], fits["covariance_valid"], config.fit_filters
        )
        assert fits["filter_pass"].sum() < quality_only.sum()


def test_optical_prior_collapse_reuses_enhancement(scene):
    from scipy.spatial.distance import cdist, pdist

    image, config, truth = scene
    # Two resolved emitters lie inside the configured optical merge distance.
    secondary = truth.copy()
    secondary[0] *= 0.7
    secondary[1] += 0.66
    secondary[6] = 0
    image = image + gaussian3d_psf_model().model(
        get_coords(image.shape, config.voxel_size_zyx_um, origin=config.origin_zyx_um),
        secondary,
    ).astype(np.float32)
    config = config.model_copy(update={"wavelength_um": 2.4})
    raw_config = config.model_copy(update={"collapse_close_priors": False})
    detector = Detector()
    try:
        raw, enhanced = detector.detect(image, raw_config)
        merged, cached = detector.detect(image, config)
        assert cached is enhanced and 0 < len(merged) < len(raw)
        distances = np.asarray(optical_parameters(config)["collapse_distance_zyx_um"])
        centers = merged * config.voxel_size_zyx_um
        assert np.all(
            (pdist(centers[:, 1:]) > distances[1]) | (pdist(centers[:, :1]) > distances[0])
        )
        raw_centers = raw * config.voxel_size_zyx_um
        nearby = (cdist(raw_centers[:, 1:], centers[:, 1:]) <= distances[1]) & (
            cdist(raw_centers[:, :1], centers[:, :1]) <= distances[0]
        )
        raw_pixels = raw.astype(int)[:, 1:] if config.fit_model == "2d" else raw.astype(int)
        merged_pixels = (
            merged.astype(int)[:, 1:] if config.fit_model == "2d" else merged.astype(int)
        )
        raw_strengths = enhanced[tuple(raw_pixels.T)]
        strengths = enhanced[tuple(merged_pixels.T)]
        assert np.all(np.any(nearby & (strengths[None] >= raw_strengths[:, None]), axis=1))
        narrower = config.model_copy(update={"wavelength_um": config.wavelength_um / 2})
        narrower_points, cached = detector.detect(image, narrower)
        assert cached is enhanced and len(narrower_points) > len(merged)
        restored, cached = detector.detect(image, raw_config)
        assert cached is enhanced
        np.testing.assert_array_equal(restored, raw)
        streamed, _ = detector.detect(image, config, cache=False)
        np.testing.assert_array_equal(streamed, merged)
    finally:
        detector.release()


def test_edge_filter_reuses_enhancement_and_streaming_policy(scene):
    image, config, _ = scene
    detector = Detector()
    try:
        original, enhanced = detector.detect(image, config)
        filtered_config = config.model_copy(update={"exclude_edge_centroids": True})
        filtered, reused = detector.detect(image, filtered_config)
        assert reused is enhanced and len(filtered)
        dimensions = range(1 if config.fit_model == "2d" else 0, 3)
        for axis in dimensions:
            assert filtered[:, axis].min() >= filtered_config.edge_margin_px
            assert filtered[:, axis].max() <= image.shape[axis] - 1 - filtered_config.edge_margin_px
        if config.fit_model == "2d":
            np.testing.assert_array_equal(filtered[:, 0], 0)
        wider_config = filtered_config.model_copy(update={"edge_margin_px": 3})
        wider, reused = detector.detect(image, wider_config)
        assert reused is enhanced and len(wider)
        for axis in dimensions:
            assert wider[:, axis].min() >= 3
            assert wider[:, axis].max() <= image.shape[axis] - 4
        restored, reused = detector.detect(image, config)
        assert reused is enhanced
        np.testing.assert_array_equal(restored, original)
        streamed, _ = detector.detect(image, filtered_config, cache=False)
        np.testing.assert_array_equal(streamed, filtered)
    finally:
        detector.release()


def test_streaming_detection_reuses_weights_without_retaining_images(scene):
    import gc
    import weakref

    image, config, _ = scene
    detector = Detector()
    try:
        current = image.copy()
        pixels = weakref.ref(current)
        points, enhanced = detector.detect(current, config, cache=False)
        assert len(points)
        enhancement = weakref.ref(enhanced)
        model = detector._detector
        del current, enhanced
        gc.collect()
        assert pixels() is None and enhancement() is None
        following = image.copy()
        points = detector.detect(following, config, cache=False)[0]
        assert len(points) and detector._detector is model
    finally:
        detector.release()


@pytest.mark.parametrize("mode", ["2d", "3d"])
def test_zero_padded_sloping_edges_are_excluded_on_every_face(mode):
    shape = (1, 25, 65) if mode == "2d" else (25, 65, 49)
    axes = (1, 2) if mode == "2d" else (0, 1)
    spacing, origin = (0.2, 0.1, 0.11), (-2.0, 3.0, -4.0)
    grid = np.ogrid[tuple(slice(n) for n in shape)]
    # Simulate a deskewed acquisition: a 30-pixel slab sheared by one
    # secondary-axis pixel per primary-axis plane, with zero padding outside.
    support = (grid[axes[1]] - grid[axes[0]] >= 3) & (grid[axes[1]] - grid[axes[0]] <= 32)
    center = np.array([0, 12, 29] if mode == "2d" else [12, 29, 24])
    emitters = [center.copy()]
    for plane in (1, 23):
        point = center.copy()
        point[list(axes)] = (plane, plane + 17)
        emitters.append(point)
    for offset in (4, 31):
        point = center.copy()
        point[axes[1]] = point[axes[0]] + offset
        emitters.append(point)
    if mode == "3d":
        for x in (1, 47):
            point = center.copy()
            point[2] = x
            emitters.append(point)
    model = gaussian3d_psf_model()
    coords = get_coords(shape, spacing, origin=origin)
    image = np.full(shape, 5, np.float32)
    for point in emitters:
        physical = np.asarray(origin) + point * spacing
        image += model.model(
            coords, np.array([5000, *physical[::-1], 0.16, 1 if mode == "2d" else 0.35, 0])
        ).astype(np.float32)
    image[~np.broadcast_to(support, shape)] = 0
    # Dark interior pixels are valid data, not an acquisition boundary.
    hole = (
        (grid[axes[0]] >= 8) & (grid[axes[0]] < 10) & (grid[axes[1]] >= 27) & (grid[axes[1]] < 29)
    )
    image[np.broadcast_to(hole, shape)] = 0
    config = LocalizationConfig(fit_model=mode, blend_3d=True, edge_margin_px=4)
    detector = Detector()
    try:
        raw, enhanced = detector.detect(image, config)
        offset = raw[:, axes[1]] - raw[:, axes[0]]
        # Exercise both sloping faces and both primary-axis faces. A rectangular
        # array-bound check leaves the sloping-face detections behind.
        assert np.any(offset < 8) and np.any(offset > 27)
        assert np.any(raw[:, axes[0]] < 4) and np.any(raw[:, axes[0]] > 20)
        if mode == "3d":
            assert np.any(raw[:, 2] < 4) and np.any(raw[:, 2] > 44)
        enabled = config.model_copy(update={"exclude_edge_centroids": True})
        filtered, reused = detector.detect(image, enabled)
        assert reused is enhanced and len(filtered)
        offset = filtered[:, axes[1]] - filtered[:, axes[0]]
        assert np.all((offset >= 8) & (offset <= 27))
        assert np.all((filtered[:, axes[0]] >= 4) & (filtered[:, axes[0]] <= 20))
        if mode == "3d":
            assert np.all((filtered[:, 2] >= 4) & (filtered[:, 2] <= 44))
        else:
            np.testing.assert_array_equal(filtered[:, 0], 0)
        assert np.min(np.linalg.norm(filtered - center, axis=1)) <= 1
        streamed, _ = detector.detect(image, enabled, cache=False)
        np.testing.assert_array_equal(streamed, filtered)
    finally:
        detector.release()
