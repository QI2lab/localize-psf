"""Real CUDA integration of shared preview policy, readers, checkpoints and CLI."""

import json
import subprocess
import sys
from concurrent.futures import CancelledError
from threading import Event

import numpy as np
import pytest

from localize_psf.config import LocalizationConfig, Parameter, PreviewSettings
from localize_psf.fit_psf import gaussian3d_psf_model
from localize_psf.localize import get_coords
from localize_psf.processing import Detector, fit_spots, prepare_volume

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


def test_preview_batch_parity_and_original_counts(localization, scene):
    image, config, truth = scene
    config = config.model_copy(update={"exclude_edge_centroids": True})
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
        {"intensity_threshold": 0.6},
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
    assert identity["preprocessing_policy"].endswith("fit-original-counts")
    assert identity["edge_margin_px"] == 0


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
