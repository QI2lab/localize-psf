"""Unit tests of production configuration validation and serialization."""

import json

import pytest
from pydantic import ValidationError

from localize_psf.config import FitFilters, LocalizationConfig, Parameter, PreviewSettings

pytestmark = pytest.mark.unit


def test_strict_json_roundtrip(tmp_path):
    defaults = LocalizationConfig()
    assert defaults.ufish_weights == "finetune_models/v1.0.1-suntag_model.pth"
    assert defaults.intensity_threshold == 0.1
    assert defaults.ufish_median_filter and defaults.exclude_edge_centroids
    assert defaults.edge_margin_px == 1
    config = LocalizationConfig(
        blend_3d=False,
        ufish_median_filter=False,
        intensity_threshold=0.75,
        collapse_close_priors=False,
        exclude_edge_centroids=False,
        edge_margin_px=3,
        return_covariance=True,
        fit_filters=FitFilters(
            amplitude_min=100, amplitude_max=1000, covariance="pass", convergence="fail"
        ),
    )
    path = tmp_path / "settings.json"
    config.save(path)
    assert LocalizationConfig.load(path) == config
    saved = json.loads(path.read_text())
    assert saved["fit_parameters"]["cx"]["lower"] is None
    assert "Infinity" not in path.read_text()
    assert not saved["blend_3d"]
    assert not saved["ufish_median_filter"]
    missing_options = dict(saved)
    for name in ("ufish_median_filter", "exclude_edge_centroids", "edge_margin_px"):
        missing_options.pop(name)
    restored = LocalizationConfig.model_validate_json(json.dumps(missing_options))
    assert restored.ufish_median_filter and restored.exclude_edge_centroids
    assert restored.edge_margin_px == 1
    assert not saved["collapse_close_priors"]
    assert LocalizationConfig().collapse_close_priors
    assert not saved["exclude_edge_centroids"]
    assert saved["edge_margin_px"] == 3
    assert saved["fit_filters"] == config.fit_filters.model_dump()


@pytest.mark.parametrize(
    "values",
    [
        {"amplitude_min": -1},
        {"amplitude_max": float("inf")},
        {"amplitude_min": 2, "amplitude_max": 1},
        {"covariance": "unknown"},
        {"convergence": True},
        {"amplitude_min": "5"},
    ],
)
def test_invalid_fit_filters(values):
    with pytest.raises(ValidationError):
        FitFilters(**values)


def test_covariance_filter_requires_diagnostics():
    with pytest.raises(ValidationError, match="return_covariance"):
        LocalizationConfig(fit_filters=FitFilters(covariance="fail"))
    assert LocalizationConfig().fit_filters == FitFilters()


@pytest.mark.parametrize(
    "values",
    [
        {"lower": 2, "upper": 1},
        {"lower": 1, "upper": 1},
        {"fixed": True},
        {"lower": 1, "initial": 0},
        {"upper": 0, "initial": 1},
        {"lower": float("nan")},
    ],
)
def test_invalid_parameter_controls(values):
    with pytest.raises(ValidationError):
        Parameter(**values)


@pytest.mark.parametrize(
    "values",
    [
        {"axes": "xyy"},
        {"axes": "tz"},
        {"axes": "abc"},
        {"crop_zyx_px": (0, 1, 5, 3, 0, 10)},
        {"crop_zyx_px": (-1, 1, 0, 3, 0, 10)},
    ],
)
def test_invalid_axis_or_crop(values):
    with pytest.raises(ValidationError):
        PreviewSettings(**values)


@pytest.mark.parametrize(
    "values",
    [
        {"schema_version": 2},
        {"voxel_size_zyx_um": (0, 1, 1)},
        {"fit_parameters": {}},
        {"relative_sigma_range": (2, 3)},
        {"numerical_aperture": 2, "refractive_index": 1},
        {"unused_setting": 1},
    ],
)
def test_invalid_configuration(values):
    with pytest.raises(ValidationError):
        LocalizationConfig(**values)


@pytest.mark.parametrize("name,lower", [("sxy", 0), ("amp", -1)])
def test_parameter_domains_require_physical_lower_limits(name, lower):
    defaults = LocalizationConfig()
    values = defaults.model_dump()
    values["fit_parameters"][name]["lower"] = lower
    with pytest.raises(ValidationError, match="physical lower bound"):
        LocalizationConfig.model_validate(values)


def test_optical_sigma_interval_cannot_collapse():
    with pytest.raises(ValidationError, match="ordered"):
        LocalizationConfig(relative_sigma_range=(1, 1))


@pytest.mark.parametrize(
    "values",
    [
        {"schema_version": True},
        {"schema_version": 1.0},
        {"schema_version": "1"},
        {"batch_axes": "true"},
        {"roi_batch_size": 1.5},
        {"blend_3d": 1},
        {"ufish_median_filter": 1},
        {"ufish_median_filter": "true"},
        {"exclude_edge_centroids": "true"},
        {"exclude_edge_centroids": 1},
        {"collapse_close_priors": "true"},
        {"collapse_close_priors": 1},
        {"collapse_fwhm_multiplier": 2.0},
        {"edge_margin_px": 0},
        {"edge_margin_px": -1},
        {"edge_margin_px": 1.5},
        {"edge_margin_px": "2"},
        {"edge_margin_px": True},
        {"intensity_threshold": "0.5"},
        {"preview": {"axes": "zyx", "timepoint": "0"}},
        {"unsupported_preprocessing": "normalize"},
    ],
)
def test_version_one_json_rejects_coercion_and_unsupported_settings(values):
    with pytest.raises(ValidationError):
        LocalizationConfig.model_validate_json(json.dumps(dict(schema_version=1) | values))


@pytest.mark.example
def test_exported_example_roundtrip(tmp_path):
    from pathlib import Path

    supplied = Path(__file__).parents[1] / "examples/localization.json"
    config = LocalizationConfig.load(supplied)
    config = config.model_copy(update={"input_uri": str(tmp_path / "a's image.ome.tif")})
    destination = tmp_path / "a's current settings.json"
    config.save(destination)
    assert LocalizationConfig.load(destination) == config
