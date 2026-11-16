"""Unit tests of production configuration validation and serialization."""

import json

import pytest
from pydantic import ValidationError

from localize_psf.config import LocalizationConfig, Parameter, PreviewSettings

pytestmark = pytest.mark.unit


def test_strict_json_roundtrip(tmp_path):
    config = LocalizationConfig(blend_3d=False, intensity_threshold=0.75)
    path = tmp_path / "settings.json"
    config.save(path)
    assert LocalizationConfig.load(path) == config
    saved = json.loads(path.read_text())
    assert saved["fit_parameters"]["cx"]["lower"] is None
    assert "Infinity" not in path.read_text()
    assert not saved["blend_3d"]


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
