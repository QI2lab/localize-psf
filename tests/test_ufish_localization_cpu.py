"""CPU unit and integration tests using the production localization code."""

import json

import numpy as np
import pytest

pytestmark = [pytest.mark.cpu, pytest.mark.example]


@pytest.mark.unit
@pytest.mark.parametrize(
    "axes,selection,missing",
    [
        ("tzyx", {}, "timepoint"),
        ("czyx", {}, "channel"),
        ("tczyx", {"timepoint": 0}, "channel"),
        ("tczyx", {"channel": 0}, "timepoint"),
    ],
)
def test_selection_requires_present_axes(localization, axes, selection, missing):
    with pytest.raises(ValueError, match=f"explicit {missing}"):
        localization.localize_ufish_prior(
            np.ones((1,) * (len(axes) - 3) + (7, 9, 11)),
            dict(axes=axes, voxel_size_zyx_um=(1, 1, 1), batch_axes=False, **selection),
        )


@pytest.mark.unit
def test_array_selection_and_metadata(localization):
    data = np.arange(2 * 3 * 7 * 9 * 11).reshape(2, 3, 7, 9, 11)
    view, keys = localization._prepare_data(
        data,
        dict(
            axes="tczyx",
            batch_axes=False,
            timepoint=1,
            channel=2,
            psf_diagnostics=False,
        ),
    )
    np.testing.assert_array_equal(np.asarray(view[1, 2]), data[1, 2])
    assert keys == ((1, 2),) and np.shares_memory(view, data)
    with pytest.raises(ValueError, match="4D array input requires axes"):
        localization._prepare_data(data[:, 0], {})


@pytest.mark.unit
def test_focus_surface(localization):
    x, y = np.random.default_rng(18).uniform(-4, 4, (2, 20))
    z = 2 + 0.03 * x - 0.04 * y  # Simulated focus plane.
    result = localization._focus_diagnostics(np.column_stack((z, y, x)))
    np.testing.assert_allclose(result["plane_parameters"], [2, 0.03, -0.04], atol=1e-12)
    np.testing.assert_allclose(result["gradient_magnitude"], 0.05)
    np.testing.assert_allclose(result["residual_after_curvature_um"], 0, atol=1e-12)
    assert not localization._focus_diagnostics(np.zeros((2, 3)))["available"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "missing", ["numerical_aperture", "wavelength_um", "refractive_index", "relative_sigma_range"]
)
def test_optical_inputs_are_required(localization, missing):
    optics = dict(
        numerical_aperture=1.3,
        wavelength_um=0.515,
        refractive_index=1.51,
        relative_sigma_range=(0.5, 2),
    )
    optics[missing] = None
    with pytest.raises(ValueError, match=missing):
        localization.localize_ufish_prior(
            np.ones((7, 9, 11)), dict(optics, voxel_size_zyx_um=(0.3, 0.1, 0.1))
        )


@pytest.mark.unit
@pytest.mark.parametrize("selection", [{}, {"batch_axes": False, "timepoint": 0}])
def test_timelapse_psf_diagnostics_are_rejected(localization, selection):
    with pytest.raises(ValueError, match="time-lapse"):
        localization.localize_ufish_prior(
            np.ones((2, 7, 9, 11)),
            dict(axes="tzyx", voxel_size_zyx_um=(0.3, 0.1, 0.1), psf_diagnostics=True, **selection),
        )


@pytest.mark.integration
@pytest.mark.parametrize("timepoint", [0, 1])
@pytest.mark.parametrize("channel", [0, 1])
def test_selected_volume_roi_packing(localization, timepoint, channel):
    data = np.arange(2 * 2 * 7 * 9 * 11, dtype=np.uint16).reshape(2, 2, 7, 9, 11)
    coords = localization.get_coords(data.shape[-3:], (0.3, 0.2, 0.1), origin=(-4, 2, -3))
    rois = np.array([[0, 2, 0, 3, 0, 4], [2, 7, 4, 9, 6, 11]])
    values, coordinates, counts = localization.prepare_rois(data[timepoint, channel], coords, rois)
    for i, roi in enumerate(rois):
        selected = tuple(slice(start, stop) for start, stop in roi.reshape(3, 2))
        expected = data[timepoint, channel][selected].ravel()
        assert counts[i] == len(expected)
        np.testing.assert_array_equal(values[i, : counts[i]], expected)
        assert np.isnan(values[i, counts[i] :]).all()
        for actual, field in zip(coordinates, coords, strict=True):
            expected = np.broadcast_to(field, data.shape[-3:])[selected].ravel()
            np.testing.assert_array_equal(actual[i, : counts[i]], expected)
            assert np.isnan(actual[i, counts[i] :]).all()


@pytest.mark.integration
def test_parquet_update_preserves_the_previous_file_after_invalid_input(localization, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "centroids.parquet"
    initial = pa.table(dict(timepoint=[0], channel=[1], z_px=[2.0], y_px=[3.0], x_px=[4.0]))
    updated = pa.concat_tables([initial, initial])
    localization._write_parquet(initial.to_reader(), path)

    invalid = pa.RecordBatchReader.from_batches(initial.schema, [pa.record_batch({"bad": [1]})])
    with pytest.raises(ValueError, match="(?i)schema"):
        localization._write_parquet(invalid, path)
    assert pq.read_table(path).equals(initial)
    localization._write_parquet(updated.to_reader(), path)
    assert pq.read_table(path).equals(updated)
    assert {p.name for p in tmp_path.iterdir()} == {"centroids.parquet"}


@pytest.mark.integration
@pytest.mark.parametrize("version", ["0.5", "0.6"])
@pytest.mark.parametrize("series_count", [0, 1, 2])
def test_tensorstore_metadata_and_cli_selection(localization, tmp_path, version, series_count):
    import tensorstore as ts
    from typer.testing import CliRunner
    from yaozarrs import DimSpec, convert_version, open_group, v05
    from yaozarrs.write import v05 as write05
    from yaozarrs.write import v06 as write06

    counts = np.arange(2 * 2 * 7 * 9 * 11, dtype=np.uint16).reshape(2, 2, 7, 9, 11)
    dims = [DimSpec(name="t", size=2, unit="second"), DimSpec(name="c", size=2)]
    dims += [
        DimSpec(name=n, size=s, scale=d, unit=u)
        for n, s, d, u in zip(
            "zyx",
            (7, 9, 11),
            (300, 0.2, 0.1),
            ("nanometer", "micrometer", "micrometer"),
            strict=True,
        )
    ]
    scale = v05.Multiscale.from_dims(dims, n_levels=2)
    scale.datasets[0].coordinateTransformations.append(
        v05.TranslationTransformation(translation=[0, 0, -400, 2, -1])
    )
    scale.coordinateTransformations = [
        v05.ScaleTransformation(scale=[1, 1, 3, 4, 2]),
        v05.TranslationTransformation(translation=[0, 0, 100, 1, 0.5]),
    ]
    path = tmp_path / "counts.ome.zarr"
    coarse = counts[..., ::2, ::2, ::2]
    image = convert_version(v05.Image(multiscales=[scale]), version)
    writer = write05 if version == "0.5" else write06
    if series_count:
        images = {str(i): (image, [counts, coarse]) for i in range(series_count)}
        writer.write_bioformats2raw(path, images, writer="tensorstore")
    else:
        writer.write_image(path, image, [counts, coarse], writer="tensorstore")
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps(dict(batch_axes=False, timepoint=1, psf_diagnostics=False)),
        encoding="utf-8",
    )
    app, runner = localization.app, CliRunner()
    arguments = [str(path), "--metadata", str(metadata)]
    arguments += ["--output-dir", str(tmp_path / "result")]
    result = runner.invoke(app, arguments)
    if series_count == 2:
        assert result.exit_code == 2 and "pass an image group path" in result.output
        return
    assert result.exit_code == 2 and "explicit channel" in result.output
    assert not (tmp_path / "result").exists()
    group = open_group(str(path / "0" if series_count else path))
    stored = convert_version(group.ome_metadata(), "0.5").multiscales[0]
    spatial = localization._spatial_metadata(stored)
    data, _ = localization._prepare_data(
        group[stored.datasets[0].path].to_tensorstore(),
        dict(spatial, batch_axes=True, psf_diagnostics=False),
    )
    assert isinstance(data, ts.TensorStore)
    for t, c in np.ndindex(2, 2):
        np.testing.assert_array_equal(np.asarray(data[t, c]), counts[t, c])
    np.testing.assert_allclose(spatial["voxel_size_zyx_um"], (0.9, 0.8, 0.2))
    np.testing.assert_allclose(spatial["origin_zyx_um"], (-1.1, 9, -1.5))
