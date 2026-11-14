"""Integration of real CUDA U-FISH, GPUfit, localization, exports and Typer."""

import json

import numpy as np
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.integration, pytest.mark.example]


def test_localization_and_cli(localization, tmp_path, monkeypatch):
    import pyarrow.parquet as pq
    import pygpufit.gpufit as gf
    import tifffile
    from typer.testing import CliRunner
    from ufish.api import UFish
    from yaozarrs import DimSpec, v05
    from yaozarrs.write.v05 import write_bioformats2raw

    spacing, origin, shape = (0.2, 0.1, 0.1), (-4, 2, -3), (41, 51, 61)
    coords = localization.get_coords(shape, spacing, origin=origin)
    model = localization.gaussian3d_psf_model()
    centers = (
        np.asarray(origin)
        + np.array([[20, 16, 16], [21, 34, 40], [18, 35, 18], [22, 15, 42], [1, 1, 1]]) * spacing
    )
    image = np.full(shape, 4.0)
    for center in centers:
        image += model.model(coords, [100, *center[::-1], 0.14, 0.25, 4]) - 4
    # Simulated bright beads; detection, fitting and diagnostics use production code.
    data = np.random.default_rng(18).poisson(np.stack([image * i for i in (100, 125, 150, 175)]))
    data = data.reshape(2, 2, *shape)
    data[1, 0] = 0
    optics = dict(
        numerical_aperture=1.3,
        wavelength_um={0: 0.6, 1: 0.65},
        refractive_index=1.4,
        relative_sigma_range=(0.5, 2),
    )
    events = []

    def record(target, name, event):
        original = getattr(target, name)

        def call(*args, **kwargs):
            events.append(event)
            if event == "infer":
                assert next(args[0].model.parameters()).is_cuda
            return original(*args, **kwargs)

        monkeypatch.setattr(target, name, call)

    record(UFish, "predict", "infer")
    record(gf, "fit_constrained", "fit")
    result = localization.localize_ufish_prior(
        data,
        dict(optics, voxel_size_zyx_um=spacing, origin_zyx_um=origin, fit_roi_size_um=(1.4, 1, 1)),
    )
    assert events[:4] == ["infer"] * 4 and set(events[4:]) == {"fit"}
    np.testing.assert_array_equal(
        result.fit_results["candidate_ids"], np.arange(len(result.candidate_channels))
    )
    for key, volume in result.volume_results.items():
        ids = volume["candidate_ids"][result.fit_results["success"][volume["candidate_ids"]]]
        assert not volume["focus"]["available"] and volume["average_psf"] is None
        if key == (1, 0):
            assert len(ids) == 0
            continue
        assert len(ids) >= 4
        interior = ids[~result.fit_results["covariance_flags"]["truncated"][ids]]
        assert result.fit_results["covariance_valid"][interior].all()
        assert np.isfinite(result.fit_results["chi_sqrs"][ids]).all()
        np.testing.assert_array_equal(result.candidate_timepoints[ids], key[0])
        np.testing.assert_array_equal(result.candidate_channels[ids], key[1])
        fitted = result.fit_results["fit_params"][ids][:, (3, 2, 1)]
        assert (
            np.linalg.norm(fitted[:, None] - centers[None, :-1], axis=-1).min(axis=0) < 0.05
        ).all()
        bounds = localization.gaussian_width_bounds(
            1.3,
            optics["wavelength_um"][key[1]],
            1.4,
            spacing,
            relative_sigma_range=(0.5, 2),
            sampling_floor_pixels=0,
            use_gpu=False,
        )
        for actual, name in zip(result.fit_results["bounds"], ("lower", "upper"), strict=True):
            expected = bounds[f"{name}_sigma_zyx_um"][[1, 0]]
            np.testing.assert_allclose(actual[ids, 4:6], np.broadcast_to(expected, (len(ids), 2)))

    localization.save_localizations(result, tmp_path / "localizations")
    saved = pq.read_table(tmp_path / "localizations/localizations.parquet")
    for name in ("fit_params", "covariance", "center_covariance_zyx"):
        np.testing.assert_allclose(
            np.asarray(saved[name].to_pylist()), result.fit_results[name], equal_nan=True
        )
    summary = json.loads(saved.schema.metadata[b"localize_psf"])
    assert set(summary["volumes"]) == {"t0_c0", "t0_c1", "t1_c0", "t1_c1"}

    path, output, metadata = (
        tmp_path / "beads.ome.zarr",
        tmp_path / "localization",
        tmp_path / "metadata.json",
    )
    dims = [DimSpec(name="c", size=2)] + [
        DimSpec(name=n, size=s, scale=d, unit="micrometer")
        for n, s, d in zip("zyx", shape, spacing, strict=True)
    ]
    scale = v05.Multiscale.from_dims(dims)
    scale.datasets[0].coordinateTransformations.append(
        v05.TranslationTransformation(translation=[0, *origin])
    )
    write_bioformats2raw(
        path, {"0": (v05.Image(multiscales=[scale]), [data[0]])}, writer="tensorstore"
    )
    metadata.write_text(
        json.dumps(
            dict(
                optics,
                batch_axes=False,
                channel=1,
                psf_diagnostics=True,
                fit_roi_size_um=[1.4, 1, 1],
                psf_roi_size_px=[6, 10, 10],
            )
        ),
        encoding="utf-8",
    )
    arguments = [str(path), "--metadata", str(metadata), "--output-dir", str(output)]
    open_figures = localization.plt.get_fignums()
    cli = CliRunner().invoke(localization.app, arguments)
    assert cli.exit_code == 0, (cli.output, cli.exception)
    assert localization.plt.get_fignums() == open_figures
    saved = pq.read_table(output / "localizations.parquet")
    assert "covariance" in saved.column_names
    summary = json.loads(saved.schema.metadata[b"localize_psf"])
    assert set(summary["volumes"]) == {"t0_c1"}
    assert not summary["metadata"]["batch_axes"] and summary["metadata"]["psf_diagnostics"]
    assert summary["metadata"]["fit_roi_size_um"] == [1.4, 1, 1]
    assert summary["metadata"]["psf_roi_size_px"] == [7, 11, 11]
    assert summary["volumes"]["t0_c1"]["focus"]["available"]
    assert {p.name for p in output.iterdir()} == {
        "localizations.parquet",
        "psf.tif",
        "focus_gradient_and_field_curvature.png",
    }
    with tifffile.TiffFile(output / "psf.tif") as psf:
        assert psf.series[0].axes == "ZYX" and psf.asarray().shape == (7, 11, 11)
        assert psf.imagej_metadata["unit"] == "um" and psf.imagej_metadata["spacing"] == spacing[0]
        np.testing.assert_allclose(np.nanmax(psf.asarray()[3]), 1)

    empty = localization.localize_ufish_prior(
        np.zeros(shape), dict(optics, voxel_size_zyx_um=spacing, origin_zyx_um=origin)
    )
    assert empty.candidate_centroids_zyx_px.shape == (0, 3)
    assert empty.fit_results["fit_params"].shape == (0, 7)
    assert empty.fit_results["covariance"].shape == (0, 7, 7)
