"""Integration of upstream image readers and stored physical coordinate metadata."""

import numpy as np
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.cpu, pytest.mark.example]


def test_ome_tiff_axis_order_channels_units_and_translations(localization, tmp_path):
    import tifffile

    shape = (2, 3, 2, 11, 13)
    counts = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    z_planes = [z for t, z, c in np.ndindex(shape[:3])]
    path = tmp_path / "counts.ome.tif"
    tifffile.imwrite(
        path,
        counts,
        ome=True,
        photometric="minisblack",
        metadata={
            "axes": "TZCYX",
            "PhysicalSizeX": 130,
            "PhysicalSizeXUnit": "nm",
            "PhysicalSizeY": 110,
            "PhysicalSizeYUnit": "nm",
            "PhysicalSizeZ": 200,
            "PhysicalSizeZUnit": "nm",
            "Channel": {"Name": ["alpha", "beta"]},
            "Plane": {
                "PositionZ": [-1300 + 200 * z for z in z_planes],
                "PositionZUnit": ["nm"] * 12,
                "PositionY": [2.0] * 12,
                "PositionX": [-3.0] * 12,
            },
        },
    )
    data, geometry = localization.open_image(str(path))
    assert geometry["axes"] == "tzcyx"
    assert geometry["channel_names"] == ["alpha", "beta"]
    np.testing.assert_allclose(geometry["voxel_size_zyx_um"], (0.2, 0.11, 0.13))
    np.testing.assert_allclose(geometry["origin_zyx_um"], (-1.3, 2, -3))
    data, keys = localization._prepare_data(
        data, dict(geometry, batch_axes=False, timepoint=1, channel=1, psf_diagnostics=False)
    )
    assert keys == ((1, 1),)
    np.testing.assert_array_equal(data[1, 1], counts[1, :, 1])


def test_ome_zarr_highest_resolution_transforms_and_spatial_axis_order(localization, tmp_path):
    from yaozarrs import DimSpec, v05
    from yaozarrs.write.v05 import write_image

    shape = (2, 2, 17, 19, 5)
    counts = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    dims = [
        DimSpec(name=a, size=n, scale=s, unit="nanometer" if a in "zyx" else None)
        for a, n, s in zip("tcyxz", shape, (1, 1, 110, 130, 200), strict=True)
    ]
    scale = v05.Multiscale.from_dims(dims, n_levels=2)
    scale.coordinateTransformations = [
        v05.ScaleTransformation(scale=[1, 1, 2, 2, 2]),
        v05.TranslationTransformation(translation=[0, 0, 2000, -3000, -1300]),
    ]
    for dataset in scale.datasets:
        dataset.coordinateTransformations.append(
            v05.TranslationTransformation(translation=[0, 0, 100, -200, 300])
        )
    path = tmp_path / "counts.ome.zarr"
    write_image(
        path,
        v05.Image(multiscales=[scale]),
        [counts, counts[:, :, ::2, ::2, ::2]],
        writer="tensorstore",
    )
    data, geometry = localization.open_image(str(path))
    assert data.shape == shape and geometry["axes"] == "tcyxz"
    np.testing.assert_allclose(geometry["voxel_size_zyx_um"], (0.4, 0.22, 0.26))
    np.testing.assert_allclose(geometry["origin_zyx_um"], (-0.7, 2.2, -3.4))
    data, keys = localization._prepare_data(
        data,
        dict(
            geometry,
            batch_axes=True,
            timepoint=1,
            channel=1,
            schema_version=1,
            psf_diagnostics=False,
        ),
    )
    assert keys == ((0, 0), (0, 1), (1, 0), (1, 1))
    np.testing.assert_array_equal(np.asarray(data[1, 1]), counts[1, 1].transpose(2, 0, 1))
