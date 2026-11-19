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


@pytest.mark.parametrize("axes", ["tczyx", "tcyxz"])
@pytest.mark.parametrize(
    "timepoint,channel,roi_bounds",
    [(1, 1, [[4, 6, 0, 2, 0, 3], [4, 7, 0, 3, 0, 4]]), (0, 1, [[0, 2, 4, 6, 0, 2]])],
)
def test_lazy_roi_reads_skip_unselected_chunks(
    localization, tmp_path, axes, timepoint, channel, roi_bounds
):
    import tensorstore as ts
    from yaozarrs import DimSpec, open_group, v05
    from yaozarrs.write.v05 import write_image

    shape = (2, 2, 8, 8, 8)
    counts = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    dims = [
        DimSpec(name=a, size=n, unit="micrometer" if a in "zyx" else None)
        for a, n in zip(axes, shape, strict=True)
    ]
    path = tmp_path / "lazy.ome.zarr"
    write_image(
        path,
        v05.Image(multiscales=[v05.Multiscale.from_dims(dims)]),
        [counts],
        writer="tensorstore",
        chunks=(1, 1, 4, 4, 4),
    )
    # A genuine unreadable chunk proves that opening/selecting/packing does not
    # read the whole image or use an unrelated origin pixel for ROI padding.
    stored = open_group(str(path))["0"].to_tensorstore()
    stored.kvstore.write(b"c/1/1/0/0/0", b"invalid chunk").result()
    data, metadata = localization.open_image(str(path))
    assert isinstance(data, ts.TensorStore)
    data, _ = localization._prepare_data(
        data, dict(metadata, batch_axes=True, psf_diagnostics=False)
    )
    assert isinstance(data, ts.TensorStore)
    rois = np.array(roi_bounds)
    coords = localization.get_coords(shape[-3:], (0.2, 0.1, 0.15), origin=(-4, 2, -3))
    values, coordinates, sizes = localization.prepare_rois(data[timepoint, channel], coords, rois)
    expected = counts.transpose(tuple(axes.index(a) for a in "tczyx"))
    for i, roi in enumerate(rois):
        region = tuple(slice(int(start), int(stop)) for start, stop in roi.reshape(3, 2))
        np.testing.assert_array_equal(
            values[i, : sizes[i]], expected[timepoint, channel][region].ravel()
        )
        assert np.isnan(values[i, sizes[i] :]).all()
        for actual, vector in zip(coordinates, coords, strict=True):
            np.testing.assert_array_equal(
                actual[i, : sizes[i]], np.broadcast_to(vector, shape[-3:])[region].ravel()
            )
    with pytest.raises(ValueError, match="Invalid blosc-compressed data"):
        np.asarray(data[1, 1])


@pytest.mark.parametrize("explicit_group", [False, True])
def test_collection_root_selects_highest_resolution_without_xml(
    localization, tmp_path, explicit_group
):
    import tensorstore as ts
    from yaozarrs import DimSpec, open_group, v05
    from yaozarrs.write.v05 import write_bioformats2raw

    counts = np.arange(2 * 2 * 7 * 19 * 21, dtype=np.uint16).reshape(2, 2, 7, 19, 21)
    dims = [
        DimSpec(name=axis, size=size, scale=scale, unit="micrometer" if axis in "zyx" else None)
        for axis, size, scale in zip(
            "tczyx", counts.shape, (0.015, 1, 0.23, 0.11, 0.13), strict=True
        )
    ]
    multiscale = v05.Multiscale.from_dims(dims, n_levels=2)
    for dataset, name in zip(multiscale.datasets, ("fine", "coarse"), strict=True):
        dataset.path = name
        dataset.coordinateTransformations.append(
            v05.TranslationTransformation(translation=[0, 0, -2, 3, -4])
        )
    path = tmp_path / "pyramid.ome.zarr"
    write_bioformats2raw(
        path,
        {"0": (v05.Image(multiscales=[multiscale]), [counts, counts[:, :, ::2, ::2, ::2]])},
        writer="tensorstore",
        chunks=(1, 1, 4, 8, 8),
    )
    assert not (path / "OME" / "METADATA.ome.xml").exists()
    # A genuinely unreadable lower-resolution chunk must never affect opening
    # or reading the selected full-resolution image.
    coarse = open_group(str(path))["0/coarse"].to_tensorstore()
    coarse.kvstore.write(b"c/0/0/0/0/0", b"invalid chunk").result()
    source = str(path / "0" if explicit_group else path)
    assert localization.ome_zarr_image_paths(source) == [str(path / "0")]
    data, metadata = localization.open_image(source)
    assert isinstance(data, ts.TensorStore) and data.shape == counts.shape
    assert metadata["axes"] == "tczyx"
    np.testing.assert_allclose(metadata["voxel_size_zyx_um"], (0.23, 0.11, 0.13))
    np.testing.assert_allclose(metadata["origin_zyx_um"], (-2, 3, -4))
    np.testing.assert_array_equal(np.asarray(data[1, 1]), counts[1, 1])
    with pytest.raises(ValueError, match="Invalid blosc-compressed data"):
        np.asarray(coarse[0, 0])


def test_image_groups_remain_distinct_from_pyramid_resolutions(localization, tmp_path):
    from yaozarrs import DimSpec, v05
    from yaozarrs.write.v05 import write_bioformats2raw

    counts = np.ones((3, 5, 7), dtype=np.uint16)
    dims = [
        DimSpec(name=axis, size=size, unit="micrometer")
        for axis, size in zip("zyx", counts.shape, strict=True)
    ]
    image = v05.Image(multiscales=[v05.Multiscale.from_dims(dims)])
    path = tmp_path / "positions.ome.zarr"
    write_bioformats2raw(
        path, {"1": (image, [counts]), "0": (image, [counts])}, writer="tensorstore"
    )
    assert localization.ome_zarr_image_paths(str(path)) == [str(path / "1"), str(path / "0")]
    with pytest.raises(ValueError, match="Collection has 2 images"):
        localization.open_image(str(path))
