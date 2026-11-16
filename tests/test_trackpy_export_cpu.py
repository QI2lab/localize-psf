"""Trackpy exports use fitted positions and retain converged, covariance-invalid fits."""

import json

import numpy as np
import pytest

pytestmark = [pytest.mark.cpu, pytest.mark.integration, pytest.mark.example]


def test_trackpy_export(localization, tmp_path):
    import pandas as pd
    import pyarrow as pa
    import trackpy as tp

    # Simulated fit evidence with a failed fit and an unavailable covariance.
    parameters = [
        [12, -2.5, 3, -3.4, 0.1, 0.3, 4],
        [100, -2.4, 3, -3.3, 0.2, 0.4, 5],
        [14, -2.45, 3.04, -3.35, 0.12, 0.33, 5],
        [20, -2.5, 3, -3.4, 0.1, 0.3, 4],
    ]
    covariance = np.tile(np.diag([0.0004, 0.0001, 0.0009]), (4, 1, 1))
    covariance[1:3] = np.nan
    metadata = dict(voxel_size_zyx_um=[0.3, 0.2, 0.1], origin_zyx_um=[-4, 2, -3])
    table = pa.table(
        dict(
            fit_params=pa.array(parameters, type=pa.list_(pa.float64(), 7)),
            center_covariance_zyx=pa.array(
                covariance.tolist(), type=pa.list_(pa.list_(pa.float64(), 3), 3)
            ),
            success=[True, False, True, True],
            covariance_valid=[True, False, False, True],
            candidate_timepoints=[10, 11, 11, 10],
            candidate_channels=[0, 0, 0, 1],
        )
    ).replace_schema_metadata({b"localize_psf": json.dumps(dict(metadata=metadata)).encode()})
    source = tmp_path / "localizations.parquet"
    localization._write_parquet(table.to_reader(max_chunksize=2), source)
    output = localization.save_trackpy(source)
    assert output == tmp_path / "trackpy.parquet" and source.exists()
    features = pd.read_parquet(output)
    assert features["frame"].tolist() == [10, 11, 10]
    assert features["channel"].tolist() == [0, 0, 1]
    assert features["signal"].tolist() == [12, 14, 20]
    assert features["covariance_valid"].tolist() == [True, False, True]
    assert not {"mass", "raw_mass", "raw_intensity", "amplitude"} & set(features)
    np.testing.assert_allclose(
        features[["x", "y", "z"]], [[5, 5, 2], [5.5, 5.2, 13 / 6], [5, 5, 2]]
    )
    np.testing.assert_allclose(
        features[["xum", "yum", "zum"]], [[-2.5, 3, -3.4], [-2.45, 3.04, -3.35], [-2.5, 3, -3.4]]
    )
    np.testing.assert_allclose(
        features[["size_x", "size_y", "size_z"]], [[1, 0.5, 1], [1.2, 0.6, 1.1], [1, 0.5, 1]]
    )
    np.testing.assert_allclose(
        features.loc[[0, 2], ["ep_x", "ep_y", "ep_z"]], np.tile([0.3, 0.05, 1 / 15], (2, 1))
    )
    assert features.loc[1, ["ep_x", "ep_y", "ep_z"]].isna().all()
    linked = tp.link_df(features[features["channel"] == 0], 0.5, pos_columns=["xum", "yum", "zum"])
    assert linked["particle"].nunique() == 1

    localization._write_parquet(table.slice(0, 0).to_reader(), source)
    empty = pd.read_parquet(localization.save_trackpy(source))
    assert empty.empty and empty.columns.equals(features.columns)

    without_covariance = table.drop(["center_covariance_zyx", "covariance_valid"])
    localization._write_parquet(without_covariance.to_reader(), source)
    features = pd.read_parquet(localization.save_trackpy(source))
    assert features["signal"].tolist() == [12, 14, 20]
    assert not features["covariance_valid"].any()
    assert features[["ep_x", "ep_y", "ep_z"]].isna().all().all()
