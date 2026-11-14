"""Microscopy dispatch contracts and numerical acceptance for GPUfit ABI 3."""

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from localize_psf import batch_fitting
from localize_psf import fit_psf as psf

pytest_plugins = ["test_gpu_psf_derivatives"]


@pytest.fixture
def native_binding(monkeypatch):
    """Expose the public GPUfit contract without loading CUDA or a native binary."""
    models = {
        "GAUSS_3D_ARB": {"id": 15, "parameters": 7},
        "GAUSS_LOR_3D_ARB": {"id": 16, "parameters": 7},
        "GAUSS_3D_ROT_ARB": {"id": 17, "parameters": 11},
        "GAUSS_3D_ASYM_ARB": {"id": 18, "parameters": 8},
    }
    manifest = dict(
        abi_version=3,
        precision="float32",
        constraints_layout="fit-major-interleaved-lower-upper",
        models=models,
    )
    gf = ModuleType("pygpufit.gpufit")
    gf.ModelID = SimpleNamespace(**{name: values["id"] for name, values in models.items()})
    gf.EstimatorID = SimpleNamespace(LSE=0, MLE=1)
    gf.get_capabilities = lambda: manifest
    gf.cuda_available = lambda: True
    calls = []

    def record(data, weights, model_id, initial, *constraints, **options):
        calls.append((data, weights, model_id, initial, constraints, options))
        # Distinct results expose any regrouping that loses candidate alignment.
        parameters = initial.copy()
        parameters[:, 0] += 1
        return parameters, np.zeros(len(data), np.int32), data[:, 0], np.ones(len(data)), 0

    gf.fit = gf.fit_constrained = record
    package = ModuleType("pygpufit")
    package.gpufit = gf
    monkeypatch.setitem(sys.modules, "pygpufit", package)
    monkeypatch.setitem(sys.modules, "pygpufit.gpufit", gf)
    return gf, manifest, calls


@pytest.mark.parametrize(
    "model, expected_id, sigmas",
    [
        (psf.gaussian3d_psf_model(minimum_sigmas=(0.04, 0.08)), 15, [0.04, 0.08]),
        (psf.gaussian_lorentzian_psf_model(), 16, []),
        (
            psf.gaussian3d_asymmetric_rotated_pixelated(minimum_sigmas=(0.03, 0.04, 0.08)),
            17,
            [0.03, 0.04, 0.08],
        ),
        (
            psf.gaussian3d_asymmetric_pixelated(minimum_sigmas=(0.03, 0.04, 0.08)),
            18,
            [0.03, 0.04, 0.08],
        ),
    ],
)
@pytest.mark.parametrize("estimator", ["LSE", "MLE"])
def test_microscopy_dispatch_and_grouped_buffers(
    native_binding, model, expected_id, sigmas, estimator
):
    gf, _, calls = native_binding
    sizes = np.array([5, 3, 4, 2])
    valid = np.arange(5)[None, :] < sizes[:, None]
    data = np.where(valid, np.arange(20).reshape(4, 5) + 1, np.nan)
    coordinates = tuple(np.where(valid, data + offset, np.nan) for offset in (10, 20, 30))
    initial = np.ones((4, model.nparams), np.float32)
    initial[:, 0] = [10, 20, 30, 40]
    lower = np.full_like(initial, -np.inf)
    upper = np.full_like(initial, np.inf)
    lower[[0, 2], 0] = [1, 3]
    upper[[0, 2], 0] = [100, 300]
    fixed = np.zeros(model.nparams, bool)
    fixed[-1] = True
    weights = np.where(valid, 0.5, np.nan) if estimator == "LSE" else None
    result = {
        "fit_params": initial.copy(),
        "fit_states": np.empty(4, np.int32),
        "chi_sqrs": np.empty(4),
        "niters": np.empty(4),
    }
    batch_fitting._fit_gpu(
        result,
        data,
        coordinates,
        sizes,
        valid,
        initial,
        lower,
        upper,
        fixed,
        weights,
        model,
        estimator,
        1e-6,
        90,
    )
    assert len(calls) == (4 if estimator == "MLE" else 2)
    assert {bool(call[4]) for call in calls} == {False, True}
    for packed_data, packed_weights, model_id, starts, constraints, options in calls:
        assert model_id == expected_id
        indices = (starts[:, 0] / 10 - 1).astype(int)
        npoints = packed_data.shape[1]
        group_valid = valid[indices, :npoints]
        np.testing.assert_array_equal(
            packed_data, np.where(group_valid, data[indices, :npoints], 0)
        )
        assert packed_data.dtype == np.float32 and packed_data.flags.c_contiguous
        if estimator == "LSE":
            np.testing.assert_array_equal(packed_weights, np.where(group_valid, 0.5, 0))
        else:
            assert packed_weights is None
            np.testing.assert_array_equal(sizes[indices], npoints)
        expected_info = np.concatenate(
            [
                np.stack(
                    [np.where(group_valid, c[indices, :npoints], 0) for c in coordinates[::-1]],
                    axis=1,
                ).ravel(),
                sizes[indices],
                sigmas,
            ]
        ).astype(np.float32)
        np.testing.assert_array_equal(options["user_info"], expected_info)
        assert options["user_info"].dtype == np.float32
        assert options["estimator_id"] == getattr(gf.EstimatorID, estimator)
        assert options["tolerance"] == 1e-6 and options["max_number_iterations"] == 90
        np.testing.assert_array_equal(options["parameters_to_fit"], ~fixed)
        if constraints:
            endpoints, types = constraints
            assert endpoints.dtype == np.float32 and endpoints.flags.c_contiguous
            np.testing.assert_array_equal(
                endpoints[:, :2], np.column_stack((lower[indices, 0], upper[indices, 0]))
            )
            np.testing.assert_array_equal(endpoints[:, 2:], 0)
            np.testing.assert_array_equal(types, [3, *([0] * (model.nparams - 1))])
    np.testing.assert_array_equal(result["fit_params"][:, 0], initial[:, 0] + 1)
    np.testing.assert_array_equal(result["chi_sqrs"], data[:, 0])
    assert result["success"].all()


@pytest.mark.parametrize(
    "incompatible",
    [
        "abi1",
        "abi2",
        "precision",
        "layout",
        "model_id",
        "parameters",
        "missing_api",
        "missing_model",
    ],
)
def test_reject_incompatible_gpufit_before_dispatch(native_binding, incompatible):
    gf, manifest, calls = native_binding
    if incompatible.startswith("abi"):
        manifest["abi_version"] = int(incompatible[-1])
        gf.ModelID.GAUSS_3D_ARB = 13
        manifest["models"]["GAUSS_3D_ARB"]["id"] = 13
    elif incompatible == "precision":
        manifest["precision"] = "float64"
    elif incompatible == "layout":
        manifest.pop("constraints_layout")
    elif incompatible == "model_id":
        gf.ModelID.GAUSS_3D_ARB = 13
    elif incompatible == "parameters":
        manifest["models"]["GAUSS_3D_ARB"]["parameters"] = 6
    elif incompatible == "missing_api":
        del gf.get_capabilities
    else:
        del gf.ModelID.GAUSS_3D_ARB
    gf.cuda_available = lambda: pytest.fail("incompatible binding must fail before CUDA dispatch")
    with pytest.raises(RuntimeError, match="QI2lab GPUfit 1.2.0.*ABI 3"):
        batch_fitting._fit_gpu(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            psf.gaussian3d_psf_model(),
            "LSE",
            None,
            100,
        )
    assert not calls


@pytest.mark.gpu
@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("estimator", ["LSE", "MLE"])
def test_native_microscopy_models(microscopy_model, bounded, estimator):
    pytest.importorskip("pygpufit.gpufit")
    model, truth, coords = microscopy_model
    # Fit distinct physical coordinate grids and unequal valid lengths in one call.
    coordinates = tuple(np.stack([c, c + offset]) for c, offset in zip(coords, (0.1, -0.2, 0.3)))
    parameters = np.stack([truth, truth])
    parameters[1, 1:4] += [0.3, -0.2, 0.1]
    parameters[1, 0] += 10
    sizes = np.array([len(coords[0]), len(coords[0]) - 37])
    data = np.array(
        [model.model(tuple(c[i] for c in coordinates), p) for i, p in enumerate(parameters)]
    )
    data[1, sizes[1] :] = np.nan
    for c in coordinates:
        c[1, sizes[1] :] = np.nan
    starts = parameters.copy()
    starts[:, 1:4] += 0.02
    bounds = None
    if bounded:
        lower = np.full_like(parameters, -np.inf)
        upper = np.full_like(parameters, np.inf)
        lower[:, 0] = [1, 2]
        upper[:, 0] = [150, 160]
        lower[:, 4 : model.parameter_names.index("bg")] = 0.05
        bounds = lower, upper
    result = batch_fitting.fit_rois(
        data,
        coordinates,
        sizes,
        starts,
        model=model,
        use_gpu=True,
        bounds=bounds,
        estimator=estimator,
        tolerance=1e-7,
        max_number_iterations=200,
    )
    assert result["success"].all()
    np.testing.assert_allclose(result["fit_params"], parameters, atol=0.003, rtol=0.003)
    np.testing.assert_array_equal(result["candidate_ids"], [0, 1])
