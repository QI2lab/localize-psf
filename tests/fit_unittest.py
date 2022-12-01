"""
Test Jacobians of fit functions
"""

import unittest
import numpy as np
from localize_psf import fit

class Test_psf(unittest.TestCase):

    def setUp(self):
        pass


    def test_gauss1d(self):
        model = fit.gauss1d()
        params = np.array([1.235235, 5.236236, 2.236236236, 0.3246346])
        x = np.linspace(0, 10, 101)
        coords = (x,)

        jn, jcalc = model.test_jacobian(coords, params)

        for ii in range(model.nparams):
            np.testing.assert_allclose(jn[ii], jcalc[ii], atol=1e-8, rtol=1e-5,
                                       err_msg=f"jacobian test failed for parameter {model.parameter_names[ii]:s}")


    def test_gauss2d(self):
        model = fit.gauss2d(use_sigma_ratio_parameterization=False)
        params = np.array([1.235235, 5.236236, 7.3236236236, 2.236236236, 1.236236, 0.3246346, np.pi / 7])
        x = np.linspace(0, 10, 101)
        xx, yy = np.meshgrid(x, x)
        coords = (yy, xx)

        jn, jcalc = model.test_jacobian(coords, params)

        for ii in range(model.nparams):
            np.testing.assert_allclose(jn[ii], jcalc[ii], atol=1e-8, rtol=1e-5,
                                       err_msg=f"jacobian test failed for parameter {model.parameter_names[ii]:s}")


    def test_gauss2d_sigma_ratio(self):
        model = fit.gauss2d(use_sigma_ratio_parameterization=True)
        params = np.array([1.235235, 5.236236, 7.3236236236, 2.236236236, 1.0003333, 0.3246346, np.pi / 7])
        x = np.linspace(0, 10, 101)
        xx, yy = np.meshgrid(x, x)
        coords = (yy, xx)

        jn, jcalc = model.test_jacobian(coords, params)

        for ii in range(model.nparams):
            np.testing.assert_allclose(jn[ii], jcalc[ii], atol=1e-8, rtol=1e-5,
                                       err_msg=f"jacobian test failed for parameter {model.parameter_names[ii]:s}")


    def test_gauss3d(self):
        model = fit.gauss3d()
        params = np.array([1.235235, 5.236236, 7.3236236236, 2.236236236, 3.2362362, 0.3246346, 0.2236236])
        x = np.linspace(0, 10, 101)
        coords = np.meshgrid(x, x, x, indexing="ij")

        jn, jcalc = model.test_jacobian(coords, params)

        for ii in range(model.nparams):
            np.testing.assert_allclose(jn[ii], jcalc[ii], atol=1e-8, rtol=1e-5,
                                       err_msg=f"jacobian test failed for parameter {model.parameter_names[ii]:s}")

    def test_gauss3d_min_sigmas(self):
        model = fit.gauss3d(minimum_sigmas=(0.5, 0.5))
        params = np.array([1.235235, 5.236236, 7.3236236236, 2.236236236, 3.2362362, 0.3246346, 0.2236236])
        x = np.linspace(0, 10, 101)
        coords = np.meshgrid(x, x, x, indexing="ij")

        jn, jcalc = model.test_jacobian(coords, params)

        for ii in range(model.nparams):
            np.testing.assert_allclose(jn[ii], jcalc[ii], atol=1e-8, rtol=1e-5,
                                       err_msg=f"jacobian test failed for parameter {model.parameter_names[ii]:s}")

    def test_gauss3d_asymmetric(self):
        model = fit.gauss3d_asymmetric(use_sigma_ratio_parameterization=False)
        params = np.array([1.235235, 5.236236, 7.3236236236, 2.236236236, 3.2362362, 2.236236,  0.3246346, 0.2236236])
        x = np.linspace(0, 10, 101)
        coords = np.meshgrid(x, x, x, indexing="ij")

        jn, jcalc = model.test_jacobian(coords, params)

        for ii in range(model.nparams):
            np.testing.assert_allclose(jn[ii], jcalc[ii], atol=1e-8, rtol=1e-5,
                                       err_msg=f"jacobian test failed for parameter {model.parameter_names[ii]:s}")

    def test_gauss3d_asymmetric_min_sigmas(self):
        model = fit.gauss3d_asymmetric(use_sigma_ratio_parameterization=False, minimum_sigmas=(0.5, 0.6, 0.7))
        params = np.array([1.235235, 5.236236, 7.3236236236, 2.236236236, 3.2362362, 2.236236,  0.3246346, 0.2236236])
        x = np.linspace(0, 10, 101)
        coords = np.meshgrid(x, x, x, indexing="ij")

        jn, jcalc = model.test_jacobian(coords, params)

        for ii in range(model.nparams):
            np.testing.assert_allclose(jn[ii], jcalc[ii], atol=1e-8, rtol=1e-5,
                                       err_msg=f"jacobian test failed for parameter {model.parameter_names[ii]:s}")

    def test_gauss3d_asymmetric_ratio(self):
        model = fit.gauss3d_asymmetric(use_sigma_ratio_parameterization=True)
        params = np.array([1.235235, 5.236236, 7.3236236236, 2.236236236, 3.2362362, 2.236236,  0.3246346, 0.2236236])
        x = np.linspace(0, 10, 101)
        coords = np.meshgrid(x, x, x, indexing="ij")

        jn, jcalc = model.test_jacobian(coords, params)

        for ii in range(model.nparams):
            np.testing.assert_allclose(jn[ii], jcalc[ii], atol=1e-8, rtol=1e-5,
                                       err_msg=f"jacobian test failed for parameter {model.parameter_names[ii]:s}")

    def test_gauss3d_asymmetric_ratio_min_sigmas(self):
        model = fit.gauss3d_asymmetric(use_sigma_ratio_parameterization=True, minimum_sigmas=(0.5, 0.6, 0.7))
        params = np.array([1.235235, 5.236236, 7.3236236236, 2.236236236, 3.2362362, 2.236236,  0.3246346, 0.2236236])
        x = np.linspace(0, 10, 101)
        coords = np.meshgrid(x, x, x, indexing="ij")

        jn, jcalc = model.test_jacobian(coords, params)

        for ii in range(model.nparams):
            np.testing.assert_allclose(jn[ii], jcalc[ii], atol=1e-8, rtol=1e-5,
                                       err_msg=f"jacobian test failed for parameter {model.parameter_names[ii]:s}")

    def test_gauss3d_rotated(self):
        model = fit.rotated_model(fit.gauss3d(), center_inds=(3, 2, 1))
        params = np.array([1.235235, 5.236236, 7.3236236236, 2.236236236, 3.2362362, 0.3246346, 0.2236236, np.pi * 0.236236, np.pi * 0.1112, np.pi * 0.3])
        x = np.linspace(0, 10, 101)
        coords = np.meshgrid(x, x, x, indexing="ij")

        jn, jcalc = model.test_jacobian(coords, params)

        for ii in range(model.nparams):
            np.testing.assert_allclose(jn[ii], jcalc[ii], atol=1e-8, rtol=1e-5,
                                       err_msg=f"jacobian test failed for parameter {model.parameter_names[ii]:s}")

    def test_gauss3d_rotated_fixed_angles(self):
        model_free = fit.rotated_model(fit.gauss3d(), center_inds=(3, 2, 1))
        model = fit.fixed_parameter_model(model_free, fixed_inds=(8, 9), fixed_values=(0, 0))
        params = np.array([1.235235, 5.236236, 7.3236236236, 2.236236236, 3.2362362, 0.3246346, 0.2236236, np.pi * 0.236236])
        x = np.linspace(0, 10, 101)
        coords = np.meshgrid(x, x, x, indexing="ij")

        jn, jcalc = model.test_jacobian(coords, params)

        for ii in range(model.nparams):
            np.testing.assert_allclose(jn[ii], jcalc[ii], atol=1e-8, rtol=1e-5,
                                       err_msg=f"jacobian test failed for parameter {model.parameter_names[ii]:s}")


if __name__ == "__main__":
    unittest.main()