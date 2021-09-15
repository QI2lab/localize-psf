import unittest

import numpy as np
import fit

class Test_psf(unittest.TestCase):

    def setUp(self):
        pass

    def test_gauss2d(self):
        nx, ny = 51, 51
        xx, yy = np.meshgrid(range(nx), range(ny))
        ps = [1, xx.mean() + 0.34764376, yy.mean() - 6.46346, 10.347437, 5.4376347, 15 * np.pi/180, 1.2462436]
        num_p = len(ps)

        dp = 1e-5

        jac = fit.gauss2d_jacobian(xx, yy, ps)
        jac_est = [[]] * num_p
        for ii in range(num_p):
            ps_dp = np.array(ps, copy=True)
            ps_dp[ii] -= dp

            jac_est[ii] = 1 / dp * (fit.gauss2d(xx, yy, ps) - fit.gauss2d(xx, yy, ps_dp))
            # print(np.max(np.abs(jac_est[ii] - jac[ii])))
            np.testing.assert_allclose(jac[ii], jac_est[ii], atol=1e-5)

    def test_gauss3d(self):
        nx, ny, nz = 51, 51, 51
        zz, yy, xx = np.meshgrid(range(nz), range(ny), range(nx), indexing='ij')
        ps = [1, xx.mean() + 0.34764376, yy.mean() - 6.46346, zz.mean() + 8.3476436,
              10.347437, 5.4376347, 3.11111, 1.2462436,
              15 * np.pi / 180, 30*np.pi/180, -8 * np.pi/180]
        num_p = len(ps)

        dp = 1e-7

        jac = fit.gauss3d_jacobian(xx, yy, zz, ps)
        jac_est = [[]] * num_p
        for ii in range(num_p):
            ps_dp = np.array(ps, copy=True)
            ps_dp[ii] -= dp

            jac_est[ii] = 1 / dp * (fit.gauss3d(xx, yy, zz, ps) - fit.gauss3d(xx, yy, zz, ps_dp))
            # print(np.max(np.abs(jac_est[ii] - jac[ii])))
            np.testing.assert_allclose(jac[ii], jac_est[ii], atol=1e-5)