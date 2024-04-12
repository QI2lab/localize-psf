"""
Test camera binning function
"""

import unittest
import numpy as np
from localize_psf import camera


class TestCam(unittest.TestCase):

    def setUp(self):
        pass

    def testBinAdjoint(self):
        # test <w | B*v> = <Badj * w | v>

        nx = 512
        ny = 412

        nbin = 2

        w = np.random.rand(ny // nbin, nx // nbin)
        v = np.random.rand(ny, nx)

        # sum mode
        Bv = camera.bin(v, [nbin, nbin], mode="sum")
        Badj_w = camera.bin_adjoint(w, [nbin, nbin], mode="sum")

        prod1 = np.sum(w.conj() * Bv)
        prod2 = np.sum(Badj_w.conj() * v)

        self.assertAlmostEqual(prod1, prod2, 9)

        # mean mode
        Bv_mean = camera.bin(v, [nbin, nbin], mode="mean")
        Badj_w_mean = camera.bin_adjoint(w, [nbin, nbin], mode="mean")

        prod1_mean = np.sum(w.conj() * Bv_mean)
        prod2_mean = np.sum(Badj_w_mean.conj() * v)

        self.assertAlmostEqual(prod1_mean, prod2_mean, 9)


if __name__ == "__main__":
    unittest.main()
