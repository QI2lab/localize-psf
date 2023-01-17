"""
Test camera binning function
"""

import unittest
import numpy as np
from localize_psf import camera

class Test_psf(unittest.TestCase):

    def setUp(self):
        pass

    def testBinAdjoint(self):
        # test <w | B*v> = <Badj * w | v>

        nx = 512
        ny = 412

        nbin = 2

        w = np.random.rand(ny // nbin, nx // nbin)
        v = np.random.rand(ny, nx)

        Bv = camera.bin(v, [nbin, nbin], mode="sum")
        Badj_w = camera.bin_adjoint(w, [nbin, nbin], mode="sum")

        prod1 = np.sum(w.conj() * Bv)
        prod2 = np.sum(Badj_w.conj() * v)

        self.assertAlmostEqual(prod1, prod2, 9)