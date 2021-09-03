import unittest

import numpy as np
from numpy import fft
import fit_psf

class Test_psf(unittest.TestCase):

    def setUp(self):
        pass

    def test_otf2psf(self):
        """
        Test otf2psf() by verifying that the ideal circular aperture otf obtained with circ_aperture_otf() produces
        the correct psf, obtained by airy_fn()
        :return:
        """
        na = 1.3
        wavelength = 0.465
        dx = 0.065
        nx = 101
        dy = dx
        ny = nx

        fxs = fft.fftshift(fft.fftfreq(nx, dx))
        fys = fft.fftshift(fft.fftfreq(ny, dy))
        dfx = fxs[1] - fxs[0]
        dfy = fys[1] - fys[0]

        otf = fit_psf.circ_aperture_otf(np.expand_dims(fxs, axis=0), np.expand_dims(fys, axis=1), na, wavelength)
        psf, (ys, xs) = fit_psf.otf2psf(otf, (dfy, dfx))
        psf = psf / psf.max()

        xb, yb, zb = np.broadcast_arrays(np.expand_dims(xs, axis=(0, 1)),
                                  np.expand_dims(ys, axis=(0, 2)),
                                  np.expand_dims(np.array([0]), axis=(1, 2)))
        psf_true = fit_psf.born_wolf_psf(xb, yb, zb, [1, 0, 0, 0, na, 0], wavelength, 1)

        self.assertAlmostEqual(np.max(np.abs(psf - psf_true)), 0, 4)


if __name__ == "__main__":
    unittest.main()