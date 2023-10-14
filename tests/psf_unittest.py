import unittest
import numpy as np
from numpy.testing import assert_allclose
from numpy import fft
from localize_psf import fit_psf
from localize_psf.camera import bin

class Test_psf(unittest.TestCase):

    def setUp(self):
        pass

    @unittest.skip("need to correct for recent code changes")
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


    def test_oversample_voxel(self):
        dxy = 0.065
        nxy = 2*3*5*7

        for nbin in [2, 3, 5, 7]:
            yyb, xxb = fit_psf.get_psf_coords((nxy // nbin, nxy // nbin),
                                              (dxy * nbin, dxy * nbin),
                                              broadcast=True)
            yy, xx = fit_psf.oversample_voxel((yyb, xxb),
                                              (dxy * nbin, dxy * nbin),
                                              sf=nbin,
                                              expand_along_extra_dim=False)

            dxs = xx[0, 1:] - xx[0, :-1]
            dys = yy[1:, 0] - yy[:-1, 0]

            assert_allclose(dxs, dxs[0], atol=1e-12)
            assert_allclose(dys, dys[0], atol=1e-12)
            assert_allclose(xxb, bin(xx, bin_sizes=(nbin, nbin), mode="mean"), atol=1e-12)
            assert_allclose(yyb, bin(yy, bin_sizes=(nbin, nbin), mode="mean"), atol=1e-12)

if __name__ == "__main__":
    unittest.main()