"""
Tests for affine.py
"""
import unittest
import numpy as np
from scipy import fft
import scipy.signal
import scipy.optimize
import affine

def get_phase_realspace(img, frq, dxy, phase_guess=0):
    """
    Determine phase of pattern with a given frequency. Matches +cos(2*pi* f.*r + phi), where the origin
    is taken to be in the center of the image, or more precisely using the same coordinates as the fft
    assumes.

    To obtain the correct phase, it is necessary to have a very good frq_guess of the frequency.
    However, obtaining accurate relative phases is much less demanding.

    :param img: 2D array, must be positive
    :param frq: [fx, fy]. Should be frequency (not angular frequency).
    :param dxy: pixel size (um)
    :param phase_guess: optional guess for phase

    :return phase_fit: fitted value for the phase
    """
    if np.any(img < 0):
        raise ValueError('img must be strictly positive.')

    # assume origin is at the edge
    x = np.arange(img.shape[1]) * dxy
    y = np.arange(img.shape[0]) * dxy

    xx, yy = np.meshgrid(x, y)

    def fn(phi): return -np.cos(2 * np.pi * (frq[0] * xx + frq[1] * yy) + phi)
    def fn_deriv(phi): return np.sin(2 * np.pi * (frq[0] * xx + frq[1] * yy) + phi)
    def min_fn(phi): return np.sum(fn(phi) * img)
    def jac_fn(phi): return np.asarray([np.sum(fn_deriv(phi) * img),])

    # using jacobian makes faster and more robust
    result = scipy.optimize.minimize(min_fn, phase_guess, jac=jac_fn)
    phi_fit = np.mod(result.x, 2 * np.pi)

    return phi_fit


class Test_affine(unittest.TestCase):

    def setUp(self):
        pass

    def test_xform_sinusoid_params(self):
        """
        test the xform_sinusoid_params() function by constructing sinusoid pattern and passing through an affine
        transformation. Compare the resulting frequency determined numerically with the resulting frequency determined
        from the initial frequency + affine parameters
        :return:
        """

        # define object space parameters
        fobj = np.array([0.08333333, 0.08333333])
        phase_obj = 5.497787143782089
        fn = lambda x, y: 1 + np.cos(2 * np.pi * (fobj[0] * x + fobj[1] * y) + phase_obj)

        # define affine transform
        xform = affine.params2xform([1.4296003114502853, 2.3693263411981396, 2671.39109,
                                     1.4270495211450602, 2.3144621088632635, 790.402632])

        # sinusoid parameter transformation
        fxi, fyi, phase_img = affine.xform_sinusoid_params(fobj[0], fobj[1], phase_obj, xform)
        fimg = np.array([fxi, fyi])

        # compared with phase from fitting image directly
        out_coords = np.meshgrid(range(2048), range(2048))
        img = affine.xform_fn(fn, xform)(*out_coords)

        phase_fit = float(get_phase_realspace(img, fimg, 1))

        # todo: could also test frequencies if wanted...

        self.assertAlmostEqual(phase_img, phase_fit, 5)

    def test_xform_phase_translation(self):
        """
        Test function xform_phase_translation() function by defining sinusoid image and then translating. Compare numerically
        determined phase with that given by xform_phase_translation().
        :return:
        """
        fobj = np.array([0.08333333, 0.08333333])
        phase_obj = 5.497787143782089
        fn = lambda x, y: 1 + np.cos(2 * np.pi * (fobj[0] * x + fobj[1] * y) + phase_obj)

        # center of "new" coordinates in "old" coordinates
        # xn = xo - cx
        # yn = yo - cy
        cx = -100.62362
        cy = 0.3743743
        phase_xlated = affine.xform_phase_translation(fobj[0], fobj[1], phase_obj, [cx, cy])

        # fn of new coordinates
        # xo = xn + cx
        fn_xlated = lambda xn, yn: fn(xn + cx, yn + cy)

        x_new, y_new = np.meshgrid(range(500), range(500))
        img_new = fn_xlated(x_new, y_new)

        phase_xlated_test = float(get_phase_realspace(img_new, fobj, 1))

        self.assertAlmostEqual(phase_xlated, phase_xlated_test, 3)

    def test_xform_phase_roi(self):
        """
        Test function xform_phase_translation() function by defining sinusoid image and then cropping. Compare numerically
        determined phase with that given by xform_phase_translation().
        :return:
        """
        fobj = np.array([0.08333333, 0.08333333])
        phase_obj = 5.497787143782089
        fn = lambda x, y: 1 + np.cos(2 * np.pi * (fobj[0] * x + fobj[1] * y) + phase_obj)
        xo, yo = np.meshgrid(range(500), range(500))
        img = fn(xo, yo)

        # get ROI phase from function
        # center of "new" coordinates in "old" coordinates
        # xn = xo - cx
        # yn = yo - cy
        roi = [30, 450, 100, 210]
        phase_roi = affine.xform_phase_translation(fobj[0], fobj[1], phase_obj, [roi[2], roi[0]])

        # determine ROI phase from fitting
        img_roi = img[roi[0]:roi[1], roi[2]:roi[3]]
        phase_roi_test = float(get_phase_realspace(img_roi, fobj, 1))

        self.assertAlmostEqual(phase_roi, phase_roi_test, 8)

    def test_xform_sinusoid_params_roi(self):
        """
        Test function xform_sinusoid_params_roi() by constructing sinusoid pattern and passing through an affine
        transformation. Compare the resulting frequency determined numerically with the resulting frequency determined
        from the initial frequency + affine parameters
        :return:
        """
        # define object space parameters
        # roi_img = [0, 2048, 0, 2048]
        roi_img = [512, 788, 390, 871]

        fobj = np.array([0.08333333, 0.08333333])
        phase_obj = 5.497787143782089
        fn = lambda x, y: 1 + np.cos(2 * np.pi * (fobj[0] * x + fobj[1] * y) + phase_obj)

        # define affine transform
        xform = affine.params2xform([1.4296003114502853, 2.3693263411981396, 2671.39109,
                                     1.4270495211450602, 2.3144621088632635, 790.402632])

        # sinusoid parameter transformation
        fxi, fyi, phase_roi = affine.xform_sinusoid_params_roi(fobj[0], fobj[1], phase_obj, None, roi_img, xform,
                                                               input_origin="edge", output_origin="edge")
        fimg = np.array([fxi, fyi])

        # FFT phase
        _, _, phase_roi_ft = affine.xform_sinusoid_params_roi(fobj[0], fobj[1], phase_obj, None, roi_img, xform,
                                                              input_origin="edge", output_origin="fft")

        # compared with phase from fitting image directly
        out_coords = np.meshgrid(range(roi_img[2], roi_img[3]), range(roi_img[0], roi_img[1]))
        # img = affine.xform_fn(fn, xform, out_coords)
        img = affine.xform_fn(fn, xform)(*out_coords)
        phase_fit_roi = float(get_phase_realspace(img, fimg, 1, phase_guess=phase_roi))

        # phase FFT
        ny, nx = img.shape
        window = np.expand_dims(scipy.signal.windows.hann(nx), axis=0) * \
                 np.expand_dims(scipy.signal.windows.hann(ny), axis=1)
        img_ft = fft.fftshift(fft.fft2(fft.ifftshift(img * window)))
        fx = fft.fftshift(fft.fftfreq(nx, 1))
        fy = fft.fftshift(fft.fftfreq(ny, 1))

        xind = np.argmin(np.abs(fx - fimg[0]))
        yind = np.argmin(np.abs(fy - fimg[1]))
        peak = np.mean(img_ft[yind-1:yind+2, xind-1:xind+2])
        phase_fit_roi_ft = np.mod(np.angle(peak), 2 * np.pi)

        # peak2 = tools.get_peak_value(img_ft, fx, fy, fimg, 2)
        # phase_fit_roi_ft2 = np.mod(np.angle(peak), 2*np.pi)

        # accuracy is limited by frequency fitting routine...
        self.assertAlmostEqual(phase_roi, phase_fit_roi, 1)
        # probably limited by peak height finding routine
        self.assertAlmostEqual(phase_roi_ft, phase_fit_roi_ft, 2)

if __name__ == "__main__":
    unittest.main()
