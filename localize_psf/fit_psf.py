"""
Tools for working with point-spread functions and optical transfer functions.

Functions for estimating PSF's from images of fluorescent beads (z-stacks or single planes). Useful for generating
experimental PSF's from the average of many beads and fitting 2D and 3D PSF models to beads.
"""
import warnings
import numpy as np
from scipy.ndimage import shift
from scipy.special import j0, j1
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve
from scipy import fft
from localize_psf import affine, rois

# most of the functions don't require this module, and it does not easily pip install,
# so don't require it. Probably should enforce some reasonable behavior on the functions
# that require it...
# https://pypi.org/project/psfmodels/
try:
    import psfmodels as psfm
    psfmodels_available = True
except ImportError:
    psfmodels_available = False


def blur_img_otf(ground_truth: np.ndarray,
                 otf: np.ndarray,
                 apodization: np.ndarray = 1):
    """
    Blur image with OTF

    :param ground_truth:
    :param otf: optical transfer function evalated at the FFT frequencies (with f=0 near the center of the array)
    :return img_blurred:
    """
    gt_ft = fft.fftshift(fft.fftn(fft.ifftshift(ground_truth)))
    img_blurred = fft.fftshift(fft.ifftn(fft.ifftshift(gt_ft * otf * apodization)))

    return img_blurred


def blur_img_psf(ground_truth: np.ndarray,
                 psf: np.ndarray,
                 apodization: np.ndarray = 1):
    """
    Blur image with PSF

    :param ground_truth:
    :param psf: point-spread function. this array should be centered at ny//2, nx//2
    :return blurred_img:
    """
    # todo: allow PSF of different size than image
    otf, _ = psf2otf(psf)

    return blur_img_otf(ground_truth, otf, apodization)


# tools for converting between different otf/psf representations
def otf2psf(otf: np.ndarray,
            dfs: list[float] = 1):
    """
    Compute the point-spread function from the optical transfer function
    :param otf: otf, as a 1D, 2D or 3D array. Assumes that f=0 is near the center of the array, and frequency are
    arranged by the FFT convention
    :param dfs: (dfz, dfy, dfx), (dfy, dfx), or (dfx). If only a single number is provided, will assume these are the
    same
    :return psf, coords: where coords = (z, y, x)
    """

    if isinstance(dfs, (int, float)) and otf.ndim > 1:
        dfs = [dfs] * otf.ndim

    if len(dfs) != otf.ndim:
        raise ValueError("dfs length must be otf.ndim")

    shape = otf.shape
    drs = np.array([1 / (df * n) for df, n in zip(shape, dfs)])
    coords = [fft.fftshift(fft.fftfreq(n, 1 / (dr * n))) for n, dr in zip(shape, drs)]

    psf = fft.fftshift(fft.ifftn(fft.ifftshift(otf))).real

    return psf, coords


def psf2otf(psf: np.ndarray,
            drs: list[float] = 1):
    """
    Compute the optical transfer function from the point-spread function

    :param psf: psf, as a 1D, 2D or 3D array. Assumes that r=0 is near the center of the array, and positions
    are arranged by the FFT convention
    :param drs: (dz, dy, dx), (dy, dx), or (dx). If only a single number is provided, will assume these are the
    same
    :return otf, coords: where coords = (fz, fy, fx)
    """

    if isinstance(drs, (int, float)) and psf.ndim > 1:
        drs = [drs] * psf.ndim

    if len(drs) != psf.ndim:
        raise ValueError("drs length must be psf.ndim")

    shape = psf.shape
    coords = [fft.fftshift(fft.fftfreq(n, dr)) for n, dr in zip(shape, drs)]

    otf = fft.fftshift(fft.fftn(fft.ifftshift(psf)))

    return otf, coords


def symm_fn_1d_to_2d(arr,
                     fs,
                     fmax: float,
                     npts: int):
    """
    Convert a 1D function which is symmetric wrt to the radial variable to a 2D matrix.
    Useful helper function when computing PSFs from 2D OTFs
    :param arr:
    :param fs:
    :param fmax:
    :param npts:
    :return arr_out, fxs, fys:
    """

    ny = 2 * npts + 1
    nx = 2 * npts + 1
    # fmax = fs.max()
    dx = 1 / (2 * fmax)
    dy = dx

    not_nan = np.logical_not(np.isnan(arr))

    fxs = fft.fftshift(fft.fftfreq(nx, dx))
    fys = fft.fftshift(fft.fftfreq(ny, dy))
    fmag = np.sqrt(fxs[None, :]**2 + fys[:, None]**2)
    to_interp = np.logical_and(fmag >= fs[not_nan].min(), fmag <= fs[not_nan].max())

    arr_out = np.zeros((ny, nx), dtype=arr.dtype)
    arr_out[to_interp] = interp1d(fs[not_nan], arr[not_nan])(fmag[to_interp])

    return arr_out, fxs, fys


def atf2otf(atf,
            dx: float = None,
            wavelength: float = 0.5,
            ni: float = 1.5,
            defocus_um: float = 0,
            fx=None,
            fy=None):
    """
    Get incoherent transfer function (OTF) from autocorrelation of coherent transfer function (ATF)

    :param atf:
    :param dx:
    :param wavelength:
    :param ni:
    :param defocus_um:
    :param fx:
    :param fy:
    :return otf, atf_defocus:
    """
    ny, nx = atf.shape

    if defocus_um != 0:
        if fx is None:
            fx = fft.fftshift(fft.fftfreq(nx, dx))
        if fy is None:
            fy = fft.fftshift(fft.fftfreq(ny, dx))

        if dx is None or wavelength is None or ni is None:
            raise TypeError("if defocus != 0, dx, wavelength, ni must be provided")

        k = 2*np.pi / wavelength * ni
        kperp = np.sqrt(np.array(k**2 - (2 * np.pi)**2 * (fx[None, :]**2 + fy[:, None]**2), dtype=np.complex))
        defocus_fn = np.exp(1j * defocus_um * kperp)
    else:
        defocus_fn = 1

    atf_defocus = atf * defocus_fn
    # if even number of frequencies, we must translate otf_c by one so that f and -f match up
    otf_c_minus_conj = np.roll(np.roll(np.flip(atf_defocus, axis=(0, 1)), np.mod(ny + 1, 2), axis=0),
                               np.mod(nx + 1, 2), axis=1).conj()

    otf = fftconvolve(atf_defocus, otf_c_minus_conj, mode='same') / np.sum(np.abs(atf) ** 2)
    return otf, atf_defocus


# circular aperture functions
def circ_aperture_atf(fx,
                      fy,
                      na: float,
                      wavelength: float):
    """
    Amplitude transfer function for circular aperture

    @param fx:
    @param fy:
    @param na:
    @param wavelength:
    @return atf:
    """
    fmax = 0.5 / (0.5 * wavelength / na)

    # ff = np.sqrt(fx[None, :]**2 + fy[:, None]**2)
    ff = np.sqrt(fx**2 + fy**2)

    atf = np.ones(ff.shape)
    atf[ff > fmax] = 0

    return atf


def circ_aperture_otf(fx,
                      fy,
                      na: float,
                      wavelength: float):
    """
    OTF for roi_size circular aperture

    :param fx:
    :param fy:
    :param na: numerical aperture
    :param wavelength: in um
    :return otf:
    """
    # maximum frequency imaging system can pass
    fmax = 1 / (0.5 * wavelength / na)

    # freq data
    fx = np.asarray(fx)
    fy = np.asarray(fy)
    ff = np.asarray(np.sqrt(fx**2 + fy**2))

    with np.errstate(invalid='ignore'):
        # compute otf
        otf = np.asarray(2 / np.pi * (np.arccos(ff / fmax) - (ff / fmax) * np.sqrt(1 - (ff / fmax)**2)))
        otf[ff > fmax] = 0

    return otf


# helper functions for converting between NA and peak widths
def na2fwhm(na: float,
            wavelength: float):
    """
    Convert numerical aperture to full-width at half-maximum, assuming an Airy-function PSF

    FWHM ~ 0.51 * wavelength / na

    :param na: numerical aperture
    :param wavelength:
    :return fwhm: in same units as wavelength
    """
    fwhm = 1.6163399561827614 / np.pi * wavelength / na
    return fwhm


def fwhm2na(wavelength: float,
            fwhm: float):
    """
    Convert full-width half-maximum PSF value to the equivalent numerical aperture. Inverse function of na2fwhm

    @param wavelength:
    @param fwhm:
    @return na:
    """
    na = 1.6163399561827614 / np.pi * wavelength / fwhm
    return na


def na2sxy(na: float,
           wavelength: float):
    """
    Convert numerical aperture to standard deviation, assuming the numerical aperture and the sigma
    are related as in the Airy function PSF

    :param na:
    :param wavelength:
    :return sigma:
    """
    fwhm = na2fwhm(na, wavelength)
    sigma = 1.49686886 / 1.6163399561827614 / 2 * fwhm
    # 2 * sqrt{2*log(2)} * sigma = 0.5 * wavelength / NA
    # sigma = na2fwhm(na, wavelength) / (2*np.sqrt(2 * np.log(2)))
    return sigma


def sxy2na(wavelength: float,
           sigma_xy: float):
    """
    Convert sigma xy value to equivalent numerical aperture, assuming these are related as in the Airy function PSF
    @param wavelength:
    @param sigma_xy:
    @return fwhm:
    """
    fwhm = 2 * 1.6163399561827614 / 1.49686886 * sigma_xy
    # fwhm = na2fwhm(na, wavelength)
    # fwhm = sigma * (2*np.sqrt(2 * np.log(2)))
    return fwhm2na(wavelength, fwhm)


def na2sz(na: float,
          wavelength: float,
          ni: float):
    """
    Convert numerical aperture to equivalent sigma-z value,

    @param na: numerical aperture
    @param wavelength:
    @param ni: index of refraction
    @return sz:
    """
    # todo: believe this is a gaussian approx. Find reference
    return np.sqrt(6) / np.pi * ni * wavelength / na ** 2


def sz2na(sigma_z: float,
          wavelength: float,
          ni: float):
    """
    Convert sigma-z value to equivalent numerical aperture

    todo: believe this is a gaussian approx. Find reference
    @param wavelength:
    @param sigma_z:
    @param ni: index of refraction
    @ return na:
    """
    return np.sqrt(np.sqrt(6) / np.pi * ni * wavelength / sigma_z)


# PSF models
class psf_model:
    """
    TODO: should I include a fit method in this class?
    """
    def __init__(self,
                 param_names: list[str],
                 dc: float = None,
                 sf: int = 1,
                 angles: tuple[float] = (0., 0., 0.),
                 has_jacobian: bool = False):
        """

        PSF functions, accounting for image pixelation along an arbitrary direction

        vectorized, i.e. can rely on obeying broadcasting rules for x,y,z

        @param param_names:
        @param dc: pixel size
        @param sf: factor to oversample pixels. The value of each pixel is determined by averaging sf**2 equally spaced
        @param angles: Euler angles describing orientation of pixel to resample
        @param has_jacobian:
        """



        if not isinstance(param_names, list):
            raise ValueError("param_names must be a list of strings")

        self.parameter_names = param_names
        self.nparams = len(param_names)

        if not isinstance(sf, int):
            raise ValueError("sf must be an integer")
        self.sf = sf

        if sf != 1 and dc is None:
            raise Exception("If sf != 1, then a value for dc must be provided")
        self.dc = dc

        self.angles = angles
        self.has_jacobian = has_jacobian



    def model(self,
              coords: tuple[np.ndarray],
              params: np.ndarray):
        pass

    def jacobian(self,
                 coords: tuple[np.ndarray],
                 params: np.ndarray):
        pass

    def test_jacobian(self,
                      coords: tuple[np.ndarray],
                      params: np.ndarray,
                      dp: float = 1e-7):
        # numerical test for jacobian
        jac_calc = self.jacobian(coords, params)
        jac_numerical = []
        for ii in range(self.nparams):
            dp_now = np.zeros(self.nparams)
            dp_now[ii] = dp * 0.5

            jac_numerical.append((self.model(coords, params + dp_now) - self.model(coords, params - dp_now)) / dp)

        return jac_numerical, jac_calc


    def estimate_parameters(self,
                            img: np.ndarray,
                            coords: tuple[np.ndarray]):
        pass

    def normalize_parameters(self,
                             params):
        """
        Return parameters in a standardized format, when there can be ambiguity. For example,
        a Gaussian model may include a standard deviation parameter. Since only the square of this quantity enters
        the model, a fit may return a negative value for standard deviation. In that case, this function
        would return the absolute value of the standard deviation
        @param params:
        @return:
        """
        return params


class gaussian3d_psf_model(psf_model):
    """
    Gaussian approximation to PSF.

    Since a diffraction limited PSF does not trully have a Gaussian form, we must choose some metric measuring
    the difference between the real PSF and the Gaussian PSF. For example, minimizing the difference between
    the two using an L1 metric results in the estimate
    sigma_xy = 0.22 * lambda / NA.
    See https://doi.org/10.1364/AO.46.001819 for more details.

    TWee arrive at a similar estimate from equating the FWHM of the Gaussian and the airy function.
    FWHM = 2 * sqrt{2*log(2)} * sigma ~ 0.51 * wavelength / NA

    sigma_z = np.sqrt(6) / np.pi * ni * wavelength / NA ** 2
    """
    def __init__(self, dc: float = None, sf=1, angles=(0., 0., 0.)):
        super().__init__(["A", "cx", "cy", "cz", "sxy", "sz", "bg"],
                         dc=dc, sf=sf, angles=angles, has_jacobian=True)

    def model(self, coords: tuple[np.ndarray], p: np.ndarray):
        z, y, x, = coords
        # oversample points in pixel
        xx_s, yy_s, zz_s = oversample_pixel(x, y, z, self.dc, sf=self.sf, euler_angles=self.angles)

        # calculate psf at oversampled points
        psf_s = np.exp(-(xx_s - p[1]) ** 2 / 2 / p[4] ** 2
                       -(yy_s - p[2]) ** 2 / 2 / p[4] ** 2
                       -(zz_s - p[3]) ** 2 / 2 / p[5] ** 2
                       )

        # normalization is such that the predicted peak value of the PSF ignoring binning would be p[0]
        psf = p[0] * np.mean(psf_s, axis=-1) + p[-1]

        return psf

    def jacobian(self, coords: tuple[np.ndarray], p: np.ndarray):
        z, y, x = coords

        # oversample points in pixel
        xx_s, yy_s, zz_s = oversample_pixel(x, y, z, self.dc, sf=self.sf, euler_angles=self.angles)

        # use sxy * |sxy| instead of sxy**2 to enforce sxy > 0
        psf_s = np.exp(-(xx_s - p[1]) ** 2 / 2 / p[4]**2
                       -(yy_s - p[2]) ** 2 / 2 / p[4]**2
                       -(zz_s - p[3]) ** 2 / 2 / p[5]**2
                       )

        # normalization is such that the predicted peak value of the PSF ignoring binning would be p[0]
        # psf = p[0] * psf_sum + p[-1]

        bcast_shape = (x + y + z).shape
        # [A, cx, cy, cz, sxy, sz, bg]
        jac = [np.mean(psf_s, axis=-1),
               p[0] * np.mean(2 * (xx_s - p[1]) / 2 / p[4]**2 * psf_s, axis=-1),
               p[0] * np.mean(2 * (yy_s - p[2]) / 2 / p[4]**2 * psf_s, axis=-1),
               p[0] * np.mean(2 * (zz_s - p[3]) / 2 / p[5]**2 * psf_s, axis=-1),
               p[0] * np.mean((2 / p[4] ** 3 * (xx_s - p[1]) ** 2 / 2 +
                               2 / p[4] ** 3 * (yy_s - p[2]) ** 2 / 2) * psf_s, axis=-1),
               p[0] * np.mean( 2 / p[5] ** 3 * (zz_s - p[3]) ** 2 / 2 * psf_s, axis=-1),
               np.ones(bcast_shape)]

        return jac

    def estimate_parameters(self, img: np.ndarray, coords: tuple[np.ndarray]):
        z, y, x = coords

        # subtract smallest value so positive
        # img_temp = img - np.nanmean(img)
        # to_use = np.logical_and(np.logical_not(np.isnan(img_temp)), img_temp > 0)
        img_temp = img - np.nanmin(img)
        to_use = np.logical_and(np.logical_not(np.isnan(img_temp)), img_temp > 0)

        if img.ndim != len(coords):
            raise ValueError("len(coords) != img.ndim")

        # compute moments
        c1s = np.zeros(img.ndim)
        c2s = np.zeros(img.ndim)
        for ii in range(img.ndim):
            c1s[ii] = np.sum(img_temp[to_use] * coords[ii][to_use]) / np.sum(img_temp[to_use])
            c2s[ii] = np.sum(img_temp[to_use] * coords[ii][to_use] ** 2) / np.sum(img_temp[to_use])

        sigmas = np.sqrt(c2s - c1s ** 2)
        sxy = np.mean(sigmas[:2])
        sz = sigmas[2]

        guess_params = np.concatenate((#np.array([np.nanmax(img) - np.nanmean(img)]),
                                       np.array([np.nanmax(img) - np.nanmin(img)]), # may be too large ... but typically have problems with being too small
                                       np.flip(c1s),
                                       np.array([sxy]),
                                       np.array([sz]),
                                       np.array([np.nanmean(img)])
                                       ),
                                      )

        return self.normalize_parameters(guess_params)

    def normalize_parameters(self, params):
        norm_params = np.array([params[0], params[1], params[2], params[3],
                                np.abs(params[4]), np.abs(params[5]),
                                params[6]])

        return norm_params

class asymmetric_gaussian3d(psf_model):
    def __init__(self, dc: float = None, sf=1, angles=(0., 0., 0.)):
        super().__init__(["A", "cx", "cy", "cz", "sx", "sy/sx", "sz", "theta_xy", "bg"],
                         dc=dc, sf=sf, angles=angles, has_jacobian=True)

    def model(self, coords: tuple[np.ndarray], p: np.ndarray):
        z, y, x, = coords

        # oversample points in pixel
        xx_s, yy_s, zz_s = oversample_pixel(x, y, z, self.dc, sf=self.sf, euler_angles=self.angles)

        # rotated coordinates
        xx_rot = np.cos(p[7]) * (xx_s - p[1]) - np.sin(p[7]) * (yy_s - p[2])
        yy_rot = np.cos(p[7]) * (yy_s - p[2]) + np.sin(p[7]) * (xx_s - p[1])

        # calculate psf at oversampled points
        psf_s = np.exp(-xx_rot ** 2 / 2 / p[4] ** 2
                       -yy_rot ** 2 / 2 / (p[4] * p[5]) ** 2
                       -(zz_s - p[3]) ** 2 / 2 / p[6] ** 2)

        # normalization is such that the predicted peak value of the PSF ignoring binning would be p[0]
        psf = p[0] * np.mean(psf_s, axis=-1) + p[8]

        return psf

    def jacobian(self, coords: tuple[np.ndarray], p: np.ndarray):
        z, y, x, = coords
        # oversample points in pixel
        xx_s, yy_s, zz_s = oversample_pixel(x, y, z, self.dc, sf=self.sf, euler_angles=self.angles)

        # rotated coordinates
        xx_rot = np.cos(p[7]) * (xx_s - p[1]) - np.sin(p[7]) * (yy_s - p[2])
        yy_rot = np.cos(p[7]) * (yy_s - p[2]) + np.sin(p[7]) * (xx_s - p[1])

        dx_dphi = -np.sin(p[7]) * (xx_s - p[1]) - np.cos(p[7]) * (yy_s - p[2])
        dy_dphi = -np.sin(p[7]) * (yy_s - p[2]) + np.cos(p[7]) * (xx_s - p[1])

        # calculate psf at oversampled points
        psf_s = np.exp(-xx_rot ** 2 / 2 / p[4] ** 2
                       -yy_rot ** 2 / 2 / (p[4] * p[5]) ** 2
                       -(zz_s - p[3]) ** 2 / 2 / p[6] ** 2)

        dpsf_dcx = 2 * xx_rot * np.cos(p[7]) / 2 / p[4]**2 + \
                   2 * yy_rot * np.sin(p[7]) / 2 / (p[4] * p[5])**2

        dpsf_dcy = -2 * xx_rot * np.sin(p[7]) / 2 / p[4] ** 2 + \
                    2 * yy_rot * np.cos(p[7]) / 2 / (p[4] * p[5]) ** 2

        dpsf_dcz = 2 * (zz_s - p[3]) / 2 / p[6] ** 2

        dpsf_dsx = 2 / p[4] ** 3 * xx_rot ** 2 / 2 + \
                   2 / p[4] ** 3 * yy_rot ** 2 / 2 / p[5] ** 2

        dpsf_dsrat = 2 / p[5] ** 3 * yy_rot**2 / 2 / p[4] ** 2

        dpsf_dsz = 2 / p[6] ** 3 * (zz_s - p[3]) ** 2 / 2

        dpsf_dtheta = 2 * xx_rot / 2 / p[4] ** 2 * dx_dphi + \
                      2 * yy_rot / 2 / (p[4] * p[5]) ** 2 * dy_dphi

        bcast_shape = (x + y + z).shape
        jac = [np.mean(psf_s, axis=-1),  # A
               p[0] * np.mean(dpsf_dcx * psf_s, axis=-1),  # cx
               p[0] * np.mean(dpsf_dcy * psf_s, axis=-1),  # cy
               p[0] * np.mean(dpsf_dcz * psf_s, axis=-1),  # cz
               p[0] * np.mean(dpsf_dsx * psf_s, axis=-1),  # sx
               p[0] * np.mean(dpsf_dsrat * psf_s, axis=-1),  # sy/sx
               p[0] * np.mean(dpsf_dsz * psf_s, axis=-1),  # sz
               p[0] * np.mean(dpsf_dtheta * psf_s, axis=-1), # theta
               np.ones(bcast_shape)  # bg
               ]

        return jac

    def estimate_parameters(self, img: np.ndarray, coords: tuple[np.ndarray]):
        z, y, x = coords

        # subtract smallest value so positive
        # img_temp = img - np.nanmean(img)
        img_temp = img
        to_use = np.logical_and(np.logical_not(np.isnan(img_temp)), img_temp > 0)

        if img.ndim != len(coords):
            raise ValueError("len(coords) != img.ndim")

        # compute moments
        c1s = np.zeros(img.ndim)
        c2s = np.zeros(img.ndim)
        for ii in range(img.ndim):
            c1s[ii] = np.sum(img_temp[to_use] * coords[ii][to_use]) / np.sum(img_temp[to_use])
            c2s[ii] = np.sum(img_temp[to_use] * coords[ii][to_use] ** 2) / np.sum(img_temp[to_use])

        sigmas = np.sqrt(c2s - c1s ** 2)

        guess_params = np.concatenate((np.array([np.nanmax(img) - np.nanmean(img)]),
                                       np.flip(c1s),
                                       np.flip(sigmas),
                                       np.array([0.]),
                                       np.array([np.nanmean(img)])
                                       ),
                                      )

        return self.normalize_parameters(guess_params)

    def normalize_parameters(self, params):
        norm_params = np.array([params[0], params[1], params[2], params[3],
                                np.abs(params[4]), np.abs(params[5]), np.abs(params[6]),
                                params[7], params[8]])

        return norm_params

class gaussian2d_psf_model(psf_model):
    """
    Gaussian approximation to PSF. Matches well for equal peak intensity, but some deviations in area.
    See https://doi.org/10.1364/AO.46.001819 for more details.
    sigma_xy = 0.22 * lambda / NA.
    This comes from equating the FWHM of the Gaussian and the airy function.
    FWHM = 2 * sqrt{2*log(2)} * sigma ~ 0.51 * wavelength / NA
    """
    def __init__(self, dc: float = None, sf=1, angles=(0., 0., 0.)):
        super().__init__(["A", "cx", "cy", "sxy", "bg"],
                         dc=dc, sf=sf, angles=angles, has_jacobian=True)

        self.gauss3d = gaussian3d_psf_model(dc=dc, sf=sf, angles=angles)

    def model(self, coords: tuple[np.ndarray], params: np.ndarray):
        """

        @param coords: [amplitude, cx, cy, sxy, bg]
        @param params:
        @return:
        """
        y, x = coords
        bcast_shape = np.broadcast_shapes(y.shape, x.shape)
        z = np.zeros(bcast_shape)

        p3d = np.array([params[0], params[1], params[2], 0., params[3], 1., params[4]])

        return self.gauss3d.model((z, y, x), p3d)

    def jacobian(self, coords: tuple[np.ndarray], params: np.ndarray):
        y, x = coords
        bcast_shape = np.broadcast_shapes(y.shape, x.shape)
        z = np.zeros(bcast_shape)

        p3d = np.array([params[0], params[1], params[2], 0., params[3], 1., params[4]])

        j3d = self.gauss3d.jacobian((z, y, x), p3d)

        j2d = j3d[:3] + [j3d[4]] + [j3d[6]]

        return j2d

    def estimate_parameters(self, img: np.ndarray, coords: tuple[np.ndarray]):
        y, x = coords
        bcast_shape = np.broadcast_shapes(y.shape, x.shape)
        z = np.zeros(bcast_shape)

        p3d = self.gauss3d.estimate_parameters(img, (z, y, x))

        p2d = np.concatenate((p3d[:3],
                              np.array([p3d[4]]),
                              np.array([p3d[6]])
                              )
                             )
        return self.normalize_parameters(p2d)

    def normalize_parameters(self, params):
        norm_params = np.array([params[0], params[1], params[2], np.abs(params[3]), params[4]])
        return norm_params

class gaussian_lorentzian_psf_model(psf_model):
    """
    Gaussian-Lorentzian PSF model. One difficulty with the Gaussian PSF is the weight is not the same in every z-plane,
    as we expect it should be. The Gaussian-Lorentzian form remedies this, but the functional form
    is no longer separable
    """
    def __init__(self, dc=None, sf=1, angles=(0., 0., 0.)):
        super().__init__(["A", "cx", "cy", "cz", "sxy", "hwhm_z", "bg"],
                         dc=dc, sf=sf, angles=angles, has_jacobian=True)

    def model(self, coords, p):
        (z, y, x) = coords

        # oversample points in pixel
        xx_s, yy_s, zz_s = oversample_pixel(x, y, z, self.dc, sf=self.sf, euler_angles=self.angles)

        # calculate psf at oversampled points
        lor_factor = 1 + (zz_s - p[3]) ** 2 / p[5] ** 2
        psf_s = np.exp(- ((xx_s - p[1]) ** 2 + (yy_s - p[2]) ** 2) / (2 * p[4] ** 2 * lor_factor)) / lor_factor

        # normalization is such that the predicted peak value of the PSF ignoring binning would be p[0]
        psf = p[0] * np.mean(psf_s, axis=-1) + p[6]

        return psf

    def jacobian(self, coords, p):
        z, y, x = coords

        # oversample points in pixel
        xx_s, yy_s, zz_s = oversample_pixel(x, y, z, self.dc, sf=self.sf, euler_angles=self.angles)

        lor = 1 + (zz_s - p[3]) ** 2 / p[5] ** 2
        r_sqr = (xx_s - p[1]) ** 2 + (yy_s - p[2]) ** 2
        factor = np.exp(- r_sqr / (2 * p[4] ** 2 * lor)) / lor

        bcast_shape = (x + y + z).shape
        jac = [np.mean(factor, axis=-1),  # amp
               p[0] * np.mean((x - p[1]) / p[4] ** 2 / lor * factor, axis=-1),  # cx
               p[0] * np.mean((y - p[2]) / p[4] ** 2 / lor * factor, axis=-1),  # cy
               p[0] * np.mean(-(z - p[3]) / p[5] ** 2 / lor ** 2 * (r_sqr / 2 / p[4] ** 2) * factor, axis=-1) +
               p[0] * np.mean(factor / lor * 2 * (z - p[3]) / p[5] ** 2, axis=-1),  # cz
               p[0] * np.mean(factor * r_sqr / lor / p[4] ** 3, axis=-1),  # sxy
               p[0] * np.mean(factor * r_sqr / 2 / p[4] ** 2 / lor ** 2 * 2 * (z - p[3]) ** 2 / p[5] ** 3, axis=-1) +
               p[0] * np.mean(factor / lor * 2 * (z - p[3]) ** 2 / p[5] ** 3, axis=-1),  # hwhm_z
               np.ones(bcast_shape)  # bg
               ]

        return jac

    def estimate_parameters(self, img: np.ndarray, coords: tuple[np.ndarray]):
        z, y, x = coords

        # subtract smallest value so positive
        img_temp = img - np.nanmean(img)
        to_use = np.logical_and(np.logical_not(np.isnan(img_temp)), img_temp > 0)

        if img.ndim != len(coords):
            raise ValueError("len(coords) != img.ndim")

        # compute moments
        c1s = np.zeros(img.ndim)
        c2s = np.zeros(img.ndim)
        for ii in range(img.ndim):
            c1s[ii] = np.sum(img_temp[to_use] * coords[ii][to_use]) / np.sum(img_temp[to_use])
            c2s[ii] = np.sum(img_temp[to_use] * coords[ii][to_use] ** 2) / np.sum(img_temp[to_use])

        sigmas = np.sqrt(c2s - c1s ** 2)
        sxy = np.mean(sigmas[:2])
        sz = sigmas[2]

        guess_params = np.concatenate((np.array([np.nanmax(img) - np.nanmean(img)]),
                                       np.flip(c1s),
                                       np.array([sxy]),
                                       np.array([sz]),
                                       np.array([np.nanmean(img)])
                                       ),
                                      )

        return self.normalize_parameters(guess_params)

    def normalize_parameters(self, params):
        norm_params = np.array([params[0], params[1], params[2], params[3],
                                np.abs(params[4]), np.abs(params[5]), params[6]])
        return norm_params

class born_wolf_psf_model(psf_model):
    """
    Born-wolf PSF function evaluated using Airy function if in-focus, and axial function if along the axis.
    Otherwise evaluated using numerical integration.
    """
    def __init__(self, wavelength: float, ni: float, dc: float = None, sf=1, angles=(0., 0., 0.)):
        """

        @param wavelength:
        @param ni: refractive index
        @param dc:
        @param sf:
        @param angles:
        """

        # TODO is it better to put wavelength and ni as arguments to model or as class members?
        # advantage to being model parameters is could conceivable fit
        self.wavelength = wavelength
        self.ni = ni

        if sf != 1:
            raise NotImplementedError("Only implemented for sf=1")

        super().__init__(["A", "cx", "cy", "cz", "na", "bg"],
                         dc=dc, sf=sf, angles=angles, has_jacobian=False)

    def model(self, coords: tuple[np.ndarray], p: np.ndarray):
        """

        @param coords:
        @param p:
        @param wavelength:
        @param ni:
        @return:
        """
        z, y, x = coords

        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        z = np.atleast_1d(z)
        x, y, z = np.broadcast_arrays(x, y, z)

        k = 2 * np.pi / self.wavelength
        rr = np.sqrt((x - p[1]) ** 2 + (y - p[2]) ** 2)

        psfs = np.zeros(rr.shape) * np.nan
        is_in_focus = (z == p[3])
        is_on_axis = (rr == 0)

        # ################################
        # evaluate in-focus portion using airy function, which is much faster than integrating
        # ################################
        def airy_fn(rho):
            val = p[0] * 4 * np.abs(j1(rho * k * p[4]) / (rho * k * p[4])) ** 2 + p[5]
            val[rho == 0] = p[0] + p[4]
            return val

        with np.errstate(invalid="ignore"):
            psfs[is_in_focus] = airy_fn(rr[is_in_focus])

        # ################################
        # evaluate on axis portion using exact expression
        # ################################
        def axial_fn(z):
            val = p[0] * 4 * (2 * self.ni ** 2) / (k ** 2 * p[4] ** 4 * (z - p[3]) ** 2) * \
                  (1 - np.cos(0.5 * k * (z - p[3]) * p[4] ** 2 / self.ni)) + p[5]
            val[z == p[3]] = p[0] + p[5]
            return val

        with np.errstate(invalid="ignore"):
            psfs[is_on_axis] = axial_fn(z[is_on_axis])

        # ################################
        # evaluate out of focus portion using integral
        # ################################
        if not np.all(is_in_focus):

            def integrand(rho, r, z):
                return rho * j0(k * r * p[4] * rho) * np.exp(-1j * k * (z - p[3]) * p[4] ** 2 * rho ** 2 / (2 * self.ni))

            # like this approach because allows rr, z, etc. to have arbitrary dimension
            already_evaluated = np.logical_or(is_in_focus, is_in_focus)
            for ii, (r, zc, already_eval) in enumerate(zip(rr.ravel(), z.ravel(), already_evaluated.ravel())):
                if already_eval:
                    continue

                int_real = quad(lambda rho: integrand(rho, r, zc).real, 0, 1)[0]
                int_img = quad(lambda rho: integrand(rho, r, zc).imag, 0, 1)[0]

                coords = np.unravel_index(ii, rr.shape)
                psfs[coords] = p[0] * 4 * (int_real ** 2 + int_img ** 2) + p[5]

        return psfs


    def estimate_parameters(self, img: np.ndarray, coords: tuple[np.ndarray]):

        gauss3d = gaussian3d_psf_model(dc=self.dc, sf=self.sf, angles=self.angles)
        p3d_gauss = gauss3d.estimate_parameters(img, coords)

        na = sxy2na(self.wavelength, p3d_gauss[4])

        params_guess = np.concatenate((p3d_gauss[:4],
                                       np.array([na]),
                                       np.array([p3d_gauss[-1]])
                                       )
                                      )

        return params_guess


class model(psf_model):
    """
    Wrapper function for evaluating different PSF models where only the coordinate grid information is given
    (i.e. nx and dxy) and not the actual coordinates. The real coordinates can be obtained using get_psf_coords().

    This class primarily exists to wrap the psfmodel functions, but

    The size of these functions is parameterized by the numerical aperture, instead of the sigma or other size
    parameters that are only convenient for some specific models of the PSF

    For vectorial or gibson-lanni PSF's, this wraps the functions in the psfmodels package
    (https://pypi.org/project/psfmodels/) with added ability to shift the center of these functions away from the
    center of the ROI, which is useful when fitting them.

    For 'gaussian', it wraps the gaussian3d_pixelated_psf() function. More details about the relationship between
    the Gaussian sigma and the numerical aperture can be found here: https://doi.org/10.1364/AO.46.001819
    """
    def __init__(self, wavelength, ni, model_name="vectorial", dc: float = None, sf=1, angles=(0., 0., 0.)):
        """

        @param wavelength:
        @param ni: index of refraction
        @param model_name: 'gaussian', 'gibson-lanni', 'born-wolf', or 'vectorial'. 'gibson-lanni' relies on the
        psdmodels function scalar_psf(), while 'vectorial' relies on the psfmodels function vectorial_psf()
        @param dc:
        @param sf:
        @param angles:
        """
        self.wavelength = wavelength
        self.ni = ni

        allowed_models = ["gibson-lanni", "vectorial", "born-wolf", "gaussian"]
        if model_name not in allowed_models:
            raise ValueError(f"model={model_name:s} was not an allowed value. Allowed values are {allowed_models}")

        if not psfmodels_available and (model_name == "vectorial" or model_name == "gibson-lanni"):
            raise NotImplementedError(f"model={model_name:s} selected but psfmodels is not installed")
        self.model_name = model_name

        if sf != 1:
            raise NotImplementedError("not yet implemented for sf=/=1")

        super().__init__(["A", "cx", "cy", "cz", "na", "bg"],
                         dc=dc, sf=sf, angles=angles, has_jacobian=False)

    def model(self, coords: tuple[np.ndarray], p: np.ndarray, **kwargs):
        """
        Unlike other model functions this ONLY works if coords are of the same style as obtained from
        get_psf_coords()

        @param coords: (z, y, x). Coordinates must be exactly as obtained from get_psf_coords() with
        nx=ny and dx=dy.
        @param p:
        @param kwargs: keywords passed through to vectorial_psf() or scalar_psf(). Note that in most cases
        these will shift the best focus PSF away from z=0
        @return:
        """
        zz, y, x = coords
        z = zz[:, 0, 0]

        dxy = x[0, 0, 1] - x[0, 0, 0]
        nx = x.shape[2]

        if 'NA' in kwargs.keys():
            raise ValueError("'NA' is not allowed to be passed as a named parameter. It is specified in p.")

        if self.model_name == 'vectorial':
            model_params = {'NA': p[4], 'sf': self.sf, 'ni': self.ni, 'ni0': self.ni}
            model_params.update(kwargs)

            psf_norm = psfm.vectorial_psf(0, 1, dxy, wvl=self.wavelength, params=model_params, normalize=False)
            val = psfm.vectorial_psf(z - p[3], nx, dxy, wvl=self.wavelength, params=model_params, normalize=False)
            val = p[0] / psf_norm * shift(val, [0, p[2] / dxy, p[1] / dxy], mode='nearest') + p[5]

        elif self.model_name == 'gibson-lanni':
            model_params = {'NA': p[4], 'sf': self.sf, 'ni': self.ni, 'ni0': self.ni}
            model_params.update(kwargs)

            psf_norm = psfm.scalar_psf(0, 1, dxy, wvl=self.wavelength, params=model_params, normalize=False)
            val = psfm.scalar_psf(z - p[3], nx, dxy, wvl=self.wavelength, params=model_params, normalize=False)
            val = p[0] / psf_norm * shift(val, [0, p[2] / dxy, p[1] / dxy], mode='nearest') + p[5]

        elif self.model_name == "born-wolf":
            bw_model = born_wolf_psf_model(wavelength=self.wavelength, ni=self.ni, dc=self.dc, sf=self.sf, angles=self.angles)
            val = bw_model.model(coords, p)

        elif self.model_name == 'gaussian':
            # transform NA to sigmas
            p_gauss = [p[0], p[1], p[2], p[3], 0.22 * self.wavelength / p[4],
                       np.sqrt(6) / np.pi * self.ni * self.wavelength / p[4] ** 2,
                       p[5]]

            gauss_model = gaussian3d_psf_model(dc=self.dc, sf=self.sf, angles=self.angles)

            # normalize so that amplitude parameter is actually amplitude
            psf_norm = gauss_model.model((p[3], p[2], p[1]), p_gauss) - p[5]
            val = p[0] / psf_norm * (gauss_model.model((z, y, x), p_gauss) - p[5]) + p[5]
        else:
            raise ValueError(f"model_name was '{self.model_name:s}',"
                             f" but must be 'vectorial', 'gibson-lanni', 'born-wolf', or 'gaussian'")


        return val

    def estimate_parameters(self, img: np.ndarray, coords: tuple[np.ndarray]):

        gauss3d = gaussian3d_psf_model(dc=self.dc, sf=self.sf, angles=self.angles)
        p3d_gauss = gauss3d.estimate_parameters(img, coords)

        na = sxy2na(self.wavelength, p3d_gauss[4])

        params_guess = np.concatenate((p3d_gauss[:4],
                                       np.array([na]),
                                       np.array([p3d_gauss[-1]])
                                       )
                                      )

        return params_guess


# utility functions
def oversample_pixel(x: np.ndarray,
                     y: np.ndarray,
                     z: np.ndarray,
                     ds: float,
                     sf: int,
                     euler_angles: tuple[float] = (0., 0., 0.)):
    """
    Generate coordinates to oversample a pixel on a 2D grid.

    Suppose we have a set of pixels centered at points given by x, y, z. Generate sf**2 points in this pixel equally
    spaced about the center. Allow the pixel to be orientated in an arbitrary direction with respect to the coordinate
    system. The pixel rotation is described by the Euler angles (psi, theta, phi), where the pixel "body" frame
    is a square with xy axis orientated along the legs of the square with z-normal to the square

    Compare with oversample_voxel() which works on a 3D grid with a fixed orientation. oversample_pixel() works
    on a 2D grid with an arbitrary orientation.

    :param x: x-coordinate with shape such that can be broadcast with y and z. e.g. z.shape = [nz, 1, 1];
     y.shape = [1, ny, 1]; x.shape = [1, 1, nx]
    :param y:
    :param z:
    :param ds: pixel size
    :param sf: sample factor
    :param euler_angles: [phi, theta, psi] where phi and theta are the polar angles describing the normal of the pixel,
    and psi describes the rotation of the pixel about its normal
    :return xx_s, yy_s, zz_s:

    """
    # generate new points in pixel, each of which is centered about an equal area of the pixel, so summing them is
    # giving an approximation of the integral
    if sf > 1:
        pts = np.arange(1 / (2*sf), 1 - 1 / (2*sf), 1 / sf) - 0.5
        xp, yp = np.meshgrid(ds * pts, ds * pts)
        zp = np.zeros(xp.shape)

        # rotate points to correct position using normal vector
        # for now we will fix x, but lose generality
        mat = affine.euler_mat(*euler_angles)
        result = mat.dot(np.concatenate((xp.ravel()[None, :],
                                         yp.ravel()[None, :],
                                         zp.ravel()[None, :]), axis=0))
        xs, ys, zs = result

        # now must add these to each point x, y, z
        xx_s = x[..., None] + xs[None, ...]
        yy_s = y[..., None] + ys[None, ...]
        zz_s = z[..., None] + zs[None, ...]
    else:
        xx_s = np.expand_dims(x, axis=-1)
        yy_s = np.expand_dims(y, axis=-1)
        zz_s = np.expand_dims(z, axis=-1)

    return xx_s, yy_s, zz_s


def oversample_voxel(coords: tuple[np.ndarray],
                     drs: tuple[float],
                     sf: int = 3):
    """
    Get coordinates to oversample a voxel on a 3D grid

    Compare with oversample_pixel(), which performs oversampling on a 2D grid.

    :param coords: tuple of coordinates, e.g. (z, y, x)
    :param drs: tuple giving voxel size (dz, dy, dx)
    :param sf: sampling factor. Assumed to be same for all directions
    :return coords_upsample: tuple of coordinates, e.g. (z_os, y_os, x_os). e.g. x_os has one more dimension than x
    with  this extra dimension giving the oversampled coordinates
    """
    pts = np.arange(1 / (2 * sf), 1 - 1 / (2 * sf), 1 / sf) - 0.5
    pts_dims = np.meshgrid(*[pts * dr for dr in drs], indexing="ij")

    coords_upsample = [np.expand_dims(c, axis=-1) + np.expand_dims(np.ravel(r), axis=0)
                       for c, r in zip(coords, pts_dims)]
    # now must add these to each point x, y, z
    return coords_upsample


def get_psf_coords(ns: list[int],
                   drs: list[float],
                   broadast: bool = False):
    """
    Get centered coordinates for PSFmodels style PSF's from step size and number of coordinates
    :param ns: list of number of points
    :param drs: list of step sizes
    :return coords: list of coordinates [zs, ys, xs, ...]
    """
    ndims = len(drs)
    coords = [np.expand_dims(d * (np.arange(n) - (n // 2)),
                             axis=tuple(range(ii)) + tuple(range(ii+1, ndims)))
              for ii, (n, d) in enumerate(zip(ns, drs))]

    if broadast:
        # return arrays instead of views coords
        coords = [np.array(c, copy=True) for c in np.broadcast_arrays(*coords)]

    return coords


def average_exp_psfs(imgs: np.ndarray,
                     coords: tuple[np.ndarray],
                     centers: np.ndarray,
                     roi_sizes: tuple[int],
                     backgrounds=None):
    """
    Get experimental psf from imgs by averaging many localizations (after pixel shifting).

    :param imgs:z-stack of images
    :param coords: (z, y, x) of full image. Must be broadcastable to full image size
    :param centers: n x 3 array, (cz, cy, cx)
    :param roi_sizes: [sz, sy, sx]
    :param backgrounds: values to subtracted from each ROI

    :return psf_mean, psf_coords, otf_mean, otf_coords:
    """

    # if np.any(np.mod(np.array(roi_sizes), 2) == 0):
    #     raise ValueError("roi_sizes must be odd")

    z, y, x, = coords
    dz = z[1, 0, 0] - z[0, 0, 0]
    dy = y[0, 1, 0] - y[0, 0, 0]
    dx = x[0, 0, 1] - x[0, 0, 0]

    # set up array to hold psfs
    nrois = len(centers)
    if backgrounds is None:
        backgrounds = np.zeros(nrois)

    psf_shifted = np.zeros((nrois, roi_sizes[0], roi_sizes[1], roi_sizes[2])) * np.nan
    # coordinates
    z_psf, y_psf, x_psf = get_psf_coords(roi_sizes, [dz, dy, dx], broadast=True)

    zc_pix_psf = np.argmin(np.abs(z_psf[:, 0, 0]))
    yc_pix_psf = np.argmin(np.abs(y_psf[0, :, 0]))
    xc_pix_psf = np.argmin(np.abs(x_psf[0, 0, :]))

    # loop over rois and shift psfs so they are centered
    for ii in range(nrois):
        # get closest pixels to center
        xc_pix = np.argmin(np.abs(x - centers[ii, 2]))
        yc_pix = np.argmin(np.abs(y - centers[ii, 1]))
        zc_pix = np.argmin(np.abs(z - centers[ii, 0]))

        # cut roi from image
        roi_unc = rois.get_centered_roi((zc_pix, yc_pix, xc_pix), roi_sizes)
        roi = rois.get_centered_roi((zc_pix, yc_pix, xc_pix),
                                    roi_sizes,
                                    min_vals=[0, 0, 0],
                                    max_vals=imgs.shape)
        img_roi = rois.cut_roi(roi, imgs)

        zroi = rois.cut_roi(roi, z)
        yroi = rois.cut_roi(roi, y)
        xroi = rois.cut_roi(roi, x)

        cx_pix_roi = (roi[5] - roi[4]) // 2
        cy_pix_roi = (roi[3] - roi[2]) // 2
        cz_pix_roi = (roi[1] - roi[0]) // 2

        xshift_pix = (xroi[0, 0, cx_pix_roi] - centers[ii, 2]) / dx
        yshift_pix = (yroi[0, cy_pix_roi, 0] - centers[ii, 1]) / dy
        zshift_pix = (zroi[cz_pix_roi, 0, 0] - centers[ii, 0]) / dz

        # get coordinates
        img_roi_shifted = shift(np.array(img_roi, dtype=float), [zshift_pix, yshift_pix, xshift_pix],
                                    mode="constant", cval=-1)
        img_roi_shifted[img_roi_shifted == -1] = np.nan

        # put into array in appropriate positions
        zstart = zc_pix_psf - cz_pix_roi
        zend = zstart + (roi[1] - roi[0])
        ystart = yc_pix_psf - cy_pix_roi
        yend = ystart + (roi[3] - roi[2])
        xstart = xc_pix_psf - cx_pix_roi
        xend = xstart + (roi[5] - roi[4])

        psf_shifted[ii, zstart:zend, ystart:yend, xstart:xend] = img_roi_shifted - backgrounds[ii]


    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        with np.errstate(divide='ignore', invalid='ignore'):
            psf_mean = np.nanmean(psf_shifted, axis=0)

    # the above doesn't do a good enough job of normalizing PSF
    max_val = np.nanmax(psf_mean[psf_mean.shape[0]//2])
    psf_mean = psf_mean / max_val

    # get otf
    otf_mean, ks = psf2otf(psf_mean, drs=(dz, dy, dx))
    kz, ky, kx = np.meshgrid(*ks, indexing="ij")

    return psf_mean, (z_psf, y_psf, x_psf), otf_mean, (kz, ky, kx)
