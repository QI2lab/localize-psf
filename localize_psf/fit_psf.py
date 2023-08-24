"""
Tools for working with point-spread functions and optical transfer functions.

Functions for estimating PSF's from images of fluorescent beads (z-stacks or single planes). Useful for generating
experimental PSF's from the average of many beads and fitting 2D and 3D PSF models to beads.
"""
from typing import Union, Optional
import warnings
import numpy as np
from scipy.ndimage import shift
from scipy.special import j0, j1
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.signal import fftconvolve, convolve
from scipy import fft
from localize_psf import affine, rois, fit

# most of the functions don't require this module, and it does not easily pip install,
# so don't require it. Probably should enforce some reasonable behavior on the functions
# that require it...
# https://pypi.org/project/psfmodels/
_psfmodels_available = True
try:
    import psfmodels as psfm
except ImportError:
    _psfmodels_available = False

_cupy_available = True
try:
    import cupy as cp
    from cupyx.scipy.signal import convolve as convolve_gpu
except ImportError:
    cp = np
    _cupy_available = False

array = Union[np.ndarray, cp.ndarray]


def blur_img_otf(ground_truth: array,
                 otf: array,
                 apodization: Optional[array] = None) -> array:
    """
    Blur image with OTF. OTF must be on a grid with coordinates obtained from
    fx = fft.fftshift(fft.fftfrq(nx, d=dx))

    :param ground_truth: NumPy or CuPy array. If CuPy array operations will be performed on the GPU
    :param otf: optical transfer function evalated at the FFT frequencies (with f=0 near the center of the array)
    :param apodization:
    :return img_blurred:
    """

    if isinstance(ground_truth, cp.ndarray):
        xp = cp
    else:
        xp = np

    if apodization is None:
        apodization = 1.

    otf = xp.asarray(otf)
    apodization = xp.asarray(apodization)

    gt_ft = xp.fft.fftshift(xp.fft.fftn(xp.fft.ifftshift(ground_truth)))
    img_blurred = xp.fft.fftshift(xp.fft.ifftn(xp.fft.ifftshift(gt_ft * otf * apodization)))

    return img_blurred


def blur_img_psf(ground_truth: array,
                 psf: array,
                 apodization: Optional[array] = None) -> array:
    """
    Blur image with PSF. For odd-sized PSF's, there is no ambiguity about the convolution process. For even PSF's,
    we have to adopt some convention. We choose the PSF coordinate grid to be x = (arange(nx) - nx // 2) * dx.
    Which is appropriate for directly blurring using an OTF.

    Note that this grid is different than what one would adopt if using scipy.signal(mode="same"). In that case
    the PSF should be shifted one pixel up and to the left wrt to the convention used here

    :param ground_truth:
    :param psf: point-spread function, must have same number of dimensions as ground_truth, but possibly
      different size. This array should be centered at coordinate (ny//2, nx//2).
    :return blurred_img:
    """

    if isinstance(ground_truth, cp.ndarray) and _cupy_available:
        xp = cp
        conv = convolve_gpu
    else:
        xp = np
        conv = convolve

    psf = xp.asarray(psf)

    # since GPU convolve only implemented for 1D arrays, need to ensure psf correct size
    if xp == cp and _cupy_available:
        ns_after = [(n - m) // 2 for n, m in zip(ground_truth.shape, psf.shape)]
        ns_before = [na + 1 if na != 0 else 0 for na in ns_after]
        pad_sizes = [(nb, na) for nb, na in zip(ns_before, ns_after)]

        psf = xp.pad(psf,
                     pad_sizes,
                     mode="constant",
                     constant_values=0)

        if psf.shape != ground_truth.shape:
            raise ValueError()

    if psf.shape == ground_truth.shape:
        otf, _ = psf2otf(psf)
        img_blurred = blur_img_otf(ground_truth, otf, apodization=apodization)
    else:
        # todo: problem, GPU version only implemented for 1D arrays
        ns = ground_truth.shape
        ms = psf.shape
        slices = tuple([slice(m//2, m//2 + n) for m, n in zip(ms, ns)])
        img_blurred = conv(ground_truth, psf, mode="full")[slices]

    return img_blurred


# tools for converting between different otf/psf representations
def otf2psf(otf: array,
            dfs: list[float] = 1,
            apodization: Optional[array] = None) -> (array, list[array]):
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

    if apodization is None:
        apodization = 1.

    if isinstance(otf, cp.ndarray):
        xp = cp
    else:
        xp = np

    apodization = xp.asarray(apodization)

    shape = otf.shape
    drs = np.array([1 / (df * n) for df, n in zip(shape, dfs)])
    coords = [xp.fft.fftshift(xp.fft.fftfreq(n, 1 / (dr * n))) for n, dr in zip(shape, drs)]

    psf = xp.fft.fftshift(xp.fft.ifftn(xp.fft.ifftshift(otf * apodization))).real

    return psf, coords


def psf2otf(psf: array,
            drs: list[float] = 1,
            apodization: Optional[array] = None) -> (array, list[array]):
    """
    Compute the optical transfer function from the point-spread function

    :param psf: psf, as a 1D, 2D or 3D array. Assumes that r=0 is near the center of the array, and positions
      are arranged by the FFT convention
    :param drs: (dz, dy, dx), (dy, dx), or (dx). If only a single number is provided, will assume these are the same
    :return otf, coords: where coords = (fz, fy, fx)
    """

    if isinstance(drs, (int, float)) and psf.ndim > 1:
        drs = [drs] * psf.ndim

    if len(drs) != psf.ndim:
        raise ValueError("drs length must be psf.ndim")

    if isinstance(psf, cp.ndarray):
        xp = cp
    else:
        xp = np

    if apodization is None:
        apodization = 1.

    apodization = xp.asarray(apodization)

    shape = psf.shape
    coords = [xp.fft.fftshift(xp.fft.fftfreq(n, dr)) for n, dr in zip(shape, drs)]

    otf = xp.fft.fftshift(xp.fft.fftn(xp.fft.ifftshift(psf * apodization)))

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


def atf2otf(atf: np.ndarray,
            dx: Optional[float] = None,
            wavelength: float = 0.5,
            ni: float = 1.5,
            defocus_um: float = 0,
            fx: Optional[np.ndarray] = None,
            fy: Optional[np.ndarray] = None):
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
def circ_aperture_atf(fx: array,
                      fy: array,
                      na: float,
                      wavelength: float) -> array:
    """
    Amplitude transfer function for circular aperture

    :param fx:
    :param fy:
    :param na:
    :param wavelength:
    :return atf:
    """

    if isinstance(fx, cp.ndarray):
        xp = cp
    else:
        xp = np
    fy = xp.ndarray(fy)

    fmax = 0.5 / (0.5 * wavelength / na)

    # ff = np.sqrt(fx[None, :]**2 + fy[:, None]**2)
    ff = xp.sqrt(fx**2 + fy**2)

    atf = xp.ones(ff.shape)
    atf[ff > fmax] = 0

    return atf


def circ_aperture_otf(fx: array,
                      fy: array,
                      na: float,
                      wavelength: float) -> array:
    """
    OTF for roi_size circular aperture

    :param fx: numpy or cupy ndarray
    :param fy:
    :param na: numerical aperture
    :param wavelength: in um
    :return otf: numpy or cupy ndarray
    """
    # maximum frequency imaging system can pass
    fmax = 2 * na / wavelength

    if isinstance(fx, cp.ndarray) and _cupy_available:
        xp = cp
    else:
        xp = np

    # freq data
    fx = xp.asarray(fx)
    fy = xp.asarray(fy)
    ff = xp.sqrt(fx**2 + fy**2)

    # compute otf
    otf = xp.zeros(ff.shape)
    to_use = ff <= fmax
    with np.errstate(invalid='ignore'):
        ff_use = ff[to_use]
        otf[to_use] = 2 / np.pi * (xp.arccos(ff_use / fmax) - (ff_use / fmax) * xp.sqrt(1 - (ff_use / fmax)**2))

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

    :param wavelength:
    :param fwhm:
    :return na:
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

    :param wavelength:
    :param sigma_xy:
    :return fwhm:
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

    :param na: numerical aperture
    :param wavelength:
    :param ni: index of refraction
    :return sz:
    """
    # todo: believe this is a gaussian approx. Find reference
    return np.sqrt(6) / np.pi * ni * wavelength / na ** 2


def sz2na(sigma_z: float,
          wavelength: float,
          ni: float):
    """
    Convert sigma-z value to equivalent numerical aperture

    todo: believe this is a gaussian approx. Find reference
    :param wavelength:
    :param sigma_z:
    :param ni: index of refraction
    : return na:
    """
    return np.sqrt(np.sqrt(6) / np.pi * ni * wavelength / sigma_z)


# PSF models
class pixelated_psf_model(fit.coordinate_model):

    def __init__(self,
                 param_names: list[str],
                 dc: Optional[float] = None,
                 sf: int = 1,
                 angles: tuple[float] = (0., 0., 0.),
                 has_jacobian: bool = False,
                 ndims: int = 3
                 ):
        """
        PSF functions, accounting for image pixelation along an arbitrary direction.
        vectorized, i.e. can rely on obeying broadcasting rules for x,y,z
        # todo: want any easy way to create pixelated model from a fit.coordinate_model

        :param param_names:
        :param dc: pixel size
        :param sf: factor to oversample pixels. The value of each pixel is determined by averaging sf**2 equally spaced
        :param angles: Euler angles describing orientation of pixel to resample
        :param has_jacobian:
        :param ndims: specifies the number of spatial dimensions for this model. coordinates should always be a
          tuple with length ndims
        """

        super().__init__(param_names, ndims, has_jacobian=has_jacobian)

        if not isinstance(sf, int):
            raise ValueError("sf must be an integer")
        self.sf = sf

        if sf != 1 and dc is None:
            raise Exception("If sf != 1, then the pixel size, dc, must not be None")
        self.dc = dc

        self.angles = angles


class from_coordinate_model(pixelated_psf_model):
    """
    Helper class to convert any preexisting 3D coordinate model to a pixelated model
    """
    def __init__(self,
                 model: fit.coordinate_model,
                 dc: Optional[float] = None,
                 sf: int = 1,
                 angles: tuple[float] = (0., 0., 0.)
                 ):

        param_names = model.parameter_names
        has_jacobian = model.has_jacobian
        ndims = model.ndim

        if ndims != 3:
            raise ValueError(f"pixel oversampling only implemented for 3D models,"
                             f" but provided moel has ndims={ndims:d}")

        super().__init__(param_names,
                         dc=dc,
                         sf=sf,
                         angles=angles,
                         has_jacobian=has_jacobian,
                         ndims=ndims)
        self.coord_model = model

        # copy any attributes that don't overlap
        for k, v in model.__dict__.items():
            if not hasattr(self, k):
                setattr(self, k, v)

    def model(self,
              coordinates: tuple[np.ndarray],
              parameters: np.ndarray) -> np.ndarray:

        z, y, x, = coordinates
        # oversample points in pixel
        xx_s, yy_s, zz_s = oversample_pixel(x, y, z, self.dc, sf=self.sf, euler_angles=self.angles)

        # calculate psf at oversampled points
        psf_s = self.coord_model.model((zz_s, yy_s, xx_s), parameters)
        # average over those points
        psf = np.mean(psf_s, axis=-1)

        return psf


    def jacobian(self,
                 coordinates: tuple[np.ndarray],
                 parameters: np.ndarray) -> list[np.ndarray]:

        z, y, x = coordinates

        # oversample points in pixel
        xx_s, yy_s, zz_s = oversample_pixel(x, y, z, self.dc, sf=self.sf, euler_angles=self.angles)

        jac_os = self.coord_model.jacobian((zz_s, yy_s, xx_s), parameters)

        jac = [np.mean(j, axis=-1) for j in jac_os]

        return jac


    def estimate_parameters(self,
                            data: np.ndarray,
                            coordinates: tuple[np.ndarray],
                            num_preserved_dims: int = 0):
        return self.coord_model.estimate_parameters(data, coordinates, num_preserved_dims)


    def estimate_bounds(self,
                        coordinates: tuple[np.ndarray]) -> (tuple[float], tuple[float]):
        return self.coord_model.estimate_bounds(coordinates)


    def normalize_parameters(self,
                             parameters) -> np.ndarray:
        return self.coord_model.normalize_parameters(parameters)


class gaussian3d_psf_model(from_coordinate_model):
    """
    Gaussian approximation to PSF.

    Since a diffraction limited PSF does not truly have a Gaussian form, we must choose some metric measuring
    the difference between the real PSF and the Gaussian PSF. For example, minimizing the difference between
    the two using an L1 metric results in the estimate
    sigma_xy = 0.22 * lambda / NA.
    See https://doi.org/10.1364/AO.46.001819 for more details.

    We arrive at a similar estimate from equating the FWHM of the Gaussian and the airy function.
    FWHM = 2 * sqrt{2*log(2)} * sigma ~ 0.51 * wavelength / NA

    sigma_z = np.sqrt(6) / np.pi * ni * wavelength / NA ** 2
    """
    def __init__(self,
                 dc: Optional[float] = None,
                 sf: int = 1,
                 angles: tuple[float] = (0., 0., 0.),
                 minimum_sigmas: tuple[float] = (0., 0.)
                 ):

        super().__init__(fit.gauss3d(minimum_sigmas=minimum_sigmas), dc=dc, sf=sf, angles=angles)


class gaussian3d_asymmetric_pixelated(from_coordinate_model):
    """
    3D gaussian with equal sigma_x and sigma_y
    """
    def __init__(self,
                 dc: Optional[float] = None,
                 sf: int = 1,
                 angles: tuple[float] = (0., 0., 0.),
                 minimum_sigmas: tuple[float] = (0., 0., 0.)
                 ):

        super().__init__(fit.gauss3d_asymmetric(minimum_sigmas=minimum_sigmas), dc=dc, sf=sf, angles=angles)


class gaussian3d_rotated_pixelated(from_coordinate_model):
    """
    3D gaussian with equal sigma_x and sigma_y rotated by an arbitrary angle
    """
    def __init__(self,
                 dc: Optional[float] = None,
                 sf: int = 1,
                 angles: tuple[float] = (0., 0., 0.),
                 minimum_sigmas: tuple[float] = (0., 0.)
                 ):

        gauss3d = fit.gauss3d(minimum_sigmas=minimum_sigmas)
        gauss3d_rotated = fit.rotated_model_3d(gauss3d, (3, 2, 1))

        super().__init__(gauss3d_rotated, dc=dc, sf=sf, angles=angles)


class gaussian3d_asymmetric_rotated_pixelated(from_coordinate_model):
    """
    3D gaussian with arbitrary sigma_x, sigma_y, sigma_z rotated by an arbitrary angle
    """

    def __init__(self,
                 dc: Optional[float] = None,
                 sf: int = 1,
                 angles: tuple[float] = (0., 0., 0.),
                 minimum_sigmas: tuple[float] = (0., 0., 0.)
                 ):
        model_rotated = fit.rotated_model_3d(fit.gauss3d_asymmetric(minimum_sigmas=minimum_sigmas), (3, 2, 1))

        super().__init__(model_rotated, dc=dc, sf=sf, angles=angles)


class gaussian2d_psf_model(pixelated_psf_model):
    """
    Gaussian approximation to PSF. Matches well for equal peak intensity, but some deviations in area.
    See https://doi.org/10.1364/AO.46.001819 for more details.
    sigma_xy = 0.22 * lambda / NA.
    This comes from equating the FWHM of the Gaussian and the airy function.
    FWHM = 2 * sqrt{2*log(2)} * sigma ~ 0.51 * wavelength / NA
    """
    def __init__(self,
                 dc: Optional[float] = None,
                 sf: int = 1,
                 angles: tuple[float] = (0., 0., 0.)):
        super().__init__(["A", "cx", "cy", "sxy", "bg"],
                         dc=dc, sf=sf, angles=angles, has_jacobian=True, ndims=2)

        self.gauss3d = gaussian3d_psf_model(dc=dc, sf=sf, angles=angles)

    def model(self, coords: tuple[np.ndarray], params: np.ndarray):
        """

        :param coords: [amplitude, cx, cy, sxy, bg]
        :param params:
        :return:
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

    def estimate_parameters(self,
                            img: np.ndarray,
                            coords: tuple[np.ndarray],
                            num_preserved_dims: int = 0):


        if num_preserved_dims != 0:
            raise NotImplementedError()

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
        norm_params = np.array(params, copy=True)
        norm_params[..., 3] = np.abs(norm_params[..., 3])
        return norm_params


class gaussian_lorentzian_psf_model(pixelated_psf_model):
    """
    Gaussian-Lorentzian PSF model. One difficulty with the Gaussian PSF is the weight is not the same in every z-plane,
    as we expect it should be. The Gaussian-Lorentzian form remedies this, but the functional form
    is no longer separable
    """
    def __init__(self, dc=None, sf=1, angles=(0., 0., 0.)):
        super().__init__(["A", "cx", "cy", "cz", "sxy", "hwhm_z", "bg"],
                         dc=dc, sf=sf, angles=angles, has_jacobian=True, ndims=3)

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

    def estimate_parameters(self,
                            img: np.ndarray,
                            coords: tuple[np.ndarray],
                            num_preserved_dims: int = 0):

        if num_preserved_dims != 0:
            raise NotImplementedError()

        z, y, x = coords

        # subtract smallest value so positive
        img_temp = img - np.nanmean(img)
        to_use = np.logical_and(np.logical_not(np.isnan(img_temp)), img_temp > 0)

        if self.ndim != len(coords):
            raise ValueError("len(coords) != self.ndim")

        # compute moments
        c1s = np.zeros(self.ndim)
        c2s = np.zeros(self.ndim)
        isum = np.sum(img_temp[to_use])
        for ii in range(self.ndim):
            c1s[ii] = np.sum((img_temp * coords[ii])[to_use]) / isum
            c2s[ii] = np.sum((img_temp * coords[ii]**2)[to_use]) / isum

        sigmas = np.sqrt(c2s - c1s ** 2)
        sz = sigmas[0]
        sxy = np.mean(sigmas[1:])

        guess_params = np.concatenate((np.array([np.nanmax(img) - np.nanmean(img)]),
                                       np.flip(c1s),
                                       np.array([sxy]),
                                       np.array([sz]),
                                       np.array([np.nanmean(img)])
                                       ),
                                      )

        return self.normalize_parameters(guess_params)

    def normalize_parameters(self, params):
        norm_params = np.array(params, copy=True)
        norm_params[..., 4] = np.abs(norm_params[..., 4])
        norm_params[..., 5] = np.abs(norm_params[..., 5])
        return norm_params


class born_wolf_psf_model(pixelated_psf_model):
    """
    Born-wolf PSF function evaluated using Airy function if in-focus, and axial function if along the axis.
    Otherwise evaluated using numerical integration.
    """
    def __init__(self,
                 wavelength: float,
                 ni: float,
                 dc: Optional[float] = None,
                 sf: int = 1,
                 angles: tuple[float] = (0., 0., 0.)):
        """

        :param wavelength:
        :param ni: refractive index
        :param dc:
        :param sf:
        :param angles:
        """

        # TODO is it better to put wavelength and ni as arguments to model or as class members?
        # advantage to being model parameters is could conceivable fit
        self.wavelength = wavelength
        self.ni = ni

        if sf != 1:
            raise NotImplementedError("Only implemented for sf=1")

        super().__init__(["A", "cx", "cy", "cz", "na", "bg"],
                         dc=dc, sf=sf, angles=angles, has_jacobian=False, ndims=3)

    def model(self, coords: tuple[np.ndarray], p: np.ndarray):
        """

        :param coords:
        :param p:
        :param wavelength:
        :param ni:
        :return:
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


    def estimate_parameters(self,
                            img: np.ndarray,
                            coords: tuple[np.ndarray],
                            num_preserved_dims: int = 0):

        if num_preserved_dims != 0:
            raise NotImplementedError()

        gauss3d = gaussian3d_psf_model(dc=self.dc, sf=self.sf, angles=self.angles)
        p3d_gauss = gauss3d.estimate_parameters(img, coords)

        na = sxy2na(self.wavelength, p3d_gauss[4])

        params_guess = np.concatenate((p3d_gauss[:4],
                                       np.array([na]),
                                       np.array([p3d_gauss[-1]])
                                       )
                                      )

        return params_guess


class gridded_psf_model(pixelated_psf_model):
    """
    Wrapper function for evaluating different PSF models which are constrained to be on gridded coordinates. Therefore
    only grid parameters (i.e. nx and dxy) are provided and not the actual coordinates.
    The real coordinates can be obtained using get_psf_coords().

    This class primarily exists to wrap the psfmodel functions, but

    The size of these functions is parameterized by the numerical aperture, instead of the sigma or other size
    parameters that are only convenient for some specific models of the PSF

    For vectorial or gibson-lanni PSF's, this wraps the functions in the psfmodels package
    (https://pypi.org/project/psfmodels/) with added ability to shift the center of these functions away from the
    center of the ROI, which is useful when fitting them.

    For 'gaussian', it wraps the gaussian3d_pixelated_psf() function. More details about the relationship between
    the Gaussian sigma and the numerical aperture can be found here: https://doi.org/10.1364/AO.46.001819
    """
    def __init__(self,
                 wavelength: float,
                 ni: float,
                 model_name: str = "vectorial",
                 dc: Optional[float] = None,
                 sf: int = 1,
                 angles: tuple[float] = (0., 0., 0.)):
        """

        :param wavelength:
        :param ni: index of refraction
        :param model_name: 'gaussian', 'gibson-lanni', 'born-wolf', or 'vectorial'. 'gibson-lanni' relies on the
        psdmodels function scalar_psf(), while 'vectorial' relies on the psfmodels function vectorial_psf()
        :param dc:
        :param sf:
        :param angles:
        """
        self.wavelength = wavelength
        self.ni = ni

        allowed_models = ["gibson-lanni", "vectorial", "born-wolf", "gaussian"]
        if model_name not in allowed_models:
            raise ValueError(f"model={model_name:s} was not an allowed value. Allowed values are {allowed_models}")

        if not _psfmodels_available and (model_name == "vectorial" or model_name == "gibson-lanni"):
            raise NotImplementedError(f"model={model_name:s} selected but psfmodels is not installed")
        self.model_name = model_name

        if sf != 1:
            raise NotImplementedError("not yet implemented for sf=/=1")

        super().__init__(["A", "cx", "cy", "cz", "na", "bg"],
                         dc=dc, sf=sf, angles=angles, has_jacobian=False, ndims=3)

    def model(self,
              coords: tuple[np.ndarray],
              p: np.ndarray, **kwargs):
        """
        Unlike other model functions this ONLY works if coords are of the same style as obtained from
        get_psf_coords()

        :param coords: (z, y, x). Coordinates must be exactly as obtained from get_psf_coords() with nx=ny and dx=dy.
        :param p:
        :param kwargs: keywords passed through to vectorial_psf() or scalar_psf(). Note that in most cases
          these will shift the best focus PSF away from z=0
        :return:
        """
        zz, y, x = coords
        z = zz[:, 0, 0]

        dxy = x[0, 0, 1] - x[0, 0, 0]
        nz = zz.shape[0]
        ny = y.shape[1]
        nx = x.shape[2]

        if 'NA' in kwargs.keys():
            raise ValueError("'NA' is not allowed to be passed as a named parameter. It is specified in p.")

        model_params = {'NA': p[4], 'ni': self.ni, 'ni0': self.ni, "sf": self.sf}  # 'sf': self.sf
        model_params.update(kwargs)

        if self.model_name == 'vectorial':
            if nx != ny:
                raise ValueError("only nx=ny is supported")

            if ny % 2 == 0 and self.sf != 1:
                raise ValueError(f"psfmodel model has even sizes ({ny:d}, {nx:d}), but this is not compatible with sf != 1")

            psf_norm = psfm.vectorial_psf(0, 1, dxy, wvl=self.wavelength, params=model_params, normalize=False)
            val = psfm.vectorial_psf(z - p[3], nx, dxy, wvl=self.wavelength, params=model_params, normalize=False)

            # add 1 to correct centering, since PSFmodels naturally centered at (n-1)//2, but coordinates centered at n//2
            if ny % 2 == 0:
                correction = 1
            else:
                correction = 0
            val = p[0] / psf_norm * shift(val, [0, p[2] / dxy + correction, p[1] / dxy + correction], mode='nearest') + p[5]

        elif self.model_name == 'gibson-lanni':
            if nx != ny:
                raise ValueError("only nx=ny is supported")

            if ny % 2 == 0 and self.sf != 1:
                raise ValueError(
                    f"psfmodel model has even sizes ({ny:d}, {nx:d}), but this is not compatible with sf != 1")

            psf_norm = psfm.scalar_psf(0, 1, dxy, wvl=self.wavelength, params=model_params, normalize=False)
            val = psfm.scalar_psf(z - p[3], nx, dxy, wvl=self.wavelength, params=model_params, normalize=False)
            # add 1 to correct centering, since PSFmodels naturally centered at (n-1)//2, but coordinates centered at n//2
            if ny % 2 == 0:
                correction = 1
            else:
                correction = 0
            val = p[0] / psf_norm * shift(val, [0, p[2] / dxy + correction, p[1] / dxy + correction], mode='nearest') + p[5]

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
            val = p[0] / psf_norm * (gauss_model.model(coords, p_gauss) - p[5]) + p[5]
        else:
            raise ValueError(f"model_name was '{self.model_name:s}',"
                             f" but must be 'vectorial', 'gibson-lanni', 'born-wolf', or 'gaussian'")


        return val

    def estimate_parameters(self,
                            img: np.ndarray,
                            coords: tuple[np.ndarray],
                            num_preserved_dims: int = 0):

        if num_preserved_dims != 0:
            raise NotImplementedError()

        gauss3d = gaussian3d_psf_model(dc=self.dc, sf=self.sf, angles=self.angles)
        p3d_gauss = gauss3d.estimate_parameters(img, coords, num_preserved_dims)

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
        # pts = np.arange(1 / (2*sf), 1 - 1 / (2*sf), 1 / sf) - 0.5 # todo: this version only correct is sf odd
        pts = np.arange(1 / (2 * sf), 1, 1 / sf) - 0.5

        if len(pts) != sf:
            raise ValueError()

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
    # pts = np.arange(1 / (2 * sf), 1 - 1 / (2 * sf), 1 / sf) - 0.5
    pts = np.arange(1 / (2 * sf), 1, 1 / sf) - 0.5

    if len(pts) != sf:
        raise ValueError()

    pts_dims = np.meshgrid(*[pts * dr for dr in drs], indexing="ij")

    coords_upsample = [np.expand_dims(c, axis=-1) + np.expand_dims(np.ravel(r), axis=0)
                       for c, r in zip(coords, pts_dims)]
    # now must add these to each point x, y, z
    return coords_upsample


def get_psf_coords(ns: list[int],
                   drs: list[float],
                   broadcast: bool = False):
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

    if broadcast:
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

    :param imgs: z-stack of images
    :param coords: (z, y, x) of full image. Must be broadcastable to full image size
    :param centers: n x 3 array, (cz, cy, cx)
    :param roi_sizes: [sz, sy, sx] in pixels
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
    z_psf, y_psf, x_psf = get_psf_coords(roi_sizes, [dz, dy, dx], broadcast=True)

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
        roi_unc = rois.get_centered_rois((zc_pix, yc_pix, xc_pix), roi_sizes)
        roi = rois.get_centered_rois((zc_pix, yc_pix, xc_pix),
                                    roi_sizes,
                                    min_vals=[0, 0, 0],
                                    max_vals=imgs.shape)[0]
        img_roi = rois.cut_roi(roi, imgs)[0]

        zroi = rois.cut_roi(roi, z)[0]
        yroi = rois.cut_roi(roi, y)[0]
        xroi = rois.cut_roi(roi, x)[0]

        cx_pix_roi = (roi[5] - roi[4]) // 2
        cy_pix_roi = (roi[3] - roi[2]) // 2
        cz_pix_roi = (roi[1] - roi[0]) // 2

        xshift_pix = (xroi[0, 0, cx_pix_roi] - centers[ii, 2]) / dx
        yshift_pix = (yroi[0, cy_pix_roi, 0] - centers[ii, 1]) / dy
        zshift_pix = (zroi[cz_pix_roi, 0, 0] - centers[ii, 0]) / dz

        # get coordinates
        img_roi_shifted = shift(np.array(img_roi, dtype=float),
                                [zshift_pix, yshift_pix, xshift_pix],
                                mode="constant",
                                cval=-1)
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
