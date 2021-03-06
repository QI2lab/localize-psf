"""
Functions for estimating PSF's from images of fluorescent beads (z-stacks or single planes). Useful for generating
experimental PSF's from the average of many beads and fitting 2D and 3D PSF models to beads. Also includes tools for
working withpoint-spread functions and optical transfer functions more generally.
"""
import os
import copy
import numpy as np
import scipy.ndimage as ndi
import scipy.special as sp
import scipy.integrate
import scipy.interpolate
import scipy.signal
from scipy import fft
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from localize_psf import fit
from localize_psf import affine
from localize_psf import rois

# most of the functions don't require this module, and it does not easily pip install,
# so don't require it. Probably should enforce some reasonable behavior on the functions
# that require it...
# https://pypi.org/project/psfmodels/
try:
    import psfmodels as psfm
    psfmodels_available = True
except ImportError:
    psfmodels_available = False


def blur_img_otf(ground_truth, otf, apodization=1):
    """
    Blur image with OTF

    :param ground_truth:
    :param otf: optical transfer function evalated at the FFT frequencies (with f=0 near the center of the array)
    :return img_blurred:
    """
    gt_ft = fft.fftshift(fft.fftn(fft.ifftshift(ground_truth)))
    img_blurred = fft.fftshift(fft.ifftn(fft.ifftshift(gt_ft * otf * apodization)))

    return img_blurred


def blur_img_psf(ground_truth, psf, apodization):
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
def otf2psf(otf, dfs=1):
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


def psf2otf(psf, drs=1):
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


def symm_fn_1d_to_2d(arr, fs, fmax, npts):
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
    arr_out[to_interp] = scipy.interpolate.interp1d(fs[not_nan], arr[not_nan])(fmag[to_interp])

    return arr_out, fxs, fys


def atf2otf(atf, dx=None, wavelength=0.5, ni=1.5, defocus_um=0, fx=None, fy=None):
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

    otf = scipy.signal.fftconvolve(atf_defocus, otf_c_minus_conj, mode='same') / np.sum(np.abs(atf) ** 2)
    return otf, atf_defocus


# circular aperture functions
def circ_aperture_atf(fx, fy, na, wavelength):
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


def circ_aperture_otf(fx, fy, na, wavelength):
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
def na2fwhm(na, wavelength):
    """
    Convert numerical aperture to full-width at half-maximum, assuming an Airy-function PSF

    FWHM ~ 0.51 * wavelength / na

    :param na: numerical aperture
    :param wavelength:
    :return fwhm: in same units as wavelength
    """
    fwhm = 1.6163399561827614 / np.pi * wavelength / na
    return fwhm


def fwhm2na(wavelength, fwhm):
    """
    Convert full-width half-maximum PSF value to the equivalent numerical aperture. Inverse function of na2fwhm

    @param wavelength:
    @param fwhm:
    @return na:
    """
    na = 1.6163399561827614 / np.pi * wavelength / fwhm
    return na


def na2sxy(na, wavelength):
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


def sxy2na(wavelength, sigma_xy):
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


def na2sz(na, wavelength, ni):
    """
    Convert numerical aperture to equivalent sigma-z value,

    @param na: numerical aperture
    @param wavelength:
    @param ni: index of refraction
    @return sz:
    """
    # todo: believe this is a gaussian approx. Find reference
    return np.sqrt(6) / np.pi * ni * wavelength / na ** 2


def sz2na(sigma_z, wavelength, ni):
    """
    Convert sigma-z value to equivalent numerical aperture

    todo: believe this is a gaussian approx. Find reference
    @param wavelength:
    @param sigma_z:
    @param ni: index of refraction
    @ return na:
    """
    return np.sqrt(np.sqrt(6) / np.pi * ni * wavelength / sigma_z)


# different PSF model functions
def gaussian2d_psf(x, y, p, sf=1):
    """
    2D Gaussian approximation to airy function. Matches well for equal peak intensity, but then area will not match.
    :param x:
    :param y:
    :param p: [A, cx, cy, NA, bg]
    :param sf:
    :return value:
    """

    return gaussian3d_psf(x, y, np.array([0]), [p[0], p[1], p[2], 0, p[3], 1, p[4]], sf=sf, angles=(0., 0., 0.))


def gaussian3d_psf(x, y, z, dc, p, sf=1, angles=(0., 0., 0.)):
    """
    Gaussian function, accounting for image pixelation in the xy plane.

    vectorized, i.e. can rely on obeying broadcasting rules for x,y,z

    :param x:
    :param y:
    :param z: coordinates of z-planes to evaluate function at
    :param dc: pixel size
    :param p: [A, cx, cy, cz, sxy, sz, bg]
    :param sf: factor to oversample pixels. The value of each pixel is determined by averaging sf**2 equally spaced
    points in the pixel.
    :param angles: orientation of pixel to resample
    :return:
    """

    if not isinstance(sf, int):
        raise ValueError("sf must be an integer")

    # oversample points in pixel
    xx_s, yy_s, zz_s = oversample_pixel(x, y, z, dc, sf=sf, euler_angles=angles)

    # calculate psf at oversampled points
    psf_s = np.exp(-(xx_s - p[1]) ** 2 / 2 / p[4] ** 2
                   -(yy_s - p[2]) ** 2 / 2 / p[4] ** 2
                   -(zz_s - p[3]) ** 2 / 2 / p[5] ** 2)

    # normalization is such that the predicted peak value of the PSF ignoring binning would be p[0]
    psf = p[0] * np.mean(psf_s, axis=-1) + p[-1]

    return psf


def gaussian3d_psf_jac(x, y, z, dc, p, sf, angles=(0., 0., 0.)):
    """
    Jacobian of gaussian3d_pixelated_psf()

    :param x:
    :param y:
    :param z: coordinates of z-planes to evaluate function at
    :param dc: pixel size
    :param p: [A, cx, cy, cz, sxy, sz, bg]
    :param sf: factor to oversample pixels. The value of each pixel is determined by averaging sf**2 equally spaced
    points in the pixel.
    :param angles:
    :return jacobian:
    """

    # oversample points in pixel
    xx_s, yy_s, zz_s = oversample_pixel(x, y, z, dc, sf=sf, euler_angles=angles)

    psf_s = np.exp(-(xx_s - p[1]) ** 2 / 2 / p[4] ** 2
                   -(yy_s - p[2]) ** 2 / 2 / p[4] ** 2
                   -(zz_s - p[3]) ** 2 / 2 / p[5] ** 2)

    # normalization is such that the predicted peak value of the PSF ignoring binning would be p[0]
    # psf = p[0] * psf_sum + p[-1]

    bcast_shape = (x + y + z).shape
    # [A, cx, cy, cz, sxy, sz, bg]
    jac = [np.mean(psf_s, axis=-1),
           p[0] * np.mean(2 * (xx_s - p[1]) / 2 / p[4]**2 * psf_s, axis=-1),
           p[0] * np.mean(2 * (yy_s - p[2]) / 2 / p[4]**2 * psf_s, axis=-1),
           p[0] * np.mean(2 * (zz_s - p[3]) / 2/ p[5]**2 * psf_s, axis=-1),
           p[0] * np.mean((2 / p[4]**3 * (xx_s - p[1])**2 / 2 +
                           2 / p[4]**3 * (yy_s - p[2])**2 / 2) * psf_s, axis=-1),
           p[0] * np.mean(2 / p[5]**3 * (zz_s - p[3])**2 / 2 * psf_s, axis=-1),
           np.ones(bcast_shape)]

    return jac


def gaussian_lorentzian_psf(x, y, z, dc, p, sf=1, angles=(0., 0., 0.)):
    """
    Gaussian-Lorentzian PSF model. One difficulty with the Gaussian PSF is the weight is not the same in every z-plane,
    as we expect it should be. The Gaussian-Lorentzian form remedies this, but the functional form
    is no longer separable

    @param x:
    @param y:
    @param z:
    @param float dc:
    @param p: [amplitude, cx, cy, cz, sxy, hwhm_z, bg]
    @param int sf:
    @param angles:
    @return value:
    """

    if not isinstance(sf, int):
        raise ValueError("sf must be an integer")

    if sf != 1 and dc is None:
        raise Exception("If sf != 1, then a value for dc must be provided")

    # oversample points in pixel
    xx_s, yy_s, zz_s = oversample_pixel(x, y, z, dc, sf=sf, euler_angles=angles)

    # calculate psf at oversampled points
    lor_factor = 1 + (zz_s - p[3]) ** 2 / p[5] ** 2
    psf_s = np.exp(- ((xx_s - p[1]) ** 2 + (yy_s - p[2]) ** 2) / (2 * p[4] ** 2 * lor_factor)) / lor_factor

    # normalization is such that the predicted peak value of the PSF ignoring binning would be p[0]
    psf = p[0] * np.mean(psf_s, axis=-1) + p[6]

    return psf


def gaussian_lorentzian_psf_jac(x, y, z, dc, p, sf=1, angles=(0., 0., 0.)):
    """
    Get jacobian of gaussian_lorentzian_psf()
    @param x:
    @param y:
    @param z:
    @param dc:
    @param p:
    @param sf:
    @param angles:
    @return jacobian:
    """
    if not isinstance(sf, int):
        raise ValueError("sf must be an integer")

    if sf != 1 and dc is None:
        raise Exception("If sf != 1, then a value for dc must be provided")

    # oversample points in pixel
    xx_s, yy_s, zz_s = oversample_pixel(x, y, z, dc, sf=sf, euler_angles=angles)

    lor = 1 + (zz_s - p[3]) ** 2 / p[5] ** 2
    r_sqr = (xx_s - p[1]) ** 2 + (yy_s - p[2]) ** 2
    factor = np.exp(- r_sqr / (2 * p[4] ** 2 * lor)) / lor

    bcast_shape = (x + y + z).shape
    jac = [np.mean(factor, axis=-1),
           p[0] * np.mean((x - p[1]) / p[4]**2 / lor * factor, axis=-1),
           p[0] * np.mean((y - p[2]) / p[4]**2 / lor * factor, axis=-1),
           p[0] * np.mean(-(z - p[3]) / p[5]**2 / lor**2 * (r_sqr / 2 / p[4]**2) * factor, axis=-1) +
           p[0] * np.mean(factor / lor * 2 * (z - p[3]) / p[5]**2, axis=-1),
           p[0] * np.mean(factor * r_sqr / lor / p[4]**3, axis=-1),
           p[0] * np.mean(factor * r_sqr / 2 / p[4]**2 / lor**2 * 2 * (z - p[3])**2 / p[5]**3, axis=-1) +
           p[0] * np.mean(factor / lor * 2 * (z - p[3])**2 / p[5]**3, axis=-1),
           np.ones(bcast_shape)
           ]

    return jac


def born_wolf_psf(x, y, z, p, wavelength, ni, sf=1):
    """
    Born-wolf PSF function evaluated using Airy function if in-focus, and axial function if along the axis.
    Otherwise evaluated using numerical integration.

    :param x: in um
    :param y: in um
    :param z: in um
    :param p: [A, cx, cy, cz, NA, bg]
    :param float wavelength: in um
    :param float ni: index of refraction
    :param int sf:
    :return value:
    """
    if sf != 1:
        raise NotImplementedError("Only implemented for sf=1")

    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    x, y, z = np.broadcast_arrays(x, y, z)

    k = 2 * np.pi / wavelength
    rr = np.sqrt((x - p[1]) ** 2 + (y - p[2]) ** 2)

    psfs = np.zeros(rr.shape) * np.nan
    is_in_focus = (z == p[3])
    is_on_axis = (rr == 0)

    # ################################
    # evaluate in-focus portion using airy function, which is much faster than integrating
    # ################################
    def airy_fn(rho):
        val = p[0] * 4 * np.abs(sp.j1(rho * k * p[4]) / (rho * k * p[4]))**2 + p[5]
        val[rho == 0] = p[0] + p[4]
        return val

    with np.errstate(invalid="ignore"):
        psfs[is_in_focus] = airy_fn(rr[is_in_focus])

    # ################################
    # evaluate on axis portion using exact expression
    # ################################
    def axial_fn(z):
        val = p[0] * 4 * (2 * ni ** 2) / (k ** 2 * p[4] ** 4 * (z - p[3]) ** 2) * \
               (1 - np.cos(0.5 * k * (z - p[3]) * p[4] ** 2 / ni)) + p[5]
        val[z == p[3]] = p[0] + p[5]
        return val

    with np.errstate(invalid="ignore"):
        psfs[is_on_axis] = axial_fn(z[is_on_axis])

    # ################################
    # evaluate out of focus portion using integral
    # ################################
    if not np.all(is_in_focus):

        def integrand(rho, r, z):
            return rho * sp.j0(k * r * p[4] * rho) * np.exp(-1j * k * (z - p[3]) * p[4]**2 * rho**2 / (2 * ni))

        # like this approach because allows rr, z, etc. to have arbitrary dimension
        already_evaluated = np.logical_or(is_in_focus, is_in_focus)
        for ii, (r, zc, already_eval) in enumerate(zip(rr.ravel(), z.ravel(), already_evaluated.ravel())):
            if already_eval:
                continue

            int_real = scipy.integrate.quad(lambda rho: integrand(rho, r, zc).real, 0, 1)[0]
            int_img = scipy.integrate.quad(lambda rho: integrand(rho, r, zc).imag, 0, 1)[0]

            coords = np.unravel_index(ii, rr.shape)
            psfs[coords] = p[0] * 4 * (int_real ** 2 + int_img ** 2) + p[5]

    return psfs


# utility functions
def oversample_pixel(x, y, z, ds, sf, euler_angles=(0., 0., 0.)):
    """
    Suppose we have a set of pixels centered at points given by x, y, z. Generate sf**2 points in this pixel equally
    spaced about the center. Allow the pixel to be orientated in an arbitrary direction with respect to the coordinate
    system. The pixel rotation is described by the Euler angles (psi, theta, phi), where the pixel "body" frame
    is a square with xy axis orientated along the legs of the square with z-normal to the square

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
        result = mat.dot(np.concatenate((xp.ravel()[None, :], yp.ravel()[None, :], zp.ravel()[None, :]), axis=0))
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


def oversample_voxel(coords, drs, sf=3):
    """
    Get pointsss to oversample a voxel

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


# main functions for fitting/plotting PSFs
def get_psf_coords(ns, drs):
    """
    Get centered coordinates for PSFmodels style PSF's from step size and number of coordinates
    :param ns: list of number of points
    :param drs: list of step sizes
    :return coords: list of coordinates [zs, ys, xs, ...]
    """
    ndims = len(drs)
    coords = [np.expand_dims(d * (np.arange(n) - n // 2), axis=tuple(range(ii)) + tuple(range(ii+1, ndims)))
              for ii, (n, d) in enumerate(zip(ns, drs))]

    return coords


def model_psf(nx, dxy, z, p, wavelength, ni, sf=1, model='vectorial', **kwargs):
    """
    Wrapper function for evaluating different PSF models where only the coordinate grid information is given
    (i.e. nx and dxy) and not the actual coordinates. The real coordinates can be obtained using get_psf_coords().

    The size of these functions is parameterized by the numerical aperture, instead of the sigma or other size
    parameters that are only convenient for some specific models of the PSF

    For vectorial or gibson-lanni PSF's, this wraps the functions in the psfmodels package
    (https://pypi.org/project/psfmodels/) with added ability to shift the center of these functions away from the
    center of the ROI, which is useful when fitting them.

    For 'gaussian', it wraps the gaussian3d_pixelated_psf() function. More details about the relationship between
    the Gaussian sigma and the numerical aperture can be found here: https://doi.org/10.1364/AO.46.001819

    todo: need to implement index of refraction everywhere?

    :param nx: number of points to be sampled in x- and y-directions
    :param dxy: pixel size in um
    :param z: z positions in um
    :param p: [A, cx, cy, cz, NA, bg]
    :param float wavelength: wavelength in um
    :param float ni: index of refraction
    :param int sf:
    :param model: 'gaussian', 'gibson-lanni', 'born-wolf', or 'vectorial'. 'gibson-lanni'
     relies on the psfmodels function
    scalar_psf(), while 'vectorial' relies on the psfmodels function vectorial_psf()
    :param kwargs: keywords passed through to vectorial_psf() or scalar_psf()
    :return:
    """
    if 'NA' in kwargs.keys():
        raise ValueError("'NA' is not allowed to be passed as a named parameter. It is specified in p.")

    model_params = {'NA': p[4], 'sf': sf}
    model_params.update(kwargs)

    if model == 'vectorial':
        if not psfmodels_available:
            raise NotImplementedError("vectorial model selected but psfmodels is not installed")
        if sf != 1:
            raise NotImplementedError('vectorial model not implemented for sf=/=1')

        psf_norm = psfm.vectorial_psf(0, 1, dxy, wvl=wavelength, params=model_params, normalize=False)
        val = psfm.vectorial_psf(z - p[3], nx, dxy, wvl=wavelength, params=model_params, normalize=False)
        val = p[0] / psf_norm * ndi.shift(val, [0, p[2] / dxy, p[1] / dxy], mode='nearest') + p[5]
    elif model == 'gibson-lanni':
        if not psfmodels_available:
            raise NotImplementedError("gibson-lanni model selected but psfmodels is not installed")
        if sf != 1:
            raise NotImplementedError('gibson-lanni model not implemented for sf=/=1')

        psf_norm = psfm.scalar_psf(0, 1, dxy, wvl=wavelength, params=model_params, normalize=False)
        val = psfm.scalar_psf(z - p[3], nx, dxy, wvl=wavelength, params=model_params, normalize=False)
        val = p[0] / psf_norm * ndi.shift(val, [0, p[2] / dxy, p[1] / dxy], mode='nearest') + p[5]
    elif model == "born-wolf":
        if sf != 1:
            raise NotImplementedError('gibson-lanni model not implemented for sf=/=1')

        y, x, = get_psf_coords([nx, nx], [dxy, dxy])
        y = np.expand_dims(y, axis=0)
        x = np.expand_dims(x, axis=0)
        z = np.expand_dims(np.array(z, copy=True), axis=(1, 2))

        val = born_wolf_psf(x, y, z, p, wavelength, ni, sf=sf)
    elif model == 'gaussian':
        # Gaussian approximation to PSF. Matches well for equal peak intensity, but some deviations in area.
        # See https://doi.org/10.1364/AO.46.001819 for more details.
        # sigma_xy = 0.22 * lambda / NA.
        # This comes from equating the FWHM of the Gaussian and the airy function.
        # FWHM = 2 * sqrt{2*log(2)} * sigma ~ 0.51 * wavelength / NA
        # transform NA to sigmas
        p_gauss = [p[0], p[1], p[2], p[3],
                   0.22 * wavelength / p[4],
                   np.sqrt(6) / np.pi * ni * wavelength / p[4] ** 2,
                   p[5]]

        y, x, = get_psf_coords([nx, nx], [dxy, dxy])
        y = np.expand_dims(y, axis=0)
        x = np.expand_dims(x, axis=0)
        z = np.expand_dims(np.array(z, copy=True), axis=(1, 2))

        # normalize so that peak amplitude is actually
        psf_norm = gaussian3d_psf(p[1], p[2], p[3], dxy, p_gauss, sf, angles=(0., 0., 0.)) - p[5]
        val = p[0] / psf_norm * (gaussian3d_psf(x, y, z, dxy, p_gauss, sf, angles=(0., 0., 0.)) - p[5]) + p[5]
    else:
        raise ValueError("model must be 'gibson-lanni', 'vectorial', 'born-wolf', or 'gaussian' but was '%s'" % model)

    return val


def fit_psfmodel(img, dxy, dz, wavelength, ni, sf=1, model='vectorial',
                 init_params=None, fixed_params=None, sd=None, bounds=None):
    """
    3D non-linear least squares fit using one of the point spread function models from psfmodels package.

    The x/y coordinates are assumed to match the convention of get_coords(), i.e. they are (arange(nx) / nx//2) * d

    # todo: make sure ni implemented correctly. if want to use different ni, have to be careful because this will
    # todo: shift the focus position away from z=0
    # todo: make sure oversampling (sf) works correctly with all functions

    :param img: nz x ny x nx image stack
    :param float dxy: dx and dy in um
    :param float dz: dz in um
    :param float wavelength: wavelength in um
    :param float ni: index of refraction
    :param sf:
    :param model: 'gaussian', 'gibson-lanni', or 'vectorial'
    :param init_params: [A, cx, cy, cz, NA, bg]
    :param list[bool] fixed_params:
    :param sd: standard deviations if img is derived from averaged pictures
    :param bounds:
    :return result, fit_fn: result is a dictionary object, and fit_fn takes arguments x and y
    """

    # get coordinates
    z, y, x = get_psf_coords(img.shape, [dz, dxy, dxy])
    z = z[:, 0, 0]

    # check size
    nz, ny, nx = img.shape
    if not ny == nx:
        raise ValueError('x- and y-size of img must be equal')

    if not np.mod(nx, 2) == 1:
        raise ValueError('x- and y-size of img must be odd')

    # get default initial parameters
    if init_params is None:
        init_params = [None] * 6
    else:
        init_params = copy.deepcopy(init_params)

    # use default values for any params that are None
    if np.any([ip is None for ip in init_params]):
        # exclude nans
        to_use = np.logical_not(np.isnan(img))

        bg = np.mean(img[to_use].ravel())
        amp = np.max(img[to_use].ravel()) - bg

        cz, cy, cx = fit.get_moments(img, order=1, coords=(np.expand_dims(z, axis=(1, 2)), y, x))

        # iz = np.argmin(np.abs(z - cz))
        # m2y, m2x = tools.get_moments(img[iz], order=2, coords=[y, x])
        # sx = np.sqrt(m2x - cx ** 2)
        # sy = np.sqrt(m2y - cy ** 2)
        # # from gaussian approximation
        # na_guess = 0.22 * wavelength / np.sqrt(sx * sy)
        na_guess = 1

        ip_default = [amp, cx, cy, cz, na_guess, bg]

        for ii in range(len(init_params)):
            if init_params[ii] is None:
                init_params[ii] = ip_default[ii]

    # set bounds
    if bounds is None:
        # allow for 2D fitting by making z bounds eps apart
        if z.min() == z.max():
            zlow = z.min() - 1e-12
            zhigh = z.max() + 1e-12
        else:
            zlow = z.min()
            zhigh = z.max()

        # NA must be <= index of refraction
        bounds = ((0, x.min(), y.min(), zlow, 0, -np.inf),
                  (np.inf, x.max(), y.max(), zhigh, ni, np.inf))

    # do fitting
    def model_fn(z, nx, dxy, p): return model_psf(nx, dxy, z, p, wavelength, ni, sf=sf, model=model)

    result = fit.fit_model(img, lambda p: model_fn(z, nx, dxy, p), init_params,
                           fixed_params=fixed_params, sd=sd, bounds=bounds, jac='3-point', x_scale='jac')

    # model function at fit parameters
    def fit_fn(z, nx, dxy): return model_fn(z, nx, dxy, result['fit_params'])

    return result, fit_fn


def plot_psfmodel_fit(imgs, dx, dz, wavelength, ni, sf, fit_params, model='vectorial',
                      gamma=1., figsize=(18, 10), save_dir=None, label='', **kwargs):
    """
    Plot data and fit obtained from fit_psfmodel().

    Multiple different fits can be plotted if fit_params, chi_sqrs, cov, and model are provided as lists.

    :param imgs: 3D image stack
    :param float dx: pixel size in um
    :param float dz: space between z-planes in um
    :param float float wavelength:
    :param float ni:
    :param int sf:
    :param fit_params: nfits x nparams array
    :param str model: 'vectorial', 'gibson-lanni', 'born-wolf', or 'gaussian'
    :param float gamma:
    :param tuple(float, float) figsize: (sx, sy)
    :param str save_dir: if not None, then a png of figure will be saved in the provided directory
    :param str label: label to add to the start of the file name, if saving
    :param kwargs: additional keyword arguments are passed to plt.figure()
    :return figure_handle:
    """

    info = "%s, %s, sf=%d" % (label, model, sf)
    fp_names = ["A", "cx", "cy", "cz", "NA", "bg"]
    coords = get_psf_coords(imgs.shape, (dz, dx, dx))
    imgs_fit = model_psf(imgs.shape[-1], dx, coords[0][:, 0, 0], fit_params, wavelength, ni, sf, model=model)

    figh = plot_psf_fit(imgs, imgs_fit, coords, fit_params, fit_param_names=fp_names, label=info,
                        gamma=gamma, figsize=figsize, save_dir=save_dir, **kwargs)

    return figh


# get real PSF
def get_exp_psf(imgs, coords, centers, roi_sizes, backgrounds=None):
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
    dx = x[0, 0, 1] - x[0, 0, 0]

    # set up array to hold psfs
    nrois = len(centers)
    if backgrounds is None:
        backgrounds = np.zeros(nrois)

    psf_shifted = np.zeros((nrois, roi_sizes[0], roi_sizes[1], roi_sizes[2])) * np.nan
    # coordinates
    z_psf, y_psf, x_psf = get_psf_coords(roi_sizes, [dz, dx, dx])

    zc_pix_psf = np.argmin(np.abs(z_psf))
    yc_pix_psf = np.argmin(np.abs(y_psf))
    xc_pix_psf = np.argmin(np.abs(x_psf))

    # loop over rois and shift psfs so they are centered
    for ii in range(nrois):
        # get closest pixels to center
        xc_pix = np.argmin(np.abs(x - centers[ii, 2]))
        yc_pix = np.argmin(np.abs(y - centers[ii, 1]))
        zc_pix = np.argmin(np.abs(z - centers[ii, 0]))

        # cut roi from image
        roi_unc = rois.get_centered_roi((zc_pix, yc_pix, xc_pix), roi_sizes)
        roi = rois.get_centered_roi((zc_pix, yc_pix, xc_pix), roi_sizes, min_vals=[0, 0, 0], max_vals=imgs.shape)
        img_roi = imgs[roi[0]:roi[1], roi[2]:roi[3], roi[4]:roi[5]]

        zroi = z[roi[0]:roi[1], :, :]
        yroi = y[:, roi[2]:roi[3], :]
        xroi = x[:, :, roi[4]:roi[5]]

        cx_pix_roi = (roi[5] - roi[4]) // 2
        cy_pix_roi = (roi[3] - roi[2]) // 2
        cz_pix_roi = (roi[1] - roi[0]) // 2

        xshift_pix = (xroi[0, 0, cx_pix_roi] - centers[ii, 2]) / dx
        yshift_pix = (yroi[0, cy_pix_roi, 0] - centers[ii, 1]) / dx
        zshift_pix = (zroi[cz_pix_roi, 0, 0] - centers[ii, 0]) / dz

        # get coordinates
        img_roi_shifted = ndi.shift(np.array(img_roi, dtype=float), [zshift_pix, yshift_pix, xshift_pix],
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

    with np.errstate(divide='ignore', invalid='ignore'):
        psf_mean = np.nanmean(psf_shifted, axis=0)

    # the above doesn't do a good enough job of normalizing PSF
    max_val = np.nanmax(psf_mean[psf_mean.shape[0]//2])
    psf_mean = psf_mean / max_val

    # get otf
    otf_mean, ks = psf2otf(psf_mean, drs=(dz, dx, dx))
    kz, ky, kx = ks

    return psf_mean, (z_psf, y_psf, x_psf), otf_mean, (kz, ky, kx)


# other display functions
def plot_psf_fit(imgs, imgs_fit, coords, fit_params, fit_param_names=None, label="",
                 figsize=(18, 10), gamma=1, save_dir=None, **kwargs):
    """
    Plot PSF model compared with real data. Intended to be agnostic of fit parameters

    @param imgs:
    @param imgs_fit:
    @param coords:
    @param fit_params:
    @param fit_param_names:
    @param label:
    @param figsize:
    @param gamma:
    @param save_dir:
    @param kwargs:
    @return:
    """

    # get image size and central pixels
    nz, ny, nx = imgs.shape

    zc_pix = nz // 2
    yc_pix = ny // 2
    xc_pix = nx // 2

    # get coordinates
    z, y, x, = coords
    dz = z[1, 0, 0] - z[0, 0, 0]
    dy = y[0, 1, 0] - y[0, 0, 0]
    dx = x[0, 0, 1] - x[0, 0, 0]

    # other useful coordinate info
    extent_xy = [x[0, 0, 0] - 0.5 * dx, x[0, 0, -1] + 0.5 * dx, y[0, -1, 0] + 0.5 * dy, y[0, 0, 0] - 0.5 * dy]
    extent_xz = [x[0, 0, 0] - 0.5 * dx, x[0, 0, -1] + 0.5 * dx, z[-1, 0, 0] + 0.5 * dz, z[0, 0, 0] - 0.5 * dz]
    extent_zy = [z[0, 0, 0] - 0.5 * dz, z[-1, 0, 0] + 0.5 * dz, y[0, -1, 0] + 0.5 * dy, y[0, 0, 0] - 0.5 * dy]

    # plot results
    ttl_str = "%s\n" % label
    for ii in range(len(fit_params)):
        if fit_param_names is None:
            ttl_str += "%0.3f, " % fit_params[ii]
        else:
            ttl_str += "%s=%0.3f, " % (fit_param_names[ii], fit_params[ii])

    figh = plt.figure(figsize=figsize, **kwargs)
    grid = plt.GridSpec(2, 4, wspace=0.5, hspace=0.5)
    plt.suptitle(ttl_str)

    # XY-plane
    ax = plt.subplot(grid[0, 1])
    ax.imshow(imgs[zc_pix], extent=extent_xy, cmap="bone", norm=PowerNorm(gamma=gamma))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Data")

    ax = plt.subplot(grid[0, 3])
    ax.imshow(imgs_fit[zc_pix], extent=extent_xy, cmap="bone", norm=PowerNorm(gamma=gamma))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Fit")

    # XZ-plane
    ax = plt.subplot(grid[1, 1])
    ax.imshow(imgs[:, yc_pix], extent=extent_xz, cmap="bone", norm=PowerNorm(gamma=gamma))
    ax.set_xlabel("x ($\mu m$)")
    ax.set_ylabel("z ($\mu m$)")

    ax = plt.subplot(grid[1, 3])
    ax.imshow(imgs_fit[:, yc_pix], extent=extent_xz, cmap="bone", norm=PowerNorm(gamma=gamma))
    ax.set_xlabel("x ($\mu m$)")
    ax.set_ylabel("z ($\mu m$)")

    # YZ-plane
    ax = plt.subplot(grid[0, 0])
    ax.imshow(np.transpose(imgs[:, :, xc_pix]), extent=extent_zy, cmap="bone", norm=PowerNorm(gamma=gamma))
    ax.set_xlabel("z ($\mu m$)")
    ax.set_ylabel("y ($\mu m$)")

    ax = plt.subplot(grid[0, 2])
    ax.imshow(np.transpose(imgs_fit[:, :, xc_pix]), extent=extent_zy, cmap="bone", norm=PowerNorm(gamma=gamma))
    ax.set_xlabel("z ($\mu m$)")
    ax.set_ylabel("y ($\mu m$)")

    # XY cuts
    ax = plt.subplot(grid[1, 0])
    ax.plot(np.sqrt(np.expand_dims(x, axis=0) ** 2 + np.expand_dims(y, axis=1) ** 2).ravel(), imgs[zc_pix].ravel(), 'g.')
    ax.plot(y.ravel(), imgs[zc_pix, :, xc_pix], 'b.')
    ax.plot(x.ravel(), imgs[zc_pix, yc_pix, :], 'k.')
    ax.plot(y.ravel(), imgs_fit[zc_pix, :, xc_pix], 'b')
    ax.plot(x.ravel(), imgs_fit[zc_pix, yc_pix, :], 'k')
    ax.set_xlabel("xy-position ($\mu m$)")
    ax.set_ylabel("amplitude")
    ax.legend(["all", "y-cut", "x-cut"])

    # z cuts
    ax = plt.subplot(grid[1, 2])
    ax.plot(z.ravel(), imgs[:, yc_pix, xc_pix], 'b.')
    ax.plot(z.ravel(), imgs_fit[:, yc_pix, xc_pix], 'b')
    ax.set_xlabel("z-position ($\mu m$)")
    ax.set_ylabel("amplitude")

    # optional saving
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        figh.savefig(os.path.join(save_dir, "%s.png" % label))
        plt.close(figh)

    return figh


def plot_fit_stats(fit_params, figsize=(18, 10), **kwargs):
    """
    Plot statistics for a list of fit result dictionaries

    :param fit_params: N x 6 list of localization fit parameters
    :param figsize:
    :param kwargs: passed to plt.figure()
    :return figh: figure handle
    """

    # fit parameter summary
    figh = plt.figure(figsize=figsize, **kwargs)
    plt.suptitle("Localization fit parameter summary")
    grid = plt.GridSpec(2, 2, hspace=1, wspace=0.5)

    # amplitude vs sxy
    ax = plt.subplot(grid[0, 0])
    ax.plot(fit_params[:, 4], fit_params[:, 0], '.')
    ax.set_xlabel(r"$\sigma_{xy}$ ($\mu m$)")
    ax.set_ylabel("amp")

    # sxy vs sz
    ax = plt.subplot(grid[0, 1])
    ax.plot(fit_params[:, 4], fit_params[:, 5], '.')
    ax.set_xlabel(r"$\sigma_{xy}$ ($\mu m$)")
    ax.set_ylabel(r"$\sigma_{z}$ ($\mu m$)")

    # sxy vs bg
    ax = plt.subplot(grid[1, 1])
    ax.plot(fit_params[:, 4], fit_params[:, 6], '.')
    ax.set_xlabel(r"$\sigma_{xy}$ ($\mu m$)")
    ax.set_ylabel(r"$\sigma_{z}$ ($\mu m$)")

    return figh
