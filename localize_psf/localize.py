"""
Code for localizing and fitting (typically diffraction limited) spots/beads

The fitting code can be run on a CPU using multiprocessing with joblib, or on a GPU using custom modifications
to GPUfit which can be found at https://github.com/QI2lab/Gpufit. To use the GPU code, you must download and
compile this repository and install the python bindings.
"""
from typing import Union, Optional, Sequence
from pathlib import Path
import time
import warnings
import zarr
import numpy as np
import scipy.signal
import scipy.ndimage
import dask
from dask.diagnostics import ProgressBar
from numba import njit, prange
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, LinearSegmentedColormap, Normalize
import localize_psf.rois as roi_fns
from localize_psf import fit
import localize_psf.fit_psf as psf

# for filtering on GPU
try:
    import cupy as cp
    import cupyx.scipy.signal
    import cupyx.scipy.ndimage
    _cupy_available = True
except ImportError:
    cp = np
    _cupy_available = False

# custom GPUFit for fitting on GPU
try:
    import pygpufit.gpufit as gf
    _gpufit_available = True
except ImportError:
    _gpufit_available = False

array = Union[np.ndarray, cp.ndarray]


def get_coords(sizes: Sequence[int],
               drs: Sequence[float],
               broadcast: bool = False) -> tuple[np.ndarray[float]]:
    """
    Regularly spaced coordinates which can be broadcast to full size.

    For example, if sizes = (nz, ny, nx) and drs = (1, 1, 1) then
    coords0.size = (nz, 1, 1)
    coords1.size = (1, ny, 1)
    coords2.size = (1, 1, nx)
    this arrays are broadcastable to size (nz, ny, nx). If the broadcast arrays are desired, they can be obtained by:
    >>> coords_bcast = np.broadcast_arrays(coords0, coords1, coords2)
    >>> coords0_bc, coords1_bc, coords2_bc = [np.array(c, copy=True) for c in coords_bcast]
    note that the second line is necessary because np.broadcast_arrays() produces arrays with references to the
    original entries, so assigning to these arrays can produce surprising results.

    :param sizes: (s0, s1, ..., sn)
    :param drs: (dr0, dr1, ..., drn)
    :param broadcast: whether to expand all arrays to full size, or keep as 1D arrays with singleton dimensions
      that will be automatically broadcast during arithmetic
    :return coords: (coords0, coords1, ..., coordsn)
    """
    ndims = len(drs)
    coords = [np.expand_dims(np.arange(sz, dtype=float) * dr, axis=list(range(ii)) + list(range(ii + 1, ndims)))
              for ii, (sz, dr) in enumerate(zip(sizes, drs))]

    if broadcast:
        # this produces copies of the arrays instead of views
        coords = [np.array(c, copy=True) for c in np.broadcast_arrays(*coords)]

    return tuple(coords)


def get_nearest_pixel(centers: np.ndarray[float],
                      drs: np.ndarray[float]) -> np.ndarray[int]:
    """
    Get nearest pixel indices for centers given in real coordinates

    :param centers:
    :param drs:
    :return indices:
    """
    drs = np.asarray(drs)

    return np.rint(centers / drs).astype(int)

@njit(parallel=True)
def prepare_rois(image: np.ndarray,
                 coords: tuple[np.ndarray, np.ndarray, np.ndarray],
                 rois: np.ndarray[int]) -> (np.ndarray[float], tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float]], np.ndarray[int]):
    """
    Cut ROI out of image and coordinate arrays and insert into nroi x nmax_roi_size array which is nan padded

    :param image: image
    :param coords:
    :param rois:
    :return img_rois, roi_coords, roi_sizes:
    """

    nrois = len(rois)
    z, y, x = coords

    # numba does not support prod with axis argument
    sizes = (rois[..., 1] - rois[..., 0]) * (rois[..., 3] - rois[..., 2]) * (rois[..., 5] - rois[..., 4])
    nmax_roi_size = np.max(sizes)

    img_rois = np.ones((nrois, nmax_roi_size)) * np.nan
    x_rois = np.ones((nrois, nmax_roi_size)) * np.nan
    y_rois = np.ones((nrois, nmax_roi_size)) * np.nan
    z_rois = np.ones((nrois, nmax_roi_size)) * np.nan

    nrois = len(rois)
    for rr in prange(nrois):
        roi = rois[rr]

        nz = roi[1] - roi[0]
        ny = roi[3] - roi[2]
        nx = roi[5] - roi[4]
        n_size_roi = nz * ny * nx

        img_rois[rr, :n_size_roi] = image[roi[0]:roi[1], roi[2]:roi[3], roi[4]:roi[5]].ravel()

        # numba compatible equivalent of np.array_broadcast()
        for ii in prange(nz):
            for jj in prange(ny):
                for kk in prange(nx):
                    counter = kk + nx * jj + ny * nx * ii
                    x_rois[rr, counter] = x[0, 0, roi[4] + kk]
                    y_rois[rr, counter] = y[0, roi[2] + jj, 0]
                    z_rois[rr, counter] = z[roi[0] + ii, 0, 0]

    return img_rois, (z_rois, y_rois, x_rois), sizes

def get_roi(center: Sequence[float],
            img: np.ndarray,
            coords: Sequence[np.ndarray],
            sizes: tuple[int]):
    """
    Find ROI which is nearly centered on center. Since center may not correspond to a pixel location, and the
    size of the ROI may not be odd, center will not be the exact center of the ROI

    :param center: [c_0, c_1, ..., c_n] in same units as x, y, z.
    :param img: array of arbitrary size, m0 x m1 x ... x mn
    :param coords: (coords0, coords1, ..., coordsN)
    :param sizes: [i0, i1, ... in] integers
    :return roi, img_roi, coords_roi:
    """

    warnings.warn("get_roi() is deprecated and will be removed soon. Please use prepare_rois() instead")

    # todo: deprecate in favor of vectorized finding ROIs and prepare_rois()

    ndims = img.ndim
    # get closest coordinates to desired center of roi
    ics = [np.argmin(np.abs(r.ravel() - c)) for r, c in zip(coords, center)]

    roi = roi_fns.get_centered_rois(ics, sizes, min_vals=[0]*ndims, max_vals=img.shape)[0]

    # get coordinates as arrays which only have nonunit size along one direction
    coords_roi = [c[tuple([slice(None)] * ii + [slice(roi[2*ii], roi[2*ii + 1])] + [slice(None)] * (ndims - 1 - ii))]
                  for ii, c in enumerate(coords)]
    # broadcast to full arrays, essentially meshgrid
    coords_roi = np.broadcast_arrays(*coords_roi)

    img_roi = roi_fns.cut_roi(roi, img)[0]

    return roi, img_roi, coords_roi


def get_filter_kernel(sigmas: Sequence[float],
                      drs: Sequence[float],
                      sigma_cutoff: int = 2) -> np.ndarray:
    """
    Gaussian filter kernel for arbitrary dimensions. If drs or sigmas are zero along one dimension, then the kernel
    will have unit length and weight along this direction

    :param sigmas: (sigma_0, sigma_1, ..., sigma_n)
    :param drs: (dr_1, dr_2, ..., dr_n) pixel sizes along each dimension
    :param sigma_cutoff: single number or list [s0, s1, ..., sn] giving after how many sigmas we cutoff the kernel
    :return kernel:
    """

    ndims = len(drs)

    if isinstance(sigma_cutoff, (int, float)):
        sigma_cutoff = [sigma_cutoff] * ndims

    # convert to arrays
    drs = np.array(drs, copy=True)
    sigmas = np.array(sigmas, copy=True)
    sigma_cutoff = np.array(sigma_cutoff, copy=True)

    # compute kernel size
    with np.errstate(invalid="ignore"):
        nks = 2 * np.round(sigmas / drs * sigma_cutoff) + 1
        nks[np.isnan(nks)] = 1
        nks = nks.astype(int)
        # nks = [2 * int(np.round(sig / dr * sig_cut)) + 1 for sig, dr, sig_cut in zip(sigmas, drs, sigma_cutoff)]

    # now need to correct sigma = 0, as this would result in invalid kernel. Doesn't matter what value we set because
    # coordinates will all be zeros
    sigmas[sigmas == 0] = 1

    # get coordinates to evaluate kernel at
    coords = [np.expand_dims(np.arange(nk) * dr, axis=list(range(ii)) + list(range(ii + 1, ndims))) for
              ii, (nk, dr) in enumerate(zip(nks, drs))]

    coords = [c - np.mean(c) for c in coords]

    kernel = np.exp(sum([-rk ** 2 / 2 / sig ** 2 for rk, sig in zip(coords, sigmas)]))
    kernel = kernel / np.sum(kernel)

    return kernel


def filter_convolve(imgs: array,
                    kernel: array) -> array:
    """
    Convolution filter using kernel with GPU support. To avoid roll-off effects at the edge, the convolved
    result is "normalized" by being divided by the kernel convolved with an array of ones.

    This function can be run on either the GPU or the CPU. To run on the GPU, ensure that imgs is a cupy array

    :param imgs: images to be convolved
    :param kernel: kernel to be convolved. Does not need to be the same shape as image.
    :param bool use_gpu: if True, do convolution on GPU. If false, do on CPU.
    :return imgs_filtered:
    """
    # todo: check and make sure kernel and imgs are compatible. e.g. that kernel is smaller than image in all dims
    # todo: estimate how much memory convolution requires? Much more than I expect...
    # todo: possibly because issue with fft plan caching, which is resolved by forcing cache to zero and clearing it

    use_gpu = isinstance(imgs, cp.ndarray) and _cupy_available

    if use_gpu:
        xp = cp
        convolve = cupyx.scipy.signal.fftconvolve
        cp.fft._cache.PlanCache(memsize=0)
    else:
        xp = np
        convolve = scipy.signal.fftconvolve

    kernel = xp.asarray(kernel)
    imgs = xp.asarray(imgs)

    # convolve, and deal with edges by normalizing
    imgs_filtered = convolve(imgs, kernel, mode="same")
    norm = convolve(xp.ones(imgs.shape), kernel, mode="same")
    imgs_filtered /= norm

    if use_gpu:
        cache = cp.fft.config.get_plan_cache()
        cache.clear()

    return imgs_filtered


def get_max_filter_footprint(min_separations: Sequence[float],
                             drs: Sequence[float]) -> np.ndarray:
    """
    Get footprint for maximum filter. This is a binary mask which is True at points included in the mask
    and False at other points. For doing a square maximum filter can choose a footprint of only Trues, but cannot
    do this for more complex shapes

    :param min_separations: (size_0, size_1, ..., size_n)
    :param drs: (dr_0, ..., dr_n)
    :return footprint: boolean mask
    """

    min_sep_allowed = np.array(min_separations)
    drs = np.array(drs)

    ns = np.ceil(min_sep_allowed / drs).astype(int)
    # ensure at least size 1
    ns[ns == 0] = 1
    # ensure odd
    ns += (1 - np.mod(ns, 2))

    footprint = np.ones(ns, dtype=bool)

    return footprint


def find_peak_candidates(imgs: array,
                         footprint: array,
                         threshold: float,
                         mask: Optional[np.ndarray] = None) -> (np.ndarray, np.ndarray):
    """
    Find peak candidates in image using maximum filter. This can be run on either the GPU or the CPU.

    :param imgs: 2D or 3D array. If this is a CuPy array, function will be run on the GPU
    :param footprint: footprint to use for maximum filter. Array should have same number of dimensions as imgs.
      This can be obtained from get_max_filter_footprint()
    :param threshold: only pixels with values greater than or equal to the threshold will be considered
    :param mask:
    :return inds, amps: np.array([[i0, i1, i2], ...]) array indices of local maxima
    """
    use_gpu_filter = isinstance(imgs, cp.ndarray) and _cupy_available

    if use_gpu_filter:
        xp = cp
        max_filter = cupyx.scipy.ndimage.maximum_filter
    else:
        xp = np
        max_filter = scipy.ndimage.maximum_filter

    if mask is None:
        mask = xp.ones(imgs.shape, dtype=bool)

    mask = xp.asarray(mask)
    imgs = xp.asarray(imgs)
    footprint = xp.asarray(footprint)

    img_max_filtered = max_filter(imgs, footprint=footprint)
    # don't use reduce because CuPy doesn't support it
    is_max = xp.logical_and(xp.logical_and(imgs == img_max_filtered, imgs >= threshold), mask)

    amps = imgs[is_max]
    inds = xp.argwhere(is_max)

    return inds, amps


def filter_nearby_peaks(centers: np.ndarray,
                        min_xy_dist: float,
                        min_z_dist: float,
                        mode: str = "keep-one",
                        weights: Optional[np.ndarray] = None,
                        nmax: int = 10000) -> (np.ndarray, np.ndarray):
    """
    Combine multiple center positions into a reduced set, where assume all centers separated by no more than
    min_xy_dist and min_z_dist come from the same feature.

    This function treats xy and z directions separately. Centers must be close in both to be filtered.

    :param centers: N x 3 array [cz, cy, cx]
    :param min_xy_dist:
    :param min_z_dist:
    :param mode: "average", "keep-one", or "remove"
    :param weights: only used in "average" mode. If weights are provided, a weighted average between nearby
      points is computed
    :param nmax: maximum number of centers to be processed at once. If the number of centers exceeds this size,
      then the problem will be split in half (recursively) and solved on the subregions. Then the overlap zone
      between these two regions will be checked and the results will be combined
    :return centers_unique: array of unique center coordinates
    :return inds: index into the initial array to produce centers_unique. In mode is "keep-one" or "remove"
      then centers_unique = centers[inds]. If mode is "average", this will not be true as centers_unique will
      not be elements of centers. However, centers[inds] will correspond to one point which was averaged to produce
      the corresponding element of centers_unique
    """

    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("centers should be a nx3 array where columns are cz, cy, cx")

    centers_unique = np.array(centers, copy=True)
    inds = np.arange(len(centers), dtype=int)

    if weights is None:
        weights = np.ones(len(centers_unique))

    # only need to act if minimum distances are non-zero and we have any centers to deal with
    if (min_xy_dist > 0 or min_z_dist > 0) and centers_unique.size != 0:
        # if number of points is large, divide problem into subproblems, solve each of these, and combine results.

        # check and make sure region is large enough (relative to the minimum distances) to be divide
        # limits of data
        clims = np.stack((np.min(centers, axis=0),
                          np.max(centers, axis=0)), axis=1)

        # find ranges as fraction of min_dist
        min_dists = np.array([min_z_dist, min_xy_dist, min_xy_dist])

        with np.errstate(invalid="ignore"):
            # todo: does this have a problem if min_dists = 0?
            ranges = (clims[:, 1] - clims[:, 0]) / min_dists

        if len(centers_unique) > nmax and not np.all(ranges <= 2):
            if mode == "average":
                raise NotImplementedError("mode='average' is not implemented with nmax < np.inf. Set nmax to np.inf")

            ind_red_dim = np.argmax(ranges)

            # divide into two regions and solve separately
            in1 = np.logical_and(centers_unique[:, ind_red_dim] >= clims[ind_red_dim, 0],
                                 centers_unique[:, ind_red_dim] < np.mean(clims[ind_red_dim]))

            in2 = np.logical_and(centers_unique[:, ind_red_dim] >= np.mean(clims[ind_red_dim]),
                                 centers_unique[:, ind_red_dim] <= clims[ind_red_dim, 1])

            if np.any(in1):
                cu1, i1 = filter_nearby_peaks(centers_unique[in1], min_xy_dist, min_z_dist, mode=mode)
            else:
                cu1 = np.zeros((0, 3))
                i1 = np.zeros((0, 3))

            if np.any(in2):
                cu2, i2 = filter_nearby_peaks(centers_unique[in2], min_xy_dist, min_z_dist, mode=mode)
            else:
                cu2 = np.zeros((0, 3))
                i2 = np.zeros((0, 3))

            full_inds = np.arange(len(centers_unique), dtype=int)
            centers_unique_sectors = np.concatenate((cu1, cu2))
            inds_sectors = np.concatenate((full_inds[in1][i1], full_inds[in2][i2]))

            # take care of any non-unique points in the overlap region
            in_overlap = np.logical_and(centers_unique_sectors[:, ind_red_dim] >= np.mean(clims[ind_red_dim]) - min_dists[ind_red_dim],
                                        centers_unique_sectors[:, ind_red_dim] <= np.mean(clims[ind_red_dim]) + min_dists[ind_red_dim])

            if np.any(in_overlap):
                centers_unique_overlap, i_overlap = filter_nearby_peaks(centers_unique_sectors[in_overlap], min_xy_dist, min_z_dist, mode=mode)

                # get full centers by adding any that were not in the overlap region with the reduced set from the overlap region
                centers_unique_sectors = np.concatenate((centers_unique_sectors[np.logical_not(in_overlap)],
                                                         centers_unique_overlap))
                inds_sectors = np.concatenate((inds_sectors[np.logical_not(in_overlap)],
                                               full_inds[inds_sectors][in_overlap][i_overlap]))

            # full results
            centers_unique = centers_unique_sectors
            inds = inds_sectors

        else:
            # loop through points, at each step removing any duplicates and shrinking our list
            # after looping through a point, it cannot be subsequently removed because the relations we are checking
            # are symmetric.
            # todo: is it possible this can fail in "average" mode?
            # todo: looks like this might be easier if use some tools from scipy.spatial, scipy.spatial.cKDTree
            counter = 0
            while counter < len(centers_unique):
                # compute distances to all other beads
                z_dists = np.abs(centers_unique[counter][0] - centers_unique[:, 0])
                xy_dists = np.sqrt((centers_unique[counter][1] - centers_unique[:, 1]) ** 2 +
                                   (centers_unique[counter][2] - centers_unique[:, 2]) ** 2)

                # beads which are close enough we will combine
                combine = np.logical_and(z_dists <= min_z_dist, xy_dists <= min_xy_dist)
                if mode == "average":
                    denom = np.nansum(np.logical_not(np.isnan(np.sum(centers_unique[combine], axis=1))) * weights[combine])
                    # compute new center from average and reset that position in the list
                    centers_unique[counter] = np.nansum(centers_unique[combine] * weights[combine][:, None], axis=0, dtype=float) / denom
                    weights[counter] = denom
                    combine[counter] = False
                elif mode == "keep-one":
                    # don't want to remove the point itself
                    combine[counter] = False
                elif mode == "remove":
                    pass
                else:
                    raise ValueError("mode must be 'average', 'keep-one', or 'remove' but was '%s'" % mode)

                # remove points from lists
                inds = inds[np.logical_not(combine)]
                centers_unique = centers_unique[np.logical_not(combine)]
                weights = weights[np.logical_not(combine)]

                counter += 1

    return centers_unique, inds


def localize2d(img: np.ndarray,
               mode: str = "radial-symmetry"):
    """
    Perform 2D localization using the radial symmetry approach of https://doi.org/10.1038/nmeth.2071

    :param img: 2D image of size ny x nx
    :param mode: 'radial-symmetry' or 'centroid'
    :return xc, yc:
    """
    if img.ndim != 2:
        raise ValueError("img must be a 2D array, but was %dD" % img.ndim)

    ny, nx = img.shape
    x = np.arange(nx)
    y = np.arange(ny)

    if mode == "centroid":
        xc = np.sum(img * x[None, :]) / np.sum(img)
        yc = np.sum(img * y[:, None]) / np.sum(img)
    elif mode == "radial-symmetry":
        # gradients taken at point between four pixels, i.e. (xk, yk) = (j + 0.5, i + 0.5)
        # using the Roberts cross operator
        yk = 0.5 * (y[:-1] + y[1:])
        xk = 0.5 * (x[:-1] + x[1:])
        # gradients along 45 degree rotated directions
        grad_uk = img[1:, 1:] - img[:-1, :-1]
        grad_vk = img[1:, :-1] - img[:-1, 1:]
        grad_xk = 1 / np.sqrt(2) * (grad_uk - grad_vk)
        grad_yk = 1 / np.sqrt(2) * (grad_uk + grad_vk)
        with np.errstate(invalid="ignore", divide="ignore"):
            # slope of the gradient at this point
            mk = grad_yk / grad_xk
            mk[np.isnan(mk)] = np.inf

            # compute weights by (1) increasing weight where gradient is large and (2) decreasing weight for points far away
            # from the centroid (as small slope errors can become large as the line is extended to the centroi)
            # approximate distance between (xk, yk) and (xc, yc) by assuming (xc, yc) is centroid of the gradient
            grad_norm = np.sqrt(grad_xk**2 + grad_yk**2)
            centroid_grad_norm_x = np.sum(xk[None, :] * grad_norm) / np.sum(grad_norm)
            centroid_grad_norm_y = np.sum(yk[:, None] * grad_norm) / np.sum(grad_norm)
            dk_centroid = np.sqrt((yk[:, None] - centroid_grad_norm_y)**2 + (xk[None, :] - centroid_grad_norm_x)**2)
            # weights
            wk = grad_norm**2 / dk_centroid

            # def chi_sqr(xc, yc):
            #     val = ((yk[:, None] - yc) - mk * (xk[None, :] - xc))**2 / (mk**2 + 1) * wk
            #     val[np.isinf(mk)] = (np.tile(xk[None, :], [yk.size, 1])[np.isinf(mk)] - xc)**2
            #     return np.sum(val)

            # line passing through through (xk, yk) with slope mk is y = yk + mk*(x - xk)
            # minimimum distance of points (xc, yc) is dk**2 = [(yk - yc) - mk*(xk -xc)]**2 / (mk**2 + 1)
            # must handle the case mk -> infinity separately. In this case dk**2 -> (xk - xc)**2
            # minimize chi^2 = \sum_k dk**2 * wk
            # minimizing for xc, yc gives a matrix equation
            # [[A, B], [C, D]] * [[xc], [yc]] = [[E], [F]]
            # in case the slope is infinite, need to take the limit of the sum manually
            summand_a = -mk ** 2 * wk / (mk ** 2 + 1)
            summand_a[np.isinf(mk)] = wk[np.isinf(mk)]
            A = np.sum(summand_a)

            summand_b = mk * wk / (mk**2 + 1)
            summand_b[np.isinf(mk)] = 0
            B = np.sum(summand_b)
            C = -B

            D = np.sum(wk / (mk**2 + 1))

            summand_e = (mk * wk * (yk[:, None] - mk * xk[None, :])) / (mk**2 + 1)
            summand_e[np.isinf(mk)] = - (wk * xk[None, :])[np.isinf(mk)]
            E = np.sum(summand_e)

            summand_f = (yk[:, None] - mk * xk[None, :]) * wk / (mk**2 + 1)
            summand_f[np.isinf(mk)] = 0
            F = np.sum(summand_f)

            xc = (D * E - B * F) / (A*D - B*C)
            yc = (-C * E + A * F) / (A*D - B*C)
    else:
        raise ValueError("mode must be 'centroid' or 'radial-symmetry', but was '%s'" % mode)

    return xc, yc


def localize3d(img: np.ndarray,
               mode: str = "radial-symmetry"):
    """
    Perform 3D localization using an extension of the radial symmetry approach of https://doi.org/10.1038/nmeth.2071

    :param img: 3D image of size nz x ny x nx
    :param str mode: 'radial-symmetry' or 'centroid'
    :return xc, yc, zc:
    """
    if img.ndim != 3:
        raise ValueError("img must be a 3D array, but was %dD" % img.ndim)

    nz, ny, nx = img.shape
    x = np.arange(nx)[None, None, :]
    y = np.arange(ny)[None, :, None]
    z = np.arange(nz)[:, None, None]

    if mode == "centroid":
        xc = np.sum(img * x) / np.sum(img)
        yc = np.sum(img * y) / np.sum(img)
        zc = np.sum(img * z) / np.sum(img)
    elif mode == "radial-symmetry":
        yk = 0.5 * (y[:, :-1, :] + y[:, 1:, :])
        xk = 0.5 * (x[:, :, :-1] + x[:, :, 1:])
        zk = 0.5 * (z[:-1] + z[1:])
        coords = (zk, yk, xk)

        # take a cube of 8 voxels, and compute gradients at the center, using the four pixel diagonals that pass
        # through the center
        grad_n1 = img[1:, 1:, 1:] - img[:-1, :-1, :-1]
        n1 = np.array([1, 1, 1]) / np.sqrt(3)  # vectors go [nz, ny, nx]
        grad_n2 = img[1:, :-1, 1:] - img[:-1, 1:, :-1]
        n2 = np.array([1, -1, 1]) / np.sqrt(3)
        grad_n3 = img[1:, :-1, :-1] - img[:-1, 1:, 1:]
        n3 = np.array([1, -1, -1]) / np.sqrt(3)
        grad_n4 = img[1:, 1:, :-1] - img[:-1, :-1, 1:]
        n4 = np.array([1, 1, -1]) / np.sqrt(3)

        # compute the gradient xyz components
        # 3 unknowns and 4 eqns, so use pseudo-inverse to optimize overdetermined system
        mat = np.concatenate((n1[None, :], n2[None, :], n3[None, :], n4[None, :]), axis=0)
        gradk = np.linalg.pinv(mat).dot(
            np.concatenate((grad_n1.ravel()[None, :], grad_n2.ravel()[None, :],
                            grad_n3.ravel()[None, :], grad_n4.ravel()[None, :]), axis=0))
        gradk = np.reshape(gradk, [3, zk.size, yk.size, xk.size])

        # compute weights by (1) increasing weight where gradient is large and (2) decreasing weight for points far away
        # from the centroid (as small slope errors can become large as the line is extended to the centroi)
        # approximate distance between (xk, yk) and (xc, yc) by assuming (xc, yc) is centroid of the gradient
        grad_norm = np.sqrt(np.sum(gradk**2, axis=0))
        centroid_gns = np.array([np.sum(zk * grad_norm), np.sum(yk * grad_norm), np.sum(xk * grad_norm)]) / np.sum(grad_norm)
        dk_centroid = np.sqrt((zk - centroid_gns[0]) ** 2 + (yk - centroid_gns[1]) ** 2 + (xk - centroid_gns[2]) ** 2)
        # weights
        wk = grad_norm ** 2 / dk_centroid

        # in 3D, parameterize a line passing through point Po along normal n by
        # V(t) = Pk + n * t
        # distance between line and point Pc minimized at
        # tmin = -\sum_{i=1}^3 (Pk_i - Pc_i) / \sum_i n_i^2
        # dk^2 = \sum_k \sum_i (Pk + n * tmin - Pc)^2
        # again, we want to minimize the quantity
        # chi^2 = \sum_k dk^2 * wk
        # so we take the derivatives of chi^2 with respect to Pc_x, Pc_y, and Pc_z, which gives a system of linear
        # equations, which we can recast into a matrix equation
        # np.array([[A, B, C], [D, E, F], [G, H, I]]) * np.array([[Pc_z], [Pc_y], [Pc_x]]) = np.array([[J], [K], [L]])
        nk = gradk / np.linalg.norm(gradk, axis=0)

        # def chi_sqr(xc, yc, zc):
        #     cs = (zc, yc, xc)
        #     chi = 0
        #     for ii in range(3):
        #         chi += np.sum((coords[ii] + nk[ii] * (cs[jj] - coords[jj]) - cs[ii]) ** 2 * wk)
        #     return chi

        # build 3x3 matrix from above
        mat = np.zeros((3, 3))
        for ll in range(3):  # rows of matrix
            for ii in range(3):  # columns of matrix
                if ii == ll:
                    mat[ll, ii] += np.sum(-wk * (nk[ii] * nk[ll] - 1))
                else:
                    mat[ll, ii] += np.sum(-wk * nk[ii] * nk[ll])

                for jj in range(3):  # internal sum
                    if jj == ll:
                        mat[ll, ii] += np.sum(wk * nk[ii] * nk[jj] * (nk[jj] * nk[ll] - 1))
                    else:
                        mat[ll, ii] += np.sum(wk * nk[ii] * nk[jj] * nk[jj] * nk[ll])

        # build vector from above
        vec = np.zeros((3, 1))
        coord_sum = zk * nk[0] + yk * nk[1] + xk * nk[2]
        for ll in range(3):  # sum over J, K, L
            for ii in range(3):  # internal sum
                if ii == ll:
                    vec[ll] += -np.sum((coords[ii] - nk[ii] * coord_sum) * (nk[ii] * nk[ll] - 1) * wk)
                else:
                    vec[ll] += -np.sum((coords[ii] - nk[ii] * coord_sum) * nk[ii] * nk[ll] * wk)

        # invert matrix
        zc, yc, xc = np.linalg.inv(mat).dot(vec)
    else:
        raise ValueError("mode must be 'centroid' or 'radial-symmetry', but was '%s'" % mode)

    return xc, yc, zc


# @profile
def fit_rois(img_rois: np.ndarray,
             coords_rois: tuple[np.ndarray, np.ndarray, np.ndarray],
             roi_sizes: np.ndarray[int],
             init_params: np.ndarray,
             max_number_iterations: int = 100,
             tolerance: Optional[float] = None,
             estimator: str = "LSE",
             fixed_params: Optional[np.ndarray] = None,
             guess_bounds: bool = False,
             use_gpu: bool = _gpufit_available,
             debug: bool = False,
             verbose: bool = False,
             model: psf.pixelated_psf_model = psf.gaussian3d_psf_model()) -> dict:
    """
    Fit rois to different model functions. Can use either CPU parallelization with dask or GPU parallelization
    using gpufit.

    For help cutting ROI's from an image and converting them to the correct format, use the helper function
    prepare_rois()

    :param img_rois: array of image rois of size nroi x nmax_roi_size. Each ROI should be flattened, padded
     with NaN's, and inserted along the 0th dimension.
    :param coords_rois: (z_rois, y_rois, ....) the coordinate arrays z_rois should have the same shape as img_rois
    :param roi_sizes: array giving size of each ROI
    :param init_params: initial parameters for fits, size nfits x model.nparams
    :param max_number_iterations: maximum number of iterations to be used for each fit
    :param tolerance: only for GPUFIT. Default is 1e-4.
    :param estimator: "LSE" or "MLE", only for GPUFIT
    :param model: "gaussian", "rotated-gaussian", "gaussian-lorentzian"
    :param fixed_params: For entries which are True, the fit function will force that parameter to be identical
      to the value in init_params. For entries which are False, the fit function will determine the optimal value.
      only supports fixing/unfixing each parameter for all fits
    :param guess_bounds: ((lower_bounds), (upper_bounds)) where lower_bounds and upper_bounds are each
      lists/tuples/arrays of length nparams
    :param use_gpu: whether to perform fitting on the GPU. If true, then GPUfit must be installed
    :param debug:
    :param verbose:
    :param model: model to use for PSF fitting. If doing this on the CPU, use implementations in fit_psf.py, otherwise
      model must have a corresponding version in GPU fit.
    :return fit_results: dictionary of fit results
    """

    if guess_bounds and use_gpu:
        warnings.warn("use_gpu selected for fitting, but unsupported option guess_bounds selected."
                      "guess_bounds will be ignored")

    zrois, yrois, xrois = coords_rois

    if not use_gpu:
        tstart = time.perf_counter()

        if debug:
            results = []
            for ii in range(len(img_rois)):
                results.append(model.fit(img_rois[ii],
                                         (zrois[ii], yrois[ii], xrois[ii]),
                                         init_params[ii],
                                         fixed_params=fixed_params,
                                         guess_bounds=guess_bounds,
                                         max_nfev=max_number_iterations)
                               )

        else:
            # forced to switch to dask form joblib because joblib use pickling to exchange info between process
            # and functions (which are arguments to fit_gauss_roi) are not pickle-able
            delayed = []
            for ii in range(len(img_rois)):
                delayed.append(dask.delayed(model.fit)(img_rois[ii],
                                                       (zrois[ii], yrois[ii], xrois[ii]),
                                                       init_params=init_params[ii],
                                                       fixed_params=fixed_params,
                                                       guess_bounds=guess_bounds,
                                                       max_nfev=max_number_iterations)
                               )

            if verbose:
                with ProgressBar():
                    results = dask.compute(*delayed)
            else:
                results = dask.compute(*delayed)

        fit_t = (time.perf_counter() - tstart)
        fit_params = np.asarray([r["fit_params"] for r in results])
        chi_sqrs = np.asarray([r["chi_squared"] for r in results])
        fit_states = np.asarray([r["status"] for r in results])
        niters = np.asarray([r["nfev"] for r in results])
        fit_states_key = results[0]["status_codes"]

    else:
        if model.sf != 1:
            raise NotImplementedError("sampling factors other than 1 are not implemented for GPU fitting")

        # resolve GPUfit model
        models_mapping = ((psf.gaussian3d_psf_model, gf.ModelID.GAUSS_3D_ARB),
                          (psf.gaussian_lorentzian_psf_model, gf.ModelID.GAUSS_LOR_3D_ARB),
                          (psf.gaussian3d_asymmetric_rotated_pixelated, gf.ModelID.GAUSS_3D_ROT_ARB),
                          (psf.gaussian3d_asymmetric_pixelated, gf.ModelID.GAUSS_3D_ASYM_ARB)
                          )
        model_id = None
        for mod, mod_gpu in models_mapping:
            if isinstance(model, mod):
                model_id = mod_gpu

        if model_id is None:
            raise NotImplementedError(f"model of type {type(model)} has not been implemented in gpufit."
                                      f"The models which have been implemented are {[a for a, b in models_mapping]}")

        # build GPUfit data
        data = img_rois.astype(np.float32)
        nfits, n_pts_per_fit = data.shape

        # build user data
        coords = np.stack((xrois, yrois, zrois), axis=1).ravel()
        user_info = np.concatenate((coords.astype(np.float32),
                                    roi_sizes.astype(np.float32)))

        # some models have extra non-fit parameters appended at end of user_info
        if model_id == gf.ModelID.GAUSS_3D_ARB or \
           model_id == gf.ModelID.GAUSS_3D_ROT_ARB or \
           model_id == gf.ModelID.GAUSS_3D_ASYM_ARB:

            user_info = np.concatenate((user_info,
                                        np.array(model.minimum_sigmas).astype(np.float32)))

        # initial parameters
        init_params = init_params.astype(np.float32)

        nparams = model.nparams

        # check arguments
        if data.ndim != 2:
            raise ValueError(f"data.ndim should = 2 but was {data.ndim:d}")
        if init_params.ndim != 2 or init_params.shape != (nfits, nparams):
            raise ValueError(f"init_params should have shape ({nfits:d}, {nparams:d}), but had shape {init_params.shape}")
        # todo: this now depends on the model
        # if user_info.ndim != 1 or user_info.size != (3 * nfits * n_pts_per_fit + nfits):
        #     raise ValueError(f"user_info should have size ({3 * nfits * n_pts_per_fit + nfits:d}), but had size {user_info.size:d}")

        if estimator == "MLE":
            est_id = gf.EstimatorID.MLE
        elif estimator == "LSE":
            est_id = gf.EstimatorID.LSE
        else:
            raise ValueError(f"'estimator' must be 'MLE' or 'LSE' but was '{estimator:s}'")

        # set which parameters to fit/fix
        if fixed_params is None:
            fixed_params = np.zeros(nparams, dtype=bool)

        params_to_fit = np.logical_not(np.array(fixed_params)).astype(np.int32)

        # do fitting
        fit_params, fit_states, chi_sqrs, niters, fit_t = gf.fit(data,
                                                                 None,
                                                                 model_id,
                                                                 init_params,
                                                                 tolerance=tolerance,
                                                                 max_number_iterations=max_number_iterations,
                                                                 estimator_id=est_id,
                                                                 parameters_to_fit=params_to_fit,
                                                                 user_info=user_info)

        # defined in Gpufit/constants.h
        fit_states_key = {"converged": 0,
                          "max_iteration": 1,
                          "singular_hessian": 2,
                          "neg_curvature_mle": 3,
                          "gpu_not_ready": 4}

    # ensure e.g. Gaussian sigmas are > 0
    fit_params = model.normalize_parameters(fit_params)

    # collect results
    fit_results = {"fit_params": fit_params,
                   "init_params": init_params,
                   "fit_states": fit_states,
                   "fit_states_key": fit_states_key,
                   "chi_sqrs": chi_sqrs,
                   "niters": niters,
                   "fit_time": fit_t,
                   "model": repr(model)}

    return fit_results


def plot_fit_roi(fit_params: list[float],
                 roi: list[int],
                 imgs: np.ndarray,
                 coords: Optional[tuple[np.ndarray]] = None,
                 init_params: Optional[np.ndarray] = None,
                 model: psf.pixelated_psf_model = psf.gaussian3d_psf_model(),
                 string: Optional[str] = None,
                 same_color_scale: bool = True,
                 vmin: Optional[float] = None,
                 vmax: Optional[float] = None,
                 cmap="bone",
                 gamma: float = 1.,
                 scale_z_display: float = 1.,
                 figsize: tuple[float] = (16, 8),
                 prefix: str = "",
                 save_dir: Optional[str] = None):
    """
    Plot results obtained from fitting functions fit_gauss_roi() or fit_gauss_rois()

    :param fit_params:
    :param roi: [zstart, zend, ystart, yend, xstart, xend]
    :param imgs: full image, such that imgs[zstart:zend, ystart:yend, xstart:xend] is the region that was fit
    :param coords: (z, y, x) broadcastable to same size as imgs
    :param init_params: initial parameters used in fit, optional
    :param model:
    :param string:
    :param same_color_scale: whether to use same color scale for data and fits
    :param vmin:
    :param vmax:
    :param cmap:
    :param gamma:
    :param scale_z_display:
    :param figsize: (sx, sz)
    :param prefix: prefix prepended before save name
    :param save_dir: if None, do not save results
    :return figh:
    """

    nz, ny, nx = imgs.shape
    if coords is None:
        coords = np.meshgrid(range(nz), range(ny), range(nx), indexing="ij")

    z, y, x = coords
    # extract useful coordinate info
    dc = x[0, 0, 1] - x[0, 0, 0]

    if z.shape[0] > 1:
        dz = z[1, 0, 0] - z[0, 0, 0]
    else:
        # dz = dc
        dz = dc * (roi[5] - roi[4] + 1) / 10

    if init_params is not None:
        center_guess = np.array([init_params[3], init_params[2], init_params[1]])

    center_fit = np.array([fit_params[3], fit_params[2], fit_params[1]])

    # get ROI and coordinates
    img_roi = roi_fns.cut_roi(roi, imgs)[0]
    x_roi = roi_fns.cut_roi(roi, x)[0]
    y_roi = roi_fns.cut_roi(roi, y)[0]
    z_roi = roi_fns.cut_roi(roi, z)[0]

    if vmin is None:
        vmin = np.percentile(img_roi[np.logical_not(np.isnan(img_roi))], 1)

    if vmax is None:
        vmax = np.percentile(img_roi[np.logical_not(np.isnan(img_roi))], 99.9)

    # git fit
    img_fit = model.model((z_roi, y_roi, x_roi), fit_params)

    # set extents
    extent_xy = [x_roi[0, 0, 0] - 0.5 * dc, x_roi[0, 0, -1] + 0.5 * dc,
                 y_roi[0, -1, 0] + 0.5 * dc, y_roi[0, 0, 0] - 0.5 * dc]

    extent_xz = [x_roi[0, 0, 0] - 0.5 * dc, x_roi[0, 0, -1] + 0.5 * dc,
                 z_roi[-1, 0, 0] + 0.5 * dz, z_roi[0, 0, 0] - 0.5 * dz]

    extent_zy = [z_roi[0, 0, 0] - 0.5 * dz, z_roi[-1, 0, 0] + 0.5 * dz,
                 y_roi[0, -1, 0] + 0.5 * dc, y_roi[0, 0, 0] - 0.5 * dc]

    wx = extent_xy[1] - extent_xy[0]
    wy = extent_xy[2] - extent_xy[3]
    wz = extent_xz[2] - extent_xz[3]

    # ################################
    # plot results interpolated on regular grid
    # ################################
    figh_interp = plt.figure(figsize=figsize)
    st_str = f"Fit, max projections, interpolated, ROI = {roi}"

    st_str += f"\n{'fit': <10}" + ", ".join([f"{model.parameter_names[ii]:s}={fit_params[ii]:3.4f}" for ii in range(len(fit_params))])
    if init_params is not None:
        st_str += f"\n{'guess': <10}" + ", ".join([f"{model.parameter_names[ii]:s}={init_params[ii]:3.4f}" for ii in range(len(fit_params))])

    if string is not None:
        st_str += "\n" + string

    figh_interp.suptitle(st_str)

    grid = figh_interp.add_gridspec(nrows=2, height_ratios=[1, wz / wy * scale_z_display], hspace=0,
                                    ncols=7, width_ratios=[wz / wx * scale_z_display, 1, 0.2, wz / wx * scale_z_display, 1, 0.2, 0.2], wspace=0)

    # ################################
    # XY, data
    # ################################
    ax = figh_interp.add_subplot(grid[0, 1])
    im = ax.imshow(np.nanmax(img_roi, axis=0),
                   extent=extent_xy,
                   cmap=cmap,
                   norm=PowerNorm(vmin=vmin, vmax=vmax, gamma=gamma))

    ax.plot(center_fit[2], center_fit[1], 'm+')
    if init_params is not None:
        ax.plot(center_guess[2], center_guess[1], 'gx')

    ax.set_ylim(extent_xy[2:4])
    ax.set_xlim(extent_xy[0:2])
    ax.set_xticks([])
    ax.set_yticks([])

    # ################################
    # XZ, data
    # ################################
    ax = figh_interp.add_subplot(grid[1, 1])

    ax.imshow(np.nanmax(img_roi, axis=1),
              extent=extent_xz,
              cmap=cmap,
              norm=PowerNorm(vmin=vmin, vmax=vmax, gamma=gamma))
    ax.plot(center_fit[2], center_fit[0], 'm+')
    if init_params is not None:
        ax.plot(center_guess[2], center_guess[0], 'gx')

    ax.set_ylim(extent_xz[2:4])
    ax.set_xlim(extent_xz[0:2])

    ax.set_xlabel("X (um)")

    # ################################
    # YZ, data
    # ################################
    ax = figh_interp.add_subplot(grid[0, 0])
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')

        ax.imshow(np.nanmax(img_roi, axis=2).transpose(),
                  extent=extent_zy,
                  cmap="bone",
                  norm=PowerNorm(vmin=vmin, vmax=vmax, gamma=gamma))

    ax.plot(center_fit[0], center_fit[1], 'm+')
    if init_params is not None:
        ax.plot(center_guess[0], center_guess[1], 'gx')

    ax.set_ylim(extent_zy[2:4])
    ax.set_xlim([extent_zy[1], extent_zy[0]])
    ax.set_xlabel("Z (um)")
    ax.set_ylabel("Y (um)")

    if same_color_scale:
        vmin_fit = vmin
        vmax_fit = vmax
    else:
        vmin_fit = np.percentile(img_fit, 1)
        vmax_fit = np.percentile(img_fit, 99.9)

    # ################################
    # YX, fit
    # ################################
    ax = figh_interp.add_subplot(grid[0, 4])

    ax.imshow(np.nanmax(img_fit, axis=0),
              extent=extent_xy,
              cmap=cmap,
              norm=PowerNorm(vmin=vmin_fit, vmax=vmax_fit, gamma=gamma))

    ax.plot(center_fit[2], center_fit[1], 'm+')
    if init_params is not None:
        ax.plot(center_guess[2], center_guess[1], 'gx')

    ax.set_ylim(extent_xy[2:4])
    ax.set_xlim(extent_xy[0:2])
    ax.set_xticks([])
    ax.set_yticks([])

    # ################################
    # ZX, fit
    # ################################
    ax = figh_interp.add_subplot(grid[1, 4])

    ax.imshow(np.nanmax(img_fit, axis=1),
              extent=extent_xz,
              cmap=cmap,
              norm=PowerNorm(vmin=vmin_fit, vmax=vmax_fit, gamma=gamma))

    ax.plot(center_fit[2], center_fit[0], 'm+')
    if init_params is not None:
        ax.plot(center_guess[2], center_guess[0], 'gx')

    ax.set_ylim(extent_xz[2:4])
    ax.set_xlim(extent_xz[0:2])

    ax.set_xlabel("X (um)")

    # ################################
    # YZ, fit
    # ################################
    ax = figh_interp.add_subplot(grid[0, 3])

    ax.imshow(np.nanmax(img_fit, axis=2).transpose(),
              extent=extent_zy,
              cmap=cmap,
              norm=PowerNorm(vmin=vmin_fit, vmax=vmax_fit, gamma=gamma))
    ax.plot(center_fit[0], center_fit[1], 'm+')

    if init_params is not None:
        ax.plot(center_guess[0], center_guess[1], 'gx')

    ax.set_ylim(extent_zy[2:4])
    ax.set_xlim([extent_zy[1], extent_zy[0]])

    ax.set_xlabel("Z (um)")
    ax.set_ylabel("Y (um)")

    # ################################
    # colorbar
    # ################################
    ax = figh_interp.add_subplot(grid[:, -1])
    plt.colorbar(im, cax=ax)

    # ################################
    # saving
    # ################################
    if save_dir is not None:
        figh_interp.savefig(Path(save_dir) / f"{prefix:s}.png")
        plt.close(figh_interp)

    return figh_interp


# filter fit parameters
class filter:
    """
    Filter fit results to identfy "good" and "bad" fits
    """
    def __init__(self, fn, condition_names):
        self._fn = fn
        self.condition_names = condition_names

    def filter(self, fit_params, *args, **kwargs):
        """
        Filter function requis parameters and can optionally accept other arguments, e.g. if want
        to compare fit_params with initial parameters or etc. Filter functions are free to ignore all
        parameters besides fit_params, but must accept arbitrary number of arguments.

        :param fit_params:
        :param args: additional arguments which must all be arrays and must all have first dimensions of the same length
        :param kwargs: key-word arguments, which should be objects which apply to all points. This matter mostly
          if you want to us the __mul__ method. When applying filters where order matters, each element of *args
          will be reduced to an array which only keeps those elements that passed the previous filter, but
          **kwargs will not be touched
        :return:
        """

        if fit_params.ndim == 1:
            fit_params = np.expand_dims(fit_params, axis=0)

        conditions = self._fn(fit_params, *args, **kwargs)

        if conditions.ndim == 1:
            conditions = np.expand_dims(conditions, axis=0)

        return conditions

    def __add__(self, other):
        """
        return new filter which concatenates the results from two other filters

        :param other:
        :return:
        """

        def both_filter(fit_params, *args, **kwargs):
            c1 = self.filter(fit_params, *args, **kwargs)
            c2 = other.filter(fit_params, *args, **kwargs)

            conditions = np.concatenate((c1, c2), axis=1)

            return conditions

        return filter(both_filter, self.condition_names + other.condition_names)

    def __mul__(self, other):
        """
        apply rightmost filter first, then apply left filter to only those points which succeeded.

        :param other:
        :return:
        """

        def sequential_filter(fit_params, *args, **kwargs):
            c1 = other.filter(fit_params, *args, **kwargs)
            tk1 = np.logical_and.reduce(c1, axis=1)

            args_red = [a[tk1] for a in args]
            c2_reduced = self.filter(fit_params[tk1], *args_red, **kwargs)

            # set array on full space
            c2 = np.zeros((c1.shape[0], c2_reduced.shape[1]), dtype=bool)
            c2[tk1, :] = c2_reduced

            conditions = np.concatenate((c1, c2), axis=1)

            return conditions

        return filter(sequential_filter, other.condition_names + self.condition_names)


class no_filter(filter):
    """
    Filter which accepts all entries
    """
    def __init__(self):
        self.condition_names = ["none"]

    def filter(self, fit_params, *args, **kwargs):
        conditions = np.ones((len(fit_params), 1), dtype=bool)
        return conditions


class range_filter(filter):
    """
    Filter based on value being in a certain range
    """
    def __init__(self,
                 low: float,
                 high: float,
                 index: int,
                 name: str):

        self.low = low
        self.high = high
        self.index = index
        self.condition_names = [f"{name:s} too small", f"{name:s} too large"]

    def filter(self, fit_params, *args, **kwargs):
        conditions = np.stack((fit_params[:, self.index] >= self.low,
                               fit_params[:, self.index] <= self.high), axis=1)

        return conditions


class proximity_filter(filter):
    """
    Filter spots based on proximity to some other array. e.g. based on the distance from the initial guess position
    to the final fit position
    """
    def __init__(self, indices, min_dist, max_dist, name):
        self.indices = tuple(indices)
        self.min_dist = min_dist
        self.max_dist = max_dist
        self.condition_names = [f"{name:s} deviation too small", f"{name:s} deviation too large"]

    def filter(self, fit_params, ref_params, *args, **kwargs):
        dists = np.linalg.norm(ref_params[:, self.indices] - fit_params[:, self.indices], axis=1)
        conditions = np.stack((dists >= self.min_dist, dists <= self.max_dist), axis=1)

        return conditions


class unique_filter(filter):
    """
    Filter spots so that only keep one in a certain area, to avoid one spot being picked up multiple times by max filter
    """
    def __init__(self,
                 dxy_min_dist: float,
                 dz_min_dist: float,
                 name="not unique",
                 center_indices: tuple[int] = (3, 2, 1)):
        """

        :param dxy_min_dist:
        :param dz_min_dist:
        :param name:
        :param center_indices: indices of cz, cy, cx in model respectively
        """

        self.dxy_min_dist = dxy_min_dist
        self.dz_min_dist = dz_min_dist
        self.center_indices = center_indices
        self.condition_names = [f"{name:s}"]

    def filter(self, fit_params, *args, **kwargs):
        """

        :param fit_params: we assume that cx = fit_params[:, 1], cy = fit_params[:, 2], cz = fit_params[:, 3]
        :param args:
        :param kwargs:
        :return:
        """
        _, unique_inds = filter_nearby_peaks(fit_params[:, self.center_indices],
                                             self.dxy_min_dist,
                                             self.dz_min_dist,
                                             mode="keep-one")
        conditions = np.zeros((fit_params.shape[0], 1), dtype=bool)
        conditions[unique_inds] = True

        return conditions


def get_param_filter(coords: Sequence[np.ndarray],
                     fit_dist_max_err: Sequence[float] = (np.inf, np.inf),
                     min_spot_sep: Sequence[float] = (0., 0.),
                     sigma_bounds: tuple[Sequence[float], Sequence[float]] = ((0., 0.), (np.inf, np.inf)),
                     amp_bounds: Sequence[float] = (0., np.inf),
                     dist_boundary_min: Sequence[float] = (0., 0.)):
    """
    Simple composite filter testing bounds of fit parameters

    :param coords: (z, y, x)
    :param fit_dist_max_err: (dmax_z, dmax_xy) maximum distance between fit points allowed.
    :param min_spot_sep: (dz_min, dxy_min)
    :param sigma_bounds: ((sz_min, sxy_min), (sz_max, sxy_max))
    :param amp_bounds: (amp_min, amp_max)
    :param dist_boundary_min: (dz_min, dxy_min)
    :return filter:
    """

    z, y, x = coords

    # in bounds
    dz_min, dxy_min = dist_boundary_min

    filter_in_bounds = range_filter(x.min() + dxy_min, x.max() - dxy_min, 1, "x-position") + \
                       range_filter(y.min() + dxy_min, y.max() - dxy_min, 2, "y-position") + \
                       range_filter(z.min() + dz_min, z.max() - dz_min, 3, "z-position")

    # size
    (sz_min, sxy_min), (sz_max, sxy_max) = sigma_bounds

    filter_size = range_filter(sxy_min, sxy_max, 4, "xy-size") + \
                  range_filter(sz_min, sz_max, 5, "z-size")

    # amplitude
    filter_amp = range_filter(amp_bounds[0], amp_bounds[1], 0, "amplitude")

    # proximity to initial guess
    # maximum distance between fit center and guess center
    z_err_fit_max, xy_fit_err_max = fit_dist_max_err

    filter_proximity = proximity_filter((1, 2), 0, xy_fit_err_max, "xy") + \
                       proximity_filter((3,), 0, z_err_fit_max, "z")

    #
    dz, dxy = min_spot_sep
    filter_unique = unique_filter(dxy, dz)

    # full filter
    filter_combined = filter_unique * (filter_in_bounds + filter_size + filter_amp + filter_proximity)

    return filter_combined


def get_param_filter_model(model: psf.pixelated_psf_model,
                           fit_dist_max_err: Sequence[float],
                           min_spot_sep: Sequence[float] = (0, 0),
                           param_bounds: Optional[tuple[Sequence[float], Sequence[float]]] = None,
                           center_param_inds: tuple[int] = (3, 2, 1),
                           ):
    """
    Simple composite filter testing bounds of fit parameters

    :param model: fit model
    :param fit_dist_max_err:
    :param min_spot_sep:
    :param param_bounds:
    :param center_param_inds:
    :return:
    """

    filter = no_filter()

    # filter on parameters
    for ii in range(model.nparams):
        filter += range_filter(param_bounds[0][ii], param_bounds[1][ii], ii, model.parameter_names[ii])

    # proximity to initial guess
    # maximum distance between fit center and guess center
    z_err_fit_max, xy_fit_err_max = fit_dist_max_err

    filter += proximity_filter(center_param_inds[1:], 0, xy_fit_err_max, "xy") + \
              proximity_filter((center_param_inds[0],), 0, z_err_fit_max, "z")

    # spots separated by distance
    dz, dxy = min_spot_sep
    filter_unique = unique_filter(dxy, dz, center_indices=center_param_inds)

    filter = filter_unique * filter

    return filter


def filter_localizations(fit_params: np.ndarray,
                         init_params: np.ndarray,
                         coords: Sequence[np.ndarray],
                         fit_dist_max_err: Sequence[float],
                         min_spot_sep: Sequence[float],
                         sigma_bounds: tuple[Sequence[float], Sequence[float]],
                         amp_min: float = 0,
                         dist_boundary_min: Sequence[float] = (0, 0)):
    """
    Given a list of fit parameters, return boolean arrays indicating which fits pass/fail given a variety
    of tests for reasonability

    :param fit_params: nfits x 7, results of fitting where the parameters are
      [amplitude, center x, center y, center z, size xy, size z, background]. Most commonly these values come from
      a 3D Gaussian fit. But other models can be used as well.
    :param init_params: nfits x 7, initial guess parameters used in fits
    :param coords: (z, y, x)
    :param fit_dist_max_err: (dz_max, dxy_max) maximum distance between guess value and fit value
    :param min_spot_sep: (dz, dxy) assume points separated by less than this distance come from one spot
    :param sigma_bounds: ((sz_min, sxy_min), (sz_max, sxy_max)) exclude fits with sigmas that fall outside
      these ranges
    :param amp_min: exclude fits with smaller amplitude
    :param dist_boundary_min: (dz_min, dxy_min)
    :return: (to_keep, conditions, condition_names, filter_settings)
    """
    # todo: deprecate this and replace with the filter class objects as did in localize_beads_generic

    filter_settings = {"fit_dist_max_err": fit_dist_max_err,
                       "min_spot_sep": min_spot_sep,
                       "sigma_bounds": sigma_bounds,
                       "amp_min": amp_min,
                       "dist_boundary_min": dist_boundary_min}

    z, y, x = coords
    centers_guess = init_params[:, (3, 2, 1)]
    centers_fit = fit_params[:, (3, 2, 1)]

    # ###################################################
    # only keep points if size and position were reasonable
    # ###################################################
    dz_min, dxy_min = dist_boundary_min

    in_bounds = np.logical_and.reduce((fit_params[:, 1] >= x.min() + dxy_min,
                                       fit_params[:, 1] <= x.max() - dxy_min,
                                       fit_params[:, 2] >= y.min() + dxy_min,
                                       fit_params[:, 2] <= y.max() - dxy_min,
                                       fit_params[:, 3] >= z.min() + dz_min,
                                       fit_params[:, 3] <= z.max() - dz_min))

    # maximum distance between fit center and guess center
    z_err_fit_max, xy_fit_err_max = fit_dist_max_err
    center_close_to_guess_xy = np.sqrt((centers_guess[:, 2] - fit_params[:, 1])**2 +
                                       (centers_guess[:, 1] - fit_params[:, 2])**2) <= xy_fit_err_max
    center_close_to_guess_z = np.abs(centers_guess[:, 0] - fit_params[:, 3]) <= z_err_fit_max

    # maximum/minimum sigmas AND combine all conditions
    (sz_min, sxy_min), (sz_max, sxy_max) = sigma_bounds
    conditions = np.stack((in_bounds,
                           center_close_to_guess_xy,
                           center_close_to_guess_z,
                           fit_params[:, 4] <= sxy_max,
                           fit_params[:, 4] >= sxy_min,
                           fit_params[:, 5] <= sz_max,
                           fit_params[:, 5] >= sz_min,
                           fit_params[:, 0] >= amp_min), axis=1)

    condition_names = ["in_bounds",
                       "center_close_to_guess_xy",
                       "center_close_to_guess_z",
                       "xy_size_small_enough",
                       "xy_size_big_enough",
                       "z_size_small_enough",
                       "z_size_big_enough",
                       "amp_ok"]

    to_keep_temp = np.logical_and.reduce(conditions, axis=1)

    # ###################################################
    # check for unique points
    # ###################################################

    dz, dxy = min_spot_sep
    if np.sum(to_keep_temp) > 0:

        # only keep unique center if close enough
        _, unique_inds = filter_nearby_peaks(centers_fit[to_keep_temp], dxy, dz, mode="keep-one")

        # unique mask for those in to_keep_temp
        is_unique = np.zeros(np.sum(to_keep_temp), dtype=bool)
        is_unique[unique_inds] = True

        # get indices of non-unique points among all points
        not_unique_inds_full = np.arange(len(to_keep_temp), dtype=int)[to_keep_temp][np.logical_not(is_unique)]

        # get mask in full space
        unique = np.ones(len(fit_params), dtype=bool)
        unique[not_unique_inds_full] = False
    else:
        unique = np.ones(len(fit_params), dtype=bool)

    conditions = np.concatenate((conditions, np.expand_dims(unique, axis=1)), axis=1)
    condition_names += ["unique"]
    to_keep = np.logical_and(to_keep_temp, unique)

    return to_keep, conditions, condition_names, filter_settings

# @profile
def localize_beads_generic(imgs: np.ndarray,
                           drs: tuple[float],
                           threshold: float,
                           roi_size: Sequence[float] = (4., 2., 2.),
                           filter_sigma_small: Sequence[float] = (1., 0.1, 0.1),
                           filter_sigma_large: Sequence[float] = (10., 5., 5.),
                           min_spot_sep: Sequence[float] = (0., 0.),
                           filter: Optional[filter] = None,
                           mask: Optional[np.ndarray] = None,
                           average_duplicates_before_fit: bool = True,
                           max_nfit_iterations: int = 100,
                           fit_filtered_images: bool = False,
                           use_gpu_fit: bool = _gpufit_available,
                           use_gpu_filter: bool = _cupy_available,
                           verbose: bool = True,
                           model: psf.pixelated_psf_model = psf.gaussian3d_psf_model(),
                           guess_bounds: bool = False,
                           debug: bool = False,
                           return_filtered_images: bool = False,
                           model_zsize_index: int = 5,
                           model_zposition_index: int = 3,
                           **kwargs):
    """
    Given an image consisting of diffraction limited features and background, identify the diffraction limited features
    using the following procedure:
    (1) Obtain a filtered image using a difference-of-Gaussians filter
    (2) Identify candidate spots from the filtered image using a threshold and maximum filter
    (3) Fit candidate spots to a 2D or 3D Gaussian function. Note the fitting is done on the raw image, not the
    filtered image
    (4) Filter out likely candidate spots based on the results of the fitting
    the various parameters used in this function are set in terms of real units, i.e. um, and not pixels. To use
    pixel units, set dxy=dz=1

    :param imgs: an image of size ny x nx or an image stack of size nz x ny x nx
    :param drs: (dz, dy, dx))
    :param threshold: threshold used for identifying spots. This is applied after filtering of image
    :param roi_size: (sz, sy, sx) in um
    :param filter_sigma_small: (sz, sy, sx) small sigmas to be used in difference-of-Gaussian filter. Roughly speaking,
      features which are smaller than these sigmas will be high pass filtered out. To turn off this filter, set to None
    :param filter_sigma_large: (sz, sy, sx) large sigmas to be used in difference-of-Gaussian filter. Roughly speaking,
      features which are larger than these sigmas will be low pass filtered out. To turn off this filter, set to None
    :param min_spot_sep: (dz, dxy) minimum separation allowed between adjacent peaks. This is used to determine
      (1) the size of the maximum filter used to identify spot candidates and (2) as the threshold distance for
      combining two nearby spots. These must be larger than the pixel sizes along the corresponding dimensions
    :param filter: filter will be applied with args = [fit_params, ref_params, chi_sqrs, niters, rois]
      and kwargs "image" and "image_filtered"
    :param mask: optionally boolean array of same size as image which indicates where to search for peaks
    :param average_duplicates_before_fit: test if points are "unique" within region defined by min_spot_sep before fitting
    :param max_nfit_iterations: maximum number of iterations in fitting function
    :param fit_filtered_images: whether to perform fitting on raw images or filtered images
    :param use_gpu_fit: whether to do spot fitting on the GPU
    :param use_gpu_filter: whether to do difference-of-Gaussian filtering on GPU
    :param verbose: whether to print information
    :param model: an instance of a class derived from psf.psf_model. The model describes the PSF and how to 'fit' data
      to it. Information e.g. about pixelation can be provided to the model. See psf.psf_model and the derived classes
      for more details
    :param guess_bounds: whether to use bounds for each ROI guessed from the coordinates. If so, will use
      bound guesses from model.estimate_bounds().
    :param debug:
    :param return_filtered_images:
    :param **kwargs: passed through to fit_rois() function
    :return coords, fit_results, imgs_filtered: coords = (z, y, x)
    """

    # ###################################
    # process/regularize input parameters
    # ###################################
    # check if is 2D
    data_is_2d = (imgs.ndim == 2)

    if imgs.ndim == 2:
        imgs = np.expand_dims(imgs, axis=0)

    if mask is not None and mask.ndim == 2:
        mask = np.expand_dims(mask, axis=0)

    dz, dy, dx = drs
    if data_is_2d:
        dz = 1  # dz not meaningful in this case

    roi_size = np.array(roi_size, copy=True)
    if data_is_2d:
        roi_size[0] = 1

    min_spot_sep = np.array(min_spot_sep, copy=True)
    if data_is_2d:
        min_spot_sep[0] = 0

    if filter_sigma_large is not None:
        filter_sigma_large = np.array(filter_sigma_large, copy=True)
        if data_is_2d:
            filter_sigma_large[0] = 0

    if filter_sigma_small is not None:
        filter_sigma_small = np.array(filter_sigma_small, copy=True)
        if data_is_2d:
            filter_sigma_small[0] = 0

    # unpack arguments
    z, y, x = get_coords(imgs.shape, (dz, dy, dx))
    dz_min_sep, dxy_min_sep = min_spot_sep
    roi_size_pix = roi_fns.get_roi_size(roi_size, [dz, dy, dx], ensure_odd=True)

    if dz_min_sep < dz and not data_is_2d:
        raise ValueError(f"minimum separation along the z-direction was {dz_min_sep:.2f},"
                         f" but this must be greater than the z-pixel size={dz:.2f}")

    if dxy_min_sep < dx or dxy_min_sep < dy:
        raise ValueError(f"minimum separation along the xy-direction was {dxy_min_sep:.2f},"
                         f" but this must be greater than the x- and y-pixel sizes=({dx:.2f}, {dy:.2f})")

    # ###################################
    # filter images
    # ###################################
    tstart = time.perf_counter()

    if use_gpu_filter:
        xp = cp
    else:
        xp = np

    imgs = xp.asarray(imgs)

    if filter_sigma_small is not None:
        ks = xp.asarray(get_filter_kernel(filter_sigma_small, (dz, dy, dx)))
        imgs_hp = filter_convolve(imgs, ks)
    else:
        imgs_hp = xp.asarray(imgs)

    if filter_sigma_large is not None:
        kl = xp.asarray(get_filter_kernel(filter_sigma_large, (dz, dy, dx)))
        imgs_lp = filter_convolve(imgs, kl)
    else:
        imgs_lp = 0

    imgs_filtered = imgs_hp - imgs_lp

    if verbose:
        print(f"filtered image in {time.perf_counter() - tstart:.2f}s")

    # ###################################
    # identify candidate beads
    # ###################################
    tstart = time.perf_counter()

    footprint = get_max_filter_footprint((dz_min_sep, dxy_min_sep, dxy_min_sep), (dz, dy, dx))
    centers_guess_inds, _ = find_peak_candidates(imgs_filtered, footprint, threshold, mask=mask)

    if use_gpu_filter and _cupy_available:
        imgs = imgs.get()
        imgs_filtered = imgs_filtered.get()
        centers_guess_inds = centers_guess_inds.get()

    # real coordinates
    centers_guess = np.stack((z[centers_guess_inds[:, 0], 0, 0],
                              y[0, centers_guess_inds[:, 1], 0],
                              x[0, 0, centers_guess_inds[:, 2]]),
                             axis=1)

    if verbose:
        print(f"identified {len(centers_guess_inds):d} candidates"
              f" in {time.perf_counter() - tstart:.2f}s")

    # ###################################
    # identify candidate beads
    # ###################################
    if len(centers_guess_inds) != 0:

        if average_duplicates_before_fit:
            # ###################################################
            # average multiple points too close together. Necessary bc if naive threshold, may identify several points
            # from same spot. Particularly important if spots have very different brightness levels.
            # ###################################################
            tstart = time.perf_counter()

            inds = np.ravel_multi_index(centers_guess_inds.transpose(), imgs_filtered.shape)
            weights = imgs_filtered.ravel()[inds]
            centers_guess, inds_comb = filter_nearby_peaks(centers_guess,
                                                           dxy_min_sep,
                                                           dz_min_sep,
                                                           weights=weights,
                                                           mode="average",
                                                           nmax=np.inf)

            if verbose:
                print(f"Found {len(centers_guess):d} points separated by"
                      f" dxy > {dxy_min_sep:0.5g} and dz > {dz_min_sep:0.5g}"
                      f" in {time.perf_counter() - tstart:.1f}s")

        # ###################################################
        # prepare ROIs
        # ###################################################
        tstart = time.perf_counter()

        if fit_filtered_images:
            imgs_fit = imgs_filtered
        else:
            imgs_fit = imgs

        roi_centers = get_nearest_pixel(centers_guess, (dz, dy, dx))
        rois = roi_fns.get_centered_rois(roi_centers, roi_size_pix, [0, 0, 0], imgs_fit.shape)
        img_rois, roi_coords, roi_sizes = prepare_rois(imgs_fit, (z, y, x), rois)

        # ###################################################
        # determine initial guess values for fits
        # ###################################################
        # init_params = np.stack([model.estimate_parameters(img_rois[ii], (zrois[ii], yrois[ii], xrois[ii]))
        #                         for ii in range(len(img_rois))], axis=0)
        init_params = model.estimate_parameters(img_rois, roi_coords, num_preserved_dims=1)

        if np.any(np.isnan(init_params)):
            raise ValueError("one or more init_params was NaN")

        if verbose:
            print(f"Prepared {len(rois):d} rois and estimated initial parameters"
                  f" in {time.perf_counter() - tstart:.2f}s")

        # ###################################################
        # perform localization fitting
        # ###################################################
        if verbose:
            print(f"starting fitting for {centers_guess.shape[0]:d} rois")
        tstart = time.perf_counter()

        # fix some parameters if desired
        fixed_params = [False] * model.nparams
        # if 2D, don't want to fit cz or sz
        if data_is_2d:
            if model.parameter_names[model_zsize_index] != "sz":
                # todo: should probably remove this check...
                raise ValueError(f"Data was 2D, but model {str(model):s} is not supported because"
                                 f" parameter {model_zsize_index:d} is not 'sz'.")
            # fix sigma-z
            fixed_params[model_zsize_index] = True
            init_params[:, model_zsize_index] = 1.

            if model.parameter_names[model_zposition_index] != "cz":
                # todo: should probably remove this check
                raise ValueError(f"Data was 2D, but model {str(model):s} is not supported because"
                                 f" parameter {model_zposition_index:d} is not 'cz'.")
            # fix cz
            fixed_params[model_zposition_index] = True
            init_params[:, model_zposition_index] = z[0, 0, 0]

        fit_results = fit_rois(img_rois,
                               roi_coords,
                               roi_sizes,
                               init_params,
                               max_number_iterations=max_nfit_iterations,
                               estimator="LSE",
                               fixed_params=fixed_params,
                               model=model,
                               guess_bounds=guess_bounds,
                               use_gpu=use_gpu_fit,
                               debug=debug,
                               verbose=verbose,
                               **kwargs
                               )

        if verbose:
            print(f"Localization took {time.perf_counter() - tstart:.2f}s")

        # ###################################################
        # filter fits
        # ###################################################
        tstart_filter = time.perf_counter()

        if filter is None:
            filter = no_filter()

        # add any other parameters which could be useful for the filter function to accept
        # filter functions are free to ignore all parameters besides fit_params,
        # but must accept arbitrary number of arguments and pass this through to other filters
        conditions = filter.filter(fit_results["fit_params"],
                                   init_params,
                                   fit_results["chi_sqrs"],
                                   fit_results["niters"],
                                   rois,
                                   image=imgs,
                                   image_filtered=imgs_filtered)

        to_keep = np.logical_and.reduce(conditions, axis=1)
        condition_names = filter.condition_names
        filter_settings = {}  # todo: populate

        # update fit results with more info
        fit_results.update({"rois": rois,
                            "to_keep": to_keep,
                            "conditions": conditions,
                            "condition_names": condition_names,
                            "filter_settings": filter_settings
                            })

        if verbose:
            num_rejected = len(conditions) - np.sum(conditions, axis=0)
            for rr in range(len(filter.condition_names)):
                print(f"Rejected {num_rejected[rr]:d} fits because {filter.condition_names[rr]}")

            print(f"Identified {np.sum(to_keep):d} likely candidates in {time.perf_counter() - tstart_filter:.3f}s")

    else:
        fit_results = None

    if return_filtered_images:
        return (z, y, x), fit_results, imgs_filtered
    else:
        return (z, y, x), fit_results, None


def localize_beads(imgs: np.ndarray,
                   dxy: float,
                   dz: float,
                   threshold: float,
                   roi_size: tuple[float] = (4, 2, 2),
                   filter_sigma_small: tuple[float] = (1, 0.1, 0.1),
                   filter_sigma_large: tuple[float] = (10, 5, 5),
                   min_spot_sep: tuple[float] = (0, 0),
                   sigma_bounds: tuple[tuple[float], tuple[float]] = ((0, 0), (np.inf, np.inf)),
                   fit_amp_min: float = 0,
                   fit_dist_max_err: tuple[float] = (np.inf, np.inf),
                   dist_boundary_min: tuple[float] = (0, 0),
                   max_nfit_iterations: int = 100,
                   fit_filtered_images: bool = False,
                   use_gpu_fit: bool = _gpufit_available,
                   use_gpu_filter: bool = _cupy_available,
                   return_filtered_images: bool = False,
                   verbose: bool = True,
                   **kwargs):
    """
    Wrapper around localize_beads_generic() which also takes all filter parameters as arguments. Mostly for
    historical convenience. Avoids the need to instantiate a separate filter object.

    For a description of the parameters see localize_beads_generic() and get_param_filter()
    """

    shape = imgs.shape
    if imgs.ndim == 2:
        shape = (1,) + shape
        dz = 1.

    coords = get_coords(shape, (dz, dxy, dxy))
    filter = get_param_filter(coords, fit_dist_max_err, min_spot_sep, sigma_bounds, (fit_amp_min, np.inf), dist_boundary_min)

    return localize_beads_generic(imgs,
                                  drs=(dz, dxy, dxy),
                                  threshold=threshold,
                                  roi_size=roi_size,
                                  filter_sigma_small=filter_sigma_small,
                                  filter_sigma_large=filter_sigma_large,
                                  min_spot_sep=min_spot_sep,
                                  filter=filter,
                                  max_nfit_iterations=max_nfit_iterations,
                                  fit_filtered_images=fit_filtered_images,
                                  use_gpu_fit=use_gpu_fit,
                                  use_gpu_filter=use_gpu_filter,
                                  return_filtered_images=return_filtered_images,
                                  verbose=verbose,
                                  **kwargs)


def plot_bead_locations(imgs: np.ndarray,
                        center_lists: list[np.ndarray],
                        title: str = "",
                        color_lists: Optional[list[str]] = None,
                        color_limits: Optional[list[list[float]]] = None,
                        legend_labels: Optional[list[str]] = None,
                        weights: Optional[list[np.ndarray]] = None,
                        cbar_labels: Optional[list[str]] = None,
                        coords: Optional[list] = None,
                        vlims_percentile: tuple[float] = (0.01, 99.99),
                        gamma: float = 1,
                        **kwargs):
    """
    Plot center locations over 2D image or max projection of 3D image. Supports plotting multiple different sets
    of center locations, and using different colors to indicate properties of the different centers e.g. sigma,
    amplitude, modulation depth, etc.

    :param imgs: 3D or 2D array. Dimensions order Z, Y, X
    :param center_lists: [center_array_1, center_array_2, ...] where each center_array is a numpy array of size N_i x 3
      consisting of triples of center values giving cz, cy, cx
    :param str title: title of plot
    :param color_lists: list of colors for each series to be plotted in
    :param color_limits: [[vmin_1, vmax_1], ...]
    :param legend_labels: labels for each series
    :param weights: list of arrays [w_1, ..., w_n], with w_i the same size as center_array_i, giving the intensity of
      the color to be plotted
    :param cbar_labels: list of labels for color bars
    :param coords:
    :param vlims_percentile: (percentile_min, percentile_max) used to set color scale of image
    :param gamma: gamma to use when displaying image
    :return figure_handle:
    """

    if not isinstance(center_lists, list):
        center_lists = [center_lists]
    nlists = len(center_lists)

    if color_lists is None:
        cmap = plt.cm.get_cmap('hsv')
        color_lists = []
        for ii in range(nlists):
            color_lists.append(cmap(ii / nlists))

    if not isinstance(color_lists, (list, tuple)):
        color_lists = [color_lists]

    if legend_labels is None:
        legend_labels = list(map(lambda x: "series #" + str(x) + " %d pts" % len(center_lists[x]), range(nlists)))

    if weights is None:
        weights = [np.ones(len(cs)) for cs in center_lists]

    if not isinstance(weights, (list, tuple)):
        weights = [weights]

    if cbar_labels is None:
        cbar_labels = ["" for cs in center_lists]

    if not isinstance(cbar_labels, (list, tuple)):
        cbar_labels = [cbar_labels]

    if imgs.ndim == 3:
        img_max_proj = np.nanmax(imgs, axis=0)
    else:
        img_max_proj = imgs

    # get extent from coordinates
    if coords is None:
        xx, yy = np.meshgrid(range(img_max_proj.shape[1]), range(img_max_proj.shape[0]))
    else:
        yy, xx = coords

    dx = xx[0, 1] - xx[0, 0]
    dy = yy[1, 0] - yy[0, 0]

    extent_xy = [xx.min() - 0.5 * dx, xx.max() + 0.5 * dx,
                 yy.max() + 0.5 * dy, xx.min() - 0.5 * dy]

    # create figure
    figh = plt.figure(**kwargs)
    figh.suptitle(title)
    ax = figh.add_subplot(1, 1, 1)

    # plot image
    vmin = np.percentile(img_max_proj, vlims_percentile[0])
    vmax = np.percentile(img_max_proj, vlims_percentile[1])

    im = ax.imshow(img_max_proj,
                   norm=PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax),
                   cmap=plt.cm.get_cmap("bone"),
                   extent=extent_xy)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel(f"Image intensity (counts), gamma={gamma:.2f}")

    # plot centers
    for ii in range(nlists):
        if color_limits is None:
            vmin = 0
            vmax = np.nanmax(weights[ii])
        else:
            vmin = color_limits[ii][0]
            vmax = color_limits[ii][1]

        cmap_color = LinearSegmentedColormap.from_list("test", [[0.5, 0.5, 0.5], color_lists[ii]])
        cs = cmap_color((weights[ii] - vmin) / (vmax - vmin))

        ax.scatter(center_lists[ii][:, 2], center_lists[ii][:, 1], facecolor='none', edgecolor=cs, marker='o')

        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap_color))
        cbar.ax.set_ylabel(cbar_labels[ii])

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.legend(legend_labels)

    return figh


def autofit_psfs(imgs: np.ndarray,
                 psf_roi_size: list[float],
                 dxy: float,
                 dz: float,
                 summary_model: psf.pixelated_psf_model = psf.gridded_psf_model(wavelength=0.532, ni=1.5, model_name="gaussian"),
                 threshold: float = 100.,
                 min_spot_sep: tuple[float] = (0., 0.),
                 filter_sigma_small: tuple[float] = (1, 0.5, 0.5),
                 filter_sigma_large: tuple[float] = (3, 5, 5),
                 sigma_bounds: tuple[tuple[float]] = ((0., 0.), (np.inf, np.inf)),
                 amp_bounds: tuple[float] = (0, np.inf),
                 roi_size_loc=(2, 3, 3),
                 dist_boundary_min: tuple[float] = (0., 0.),
                 localization_model: psf.pixelated_psf_model = psf.gaussian3d_psf_model(),
                 max_number_iterations: int = 100,
                 fit_dist_max_err: tuple[float] = (np.inf, np.inf),
                 num_localizations_to_plot: int = 5,
                 psf_percentiles: tuple[float] = (5,),
                 plot_results: bool = True,
                 only_plot_good_fits: bool = True,
                 plot_filtered_image: bool = False,
                 use_gpu_filter: bool = False,
                 use_gpu_fit: bool = False,
                 verbose: bool = False,
                 gamma: float = 0.5,
                 save_dir: Optional[str] = None,
                 figsize=(18, 10),
                 **kwargs) -> dict:
    """
    Given a 2D or 3D image, identify and localize diffraction limited spots. Aggregate the results to create
    an experimental point-spread function and fit the average PSF to a model function.

    :param imgs: nz x nx x ny image
    :param psf_roi_size: s(sz, sy, sx) size in um of  ROI to determine PSFs on. This is not necessarily the same as the size of the
      ROI's used during localization
    :param dxy: pixel size in um
    :param dz: z-plane spacing in um
    :param summary_model: This model is used to fit the averaged PSF's
    :param threshold: threshold pixel value to identify peaks. Note: this is applied to the filtered image,
      and is not directly comparable to the values in the raw image array
    :param min_spot_sep: (sz, sxy) minimum spot separation between different beads in um
    :param filter_sigma_small: (sz, sy, sx) sigmas of Gaussian filter used to smooth image in um
    :param filter_sigma_large: (sz, sy, sx) sigmas of Gaussian filter used to removed background in um
    :param sigma_bounds: ((sz_min, sxy_min), (sz_max, sxy_max)) in um. exclude fits with sigmas that fall outside
      these ranges
    :param amp_bounds:
    :param roi_size_loc: (sz, sy, sx) size of ROI to used in localization, in um
    :param dist_boundary_min:
    :param localization_model:
    :param max_number_iterations:
    :param fit_dist_max_err:
    :param num_localizations_to_plot: number of ROI's to plot
    :param psf_percentiles: calculate the averaged PSF from the smallest supplied percentage of spots. When
      a tuple is given, compute PSF's corresponding to each supplied percentage.
    :param plot_results: optionally plot diagnostics
    :param only_plot_good_fits: when plotting ROI, plot only fits that passed all filtering tests or plot all fits
    :param plot_filtered_image: plot ROI's against filtered image also
    :param use_gpu_filter:
    :param use_gpu_fit:
    :param gamma: gamma to use when plotting
    :param save_dir: directory to save diagnostic plots. If None, these will not be saved
    :param figsize: (sx, sy)
    :param **kwargs: passed through to plt.figure()
    :return: dictionary object with entries coords, fit_params, init_params, rois, to_keep, conditions,
      condition_names, filter_settings, fit_states, chi_sqrs, niters,  psfs_real, psf_coords, otfs_real,
      otf_coords, psf_percentiles, fit_params_real
    """

    plt.switch_backend("agg")
    plt.ioff()

    # todo: correct this function for many recent changes

    if isinstance(psf_percentiles, (int, float)):
        psf_percentiles = [psf_percentiles]

    saving = False
    if save_dir is not None:
        saving = True
        save_dir = Path(save_dir)

        save_dir.mkdir(exist_ok=True)

    # ###################################
    # do localization
    # ###################################
    z, y, x = get_coords(imgs.shape, (dz, dxy, dxy))

    filter = get_param_filter((z, y, x),
                              fit_dist_max_err=fit_dist_max_err,
                              min_spot_sep=min_spot_sep,
                              sigma_bounds=sigma_bounds,
                              amp_bounds=amp_bounds,
                              dist_boundary_min=dist_boundary_min)

    coords, fit_results, imgs_filtered = localize_beads_generic(imgs,
                                                                drs=(dz, dxy, dxy),
                                                                threshold=threshold,
                                                                roi_size=roi_size_loc,
                                                                filter_sigma_small=filter_sigma_small,
                                                                filter_sigma_large=filter_sigma_large,
                                                                min_spot_sep=min_spot_sep,
                                                                filter=filter,
                                                                model=localization_model,
                                                                max_nfit_iterations=max_number_iterations,
                                                                use_gpu_filter=use_gpu_filter,
                                                                use_gpu_fit=use_gpu_fit,
                                                                verbose=verbose)

    fit_params = fit_results["fit_params"]
    init_params = fit_results["init_params"]
    rois = fit_results["rois"]
    to_keep = fit_results["to_keep"]
    conditions = fit_results["conditions"]
    condition_names = fit_results["condition_names"]
    filter_settings = fit_results["filter_settings"]
    fit_states = fit_results["fit_states"]
    chi_sqrs = fit_results["chi_sqrs"]
    niters = fit_results["niters"]

    no_psfs_found = not np.any(to_keep)
    if no_psfs_found:
        warnings.warn("no spots were localized")

    # ###################################
    # plot individual localizations
    # ###################################
    if plot_results:
        print("plotting ROI's")
        tstart_plot_roi = time.perf_counter()

        if only_plot_good_fits:
            ind_to_plot = np.arange(len(to_keep), dtype=int)[to_keep][:num_localizations_to_plot]
        else:
            ind_to_plot = np.arange(len(to_keep), dtype=int)[:num_localizations_to_plot]

        delayed = []

        if plot_filtered_image:
            im_to_plot = imgs_filtered
        else:
            im_to_plot = imgs

        for ind in ind_to_plot:
            delayed.append(dask.delayed(plot_fit_roi)(fit_params[ind],
                                                      rois[ind],
                                                      im_to_plot,
                                                      coords,
                                                      init_params[ind],
                                                      figsize=figsize,
                                                      prefix="localization_roi_%d" % ind,
                                                      string="filter conditions = " + " ".join(["%d," % c for c in conditions[ind]]),
                                                      save_dir=save_dir
                                                      )
                           )

        # with ProgressBar():
        results = dask.compute(*delayed)

        print(f"plotting took {time.perf_counter() - tstart_plot_roi:.2f}s")

    # ###################################
    # plot fit statistics
    # ###################################
    if plot_results and not no_psfs_found:

        # fit parameter summary
        figh = plt.figure(figsize=figsize, **kwargs)
        figh.suptitle("Localization fit parameter summary")
        grid = figh.add_gridspec(nrows=localization_model.nparams - 1, hspace=0.3,
                                 ncols=localization_model.nparams - 1, wspace=0.3)

        for ii in range(localization_model.nparams):
            for jj in range(ii + 1, localization_model.nparams):
                ax = figh.add_subplot(grid[ii, jj - 1])
                ax.plot(fit_params[to_keep, ii], fit_params[to_keep, jj], '.')
                ax.set_xlabel(localization_model.parameter_names[ii])
                ax.set_ylabel(localization_model.parameter_names[jj])

        if saving:
            figh.savefig(Path(save_dir) / "fit_stats.png")
            plt.close(figh)

    # ###################################
    # get and plot experimental PSFs
    # ###################################
    nps = len(psf_percentiles)
    psf_roi_size_pix = np.round(np.array(psf_roi_size) / np.array([dz, dxy, dxy])).astype(int)
    psf_roi_size_pix += (1 - np.mod(psf_roi_size_pix, 2))

    psfs_real = np.zeros((nps,) + tuple(psf_roi_size_pix))
    otfs_real = np.zeros(psfs_real.shape, dtype=complex)
    fit_params_real = np.zeros((nps, summary_model.nparams))
    psf_coords = None
    otf_coords = None

    if not no_psfs_found:
        for ii in range(len(psf_percentiles)):
            # only keep smallest so many percent of spots
            sigma_max = np.percentile(fit_params[:, 4][to_keep], psf_percentiles[ii])
            to_use = np.logical_and(to_keep, fit_params[:, 4] <= sigma_max)

            # get centers
            centers = np.stack((fit_params[:, 3][to_use],
                                fit_params[:, 2][to_use],
                                fit_params[:, 1][to_use]), axis=1)

            # find average experimental psf/otf
            psfs_real[ii], psf_coords, otfs_real[ii], otf_coords = psf.average_exp_psfs(imgs, (z, y, x), centers, psf_roi_size_pix,
                                                                                        backgrounds=fit_params[:, 5][to_use])

            # fit average experimental psf
            def fn(p): return summary_model.model(psf_coords, p)
            init_params = summary_model.estimate_parameters(psfs_real[ii], psf_coords)

            results = fit.fit_model(psfs_real[ii], fn, init_params, jac='3-point', x_scale='jac')
            fit_params_real[ii] = results["fit_params"]

            if plot_results:
                figh = plot_fit_roi(fit_params_real[ii],
                                    [0, psfs_real[ii].shape[0],
                                       0, psfs_real[ii].shape[1],
                                       0, psfs_real[ii].shape[2]],
                                    psfs_real[ii],
                                    psf_coords,
                                    model=summary_model,
                                    string=f"smallest {psf_percentiles[ii]:.0f} percent,"
                                             f" {type(summary_model)}, sf={summary_model.sf}",
                                    vmin=0,
                                    vmax=1,
                                    gamma=gamma,
                                    figsize=figsize,
                                    **kwargs)

                if saving:
                    figh.savefig(Path(save_dir) / f"experimental_psf_smallest_{psf_percentiles[ii]:.2f}.png")
                    plt.close(figh)

    # ###################################
    # plot localization positions
    # ###################################
    if plot_results and not no_psfs_found:
        centers_all = np.stack((fit_params[:, 3],
                                fit_params[:, 2],
                                fit_params[:, 1]), axis=1)

        figh = plot_bead_locations(imgs,
                                   [centers_all, centers_all[to_keep]],
                                   weights=[np.ones(len(centers_all)), fit_params[to_keep, 4]],
                                   cbar_labels=["all fits", r"kept fits, $\sigma_{xy} (\mu m)$"],
                                   title="Max intensity projection and size from 2D fit versus position",
                                   coords=np.meshgrid(y, x, indexing="ij"),
                                   gamma=gamma,
                                   figsize=figsize)

        if saving:
            figh.savefig(Path(save_dir) / "sigma_versus_position.png")
            plt.close(figh)

    # convert coords to array we can save in zarr file
    coords_bcast = np.stack([np.array(c, copy=True) for c in np.broadcast_arrays(*coords)], axis=0)

    if psf_coords is not None:
        psf_coords_bcast = np.stack([np.array(c, copy=True) for c in np.broadcast_arrays(*psf_coords)], axis=0)

    if otf_coords is not None:
        otf_coords_bcast = np.stack([np.array(c, copy=True) for c in np.broadcast_arrays(*otf_coords)], axis=0)

    data = {"coords": coords_bcast,
            "fit_params": fit_params,
            "init_params": init_params,
            "rois": rois,
            "to_keep": to_keep,
            "conditions": conditions,
            "condition_names": condition_names,
            # "filter_settings": filter_settings,
            "fit_states": fit_states,
            "chi_sqrs": chi_sqrs,
            "niterations": niters,
            "psfs_real": psfs_real,
            "psf_coords": psf_coords_bcast,
            "otfs_real": otfs_real,
            "otf_coords": otf_coords_bcast,
            "psf_percentiles": np.array(psf_percentiles),
            "fit_params_real": fit_params_real}

    if saving:
        z = zarr.open(Path(save_dir) / "localization_results.zarr", "w")
        for k, v in data.items():
            if v is not None:
                z.array(k, v, compressor="none")

    return data
