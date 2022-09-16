"""
Code for localizing and fitting (typically diffraction limited) spots/beads

The fitting code can be run on a CPU using multiprocessing with joblib, or on a GPU using custom modifications
to GPUfit which can be found at https://github.com/QI2lab/Gpufit. To use the GPU code, you must download and
compile this repository and install the python bindings.
"""
from pathlib import Path
import time
import warnings
import zarr
import numpy as np
import scipy.signal
import scipy.ndimage
import joblib # todo: remove in favor of dask
import dask
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
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

# custom GPUFit for fitting on GPU
try:
    import pygpufit.gpufit as gf
    GPUFIT_AVAILABLE = True
except ImportError:
    GPUFIT_AVAILABLE = False


def get_coords(sizes, drs):
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
    :return coords: (coords0, coords1, ..., coordsn)
    """
    ndims = len(drs)
    coords = [np.expand_dims(np.arange(sz) * dr, axis=list(range(ii)) + list(range(ii + 1, ndims)))
              for ii, (sz, dr) in enumerate(zip(sizes, drs))]

    return coords


def get_roi(center, img, coords, sizes):
    """
    Find ROI which is nearly centered on center. Since center may not correspond to a pixel location, and the
    size of the ROI may not be odd, center will not be the exact center of the ROI

    :param center: [c_0, c_1, ..., c_n] in same units as x, y, z.
    :param img: array of arbitrary size, m0 x m1 x ... x mn
    :param coords: (coords0, coords1, ..., coordsN)
    :param sizes: [i0, i1, ... in] integers
    :return roi, img_roi, coords_roi:
    """
    ndims = img.ndim
    # get closest coordinates to desired center of roi
    ics = [np.argmin(np.abs(r.ravel() - c)) for r, c in zip(coords, center)]

    roi = np.array(roi_fns.get_centered_roi(ics, sizes, min_vals=[0]*ndims, max_vals=img.shape))

    # get coordinates as arrays which only have nonunit size along one direction
    coords_roi = [c[tuple([slice(None)] * ii + [slice(roi[2*ii], roi[2*ii + 1])] + [slice(None)] * (ndims - 1 - ii))]
                  for ii, c in enumerate(coords)]
    # broadcast to full arrays, essentially meshgrid
    coords_roi = np.broadcast_arrays(*coords_roi)

    img_roi = roi_fns.cut_roi(roi, img)

    return roi, img_roi, coords_roi


def get_filter_kernel(sigmas, drs, sigma_cutoff=2):
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


def filter_convolve(imgs, kernel, use_gpu=CUPY_AVAILABLE):
    """
    Convolution filter using kernel with GPU support. To avoid roll-off effects at the edge, the convolved
    result is "normalized" by being divided by the kernel convolved with an array of ones.

    :param imgs: images to be convolved
    :param kernel: kernel to be convolved. Does not need to be the same shape as image.
    :param bool use_gpu: if True, do convolution on GPU. If false, do on CPU.
    :return imgs_filtered:
    """

    # todo: estimate how much memory convolution requires? Much more than I expect...

    # convolve, and deal with edges by normalizing
    if use_gpu:
        kernel_cp = cp.asarray(kernel, dtype=cp.float32)
        imgs_cp = cp.asarray(imgs, dtype=cp.float32)
        imgs_filtered_cp = cupyx.scipy.signal.fftconvolve(imgs_cp, kernel_cp, mode="same")
        imgs_filtered = cp.asnumpy(imgs_filtered_cp)

        imgs_cp = None
        del imgs_cp

        imgs_filtered_cp = None
        del imgs_filtered_cp

        norm_cp = cupyx.scipy.signal.fftconvolve(cp.ones(imgs.shape), kernel_cp, mode="same")
        norm = cp.asnumpy(norm_cp)

        imgs_filtered = imgs_filtered / norm

        kernel_cp = None
        del kernel_cp

        norm_cp = None
        del norm_cp

        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()
        # pinned_mempool = cp.get_default_pinned_memory_pool()

    else:
        imgs_filtered = scipy.signal.fftconvolve(imgs, kernel, mode="same") / \
                        scipy.signal.fftconvolve(np.ones(imgs.shape), kernel, mode="same")

    # this method too slow for large filter sizes
    # imgs_filtered = scipy.ndimage.convolve(imgs, kernel, mode="constant", cval=0) / \
    #                 scipy.ndimage.convolve(np.ones(imgs.shape), kernel, mode="constant", cval=0)

    return imgs_filtered


def get_max_filter_footprint(min_separations, drs):
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

    # ns = [int(np.ceil(sz / dr)) for sz, dr in zip(min_sep_allowed, drs)]
    # ensure odd
    # ns = [n if np.mod(n, 2) == 0 else n + 1 for n in ns]

    footprint = np.ones(ns, dtype=bool)

    return footprint


def find_peak_candidates(imgs, footprint, threshold, mask=None, use_gpu_filter=CUPY_AVAILABLE):
    """
    Find peak candidates in image using maximum filter

    :param imgs: 2D or 3D array
    :param footprint: footprint to use for maximum filter. Array should have same number of dimensions as imgs.
    This can be obtained from get_max_filter_footprint()
    :param float threshold: only pixels with values greater than or equal to the threshold will be considered
    :param bool use_gpu_filter: whether or not to do maximum filter on GPU
    :return centers_guess_inds, img_vals: np.array([[i0, i1, i2], ...]) array indices of local maxima
    """
    if mask is None:
        mask = np.ones(imgs.shape, dtype=bool)

    if use_gpu_filter:
        img_max_filtered = cp.asnumpy(
            cupyx.scipy.ndimage.maximum_filter(cp.asarray(imgs, dtype=cp.float32), footprint=cp.asarray(footprint)))
        # need to compare imgs as float32 because img_max_filtered will be ...
        is_max = np.logical_and.reduce((imgs.astype(np.float32) == img_max_filtered, imgs >= threshold, mask))
    else:
        img_max_filtered = scipy.ndimage.maximum_filter(imgs, footprint=footprint)
        is_max = np.logical_and.reduce((imgs == img_max_filtered, imgs >= threshold, mask))

    amps = imgs[is_max]
    centers_guess_inds = np.argwhere(is_max)

    return centers_guess_inds, amps


def filter_nearby_peaks(centers: np.ndarray, min_xy_dist: float, min_z_dist: float, mode: str = "keep-one",
                        weights: np.ndarray = None, nmax: int = 10000) -> (np.ndarray, np.ndarray):
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

    :return centers_unique: array of unique centers
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

    # only need to act if minimum distances are non-zero
    if min_xy_dist > 0 or min_z_dist > 0:
        if len(centers_unique) > nmax:
            if mode == "average":
                raise NotImplementedError("mode='average' is not implemented with nmax < np.inf. Set nmax to np.inf")

            # todo: maybe I should check that dimension sizes are similar before dividing. If have very asymmetric
            # todo: area it might be better to e.g. make more divisions along one dimension only
            # todo: on second thought, I think the right approach is in each step to only divide once, and always divide the largest dimension

            # if number of inputs is large, divide problem into subproblems, solve each of these, and combine results.
            xlims = [np.min(centers[:, 2]), np.max(centers[:, 2])]
            ylims = [np.min(centers[:, 1]), np.max(centers[:, 1])]
            zlims = [np.min(centers[:, 0]), np.max(centers[:, 0])]

            full_inds = np.arange(len(centers_unique), dtype=int)
            centers_unique_sectors = []
            inds_sectors = []

            # divide space into octants, if each direction is large enough
            if (zlims[1] - zlims[0]) > 2 * min_z_dist:
                zedges = [zlims[0], 0.5 * (zlims[0] + zlims[1]), zlims[1] + min_z_dist]
            else:
                zedges = [zlims[0], zlims[1] + min_z_dist]

            if (ylims[1] - ylims[0]) > 2 * min_xy_dist:
                yedges = [ylims[0], 0.5 * (ylims[0] + ylims[1]), ylims[1] + min_xy_dist]
            else:
                yedges = [ylims[0], ylims[1] + min_xy_dist]

            if (xlims[1] - xlims[0]) > 2 * min_xy_dist:
                xedges = [xlims[0], 0.5 * (xlims[0] + xlims[1]), xlims[1] + min_xy_dist]
            else:
                xedges = [xlims[0], xlims[1] + min_xy_dist]

            # solve sectors independently
            for ii in range(len(xedges) - 1):
                for jj in range(len(yedges) - 1):
                    for kk in range(len(zedges) - 1):
                        to_use = np.logical_and.reduce((centers_unique[:, 0] >= zedges[kk], centers_unique[:, 0] < zedges[kk + 1],
                                                        centers_unique[:, 1] >= yedges[jj], centers_unique[:, 1] < yedges[jj + 1],
                                                        centers_unique[:, 2] >= xedges[ii], centers_unique[:, 2] < xedges[ii + 1]))

                        if np.any(to_use):
                            cu, i = filter_nearby_peaks(centers_unique[to_use], min_xy_dist, min_z_dist, mode=mode)
                            centers_unique_sectors.append(cu)
                            inds_sectors.append(full_inds[to_use][i])

            centers_unique_sectors = np.concatenate(centers_unique_sectors, axis=0)
            inds_sectors = np.concatenate(inds_sectors)

            # check overlap regions between sectors
            for ii in range(len(xedges) - 2):
                to_use = np.logical_and(centers_unique_sectors[:, 2] >= xedges[ii + 1] - min_xy_dist,
                                        centers_unique_sectors[:, 2] < xedges[ii + 1] + min_xy_dist)
                if np.any(to_use):
                    centers_unique_overlap, i = filter_nearby_peaks(centers_unique_sectors[to_use], min_xy_dist, min_z_dist, mode=mode)

                    # get full centers by adding any that were not in the overlap region with the reduced set from the overlap region
                    centers_unique_sectors = np.concatenate((centers_unique_sectors[np.logical_not(to_use)], centers_unique_overlap))
                    inds_sectors = np.concatenate((inds_sectors[np.logical_not(to_use)], full_inds[inds_sectors][to_use][i]))

            for ii in range(len(yedges) - 2):
                to_use = np.logical_and(centers_unique_sectors[:, 1] >= yedges[ii + 1] - min_xy_dist,
                                        centers_unique_sectors[:, 1] < yedges[ii + 1] + min_xy_dist)

                if np.any(to_use):
                    centers_unique_overlap, i = filter_nearby_peaks(centers_unique_sectors[to_use], min_xy_dist, min_z_dist, mode=mode)

                    centers_unique_sectors = np.concatenate((centers_unique_sectors[np.logical_not(to_use)], centers_unique_overlap))
                    inds_sectors = np.concatenate((inds_sectors[np.logical_not(to_use)], full_inds[inds_sectors][to_use][i]))

            for ii in range(len(zedges) - 2):
                to_use = np.logical_and(centers_unique_sectors[:, 0] >= zedges[ii + 1] - min_z_dist,
                                        centers_unique_sectors[:, 0] < zedges[ii + 1] + min_z_dist)
                if np.any(to_use):
                    centers_unique_overlap, i = filter_nearby_peaks(centers_unique_sectors[to_use], min_xy_dist, min_z_dist, mode=mode)

                    centers_unique_sectors = np.concatenate((centers_unique_sectors[np.logical_not(to_use)], centers_unique_overlap))
                    inds_sectors = np.concatenate((inds_sectors[np.logical_not(to_use)], full_inds[inds_sectors][to_use][i]))

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


# localize using radial symmetry
def localize2d(img, mode="radial-symmetry"):
    """
    Perform 2D localization using the radial symmetry approach of https://doi.org/10.1038/nmeth.2071

    :param img: 2D image of size ny x nx
    :param str mode: 'radial-symmetry' or 'centroid'

    :return xc:
    :return yc:
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


def localize3d(img, mode="radial-symmetry"):
    """
    Perform 3D localization using an extension of the radial symmetry approach of https://doi.org/10.1038/nmeth.2071

    :param img: 3D image of size nz x ny x nx
    :param str mode: 'radial-symmetry' or 'centroid'

    :return xc:
    :return yc:
    :return zc:
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
        n1 = np.array([1, 1, 1]) / np.sqrt(3) # vectors go [nz, ny, nx]
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
        for ll in range(3): # rows of matrix
            for ii in range(3): # columns of matrix
                if ii == ll:
                    mat[ll, ii] += np.sum(-wk * (nk[ii] * nk[ll] - 1))
                else:
                    mat[ll, ii] += np.sum(-wk * nk[ii] * nk[ll])

                for jj in range(3): # internal sum
                    if jj == ll:
                        mat[ll, ii] += np.sum(wk * nk[ii] * nk[jj] * (nk[jj] * nk[ll] - 1))
                    else:
                        mat[ll, ii] += np.sum(wk * nk[ii] * nk[jj] * nk[jj] * nk[ll])

        # build vector from above
        vec = np.zeros((3, 1))
        coord_sum = zk * nk[0] + yk * nk[1] + xk * nk[2]
        for ll in range(3): # sum over J, K, L
            for ii in range(3): # internal sum
                if ii == ll:
                    vec[ll] += -np.sum((coords[ii] - nk[ii] * coord_sum) * (nk[ii] * nk[ll] - 1) * wk)
                else:
                    vec[ll] += -np.sum((coords[ii] - nk[ii] * coord_sum) * nk[ii] * nk[ll] * wk)

        # invert matrix
        zc, yc, xc = np.linalg.inv(mat).dot(vec)
    else:
        raise ValueError("mode must be 'centroid' or 'radial-symmetry', but was '%s'" % mode)

    return xc, yc, zc


# localize using Gaussian fit
def fit_gauss_roi(img_roi,
                  coords,
                  init_params,
                  fixed_params=None,
                  bounds=None,
                  model=psf.gaussian3d_psf_model()):
    """
    Fit a single ROI to a 3D gaussian function.

    :param img_roi: array of size nz x ny x nx which will be fit
    :param coords: (z_roi, y_roi, x_roi). These coordinate arrays must be broadcastable to the same size as img_roi
    :param init_params: array of length 7, where the parameters are
    [amplitude, center-x, center-y, center-z, sigma-xy, sigma_z, offset]
    :param fixed_params: boolean array of length 7. For entries which are True, the fit function will force
    that parameter to be identical to the value in init_params. For entries which are False, the fit function
    will determine the optimal value
    :param bounds: ((lower_bounds), (upper_bounds)) where lower_bounds and upper_bounds are each lists/tuples/arrays
    of length 7.
    :param model:
    :return results: dictionary object containing information about fitting
    """
    z_roi, y_roi, x_roi = coords

    # if img_roi is 2D and z- dimension size is 0, treat as 3D
    if img_roi.ndim == 2 and z_roi.shape[0] == 1:
        img_roi = np.expand_dims(img_roi, axis=0)

    # img_roi must be 3D
    if img_roi.ndim != 3:
        raise ValueError(f"img_roi must have 3 dimensions but had {img_roi.ndim:d}")

    # localization functions
    def model_fn(params): return model.model((z_roi, y_roi, x_roi), params)

    if model.has_jacobian:
        def jac_fn(p): return model.jacobian((z_roi, y_roi, x_roi), p)
    else:
        jac_fn = None

    # do fitting
    results = fit.fit_model(img_roi, model_fn, init_params,
                            bounds=bounds,
                            fixed_params=fixed_params,
                            model_jacobian=jac_fn)

    return results


def fit_gauss_rois(img_rois: np.ndarray,
                   coords_rois: tuple[np.ndarray],
                   init_params: np.ndarray,
                   max_number_iterations: int = 100,
                   estimator: str = "LSE",
                   fixed_params: np.ndarray = None,
                   use_gpu: bool = GPUFIT_AVAILABLE,
                   verbose: bool = True,
                   debug: bool = False,
                   model=psf.gaussian3d_psf_model()):
    """
    Fit rois. Can use either CPU parallelization with joblib or GPU parallelization using gpufit

    :param img_rois: list of image rois
    :param coords_rois: ((z0, y0, x0), (z1, y1, x1), ....)
    :param init_params: initial parameters for fits, size nfits x nparams
    :param int max_number_iterations: maximum number of iterations to be used for each fit
    :param estimator: "LSE" or "MLE"
    :param model: "gaussian", "rotated-gaussian", "gaussian-lorentzian"
    :param list[bool] fixed_params: length nparams vector of parameters to fix. only supports fixing/unfixing
    each parameter for all fits
    :param bool use_gpu:
    :return fit_params, fit_states, chi_sqrs, niters, fit_t:
    """

    zrois, yrois, xrois = coords_rois

    for ii in range(len(img_rois)):
        if img_rois[ii].ndim != 3:
            raise ValueError(f"img_rois position {ii:d} was not 3-dimensional")


    if not use_gpu:

        verbose_joblib = 0
        if verbose:
            verbose_joblib = 1

        tstart = time.perf_counter()

        if debug:
            results = []
            for ii in range(len(img_rois)):
                results.append(fit_gauss_roi(img_rois[ii],
                                             (zrois[ii], yrois[ii], xrois[ii]),
                                             init_params=init_params[ii],
                                             fixed_params=fixed_params,
                                             model=model))
        else:
            # results = joblib.Parallel(n_jobs=-1, verbose=verbose_joblib, timeout=None)(
            #     joblib.delayed(fit_gauss_roi)(img_rois[ii],
            #                                   (zrois[ii], yrois[ii], xrois[ii]),
            #                                   init_params=init_params[ii],
            #                                   fixed_params=fixed_params,
            #                                   sf=sf,
            #                                   dc=dc,
            #                                   angles=angles,
            #                                   fn=fn,
            #                                   jacobian=jacobian)
            #     for ii in range(len(img_rois)))

            # forced to switch to dask form joblib because joblib use pickling to exchange info between process
            # and functions (which are arguments to fit_gauss_roi) are not pickleable
            delayed = []
            for ii in range(len(img_rois)):
                delayed.append(dask.delayed(fit_gauss_roi)(img_rois[ii],
                                             (zrois[ii], yrois[ii], xrois[ii]),
                                             init_params=init_params[ii],
                                             fixed_params=fixed_params,
                                             model=model))

            # with ProgressBar():
            results = dask.compute(*delayed)



        fit_t = (time.perf_counter() - tstart)
        fit_params = np.asarray([r["fit_params"] for r in results])
        chi_sqrs = np.asarray([r["chi_squared"] for r in results])
        fit_states = np.asarray([r["status"] for r in results])
        niters = np.asarray([r["nfev"] for r in results])

    else:
        if model.sf != 1:
            raise NotImplementedError("sampling factors other than 1 are not implemented for GPU fitting")
        # todo: if requires more memory than GPU has, split into chunks

        if isinstance(model, psf.gaussian3d_psf_model):
            model_id = gf.ModelID.GAUSS_3D_ARB
        elif isinstance(model, psf.gaussian_lorentzian_psf_model):
            model_id = gf.ModelID.GAUSS_LOR_3D_ARB
        else:
            raise NotImplementedError("only 'gaussian3d_psf' and 'gaussian_lorentzian_psf' have"
                                      " corresponding GPU implementations")

        nparams = model.nparams

        # todo: implemented on GPU but not CPU
        # elif fn == 0:
        #     model_id = gf.ModelID.GAUSS_3D_ROT_ARB

        # ensure arrays are row vectors
        xrois, yrois, zrois, img_rois = zip(*[(xr.ravel()[None, :],
                                               yr.ravel()[None, :],
                                               zr.ravel()[None, :],
                                               ir.ravel()[None, :])
                                               for xr, yr, zr, ir in zip(xrois, yrois, zrois, img_rois)])

        # get ROI sizes
        roi_sizes = np.array([ir.size for ir in img_rois])
        nmax = roi_sizes.max()

        # pad ROI's to make sure all ROI's same size
        img_rois = [np.pad(ir, ((0, 0), (0, nmax - ir.size)), mode="constant") for ir in img_rois]

        # build ROI data
        data = np.concatenate(img_rois, axis=0)
        data = data.astype(np.float32)
        nfits, n_pts_per_fit = data.shape

        # build user info, which stores information about the coordinates
        coords = [np.concatenate(
            (np.pad(xr.ravel(), (0, nmax - xr.size)),
             np.pad(yr.ravel(), (0, nmax - yr.size)),
             np.pad(zr.ravel(), (0, nmax - zr.size))
             ))
            for xr, yr, zr in zip(xrois, yrois, zrois)]
        coords = np.concatenate(coords)

        user_info = np.concatenate((coords.astype(np.float32), roi_sizes.astype(np.float32)))

        # initial parameters
        init_params = init_params.astype(np.float32)

        # check arguments
        if data.ndim != 2:
            raise ValueError("data wrong dimension")
        if init_params.ndim != 2 or init_params.shape != (nfits, nparams):
            raise ValueError("init_params had wrong shape or dimension")
        if user_info.ndim != 1 or user_info.size != (3 * nfits * n_pts_per_fit + nfits):
            raise ValueError("user_info had wrong shape or dimension")

        if estimator == "MLE":
            est_id = gf.EstimatorID.MLE
        elif estimator == "LSE":
            est_id = gf.EstimatorID.LSE
        else:
            raise ValueError("'estimator' must be 'MLE' or 'LSE' but was '%s'" % estimator)

        # set which parameters to fit/fix
        if fixed_params is None:
            fixed_params = np.zeros((nparams), dtype=bool)

        params_to_fit = fixed_params.astype(np.int32)

        # do fitting
        fit_params, fit_states, chi_sqrs, niters, fit_t = gf.fit(data,
                                                                 None,
                                                                 model_id,
                                                                 init_params,
                                                                 max_number_iterations=max_number_iterations,
                                                                 estimator_id=est_id,
                                                                 parameters_to_fit=params_to_fit,
                                                                 user_info=user_info)

        # correct sigmas in case negative
        # todo: think better to do this in external function bc might not be same in all models
        fit_params[:, 4] = np.abs(fit_params[:, 4])
        fit_params[:, 5] = np.abs(fit_params[:, 5])

    return fit_params, fit_states, chi_sqrs, niters, fit_t


def plot_gauss_roi(fit_params: list[float],
                   roi: list[int],
                   imgs: np.ndarray,
                   coords: tuple[np.ndarray] = None,
                   init_params: np.ndarray = None,
                   model=psf.gaussian3d_psf_model(),
                   string: str = None,
                   same_color_scale: bool = True,
                   vmin: float = None,
                   vmax: float = None,
                   cmap="bone",
                   gamma: float = 1.,
                   figsize: tuple[float] = (16, 8),
                   prefix: str = "",
                   save_dir=None):
    """
    Plot results obtained from fitting functions fit_gauss_roi() or fit_gauss_rois()
    :param fit_params:
    :param roi: [zstart, zend, ystart, yend, xstart, xend]
    :param imgs: full image, such that imgs[zstart:zend, ystart:yend, xstart:xend] is the region that was fit
    :param coords: (z, y, x) broadcastable to same size as imgs
    :param init_params: initial parameters used in fit, optional
    :param bool same_color_scale: whether or not to use same color scale for data and fits
    :param model:
    :param vmin:
    :param vmax:
    :param cmap:
    :param figsize: (sx, sz)
    :param str prefix: prefix prepended before save name
    :param str save_dir: if None, do not save results
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
    img_roi = roi_fns.cut_roi(roi, imgs)
    x_roi = roi_fns.cut_roi(roi, x)
    y_roi = roi_fns.cut_roi(roi, y)
    z_roi = roi_fns.cut_roi(roi, z)

    if vmin is None:
        vmin = np.percentile(img_roi[np.logical_not(np.isnan(img_roi))], 1)

    if vmax is None:
        vmax = np.percentile(img_roi[np.logical_not(np.isnan(img_roi))], 99.9)

    # git fit
    img_fit = model.model((z_roi, y_roi, x_roi), fit_params)

    # set extents
    extent_yx = [y_roi[0, 0, 0] - 0.5 * dc, y_roi[0, -1, 0] + 0.5 * dc,
                 x_roi[0, 0, 0] - 0.5 * dc, x_roi[0, 0, -1] + 0.5 * dc]

    extent_zx = [z_roi[0, 0, 0] - 0.5 * dz, z_roi[-1, 0, 0] + 0.5 * dz,
                 x_roi[0, 0, 0] - 0.5 * dc, x_roi[0, 0, -1] + 0.5 * dc]

    extent_yz = [y_roi[0, 0, 0] - 0.5 * dc, y_roi[0, -1, 0] + 0.5 * dc,
                 z_roi[0, 0, 0] - 0.5 * dz, z_roi[-1, 0, 0] + 0.5 * dz]

    wx = extent_yx[3] - extent_yx[2]
    wy = extent_yx[1] - extent_yx[0]
    wz = extent_zx[1] - extent_zx[0]

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

    grid = figh_interp.add_gridspec(nrows=2, height_ratios=[1, wz / wy], hspace=0,
                                    ncols=7, width_ratios=[wz / wx, 1, 0.2, wz / wx, 1, 0.2, 0.2], wspace=0)

    # ################################
    # XY, data
    # ################################
    ax = figh_interp.add_subplot(grid[0, 1])
    im = ax.imshow(np.nanmax(img_roi, axis=0).transpose(), origin="lower", extent=extent_yx, cmap=cmap,
                   norm=PowerNorm(vmin=vmin, vmax=vmax, gamma=gamma))

    ax.plot(center_fit[1], center_fit[2], 'mx')
    if init_params is not None:
        ax.plot(center_guess[1], center_guess[2], 'gx')

    ax.set_ylim(extent_yx[2:4])
    ax.set_xlim(extent_yx[0:2])
    ax.set_xticks([])
    ax.set_yticks([])

    # ################################
    # XZ, data
    # ################################
    ax = figh_interp.add_subplot(grid[0, 0])

    ax.imshow(np.nanmax(img_roi, axis=1).transpose(), origin="lower", extent=extent_zx, cmap=cmap,
              norm=PowerNorm(vmin=vmin, vmax=vmax, gamma=gamma))
    ax.plot(center_fit[0], center_fit[2], 'mx')
    if init_params is not None:
        ax.plot(center_guess[0], center_guess[2], 'gx')
    ax.set_ylim(extent_zx[2:4])
    ax.set_xlim(extent_zx[0:2])

    ax.set_ylabel("X (um)")

    # ################################
    # YZ, data
    # ################################
    ax = figh_interp.add_subplot(grid[1, 1])
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')

        ax.imshow(np.nanmax(img_roi, axis=2),  origin="lower", extent=extent_yz, cmap="bone",
                  norm=PowerNorm(vmin=vmin, vmax=vmax, gamma=gamma))

    ax.plot(center_fit[1], center_fit[0], 'mx')
    if init_params is not None:
        ax.plot(center_guess[1], center_guess[0], 'gx')

    ax.set_ylim(extent_yz[2:4])
    ax.set_xlim(extent_yz[0:2])
    ax.set_xlabel("Y (um)")
    ax.set_ylabel("Z (um)")

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
    extent = [y_roi[0, 0, 0] - 0.5 * dc, y_roi[0, -1, 0] + 0.5 * dc,
              x_roi[0, 0, 0] - 0.5 * dc, x_roi[0, 0, -1] + 0.5 * dc]
    ax.imshow(np.nanmax(img_fit, axis=0).transpose(), origin="lower", extent=extent, cmap=cmap,
              norm=PowerNorm(vmin=vmin_fit, vmax=vmax_fit, gamma=gamma))
    ax.plot(center_fit[1], center_fit[2], 'mx')
    if init_params is not None:
        ax.plot(center_guess[1], center_guess[2], 'gx')

    ax.set_ylim(extent[2:4])
    ax.set_xlim(extent[0:2])
    ax.set_xticks([])
    ax.set_yticks([])

    # ################################
    # ZX, fit
    # ################################
    ax = figh_interp.add_subplot(grid[0, 3])
    extent = [z_roi[0, 0, 0] - 0.5 * dz, z_roi[-1, 0, 0] + 0.5 * dz,
              x_roi[0, 0, 0] - 0.5 * dc, x_roi[0, 0, -1] + 0.5 * dc]
    ax.imshow(np.nanmax(img_fit, axis=1).transpose(), origin="lower", extent=extent, cmap=cmap,
              norm=PowerNorm(vmin=vmin_fit, vmax=vmax_fit, gamma=gamma))
    ax.plot(center_fit[0], center_fit[2], 'mx')
    if init_params is not None:
        ax.plot(center_guess[0], center_guess[2], 'gx')
    ax.set_ylim(extent[2:4])
    ax.set_xlim(extent[0:2])

    ax.set_ylabel("X (um)")

    # ################################
    # YZ, fit
    # ################################
    ax = figh_interp.add_subplot(grid[1, 4])
    extent = [y_roi[0, 0, 0] - 0.5 * dc, y_roi[0, -1, 0] + 0.5 * dc,
              z_roi[0, 0, 0] - 0.5 * dz, z_roi[-1, 0, 0] + 0.5 * dz]
    ax.imshow(np.nanmax(img_fit, axis=2), origin="lower", extent=extent, cmap=cmap,
              norm=PowerNorm(vmin=vmin_fit, vmax=vmax_fit, gamma=gamma))
    ax.plot(center_fit[1], center_fit[0], 'mx')
    if init_params is not None:
        ax.plot(center_guess[1], center_guess[0], 'gx')
    ax.set_ylim(extent[2:4])
    ax.set_xlim(extent[0:2])

    ax.set_xlabel("Y (um)")
    ax.set_ylabel("Z (um)")

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
        to comapre fit_params with initial parameters or etc.

        filter functions are free to ignore all parameters besides fit_params,
        but must accept arbitrary number of arguments
        @param fit_params:
        @param args: additional arguments which must all be arrays and must all have first dimensions of the same length
        @param kwargs: key-word arguments, which should be objects which apply to all points. This matter mostly
        if you want to us the __mul__ method. When applying filters where order matters, each element of *args
        will be reduced to an array which only keeps those elements that passed the previous filter, but
        **kwargs will not be touched
        @return:
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
        @param other:
        @return:
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
        @param other:
        @return:
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
    def __init__(self):
        self.condition_names = ["none"]

    def filter(self, fit_params, *args, **kwargs):
        conditions = np.ones((len(fit_params), 1), dtype=bool)
        return conditions


class range_filter(filter):
    """
    Filter based on value being in a certain range
    """
    def __init__(self, low, high, index, name):
        self.low = low
        self.high = high
        self.index = index
        self.condition_names = [f"{name:s} too small", f"{name:s} too large"]

    def filter(self, fit_params, *args, **kwargs):
        conditions = np.stack((fit_params[:, self.index] >= self.low, fit_params[:, self.index] <= self.high), axis=1)

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
    def __init__(self, dxy_min_dist, dz_min_dist, name="unique"):
        self.dxy_min_dist = dxy_min_dist
        self.dz_min_dist = dz_min_dist
        self.condition_names = [f"{name:s}"]

    def filter(self, fit_params, *args, **kwargs):
        _, unique_inds = filter_nearby_peaks(fit_params[:, (3, 2, 1)],
                                             self.dxy_min_dist,
                                             self.dz_min_dist,
                                             mode="keep-one")
        conditions = np.zeros((fit_params.shape[0], 1), dtype=bool)
        conditions[unique_inds] = True

        return conditions


def get_param_filter(coords: tuple[np.ndarray],
                     fit_dist_max_err: tuple[float],
                     min_spot_sep: tuple[float],
                     sigma_bounds: tuple[tuple[float], tuple[float]],
                     amp_bounds: tuple[float] = (0, 0),
                     dist_boundary_min: tuple[float] = (0, 0)):
    """
    Simple composite filter testing bounds of fit parameters
    @param coords: (z, y, x)
    @param fit_dist_max_err: (dmax_z, dmax_xy) maximum distance between
    @param min_spot_sep: (dz_min, dxy_min)
    @param sigma_bounds: ((sz_min, sxy_min), (sz_max, sxy_max))
    @param amp_bounds: (amp_min, amp_max)
    @param dist_boundary_min: (dz_min, dxy_min)
    @return filter:
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


def filter_localizations(fit_params: np.ndarray,
                         init_params: np.ndarray,
                         coords: tuple[np.ndarray],
                         fit_dist_max_err: tuple[float],
                         min_spot_sep: tuple[float],
                         sigma_bounds: tuple[tuple[float], tuple[float]],
                         amp_min: float = 0,
                         dist_boundary_min: tuple[float] = (0, 0)):
    """
    Given a list of fit parameters, return boolean arrays indicating which fits pass/fail given a variety
    of tests for reasonability

    :param fit_params: nfits x 7, results of fitting where the parameters are
     [amplitude, center x, center y, center z, size xy, size z, background]. Most commonly these values come from
     a 3D Gaussian fit. But other models can be used as well.
    :param init_params: nfits x 7, initial guess parameters used in fits
    :param coords: (z, y, x)
    :param fit_dist_max_err = (dz_max, dxy_max) maximum distance between guess value and fit value
    :param min_spot_sep: (dz, dxy) assume points separated by less than this distance come from one spot
    :param sigma_bounds: ((sz_min, sxy_min), (sz_max, sxy_max)) exclude fits with sigmas that fall outside
    these ranges
    :param amp_min: exclude fits with smaller amplitude
    :param dist_boundary_min: (dz_min, dxy_min)
    :return to_keep, conditions, condition_names, filter_settings:
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


def localize_beads_generic(imgs: np.ndarray,
                           drs: tuple[float],
                           threshold: float,
                           roi_size: tuple[float] = (4, 2, 2),
                           filter_sigma_small: tuple[float] = (1, 0.1, 0.1),
                           filter_sigma_large: tuple[float] = (10, 5, 5),
                           min_spot_sep: tuple[float] = (0, 0),
                           filter=None,
                           mask=None,
                           max_nfit_iterations: int = 100,
                           fit_filtered_images: bool = False,
                           use_gpu_fit: bool = GPUFIT_AVAILABLE,
                           use_gpu_filter: bool = CUPY_AVAILABLE,
                           verbose: bool = True,
                           model=psf.gaussian3d_psf_model(),
                           debug=False):
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

    @param imgs: an image of size ny x nx or an image stack of size nz x ny x nx
    @param drs: (dz, dy, dx))
    @param threshold: threshold used for identifying spots. This is applied after filtering of image
    @param roi_size: (sz, sy, sx) in um
    @param filter_sigma_small: (sz, sy, sx) small sigmas to be used in difference-of-Gaussian filter. Roughly speaking,
    features which are smaller than these sigmas will be high pass filtered out. To turn off this filter, set to None
    @param filter_sigma_large: (sz, sy, sx) large sigmas to be used in difference-of-Gaussian filter. Roughly speaking,
    features which are large than these sigmas will be low pass filtered out. To turn off this filter, set to None
    @param min_spot_sep: (dz, dxy) minimum separation allowed between adjacent peaks
    @param filter: filter will be applied with args = [fit_params, ref_params, chi_sqrs, niters, rois]
    and kwargs "image" and "image_filtered"
    @param mask: optionally boolean array of same size as image which indicates where to search for peaks
    @param sigma_bounds: ((sz_min, sxy_min), (sz_max, sxy_max))
    @param max_nfit_iterations: maximum number of iterations in fitting function
    @param fit_filtered_images: whether to perform fitting on raw images or filtered images
    @param use_gpu_fit: whether or not to do spot fitting on the GPU
    @param use_gpu_filter: whether or not to do difference-of-Gaussian filtering on GPU
    @param verbose: whether or not to print information
    @param model: an instance of a class derived from psf.psf_model. The model describes the PSF and how to 'fit' data
    to it. Information e.g. about pixelation can be provided to the model. See psf.psf_model and the derived classes
    for more details
    @return coords, fit_results, imgs_filtered: coords = (z, y, x)
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
    roi_size_pix = roi_fns.get_roi_size(roi_size, 0.5 * (dx + dy), dz, ensure_odd=True)

    # ###################################
    # filter images
    # ###################################
    tstart = time.perf_counter()

    # imgs = cp.asnumpy(cupyx.scipy.ndimage.median_filter(cp.array(imgs), size=(3, 3, 3)))

    if filter_sigma_small is not None:
        ks = get_filter_kernel(filter_sigma_small, (dz, dy, dx))
        imgs_hp = filter_convolve(imgs, ks, use_gpu=use_gpu_filter)
    else:
        imgs_hp = imgs

    if filter_sigma_large is not None:
        kl = get_filter_kernel(filter_sigma_large, (dz, dy, dx))
        imgs_lp = filter_convolve(imgs, kl, use_gpu=use_gpu_filter)
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
    centers_guess_inds, amps = find_peak_candidates(imgs_filtered, footprint, threshold, mask=mask)

    # todo: could autothreshold by trying to fit # of points versus threshold

    # real coordinates
    centers_guess = np.stack((z[centers_guess_inds[:, 0], 0, 0],
                              y[0, centers_guess_inds[:, 1], 0],
                              x[0, 0, centers_guess_inds[:, 2]]), axis=1)

    if verbose:
        print(f"identified {len(centers_guess_inds):d} candidates"
              f" in {time.perf_counter() - tstart:.2f}s")

    # ###################################
    # identify candidate beads
    # ###################################
    if len(centers_guess_inds) != 0:
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

        amps = amps[inds_comb]

        if verbose:
            print(f"Found {len(centers_guess):d} points separated by"
                  f" dxy > {dxy_min_sep:0.5g} and dz > {dz_min_sep:0.5g}"
                  f" in {time.perf_counter() - tstart:.1f}s")

        # ###################################################
        # prepare ROIs
        # ###################################################
        tstart = time.perf_counter()

        if fit_filtered_images:
            rois, img_rois, coords = zip(*[get_roi(c, imgs_filtered, (z, y, x), roi_size_pix) for c in centers_guess])
        else:
            rois, img_rois, coords = zip(*[get_roi(c, imgs, (z, y, x), roi_size_pix) for c in centers_guess])

        zrois, yrois, xrois = zip(*coords)
        rois = np.asarray(rois)

        # ###################################################
        # determine initial guess values for fits
        # ###################################################
        init_params = np.stack([model.estimate_parameters(img_rois[ii], (zrois[ii], yrois[ii], xrois[ii]))
                                for ii in range(len(img_rois))], axis=0)


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

        fixed_params = [False] * model.nparams

        # if 2D, don't want to fit cz or sz
        if data_is_2d:
            fixed_params[5] = True
            fixed_params[3] = True

        fit_params, fit_states, chi_sqrs, niters, fit_t = fit_gauss_rois(img_rois,
                                                                         (zrois, yrois, xrois),
                                                                         init_params,
                                                                         max_nfit_iterations,
                                                                         estimator="LSE",
                                                                         fixed_params=fixed_params,
                                                                         use_gpu=use_gpu_fit,
                                                                         verbose=verbose,
                                                                         model=model,
                                                                         debug=debug)

        tend = time.perf_counter()
        if verbose:
            print("Localization took %0.2fs" % (tend - tstart))

        # ###################################################
        # filter fits
        # ###################################################
        if filter is None:
            filter = no_filter()

        # add any other parameters which could be useful for the filter function to accept
        # filter functions are free to ignore all parameters besides fit_params,
        # but must accept arbitrary number of arguments and pass this through to other filters
        conditions = filter.filter(fit_params, init_params, chi_sqrs, niters, rois, image=imgs, image_filtered=imgs_filtered)

        to_keep = np.logical_and.reduce(conditions, axis=1)
        condition_names = filter.condition_names
        filter_settings = {}

        if verbose:
            print(f"Identified {np.sum(to_keep):d} likely candidates")

    else:
        fit_params = np.zeros((0, 7), dtype=float)
        init_params = np.zeros((0, 7), dtype=float)
        rois = np.zeros((0, 6), dtype=int)
        to_keep = None
        conditions = None
        condition_names = None
        filter_settings = None
        fit_states = None
        chi_sqrs = None
        niters = None

    fit_results = {"fit_params": fit_params,
                   "init_params": init_params,
                   "rois": rois,
                   "to_keep": to_keep,
                   "conditions": conditions,
                   "condition_names": condition_names,
                   "filter_settings": filter_settings,
                   "fit_states": fit_states,
                   "chi_sqrs": chi_sqrs,
                   "niters": niters}

    return (z, y, x), fit_results, imgs_filtered


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
                   use_gpu_fit: bool = GPUFIT_AVAILABLE,
                   use_gpu_filter: bool = CUPY_AVAILABLE,
                   verbose: bool = True):
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
                                  verbose=verbose)


def plot_bead_locations(imgs: np.ndarray,
                        center_lists: list[np.ndarray],
                        title: str = "",
                        color_lists: list[str] = None,
                        color_limits: list[list[float]] = None,
                        legend_labels: list[str] = None,
                        weights: list[np.ndarray] = None,
                        cbar_labels: list[str] = None,
                        coords: list = None,
                        vlims_percentile: tuple[float] = (0.01, 99.99),
                        gamma: float = 1,
                        **kwargs):
    """
    Plot center locations over 2D image or max projection of 3D image. Supports plotting multiple different sets
    of center locations, and using different colors to indicate properties of the different centers e.g. sigma,
    amplitude, modulation depth, etc.

    :param imgs: np.array either 3D or 2D. Dimensions order Z, Y, X
    :param center_lists: [center_array_1, center_array_2, ...] where each center_array is a numpy array of size N_i x 3
    consisting of triples of center values giving cz, cy, cx
    :param str title: title of plot
    :param color_lists: list of colors for each series to be plotted in
    :param color_limits: [[vmin_1, vmax_1], ...]
    :param legend_labels: labels for each series
    :param weights: list of arrays [w_1, ..., w_n], with w_i the same size as center_array_i, giving the intensity of
    the color to be plotted
    :param cbar_labels: list of labels for color bars
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

    im = ax.imshow(img_max_proj, norm=PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax), cmap=plt.cm.get_cmap("bone"), extent=extent_xy)
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
                 dx: float,
                 dz: float,
                 wavelength: float,
                 ni: float = 1.5,
                 model: str = 'vectorial',
                 threshold: float = 100.,
                 min_spot_sep: tuple[float] = (0., 0.),
                 filter_sigma_small: tuple[float] = (1, 0.5, 0.5),
                 filter_sigma_large: tuple[float] = (3, 5, 5),
                 sigma_bounds: tuple[tuple[float]] = ((0., 0.), (np.inf, np.inf)),
                 roi_size_loc=(13, 21, 21),
                 fit_amp_thresh: float = 100.,
                 dist_boundary_min: tuple[float] = (0., 0.),
                 max_number_iterations: int = 100,
                 fit_dist_max_err: tuple[float] = (np.inf, np.inf),
                 num_localizations_to_plot: int = 5,
                 psf_percentiles: tuple[float] = (5,),
                 plot: bool = True,
                 only_plot_good_fits: bool = True,
                 plot_filtered_image: bool = False,
                 use_gpu_filter: bool = False,
                 gamma: float = 0.5,
                 save_dir: str = None,
                 figsize=(18, 10),
                 **kwargs):
    """
    Given a 2D or 3D image, identify and localize diffraction limited spots. Aggregate the results to create
    an experimental point-spread function and fit the average PSF to a model function.

    :param imgs: nz x nx x ny image
    :param psf_roi_size: [nz, ny, nx] ROI to determine PSFs on. This is not necessarily the same as the size of the
    ROI's used during localization
    :param dx: pixel size in um
    :param dz: z-plane spacing in um
    :param wavelength: wavelength in um
    :param ni: index of refraction of medium
    :param model: "vectorial", "gibson-lanni", "born-wolf", or "gaussian". This model is used to fit the
    averaged PSF's, but a gaussian model is always used for localization
    :param threshold: threshold pixel value to identify peaks. Note: this is applied to the filtered image,
    and is not directly comparable to the values in the raw image array
    :param min_spot_sep: (sz, sxy) minimum spot separation between different beads in um
    :param filter_sigma_small: (sz, sy, sx) sigmas of Gaussian filter used to smooth image in um
    :param filter_sigma_large: (sz, sy, sx) sigmas of Gaussian filter used to removed background in um
     :param sigma_bounds: ((sz_min, sxy_min), (sz_max, sxy_max)) in um. exclude fits with sigmas that fall outside
    these ranges
    :param roi_size_loc: (sz, sy, sx) size of ROI to used in localization, in um
    :param float fit_amp_thresh: only consider spots which have fit values larger tha this amplitude
    :param dist_boundary_min:
    :param fit_dist_max_err:
    :param int num_localizations_to_plot: number of ROI's to plot
    :param tuple psf_percentiles: calculate the averaged PSF from the smallest supplied percentage of spots. When
    a tuple is given, compute PSF's corresponding to each supplied percentage.
    :param bool plot: optionally plot diagnostics
    :param bool only_plot_good_fits: when plotting ROI, plot only fits that passed all filtering tests or plot all fits
    :param bool plot_filtered_image: plot ROI's against filtered image also
    :param float gamma: gamma to use when plotting
    :param str save_dir: directory to save diagnostic plots. If None, these will not be saved
    :param figsize: (sx, sy)
    :param **kwargs: passed through to plt.figure()

    :return coords, fit_params, init_params, rois, \
           to_keep, conditions, condition_names, filter_settings,\
           fit_states, chi_sqrs, niters, \
           psfs_real, psf_coords, otfs_real, otf_coords, psf_percentiles, fit_params_real:
    """

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
    z, y, x = get_coords(imgs.shape, (dz, dx, dx))

    filter = get_param_filter((z, y, x),
                              fit_dist_max_err=fit_dist_max_err,
                              min_spot_sep=min_spot_sep,
                              sigma_bounds=sigma_bounds,
                              amp_bounds=(0, fit_amp_thresh),
                              dist_boundary_min=dist_boundary_min)

    coords, fit_results, imgs_filtered = localize_beads_generic(imgs,
                                                                drs=(dz, dx, dx),
                                                                threshold=threshold,
                                                                roi_size=roi_size_loc,
                                                                filter_sigma_small=filter_sigma_small,
                                                                filter_sigma_large=filter_sigma_large,
                                                                min_spot_sep=min_spot_sep,
                                                                filter=filter,
                                                                max_nfit_iterations=max_number_iterations,
                                                                use_gpu_filter=use_gpu_filter)

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
    if plot:
        print("plotting ROI's")
        tstart_plot_roi = time.perf_counter()

        if only_plot_good_fits:
            ind_to_plot = np.arange(len(to_keep), dtype=int)[to_keep][:num_localizations_to_plot]
        else:
            ind_to_plot = np.arange(len(to_keep), dtype=int)[:num_localizations_to_plot]

        results = joblib.Parallel(n_jobs=-1, verbose=10, timeout=None)(
            joblib.delayed(plot_gauss_roi)(fit_params[ind], rois[ind], imgs, coords, init_params[ind],
                                           figsize=figsize,
                                           prefix="localization_roi_%d" % ind,
                                           title="filter conditions = " + " ".join(
                                                ["%d," % c for c in conditions[ind]]),
                                           save_dir=save_dir)
            for ind in ind_to_plot
        )

        if plot_filtered_image:
            results = joblib.Parallel(n_jobs=-1, verbose=10, timeout=None)(
                joblib.delayed(plot_gauss_roi)(fit_params[ind], rois[ind], imgs_filtered, coords,
                                               init_params[ind], same_color_scale=False,
                                               figsize=figsize,
                                               prefix="localization_roi_%d_filtered" % ind,
                                               title="filter conditions = " + " ".join(
                                                    ["%d," % c for c in conditions[ind]]),
                                               save_dir=save_dir)
                for ind in ind_to_plot
            )

        print("plotting took %0.2fs" % (time.perf_counter() - tstart_plot_roi))

    # ###################################
    # plot fit statistics
    # ###################################
    if plot and not no_psfs_found:
        # figh = psf.plot_fit_stats(fit_params[to_keep], figsize=figsize, **kwargs)

        # fit parameter summary
        figh = plt.figure(figsize=figsize, **kwargs)
        plt.suptitle("Localization fit parameter summary")
        grid = plt.GridSpec(2, 2, hspace=1, wspace=0.5)

        # amplitude vs sxy
        ax = plt.subplot(grid[0, 0])
        ax.plot(fit_params[to_keep, 4], fit_params[to_keep, 0], '.')
        ax.set_xlabel(r"$\sigma_{xy}$ ($\mu m$)")
        ax.set_ylabel("amp")

        # sxy vs sz
        ax = plt.subplot(grid[0, 1])
        ax.plot(fit_params[to_keep, 4], fit_params[to_keep, 5], '.')
        ax.set_xlabel(r"$\sigma_{xy}$ ($\mu m$)")
        ax.set_ylabel(r"$\sigma_{z}$ ($\mu m$)")

        # sxy vs bg
        ax = plt.subplot(grid[1, 1])
        ax.plot(fit_params[to_keep, 4], fit_params[to_keep, 6], '.')
        ax.set_xlabel(r"$\sigma_{xy}$ ($\mu m$)")
        ax.set_ylabel(r"$\sigma_{z}$ ($\mu m$)")


        if saving:
            figh.savefig(Path(save_dir) / "fit_stats.png")
            plt.close(figh)

    # ###################################
    # get and plot experimental PSFs
    # ###################################
    nps = len(psf_percentiles)
    psf_roi_size_pix = np.round(np.array(psf_roi_size) / np.array([dz, dx, dx])).astype(int)
    psf_roi_size_pix += (1 - np.mod(psf_roi_size_pix, 2))

    psfs_real = np.zeros((nps,) + tuple(psf_roi_size_pix))
    otfs_real = np.zeros(psfs_real.shape, dtype=complex)
    fit_params_real = np.zeros((nps, 6))
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

            # find experiment psf/otf
            psfs_real[ii], psf_coords, otfs_real[ii], otf_coords = psf.get_exp_psf(imgs, (z, y, x), centers, psf_roi_size,
                                                                               backgrounds=fit_params[:, 5][to_use])

            results, _ = psf.fit_psfmodel(psfs_real[ii], dx, dz, wavelength, ni, 1, model=model)
            fit_params_real[ii] = results["fit_params"]

            if plot:
                figh = plot_gauss_roi(fit_params_real[ii],
                                      [0, psfs_real[ii].size[0],
                                       0, psfs_real[ii].size[1],
                                       0, psfs_real[ii].size[2]],
                                      psfs_real[ii],
                                      coords,
                                      model=model,
                                      string=f"smallest {psf_percentiles[ii]:.0f} percent, {type(model)}, sf={model.sf}",
                                      gamma=gamma,
                                      figsize=figsize,
                                      **kwargs)

                # figh = psf.plot_psf_fit(psfs_real[ii],
                #                         model.model(coords, fit_params_real[ii]),
                #                         coords,
                #                         fit_params,
                #                         fit_param_names=model.parameter_names,
                #                         label=f"smallest {psf_percentiles[ii]:.0f} percent, {type(model)}, sf={model.sf}",
                #                         gamma=gamma,
                #                         figsize=figsize,
                #                         **kwargs)

                if saving:
                    figh.savefig(Path(save_dir) / "experimental_psf_smallest_{psf_percentiles[ii]:.2f}.png")
                    plt.close(figh)

    # ###################################
    # plot localization positions
    # ###################################
    if plot and not no_psfs_found:
        centers_all = np.stack((fit_params[:, 3],
                                fit_params[:, 2],
                                fit_params[:, 1]), axis=1)

        extent = [x.min() - 0.5 * dx, x.max() + 0.5 * dx, y.max() + 0.5 * dx, y.min() - 0.5 * dx]

        figh = plot_bead_locations(imgs, [centers_all, centers_all[to_keep]],
                                   weights=[np.ones(len(centers_all)), fit_params[to_keep, 4]],
                                   cbar_labels=["all fits", r"kept fits, $\sigma_{xy} (\mu m)$"],
                                   title="Max intensity projection and size from 2D fit versus position",
                                   extent=extent, gamma=gamma, figsize=figsize)

        if saving:
            figh.savefig(Path(save_dir) / "sigma_versus_position.png")
            plt.close(figh)

    data = {"coords": coords,
            "fit_params": fit_params,
            "init_params": init_params,
            "rois": rois,
            "to_keep": to_keep,
            "conditions": conditions,
            "condition_names": condition_names,
            "filter_settings": filter_settings,
            "fit_states": fit_states,
            "chi_sqrs": chi_sqrs,
            "niterations": niters,
            "psfs_real": psfs_real,
            "psf_coords": psf_coords,
            "otfs_real": otfs_real,
            "otf_coords": otf_coords,
            "psf_percentiles": psf_percentiles,
            "fit_params_real": fit_params_real}

    if saving:
        z = zarr.open(Path(save_dir) / "localization_results.zarr", "w")
        for k, v in data.items():
            z.array(k, v, compressor="none")

    return data
