"""
Code for localizing and fitting (typically diffraction limited) spots/beads

The fitting code can be run on a CPU using multiprocessing with joblib, or on a GPU using custom modifications
to GPUfit which can be found at https://github.com/QI2lab/Gpufit. To use the GPU code, you must download and
compile this repository and install the python bindings.
"""
import os
import time
import warnings
import pickle
import numpy as np
import scipy.signal
import scipy.ndimage
import joblib
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


def find_peak_candidates(imgs, footprint, threshold, use_gpu_filter=CUPY_AVAILABLE):
    """
    Find peak candidates in image using maximum filter

    :param imgs: 2D or 3D array
    :param footprint: footprint to use for maximum filter. Array should have same number of dimensions as imgs.
    This can be obtained from get_max_filter_footprint()
    :param float threshold: only pixels with values greater than or equal to the threshold will be considered
    :param bool use_gpu_filter: whether or not to do maximum filter on GPU
    :return centers_guess_inds: np.array([[i0, i1, i2], ...]) array indices of local maxima
    """

    if use_gpu_filter:
        img_max_filtered = cp.asnumpy(
            cupyx.scipy.ndimage.maximum_filter(cp.asarray(imgs, dtype=cp.float32), footprint=cp.asarray(footprint)))
        # need to compare imgs as float32 because img_max_filtered will be ...
        is_max = np.logical_and(imgs.astype(np.float32) == img_max_filtered, imgs >= threshold)
    else:
        img_max_filtered = scipy.ndimage.maximum_filter(imgs, footprint=footprint)
        is_max = np.logical_and(imgs == img_max_filtered, imgs >= threshold)

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
def fit_gauss_roi(img_roi, coords, init_params=None, fixed_params=None, bounds=None, sf=1, dc=None, angles=None):
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
    :param int sf: over-sampling factor
    :param float dc: pixel size, only required if sf > 1
    :param angles:
    :return results: dictionary object containing information about fitting
    """
    z_roi, y_roi, x_roi = coords

    # if img_roi is 2D and z- dimension size is 0, treat as 3D
    if img_roi.ndim == 2 and z_roi.shape[0] == 1:
        img_roi = np.expand_dims(img_roi, axis=0)

    # img_roi must be 3D
    if img_roi.ndim != 3:
        raise ValueError(f"img_roi must have 3 dimensions but had {img_roi.ndim:d}")

    to_use = np.logical_not(np.isnan(img_roi))
    x_roi_full, y_roi_full, z_roi_full = np.broadcast_arrays(x_roi, y_roi, z_roi)

    if init_params is None:
        init_params = [None] * 7

    if np.any([ip is None for ip in init_params]):
        # set initial parameters
        min_val = np.nanmin(img_roi)
        img_roi -= min_val  # so will get ok values for moments
        mx1 = np.nansum(img_roi * x_roi) / np.nansum(img_roi)
        mx2 = np.nansum(img_roi * x_roi ** 2) / np.nansum(img_roi)
        my1 = np.nansum(img_roi * y_roi) / np.nansum(img_roi)
        my2 = np.nansum(img_roi * y_roi ** 2) / np.nansum(img_roi)
        sxy = np.sqrt(np.sqrt(my2 - my1 ** 2) * np.sqrt(mx2 - mx1 ** 2))
        mz1 = np.nansum(img_roi * z_roi) / np.nansum(img_roi)
        mz2 = np.nansum(img_roi * z_roi ** 2) / np.nansum(img_roi)
        sz = np.sqrt(mz2 - mz1 ** 2)
        img_roi += min_val  # put back to before

        ip_default = [np.nanmax(img_roi) - np.nanmean(img_roi), mx1, my1, mz1, sxy, sz, np.nanmean(img_roi)]

        # set any parameters that were None to the default values
        for ii in range(len(init_params)):
            if init_params[ii] is None:
                init_params[ii] = ip_default[ii]

    if bounds is None:
        # set bounds
        bounds = [[0, x_roi_full[to_use].min(), y_roi_full[to_use].min(), z_roi_full[to_use].min(), 0, 0, -np.inf],
                  [np.inf, x_roi_full[to_use].max(), y_roi_full[to_use].max(), z_roi_full[to_use].max(), np.inf, np.inf,
                   np.inf]]

        # ensure guesses were not already outside bounds. If they were, reset bounds
        for ii in range(len(init_params)):
            if init_params[ii] < bounds[0][ii]:
                bounds[0][ii] = -np.inf
            if init_params[ii] > bounds[1][ii]:
                bounds[1][ii] = np.inf

    # gaussian fitting localization
    def model_fn(p):
        return psf.gaussian3d_psf(x_roi, y_roi, z_roi, dc, p, sf=sf, angles=angles)

    def jac_fn(p):
        return psf.gaussian3d_psf_jac(x_roi, y_roi, z_roi, dc, p, sf=sf, angles=angles)

    # do fitting
    results = fit.fit_model(img_roi, model_fn, init_params, bounds=bounds,
                            fixed_params=fixed_params, model_jacobian=jac_fn)

    return results


def fit_gauss_rois(img_rois, coords_rois, init_params, max_number_iterations=100,
                   sf=1, dc=None, angles=None, estimator="LSE", model="gaussian",
                   fixed_params=None, use_gpu=GPUFIT_AVAILABLE, verbose=True, debug=False):
    """
    Fit rois. Can use either CPU parallelization with joblib or GPU parallelization using gpufit

    :param img_rois: list of image rois
    :param coords_rois: ((z0, y0, x0), (z1, y1, x1), ....)
    :param init_params: initial parameters for fits, size nfits x nparams
    :param int max_number_iterations: maximum number of iterations to be used for each fit
    :param int sf: oversampling factor for simulating pixelation
    :param float dc: pixel size, only used for oversampling
    :param (float, float, float) angles: euler angles describing pixel orientation. Only used for oversampling.
    :param estimator: "LSE" or "MLE"
    :param model: "gaussian", "rotated-gaussian", "gaussian-lorentzian"
    :param list[bool] fixed_params: length nparams vector of parameters to fix. only supports fixing/unfixing
    each parameter for all fits
    :param bool use_gpu:
    :return fit_params, fit_states, chi_sqrs, niters, fit_t:
    """

    if fixed_params is None:
        fixed_params = [False] * 7

    zrois, yrois, xrois = coords_rois

    for ii in range(len(img_rois)):
        if img_rois[ii].ndim != 3:
            raise ValueError(f"img_rois position {ii:d} was not 3-dimensional")


    if not use_gpu:
        if model != "gaussian":
            raise NotImplementedError("only model = 'gaussian' implemented for non gpu")

        verbose_joblib = 0
        if verbose:
            verbose_joblib = 1

        tstart = time.perf_counter()

        if debug:
            results = []
            for ii in range(len(img_rois)):
                results.append(fit_gauss_roi(img_rois[ii], (zrois[ii], yrois[ii], xrois[ii]),
                                              init_params=init_params[ii],
                                              fixed_params=fixed_params, sf=sf, dc=dc, angles=angles))
        else:
            results = joblib.Parallel(n_jobs=-1, verbose=verbose_joblib, timeout=None)(
                joblib.delayed(fit_gauss_roi)(img_rois[ii], (zrois[ii], yrois[ii], xrois[ii]),
                                              init_params=init_params[ii],
                                              fixed_params=fixed_params, sf=sf, dc=dc, angles=angles)
                for ii in range(len(img_rois)))
        tend = time.perf_counter()

        fit_params = np.asarray([r["fit_params"] for r in results])
        chi_sqrs = np.asarray([r["chi_squared"] for r in results])
        fit_states = np.asarray([r["status"] for r in results])
        niters = np.asarray([r["nfev"] for r in results])
        fit_t = (tend - tstart)

    else:
        if sf != 1:
            raise NotImplementedError("sampling factors other than 1 are not implemented for GPU fitting")
        # todo: if requires more memory than GPU has, split into chunks

        # ensure arrays are row vectors
        xrois, yrois, zrois, img_rois = zip(
            *[(xr.ravel()[None, :], yr.ravel()[None, :], zr.ravel()[None, :], ir.ravel()[None, :])
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

        # build user info
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
        if init_params.ndim != 2 or init_params.shape != (nfits, 7):
            raise ValueError("init_params had wrong shape or dimension")
        if user_info.ndim != 1 or user_info.size != (3 * nfits * n_pts_per_fit + nfits):
            raise ValueError("user_info had wrong shape or dimension")

        if estimator == "MLE":
            est_id = gf.EstimatorID.MLE
        elif estimator == "LSE":
            est_id = gf.EstimatorID.LSE
        else:
            raise ValueError("'estimator' must be 'MLE' or 'LSE' but was '%s'" % estimator)

        if model == "gaussian":
            model_id = gf.ModelID.GAUSS_3D_ARB
        elif model == "gaussian-lorentzian":
            model_id = gf.ModelID.GAUSS_LOR_3D_ARB
        elif model == "rotated-gaussian":
            model_id = gf.ModelID.GAUSS_3D_ROT_ARB
        else:
            raise ValueError("'model' must be 'gaussian' or 'gaussian-lorentzian' but was '%s'" % model)

        # set which parameters to fit/fix
        params_to_fit = np.array([not fp for fp in fixed_params], dtype=np.int32)
        # do fitting
        fit_params, fit_states, chi_sqrs, niters, fit_t = gf.fit(data, None, model_id, init_params,
                                                                 max_number_iterations=max_number_iterations,
                                                                 estimator_id=est_id,
                                                                 parameters_to_fit=params_to_fit,
                                                                 user_info=user_info)

        # correct sigmas in case negative
        fit_params[:, 4] = np.abs(fit_params[:, 4])
        fit_params[:, 5] = np.abs(fit_params[:, 5])

    return fit_params, fit_states, chi_sqrs, niters, fit_t


def plot_gauss_roi(fit_params, roi, imgs, coords, init_params=None, same_color_scale=True,
                   fit_fn=None,
                   figsize=(16, 8), prefix="", save_dir=None):
    """
    Plot results obtained from fitting functions fit_gauss_roi() or fit_gauss_rois()
    :param fit_params:
    :param roi: [zstart, zend, ystart, yend, xstart, xend]
    :param imgs: full image, such that imgs[zstart:zend, ystart:yend, xstart:xend] is the region that was fit
    :param coords: (z, y, x) broadcastable to same size as imgs
    :param init_params: initial parameters used in fit, optional
    :param bool same_color_scale: whether or not to use same color scale for data and fits
    :param fit_fn: function used for fitting. Must have arguments (x, y, z, dc, fit_params, sf=1). Default
     fit function is psf.gaussian3d_psf()
    :param figsize: (sx, sz)
    :param str prefix: prefix prepended before save name
    :param str save_dir: if None, do not save results
    :return:
    """
    if fit_fn is None:
        fit_fn = psf.gaussian3d_psf

    z, y, x = coords
    # extract useful coordinate info
    dc = x[0, 0, 1] - x[0, 0, 0]

    if z.shape[0] > 1:
        dz = z[1, 0, 0] - z[0, 0, 0]
    else:
        dz = dc

    if init_params is not None:
        center_guess = np.array([init_params[3], init_params[2], init_params[1]])

    center_fit = np.array([fit_params[3], fit_params[2], fit_params[1]])

    # get ROI and coordinates
    img_roi = roi_fns.cut_roi(roi, imgs)
    # x_roi = x[:, :, roi[4]:roi[5]]
    # y_roi = y[:, roi[2]:roi[3], :]
    # z_roi = z[roi[0]:roi[1], :, :]
    x_roi = roi_fns.cut_roi(roi, x)
    y_roi = roi_fns.cut_roi(roi, y)
    z_roi = roi_fns.cut_roi(roi, z)

    vmin_roi = np.percentile(img_roi[np.logical_not(np.isnan(img_roi))], 1)
    vmax_roi = np.percentile(img_roi[np.logical_not(np.isnan(img_roi))], 99.9)

    # git fit
    img_fit = fit_fn(x_roi, y_roi, z_roi, dc, fit_params, sf=1)

    # ################################
    # plot results interpolated on regular grid
    # ################################
    figh_interp = plt.figure(figsize=figsize)
    st_str = f"Fit, max projections, interpolated, ROI = {roi}"

    st_str += f"\n{'fit': <10}" + "A=%3.3f, cx=%3.5f, cy=%3.5f, cz=%3.5f, sxy=%3.5f, sz=%3.5f, bg=%3.3f" % tuple(fit_params)
    if init_params is not None:
        st_str += f"\n{'guess': <10}" + "A=%3.3f, cx=%3.5f, cy=%3.5f, cz=%3.5f, sxy=%3.5f, sz=%3.5f, bg=%3.3f" % tuple(init_params)

    figh_interp.suptitle(st_str)

    grid = figh_interp.add_gridspec(2, 4)

    # ################################
    # XY, data
    # ################################
    ax = figh_interp.add_subplot(grid[0, 1])
    extent = [y_roi[0, 0, 0] - 0.5 * dc, y_roi[0, -1, 0] + 0.5 * dc,
              x_roi[0, 0, 0] - 0.5 * dc, x_roi[0, 0, -1] + 0.5 * dc]
    ax.imshow(np.nanmax(img_roi, axis=0).transpose(), vmin=vmin_roi, vmax=vmax_roi, origin="lower",
               extent=extent, cmap="bone")

    ax.plot(center_fit[1], center_fit[2], 'mx')
    if init_params is not None:
        ax.plot(center_guess[1], center_guess[2], 'gx')

    ax.set_ylim(extent[2:4])
    ax.set_xlim(extent[0:2])
    ax.set_xlabel("Y (um)")
    ax.set_ylabel("X (um)")
    ax.set_title("XY")

    # ################################
    # XZ, data
    # ################################
    ax = figh_interp.add_subplot(grid[0, 0])
    extent = [z_roi[0, 0, 0] - 0.5 * dz, z_roi[-1, 0, 0] + 0.5 * dz,
              x_roi[0, 0, 0] - 0.5 * dc, x_roi[0, 0, -1] + 0.5 * dc]
    ax.imshow(np.nanmax(img_roi, axis=1).transpose(), vmin=vmin_roi, vmax=vmax_roi, origin="lower",
               extent=extent, cmap="bone")
    ax.plot(center_fit[0], center_fit[2], 'mx')
    if init_params is not None:
        ax.plot(center_guess[0], center_guess[2], 'gx')
    ax.set_ylim(extent[2:4])
    ax.set_xlim(extent[0:2])
    ax.set_xlabel("Z (um)")
    ax.set_ylabel("X (um)")
    ax.set_title("XZ")

    # ################################
    # YZ, data
    # ################################
    ax = figh_interp.add_subplot(grid[1, 1])
    extent = [y_roi[0, 0, 0] - 0.5 * dc, y_roi[0, -1, 0] + 0.5 * dc,
              z_roi[0, 0, 0] - 0.5 * dz, z_roi[-1, 0, 0] + 0.5 * dz]
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')

        ax.imshow(np.nanmax(img_roi, axis=2), vmin=vmin_roi, vmax=vmax_roi, origin="lower", extent=extent, cmap="bone")

    ax.plot(center_fit[1], center_fit[0], 'mx')
    if init_params is not None:
        ax.plot(center_guess[1], center_guess[0], 'gx')

    ax.set_ylim(extent[2:4])
    ax.set_xlim(extent[0:2])
    ax.set_xlabel("Y (um)")
    ax.set_ylabel("Z (um)")
    ax.set_title("YZ")

    if same_color_scale:
        vmin_fit = vmin_roi
        vmax_fit = vmax_roi
    else:
        vmin_fit = np.percentile(img_fit, 1)
        vmax_fit = np.percentile(img_fit, 99.9)

    # ################################
    # YX, fit
    # ################################
    ax = figh_interp.add_subplot(grid[0, 3])
    extent = [y_roi[0, 0, 0] - 0.5 * dc, y_roi[0, -1, 0] + 0.5 * dc,
              x_roi[0, 0, 0] - 0.5 * dc, x_roi[0, 0, -1] + 0.5 * dc]
    ax.imshow(np.nanmax(img_fit, axis=0).transpose(), vmin=vmin_fit, vmax=vmax_fit,
               origin="lower", extent=extent, cmap="bone")
    ax.plot(center_fit[1], center_fit[2], 'mx')
    if init_params is not None:
        ax.plot(center_guess[1], center_guess[2], 'gx')
    ax.set_ylim(extent[2:4])
    ax.set_xlim(extent[0:2])
    ax.set_xlabel("Y (um)")
    ax.set_ylabel("X (um)")

    # ################################
    # ZX, fit
    # ################################
    ax = figh_interp.add_subplot(grid[0, 2])
    extent = [z_roi[0, 0, 0] - 0.5 * dz, z_roi[-1, 0, 0] + 0.5 * dz,
              x_roi[0, 0, 0] - 0.5 * dc, x_roi[0, 0, -1] + 0.5 * dc]
    ax.imshow(np.nanmax(img_fit, axis=1).transpose(), vmin=vmin_fit, vmax=vmax_fit,
               origin="lower", extent=extent, cmap="bone")
    ax.plot(center_fit[0], center_fit[2], 'mx')
    if init_params is not None:
        ax.plot(center_guess[0], center_guess[2], 'gx')
    ax.set_ylim(extent[2:4])
    ax.set_xlim(extent[0:2])
    ax.set_xlabel("Z (um)")
    ax.set_ylabel("X (um)")

    # ################################
    # YZ, fit
    # ################################
    ax = figh_interp.add_subplot(grid[1, 3])
    extent = [y_roi[0, 0, 0] - 0.5 * dc, y_roi[0, -1, 0] + 0.5 * dc,
              z_roi[0, 0, 0] - 0.5 * dz, z_roi[-1, 0, 0] + 0.5 * dz]
    ax.imshow(np.nanmax(img_fit, axis=2), vmin=vmin_fit, vmax=vmax_fit,
               origin="lower", extent=extent, cmap="bone")
    ax.plot(center_fit[1], center_fit[0], 'mx')
    if init_params is not None:
        ax.plot(center_guess[1], center_guess[0], 'gx')
    ax.set_ylim(extent[2:4])
    ax.set_xlim(extent[0:2])
    ax.set_xlabel("Y (um)")
    ax.set_ylabel("Z (um)")

    if save_dir is not None:
        figh_interp.savefig(os.path.join(save_dir, "%s.png" % prefix))
        plt.close(figh_interp)

    return figh_interp


def filter_localizations(fit_params: np.ndarray, init_params: np.ndarray, coords: tuple[np.ndarray],
                         fit_dist_max_err: tuple[float], min_spot_sep: tuple[float],
                         sigma_bounds: tuple[tuple[float], tuple[float]], amp_min: float = 0,
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

    filter_settings = {"fit_dist_max_err": fit_dist_max_err, "min_spot_sep": min_spot_sep,
                       "sigma_bounds": sigma_bounds, "amp_min": amp_min, "dist_boundary_min": dist_boundary_min}

    z, y, x = coords
    centers_guess = np.concatenate((init_params[:, 3][:, None], init_params[:, 2][:, None],
                                    init_params[:, 1][:, None]), axis=1)
    centers_fit = np.concatenate((fit_params[:, 3][:, None], fit_params[:, 2][:, None],
                                  fit_params[:, 1][:, None]), axis=1)

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


def localize_beads(imgs: np.ndarray, dxy: float, dz: float, threshold: float,
                   roi_size: tuple[float] = (4, 2, 2),
                   filter_sigma_small: tuple[float] = (1, 0.1, 0.1),
                   filter_sigma_large: tuple[float] = (10, 5, 5),
                   min_spot_sep: tuple[float] = (0, 0),
                   sigma_bounds: tuple[tuple[float], tuple[float]] = ((0, 0), (np.inf, np.inf)),
                   fit_amp_min: float = 0, fit_dist_max_err: tuple[float] = (np.inf, np.inf),
                   dist_boundary_min: tuple[float] = (0, 0), max_nfit_iterations: int = 100,
                   fit_filtered_images: bool = False, sf: int=1,
                   use_gpu_fit: bool = GPUFIT_AVAILABLE, use_gpu_filter: bool = CUPY_AVAILABLE, verbose: bool = True):
    """
    Given an image consisting of diffraction limited spots and background, identify the diffraction limit spots using
    the following procedure
    (1) Obtain a filtered image using a difference-of-Gaussians filter
    (2) Identify candidate spots from the filtered image using a threshold and maximum filter
    (3) Fit candidate spots to a 2D or 3D Gaussian function. Note the fitting is done on the raw image, not the
    filtered image
    (4) Filter out likely candidate spots based on the results of the fitting
    the various parameters used in this function are set in terms of real units, i.e. um, and not pixels. To use
    pixel units, set dxy=dz=1

    @param imgs: an image of size ny x nx or an image stack of size nz x ny x nx
    @param dxy: xy-pixel spacing in um
    @param dz: z-plane spacing in um
    @param threshold: threshold used for identifying spots. This is applied after filtering of image
    @param roi_size: (sz, sy, sx) in um
    @param filter_sigma_small: (sz, sy, sx) small sigmas to be used in difference-of-Gaussian filter. Roughly speaking,
    features which are smaller than these sigmas will be high pass filtered out.
    @param filter_sigma_large: (sz, sy, sx) large sigmas to be used in difference-of-Gaussian filter. Roughly speaking,
    features which are large than these sigmas will be low pass filtered out.
    @param min_spot_sep: (dz, dxy) minimum separation allowed between adjacent peaks
    @param sigma_bounds: ((sz_min, sxy_min), (sz_max, sxy_max))
    @param fit_amp_min: minimum amplitude value for fit to be kept
    @param fit_dist_max_err: (dz_max, dxy_max) maximum distance between guess value and fit value
    @param dist_boundary_min: (dz_min, dxy_min) filter out spots which are closer to the boundary than this
    @param max_nfit_iterations: maximum number of iterations in fitting function
    @param fit_filtered_images: whether to perform fitting on raw images or filtered images
    @param sf: oversampling factored used in PSF model to simulate pixelation
    @param bool use_gpu_fit: whether or not to do spot fitting on the GPU
    @param bool use_gpu_filter: whether or not to do difference-of-Gaussian filtering on GPU
    @param bool verbose: whether or not to print information
    @return coords, fit_params, init_params, rois, to_keep, conditions, condition_names, filter_settings,
    fit_states, chi_sqrs, niters, imgs_filtered:
    coords = (z, y, x)
    """

    # make sure inputs correct types
    roi_size = np.array(roi_size, copy=True)
    min_spot_sep = np.array(min_spot_sep, copy=True)
    filter_sigma_large = np.array(filter_sigma_large, copy=True)
    filter_sigma_small = np.array(filter_sigma_small, copy=True)
    dist_boundary_min = np.array(dist_boundary_min, copy=True)

    # check if is 2D
    if imgs.ndim == 2:
        imgs = np.expand_dims(imgs, axis=0)

    data_is_2d = imgs.shape[0] == 1

    if data_is_2d:
        roi_size[0] = 1
        filter_sigma_large[0] = 0
        filter_sigma_small[0] = 0
        min_spot_sep[0] = 0
        dist_boundary_min[0] = 0
        dz = 1 # dz not meaningful in this case

    # unpack arguments
    z, y, x = get_coords(imgs.shape, (dz, dxy, dxy))
    dz_min_sep, dxy_min_sep = min_spot_sep
    roi_size_pix = roi_fns.get_roi_size(roi_size, dxy, dz, ensure_odd=True)

    # ###################################
    # filter images
    # ###################################
    tstart = time.perf_counter()

    # imgs = cp.asnumpy(cupyx.scipy.ndimage.median_filter(cp.array(imgs), size=(3, 3, 3)))

    ks = get_filter_kernel(filter_sigma_small, (dz, dxy, dxy))
    kl = get_filter_kernel(filter_sigma_large, (dz, dxy, dxy))
    imgs_hp = filter_convolve(imgs, ks, use_gpu=use_gpu_filter)
    imgs_lp = filter_convolve(imgs, kl, use_gpu=use_gpu_filter)
    imgs_filtered = imgs_hp - imgs_lp

    if verbose:
        print("filtered image in %0.2fs" % (time.perf_counter() - tstart))

    # ###################################
    # identify candidate beads
    # ###################################
    tstart = time.perf_counter()

    footprint = get_max_filter_footprint((dz_min_sep, dxy_min_sep, dxy_min_sep), (dz, dxy, dxy))
    centers_guess_inds, amps = find_peak_candidates(imgs_filtered, footprint, threshold)

    # real coordinates
    centers_guess = np.stack((z[centers_guess_inds[:, 0], 0, 0],
                              y[0, centers_guess_inds[:, 1], 0],
                              x[0, 0, centers_guess_inds[:, 2]]), axis=1)

    if verbose:
        print("identified %d candidates in %0.2fs" % (len(centers_guess_inds), time.perf_counter() - tstart))

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
        centers_guess, inds_comb = filter_nearby_peaks(centers_guess, dxy_min_sep, dz_min_sep,
                                                       weights=weights, mode="average", nmax=np.inf)

        amps = amps[inds_comb]
        if verbose:
            print("Found %d points separated by dxy > %0.5g and dz > %0.5g in %0.1fs" %
                  (len(centers_guess), dxy_min_sep, dz_min_sep, time.perf_counter() - tstart))

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

        # extract guess values
        bgs = np.array([np.mean(r) for r in img_rois])
        sxs = np.zeros(len(img_rois))
        sys = np.zeros(len(img_rois))
        szs = np.zeros(len(img_rois))
        for ii in range(len(img_rois)):
            cg = centers_guess[ii]
            ir = img_rois[ii] * (img_rois[ii] >= 0)
            sxs[ii] = np.sqrt(np.sum(ir * (xrois[ii] - cg[2]) ** 2) / np.sum(ir))
            sys[ii] = np.sqrt(np.sum(ir * (yrois[ii] - cg[1]) ** 2) / np.sum(ir))

            if data_is_2d:
                szs[ii] = 1
            else:
                szs[ii] = np.sqrt(np.sum(ir * (zrois[ii] - cg[0]) ** 2) / np.sum(ir))

        # sxs = np.array([np.sqrt(np.sum(ir * (xr - cg[2]) ** 2) / np.sum(ir))
        #                 for ir, xr, cg in zip(img_rois, xrois, centers_guess)])
        # sys = np.array([np.sqrt(np.sum(ir * (yr - cg[1]) ** 2) / np.sum(ir))
        #                 for ir, yr, cg in zip(img_rois, yrois, centers_guess)])

        if data_is_2d:
            cz_guess = np.zeros(len(img_rois))
        else:
            cz_guess = centers_guess[:, 0]

        # get initial parameter guesses
        init_params = np.stack((amps,
                                centers_guess[:, 2], centers_guess[:, 1], cz_guess,
                                0.5 * (sxs + sys), szs, bgs), axis=1)

        if np.any(np.isnan(init_params)):
            raise ValueError("one or more init_params was nan")

        if verbose:
            print("Prepared %d rois and estimated initial parameters in %0.2fs" %
                  (len(rois), time.perf_counter() - tstart))

        # ###################################################
        # localization
        # ###################################################
        if verbose:
            print("starting fitting for %d rois" % centers_guess.shape[0])
        tstart = time.perf_counter()

        fixed_params = [False] * 7

        # if 2D, don't want to fit cz or sz
        if data_is_2d:
            fixed_params[5] = True
            fixed_params[3] = True

        fit_params, fit_states, chi_sqrs, niters, fit_t = fit_gauss_rois(img_rois, (zrois, yrois, xrois),
                                                                         init_params, max_nfit_iterations,
                                                                         estimator="LSE",
                                                                         sf=sf, dc=dxy, angles=(0., 0., 0.),
                                                                         fixed_params=fixed_params,
                                                                         use_gpu=use_gpu_fit, verbose=verbose)

        tend = time.perf_counter()
        if verbose:
            print("Localization took %0.2fs" % (tend - tstart))

        # ###################################################
        # filter fits
        # ###################################################
        to_keep, conditions, condition_names, filter_settings = \
            filter_localizations(fit_params, init_params, (z, y, x), fit_dist_max_err, min_spot_sep,
                                 sigma_bounds, fit_amp_min, dist_boundary_min)

        if verbose:
            print("Identified %d likely candidates" % np.sum(to_keep))

    else:
        fit_params = np.zeros((0, 7))
        init_params = np.zeros((0, 7))
        rois = np.zeros((0, 6))
        to_keep = None
        conditions = None
        condition_names = None
        filter_settings = None
        fit_states = None
        chi_sqrs = None
        niters = None

    return (z, y, x), fit_params, init_params, rois,\
           to_keep, conditions, condition_names, filter_settings,\
           fit_states, chi_sqrs, niters, imgs_filtered


def plot_bead_locations(imgs: np.ndarray, center_lists: list[np.ndarray],
                        title: str = "", color_lists: list[str] = None,
                        color_limits: list[list[float]] = None, legend_labels: list[str] = None,
                        weights: list[np.ndarray] = None, cbar_labels: list[str] = None, coords: list = None,
                        vlims_percentile: tuple[float] = (0.01, 99.99), gamma: float = 1, **kwargs):
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


def autofit_psfs(imgs: np.ndarray, psf_roi_size: list[float], dx: float, dz: float, wavelength: float, ni: float = 1.5,
                 model: str = 'vectorial', sf: int = 1, threshold: float = 100., min_spot_sep: tuple[float] = (0., 0.),
                 filter_sigma_small: tuple[float] = (1, 0.5, 0.5), filter_sigma_large: tuple[float] = (3, 5, 5),
                 sigma_bounds: tuple[tuple[float]] = ((0., 0.), (np.inf, np.inf)), roi_size_loc=(13, 21, 21),
                 fit_amp_thresh: float = 100., dist_boundary_min: tuple[float] = (0., 0.),
                 max_number_iterations: int = 100, fit_dist_max_err: tuple[float] = (np.inf, np.inf),
                 num_localizations_to_plot: int = 5, psf_percentiles: tuple[float] = (5,),
                 plot: bool = True, only_plot_good_fits: bool = True, plot_filtered_image: bool = False,
                 use_gpu_filter: bool = False, gamma: float = 0.5, save_dir: str = None,
                 figsize=(18, 10), **kwargs):
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
    :param sf: sampling factor for oversampling along each dimension
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

    if isinstance(psf_percentiles, (int, float)):
        psf_percentiles = [psf_percentiles]

    saving = False
    if save_dir is not None:
        saving = True

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

    # ###################################
    # do localization
    # ###################################
    z, y, x = get_coords(imgs.shape, (dz, dx, dx))

    coords, fit_params, init_params, rois,\
    to_keep, conditions, condition_names, filter_settings,\
    fit_states, chi_sqrs, niters, imgs_filtered = \
        localize_beads(imgs, dx, dz, threshold, roi_size_loc, filter_sigma_small, filter_sigma_large,
                       min_spot_sep, sigma_bounds, fit_amp_thresh, fit_dist_max_err=fit_dist_max_err,
                       dist_boundary_min=dist_boundary_min, max_nfit_iterations=max_number_iterations,
                       use_gpu_filter=use_gpu_filter)

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
        figh = psf.plot_fit_stats(fit_params[to_keep], figsize=figsize, **kwargs)

        if saving:
            fname = os.path.join(save_dir, "fit_stats.png")
            figh.savefig(fname)
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

            results, _ = psf.fit_psfmodel(psfs_real[ii], dx, dz, wavelength, ni, sf, model=model)
            fit_params_real[ii] = results["fit_params"]

            if plot:
                figh = psf.plot_psfmodel_fit(psfs_real[ii], dx, dz, wavelength, ni, sf, fit_params_real[ii], model=model,
                                         gamma=gamma, figsize=figsize,
                                         label="smallest %d percent" % psf_percentiles[ii])

                if saving:
                    fname = os.path.join(save_dir, "experimental_psf_smallest_%0.2f.png" % psf_percentiles[ii])
                    figh.savefig(fname)
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
            fname = os.path.join(save_dir, "sigma_versus_position.png")
            figh.savefig(fname)
            plt.close(figh)

    if saving:
        data = {"coords": coords, "fit_params": fit_params, "init_params": init_params,
                "rois": rois, "to_keep": to_keep, "conditions": conditions,
                "condition_names": condition_names, "filter_settings": filter_settings,
                "fit_states": fit_states, "chi_sqrs": chi_sqrs, "niterations": niters,
                "psfs_real": psfs_real, "psf_coords": psf_coords, "otfs_real": otfs_real, "otf_coords": otf_coords,
                "psf_percentiles": psf_percentiles, "fit_params_real": fit_params_real}

        fname = os.path.join(save_dir, "localization_results.pkl")
        with open(fname, "wb") as f:
            pickle.dump(data, f)

    return coords, fit_params, init_params, rois, \
           to_keep, conditions, condition_names, filter_settings,\
           fit_states, chi_sqrs, niters, \
           psfs_real, psf_coords, otfs_real, otf_coords, psf_percentiles, fit_params_real
