"""
Code for localizing and fitting spots
"""
import os
import time
import warnings
import numpy as np
import scipy.signal
import joblib
import matplotlib.pyplot as plt
import rois as roi_fns
import fit
import fit_psf as psf

# for filtering on GPU
try:
    import cupy as cp
    import cupyx.scipy.signal
    import cupyx.scipy.ndimage
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

# for fitting on GPU
# custom code found at https://github.com/QI2lab/Gpufit
# this must be compiled from source, and then the python package installed from there
try:
    import pygpufit.gpufit as gf
    GPUFIT_AVAILABLE = True
except ImportError:
    GPUFIT_AVAILABLE = False

def get_coords(sizes, drs):
    """
    Regularly spaced coordinates which can be broadcast to full size
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
    Gaussian filter kernel for arbitrary dimensions

    :param sigmas: (sigma_0, sigma_1, ..., sigma_n)
    :param drs: (dr_1, dr_2, ..., dr_n) pixel sizes along each dimension
    :param sigma_cutoff: single number or list [s0, s1, ..., sn] giving after how many sigmas we cutoff the kernel
    :return kernel:
    """

    ndims = len(drs)

    if isinstance(sigma_cutoff, (int, float)):
        sigma_cutoff = [sigma_cutoff] * ndims


    # compute kernel size
    nks = [2 * int(np.round(sig / dr * sig_cut)) + 1 for sig, dr, sig_cut in zip(sigmas, drs, sigma_cutoff)]

    # get coordinates to evaluate kernel at
    coords = [np.expand_dims(np.arange(nk) * dr, axis=list(range(ii)) + list(range(ii + 1, ndims))) for ii, (nk, dr) in enumerate(zip(nks, drs))]

    coords = [c - np.mean(c) for c in coords]

    kernel = np.exp(sum([-rk ** 2 / 2 / sig ** 2 for rk, sig in zip(coords, sigmas)]))
    kernel = kernel / np.sum(kernel)

    return kernel


def filter_convolve(imgs, kernel, use_gpu=CUPY_AVAILABLE):
    """
    Convolution filter using kernel with GPU support

    # todo: how much memory does convolution require? Much more than I expect...

    :param imgs: images to be convolved
    :param kernel: kernel to be convolved
    :param bool use_gpu:
    :return imgs_filtered:
    """

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


def get_max_filter_footprint(min_sep_allowed, drs):
    """
    Get footprint for maximum filter
    :param min_sep_allowed: (size_0, size_1, ..., size_n)
    :param drs: (dr_0, ..., dr_n
    :return footprint: square matrix, boolean mask
    """

    ns = [int(np.ceil(sz / dr)) for sz, dr in zip(min_sep_allowed, drs)]
    # ensure odd
    ns = [n if np.mod(n, 2) == 0 else n + 1 for n in ns]

    footprint = np.ones(ns, dtype=bool)

    return footprint


def find_peak_candidates(imgs, footprint, threshold, use_gpu_filter=CUPY_AVAILABLE):
    """
    Find peak candidates using maximum filter

    :param imgs: 2D or 3D array
    :param footprint: footprint to use for maximum filter. Array should have same number of dimensions as imgs
    :param threshold: only pixels with values greater than or equal to the threshold will be considered
    :param use_gpu_filter:

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


def filter_nearby_peaks(centers, min_xy_dist, min_z_dist, mode="average", weights=None):
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
        combine = np.logical_and(z_dists < min_z_dist, xy_dists < min_xy_dist)
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
    :param mode: 'radial-symmetry' or 'centroid'

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
    :param mode: 'radial-symmetry' or 'centroid'

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
    Fit single ROI

    :param img_roi:
    :param coords: (z_roi, y_roi, x_roi)
    :param init_params:
    :param fixed_params:
    :param bounds:
    :return:
    """
    z_roi, y_roi, x_roi = coords

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
    results = fit.fit_model(img_roi, model_fn, init_params, bounds=bounds, fixed_params=fixed_params, model_jacobian=jac_fn)

    return results


def fit_gauss_rois(img_rois, coords_rois, init_params, max_number_iterations=100,
                   sf=1, dc=None, angles=None, estimator="LSE", model="gaussian",
                   fixed_params=None, use_gpu=GPUFIT_AVAILABLE):
    """
    Fit rois. Can use either CPU parallelization with joblib or GPU parallelization using gpufit

    :param img_rois: list of image rois
    :param coords_rois: ((z0, y0, x0), (z1, y1, x1), ....)
    :param init_params:
    :param max_number_iterations:
    :param int sf: oversampling factor for simulating pixelation
    :param float dc: pixel size, only used for oversampling
    :param (float, float, float) angles: euler angles describing pixel orientation. Only used for oversampling.
    :param estimator: "LSE" or "MLE"
    :param model: "gaussian" or "gaussian-lorentzian"
    :param bool use_gpu:
    :return:
    """

    if fixed_params is None:
        fixed_params = [False] * 7

    zrois, yrois, xrois = coords_rois

    if not use_gpu:
        if model != "gaussian":
            raise NotImplementedError("only gaussian implemented for non gpu")

        results = joblib.Parallel(n_jobs=-1, verbose=1, timeout=None)(
            joblib.delayed(fit_gauss_roi)(img_rois[ii], (zrois[ii], yrois[ii], xrois[ii]), init_params=init_params[ii],
                                          fixed_params=fixed_params, sf=sf, dc=dc, angles=angles)
            for ii in range(len(img_rois)))

        fit_params = np.asarray([r["fit_params"] for r in results])
        chi_sqrs = np.asarray([r["chi_squared"] for r in results])
        fit_states = np.asarray([r["status"] for r in results])
        niters = np.asarray([r["nfev"] for r in results])
        fit_t = np.zeros(niters.shape)  # todo: what to set this to?

    else:
        if sf != 1:
            raise NotImplementedError("sampling factors other than 1 are not implemented for GPU fitting")
        # todo: if requires more memory than GPU has, split into chunks

        # ensure arrays row vectors
        xrois, yrois, zrois, img_rois = zip(
            *[(xr.ravel()[None, :], yr.ravel()[None, :], zr.ravel()[None, :], ir.ravel()[None, :])
              for xr, yr, zr, ir in zip(xrois, yrois, zrois, img_rois)])

        # setup rois
        roi_sizes = np.array([ir.size for ir in img_rois])
        nmax = roi_sizes.max()

        # pad ROI's to make sure all rois same size
        img_rois = [np.pad(ir, ((0, 0), (0, nmax - ir.size)), mode="constant") for ir in img_rois]

        # build roi data
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
            raise ValueError
        if init_params.ndim != 2 or init_params.shape != (nfits, 7):
            raise ValueError
        if user_info.ndim != 1 or user_info.size != (3 * nfits * n_pts_per_fit + nfits):
            raise ValueError

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
        else:
            raise ValueError("'model' must be 'gaussian' or 'gaussian-lorentzian' but was '%s'" % model)

        params_to_fit = np.array([not fp for fp in fixed_params], dtype=np.int32)
        fit_params, fit_states, chi_sqrs, niters, fit_t = gf.fit(data, None, model_id, init_params,
                                                                 max_number_iterations=max_number_iterations,
                                                                 estimator_id=est_id,
                                                                 parameters_to_fit=params_to_fit,
                                                                 user_info=user_info)

        # correct sigmas in case negative
        fit_params[:, 4] = np.abs(fit_params[:, 4])
        fit_params[:, 5] = np.abs(fit_params[:, 5])

    return fit_params, fit_states, chi_sqrs, niters, fit_t


def plot_gauss_roi(fit_params, roi, imgs, coords, init_params=None, same_color_scale=True, figsize=(16, 8),
                   prefix="", save_dir=None):
    """
    plot results from fit_roi()
    :param fit_params:
    :param roi: [zstart, zend, ystart, yend, xstart, xend]
    :param imgs:
    :param coords: (z, y, x) broadcastable to same size as imgs
    :param figsize:
    :return:
    """
    z, y, x = coords
    # extract useful coordinate info
    dc = x[0, 0, 1] - x[0, 0, 0]
    dz = z[1, 0, 0] - z[0, 0, 0]

    if init_params is not None:
        center_guess = np.array([init_params[3], init_params[2], init_params[1]])

    center_fit = np.array([fit_params[3], fit_params[2], fit_params[1]])

    # get ROI and coordinates
    img_roi = roi_fns.cut_roi(roi, imgs)
    x_roi = x[:, :, roi[4]:roi[5]]
    y_roi = y[:, roi[2]:roi[3], :]
    z_roi = z[roi[0]:roi[1], :, :]

    vmin_roi = np.percentile(img_roi[np.logical_not(np.isnan(img_roi))], 1)
    vmax_roi = np.percentile(img_roi[np.logical_not(np.isnan(img_roi))], 99.9)

    # git fit
    img_fit = psf.gaussian3d_psf(x_roi, y_roi, z_roi, dc, fit_params, sf=1)

    # ################################
    # plot results interpolated on regular grid
    # ################################
    figh_interp = plt.figure(figsize=figsize)
    st_str = "Fit, max projections, interpolated, ROI = [%d, %d, %d, %d, %d, %d]\n" \
             "         A=%3.3f, cx=%3.5f, cy=%3.5f, cz=%3.5f, sxy=%3.5f, sz=%3.5f, bg=%3.3f" % \
             (roi[0], roi[1], roi[2], roi[3], roi[4], roi[5],
              fit_params[0], fit_params[1], fit_params[2], fit_params[3], fit_params[4], fit_params[5], fit_params[6])
    if init_params is not None:
        st_str += "\nguess A=%3.3f, cx=%3.5f, cy=%3.5f, cz=%3.5f, sxy=%3.5f, sz=%3.5f, bg=%3.3f" % \
                  (init_params[0], init_params[1], init_params[2], init_params[3], init_params[4], init_params[5],
                   init_params[6])
    plt.suptitle(st_str)

    grid = plt.GridSpec(2, 4)

    ax = plt.subplot(grid[0, 1])
    extent = [y_roi[0, 0, 0] - 0.5 * dc, y_roi[0, -1, 0] + 0.5 * dc, x_roi[0, 0, 0] - 0.5 * dc, x_roi[0, 0, -1] + 0.5 * dc]
    plt.imshow(np.nanmax(img_roi, axis=0).transpose(), vmin=vmin_roi, vmax=vmax_roi, origin="lower",
               extent=extent, cmap="bone")
    plt.plot(center_fit[1], center_fit[2], 'mx')
    if init_params is not None:
        plt.plot(center_guess[1], center_guess[2], 'gx')
    ax.set_ylim(extent[2:4])
    ax.set_xlim(extent[0:2])
    plt.xlabel("Y (um)")
    plt.ylabel("X (um)")
    plt.title("XY")

    ax = plt.subplot(grid[0, 0])
    extent = [z_roi[0, 0, 0] - 0.5 * dz, z_roi[-1, 0, 0] + 0.5 * dz, x_roi[0, 0, 0] - 0.5 * dc, x_roi[0, 0, -1] + 0.5 * dc]
    plt.imshow(np.nanmax(img_roi, axis=1).transpose(), vmin=vmin_roi, vmax=vmax_roi, origin="lower",
               extent=extent, cmap="bone")
    plt.plot(center_fit[0], center_fit[2], 'mx')
    if init_params is not None:
        plt.plot(center_guess[0], center_guess[2], 'gx')
    ax.set_ylim(extent[2:4])
    ax.set_xlim(extent[0:2])
    plt.xlabel("Z (um)")
    plt.ylabel("X (um)")
    plt.title("XZ")

    ax = plt.subplot(grid[1, 1])
    extent = [y_roi[0, 0, 0] - 0.5 * dc, y_roi[0, -1, 0] + 0.5 * dc, z_roi[0, 0, 0] - 0.5 * dz, z_roi[-1, 0, 0] + 0.5 * dz]
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')

        plt.imshow(np.nanmax(img_roi, axis=2), vmin=vmin_roi, vmax=vmax_roi, origin="lower", extent=extent, cmap="bone")
    plt.plot(center_fit[1], center_fit[0], 'mx')
    if init_params is not None:
        plt.plot(center_guess[1], center_guess[0], 'gx')
    ax.set_ylim(extent[2:4])
    ax.set_xlim(extent[0:2])
    plt.xlabel("Y (um)")
    plt.ylabel("Z (um)")
    plt.title("YZ")

    if same_color_scale:
        vmin_fit = vmin_roi
        vmax_fit = vmax_roi
    else:
        vmin_fit = np.percentile(img_fit, 1)
        vmax_fit = np.percentile(img_fit, 99.9)

    ax = plt.subplot(grid[0, 3])
    extent = [y_roi[0, 0, 0] - 0.5 * dc, y_roi[0, -1, 0] + 0.5 * dc, x_roi[0, 0, 0] - 0.5 * dc, x_roi[0, 0, -1] + 0.5 * dc]
    plt.imshow(np.nanmax(img_fit, axis=0).transpose(), vmin=vmin_fit, vmax=vmax_fit, origin="lower", extent=extent, cmap="bone")
    plt.plot(center_fit[1], center_fit[2], 'mx')
    if init_params is not None:
        plt.plot(center_guess[1], center_guess[2], 'gx')
    ax.set_ylim(extent[2:4])
    ax.set_xlim(extent[0:2])
    plt.xlabel("Y (um)")
    plt.ylabel("X (um)")

    ax = plt.subplot(grid[0, 2])
    extent = [z_roi[0, 0, 0] - 0.5 * dz, z_roi[-1, 0, 0] + 0.5 * dz, x_roi[0, 0, 0] - 0.5 * dc, x_roi[0, 0, -1] + 0.5 * dc]
    plt.imshow(np.nanmax(img_fit, axis=1).transpose(), vmin=vmin_fit, vmax=vmax_fit, origin="lower", extent=extent, cmap="bone")
    plt.plot(center_fit[0], center_fit[2], 'mx')
    if init_params is not None:
        plt.plot(center_guess[0], center_guess[2], 'gx')
    ax.set_ylim(extent[2:4])
    ax.set_xlim(extent[0:2])
    plt.xlabel("Z (um)")
    plt.ylabel("X (um)")

    ax = plt.subplot(grid[1, 3])
    extent = [y_roi[0, 0, 0] - 0.5 * dc, y_roi[0, -1, 0] + 0.5 * dc, z_roi[0, 0, 0] - 0.5 * dz, z_roi[-1, 0, 0] + 0.5 * dz]
    plt.imshow(np.nanmax(img_fit, axis=2), vmin=vmin_fit, vmax=vmax_fit, origin="lower", extent=extent, cmap="bone")
    plt.plot(center_fit[1], center_fit[0], 'mx')
    if init_params is not None:
        plt.plot(center_guess[1], center_guess[0], 'gx')
    ax.set_ylim(extent[2:4])
    ax.set_xlim(extent[0:2])
    plt.xlabel("Y (um)")
    plt.ylabel("Z (um)")

    # plt.show(figh_interp)
    if save_dir is not None:
        figh_interp.savefig(os.path.join(save_dir, "%s.png" % prefix))
        plt.close(figh_interp)

    return figh_interp


def filter_localizations(fit_params, init_params, coords, fit_dist_max_err, min_spot_sep,
                         sigma_bounds, amp_min=0, dist_boundary_min=(0, 0)):
    """
    :param fit_params:
    :param init_params:
    :param coords: (z, y, x)
    :param fit_dist_max_err = (dz_max, dxy_max)
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
    centers_guess = np.concatenate((init_params[:, 3][:, None], init_params[:, 2][:, None], init_params[:, 1][:, None]), axis=1)
    centers_fit = np.concatenate((fit_params[:, 3][:, None], fit_params[:, 2][:, None], fit_params[:, 1][:, None]), axis=1)

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
    conditions = np.stack((in_bounds, center_close_to_guess_xy, center_close_to_guess_z,
                            fit_params[:, 4] <= sxy_max, fit_params[:, 4] >= sxy_min,
                            fit_params[:, 5] <= sz_max, fit_params[:, 5] >= sz_min,
                            fit_params[:, 0] >= amp_min), axis=1)


    condition_names = ["in_bounds", "center_close_to_guess_xy", "center_close_to_guess_z",
                       "xy_size_small_enough", "xy_size_big_enough", "z_size_small_enough",
                       "z_size_big_enough", "amp_ok"]

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

#
def localize_beads(imgs, dx, dz, threshold, roi_size, filter_sigma_small, filter_sigma_large,
                   min_spot_sep, sigma_bounds, fit_amp_min, fit_dist_max_err=(np.inf, np.inf), dist_boundary_min=(0, 0),
                   use_gpu_fit=GPUFIT_AVAILABLE, use_gpu_filter=CUPY_AVAILABLE):
    """

    @param imgs:
    @param dx:
    @param dz:
    @param threshold:
    @param roi_size:
    @param filter_sigma_small: (sz, sy, sx)
    @param filter_sigma_large: ()
    @param min_spot_sep: (dz, dxy)
    @param sigma_bounds: ((sz_min, sxy_min), (sz_max, sxy_max))
    @param fit_amp_min:
    @param fit_dist_max_err:
    @param dist_boundary_min:
    @param use_gpu_fit:
    @param use_gpu_filter:
    @return (z, y, x), fit_params, init_params, rois, to_keep, conditions, condition_names, filter_settings:
    """
    # todo: need to ensure can also do 2D ... see e.g. 2021_07_09_localize_zhang_data.py

    if imgs.ndim == 2:
        imgs = np.expand_dims(imgs, axis=0)

    if imgs.shape[0] == 1:
        data_is_2d = True
    else:
        data_is_2d = False

    z, y, x = get_coords(imgs.shape, (dz, dx, dx))
    dz_min, dxy_min = min_spot_sep
    roi_size_pix = roi_fns.get_roi_size(roi_size, dx, dz, ensure_odd=True)

    # ###################################
    # filter images
    # ###################################
    tstart = time.perf_counter()

    ks = get_filter_kernel(filter_sigma_small, (dz, dx, dx))
    kl = get_filter_kernel(filter_sigma_large, (dz, dx, dx))
    imgs_hp = filter_convolve(imgs, ks, use_gpu=use_gpu_filter)
    imgs_lp = filter_convolve(imgs, kl, use_gpu=use_gpu_filter)
    imgs_filtered = imgs_hp - imgs_lp

    print("filtered image in %0.2fs" % (time.perf_counter() - tstart))

    # ###################################
    # identify candidate beads
    # ###################################
    tstart = time.perf_counter()

    footprint = get_max_filter_footprint((dz_min, dxy_min, dxy_min), (dz, dx, dx))
    centers_guess_inds, amps = find_peak_candidates(imgs_filtered, footprint, threshold)

    # real coordinates
    centers_guess = np.stack((z[centers_guess_inds[:, 0], 0, 0],
                              y[0, centers_guess_inds[:, 1], 0],
                              x[0, 0, centers_guess_inds[:, 2]]), axis=1)

    print("identified %d candidates in %0.2fs" % (len(centers_guess_inds), time.perf_counter() - tstart))

    # ###################################
    # identify candidate beads
    # ###################################
    if len(centers_guess_inds) == 0:
        return None
    else:
        # ###################################################
        # average multiple points too close together. Necessary bc if naive threshold, may identify several points
        # from same spot. Particularly important if spots have very different brightness levels.
        # ###################################################
        tstart = time.perf_counter()

        inds = np.ravel_multi_index(centers_guess_inds.transpose(), imgs_filtered.shape)
        weights = imgs_filtered.ravel()[inds]
        centers_guess, inds_comb = filter_nearby_peaks(centers_guess, dxy_min, dz_min, weights=weights, mode="average")

        amps = amps[inds_comb]
        print("Found %d points separated by dxy > %0.5g and dz > %0.5g in %0.1fs" %
              (len(centers_guess), dxy_min, dz_min, time.perf_counter() - tstart))

        # ###################################################
        # prepare ROIs
        # ###################################################
        tstart = time.perf_counter()

        rois, img_rois, coords = zip(*[get_roi(c, imgs, (z, y, x), roi_size_pix) for c in centers_guess])
        zrois, yrois, xrois = zip(*coords)
        rois = np.asarray(rois)
        nsizes = (rois[:, 1] - rois[:, 0]) * (rois[:, 3] - rois[:, 2]) * (rois[:, 5] - rois[:, 4])
        nfits = len(rois)

        # extract guess values
        bgs = np.array([np.mean(r) for r in img_rois])
        sxs = np.array([np.sqrt(np.sum(ir * (xr - cg[2]) ** 2) / np.sum(ir)) for ir, xr, cg in
                        zip(img_rois, xrois, centers_guess)])
        sys = np.array([np.sqrt(np.sum(ir * (yr - cg[1]) ** 2) / np.sum(ir)) for ir, yr, cg in
                        zip(img_rois, yrois, centers_guess)])
        sxys = 0.5 * (sxs + sys)

        if data_is_2d:
            cz_guess = np.zeros(len(img_rois))
            szs = np.ones(len(img_rois))
        else:
            cz_guess = centers_guess[:, 0]
            szs = np.array([np.sqrt(np.sum(ir * (zr - cg[0]) ** 2) / np.sum(ir)) for ir, zr, cg in
                            zip(img_rois, zrois, centers_guess)])

        # get initial parameter guesses
        init_params = np.stack((amps,
                                centers_guess[:, 2], centers_guess[:, 1], cz_guess,
                                sxys, szs, bgs), axis=1)

        print("Prepared %d rois and estimated initial parameters in %0.2fs" % (len(rois), time.perf_counter() - tstart))

        # ###################################################
        # localization
        # ###################################################
        print("starting fitting for %d rois" % centers_guess.shape[0])
        tstart = time.perf_counter()

        fixed_params = [False] * 7

        # if 2D, don't want to fit cz or sz
        if data_is_2d:
            fixed_params[5] = True
            fixed_params[3] = True

        fit_params, fit_states, chi_sqrs, niters, fit_t = fit_gauss_rois(img_rois, (zrois, yrois, xrois),
                                                                         init_params, estimator="LSE",
                                                                         sf=1, dc=dx, angles=(0., 0., 0.),
                                                                         fixed_params=fixed_params,
                                                                         use_gpu=use_gpu_fit)

        tend = time.perf_counter()
        print("Localization took %0.2fs" % (tend - tstart))


        # ###################################################
        # filter fits
        # ###################################################
        to_keep, conditions, condition_names, filter_settings = \
            filter_localizations(fit_params, init_params, (z, y, x), fit_dist_max_err, min_spot_sep,
                                 sigma_bounds, fit_amp_min, dist_boundary_min)

        print("Identified %d likely candidates" % np.sum(to_keep))

        return (z, y, x), fit_params, init_params, rois, to_keep, conditions, condition_names, filter_settings
