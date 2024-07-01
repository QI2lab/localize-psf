"""
Code for working with affine transformations in 2D and 3D

Determine affine transformation mapping object space to image space.
The affine transformation (in homogeneous coordinates) is represented by a matrix, T
[[c0_i], [c0_i], [1]] = T * [[c0_o], [c1_0o], [1]]
Given a function defined on object space, g(xo, yo), we can define a corresponding function on image space
gi(c0_i, c1_i) = g(T^{-1} [[c0_i], [c1_i], [1]])
"""
from typing import Optional, Union
from collections.abc import Sequence
from warnings import warn
from joblib import Parallel, delayed
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from localize_psf.fit import fit_least_squares

try:
    import cupy as cp
    import cupyx.scipy.RegularGridInterpolator as RegularGridInterpolatorGPU
except ImportError:
    cp = None
    RegularGridInterpolatorGPU = None

if cp:
    array = Union[np.ndarray, cp.ndarray]
else:
    array = np.ndarray


def xform2params(affine_mat: np.ndarray) -> np.ndarray:
    """
    Parametrize 2D affine transformation in terms of rotation angles, magnifications, and offsets.
    T = [[m0 * cos(t0), -m1 * sin(t1), v0],
         [m0 * sin(t0),  m1 * cos(t1), v1],
         [   0        ,    0         , 1]]

    :param affine_mat:
    :return params: [m0, t0, v0 ,m1, t1, v1]
    """
    if affine_mat.shape != (3, 3):
        raise ValueError("xform2params only works with 2D affine transformations (i.e. 3x3 matrices)")

    # get offsets
    v0 = affine_mat[0, -1]
    v1 = affine_mat[1, -1]

    # get rotation and scale for x-axis
    t0 = np.angle(affine_mat[0, 0] + 1j * affine_mat[1, 0])
    m0 = np.nanmean([affine_mat[0, 0] / np.cos(t0), affine_mat[1, 0] / np.sin(t0)])

    # get rotation and scale for y-axis
    t1 = np.angle(affine_mat[1, 1] - 1j * affine_mat[0, 1])
    m1 = np.nanmean([affine_mat[1, 1] / np.cos(t1), -affine_mat[0, 1] / np.sin(t1)])

    return np.array([m0, t0, v0, m1, t1, v1])


def params2xform(params: Sequence[float]) -> np.ndarray:
    """
    Construct a 2D affine transformation from parameters. Inverse function for xform2params()
    T = [[m0 * cos(t0), -m1 * sin(t1), v0],
         [m0 * sin(t0),  m1 * cos(t1), v1],
         [   0        ,    0         , 1]]

    :param params: [M0, t0, v0 ,M1, t1, v1]
    :return affine_xform:
    """
    ma, ta, va, mb, tb, vb = params
    affine_xform = np.array([[ma * np.cos(ta), -mb * np.sin(tb), va],
                             [ma * np.sin(ta),  mb * np.cos(tb), vb],
                             [0              ,  0              ,  1]])

    return affine_xform


def rotation2xform(angle: float,
                   center: Sequence[float]) -> np.ndarray:
    """
    Get 2D affine transformation corresponding to a rotation by an angle about a given center

    :param angle: angle in radians
    :param center: (cx, cy)
    :return xform:
    """
    # think of this xform as Rot Matrix * [[X - cx], [Y - cy]] + [[cx], [cy]]
    cx, cy = center
    xform = params2xform([1, angle, cx, 1, angle, cy])
    extra_offset = -xform[:2, :2].dot(np.array([[cy], [cx]]))
    xform[:-1, -1] += extra_offset.ravel()

    return xform


def xform_mat(mat_obj: array,
              xform: array,
              img_coords: Sequence[array],
              mode: str = 'nearest') -> array:
    """
    Given a matrix defined on object space coordinates, M[c0_obj, c1_obj], calculate corresponding matrix at image
    space coordinates. This is given by
    M'[c0_img, c1_img] = M[ T^{-1} * [c0_img, c1_img] ]
    where T is the affine transformation from object space to image space.
    Object coordinates are assumed to be on a regular pixel grid [0, ..., n0-1], ...
    This function is a wrapper for scipy.optimize.RegularGridInterpolator

    :param mat_obj: matrix in object space
    :param xform: affine transformation which takes object space coordinates as input, [yi, xi] = T * [yo, xo].
    :param img_coords: (c0_i, c1_i, ..., cn_i) list of coordinate arrays where the image-space matrix is to be
      evaluated. All coordinate arrays must be the same shape. i.e., c0_i.shape = c1_i.shape.
    :param mode: passed through as the `method` argument to RegularGridInterpolator. Options include "nearest",
      "linear", and "cubic".
    :return mat_img: matrix in image space, M'[c0_i, c1_i]
    """
    if cp and isinstance(mat_obj, cp.ndarray):
        xp = cp
        Interp = RegularGridInterpolatorGPU
    else:
        xp = np
        Interp = RegularGridInterpolator

    # image space coordinates
    output_shape = img_coords[0].shape
    coords_img = xp.stack([xp.asarray(ic).ravel() for ic in img_coords], axis=1)

    # corresponding object space coordinates
    xform_inv = xp.linalg.inv(xp.asarray(xform))
    coords_obj = [xp.reshape(c, output_shape)
                           for c in xform_points(coords_img, xform_inv).transpose()]
    co = np.stack(coords_obj, axis=-1)

    # object space range
    coords_obj_range = [xp.arange(s) for s in mat_obj.shape]

    # get matrix in image space
    mat_img = Interp(coords_obj_range,
                     mat_obj,
                     bounds_error=False,
                     fill_value=np.nan,
                     method=mode)(co)

    return mat_img


def xform_fn(fn: callable,
             xform: np.ndarray) -> callable:
    """
    Given a function f(c0_o, c1_o, ...) on object space coordinates and an affine transformation,
    (c0_i, c1_i, ...) = T * (c0_o, c1_o, ...), create the function
    f_i(c0_i, c1_i, ...) = f( T^{-1}(c0_o, c1_o, ...) )
    Evaluate f_i at ci by calling xform_fn(fn, xform)(*ci)

    :param fn: function on object space coordines, f(c_o)
    :param xform: affine transformation matrix which takes points in object space to points in image space
    :return fn_out: function of image space coordinates (c0_i, c1_i, ...)
    """

    xform_inv = np.linalg.inv(xform)

    def fn_out(*out_coords):
        out_coords = [np.array(oc) for oc in out_coords]
        output_shape = out_coords[0].shape

        ci = np.stack([ic.ravel() for ic in out_coords], axis=1)
        co = xform_points(ci, xform_inv).transpose()
        co = [np.reshape(c, output_shape) for c in co]

        return fn(*co)

    return fn_out


def xform_points(coords: array,
                 xform: array) -> array:
    """
    Transform coordinates of arbitrary dimension under the action of an affine transformation

    :param coords: array of shape n0 x n1 x ... nm x ndim
    :param xform: affine transform matrix of shape (ndim + 1) x (ndim + 1)
    :return coords_out: n0 x n1 x ... nm x ndim
    """
    if cp and isinstance(coords, cp.ndarray):
        xp = cp
    else:
        xp = np

    ndims = coords.shape[-1]
    coords_in = xp.stack([coords[..., ii].ravel() for ii in range(ndims)] +
                         [xp.ones(coords[..., 0].size)],
                         axis=0)

    # trim off homogeneous coordinate row and reshape
    coords_out = xp.asarray(xform).dot(coords_in)[:-1].transpose().reshape(coords.shape)

    return coords_out


def xform_sinusoid_params(f0_obj: Union[float, np.ndarray],
                          f1_obj: Union[float, np.ndarray],
                          phi_obj: Union[float, np.ndarray],
                          affine_mat: np.ndarray) -> (Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray],):
    """
    Given a sinusoid function of object space,
    cos[2pi f0 * c0_obj + 2pi f1 * c1_obj + phi_o],
    and an affine transformation mapping object space to image space, [c0_img, c1_img] = A * [c0_obj, c1_obj]
    find the frequency and phase parameters for the corresponding function on image space,
    cos[2pi f0_img * c0_img + 2pi f1_img * y1_img + phi_i]
    f0_obj, f1_obj, and phi_obj should be broadcastable to the same shape

    :param f0_obj: 0th coordinate component of frequency (in units 1/cycles) in object space
    :param f1_obj: 1st coordinate component of frequency (in units of 1/cycles) in object space
    :param phi_obj: phase in object space
    :param affine_mat: affine transformation homogeneous coordinate matrix transforming
     points in object space to image space
    :return f0_img, f1_img, phi_mig: frequency components and phase in image space
    """
    # todo: should twe accept an array frq_obj of shape n0 x n1 x ... x nk x ndim
    # todo: and support arbitary dimensional sinusoids?
    affine_inv = np.linalg.inv(affine_mat)
    f0_img = f0_obj * affine_inv[0, 0] + f1_obj * affine_inv[1, 0]
    f1_img = f0_obj * affine_inv[0, 1] + f1_obj * affine_inv[1, 1]
    phi_img = np.mod(phi_obj +
                     2 * np.pi * f0_obj * affine_inv[0, 2] +
                     2 * np.pi * f1_obj * affine_inv[1, 2],
                     2 * np.pi)

    return f0_img, f1_img, phi_img


def fit_xform_points(from_pts: np.ndarray,
                     to_pts: np.ndarray,
                     translate_only: bool = False) -> (np.ndarray, np.ndarray):
    """
    Solve for an affine transformation of arbitrary dimensions, where the transformation
    T = [[A, b], [0, ..., 0, 1]],
    to_pts = A * from_pts + b, or
    to_pts = T * [[from_pts], [1]]
    i.e. A gives the rotational part of the affine transformation, and b gives the offset.

    :param from_pts: npts x ndims array, where each column gives coordinates for a different dimension, e.g. first
     column is x, second is y,...
    :param to_pts: npts x ndims array. The desired affine transformation acts on from_pts to produce to_pts
    :param translate_only: if True do not allow scaling/shearing in affine transformation, only allow translation
    :return affine_mat, vars: affine_mat is an (ndim + 1) x (ndim + 1) matrix which act on homogeneous coordinates.
     To transform coordinates using this affine transformation use xform_points(). vars are the estimated
     variances of the affine transformation matrix entries
    """
    # todo: could add ability to fix certain parameters, but tricky to do this for 2D/3D dims in same code

    # interpret as ndims x npts, i.e. rows are coord_0, coord_1, ..., coord_(n-1)
    from_pts = np.asarray(from_pts).transpose()
    to_pts = np.asarray(to_pts).transpose()
    ndim, npts = to_pts.shape

    if (npts < 3 * ndim and not translate_only) or (npts < ndim and translate_only):
        raise ValueError("Not enough points provided to determine affine transformation."
                         " Need 3*ndim points if translate_only=False, else ndim points")

    # augment points with a row of ones
    # rows are coord_0, coord_1, ..., coord_(n-1), 1
    from_pts_aug = np.concatenate((from_pts, np.ones((1, npts))), axis=0)

    # Solve for affine matrix as a linear least squares problem or actually two separate problems:
    # [coord_0_from, coord_1_from, 1] * M1 = coord_0_to; M1 = [[A], [B], [C]]
    # [coord_0_from, coord_1_from, 1] * M2 = coord_1_to; M2 = [[D], [E], [F]]
    # and this naturally generalizes to any number of dimensions
    affine_mat = np.zeros((ndim + 1, ndim + 1))
    affine_mat[-1, -1] = 1

    vars = np.zeros(affine_mat.shape)
    for ii in range(ndim):
        if not translate_only:
            params_temp, residuals, rank, svals = np.linalg.lstsq(from_pts_aug.transpose(),
                                                                  to_pts[ii],
                                                                  rcond=None)
            affine_mat[ii] = params_temp

            # variances of fit parameters
            xt_x_inv = np.linalg.inv(from_pts_aug.dot(from_pts_aug.transpose()))
            var_sample = residuals / (npts - (ndim + 1))
            try:
                vars[ii] = np.diag(xt_x_inv) * var_sample
            except ValueError:
                vars[ii] = np.nan
        else:
            params_temp, residuals, rank, svals = \
                np.linalg.lstsq(np.expand_dims(from_pts_aug[-1], axis=1),
                                to_pts[ii] - from_pts[ii],
                                rcond=None)
            affine_mat[ii, -1] = params_temp
            affine_mat[ii, ii] = 1

            xt_x_inv = 1 / np.sum(from_pts_aug[-1]**2)
            var_sample = residuals / (npts - (ndim + 1))
            vars[ii, -1] = xt_x_inv * var_sample

    return affine_mat, vars


def fit_xform_points_ransac(from_pts: np.ndarray,
                            to_pts: np.ndarray,
                            dist_err_max: float = 0.3,
                            niterations: int = 1000,
                            njobs: int = 1,
                            n_inliers_stop: int = np.inf,
                            translate_only: bool = False) -> (np.ndarray, ):
    """
    Determine affine transformation using the random sample consensus (RANSAC) algorithm. This approach
    is more robust to false correspondences between some of the from_pts and to_pts.

    The entries in the arrays from_pts and to_pts are assumed to be potential correspondences. If one point in
    from_pts has two potential identifications in to_pts, then include this point twice in the array from_pts
    corresponding to two different points in to_pts.

    :param from_pts: array of shape n0 x ... x nm x ndims
    :param to_pts: array of shape n0 x ... x nm x ndims of points proposed to correspond to from_pts
    :param dist_err_max: maximum distance error for points to be considered "inliers"
    :param niterations: number of iterations of RANSAC
    :param njobs: passed through to joblib to set number of cores to use
    :param n_inliers_stop: stop iterating when find at least this many inliers
    :param translate_only: only fit the translation parameters in the affine transformation
    :return xform_best, inliers_best, err_best, vars_best:
    """

    if njobs > 1:
        niters_each = int(np.ceil(niterations / njobs))
        results = Parallel(n_jobs=-1, verbose=0, timeout=None)(
                  delayed(fit_xform_points_ransac)(from_pts,
                                                   to_pts,
                                                   dist_err_max=dist_err_max,
                                                   niterations=niters_each,
                                                   njobs=1,
                                                   n_inliers_stop=n_inliers_stop,
                                                   translate_only=translate_only)
                  for ii in range(njobs))

        xforms, inliers, errs, vars = zip(*results)

        # find best result from the
        ind_best = np.argmax([np.sum(ins) for ins in inliers])
        xform_best = xforms[ind_best]
        inliers_best = inliers[ind_best]
        error_best = errs[ind_best]
        vars_best = vars[ind_best]

    else:
        # npts, ndims = from_pts.shape
        ndims = from_pts.shape[-1]
        pts_shape = from_pts.shape[:-1]
        npts = np.prod(pts_shape)

        if translate_only:
            ninit_pts = ndims
        else:
            ninit_pts = 3 * ndims

        error_best = np.inf
        xform_best = None
        vars_best = None
        # inliers_best = np.zeros(npts, dtype=bool)
        inliers_best = np.zeros(pts_shape, dtype=bool)
        for ii in range(niterations):
            # get initial proposed points and determine transformation
            is_inlier_prop = np.zeros(pts_shape, dtype=bool)
            is_inlier_prop.ravel()[np.sort(np.random.choice(np.arange(npts),
                                                            size=ninit_pts,
                                                            replace=False))] = True
            not_inlier_prop = np.logical_not(is_inlier_prop)

            xform_prop, _ = fit_xform_points(from_pts[is_inlier_prop],
                                             to_pts[is_inlier_prop],
                                             translate_only=translate_only)

            # get distance errors of other points to determine if inliers or outliers
            dist_errs = np.linalg.norm(to_pts[not_inlier_prop] -
                                       xform_points(from_pts[not_inlier_prop], xform_prop),
                                       axis=-1)

            is_inlier_prop[not_inlier_prop] = dist_errs < dist_err_max

            # refit using all inliers and compute final error and inliers
            xform_prop, vars_prop = fit_xform_points(from_pts[is_inlier_prop],
                                                     to_pts[is_inlier_prop],
                                                     translate_only=translate_only)

            dist_errs = np.linalg.norm(to_pts - xform_points(from_pts, xform_prop), axis=1)

            is_inlier_prop = dist_errs < dist_err_max
            if not np.any(is_inlier_prop):
                continue

            model_err = np.mean(dist_errs[is_inlier_prop])

            # if model_err < error_best:
            if np.sum(inliers_best) < np.sum(is_inlier_prop):
                error_best = model_err
                xform_best = np.array(xform_prop, copy=True)
                vars_best = vars_prop
                inliers_best = np.array(is_inlier_prop, copy=True)

            if np.sum(inliers_best) >= n_inliers_stop:
                break

    return xform_best, inliers_best, error_best, vars_best


def fit_xform_img(mat_obj: np.ndarray,
                  mat_img: np.ndarray,
                  init_params: Optional[Sequence[float]] = None,
                  fixed_params: Optional[Sequence[bool]] = None,
                  bounds: Optional[Sequence[Sequence[float]]] = None) -> (dict, np.ndarray):
    """
    Fit affine transformation by comparing image with transformed image

    :param mat_obj: array of size ny_o x nx_o in object space
    :param mat_img: array of size ny_i x nx_i in image space
    :param init_params: let t by the affine transformation matrix which acts on object space coordinates to prouce
     image space coordinates. Then the initial parameters are
     [amplitude, background, t[0, 0], t[0, 1], t[0, 2], t[1, 0], t[1, 1], t[1, 2]]
    :param fixed_params:
    :param bounds:
    :return results, xform:
    """

    # ensure we are working with float arrays
    mat_obj = np.array(mat_obj, dtype=float, copy=True)
    mat_img = np.array(mat_img, dtype=float, copy=True)

    if init_params is None:
        init_params = np.array([1, 0, 1, 0, 0, 0, 1, 0])

    def p2xform(p): return np.array([[p[2], p[3], p[4]],
                                     [p[5], p[6], p[7]],
                                     [0   , 0   , 1]])

    def err_fn(p):
        img_coords = np.meshgrid(np.arange(mat_img.shape[1]), np.arange(mat_img.shape[0]), indexing="ij")
        diff = mat_img.ravel() - (p[0] * xform_mat(mat_obj, p2xform(p), img_coords, mode='linear').ravel() + p[1])
        diff[np.isnan(diff)] = 0
        return diff

    results = fit_least_squares(err_fn,
                                init_params,
                                fixed_params=fixed_params,
                                bounds=bounds)
    xform = p2xform(results["fit_params"])

    return results, xform
