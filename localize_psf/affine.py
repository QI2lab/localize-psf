"""
Code for working with affine transformations in 2D and 3D

Determine affine transformation mapping object space to image space.
The affine transformation (in homogeneous coordinates) is represented by a matrix,
[[xi], [yi], [1]] = T * [[xo], [yo], [1]]

Given a function defined on object space, g(xo, yo), we can define a corresponding function on image space
gi(xi, yi) = g(T^{-1} [[xi], [yi], [1]])
"""
from typing import Optional, Union
from collections.abc import Sequence
from joblib import Parallel, delayed
import numpy as np
from scipy.interpolate import RectBivariateSpline
from localize_psf.fit import fit_least_squares

try:
    import cupy as cp
except ImportError:
    cp = None

if cp:
    array = Union[np.ndarray, cp.ndarray]
else:
    array = np.ndarray


def xform2params(affine_mat: np.ndarray) -> np.ndarray:
    """
    Parametrize 2D affine transformation in terms of rotation angles, magnifications, and offsets.
    T = [[Mx * cos(tx), -My * sin(ty), vx],
         [Mx * sin(tx),  My * cos(ty), vy],
         [   0        ,    0        , 1]]

    Both theta_x and theta_y are measured CCW from the x-axis

    :param affine_mat:
    :return [mx, theta_x, vx, my, theta_y, vy]:
    """
    if affine_mat.shape != (3, 3):
        raise ValueError("xform2params only works with 2D affine transformations (i.e. 3x3 matrices)")

    # get offsets
    vx = affine_mat[0, -1]
    vy = affine_mat[1, -1]

    # get rotation and scale for x-axis
    theta_x = np.angle(affine_mat[0, 0] + 1j * affine_mat[1, 0])
    mx = np.nanmean([affine_mat[0, 0] / np.cos(theta_x), affine_mat[1, 0] / np.sin(theta_x)])

    # get rotation and scale for y-axis
    theta_y = np.angle(affine_mat[1, 1] - 1j * affine_mat[0, 1])
    my = np.nanmean([affine_mat[1, 1] / np.cos(theta_y), -affine_mat[0, 1] / np.sin(theta_y)])

    return np.array([mx, theta_x, vx, my, theta_y, vy])


def params2xform(params: Sequence[float]) -> np.ndarray:
    """
    Construct a 2D affine transformation from parameters. Inverse function for xform2params()

    T = Ma * cos(ta), -Mb * sin(tb), va
        Ma * sin(ta),  Mb * cos(tb), vb
           0        ,    0        , 1

    :param params: [Ma, ta, va ,Mb, tb, vb]
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
    Given a matrix defined on object space coordinates, M[yo, xo], calculate corresponding matrix at image
    space coordinates. This is given by (roughly speaking)
    M'[yi, xi] = M[ T^{-1} * [xi, yi] ]
    Object coordinates are assumed to be [0, ..., nx-1] and [0, ..., ny-1]

    :param mat_obj: matrix in object space
    :param xform: affine transformation which takes object space coordinates as input, [yi, xi] = T * [xo, yo]
    :param img_coords: (c1, c0) list of coordinate arrays where the image-space matrix is to be evaluated. All
      coordinate arrays must be the same shape. i.e., xi.shape = yi.shape.
    :param str mode: 'nearest' or 'interp'. 'interp' will produce better results if e.g. looking at phase content after
      affine transformation.
    :return mat_img: matrix in image space, M'[yi, xi]
    """
    if cp and isinstance(mat_obj, cp.ndarray):
        xp = cp
        if mode == "interp":
            raise NotImplementedError("mode 'interp' is not implemented for CuPy arrays")
    else:
        xp = np

    if mat_obj.ndim != 2:
        raise ValueError("img_obj must be a 2D array")

    # image space coordinates
    output_shape = img_coords[0].shape
    coords_img = xp.stack([xp.asarray(ic).ravel() for ic in img_coords], axis=1)

    # get corresponding object space coordinates
    xform_inv = xp.linalg.inv(xp.asarray(xform))

    coords_obj_from_img = [xp.reshape(c, output_shape)
                           for c in xform_points(coords_img, xform_inv).transpose()]

    # only use points with coords in image
    coords_obj_bounds = [xp.arange(mat_obj.shape[1]),
                         xp.arange(mat_obj.shape[0])]

    # since CuPy logical_and() does not support reduce
    to_use = xp.ones(coords_obj_from_img[0].shape, dtype=bool)
    for ii in range(mat_obj.ndim):
        to_use[coords_obj_from_img[ii] < coords_obj_bounds[ii].min()] = False
        to_use[coords_obj_from_img[ii] > coords_obj_bounds[ii].max()] = False

    # get matrix in image space
    if mode == 'nearest':
        # find the closest point in image to each output point
        inds = [tuple(xp.array(xp.round(oc[to_use]), dtype=int))
                for oc in coords_obj_from_img]
        inds.reverse()

        # evaluate matrix
        mat_img = xp.zeros(output_shape) * xp.nan
        mat_img[to_use] = mat_obj[tuple(inds)]

    elif mode == 'interp':
        mat_img = RectBivariateSpline(*coords_obj_bounds, mat_obj.transpose()).ev(*coords_obj_from_img)
        mat_img[xp.logical_not(to_use)] = np.nan
    else:
        raise ValueError(f"'mode' must be 'nearest' or 'interp' but was '{mode:s}'")

    return mat_img


def xform_fn(fn: callable,
             xform: np.ndarray) -> callable:
    """
    Given a function f(xo, yo) and an affine transformation, (xi, yi, ...) = T * (xo, yo, ...), create the function
    f'(xi, yi, ...) = f( T^{-1}(xo, yo, ...) )

    Given a set of output coordinates, ci, evaluate f' at ci by calling xform_fn(fn, xform)(*ci)

    :param fn: function on object space, fn(xo, yo)
    :param xform: affine transformation matrix which takes points in object space to points in image space,
     (xi, yi, ...) = T * (xo, yo, ...)
    :return fn_out: function of coordinates (xi, yi, ...)
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


def xform_sinusoid_params(fx_obj: Union[float, np.ndarray],
                          fy_obj: Union[float, np.ndarray],
                          phi_obj: Union[float, np.ndarray],
                          affine_mat: np.ndarray) -> (Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray],):
    """
    Given a sinusoid function of object space,
    cos[2pi f_x * xo + 2pi f_y * yo + phi_o],
    and an affine transformation mapping object space to image space, [xi, yi] = A * [xo, yo]
    find the frequency and phase parameters for the corresponding function on image space,
    cos[2pi f_xi * xi + 2pi f_yi * yi + phi_i]

    :param fx_obj: x-component of frequency (in units 1/cycles) in object space
    :param fy_obj: y-component of frequency (in units of 1/cycles) in object space
    :param phi_obj: phase in object space
    :param affine_mat: affine transformation homogeneous coordinate matrix transforming
     points in object space to image space
    :return fx_img, fy_img, phi_mig: frequency components and phase in image space
    """
    affine_inv = np.linalg.inv(affine_mat)
    fx_img = fx_obj * affine_inv[0, 0] + fy_obj * affine_inv[1, 0]
    fy_img = fx_obj * affine_inv[0, 1] + fy_obj * affine_inv[1, 1]
    phi_img = np.mod(phi_obj +
                     2 * np.pi * fx_obj * affine_inv[0, 2] +
                     2 * np.pi * fy_obj * affine_inv[1, 2],
                     2 * np.pi)

    return fx_img, fy_img, phi_img


def xform_sinusoid_params_roi(fx: float,
                              fy: float,
                              phase: float,
                              affine_mat: np.ndarray,
                              object_size: Optional[Sequence[int]] = None,
                              img_roi: Optional[Sequence[int]] = None,
                              input_origin_fft: bool = True,
                              output_origin_fft: bool = True) -> (float, float, float):
    """
    Transform sinusoid parameter from object space to a region of interest in image space.

    This is an unfortunately complicated function because we have five coordinate systems to worry about
    o: object space coordinates with origin at the corner of the DMD pattern
    o': object space coordinates assumed by fft functions
    i: image space coordinates, with origin at corner of the camera
    r: roi coordinates with origin at the edge of the roi
    r': roi coordinates, with origin near the center of the roi (coordinates for fft)
    The frequencies don't care about the coordinate origin, but the phase does

    :param fx: x-component of frequency in object space
    :param fy: y-component of frequency in object space
    :param phase: phase of pattern in object space coordinates system o or o'.
    :param object_size: [sy, sx], size of object space, required to define origin of o'
    :param img_roi: [ystart, yend, xstart, xend], region of interest in image space. Note: this region does
     not include the pixels at yend and xend! In coordinates with integer values the pixel centers, it is the area
     [ystart - 0.5*dy, yend-0.5*dy] x [xstart -0.5*dx, xend - 0.5*dx]
    :param affine_mat: affine transformation matrix, which takes points from o -> i
    :param input_origin_fft: True if phase is provided in coordinate system o', or "edge" if provided in
     coordinate system o
    :param output_origin_fft: True if output phase should be in coordinate system r' or "edge" if in
     coordinate system r
    :return fx_xform, fy_xform, phi_xform: frequency components and phase in coordinate system r or r' depending
     on the value of output_origin_fft
    """
    # todo: think it would be better to construct these affine transformations as needed in code elsewhere,
    # todo: and simply call xform_sinusoid_params()
    # todo: remove this function once that is completed

    if input_origin_fft:
        if object_size is None:
            raise ValueError("if input_origin_fft = True, then object size must be provided")
        xform_input2edge = params2xform([1, 0, (object_size[1] // 2),
                                         1, 0, (object_size[0] // 2)])
    else:
        xform_input2edge = params2xform([1, 0, 0, 1, 0, 0])

    xform_full2roi = params2xform([1, 0, -img_roi[2],
                                   1, 0, -img_roi[0]])
    if output_origin_fft:
        # origin of rp-coordinate system, written in the i-coordinate system
        xform_edge2output = params2xform([1, 0, -((img_roi[3] - img_roi[2]) // 2),
                                          1, 0, -((img_roi[1] - img_roi[0]) // 2)])
    else:
        xform_edge2output = params2xform([1, 0, 0, 1, 0, 0])

    xform_full = xform_edge2output.dot(xform_full2roi.dot(affine_mat.dot(xform_input2edge)))

    return xform_sinusoid_params(fx, fy, phase, xform_full)


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
            except:
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
        img_coords = np.meshgrid(np.arange(mat_img.shape[1]), np.arange(mat_img.shape[0]))
        diff = mat_img.ravel() - (p[0] * xform_mat(mat_obj, p2xform(p), img_coords, mode='interp').ravel() + p[1])
        diff[np.isnan(diff)] = 0
        return diff

    results = fit_least_squares(err_fn,
                                init_params,
                                fixed_params=fixed_params,
                                bounds=bounds)
    xform = p2xform(results["fit_params"])

    return results, xform
