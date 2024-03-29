"""
Code for working with affine transformations in 2D and 3D

Determine affine transformation mapping object space to image space.
The affine transformation (in homogeneous coordinates) is represented by a matrix,
[[xi], [yi], [1]] = T * [[xo], [yo], [1]]

Given a function defined on object space, g(xo, yo), we can define a corresponding function on image space
gi(xi, yi) = g(T^{-1} [[xi], [yi], [1]])
"""
from typing import Optional, Sequence
import joblib
import numpy as np
from numpy import fft
from scipy.interpolate import RectBivariateSpline
from localize_psf import fit


def xform2params(affine_mat: np.ndarray) -> np.ndarray:
    """
    Parametrize 2D affine transformation in terms of rotation angles, magnifications, and offsets.
    T = [[Mx * cos(tx), -My * sin(ty), vx],
         [Mx * sin(tx),  My * cos(ty), vy],
         [   0        ,    0        , 1]]

    Both theta_x and theta_y are measured CCW from the x-axis

    :param np.array affine_mat:

    :return list[float]: [mx, theta_x, vx, my, theta_y, vy]
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

    T = Mx * cos(tx), -My * sin(ty), vx
        Mx * sin(tx),  My * cos(ty), vy
           0        ,    0        , 1

    :param list[float] params: [Mx, theta_x, vx ,My, theta_y, vy]

    :return np.array affine_xform:
    """
    # read parameters
    mx = params[0]
    theta_x = params[1]
    vx = params[2]
    my = params[3]
    theta_y = params[4]
    vy = params[5]

    # construct affine xform
    affine_xform = np.array([[mx * np.cos(theta_x), -my * np.sin(theta_y), vx],
                             [mx * np.sin(theta_x),  my * np.cos(theta_y), vy],
                             [0                   ,  0                   ,  1]])

    return affine_xform


def rotation2xform(angle: float,
                   center: Sequence[float]) -> np.ndarray:
    """
    Get 2D transform corresponding to a rotation by a given angle about a given center

    :param angle:
    :param center:
    :return:
    """

    # think of this xform as
    # Rot Matrix * [[X - cx], [Y - cy]] + [[cx], [cy]]

    cx, cy = center
    xform = params2xform([1, angle, cx, 1, angle, cy])
    extra_offset = -xform[:2, :2].dot(np.array([[cy], [cx]]))
    xform[:-1, -1] += extra_offset.ravel()

    return xform


# transform functions/matrices under action of affine transformation
def xform_mat(mat_obj: np.ndarray,
              xform: np.ndarray,
              img_coords: tuple[np.ndarray],
              mode: str = 'nearest') -> np.ndarray:
    """
    Given a matrix defined on object space coordinates, M[yo, xo], calculate corresponding matrix at image
    space coordinates. This is given by (roughly speaking)
    M'[yi, xi] = M[ T^{-1} * [xi, yi] ]

    Object coordinates are assumed to be [0, ..., nx-1] and [0, ..., ny-1]
    # todo: want object coordinates to be on a grid, but don't want to force a specific one like this ...

    :param mat_obj: matrix in object space
    :param xform: affine transformation which takes object space coordinates as input, [yi, xi] = T * [xo, yo]
    :param img_coords: (c1, c0) list of coordinate arrays where the image-space matrix is to be evaluated. All
    coordinate arrays must be the same shape. i.e., xi.shape = yi.shape.
    :param str mode: 'nearest' or 'interp'. 'interp' will produce better results if e.g. looking at phase content after
    affine transformation.

    :return mat_img: matrix in image space, M'[yi, xi]
    """
    if mat_obj.ndim != 2:
        raise ValueError("img_obj must be a 2D array")

    # image space coordinates
    output_shape = img_coords[0].shape
    coords_img = np.stack([ic.ravel() for ic in img_coords], axis=1)

    # get corresponding object space coordinates
    xform_inv = np.linalg.inv(xform)

    coords_obj_from_img = xform_points(coords_img, xform_inv).transpose()
    coords_obj_from_img = [np.reshape(c, output_shape) for c in coords_obj_from_img]

    # only use points with coords in image
    coords_obj_bounds = [np.arange(mat_obj.shape[1]), np.arange(mat_obj.shape[0])]

    to_use = np.logical_and.reduce([np.logical_and(oc >= np.min(ocm),
                                                   oc <= np.max(ocm))
                                    for oc, ocm in zip(coords_obj_from_img, coords_obj_bounds)])

    # get matrix in image space
    if mode == 'nearest':
        # find closest point in image to each output point
        inds = [tuple(np.array(np.round(oc[to_use]), dtype=int)) for oc in coords_obj_from_img]
        inds.reverse()

        # evaluate matrix
        mat_img = np.zeros(output_shape) * np.nan
        mat_img[to_use] = mat_obj[tuple(inds)]

    elif mode == 'interp':
        mat_img = RectBivariateSpline(*coords_obj_bounds, mat_obj.transpose()).ev(*coords_obj_from_img)
        mat_img[np.logical_not(to_use)] = np.nan
    else:
        raise ValueError("'mode' must be 'nearest' or 'interp' but was '%s'" % mode)

    return mat_img


def xform_fn(fn: callable,
             xform: np.ndarray):
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


def xform_points(coords: np.ndarray,
                 xform: np.ndarray) -> np.ndarray:
    """
    Transform coordinates of arbitrary dimension under the action of an affine transformation

    :param coords: array of shape n0 x n1 x ... nm x ndim
    :param xform: affine transform matrix of shape (ndim + 1) x (ndim + 1)
    :return coords_out: n0 x n1 x ... nm x ndim
    """
    # coords_in = np.concatenate((coords.transpose(), np.ones((1, coords.shape[0]))), axis=0)
    # clip off extra dimension and return
    # coords_out = xform.dot(coords_in)[:-1].transpose()

    ndims = coords.shape[-1]
    coords_in = np.stack([coords[..., ii].ravel() for ii in range(ndims)] + [np.ones((coords[..., 0].size))], axis=0)

    # trim off homogeneous coordinate row and reshape
    coords_out = xform.dot(coords_in)[:-1].transpose().reshape(coords.shape)

    return coords_out


# modify affine xform
def xform_shift_center(xform: np.ndarray,
                       cobj_new: Optional[Sequence[float]] = None,
                       cimg_new: Optional[Sequence[float]] = None) -> np.ndarray:
    """
    Modify affine transform for coordinate shift in object or image space.

    Useful e.g. for changing region of interest

    Ro_new = Ro_old - Co
    Ri_new = Ri_old - Ci

    :param xform:
    :param cobj_new: [cox, coy]
    :param cimg_new: [cix, ciy]
    :return:
    """
    # todo ... this should be implemented by multiplying affine matrices ...

    xform = np.array(xform, copy=True)

    if cobj_new is None:
        cobj_new = [0, 0]
    cox, coy = cobj_new

    xform[0, 2] = xform[0, 2] + xform[0, 0] * cox + xform[0, 1] * coy
    xform[1, 2] = xform[1, 2] + xform[1, 0] * cox + xform[1, 1] * coy

    if cimg_new is None:
        cimg_new = [0, 0]
    cix, ciy = cimg_new

    xform[0, 2] = xform[0, 2] - cix
    xform[1, 2] = xform[1, 2] - ciy

    return xform


# transform sinusoid parameters for coordinate shifts
def phase_edge2fft(frq: Sequence[float],
                   phase: float,
                   img_shape: Sequence[int],
                   dx: float = 1.):
    """
    Give a sinusoidal pattern where we have defined the edge of the image to be x=0, and given the phase determined
    using this coordinate choice, transform the phase to the value referenced to near the center of the image
    (i.e. using the coordinate conventions of the discrete Fourier transform)

    :param frq:
    :param phase:
    :param img_shape:
    :param dx:

    :return phase_fft:
    """
    nx = img_shape[1]
    xft = fft.fftshift(fft.fftfreq(nx, 1 / (dx*nx)))
    ny = img_shape[0]
    yft = fft.fftshift(fft.fftfreq(ny, 1 / (dx*ny)))

    # xft = tools.get_fft_pos(img_shape[1], dt=dx, centered=True, mode="symmetric")
    # yft = tools.get_fft_pos(img_shape[0], dt=dx, centered=True, mode="symmetric")
    phase_fft = xform_phase_translation(frq[0], frq[1], phase, [-xft[0], -yft[0]])

    return phase_fft


def phase_fft2edge(frq: Sequence[float],
                   phase: float,
                   img_shape: Sequence[int],
                   dx: float = 1.):
    """

    :param frq:
    :param phase:
    :param img_shape:
    :param dx:

    :return phase_edge:
    """

    nx = img_shape[1]
    xft = fft.fftshift(fft.fftfreq(nx, 1 / (dx*nx)))
    ny = img_shape[0]
    yft = fft.fftshift(fft.fftfreq(ny, 1 / (dx*ny)))

    phase_edge = xform_phase_translation(frq[0], frq[1], phase, [xft[0], yft[0]])

    return phase_edge


def xform_phase_translation(fx: float,
                            fy: float,
                            phase: float,
                            shifted_center: Sequence[float]):
    """
    Transform sinusoid phase based on translating coordinate center. If we make the transformation,
    x' = x - cx
    y' = y - cy
    then the phase transforms
    phase' = phase + 2*pi * (fx * cx + fy * cy)

    :param fx: x-component of frequency
    :param fy: y-component of frequency
    :param phase:
    :param shifted_center: shifted center in initial coordinates, [cx, cy]

    :return phase_shifted:
    """

    cx, cy = shifted_center
    phase_shifted = np.mod(phase + 2*np.pi * (fx * cx + fy * cy), 2*np.pi)
    return phase_shifted


# transform sinusoid parameters under full affine transformation
def xform_sinusoid_params(fx_obj: float,
                          fy_obj: float,
                          phi_obj: float,
                          affine_mat: np.ndarray) -> (float, float, float):
    """
    Given a sinusoid function of object space,
    cos[2pi f_x * xo + 2pi f_y * yo + phi_o],
    and an affine transformation mapping object space to image space, [xi, yi] = A * [xo, yo]
    find the frequency and phase parameters for the corresponding function on image space,
    cos[2pi f_xi * xi + 2pi f_yi * yi + phi_i]

    :param fx_obj: x-component of frequency in object space
    :param fy_obj: y-component of frequency in object space
    :param phi_obj: phase in object space
    :param affine_mat: affine transformation homogeneous coordinate matrix transforming
     points in object space to image space

    :return fx_img: x-component of frequency in image space
    :return fy_img: y-component of frequency in image space
    :return phi_img: phase in image space
    """
    affine_inv = np.linalg.inv(affine_mat)
    fx_img = fx_obj * affine_inv[0, 0] + fy_obj * affine_inv[1, 0]
    fy_img = fx_obj * affine_inv[0, 1] + fy_obj * affine_inv[1, 1]
    phi_img = np.mod(phi_obj + 2 * np.pi * fx_obj * affine_inv[0, 2] + 2 * np.pi * fy_obj * affine_inv[1, 2], 2 * np.pi)

    return fx_img, fy_img, phi_img


def xform_sinusoid_params_roi(fx: float,
                              fy: float,
                              phase: float,
                              object_size: list[int],
                              img_roi: list[int],
                              affine_mat: np.ndarray,
                              input_origin: str = "fft",
                              output_origin: str = "fft"):
    """
    Transform sinusoid parameter from object space to a region of interest in image space.

    # todo: would it be more appropriate to put this function in sim_reconstruction.py?

    This is an unfortunately complicated function because we have five coordinate systems to worry about
    o: object space coordinates with origin at the corner of the DMD pattern
    o': object space coordinates assumed by fft functions
    i: image space coordinates, with origin at corner of the camera
    r: roi coordinates with origin at the edge of the roi
    r': roi coordinates, with origin near the center of the roi (coordinates for fft)
    The frequencies don't care about the coordinate origin, but the phase does

    :param float fx: x-component of frequency in object space
    :param float fy: y-component of frequency in object space
    :param float phase: phase of pattern in object space coordinates system o or o'.
    :param list[int] object_size: [sy, sx], size of object space, required to define origin of o'
    :param list[int] img_roi: [ystart, yend, xstart, xend], region of interest in image space. Note: this region does
     not include the pixels at yend and xend! In coordinates with integer values the pixel centers, it is the area
    [ystart - 0.5*dy, yend-0.5*dy] x [xstart -0.5*dx, xend - 0.5*dx]
    :param np.array affine_mat: affine transformation matrix, which takes points from o -> i
    :param str input_origin: "fft" if phase is provided in coordinate system o', or "edge" if provided in
     coordinate system o
    :param str output_origin: "fft" if output phase should be in coordinate system r' or "edge" if in
     coordinate system r

    :return fx_xform: x-component of frequency in coordinate system r'
    :return fy_xform: y-component of frequency in coordinates system r'
    :return phi_xform: phase in coordinates system r or r' (depending on the value of output_origin)
    """

    if input_origin == "fft":
        phase_o = phase_fft2edge([fx, fy], phase, object_size, dx=1)
        # xft = tools.get_fft_pos(object_size[1])
        # yft = tools.get_fft_pos(object_size[0])
        # phase_o = xform_phase_translation(fx, fy, phase, [xft[0], yft[0]])
    elif input_origin == "edge":
        phase_o = phase
    else:
        raise ValueError("input origin must be 'fft' or 'edge' but was '%s'" % input_origin)

    # affine transformation, where here we take coordinate origins at the corners
    fx_xform, fy_xform, phase_i = xform_sinusoid_params(fx, fy, phase_o, affine_mat)

    if output_origin == "edge":
        phase_r = xform_phase_translation(fx_xform, fy_xform, phase_i, [img_roi[2], img_roi[0]])
        phase_xform = phase_r
    elif output_origin == "fft":
        # transform so that phase is relative to center of ROI
        ystart, yend, xstart, xend = img_roi

        nx = xend - xstart
        x_rp = fft.fftshift(fft.fftfreq(nx, 1 / nx))
        ny = yend - ystart
        y_rp = fft.fftshift(fft.fftfreq(ny, 1/ny))
        # x_rp = tools.get_fft_pos(xend - xstart, dt=1, centered=True, mode="symmetric")
        # y_rp = tools.get_fft_pos(yend - ystart, dt=1, centered=True, mode="symmetric")

        # origin of rp-coordinate system, written in the i-coordinate system
        cx = xstart - x_rp[0]
        cy = ystart - y_rp[0]

        phase_rp = xform_phase_translation(fx_xform, fy_xform, phase_i, [cx, cy])
        phase_xform = phase_rp
    else:
        raise ValueError("output_origin must be 'fft' or 'edge' but was '%s'" % output_origin)

    return fx_xform, fy_xform, phase_xform


# fit affine transformation
def fit_xform_points(from_pts,
                     to_pts,
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

    :return affine_mat: affine matrix. This is an (ndim + 1) x (ndim + 1) matrix which act on homogeneous coordinates.
    To transform coordinates using this affine transformation use xform_points()
    :return vars: estimated variances of the affine transformation matrix entries
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
            params_temp, residuals, rank, svals = np.linalg.lstsq(from_pts_aug.transpose(), to_pts[ii], rcond=None)
            affine_mat[ii] = params_temp

            # variances of fit parameters
            xt_x_inv = np.linalg.inv(from_pts_aug.dot(from_pts_aug.transpose()))
            var_sample = residuals / (npts - (ndim + 1))
            vars[ii] = np.diag(xt_x_inv) * var_sample
        else:
            params_temp, residuals, rank, svals = \
                np.linalg.lstsq(np.expand_dims(from_pts_aug[-1], axis=1), to_pts[ii] - from_pts[ii], rcond=None)
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

    :result xform_best, inliers_best, err_best, vars_best:
    """

    if njobs > 1:
        niters_each = int(np.ceil(niterations / njobs))
        results = joblib.Parallel(n_jobs=-1, verbose=0, timeout=None)(
                  joblib.delayed(fit_xform_points_ransac)(from_pts,
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
            is_inlier_prop.ravel()[np.sort(np.random.choice(np.arange(npts), size=ninit_pts, replace=False))] = True
            not_inlier_prop = np.logical_not(is_inlier_prop)

            xform_prop, _ = fit_xform_points(from_pts[is_inlier_prop],
                                             to_pts[is_inlier_prop],
                                             translate_only=translate_only)

            # get distance errors of other points to determine if inliers or outliers
            dist_errs = np.linalg.norm(to_pts[not_inlier_prop] - xform_points(from_pts[not_inlier_prop], xform_prop), axis=-1)

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

    :param mat_obj: matrix in object space
    :param mat_img: matrix in image space
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

    results = fit.fit_least_squares(err_fn, init_params, fixed_params=fixed_params, bounds=bounds)
    xform = p2xform(results["fit_params"])

    return results, xform


# ######################
# rotation matrices
# ######################
def get_rot_mat(rot_axis: Sequence[float],
                gamma: float) -> np.ndarray:
    """
    Get matrix which rotates points about the specified axis by the given angle. Think of this rotation matrix
    as acting on unit vectors, and hence its inverse R^{-1} transforms regular vectors. Therefore, we define
    this matrix such that it rotates unit vectors in a lefthanded sense about the given axis for positive gamma.
    e.g. when rotating about the z-axis this becomes
    [[cos(gamma), -sin(gamma), 0],
     [sin(gamma), cos(gamma), 0],
     [0, 0, 1]]
    since vectors are acted on by the inverse matrix, they rotated in a righthanded sense about the given axis.

    :param rot_axis: unit vector specifying axis to rotate about, [nx, ny, nz]
    :param float gamma: rotation angle in radians to transform point. A positive angle corresponds right-handed rotation
    about the given axis
    :return mat: 3x3 rotation matrix
    """
    if np.abs(np.linalg.norm(rot_axis) - 1) > 1e-12:
        raise ValueError("rot_axis must be a unit vector")

    nx, ny, nz = rot_axis
    mat = np.array([[nx**2 * (1 - np.cos(gamma)) + np.cos(gamma), nx * ny * (1 - np.cos(gamma)) - nz * np.sin(gamma), nx * nz * (1 - np.cos(gamma)) + ny * np.sin(gamma)],
                    [nx * ny * (1 - np.cos(gamma)) + nz * np.sin(gamma), ny**2 * (1 - np.cos(gamma)) + np.cos(gamma), ny * nz * (1 - np.cos(gamma)) - nx * np.sin(gamma)],
                    [nx * nz * (1 - np.cos(gamma)) - ny * np.sin(gamma), ny * nz * (1 - np.cos(gamma)) + nx * np.sin(gamma), nz**2 * (1 - np.cos(gamma)) + np.cos(gamma)]])
    return mat


def get_rot_mat_angle_axis(rot_mat: np.ndarray) -> (np.ndarray, float):
    """
    Given a rotation matrix, determine the axis it rotates about and the angle it rotates through. This is
    the inverse function for get_rot_mat()

    Note that get_rot_mat_angle_axis(get_rot_mat(axis, angle)) can return either axis, angle or -axis, -angle
    as these two rotation matrices are equivalent

    :param rot_mat:
    :return rot_axis, angle:
    """
    if np.linalg.norm(rot_mat.dot(rot_mat.transpose()) - np.identity(rot_mat.shape[0])) > 1e-12:
        raise ValueError("rot_mat was not a valid rotation matrix")

    eig_vals, eig_vects = np.linalg.eig(rot_mat)

    # rotation matrix must have one eigenvalue that is 1 to numerical precision
    ind = np.argmin(np.abs(eig_vals - 1))

    # construct basis with e3 = rotation axis
    e3 = eig_vects[:, ind].real

    if np.linalg.norm(np.cross(np.array([0, 1, 0]), e3)) != 0:
        e1 = np.cross(np.array([0, 1, 0]), e3)
    else:
        e1 = np.cross(np.array([1, 0, 0]), e3)
    e1 = e1 / np.linalg.norm(e1)

    e2 = np.cross(e3, e1)

    # basis change matrix to look like rotation about z-axis
    mat_basis_change = np.vstack((e1, e2, e3)).transpose()

    # transformed rotation matrix
    r_bc = np.linalg.inv(mat_basis_change).dot(rot_mat.dot(mat_basis_change))
    angle = np.arcsin(r_bc[1, 0]).real
    # angle = np.arcsin(r_bc[0, 1]).real

    return e3, angle


def euler_mat(phi: float,
              theta: float,
              psi: float) -> np.ndarray:
    """
    Define our Euler angles connecting the body frame to the space/lab frame by
    r_lab = U_z(phi) * U_y(theta) * U_z(psi) * r_body
    The coordinates are column vectors r = [[x], [y], [z]], so
    U_z(phi) = [[cos(phi), -sin(phi), 0], [sin(phi), cos(phi), 0], [0, 0, 1]]
    U_y(theta) = [[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]]

    Consider the z-axis in the body frame. This axis is then orientated at
    [cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta)]
    in the space frame. i.e. phi, theta are the usual polar angles. psi represents a rotation of the object
    about its own axis.

    :param phi:
    :param theta:
    :param psi:
    :return euler_mat: U_z(phi) * U_y(theta) * U_z(psi)
    """
    euler_mat = np.array([[np.cos(phi) * np.cos(theta) * np.cos(psi) - np.sin(phi) * np.sin(psi),
                          -np.cos(phi) * np.cos(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi),
                           np.cos(phi) * np.sin(theta)],
                          [np.sin(phi) * np.cos(theta) * np.cos(psi) + np.cos(phi) * np.sin(psi),
                          -np.sin(phi) * np.cos(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi),
                           np.sin(phi) * np.sin(theta)],
                          [-np.sin(theta) * np.cos(psi), np.sin(theta) * np.sin(psi), np.cos(theta)]])

    return euler_mat


def euler_mat_inv(phi: float,
                  theta: float,
                  psi: float) -> np.ndarray:
    """
    r_body = U_z(-psi) * U_y(-theta) * U_z(-phi) * r_lab

    :param phi:
    :param theta:
    :param psi:
    :return dphi, dtheta, dsi:
    """
    return euler_mat(-psi, -theta, -phi)


def euler_mat_derivatives(phi: float,
                          theta: float,
                          psi: float) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Derivative of Euler matrix with respect to Euler angles

    :param phi:
    :param theta:
    :param psi:
    :return dphi, dtheta, dsi:
    """
    dphi = np.array([[-np.sin(phi) * np.cos(theta) * np.cos(psi) - np.cos(phi) * np.sin(psi),
                       np.sin(phi) * np.cos(theta) * np.sin(psi) - np.cos(phi) * np.cos(psi),
                      -np.sin(phi) * np.sin(theta)],
                     [ np.cos(phi) * np.cos(theta) * np.cos(psi) - np.sin(phi) * np.sin(psi),
                      -np.cos(phi) * np.cos(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi),
                       np.cos(phi) * np.sin(theta)],
                     [0, 0, 0]])
    dtheta = np.array([[-np.cos(phi) * np.sin(theta) * np.cos(psi),
                         np.cos(phi) * np.sin(theta) * np.sin(psi),
                         np.cos(phi) * np.cos(theta)],
                       [-np.sin(phi) * np.sin(theta) * np.cos(psi),
                         np.sin(phi) * np.sin(theta) * np.sin(psi),
                         np.sin(phi) * np.cos(theta)],
                       [-np.cos(theta) * np.cos(psi), np.cos(theta) * np.sin(psi), -np.sin(theta)]])
    dpsi = np.array([[-np.cos(phi) * np.cos(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi),
                      -np.cos(phi) * np.cos(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi),
                      0],
                     [-np.sin(phi) * np.cos(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi),
                      -np.sin(phi) * np.cos(theta) * np.cos(psi) - np.cos(phi) * np.sin(psi),
                      0],
                     [np.sin(theta) * np.sin(psi), np.sin(theta) * np.cos(psi), 0]])

    return dphi, dtheta, dpsi


def euler_mat_inv_derivatives(phi: float,
                              theta: float,
                              psi: float) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Derivative of inverse Euler matrix with respect to Euler angles

    :param phi:
    :param theta:
    :param psi:
    :return dphi, dtheta, dpsi:
    """
    d1, d2, d3 = euler_mat_derivatives(-psi, -theta, -phi)
    dphi = -d3
    dtheta = -d2
    dpsi = -d1

    return dphi, dtheta, dpsi
