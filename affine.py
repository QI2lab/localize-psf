"""
Code for working with affine transformations in 2D and 3D

Determine affine transformation mapping object space to image space.
The affine transformation (in homogeneous coordinates) is represented by a matrix,
[[xi], [yi], [1]] = T * [[xo], [yo], [1]]

Given a function defined on object space, g(xo, yo), we can define a corresponding function on image space
gi(xi, yi) = g(T^{-1} [[xi], [yi], [1]])
"""

import numpy as np
from numpy import fft
import copy
from scipy import optimize
import scipy.interpolate

def xform2params(affine_mat):
    """
    Parametrize affine transformation in terms of rotation angles, magnifications, and offsets.
    T = Mx * cos(tx), -My * sin(ty), vx
        Mx * sin(tx),  My * cos(ty), vy
           0        ,    0        , 1

    Both theta_x and theta_y are measured CCW from the x-axis

    :param np.array affine_mat:

    :return list[float]: [mx, theta_x, vx, my, theta_y, vy]
    """
    # get offsets
    vx = affine_mat[0, -1]
    vy = affine_mat[1, -1]

    # get rotation and scale for x-axis
    theta_x = np.angle(affine_mat[0, 0] + 1j * affine_mat[1, 0])
    mx = np.nanmean([affine_mat[0, 0] / np.cos(theta_x), affine_mat[1, 0] / np.sin(theta_x)])

    # get rotation and scale for y-axis
    theta_y = np.angle(affine_mat[1, 1] - 1j * affine_mat[0, 1])
    my = np.nanmean([affine_mat[1, 1] / np.cos(theta_y), -affine_mat[0, 1] / np.sin(theta_y)])

    return [mx, theta_x, vx, my, theta_y, vy]


def inv_xform2params(affine_mat_inv):
    """
    Compute parameters of affine transform from the inverse matrix

    T = Mx * cos(tx), -My * sin(ty), vx
        Mx * sin(tx),  My * cos(ty), vy
           0        ,    0        , 1

    T^{-1} = |1/Mx,    0,  0 |   | cos(ty)/F, sin(ty)/F, 0|    |1, 0, -vx|
             |  0 ,  1/My, 0 | * |-sin(tx)/F, cos(tx)/F, 0|  * |0, 1, -vy|
             |  0,     0 , 1 |   |     0    ,     0    , 1|    |0, 0,  0|

           = cos(ty)/(Mx*cos(tx-ty), sin(ty)/(My*cos(tx-ty)), -( vx*cos(ty) + vy*sin(ty))/(Mx*cos(tx-ty))
            -sin(tx)/(My*cos(tx-ty), cos(tx)/(My*cos(tx-ty)), -(-Vx*sin(tx) + vy*cos(tx))/(My*cos(tx-ty))
                     0             ,             0          ,                  1

    :param np.array affine_mat_inv:

    :return list[float]: [mx, theta_x, vx, my, theta_y, vy]
    """

    # todo: why not just invert the matrix and apply xform2params?

    # this works as long as np.cos(theta_x - theta_y) > 0
    theta_x = np.angle(affine_mat_inv[1, 1] - 1j * affine_mat_inv[1, 0])
    theta_y = np.angle(affine_mat_inv[0, 0] + 1j * affine_mat_inv[0, 1])
    if np.cos(theta_x - theta_y) < 0:
        theta_x = theta_x + np.pi
        theta_y = theta_y + np.pi

    mx = 1 / (np.cos(theta_x - theta_y) * np.nanmean([affine_mat_inv[0, 0] / np.cos(theta_y),
                                                      affine_mat_inv[0, 1] / np.sin(theta_y)]))
    my = 1 / (np.cos(theta_x - theta_y) * np.nanmean([-affine_mat_inv[1, 0] / np.sin(theta_x),
                                                      affine_mat_inv[1, 1] / np.cos(theta_x)]))

    # invert the matrix we've found so far, and whats left is just the offsets
    mat = params2xform([mx, theta_x, 0, my, theta_y, 0])
    shift_mat = mat.dot(affine_mat_inv)
    vx = -shift_mat[0, 2]
    vy = -shift_mat[1, 2]

    return [mx, theta_x, vx, my, theta_y, vy]


def params2xform(params):
    """
    Construct affine transformation from parameters. Inverse function for xform2params()

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


# transform functions/matrices under action of affine transformation
def affine_xform_mat(mat, xform, img_coords, mode='nearest'):
    """
    Given roi_size matrix defined on object space coordinates, i.e. M[yo, xo], calculate corresponding matrix at image
    space coordinates, M'[yi, xi] = M[ T^{-1} * [xi, yi] ]

    Object coordinates are assumed to be [0, ..., nx-1] and [0, ..., ny-1]

    # todo: change out_shape to a coordinate argument so I can directly generate e.g. a region of interest or etc.
    # todo: rename xform_mat

    :param np.array mat: matrix in object space
    :param np.array xform: affine transformation which takes object space coordinates as input, [yi, xi] = T * [xo, yo]
    :param img_coords: (xi, yi) coordinates where the transformed matrix is evaluated.
    :param str mode: 'nearest' or 'interp'. 'interp' will produce better results if e.g. looking at phase content after
    affine transformation.

    :return mat_out: matrix in image space
    """

    # if obj_coords is None:
    #     xo = np.arange(mat.shape[1])
    #     yo = np.arange(mat.shape[0])
    # else:
    #     xo, yo = obj_coords
    xo = np.arange(mat.shape[1])
    yo = np.arange(mat.shape[0])

    # xi, yi = img_coords
    # xixi, yiyi = np.meshgrid(xi, yi)
    xixi, yiyi = img_coords
    mat_out = np.zeros(xixi.shape)
    # xixi, yiyi = np.meshgrid(range(out_shape[1]), range(out_shape[0]))

    coords_i_aug = np.concatenate((xixi.ravel()[None, :], yiyi.ravel()[None, :], np.ones((1, xixi.size))), axis=0)
    # get corresponding object space coordinates
    coords_o = np.linalg.inv(xform).dot(coords_i_aug)[:2, :]
    xos = np.reshape(coords_o[0, :], xixi.shape)
    yos = np.reshape(coords_o[1, :], xixi.shape)

    if mode == 'nearest':
        # xos = np.round(xos)
        # yos = np.round(yos)

        # only use points with coords in image
        to_use_x = np.logical_and(xos >= np.min(xo), xos <= np.max(xo))
        to_use_y = np.logical_and(yos >= np.min(yo), yos < np.max(yo))
        to_use = np.logical_and(to_use_x, to_use_y)

        # find closest point in matrix to each output point
        # inds_y = tuple([np.argmin(np.abs(y - yo)) for y in yos[to_use]])
        # inds_x = tuple([np.argmin(np.abs(x - xo)) for x in xos[to_use]])

        # inds_y = tuple([int(np.round(y)) for y in yos[to_use]])
        # inds_x = tuple([int(np.round(x)) for x in xos[to_use]])

        inds_y = np.array(np.round(yos[to_use]), dtype=np.int)
        inds_x = np.array(np.round(xos[to_use]), dtype=np.int)

        inds = (tuple(inds_y), tuple(inds_x))
        mat_out[to_use] = mat[inds]

    elif mode == 'interp':
        # only use points with coords in image
        to_use_x = np.logical_and(xos >= 0, xos < mat.shape[1])
        to_use_y = np.logical_and(yos >= 0, yos < mat.shape[0])
        to_use = np.logical_and(to_use_x, to_use_y)

        mat_out = scipy.interpolate.RectBivariateSpline(xo, yo, mat.transpose()).ev(xos, yos)
        mat_out[np.logical_not(to_use)] = 0
    else:
        raise ValueError("'mode' must be 'nearest' or 'interp' but was %s" % mode)

    return mat_out


def xform_fn(fn, xform, out_coords):
    """
    Given a function on object space, evaluate the corresponding image space function at out_coords

    :param fn: function on object space. fn(x, y)
    :param xform: affine transformation matrix which takes points in object space to points in image space
    :param out_coords: (x_img, y_img) coordinates in image space. x_img and y_img must be the same size

    :return img: function evaluated at desired image space coordinates
    """

    x_img, y_img = out_coords
    xform_inv = np.linalg.inv(xform)

    coords_i = np.concatenate((x_img.ravel()[None, :], y_img.ravel()[None, :], np.ones((1, y_img.size))), axis=0)
    coords_o = xform_inv.dot(coords_i)
    x_o = coords_o[0]
    y_o = coords_o[1]

    img = np.reshape(fn(x_o, y_o), x_img.shape)

    return img


def xform_points(xs, ys, xform):
    """
    Transform a set of coordinates under the action of the affine transformation.
    :param xs: array of x-coordinates of arbitrary shape
    :param ys: array of y-coordinates, same shape as xs
    :param xform: affine transformation, 3x3 matrix

    :return xs_out: x-coordinates after affine transformation. Same shape as xs
    :return ys_out: y-coordinates after affine transformation. Same shape as xs.
    """
    coord_vec_in = np.concatenate((xs.ravel()[None, :], ys.ravel()[None, :], np.ones((1, xs.size))), axis=0)
    coord_vec_out = xform.dot(coord_vec_in)
    xs_out = coord_vec_out[0, :].reshape(xs.shape)
    ys_out = coord_vec_out[1, :].reshape(ys.shape)

    return xs_out, ys_out


def xform_points2(coords, xform):
    """
    Transform coordinates of arbitrary dimension under the action of an affine transformation

    # TODO: replace xform_points() with this function

    :param coords: given as an N x Ndim matrix
    :param xform: affine transform matrix
    """
    coord_vec_in = np.concatenate((coords.transpose(), np.ones((1, coords.shape[0]))), axis=0)
    coord_vec_out = xform.dot(coord_vec_in)[:-1].transpose()

    return coord_vec_out


# modify affine xform
def xform_shift_center(xform, cobj_new=None, cimg_new=None):
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

    xform = copy.deepcopy(xform)

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
def phase_edge2fft(frq, phase, img_shape, dx=1):
    """

    :param list[float] or np.array frq:
    :param float phase:
    :param tuple or list img_shape:
    :param float dx:

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


def phase_fft2edge(frq, phase, img_shape, dx=1):
    """

    :param list[float] or np.array frq:
    :param float phase:
    :param tuple or list img_shape:
    :param float dx:

    :return phase_edge:
    """

    nx = img_shape[1]
    xft = fft.fftshift(fft.fftfreq(nx, 1 / (dx*nx)))
    ny = img_shape[0]
    yft = fft.fftshift(fft.fftfreq(ny, 1 / (dx*ny)))

    # xft = tools.get_fft_pos(img_shape[1], dt=dx, centered=True, mode="symmetric")
    # yft = tools.get_fft_pos(img_shape[0], dt=dx, centered=True, mode="symmetric")
    phase_edge = xform_phase_translation(frq[0], frq[1], phase, [xft[0], yft[0]])

    return phase_edge


def xform_phase_translation(fx, fy, phase, shifted_center):
    """
    Transform sinusoid phase based on translating coordinate center. If we make the transformation,
    x' = x - cx
    y' = y - cy
    then the phase transforms
    phase' = phase + 2*pi * (fx * cx + fy * cy)

    :param float fx: x-component of frequency
    :param float fy: y-component of frequency
    :param float phase:
    :param list[float] shifted_center: shifted center in initial coordinates, [cx, cy]

    :return phase_shifted:
    """

    cx, cy = shifted_center
    phase_shifted = np.mod(phase + 2*np.pi * (fx * cx + fy * cy), 2*np.pi)
    return phase_shifted


# transform sinusoid parameters under full affine transformation
def xform_sinusoid_params(fx_obj, fy_obj, phi_obj, affine_mat):
    """
    Given a sinusoid function of object space,
    cos[2pi f_x * x_o + 2pi f_y * y_o + phi],
    find the frequency and phase parameters for the corresponding function on image space,
    cos[2pi f_xi * x_i + 2pi f_yi * yi + phi_i]

    :param float fx_obj: x-component of frequency in object space
    :param float fy_obj: y-component of frequency in object space
    :param float phi_obj: phase in object space
    :param np.array affine_mat: affine transformation homogeneous coordinate matrix transforming
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


def xform_sinusoid_params_roi(fx, fy, phase, object_size, img_roi, affine_mat,
                              input_origin="fft", output_origin="fft"):
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
    :param list[int] img_roi: [ystart, yend, xstart, xend], region of interest in image space. Note: this region does not include
    the pixels at yend and xend! In coordinates with integer values the pixel centers, it is the area
    [ystart - 0.5*dy, yend-0.5*dy] x [xstart -0.5*dx, xend - 0.5*dx]
    :param np.array affine_mat: affine transformation matrix, which takes points from o -> i
    :param str input_origin: "fft" if phase is provided in coordinate system o', or "edge" if provided in coordinate sysem o
    :param str output_origin: "fft" if output phase should be in coordinate system r' or "edge" if in coordinate system r

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


# deprecate this fn in favor of xform_sinusoid_params_roi()
def get_roi_sinusoid_params(roi, fs, phi, dr=None):
    """
    choosing a region of interest (ROI) amounts to making a coordinate transform

    # todo: probably better to write this more generally for a change of center, rather than narrowly as roi shift

    :param list[int] roi: [r1_start, r1_end, r2_start, r2_end, r3_start, ..., rn_end]
    :param fs: [f1, f2, ..., fn]
    :param float phi:
    :param list[float] dr: [dr1, dr2, ..., drn]
    :return:
    """

    if dr is None:
        dr = [1] * len(fs)

    if isinstance(dr, (int, float)):
        dr = [dr]

    if len(dr) == 1 and len(fs) > 1:
        dr = dr * len(fs)

    phi_roi = phi
    for ii in range(len(fs)):
        phi_roi = phi_roi + 2 * np.pi * fs[ii] * roi[2*ii] * dr[ii]

    return np.mod(phi_roi, 2*np.pi)


# fit affine transformation
def fit_affine_xform_points(from_pts, to_pts):
    """
    Solve for affine transformation T = [[A, b], [0, ..., 0, 1]], satisfying
    to_pts = A * from_pts + b, or
    to_pts_aug = T * from_pts_aug
    Put this in roi_size form where Gaussian elimination is applicable by taking transpose of this,
    from_pts_aug^t * T^t = to_pts_aug^t

    This works for any dimension

    Based on a`function <https://elonen.iki.fi/code/misc-notes/affine-fit/>` written by Jarno Elonen
    <elonen@iki.fi> in 2007 (Placed in Public Domain),
    which was in turn based on the paper "Fitting affine and orthogonal transformations between
    two sets of points" Mathematical Communications 9 27-34 (2004) by Helmuth Späth, available
    `here <https://hrcak.srce.hr/712>`

    :param from_pts: npts x ndims array, where each column gives coordinates for a different dimension, e.g. first
     column is x, second is y,...
    :param to_pts: same format as from_pts
    :return soln, affine_mat:
    """
    # todo: could add ability to fix certain parameters, but tricky to do this for 2D/3D dims in same code

    # input and output points as arrays
    # rows are x,y points
    q = np.asarray(from_pts).transpose()
    p = np.asarray(to_pts).transpose()

    # augmented points
    # rows are x, y, 1
    ones_row = np.ones((1, q.shape[1]))
    q_aug = np.concatenate((q, ones_row), axis=0)

    if False:
        # convert to a full rank matrix equation
        # solve using gaussian elimination. soln = [A, b]
        # c = [[xf.xt, xf.yt],
        #      [yf.yt, yf.xt],
        #      [xf.1, yf.1]]
        c = q_aug.dot(p.transpose())
        # d = [[xf.xf, xf.yf, xf.1],
        #      [yf.xf, yf.yf, yf.1],
        #      [1.xf, 1.yf, 1.1]]
        d = q_aug.dot(q_aug.transpose())
        # now we expect
        # d.x = c, where
        # x = [[A, D],
        #      [B, E],
        #      [C, F]]
        # where the affine matrix we are interested in is
        # [[A, B, C],
        #  [D, E, F],
        #  [0, 0, 1]]
        soln = np.linalg.solve(d, c).transpose()

        # construct affine matrix in unprojected space
        btm_row = np.concatenate((np.zeros((1, soln.shape[1] - 1)), np.ones((1, 1))), axis=1)
        affine_mat = np.concatenate((soln, btm_row), axis=0)

    else:
        # Can also solve as a least squares problem ... is this any different?
        # actually two separate problems: [X_from, Y_from, 1] * M1 = X_to; M1 = [[A], [B], [C]]
        #                                 [X_from, Y_from, 1] * M2 = Y_to; M2 = [[D], [E], [F]]
        affine_mat = np.zeros((p.shape[0] + 1, p.shape[0] + 1))
        for ii in range(p.shape[0]):
            row_temp, res, rank, s = np.linalg.lstsq(q_aug.transpose(), p[ii], rcond=None)
            affine_mat[ii] = row_temp
        affine_mat[-1, -1] = 1
        soln = 0

    # todo: remove soln from here...
    return affine_mat


def fit_affine_xform_mask(img, mask, init_params=None):
    """
    Fit affine transformation by comparing img with transformed images of mask
    :param img:
    :param mask:
    :param init_params:

    :return np.array pfit:
    """

    if init_params is None:
        init_params = [1, 0, 0, 0, 1, 0]

    raise NotImplementedError("Function not finished!")
    # todo: need to binarize img
    # todo: OR maybe better idea: look at cross correlation and maximize this
    xform_fn = lambda p: np.array([[p[0], p[1], p[2]], [p[3], p[4], p[5]], [0, 0, 1]])

    # err_fn = lambda p: img.ravel() - affine_xform_mat(mask, xform_fn(p), img.shape, mode='nearest').ravel()
    # fit_dict = optimize.least_squares(err_fn, init_params)
    img_sum = np.sum(img)

    img_coords = np.meshgrid(range(img.shape[1], img.shape[0]))
    min_fn = lambda p: -np.sum(img.ravel() * affine_xform_mat(mask, xform_fn(p), img_coords, mode='interp').ravel()) / \
                       img_sum / np.sum(affine_xform_mat(mask, xform_fn(p), img_coords, mode='interp'))

    fit_dict = optimize.minimize(min_fn, init_params)
    pfit = fit_dict['x']

    return pfit

