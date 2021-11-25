"""
Tools for fitting data using non-linear least squares. The primary fitting functions fit_model() and
fit_least_squares() are wrappers for scipy.optimize.least_squares() which additionally handle fixing
parameters and calculating standard uncertainties. In addition, various commonly used fit functions are collected here,
primarily 1D, 2D, and 3D gaussians allowing for arbitrary rotations.
"""
import copy
import numpy as np
import scipy.optimize
import affine


def fit_model(img, model_fn, init_params, fixed_params=None, sd=None, bounds=None, model_jacobian=None, **kwargs):
    """
    Fit 2D model function to an image. Any Nan values in the image will be ignored. This function is a wrapper for
    for the non-linear least squares fit function scipy.optimize.least_squares() which additionally handles fixing
    parameters and calculating fit uncertainty.

    :param np.array img: nd array
    :param model_fn: function f(p)
    :param list[float] init_params: p = [p1, p2, ..., pn]
    :param list[boolean] fixed_params: list of boolean values, same size as init_params. If None,
     no parameters will be fixed.
    :param sd: uncertainty in parameters y. e.g. if experimental curves come from averages then should be the standard
    deviation of the mean. If None, then will use a value of 1 for all points. As long as these values are all the same
    they will not affect the optimization results, although they will affect chi squared.
    :param tuple[tuple[float]] bounds: (lbs, ubs). If None, -/+ infinity used for all parameters.
    :param model_jacobian: Jacobian of the model function as a list, [df/dp[0], df/dp[1], ...]. If None,
     no jacobian used.
    :param kwargs: additional key word arguments will be passed through to scipy.optimize.least_squares
    :return dict results:
    """
    to_use = np.logical_not(np.isnan(img))

    # if all sd's are nan or zero, set to 1
    if sd is None or np.all(np.isnan(sd)) or np.all(sd == 0):
        sd = np.ones(img.shape)

    # handle uncertainties that will cause fitting to fail
    if np.any(sd == 0) or np.any(np.isnan(sd)):
        sd[sd == 0] = np.nanmean(sd[sd != 0])
        sd[np.isnan(sd)] = np.nanmean(sd[sd != 0])

    # function to be optimized
    def err_fn(p): return np.divide(model_fn(p)[to_use].ravel() - img[to_use].ravel(), sd[to_use].ravel())

    # if it was passed, use model jacobian
    if model_jacobian is not None:
        def jac_fn(p): return [v[to_use] / sd[to_use] for v in model_jacobian(p)]
    else:
        jac_fn = None

    results = fit_least_squares(err_fn, init_params, fixed_params=fixed_params, bounds=bounds,
                                model_jacobian=jac_fn, **kwargs)

    return results


def fit_least_squares(model_fn, init_params, fixed_params=None, bounds=None, model_jacobian=None, **kwargs):
    """
    Wrapper for non-linear least squares fit function scipy.optimize.least_squares which handles fixing parameters
    and calculating fit uncertainty.

    :param model_fn: function of model parameters p which returns an array, where the sum of squares of this array is
    minimized. e.g. if we have a set of data points x_i and we make measurements y_i with uncertainties sigma_i,
    and we have a model m(p, x_i)
     then f(p) = [(m(p, x_i) - y_i) / sigma_i]
    :param list[float] init_params: p = [p1, p2, ..., pn]
    :param list[boolean] fixed_params: list of boolean values, same size as init_params. If None,
     no parameters will be fixed.
    :param tuple[tuple[float]] bounds: (lbs, ubs). If None, -/+ infinity used for all parameters.
    :param model_jacobian: Jacobian of the model function as a list, [df/dp[0], df/dp[1], ...]. If None,
     no jacobian used.
    :param kwargs: additional key word arguments will be passed through to scipy.optimize.least_squares

    :return results: dictionary object. Uncertainty can be obtained from the square rootsof the diagonals of the
     covariance matrix, but these will only be meaningful if variances were appropriately provided for the cost function
    """

    # get default fixed parameters
    if fixed_params is None:
        fixed_params = [False for _ in init_params]

    # default bounds
    if bounds is None:
        bounds = (tuple([-np.inf] * len(init_params)), tuple([np.inf] * len(init_params)))

    init_params = np.array(init_params, copy=True)
    # ensure initial parameters within bounds, if not fixed
    for ii in range(len(init_params)):
        if (init_params[ii] < bounds[0][ii] or init_params[ii] > bounds[1][ii]) and not fixed_params[ii]:
            raise ValueError("Initial parameter at index %d had value %0.2g, which was outside of bounds (%0.2g, %0.2g"
                             % (ii, init_params[ii], bounds[0][ii], bounds[1][ii]))

    if np.any(np.isnan(init_params)):
        raise ValueError("init_params cannot include nans")

    if np.any(np.isnan(bounds)):
        raise ValueError("bounds cannot include nans")

    # if some parameters are fixed, we need to hide them from the fit function to produce correct covariance, etc.
    # Idea: map the "reduced" (i.e. not fixed) parameters onto the full parameter list.
    # do this by looking at each parameter. If it is supposed to be "fixed" substitute the initial parameter. If not,
    # then get the next value from pfree. We find the right index of pfree by summing the number of previously unfixed parameters
    free_inds = [int(np.sum(np.logical_not(fixed_params[:ii]))) for ii in range(len(fixed_params))]

    def pfree2pfull(pfree):
        return np.array([pfree[free_inds[ii]] if not fp else init_params[ii] for ii, fp in enumerate(fixed_params)])

    # map full parameters to reduced set
    def pfull2pfree(pfull): return np.array([p for p, fp in zip(pfull, fixed_params) if not fp])

    # function to minimize the sum of squares of, now as a function of only the free parameters
    def err_fn_pfree(pfree): return model_fn(pfree2pfull(pfree))

    if model_jacobian is not None:
        def jac_fn_free(pfree): return pfull2pfree(model_jacobian(pfree2pfull(pfree))).transpose()
    init_params_free = pfull2pfree(init_params)
    bounds_free = (tuple(pfull2pfree(bounds[0])), tuple(pfull2pfree(bounds[1])))

    # non-linear least squares fit
    if model_jacobian is None:
        fit_info = scipy.optimize.least_squares(err_fn_pfree, init_params_free, bounds=bounds_free, **kwargs)
    else:
        fit_info = scipy.optimize.least_squares(err_fn_pfree, init_params_free, bounds=bounds_free,
                                                jac=jac_fn_free, x_scale='jac', **kwargs)
    pfit = pfree2pfull(fit_info['x'])

    # calculate chi squared
    nfree_params = np.sum(np.logical_not(fixed_params))
    # red_chi_sq = np.sum(np.square(model_fn(pfit))) / (model_fn(init_params).size - nfree_params)
    # scipy.optimize.least_squares minimizes s = 0.5 * \sum |fn(x_i)|^2, so need a factor of two to correct their cost
    red_chi_sq = (2 * fit_info["cost"]) / (model_fn(init_params).size - nfree_params)

    # calculate covariances
    try:
        jacobian = fit_info['jac']
        cov_free = red_chi_sq * np.linalg.inv(jacobian.transpose().dot(jacobian))
    except np.linalg.LinAlgError:
        cov_free = np.nan * np.zeros((jacobian.shape[1], jacobian.shape[1]))

    cov = np.nan * np.zeros((len(init_params), len(init_params)))
    ii_free = 0
    for ii, fpi in enumerate(fixed_params):
        jj_free = 0
        for jj, fpj in enumerate(fixed_params):
            if not fpi and not fpj:
                cov[ii, jj] = cov_free[ii_free, jj_free]
                jj_free += 1
                if jj_free == nfree_params:
                    ii_free += 1

    result = {'fit_params': pfit, 'chi_squared': red_chi_sq, 'covariance': cov,
              'init_params': init_params, 'fixed_params': fixed_params, 'bounds': bounds,
              'cost': fit_info['cost'], 'optimality': fit_info['optimality'],
              'nfev': fit_info['nfev'], 'njev': fit_info['njev'], 'status': fit_info['status'],
              'success': fit_info['success'], 'message': fit_info['message']}

    return result


def get_moments(img, order=1, coords=None, dims=None):
    """
    Calculate moments of distribution of arbitrary size

    :param img: distribution from which moments are calculated
    :param order: order of moments to be calculated
    :param coords: list of coordinate arrays for each dimension e.g. (y, x), where y, x etc. are broadcastable to the
    same size as img
    :param dims: dimensions to be summed over. For example, given roi_size 3D array of size Nz x Ny x Nz,
     calculate the 2D moments of each slice by setting dims = [1, 2]
    :return moments:
    """

    # todo: does not compute any cross moments, e.g. X*Y

    if dims is None:
        dims = range(img.ndim)

    if coords is None:
        coords = [np.arange(s) for ii, s in enumerate(img.shape) if ii in dims]

    # ensure coords are float arrays to avoid overflow issues
    coords = [np.array(c, dtype=float) for c in coords]

    if len(dims) != len(coords):
        raise ValueError('dims and coordinates must have the same length')

    # weight summing only over certain dimensions
    w = np.nansum(img, axis=tuple(dims), dtype=float)

    # as trick to avoid having to meshgrid any of the coordinates, we can use NumPy's array broadcasting. Because this
    # looks at the trailing array dimensions, we need to swap our desired axis to be the last dimension, multiply by the
    # coordinates to do the broadcasting, and then swap back
    # moments = [np.nansum(np.swapaxes(np.swapaxes(img, ii, img.ndim-1) * c**order, ii, img.ndim-1),
    #            axis=tuple(dims), dtype=float) / w
    #            for ii, c in zip(dims, coords)]
    moments = [np.nansum(img * c ** order, axis=tuple(dims), dtype=float) / w for c in coords]

    return moments


# fit data to gaussians
def fit_gauss1d(y, init_params=None, fixed_params=None, sd=None, x=None, bounds=None, **kwargs):
    """
    Fit 1D Gaussian. This is a wrapper for fit_model() which additionally computes reasonably parameter guess values
    from the input data y.

    :param y:
    :param init_params: [A, cx, sx, bg]
    :param fixed_params:
    :param sd:
    :param x:
    :param bounds:
    :return results, fit_function:
    """

    # get coordinates if not provided
    if x is None:
        x = np.arange(len(y))

    # get default initial parameters
    if init_params is None:
        init_params = [None] * 4
    else:
        init_params = copy.deepcopy(init_params)

    # guess reasonable parameters if not provided
    if np.any([ip is None for ip in init_params]):
        to_use = np.logical_not(np.isnan(y))

        bg = np.nanmean(y.ravel())
        amp = np.max(y[to_use].ravel()) - bg

        cx, = get_moments(y, order=1, coords=[x])
        m2x, = get_moments(y, order=2, coords=[x])
        sx = np.sqrt(m2x - cx ** 2)

        ip_default = [amp, cx, sx, bg]

        # set any parameters that were None to the default values
        for ii in range(len(init_params)):
            if init_params[ii] is None:
                init_params[ii] = ip_default[ii]

    if bounds is None:
        bounds = ((-np.inf, x.min(), 0, -np.inf),
                  (np.inf, x.max(), x.max() - x.min(), np.inf))

    def fn(p): return gauss2d(x, np.zeros(x.shape), [p[0], p[1], 0, p[2], 1, p[3], 0])
    def jacob_fn(p): return gauss2d_jacobian(x, np.zeros(x.shape), [p[0], p[1], 0, p[2], 1, p[3], 0])

    result = fit_model(y, fn, init_params, fixed_params=fixed_params,
                       sd=sd, bounds=bounds, model_jacobian=jacob_fn, **kwargs)

    pfit = result['fit_params']
    def fit_fn(x): return gauss2d(x, np.zeros(x.shape), [pfit[0], pfit[1], 0, pfit[2], 1, pfit[3], 0])

    return result, fit_fn


def fit_gauss2d(img, init_params=None, fixed_params=None, sd=None, xx=None, yy=None, bounds=None, **kwargs):
    """
    Fit 2D gaussian function. The angle theta is defined clockwise from the x- (or y-) axis. NOTE: be careful
    with this when looking at results using e.g. matplotlib.imshow, as this will display the negative y-axis on top.

    This is a wrapper for fit_model() which additionally computes reasonably parameter guess values
    from the input data img.

    :param img: 2D image to fit
    :param init_params: [A, cx, cy, sx, sy, bg, theta]
    :param fixed_params: list of boolean values, same size as init_params.
    :param sd: uncertainty in parameters y. e.g. if experimental curves come from averages then should be the standard
    deviation of the mean
    :param xx: 2D array, same size as image (use this instead of 1D array because want to preserve ability to fit on
    non-regularly spaced grids, etc.)
    :param yy:
    :param bounds: (lbs, ubs)
    :return dict results, fit_fn:
    """

    # get coordinates if not provided
    if xx is None or yy is None:
        xx, yy = np.meshgrid(range(img.shape[1]), range(img.shape[0]))

    # get default initial parameters
    if init_params is None:
        init_params = [None] * 7
    else:
        init_params = copy.deepcopy(init_params)

    # guess reasonable parameters if not provided
    if np.any([ip is None for ip in init_params]):
        to_use = np.logical_not(np.isnan(img))

        bg = np.nanmean(img.ravel())
        amp = np.max(img[to_use].ravel()) - bg

        cy, cx = get_moments(img, order=1, coords=(yy, xx))
        m2y, m2x = get_moments(img, order=2, coords=(yy, xx))
        with np.errstate(invalid='ignore'):
            sx = np.sqrt(m2x - cx ** 2)
            sy = np.sqrt(m2y - cy ** 2)

        ip_default = [amp, cx, cy, sx, sy, bg, 0]

        # set any parameters that were None to the default values
        for ii in range(len(init_params)):
            if init_params[ii] is None:
                init_params[ii] = ip_default[ii]

    # replace any bounds which are none with default guesses
    lbs_default = (-np.inf, xx.min(), yy.min(), 0, 0, -np.inf, -np.inf)
    ubs_default = (np.inf, xx.max(), yy.max(), xx.max() - xx.min(), yy.max() - yy.min(), np.inf, np.inf)

    if bounds is None:
        bounds = (lbs_default, ubs_default)
    else:
        lbs = tuple([b if b is not None else lbs_default[ii] for ii, b in enumerate(bounds[0])])
        ubs = tuple([b if b is not None else ubs_default[ii] for ii, b in enumerate(bounds[1])])
        bounds = (lbs, ubs)

    # do fitting
    result = fit_model(img, lambda p: gauss2d(xx, yy, p), init_params, fixed_params=fixed_params,
                       sd=sd, bounds=bounds, model_jacobian=lambda p: gauss2d_jacobian(xx, yy, p), **kwargs)

    # model function
    def fit_fn(x, y): return gauss2d(x, y, result['fit_params'])

    return result, fit_fn


def fit_sum_gauss2d(img, ngaussians, init_params, fixed_params=None, sd=None, xx=None, yy=None, bounds=None, **kwargs):
    """
    Fit 2D gaussian function. The angle theta is defined clockwise from the x- (or y-) axis. NOTE: be careful
    with this when looking at results using e.g. matplotlib.imshow, as this will display the negative y-axis on top.

    :param img: 2D image to fit
    :param in ngaussians: number of Gaussians
    :param init_params: [A1, cx1, cy1, sx1, sy1, theta1, A2, cx2, ..., thetan, bg]
    :param fixed_params: list of boolean values, same size as init_params.
    :param sd: uncertainty in parameters y. e.g. if experimental curves come from averages then should be the standard
    deviation of the mean
    :param xx: 2D array, same size as image (use this instead of 1D array because want to preserve ability to fit on
    non-regularly spaced grids, etc.)
    :param yy:
    :param bounds: (lbs, ubs)
    :return result, fit_function:
    """

    # get coordinates if not provided
    if xx is None or yy is None:
        xx, yy = np.meshgrid(range(img.shape[1]), range(img.shape[0]))

    nparams = 6 * ngaussians + 1
    # get default initial parameters
    if init_params is None:
        init_params = [None] * nparams
    else:
        init_params = copy.deepcopy(init_params)

    if bounds is None:
        bounds = [[-np.inf, xx.min(), yy.min(), 0, 0, -np.inf] * ngaussians + [-np.inf],
                  [ np.inf, xx.max(), yy.max(), xx.max() - xx.min(), yy.max() - yy.min(), np.inf] * ngaussians + [np.inf]]

    result = fit_model(img, lambda p: sum_gauss2d(xx, yy, p), init_params, fixed_params=fixed_params,
                       sd=sd, bounds=bounds, model_jacobian=lambda p: sum_gauss2d_jacobian(xx, yy, p), **kwargs)

    pfit = result['fit_params']

    def fn(x, y):
        return sum_gauss2d(x, y, pfit)

    return result, fn


def fit_half_gauss1d(y, init_params=None, fixed_params=None, sd=None, x=None, bounds=None, **kwargs):
    """
    Fit function that has two Gaussian halves with different sigmas and offsets but match smoothly at cx

    :param y:
    :param init_params: [A1, cx, sx1, bg1, sx2, bg2]
    :param fixed_params:
    :param sd:
    :param x:
    :param bounds:
    :return result, fit_function:
    """

    # get coordinates if not provided
    if x is None:
        x = np.arange(len(y))

    # get default initial parameters
    if init_params is None:
        init_params = [None] * 6
    else:
        # init_params = copy.deepcopy(init_params)
        init_params = [p for p in init_params]

    # guess reasonable parameters if not provided
    if np.any([ip is None for ip in init_params]):
        to_use = np.logical_not(np.isnan(y))

        bg = np.nanmean(y.ravel())
        amp = np.max(y[to_use].ravel()) - bg

        cx, = get_moments(y, order=1, coords=[x])
        m2x, = get_moments(y, order=2, coords=[x])
        sx = np.sqrt(m2x - cx ** 2)

        ip_default = [amp, cx, sx, bg, sx, bg]

        # set any parameters that were None to the default values
        for ii in range(len(init_params)):
            if init_params[ii] is None:
                init_params[ii] = ip_default[ii]

    if bounds is None:
        bounds = ((-np.inf, x.min(), 0, -np.inf, 0, -np.inf),
                  (np.inf, x.max(), x.max() - x.min(), np.inf, x.max() - x.min(), np.inf))

    def hg_fn(x, p): return (p[0] * np.exp(-(x - p[1])**2 / (2*p[2]**2)) + p[3]) * (x < p[1]) + \
                            ((p[0] + p[3] - p[5]) * np.exp(-(x - p[1])**2 / (2*p[4]**2)) + p[5]) * (x >= p[1])

    result = fit_model(y, lambda p: hg_fn(x, p), init_params, fixed_params=fixed_params, sd=sd, bounds=bounds, **kwargs)

    pfit = result['fit_params']
    def fit_fn(x): return hg_fn(x, pfit)

    return result, fit_fn


# gaussians and jacobians
def gauss2d(x, y, p):
    """
    Rotated 2D gaussian function. The angle theta is defined clockwise from the x- (or y-) axis. NOTE: be careful
    with this when looking at results using e.g. matplotlib.imshow, as this will display the negative y-axis on top.

    :param x: x-coordinates to evaluate function at.
    :param y: y-coordinates to evaluate function at. Either same size as x, or broadcastable with x.
    :param p: [A, cx, cy, sxrot, syrot, bg, theta]
    :return value:
    """
    if len(p) != 7:
        raise ValueError("parameter list p must have length 7")

    xrot = np.cos(p[6]) * (x - p[1]) - np.sin(p[6]) * (y - p[2])
    yrot = np.cos(p[6]) * (y - p[2]) + np.sin(p[6]) * (x - p[1])
    return p[0] * np.exp(-xrot ** 2 / (2 * p[3] ** 2) - yrot ** 2 / (2 * p[4] ** 2)) + p[5]


def gauss2d_jacobian(x, y, p):
    """
    Jacobian of gauss2d()

    :param x:
    :param y:
    :param p: [A, cx, cy, sx, sy, bg, theta]
    :return value:
    """
    if len(p) != 7:
        raise ValueError("parameter list p must have length 7")

    # useful functions that show up in derivatives
    xrot = np.cos(p[6]) * (x - p[1]) - np.sin(p[6]) * (y - p[2])
    yrot = np.cos(p[6]) * (y - p[2]) + np.sin(p[6]) * (x - p[1])
    exps = np.exp(-xrot**2 / (2 * p[3] ** 2) -yrot**2 / (2 * p[4] ** 2))

    bcast_shape = (x + y).shape

    return [exps,
            p[0] * exps * (xrot / p[3]**2 * np.cos(p[6]) + yrot / p[4]**2 * np.sin(p[6])),
            p[0] * exps * (yrot / p[4]**2 * np.cos(p[6]) - xrot / p[3]**2 * np.sin(p[6])),
            p[0] * exps * xrot ** 2 / p[3] ** 3,
            p[0] * exps * yrot ** 2 / p[4] ** 3,
            np.ones(bcast_shape),
            p[0] * exps * xrot * yrot * (1 / p[3]**2 - 1 / p[4]**2)]


def gauss2d_v2(x, y, p):
    """
    Rotated 2D gaussian function. The angle theta is defined clockwise from the x- (or y-) axis. NOTE: be careful
    with this when looking at results using e.g. matplotlib.imshow, as this will display the negative y-axis on top.

    :param x: x-coordinates to evaluate function at.
    :param y: y-coordinates to evaluate function at. Either same size as x, or broadcastable with x.
    :param p: [A, cx, cy, sxrot, syrot/sxrot = anistropy, bg, theta]
    :return value:
    """
    if len(p) != 7:
        raise ValueError("parameter list p must have length 7")

    xrot = np.cos(p[6]) * (x - p[1]) - np.sin(p[6]) * (y - p[2])
    yrot = np.cos(p[6]) * (y - p[2]) + np.sin(p[6]) * (x - p[1])
    val = p[0] * np.exp(-xrot ** 2 / (2 * p[3] ** 2) - yrot ** 2 / (2 * p[3] ** 2 * p[4] ** 2)) + p[5]

    return val


def gauss2d_v2_jacobian(x, y, p):
    """
      Jacobian of gauss_fn

      :param x:
      :param y:
      :param p: [A, cx, cy, sx, sy/sx, bg, theta]
      :return value:
      """
    if len(p) != 7:
        raise ValueError("parameter list p must have length 7")

    # useful functions that show up in derivatives
    xrot = np.cos(p[6]) * (x - p[1]) - np.sin(p[6]) * (y - p[2])
    yrot = np.cos(p[6]) * (y - p[2]) + np.sin(p[6]) * (x - p[1])
    exps = np.exp(-xrot ** 2 / (2 * p[3] ** 2) - yrot ** 2 / (2 * p[3] ** 2 * p[4] ** 2))

    bcast_shape = (x + y).shape

    jac = [exps,
           p[0] * exps * (xrot / p[3] ** 2 * np.cos(p[6]) + yrot / p[3]**2 / p[4] ** 2 * np.sin(p[6])),
           p[0] * exps * (yrot / p[3] ** 2 / p[4] ** 2 * np.cos(p[6]) - xrot / p[3] ** 2 * np.sin(p[6])),
           p[0] * exps * (xrot ** 2 / p[3] ** 3 + yrot ** 2 / p[3] ** 3 / p[4]**2),
           p[0] * exps * yrot ** 2 / p[3] ** 2 / p[4] ** 3,
           np.ones(bcast_shape),
           p[0] * exps * xrot * yrot * (1 / p[3] ** 2 - 1 / (p[3] ** 2 * p[4] ** 2))]

    return jac


def sum_gauss2d(x, y, p):
    """
    Sum of n 2D gaussians
    :param x:
    :param y:
    :param p: [amp_1, cx1, cx2, sx1, sx2, theta_1, amp_2, ..., theta_n, bg]
    :return value:
    """
    if len(p) % 6 != 1:
        raise ValueError("Parameter list should have remainder 1 mod 6")

    ngaussians = (len(p) - 1) // 6

    val = 0
    for ii in range(ngaussians - 1):
        ps = np.concatenate((np.array(p[6*ii: 6*ii + 5]), np.array([0]), np.atleast_1d([p[ii * 6 + 5]])))
        val += gauss2d(x, y, ps)

    # deal with last gaussian, which also gets background term
    ps = np.concatenate((np.array(p[-7:-2]), np.atleast_1d(p[-1]), np.atleast_1d(p[-2])))
    val += gauss2d(x, y, ps)
    return val


def sum_gauss2d_jacobian(x, y, p):
    """
    Jacobian of the sum of n 2D gaussians
    :param x:
    :param y:
    :param p:
    :return jacobian:
    """
    if len(p) % 6 != 1:
        raise ValueError("Parameter array had wrong length")

    ngaussians = (len(p) - 1) // 6

    jac_list = []
    for ii in range(ngaussians - 1):
        ps = np.concatenate((np.array(p[6 * ii: 6 * ii + 5]), np.array([0]), np.atleast_1d([p[ii * 6 + 5]])))
        jac_current = gauss2d_jacobian(x, y, ps)
        jac_list += jac_current[:-2] + [jac_current[-1]]

    # deal with last gaussian, which also gets background term
    ps = np.concatenate((np.array(p[-7:-2]), np.atleast_1d(p[-1]), np.atleast_1d(p[-2])))
    jac_current = gauss2d_jacobian(x, y, ps)
    jac_list += jac_current[:-2] + [jac_current[-1]] + [jac_current[-2]]

    return jac_list


def gauss3d(x, y, z, p):
    """
    3D gaussian, with arbitrary rotation parameterized by Euler angles

    r_body = U_z(psi)^-1 U_y(theta)^-1 U_z(phi)^-1 * r_lab
    U_z(phi)^-1 = [[cos(phi), -sin(phi), 0], [sin(phi), cos(phi), 0], [0, 0, 1]]
    f_rot(r_lab) = f(r_body) = f(U^{-1} * r_lab)

    Take the z-axis in the frame of the object, and consider the z-axis in the lab frame. phi and theta describe
    how the transformation to overlap these two. psi gives the gives the angle the object is rotated about its own axis

    :param x: x-coordinates to evaluate function at. x, y, z must be the same size, or broadcastable to the same size
    :param y: y-coordinates to evaluate function at.
    :param z: z-coordinates to evaluate function at.
    :param p: [A, cx, cy, cz, sigma x_rot, sigma y_rot, sigma z_rot, bg, phi, theta, psi]
    :return value:
    """

    phi = p[8]
    theta = p[9]
    psi = p[10]
    rot_mat = affine.euler_mat_inv(phi, theta, psi)
    xrot = (x - p[1]) * rot_mat[0, 0] + (y - p[2]) * rot_mat[0, 1] + (z - p[3]) * rot_mat[0, 2]
    yrot = (x - p[1]) * rot_mat[1, 0] + (y - p[2]) * rot_mat[1, 1] + (z - p[3]) * rot_mat[1, 2]
    zrot = (x - p[1]) * rot_mat[2, 0] + (y - p[2]) * rot_mat[2, 1] + (z - p[3]) * rot_mat[2, 2]
    val = p[0] * np.exp(-xrot**2 / (2 * p[4]**2) - yrot**2 / (2 * p[5] ** 2) - zrot**2 / (2 * p[6]**2)) + p[7]

    return val


def gauss3d_jacobian(x, y, z, p):
    """
    Calculate Jacobian matrix of gauss3d

    :param x: x-coordinates to evaluate function at.
    :param y: y-coordinates to evaluate function at. Either same size as x, or broadcastable with x.
    :param z: z-coordinates
    :param p: [A, cx, cy, cz, sxrot, syrot, szrot, bg, phi, theta, psi]
    :return value:
    """
    bcast_shape = (x + y + z).shape

    phi = p[8]
    theta = p[9]
    psi = p[10]

    rot_mat = affine.euler_mat_inv(phi, theta, psi)
    dphi, dtheta, dpsi = affine.euler_mat_inv_derivatives(phi, theta, psi)

    xrot = (x - p[1]) * rot_mat[0, 0] + (y - p[2]) * rot_mat[0, 1] + (z - p[3]) * rot_mat[0, 2]
    yrot = (x - p[1]) * rot_mat[1, 0] + (y - p[2]) * rot_mat[1, 1] + (z - p[3]) * rot_mat[1, 2]
    zrot = (x - p[1]) * rot_mat[2, 0] + (y - p[2]) * rot_mat[2, 1] + (z - p[3]) * rot_mat[2, 2]
    exp = np.exp(-xrot**2 / (2 * p[4]**2) - yrot**2 / (2 * p[5] ** 2) - zrot**2 / (2 * p[6]**2))

    jac = [exp,
           p[0] * exp * (xrot / p[4]**2 * rot_mat[0, 0] + yrot / p[5]**2 * rot_mat[1, 0] + zrot / p[6]**2 * rot_mat[2, 0]),
           p[0] * exp * (xrot / p[4]**2 * rot_mat[0, 1] + yrot / p[5]**2 * rot_mat[1, 1] + zrot / p[6]**2 * rot_mat[2, 1]),
           p[0] * exp * (xrot / p[4]**2 * rot_mat[0, 2] + yrot / p[5]**2 * rot_mat[1, 2] + zrot / p[6]**2 * rot_mat[2, 2]),
           p[0] * exp * xrot ** 2 / p[4] ** 3,
           p[0] * exp * yrot ** 2 / p[5] ** 3,
           p[0] * exp * zrot ** 2 / p[6] ** 3,
           np.ones(bcast_shape),
           -p[0] * exp * (
               (xrot / p[4]**2) * ((x - p[1]) * dphi[0, 0] + (y - p[2]) * dphi[0, 1] + (z - p[3]) * dphi[0, 2]) +
               (yrot / p[5]**2) * ((x - p[1]) * dphi[1, 0] + (y - p[2]) * dphi[1, 1] + (z - p[3]) * dphi[1, 2]) +
               (zrot / p[6]**2) * ((x - p[1]) * dphi[2, 0] + (y - p[2]) * dphi[2, 1] + (z - p[3]) * dphi[2, 2])),
           -p[0] * exp * (
                   (xrot / p[4] ** 2) * ((x - p[1]) * dtheta[0, 0] + (y - p[2]) * dtheta[0, 1] + (z - p[3]) * dtheta[0, 2]) +
                   (yrot / p[5] ** 2) * ((x - p[1]) * dtheta[1, 0] + (y - p[2]) * dtheta[1, 1] + (z - p[3]) * dtheta[1, 2]) +
                   (zrot / p[6] ** 2) * ((x - p[1]) * dtheta[2, 0] + (y - p[2]) * dtheta[2, 1] + (z - p[3]) * dtheta[2, 2])),
           -p[0] * exp * (
                   (xrot / p[4] ** 2) * ((x - p[1]) * dpsi[0, 0] + (y - p[2]) * dpsi[0, 1] + (z - p[3]) * dpsi[0, 2]) +
                   (yrot / p[5] ** 2) * ((x - p[1]) * dpsi[1, 0] + (y - p[2]) * dpsi[1, 1] + (z - p[3]) * dpsi[1, 2]) +
                   (zrot / p[6] ** 2) * ((x - p[1]) * dpsi[2, 0] + (y - p[2]) * dpsi[2, 1] + (z - p[3]) * dpsi[2, 2]))
           ]

    return jac


def circle(x, y, p):
    """
    Function which attains one value within a circle and another value outside it. These regions are continuously
    stitched together by an exponential decay. This length should be non-zero for reliable fitting
    @param x: x-coordinates to evaluate function at.
    @param y:
    @param p: [cx, cy, radius, in value, out value, decay_len]
    @return value:
    """
    dist = np.sqrt((x - p[0])**2 + (y - p[1])**2)
    in_circ = p[3] * np.exp((p[2] - dist) / p[5]) + p[4]

    in_circ[dist < p[2]] = p[3]

    return in_circ


def line_piecewise(x, p):
    """
    Two piecewise lines which connect at a point
    @param x: x-positions to evaluate function
    @param p: [slope 1, y-intercept 1, slope 2, changover point]
    @return value:
    """
    l1 = p[0] * x + p[1]
    # l1(p[3]) = l2(p[3])
    b2 = (p[0] - p[2]) * p[3] + p[1]
    l2 = p[2] * x + b2

    line = l1
    line[x >= p[3]] = l2[x >= p[3]]
    return line


def sinc_squared2d(x, y, p):
    """
    Product of sinc squareds
    @param x: x-points to evaluate function
    @param y: y-points to evaluate function
    @param p: [amp, cx, cy, wx, wy, bg, theta]
    @return value:
    """

    xrot = np.cos(p[6]) * (x - p[1]) - np.sin(p[6]) * (y - p[2])
    argx = xrot * p[3]
    yrot = np.cos(p[6]) * (y - p[2]) + np.sin(p[6]) * (x - p[1])
    argy = yrot * p[4]

    xpart = (np.sin(argx) / argx)**2
    xpart[argx == 0] = 1
    ypart = (np.sin(argy) / argy)**2
    ypart[argy == 0] = 1

    val = p[0] * xpart * ypart + p[5]

    return val
