"""
Tools for fitting data using non-linear least squares. The recommend way to do this is to sub-class coordinate_model(),
which keeps track of the jacobian, parameter names, parameter estimation, etc. For one-off fitting use fit_model().
Various commonly used fit functions are collected here, primarily 1D, 2D, and 3D gaussians allowing for
arbitrary rotations.

All functions rely on fit_least_squares() which is a wrapper for scipy.optimize.least_squares()
which additionally handles fixing parameters and calculating standard uncertainties.
"""
import numpy as np
from scipy.optimize import least_squares
from localize_psf import affine


class coordinate_model():


    def __init__(self,
                 param_names: list[str],
                 ndims: int,
                 has_jacobian: bool = False):
        """
        @param param_names:
        @param has_jacobian:
        """

        if not isinstance(param_names, list):
            raise ValueError("param_names must be a list of strings")

        if not isinstance(ndims, int):
            raise ValueError("ndims must be an integer")

        if not isinstance(has_jacobian, bool):
            raise ValueError("has_jacobian must be a boolean")

        self.parameter_names = param_names
        self.nparams = len(param_names)
        self.ndims = ndims
        self.has_jacobian = has_jacobian


    def model(self,
              coordinates: tuple[np.ndarray],
              parameters: np.ndarray) -> np.ndarray:
        pass


    def jacobian(self,
                 coordinates: tuple[np.ndarray],
                 parameters: np.ndarray) -> list[np.ndarray]:
        """
        Return the jacobian matrix of the model evaluated at the given coordinates and parameters
        @param coordinates: (..., z, y, x) where all coordinate arrays must be broadcastable with each other
        @param parameters:
        @return jacobian: a list of ndarrays, where each entry in the list matches the size (after broadcasting)
        of all of the coordinate arrays
        """
        pass


    def test_jacobian(self,
                      coordinates: tuple[np.ndarray],
                      parameters: np.ndarray,
                      dp: float = 1e-7) -> (list[np.ndarray], list[np.ndarray]):
        """
        Test that the jacobian is implemented correctly by return both numerical and calculated values
        @param coordinates:
        @param parameters:
        @param dp:
        @return jac_numerical, jac_calc:
        """

        # numerical test for jacobian
        jac_calc = self.jacobian(coordinates, parameters)
        jac_numerical = []
        for ii in range(self.nparams):
            dp_now = np.zeros(self.nparams)
            dp_now[ii] = dp * 0.5

            jac_numerical.append((self.model(coordinates, parameters + dp_now) - self.model(coordinates, parameters - dp_now)) / dp)

        return jac_numerical, jac_calc


    def estimate_parameters(self,
                            data: np.ndarray,
                            coordinates: tuple[np.ndarray]):
        """
        Estimate model parameters from data
        @param data:
        @param coordinates: (..., z, y, x)
        @return estimated_parameters:
        """
        pass

    def estimate_bounds(self,
                        coordinates: tuple[np.ndarray]) -> (tuple[float], tuple[float]):
        """
        Estimate upper and lower bounds from the coordinates
        @param coordinates: (..., z, y, x)
        @return lbs, ubs:
        """
        lbs = (-np.inf,) * self.nparams
        ubs = (np.inf,) * self.nparams

        return lbs, ubs


    def normalize_parameters(self,
                             parameters) -> np.ndarray:
        """
        Return parameters in a standardized format, when there can be ambiguity. For example,
        a Gaussian model may include a standard deviation parameter. Since only the square of this quantity enters
        the model, a fit may return a negative value for standard deviation. In that case, this function
        would return the absolute value of the standard deviation
        @param params:
        @return normalized_params:
        """
        return parameters


    def fit(self,
            data: np.ndarray,
            coordinates: tuple[np.ndarray],
            init_params: list[float] = None,
            fixed_params: list[bool] = None,
            sd: np.ndarray = None,
            bounds: tuple[tuple[float]] = None,
            use_jacobian: bool = True,
            guess_bounds: bool = False,
            **kwargs) -> dict:
        """
        Fit model function to ND data. Any Nan values in the data will be ignored. This function is a wrapper
        for the non-linear least squares fit function scipy.optimize.least_squares() which additionally
        guessing parameters, guessing bounds, fixing parameters, and calculating fit uncertainty.

        :param data: data to be fit. nD array of arbitrary size
        :param coordinates: (..., zz, yy, xx) each of the coordinate arrays yy, xx, etc. must be broadcastable to the
        same shape as data
        :param init_params: [p1, p2, ..., pn]. If any entries in the list is None, that parameter will be estimated
        from the data. If None is passed instead of a list, try to guess all parameters from the data.
        :param fixed_params: list of boolean values, same size as init_params. If None, no parameters will be fixed.
        :param sd: uncertainty in parameters y. e.g. if experimental curves come from averages then should be the standard
        deviation of the mean. If None, then will use a value of 1 for all points. As long as these values are all the same
        they will not affect the optimization results, although they will affect the chi squared value reported.
        :param bounds: (lower bounds, upper bounds) where lower bounds = [lb0, lb1, ...,]
         and similar for ubs. If any bound, lbn, is None, this bound will be estimated from the data. If bounds is None
         then either all bounds will be guessed (if guess_bounds is True) or else -/+ infinity will be used for
          all parameters (i.e. no bounds)
        :param use_jacobian: force fit to ignore jacobian even if model has it
        :param guess_bounds: allow bounds to be guessed from input data
        :param kwargs: additional key word arguments will be passed through to scipy.optimize.least_squares()
        :return dict results:
        """

        to_use = np.logical_not(np.isnan(data))

        # set initial parameters that were set to None
        if np.any([ip is None for ip in init_params]):
            ip_default = self.estimate_parameters(data, coordinates)

            # set any parameters that were None to the default values
            for ii in range(len(init_params)):
                if init_params[ii] is None:
                    init_params[ii] = ip_default[ii]

        # if all sd's are nan or zero, set to 1
        if sd is None or np.all(np.isnan(sd)) or np.all(sd == 0):
            sd = np.ones(data.shape)

        # handle uncertainties that will cause fitting to fail
        if np.any(sd == 0) or np.any(np.isnan(sd)):
            sd[sd == 0] = np.nanmean(sd[sd != 0])
            sd[np.isnan(sd)] = np.nanmean(sd[sd != 0])

                # default bounds
        lbs_default, ubs_default = self.estimate_bounds(coordinates)

        if bounds is None:
            if guess_bounds:
                bounds = (lbs_default, ubs_default)
            else:
                bounds = ((-np.inf,) * self.nparams, (np.inf,) * self.nparams)
        else:
            if guess_bounds:
                lbs = tuple([b if b is not None else lbs_default[ii] for ii, b in enumerate(bounds[0])])
                ubs = tuple([b if b is not None else ubs_default[ii] for ii, b in enumerate(bounds[1])])
            else:
                lbs = tuple([b if b is not None else -np.inf for ii, b in enumerate(bounds[0])])
                ubs = tuple([b if b is not None else np.inf for ii, b in enumerate(bounds[1])])
            bounds = (lbs, ubs)


        # function to be optimized
        def err_fn(p):
            return np.divide(self.model(coordinates, p)[to_use].ravel() - data[to_use].ravel(), sd[to_use].ravel())

        # if it was passed, use model jacobian
        if self.has_jacobian and use_jacobian:
            def jac_fn(p):
                return [v[to_use] / sd[to_use] for v in self.jacobian(coordinates, p)]
        else:
            jac_fn = None

        results = fit_least_squares(err_fn,
                                    init_params,
                                    fixed_params=fixed_params,
                                    bounds=bounds,
                                    model_jacobian=jac_fn,
                                    **kwargs)

        return results


class gauss1dm(coordinate_model):
    def __init__(self):
        super().__init__(["amp", "center", "sigma", "bg"], 1, has_jacobian=True)
    def model(self,
              coordinates: tuple[np.ndarray],
              parameters: np.ndarray):
        x, = coordinates
        g = parameters[0] * np.exp(-(x - parameters[1]) ** 2 / (2 * parameters[2] ** 2)) + parameters[3]
        return g

    def jacobian(self,
                 coordinates: tuple[np.ndarray],
                 parameters: np.ndarray):

        x, = coordinates
        # useful functions that show up in derivatives
        exps = np.exp(-(x - parameters[1]) ** 2 / (2 * parameters[2] ** 2))

        jac = [exps,
               parameters[0] * exps * (x / parameters[2] ** 2 ),
               parameters[0] * exps * x ** 2 / parameters[2] ** 3,
               np.ones(x.shape),
               ]

        return jac

    def estimate_parameters(self,
                            data: np.ndarray,
                            coordinates: tuple[np.ndarray]):

        to_use = np.logical_not(np.isnan(data))

        bg = np.nanmean(data.ravel())
        amp = np.max(data[to_use].ravel()) - bg

        cx, = get_moments(data, order=1, coords=coordinates)
        m2x, = get_moments(data, order=2, coords=coordinates)
        sx = np.sqrt(m2x - cx ** 2)

        params = np.array([amp, cx, sx, bg])
        return params

    def estimate_bounds(self,
                        coordinates: tuple[np.ndarray]):
        x, = coordinates
        lbs = (-np.inf, x.min(), 0, -np.inf)
        ubs = (np.inf, x.max(), x.max() - x.min(), np.inf)
        return lbs, ubs

    def normalize_parameters(self,
                             parameters):
        param_norm = np.array(parameters, copy=True)
        param_norm[..., 2] = np.abs(param_norm[..., 2])

        return param_norm


class gauss2dm(coordinate_model):
    def __init__(self, use_sigma_ratio_parameterization=False):
        if use_sigma_ratio_parameterization:
            super().__init__(["amp", "cx", "cy", "sx", "sy/sx", "bg", "theta"], 2, has_jacobian=True)
        else:
            super().__init__(["amp", "cx", "cy", "sx", "sy", "bg", "theta"], 2, has_jacobian=True)
        self.use_sigma_ratio_parameterization = use_sigma_ratio_parameterization


    def model(self,
              coordinates: tuple[np.ndarray],
              parameters: np.ndarray):
        y, x = coordinates
        xrot = np.cos(parameters[6]) * (x - parameters[1]) - np.sin(parameters[6]) * (y - parameters[2])
        yrot = np.cos(parameters[6]) * (y - parameters[2]) + np.sin(parameters[6]) * (x - parameters[1])

        if self.use_sigma_ratio_parameterization:
            val = parameters[0] * np.exp(-xrot ** 2 / (2 * parameters[3] ** 2) - yrot ** 2 / (2 * parameters[3] ** 2 * parameters[4] ** 2)) + parameters[5]
        else:
            val = parameters[0] * np.exp(-xrot ** 2 / (2 * parameters[3] ** 2) - yrot ** 2 / (2 * parameters[4] ** 2)) + parameters[5]

        return val


    def jacobian(self,
                 coordinates: tuple[np.ndarray],
                 params: np.ndarray):

        y, x = coordinates
        bcast_shape = (x + y).shape

        # useful functions that show up in derivatives
        xrot = np.cos(params[6]) * (x - params[1]) - np.sin(params[6]) * (y - params[2])
        yrot = np.cos(params[6]) * (y - params[2]) + np.sin(params[6]) * (x - params[1])

        if self.use_sigma_ratio_parameterization:
            exps = np.exp(-xrot ** 2 / (2 * params[3] ** 2) - yrot ** 2 / (2 * params[3] ** 2 * params[4] ** 2))

            jac = [exps,
                   params[0] * exps * (xrot / params[3] ** 2 * np.cos(params[6]) + yrot / params[3] ** 2 / params[4] ** 2 * np.sin(params[6])),
                   params[0] * exps * (yrot / params[3] ** 2 / params[4] ** 2 * np.cos(params[6]) - xrot / params[3] ** 2 * np.sin(params[6])),
                   params[0] * exps * (xrot ** 2 / params[3] ** 3 + yrot ** 2 / params[3] ** 3 / params[4] ** 2),
                   params[0] * exps * yrot ** 2 / params[3] ** 2 / params[4] ** 3,
                   np.ones(bcast_shape),
                   params[0] * exps * xrot * yrot * (1 / params[3] ** 2 - 1 / (params[3] ** 2 * params[4] ** 2))]
        else:
            exps = np.exp(-xrot ** 2 / (2 * params[3] ** 2) - yrot ** 2 / (2 * params[4] ** 2))

            jac = [exps,
                   params[0] * exps * (xrot / params[3] ** 2 * np.cos(params[6]) + yrot / params[4] ** 2 * np.sin(params[6])),
                   params[0] * exps * (yrot / params[4] ** 2 * np.cos(params[6]) - xrot / params[3] ** 2 * np.sin(params[6])),
                   params[0] * exps * xrot ** 2 / params[3] ** 3,
                   params[0] * exps * yrot ** 2 / params[4] ** 3,
                   np.ones(bcast_shape),
                   params[0] * exps * xrot * yrot * (1 / params[3] ** 2 - 1 / params[4] ** 2)]

        return jac

    def estimate_parameters(self,
                            data: np.ndarray,
                            coordinates: tuple[np.ndarray]):

        to_use = np.logical_not(np.isnan(data))

        bg = np.nanmean(data.ravel())
        amp = np.max(data[to_use].ravel()) - bg

        cy, cx = get_moments(data, order=1, coords=coordinates)
        m2y, m2x = get_moments(data, order=2, coords=coordinates)
        with np.errstate(invalid='ignore'):
            sx = np.sqrt(m2x - cx ** 2)
            sy = np.sqrt(m2y - cy ** 2)

        if self.use_sigma_ratio_parameterization:
            ip_default = np.array([amp, cx, cy, sx, sy / sx, bg, 0])
        else:
            ip_default = np.array([amp, cx, cy, sx, sy, bg, 0])

        return ip_default


    def estimate_bounds(self,
                        coordinates: tuple[np.ndarray]):

        yy, xx = coordinates
        # replace any bounds which are none with default guesses
        lbs = (-np.inf, xx.min(), yy.min(), 0, 0, -np.inf, -np.inf)
        if self.use_sigma_ratio_parameterization:
            ubs = (np.inf, xx.max(), yy.max(), xx.max() - xx.min(), np.inf, np.inf, np.inf)
        else:
            ubs = (np.inf, xx.max(), yy.max(), xx.max() - xx.min(), yy.max() - yy.min(), np.inf, np.inf)

        return lbs, ubs


    def normalize_parameters(self,
                             parameters):
        param_norm = np.array(parameters, copy=True)
        param_norm[..., 3] = np.abs(param_norm[..., 3])
        param_norm[..., 4] = np.abs(param_norm[..., 4])
        param_norm[..., 6] = np.mod(param_norm[..., 6], 2*np.pi)

        return param_norm


class gauss2d_sum(coordinate_model):
    def __init__(self, ngaussians):
        param_names = ["amp", "cx", "cy", "sx", "sy/sx", "theta"] * ngaussians + ["bg"]
        super().__init__(param_names, 2, has_jacobian=True)
        self.ngaussians = int(ngaussians)

    def model(self,
              coordinates: tuple[np.ndarray],
              p: np.ndarray):

        # todo: need to check if this is implemented correctly
        val = 0
        for ii in range(self.ngaussians - 1):
            ps = np.concatenate((np.array(p[6*ii: 6*ii + 5]), np.array([0]), np.atleast_1d([p[ii * 6 + 5]])))
            val += gauss2dm().model(coordinates, ps)

        # deal with last gaussian, which also gets background term
        ps = np.concatenate((np.array(p[-7:-2]), np.atleast_1d(p[-1]), np.atleast_1d(p[-2])))
        val += gauss2dm().model(coordinates, ps)

        return val


    def jacobian(self,
                 coordinates: tuple[np.ndarray],
                 p: np.ndarray):

        jac_list = []
        for ii in range(self.ngaussians - 1):
            ps = np.concatenate((np.array(p[6 * ii: 6 * ii + 5]), np.array([0]), np.atleast_1d([p[ii * 6 + 5]])))
            jac_current = gauss2dm().jacobian(coordinates, ps)
            jac_list += jac_current[:-2] + [jac_current[-1]]

        # deal with last gaussian, which also gets background term
        ps = np.concatenate((np.array(p[-7:-2]), np.atleast_1d(p[-1]), np.atleast_1d(p[-2])))
        jac_current = gauss2dm().jacobian(coordinates, ps)
        jac_list += jac_current[:-2] + [jac_current[-1]] + [jac_current[-2]]

        return jac_list


    def normalize_parameters(self,
                             parameters):

        param_norm = np.array(parameters, copy=True)
        param_norm[..., 3::6] = np.abs(param_norm[..., 3::6])
        param_norm[..., 4::6] = np.abs(param_norm[..., 4::6])
        param_norm[..., 6::6] = np.mod(param_norm[..., 6::6], 2*np.pi)

        return param_norm


class gauss3d(coordinate_model):
    def __init__(self):
        """
        3D gaussian symmetric in xy
        """
        super().__init__(["amp", "cx", "cy", "cz", "sxy", "sz", "bg"], 3, has_jacobian=True)


    def model(self,
              coordinates: tuple[np.ndarray],
              params: np.ndarray) -> np.ndarray:

        z, y, x, = coordinates

        # calculate psf at oversampled points
        val = params[0] * np.exp(-(x - params[1]) ** 2 / 2 / params[4] ** 2
                                 -(y - params[2]) ** 2 / 2 / params[4] ** 2
                                 -(z - params[3]) ** 2 / 2 / params[5] ** 2
                                ) + params[6]

        return val


    def jacobian(self,
                 coordinates: tuple[np.ndarray],
                 params: np.ndarray) -> list[np.ndarray]:

        z, y, x = coordinates
        bcast_shape = (x + y + z).shape

        # use sxy * |sxy| instead of sxy**2 to enforce sxy > 0
        v = np.exp(-(x - params[1]) ** 2 / 2 / params[4] ** 2
                   - (y - params[2]) ** 2 / 2 / params[4] ** 2
                   - (z - params[3]) ** 2 / 2 / params[5] ** 2
                  )

        # [A, cx, cy, cz, sxy, sz, bg]
        jac = [v,
               params[0] * 2 * (x - params[1]) / 2 / params[4] ** 2 * v,
               params[0] * 2 * (y - params[2]) / 2 / params[4] ** 2 * v,
               params[0] * 2 * (z - params[3]) / 2 / params[5] ** 2 * v,
               params[0] * (2 / params[4] ** 3 * (x - params[1]) ** 2 / 2 +
                            2 / params[4] ** 3 * (y - params[2]) ** 2 / 2) * v,
               params[0] * 2 / params[5] ** 3 * (z - params[3]) ** 2 / 2 * v,
               np.ones(bcast_shape)]

        return jac


    def estimate_parameters(self,
                            data: np.ndarray,
                            coordinates: tuple[np.ndarray]):

        z, y, x = coordinates

        # subtract smallest value so positive
        img_temp = data - np.nanmin(data)
        to_use = np.logical_and(np.logical_not(np.isnan(img_temp)), img_temp > 0)

        if data.ndim != len(coordinates):
            raise ValueError("len(coords) != img.ndim")

        # compute moments
        c1s = np.zeros(data.ndim)
        c2s = np.zeros(data.ndim)
        for ii in range(data.ndim):
            c1s[ii] = np.sum(img_temp[to_use] * coordinates[ii][to_use]) / np.sum(img_temp[to_use])
            c2s[ii] = np.sum(img_temp[to_use] * coordinates[ii][to_use] ** 2) / np.sum(img_temp[to_use])

        sigmas = np.sqrt(c2s - c1s ** 2)
        sz = sigmas[0]
        sxy = np.mean(sigmas[1:])

        guess_params = np.concatenate((np.array([np.nanmax(data) - np.nanmin(data)]),
                                       np.flip(c1s),
                                       np.array([sxy]),
                                       np.array([sz]),
                                       np.array([np.nanmean(data)])
                                       ),
                                      )

        return self.normalize_parameters(guess_params)


    def estimate_bounds(self,
                        coordinates: tuple[np.ndarray]):
        z, y, x = coordinates

        lbs = (-np.inf, x.min(), y.min(), z.min(), 0, 0, -np.inf)
        ubs = (np.inf, x.max(), y.max(), z.max(), np.inf, np.inf, np.inf)
        return lbs, ubs

    def normalize_parameters(self,
                             params: np.ndarray):
        norm_params = np.array(params, copy=True)
        norm_params[..., 4] = np.abs(norm_params[..., 4])
        norm_params[..., 5] = np.abs(norm_params[..., 5])

        return norm_params


class asymmetric_gaussian3d(coordinate_model):
    def __init__(self):
        super().__init__(["A", "cx", "cy", "cz", "sx", "sy/sx", "sz", "theta_xy", "bg"],
                         3, has_jacobian=True)


    def model(self,
              coords: tuple[np.ndarray],
              params: np.ndarray):
        z, y, x, = coords

        # rotated coordinates
        xx_rot = np.cos(params[7]) * (x - params[1]) - np.sin(params[7]) * (y - params[2])
        yy_rot = np.cos(params[7]) * (y - params[2]) + np.sin(params[7]) * (x - params[1])

        vals = params[0] * np.exp(-xx_rot ** 2 / 2 / params[4] ** 2
                                  - yy_rot ** 2 / 2 / (params[4] * params[5]) ** 2
                                  - (z - params[3]) ** 2 / 2 / params[6] ** 2) + params[8]

        return vals


    def jacobian(self,
                 coords: tuple[np.ndarray],
                 params: np.ndarray):

        z, y, x, = coords
        bcast_shape = (x + y + z).shape

        p = params
        # rotated coordinates
        xx_rot = np.cos(p[7]) * (x - p[1]) - np.sin(p[7]) * (y - p[2])
        yy_rot = np.cos(p[7]) * (y - p[2]) + np.sin(p[7]) * (x - p[1])

        dx_dphi = -np.sin(p[7]) * (x - p[1]) - np.cos(p[7]) * (y - p[2])
        dy_dphi = -np.sin(p[7]) * (y - p[2]) + np.cos(p[7]) * (x - p[1])

        # calculate psf at oversampled points
        val0  = np.exp(-xx_rot ** 2 / 2 / p[4] ** 2
                       -yy_rot ** 2 / 2 / (p[4] * p[5]) ** 2
                       -(z - p[3]) ** 2 / 2 / p[6] ** 2)

        dpsf_dcx = 2 * xx_rot * np.cos(p[7]) / 2 / p[4]**2 + \
                   2 * yy_rot * np.sin(p[7]) / 2 / (p[4] * p[5])**2

        dpsf_dcy = -2 * xx_rot * np.sin(p[7]) / 2 / p[4] ** 2 + \
                    2 * yy_rot * np.cos(p[7]) / 2 / (p[4] * p[5]) ** 2

        dpsf_dcz = 2 * (z - p[3]) / 2 / p[6] ** 2

        dpsf_dsx = 2 / p[4] ** 3 * xx_rot ** 2 / 2 + \
                   2 / p[4] ** 3 * yy_rot ** 2 / 2 / p[5] ** 2

        dpsf_dsrat = 2 / p[5] ** 3 * yy_rot**2 / 2 / p[4] ** 2

        dpsf_dsz = 2 / p[6] ** 3 * (z - p[3]) ** 2 / 2

        dpsf_dtheta = 2 * xx_rot / 2 / p[4] ** 2 * dx_dphi + \
                      2 * yy_rot / 2 / (p[4] * p[5]) ** 2 * dy_dphi

        jac = [val0,  # A
               p[0] * dpsf_dcx * val0,  # cx
               p[0] * dpsf_dcy * val0,  # cy
               p[0] * dpsf_dcz * val0,  # cz
               p[0] * dpsf_dsx * val0,  # sx
               p[0] * dpsf_dsrat * val0,  # sy/sx
               p[0] * dpsf_dsz * val0,  # sz
               p[0] * dpsf_dtheta * val0, # theta
               np.ones(bcast_shape)  # bg
               ]

        return jac


    def estimate_parameters(self,
                            img: np.ndarray,
                            coords: tuple[np.ndarray]):
        z, y, x = coords

        # subtract smallest value so positive
        img_temp = img
        to_use = np.logical_and(np.logical_not(np.isnan(img_temp)), img_temp > 0)

        if img.ndim != len(coords):
            raise ValueError("len(coords) != img.ndim")

        # compute moments
        c1s = np.zeros(img.ndim)
        c2s = np.zeros(img.ndim)
        for ii in range(img.ndim):
            c1s[ii] = np.sum(img_temp[to_use] * coords[ii][to_use]) / np.sum(img_temp[to_use])
            c2s[ii] = np.sum(img_temp[to_use] * coords[ii][to_use] ** 2) / np.sum(img_temp[to_use])

        sigmas = np.sqrt(c2s - c1s ** 2)

        guess_params = np.concatenate((np.array([np.nanmax(img) - np.nanmean(img)]),
                                       np.flip(c1s),
                                       np.flip(sigmas),
                                       np.array([0.]),
                                       np.array([np.nanmean(img)])
                                       ),
                                      )

        return self.normalize_parameters(guess_params)


    def estimate_bounds(self,
                        coordinates: tuple[np.ndarray]) -> (tuple[float], tuple[float]):
        z, y, x = coordinates

        lbs = (-np.inf, x.min(), y.min(), z.min(), 0, 0, 0, -np.inf, -np.inf)
        ubs = (np.inf, x.max(), y.max(), z.max(), np.inf, np.inf, np.inf, np.inf, np.inf)
        return lbs, ubs


    def normalize_parameters(self, params):
        norm_params = np.array(params, copy=True)
        norm_params[..., 4] = np.abs(norm_params[..., 4])
        norm_params[..., 5] = np.abs(norm_params[..., 5])
        norm_params[..., 6] = np.abs(norm_params[..., 6])

        return norm_params


class gauss3d_rotated(coordinate_model):

    def __init__(self):
        """
        3D gaussian, with arbitrary rotation parameterized by Euler angles

        r_body = U_z(psi)^-1 U_y(theta)^-1 U_z(phi)^-1 * r_lab
        U_z(phi)^-1 = [[cos(phi), -sin(phi), 0], [sin(phi), cos(phi), 0], [0, 0, 1]]
        f_rot(r_lab) = f(r_body) = f(U^{-1} * r_lab)

        Take the z-axis in the frame of the object, and consider the z-axis in the lab frame. phi and theta describe
        how the transformation to overlap these two. psi gives the angle the object is rotated about its own axis

        :param p: [A, cx, cy, cz, sigma x_rot, sigma y_rot, sigma z_rot, bg, phi, theta, psi]
        :return value:
        """
        super().__init__(["amp", "cx", "cy", "cz", "sx", "sy", "sz", "bg", "phi", "theta", "psi"], 3, has_jacobian=True)


    def model(self,
              coordinates: tuple[np.ndarray],
              params: np.ndarray):

        z, y, x, = coordinates

        phi = params[8]
        theta = params[9]
        psi = params[10]
        rot_mat = affine.euler_mat_inv(phi, theta, psi)
        xrot = (x - params[1]) * rot_mat[0, 0] + (y - params[2]) * rot_mat[0, 1] + (z - params[3]) * rot_mat[0, 2]
        yrot = (x - params[1]) * rot_mat[1, 0] + (y - params[2]) * rot_mat[1, 1] + (z - params[3]) * rot_mat[1, 2]
        zrot = (x - params[1]) * rot_mat[2, 0] + (y - params[2]) * rot_mat[2, 1] + (z - params[3]) * rot_mat[2, 2]
        val = params[0] * np.exp(-xrot ** 2 / (2 * params[4] ** 2) - yrot ** 2 / (2 * params[5] ** 2) - zrot ** 2 / (2 * params[6] ** 2)) + \
              params[7]

        return val


    def jacobian(self,
                 coordinates: tuple[np.ndarray],
                 parameters: np.ndarray):

        z, y, x = coordinates
        bcast_shape = (x + y + z).shape

        p = parameters
        phi = p[8]
        theta = p[9]
        psi = p[10]

        rot_mat = affine.euler_mat_inv(phi, theta, psi)
        dphi, dtheta, dpsi = affine.euler_mat_inv_derivatives(phi, theta, psi)

        xrot = (x - p[1]) * rot_mat[0, 0] + (y - p[2]) * rot_mat[0, 1] + (z - p[3]) * rot_mat[0, 2]
        yrot = (x - p[1]) * rot_mat[1, 0] + (y - p[2]) * rot_mat[1, 1] + (z - p[3]) * rot_mat[1, 2]
        zrot = (x - p[1]) * rot_mat[2, 0] + (y - p[2]) * rot_mat[2, 1] + (z - p[3]) * rot_mat[2, 2]
        exp = np.exp(-xrot ** 2 / (2 * p[4] ** 2) - yrot ** 2 / (2 * p[5] ** 2) - zrot ** 2 / (2 * p[6] ** 2))

        jac = [exp,
               p[0] * exp * (xrot / p[4] ** 2 * rot_mat[0, 0] + yrot / p[5] ** 2 * rot_mat[1, 0] + zrot / p[6] ** 2 *
                             rot_mat[2, 0]),
               p[0] * exp * (xrot / p[4] ** 2 * rot_mat[0, 1] + yrot / p[5] ** 2 * rot_mat[1, 1] + zrot / p[6] ** 2 *
                             rot_mat[2, 1]),
               p[0] * exp * (xrot / p[4] ** 2 * rot_mat[0, 2] + yrot / p[5] ** 2 * rot_mat[1, 2] + zrot / p[6] ** 2 *
                             rot_mat[2, 2]),
               p[0] * exp * xrot ** 2 / p[4] ** 3,
               p[0] * exp * yrot ** 2 / p[5] ** 3,
               p[0] * exp * zrot ** 2 / p[6] ** 3,
               np.ones(bcast_shape),
               -p[0] * exp * (
                       (xrot / p[4] ** 2) * (
                           (x - p[1]) * dphi[0, 0] + (y - p[2]) * dphi[0, 1] + (z - p[3]) * dphi[0, 2]) +
                       (yrot / p[5] ** 2) * (
                                   (x - p[1]) * dphi[1, 0] + (y - p[2]) * dphi[1, 1] + (z - p[3]) * dphi[1, 2]) +
                       (zrot / p[6] ** 2) * (
                                   (x - p[1]) * dphi[2, 0] + (y - p[2]) * dphi[2, 1] + (z - p[3]) * dphi[2, 2])),
               -p[0] * exp * (
                       (xrot / p[4] ** 2) * (
                           (x - p[1]) * dtheta[0, 0] + (y - p[2]) * dtheta[0, 1] + (z - p[3]) * dtheta[0, 2]) +
                       (yrot / p[5] ** 2) * (
                                   (x - p[1]) * dtheta[1, 0] + (y - p[2]) * dtheta[1, 1] + (z - p[3]) * dtheta[1, 2]) +
                       (zrot / p[6] ** 2) * (
                                   (x - p[1]) * dtheta[2, 0] + (y - p[2]) * dtheta[2, 1] + (z - p[3]) * dtheta[2, 2])),
               -p[0] * exp * (
                       (xrot / p[4] ** 2) * (
                           (x - p[1]) * dpsi[0, 0] + (y - p[2]) * dpsi[0, 1] + (z - p[3]) * dpsi[0, 2]) +
                       (yrot / p[5] ** 2) * (
                                   (x - p[1]) * dpsi[1, 0] + (y - p[2]) * dpsi[1, 1] + (z - p[3]) * dpsi[1, 2]) +
                       (zrot / p[6] ** 2) * (
                                   (x - p[1]) * dpsi[2, 0] + (y - p[2]) * dpsi[2, 1] + (z - p[3]) * dpsi[2, 2]))
               ]

        return jac


    def estimate_parameters(self,
                            data: np.ndarray,
                            coordinates: tuple[np.ndarray]):
        z, y, x = coordinates

        # subtract smallest value so positive
        img_temp = data
        to_use = np.logical_and(np.logical_not(np.isnan(img_temp)), img_temp > 0)

        if data.ndim != len(coordinates):
            raise ValueError("len(coords) != img.ndim")

        # compute moments
        c1s = np.zeros(data.ndim)
        c2s = np.zeros(data.ndim)
        for ii in range(data.ndim):
            c1s[ii] = np.sum(img_temp[to_use] * coordinates[ii][to_use]) / np.sum(img_temp[to_use])
            c2s[ii] = np.sum(img_temp[to_use] * coordinates[ii][to_use] ** 2) / np.sum(img_temp[to_use])

        sigmas = np.sqrt(c2s - c1s ** 2)

        guess_params = np.concatenate((np.array([np.nanmax(data) - np.nanmean(data)]),
                                       np.flip(c1s),
                                       np.flip(sigmas),
                                       np.array([np.nanmean(data)]),
                                       np.array([0.]),
                                       np.array([0.]),
                                       np.array([0.]),
                                       ),
                                      )

        return self.normalize_parameters(guess_params)


    def estimate_bounds(self,
                        coordinates: tuple[np.ndarray]):
        z, y, x = coordinates

        lbs = (-np.inf, x.min(), y.min(), z.min(), 0, 0, 0, -np.inf, -np.inf, -np.inf, -np.inf)
        ubs = (np.inf, x.max(), y.max(), z.max(), np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf)
        return lbs, ubs


    def normalize_parameters(self,
                             parameters):
        param_norm = np.array(parameters, copy=True)
        # normalize sigmas
        param_norm[..., 4] = np.abs(param_norm[..., 4])
        param_norm[..., 5] = np.abs(param_norm[..., 5])
        param_norm[..., 6] = np.abs(param_norm[..., 6])
        # normalize angles
        # param_norm[..., 8] = np.mod(param_norm[..., 8], 2*np.pi)
        # param_norm[..., 9] = np.mod(param_norm[..., 9], 2*np.pi)
        # param_norm[..., 10] = np.mode(param_norm[..., 10], 2*np.pi)

        return param_norm


class line_piecewisem(coordinate_model):

    def __init__(self):
        super().__init__(["slope1", "y intercept1", "slope2", "crossover"], 1, has_jacobian=True)

    def model(self,
              coordinates: tuple[np.ndarray],
              parameters: np.ndarray) -> np.ndarray:
        """
        Two piecewise lines which connect at a point
        @param x: x-positions to evaluate function
        @param p: [slope 1, y-intercept 1, slope 2, changover point]
        @return value:
        """

        x, = coordinates
        p = parameters

        # first part of the line
        l1 = p[0] * x + p[1]

        # second part of the line
        # l1(p[3]) = l2(p[3])
        b2 = (p[0] - p[2]) * p[3] + p[1]
        l2 = p[2] * x + b2

        # full line
        line = l1
        line[x >= p[3]] = l2[x >= p[3]]

        return line

    def jacobian(self,
                 coordinates: tuple[np.ndarray],
                 parameters: np.ndarray) -> list[np.ndarray]:

        x, = coordinates
        p = parameters

        on_line2 = (x >= p[3])

        # slope 1
        dl_ds1 = x
        dl_ds1[on_line2] = p[3] * np.ones(x[on_line2].shape)

        # offset
        dl_do1 = np.ones(x.shape)

        # slope 2
        dl_ds2 = np.zeros(x.shape)
        dl_ds2[on_line2] = -p[3] * np.ones(x[on_line2].shape) + x[on_line2]

        # crossover
        dl_dc = np.zeros(x.shape)

        jac = [dl_ds1, dl_do1, dl_ds1, dl_dc]

        return jac



    def estimate_parameters(self,
                            data: np.ndarray,
                            coordinates: tuple[np.ndarray]):
        pass


def fit_model(img: np.ndarray,
              model_fn,
              init_params: list[float],
              fixed_params: list[bool] = None,
              sd: np.ndarray = None,
              bounds: tuple[tuple[float]] = None,
              model_jacobian = None,
              **kwargs) -> dict:
    """
    Fit 2D model function to an image. Any Nan values in the image will be ignored. This function is a wrapper for
    for the non-linear least squares fit function scipy.optimize.least_squares() which additionally handles fixing
    parameters and calculating fit uncertainty.

    This function is kep for convenience when working with one-off models. Preferred method is to subclass
    coordinate_model()

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


    has_jacobian = model_jacobian is not None

    class mtemp(coordinate_model):
        def __init__(self):
            super().__init__(["p"] * len(init_params), 0, has_jacobian=has_jacobian)

        def model(self,
                  coordinates: tuple[np.ndarray],
                  parameters: np.ndarray) -> np.ndarray:
            return model_fn(parameters)

        def jacobian(self,
                 coordinates: tuple[np.ndarray],
                 parameters: np.ndarray) -> list[np.ndarray]:
            return model_jacobian(parameters)


    results = mtemp().fit(img,
                          None,
                          init_params=init_params,
                          fixed_params=fixed_params,
                          sd=sd,
                          bounds=bounds,
                          guess_bounds=False,
                          **kwargs
                          )

    return results


def fit_least_squares(model_fn,
                      init_params: list[float],
                      fixed_params: list[bool] = None,
                      bounds: tuple[tuple[float]] = None,
                      model_jacobian = None,
                      **kwargs) -> dict:
    """
    Wrapper for non-linear least squares fit function scipy.optimize.least_squares which handles fixing parameters
    and calculating fit uncertainty.

    :param model_fn: function of model parameters p which returns an array, where the sum of squares of this array is
    minimized. e.g. if we have a set of data points x_i and we make measurements y_i with uncertainties sigma_i,
    and we have a model m(p, x_i)
     then f(p) = [(m(p, x_i) - y_i) / sigma_i]
    :param init_params: p = [p1, p2, ..., pn]
    :param fixed_params: list of boolean values, same size as init_params. If None,
     no parameters will be fixed.
    :param  bounds: (lbs, ubs). If None, -/+ infinity used for all parameters.
    :param model_jacobian: Jacobian of the model function as a list, [df/dp[0], df/dp[1], ...]. If None,
     no jacobian used.
    :param kwargs: additional key word arguments will be passed through to scipy.optimize.least_squares()

    :return results: dictionary object. Uncertainty can be obtained from the square rootsof the diagonals of the
     covariance matrix, but these will only be meaningful if variances were appropriately provided for the cost function
    """

    # ###########################
    # check input parameters
    # ###########################

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

    # ###########################
    # if some parameters are fixed, we need to hide them from the fit function to produce correct covariance, etc.
    # Idea: map the "reduced" (i.e. not fixed) parameters onto the full parameter list.
    # do this by looking at each parameter. If it is supposed to be "fixed" substitute the initial parameter. If not,
    # then get the next value from pfree. We find the right index of pfree by summing the number of previously unfixed parameters
    # ###########################
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

    # ###########################
    # non-linear least squares fit
    # ###########################
    if model_jacobian is None:
        fit_info = least_squares(err_fn_pfree,
                                 init_params_free,
                                 bounds=bounds_free,
                                 **kwargs)
    else:
        fit_info = least_squares(err_fn_pfree,
                                 init_params_free,
                                 bounds=bounds_free,
                                 jac=jac_fn_free,
                                 x_scale='jac',
                                 **kwargs)
    pfit = pfree2pfull(fit_info['x'])

    # ###########################
    # calculate chi squared
    # ###########################
    nfree_params = np.sum(np.logical_not(fixed_params))
    # scipy.optimize.least_squares minimizes s = 0.5 * \sum |fn(x_i)|^2, so need a factor of two to correct their cost
    red_chi_sq = (2 * fit_info["cost"]) / (model_fn(init_params).size - nfree_params)

    # ###########################
    # calculate covariance matrix
    # ###########################
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

    # ###########################
    # store results
    # ###########################
    status_codes = {"improper input parameters status returned from MINPACK": -1,
                    "the maximum number of function evaluations is exceeded": 0,
                    "gtol termination condition is satisfied": 1,
                    "ftol termination condition is satisfied": 2,
                    "xtol termination condition is satisfied": 3,
                    "Both ftol and xtol termination conditions are satisfied": 4}

    result = {'fit_params': pfit,
              'chi_squared': red_chi_sq,
              'covariance': cov,
              'init_params': init_params,
              'fixed_params': fixed_params,
              'bounds': bounds,
              'cost': fit_info['cost'],
              'optimality': fit_info['optimality'],
              'nfev': fit_info['nfev'],
              'njev': fit_info['njev'],
              'status': fit_info['status'],
              'status_codes': status_codes,
              'success': fit_info['success'],
              'message': fit_info['message']}

    return result


def get_moments(img: np.ndarray,
                order: int = 1,
                coords: tuple[np.ndarray] = None,
                dims: list[int] = None) -> list[float]:
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
    moments = [np.nansum(img * c ** order, axis=tuple(dims), dtype=float) / w for c in coords]

    return moments


# todo: convert these to class models
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
