"""
Tools for fitting data using non-linear least squares. The recommended way to do this is to sub-class coordinate_model(),
which keeps track of the jacobian, parameter names, parameter estimation, etc. For one-off fitting use fit_model().
Various commonly used fit functions are collected here, primarily 1D, 2D, and 3D gaussians allowing for
arbitrary rotations.

All functions rely on fit_least_squares() which is a wrapper for scipy.optimize.least_squares()
which additionally handles fixing parameters and calculating standard uncertainties.
"""
from typing import Optional, Sequence
import numpy as np
from scipy.optimize import least_squares
from localize_psf import affine


class coordinate_model():
    """
    Basic model for dealing with functions of coordinates. Coordinates are given as tuples (c0, c1, ..., cn)
    e.g. for 3D models (z, y, x) where ci are broadcastable to the same shape. m-dimensional models
    should accept n-dimensional data and ignore all but the last m-dimensions
    """

    def __init__(self,
                 param_names: list[str],
                 ndims: int,
                 has_jacobian: bool = False):
        """

        :param param_names:
        :param has_jacobian:
        """

        if not isinstance(param_names, list):
            raise ValueError("param_names must be a list of strings")

        if not isinstance(ndims, int):
            raise ValueError("ndims must be an integer")

        if not isinstance(has_jacobian, bool):
            raise ValueError("has_jacobian must be a boolean")

        self.parameter_names = param_names
        self.nparams = len(param_names)
        self.ndim = ndims
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

        :param coordinates: (..., z, y, x) where all coordinate arrays must be broadcastable with each other
        :param parameters:
        :return jacobian: a list of ndarrays, where each entry in the list matches the size (after broadcasting)
          of all of the coordinate arrays
        """
        pass


    def test_jacobian(self,
                      coordinates: tuple[np.ndarray],
                      parameters: np.ndarray,
                      dp: float = 1e-7) -> (list[np.ndarray], list[np.ndarray]):
        """
        Test that the jacobian is implemented correctly by return both numerical and calculated values

        :param coordinates:
        :param parameters:
        :param dp:
        :return jac_numerical, jac_calc:
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
                            coordinates: tuple[np.ndarray],
                            num_preserved_dims: int = 0) -> np.ndarray:
        """
        Estimate model parameters from data. This function should support coordinate arrays being broadcastable
        with data (but not actually the same size)

        To process many data sets in parallel,

        :param data: array of arbitrary shape
        :param coordinates: (..., z, y, x) where each coordinate should match the shape of data
        :param num_preserved_dims: number of dimensions to preserve during parameter estimation. Setting this
        to a non-zero value allows parameter estimation for multiple different data sets to happen
        simultaneously
        :return estimated_parameters:
        """
        pass

    def estimate_bounds(self,
                        coordinates: tuple[np.ndarray]) -> (tuple[float], tuple[float]):
        """
        Estimate upper and lower bounds from the coordinates

        :param coordinates: (..., z, y, x)
        :return lbs, ubs:
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

        This method should accept an array of parameters of size nfits x nparams and normalize them all

        :param params:
        :return normalized_params:
        """
        return parameters


    def fit(self,
            data: np.ndarray,
            coordinates: tuple[np.ndarray],
            init_params: Optional[list[float]] = None,
            fixed_params: Optional[list[bool]] = None,
            sd: Optional[np.ndarray] = None,
            bounds: Optional[tuple[tuple[float]]] = None,
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
        :return results:
        """

        to_use = np.logical_not(np.isnan(data))

        # set initial parameters that were set to None
        if init_params is None:
            init_params = self.estimate_parameters(data, coordinates)
        elif np.any([ip is None for ip in init_params]):
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
        # todo: handle complex functions
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


class rotated_model_2d(coordinate_model):
    """
    Take any 2D model and parameterize its arbitrary rotation using angle theta

    This is a helper function to avoid needing to write rotation code more than once
    """

    def __init__(self,
                 model: coordinate_model,
                 center_inds: tuple[int]):
        """

        :param model:
        :param center_inds: (cy_index, cx_index)
        """

        param_names = model.parameter_names + ["phi"]
        has_jacobian = model.has_jacobian
        ndims = model.ndim

        if model.ndim != 2:
            raise ValueError(f"model.ndim = {model.ndim:d}, but only 2D models are supported")

        super().__init__(param_names,
                         has_jacobian=has_jacobian,
                         ndims=ndims)

        self.base_model = model
        self.center_inds = center_inds

        # copy any attributes that don't overlap
        for k, v in model.__dict__.items():
            if not hasattr(self, k):
                setattr(self, k, v)

    def model(self,
              coordinates: tuple[np.ndarray],
              parameters: np.ndarray) -> np.ndarray:
        y, x = coordinates[-2:]

        phi = parameters[-1]
        rot_mat = affine.euler_mat_inv(phi, 0, 0)[:2, :2]

        cx = parameters[self.center_inds[1]]
        cy = parameters[self.center_inds[0]]

        # rotated coordinates
        xrot = (x - cx) * rot_mat[0, 0] + (y - cy) * rot_mat[0, 1]
        yrot = (x - cx) * rot_mat[1, 0] + (y - cy) * rot_mat[1, 1]

        # evaluate base model at rotated coordinates
        params_base = np.array(parameters[:-1], copy=True)
        params_base[self.center_inds[1]] = 0
        params_base[self.center_inds[0]] = 0

        return self.base_model.model((yrot, xrot), params_base)

    def jacobian(self,
                 coordinates: tuple[np.ndarray],
                 parameters: np.ndarray) -> list[np.ndarray]:
        y, x = coordinates[-2:]

        phi = parameters[-1]
        rot_mat = affine.euler_mat_inv(phi, 0, 0)[:2, :2]

        cx = parameters[self.center_inds[1]]
        cy = parameters[self.center_inds[0]]

        # rotated coordinates
        xrot = (x - cx) * rot_mat[0, 0] + (y - cy) * rot_mat[0, 1]
        yrot = (x - cx) * rot_mat[1, 0] + (y - cy) * rot_mat[1, 1]

        dxrot_dcx = -rot_mat[0, 0]
        dxrot_dcy = -rot_mat[0, 1]

        dyrot_dcx = -rot_mat[1, 0]
        dyrot_dcy = -rot_mat[1, 1]

        # evaluate base model at rotated coordinates
        params_base = np.array(parameters[:-1], copy=True)
        params_base[self.center_inds[1]] = 0.
        params_base[self.center_inds[0]] = 0.
        jac_base = self.base_model.jacobian((yrot, xrot), params_base)

        # need to correct jacobian with
        # (1) derivatives of rotated coordinates wrt centers
        # (2) derivatives wrt Euler angles
        j_cx = np.array(jac_base[self.center_inds[1]], copy=True)
        j_cy = np.array(jac_base[self.center_inds[0]], copy=True)

        # need negative sign, because thinking the j_cx and etc. terms as actually taking derivative wrt x, y, z coords
        # since these enter in the same way as cx, cy, cz, but with opposite sign
        jac_base[self.center_inds[1]] = -(j_cx * dxrot_dcx + j_cy * dyrot_dcx)
        jac_base[self.center_inds[0]] = -(j_cx * dxrot_dcy + j_cy * dyrot_dcy)

        # euler angle derivatives
        dphi = affine.euler_mat_inv_derivatives(phi, 0, 0)[0][:2, :2]
        dxrot_dphi = (x - cx) * dphi[0, 0] + (y - cy) * dphi[0, 1]
        dyrot_dphi = (x - cx) * dphi[1, 0] + (y - cy) * dphi[1, 1]

        # need negative sign, because thinking the j_cx and etc. terms as actually taking derivative wrt x, y, z coords
        # since these enter in the same way as cx, cy, cz, but with opposite sign
        jphi = -(j_cx * dxrot_dphi + j_cy * dyrot_dphi)
        jac = jac_base + [jphi]

        return jac

    def estimate_parameters(self,
                            data: np.ndarray,
                            coordinates: tuple[np.ndarray],
                            num_preserved_dims: int = 0):
        pguess_no_rot = self.base_model.estimate_parameters(data, coordinates, num_preserved_dims)
        pguess = np.concatenate((pguess_no_rot, np.array([0.])))
        return pguess

    def estimate_bounds(self,
                        coordinates: tuple[np.ndarray]) -> (tuple[float], tuple[float]):
        lbs_no_rot, ubs_no_rot = self.base_model.estimate_bounds(coordinates)
        lbs_angles = (-np.inf,)
        ubs_angles = (np.inf,)

        lbs = lbs_no_rot + lbs_angles
        ubs = ubs_no_rot + ubs_angles

        return lbs, ubs

    def normalize_parameters(self,
                             parameters) -> np.ndarray:
        normalized_params_no_rot = self.base_model.normalize_parameters(parameters[..., :-1])

        if parameters.ndim > 1:
            normalized_params = np.concatenate((normalized_params_no_rot, np.mod(parameters[..., -1:], 2*np.pi)), axis=1)
        else:
            normalized_params = np.concatenate((normalized_params_no_rot, np.mod(parameters[..., -1:], 2*np.pi)), axis=0)

        return normalized_params


class rotated_model_3d(coordinate_model):
    """
    Take any 3D model and parameterize its arbitrary rotation using by Euler angles.

    This is a helper function to avoid needing to write rotation code more than once

    r_body = U_z(psi)^-1 U_y(theta)^-1 U_z(phi)^-1 * r_lab
    U_z(phi)^-1 = [[cos(phi), -sin(phi), 0], [sin(phi), cos(phi), 0], [0, 0, 1]]
    f_rot(r_lab) = f(r_body) = f(U^{-1} * r_lab)

    Take the z-axis in the frame of the object, and consider the z-axis in the lab frame. phi and theta describe
    how the transformation to overlap these two. psi gives the angle the object is rotated about its own axis
    """

    def __init__(self,
                 model: coordinate_model,
                 center_inds: tuple[int]):
        """

        :param model:
        :param center_inds: center_inds: (cz_index, cy_index, cx_index)
        """

        # if model.ndim != 3:
        #     raise ValueError(f"model.ndim = {model.ndim:d}, but only 3D models are supported")

        param_names = model.parameter_names + ["phi", "theta", "psi"]
        has_jacobian = model.has_jacobian
        ndims = model.ndim

        if ndims != 3:
            raise ValueError(f"pixel oversampling only implemented for 3D models,"
                             f" but provided model has ndims={ndims:d}")

        super().__init__(param_names,
                         has_jacobian=has_jacobian,
                         ndims=ndims)

        self.base_model = model
        self.center_inds = center_inds

        # copy any attributes that don't overlap
        for k, v in model.__dict__.items():
            if not hasattr(self, k):
                setattr(self, k, v)

    def model(self,
              coordinates: tuple[np.ndarray],
              parameters: np.ndarray) -> np.ndarray:
        z, y, x = coordinates[-3:]

        phi = parameters[-3]
        theta = parameters[-2]
        psi = parameters[-1]
        rot_mat = affine.euler_mat_inv(phi, theta, psi)

        cx = parameters[self.center_inds[2]]
        cy = parameters[self.center_inds[1]]
        cz = parameters[self.center_inds[0]]

        # rotated coordinates
        xrot = (x - cx) * rot_mat[0, 0] + (y - cy) * rot_mat[0, 1] + (z - cz) * rot_mat[0, 2]
        yrot = (x - cx) * rot_mat[1, 0] + (y - cy) * rot_mat[1, 1] + (z - cz) * rot_mat[1, 2]
        zrot = (x - cx) * rot_mat[2, 0] + (y - cy) * rot_mat[2, 1] + (z - cz) * rot_mat[2, 2]

        # evaluate base model at rotated coordinates
        params_base = np.array(parameters[:-3], copy=True)
        params_base[self.center_inds[2]] = 0
        params_base[self.center_inds[1]] = 0
        params_base[self.center_inds[0]] = 0

        return self.base_model.model((zrot, yrot, xrot), params_base)

    def jacobian(self,
                 coordinates: tuple[np.ndarray],
                 parameters: np.ndarray) -> list[np.ndarray]:
        z, y, x = coordinates[-3:]

        phi = parameters[-3]
        theta = parameters[-2]
        psi = parameters[-1]
        rot_mat = affine.euler_mat_inv(phi, theta, psi)

        cx = parameters[self.center_inds[2]]
        cy = parameters[self.center_inds[1]]
        cz = parameters[self.center_inds[0]]

        # rotated coordinates
        xrot = (x - cx) * rot_mat[0, 0] + (y - cy) * rot_mat[0, 1] + (z - cz) * rot_mat[0, 2]
        yrot = (x - cx) * rot_mat[1, 0] + (y - cy) * rot_mat[1, 1] + (z - cz) * rot_mat[1, 2]
        zrot = (x - cx) * rot_mat[2, 0] + (y - cy) * rot_mat[2, 1] + (z - cz) * rot_mat[2, 2]

        dxrot_dcx = -rot_mat[0, 0]
        dxrot_dcy = -rot_mat[0, 1]
        dxrot_dcz = -rot_mat[0, 2]

        dyrot_dcx = -rot_mat[1, 0]
        dyrot_dcy = -rot_mat[1, 1]
        dyrot_dcz = -rot_mat[1, 2]

        dzrot_dcx = -rot_mat[2, 0]
        dzrot_dcy = -rot_mat[2, 1]
        dzrot_dcz = -rot_mat[2, 2]

        # evaluate base model at rotated coordinates
        params_base = np.array(parameters[:-3], copy=True)
        params_base[self.center_inds[2]] = 0.
        params_base[self.center_inds[1]] = 0.
        params_base[self.center_inds[0]] = 0.
        jac_base = self.base_model.jacobian((zrot, yrot, xrot), params_base)

        # need to correct jacobian with
        # (1) derivatives of rotated coordinates wrt centers
        # (2) derivatives wrt Euler angles
        j_cx = np.array(jac_base[self.center_inds[2]], copy=True)
        j_cy = np.array(jac_base[self.center_inds[1]], copy=True)
        j_cz = np.array(jac_base[self.center_inds[0]], copy=True)

        # need negative sign, because thinking the j_cx and etc. terms as actually taking derivative wrt x, y, z coords
        # since these enter in the same way as cx, cy, cz, but with opposite sign
        jac_base[self.center_inds[2]] = -(j_cx * dxrot_dcx + j_cy * dyrot_dcx + j_cz * dzrot_dcx)
        jac_base[self.center_inds[1]] = -(j_cx * dxrot_dcy + j_cy * dyrot_dcy + j_cz * dzrot_dcy)
        jac_base[self.center_inds[0]] = -(j_cx * dxrot_dcz + j_cy * dyrot_dcz + j_cz * dzrot_dcz)

        # euler angle derivatives
        dphi, dtheta, dpsi = affine.euler_mat_inv_derivatives(phi, theta, psi)
        dxrot_dphi = (x - cx) * dphi[0, 0] + (y - cy) * dphi[0, 1] + (z - cz) * dphi[0, 2]
        dyrot_dphi = (x - cx) * dphi[1, 0] + (y - cy) * dphi[1, 1] + (z - cz) * dphi[1, 2]
        dzrot_dphi = (x - cx) * dphi[2, 0] + (y - cy) * dphi[2, 1] + (z - cz) * dphi[2, 2]

        dxrot_dtheta = (x - cx) * dtheta[0, 0] + (y - cy) * dtheta[0, 1] + (z - cz) * dtheta[0, 2]
        dyrot_dtheta = (x - cx) * dtheta[1, 0] + (y - cy) * dtheta[1, 1] + (z - cz) * dtheta[1, 2]
        dzrot_dtheta = (x - cx) * dtheta[2, 0] + (y - cy) * dtheta[2, 1] + (z - cz) * dtheta[2, 2]

        dxrot_dpsi = (x - cx) * dpsi[0, 0] + (y - cy) * dpsi[0, 1] + (z - cz) * dpsi[0, 2]
        dyrot_dpsi = (x - cx) * dpsi[1, 0] + (y - cy) * dpsi[1, 1] + (z - cz) * dpsi[1, 2]
        dzrot_dpsi = (x - cx) * dpsi[2, 0] + (y - cy) * dpsi[2, 1] + (z - cz) * dpsi[2, 2]

        # need negative sign, because thinking the j_cx and etc. terms as actually taking derivative wrt x, y, z coords
        # since these enter in the same way as cx, cy, cz, but with opposite sign
        jphi   = -(j_cx * dxrot_dphi   + j_cy * dyrot_dphi   + j_cz * dzrot_dphi)
        jtheta = -(j_cx * dxrot_dtheta + j_cy * dyrot_dtheta + j_cz * dzrot_dtheta)
        jpsi   = -(j_cx * dxrot_dpsi   + j_cy * dyrot_dpsi   + j_cz * dzrot_dpsi)

        jac = jac_base + [jphi, jtheta, jpsi]

        return jac

    def estimate_parameters(self,
                            data: np.ndarray,
                            coordinates: tuple[np.ndarray],
                            num_preserved_dims: int = 0):
        pguess_no_rot = self.base_model.estimate_parameters(data, coordinates, num_preserved_dims)
        pguess = np.concatenate((pguess_no_rot, np.array([0., 0., 0.])))
        return pguess

    def estimate_bounds(self,
                        coordinates: tuple[np.ndarray]) -> (tuple[float], tuple[float]):
        lbs_no_rot, ubs_no_rot = self.base_model.estimate_bounds(coordinates)
        lbs_angles = (-np.inf, -np.inf, -np.inf)
        ubs_angles = (np.inf, np.inf, np.inf)

        lbs = lbs_no_rot + lbs_angles
        ubs = ubs_no_rot + ubs_angles

        return lbs, ubs

    def normalize_parameters(self,
                             parameters) -> np.ndarray:
        normalized_params_no_rot = self.base_model.normalize_parameters(parameters[..., :-3])

        if parameters.ndim > 1:
            normalized_params = np.concatenate((normalized_params_no_rot, parameters[..., -3:]), axis=1)
        else:
            normalized_params = np.concatenate((normalized_params_no_rot, parameters[..., -3:]), axis=0)

        return normalized_params


class fixed_parameter_model(coordinate_model):
    """
    Create a new model by fixing some parameters in a different model
    """

    def __init__(self,
                 model,
                 fixed_inds: tuple[int],
                 fixed_values: tuple[float]):
        self.base_model = model
        self.fixed_inds = fixed_inds
        self.fixed_mask = np.array([True if ii in fixed_inds else False for ii in range(model.nparams)])
        self.fixed_values = fixed_values

        self.unfixed_inds = tuple([ii for ii in range(model.nparams) if ii not in fixed_inds])
        self.unfixed_mask = np.logical_not(self.fixed_mask)

        param_names_not_fixed = [p for ii, p in enumerate(model.parameter_names) if ii not in fixed_inds]

        super().__init__(param_names_not_fixed,
                         has_jacobian=model.has_jacobian,
                         ndims=model.ndim)

    def model(self,
              coordinates: tuple[np.ndarray],
              parameters: np.ndarray) -> np.ndarray:

        p = np.zeros(self.base_model.nparams)
        p[self.fixed_mask] = np.array(self.fixed_values)
        p[self.unfixed_mask] = parameters

        return self.base_model.model(coordinates, p)

    def jacobian(self,
                 coordinates: tuple[np.ndarray],
                 parameters: np.ndarray) -> list[np.ndarray]:

        p = np.zeros(self.base_model.nparams)
        p[self.fixed_mask] = np.array(self.fixed_values)
        p[self.unfixed_mask] = parameters

        jac = [j for ii, j in enumerate(self.base_model.jacobian(coordinates, p)) if ii not in self.fixed_inds]
        return jac

    def estimate_parameters(self,
                            data: np.ndarray,
                            coordinates: tuple[np.ndarray],
                            num_preserved_dims: int = 0):

        p = self.base_model.estimate_parameters(data, coordinates, num_preserved_dims)
        p_out = p[self.unfixed_mask]

        return p_out

    def estimate_bounds(self,
                        coordinates: tuple[np.ndarray]) -> (tuple[float], tuple[float]):


        lbs_full, ubs_full = self.base_model.estimate_bounds(coordinates)
        lbs = tuple([lb for ii, lb in enumerate(lbs_full) if ii not in self.fixed_inds])
        ubs = tuple([ub for ii, ub in enumerate(ubs_full) if ii not in self.fixed_inds])

        return lbs, ubs

    def normalize_parameters(self,
                             parameters) -> np.ndarray:

        p = np.zeros(parameters.shape[:-1] + (self.base_model.nparams,))
        p[..., self.fixed_mask] = np.array(self.fixed_values)
        p[..., self.unfixed_mask] = parameters

        p_norm = self.base_model.normalize_parameters(p)
        p_out = p_norm[..., self.unfixed_mask]

        return p_out


class gauss1d(coordinate_model):
    def __init__(self):
        super().__init__(["amp", "center", "sigma", "bg"],
                         ndims=1,
                         has_jacobian=True)

    def model(self,
              coordinates: tuple[np.ndarray],
              parameters: np.ndarray):
        x, = coordinates[-1:]
        g = parameters[0] * np.exp(-(x - parameters[1]) ** 2 / (2 * parameters[2] ** 2)) + parameters[3]
        return g

    def jacobian(self,
                 coordinates: tuple[np.ndarray],
                 parameters: np.ndarray):

        amp, c, sig, bg = parameters

        x, = coordinates[-1:]
        # useful functions that show up in derivatives
        exps = np.exp(-(x - c) ** 2 / (2 * sig ** 2))

        jac = [exps,
               amp * exps * (x - c) / parameters[2] ** 2,
               amp * exps * (x - c) ** 2 / parameters[2] ** 3,
               np.ones(x.shape),
               ]

        return jac

    def estimate_parameters(self,
                            data: np.ndarray,
                            coordinates: tuple[np.ndarray],
                            num_preserved_dims: int = 0):

        if num_preserved_dims != 0:
            raise NotImplementedError()

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
        x, = coordinates[-1:]
        lbs = (-np.inf, x.min(), 0, -np.inf)
        ubs = (np.inf, x.max(), x.max() - x.min(), np.inf)
        return lbs, ubs

    def normalize_parameters(self,
                             parameters):
        param_norm = np.array(parameters, copy=True)
        param_norm[..., 2] = np.abs(param_norm[..., 2])

        return param_norm


class gauss2d(coordinate_model):
    def __init__(self, use_sigma_ratio_parameterization=False):
        if use_sigma_ratio_parameterization:
            super().__init__(["amp", "cx", "cy", "sx", "sy/sx", "bg", "theta"], 2, has_jacobian=True)
        else:
            super().__init__(["amp", "cx", "cy", "sx", "sy", "bg", "theta"], 2, has_jacobian=True)
        self.use_sigma_ratio_parameterization = use_sigma_ratio_parameterization


    def model(self,
              coordinates: tuple[np.ndarray],
              parameters: np.ndarray):
        y, x = coordinates[-2:]
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

        y, x = coordinates[-2:]
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
                            coordinates: tuple[np.ndarray],
                            num_preserved_dims: int = 0):

        if num_preserved_dims != 0:
            raise NotImplementedError()

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

        yy, xx = coordinates[-2:]
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


class ellipsoid2d(coordinate_model):
    def __init__(self, decay_length):
        """
        2D ellipsoid

        :param decay_length:
        """
        self.decay_length = decay_length
        super().__init__(["amp", "cx", "cy", "ax", "ay", "bg"], 2, has_jacobian=True)

    def model(self,
              coordinates: tuple[np.ndarray],
              parameters: np.ndarray) -> np.ndarray:
        y, x = coordinates[-2:]
        bcast_shape = (x + y).shape
        amp, cx, cy, ax, ay, bg = parameters

        surface_val = (x - cx) ** 2 / ax ** 2 + (y - cy) ** 2 / ay ** 2
        inside = surface_val <= 1

        val = np.zeros(bcast_shape)
        val[inside] = amp
        val[np.logical_not(inside)] = amp * np.exp(-surface_val[np.logical_not(inside)] / self.decay_length)
        val += bg

        return val

    def jacobian(self,
                 coordinates: tuple[np.ndarray],
                 parameters: np.ndarray) -> list[np.ndarray]:
        y, x = coordinates[-2:]
        bcast_shape = (x + y).shape
        amp, cx, cy, ax, ay, bg = parameters

        # building blocks of jacobian
        surface_val = (x - cx) ** 2 / ax ** 2 + (y - cy) ** 2 / ay ** 2
        ds_dcx = - 2 * (x - cx) / ax ** 2
        ds_dcy = - 2 * (y - cy) / ay ** 2
        ds_dax = - 2 * (x - cx)**2 / ax ** 3
        ds_day = - 2 * (y - cy)**2 / ay ** 3

        exp_decay = np.exp(-surface_val / self.decay_length)
        inside = surface_val <= 1
        outside = np.logical_not(inside)

        dexp_ds = np.zeros(bcast_shape)
        dexp_ds[outside] = amp * -1 / self.decay_length * exp_decay[outside]

        # jacobian components
        df_damp = np.ones(bcast_shape)
        df_damp[outside] = exp_decay[outside]

        jac = [df_damp,
               dexp_ds * ds_dcx,
               dexp_ds * ds_dcy,
               dexp_ds * ds_dax,
               dexp_ds * ds_day,
               np.ones(bcast_shape)]

        return jac

    def estimate_parameters(self,
                            data: np.ndarray,
                            coordinates: tuple[np.ndarray],
                            num_preserved_dims: int = 0):
        return gauss2d().estimate_parameters(data, coordinates, num_preserved_dims)[..., :-1]

    def estimate_bounds(self,
                        coordinates: tuple[np.ndarray]) -> (tuple[float], tuple[float]):
        lbs, ubs = gauss2d().estimate_bounds(coordinates)
        return lbs[:-1], ubs[:-1]


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
            val += gauss2d().model(coordinates, ps)

        # deal with last gaussian, which also gets background term
        ps = np.concatenate((np.array(p[-7:-2]), np.atleast_1d(p[-1]), np.atleast_1d(p[-2])))
        val += gauss2d().model(coordinates, ps)

        return val


    def jacobian(self,
                 coordinates: tuple[np.ndarray],
                 p: np.ndarray):

        jac_list = []
        for ii in range(self.ngaussians - 1):
            ps = np.concatenate((np.array(p[6 * ii: 6 * ii + 5]), np.array([0]), np.atleast_1d([p[ii * 6 + 5]])))
            jac_current = gauss2d().jacobian(coordinates, ps)
            jac_list += jac_current[:-2] + [jac_current[-1]]

        # deal with last gaussian, which also gets background term
        ps = np.concatenate((np.array(p[-7:-2]), np.atleast_1d(p[-1]), np.atleast_1d(p[-2])))
        jac_current = gauss2d().jacobian(coordinates, ps)
        jac_list += jac_current[:-2] + [jac_current[-1]] + [jac_current[-2]]

        return jac_list


    def normalize_parameters(self,
                             parameters):

        param_norm = np.array(parameters, copy=True)
        param_norm[..., 3::6] = np.abs(param_norm[..., 3::6])
        param_norm[..., 4::6] = np.abs(param_norm[..., 4::6])
        param_norm[..., 6::6] = np.mod(param_norm[..., 6::6], 2*np.pi)

        return param_norm


# todo: should derive this from gauss3d_asymmetric?
class gauss3d(coordinate_model):
    def __init__(self,
                 minimum_sigmas: tuple[float] = (0., 0.)):
        """
        3D gaussian symmetric in xy
        """
        self.minimum_sigmas = minimum_sigmas
        super().__init__(["amp", "cx", "cy", "cz", "sxy", "sz", "bg"], ndims=3, has_jacobian=True)


    def model(self,
              coordinates: tuple[np.ndarray],
              params: np.ndarray) -> np.ndarray:

        z, y, x, = coordinates[-3:]
        amp, cx, cy, cz, sxy, sz, bg = params
        sxy_min, sz_min = self.minimum_sigmas

        # calculate psf at oversampled points
        val = amp * np.exp(-(x - cx) ** 2 / 2 / (sxy_min ** 2 + sxy ** 2)
                           -(y - cy) ** 2 / 2 / (sxy_min ** 2 + sxy ** 2)
                           -(z - cz) ** 2 / 2 / (sz_min ** 2 + sz ** 2)
                           ) + bg

        return val


    def jacobian(self,
                 coordinates: tuple[np.ndarray],
                 params: np.ndarray) -> list[np.ndarray]:

        z, y, x = coordinates[-3:]
        amp, cx, cy, cz, sxy, sz, bg = params
        sxy_min, sz_min = self.minimum_sigmas
        bcast_shape = (x + y + z).shape

        # use sxy * |sxy| instead of sxy**2 to enforce sxy > 0
        v = np.exp(-(x - cx) ** 2 / 2 / (sxy_min ** 2 + sxy ** 2)
                   -(y - cy) ** 2 / 2 / (sxy_min ** 2 + sxy ** 2)
                   -(z - cz) ** 2 / 2 / (sz_min ** 2 + sz ** 2)
                  )

        # [A, cx, cy, cz, sxy, sz, bg]
        jac = [v,
               amp * v * 2 * (x - cx) / 2 / (sxy_min ** 2 + sxy ** 2),
               amp * v * 2 * (y - params[2]) / 2 / (sxy_min ** 2 + sxy ** 2),
               amp * v * 2 * (z - params[3]) / 2 / (sz_min ** 2 + sz ** 2),
               amp * v * (2 * sxy / (sxy_min ** 2 + sxy ** 2) ** 2) * ((x - cx) ** 2 / 2 + (y - cy) ** 2 / 2),
               amp * v * 2 * sz / (sz_min ** 2 + sz ** 2) ** 2 * (z - cz) ** 2 / 2,
               np.ones(bcast_shape)]

        return jac


    def estimate_parameters(self,
                            data: np.ndarray,
                            coordinates: tuple[np.ndarray],
                            num_preserved_dims: int = 0):

        if self.ndim != len(coordinates):
            raise ValueError("len(coords) != model dimensions")

        z, y, x = coordinates[-3:]
        data_ndim = data.ndim


        average_axes = tuple(range(num_preserved_dims, data_ndim))

        # subtract smallest value so positive
        img_temp = data - np.nanmin(data, axis=average_axes, keepdims=True)
        img_temp[img_temp <= 0] = np.nan

        # compute moments
        isum = np.nansum(img_temp, axis=average_axes)
        m1x = np.nansum((img_temp * x), axis=average_axes) / isum
        m2x = np.nansum((img_temp * x ** 2), axis=average_axes) / isum
        sx = np.sqrt(m2x - m1x ** 2)
        m1y = np.nansum((img_temp * y), axis=average_axes) / isum
        m2y = np.nansum((img_temp * y ** 2), axis=average_axes) / isum
        sy = np.sqrt(m2y - m1y ** 2)
        m1z = np.nansum((img_temp * z), axis=average_axes) / isum
        m2z = np.nansum((img_temp * z ** 2), axis=average_axes) / isum
        sz = np.sqrt(m2z - m1z ** 2)

        # if e.g. all img_temp values are the same, sigmas can be zero and to machine precision can get NaN. avoid this
        # if np.isnan(sz):
        #     sz = 0.5 * (np.max(z, axis=average_axes) - np.min(z, axis=average_axes))

        # if np.isnan(sxy):
        #     sxy = 0.5 * (0.5 * (np.max(x, axis=average_axes) - np.min(x, axis=average_axes)) +
        #                  0.5 * (np.max(y, axis=average_axes) - np.min(y, axis=average_axes))
        #                  )

        guess_params = np.stack((np.nanmax(data, axis=average_axes) - np.nanmin(data, axis=average_axes),
                                 m1x, m1y, m1z, 0.5 * (sx + sy), sz,
                                 np.nanmean(data, axis=average_axes)), axis=-1)

        return self.normalize_parameters(guess_params)


    def estimate_bounds(self,
                        coordinates: tuple[np.ndarray]):
        z, y, x = coordinates[-3:]

        lbs = (-np.inf, x.min(), y.min(), z.min(), 0, 0, -np.inf)
        ubs = (np.inf, x.max(), y.max(), z.max(), np.inf, np.inf, np.inf)
        return lbs, ubs

    def normalize_parameters(self,
                             params: np.ndarray):
        norm_params = np.array(params, copy=True)
        norm_params[..., 4] = np.abs(norm_params[..., 4])
        norm_params[..., 5] = np.abs(norm_params[..., 5])

        return norm_params


class gauss3d_asymmetric(coordinate_model):
    def __init__(self,
                 use_sigma_ratio_parameterization: bool = False,
                 minimum_sigmas: tuple[float] = (0., 0., 0.)):
        self.use_sigma_ratio_parameterization = use_sigma_ratio_parameterization
        self.minimum_sigmas = minimum_sigmas

        if use_sigma_ratio_parameterization:
            super().__init__(["A", "cx", "cy", "cz", "sx", "sy/sx", "sz", "bg"],
                             3, has_jacobian=True)
        else:
            super().__init__(["A", "cx", "cy", "cz", "sx", "sy", "sz", "bg"],
                             3, has_jacobian=True)


    def model(self,
              coords: tuple[np.ndarray],
              params: np.ndarray):

        z, y, x, = coords[-3:]
        sx_min, sy_min, sz_min = self.minimum_sigmas

        if self.use_sigma_ratio_parameterization:
            amp, cx, cy, cz, sx, ratio, sz, bg = params
            sy = sx * ratio
        else:
            amp, cx, cy, cz, sx, sy, sz, bg = params

        vals = amp * np.exp(- (x - cx) ** 2 / 2 / (sx_min ** 2 + sx ** 2)
                            - (y - cy) ** 2 / 2 / (sy_min ** 2 + sy ** 2)
                            - (z - cz) ** 2 / 2 / (sz_min ** 2 + sz ** 2)) + bg

        return vals


    def jacobian(self,
                 coords: tuple[np.ndarray],
                 params: np.ndarray):

        z, y, x, = coords[-3:]
        sx_min, sy_min, sz_min = self.minimum_sigmas
        bcast_shape = (x + y + z).shape

        if self.use_sigma_ratio_parameterization:
            amp, cx, cy, cz, sx, ratio, sz, bg = params
            sy = sx * ratio
        else:
            amp, cx, cy, cz, sx, sy, sz, bg = params

        # calculate psf at oversampled points
        val0 = np.exp(-(x - cx) ** 2 / 2 / (sx_min ** 2 + sx ** 2)
                      -(y - cy) ** 2 / 2 / (sy_min ** 2 + sy ** 2)
                      -(z - cz) ** 2 / 2 / (sz_min ** 2 + sz ** 2))

        dpsf_dcx = 2 * (x - cx) / 2 / (sx_min ** 2 + sx ** 2)
        dpsf_dcy = 2 * (y - cy) / 2 / (sy_min ** 2 + sy ** 2)
        dpsf_dcz = 2 * (z - cz) / 2 / (sz_min ** 2 + sz ** 2)

        if self.use_sigma_ratio_parameterization:
            dpsf_dsx = 2 * sx / (sx_min ** 2 + sx ** 2) ** 2 * (x - cx) ** 2 / 2 + \
                       2 * sx * ratio ** 2 / (sy_min ** 2 + ratio ** 2 * sx ** 2) ** 2 * (y - cy) ** 2 / 2
        else:
            dpsf_dsx = 2 * sx / (sx_min ** 2 + sx ** 2) ** 2 * (x - cx) ** 2 / 2

        if self.use_sigma_ratio_parameterization:
            dpsf_dsy_like = 2 * ratio * sx ** 2 / (sy_min ** 2 + ratio ** 2 * sx ** 2)**2 * (y - cy) ** 2 / 2
        else:
            dpsf_dsy_like = 2 * sy / (sy_min ** 2 + sy ** 2) ** 2 * (y - cy) ** 2 / 2

        dpsf_dsz = 2 * sz / (sz_min ** 2 + sz ** 2) ** 2 * (z - cz) ** 2 / 2

        jac = [val0,  # A
               amp * dpsf_dcx * val0,  # cx
               amp * dpsf_dcy * val0,  # cy
               amp * dpsf_dcz * val0,  # cz
               amp * dpsf_dsx * val0,  # sx
               amp * dpsf_dsy_like * val0,  # sy/sx or sy
               amp * dpsf_dsz * val0,  # sz
               np.ones(bcast_shape)  # bg
               ]

        return jac


    def estimate_parameters(self,
                            img: np.ndarray,
                            coords: tuple[np.ndarray],
                            num_preserved_dims: int = 0):

        if num_preserved_dims != 0:
            raise NotImplementedError()

        z, y, x = coords[-3:]

        # subtract smallest value so positive
        img_temp = img
        to_use = np.logical_and(np.logical_not(np.isnan(img_temp)), img_temp > 0)

        if self.ndim != len(coords):
            raise ValueError("len(coords) != img.ndim")

        # compute moments
        c1s = np.zeros(img.ndim)
        c2s = np.zeros(img.ndim)
        isum = np.sum(img_temp[to_use])
        for ii in range(img.ndim):
            c1s[ii] = np.sum((img_temp * coords[ii])[to_use]) / isum
            c2s[ii] = np.sum((img_temp * coords[ii]**2)[to_use]) / isum

        # sz, sy, sx
        sigmas = np.sqrt(c2s - c1s ** 2)

        if np.isnan(sigmas[0]):
            sigmas[0] = 0.5 * (z.max() - z.min())
        if np.isnan(sigmas[1]):
            sigmas[1] = 0.5 * (y.max() - y.min())
        if np.isnan(sigmas[2]):
            sigmas[2] = 0.5 * (x.max() - x.min())

        if self.use_sigma_ratio_parameterization:
            sigmas[1] = sigmas[1] / sigmas[2]

        guess_params = np.concatenate((np.array([np.nanmax(img) - np.nanmean(img)]),
                                       np.flip(c1s),
                                       np.flip(sigmas),
                                       np.array([np.nanmean(img)])
                                       ),
                                      )

        return self.normalize_parameters(guess_params)


    def estimate_bounds(self,
                        coordinates: tuple[np.ndarray]) -> (tuple[float], tuple[float]):
        z, y, x = coordinates[-3:]

        lbs = (-np.inf, x.min(), y.min(), z.min(), 0, 0, 0, -np.inf)
        ubs = (np.inf, x.max(), y.max(), z.max(), np.inf, np.inf, np.inf, np.inf)
        return lbs, ubs


    def normalize_parameters(self, params):
        norm_params = np.array(params, copy=True)
        norm_params[..., 4] = np.abs(norm_params[..., 4])
        norm_params[..., 5] = np.abs(norm_params[..., 5])
        norm_params[..., 6] = np.abs(norm_params[..., 6])

        return norm_params


class ellipsoid3d(coordinate_model):
    def __init__(self, decay_length):
        """
        3D ellipsoid

        :param decay_length: to facilitate fitting, the value outside of the ellipsoid decays exponentially instead of
        instantly cutting off
        """
        self.decay_length = decay_length
        super().__init__(["amp", "cx", "cy", "cz", "ax", "ay", "az", "bg"], 3, has_jacobian=True)

    def model(self,
              coordinates: tuple[np.ndarray],
              parameters: np.ndarray) -> np.ndarray:
        z, y, x = coordinates[-3:]
        bcast_shape = (x + y + z).shape
        
        amp, cx, cy, cz, ax, ay, az, bg = parameters

        surface_val = (x - cx)**2 / ax**2 + (y - cy)**2 / ay**2 + (z - cz)**2 / az**2
        inside = surface_val <= 1

        val = np.zeros(bcast_shape)
        val[inside] = amp
        val[np.logical_not(inside)] = amp * np.exp(-surface_val[np.logical_not(inside)] / self.decay_length)
        val += bg

        return val

    def jacobian(self,
                 coordinates: tuple[np.ndarray],
                 parameters: np.ndarray) -> list[np.ndarray]:

        z, y, x = coordinates[-3:]
        # z, y, x = np.broadcast_arrays(*coordinates[-3:])
        bcast_shape = (x + y + z).shape

        amp, cx, cy, cz, ax, ay, az, bg = parameters

        # building blocks of jacobian
        surface_val = (x - cx) ** 2 / ax ** 2 + (y - cy) ** 2 / ay ** 2 + (z - cz) ** 2 / az ** 2
        ds_dcx = - 2 * (x - cx) / ax ** 2
        ds_dcy = - 2 * (y - cy) / ay ** 2
        ds_dcz = - 2 * (z - cz) / az ** 2
        ds_dax = - 2 * (x - cx)**2 / ax ** 3
        ds_day = - 2 * (y - cy)**2 / ay ** 3
        ds_daz = -2 * (z - cz)**2 / az ** 3

        exp_decay = np.exp(-surface_val / self.decay_length)
        inside = surface_val <= 1
        outside = np.logical_not(inside)

        dexp_ds = np.zeros(bcast_shape)
        dexp_ds[outside] = amp * -1 / self.decay_length * exp_decay[outside]

        # jacobian components
        df_damp = np.ones(bcast_shape)
        df_damp[outside] = exp_decay[outside]

        jac = [df_damp,
               dexp_ds * ds_dcx,
               dexp_ds * ds_dcy,
               dexp_ds * ds_dcz,
               dexp_ds * ds_dax,
               dexp_ds * ds_day,
               dexp_ds * ds_daz,
               np.ones(bcast_shape)]

        return jac

    def estimate_parameters(self,
                            data: np.ndarray,
                            coordinates: tuple[np.ndarray],
                            num_preserved_dims: int = 0):
        return gauss3d_asymmetric().estimate_parameters(data, coordinates, num_preserved_dims)

    def estimate_bounds(self,
                        coordinates: tuple[np.ndarray]) -> (tuple[float], tuple[float]):
        return gauss3d_asymmetric().estimate_bounds(coordinates)


class line_piecewisem(coordinate_model):

    def __init__(self):
        super().__init__(["slope1", "y intercept1", "slope2", "crossover"], 1, has_jacobian=True)

    def model(self,
              coordinates: tuple[np.ndarray],
              parameters: np.ndarray) -> np.ndarray:
        """
        Two piecewise lines which connect at a point

        :param x: x-positions to evaluate function
        :param p: [slope 1, y-intercept 1, slope 2, changover point]
        :return value:
        """

        x = coordinates[-1:]
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

        x = coordinates[-1:]
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
                            coordinates: tuple[np.ndarray],
                            num_preserved_dims: int = 0):
        pass


def fit_model(img: np.ndarray,
              model_fn,
              init_params: list[float],
              fixed_params: Optional[list[bool]] = None,
              sd: Optional[np.ndarray] = None,
              bounds: Optional[tuple[tuple[float]]] = None,
              model_jacobian: Optional[callable] = None,
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
    :return: results
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
                      fixed_params: Optional[list[bool]] = None,
                      bounds: Optional[tuple[tuple[float]]] = None,
                      model_jacobian: Optional[callable] = None,
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
    :return results: dictionary object. Uncertainty can be obtained from the square roots of the diagonals of the
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
                coords: Optional[tuple[np.ndarray]] = None,
                dims: Optional[list[int]] = None) -> list[float]:
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

    :param x: x-coordinates to evaluate function at.
    :param y:
    :param p: [cx, cy, radius, in value, out value, decay_len]
    :return value:
    """
    dist = np.sqrt((x - p[0])**2 + (y - p[1])**2)
    in_circ = p[3] * np.exp((p[2] - dist) / p[5]) + p[4]

    in_circ[dist < p[2]] = p[3]

    return in_circ


def line_piecewise(x, p):
    """
    Two piecewise lines which connect at a point

    :param x: x-positions to evaluate function
    :param p: [slope 1, y-intercept 1, slope 2, changover point]
    :return value:
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

    :param x: x-points to evaluate function
    :param y: y-points to evaluate function
    :param p: [amp, cx, cy, wx, wy, bg, theta]
    :return value:
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
