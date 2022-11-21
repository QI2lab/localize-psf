import numpy as np
import pandas as pd
import json
import time
# from tysserand import tysserand as ty
import localize_psf.rois as roi_fns
from localize_psf import fit
import localize_psf.fit_psf as psf
from localize_psf import localize


def _unpack_tuple(x):
    """
    Unpacks one-element tuples for use as return values 
    from Numpy's arraysetops module
    """
    if len(x) == 1:
        return x[0]
    else:
        return x

# ------ general FISH pipeline functions ------

def make_sigmas(na=None, ni=None, lambda_em=None,
                sz=None, sxy=None, sigma_ratio=1.6):
    """
    Compute min and max of sigmas x, y and z from physical parameters 
    or from estimated size of gaussian spot in pixel. Returned values
    are in µm if physical parameters are used, in pixels otherwise.

    Parameters
    ----------
    na : float
        Numerical aperture of the objective.
    ni : float
        Refractive index of the sample.
    lambda_em : float
        Emission wavelength in µm.
    sz : float | int
        Estimated size of spots along the z axis in pixels.
    sxy : float | int
        Estimated size of spots in the xy plane in pixels.
    sigma_ratio : float
        Ratio between big and small sigma for z or xy.
        To reproduce LoG with DoG we need sigma_big = 1.6 * sigma_small.
    """
    if na is None:
        # FWHM = 2.355 x sigma
        sigma_xy = sxy / 2.355
        sigma_z = sz / 2.355
        sigma_xy_small = sigma_xy / sigma_ratio**(1/2)
        sigma_xy_large = sigma_xy * sigma_ratio**(1/2)
        sigma_z_small = sigma_z / sigma_ratio**(1/2)
        sigma_z_large = sigma_z * sigma_ratio**(1/2)

        return sigma_z, sigma_xy, sigma_z_small, sigma_xy_small, sigma_z_large, sigma_xy_large
        
    else:
        sigma_xy = psf.na2sxy(na, lambda_em)
        sigma_z = psf.na2sz(na, lambda_em, ni)

        # TODO: add choice of coefficients
        # difference of gaussian filer
        filter_sigma_small = (0.5 * sigma_z, 0.25 * sigma_xy, 0.25 * sigma_xy)
        filter_sigma_large = (3 * sigma_z, 3 * sigma_xy, 3 * sigma_xy)
        # exclude points with sigmas outside these ranges
        sigmas_min = (0.25 * sigma_z, 0.25 * sigma_xy)
        sigmas_max = (3 * sigma_z, 4 * sigma_xy)

        return sigma_z, sigma_xy, sigmas_min, sigmas_max, filter_sigma_small, filter_sigma_large


def make_roi_sizes(sz, sxy, dz, dxy, coef_roi=2, coef_min_roi=0.5):
    """
    Compute the x/y and z sizes of ROIs to fit gaussians to spots.
    """
    if isinstance(coef_roi, (int, float)):
        coef_roi = np.array([coef_roi] * 3)
    if isinstance(coef_min_roi, (int, float)):
        coef_min_roi = np.array([coef_min_roi] * 3)
    
    roi_sizes = (coef_roi * np.array([sz, sxy, sxy])).astype(int)
    roi_sizes_pix = roi_fns.get_roi_size(roi_sizes, dxy, dz, ensure_odd=True)
    min_roi_sizes_pix = roi_sizes_pix * coef_min_roi

    return roi_sizes, roi_sizes_pix, min_roi_sizes_pix


def make_min_spot_sep(sigma_z, sigma_xy, coef_sep=(3, 3)):
    # assume points closer together than this come from a single bead
    min_spot_sep = (coef_sep[0] * sigma_z, coef_sep[1] * sigma_xy)
    return min_spot_sep


# no longer used for direct implementation of localize_psf
# will merge both method in the future
# def get_roi_coordinates(centers, sizes, max_coords_val, min_sizes, return_sizes=True):
#     """
#     Make pairs of (z, y, x) coordinates defining an ROI.
    
#     Parameters
#     ----------
#     centers : ndarray, dtype int
#         Centers of future ROIs, a Nx3 array.
#     sizes : array or list
#         Size of ROIs in each dimensions.
#     max_coords_val : array or list
#         Maximum value of coordinates in each dimension,
#         typically the original image shape - 1.
#     min_sizes : array or list
#         Minimum size of ROIs in each dimension.
    
#     Returns
#     -------
#     roi_coords : ndarray
#         Pairs of point coordinates, a 2xNx3 array.
#     roi_coords : ndarray
#         Shape of each ROI, Nx3 array.
#     """
    
#     # make raw coordinates
#     min_coords = centers - sizes / 2
#     max_coords = centers + sizes / 2
#     coords = np.stack([min_coords, max_coords]).astype(int)
#     # clean min and max values of coordinates
#     coords[coords < 0] = 0
#     for i in range(3):
#         coords[1, coords[1, :, i] > max_coords_val[i], i] = max_coords_val[i]
#     # delete small ROIs
#     roi_sizes = coords[1, :, :] - coords[0, :, :]
#     select = ~np.any([roi_sizes[:, i] < min_sizes[i] for i in range(3)], axis=0)
#     coords = coords[:, select, :]
#     # swap axes for latter convenience
#     roi_coords = np.swapaxes(coords, 0, 1)
    
#     if return_sizes:
#         roi_sizes = roi_sizes[select, :]
#         return roi_coords, roi_sizes
#     else:
#         return roi_coords

    
# def extract_ROI(img, coords):
#     """
#     Extract a portion of an image given by the coordinates of 2 points.
    
#     Parameters
#     ----------
#     img : ndarray, dimension 3
#         The image from which the ROI is extracted.
#     coords : ndarry, shape (2, 3)
#         The 2 coordinates of the 3 dimensional points at the corner of the ROI.
    
#     Returns
#     -------
#     roi : ndarray
#         A region of interest of the original image.
#     """
    
#     z0, y0, x0 = coords[0]
#     z1, y1, x1 = coords[1]
#     roi = img[z0:z1, y0:y1, x0:x1]
#     return roi


def find_mask_borders(coords, mask_shape, img_shape, min_sizes=None):
    """
    Compute the min and max indices in all dimensions of a mask to extract
    its values. The mask's center coordinates can get too close to the 
    border of an image, thus the indices need to change accordingly.

    Returns
    -------
    coords : ndarray
        Coordinates of the center position of mask.
    mask_shape : array
        Shape of mask.
    img_shape : array
        Shape of image where mask is applied.
    min_sizes : array or list
        Minimum size of masks in each dimension.

    Example
    -------
    >>> img = np.arange(100).reshape(10, 10)
    >>> coords = np.array[[5, 5], [0, 0], [8, 6]]
    >>> mask = np.ones(9).reshape(3, 3)
    """

    n_dim = coords.shape[1]
    # make sure we manipulate arrays and not tuples or lists
    mask_shape = np.array(mask_shape)
    img_shape = np.array(img_shape)

    # we can't broadcast arrays in min()
    # we need to build a nb_coords x nb_dims x 2 array
    min_id = (mask_shape / 2 - coords).astype(int)
    min_ref = np.zeros_like(min_id, dtype=int)
    min_id = np.max(np.stack([min_id, min_ref]), axis=0)

    max_id = (mask_shape / 2 - coords + img_shape).astype(int)
    max_ref = (mask_shape * np.ones((len(coords), 1))).astype(int)
    max_id = np.min(np.stack([max_id, max_ref]), axis=0)

    # delete small masks
    if min_sizes is None:
        min_sizes = mask_shape
    mask_sizes = max_id - min_id
    select = ~np.any([mask_sizes[:, i] < min_sizes[i] for i in range(n_dim)], axis=0)
    coords = coords[select, :]
    min_id = min_id[select, :]
    max_id = max_id[select, :]

    return coords, min_id, max_id


def extract_masked_spot(img, coords, mask, min_sizes=None):
    """
    Extract a tight portions of image around points, which shape is
    given by a boolean mask.

    Parameters
    ----------
    img : ndarray
        The image where pixel values are exctracted.
    coords : ndarray
        Coordinates of the center positions of mask.
    mask : ndarray
        Boolean mask to select pixel values.
    min_sizes : list, tuple or array
        Minimum size of masks that are cut when crossing the border if the image.

    Returns
    -------
    values : array
        Intensity values of masked pixels
    px_coords : list of arrays
        Coordinates of masked pixels in the z, y and x dimension.

    Example
    -------
    >>> img = np.arange(100).reshape(10, 10)
    >>> coords = np.array([[4, 4], [0, 0], [9, 0], [0, 9], [9, 9], [9, 6]])
    >>> mask = np.ones((3, 3), dtype=bool)
    >>> rois_values, px_coords_y, px_coords_x = extract_masked_spot(img, coords, mask, min_sizes=[2, 2])
    >>> print(rois_values)
    [array([33, 34, 35, 43, 44, 45, 53, 54, 55]), 
    array([ 0,  1, 10, 11]), array([88, 89, 98, 99]),
    array([85, 86, 87, 95, 96, 97])]
    >>> print(px_coords_y)
    [array([3, 3, 3, 4, 4, 4, 5, 5, 5]), 
    array([0, 0, 1, 1]), array([8, 8, 9, 9]), 
    array([8, 8, 8, 9, 9, 9])]
    >>> print(px_coords_x)
    [array([3, 4, 5, 3, 4, 5, 3, 4, 5]), 
    array([0, 1, 0, 1]), 
    array([8, 9, 8, 9]), 
    array([5, 6, 7, 5, 6, 7])]
    # in 3D:
    >>> img = np.arange(4**3).reshape(4, 4, 4)
    >>> coords = np.array([[1, 1, 1], [0, 0, 0], [3, 3, 3]])
    >>> mask = np.ones((3, 3, 3), dtype=bool)
    >>> rois_values, px_coords_z, px_coords_y, px_coords_x = fish.extract_masked_spot(img, coords, mask, min_sizes=[2, 2, 2])
    >>> print(rois_values)
    [array([ 0,  1,  2,  4,  5,  6,  8,  9, 10, 16, 17, 18, 20, 21, 22, 24, 25, 26, 32, 33, 34, 36, 37, 38, 40, 41, 42]), 
    array([ 0,  1,  4,  5, 16, 17, 20, 21]), 
    array([42, 43, 46, 47, 58, 59, 62, 63])]
    >>> print(px_coords_z)
    [array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2]), 
    array([0, 0, 0, 0, 1, 1, 1, 1]), 
    array([2, 2, 2, 2, 3, 3, 3, 3])]
    >>> print(px_coords_y)
    [array([0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2]), 
    array([0, 0, 1, 1, 0, 0, 1, 1]), 
    array([2, 2, 3, 3, 2, 2, 3, 3])]
    >>> print(px_coords_x)
    [array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]), 
    array([0, 1, 0, 1, 0, 1, 0, 1]), 
    array([2, 3, 2, 3, 2, 3, 2, 3])]
    """

    n_dim = coords.shape[1]
    mask_shape = np.array(mask.shape)
    # find indices to cut mask correctly at borders
    coords, min_id, max_id = find_mask_borders(coords, mask_shape, img.shape, min_sizes)

    # ------ extract values of img from masks as 1D arrays -------
    rois_values = []
    px_coords_z = []
    px_coords_y = []
    px_coords_x = []
    # compute limits of ROIs from center coordinates and masks sizes
    # the astype(int) int the niddle is important for rouding/truncating matters
    min_roi = (coords - (mask_shape / 2).astype(int) + min_id).astype(int)
    max_roi = (coords - (mask_shape / 2).astype(int) + max_id).astype(int)
    if n_dim == 3:
        for i in range(len(coords)):
            # extract an roi from the image
            roi = img[min_roi[i, 0]:max_roi[i, 0], min_roi[i, 1]:max_roi[i, 1], min_roi[i, 2]:max_roi[i, 2]]
            # extract the usable region of mask
            roi_mask = mask[min_id[i, 0]:max_id[i, 0], min_id[i, 1]:max_id[i, 1], min_id[i, 2]:max_id[i, 2]]
            # exctract intensity values
            roi_values = roi[roi_mask]
            # make the corresponding coordinates
            uniq_x = np.arange(start=min_roi[i, 0], stop=max_roi[i, 0])
            uniq_y = np.arange(start=min_roi[i, 1], stop=max_roi[i, 1])
            uniq_z = np.arange(start=min_roi[i, 2], stop=max_roi[i, 2])
            roi_coords_z, roi_coords_y, roi_coords_x = np.meshgrid(uniq_x, uniq_y, uniq_z, indexing='ij')
            mask_coords_z = roi_coords_z[roi_mask]
            mask_coords_y = roi_coords_y[roi_mask]
            mask_coords_x = roi_coords_x[roi_mask]
            # save the 1D arrays
            rois_values.append(roi_values)
            px_coords_z.append(mask_coords_z)
            px_coords_y.append(mask_coords_y)
            px_coords_x.append(mask_coords_x)
        px_coords = [px_coords_z, px_coords_y, px_coords_x]
    
    elif n_dim == 2:
        for i in range(len(coords)):
            # extract an roi from the image
            roi = img[min_roi[i, 0]:max_roi[i, 0], min_roi[i, 1]:max_roi[i, 1]]
            # extract the usable region of mask
            roi_mask = mask[min_id[i, 0]:max_id[i, 0], min_id[i, 1]:max_id[i, 1]]
            # exctract intensity values
            roi_values = roi[roi_mask]
            # make the corresponding coordinates
            uniq_x = np.arange(start=min_roi[i, 0], stop=max_roi[i, 0])
            uniq_y = np.arange(start=min_roi[i, 1], stop=max_roi[i, 1])
            roi_coords_y, roi_coords_x = np.meshgrid(uniq_x, uniq_y, indexing='ij')
            mask_coords_y = roi_coords_y[roi_mask]
            mask_coords_x = roi_coords_x[roi_mask]
            # save the 1D arrays
            rois_values.append(roi_values)
            px_coords_y.append(mask_coords_y)
            px_coords_x.append(mask_coords_x)
        px_coords = [px_coords_y, px_coords_x]

    return rois_values, px_coords


def filter_dog(img, filter_sigma_small, filter_sigma_large, 
               pixel_sizes, sigma_cutoff=2):
    """
    Filter an image (convolve) with a Difference of Gaussian kernel.
    """

    kernel_small = localize.get_filter_kernel(filter_sigma_small, pixel_sizes, sigma_cutoff=sigma_cutoff)
    kernel_large = localize.get_filter_kernel(filter_sigma_large, pixel_sizes, sigma_cutoff=sigma_cutoff)
    img_high_pass = localize.filter_convolve(img, kernel_small)
    img_low_pass = localize.filter_convolve(img, kernel_large)
    img_filtered = img_high_pass - img_low_pass

    return img_filtered


# def threshold_image(img, threshold):
#     img_thresholded = np.array(img, copy=True)
#     img_thresholded[img_thresholded < threshold] = 0
#     return img_thresholded

def adapt_2D_data(img, roi_size, filter_sigma_small, filter_sigma_large, min_spot_sep):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=0)
        data_is_2d = True
    else:
        data_is_2d = False

    if data_is_2d:
        roi_size[0] = 0
        filter_sigma_large[0] = 0
        filter_sigma_small[0] = 0
        min_spot_sep[0] = 0

    return data_is_2d, img, roi_size, filter_sigma_small, filter_sigma_large, min_spot_sep


def find_peaks(img_filtered, threshold, min_spot_sep, pixel_sizes):
    """
    Detect peaks, presumably in a thresholded DoG filtered image.
    """
    # footprint = localize.get_max_filter_footprint(min_separations=min_separations, drs=(1,1,1))
    # # array of size nz, ny, nx of True
    # # or use ndi.maximum_filter(img, footprint=np.ones(min_separations))
    # centers_guess_inds, amps = localize.find_peak_candidates(img, footprint, threshold=threshold, use_gpu_filter=use_gpu_filter)

    # unpack arguments
    z, y, x = localize.get_coords(img_filtered.shape, pixel_sizes)
    # identify candidate spots
    if len(img_filtered.shape) == 3 and len(min_spot_sep) == 2:
        min_spot_sep = np.array([min_spot_sep[0], min_spot_sep[1], min_spot_sep[1]])
    footprint = localize.get_max_filter_footprint(min_spot_sep, pixel_sizes)
    centers_guess_inds, amps = localize.find_peak_candidates(img_filtered, footprint, threshold)

    # real coordinates
    centers_guess = np.stack((z[centers_guess_inds[:, 0], 0, 0],
                              y[0, centers_guess_inds[:, 1], 0],
                              x[0, 0, centers_guess_inds[:, 2]]), axis=1)
    return (z, y, x), centers_guess_inds, centers_guess, amps


def filter_peaks_distance(centers_guess_inds, centers_guess, amps, img_filtered, dxy_min_sep, dz_min_sep):
    inds = np.ravel_multi_index(centers_guess_inds.transpose(), img_filtered.shape)
    weights = img_filtered.ravel()[inds]
    centers_guess, inds_comb = localize.filter_nearby_peaks(centers_guess, dxy_min_sep, dz_min_sep,
                                                            weights=weights, mode="average")
    amps = amps[inds_comb]
    return centers_guess, amps


def merge_peaks(coords, max_z, max_xy, weights=None, method='all_dist', verbose=True):
    """
    Merge peaks that are close to each other.

    Parameters
    ----------
    weight : ndarray
        Either weights of spots or image used to compute their weight.
    """

    if method == 'network':
        merged_centers_inds = localize.filter_nearby_peaks(coords, max_z, max_xy, weights)
        if weights is not None:
            # need ravel_multi_index to get pixel values of weights at several 3D coordinates
            amplitudes_id = np.ravel_multi_index(merged_centers_inds.astype(int).transpose(), weights.shape)
            amps = weights.ravel()[amplitudes_id]
    elif method == 'all_dist':
        merged_centers_inds, inds_comb = localize.filter_nearby_peaks(coords, max_xy, max_z, weights=weights, mode="average")
        if weights is not None:
            amps = weights[inds_comb]
    if verbose:
        nb_points = len(merged_centers_inds)
        print(f"Found {nb_points} points separated by dxy > {max_xy} and dz > {max_z}")
    if weights is not None:
        return merged_centers_inds, amps
    else:
        return merged_centers_inds


def detect_dog_spots(img, sigma_xy_small, sigma_xy_large, 
                    sigma_z_small, sigma_z_large, dog_thresh,
                    min_separations, sigma_cutoff, 
                    return_amplitudes=True, merge_peaks=False, 
                    merge_params=None, verbose=0):
    """
    
    """

    if verbose > 0: print("Filtering with DoG kernel")
    img_filtered = filter_dog(img, sigma_xy_small, sigma_xy_large, 
                              sigma_z_small, sigma_z_large, sigma_cutoff=sigma_cutoff)
    if verbose > 0: print("Thresholding")
    img_filtered = threshold_image(img_filtered, dog_thresh)

    if verbose > 0: print("Finding peaks")
    centers_guess_inds, amps = find_peaks(img, dog_thresh, min_separations, use_gpu_filter=False)

    if merge_peaks:
        if verbose > 0: print("Merging peaks")
        if merge_params is None:
            merge_params = {'weights': None, 
                            'method': 'network', 
                            'verbose': True}
        if 'weigths' in merge_params.keys() and merge_params['weigths'] is not None:
            centers_guess_inds, amps = merge_peaks(centers_guess_inds, merge_peaks_z, 
                                                   merge_peaks_xy, **merge_params)
        else:
            centers_guess_inds = merge_peaks(centers_guess_inds, merge_peaks_z, 
                                             merge_peaks_xy, **merge_params)
    if return_amplitudes:
        return centers_guess_inds, amps
    else:
        return centers_guess_inds


# def shift_coordinates(spots_coords, tile_coords, format='pair'):
    
#     if format == 'pair':
#         spots_coords = spots_coords + tile_coords[:, 0, :]
#     elif format == 'single':
#         spots_coords = spots_coords + tile_coords
#     return spots_coords


# def fit_gaussian_spot_legacy(img, centers, fit_roi_sizes, min_fit_roi_sizes, amps, sigma_xy, sigma_z, 
#                       return_fit_vars=True, verbose=0):
    
#     # we don't return roi_sizes because we would have to manage it in
#     # the detect_spots_tile function, whereas another method could not output it
#     if verbose > 0: print("Computing ROIs' coordinates")
#     roi_coords, roi_sizes  = get_roi_coordinates(
#         centers = centers, 
#         sizes = fit_roi_sizes, 
#         max_coords_val = np.array(img.shape) - 1, 
#         min_sizes = min_fit_roi_sizes,
#     )
#     # guess center *in* each individual ROI
#     centers_guess = (roi_sizes / 2)
    
#     # Gaussian fit to find center of each spot
#     all_res = []
#     chi_squared = []
#     for i in range(len(roi_coords)):
#         # extract ROI
#         roi = extract_ROI(img, roi_coords[i])
#         # fit gaussian in ROI
#         init_params = np.array([
#             amps[i], 
#             centers_guess[i, 2],
#             centers_guess[i, 1],
#             centers_guess[i, 0],
#             sigma_xy, 
#             sigma_z, 
#             roi.min(),
#         ])
#         fit_results = localize.fit_gauss_roi(
#             roi, 
#             (localize.get_coords(roi_sizes[i], drs=[1, 1, 1])), 
#             init_params,
#             fixed_params=np.full_like(init_params, False),
#         )
#         chi_squared.append(fit_results['chi_squared'])
#         all_res.append(fit_results['fit_params'])
        
#     # process all the results
#     all_res = np.array(all_res)
#     amplitudes = all_res[:, 0]
#     centers = all_res[:, 3:0:-1]
#     sigmas_xy = all_res[:, 4]
#     sigmas_z = all_res[:, 5]
#     # offsets = all_res[:, 6]
#     chi_squared = np.array(chi_squared)
#     # distances from initial guess
#     dist_center = np.sqrt(np.sum((centers - centers_guess)**2, axis=1))
#     # add origin coordinates of each ROI
#     centers = centers + roi_coords[:, 0, :]
#     # composed variables for filtering
#     sigma_ratios = sigmas_z / sigmas_xy
    
#     fit_vars = {
#     'amplitudes': amplitudes,
#     'sigmas_xy': sigmas_xy,
#     'sigmas_z': sigmas_z,
#     # 'offsets': offsets,
#     'chi_squared': chi_squared,
#     'dist_center': dist_center,
#     'sigma_ratios': sigma_ratios,
#     }        
    
#     if return_fit_vars:
#         return centers, fit_vars
#     else:
#         return centers

def fit_gaussian_spot(img, centers_guess, axes, roi_size_pix, amps, sigma_xy, sigma_z, 
                      dc, sf=1, angles=(0., 0., 0.), data_is_2d=False, 
                      return_fit_vars=True, verbose=0):
    """
    axes : (z, y, x)
    """
    
    # ------------ Prepare ROIs ------------
    # we don't return roi_sizes because we would have to manage it in
    # the detect_spots_tile function, whereas another method could not output it
    if verbose > 0: print("Computing ROIs' coordinates")
    # localize_psf method:
    rois, img_rois, coords = zip(*[localize.get_roi(c, img, axes, roi_size_pix) for c in centers_guess])
    zrois, yrois, xrois = zip(*coords)
    rois = np.asarray(rois)

    # extract guess values
    # TODO: compare fit quality and speed with simpler guesses
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

    # ------------ Fit ------------
    if verbose:
        print("starting fitting for %d rois" % centers_guess.shape[0])
    tstart = time.perf_counter()

    fixed_params = [False] * 7

    # if 2D, don't want to fit cz or sz
    if data_is_2d:
        fixed_params[5] = True
        fixed_params[3] = True

    fit_params, fit_states, chi_squared, niters, fit_t = localize.fit_gauss_rois(img_rois, (zrois, yrois, xrois),
                                                                              init_params, estimator="LSE",
                                                                              sf=sf, dc=dc, angles=angles,
                                                                              fixed_params=fixed_params)

    tend = time.perf_counter()
    if verbose:
        print("Localization took %0.2fs" % (tend - tstart))
        
    # process all the results
    all_res = np.array(fit_params)
    amplitudes = all_res[:, 0]
    centers = all_res[:, 3:0:-1]
    sigmas_xy = all_res[:, 4]
    sigmas_z = all_res[:, 5]
    offsets = all_res[:, 6]
    chi_squared = np.array(chi_squared)
    # distances from initial guess
    dist_center = np.sqrt(np.sum((centers - centers_guess)**2, axis=1))
    # composed variables for filtering
    sigma_ratios = sigmas_z / sigmas_xy
    
    fit_vars = {
    'amplitudes': amplitudes,
    'sigmas_xy': sigmas_xy,
    'sigmas_z': sigmas_z,
    'offsets': offsets,
    'chi_squared': chi_squared,
    'dist_center': dist_center,
    'sigma_ratios': sigma_ratios,
    }        
    
    if return_fit_vars:
        return centers, fit_vars
    else:
        return centers


def detect_spots_tile(tile, tile_coords=None, 
                      roi_method=detect_dog_spots, roi_kwargs=None,
                      center_method=fit_gaussian_spot, center_kwargs=None,
                      filter_method=None, filter_kwargs=None):
    """
    Find spots in a region of interest.
    
    Parameters
    ----------
    tile : numpy.ndarray
        A ND image where we want to find spots.
    tile_coords : Union[Tuple, List, np.ndarray]
        The coordinates of the 'lowest' corner of the tile in the
        bigger image it is extracted from.
    roi_method : fct
        Method used to define ROI around detect potential spots.
    roi_kwargs : dict
        Optional arguments for the blob detection method.
    center_method : fct
        Method used to decipher more precisely spots coordinates
    center_kwargs : dict
        Optional arguments for the center method.
    filter_method : str
        Method used to discard wrong spots.
    filter_kwargs : str
        Optional arguments for the filter method.
    
    Returns
    -------
    spots_coords : numpy.ndarray
        The coordinates in the bigger image reference system of all spots.
    """
    
    if roi_method is not None:
        rois_coords = roi_method(tile, roi_kwargs)
        spots_coords = center_method(tile, rois_coords, **center_kwargs)
    else:
        # case where no pre-detection of ROIs is needed
        spots_coords = center_method(tile, **center_kwargs)
        
    if tile_coords is not None:
        spots_coords = shift_coordinates(spots_coords, tile_coords)
    
    if filter_method is not None:
        spots_coords = filter_method(spots_coords, **filter_kwargs)
    
    return spots_coords

def merge_spots_coords(all_coords):
    """
    Merge a list of spots coordinates into a single array.
    Useful to aggregate data analyses distributed on several places.
    """
    return np.vstack(all_coords)


def make_filter_spots(filter_vars, filter_params):
    """
    Filter out spots based on gaussian fit results.
    
    Parameters
    ----------
    filter_vars : dict
        Variables used to compute boolean filters.
    filter_params : dict
        Parameters (thresholds) applied to their corresponding variables.
    
    Return
    ------
    spot_select : array
        Bolean vector of spots to keep.
    """

    # list of boolean filters for all spots thresholds
    selectors = []
    if filter_params['use_amplitude_min']:
        selectors.append(filter_vars['amplitudes'] >= filter_params['amplitude_range'][0])
    if filter_params['use_amplitude_max']:
        selectors.append(filter_vars['amplitudes'] <= filter_params['amplitude_range'][1])
    if filter_params['use_sigma_xy_min']:
        selectors.append(filter_vars['sigmas_xy'] >= filter_params['sigma_xy_range'][0])
    if filter_params['use_sigma_xy_max']:
        selectors.append(filter_vars['sigmas_xy'] <= filter_params['sigma_xy_range'][1])
    if filter_params['use_sigma_z_min']:
        selectors.append(filter_vars['sigmas_z'] >= filter_params['sigma_z_range'][0])
    if filter_params['use_sigma_z_max']:
        selectors.append(filter_vars['sigmas_z'] <= filter_params['sigma_z_range'][1])
    if filter_params['use_sigma_ratio_min']:
        selectors.append(filter_vars['sigma_ratios'] >= filter_params['sigma_ratio_range'][0])
    if filter_params['use_sigma_ratio_max']:
        selectors.append(filter_vars['sigma_ratios'] <= filter_params['sigma_ratio_range'][1])
    if filter_params['use_chi_squared']:
        selectors.append(filter_vars['chi_squared'] >= filter_params['chi_squared'])
    if filter_params['use_dist_center']:
        selectors.append(filter_vars['dist_center'] <= filter_params['dist_center'])

    if len(selectors) == 0:
        print("No filter is active")
        return None
    else:
        spot_select = np.logical_and.reduce(selectors)
        return spot_select


def apply_filter_spots(spots_coords, spot_select):
    return spots_coords[spot_select, :]


def make_detection_df(centers, fit_vars=None, spot_select=None):

    if centers.shape[1] == 3:
        df_spots = pd.DataFrame({
            'z': centers[:,0],
            'y': centers[:,1],
            'x': centers[:,2],
        })
    else:
        df_spots = pd.DataFrame({
            'y': centers[:,0],
            'x': centers[:,1],
        })
    if fit_vars is not None:
        df_spots = df_spots.join(pd.DataFrame(fit_vars))
    if spot_select is None:
        spot_select = np.full(len(centers), np.nan)

    return df_spots


# ------ Nearby spot merging functions ------

def compute_distances(source, target, method='xy_z_orthog', dist_fct='euclidian', tilt_vector=None):
    """
    Parameters
    ----------
    source : ndarray
        Coordinates of the first set of points.
    target : ndarray
        Coordinates of the second set of points.
    method : str
        Method used to compute distances. If 'xyz', standard distances are computed considering all axes
        simultaneously. If 'xy_z_orthog' 2 distances are computed, for the xy pkane and along the z axis 
        respectively. If 'xy_z_tilted' 2 distances are computed for the tilted plane and  its normal axis.
    
    Example
    -------
    >>> source = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    >>> target = np.array([[0, 0, 0], [-3, 0, 2], [0, 0, 10]])
    >>> distance(source, target)
        (array([0, 4, 0]), array([0., 2., 5.]))
    >>> distance(source, target, dist_fct='L1')
        (array([0, 4, 0]), array([0, 2, 7]))
    
    """
    if method == 'xyz':
        if dist_fct == 'euclidian':
            dist = np.sqrt(np.sum((source - target)**2, axis=1))
        elif dist_fct == 'L1':
            dist = np.sum((source - target), axis=1)
        else:
            dist = dist_fct(source, target, axis=1)
        return dist
    elif method == 'xy_z_orthog':
        if dist_fct == 'euclidian':
            dist_xy = np.sqrt(np.sum((source[:, 1:] - target[:, 1:])**2, axis=1))
            dist_z = np.abs(source[:, 0] - target[:, 0])
        elif dist_fct == 'L1':
            dist_xy = np.sum(np.abs((source[:, 1:]  - target[:, 1:])), axis=1)
            dist_z = np.abs(source[:, 0] - target[:, 0])
        else:
            dist_xy = dist_fct(source[:, 1:], target[:, 1:], axis=1)
            dist_z = dist_fct(source[:, 0], target[:, 0])
        return dist_z, dist_xy
    elif method == 'xy_z_tilted':
        raise NotImplementedError("Method 'xy_z_tilted' will be implemented soon")


def cut_graph_bidistance(dist_z, dist_xy, max_z, max_xy, pairs=None):
    """
    Apply 2 thresholds on distances, along the z axis and in the xy plane,
    to cut a graph of closest neighbors, i.e. to trim edges.

    Parameters
    ----------
    dist_z : array
        Distances between nodes along the z axis.
    dist_xy : array
        Distances between nodes in the xy plane.
    max_z : float
        Distance threshold along the z axis.
    max_xy : float
        Distance threshold in the xy plane.
    pairs : ndarray, optionnal
        Array of pairs of nodes' indices defining the network, of shape nb_nodes x 2.
        If not None, this array is filtered and returned in addition to the boolean filter.
    
    Returns
    -------
    select : array
        Boolean filter used to select pairs of nodes considered as close to each other.
    filtered_pairs : ndarray
        Filtered array of pairs of nodes' indices that are close to each other.
    """

    select = np.logical_and(dist_z <= max_z, dist_xy  <= max_xy)
    if pairs is not None:
        filtered_pairs = pairs[select, :]
        return select, pairs
    else:
        return select


def find_neighbors(pairs, n):
    """
    Return the list of neighbors of a node in a network defined 
    by edges between pairs of nodes. 
    
    Parameters
    ----------
    pairs : array_like
        Pairs of nodes' id that define the network's edges.
    n : int
        The node for which we look for the neighbors.
        
    Returns
    -------
    neigh : array_like
        The indices of neighboring nodes.
    """
    
    left_neigh = pairs[pairs[:,1] == n, 0]
    right_neigh = pairs[pairs[:,0] == n, 1]
    neigh = np.hstack( (left_neigh, right_neigh) ).flatten()
    
    return neigh


def neighbors_k_order(pairs, n, order):
    """
    Return the list of up the kth neighbors of a node 
    in a network defined by edges between pairs of nodes
    
    Parameters
    ----------
    pairs : array_like
        Pairs of nodes' id that define the network's edges.
    n : int
        The node for which we look for the neighbors.
    order : int
        Max order of neighbors.
        
    Returns
    -------
    all_neigh : list
        The list of lists of 1D array neighbor and the corresponding order
    
    
    Examples
    --------
    >>> pairs = np.array([[0, 10],
                        [0, 20],
                        [0, 30],
                        [10, 110],
                        [10, 210],
                        [10, 310],
                        [20, 120],
                        [20, 220],
                        [20, 320],
                        [30, 130],
                        [30, 230],
                        [30, 330],
                        [10, 20],
                        [20, 30],
                        [30, 10],
                        [310, 120],
                        [320, 130],
                        [330, 110]])
    >>> neighbors_k_order(pairs, 0, 2)
    [[array([0]), 0],
     [array([10, 20, 30]), 1],
     [array([110, 120, 130, 210, 220, 230, 310, 320, 330]), 2]]
    """
    
    # all_neigh stores all the unique neighbors and their oder
    all_neigh = [[np.array([n]), 0]]
    unique_neigh = np.array([n])
    
    for k in range(order):
        # detected neighbor nodes at the previous order
        last_neigh = all_neigh[k][0]
        k_neigh = []
        for node in last_neigh:
            # aggregate arrays of neighbors for each previous order neighbor
            neigh = np.unique(find_neighbors(pairs, node))
            k_neigh.append(neigh)
        # aggregate all unique kth order neighbors
        if len(k_neigh) > 0:
            k_unique_neigh = np.unique(np.concatenate(k_neigh, axis=0))
            # select the kth order neighbors that have never been detected in previous orders
            keep_neigh = np.in1d(k_unique_neigh, unique_neigh, invert=True)
            k_unique_neigh = k_unique_neigh[keep_neigh]
            # register the kth order unique neighbors along with their order
            all_neigh.append([k_unique_neigh, k+1])
            # update array of unique detected neighbors
            unique_neigh = np.concatenate([unique_neigh, k_unique_neigh], axis=0)
        else:
            break
        
    return all_neigh


def flatten_neighbors(all_neigh):
    """
    Convert the list of neighbors 1D arrays with their order into
    a single 1D array of neighbors.

    Parameters
    ----------
    all_neigh : list
        The list of lists of 1D array neighbor and the corresponding order.

    Returns
    -------
    flat_neigh : array_like
        The indices of neighboring nodes.
        
    Examples
    --------
    >>> all_neigh = [[np.array([0]), 0],
                     [np.array([10, 20, 30]), 1],
                     [np.array([110, 120, 130, 210, 220, 230, 310, 320, 330]), 2]]
    >>> flatten_neighbors(all_neigh)
    array([  0,  10,  20,  30, 110, 120, 130, 210, 220, 230, 310, 320, 330])
        
    Notes
    -----
    Code from the mosna library https://github.com/AlexCoul/mosna
    """
    
    list_neigh = []
    for neigh, order in all_neigh:
        list_neigh.append(neigh)
    flat_neigh = np.concatenate(list_neigh, axis=0)

    return flat_neigh


def merge_nodes(coords, weight):
    """
    Merge nodes coordinates by averaging them.

    Parameters
    ----------
    coords : ndarray
        Coordinates of nodes, array of shape nb_nodes x 3.
    weight : array
        Weight of nodes for coordinates averaging, 
        array fo shape nb_nodes x 1.

    Returns
    -------
    merged_coords : ndarray
        Coordinates of merged nodes.
    
    Examples
    --------
    >>> coords = np.array([[0, 0, 0], [2, -4, 8]])
    >>> weight = np.array([1, 1]).reshape((len(coords), -1))
    >>> merge_nodes(coords, weight)
    array([ 1., -2.,  4.])
    """

    merged_coords = np.sum(coords * weight, axis=0) / weight.sum()
    return merged_coords


def merge_cluster_nodes(coords, pairs, weights=None, split_big_clust=False, cluster_size=None):
    """
    Merge nodes that are in the same connected cluster, for all cluster in a graph.

    Parameters
    ----------
    coords : ndarray
        Coordinates of nodes, array of shape nb_nodes x 3.
    pairs : ndarray
        Array of pairs of nodes' indices defining the network, of shape nb_nodes x 2.
    weight : array
        Weight of nodes for coordinates averaging. The image intensity at nodes
        coordinates can be used as weights.

    Returns
    -------
    merged_coords : ndarray
        Coordinates of merged nodes.
    """

    if split_big_clust and cluster_size is None:
        raise ValueError("`cluster_size` has to be given to split big clusters")
    nb_nodes = len(coords)
    if weights is None:
        weights = np.ones(nb_nodes)
    # make list of nodes indices on which we will iterate
    iter_nodes = np.arange(nb_nodes)
    # variable storing new merged coordinates
    merged_coords = []
    # variable storing nodes that have been added, directly or undirectly (merged)
    remaining = np.full(shape=nb_nodes, fill_value=True, dtype=bool)
    # for each node, detect all its connected neighbors, even indirectly
    for node_id in iter_nodes:
        # check if we have processed this node
        if remaining[node_id]:
            detected_neighbors = flatten_neighbors(neighbors_k_order(pairs, node_id, nb_nodes))
            # mark these coordinates nodes indices to avoid reprocessing the same neighbors
            select = np.isin(iter_nodes, detected_neighbors, assume_unique=True, invert=True)
            remaining[~select] = False
            # merge nodes coordinates
            if len(detected_neighbors) == 1:
                merged_coords.append(coords[node_id])
            else:
                # detect if cluster likely contains multiple spots
                if split_big_clust:
                    # work on it latter, for now use small distance thresholds
                    # actually merge peaks
                    cluster_coords = merge_nodes(coords[detected_neighbors], 
                                                 weights[detected_neighbors].reshape(-1, 1))
                else:
                    cluster_coords = merge_nodes(coords[detected_neighbors], 
                                                 weights[detected_neighbors].reshape(-1, 1))
                merged_coords.append(cluster_coords)
    merged_coords = np.vstack(merged_coords)
    return merged_coords


# def filter_nearby_peaks(coords, max_z, max_xy, weight_img=None,
#                         split_big_clust=False, cluster_size=None):
#     """
#     Merge nearby peaks in an image by building a radial distance graph and cutting it given
#     distance thresholds in the xy plane and along the z axis.

#     Parameters
#     ----------
#     coords : ndarray
#         Coordinates of nodes, array of shape nb_nodes x 3.
#     max_z : float
#         Distance threshold along the z axis.
#     max_xy : float
#         Distance threshold in the xy plane.
#     weight_img : ndarray
#         Image used to find peaks, now used to weight peaks coordinates during merge.
#         If None, equal weight is given to peaks coordinates.
#     split_big_clust : bool
#         If True, cluster big enough to contain multiple objects of interest (like spots)
#         are split into sub-clusters.
#     cluster_size : list | array
#         The threshold z and x/y size of clusters above which they are split.
    
#     Returns
#     -------
#     merged_coords : ndarray
#         The coordinates of merged peaks.
#     """

    # TODO: check coords without neighbors
    # TODO: check what happens to spots without neighbors

    # build the radial distance network using the bigest radius: max distance along z axis
    pairs = ty.build_rdn(coords=coords, r=max_z)
    if len(pairs) == 0:
        # all nodes are well separated from each other, do nothing
        merged_coords = coords
    else:
        # fuse nodes that have too close neighbors
        source = coords[pairs[:, 0]]
        target = coords[pairs[:, 1]]
        # compute the 2 distances arrays
        dist_z, dist_xy = compute_distances(source, target)
        # perform grph cut from the 2 distance thresholds
        _, pairs = cut_graph_bidistance(dist_z, dist_xy, max_z, max_xy, pairs=pairs)

        if weight_img is not None:
            # need ravel_multi_index to get pixel values of weight_img at several 3D coordinates
            amplitudes_id = np.ravel_multi_index(coords.transpose(), weight_img.shape)
            weights = weight_img.ravel()[amplitudes_id]
        else:
            weights = None  # array of ones will be generated in merge_cluster_nodes
        # merge nearby nodes coordinates
        merged_coords = merge_cluster_nodes(coords, pairs, weights,
                                            split_big_clust=split_big_clust, 
                                            cluster_size=cluster_size)
        # # merge originally isolated spots and newly fused spots
        # if select.sum() > 0:
        #     merged_coords = np.vstack([keep_coords, merged_coords])

#     return merged_coords

# ------ parameters saving and loading ------

def load_parameters(path_load, trim_widget_label=True, module_variables=None):
    """
    Load spot detection pipeline parameters.

    Parameters
    ----------
    path_load : str
        Path to the json file containing parameters.
    trim_widget_label : bool
        If True, parameters name containing labels related to
        PyQt widget are removed or replaced.
    module_variables : None | str
        If not None, defines all parameters as a module's variable, which
        name (like '__main__') is given by `module_variables`.
        One can use `sys.modules[__name__]` to obtain the module's name.
    
    Returns
    -------
    params : dict
        Parameters of the spot detection pipeline
    """

    with open(path_load, "r") as read_file:
        params = json.load(read_file)
    
    if trim_widget_label:
        for key, val in list(params.items()):   # copy as list as dictionnary changes size
            new_key = key.replace('txt_', '').\
                          replace('sld_', '').\
                          replace('filter_', '').\
                          replace('chk_', 'use_')
            params[new_key] = params[key]
            del params[key]

    if not ('sigma_z' in params and 'sigma_xy' in params):
        if trim_widget_label:
            sz = params['spot_size_z']
            sxy = params['spot_size_xy']
            sigma_ratio = params['sigma_ratio']
        else:
            sz = params['txt_spot_size_z']
            sxy = params['txt_spot_size_xy']
            sigma_ratio = params['txt_sigma_ratio']

        params['sigma_z'], params['sigma_xy'] = make_sigmas(sz, sxy, sigma_ratio)[:2]
    
    if module_variables is not None:
        for key, val in params.items():
            setattr(module_variables, key, val)
    
    return params
