"""
Functions to decode barcoded FISH experiments.
"""
import numpy as np
import pandas as pd
import json
import os
import time
import warnings
import numpy as np
import scipy.distance
import joblib
import matplotlib.pyplot as plt

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