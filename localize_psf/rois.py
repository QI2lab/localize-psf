"""
Tools for dealing with regions of interest (ROI's)
"""
from typing import Optional, Union
from collections.abc import Sequence
import numpy as np
import warnings
from numba import njit


def roi2global(coords_roi: Union[Sequence[float], Sequence[int]],
               roi: Sequence[int],
               ensure_in_roi: bool = False) -> np.ndarray:
    """
    Convert from ROI coordinates to global coordinates. i.e. if we have an array M then
    ROI(M)[c1, c2, ..., cn] = M[c1_full, c2_full, ..., cn_full]
    Inverse function of global2roi().

    :param coords_roi: an ncenters x ndims array e.g.
      [[c1, c2, ..., cn], [...]]
    :param roi: a 2*ndims or ncenters x 2*ndims array e.g.
      [c1_start, c1_end, c2_start, c2_end, ..., cn_start, cn_end]
    :param ensure_in_roi: whether to set values outside of ROI to -1.
    :return coords_full: [c1_full, c2_full, ..., cn_full]
    """
    coords_roi = np.asarray(coords_roi)
    roi = np.asarray(roi, dtype=int)

    coords_full = coords_roi + roi[..., ::2]

    if ensure_in_roi:
        # points outside of ROI to nan
        coords_full[coords_full >= roi[..., 1::2]] = -1

    return coords_full


def global2roi(coords_full: Union[Sequence[float], Sequence[int]],
               roi: Sequence[int],
               ensure_in_roi: bool = False) -> np.ndarray:
    """
    Convert from global coordinates to ROI coordinates. i.e. if we have an array M, then
    M[c1, c2, ..., cn] = ROI(M)[c1_xform, c2_xform, ..., cn_xform]
    Inverse function of roi2global()

    :param coords_full: [c1, c2, ..., cn]
    :param roi: [c1_start, c1_end, c2_start, c2_end, ..., cn_start, cn_end]
    :param ensure_in_roi: whether to set values outside of ROI to -1.
    :return coords_roi: [c1_xform, c2_xform, ..., cn_xform]
    """

    coords_full = np.asarray(coords_full)
    roi = np.asarray(roi, dtype=int)

    coords_roi = coords_full - roi[..., ::2]

    if ensure_in_roi:
        coords_roi[coords_full >= roi[..., 1::2]] = -1

    return coords_roi


def get_centered_roi(centers: Union[Sequence[float], Sequence[int]],
                     sizes: Sequence[int],
                     min_vals: Optional[Sequence[int]] = None,
                     max_vals: Optional[Sequence[int]] = None):
    """
    Get end points of an roi centered about centers (as close as possible) with length sizes.
    If the ROI size is odd, the ROI will be perfectly centered. Otherwise, the centering will
    be approximation

    roi = [start_0, end_0, start_1, end_1, ..., start_n, end_n]

    Slicing an array as A[start_0:end_0, start_1:end_1, ...] gives the desired ROI.
    Note that following python array indexing convention end_i are NOT contained in the ROI

    :param centers: list of centers [c1, c2, ..., cn]
    :param sizes: list of sizes [s1, s2, ..., sn]
    :param min_vals: list of minimimum allowed index values for each dimension
    :param max_vals: list of maximum allowed index values for each dimension
    :return roi: [start_0, end_0, start_1, end_1, ..., start_n, end_n]
    """

    warnings.warn("get_centered_roi() is deprecated and will be removed soon. Please use get_centered_rois() instead.")

    roi = []
    # for c, n in zip(centers, sizes):
    for ii in range(len(centers)):
        c = centers[ii]
        n = sizes[ii]

        # get ROI closest to centered
        end_test = np.round(c + (n - 1) / 2) + 1
        end_err = np.mod(end_test, 1)
        start_test = np.round(c - (n - 1) / 2)
        start_err = np.mod(start_test, 1)

        if end_err > start_err:
            start = start_test
            end = start + n
        else:
            end = end_test
            start = end - n

        if min_vals is not None:
            if start < min_vals[ii]:
                start = min_vals[ii]

        if max_vals is not None:
            if end > max_vals[ii]:
                end = max_vals[ii]

        roi.append(int(start))
        roi.append(int(end))

    return roi


def get_centered_rois(centers: Union[np.ndarray[int], np.ndarray[float]],
                      sizes: np.ndarray[int],
                      min_vals: Optional[np.ndarray[int]] = None,
                      max_vals: Optional[np.ndarray[int]] = None) -> np.ndarray[int]:
    """
    Get end points of an roi centered about centers (as close as possible) with length sizes.
    If the ROI size is odd, the ROI will be perfectly centered. Otherwise, the centering will
    be approximation

    roi = [start_0, end_0, start_1, end_1, ..., start_n, end_n]

    Slicing an array as A[start_0:end_0, start_1:end_1, ...] gives the desired ROI.
    Note that following python array indexing convention end_i are NOT contained in the ROI

    :param centers: num_rois x ndims list of centers [[a0, a1, ..., an], [b0, b1, ..., bn], ...]
      Will also accept multiple initial dimensions
    :param sizes: num_rois x ndims ROI size for each center and dimension. Broadcastable to same size as centers
    :param min_vals: num_rois x ndims. Broadcastable to same size as centers
    :param max_vals: num_rois x ndims. Broadcastable to same size as centers
    :return rois: num_rois x 2*ndims
    """

    centers = np.atleast_2d(centers)
    sizes = np.atleast_2d(sizes)

    first_shape = centers.shape[0:-1]
    ndim = centers.shape[-1]

    if min_vals is None:
        if np.issubdtype(centers.dtype, float):
            min_vals = np.full_like(centers, -np.inf)
        elif np.issubdtype(centers.dtype, np.integer):
            min_vals = np.full_like(centers, np.iinfo(centers.dtype).min)
        else:
            raise ValueError()

    if max_vals is None:
        if np.issubdtype(centers.dtype, float):
            max_vals = np.full_like(centers, np.inf)
        elif np.issubdtype(centers.dtype, np.integer):
            max_vals = np.full_like(centers, np.iinfo(centers.dtype).max)
        else:
            raise ValueError()

    min_vals = np.atleast_2d(min_vals)
    max_vals = np.atleast_2d(max_vals)

    # check which
    end_test = np.rint(centers + (sizes - 1) / 2) + 1
    end_err = np.mod(end_test, 1)
    start_test = np.rint(centers - (sizes - 1) / 2)
    start_err = np.mod(start_test, 1)

    mask = end_err > start_err
    start = np.where(mask, start_test, end_test - sizes)
    end = start + sizes

    start = np.maximum(start, min_vals)
    end = np.minimum(end, max_vals)

    # rois = np.stack((start, end), axis=-1).reshape((nroi, 2*ndim)).astype(int)
    rois = np.stack((start, end), axis=-1).astype(int).reshape(first_shape + (2 * ndim,))

    return rois


def cut_roi(rois: Sequence[int],
            arr: np.ndarray,
            axes: Optional[Sequence[int]] = None,
            use_numba: bool = False) -> list[np.ndarray]:
    """
    Return regions-of-interest from an array of arbitrary dimension.

    This function supports arrays that are broadcastable to the size of an appropriate array. i.e. if any
    dimension has length 1 then that dimensions will be left alone

    :param rois: [[a0_start, a0_end, a1_start, a1_end, ..., am_start, am_end], [b0_start, ...], ...]
    :param arr: array, which must have dimension m or greater
    :param axes: which axes are to be sliced by the ROI. Be default these are the last m axes of the array
      dimensions. If these are allowed, they will not be affected by the slicing operations but will remain unit size
    :param use_numba: use numba to accelerate this process. This is only supported for ROI's
      of 2- and 3-dimensions. NOTE: due to the compile time of the numba functions, this approach is only likely
      to be faster for very large numbers of ROIs
    :return: list of rois
    """

    ndim = arr.ndim

    rois = np.atleast_2d(rois)
    nroi_dim = rois.shape[-1] // 2

    if rois.ndim == 1:
        nrois = 1
    else:
        nrois = rois.shape[0]

    if not np.mod(rois.shape[-1], 2) == 0:
        raise ValueError("roi array length must be even")

    if nroi_dim > ndim:
        raise ValueError(f"roi has dimension {nroi_dim:d}, which is too large for array of dimension {arr.ndim:d}")

    if use_numba and axes is not None:
        raise ValueError("axes argument is not supported when use_numba=True")

    # default to last nroi_dim axes of array
    if axes is None:
        axes = np.arange(-nroi_dim, 0)
    axes = np.asarray(axes, dtype=int)
    axes[axes < 0] += ndim

    if len(axes) != nroi_dim:
        raise ValueError(f"number of axes {len(axes):d} is not equal to number of roi dimensions {nroi_dim:d}")

    numba_fn_map = {"2": _cut_rois2d,
                    "3": _cut_rois3d}

    if use_numba:
        if str(nroi_dim) in numba_fn_map.keys():
            arrs = numba_fn_map[str(nroi_dim)](rois, arr)
        else:
            raise NotImplementedError(f"use_numba was True, but the requested ROI had {nroi_dim:d} dimensions."
                                      f"numba acceleration is only supported for ROIs of {numba_fn_map.keys()} dimensions")
    else:
        # note: cannot accelerate this with numba because don't know in advance how many dimensions we have
        arrs = []
        for rr in range(nrois):
            roi = rois[rr]

            # base is entire array
            slices = [slice(0, arr.shape[ii]) for ii in range(arr.ndim)]
            # update whichever axes need updating
            for ii, ax in enumerate(axes):
                # get slices, unless array has unit size over this dimension, and then we will assume is broadcasting ...
                if arr.shape[ax] == 1:
                    slices[int(ax)] = slice(0, 1)
                else:
                    slices[int(ax)] = slice(roi[2 * ii], roi[2 * ii + 1])
            arrs.append(arr[tuple(slices)])

    return arrs


@njit()
def _cut_rois3d(rois: Sequence[int],
                arr: np.ndarray) -> list[np.ndarray]:
    """
    numba accelerated helper function for cut_roi()

    :param rois:
    :param arr:
    :return:
    """

    rois = np.atleast_2d(rois)
    nrois = rois.shape[0]

    arrs = []

    if arr.shape[-3] == 1 and arr.shape[-2] == 1 and arr.shape[-1] == 1:
        for rr in range(nrois):
            roi = rois[rr]
            arrs.append(arr)
    elif arr.shape[-3] == 1 and arr.shape[-2] == 1 and arr.shape[-1] != 1:
        for rr in range(nrois):
            roi = rois[rr]
            arrs.append(arr[..., :, :, roi[4]:roi[5]])
    elif arr.shape[-3] == 1 and arr.shape[-2] != 1 and arr.shape[-1] == 1:
        for rr in range(nrois):
            roi = rois[rr]
            arrs.append(arr[..., :, roi[2]:roi[3], :])
    elif arr.shape[-3] != 1 and arr.shape[-2] == 1 and arr.shape[-1] == 1:
        for rr in range(nrois):
            roi = rois[rr]
            arrs.append(arr[..., roi[0]:roi[1], :, ])
    elif arr.shape[-3] == 1 and arr.shape[-2] != 1 and arr.shape[-1] != 1:
        for rr in range(nrois):
            roi = rois[rr]
            arrs.append(arr[..., :, roi[2]:roi[3], roi[4]:roi[5]])
    elif arr.shape[-3] != 1 and arr.shape[-2] != 1 and arr.shape[-1] == 1:
        for rr in range(nrois):
            roi = rois[rr]
            arrs.append(arr[..., roi[0]:roi[1], roi[2]:roi[3], :])
    elif arr.shape[-3] != 1 and arr.shape[-2] == 1 and arr.shape[-1] != 1:
        for rr in range(nrois):
            roi = rois[rr]
            arrs.append(arr[..., roi[0]:roi[1], :, roi[4]:roi[5]])
    else:
        for rr in range(nrois):
            roi = rois[rr]
            arrs.append(arr[..., roi[0]:roi[1], roi[2]:roi[3], roi[4]:roi[5]])

    return arrs


@njit()
def _cut_rois2d(rois: Sequence[int],
                arr: np.ndarray) -> list[np.ndarray]:
    """
    numba accelerated helper function for cut_roi()

    :param rois:
    :param arr:
    :return:
    """

    rois = np.atleast_2d(rois)
    nrois = rois.shape[0]

    arrs = []
    if arr.shape[-2] == 1 and arr.shape[-1] == 1:
        for rr in range(nrois):
            roi = rois[rr]
            arrs.append(arr)
    elif arr.shape[-2] == 1 and arr.shape[-1] != 1:
        for rr in range(nrois):
            roi = rois[rr]
            arrs.append(arr[..., :, roi[2]:roi[3]])
    elif arr.shape[-2] != 1 and arr.shape[-1] == 1:
        for rr in range(nrois):
            roi = rois[rr]
            arrs.append(arr[..., roi[0]:roi[1], :])
    else:
        for rr in range(nrois):
            roi = rois[rr]
            arrs.append(arr[..., roi[0]:roi[1], roi[2]:roi[3]])

    return arrs


def get_roi_size(sizes: Union[Sequence[float], Sequence[int]],
                 drs: Sequence[float],
                 ensure_odd: bool = True) -> np.ndarray:
    """
    Get closest larger ROI size in pixels given a set of sizes in real units

    :param sizes: ROI sizes in real units. Either an ndim array or n x ndim array i.e.
       [[s0, s1, ...], [...], ...]
    :param drs: pixel size along each dimension. Either an ndim array or an n x ndim array i.e.
      [dr0, dr1, ...]
    :param ensure_odd: force ROI sizes to be odd
    :return roi_sizes: [[n0, n1, n2, ...], [...], ...]
    """

    sizes = np.asarray(sizes)
    drs = np.asarray(drs)

    roi_sizes = np.ceil(sizes / drs).astype(int)
    if ensure_odd:
        roi_sizes[np.mod(roi_sizes, 2) == 0] += 1

    return roi_sizes
