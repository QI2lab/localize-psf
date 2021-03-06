"""
Tools for dealing with regions of interest (ROI's)
"""
import numpy as np


def roi2global(coords_roi: list[float], roi: list[int]):
    """
    Convert from ROI coordinates to global coordinates. i.e. if we have an array M then
    ROI(M)[c1, c2, ..., cn] = M[c1_full, c2_full, ..., cn_full]
    Inverse function of global2roi().

    :param coords_roi: [c1, c2, ..., cn]
    :param roi: [c1_start, c1_end, c2_start, c2_end, ..., cn_start, cn_end]
    :return coords_full: [c1_full, c2_full, ..., cn_full]
    """
    coords_full = []
    for ii, c in enumerate(coords_roi):
        coords_full.append(roi[2*ii] + c)

    return coords_full


def global2roi(coords_full: list[float], roi: list[int]):
    """
    Convert from global coordinates to ROI coordinates. i.e. if we have an array M, then
    M[c1, c2, ..., cn] = ROI(M)[c1_xform, c2_xform, ..., cn_xform]
    Inverse function of roi2global()

    :param coords_full: [c1, c2, ..., cn]
    :param roi: [c1_start, c1_end, c2_start, c2_end, ..., cn_start, cn_end]
    :return coords_roi: [c1_xform, c2_xform, ..., cn_xform]
    """

    coords_roi = []
    for ii, c in enumerate(coords_full):
        coords_roi.append(c - roi[2*ii])

    return coords_roi


def get_centered_roi(centers: list, sizes: list[int], min_vals: list[int] = None, max_vals: list[int] = None):
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


def cut_roi(roi: list[int], arr: np.ndarray, axes: tuple[int] = None, allow_broadcastable_arrays: bool = True) -> np.ndarray:
    """
    Return region of interest from an array

    @param roi: [a0_start, a0_end, a1_start, a1_end, ..., am_start, am_end]
    @param arr: array, which must have dimension m or greater
    @param axes: which axes are to be sliced by the ROI. Be default these are the last m axes of the array
     dimensions. If these are allowed, they will not be affected by the slicing operations but will remain unit size
    @param allow_broadcastable_arrays: whether or not to accept arrays which have size 1 along some of the
    @return arr_roi: array roi
    """
    if not np.mod(len(roi), 2) == 0:
        raise ValueError("roi array length must be even")

    nroi_dim = len(roi) // 2
    if nroi_dim > arr.ndim:
        raise ValueError("roi has dimension %d, which is too large for array of dimension %d" % (nroi_dim, arr.ndim))

    # default to last nroi_dim axes of array
    if axes is None:
        axes = np.arange(-nroi_dim, 0)
    axes[axes < 0] = axes[axes < 0] + arr.ndim

    # base is entire array
    slices = [slice(0, arr.shape[ii]) for ii in range(arr.ndim)]
    # update whichever axes need updating
    for ii, ax in enumerate(axes):
        # get slices, unless array has unit size over this dimension and then we will assume is broadcasting ...
        if allow_broadcastable_arrays and arr.shape[ax] == 1:
            slices[ax] = slice(0, 1)
        else:
            slices[ax] = slice(roi[2*ii], roi[2*ii + 1])

    return arr[tuple(slices)]


def get_roi_size(sizes: list[float], dc: float, dz: float, ensure_odd: bool = True):
    """
    Get closest larger ROI size in pixels given a set of sizes in real units

    @param sizes: [sz, sy, sx], ROI sizes in real units
    @param dc: pixel size
    @param dz: z-spacing size
    @param bool ensure_odd: enforce only odd ROI sizes if true
    @return roi_sizes: [n0, n1, n2]
    """
    n0 = int(np.ceil(sizes[0] / dz))
    n1 = int(np.ceil(sizes[1] / dc))
    n2 = int(np.ceil(sizes[2] / dc))

    if ensure_odd:
        n0 += (1 - np.mod(n0, 2))
        n1 += (1 - np.mod(n1, 2))
        n2 += (1 - np.mod(n2, 2))

    roi_sizes = [n0, n1, n2]

    return roi_sizes
