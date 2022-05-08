# %%
"""
Pipeline to find spot in a single non-skewed xyz image tile.
The global plan is:
  - [x] spot detection working on single non-skewed tile, no GPU, no multiprocess, no coordinates change, no file handling
    - [x] extract single ct xyz tile with spots
  - [x] add multiple tiles (x, y, z vary), "stich" results with orthogonal change of coordinates
  - [ ] add multiple time steps, channels (need different parameters per channel)
  - [x] add Dask support per xyztc tile and merge results
  - [ ] add GPU support per tile? Manage conflict with multi-processes tile handling.
  - [ ] add support for skewed tiles
    - [x] extract tilted tile
"""

# %%

import numpy as np
import matplotlib.pyplot as plt
import napari
import scipy.signal
import scipy.ndimage

from pathlib import Path
import warnings
import time
import os
import sys
import joblib
import gc

from tifffile import tifffile
import zarr
import dask.array as da
from dask_image.imread import imread
from dask import delayed
from skimage.io.collection import alphanumeric_key
from pycromanager import Dataset
import napari
from napari.qt.threading import thread_worker
from magicgui import magicgui
# from matplotlib.colors import PowerNorm, LinearSegmentedColormap, Normalize
from tiler import Tiler
from tysserand import tysserand as ty
# from mosna import mosna as mo

import localize_psf.rois as roi_fns
from localize_psf import fit
import localize_psf.fit_psf as psf
from localize_psf import localize
import localize_skewed
import image_post_processing as ipp
from image_post_processing import deskew

# %% [markdown]
# ## On single deskewed tile

# %% [markdown]
# ### Extract data

# %%
dir_load = Path('../../../from_server/example_image_deskewed')
round = 1
channel = 2
tile = 0

path_im = dir_load / 'round-{}_channel-{}_tile-{}.zarr'.format(round, channel, tile)
im = da.from_zarr(str(path_im))
# im = da.moveaxis(im, 1, 0)
print(im.shape)
# img = im[:, 1000:1512, 512:1024].compute()
# img = im[:, 1000:1256, 512:768].compute()
start_x = 512
start_y = 1000
size_xy = 128
img = im[128:2*128, start_y:(start_y+size_xy), start_x:(start_x+size_xy)].compute()
print(img.shape)
mini = img.min() # 0
maxi = img.max()
# viewer = napari.Viewer()
# viewer.add_image(
#     img, 
#     contrast_limits=[mini, maxi], 
#     name='ch_' + str(channel), 
#     # colormap=color, 
#     blending='additive',
#     )

# %% [markdown]
# ### Parameters from Peter's code
# 
# Code is:
# ```python                   
# ###############################
# identify candidate points in opm data
# ###############################
# sigma_xy = 0.22 * emission_wavelengths[ch] / na
# sigma_z = np.sqrt(6) / np.pi * ni * emission_wavelengths[ch] / na ** 2
# sigma_xy = psf.na2sxy(na, emission_wavelengths[ch])
# sigma_z = psf.na2sz(na, emission_wavelengths[ch], ni)
# 
# difference of gaussian filer
# filter_sigma_small = (0.5 * sigma_z, 0.25 * sigma_xy, 0.25 * sigma_xy)
# filter_sigma_large = (3 * sigma_z, 3 * sigma_xy, 3 * sigma_xy)
# fit roi size
# roi_size = (5 * sigma_z, 12 * sigma_xy, 12 * sigma_xy)
# assume points closer together than this come from a single bead
# min_spot_sep = (3 * sigma_z, 3 * sigma_xy)
# exclude points with sigmas outside these ranges
# sigmas_min = (0.25 * sigma_z, 0.25 * sigma_xy)
# sigmas_max = (3 * sigma_z, 4 * sigma_xy)
# ```
# 
# With na=1.0 and ni=1.4  
# For ch 0:  
#     - emission_wavelengths: 0.515  
#     - sigma_xy: 0.123  
#     - sigma_z: 0.562  
#     - filter_sigma_small: [0.281 0.031 0.031]  
#     - filter_sigma_large: [1.686 0.368 0.368]  
#     - roi_size: [2.811 1.472 1.472]  
#     - min_spot_sep: [1.686 0.368]  
#     - sigmas_min: [0.141 0.031]  
#     - sigmas_max: [1.686 0.491]  
# 
# For ch 1:  
#     - emission_wavelengths: 0.6  
#     - sigma_xy: 0.143  
#     - sigma_z: 0.655  
#     - filter_sigma_small: [0.327 0.036 0.036]  
#     - filter_sigma_large: [1.965 0.429 0.429]  
#     - roi_size: [3.275 1.715 1.715]  
#     - min_spot_sep: [1.965 0.429]  
#     - sigmas_min: [0.164 0.036]  
#     - sigmas_max: [1.965 0.572]  
# 
# For ch 2:  
#     - emission_wavelengths: 0.68  
#     - sigma_xy: 0.162  
#     - sigma_z: 0.742  
#     - filter_sigma_small: [0.371 0.04  0.04 ]  
#     - filter_sigma_large: [2.227 0.486 0.486]  
#     - roi_size: [3.711 1.944 1.944]  
#     - min_spot_sep: [2.227 0.486]  
#     - sigmas_min: [0.186 0.04 ]  
#     - sigmas_max: [2.227 0.648]  
# 

# %% [markdown]
# ### DoG filter

# %%

# size of spots in pixels
sx = sy = 5
sz = 20
# FWHM = 2.355 x sigma
sigma_xy = sx / 2.355
sigma_z = sz / 2.355
# to reproduce LoG with Dog we need sigma_big = 1.6 * sigma_small
sigma_xy_small = sigma_xy / 1.6**(1/2)
sigma_xy_large = sigma_xy * 1.6**(1/2)
sigma_z_small = sigma_z / 1.6**(1/2)
sigma_z_large = sigma_z * 1.6**(1/2)

filter_sigma_small = (sigma_z_small, sigma_xy_small, sigma_xy_small)
filter_sigma_large = (sigma_z_large, sigma_xy_large, sigma_xy_large)

# %%
print(f"  - sx: {np.round(sx, 3)}  ")
print(f"  - sz: {np.round(sz, 3)}  ")
print(f"  - sigma_xy: {np.round(sigma_xy, 3)}  ")
print(f"  - sigma_z: {np.round(sigma_z, 3)}  ")
print(f"  - sigma_xy_small: {np.round(sigma_xy_small, 3)}  ")
print(f"  - sigma_xy_large: {np.round(sigma_xy_large, 3)}  ")
print(f"  - sigma_z_small: {np.round(sigma_z_small, 3)}  ")
print(f"  - sigma_z_large: {np.round(sigma_z_large, 3)}  ")

# %%
pixel_sizes = (1, 1, 1)
kernel_small = localize.get_filter_kernel(filter_sigma_small, pixel_sizes, sigma_cutoff=2)
kernel_large = localize.get_filter_kernel(filter_sigma_large, pixel_sizes, sigma_cutoff=2)

# viewer = napari.Viewer()
# viewer.add_image(kernel_small, name='kernel small', colormap='green', blending='additive')
# viewer.add_image(kernel_large, name='kernel large', colormap='red', blending='additive')

# %%Can we skip the second convulotion with an "normalized" kernel?
# im_fct = scipy.signal.fftconvolve(img, kernel_small, mode="same") / scipy.signal.fftconvolve(np.ones(img.shape), kernel_small, mode="same")
# kernel_small_normalized = kernel_small - kernel_small.mean()
# im_normalized = scipy.signal.fftconvolve(img, kernel_small_normalized, mode="same")


# viewer = napari.Viewer()
# viewer.add_image(im_fct, name='im_fct')
# viewer.add_image(im_normalized, name='im_normalized')

# %%

img_high_pass = localize.filter_convolve(img, kernel_small, use_gpu=False)
img_low_pass = localize.filter_convolve(img, kernel_large, use_gpu=False)
img_filtered = img_high_pass - img_low_pass

# %%

viewer = napari.Viewer()
viewer.add_image(img_high_pass, name='img_high_pass')
viewer.add_image(img_low_pass, name='img_low_pass')
viewer.add_image(img, name='img')
viewer.add_image(img_filtered, name='img_filtered')

# %% [markdown]
# ### Threshold DoG and local max

# %%
# threshold found with Napari
dog_thresh = 4
img_filtered[img_filtered < dog_thresh] = 0

# %%
min_separations = (10, 3, 3)

footprint = localize.get_max_filter_footprint(min_separations=min_separations, drs=pixel_sizes)
# array of size nz, ny, nx of True

# %%
# to test:
# maxis = scipy.ndimage.maximum_filter(img_filtered, footprint)
# maxis = scipy.ndimage.maximum_filter(img_filtered, footprint=np.ones(min_separations))

# %%
# np.unique(maxis)

# %%
footprint.shape

# %%
# we could remove the thresholding within each find_peak_candidates call
# no: ndimage.maximum_filter returns same image size with real values, need image == im_max
# thus need to filter with threshold to avoid zeros or low values
# TODO: use gradient on whole image could speed up global process
centers_guess_inds, amps = localize.find_peak_candidates(img_filtered, footprint, threshold=dog_thresh)

# %%
centers_guess_inds

# %%
viewer = napari.Viewer()
viewer.add_image(img, name='img')
viewer.add_image(img_filtered, name='img_filtered')
viewer.add_points(centers_guess_inds, name='local maxis', blending='additive', size=3, face_color='r')

# %% [markdown]
# ### Merge peaks

# %% [markdown]
# Here there is not multiple local maxima per spot, so we will make some artificially.

# %%
nb_peaks = len(centers_guess_inds)
sampled_peaks = centers_guess_inds[np.random.choice(nb_peaks, size=int(1.5 * nb_peaks))]
max_shift_z = 20
max_shift_xy = 5
nb_samples = len(sampled_peaks)
z_shifts = np.random.choice(np.arange(-max_shift_z, max_shift_z+1), size=(nb_samples, 1))#.reshape((nb_samples, -1))
xy_shifts = np.random.choice(np.arange(-max_shift_xy, max_shift_xy+1), size=(nb_samples, 2))
peaks_shift = np.hstack([z_shifts, xy_shifts])
shifted_samples = sampled_peaks + peaks_shift
centers_guess_inds_duplic = np.vstack([centers_guess_inds, shifted_samples])

# make sure new spots are not out of the image
for i in range(3):
    centers_guess_inds_duplic[centers_guess_inds_duplic[:, i] < 0, i] = 0
    centers_guess_inds_duplic[centers_guess_inds_duplic[:, i] >= img.shape[i], i] = img.shape[i] -1 

# %%
viewer = napari.Viewer()
viewer.add_image(img, name='img')
viewer.add_points(centers_guess_inds, name='local maxis', blending='additive', size=3, face_color='r')
viewer.add_points(centers_guess_inds_duplic, name='centers_guess_inds_duplic', blending='additive', size=3, face_color='g')

# %% [markdown]
# #### Peaks merging functions

# %% [markdown]
# The idea is to not loop through all peaks, and compute for each of them the distances with all other peaks, but instead build a kNN or radial distance graph.  
# Then we will compute distances along the z axis and in the xy plane, and use 2 distance thresholds to define peaks that are 'nearby'.  
# Then we cut the graph: we trim edges between non nearby peaks.  
# Then we merge peaks that are in the same connected cluster.   
# 
# kNN graph may be problematic, as we may choose a too low k producing oversplit clusters, for instance if 2 pairs of points are in the same spot, but within each pair distances are smaller than across pairs, and we took k=1.  
# Thus we will use the radial distance method, taking the longest distance, the one along the z axis, to build an over connected graph that we will then cut with the bi-distance criterion.

# %%
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

    tot_weight = weight.sum()
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

    nb_nodes = len(coords)
    if weights is None:
        weights = np.ones(nb_nodes)
    # make list of nodes indices on which we will iterate
    iter_nodes = np.arange(nb_nodes)
    # variable storing new merged coordinates
    merged_coords = []
    # for each node, detect all its connected neighbors, even indirectly
    for i in np.arange(nb_nodes):
        # check if we have processed all nodes
        if i >= len(iter_nodes):
            break
        else:
            node_id = iter_nodes[i]
            detected_neighbors = flatten_neighbors(neighbors_k_order(pairs, node_id, nb_nodes))
            # delete these coordinates nodes indices to avoid reprocessing the same neighbors
            select = np.isin(iter_nodes, detected_neighbors, assume_unique=True, invert=True)
            iter_nodes = iter_nodes[select]
            # merge nodes coordinates
            if len(detected_neighbors) == 1:
                merged_coords.append(coords[node_id])
            else:
                # detect if cluster likely contains multiple spots
                if split_big_clust:
                    if cluster_size is None:
                        raise ValueError("`cluster_size` has to be given to split big clusters")
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


def filter_nearby_peaks(coords, max_z, max_xy, weight_img=None,
                        split_big_clust=False, cluster_size=None):
    """
    Merge nearby peaks in an image by building a radial distance graph and cutting it given
    distance thresholds in the xy plane and along the z axis.

    Parameters
    ----------
    coords : ndarray
        Coordinates of nodes, array of shape nb_nodes x 3.
    max_z : float
        Distance threshold along the z axis.
    max_xy : float
        Distance threshold in the xy plane.
    weight_img : ndarray
        Image used to find peaks, now used to weight peaks coordinates during merge.
        If None, equal weight is given to peaks coordinates.
    split_big_clust : bool
        If True, cluster big enough to contain multiple objects of interest (like spots)
        are split into sub-clusters.
    cluster_size : list | array
        The threshold z and x/y size of clusters above which they are split.
    
    Returns
    -------
    merged_coords : ndarray
        The coordinates of merged peaks.
    """

    # build the radial distance network using the bigest radius: max distance along z axis
    pairs = ty.build_rdn(coords=coords, r=max_z)
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

    return merged_coords


# %%
max_z = 10
max_xy = 3
pairs = ty.build_rdn(coords=centers_guess_inds_duplic, r=max_z)

# %%
pairs.shape

# %%
centers_guess_inds_duplic.shape

# %%
source = centers_guess_inds_duplic[pairs[:, 0]]
target = centers_guess_inds_duplic[pairs[:, 1]]

dist_z, dist_xy = compute_distances(source, target)

# %%
select, pairs = cut_graph_bidistance(dist_z, dist_xy, max_z, max_xy, pairs=pairs)
print(pairs.shape)

# %%
viewer = napari.Viewer()
viewer.add_image(img, name='img')
viewer.add_points(centers_guess_inds, name='local maxis', blending='additive', size=3, face_color='r')
viewer.add_points(centers_guess_inds_duplic, name='centers_guess_inds_duplic', blending='additive', size=3, face_color='g')

napari_coords = ty.convert_nodes_tys_to_nap(centers_guess_inds_duplic)
# napari_edges = ty.convert_edges_tys_to_nap(centers_guess_inds_duplic, pairs)

def make_annotation_dict(coords, pairs=None,
                         nodes_class=None,
                         nodes_class_color_mapper=None,
                         ):
    """
    Create a dictionnary of annotations from tysserand network objects.
    """

    annotations = {}
    new_nodes = ty.convert_nodes_tys_to_nap(coords)
    annotations['nodes_coords'] = new_nodes
    if nodes_class is not None:
        annotations['nodes_class'] = nodes_class
    if nodes_class_color_mapper is not None:
        annotations['nodes_class_color_mapper'] = nodes_class_color_mapper
    if pairs is not None:
        annotations['edges_coords'] = pairs # convert_edges_tys_to_nap(new_nodes, pairs)
    return annotations

# annotations = make_annotation_dict(
#     napari_coords, pairs=napari_edges,
# )
annotations = ty.make_annotation_dict(
    napari_coords, pairs=pairs,
)
ty.add_annotations(viewer, annotations)

# %%
# need ravel_multi_index to get pixel values of img at several 3D coordinates
amps_id = np.ravel_multi_index(centers_guess_inds_duplic.transpose(), img.shape)
amps = img.ravel()[amps_id]

merged_coords = merge_cluster_nodes(centers_guess_inds_duplic, pairs, weights=amps)
print(merged_coords)
print(merged_coords.shape)

# %%
# the function doesn't output pairs, run cells above to get it
# merged_coords = filter_nearby_peaks(centers_guess_inds_duplic, max_z=15, max_xy=4, weight_img=img)

viewer = napari.Viewer()
viewer.add_image(img, name='img')
viewer.add_points(centers_guess_inds, name='local maxis', blending='additive', size=3, face_color='r')
viewer.add_points(centers_guess_inds_duplic, name='centers_guess_inds_duplic', blending='additive', size=3, face_color='g')
viewer.add_points(merged_coords, name='merged_coords', blending='additive', size=3, face_color='b')

napari_coords = ty.convert_nodes_tys_to_nap(centers_guess_inds_duplic)
annotations = ty.make_annotation_dict(
    napari_coords, pairs=pairs,
)
ty.add_edges(viewer, annotations)

# %% [markdown]
# ### Try Peter's function

# %%
max_z = 10
max_xy = 3

# need ravel_multi_index to get pixel values of img at several 3D coordinates
amps_id = np.ravel_multi_index(centers_guess_inds_duplic.transpose(), img.shape)
amps = img.ravel()[amps_id]

# %%

centers_merged, inds_comb = localize.filter_nearby_peaks(centers_guess_inds_duplic, max_xy, max_z, weights=amps,
                                                         mode="average")

amps_merged = amps[inds_comb]
print("Found %d points separated by dxy > %0.5g and dz > %0.5g" %
      (len(centers_merged), max_xy, max_z))

# %%
viewer = napari.Viewer()
viewer.add_image(img, name='img')
viewer.add_points(centers_guess_inds, name='local maxis', blending='additive', size=3, face_color='r')
viewer.add_points(centers_guess_inds_duplic, name='centers_guess_inds_duplic', blending='additive', size=3, face_color='g')
viewer.add_points(centers_merged, name='centers_merged', blending='additive', size=3, face_color='b')

# %% [markdown]
# ### Fit gaussian

# %%
def get_roi_coordinates(centers, sizes, max_coords_val, min_sizes, return_sizes=True):
    """
    Make pairs of (z, y, x) coordinates defining an ROI.
    
    Parameters
    ----------
    centers : ndarray, dtype int
        Centers of future ROIs, a Nx3 array.
    sizes : array or list
        Size of ROIs in each dimensions.
    max_coords_val : array or list
        Maximum value of coordinates in each dimension,
        typically the original image shape - 1.
    min_sizes : array or list
        Minimum size of ROIs in each dimension.
    
    Returns
    -------
    roi_coords : ndarray
        Pairs of point coordinates, a 2xNx3 array.
    roi_coords : ndarray
        Shape of each ROI, Nx3 array.
    """
    
    # make raw coordinates
    min_coords = centers - sizes / 2
    max_coords = centers + sizes / 2
    coords = np.stack([min_coords, max_coords]).astype(int)
    # clean min and max values of coordinates
    coords[coords < 0] = 0
    for i in range(3):
        coords[1, coords[1, :, i] > max_coords_val[i], i] = max_coords_val[i]
    # delete small ROIs
    roi_sizes = coords[1, :, :] - coords[0, :, :]
    select = ~np.any([roi_sizes[:, i] <= min_sizes[i] for i in range(3)], axis=0)
    coords = coords[:, select, :]
    # swap axes for latter convenience
    roi_coords = np.swapaxes(coords, 0, 1)
    
    if return_sizes:
        roi_sizes = roi_sizes[select, :]
        return roi_coords, roi_sizes
    else:
        return roi_coords

    
def extract_ROI(img, coords):
    """
    Extract a portion of an image given by the coordinates of 2 points.
    
    Parameters
    ----------
    img : ndarray, dimension 3
        The i;age from which the ROI is extracted.
    coords : ndarry, shape (2, 3)
        The 2 coordinates of the 3 dimensional points at the corner of the ROI.
    
    Returns
    -------
    roi : ndarray
        A region of interest of the original image.
    """
    
    z0, y0, x0 = coords[0]
    z1, y1, x1 = coords[1]
    roi = img[z0:z1, y0:y1, x0:x1]
    return roi

# %%
# centers = centers_guess_inds
# size = 2 * np.array([sz, sy, sx])
# min_coords = centers - size / 2
# max_coords = centers + size / 2
# coords = np.stack([min_coords, max_coords]).astype(int)
# max_coords_val = np.array(img.shape) - 1
# coords[coords < 0] = 0
# for i in range(3):
#     coords[1, coords[1, :, i] > max_coords_val[i], i] = max_coords_val[i]
# min_sizes = [sz, sy, sx]
# select = ~np.any([roi_sizes[:, i] <= min_sizes[i] for i in range(3)], axis=0)

# %%
# roi_sizes = coords[1, :, :] - coords[0, :, :]

# %%
# coords[:, 0, :]

# %%
# fit_roi_sizes = np.array([1.3, 1, 1]) * np.array([sz, sy, sx])
fit_roi_sizes = 2* np.array([sz, sy, sx])
# min_fit_roi_sizes = fit_roi_sizes * 0.7
min_fit_roi_sizes = fit_roi_sizes * 0.5

roi_coords, roi_sizes = get_roi_coordinates(
    centers = centers_guess_inds, 
    sizes = fit_roi_sizes, 
    max_coords_val = np.array(img.shape) - 1, 
    min_sizes = min_fit_roi_sizes,
)
nb_rois = roi_coords.shape[0]

# %%
nb_rois

# %%
viewer = napari.Viewer()
viewer.add_image(img, name='img')
viewer.add_image(img_filtered, name='img_filtered')
viewer.add_points(roi_coords[:, 0, :], name='ROI start', blending='additive', size=3, face_color='r')
viewer.add_points(roi_coords[:, 1, :], name='ROI end', blending='additive', size=3, face_color='g')

# %%
# viewer = napari.Viewer()
# # all_rois = np.stack(extract_ROI(img, roi_coords[i]) for i in range(nb_rois))
# # viewer.add_image(all_rois, name='all rois')
# for i in range(nb_rois):
#     roi = extract_ROI(img, roi_coords[i])
#     viewer.add_image(roi, name=f'roi {i}', blending='additive')

# %%
i = 0
# im_fitted = img_high_pass - img_low_pass
im_fitted = img

roi = extract_ROI(im_fitted, roi_coords[i])
roi_gauss = extract_ROI(img_high_pass, roi_coords[i])

# %%
viewer = napari.Viewer()
viewer.add_image(roi, name='roi')
viewer.add_image(roi_gauss, name='roi gauss')

# %%
centers_guess = (roi_sizes / 2)

# %%
init_params = np.array([
    amps[i], 
    centers_guess[i, 2],
    centers_guess[i, 1],
    centers_guess[i, 0],
    sigma_xy, 
    sigma_z, 
    roi.min(),
])

# %%
theta = 0.

fit_results = localize.fit_gauss_roi(
    roi, 
    (localize.get_coords(roi_sizes[i], drs=[1, 1, 1])), 
    init_params,
)
fit_results

# %%
amplitude, center_x, center_y, center_z, sigma_xy, sigma_z, offset = fit_results['fit_params']

# %%
viewer = napari.Viewer()
viewer.add_image(roi, name='roi')
viewer.add_image(roi_gauss, name='roi gauss')
viewer.add_points([center_z, center_y, center_x], name='fitted center', blending='additive', size=2, face_color='r')

# %%
# # using img_high_pass or img_filtered gives really bad results
# # I'd like to understand why fitted centered are all shifted
# im_fitted = img #img_high_pass - img_low_pass # img

# fit_results_rois = np.zeros((nb_rois, 8))
# for i in range(nb_rois):
#     # extract ROI
#     roi = extract_ROI(im_fitted, roi_coords[i])
#     # fit gaussian in ROI
#     init_params = np.array([
#         amps[i], 
#         centers_guess[i, 2],
#         centers_guess[i, 1],
#         centers_guess[i, 0],
#         sigma_xy, 
#         sigma_z, 
#         roi.min(),
#     ])
#     fit_results_roi = localize.fit_gauss_roi(
#         roi, 
#         (localize.get_coords(roi_sizes[i], drs=[1, 1, 1])), 
#         init_params,
#     )
#     # amplitude, center_x, center_y, center_z, sigma_xy, sigma_z, offset
#     fit_results_rois[i, :7] = fit_results_roi['fit_params']
#     fit_results_rois[i, 7] = fit_results_roi['chi_squared']
# # add origin coordinates of each ROI
# centers = fit_results_rois[:, 1:4] + roi_coords[:, 0, :]

# %%
# using img_high_pass or img_filtered gives really bad results
# I'd like to understand why fitted centered are all shifted
im_fitted = img #img_high_pass - img_low_pass # img

amplitudes = []
centers = []
sigmas = []
chi_squareds = []
all_res = []
for i in range(nb_rois):
    # extract ROI
    roi = extract_ROI(im_fitted, roi_coords[i])
    # fit gaussian in ROI
    init_params = np.array([
        amps[i], 
        centers_guess[i, 2],
        centers_guess[i, 1],
        centers_guess[i, 0],
        sigma_xy, 
        sigma_z, 
        roi.min(),
    ])
    fit_results = localize.fit_gauss_roi(
        roi, 
        (localize.get_coords(roi_sizes[i], drs=[1, 1, 1])), 
        init_params,
        fixed_params=np.full_like(init_params, False),
    )
    amplitude, center_x, center_y, center_z, sigma_xy, sigma_z, offset = fit_results['fit_params']
    amplitudes.append(amplitude)
    centers.append([center_z, center_y, center_x])
    sigmas.append([sigma_xy, sigma_z])
    chi_squareds.append(fit_results['chi_squared'])
    all_res.append(fit_results['fit_params'])
#     print(fit_results)
# add origin coordinates of each ROI
centers = np.array(centers) + roi_coords[:, 0, :]

# %%
np.array(all_res)[:,-4:]
# np.array(all_res)[:,:4]

# %% [markdown]
# Fits only amplitude if ROI is too small, covarianc is a matrix of nan

# %%
viewer = napari.Viewer()
viewer.add_image(img, name='img')
viewer.add_image(img_filtered, name='img_filtered')
# viewer.add_image(im_fitted, name='im_fitted')
viewer.add_points(centers_guess_inds, name='local maxis', blending='additive', size=3, face_color='r')
viewer.add_points(centers, name='fitted centers', blending='additive', size=3, face_color='g')
# viewer.add_points(centers, name='fitted centers', blending='additive', size=3, face_color=chi_squareds, face_colormap=cmap); # napari colormap doesn't work
# viewer.add_points(centers, name='fitted centers chi squared', blending='additive', size=3, face_color=chi_colors)
# viewer.add_points(centers, name='fitted centers sigma xy', blending='additive', size=3, face_color=sigma_xy_colors)

# %% [markdown]
# Some blobs look like real spot blobs but are actually non spot blobs, they look simimlar becaus of the DoG kernel.  
# In these blobs the diff between center from peak max and gaussian fit is noticeable.  
# For real spot blobs, sometimes the peak max seems to provide more accurarte estimation of center's coordinates, but on the real image we observe that the gaussian fit is the most accurate method with small enough ROI.
# But with too small ROI there is no real fitting, and with too big ROI one center can shift due to near spot. 

# %% [markdown]
# Now need to filter fitted spots:
#   - goodness of fit
#   - ratio sigma_z / sigma_xy
#   - maximum distance between guess value and fit value
#   - range of sigmas
#   - ramge of intensity

# %%
# from napari.utils.colormaps.colormap_utils import vispy_or_mpl_colormap
# cmap = vispy_or_mpl_colormap('plasma')
import matplotlib as mpl
cmap = mpl.cm.get_cmap('plasma')

chi_squareds = np.array(chi_squareds)
norm = mpl.colors.Normalize(vmin=chi_squareds.min(), vmax=chi_squareds.max())
chi_colors = [cmap(norm(x)) for x in chi_squareds]

sigmas = np.array(sigmas)
sigma_z_xy_ratios = sigmas[:, 0] / sigmas[:, 1]
# norm = mpl.colors.Normalize(vmin=sigmas[:, 0].min(), vmax=sigmas[:, 0].max())
# sigma_xy_colors = [cmap(norm(x)) for x in sigmas[:, 0]]
norm = mpl.colors.Normalize(vmin=sigma_z_xy_ratios.min(), vmax=sigma_z_xy_ratios.max())
sigma_ratios_colors = [cmap(norm(x)) for x in sigma_z_xy_ratios]

# %%
sigma_z_xy_ratios

# %%
plt.hist(sigma_z_xy_ratios);

# %%
plt.scatter(sigmas[:, 0], sigmas[:, 1])

# %%
viewer = napari.Viewer()
viewer.add_image(img, name='img')
viewer.add_image(img_filtered, name='img_filtered')
# viewer.add_image(im_fitted, name='im_fitted')
viewer.add_points(centers_guess_inds, name='local maxis', blending='additive', size=3, face_color='r')
# viewer.add_points(centers, name='fitted centers', blending='additive', size=3, face_color='g')
# viewer.add_points(centers, name='fitted centers', blending='additive', size=3, face_color=chi_squareds, face_colormap=cmap); # napari colormap doesn't work
# viewer.add_points(centers, name='fitted centers chi squared', blending='additive', size=3, face_color=chi_colors)
viewer.add_points(centers, name='fitted centers sigma ratios', blending='additive', size=3, face_color=sigma_ratios_colors)

# %% [markdown]
# ### Radial spot finding

# %%
i = 0
# im_fitted = img_high_pass - img_low_pass
im_fitted = img

roi = extract_ROI(im_fitted, roi_coords[i])
roi_gauss = extract_ROI(img_high_pass, roi_coords[i])

# %%
center_x, center_y, center_z = localize.localize3d(roi)

# %%
np.hstack((center_z, center_y, center_x))

# %%
viewer = napari.Viewer()
viewer.add_image(roi, name='roi')
viewer.add_image(roi_gauss, name='roi gauss')
viewer.add_points(centers_guess[i], name='guessed center', blending='additive', size=2, face_color='r')
viewer.add_points(np.hstack((center_z, center_y, center_x)), name='fitted center', blending='additive', size=2, face_color='g')

# %%
# using img_high_pass or img_filtered gives really bad results
# I'd like to understand why fitted centered are all shifted
# im_fitted = img
im_fitted = img_high_pass
# im_fitted = img_high_pass - img_low_pass

centers = np.zeros((nb_rois, 3))
for i in range(nb_rois):
    # extract ROI
    roi = extract_ROI(im_fitted, roi_coords[i])
    # radial method fitting
#     center_x, center_y, center_z = localize.localize3d(roi)
    centers[i, :] = np.hstack(localize.localize3d(roi))[::-1]
#     print(fit_results)
# reverse axes from 
# add origin coordinates of each ROI
centers = centers + roi_coords[:, 0, :]

# %%
viewer = napari.Viewer()
viewer.add_image(img, name='img')
viewer.add_image(img_filtered, name='img_filtered')
# viewer.add_image(im_fitted, name='im_fitted')
viewer.add_points(centers_guess_inds, name='local maxis', blending='additive', size=3, face_color='r')
viewer.add_points(centers, name='fitted centers', blending='additive', size=3, face_color='g')

# %% [markdown]
# Still not perfect, probably need the RANSAC version of it.  
# But how do we define the most accurate method if it's not with gaussian fitting?...

# %% [markdown]
# ## Tiling

# %% [markdown]
# Considering a good enough spot finding method, run it on several tiles and merge results
# One solution could be using Dask [map_overlap](https://docs.dask.org/en/latest/generated/dask.array.map_overlap.html)  
# but we run computation on extra areas and we need to manage how to merge results in the overlaping ares.  
# On the pther hand, the `get_roi_coordinates` has a `min_sizes` argument that we can use so with a given overlap we only keep once spots ROIs, either in one tile or the neighbooring one.  
# To do so, `overlap = min_sizes - 1`

# %% [markdown]
# ### Make coordinates

# %%
def gen_split_overlap(seq_size, chunk_size, overlap=0):
    # TODO: improve accuracy of computed sequences of indices 
    if chunk_size < 1 or overlap < 0:
        raise ValueError('check chunk_size > 1 and overlap > 0')

    if chunk_size >= seq_size:
        return [0, seq_size]

    for i in range(0, seq_size - overlap, chunk_size - overlap):
        pass


def make_tiles_coordinates(total_size, tile_size, overlap):
    """
    Make a list of pairs of coordinates that define overlaping tiles.
    
    Parameters
    ----------
    total_size : array | list | tuple
        The size of the original big image from which tiles are extracted.
    tile_size : int | array
        The size of tiles' dimensions without overlap. Dimensions are of equal 
        size if int, else each dimension size is given by the array.
    overlap : int | array | list | tuple
        Overlap between neighbooring tiles. Overlaps are of equal 
        size if int, else each overlap size is given by the iterable.
    
    Returns
    -------
    tiles_coordinates : ndarray
        Pairs of coordinates, dim (nb_tiles, 2, dim_image)
    
    Example
    -------
    >>> make_tiles_coordinates(total_size=(5, 5), tile_size=(2, 2), overlap=1)
    array([[[0, 0], [2, 2]],
           [[0, 2], [2, 4]],
           [[2, 0], [4, 2]],
           [[2, 2], [4, 4]]])
    """
    pass   
    


# %%
localize.get_coords([5, 5], [2, 3])

# %%
total_size = 10
data = np.arange(total_size)
chunk_size = 5
overlap = 3

np.arange(start=0, stop=total_size, step=chunk_size)

# coords = np.array(list(gen_split_overlap(total_size, chunk_size, overlap)))
# coords

# %%
print(data[coords[0, 0]: coords[0, 1]])
print(data[coords[1, 0]: coords[1, 1]])

# %%
tiler = Tiler(
    data_shape=(5, 5),
    tile_shape=(2, 2),
    overlap=1,
    channel_dimension=None,
    mode='irregular',
)

for i in range(4):
    print(tiler.get_tile_bbox(tile_id=i))

# %%
image = np.arange(100).reshape(10, 10)
image

# %%
tiler = Tiler(
    data_shape=image.shape,
    tile_shape=(5, 5),
    overlap=1,
    channel_dimension=None,
    mode='irregular',
)
for tile_id, tile in tiler.iterate(image):
    coords = tiler.get_tile_bbox(tile_id=tile_id)
    print(coords)
    print(tile)
    print()
    print(image[coords[0][0]: coords[1][0], coords[0][1]: coords[1][1]])
    print('\n')

# %% [markdown]
# We have now a way to get coordinatesof overlapping tiles, now we need to make a function that detect spots per ROI, and one that merges the results.

# %% [markdown]
# ### Functionalize spot detection

# %% [markdown]
# #### check Dask behaviour

# %%
from dask.distributed import Client
from dask import delayed

client = Client(n_workers=4)

# %%
def sub_function_1(arg=None):
    if arg is None:
        return "sub_function_1"
    else:
        return "sub_function_1, arg:", arg

def sub_function_2(arg):
    print("sub_function_2, arg:", arg)

def global_function(fct, fct_arg=None):
    print("executed")
    if fct_arg is None:
        return fct()
    else:
        return fct(**fct_arg)

# %%

# x = delayed(global_function)(sub_function_1, {fct_arg:'Un argument!'})
# x = delayed(global_function)(4)
# print(x)
# print(x.compute())

x = delayed(global_function)(sub_function_1, {'arg': 4})
print(x)
print(x.compute())

x = delayed(global_function)(sub_function_1)
print(x)
print(x.compute())

# %% [markdown]
# #### base functions

# %%
# size of spots in pixels
sx = sy = 5
sz = 20
# FWHM = 2.355 x sigma
sigma_xy = sx / 2.355
sigma_z = sz / 2.355
# to reproduce LoG with Dog we need sigma_big = 1.6 * sigma_small
sigma_xy_small = sigma_xy / 1.6**(1/2)
sigma_xy_large = sigma_xy * 1.6**(1/2)
sigma_z_small = sigma_z / 1.6**(1/2)
sigma_z_large = sigma_z * 1.6**(1/2)



def get_roi_coordinates(centers, sizes, max_coords_val, min_sizes, return_sizes=False):
    """
    Make pairs of (z, y, x) coordinates defining an ROI.
    
    Parameters
    ----------
    centers : ndarray, dtype int
        Centers of future ROIs, a Nx3 array.
    sizes : array or list
        Size of ROIs in each dimensions.
    max_coords_val : array or list
        Maximum value of coordinates in each dimension,
        typically the original image shape - 1.
    min_sizes : array or list
        Minimum size of ROIs in each dimension.
    
    Returns
    -------
    roi_coords : ndarray
        Pairs of point coordinates, a 2xNx3 array.
    roi_coords : ndarray
        Shape of each ROI, Nx3 array.
    """
    
    # make raw coordinates
    min_coords = centers - sizes / 2
    max_coords = centers + sizes / 2
    coords = np.stack([min_coords, max_coords]).astype(int)
    # clean min and max values of coordinates
    coords[coords < 0] = 0
    for i in range(3):
        coords[1, coords[1, :, i] > max_coords_val[i], i] = max_coords_val[i]
    # delete small ROIs
    roi_sizes = coords[1, :, :] - coords[0, :, :]
    select = ~np.any([roi_sizes[:, i] < min_sizes[i] for i in range(3)], axis=0)
    coords = coords[:, select, :]
    # swap axes for latter convenience
    roi_coords = np.swapaxes(coords, 0, 1)
    
    if return_sizes:
        roi_sizes = roi_sizes[select, :]
        return roi_coords, roi_sizes
    else:
        return roi_coords

    
def extract_ROI(img, coords):
    """
    Extract a portion of an image given by the coordinates of 2 points.
    
    Parameters
    ----------
    img : ndarray, dimension 3
        The i;age from which the ROI is extracted.
    coords : ndarry, shape (2, 3)
        The 2 coordinates of the 3 dimensional points at the corner of the ROI.
    
    Returns
    -------
    roi : ndarray
        A region of interest of the original image.
    """
    
    z0, y0, x0 = coords[0]
    z1, y1, x1 = coords[1]
    roi = img[z0:z1, y0:y1, x0:x1]
    return roi


def detect_blob_dog(img, sigma_xy_small, sigma_xy_large, 
                    sigma_z_small, sigma_z_large, dog_thresh,
                    min_separations, pixel_sizes,
                    sigma_cutoff, fit_roi_sizes, min_fit_roi_sizes,
                    return_amplitudes=True):
    """
    
    """
    filter_sigma_small = (sigma_z_small, sigma_xy_small, sigma_xy_small)
    filter_sigma_large = (sigma_z_large, sigma_xy_large, sigma_xy_large)

    kernel_small = localize.get_filter_kernel(filter_sigma_small, pixel_sizes, sigma_cutoff=sigma_cutoff)
    kernel_large = localize.get_filter_kernel(filter_sigma_large, pixel_sizes, sigma_cutoff=sigma_cutoff)

    img_high_pass = localize.filter_convolve(img, kernel_small, use_gpu=False)
    img_low_pass = localize.filter_convolve(img, kernel_large, use_gpu=False)
    img_filtered = img_high_pass - img_low_pass
    # apply threshold found with Napari
    img_filtered[img_filtered < dog_thresh] = 0

    footprint = localize.get_max_filter_footprint(min_separations=min_separations, drs=pixel_sizes)
    # array of size nz, ny, nx of True

    centers_guess_inds, amps = localize.find_peak_candidates(img_filtered, footprint, threshold=dog_thresh)

    # we don't return roi_sizes because we would have to manage it in
    # the detect_spots_tile function, whereas another method could not output it
    roi_coords = get_roi_coordinates(
        centers = centers_guess_inds, 
        sizes = fit_roi_sizes, 
        max_coords_val = np.array(img.shape) - 1, 
        min_sizes = min_fit_roi_sizes,
    )

    if return_amplitudes:
        return roi_coords, amps
    else:
        return roi_coords
    

def estimate_center_gauss(img, roi_coords, amps, sigma_xy, sigma_z):
    
    roi_sizes = roi_coords[:, 1, :] - roi_coords[:, 0, :]
    centers_guess = (roi_sizes / 2)
    
    # Gaussian fit to find center of each spot
    # amplitudes = []
    centers = []
    # sigmas = []
    # chi_squareds = []
    # all_res = []
    for i in range(len(roi_coords)):
        # extract ROI
        roi = extract_ROI(img, roi_coords[i])
        # fit gaussian in ROI
        init_params = np.array([
            amps[i], 
            centers_guess[i, 2],
            centers_guess[i, 1],
            centers_guess[i, 0],
            sigma_xy, 
            sigma_z, 
            roi.min(),
        ])
        fit_results = localize.fit_gauss_roi(
            roi, 
            (localize.get_coords(roi_sizes[i], drs=[1, 1, 1])), 
            init_params,
            fixed_params=np.full_like(init_params, False),
        )
        amplitude, center_x, center_y, center_z, sigma_xy, sigma_z, offset = fit_results['fit_params']
        # amplitudes.append(amplitude)
        centers.append([center_z, center_y, center_x])
        # sigmas.append([sigma_xy, sigma_z])
        # chi_squareds.append(fit_results['chi_squared'])
        # all_res.append(fit_results['fit_params'])
    # add origin coordinates of each ROI
    centers = np.array(centers) + roi_coords[:, 0, :]
    
    return centers


def shift_coordinates(spots_coords, tile_coords, format='pair'):
    
    if format == 'pair':
        spots_coords = spots_coords + tile_coords[:, 0, :]
    elif format == 'single':
        spots_coords = spots_coords + tile_coords
    return spots_coords
    

def detect_spots_tile(tile, tile_coords=None, 
                      roi_method=detect_blob_dog, roi_kwargs=None,
                      center_method=estimate_center_gauss, center_kwargs=None,
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

# %%
# size of spots in pixels
sx = sy = 5
sz = 20
# FWHM = 2.355 x sigma
sigma_xy = sx / 2.355
sigma_z = sz / 2.355
# to reproduce LoG with Dog we need sigma_big = 1.6 * sigma_small
sigma_xy_small = sigma_xy / 1.6**(1/2)
sigma_xy_large = sigma_xy * 1.6**(1/2)
sigma_z_small = sigma_z / 1.6**(1/2)
sigma_z_large = sigma_z * 1.6**(1/2)

fit_roi_sizes = (1.5 * np.array([sz, sy, sx])).astype(int)

tile = img
tile_coords = None
roi_method = detect_blob_dog
roi_kwargs = {
    'sigma_xy_small': sigma_xy / 1.6**(1/2),
    'sigma_xy_large': sigma_xy * 1.6**(1/2),
    'sigma_z_small': sigma_z / 1.6**(1/2),
    'sigma_z_large': sigma_z * 1.6**(1/2),
    'dog_thresh': 4,
    'min_separations': (10, 3, 3), 
    'pixel_sizes': (1, 1, 1),
    'sigma_cutoff': 2, 
    'fit_roi_sizes': fit_roi_sizes, 
    'min_fit_roi_sizes': fit_roi_sizes,
    'return_amplitudes': True,
}
center_method = estimate_center_gauss
center_kwargs = {
    'sigma_xy': sigma_xy, 
    'sigma_z': sigma_z,
}
filter_method = None
filter_kwargs = None

# %%
rois_coords, amps = roi_method(tile, **roi_kwargs)
print(rois_coords)
print(rois_coords.shape)

# %%
spots_coords = center_method(tile, rois_coords, amps, **center_kwargs)
print(spots_coords)

# %%
viewer = napari.Viewer()
viewer.add_image(img, name='img')
viewer.add_image(img_filtered, name='img_filtered')
# viewer.add_image(im_fitted, name='im_fitted')
viewer.add_points(spots_coords, name='fitted centers', blending='additive', size=3, face_color='g')

# %% [markdown]
# #### with Dask

# %%
rois_coords = delayed(roi_method)(tile, **roi_kwargs)
spots_coords = delayed(estimate_center_gauss)(tile, rois_coords, **center_kwargs)
print(spots_coords)
print(spots_coords.compute())

# %% [markdown]
# #### On multiple tiles

# %%
start_x = 512
start_y = 1000
start_z = 128
size_xy = 128 * 2
size_z  = 128 * 2
img = im[start_z:(start_z+size_z), start_y:(start_y+size_xy), start_x:(start_x+size_xy)].compute()

# %%
tile_shape = np.array([128, 128, 128]) + fit_roi_sizes
overlap = fit_roi_sizes - 1

tiler = Tiler(
    data_shape=img.shape,
    tile_shape=tile_shape,
    overlap=overlap,
    channel_dimension=None,
    mode='irregular',
)
for tile_id, tile in tiler.iterate(img):
    coords = tiler.get_tile_bbox(tile_id=tile_id)
    print(coords)
    print(tile.shape)

# %%
tiled_spots_coords = []
for tile_id, tile in tiler.iterate(img):
    print(tile_id)
    # origin coordinates of the tile
    tile_coords_ori = tiler.get_tile_bbox(tile_id=tile_id)[0]
    # get spots ROIs coordinates in the tile
    rois_coords, amps = roi_method(tile, **roi_kwargs)
    # fit tile's spots with gaussian
    spots_coords = center_method(tile, rois_coords, amps, **center_kwargs)
    # add origin coordinates to tile's spots' coordinates
    spots_coords = spots_coords + tile_coords_ori
    # save in global list of coordinates
    tiled_spots_coords.append(spots_coords)

tiled_spots_coords =  np.vstack(tiled_spots_coords)

# %%

# get spots ROIs coordinates in the tile
rois_coords, amps = roi_method(img, **roi_kwargs)
# fit tile's spots with gaussian
whole_spots_coords = center_method(img, rois_coords, amps, **center_kwargs)

# %%

# viewer = napari.Viewer()
# viewer.add_image(img, name='img')
# viewer.add_points(whole_spots_coords, name='whole_spots_coords', blending='additive', size=3, face_color='g')
# viewer.add_points(tiled_spots_coords, name='tiled_spots_coords', blending='additive', size=3, face_color='r')

# There are 3 more points in  the tilted version, apparently near the overlapping regions. Maybe the DoG is too sensitive to
# end of tiles. The points can be easily filtered out as they are clearly not on a real spot.

# %%
def get_tile_coords_ori(tiler, tile_id):
    return tiler.get_tile_bbox(tile_id=tile_id)[0]

from dask.distributed import Client
from dask import delayed

# %%
nb_cores = os.cpu_count()
client = Client(n_workers=nb_cores)
# cluster = LocalCluster(n_workers=4, threads_per_worker=2)
# client = Client(cluster, asynchronous=True)

tiled_spots_coords = []
for tile_id, tile in tiler.iterate(img):
    print(tile_id)
    # origin coordinates of the tile
    tile_coords_ori = delayed(get_tile_coords_ori)(tiler, tile_id)
    # get spots ROIs coordinates in the tile
    rois_coords, amps = delayed(roi_method, nout=2)(tile, **roi_kwargs)
    # fit tile's spots with gaussian
    spots_coords = delayed(center_method)(tile, rois_coords, amps, **center_kwargs)
    # add origin coordinates to tile's spots' coordinates
    spots_coords = delayed(shift_coordinates)(spots_coords, tile_coords_ori, 'single')
    # save in global list of coordinates
    tiled_spots_coords.append(spots_coords)

aggregated_spots_coords =  delayed(merge_spots_coords)(tiled_spots_coords)
aggregated_spots_coords.visualize()

# %%
coords = aggregated_spots_coords.compute()
client.close()

# %%
coords.shape

# %% [markdown]
# ## Include filtering

# %%
filter_params = {
    'filter_amplitude_min': 20,
    'filter_amplitude_max': False,
    'filter_sigma_xy_min': 0.5,
    'filter_sigma_xy_max': 5,
    'filter_sigma_z_min': 2,
    'filter_sigma_z_max': 20,
    'filter_sigma_ratio_min': 1,
    'filter_sigma_ratio_max': 5,
    'filter_chi_squared': 200,
    'filter_dist_center': 3,
}

# %%
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

    if filter_params['filter_amplitude_min']:
        selectors.append(filter_vars['amplitudes'] >= filter_params['filter_amplitude_min'])
    if filter_params['filter_amplitude_max']:
        selectors.append(filter_vars['amplitudes'] <= filter_params['filter_amplitude_max'])
    if filter_params['filter_sigma_xy_min']:
        selectors.append(filter_vars['sigmas_xy'] >= filter_params['filter_sigma_xy_min'])
    if filter_params['filter_sigma_xy_max']:
        selectors.append(filter_vars['sigmas_xy'] <= filter_params['filter_sigma_xy_max'])
    if filter_params['filter_sigma_z_min']:
        selectors.append(filter_vars['sigmas_z'] >= filter_params['filter_sigma_z_min'])
    if filter_params['filter_sigma_z_max']:
        selectors.append(filter_vars['sigmas_z'] <= filter_params['filter_sigma_z_max'])
    if filter_params['filter_sigma_ratio_min']:
        selectors.append(filter_vars['sigma_ratios'] >= filter_params['filter_sigma_ratio_min'])
    if filter_params['filter_sigma_ratio_max']:
        selectors.append(filter_vars['sigma_ratios'] <= filter_params['filter_sigma_ratio_max'])
    if filter_params['filter_chi_squared']:
        selectors.append(filter_vars['chi_squared'] >= filter_params['filter_chi_squared'])
    if filter_params['filter_dist_center']:
        selectors.append(filter_vars['dist_center'] <= filter_params['filter_dist_center'])

    if len(selectors) == 0:
        print("No filter is active")
    else:
        spot_select = np.logical_and.reduce(selectors)
    
    return spot_select

def apply_filter_spots(spots_coords, spot_select):
    return spots_coords[spot_select, :]

# %%
def estimate_center_gauss(img, roi_coords, amps, sigma_xy, sigma_z, return_fit_vars=True):
    
    roi_sizes = roi_coords[:, 1, :] - roi_coords[:, 0, :]
    centers_guess = (roi_sizes / 2)
    
    # Gaussian fit to find center of each spot
    all_res = []
    chi_squared = []
    for i in range(len(roi_coords)):
        # extract ROI
        roi = extract_ROI(img, roi_coords[i])
        # fit gaussian in ROI
        init_params = np.array([
            amps[i], 
            centers_guess[i, 2],
            centers_guess[i, 1],
            centers_guess[i, 0],
            sigma_xy, 
            sigma_z, 
            roi.min(),
        ])
        fit_results = localize.fit_gauss_roi(
            roi, 
            (localize.get_coords(roi_sizes[i], drs=[1, 1, 1])), 
            init_params,
            fixed_params=np.full_like(init_params, False),
        )
        chi_squared.append(fit_results['chi_squared'])
        all_res.append(fit_results['fit_params'])
        
    # process all the results
    all_res = np.array(all_res)
    amplitudes = all_res[:, 0]
    centers = all_res[:, 3:0:-1]
    sigmas_xy = all_res[:, 4]
    sigmas_z = all_res[:, 5]
    # offsets = all_res[:, 6]
    chi_squared = np.array(chi_squared)
    # distances from initial guess
    dist_center = np.sqrt(np.sum((centers - centers_guess)**2, axis=1))
    # add origin coordinates of each ROI
    centers = centers + roi_coords[:, 0, :]
    # composed variables for filtering
    sigma_ratios = sigmas_z / sigmas_xy
    
    fit_vars = {
    'amplitudes': amplitudes,
    'sigmas_xy': sigmas_xy,
    'sigmas_z': sigmas_z,
    # 'offsets': offsets,
    'chi_squared': chi_squared,
    'dist_center': dist_center,
    'sigma_ratios': sigma_ratios,
    }        
    
    if return_fit_vars:
        return centers, fit_vars
    else:
        return centers

center_method = estimate_center_gauss

# %%
client = Client(n_workers=nb_cores)

tiled_spots_coords = []
for tile_id, tile in tiler.iterate(img):
    print(tile_id)
    # origin coordinates of the tile
    tile_coords_ori = delayed(get_tile_coords_ori)(tiler, tile_id)
    # get spots ROIs coordinates in the tile
    rois_coords, amps = delayed(roi_method, nout=2)(tile, **roi_kwargs)
    # fit tile's spots with gaussian
    spots_coords, fit_vars = delayed(center_method, nout=2)(tile, rois_coords, amps, **center_kwargs)
    # make spots boolean selector
    spot_select = delayed(make_filter_spots)(fit_vars, filter_params)
    # apply filter on spots
    spots_coords = delayed(apply_filter_spots)(spots_coords, spot_select)
    # add origin coordinates to tile's spots' coordinates
    spots_coords = delayed(shift_coordinates)(spots_coords, tile_coords_ori, 'single')
    # save in global list of coordinates
    tiled_spots_coords.append(spots_coords)

aggregated_spots_coords = delayed(merge_spots_coords)(tiled_spots_coords)
aggregated_spots_coords.visualize()

# %%
coords = aggregated_spots_coords.compute()
print(f'Found {coords.shape[0]} spots')
client.close()

# %%
# It's working! I just need to select appropriate filter thresholds.

# %% [markdown]
# ## Automated spot filtering parameters selection

# %% [markdown]
# We need a method to autoatically select the best parameters combination to filter spots.  
# That requires a performance metrics.  
# We can first match spots that are within a given distance (related to PSF), we can use one distance for the xy plane, and another one for the z axis. We will need to adapt that for tilted configuration. Points can be only matched to a single other point, select the closest one.
# Then can compute true and false positives and negatives.

# %%
def match_closest_points(ref, target, k=1):
    """
    
    Example
    -------
    >>> ref = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0]])
    >>> target = np.array([[1, 1, 1], [7, 0, 0], [0, 0, 10], [5, 0, 0], [0, 11, 0]])
    >>> match_closest_points(source, target)
        array([0, 1, 0, 0, 2])
    """
    from scipy.spatial import cKDTree
    kdt_ref = KDTree(ref)

    # closest node id and discard computed distances ('_,')
    _, matched_ids = kdt_ref.query(x=target, k=k)
    return matched_ids

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


def make_counts_array(data, return_all=True):
    """
    Return a vector of counts of same size as data.
    """
    uniq, counts = np.unique(data, return_counts=True)
    if return_all:
        return uniq, counts, counts[data]
    else:
        return counts[data]


def remove_multiple_links(matched_ids, coords, dist_z, dist_xy):
    """
    
    Example
    -------
    >>> matched_ids = np.array([2, 0, 1, 0, 1, 0])
    >>> coords = np.array([[2, 2, 2], [0, 0, 0], [1, 1, 1], [0, 10, 0], [1, 1, 10], [0, 0, 10]])
    >>> dist_z = np.array([0, 0, 0, 10, 10, 10])
    >>> dist_xy = np.array([0, 0, 0, 10, 10, 10])
    >>> remove_multiple_links(matched_ids, coords, dist_z, dist_xy)
        (array([2, 0, 1]),
         array([[2, 2, 2],
                [0, 0, 0],
                [1, 1, 1]]))
    """
    matched_uniq, matched_counts, counts_array = make_counts_array(matched_ids)
    select_duplicated_array = counts_array != 1
    if select_duplicated_array.sum() == 0:
        return matched_uniq, coords
    else:
        # start with a copy of unique matches and coordinates
        matched_ids_single = matched_ids[~select_duplicated_array]
        coords_single = coords[~select_duplicated_array]
        # then add a single match and set of coordinates per duplicated match
        select_duplicated_uniq = matched_counts != 1
        for i in matched_uniq[select_duplicated_uniq]:
            select = matched_ids == i
            coords_duplicated = coords[select]
            dist_tot = dist_z[select] + dist_xy[select]
            min_id = np.argmin(dist_tot)
            matched_ids_single = np.hstack((matched_ids_single, i))
            coords_single = np.vstack((coords_single, coords_duplicated[min_id]))
        return matched_ids_single, coords_single
    
    
def match_spots(ref, target, thresh_z=15, thresh_xy=5):
    """
    
    Example
    -------
    >>> ref = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0]])
    >>> target = np.array([[11, 0, 0], [1, 0, 0], [12, 0, 0], [100, 0, 0], [1, 1, 1], [0, 12, 0]])
    >>> match_spots(ref, target, thresh_z=15, thresh_xy=5)
    (array([2, 0, 1]),
     array([[ 0, 12,  0],
            [ 1,  0,  0],
            [11,  0,  0]]))
    """
    # match all target points to reference points
    matched_ids = match_closest_points(ref, target)
    # compute separately distances in plane and along z axis
    dist_z, dist_xy = compute_distances(ref[matched_ids], target, method='xy_z_orthog', dist_fct='euclidian')
    # threshold on both distances
    select = np.logical_and(dist_z < thresh_z, dist_xy < thresh_xy)
    matched_ids = matched_ids[select]
    target = target[select]
    dist_z = dist_z[select]
    dist_xy = dist_xy[select]
    # remove multiple links from single spots
    matched_ids_single, target_single = remove_multiple_links(matched_ids, target, dist_z, dist_xy)
    return matched_ids_single, target_single

def evaluate_spot_detection_performance(nb_ref, nb_pred, matched_ids):

    # matched_ids is contained in the sets of ref spots and pred spots.
    TP = len(matched_ids)
    FP = nb_pred - len(matched_ids)
    FN = nb_ref - len(matched_ids)
    
    F1 = TP / (TP + 0.5 * (FP + FN))
    return F1

# %% [markdown]
# We will implement the automated parameters selection in the future, the priority is detecting spots in the native tilted plane.

# %% [markdown]
# ## Spot detection in tilted plane

# %% [markdown]
# ### Extract data

# %%
def open_NDTiff(path_dataset, channels=None, z_levels=None, squeeze=True):
    """
    Open an NDTiff image.

    Parameters
    ----------
    path_dataset : str
        Path of the image.
    channels : None or list(int)
        If None, load all channels, else load a list of channels.
    z_levels : None or array
        If None, load all z slices, else load a list of slices.
    
    Returns:
    img : ndarray
        A numpy ndimage.
    """

    dataset = Dataset(path_dataset)
    # use metadata to guess how to load the image
    meta = dataset.axes
    c = np.array([x for x in meta['c']])  # array([1, 2, 3])
    chan = np.array([x for x in meta['channel']])
    if chan.size == 1:
        chan_dict = {x: chan[0] for x in c}
    elif chan.size == c.size:
        shift = int(np.unique(c.min() - chan))
        chan_dict = {c[i]: chan[i] for i in range(len(c))}
    else:
        raise("`c` and `channel` don't match.")
    
    if z_levels is None:
        z_levels = np.array([x for x in meta['z']])

    # load one image to get more info
    sample = dataset.read_image(z=int(np.median(z_levels)), c=c[0], channel=chan_dict[c[0]])
    nb_z = z_levels.size
    nb_y, nb_x = sample.shape

    # iterativelly load all z planes of all channels
    if channels is None:
        channels = list(c) # convert to list for downstream compatibility
    else:
        # detect what could go wrong
        if isinstance(channels, int):
            channels = [channels]
        warn_channels = [x for x in channels if x not in c]
        if len(warn_channels) > 0:
            print("WARNING these channels are not in the dataset:")
            print(warn_channels)
        channels = [x for x in channels if x in c]
        if len(channels) == 0:
            print("WARNING there is no requested channel in the dataset, returning")
            return
    nb_ch = len(channels)
    img = np.zeros((nb_ch, nb_z, nb_y, nb_x), dtype=sample.dtype)
    for i, channel_id in enumerate(channels):
        print("    channel id: {}".format(channel_id))
        for z_id, z in enumerate(z_levels):
            img[i, z_id, :, :] = dataset.read_image(z=z, c=channel_id, channel=chan_dict[channel_id])
    
    if squeeze:
        img = np.squeeze(img)
    return img

# %%
dir_load = Path('../../../from_server/example_image_tilted_z0')
path_im = dir_load / '16plex_lung_r0000_y0000_z0000_1'

channel = [2]

im = open_NDTiff(
    path_im.as_posix(),
    channels=channel,
    squeeze=True)
print(im.shape)
# np.squeeze in open_NDTiff exchanged z and y axes
# im = np.swapaxes(im, -3, -2)
# print(im.shape)

# %%
start_x = 800
start_y = 1500
start_z = 0
size_x = 512
size_y = 800
size_z = 256

# There is a difference of definition between z and y for the microscope and for python:
# tilted xy images are aquired while the stage is moving
# successive images are stacked in a third "z" dimension
# but in the physical space this is the y direction, the physical z is the y axis of the image

# img = im[start_z:start_z+size_z, start_y:start_y+size_y, start_x:start_x+size_x]
img = im[start_y:start_y+size_y, start_z:start_z+size_z, start_x:start_x+size_x]
mini = img.min()
maxi = img.max()

# %%
viewer = napari.Viewer()
# viewer.add_image(img_high_pass, name='img_high_pass')
# viewer.add_image(img_low_pass, name='img_low_pass')
viewer.add_image(img, name='img')

# %%
from localize_psf import data_io

scan_data_path = dir_load / "scan_metadata.csv"
scan_data = data_io.read_metadata(scan_data_path)

nt = scan_data["num_t"]
nyp = scan_data["y_pixels"]
nxp = scan_data["x_pixels"]
dc = scan_data["pixel_size"] / 1000
dstage = scan_data["scan_step"] / 1000
theta = scan_data["theta"] * np.pi / 180
normal = np.array([0, -np.sin(theta), np.cos(theta)])  # normal of camera pixel

num_r = scan_data['num_r']
num_y = scan_data['num_y']
num_z = scan_data['num_z']
num_ch = scan_data['num_ch']
num_images = scan_data['scan_axis_positions']
excess_images = scan_data['excess_scan_positions']
nimgs_per_vol = num_images + excess_images

# trapezoid volume
volume_um3 = (dstage * nimgs_per_vol) * (dc * np.sin(theta) * nyp) * (dc * nxp)

# ###############################
# load/set parameters for all datasets
# ###############################
chunk_size_planes = 1000
chunk_size_x = 300
# chunk_size_planes = 300
# chunk_size_x = 150
chunk_overlap = 10
channel_to_use = [False, True, True]
excitation_wavelengths = np.array([0.488, 0.561, 0.635])
emission_wavelengths = np.array([0.515, 0.600, 0.680])
thresholds = np.array([np.nan, 100, 25])
fit_thresholds = np.array([np.nan, 100, 25])
na = 1.
ni = 1.4

# %% [markdown]
# ### DoG filter

# %%
ch = 2
# Peter's code

# sigma_xy = 0.22 * emission_wavelengths[ch] / na
# sigma_z = np.sqrt(6) / np.pi * ni * emission_wavelengths[ch] / na ** 2
sigma_xy = psf.na2sxy(na, emission_wavelengths[ch])
sigma_z = psf.na2sz(na, emission_wavelengths[ch], ni)

# difference of gaussian filer
filter_sigma_small = (0.5 * sigma_z, 0.25 * sigma_xy, 0.25 * sigma_xy)
filter_sigma_large = (3 * sigma_z, 3 * sigma_xy, 3 * sigma_xy)
# fit roi size
roi_size = (5 * sigma_z, 12 * sigma_xy, 12 * sigma_xy)
# assume points closer together than this come from a single bead
min_spot_sep = (3 * sigma_z, 3 * sigma_xy)
# exclude points with sigmas outside these ranges
sigmas_min = (0.25 * sigma_z, 0.25 * sigma_xy)
sigmas_max = (3 * sigma_z, 4 * sigma_xy)

# %%
print(f"  - dc: {np.round(dc, 3)}  ")
print(f"  - theta: {np.round(theta, 3)}  ")
print(f"  - dstage: {np.round(dstage, 3)}  ")
print(f"  - sigma_xy: {np.round(sigma_xy, 3)}  ")
print(f"  - sigma_z: {np.round(sigma_z, 3)}  ")
print(f"  - sigma_xy_small: {np.round(sigmas_min[1], 3)}  ")
print(f"  - sigma_xy_large: {np.round(sigmas_max[1], 3)}  ")
print(f"  - sigma_z_small: {np.round(sigmas_min[0], 3)}  ")
print(f"  - sigma_z_large: {np.round(sigmas_max[0], 3)}  ")

# %%
ks = localize_skewed.get_filter_kernel_skewed(filter_sigma_small, dc, theta, dstage, sigma_cutoff=2)
kl = localize_skewed.get_filter_kernel_skewed(filter_sigma_large, dc, theta, dstage, sigma_cutoff=2)
# imgs_hp = localize.filter_convolve(img, ks, use_gpu=True)
# imgs_lp = localize.filter_convolve(img, kl, use_gpu=True)
imgs_hp = localize.filter_convolve(img, np.flip(np.swapaxes(ks, 0, 1), axis=0), use_gpu=True)
imgs_lp = localize.filter_convolve(img, np.flip(np.swapaxes(kl, 0, 1), axis=0), use_gpu=True)
imgs_filtered = imgs_hp - imgs_lp

# %%

viewer = napari.Viewer()
# viewer.add_image(ks, name='kernel small', colormap='green', blending='additive')
# viewer.add_image(kl, name='kernel large', colormap='red', blending='additive')
# viewer.add_image(np.flip(kl, axis=1), name='kernel large flipped', colormap='blue', blending='additive')
# viewer.add_image(np.swapaxes(kl, 0, 1), name='kernel large swap 0-1', blending='additive')
viewer.add_image(np.flip(np.swapaxes(ks, 0, 1), axis=0), name='kernel small swap 0-1 flip 0', blending='additive')  # that's the good one!
viewer.add_image(np.flip(np.swapaxes(kl, 0, 1), axis=0), name='kernel large swap 0-1 flip 0', blending='additive')  # that's the good one!
# but the big kernel looks really big compared to tilted spots
viewer.add_image(img, name='img', blending='additive')
viewer.add_image(imgs_filtered, name='imgs_filtered')

# %%
# Choose by hand size of spots

# size of spots in pixels
sx = sy = 0.7
sz = 2
# FWHM = 2.355 x sigma
sigma_xy = sx / 2.355
sigma_z = sz / 2.355
# to reproduce LoG with Dog we need sigma_big = 1.6 * sigma_small
sigma_xy_small = sigma_xy / 1.6**(1/2)
sigma_xy_large = sigma_xy * 1.6**(1/2)
sigma_z_small = sigma_z / 1.6**(1/2)
sigma_z_large = sigma_z * 1.6**(1/2)

filter_sigma_small = (sigma_z_small, sigma_xy_small, sigma_xy_small)
filter_sigma_large = (sigma_z_large, sigma_xy_large, sigma_xy_large)

print(f"  - sx: {np.round(sx, 3)}  ")
print(f"  - sz: {np.round(sz, 3)}  ")
print(f"  - sigma_xy: {np.round(sigma_xy, 3)}  ")
print(f"  - sigma_z: {np.round(sigma_z, 3)}  ")
print(f"  - sigma_xy_small: {np.round(sigma_xy_small, 3)}  ")
print(f"  - sigma_xy_large: {np.round(sigma_xy_large, 3)}  ")
print(f"  - sigma_z_small: {np.round(sigma_z_small, 3)}  ")
print(f"  - sigma_z_large: {np.round(sigma_z_large, 3)}  ")

# %%
# Using Peter's functions for spot size, but with
# standard DoG parameters

# Peter's function
sigma_xy = psf.na2sxy(na, emission_wavelengths[ch])
sigma_z = psf.na2sz(na, emission_wavelengths[ch], ni)
# to reproduce LoG with Dog we need sigma_big = 1.6 * sigma_small
sigma_xy_small = sigma_xy / 1.6**(1/2)
sigma_xy_large = sigma_xy * 1.6**(1/2)
sigma_z_small = sigma_z / 1.6**(1/2)
sigma_z_large = sigma_z * 1.6**(1/2)

filter_sigma_small = (sigma_z_small, sigma_xy_small, sigma_xy_small)
filter_sigma_large = (sigma_z_large, sigma_xy_large, sigma_xy_large)

print(f"  - sigma_xy: {np.round(sigma_xy, 3)}  ")
print(f"  - sigma_z: {np.round(sigma_z, 3)}  ")
print(f"  - sigma_xy_small: {np.round(sigma_xy_small, 3)}  ")
print(f"  - sigma_xy_large: {np.round(sigma_xy_large, 3)}  ")
print(f"  - sigma_z_small: {np.round(sigma_z_small, 3)}  ")
print(f"  - sigma_z_large: {np.round(sigma_z_large, 3)}  ")

# %%
from importlib import reload
reload(localize_skewed)

# %%
def get_filter_kernel_skewed(sigmas, dc, theta, dstage, sigma_cutoff=2):
    pixel_sizes = (dc, dc, dc)
    kernel = localize.get_filter_kernel(sigmas, pixel_sizes, sigma_cutoff)
    kernel = ipp.deskew(kernel, theta, distance=dstage, pixel_size=dc)
    return kernel

# %%
# filtering
def get_filter_kernel_skewed(sigmas, dc, theta, dstage, sigma_cutoff=2):
    """
    Get gaussian filter convolution kernel in skewed coordinates

    :param sigmas: (sz, sy, sx) in the same units as dc and stage
    :param dc: pixel size
    :param theta: angle in radians
    :param dstage: stage step
    :param sigma_cutoff: number of standard deviations to include in the filter. This parameter determines the fitler size
    :return kernel:
    """
    # normalize everything to camera pixel size
    sigma_x_pix = sigmas[2] / dc
    sigma_y_pix = sigmas[2] / dc
    sigma_z_pix = sigmas[0] / dc
    nk_x = 2 * int(np.round(sigma_x_pix * sigma_cutoff)) + 1
    nk_y = 2 * int(np.round(sigma_y_pix * sigma_cutoff)) + 1
    nk_z = 2 * int(np.round(sigma_z_pix * sigma_cutoff)) + 1
    # determine how large the OPM geometry ROI needs to be to fit the desired filter
    roi_sizes = get_skewed_roi_size([nk_z, nk_y, nk_x], theta, 1, dstage / dc, ensure_odd=True)

    # get coordinates to evaluate kernel at
    xk, yk, zk = get_skewed_coords(roi_sizes, 1, dstage / dc, theta)
    xk = xk - np.mean(xk)
    yk = yk - np.mean(yk)
    zk = zk - np.mean(zk)

    kernel = np.exp(-xk ** 2 / 2 / sigma_x_pix ** 2 - yk ** 2 / 2 / sigma_y_pix ** 2 - zk ** 2 / 2 / sigma_z_pix ** 2)
    kernel = kernel / np.sum(kernel)

    return kernel

# %%
sigmas = filter_sigma_small
sigma_cutoff = 3

# normalize everything to camera pixel size
sigma_x_pix = sigmas[2] / dc
sigma_y_pix = sigmas[2] / dc
sigma_z_pix = sigmas[0] / dc
print('sigma_x_pix:', sigma_x_pix)
print('sigma_y_pix:', sigma_y_pix)
print('sigma_z_pix:', sigma_z_pix)
nk_x = 2 * int(np.round(sigma_x_pix * sigma_cutoff)) + 1
nk_y = 2 * int(np.round(sigma_y_pix * sigma_cutoff)) + 1
nk_z = 2 * int(np.round(sigma_z_pix * sigma_cutoff)) + 1
print('nk_x:', nk_x)
print('nk_y:', nk_y)
print('nk_z:', nk_z)
# determine how large the OPM geometry ROI needs to be to fit the desired filter
roi_sizes = localize_skewed.get_skewed_roi_size([nk_z, nk_y, nk_x], theta, 1, dstage / dc, ensure_odd=True)
print('roi_sizes:', roi_sizes)

# get coordinates to evaluate kernel at
xk, yk, zk = localize_skewed.get_skewed_coords(roi_sizes, 1, dstage / dc, theta)
print('xk shape:', xk.shape)
print('yk shape:', yk.shape)
print('zk shape:', zk.shape)
# xk = xk - np.mean(xk)
# yk = yk - np.mean(yk)
# zk = zk - np.mean(zk)

# %%
zk.shape

# %%
# pixel_sizes = 2# / 0.115
# dstage = .4

# kernel_small = localize_skewed.get_filter_kernel_skewed(filter_sigma_small, pixel_sizes, theta, dstage=dstage, sigma_cutoff=3)
# kernel_large = localize_skewed.get_filter_kernel_skewed(filter_sigma_large, pixel_sizes, theta, dstage=dstage, sigma_cutoff=3)
kernel_small = localize_skewed.get_filter_kernel_skewed(filter_sigma_small, dc, theta, dstage, sigma_cutoff=3)
kernel_large = localize_skewed.get_filter_kernel_skewed(filter_sigma_large, dc, theta, dstage, sigma_cutoff=3)
# kernel_small = get_filter_kernel_skewed(filter_sigma_small, dc, theta, dstage, sigma_cutoff=3)
# kernel_large = get_filter_kernel_skewed(filter_sigma_large, dc, theta, dstage, sigma_cutoff=3)
# print(kernel_large.sum())
# kernel_small = np.flip(np.swapaxes(kernel_small, 0, 1), axis=0)
# kernel_large = np.flip(np.swapaxes(kernel_large, 0, 1), axis=0)
kernel_small = np.flip(kernel_small, axis=0)
kernel_large = np.flip(kernel_large, axis=0)

# pixel_sizes = (dc, dc, dc)
# kernel_small_straight = localize.get_filter_kernel(filter_sigma_small, pixel_sizes, sigma_cutoff=2)
# kernel_large_straight = localize.get_filter_kernel(filter_sigma_large, pixel_sizes, sigma_cutoff=2)

viewer = napari.Viewer()
viewer.add_image(kernel_small, name='kernel small', colormap='green', blending='additive')
viewer.add_image(kernel_large, name='kernel large', colormap='red', blending='additive')
viewer.add_image(kernel_small_straight, name='kernel small traight', colormap='green', blending='additive')
viewer.add_image(kernel_large_straight, name='kernel large traight', colormap='red', blending='additive')
viewer.add_image(img, name='img', blending='additive')

# %%Can we skip the second convulotion with an "normalized" kernel?
# im_fct = scipy.signal.fftconvolve(img, kernel_small, mode="same") / scipy.signal.fftconvolve(np.ones(img.shape), kernel_small, mode="same")
# kernel_small_normalized = kernel_small - kernel_small.mean()
# im_normalized = scipy.signal.fftconvolve(img, kernel_small_normalized, mode="same")


# viewer = napari.Viewer()
# viewer.add_image(im_fct, name='im_fct')
# viewer.add_image(im_normalized, name='im_normalized')

# %% [markdown]
# #### Make hand tilted gaussian

# %%
pixel_sizes = (dc, dc, dc)
sigmas_small = np.array(filter_sigma_small)
sigmas_large = np.array(filter_sigma_large)
kernel_small_straight = localize.get_filter_kernel(sigmas_small, pixel_sizes, sigma_cutoff=3)
kernel_large_straight = localize.get_filter_kernel(sigmas_large, pixel_sizes, sigma_cutoff=3)
kernel_small_tilted = scipy.ndimage.rotate(kernel_small_straight, angle=30, axes=(1, 0), reshape=True)
kernel_large_tilted = scipy.ndimage.rotate(kernel_large_straight, angle=30, axes=(1, 0), reshape=True)

viewer = napari.Viewer()
viewer.add_image(kernel_small_tilted, name='kernel small tilted', colormap='green', blending='additive')
viewer.add_image(kernel_large_tilted, name='kernel large tilted', colormap='red', blending='additive')
viewer.add_image(img, name='img', blending='additive')

# %% [markdown]
# Need to take into account stage displacement, which "elongates" the image in one direction.

# %%
dilation_coef = dstage / dc
dilation_matrix = np.diag([dilation_coef, 1, 1])

kernel_small_dilated = scipy.ndimage.affine_transform(kernel_small_tilted, dilation_matrix)
kernel_large_dilated = scipy.ndimage.affine_transform(kernel_large_tilted, dilation_matrix)

viewer = napari.Viewer()
viewer.add_image(kernel_small_tilted, name='kernel small tilted', colormap='green', blending='additive')
viewer.add_image(kernel_large_tilted, name='kernel large tilted', colormap='red', blending='additive')
viewer.add_image(kernel_small_dilated, name='kernel small dilated', colormap='bop blue', blending='additive')
viewer.add_image(kernel_large_dilated, name='kernel large dilated', colormap='bop orange', blending='additive')
viewer.add_image(img, name='img', blending='additive')

# %% [markdown]
# The gaussian match the spots' size and orientation, but now we have again this pattern after the shear transformation.  
# We may need to generate a gaussian with a bigger sigma_z and rotate it by a higher angle to have an equivalent kernel, but without stripes.

# %%
def shear_sigma_theta(sigma_0, theta_0, coef):
    """
    Compute new hypothenuse and angle given by sigma and theta
    after dilating the opposite side of theta by a coefficient.
    """
    # side that is elongated
    L0 = sigma_0 * np.sin(theta_0)
    # angle contacting the hypothenuse, opposite to sheared side.
    d = sigma_0 * np.cos(theta_0)
    # elongated side
    L1 = coef * L0
    # elongated hypothenuse
    sigma_1 = (L1**2 + d**2)**0.5
    # new angle
    theta_1 = np.arcsin(L1 / sigma_1)
    
    return sigma_1, theta_1

# %%
1 / np.sin(30/180*np.pi)

# %%
# need fator of more or less 1.5 to match closer the tilt of spots
dilation_coef = dstage / dc / np.sin(30/180*np.pi) # * 1.5
sigma_z_small_elongated, theta_elongated = shear_sigma_theta(filter_sigma_small[0], theta, dilation_coef)
sigma_z_large_elongated, _ = shear_sigma_theta(filter_sigma_large[0], theta, dilation_coef)

theta_elongated = np.abs(theta_elongated * 180 / np.pi)
print(f"dilation_coef: {dilation_coef}")
print(f"theta_elongated: {theta_elongated}")

# %%
# naive tilt
kernel_small_straight = localize.get_filter_kernel(sigmas_small, pixel_sizes, sigma_cutoff=3)
kernel_large_straight = localize.get_filter_kernel(sigmas_large, pixel_sizes, sigma_cutoff=3)
kernel_small_raw_tilted = scipy.ndimage.rotate(kernel_small_straight, angle=30, axes=(1, 0), reshape=True)
kernel_large_raw_tilted = scipy.ndimage.rotate(kernel_large_straight, angle=30, axes=(1, 0), reshape=True)

# shear
sigmas_small_elongated = np.array(sigmas_small)
sigmas_small_elongated[0] = sigma_z_small_elongated
sigmas_large_elongated = np.array(sigmas_large)
sigmas_large_elongated[0] = sigma_z_large_elongated

kernel_small_straight = localize.get_filter_kernel(sigmas_small_elongated, pixel_sizes, sigma_cutoff=3)
kernel_large_straight = localize.get_filter_kernel(sigmas_large_elongated, pixel_sizes, sigma_cutoff=3)
kernel_small_tilted = scipy.ndimage.rotate(kernel_small_straight, angle=theta_elongated, axes=(1, 0), reshape=True)
kernel_large_tilted = scipy.ndimage.rotate(kernel_large_straight, angle=theta_elongated, axes=(1, 0), reshape=True)

viewer = napari.Viewer()
viewer.add_image(kernel_small_raw_tilted, name='kernel small raw tilted', colormap='green', blending='additive')
viewer.add_image(kernel_large_raw_tilted, name='kernel large raw tilted', colormap='red', blending='additive')
viewer.add_image(kernel_small_tilted, name='kernel small tilted', colormap='bop blue', blending='additive')
viewer.add_image(kernel_large_tilted, name='kernel large tilted', colormap='bop orange', blending='additive')
viewer.add_image(img, name='img', blending='additive')

# %%
kernel_small = kernel_small_tilted
kernel_large = kernel_large_tilted

# %%
img_high_pass = localize.filter_convolve(img, kernel_small, use_gpu=False)
img_low_pass = localize.filter_convolve(img, kernel_large, use_gpu=False)
# img_high_pass = localize.filter_convolve(img, np.flip(kernel_small, axis=1), use_gpu=False)
# img_low_pass = localize.filter_convolve(img, np.flip(kernel_large, axis=1), use_gpu=False)
img_filtered = img_high_pass - img_low_pass
del img_high_pass
del img_low_pass
gc.collect()

# %%
viewer = napari.Viewer()
# viewer.add_image(img_high_pass, name='img_high_pass')
# viewer.add_image(img_low_pass, name='img_low_pass')
viewer.add_image(img, name='img')
viewer.add_image(img_filtered, name='img_filtered')

# %% [markdown]
# Jump to section Threshold DoG or run next section about squeezing the image.

# %% [markdown]
# #### Squeeze image

# %% [markdown]
# Since the image is kind of elongated in the z (and y) direction, a different way than making smooth tilted gaussian kernel is to "squeeze back" the image, hoping to get smooth spots, or spots with less intensity modulation along their longest axis, to avoid detecting several local maxima per spot.

# %%
dilation_coef = dstage / dc
dilation_matrix = np.diag([1/dilation_coef, 1, 1])

img_squeezed = scipy.ndimage.affine_transform(img, dilation_matrix)

# %%
viewer = napari.Viewer()
viewer.add_image(img, name='img', blending='additive')
viewer.add_image(img_squeezed, name='img squeezed', blending='additive')

# %% [markdown]
# We still "cubes" within spots, but there is no intensity fluctuation along spots.

# %%
# naive tilt
kernel_small_straight = localize.get_filter_kernel(sigmas_small, pixel_sizes, sigma_cutoff=3)
kernel_large_straight = localize.get_filter_kernel(sigmas_large, pixel_sizes, sigma_cutoff=3)
kernel_small_raw_tilted = scipy.ndimage.rotate(kernel_small_straight, angle=30, axes=(1, 0), reshape=True)
kernel_large_raw_tilted = scipy.ndimage.rotate(kernel_large_straight, angle=30, axes=(1, 0), reshape=True)

viewer = napari.Viewer()
viewer.add_image(kernel_small_raw_tilted, name='kernel small raw tilted', colormap='green', blending='additive')
viewer.add_image(kernel_large_raw_tilted, name='kernel large raw tilted', colormap='red', blending='additive')
viewer.add_image(img_squeezed, name='img squeezed', blending='additive')

# %%


# %%


# %%


# %%


# %% [markdown]
# ### Threshold DoG and local max

# %%
# threshold found with Napari
dog_thresh = 3
img_filtered[img_filtered < dog_thresh] = 0

# %%
# footprint = localize.get_max_filter_footprint(min_separations=min_separations, drs=pixel_sizes)
# # fit roi size
roi_size = (5 * sigma_z, 12 * sigma_xy, 12 * sigma_xy)
# # assume points closer together than this come from a single bead
min_spot_sep = np.array((3 * sigma_z, 3 * sigma_xy))
dz_min, dxy_min = min_spot_sep
footprint = localize_skewed.get_skewed_footprint((dz_min, dxy_min, dxy_min), dc, dstage, theta)
print(footprint.shape)
# min_separations = footprint.shape # (10, 3, 3)
# array of size nz, ny, nx of True

# %%
viewer = napari.Viewer()
viewer.add_image(np.flip(footprint, axis=0), name='footprint', blending='additive')
viewer.add_image(kernel_small_raw_tilted, name='kernel small raw tilted', colormap='green', blending='additive')
viewer.add_image(kernel_large_raw_tilted, name='kernel large raw tilted', colormap='red', blending='additive')
viewer.add_image(kernel_small_tilted, name='kernel small tilted', colormap='bop blue', blending='additive')
viewer.add_image(kernel_large_tilted, name='kernel large tilted', colormap='bop orange', blending='additive')
viewer.add_image(img, name='img', blending='additive')

# %% [markdown]
# The footprint needs to be flipped.  
# It's a big one, maybe try a smaller one latter.

# %%
# we could remove the thresholding within each find_peak_candidates call
# no: ndimage.maximum_filter returns same image size with real values, need image == im_max
# thus need to filter with threshold to avoid zeros or low values
# TODO: use gradient on whole image could speed up global process
centers_guess_inds, amps = localize.find_peak_candidates(img_filtered, np.flip(footprint, axis=0), threshold=dog_thresh, use_gpu_filter=False)

# %%
centers_guess_inds

# %%
print(centers_guess_inds.shape)

# %%
viewer = napari.Viewer()
viewer.add_image(img, name='img')
viewer.add_image(img_filtered, name='img_filtered')
viewer.add_points(centers_guess_inds, name='local maxis', blending='additive', size=3, face_color='r')

# %% [markdown]
# ### Fit gaussian

# %%
roi_size_realspace = (5 * sigma_z, 12 * sigma_xy, 12 * sigma_xy)
fit_roi_sizes = np.array(localize_skewed.get_skewed_roi_size(roi_size_realspace, theta, dc, dstage, ensure_odd=True))
print("roi small:", kernel_small.shape)
print("roi large:", kernel_large.shape)
print("Peter's fit_roi_sizes:", fit_roi_sizes)

# %%
# fit_roi_sizes = np.array([1.3, 1, 1]) * np.array([sz, sy, sx])
fit_roi_sizes = 1 * np.array(kernel_small.shape) #np.array([sz, sy, sx])
# This method gives ROIs orthogonal to spots:
# roi_size_realspace = (5 * sigma_z, 12 * sigma_xy, 12 * sigma_xy)
# fit_roi_sizes = np.array(localize_skewed.get_skewed_roi_size(roi_size_realspace, theta, dc, dstage, ensure_odd=True))
# min_fit_roi_sizes = fit_roi_sizes * 0.7
min_fit_roi_sizes = fit_roi_sizes * 0.75


# average multiple points too close together
# quite long and not so efficient to fuse points
# either discard or tweak parameters
roi_coords, roi_sizes = get_roi_coordinates(
    centers = centers_guess_inds, 
    sizes = fit_roi_sizes, 
    max_coords_val = np.array(img.shape) - 1, 
    min_sizes = [0, 0, 0],
)
centers_guess = roi_coords[:, 0, :] + (roi_sizes / 2)
centers_guess = centers_guess.astype(int)
inds = np.ravel_multi_index(centers_guess.transpose(), img_filtered.shape)
weights = img_filtered.ravel()[inds]
# weights = img_filtered[centers_guess]
centers_guess, inds_comb = localize.filter_nearby_peaks(centers_guess, dxy_min, dz_min, weights=weights,
                                                        mode="average")

amps = amps[inds_comb]
print("Found %d points separated by dxy > %0.5g and dz > %0.5g" %
      (len(centers_guess), dxy_min, dz_min))

# %%
centers_guess

# %%
roi_coords, roi_sizes = get_roi_coordinates(
    centers = centers_guess_inds, 
    sizes = fit_roi_sizes, 
    max_coords_val = np.array(img.shape) - 1, 
    min_sizes = min_fit_roi_sizes,
)
nb_rois = roi_coords.shape[0]

# %%
nb_rois

# %%
centers_guess

# %%
viewer = napari.Viewer()
viewer.add_image(img, name='img')
viewer.add_image(img_filtered, name='img_filtered')
# viewer.add_points(centers_guess, name='centers_guess', blending='additive', size=3, face_color='r')
viewer.add_points(roi_coords[:, 0, :], name='ROI start', blending='additive', size=3, face_color='r')
viewer.add_points(roi_coords[:, 1, :], name='ROI end', blending='additive', size=3, face_color='g')

# %%
# viewer = napari.Viewer()
# # all_rois = np.stack(extract_ROI(img, roi_coords[i]) for i in range(nb_rois))
# # viewer.add_image(all_rois, name='all rois')
# for i in range(nb_rois):
#     roi = extract_ROI(img, roi_coords[i])
#     viewer.add_image(roi, name=f'roi {i}', blending='additive')

# %%
im_fitted = img
viewer = napari.Viewer()
for i in range(20):
    roi = extract_ROI(im_fitted, roi_coords[i])
    viewer.add_image(roi, name=f'roi {i}', visible=False)

# %%
i = 6
# im_fitted = img_high_pass - img_low_pass
im_fitted = img

roi = extract_ROI(im_fitted, roi_coords[i])
# roi_gauss = extract_ROI(img_high_pass, roi_coords[i])

viewer = napari.Viewer()
viewer.add_image(roi, name=f'roi {i}')
# viewer.add_image(roi_gauss, name='roi gauss')

# %%
# centers_guess = (roi_sizes / 2)
centers_guess = roi_sizes / 2

# %%
init_params = np.array([
    amps[i], 
    centers_guess[i, 2],
    centers_guess[i, 1],
    centers_guess[i, 0],
    sigma_xy, 
    sigma_z, 
    roi.min(),
])
print(init_params)

# %% [markdown]
# Do we finally use `get_roi_mask` to set to 0 or minimum of image the corners of the ROI?

# %%
# theta already defined as 30 * np.pi / 180

fit_results = localize.fit_gauss_roi(
    roi, 
    (localize.get_coords(roi_sizes[i], drs=[1, 1, 1])), 
    init_params,
#     estimator="LSE",
#     model="gaussian",
    sf=1, 
    dc=dc, 
    angles=(0., theta, 0.),
#     use_gpu=False,
)
fit_results

# %%
amplitude, center_x, center_y, center_z, sigma_xy, sigma_z, offset = fit_results['fit_params']

# %%
viewer = napari.Viewer()
viewer.add_image(roi, name='roi')
viewer.add_image(roi_gauss, name='roi gauss')
viewer.add_points([center_z, center_y, center_x], name='fitted center', blending='additive', size=2, face_color='r')

# %%
# # using img_high_pass or img_filtered gives really bad results
# # I'd like to understand why fitted centered are all shifted
# im_fitted = img #img_high_pass - img_low_pass # img

# fit_results_rois = np.zeros((nb_rois, 8))
# for i in range(nb_rois):
#     # extract ROI
#     roi = extract_ROI(im_fitted, roi_coords[i])
#     # fit gaussian in ROI
#     init_params = np.array([
#         amps[i], 
#         centers_guess[i, 2],
#         centers_guess[i, 1],
#         centers_guess[i, 0],
#         sigma_xy, 
#         sigma_z, 
#         roi.min(),
#     ])
#     fit_results_roi = localize.fit_gauss_roi(
#         roi, 
#         (localize.get_coords(roi_sizes[i], drs=[1, 1, 1])), 
#         init_params,
#     )
#     # amplitude, center_x, center_y, center_z, sigma_xy, sigma_z, offset
#     fit_results_rois[i, :7] = fit_results_roi['fit_params']
#     fit_results_rois[i, 7] = fit_results_roi['chi_squared']
# # add origin coordinates of each ROI
# centers = fit_results_rois[:, 1:4] + roi_coords[:, 0, :]

# %%
# using img_high_pass or img_filtered gives really bad results
# I'd like to understand why fitted centered are all shifted
im_fitted = img #img_high_pass - img_low_pass # img

amplitudes = []
centers = []
sigmas = []
chi_squareds = []
all_res = []
for i in range(nb_rois):
    # extract ROI
    roi = extract_ROI(im_fitted, roi_coords[i])
    # fit gaussian in ROI
    init_params = np.array([
        amps[i], 
        centers_guess[i, 2],
        centers_guess[i, 1],
        centers_guess[i, 0],
        sigma_xy, 
        sigma_z, 
        roi.min(),
    ])
    fit_results = localize.fit_gauss_roi(
        roi, 
        (localize.get_coords(roi_sizes[i], drs=[1, 1, 1])), 
        init_params,
        fixed_params=np.full_like(init_params, False),
    )
    amplitude, center_x, center_y, center_z, sigma_xy, sigma_z, offset = fit_results['fit_params']
    amplitudes.append(amplitude)
    centers.append([center_z, center_y, center_x])
    sigmas.append([sigma_xy, sigma_z])
    chi_squareds.append(fit_results['chi_squared'])
    all_res.append(fit_results['fit_params'])
#     print(fit_results)
# add origin coordinates of each ROI
centers = np.array(centers) + roi_coords[:, 0, :]

# %%
np.array(all_res)[:,-4:]
# np.array(all_res)[:,:4]

# %%
viewer = napari.Viewer()
viewer.add_image(img, name='img')
viewer.add_image(img_filtered, name='img_filtered')
# viewer.add_image(im_fitted, name='im_fitted')
viewer.add_points(centers_guess_inds, name='local maxis', blending='additive', size=3, face_color='r')
viewer.add_points(centers, name='fitted centers', blending='additive', size=3, face_color='g')
# viewer.add_points(centers, name='fitted centers', blending='additive', size=3, face_color=chi_squareds, face_colormap=cmap); # napari colormap doesn't work
# viewer.add_points(centers, name='fitted centers chi squared', blending='additive', size=3, face_color=chi_colors)
# viewer.add_points(centers, name='fitted centers sigma xy', blending='additive', size=3, face_color=sigma_xy_colors)

# %% [markdown]
# ## End

# %%
gc.collect()


