# %%
"""
Pipeline to find spot in a single non-skewed xyz image tile.
The global plan is:
  - spot detection working on single non-skewed tile, no GPU, no multiprocess, no coordinates change, no file handling
    - extract single ct xyz tile with spots
  - add multiple tiles (x, y, z vary), "stich" results with orthogonal change of coordinates
  - add multiple time steps, channels
  - add Dask support per xyztc tile and merge results
  - add GPU support per tile? Manage conflict with multi-processes tile handling.
  - add support for skewed tiles
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

from tifffile import tifffile
import zarr
import dask.array as da
from dask_image.imread import imread
from dask import delayed
from skimage.io.collection import alphanumeric_key
import pycromanager
import napari
from napari.qt.threading import thread_worker
from magicgui import magicgui
# from matplotlib.colors import PowerNorm, LinearSegmentedColormap, Normalize

import localize_psf.rois as roi_fns
from localize_psf import fit
import localize_psf.fit_psf as psf
from localize_psf import localize
import localize_skewed
import image_post_processing as ipp
from image_post_processing import deskew

# %% [markdown]
# ### Extract data

# %%

dir_load = Path('../../../from_server')
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
pixel_sizes = (1, 1, 1)
kernel_small = localize.get_filter_kernel(filter_sigma_small, pixel_sizes, sigma_cutoff=2)
kernel_large = localize.get_filter_kernel(filter_sigma_large, pixel_sizes, sigma_cutoff=2)

# viewer = napari.Viewer()
# viewer.add_image(kernel_small, name='kernel small', colormap='green', blending='additive')
# viewer.add_image(kernel_large, name='kernel large', colormap='red', blending='additive')

# # %%Can we skip the second convulotion with an "normalized" kernel?
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
# maxis = scipy.ndimage.maximum_filter(img_filtered, footprint)
maxis = scipy.ndimage.maximum_filter(img_filtered, footprint=np.ones(min_separations))

# %%
np.unique(maxis)

# %%
footprint.shape

# %%
# we could remove the threshlding within each find_peak_candidates call
# no: ndimage.maximum_filter returns same image size zith real values, need image == im_max
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
# from napari.utils.colormaps.colormap_utils import vispy_or_mpl_colormap
# cmap = vispy_or_mpl_colormap('plasma')
import matplotlib as mpl
cmap = mpl.cm.get_cmap('plasma')

chi_squareds = np.array(chi_squareds)
norm = mpl.colors.Normalize(vmin=chi_squareds.min(), vmax=chi_squareds.max())
chi_colors = [cmap(norm(x)) for x in chi_squareds]

sigmas = np.array(sigmas)
norm = mpl.colors.Normalize(vmin=sigmas[:, 0].min(), vmax=sigmas[:, 0].max())
sigma_xy_colors = [cmap(norm(x)) for x in sigmas[:, 0]]

# %%
viewer = napari.Viewer()
viewer.add_image(img, name='img')
viewer.add_image(img_filtered, name='img_filtered')
# viewer.add_image(im_fitted, name='im_fitted')
viewer.add_points(centers_guess_inds, name='local maxis', blending='additive', size=3, face_color='r')
# viewer.add_points(centers, name='fitted centers', blending='additive', size=3, face_color='g')
# viewer.add_points(centers, name='fitted centers', blending='additive', size=3, face_color=chi_squareds, face_colormap=cmap); # napari colormap doesn't work
# viewer.add_points(centers, name='fitted centers chi squared', blending='additive', size=3, face_color=chi_colors)
viewer.add_points(centers, name='fitted centers sigma xy', blending='additive', size=3, face_color=sigma_xy_colors)

# %% [markdown]
# Some blobs look like real spot blobs but are actually non spot blobs, they look simimlar becaus of the DoG kernel.  
# In these blobs the diff between center from peak max and gaussian fit is noticeable.  
# For real spot blobs, sometimes the peak max seems to provide more accurarte estimation of center's coordinates, but on the real image we observe that the gaussian fit is the most accurate method with small enough ROI.
# But with too small ROI there is no real fitting, and with too big ROI one center can shift due to near spot. 

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

# %%
