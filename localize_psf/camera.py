"""
Tools for working with or simulating camera data
"""
from typing import Optional, Union
from collections.abc import Sequence
import numpy as np
from localize_psf.fit_psf import blur_img_psf

try:
    import cupy as cp
except ImportError:
    cp = None

if cp:
    array = Union[np.ndarray, cp.ndarray]
else:
    array = np.ndarray


def adc2photons(img: array,
                gain_map: array,
                background_map: array,
                precision: float = 0.) -> array:
    """
    Convert ADC counts to photon number

    :param img:
    :param gain_map:
    :param background_map:
    :param precision:
    :return photons: array of same size as img. Will be of type float
    """

    # subtraction
    photons = (img.astype(float) - background_map) / gain_map

    # set anything less than value to zero
    photons[photons < precision] = 0.

    return photons


def simulated_img(ground_truth: array,
                  gains: Union[array, float, int],
                  offsets: Union[array, float, int],
                  readout_noise_sds: Union[array, float, int],
                  psf: Optional[array] = None,
                  photon_shot_noise: bool = True,
                  bin_size: int = 1,
                  apodization: array = 1,
                  saturation: Optional[int] = None,
                  image_is_integer: bool = True) -> (array, array):
    """
    Convert ground truth image to simulated camera image including the effects of
    photon shot noise, camera readout noise, and saturation

    :param ground_truth: Ground truth mean photon counts (before binning)
    :param gains: gains at each camera pixel (ADU/e)
    :param offsets: offsets of each camera pixel (ADU)
    :param readout_noise_sds: standard deviation characterizing readout noise at each camera pixel (ADU)
    :param psf: point-spread function. If not provided, do not blur
    :param photon_shot_noise: turn on/off photon shot-noise
    :param bin_size: bin pixels before applying Poisson/camera noise. This is to allow defining a pattern on a
      finer pixel grid.
    :param apodization: apodization used during PSF blurring
    :param saturation: set any values in final image larger than this value to this value
    :param image_is_integer: force image to be integer value
    :return img, snr:
    """

    if cp and isinstance(ground_truth, cp.ndarray):
        xp = cp
    else:
        xp = np

    ground_truth = xp.asarray(ground_truth)

    # optional blur image with PSF
    if psf is not None:
        img_blurred = blur_img_psf(ground_truth, psf, apodization=apodization).real
    else:
        img_blurred = ground_truth

    # ensure non-negative
    img_blurred[img_blurred < 0] = 0

    # resample image by binning
    bin_size_list = (1,) * (img_blurred.ndim - 2) + (bin_size, bin_size)
    img_binned = bin(img_blurred, bin_size_list, mode='sum')

    # add shot noise
    if photon_shot_noise:
        img_shot_noise = xp.random.poisson(img_binned)
    else:
        img_shot_noise = img_binned

    # generate camera noise in ADU
    readout_noise = xp.random.standard_normal(img_shot_noise.shape) * readout_noise_sds

    # convert from photons to ADU
    # todo: is this the appropriate way to convert to integer, or does it introduce some bias?
    img = gains * img_shot_noise + readout_noise + offsets

    if image_is_integer:
        img = xp.round(img).astype(int)

    img[img < 0] = 0

    if saturation is not None:
        img[img > saturation] = saturation

    # spatially resolved signal-to-noise ratio
    # get noise from adding in quadrature, assuming photon number large enough ~gaussian
    snr = (gains * img_binned) / xp.sqrt(readout_noise_sds ** 2 + gains ** 2 * img_binned)

    return img, snr


def bin(img: array,
        bin_sizes: Sequence[int],
        mode: str = "sum") -> array:
    """
    Bin image by summing or averaging adjacent pixels

    :param img: NumPy or CuPy array of size n0 x n1 x ... x n_{m-1}
    :param bin_sizes: [bk, ..., b_{m-1}] amount to bin for the last m-k dimensions of img.
      Must be shorter than img.ndim,
    :param mode: "sum" or "mean"
    :return img_binned: binned image
    """

    if cp and isinstance(img, cp.ndarray):
        xp = cp
    else:
        xp = np

    img = xp.asarray(img)

    # if bin sizes is not the same size as img dims, apply it to the last len(bin_sizes) dimensions
    bin_sizes = tuple(bin_sizes)
    bin_sizes = (1,) * (img.ndim - len(bin_sizes)) + bin_sizes

    if len(bin_sizes) != img.ndim:
        raise ValueError("img must have same number of dimensions as bin_sizes")

    for ii, (nb, nd) in enumerate(zip(bin_sizes, img.shape)):
        if nd % nb != 0 and nb != 1:
            raise ValueError(f"dimension of size {nd:d} cannot be divided into {nb:d} bins along axis {ii:d}")

    new_shape = [[nd // nb, nb] for nb, nd in zip(bin_sizes, img.shape)]
    new_shape = [v for sub_list in new_shape for v in sub_list]
    img_reshape = img.reshape(new_shape)

    sum_axes = tuple(range(1, 2*img.ndim, 2))
    if mode == "sum":
        img_binned = img_reshape.sum(axis=sum_axes)
    elif mode == "mean":
        img_binned = img_reshape.mean(axis=sum_axes)
    else:
        raise ValueError(f"'mode' must be 'sum' or 'mean' but was '{mode:s}'")

    return img_binned


def bin_adjoint(img_b: array,
                bin_sizes: Sequence[int],
                mode: str = "sum") -> array:
    """
    Binning adjoint operation. These operations are adjoing in the sense that

    <w | B*v> = <Badj * w | v>

    :param img_b: NumPy or Cupy array of size n0 x n1 x ... x n_{m-1}
    :param bin_sizes: list [by, bx]
    :param mode: "sum" or "mean"
    :return img:
    """

    if cp and isinstance(img_b, cp.ndarray):
        xp = cp
    else:
        xp = np

    ny, nx = bin_sizes

    extra_dims = img_b.ndim - 2

    if mode == "sum":
        kernel = xp.ones((1,) * extra_dims + (ny, nx))
    elif mode == "mean":
        kernel = xp.ones((1,) * extra_dims + (ny, nx)) / (ny * nx)
    else:
        raise ValueError(f"'mode' must be 'sum' or 'mean' but was '{mode:s}'")

    img = xp.kron(img_b, kernel)

    return img
