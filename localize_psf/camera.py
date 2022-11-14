import numpy as np
from localize_psf import fit_psf

_cupy_available = True
try:
    import cupy as cp
except ImportError:
    _cupy_available = False

# image simulation
def adc2photons(img: np.ndarray,
                gain_map: np.ndarray,
                background_map: np.ndarray):
    """
    Convert ADC counts to photon number
    :param img:
    :param gain_map:
    :param background_map:
    :return photons: array of same size as img. Will be of type float
    """

    # subtraction
    photons = (img.astype(float) - background_map) / gain_map

    # set anything less than zero to machine precision
    photons[photons <= 0] = np.finfo(float).eps
    return photons


def simulated_img(ground_truth: np.ndarray,
                  gains: np.ndarray,
                  offsets: np.ndarray,
                  readout_noise_sds: np.ndarray,
                  psf: np.ndarray = None,
                  photon_shot_noise: bool = True,
                  bin_size: int = 1,
                  apodization: np. ndarray = 1,
                  saturation: int = None,
                  use_gpu: bool = False):
    """
    Convert ground truth image to simulated camera image including the effects of
    photon shot noise and camera readout noise.

    :param ground_truth: Ground truth mean photon counts (before binning)
    :param gains: gains at each camera pixel (ADU/e)
    :param offsets: offsets of each camera pixel (ADU)
    :param readout_noise_sds: standard deviation characterizing readout noise at each camera pixel (ADU)
    :param psf: point-spread function
    :param photon_shot_noise: turn on/off photon shot-noise
    :param bin_size: bin pixels before applying Poisson/camera noise. This is to allow defining a pattern on a
    finer pixel grid.
    :param apodization
    :param saturation:
    :return img, snr:
    """

    if use_gpu:
        xp = cp
    else:
        xp = np

    ground_truth = xp.array(ground_truth)

    # optional blur image with PSF
    if psf is not None:
        img_blurred = fit_psf.blur_img_psf(ground_truth, psf, apodization=apodization, use_gpu=use_gpu).real
    else:
        img_blurred = ground_truth

    # ensure non-negative
    img_blurred[img_blurred < 0] = 0

    # resample image by binning
    bin_size_list = (1,) * (img_blurred.ndim - 2) + (bin_size, bin_size)
    img_blurred = bin(img_blurred, bin_size_list, mode='sum', use_gpu=use_gpu)

    # add shot noise
    if photon_shot_noise:
        img_shot_noise = xp.random.poisson(img_blurred)
    else:
        img_shot_noise = img_blurred

    # generate camera noise in ADU
    readout_noise = xp.random.standard_normal(img_shot_noise.shape) * readout_noise_sds

    # convert from photons to ADU
    # todo: is this the appropriate way to convert to integer, or does it introduce some bias?
    img = xp.round(gains * img_shot_noise + readout_noise + offsets).astype(int)
    img[img < 0] = 0

    if saturation is not None:
        img[img > saturation] = saturation


    # spatially resolved signal-to-noise ratio
    # get noise from adding in quadrature, assuming photon number large enough ~gaussian
    snr = (gains * img_blurred) / xp.sqrt(readout_noise_sds ** 2 + gains ** 2 * img_blurred)

    return img, snr

def bin(img: np.ndarray,
        bin_sizes: list[int],
        mode: str = "sum",
        use_gpu: bool = False):
    """
    bin image by combining adjacent pixels

    @param img: img of size n0 x n1 x ... x n_{m-1}
    @param bin_sizes: list [b0, ..., b_{m-1}]
    @param mode: "sum" or "mean"
    @return img_binned:
    """

    if use_gpu:
        xp = cp
    else:
        xp = np

    img = xp.array(img)

    if len(bin_sizes) != img.ndim:
        raise ValueError("img must have same number of dimensions as bin_sizes")

    for nb, nd in zip(bin_sizes, img.shape):
        if nd % nb != 0:
            raise ValueError(f"dimension of size {nd:d} cannot be divided into {nb:d} bins")

    new_shape = [[nd // nb, nb] for nb, nd in zip(bin_sizes, img.shape)]
    new_shape = [v for sub_list in new_shape for v in sub_list]
    img_reshape = img.reshape(new_shape)

    sum_axes = tuple(range(1, 2*img.ndim, 2))
    if mode == "sum":
        img_binned = img_reshape.sum(axis=sum_axes)
    elif mode == "mean":
        img_binned = img_reshape.sum(axis=sum_axes)
    else:
        raise ValueError(f"'mode' must be 'sum' or 'mean' but was '{mode:s}'")

    return img_binned
