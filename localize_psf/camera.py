import numpy as np
from localize_psf import fit_psf

# image simulation
def adc2photons(img, gain_map, background_map):
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


def simulated_img(ground_truth, gains, offsets, readout_noise_sds, psf=None, photon_shot_noise=True, bin_size=1):
    """
    Convert ground truth image to simulated camera image including the effects of
    photon shot noise and camera readout noise.

    :param ground_truth: Ground truth mean photon counts (before binning)
    :param gains: gains at each camera pixel (ADU/e)
    :param offsets: offsets of each camera pixel (ADU)
    :param readout_noise_sds: standard deviation characterizing readout noise at each camera pixel (ADU)
    :param psf: point-spread function
    :param bool photon_shot_noise: turn on/off photon shot-noise
    :param int bin_size: bin pixels before applying Poisson/camera noise. This is to allow defining a pattern on a
    finer pixel grid.

    :return img:
    :return snr:
    """

    # optional blur image with PSF
    if psf is not None:
        img_blurred = fit_psf.blur_img_psf(ground_truth, psf)
    else:
        img_blurred = ground_truth

    # ensure non-negative
    img_blurred[img_blurred < 0] = 0

    # resample image by binning
    img_blurred = bin(img_blurred, (1, bin_size, bin_size), mode='sum')

    # add shot noise
    if photon_shot_noise:
        img_shot_noise = np.random.poisson(img_blurred)
    else:
        img_shot_noise = img_blurred

    # generate camera noise in ADU
    readout_noise = np.random.standard_normal(img_shot_noise.shape) * readout_noise_sds

    # convert from photons to ADU
    img = gains * img_shot_noise + readout_noise + offsets

    # spatially resolved signal-to-noise ratio
    # get noise from adding in quadrature, assuming photon number large enough ~gaussian
    snr = (gains * img_blurred) / np.sqrt(readout_noise_sds ** 2 + gains ** 2 * img_blurred)

    return img, snr

def bin(img, bin_sizes, mode="sum"):
    """
    bin image by combining adjacent pixels

    @param img: img of size n0 x n1 x ... x n_{m-1}
    @param bin_sizes: list [b0, ..., b_{m-1}]
    @param mode: "sum" or "mean"
    @return img_binned:
    """

    if len(bin_sizes) != img.ndim:
        raise ValueError("img must have same number of dimensions as bin_sizes")

    for nb, nd in zip(bin_sizes, img.shape):
        if nd % nb != 0:
            raise ValueError("dimension of size %d cannot be divided into %d bins" % (nd, nb))

    new_shape = [[nd // nb, nb] for nb, nd in zip(bin_sizes, img.shape)]
    new_shape = [v for sub_list in new_shape for v in sub_list]
    img_reshape = img.reshape(new_shape)

    sum_axes = tuple(range(1, 2*img.ndim, 2))
    if mode == "sum":
        img_binned = np.sum(img_reshape, axis=sum_axes)
    elif mode == "mean":
        img_binned = np.mean(img_reshape, axis=sum_axes)
    else:
        raise ValueError("'mode' must be 'sum' or 'mean' but was '%s'" % mode)

    return img_binned
