import numpy as np
from localize_psf import fit_psf

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


def simulated_img(ground_truth,
                  gains,
                  offsets,
                  readout_noise_sds,
                  psf=None,
                  photon_shot_noise=True,
                  bin_size=1,
                  apodization=1,
                  saturation=None):
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
    :param apodization
    :param saturation:
    :return img, snr:
    """

    # optional blur image with PSF
    if psf is not None:
        img_blurred = fit_psf.blur_img_psf(ground_truth, psf, apodization=apodization).real
    else:
        img_blurred = ground_truth

    # ensure non-negative
    img_blurred[img_blurred < 0] = 0

    # resample image by binning
    bin_size_list = (1,) * (img_blurred.ndim - 2) + (bin_size, bin_size)
    img_blurred = bin(img_blurred, bin_size_list, mode='sum')

    # add shot noise
    if photon_shot_noise:
        img_shot_noise = np.random.poisson(img_blurred)
    else:
        img_shot_noise = img_blurred

    # generate camera noise in ADU
    readout_noise = np.random.standard_normal(img_shot_noise.shape) * readout_noise_sds

    # convert from photons to ADU
    # todo: is this the appropriate way to convert to integer, or does it introduce some bias?
    img = np.round(gains * img_shot_noise + readout_noise + offsets).astype(int)
    img[img < 0] = 0

    if saturation is not None:
        img[img > saturation] = saturation


    # spatially resolved signal-to-noise ratio
    # get noise from adding in quadrature, assuming photon number large enough ~gaussian
    snr = (gains * img_blurred) / np.sqrt(readout_noise_sds ** 2 + gains ** 2 * img_blurred)

    return img, snr

def bin(img: np.ndarray,
        bin_sizes: list[int],
        mode: str = "sum"):
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
            raise ValueError(f"dimension of size {nd:d} cannot be divided into {nb:d} bins")

    new_shape = [[nd // nb, nb] for nb, nd in zip(bin_sizes, img.shape)]
    new_shape = [v for sub_list in new_shape for v in sub_list]
    img_reshape = img.reshape(new_shape)

    sum_axes = tuple(range(1, 2*img.ndim, 2))
    if mode == "sum":
        img_binned = np.sum(img_reshape, axis=sum_axes)
    elif mode == "mean":
        img_binned = np.mean(img_reshape, axis=sum_axes)
    else:
        raise ValueError(f"'mode' must be 'sum' or 'mean' but was '{mode:s}'")

    return img_binned

# def bin(img: np.ndarray,
#         bin_size: list[int],
#         mode: str = 'sum') -> np.ndarray:
#     """
#     bin image by combining adjacent pixels
#
#     In 1D, this is a straightforward problem. The image is a vector,
#     I = (I[0], I[1], ..., I[nx-1])
#     and the binning operator is a nx/nx_bin x nx matrix
#     M = [[1, 1, ..., 1, 0, ..., 0, 0, ..., 0]
#          [0, 0, ..., 0, 1, ..., 1, 0, ..., 0]
#          ...
#          [0, ...,              0,  1, ..., 1]]
#     which has a tensor product structure, which is intuitive because we are operating on each run of x points independently.
#     M = identity(nx/nx_bin) \prod ones(1, nx_bin)
#     the binned image is obtained from matrix multiplication
#     Ib = M * I
#
#     In 2D, this situation is very similar. Here we take the image to be a row stacked vector
#     I = (I[0, 0], I[0, 1], ..., I[0, nx-1], I[1, 0], ..., I[ny-1, nx-1])
#     the binning operator is a (nx/nx_bin)*(ny/ny_bin) x nx*ny matrix which has a tensor product structure.
#
#     This time the binning matrix has dimension (nx/nx_bin * ny/ny_bin) x (nx * ny)
#     The top row starts with nx_bin 1's, then zero until position nx, and then ones until position nx + nx_bin.
#     This pattern continues, with nx_bin 1's starting at jj*nx for jj = 0,...,ny_bin-1. The second row follows a similar
#     pattern, but shifted by nx_bin pixels
#     M = [[1, ..., 1, 0, ..., 0, 1, ..., 1, 0,...]
#          [0, ..., 0, 1, ..., 1, ...
#     Again, this has tensor product structure. Notice that the first (nx/nx_bin) x nx entries are the same as the 1D case
#     and the whole matrix is constructed from blocks of these.
#     M = [identity(ny/ny_bin) \prod ones(1, ny_bin)] \prod  [identity(nx/nx_bin) \prod ones(1, nx_bin)]
#
#     Again, Ib = M*I
#
#     Probably this pattern generalizes to higher dimensions!
#
#     :param img: image to be binned
#     :param bin_size: [ny_bin, nx_bin] where these must evenly divide the size of the image
#     :param mode: either 'sum' or 'mean'
#     :return:
#     """
#     # todo: could also add ability to bin in this direction. Maybe could simplify function by allowing binning in
#     # arbitrary dimension (one mode), with another mode to bin only certain dimensions and leave others untouched.
#     # actually probably don't need to distinguish modes, this can be done by looking at bin_size.
#     # still may need different implementation for the cases, as no reason to flatten entire array to vector if not
#     # binning. But maybe this is not really a performance hit anyways with the sparse matrices?
#
#     # if three dimensional, bin each image
#     if img.ndim == 3:
#         ny_bin, nx_bin = bin_size
#         nz, ny, nx = img.shape
#
#         # size of image after binning
#         nx_s = int(nx / nx_bin)
#         ny_s = int(ny / ny_bin)
#
#         m_binned = np.zeros((nz, ny_s, nx_s))
#         for ii in range(nz):
#             m_binned[ii, :] = bin(img[ii], bin_size, mode=mode)
#
#     # bin 2D image
#     elif img.ndim == 2:
#         ny_bin, nx_bin = bin_size
#         ny, nx = img.shape
#
#         if ny % ny_bin != 0 or nx % nx_bin != 0:
#             raise ValueError('bin size must evenly divide image size.')
#
#         # size of image after binning
#         nx_s = int(nx/nx_bin)
#         ny_s = int(ny/ny_bin)
#
#         # matrix which performs binning operation on row stacked matrix
#         # need to use sparse matrices to bin even moderately sized images
#         bin_mat_x = sp.kron(sp.identity(nx_s), np.ones((1, nx_bin)))
#         bin_mat_y = sp.kron(sp.identity(ny_s), np.ones((1, ny_bin)))
#         bin_mat_xy = sp.kron(bin_mat_y, bin_mat_x)
#
#         # row stack img. img.ravel() = [img[0, 0], img[0, 1], ..., img[0, nx-1], img[1, 0], ...]
#         m_binned = bin_mat_xy.dot(img.ravel()).reshape([ny_s, nx_s])
#
#         if mode == 'sum':
#             pass
#         elif mode == 'mean':
#             m_binned = m_binned / (nx_bin * ny_bin)
#         else:
#             raise ValueError("mode must be either 'sum' or 'mean' but was '%s'" % mode)
#
#     # 1D "image"
#     elif img.ndim == 1:
#
#         nx_bin = bin_size[0]
#         nx = img.size
#
#         if nx % nx_bin != 0:
#             raise ValueError('bin size must evenly divide image size.')
#         nx_s = int(nx / nx_bin)
#
#         bin_mat_x = sp.kron(sp.identity(nx_s), np.ones((1, nx_bin)))
#         m_binned = bin_mat_x.dot(img)
#
#         if mode == 'sum':
#             pass
#         elif mode == 'mean':
#             m_binned = m_binned / nx_bin
#         else:
#             raise ValueError("mode must be either 'sum' or 'mean' but was '%s'" % mode)
#
#     else:
#         raise ValueError("Only 1D, 2D, or 3D arrays allowed")
#
#     return m_binned
