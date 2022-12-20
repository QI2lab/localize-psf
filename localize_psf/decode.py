"""
Functions to decode barcoded FISH experiments.
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
from localize_psf import fish
from scipy.spatial.distance import cdist
from itertools import permutations

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

def generate_nonbarcoded_spots(species, n_spots, tile_shape):
    """
    Parameters
    ----------
    species: list
        Name of species whose FISH spots are simulated
    n_spots : int | array | list
        Number of spots per species, identical for all species if it's an int,
        otherwise specific to each species if list of array.
    tile_shape : array | list
        Image dimensions, can be any length but 2 or 3 make sense.
    
    Returns
    -------
    coords : DataFrame
        Coordinates of all spots with their identity.
    codebook : dict
        Association from species to rounds.

    Example
    -------
    >>> species = ['a', 'b', 'c']
    >>> n_spots = 10
    >>> tile_shape = [50, 250, 500]
    >>> generate_nonbarcoded_spots(species, n_spots, tile_shape)
    """

    nb_species = len(species)
    nb_dims = len(tile_shape)
    if isinstance(n_spots, int):
        n_spots = [n_spots] * nb_species
    coords = []
    round_ids = []
    specs = []

    for round_id, (spec, n_spec) in enumerate(zip(species, n_spots)):
        spec_coords = [np.random.random(n_spec) * dim for dim in tile_shape]
        spec_coords = np.vstack(spec_coords).T
        coords.append(spec_coords)
        round_ids.extend([round_id] * n_spec)
        specs.extend([spec] * n_spec)
    coords = np.vstack(coords)

    if nb_dims == 2:
        coords_names = ['y', 'x']
    elif nb_dims == 3:
        coords_names = ['z', 'y', 'x']
    else:
        coords_names = [f'dim-{x}' for x in range(nb_dims)]
    
    coords = pd.DataFrame(data=coords, columns=coords_names)
    coords['rounds'] = round_ids
    coords['species'] = specs

    codebook = {key: var for key, var in zip(species, range(nb_species))}

    return coords, codebook

def identify_nonbarcoded_spots(coords, codebook):
    """
    Parameters
    ----------
    coords : DataFrame
        Coordinates of all spots.
    codebook : dict
        Association from species to rounds.
    
    Returns
    -------
    species : DataFrame
        Decoded FISH spots identities with their coordinates.

    Example
    -------
    >>> species_labels = ['a', 'b', 'c']
    >>> n_spots = 10
    >>> tile_shape = [50, 250, 500]
    >>> coords, codebook = generate_nonbarcoded_spots(species_labels, n_spots, tile_shape)
    >>> measured_coords = coords.drop(columns=['species'])
    >>> identify_nonbarcoded_spots(measured_coords, codebook)
    """

    # make dictionnary round --> species from dictionnary species --> round
    inv_codebook = {val: key for key, val in codebook.items()}
    species = coords.copy()
    species['species'] = species['rounds'].map(inv_codebook)
    return species

def generate_barcoded_spots(species, n_spots, tile_shape, codebook, noise=None):
    """
    Parameters
    ----------
    species: list
        Name of species whose FISH spots are simulated.
    n_spots : int | array | list
        Number of spots per species, identical for all species if it's an int,
        otherwise specific to each species if list of array.
    tile_shape : array | list
        Image dimensions, can be any length but 2 or 3 make sense.
    codebook : dict
        Association from species to rounds barcodes, given by
        a string of zeros and ones like '01001101'.
    
    Returns
    -------
    coords : DataFrame
        Coordinates of all spots with their identity.

    Example
    -------
    >>> species = ['a', 'b', 'c']
    >>> n_spots = 4
    >>> tile_shape = [25, 250, 500]
    >>> codebook = {'a': '01', 'b': '10', 'c': '11'}
    >>> generate_barcoded_spots(species, n_spots, tile_shape, codebook)
    """

    # TODO: implement noise in successive coordinates, use uniform distribution
    nb_species = len(species)
    nb_dims = len(tile_shape)
    nb_rounds = len(next(iter(codebook.values())))
    if isinstance(n_spots, int):
        n_spots = [n_spots] * nb_species
    coords = []
    true_coords = []
    true_specs = []
    round_ids = []
    specs = []

    for (spec, n_spec) in zip(species, n_spots):
        barcode = codebook[spec]
        # go from '01001101' to [1, 4, 5, 7]
        spec_rounds = [i for i, x in enumerate(barcode) if x == '1']
        # number of positive rounds
        n_pos_round = sum([int(x) for x in barcode])
        # total number of spots across images
        n_spec_tot  = n_spec * n_pos_round

        # generate coordinates, identical across rounds
        spec_coords = [np.random.random(n_spec) * dim for dim in tile_shape]
        spec_coords = np.vstack(spec_coords).T
        # save species coordinates to the list of unique ideal coordinates
        true_coords.append(spec_coords)
        # repeat and stack coordinates to match number of rounds
        for _ in range(n_pos_round - 1):
            spec_coords = np.vstack([spec_coords, spec_coords])
        coords.append(spec_coords)
        # indicate at what round coordinates are observed
        for round_id in spec_rounds:
            round_ids.extend([round_id] * n_spec)
        # indicate the ground truth species for all spots
        specs.extend([spec] * n_spec_tot)
        true_specs.extend([spec] * n_spec)
    coords = np.vstack(coords)
    true_coords = np.vstack(true_coords)

    if noise is not None:
        if nb_dims == 2:
            add_noise = np.random.uniform(low=-noise, high=noise, size=coords.shape)
        elif nb_dims == 3:
            if isinstance(noise, (int, float)):
                add_noise = np.random.uniform(low=-noise, high=noise, size=coords.shape)
            else:
                add_noise = np.hstack(
                    np.random.uniform(low=-noise[0], high=noise[0], size=(len(coords), 1)),
                    np.random.uniform(low=-noise[1], high=noise[1], size=(len(coords), 2))
                )
        coords = coords + add_noise

    if nb_dims == 2:
        coords_names = ['y', 'x']
    elif nb_dims == 3:
        coords_names = ['z', 'y', 'x']
    else:
        coords_names = [f'dim-{x}' for x in range(nb_dims)]
    
    coords = pd.DataFrame(data=coords, columns=coords_names)
    true_coords = pd.DataFrame(data=true_coords, columns=coords_names)
    coords['rounds'] = round_ids
    coords['species'] = specs
    true_coords['species'] = true_specs

    if noise is None:
        return coords
    else:
        return coords, true_coords


def array_to_dict(arr):
    return dict(enumerate(arr))


def dict_to_array(dico):
    return np.array(list(dico.values()))


def df_to_listarray(df, col_split, usecols=None):
    if usecols is None:
        usecols = df.columns
    listarray = [
        df.loc[df[col_split] == i, usecols].values for i in np.unique(df[col_split])
    ]
    return listarray


def compute_distances(
    source, target, dist_method="xy_z_orthog", metric="euclidean", tilt_vector=None
):
    """
    Parameters
    ----------
    source : ndarray
        Coordinates of the first set of points.
    target : ndarray
        Coordinates of the second set of points.
    dist_method : str
        Method used to compute distances. If 'isotropic', standard distances are computed considering all axes
        simultaneously. If 'xy_z_orthog' 2 distances are computed, for the xy plane and along the z axis
        respectively. If 'xy_z_tilted' 2 distances are computed for the tilted plane and its normal axis.

    Example
    -------
    >>> source = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    >>> target = np.array([[0, 0, 0], [-3, 0, 2], [0, 0, 10]])
    >>> compute_distances(source, target)
        (array([0, 4, 0]), array([0., 2., 5.]))
    >>> compute_distances(source, target, metric='L1')
        (array([0, 4, 0]), array([0, 2, 7]))

    """
    if dist_method == "isotropic":
        dist = cdist(source, target, metric=metric)
        return dist

    elif dist_method == "xy_z_orthog":
        dist_xy = cdist(source[:, 1:], target[:, 1:], metric=metric)
        dist_z = cdist(
            source[:, 0].reshape(-1, 1), target[:, 0].reshape(-1, 1), metric=metric
        )
        return dist_z, dist_xy

    elif dist_method == "xy_z_tilted":
        raise NotImplementedError("Method 'xy_z_tilted' will be implemented soon")


def find_neighbor_spots_in_round(
    source,
    target,
    dist_params,
    dist_method="xy_z_orthog",
    metric="euclidean",
    return_bool=False,
):
    """
    For each spot in a given round ("source"), find if there are neighbors
    in another round ("target") within a given distance.

    Parameters
    ----------
    source : ndarray
        Coordinates of spots in the source round.
    target : ndarray
        Coordinates of spots in the target round.
    dist_method : str
        Method used to compute distance between spots.
        Can be isotropic, or xy_z_orthog
    dist_params : float or array
        Threshold distance to classify spots as neighbors.
        Multiple threshold can be used depending on the method, typically 2
        to have a threshold for the xy plane and one for the z axis.
    return_bool : bool
        If True, return a vector indicating the presence of neighbors
        for spots in the source set.

    Returns
    -------
    pairs : ndarray
        Pairs of neighbors.
    has_neighb : array
        Array indicating the presence of neighbors for each spot in their source round.

    Example
    -------
    >>> source = np.array([[0, 0, 0],
                           [0, 2, 0]])
    >>> target = np.array([[0, 0, 0],
                           [1, 0, 0],
                           [0, 2, 0],
                           [0, 0, 3]])
    """

    # Compute all distances between spots of given round and all other spots of other round
    dist = compute_distances(source, target, dist_method=dist_method, metric=metric)
    # check if distances below threshold for all dimensions
    if dist_method == "xy_z_orthog":
        is_neighb = np.logical_and(dist[0] < dist_params[0], dist[1] < dist_params[1])
    elif dist_method == "isotropic":
        is_neighb = np.logical_and(dist < dist_params)

    if return_bool:
        # detect if there is any neighbor for each spot
        has_neighb = np.any(is_neighb, axis=1)
        return has_neighb
    else:
        # extract pairs of neighboring spots
        y, x = np.where(is_neighb)
        pairs = np.vstack([y, x]).T
        return pairs


def make_all_rounds_pairs(start=0, end=16):
    pairs_rounds = list(permutations(range(start, end), 2))
    return pairs_rounds


def assemble_barcodes(neighbors):
    """
    Parameters
    ----------
    neighbors : dict[dict[array]]
        Dictionnary of dictionnaries, where the fist level of keys is the
        set of source rounds, and the second level of key is the set of
        target round. Each second level value is an array indicating the
        presence of neighbors from spots in the source round to spots in
        the target round.

    Returns
    -------
    barcodes : dict[array]
        Dictionnary of barcodes found for each spot in a source round,
        keys indicate the id of the source round.

    Example
    -------
    >>> neighbors = {2: {1: [0, 1, 2, 3],
                         0: [4, 5, 6, 7]},
                     0: {1: [8, 9, 10, 11],
                         2: [12, 13, 14, 15]},
                     1: {2: [16, 17, 18, 19],
                         0: [20, 21, 22, 23]}}
    >>> assemble_barcodes(neighbors)
    {2: array([[4, 0, 1],
               [5, 1, 1],
               [6, 2, 1],
               [7, 3, 1]]),
     0: array([[ 1,  8, 12],
               [ 1,  9, 13],
               [ 1, 10, 14],
               [ 1, 11, 15]]),
     1: array([[20,  1, 16],
               [21,  1, 17],
               [22,  1, 18],
               [23,  1, 19]])}
    """

    # dictionary storing all barcodes matrices for each round
    barcodes = {}
    # for each round, stack vectors of neighbors into arrays across target rounds
    for round_source, round_targets in neighbors.items():
        # get sorted list of target round IDs
        round_ids = np.unique([i for i in round_targets.keys()])
        # initialize empty array
        nb_neigh = len(round_targets[round_ids[0]])
        nb_rounds = round_ids.size + 1  # because we consider current source round
        # get the type of data and choose between1, 1.0 and True
        fill_value = round_targets[round_ids[0]][0]
        if isinstance(fill_value, bool):
            fill_value = True
        elif isinstance(fill_value, int):
            fill_value = 1
        else:
            fill_value = 1.0
        # initilize array, which sets bits of the current source round to 1 or True
        round_barcode = np.full(shape=(nb_neigh, nb_rounds), fill_value=fill_value)
        # stack each vector in the array
        for round_id in round_ids:
            round_barcode[:, round_id] = round_targets[round_id]
        # save the array in the barcode dictionary
        barcodes[round_source] = round_barcode
    return barcodes


def clean_barcodes(barcodes, coords, min=3, max=5):
    """
    Remove barcodes and their corresponding coordinates if they
    have too few or too many positive bits.

    Parameters
    ----------
    barcodes : dict[array]
        Dictionnary of barcodes found for each spot in a source round,
        keys indicate the id of the source round.
    coords : list(arrays)
        Coordinates of all spots in rounds.
    min : int
        Minimum number of positive bits each barcode should have.
    max : int
        Maximum number of positive bits each barcode should have.

    Returns
    -------
    barcodes : dict[array]
        Dictionnary of barcodes found for each spot in a source round,
        keys indicate the id of the source round.
    coords : list(arrays)
        Coordinates of all spots in rounds.

    Example
    -------
    >>> barcodes = {0: np.array([[1, 1, 1, 1, 1, 1],
                                 [0, 0, 1, 1, 1, 0]]),
                    1: np.array([[1, 0, 0, 1, 0, 0],
                                 [1, 0, 1, 0, 1, 1]])}
    >>> coords = [np.array([[1, 2, 3],
                            [4, 5, 6]]),
                  np.array([[7, 8, 9],
                            [10, 11, 12]])]
    >>> clean_barcodes(barcodes, coords, min=3, max=5)
    ({0: array([[0, 0, 1, 1, 1, 0]]), 1: array([[1, 0, 1, 0, 1, 1]])},
    [array([[4, 5, 6]]), array([[10, 11, 12]])])
    """

    for rd_id, rd_barcode in barcodes.items():
        select = np.logical_and(
            rd_barcode.sum(axis=1) >= min, rd_barcode.sum(axis=1) <= max
        )
        # selection by key to be sure assignment is effective, necessary?
        barcodes[rd_id] = rd_barcode[select, :]
        coords[rd_id] = coords[rd_id][select, :]
    return barcodes, coords


def merge_barcodes_pairs_rounds(
    barcodes_1,
    barcodes_2,
    coords_1,
    coords_2,
    dist_params,
    dist_method="xy_z_orthog",
    metric="euclidean",
):
    """
    Merge barcodes and their corresponding coordinates in a pair of rounds
    when they are identical and close enough to each other.

    Parameters
    ----------
    barcodes_1 : ndarray
        Barcodes of first round, shape (n_barcodes, n_rounds).
    barcodes_2 : ndarray
        Barcodes of second round, shape (n_barcodes, n_rounds).
    coords_1 : ndarray
        Coordinates of barcodes of first round, shape (n_barcodes, dim_image).
    coords_2 : ndarray
        Coordinates of barcodes of second round, shape (n_barcodes, dim_image).
    dist_method : str
        Method used to compute distance between spots.
        Can be isotropic, or xy_z_orthog
    metric : str
        Metric used to compute distance between 2 points.
    dist_params : float or array
        Threshold distance to classify spots as neighbors.
        Multiple threshold can be used depending on the method, typically 2
        to have a threshold for the xy plane and one for the z axis.

    Returns
    -------
    barcodes_out : ndarray
        Merged barcodes.
    coords_out : ndarray
        Merged coordinates of barcodes.

    Example
    -------
    >>> barcodes_1 = np.array([[1, 1, 1, 1],
                               [0, 0, 0, 0]])
    >>> barcodes_2 = np.array([[1, 1, 1, 1],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0]])
    >>> coords_1 = np.array([[0, 0, 0],
                             [1, 2, 2]])
    >>> coords_2 = np.array([[0, 0, 0],
                             [0, 0, 0],
                             [2, 2, 2]])
    >>> merge_barcodes_pairs_rounds(barcodes_1, barcodes_2, coords_1, coords_2,
                                    dist_params=[0.6, 0.2])
        (array([[1, 1, 1, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]]),
        array([[0, 0, 0],
                [1, 2, 2],
                [0, 0, 0],
                [2, 2, 2]]))
    """

    # find all pairs of neighbors between the 2 rounds
    pairs = find_neighbor_spots_in_round(
        coords_1,
        coords_2,
        dist_params=dist_params,
        dist_method=dist_method,
        metric=metric,
    )
    # change to dictionary to delete entries / indices without shifting
    # element by more than one index if previous elements need to be discarded
    barcodes_2 = array_to_dict(barcodes_2)
    coords_2 = array_to_dict(coords_2)

    # very manual iteration to allow modification of the `pairs` array
    # while iterating over it
    k = 0
    while k < len(pairs):
        i, j = pairs[k]
        if np.all(barcodes_1[i] == barcodes_2[j]):
            # delete barcode and coordinates in the second set
            del barcodes_2[j]
            del coords_2[j]
            select = np.logical_or(np.arange(len(pairs)) <= k, pairs[:, 1] != j)
            pairs = pairs[select, :]
        k += 1
    # convert back to array for stacking and future distance computation
    barcodes_2 = dict_to_array(barcodes_2)
    coords_2 = dict_to_array(coords_2)

    # stack all remaining barcodes and coordinates
    barcodes_out = np.vstack([barcodes_1, barcodes_2])
    coords_out = np.vstack([coords_1, coords_2])

    return barcodes_out, coords_out


def make_pyramidal_pairs(base):
    """
    Make successive lists resulting from merging pairs in previous list,
    until a list of a unique pair is reached.

    Parameters
    ----------
    base : list
        A list of elements that will be successively merged.

    Returns
    -------
    pyramid : list
        A list of lists, each of them containing pairs of merged
        items from the previous list.

    Example
    -------
    >>> base = list(range(5))
    >>> make_pyramidal_pairs(base)
    [[0, 1, 2, 3, 4], [[0, 1], [2, 3], [4]], [[0, 2], [4]], [[0, 4]]]
    """

    # Make first level of the pyramid with base
    pyramid = [base]
    # First iteration in numbers, not pairs of number
    level = [
        [pyramid[-1][2 * i], pyramid[-1][2 * i + 1]]
        for i in range(len(pyramid[-1]) // 2)
    ]
    if len(pyramid[-1]) % 2 == 1:
        level.append([pyramid[-1][-1]])
    pyramid.append(level)
    # Next iterations on pairs of numbers
    while len(pyramid[-1]) > 1:
        level = [
            [pyramid[-1][2 * i][0], pyramid[-1][2 * i + 1][0]]
            for i in range(len(pyramid[-1]) // 2)
        ]
        if len(pyramid[-1]) % 2 == 1:
            level.append([pyramid[-1][-1][0]])
        pyramid.append(level)
    return pyramid


def merge_barcodes(
    barcodes, coords, dist_params, dist_method="xy_z_orthog", metric="euclidean"
):
    """
    Merge all barcodes and their corresponding coordinates in all rounds
    when they are identical and close enough to each other.

    Parameters
    ----------
    barcodes : ndarray
        Barcodes detected starting from spots in all rounds, shape (n_barcodes, n_rounds).
    coords : ndarray
        Coordinates of barcodes, shape (n_barcodes, dim_image).
    dist_method : str
        Method used to compute distance between spots.
        Can be isotropic, or xy_z_orthog
    metric : str
        Metric used to compute distance between 2 points.
    dist_params : float or array
        Threshold distance to classify spots as neighbors.
        Multiple threshold can be used depending on the method, typically 2
        to have a threshold for the xy plane and one for the z axis.

    Returns
    -------
    barcodes_out : ndarray
        Merged barcodes.
    coords_out : ndarray
        Merged coordinates of barcodes.

    Example
    -------
    >>> barcodes = {0: np.array([[1, 1, 1, 1],
                                 [1, 1, 0, 0],
                                 [1, 0, 1, 0],
                                 [1, 0, 0, 1]]),
            1: np.array([[1, 1, 1, 1],
                         [0, 1, 1, 0]]),
            2: np.array([[1, 1, 1, 1],
                         [1, 0, 1, 0]]),
            3: np.array([[1, 1, 1, 1],
                         [1, 1, 1, 1],
                         [1, 0, 0, 1]])}
    >>> coords = [np.zeros((len(barcodes[0]), 3), dtype='int'),
                  np.zeros((len(barcodes[1]), 3), dtype='int'),
                  np.zeros((len(barcodes[2]), 3), dtype='int'),
                  np.array([[0, 0, 0],
                            [5, 5, 5],
                            [0, 0, 0]], dtype='int')]
    >>> merge_barcodes(barcodes, coords, dist_params=[0.6, 0.2])
    """

    # get sorted list of round IDs
    round_ids = np.unique([i for i in barcodes.keys()])

    # Pyramidal merge of pairs of rounds
    pyram_levels = make_pyramidal_pairs(round_ids)
    # something like [[0, 1, 2, 3], [[0, 1], [2, 3]], [[0, 2]]]
    for level_pairs in pyram_levels[1:]:
        # for ex: [[0, 1], [2, 3]]
        for pair in level_pairs:
            # for ex: [0, 1]
            if len(pair) == 2:
                # avoid runing on a singlet
                barcodes_1 = barcodes[pair[0]]
                barcodes_2 = barcodes[pair[1]]
                coords_1 = coords[pair[0]]
                coords_2 = coords[pair[1]]

                barcodes[pair[0]], coords[pair[0]] = merge_barcodes_pairs_rounds(
                    barcodes_1,
                    barcodes_2,
                    coords_1,
                    coords_2,
                    dist_params=dist_params,
                    dist_method=dist_method,
                    metric=metric,
                )
                # # clean-up space
                barcodes[pair[1]] = None
                coords[pair[1]] = None
    return barcodes[0], coords[0]


def infer_species_from_barcodes(barcodes, codebook, method='deterministic'):
    """
    Guess the species identities of spots from their barcodes and a codebook.

    Parameters
    ----------
    barcodes : ndarray
        Barcodes detected starting from spots in all rounds, shape (n_barcodes, n_rounds).
    codebook : dict
        Association from species to rounds.
    method : str
        Method used for inference. Default is 'deterministic', where a single estimate is
        made for exactly matches between bacordes and the codebook. A bayesian method will
        be implemented soon.
    """

    if method == 'deterministic':
        # make dictionnary barcodes --> species from dictionnary species --> barcodes
        inv_codebook = {val: key for key, val in codebook.items()}
        # transform numerical barcodes to string ones
        barcodes = [''.join([str(int(i)) for i in barcode]) for barcode in barcodes]
        # actual decoding
        species = [inv_codebook[barcode] if barcode in inv_codebook.keys() else None for barcode in barcodes]

    elif method == 'bayesian':
        raise NotImplementedError("Method 'bayesian' will be implemented soon")

    return species, barcodes

def decode_spots(coords, codebook, dist_params, dist_method="xy_z_orthog", metric="euclidean",
                 clean_min=3, clean_max=5):
    """
    For each spot in a given round, find if there are neighbors
    in each other rounds within a given distance, and reconstruct barcodes from that.

    Parameters
    ----------
    coords : list(arrays)
        Coordinates of all spots in rounds.
    codebook : dict
        Association from species to rounds.
    dist_params : float or array
        Threshold distance to classify spots as neighbors.
        Multiple threshold can be used depending on the method, typically 2
        to have a threshold for the xy plane and one for the z axis.
    dist_method : str
        Method used to compute distance between spots.
        Can be isotropic, or xy_z_orthog
    metric : str
        Metric used to compute distance between 2 points.

    Returns
    -------
    species : list
        Species of decoded spots.
    coords : array
        Coordinates of decoded spots.
    barcode : array
        Reconstructed barcode from data around each spot location.

    """

    nb_rounds = len(coords)
    round_pairs = make_all_rounds_pairs(start=0, end=nb_rounds)

    # store all potential neighbors decected from each round to the other
    neighbors = {i: {} for i in range(nb_rounds)}
    for pair in round_pairs:
        neighbors[pair[0]][pair[1]] = find_neighbor_spots_in_round(
            coords[pair[0]],
            coords[pair[1]],
            dist_method=dist_method,
            metric=metric,
            dist_params=dist_params,
            return_bool=True,
        )
    barcodes = assemble_barcodes(neighbors)

    # remove barcodes that have too few or too many positive bits
    barcodes, coords = clean_barcodes(barcodes, coords, min=clean_min, max=clean_max)

    barcodes, coords = merge_barcodes(
        barcodes,
        coords,
        dist_params=dist_params,
        dist_method=dist_method,
        metric=metric,
    )

    species, barcodes = infer_species_from_barcodes(barcodes, codebook)

    return species, coords, barcodes


def plot_compare_decoded_spots(true_coords, true_species, guessed_coords, guessed_species, cmap=None, figsize=(10, 10)):
    """
    Plot ground truth and decoded species and coordinates of spots.

    Parameters
    ----------
    true_coords : array or dataframe
        Ground truth coordinates of spots. 
    true_species : list or array
        Ground truth species of FISH spots.
    guessed_coords : array or dataframe
        Output coordinates (merged and cleaned) from the spot decoding pipeline.
    guessed_species : list or array
        Output species from the spot decoding pipeline.
    cmap : dict
        Association between species and colors, default is None.
    figsize : tuple or array
        Size of the scatter plot.
    """
    fig, ax = plt.subplots(figsize=figsize)
    uniq_spec = np.unique(true_species)

    if cmap is None:
        palette = [mpl.colors.rgb2hex(x) for x in mpl.cm.get_cmap('tab20').colors]
        n_colors = len(uniq_spec)
        cmap = {x: palette[i % n_colors] for i, x in enumerate(uniq_spec)}

    if isinstance(true_species, (list, pd.Series)):
        true_species = np.array(true_species)
    if isinstance(true_coords, pd.DataFrame):
        true_coords = np.array(true_coords)
    for spec in uniq_spec:
        select = true_species == spec
        plot_coords = true_coords[select, -2:]
        ax.scatter(plot_coords[:, 1], plot_coords[:, 0], c=cmap[spec], label=f'true {spec}', marker='.')

    uniq_spec = np.unique(guessed_species)
    if isinstance(guessed_species, (list, pd.Series)):
        guessed_species = np.array(guessed_species)
    if isinstance(true_coords, pd.DataFrame):
        guessed_coords = np.array(guessed_coords)
    for spec in uniq_spec:
        select = guessed_species == spec
        plot_coords = guessed_coords[select, -2:]
        ax.scatter(plot_coords[:, 1], plot_coords[:, 0], label=f'guessed {spec}', marker='o', s=50, edgecolors=cmap[spec], facecolors='none')
    plt.legend()
    