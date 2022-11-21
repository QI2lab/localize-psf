#!/usr/bin/env python

import re
from npy2bdv import BdvEditor
import pandas as pd
import numpy as np
from pycromanager import Dataset

def read_metadata(fname):
    """
    Read data from csv file consisting of one line giving titles, and the other giving values. Return as dictionary

    :param fname:
    :return metadata:
    """
    scan_data_raw_lines = []

    with open(fname, "r") as f:
        for line in f:
            scan_data_raw_lines.append(line.replace("\n", ""))

    titles = scan_data_raw_lines[0].split(",")

    # convert values to appropriate datatypes
    vals = scan_data_raw_lines[1].split(",")
    for ii in range(len(vals)):
        if re.fullmatch("\d+", vals[ii]):
            vals[ii] = int(vals[ii])
        elif re.fullmatch("\d*.\d+", vals[ii]):
            vals[ii] = float(vals[ii])
        elif vals[ii].lower() == "False".lower():
            vals[ii] = False
        elif vals[ii].lower() == "True".lower():
            vals[ii] = True
        else:
            # otherwise, leave as string
            pass

    # convert to dictionary
    metadata = {}
    for t, v in zip(titles, vals):
        metadata[t] = v

    return metadata

def write_metadata(data_dict, save_path):
    """

    :param data_dict: dictionary of metadata entries
    :param save_path:
    :return:
    """
    pd.DataFrame([data_dict]).to_csv(save_path)


def return_data_numpy(dataset, time_axis, channel_axis, num_images, excess_images, y_pixels,x_pixels):
    """
    :param dataset: pycromanager dataset object
    :param channel_axis: integer channel index
    :param time_axis: integer time_axis
    :param num_images: integer for number of images to return 
    :param y_pixels: integer for y pixel size
    :param x_pixels: integer for x pixel size
    :return data_numpy: 3D numpy array of requested data
    """

    data_numpy = np.empty([(num_images-excess_images),y_pixels,x_pixels]).astype(np.uint16)
    j = 0
    for i in range(excess_images,num_images):
        if (time_axis is None):
            if (channel_axis is None):
                data_numpy[j,:,:] = dataset.read_image(z=i)
            else:
                data_numpy[j,:,:] = dataset.read_image(z=i, c=channel_axis)
        else:
            if (channel_axis is None):
                data_numpy[j,:,:] = dataset.read_image(z=i, t=time_axis)
            else:
                data_numpy[j,:,:] = dataset.read_image(z=i, t=time_axis, c=channel_axis)
        j = j + 1

    return data_numpy


def return_data_numpy_widefield(dataset, channel_axis, ch_BDV_idx, num_z, y_pixels,x_pixels):
    """
    :param dataset: pycromanager dataset object
    :param channel_axis: integer channel index
    :param time_axis: integer time_axis
    :param num_images: integer for number of images to return 
    :param y_pixels: integer for y pixel size
    :param x_pixels: integer for x pixel size
    :return data_numpy: 3D numpy array of requested data
    """

    data_numpy = np.empty([num_z,y_pixels,x_pixels]).astype(np.uint16)

    for i in range(num_z):
        if (channel_axis is None):
            data_numpy[i,:,:] = dataset.read_image(z=i)
        else:
            data_numpy[i,:,:] = dataset.read_image(z=i, c=channel_axis, channel=ch_BDV_idx)

    return data_numpy


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

def stitch_data(path_to_xml,iterative_flag):

    """
    :param path_to_xml: Path
        path to BDV XML. BDV H5 must be present for loading
    :param iterative_flag: Bool
        flag if multiple rounds need to be aligned
    """


    # TO DO: 1. write either pyimagej bridge + macro OR call FIJI/BigStitcher in headless mode.
    #        2. fix flipped x-axis between Python and FIJI. Easier to flip data in Python than deal with
    #           annoying affine that flips data.


def return_affine_xform(path_to_xml,r_idx,y_idx,z_idx,total_z_pos):

    """
    :param path_to_xml: Path
        path to BDV XML. BDV H5 must be present for loading
    :param r_idx: integer
        round index
    :param t_idx: integer
        time index
    :param y_idx: integer 
        y tile index
    :param z_idx: integer 
        z tile index
    :return data_numpy: NDarray
        4D numpy array of all affine transforms
    """ 

    bdv_editor = BdvEditor(str(path_to_xml))
    tile_idx = (y_idx+z_idx)+(y_idx*(total_z_pos-1))

    affine_xforms = []
    read_affine_success = True
    affine_idx = 0
    while read_affine_success:
        try:
            affine_xform = bdv_editor.read_affine(time=r_idx,illumination=0,channel=0,tile=tile_idx,angle=0,index=affine_idx)
        except:
            read_affine_success = False
        else:
            affine_xforms.append(affine_xform)
            affine_idx = affine_idx + 1
            read_affine_success = True

    return affine_xforms

def open_NDTiff(dataset, channels=None, squeeze=True):
    """
    Open an NDTiff image.

    Parameters
    ----------
    dataset : NDTiff object
        NDTiff dataset.
    channels : None or list(int)
        If None, load all channels, else load a list of channels.
    
    Returns:
    img : ndarray
        A numpy ndimage.
    """

    # use metadata to guess how to load the image
    meta = dataset.axes
    c = np.array([x for x in meta['c']])  # array([1, 2, 3])
    chan = np.array([x for x in meta['channel']])
    # print('c:', c)
    # print('chan:', chan)
    # print('unique:', np.unique(c - chan))
    # shift = np.unique(c - chan)
    if len(np.unique(chan)) == 1 and len(np.unique(c)) > 1:
        chs = {x: chan[0] for x in np.unique(c)}
    else:
        chs = {x: y for x, y in zip(np.unique(c), np.unique(chan))}
    # shift = int(np.unique(c - chan)[0])
    z_levels = np.array([x for x in meta['z']])
    # load one image to get more info
    sample = dataset.read_image(z=int(np.median(z_levels)), c=c[0], channel=chs[c[0]])
    nb_z = z_levels.size
    nb_y, nb_x = sample.shape

    # iterativelly load all z planes of all channels
    if channels is None:
        channels = list(c) # convert to list for downstream compatibility
    else:
        if isinstance(channels, int):
            channels = [channels]
        # detect what could go wrong
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
        for z in z_levels:
            img[i, z, :, :] = dataset.read_image(z=z, c=channel_id, channel=chs[channel_id])
    
    if squeeze:
        img = np.squeeze(img)
    return img