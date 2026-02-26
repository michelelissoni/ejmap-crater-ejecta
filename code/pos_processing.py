"""
Filename: pos_processing.py
Author: Michele Lissoni
Date: 2026-02-16
"""

"""

Miscellaneous functions

"""

import numpy as np
import pandas as pd

import os
import glob
import re

import xarray as xr
import rioxarray

import scipy.ndimage as sni

from pyproj import Transformer

from planet_constants import Planet_Basemap, Planet_CRS

def gridPositions(ylims, xlims, image_size, start = None, spacing=0.9):

    """
    Generate tile locations on a 2D grid.
    
    Arguments:
    - ylims, xlims (tuples): the limits of the grid portion along the row and column axes where the tiles will be generated.
    - image_size (tuple): dimensions of the tile along the row and column axes. If integer, the tile is square.
                   
    Keywords:
    - start (tuple): a (row, col) coordinate where a tile will have its corner. If not set, the tiles are initialized from a random location.
    - spacing (float): the fraction of the tile size between the side of a tile and the beginning of the next one. 
                       If for example `start = (10,12)`, `image_size=(100,200)` and `spacing=0.9`, the tiles will have their upper-left corners
                       in the cells (10+n*90,12+m*180), where n and m are integers that can assume any value (positive, negative or zero) such that
                       the tiles are contained within the `ylims` and `xlims`.
    
    Returns: 
    - pos_y, pos_x: row and col coordinates of the tile upper-left corners.
    """
    
    # Check input data
    
    if(not(len(ylims)==2 and np.issubdtype(type(ylims[0]), np.integer) and np.issubdtype(type(ylims[1]), np.integer))):
        raise ValueError("'ylims' must be a sequence of integers of length 2.")

    if(not(len(xlims)==2 and np.issubdtype(type(xlims[0]), np.integer) and np.issubdtype(type(xlims[1]), np.integer))):
        raise ValueError("'xlims' must be a sequence of integers of length 2.")
        
    # Tile size 
        
    if(isinstance(image_size, int)):
        image_size_y = image_size
        image_size_x = image_size
    elif(len(image_size)==2 and np.issubdtype(type(image_size[0]), np.integer) and np.issubdtype(type(image_size[1]), np.integer)):
        image_size_y = image_size[0]
        image_size_x = image_size[1]
    else:
        raise ValueError("'image_size' must be an integer or a sequence of integers of length 2.")
        
    # Tile spacing in grid integer units

    if(isinstance(spacing, float) and spacing<1):
        spacing_y = int(spacing*image_size_y)
        spacing_x = int(spacing*image_size_x)
    elif(isinstance(spacing, int)):
        spacing_y = spacing
        spacing_x = spacing
    elif(len(spacing)==2 and np.issubdtype(type(spacing[0]), np.integer) and np.issubdtype(type(spacing[1]), np.integer)):
        spacing_y = spacing[0]
        spacing_x = spacing[1]
    else:
        raise ValueError("'spacing' value not valid.")
        
    # Starting point

    if(start is None):
        start_y = np.random.randint(ylims[0],ylims[1]-image_size_y)
        start_x = np.random.randint(xlims[0],xlims[1]-image_size_x)
    elif(len(start)==2 and np.issubdtype(type(start[0]), np.integer) and start[0]>=ylims[0] and start[0]<ylims[1]-image_size_y and np.issubdtype(type(start[1]), np.integer)  and start[1]>=xlims[0] and start[1]<xlims[1]-image_size_x):
        start_y = start[0]
        start_x = start[1]
    else:
        raise ValueError("'start' must be either None or a sequence of two integers within the grid limits.")
        
    # Define positions
    
    x_starts = np.append(np.arange(start_x, xlims[1]-image_size_x, spacing_x, dtype=int), np.arange(start_x, xlims[0], -spacing_x, dtype=int))
    x_starts = np.unique(np.append([xlims[0],xlims[1]-image_size_x], x_starts))
    y_starts = np.append(np.arange(start_y, ylims[1]-image_size_y, spacing_y, dtype=int), np.arange(start_y, ylims[0], -spacing_y, dtype=int))
    y_starts = np.unique(np.append([ylims[0],ylims[1]-image_size_y], y_starts))

    eval_pos_y = np.repeat(y_starts, len(x_starts))
    eval_pos_x = np.tile(x_starts, len(y_starts))

    return eval_pos_y, eval_pos_x
    
def posIndex(pos_num, tmp = False, crat_FID = None, pos_digits = 5, crat_digits = 4):

    """
    Assign prefixes to the tile position indices, indicating if the position is temporary 
    (i.e. must still be transferred to the tile lists for the individual craters) and/or crater-unique
    (i.e. only ejecta for a given crater have been mapped there, not all craters).
    
    For example, let us consider a tile with initial position index 301.
    
    Non-temporary, non crater-unique: 301
    Temporary, non crater-unique: 100301
    Non-temporary, crater-unique (crat_FID: 269): 269000301
    Temporary, crater-unique (crat_FID: 269): 269100301 
    
    NOTE: crater-unique positions were not in the end used for the Mercury work.
    
    Arguments:
    - pos_num (int or integer np.array): the original position index.
    - tmp (bool or bool np.array): True if the position is temporary.
    - crat_FID (int or int np.array): if the position is crater-unique, the crater FID. Otherwise, None.
    
    If arrays, the arguments must all have the same length.

    Keywords:
    - pos_digits (int): digits devoted to the position index. Default: 5.
    - crat_digits (int): digits devoted to the crater FID for crater-unique tiles. Default: 4.
    
    Returns: 
    - pos_index (int or integer np.array): the final index.
    """

    pos_num = np.array(pos_num)
    
    tmp_prefix = 10**pos_digits
    crat_prefix = 10**(pos_digits+1+crat_digits)

    if(np.any(pos_num>tmp_prefix)):
        raise ValueError("All pos nums must have no more digits than `pos_digits`.")

    tmp = np.array(tmp)
    tmp_prefix = tmp*tmp_prefix

    crat_FID = np.ones(len(tmp), dtype=int)*(-1) if crat_FID is None else np.array(crat_FID) 
    crat_FID_present = np.isfinite(crat_FID) & (crat_FID>=0)
    
    if(np.any(crat_FID[crat_FID_present]>10**crat_digits)):
        raise ValueError("All crat FIDs must have no more digits than `crat_digits`.")    

    crat_prefix = np.array(crat_FID*10**(pos_digits+1) + crat_prefix)
    crat_prefix[~crat_FID_present] = 0

    pos_index = pos_num + tmp_prefix + crat_prefix
    pos_index = int(pos_index) if len(pos_index.shape)==0 else pos_index
    
    return pos_index
    
def posIndexUnravel(pos_index, pos_digits = 5, crat_digits = 4):

    """
    Decompose the position index into original tile number, temporary status and
    crat_FID, if crater-unique.
        
    Arguments:
    - pos_index (int or integer np.array): the final index.

    Keywords:
    - pos_digits (int): digits devoted to the position index. Default: 5.
    - crat_digits (int): digits devoted to the crater FID for crater-unique tiles. Default: 4.
    
    Returns: 
    - pos_num (int or integer np.array): the original position index.
    - tmp (bool or bool np.array): True if the position is temporary.
    - crat_FID (int or int np.array): if the position is crater-unique, the crater FID. Otherwise, -1.
    """

    pos_index = np.array(pos_index)

    # The prefixes are essentially non-zero digits followed by zeros, they can therefore be treated as integers. 

    crat_prefix = 10**(pos_digits+1+crat_digits)
    tmp_prefix = 10**(pos_digits)
    
    if(np.any(pos_index>=crat_prefix*2)):
        raise ValueError(f"The {pos_digits+1+crat_digits+1}-th digit, where present, should always be 1.")   

    # Retrieve crat_FID where present

    crat_FID_present = pos_index >= crat_prefix

    crat_FID = np.array((pos_index-crat_prefix)/10**(pos_digits+1)).astype(int)
    crat_FID[~crat_FID_present] = -1
    
    pos_index[crat_FID_present] = np.mod(pos_index[crat_FID_present], 10**(pos_digits+1))
    
    # Retrieve temporary
    
    if(np.any(pos_index>=tmp_prefix*2)):
        raise ValueError(f"The {pos_digits+1}-th digit, where present, should always be 1.")
        
    tmp = pos_index >=tmp_prefix
    
    pos_index[tmp] = np.mod(pos_index[tmp], 10**pos_digits)
    if(len(pos_index.shape)==0):
        pos_index = int(pos_index)
        tmp = bool(tmp)
        crat_FID = int(crat_FID)
    
    return pos_index, tmp, crat_FID
    
def newPositions(raster_path, image_size, ylims = None, xlims = None, lim_crs = 'raster', spacing = 0.9, start = None, pos_prefix = None, priority = 1, other_columns = None):

    """
    Generate a tile position list from a given raster.
        
    Arguments:
    - raster_path (str): the path of the raster.
    - image_size (int or tuple): dimensions of the tile along the row and column axes. If integer, the tile is square.

    Keywords:
    - ylims (numeric or tuple): the y-axis limits of the area covered by the positions. If numeric, (-ylims, ylims) is adopted.
    - xlims (numeric or tuple): the x-axis limits of the area covered by the positions. If numeric, (-xlims, xlims) is adopted.
    - lim_crs: the CRS in which the `ylims` and `xlims` are expressed. If 'raster', the raster CRS is used. Otherwise, must be a pyproj.CRS object.
               In this case, the limits will be transformed to the raster CRS by transforming the points (xlim,0) and (0,ylim). 
    - spacing (float): the fraction of the tile size between the side of a tile and the beginning of the next one (see gridPositions()).
    - start (tuple): a (row, col) coordinate where a tile will have its corner. If not set, the tiles are initialized from a random location.
    - pos_prefix (int): if defined, this quantity is summed to the indices of the tiles. It should therefore have as many trailing zeros as necessary
                        to accomodate the original tile numbers (from 0 to the number of tiles - 1).
    - priority (int): the value to which the `priority` column should be set.
    - other_columns (dict): {column name: column value}, if other columns should be added to the tile list.
    
    Returns: 
    - new_positions (pd.DataFrame): a dataframe containing the tile list (index: pos_index; 
                                    columns: 'filename', 'priority', 'y', 'x', 'height', 'width' and others...).
    """

    raster = rioxarray.open_rasterio(raster_path).astype(np.uint8)
    
    raster_shape = raster.shape
    raster_crs = raster.rio.crs
    
    y_coords = raster.coords['y'].values
    x_coords = raster.coords['x'].values
    
    # Process xlims and ylims
    
    # If none are given, they correspond to the limits of the raster
    if(xlims is None):
        xlims = [0, raster_shape[2]]
        xlim_px = True
        
    if(ylims is None):
        ylims = [0, raster_shape[1]]
        ylim_px = True
    
    pixel_error = "If you specify only one number for xlims or ylims, it cannot be a pixel coordinate, so you need to specify `lim_crs`."
     
    if((isinstance(xlims, int) or isinstance(xlims,float))):
        if(lim_crs is None):
            raise ValueError(pixel_error)
        xlims = [-xlims, xlims]
        xlim_px = False
        
    if((isinstance(ylims, int) or isinstance(ylims,float))):
        if(lim_crs is None):
            raise ValueError(pixel_error)
        ylims = [-ylims, ylims]
        ylim_px = False
        
    if(lim_crs is None):
        xlim_px = True
        ylim_px = True
    elif(lim_crs=='raster'):
        pass
    
    else:
    
        # Transform limits provided in a `lim_crs` to the raster CRS.
    
        lim_to_raster = Transformer.from_crs(lim_crs, raster_crs, always_xy=True)
        
        # This transformation is not applicable to all CRS, but it works for a lonlat to equatorial cylindrical case.
        if(not(xlim_px)):
            xlims = [lim_to_raster.transform(xlims[0], 0)[0], lim_to_raster.transform(xlims[1], 0)[0]]
            
        if(not(ylim_px)):
            ylims = [lim_to_raster.transform(0, ylims[0])[1], lim_to_raster.transform(0, ylims[1])[1]]
        
    if(not(xlim_px)):
        argsort_x = np.argsort(x_coords)
        xlims = np.sort(argsort_x[np.searchsorted(np.sort(x_coords), xlims)])
        
    if(not(ylim_px)):
        argsort_y = np.argsort(y_coords)
        ylims = np.sort(argsort_y[np.searchsorted(np.sort(y_coords), ylims)])
        
    # Generate tile positions
    pos_y, pos_x = gridPositions(ylims, xlims, image_size = image_size, spacing = spacing, start = start)
    
    # Prefix
    if(pos_prefix is None):
        pos_prefix = 0
    else:
        closest_10_power = int(10**np.ceil(np.log10(len(pos_y))))
        if(pos_prefix % closest_10_power != 0):
            raise ValueError("The position prefix should have at least as many trailing zeros as the digits of the number of positions, so that the position indices are still recognizable.")
    
    # Define tile list dataframe
    
    new_positions = pd.DataFrame({'y': pos_y,
                                  'x': pos_x},
                                  index = np.arange(len(pos_y), dtype=int) + pos_prefix)
                                  
    try:
        image_height, image_width = image_size
    except:
        image_height = image_size
        image_width = image_size
                                 
    new_positions['height'] = image_height
    new_positions['width'] = image_width
    new_positions['filename'] = raster_path
    new_positions['priority'] = priority
    
    # Add other columns
    
    if(other_columns is not None):
        for other_column, other_value in other_columns.items():
            new_positions[other_column] = other_value
        
    return new_positions
    
def emptyPosMask(height = Planet_Basemap.BASEMAP_HEIGHT, 
                 width = Planet_Basemap.BASEMAP_WIDTH,
                 extent = Planet_Basemap.BASEMAP_EXTENT, 
                 crs = Planet_CRS.CYL_CRS, 
                 other_columns = {'num_1': int},
                 band_names = ['mask']):
                 
    """
    Generate an empty raster and an empty list of tile positions.
    
    Keywords:
    - height (int): the number of rows.
    - width (int): the number of columns.
    - extent (tuple): ((min(x),max(x)),(min(y),max(y))): the raster extent.
    - crs (pyproj.CRS): the raster CRS.
    - other_columns (dict): additional columns in the list of positions.
    - band_names (list): list of band names, the raster will have as many bands as its length.
    
    Returns: 
    - mask (xr.DataArray): the empty raster
    - positions (pd.DataFrame): the empty tile list.
    """
                 
    positions = pd.DataFrame({'y': np.zeros(0, dtype=int),
                              'x': np.zeros(0, dtype=int),
                              'height': np.zeros(0, dtype=int),
                              'width': np.zeros(0, dtype=int),
                              'filename': np.zeros(0, dtype=str),
                              'priority': np.zeros(0, dtype=int)},
                              index = np.zeros(0, dtype=int))
                              
    if(other_columns is not None):
        for other_column, other_type in other_columns.items():
            positions[other_column] = np.zeros(0, dtype=other_type)
            
    xextent = np.sort(np.array(extent[0]))
    xcoords = np.linspace(xextent[0], xextent[1], width+1)
    xcoords = (xcoords[0:-1] + xcoords[1:])/2
    yextent = np.sort(np.array(extent[1]))
    ycoords = np.linspace(yextent[0], yextent[1], height+1)
    ycoords = np.flip((ycoords[0:-1] + ycoords[1:])/2)
    
    mask_arr = np.zeros((len(band_names), height, width), dtype=np.uint8)
    mask = xr.DataArray(data = mask_arr, dims=('band','y','x'), coords = (band_names, ycoords, xcoords))
    mask.attrs['long_name'] = band_names
    mask.rio.write_crs(crs, inplace=True)
    
    return mask, positions
    
def fillCyl(raster, whole_raster = None, top_left_whole = None, resolution = Planet_Basemap.BASEMAP_RESOLUTION, flat = False):

    """
    Insert a smaller raster into a larger one, with the same CRS as the basemap. 
    
    Arguments:
    - raster (xr.DataArray): the smaller raster.
    
    Keywords:
    - whole raster (xr.DataArray): the larger raster. By default, a global raster set to 0 is generated.
    - top_left_whole (xr.DataArray): the top-left corner of `whole_raster`, to be specified in case the latter is not georeferenced.
                                     By default, the bounds of `whole_raster` will be used.
    - resolution (float): the resolution in the same unit of the Equidistant Cylindrical CRS (usually meters).
                          By default, imported from the basemap.
    - flat: set to True if a 2D output is desired. By default, False.
    
    Returns: 
    - whole_raster: the larger raster with `raster` inserted. 3D by default, a 2D np.array if `flat` is True.
    """

    # Generate the larger raster (an empty copy of the basemap if `whole_raster` isn't defined).

    if(whole_raster is None):
        whole_raster, _ = emptyPosMask()
        if(flat):
            whole_raster = whole_raster.values[0,:,:]
            
    # Top-left corner

    if(top_left_whole is None and not(flat)):
        x_left_whole, _, _, y_top_whole = whole_raster.rio.bounds()
    elif(top_left_whole is None and flat):
        raise ValueError("If `top_left` is None, `flat` must be False.")
    else:
        x_left_whole, y_top_whole = top_left_whole

    _, height, width = raster.shape
    x_left, _, _, y_top = raster.rio.bounds()
    
    # The location of the top-left corner of the small raster.

    x_pos = round((x_left-x_left_whole)/resolution)
    y_pos = round((y_top_whole-y_top)/resolution)
    
    # Set small raster into large one.

    if(flat):
        whole_raster[y_pos:y_pos+height,x_pos:x_pos+width] += raster[0,:,:].values
    else:
        whole_raster[:,y_pos:y_pos+height,x_pos:x_pos+width] += raster.values

    return whole_raster
    
def cleanupInvalid(pattern, valid_values, target_wildcard_index = 0):

    """
    Delete files whose names follow a given pattern but don't have the right values in the place marked by a wildcard.
    
    Arguments:
    - pattern (str): the filename pattern, should contain the filepath and one or more wilcards (*) where different values can appear.
    - valid_values (np.array): the values that the wildcard can assume.
        
    Keywords:
    - target_wildcard_index: if there is more than one wildcard in the pattern, specify which one should be considered.
    
    Returns: 
    - invalid_files: the list of invalid filenames (which have already been deleted).
    """

    target_wildcard_index += 1

    file_list = np.array(glob.glob(pattern)) # The files matching the pattern
    
    parts = pattern.split('*') # Split the pattern at the wildcards
    regex_parts = []
    wildcard_count = 0

    for i, part in enumerate(parts):
        regex_parts.append(re.escape(part))
        if i < len(parts) - 1:  # There's a wildcard after this part
            wildcard_count += 1
            if wildcard_count == target_wildcard_index:
                regex_parts.append('(.*)')  # Capturing group for the one we want
            else:
                regex_parts.append('(?:.*)')  # Non-capturing for the rest

    # Gather the values that the chosen wildcard assumes

    regex_pattern = ''.join(regex_parts)
    
    matches = []
    for filename in file_list:
        match = re.match(regex_pattern, filename)
        if match:
            matches.append(match.group(1))
        else:
            raise RuntimeError(f"The pattern {pattern} does not work for {filename}.")
            
    # Filter the valid values
            
    matches = np.array(matches).astype(valid_values.dtype)
    
    invalid_files = file_list[~np.isin(matches, valid_values)]
    
    for filename in invalid_files:
        os.remove(filename)
            
    return invalid_files
    
def denoiseMask(mask, feat_min_size=100):

    """
    Denoise a binary raster by removing clusters of values comprising less than `feat_min_size` pixels.
    
    Arguments:
    - mask (np.array): the 2D grid of values. 
        
    Keywords:
    - feat_min_size: the minimum number of pixels for a cluster to be preserved. Default: 100
    
    Returns: 
    - denoised_mask: the denoised mask.
    """


    if(feat_min_size is None or feat_min_size==0 or feat_min_size==1):
        return mask

    # Retrieve values other than 0 or 1

    nonbinary_indices = np.nonzero(~np.isin(mask, [0,1]))
    nonbinary_values = mask[nonbinary_indices]

    # Set any indices other than 0 and 1 to 0

    mask[nonbinary_indices] = 0
    
    # Find the clusters of ones
    
    labeled_array, num_features = sni.label(mask) # Each cluster is given a different numerical label.
    feat_labels, feat_counts = np.unique(labeled_array, return_counts=True) # Number of pixels `feat_counts` per cluster
    counts, count_num = np.unique(feat_counts, return_counts=True)

    # Remove the small clusters
    
    small_labels = feat_labels[feat_counts<feat_min_size]
    denoised_mask = np.copy(mask)
    denoised_mask[np.nonzero(np.isin(labeled_array, small_labels))] = 0

    # Find the clusters of zeros

    inverse_mask = 1-denoised_mask
    inverse_mask[nonbinary_indices]=0
    labeled_array, num_features = sni.label(inverse_mask)
    
    feat_labels, feat_counts = np.unique(labeled_array, return_counts=True)
    
    counts, count_num = np.unique(feat_counts, return_counts=True)
    
    # Remove the clusters of zeros
    
    small_labels = feat_labels[feat_counts<feat_min_size]
    denoised_mask[np.nonzero(np.isin(labeled_array, small_labels))] = 1

    # Restore values other than 0 or 1

    denoised_mask[nonbinary_indices]=nonbinary_values
    
    return denoised_mask
