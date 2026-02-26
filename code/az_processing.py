"""
Filename: az_processing.py
Author: Michele Lissoni
Date: 2026-02-13
"""

"""

Functions to manage the transformation and processing of the 
data in Azimuthal Equidistant projection.

"""

import numpy as np

import rioxarray
import xarray as xr
import rasterio
import cv2

import shapely
from shapely.geometry import Polygon

from pyproj import Transformer, CRS, Geod

import os
import gc
import time

import logging

# The properties of the planet for which the Azimuthal CRS is computed must be defined in the planet_constants.py script.
from planet_constants import Planet_Properties, Planet_CRS, Planet_Basemap, getAzCRS
RADIUS = Planet_Properties.RADIUS
BASEMAP_RESOLUTION = Planet_Basemap.BASEMAP_RESOLUTION

def rotateImage(image: np.ndarray, angle_deg: float) -> np.ndarray:

    """
    Rotate an image.
    
    Arguments:
    - image (np.array): a 3D NumPy array, where the first axis contains the image bands, 
                        the second to image's vertical axis and the third the image's horizontal axis.
    - angle_deg (float): the angle in degrees.
    
    Returns: 
    - rotated: the 3D NumPy array containing the rotated image.
    """

    if(len(image.shape)!=3):
        raise ValueError('Only 3D arrays are allowed. For 2D images, turn into 3D with only one band.')    

    band_num, h, w = image.shape
    center = (w / 2, h / 2)
    
    image = np.transpose(image, (1, 2, 0)) # cv2 works with images where the bands axis is the last.

    # Rotation matrix (positive = counter-clockwise, so negate angle for clockwise)
    M = cv2.getRotationMatrix2D(center, angle_deg, scale=1.0)

    # Apply affine warp
    rotated = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    if(band_num==1 and len(rotated.shape)==2):
        rotated = rotated[np.newaxis,:,:]
    else:
        rotated = np.transpose(rotated, (2,0,1))

    return rotated

def emptyAz(center_coords, nodata = 0, band_name = 1, az_resolution = BASEMAP_RESOLUTION, dtype=np.uint8):

    """
    Generate a global, 1-band raster set to the same value everywhere in an Azimuthal Equidistant projection.
    
    Arguments:
    - center_coords (tuple): (lon, lat), the longitude and latitude of the projection's center.
    
    Keywords:
    - nodata (numerical): the value to which the raster should be set. Default: 0
    - band_name (str|int): the name of the band.
    - az_resolution: the resolution in the same unit of the Azimuthal Equidistant CRS (usually meters).
                     By default, imported from the basemap.
    - dtype: the data type of the raster, by default UInt8 (Byte).
    
    Returns: 
    - dst_da: a georeferenced xr.DataArray containing the raster.
    """

    radius = RADIUS

    center_lat = center_coords[1]
    center_lon = center_coords[0]
    
    dst_crs = getAzCRS(center_lon, center_lat) # Get azimuthal CRS

    # Raster shape

    distance_from_center = radius*np.pi
    half_size = int(distance_from_center/az_resolution)
    width = half_size*2
    height = half_size*2
    distance_from_center = az_resolution*half_size
    
    # Raster array
    
    dst_arr = np.ones((1, height, width))*nodata
    dst_arr = dst_arr.astype(dtype)

    # Raster coordinates. 
    # TO DO: compute raster transform instead.

    y_coords = np.flip(np.sort(np.append(np.arange(half_size)*az_resolution+az_resolution/2, np.arange(half_size)*az_resolution*(-1)-az_resolution/2)))
    x_coords = np.sort(np.append(np.arange(half_size)*az_resolution+az_resolution/2, np.arange(half_size)*az_resolution*(-1)-az_resolution/2))
    
    # XArray with data and coordinates.
    
    dst_da = xr.DataArray(data=dst_arr, dims=('band','y','x'), coords = ([band_name], y_coords, x_coords))
    dst_da.rio.write_crs(dst_crs, inplace=True) # Write CRS

    return dst_da
    
def fillAz(raster, center_coords, whole_raster = None, top_left_whole = None, resolution = BASEMAP_RESOLUTION, flat = False):

    """
    Insert a smaller, Azimuthal Equidistant raster into a larger (global) one.
    
    Arguments:
    - raster (xr.DataArray): the smaller, azimuthal raster.
    - center_coords (tuple): (lon, lat), the longitude and latitude of the projection's center.
                             TO DO: derive this directly from the raster.
    
    Keywords:
    - whole raster (xr.DataArray): the larger azimuthal raster. By default, a global raster set to 0 is generated.
    - top_left_whole (xr.DataArray): the top-left corner of `whole_raster`, to be specified in case the latter is not georeferenced.
                                     By default, the bounds of `whole_raster` will be used.
    - resolution (float): the resolution in the same unit of the Azimuthal Equidistant CRS (usually meters).
                          By default, imported from the basemap.
    - flat: set to True if a 2D output is desired. By default, False.
    
    Returns: 
    - whole_raster: the larger raster with `raster` inserted. 3D by default, a 2D np.array if `flat` is True.
    """

    # Generate the larger raster

    if(whole_raster is None):
        whole_raster = emptyAz(center_coords) # Create global raster
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

def reprojectAz(src_da, center_coords, axis_coords = None, nodata = 0, az_resolution = None, num_threads = 1):

    """
    Reproject a raster to Azimuthal Equidistant projection.
    
    Arguments:
    - src_da (xr.DataArray): the input raster.
    - center_coords (tuple): (lon, lat), the longitude and latitude of the azimuthal projection's center.
    
    Keywords:
    - axis_coords (tuple): if a rotated (and non-georeferenced) raster is desired where the 
                           vertical axis run from the center to a given location, specify its (lon,lat) coordinates.
                           By default, the vertical axis runs in the north-south direction
                           (NOTE: this option was ultimately never used for the ejecta mapping).
    - nodata (numeric): the no data value. Default: 0.
    - az_resolution (float): the resolution in the azimuthal projection's unit (usually meters). If None,
                             the same resolution as the input raster is used.
    - num_threads (int): increase to enable multi-threading. More than 8 threads are advised for reasonably fast performance. 
                         Default: 1.
    
    Returns: 
    - dst_da (xr.DataArray): the output azimuthal raster.
    """

    radius = RADIUS

    # Process input raster

    if(len(src_da.dims)!=3):
        raise ValueError('Only 3D Xarray DataArrays are allowed. For 2D images, turn into 3D with only one band.')
    
    src_crs = src_da.rio.crs
    src_transform = src_da.rio.transform()
    src_arr = src_da.values

    bands_name = src_da.dims[0]
    bands_coords = src_da.coords[bands_name].values

    # Define resolution if not specified

    if(az_resolution is None):
        az_resolution = (abs(src_da.rio.resolution()[0])+abs(src_da.rio.resolution()[1]))/2

    # Get azimuthal CRS

    center_lat = center_coords[1]
    center_lon = center_coords[0]
    
    dst_crs = getAzCRS(center_lon, center_lat)
    
    # Get azimuthal raster size

    distance_from_center = radius*np.pi 
    half_size = int(distance_from_center/az_resolution) # The size is the planetary semi-circumference (radius * pi)
                                                        # divided by the resolution.
                                                        # TO DO: adapt for bodies with non-negligible flattening
    width = half_size*2
    height = half_size*2
    distance_from_center = az_resolution*half_size
    
    src_arr = src_da.values
    dst_arr = np.zeros((src_arr.shape[0], height, width), dtype=src_arr.dtype)

    dst_transform = rasterio.Affine(az_resolution, 0.0, distance_from_center*(-1.0), 0.0, az_resolution*(-1), distance_from_center)

    # Reprojection: uses rasterio.warp.reproject()

    dst_arr,_ = rasterio.warp.reproject(
        src_arr,
        dst_arr,
        src_transform = src_transform,
        src_crs = src_crs,
        src_nodata = nodata, 
        dst_transform = dst_transform,
        dst_crs = dst_crs,
        dst_nodata = nodata,
        resampling = rasterio.enums.Resampling.nearest,
        num_threads = num_threads
    )

    if(axis_coords is None):
    
        # Define georeferencing if no rotation is required.
        # TO DO: rely on geotransform instead.
    
        y_coords = np.flip(np.sort(np.append(np.arange(half_size)*az_resolution+az_resolution/2, np.arange(half_size)*az_resolution*(-1)-az_resolution/2)))
        x_coords = np.sort(np.append(np.arange(half_size)*az_resolution+az_resolution/2, np.arange(half_size)*az_resolution*(-1)-az_resolution/2))
        
        dst_da = xr.DataArray(data=dst_arr, dims=(bands_name,'y','x'), coords = (bands_coords, y_coords, x_coords))
        dst_da.rio.write_crs(dst_crs, inplace=True)
        
    else:

        # Rotate raster. It will not be georeferenced,
        # since a CRS for this case cannot be defined in WKT format.
        # TO DO: look if a geotransform can be defined for this case.

        axis_lat = axis_coords[0]
        axis_lon = axis_coords[1]
        
        sphere_geod = Geod(a=radius, es=0)
        az,_,_ = sphere_geod.inv(center_lon, center_lat, axis_lon, axis_lat)

        dst_arr = rotateImage(dst_arr, az)
        
        dst_da = xr.DataArray(data=dst_arr, dims=('band','y','x'))        

    return dst_da

def reprojectBasemap(src_da, nodata = 0, num_threads = 1):

    """
    Reproject a raster to the grid of the planetary basemap (usually Equidistant Cylindrical).
    
    Arguments:
    - src_da (xr.DataArray): the input raster.
    
    Keywords:
    - nodata (numeric): the no data value. Default: 0.
    - num_threads (int): increase to enable multi-threading. More than 8 threads are advised for reasonably fast performance. 
                         Default: 1.
    
    Returns: 
    - dst_da (xr.DataArray): the output raster.
    """

    # Read input raster

    if(len(src_da.dims)!=3):
        raise ValueError('Only 3D Xarray DataArrays are allowed. For 2D images, turn into 3D with only one band.')
    
    src_crs = src_da.rio.crs
    src_transform = src_da.rio.transform()
    src_arr = src_da.values  # The raster values

    bands_name = src_da.dims[0]
    bands_coords = src_da.coords[bands_name].values

    # Basemap data

    dst_resolution = Planet_Basemap.BASEMAP_RESOLUTION # Resolution
    dst_height = Planet_Basemap.BASEMAP_HEIGHT # Height (pixels)
    dst_width = Planet_Basemap.BASEMAP_WIDTH # Width (pixels)
    dst_extent = Planet_Basemap.BASEMAP_EXTENT # Bounds
    dst_crs = Planet_CRS.CYL_CRS # CRS
    
    dst_arr = np.zeros((src_arr.shape[0], dst_height, dst_width), dtype=np.uint8)
    
    # Compute coordinates (TO DO: switch to geotransform).
    
    xextent = np.sort(np.array(dst_extent[0]))
    xcoords = np.linspace(xextent[0], xextent[1], dst_width+1)
    xcoords = (xcoords[0:-1] + xcoords[1:])/2
    yextent = np.sort(np.array(dst_extent[1]))
    ycoords = np.linspace(yextent[0], yextent[1], dst_height+1)
    ycoords = np.flip((ycoords[0:-1] + ycoords[1:])/2)
    
    # Create georeferenced destination raster
    
    dst_da = xr.DataArray(data=dst_arr, dims=(bands_name,'y','x'), coords = (bands_coords, ycoords, xcoords))
    dst_da.rio.write_crs(dst_crs, inplace=True)

    dst_transform = dst_da.rio.transform()
    
    # Fill the raster with values
    
    with rasterio.Env(CPL_LOG="/dev/null", CPL_DEBUG=False):
    
        # Rasterio prints out a huge amount of warnings
        # at the GDAL level which slow down the performance.
        # This is the only way to get rid of them. 
    
        rasterio_logger = logging.getLogger('rasterio._err')
        original_level = rasterio_logger.level
        rasterio_logger.setLevel(logging.CRITICAL)
        
        # Reprojection
        
        dst_arr,_ = rasterio.warp.reproject(
            src_arr,
            dst_arr,
            src_transform = src_transform,
            src_crs = src_crs,
            src_nodata = nodata, 
            dst_transform = dst_transform,
            dst_crs = dst_crs,
            dst_nodata = nodata,
            resampling = rasterio.enums.Resampling.nearest,
            num_threads = num_threads
        )
        
        rasterio_logger.setLevel(original_level)

    dst_da[:,:,:] = dst_arr
    
    return dst_da

def coordsToPixelsAz(coords, src_crs, center_coords, az_resolution, half_size = None, axis_coords=None):

    """
    Convert geographic coordinates to the pixel coordinates of a global azimuthal raster.
    Pixel coordinates are integers at EDGES, not at centers
    
    Arguments:
    - coords: a (x,y) tuple or a 2D np.array, with the first/second row containing the x/y (or lon/lat) coordinates.
    - src_crs: the CRS in which the coordinates are expressed.
    - center_coords (tuple): (lon, lat), the longitude and latitude of the azimuthal projection's center.
    - az_resolution (float): the resolution in the azimuthal projection's unit (usually meters). If None,
                             the same resolution as the input raster is used.
                             TO DO: maybe supplying a geotransform would be better?
                             
    Keywords:                 
    - half_size (int): the half side of the square azimuthal raster. By default, a global raster 
                       with `az_resolution` is assumed. 
    - axis_coords (tuple): if the pixel positions are for a rotated raster, where the 
                           vertical axis run from the center to a given location, 
                           specify its (lon,lat) coordinates.
                           By default, the vertical axis runs in the north-south direction
                           (NOTE: this option was ultimately never used for the ejecta mapping).
        
    Returns: 
    - x_px, y_px : the pixel positions along the horizontal and vertical axis.
    """

    radius = RADIUS

    x_coords = coords[0]
    y_coords = coords[1]

    center_lat = center_coords[1]
    center_lon = center_coords[0]
    
    dst_crs = getAzCRS(center_lon, center_lat) # Get azimuthal CRS

    # Compute azimuthal raster shape if not given

    if(half_size is None):
        distance_from_center = radius*np.pi
        half_size = int(distance_from_center/az_resolution)

    # Transform the coordinates

    src_to_dst = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    
    x_coords_az, y_coords_az = src_to_dst.transform(x_coords, y_coords)

    x_dist_center = x_coords_az/az_resolution
    y_dist_center = y_coords_az/az_resolution

    # Handle raster rotation if necessary
    # TO DO: look at appropriate geotransform?

    if(axis_coords is not None):

        axis_lat = axis_coords[0]
        axis_lon = axis_coords[1]
        
        sphere_geod = Geod(a=radius, es=0)
        az,_,_ = sphere_geod.inv(center_lon, center_lat, axis_lon, axis_lat)

        rot_matrix = np.array([
            [np.cos(az), -np.sin(az)],
            [np.sin(az), np.cos(az)]])
        
        x_dist_center, y_dist_center = np.matmul(rot_matrix, np.array([x_dist_center, y_dist_center]))

    x_px = half_size + x_dist_center
    y_px = half_size - y_dist_center

    return x_px, y_px

def findGroupArgs(values, groups, arg_type):

    """
    Given a set of values assigned to various groups,
    get the minimum or maximum of each group.
    
    Arguments:
    - values (list or np.array): the numeric values.
    - groups (list or np.array): the groups they belong to (same length as values, 
                                 same label for all elements in the same group).
    - arg_type: 'min' or 'max'.
    
    Returns: 
    - args: the argument for `values` of the minima/maxima in each group.
    """

    order = np.lexsort((values, groups)) # Sort first by group, and then within each group

    _, arg_idx = np.unique(groups[order], return_index=True) # The minima arguments of each group

    if(arg_type=='max'):
        arg_idx = np.append(arg_idx[1:]-1, len(values)-1) # The maxima arguments are the minima
                                                          # of the following group minus one.
    elif(arg_type=='min'):
        pass
    else:
        raise ValueError("`arg_type` can be 'min' or 'max'.")

    args = order[arg_idx]

    return args

def pxLinePoints(coords):

    """
    Given a polyline that runs along pixel edges,
    get the coordinates of all the pixel corners it traverses.
    
    Arguments:
    - coords (np.array): a 2D array, the first/second row indicates the x/y (column/row) pixel coordinates,
    
    Returns: 
    - coord1, coord2: the x/y (column/row) pixel coordinates of the corners.
    """

    # Number of pixels between a vertex and the next (positive or negative depending on direction).
    # Since the line segments can be either horizontal or vertical,
    # when diffs1 != 0, diffs2 = 0 and viceversa.

    diffs1 = coords[0][1:]-coords[0][0:-1] 
    diffs2 = coords[1][1:]-coords[1][0:-1]
    
    repeats = np.abs(diffs1+diffs2) # Number of pixels of each segment, regardless of direction.
                                    # The sum of `repeats` is the number of pixel corners crossed.
    
    # Distance, in pixels, of each pixel corner from the previous vertex. 
    
    rep_range = np.arange(np.sum(repeats), dtype=int)
    rep_range = rep_range - np.repeat(rep_range[np.append(0, np.cumsum(repeats)[:-1])], repeats)
    
    # Set sign and direction of distance.
    
    signs1 = np.repeat(np.sign(diffs1), repeats)
    signs2 = np.repeat(np.sign(diffs2), repeats)
    
    rep_range1 = rep_range*signs1
    rep_range2 = rep_range*signs2
    
    # Pixel corners
    
    coord1 = np.repeat(coords[0][:-1], repeats) + rep_range1
    coord2 = np.repeat(coords[1][:-1], repeats) + rep_range2
    
    coord1 = np.append(coord1, coords[0][-1])
    coord2 = np.append(coord2, coords[1][-1])
    
    return coord1, coord2

def squareFromPxCoords(coords, side_limit = None, vectorize_limit=5000):

    """
    Given a polygon whose contour runs along pixel edges, find the largest square aligned with the pixel grid
    that can be inscribed within.
    
    Used to find the biggest possible square tiles of the azimuthal mosaic 
    within the reprojected boundaries of the cylindrical tiles.
    
    Arguments:
    - coords (np.array): a 2D array, the first/second row indicates the x/y (column/row) pixel coordinates of the polygon contour.
    
    Keywords:
    - side_limit (int): upper limit of the square size. Used to ensure the azimuthal tile is not bigger than the cylindrical tile.
    - vectorize_limit (int): if there are more pixel corners in the polyline than this,
                             the calculation is done with a for loop. Otherwise, it is vectorized through NumPy. 
    
    Returns: 
    - big_square_y_pos: the y (column) pixel coordinate of the upper-left corner of the square.
    - big_square_x_pos: the x (row) pixel coordinate of the upper-left corner of the square.
    - big_square_side: the side of the square, in pixel coordinates.
    - area_ratio: the ratio between the area enclosed by the original polyline and that of the square.
    """

    coords_x, coords_y = coords
    coords_x = np.array(coords_x).astype(int)
    coords_y = np.array(coords_y).astype(int)

    x_diffs = coords_x[1:] - coords_x[0:-1]
    y_diffs = coords_y[1:] - coords_y[0:-1]
    
    # Ensure that the contour runs along pixel edges, each segment is other horizontal or vertical.
    
    if(np.sum((x_diffs!=0) & (y_diffs!=0))>0):
        raise RuntimeError('There are some oblique lines in the contour.')
    
    # Create a buffer around the polygon
    
    poly = Polygon(np.hstack([coords_x[:,np.newaxis], coords_y[:,np.newaxis]]))
    poly_buffer = poly.buffer(0.1)
    shapely.prepare(poly_buffer)
    
    poly_area = poly.area
    
    coords_x, coords_y = pxLinePoints((coords_x, coords_y)) # Get all pixel corners crossed by contour

    # Find all the possible diagonals running between two pixel corners crossed by the contour.

    if(len(coords_x)<=vectorize_limit):
    
        pairwise_diffs_x = np.abs(coords_x[:, None] - coords_x[None, :])
        
        
        pairwise_diffs_y = np.abs(coords_y[:, None] - coords_y[None, :])
        
        diag_indices1, diag_indices2 = np.nonzero(pairwise_diffs_x-pairwise_diffs_y==0)
        indices_of_different_points = np.flatnonzero(diag_indices1-diag_indices2!=0)
        diag_indices1 = diag_indices1[indices_of_different_points]
        diag_indices2 = diag_indices2[indices_of_different_points]
    
        del pairwise_diffs_x
        del pairwise_diffs_y
        gc.collect()

    else:

        diag_indices1 = np.zeros(0, dtype=int)
        diag_indices2 = np.zeros(0, dtype=int)
        
        for i in range(0,len(coords_x)):

            coords_x_aux = coords_x - coords_x[i]
            coords_y_aux = coords_y - coords_y[i]

            diag_pos = np.flatnonzero(coords_x_aux - coords_y_aux==0)
            diag_pos = diag_pos[diag_pos!=i]
            diag_neg = np.flatnonzero(coords_x_aux + coords_y_aux==0)

            diag_all = np.append(diag_pos, diag_neg)
            diag_all = diag_all[diag_all!=i]

            diag_indices1 = np.append(diag_indices1, np.repeat(i, len(diag_all)))
            diag_indices2 = np.append(diag_indices2, diag_all)
            
    # Remove duplicate diagonals, running between the same points just in reverse order
    
    _, arg_uniq = np.unique(np.sort(np.hstack([diag_indices1[:,np.newaxis], diag_indices2[:,np.newaxis]]), axis=1), axis=0, return_index=True)
    diag_indices1 = diag_indices1[arg_uniq]
    diag_indices2 = diag_indices2[arg_uniq]
    
    square_sides = np.abs(coords_x[diag_indices2] - coords_x[diag_indices1]) # Sides corresponding to diagonals
    
    # Filter out squares with side beyond `side_limit`
    
    if(side_limit is not None and side_limit>0):
        diag_indices2 = diag_indices2[square_sides<=side_limit]
        diag_indices1 = diag_indices1[square_sides<=side_limit]
        square_sides = square_sides[square_sides<=side_limit]

    # Square corners, x (row) and y (col) coordinates

    squares_x = np.hstack([
        coords_x[diag_indices1, np.newaxis],
        coords_x[diag_indices1, np.newaxis],
        coords_x[diag_indices2, np.newaxis],
        coords_x[diag_indices2, np.newaxis]])
    squares_y = np.hstack([
        coords_y[diag_indices1, np.newaxis],
        coords_y[diag_indices2, np.newaxis],
        coords_y[diag_indices2, np.newaxis],
        coords_y[diag_indices1, np.newaxis]])
    
    square_sort_indices = np.flip(np.argsort(square_sides))
    square_sides = square_sides[square_sort_indices]
    squares_x = squares_x[square_sort_indices,:]
    squares_y = squares_y[square_sort_indices,:]

    # Filter out squares that are not contained in the polygon (necessary for concave polygons).

    contained=False
    big_square_index=0
    
    while(not(contained)):
        contained_arr = shapely.contains_xy(poly_buffer, squares_x[big_square_index,:], squares_y[big_square_index,:])
        contained = np.all(contained_arr)

        big_square_index+=1

    big_square_index-=1
    
    if(big_square_index==len(square_sides)):
        raise RuntimeError('No squares.')

    # Get coordinates, side and area ratio of largest square
    
    big_square_x = squares_x[big_square_index,:]
    big_square_y = squares_y[big_square_index,:]

    big_square_y_pos = np.amin(big_square_y)
    big_square_x_pos = np.amin(big_square_x)
    big_square_side = square_sides[big_square_index]

    area_ratio = big_square_side**2/poly_area
    
    return big_square_y_pos, big_square_x_pos, big_square_side, area_ratio

def squaresInSubimagesAz(raster, positions, center_coords, side_limit = 'original', axis_coords = None, nodata=0):

    """
    Given a cylindrical raster and cylindrical tile positions,
    reproject the raster and the tile positions to azimuthal projection 
    and find largest possible tiles inscribed within the reprojected area of the cylindrical tiles.
    
    Arguments:
    - raster (xr.DataArray): the cylindrical raster.
                             TO DO: could a simple geotransform, shape and CRS be sufficient?
    - positions (tuple): (y_positions, x_positions, heights, widths), contains arrays of equal length,
                         y_ and x_positions are the row and columns of the tile upper-left corners.
    - center_coords (tuple): (lon, lat), the longitude and latitude of the azimuthal projection's center.
    
    Keywords:
    - side_limit (int or 'original'): the upper limit of the square size. If 'original', the sqrt of the original tile area.
    - axis_coords (tuple): if the azimuthal raster should be rotated, so that the 
                           vertical axis runs from the center to a given location, 
                           specify its (lon,lat) coordinates.
                           By default, the vertical axis runs in the north-south direction
                           (NOTE: this option was ultimately never used for the ejecta mapping).
    - nodata (int): the nodata value.
    
    Returns: 
    - square_y_pos: the y (column) pixel coordinates of the upper-left corners of the azimuthal tiles.
    - square_x_pos: the x (row) pixel coordinates of the upper-left corners of the azimuthal tiles.
    - square_sides: the sides of the tiles.
    - square_ratios: the ratios between the area of the azimuthal tiles and that of the reprojected cylindrical tiles.
    """

    y_positions, x_positions, heights, widths = positions

    num_positions = len(y_positions)
    
    if(num_positions==0):
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros(0, dtype=float)
    
    # To find the area covered by each tile, set all the pixels within it to a unique number for each tile.
    # Once the resulting `positions_mask` is reprojected to azimuthal projection, the area covered by the pixels with that number
    # will correspond to that of the reprojected tile.
    
    positions_mask = raster[[0],:,:].copy().astype(np.uint16)
    positions_mask[0,:,:] = 0

    for i in range(0,num_positions):
        positions_mask[0, y_positions[i]:y_positions[i]+heights[i], x_positions[i]:x_positions[i]+widths[i]]=i+1

    az_positions_mask = reprojectAz(positions_mask, center_coords, axis_coords=axis_coords, nodata=nodata)
    az_positions_mask = az_positions_mask[0,:,:].values.astype(np.uint16)
    
    # A mask where all the azimuthal pixels covered by tiles are set to 1.
    
    rasterio_mask = az_positions_mask.copy()
    rasterio_mask[np.nonzero(rasterio_mask>0)]=1
    rasterio_mask = rasterio_mask.astype(bool)

    # Convert the areas covered by pixels of uniform value (i.e. by each given tile) to polygons

    pos_poly_list = np.array(list(rasterio.features.shapes(az_positions_mask, mask = rasterio_mask)))
    pos_values = np.array([pos_poly_element[1] for pos_poly_element in pos_poly_list], dtype=int) # Tile code of each polygon
    pos_polys = np.array([shapely.geometry.shape(pos_poly_element[0]) for pos_poly_element in pos_poly_list]) # Polygons in shapely format
    pos_areas = np.array([pos_poly.area for pos_poly in pos_polys]) # Areas of the polygons
    
    # Sometimes, during reprojection, a tile area is split into multiple polygons.
    # However, there is always one big polygons and smaller ones formed by just a few pixels.
    # Therefore, in this case, the largest one is used.
    
    max_area_args = findGroupArgs(pos_areas, pos_values, 'max')
    pos_polys = pos_polys[max_area_args]
    pos_poly_list = pos_poly_list[max_area_args]
    pos_areas = pos_areas[max_area_args]
    pos_values = pos_values[max_area_args]

    if(len(max_area_args)!=num_positions):
        raise RuntimeError('The polygons are '+str(len(max_area_args))+' while the positions are '+str(num_positions))
    
    # Iterate over the tiles, for each one find the largest possible azimuthal tile 
    # inscribed within the reprojected area of the cylindrical one.
    
    square_y_pos = np.zeros(num_positions, dtype=int)
    square_x_pos = np.zeros(num_positions, dtype=int)
    square_sides = np.zeros(num_positions, dtype=int)
    square_ratios = np.zeros(num_positions)

    for i in range(0,num_positions):
        poly_coords = pos_poly_list[i][0]['coordinates'][0]
        num_vertices = len(poly_coords)
        poly_coords = np.array(poly_coords).flatten().astype(int).reshape((num_vertices,2))

        coords_x = poly_coords[:,0]
        coords_y = poly_coords[:,1]
        
        if(side_limit == 'original'):
            side_limit = int(np.sqrt(heights[i]*widths[i]))

        # Find the largest square
        square_y_pos[i], square_x_pos[i], square_sides[i], square_ratios[i] = squareFromPxCoords((coords_x, coords_y), side_limit=side_limit)
        
    return square_y_pos, square_x_pos, square_sides, square_ratios

# First attempt at a function to find the azimuthal tiles, did not work with the most deformed tiles. 
'''
def posToPixelsAz(positions, raster, center_coords, az_resolution = None, axis_coords = None):

    y_positions, x_positions, heights, widths = positions

    try:
        len_positions = len(y_positions)
        y_positions = np.array(y_positions)
        x_positions = np.array(x_positions)
        heights = np.array(heights)
        widths = np.array(widths)

    except:

        len_positions = 1
        y_positions = np.array([y_positions])
        x_positions = np.array([x_positions])
        heights = np.array([heights])
        widths = np.array([widths])

    raster_crs = raster.rio.crs
    raster_height = len(raster.coords['y'].values)
    raster_width = len(raster.coords['x'].values)
    
    resolution_y = raster.rio.resolution()[1]
    resolution_x = raster.rio.resolution()[0]
    if(az_resolution is None):
        az_resolution = (abs(resolution_y)+abs(resolution_x))/2

    corners_x = np.hstack([x_positions[:,np.newaxis], (x_positions+widths)[:,np.newaxis], (x_positions+widths)[:,np.newaxis], x_positions[:,np.newaxis]]) 
    corners_y = np.hstack([y_positions[:,np.newaxis], y_positions[:,np.newaxis], (y_positions+heights)[:,np.newaxis], (y_positions+heights)[:,np.newaxis]]) 
    corners_x = corners_x.flatten() % raster_width
    corners_y = corners_y.flatten()

    
    corners_coords_x = raster.coords['x'].values[corners_x]-resolution_x/2
    corners_coords_y = raster.coords['y'].values[corners_y]-resolution_y/2

    corners_px_x, corners_px_y = coordsToPixelsAz((corners_coords_x, corners_coords_y), 
                                                        raster_crs, 
                                                        center_coords, 
                                                        az_resolution, 
                                                        axis_coords = axis_coords)

    corners_px_x = np.reshape(corners_px_x, (len_positions, 4))
    corners_px_y = np.reshape(corners_px_y, (len_positions, 4))
    corners_px_x = np.hstack([corners_px_x, corners_px_x[:,[0]]])
    corners_px_y = np.hstack([corners_px_y, corners_px_y[:,[0]]])

    return corners_px_x, corners_px_y
'''
