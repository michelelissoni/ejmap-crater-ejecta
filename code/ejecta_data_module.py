"""
Filename: ejecta_data_module.py
Author: Michele Lissoni
Date: 2026-02-23
"""

"""

Define the data module class that organizes the data fed to the Segmentation model.

"""

import os
import numpy as np
import pandas as pd

import geopandas as gpd
from shapely import Polygon

import rioxarray
from rasterio.enums import Resampling
from rasterio.features import geometry_mask as geometryMask

import copy
import warnings

from sklearn.model_selection import StratifiedShuffleSplit

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from lightning.pytorch import LightningDataModule

from az_processing import reprojectAz, coordsToPixelsAz, emptyAz

from planet_constants import Planet_Properties, Planet_Basemap, Planet_CRS
PLANETARY_RADIUS = Planet_Properties.RADIUS
BASEMAP_RESOLUTION = Planet_Basemap.BASEMAP_RESOLUTION
LATLON_CRS = Planet_CRS.LATLON_CRS

import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

def colorAugmenter(image):

    """
    
    Color augmentation of an image with random brightness, contrast and blur changes.
    
    Arguments:
    - image: a 3D Numpy array, the first axis indicates the bands. 
    
    """

    image = np.swapaxes(image, 0, 2) # The imgaug library uses the last axis as band axis.

    contrast_augmenter = iaa.contrast.SigmoidContrast(gain=(2,7), cutoff=(0.3,0.6)) # Contrast
    brightness_augmenter = iaa.color.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-20, 20)) # Brightness
    blur_augmenter = iaa.blur.GaussianBlur(sigma=(0,2)) # Blur

    augmenter = iaa.meta.SomeOf((0,2), [ # Applies at random some of these augmenters
        contrast_augmenter,
        brightness_augmenter,
        blur_augmenter
    ], random_order=True)
    
    image_aug = augmenter(image=image)
    
    image_aug = np.swapaxes(image_aug, 0, 2)
    
    return image_aug

def geomAugmenter(image, mask):

    """
    
    Random rotation and flipping augmentation of an image.
    
    Arguments:
    - image: a 3D Numpy array, the first axis indicates the bands. 
    
    """

    image = np.swapaxes(image, 0, 2)
    
    rotate_augmenter = iaa.geometric.Affine(rotate = [0, 90, 180, 270], mode='reflect') # Rotation only at right angles, so that there are no empty zones.
    rotate_augmenter._mode_segmentation_maps = 'reflect'
    flipud_augmenter = iaa.flip.Flipud(0.5) # Flipping, with a 50% probability
    
    augmenter = iaa.meta.Sequential([ # Both augmenters implemented
        flipud_augmenter,
        rotate_augmenter,
    ], random_order=False)

    image_aug, mask_aug = augmenter(image=image, 
                                    segmentation_maps=SegmentationMapsOnImage(mask, shape=image.shape) # Rotates the mask as well
                                    )
    mask_aug = mask_aug.get_arr() # Get rotated mask
    
    image_aug = np.swapaxes(image_aug, 0, 2)

    return image_aug, mask_aug

def sigmoid_dilated(x, factor = 1, y_dil = 0.95, x_dil = 10):

    """
    
    Dilated sigmoid function (to avoid divergence of weights).
    
    """

    if(y_dil <=0 or y_dil >=1 or x_dil<= 0):
        raise ValueError("`y_dil` must be between 0 and 1 and `x_dil` must be positive.")
    
    a = np.log(1/y_dil-1)/x_dil

    return np.reciprocal(1+np.exp(x*a))*factor

def getCraterBuffer(crater_FID, craters_gdf, data_root, buffer_max, buffer_min = 0, az_resolution = BASEMAP_RESOLUTION):

    """
    
    Extract mean band values in annulus of crater.
    
    Arguments:
    - crater_FID (int): the FID of the crater.
    - craters_gdf (int): the GeoDataFrame of the craters (fresh_craters.shp).
    - data_root (str): the data root of the Segmentation model
    - buffer_max (int): the upper boundary of the annulus, in crater radii wrt to the center.
    
    Keywords:
    - buffer_min (int): the lower boundary of the annulus (default: 0, the annulus becomes a circle)
    - az_resolution (float): the resolution of the mosaic.
    
    """

    if(buffer_min >= buffer_max or buffer_min < 0):
        raise ValueError('`buffer_max` should be greater than `buffer_min` and both should be positive.')
    
    if(buffer_min == 0):
        buffer_min = -10 # Ensures that the whole interior of the crater is included
    else:
        buffer_min = buffer_min - 1 # The annulus is actually a buffer around the crater rim, so we need to subtract 1 
                                    # from the radius wrt to the center to get the buffer size 
    buffer_max = buffer_max - 1
    
    crater_radius_m = craters_gdf.loc[crater_FID, 'radius']
    crater_poly = craters_gdf.geometry.loc[crater_FID]
    center_lon = craters_gdf.loc[crater_FID, 'center_lon']
    center_lat = craters_gdf.loc[crater_FID, 'center_lat']
    
    crater_poly_x = np.array(crater_poly.exterior.coords.xy[0])
    crater_poly_y = np.array(crater_poly.exterior.coords.xy[1])
    
    # Buffer image
    
    buffer_image = rioxarray.open_rasterio(os.path.join(data_root, 'buffers', 'az_crater_'+str(crater_FID)+'.tif'), masked=True).astype(np.uint8)
    buffer_image_shape = buffer_image.shape
    
    if(buffer_image_shape[1]!=buffer_image_shape[2] or buffer_image_shape[1] % 2 != 0):
        raise RuntimeError(f"The buffer images should be square and with even pixels, but for crater {crater_FID} the image is {buffer_image_shape[1]}x{buffer_image_shape[2]}.")
    
    half_size = int(buffer_image_shape[1]/2)
    
    # Convert the crater rim to azimuthal pixel coordinates
    
    crater_poly_px_x, crater_poly_px_y = coordsToPixelsAz((crater_poly_x, crater_poly_y), 
                                                                LATLON_CRS, 
                                                                (center_lon, center_lat),
                                                                az_resolution,
                                                                half_size = half_size,
                                                                axis_coords=None)
    
    crater_poly_px = Polygon(np.hstack([crater_poly_px_x[:,np.newaxis],crater_poly_px_y[:,np.newaxis]]))
    crater_radius_px = crater_radius_m/az_resolution
    
    # Create annulus by subtracting the inner polygon from the outer one.
    
    crater_buffer_px_max = crater_poly_px.buffer(buffer_max*crater_radius_px) 
    crater_buffer_px_min = crater_poly_px.buffer(buffer_min*crater_radius_px)
    crater_buffer_px = crater_buffer_px_max.difference(crater_buffer_px_min)
    
    # Extract pixel values
    
    crater_buffer_mask = geometryMask([crater_buffer_px], buffer_image[0,:,:].shape, transform=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0), all_touched=False, invert=True)
    
    buffer_value = np.mean(buffer_image.values[:,crater_buffer_mask], axis=1)

    return buffer_value 
    

class EjectaAzDataset(Dataset):

    """
    
    A Pytorch Dataset child class, handles the labeled data for the training, validation and testing of the Segmentation model.
    
    """

    def __init__(
        self,
        data_root,
        position_df,
        distance_mode = 'absolute',
        img_augment = False,
        log_dist = True,
        img_norm = ([0,0,0], [255.,255.,255.]),
        dist_norm = (0,1),
        **kwargs
    ):
      
        """
        
        Constructor
        
        Arguments:
        - data_root (str): the path of the data root folder.
        - position_df (pd.DataFrame): the list of all tile positions
        
        Keywords:
        - distance_mode: 'absolute' (use the distance from the crater center in meters), 'ratio' (use the ratio between the distance and the crater radius),
                         'rank' (use the pixel ranking from least distant to most distant)
        - img_augment (bool): implement on-the-fly image augmentation
        - log_dist (bool): use the log10 of the distance band
        - img_norm (tuple): normalization offset and denominator for the image bands, default ([0,0,0], [255.,255.,255.])
        - dist_norm (tuple): normalization offset and denominator for the distance band, default (0,1)
        - **kwargs: keyword arguments for the parent class
        
        """
        
        super().__init__(**kwargs)
        
        self.data_root = data_root

        self.img_norm = (np.array(img_norm[0]), np.array(img_norm[1]))
        self.dist_norm = dist_norm
            
        rng = np.random.default_rng()
        random_args = np.arange(len(position_df), dtype=int)
        rng.shuffle(random_args)
        
        position_df = position_df.iloc[random_args,:] # Tiles in random order

        # Pull the data out of the dataframe and arguments save it as attributes

        pos_index_list = position_df['pos_index'].values
        crat_FID_list = position_df['crat_FID'].values
        nums_1 = position_df['num_1'].values
        h_nums_1 = position_df['h_num_1'].values
        crater_radii = position_df['radius'].values
        buffer_values = position_df.loc[:,['0','1','2']].values
        split_classes = position_df['split_class'].values
                
        self.pos_index_list = pos_index_list # Position indices
        self.crat_FID_list = crat_FID_list # Crater FIDs
        self.crater_radii = crater_radii # Crater radii
        self.buffer_values = buffer_values # Annulus values
        self.nums_1 = nums_1 # Number of ejecta pixels in the tile belonging to the tile's crater
        self.h_nums_1 = h_nums_1 # Number of ejecta pixels in the tile belonging to any crater.
        self.split_classes = split_classes # Split classes for stratified splits
        
        self.ratio_dist = distance_mode=='ratio'
        self.absolute_dist = distance_mode=='absolute'
        self.rank_dist = distance_mode=='rank'
        self.log_dist = log_dist if distance_mode!= 'rank' else False
        
        self.img_augment = img_augment
        
    def __len__(self):
        return len(self.pos_index_list)

    def weights(self, num_1_factor = 10, num_1_point95 = 10):
    
        """
        
        Compute the weights that privilege tiles containing ejecta from different craters (not used in final model).
        
        Keywords:
        - num_1_factor: a multiplicative factor for the tiles containing ejecta, controls the value to which the weight converges 
                       (has no effect if only tiles containing ejecta are used for training, as we have done).
        - num_1_point95: dilation of the sigmoid function, controls how quickly the asymptote is reached.
        
        """

        weights = np.ones(len(self))

        use_num_1 = num_1_factor is not None

        if(use_num_1):
            num_h_num_nonzero = np.flatnonzero((self.h_nums_1 > 0) & (self.nums_1 > 0)) # Tiles with ejecta from their crater and all other craters
            num_zero_h_num_nonzero = np.flatnonzero((self.h_nums_1 > 0) & (self.nums_1 == 0)) # Tiles with ejecta only from other craters
            num_h_num_zero = np.flatnonzero((self.h_nums_1 == 0) & (self.nums_1 == 0)) # Tiles with no ejecta (won't actually appear in the training data)

            weights[num_zero_h_num_nonzero] = num_1_factor # Asymptote value for tiles with ejecta from other craters
            weights[num_h_num_nonzero] = sigmoid_dilated(self.h_nums_1[num_h_num_nonzero]/self.nums_1[num_h_num_nonzero], factor = num_1_factor, y_dil = 0.95, x_dil = num_1_point95) # Larger weights the less ejecta from the tile's crater appear wrt to the ejecta from other craters.
            weights[num_h_num_zero] = sigmoid_dilated(1, factor = 1, y_dil = 0.95, x_dil = num_1_point95) # Weight for empty tiles

        weights = weights/np.sum(weights) # Normalization

        return weights

    def train_val_split(self, val_frac):
    
        """
        
        Split dataset into Train and Validation. A fraction of samples equal to `val_frac` is chosen for 
        validation using a random stratified split on the split classes.
        
        """

        # Create copies of the EjectaAzDataset where the training and validation data will be stored

        val_dataset = copy.deepcopy(self)
        train_dataset = copy.deepcopy(self)
        
        val_size = int(val_frac*len(self)) # Number of samples in the validation data
        
        split_classes = self.split_classes.copy()
        split_uniq, split_index, split_counts = np.unique(split_classes, return_index=True, return_counts=True)
        
        one_indices = split_index[split_counts==1] # If there is only one sample of a given split class, it will go into the training data
        morethanone_indices = np.delete(np.arange(len(split_classes), dtype=int), one_indices)
        split_classes = np.delete(split_classes, one_indices)

        split_uniq = np.unique(split_classes)
        if(val_size<len(split_uniq)):
    
            warnings.warn(f"`val_size` is {val_size}, the number of split classes {len(split_uniq)}. Increasing `val_size` to match.")
            val_size = len(split_uniq)


        # Split the data

        splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size)
        train_indices, val_indices = next(splitter.split(np.zeros(len(split_classes)), split_classes))
        
        train_indices = np.append(morethanone_indices[train_indices], one_indices)
        val_indices = morethanone_indices[val_indices]

        # Fill the two new Datasets

        train_dataset.pos_index_list = self.pos_index_list[train_indices]
        train_dataset.crat_FID_list = self.crat_FID_list[train_indices]
        train_dataset.crater_radii = self.crater_radii[train_indices]
        train_dataset.buffer_values = self.buffer_values[train_indices]
        train_dataset.nums_1 = self.nums_1[train_indices]
        train_dataset.h_nums_1 = self.h_nums_1[train_indices]
        train_dataset.split_classes = self.split_classes[train_indices]

        val_dataset.pos_index_list = self.pos_index_list[val_indices]
        val_dataset.crat_FID_list = self.crat_FID_list[val_indices]
        val_dataset.crater_radii = self.crater_radii[val_indices]
        val_dataset.buffer_values = self.buffer_values[val_indices]
        val_dataset.nums_1 = self.nums_1[val_indices]
        val_dataset.h_nums_1 = self.h_nums_1[val_indices]
        val_dataset.split_classes = self.split_classes[val_indices]

        return train_dataset, val_dataset

    def __getitem__(self, index):
    
        """
        
        This method controls how a data sample (the tile image, distance channel and manual mask) is retrieved from memory.
        The index is the data sample index.
        
        """
    
        pos_index = self.pos_index_list[index] # Tile position index
        crat_FID = self.crat_FID_list[index] # Tile crater
        buffer_value = self.buffer_values[index,:] # Tile annulus

        filesuffix = str(pos_index)+'_crater_'+str(crat_FID)+'.npy'
        
        image_path = os.path.join(self.data_root, 'images', 'image_'+filesuffix)
        distance_path = os.path.join(self.data_root, 'distances', 'distance_'+filesuffix)
        mask_path = os.path.join(self.data_root, 'masks', 'mask_'+filesuffix)

        mask = np.load(mask_path) # Get mask
        image = np.load(image_path) # Get image
        
        if(self.img_augment):
            image = colorAugmenter(image) # On-the-fly color augmentation
            
        image = (image - self.img_norm[0][:,np.newaxis,np.newaxis])/self.img_norm[1][:,np.newaxis,np.newaxis] # Image normalization
        
        distance_grid = np.load(distance_path)*1.0 # Distance channel
        midinput = (buffer_value - self.img_norm[0])/self.img_norm[1]
        
        crater_radius = self.crater_radii[index]
        
        # Process distance channel
        
        if(self.absolute_dist): # Absolute distance
            distance_in_midinput = np.log10(crater_radius) if self.log_dist else crater_radius # Radius value in midinput, only if distance is absolute
            midinput = np.append(midinput, (distance_in_midinput - self.dist_norm[0])/self.dist_norm[1])
            
        elif(self.rank_dist): # Ranked distance
            dist_shape = distance_grid.shape
            distance_grid = np.reshape(np.argsort(np.argsort(distance_grid, axis=None)), dist_shape)*1.0/(dist_shape[0]*dist_shape[1])
            
        else: # Ratio distance
            distance_grid = distance_grid/crater_radius            
           
        if(self.log_dist): # Distance log
            distance_grid = np.log10(distance_grid)
            
        distance_grid = (distance_grid[np.newaxis,:,:] - self.dist_norm[0])/self.dist_norm[1] # Distance normalization
        
        image = np.concatenate((image, distance_grid), axis=0) # Adding distance channel to image
        
        if(self.img_augment):
            image, mask = geomAugmenter(image, mask) # On-the-fly rotation/flipping augmentation
        
        # Converting the data to Pytorch Tensors, which can be copied from the CPU to the GPU
        
        mask_dev = torch.tensor(mask.copy(), dtype=torch.float32)
        image_dev = torch.tensor(image.copy(), dtype=torch.float32)
        midinput_dev = torch.tensor(midinput, dtype=torch.float32)
        
        sample = {
            "image": {'image': image_dev,
                      'midinput': midinput_dev},
            "mask": mask_dev,
            "filename": torch.tensor([pos_index, crat_FID])
            } # These keys are read by the Terratorch Segmentation Task and the SMP custom model.
              # "filename" in this case can be used to store any metadata
        
        return sample

class EjectaAzMosaicDataset(Dataset):

    """
    
    A Pytorch Dataset child class, handles the unlabeled data for the predictions of the Segmentation model.
    
    """

    def __init__(
        self,
        data_root,
        crater_data,
        eval_position_df,
        distance_mode = 'absolute',
        log_dist = True,
        img_norm = ([0,0,0], [255.,255.,255.]),
        dist_norm = (0,1),
        **kwargs
    ): 
    
        """
        
        Constructor
        
        Arguments:
        - data_root (str): the path of the data root folder.
        - crater_data (tuple): (crater_FID, crater annulus values, crater radius)
        - eval_position_df (pd.DataFrame): the list of tiles for prediction.
        
        Keywords:
        - distance_mode: 'absolute' (use the distance from the crater center in meters), 'ratio' (use the ratio between the distance and the crater radius),
                         'rank' (use the pixel ranking from least distant to most distant)
        - log_dist (bool): use the log10 of the distance band
        - img_norm (tuple): normalization offset and denominator for the image bands, default ([0,0,0], [255.,255.,255.])
        - dist_norm (tuple): normalization offset and denominator for the distance band, default (0,1)
        - **kwargs: keyword arguments for the parent class
        
        """

        super().__init__(**kwargs)
        
        crater_FID = crater_data[0]
        
        self.crater_FID = crater_FID
        self.buffer_value = crater_data[1].astype(float)
        self.crater_radius_m = crater_data[2]
        center_lon = crater_data[3]
        center_lat = crater_data[4]
                
        self.data_root = data_root
        self.img_norm = (np.array(img_norm[0]), np.array(img_norm[1]))
        self.dist_norm = dist_norm
        
        # Tile position data
        
        self.eval_pos_index_list = eval_position_df.index.values
        self.eval_pos_y = eval_position_df['y'].values
        self.eval_pos_x = eval_position_df['x'].values
        self.eval_height = eval_position_df['height'].values
        self.eval_width = eval_position_df['width'].values
                
        empty_mask = emptyAz((center_lon, center_lat))
        
        self.raster_half_size = int(empty_mask.shape[1]/2)
        self.empty_mask = empty_mask # The tile predictions will be written to this global mask
        self.az_resolution = BASEMAP_RESOLUTION  
        
        self.ratio_dist = distance_mode=='ratio'
        self.absolute_dist = distance_mode=='absolute'
        self.rank_dist = distance_mode=='rank'
        
        self.log_dist = log_dist if distance_mode!='rank' else False      

    def __len__(self):
        return len(self.eval_pos_y)

    def __getitem__(self, index):
    
        """
        
        This method controls how a data sample (the tile image, distance channel and manual mask) is retrieved from memory.
        The index is the data sample index.
        
        """
    
        pos_index = self.eval_pos_index_list[index] # Position index
        
        # Tile coordinates
        
        pos_y = self.eval_pos_y[index]
        pos_x = self.eval_pos_x[index]
        height = self.eval_height[index]
        width = self.eval_width[index]
        
        image_path = os.path.join(self.data_root, 'eval_images','eval_image_' + str(pos_index) + '_crater_'+str(self.crater_FID)+'.npy')
        image = np.load(image_path) # Get image
        
        image = torch.tensor((image - self.img_norm[0][:,np.newaxis,np.newaxis])/self.img_norm[1][:,np.newaxis,np.newaxis], dtype=torch.float32) # Normalization

        # Create and process distance channel

        xgrid, ygrid = np.meshgrid(np.abs(np.arange(pos_x,pos_x+width)+0.5-self.raster_half_size),
                                   np.abs(np.arange(pos_y,pos_y+height)+0.5-self.raster_half_size))
                                   
        distance_grid = np.sqrt(np.square(xgrid)+np.square(ygrid))*self.az_resolution
        buffer_value = (self.buffer_value-self.img_norm[0])/self.img_norm[1]
        
        if(self.absolute_dist): # Absolute distance
            distance_in_buffer = np.log10(self.crater_radius_m) if self.log_dist else self.crater_radius_m # Crater radius in midinput
            buffer_value = np.append(buffer_value, (distance_in_buffer - self.dist_norm[0])/self.dist_norm[1])
            
        elif(self.rank_dist): # Rank distance
            dist_shape = distance_grid.shape
            distance_grid = np.reshape(np.argsort(np.argsort(distance_grid, axis=None)), dist_shape)*1.0/(dist_shape[0]*dist_shape[1])
            
        elif(self.ratio_dist): # Ratio distance
            distance_grid = distance_grid/self.crater_radius_m
                   
        if(self.log_dist): # Log distance
            distance_grid = np.log10(distance_grid)
            
        # Copy image and distance to Tensor and concatenate them
            
        distance_grid = torch.tensor((distance_grid[np.newaxis,:,:] - self.dist_norm[0])/self.dist_norm[1], dtype=torch.float32)
        image = torch.cat((image, distance_grid), dim=0)
        
        buffer_value = torch.tensor(buffer_value, dtype=torch.float32) # Midinput to tensors
        
        sample = {
            "image": {'image': image,
                      'midinput': buffer_value},
            "filename": torch.tensor([pos_y, pos_x, height, width, self.crater_FID])
        } # These keys are read by the Terratorch Segmentation Task and the SMP custom model.
          # "filename" in this case can be used to store any metadata
        
        return sample

    def get_empty_mask(self):

        return self.empty_mask.copy()
        

class EjectaAzDataModule(LightningDataModule):

    """
    
    A LightningDataModule child class, constructs the Datasets and Dataloaders to feed data to the model.
    
    """

    def __init__(
        self,
        data_root,
        predict = False,
        only_train = True,
        test_df = None,
        train_df = None,
        buffer_bounds = (1,2),
        img_augment = False,
        distance_mode = 'absolute',
        log_dist = True,
        num_1_factor = 10,
        num_1_point95 = 10,
        norm_from_train = False,
        img_norm = ([0,0,0], [255.,255.,255.]),
        dist_norm = (0,1),
        buffer_resolution = BASEMAP_RESOLUTION,
        val_frac = 0.1,
        batch_size=4,
        num_workers=0
    ):
    
        """
        
        Constructor
        
        Arguments:
        - data_root (str): the path of the segmentation data root folder.

        Keywords:
        - predict (bool): if True, does not require training and testing tile lists and does not compute normalization parameters
        - only_train (bool): retrieve only data for TVT craters
        - test_df (pd.DataFrame): test set tiles. Not required if `predict` is True.
        - train_df (pd.DataFrame): TV set tiles. Not required if `predict` is True.
        - buffer_bounds ((float, float)): the bounds of the annulus.
        - img_augment (bool): perform on-the-fly image augmentation. 
        - distance_mode: 'absolute' (use the distance from the crater center in meters), 'ratio' (use the ratio between the distance and the crater radius),
                         'rank' (use the pixel ranking from least distant to most distant)
        - log_dist (bool): use the log10 of the distance band
        - num_1_factor, num_1_point95: parameters for the Train weights (see EjectaAzDataset.weights()). If num_1_factor is None, weights are not used.
        - norm_from_train (bool): compute normalization parameters from TV data.
        - img_norm (tuple): normalization offset and denominator for the image bands, default ([0,0,0], [255.,255.,255.])
        - dist_norm (tuple): normalization offset and denominator for the distance band, default (0,1)
        - buffer_resolution (float): resolution of the buffer image
        - val_frac (float): fraction of TV data that goes into Validation
        - batch_size (int): the batch size
        - num_workers (int): the number of subprocesses used by the dataloaders
                
        """

        super().__init__()

        self.data_root = data_root
        
        if(not(predict) and (test_df is None or train_df is None)):
            raise ValueError("`test_df` and `train_df` are mandatory if `predict` is False.")
            
        fresh_craters_gdf = gpd.read_file(os.path.join(data_root, 'fresh_craters.shp')).set_index('crat_FID')
        if(only_train or not(predict)):
            fresh_craters_gdf = fresh_craters_gdf.loc[fresh_craters_gdf['train']==1,:]
        
        if(predict):
            all_crater_FIDs = fresh_craters_gdf.index.values
        else:
            all_crater_FIDs = np.unique(np.append(train_df['crat_FID'], test_df['crat_FID']))
            fresh_craters_gdf = fresh_craters_gdf.loc[all_crater_FIDs,:]
        
        I=0
        buffer_values = np.zeros((0,3))
        
        # Get annulus values and store them in `fresh_craters_gdf`, they will be later passed to the Dataset
        
        for crat_FID in all_crater_FIDs:

            crater_radius = fresh_craters_gdf.loc[crat_FID, 'radius']
        
            buffer_value = getCraterBuffer(crat_FID, 
                                           fresh_craters_gdf, 
                                           data_root,
                                           buffer_max = buffer_bounds[1], 
                                           buffer_min = buffer_bounds[0],
                                           az_resolution = buffer_resolution)
                                           
            buffer_values = np.vstack([buffer_values, buffer_value[np.newaxis,:]]) 
             
        fresh_craters_gdf.loc[all_crater_FIDs, ['0','1','2']] = buffer_values
        
        # Compute normalizatiion parameters from TV data

        if(not(predict) and norm_from_train):
                
            for i in range(0,len(train_df)): # Retrieve all the images and distance channels from memory, flatten them and stack them
            
                pos_index = train_df['pos_index'].iloc[i]
                crat_FID = train_df['crat_FID'].iloc[i]
                
                image = np.load(os.path.join(data_root,'images','image_'+str(pos_index)+'_crater_'+str(crat_FID)+'.npy'))

                if(i==0):
                
                    if(image.shape[1]!=image.shape[2]):
                        raise RuntimeError(f'Images should be square, but instead the first one is of shape {image.shape}.')
                        
                    subimage_size_sq = image.shape[1]*image.shape[2]
                    image_bands = np.zeros((image.shape[0],len(train_df)*subimage_size_sq), dtype=np.uint8)
                    distances = np.zeros(len(train_df)*subimage_size_sq)
                                
                image = image.reshape(image.shape[0], subimage_size_sq)
                distance = np.load(os.path.join(data_root,'distances','distance_'+str(pos_index)+'_crater_'+str(crat_FID)+'.npy')).flatten()*1.0
                
                #if(distance_mode=='absolute'):
                    #crater_radius = fresh_craters_gdf.loc[crat_FID, 'radius']
                    #distance = distance*crater_radius
                if(distance_mode=='ratio'):
                    crater_radius = fresh_craters_gdf.loc[crat_FID, 'radius']
                    distance = distance/crater_radius
                
                if(log_dist):
                    distance = np.log10(distance)
    
                image_bands[:,i*subimage_size_sq:(i+1)*subimage_size_sq] = image
                distances[i*subimage_size_sq:(i+1)*subimage_size_sq] = distance
                
            # Find mean and standard deviation
                
            mean_bands = np.mean(image_bands, axis=1)
            std_bands = np.std(image_bands, axis=1, ddof=1)
            mean_distance = np.mean(distances)
            std_distance = np.std(distances, ddof=1)

            img_norm = (mean_bands, std_bands)
            dist_norm = (mean_distance, std_distance)

            self.img_norm = img_norm
            self.dist_norm = dist_norm if distance_mode!='rank' else (0.5, 0.5)

        else:

            self.img_norm = img_norm 
            self.dist_norm = dist_norm if distance_mode!='rank' else (0.0,1.0)
            
        # Store data in attributes
        
        self.train_df = train_df
        self.test_df = test_df
        self.eval_df = None
        
        self.craters_df = fresh_craters_gdf.drop('geometry', axis=1)
        
        self.val_frac = val_frac
        
        self.img_augment = img_augment

        self.distance_mode = distance_mode
        self.log_dist = log_dist
        self.num_1_factor = num_1_factor
        self.num_1_point95 = num_1_point95
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.predict_crater_FID = None
        self.setup_predict_crater_FID = None
        self.predict_crater_data = None
        
        self.empty_mask = None
        
    def get_norms(self):
    
        """
        Get normalization data
        """
    
        norm_means = np.append(np.array(self.img_norm[0]), self.dist_norm[0])
        norm_stds = np.append(np.array(self.img_norm[1]), self.dist_norm[1])
        
        norm_arr = np.vstack([norm_means[np.newaxis,:], norm_stds[np.newaxis,:]])
    
        return norm_arr

    def predict_crater(self, crater_FID, eval_df):
    
        """
        Set the crater whose ejecta map will be predicted by a trained neural network.
        """

        if(crater_FID not in self.craters_df.index.values):
            raise ValueError(f"{crater_FID} not among available craters.")

        self.predict_crater_FID = crater_FID
        self.eval_df = eval_df
        
    def setup(self, stage):
    
        """
        This function is called by a Pytorch Lightning Trainer 
        to prepare a 'fit', 'test' or 'predict' operation.
        
        It prepares the relevant Datasets.
        """

        if(stage == 'fit'):
        
            train_df = self.train_df.copy()
            train_crat_FIDs = train_df['crat_FID'].values
            train_df['radius'] = self.craters_df.loc[train_crat_FIDs, 'radius'].values
            train_df['0'] = self.craters_df.loc[train_crat_FIDs, '0'].values
            train_df['1'] = self.craters_df.loc[train_crat_FIDs, '1'].values
            train_df['2'] = self.craters_df.loc[train_crat_FIDs, '2'].values

            # TV data

            train_dataset = EjectaAzDataset(self.data_root,
                                            train_df,
                                            distance_mode = self.distance_mode,
                                            img_augment = self.img_augment,
                                            log_dist = self.log_dist,                                            
                                            img_norm = self.img_norm,
                                            dist_norm = self.dist_norm)

            # Train and Val datasets

            self.train_dataset, self.val_dataset = train_dataset.train_val_split(self.val_frac)

            # Train weights

            weights = self.train_dataset.weights(num_1_factor = self.num_1_factor,
                                                 num_1_point95 = self.num_1_point95)
            
            self.train_sampler = WeightedRandomSampler(weights, 
                                                       num_samples = len(self.train_dataset), 
                                                       replacement=True)

        elif(stage == 'test'):

            test_df = self.test_df.copy()
            
            if(len(test_df)==0):
                self.test_dataset = None
            else:            
                test_crat_FIDs = test_df['crat_FID'].values
                test_df['radius'] = self.craters_df.loc[test_crat_FIDs, 'radius'].values
                test_df['0'] = self.craters_df.loc[test_crat_FIDs, '0'].values
                test_df['1'] = self.craters_df.loc[test_crat_FIDs, '1'].values
                test_df['2'] = self.craters_df.loc[test_crat_FIDs, '2'].values
                
                # Test dataset

                self.test_dataset = EjectaAzDataset(self.data_root,
                                                    test_df,
                                                    distance_mode = self.distance_mode,
                                                    img_augment = False,
                                                    log_dist = self.log_dist,                                            
                                                    img_norm = self.img_norm,
                                                    dist_norm = self.dist_norm)
        elif(stage == 'predict'):

            if(self.predict_crater_FID is None):
                raise ValueError("Run EjectaAzDataModule.predict_crater(PREDICT_CRATER_FID) first, with PREDICT_CRATER_FID the FID of the crater you want to predict.")

            crater_data = (self.predict_crater_FID,
                           self.craters_df.loc[self.predict_crater_FID, ['0','1','2']].values,
                           self.craters_df.loc[self.predict_crater_FID, 'radius'],
                           self.craters_df.loc[self.predict_crater_FID, 'center_lon'],
                           self.craters_df.loc[self.predict_crater_FID, 'center_lat'])

            # Dataset for unlabeled data to predict

            self.predict_dataset = EjectaAzMosaicDataset(self.data_root,
                                                         crater_data,
                                                         self.eval_df,
                                                         distance_mode = self.distance_mode,
                                                         log_dist = self.log_dist,
                                                         img_norm = self.img_norm,
                                                         dist_norm = self.dist_norm)   

            self.setup_predict_crater_FID = self.predict_crater_FID
            self.empty_mask = self.predict_dataset.get_empty_mask() # Get the empty mask that will be filled afterwards
            
        else:
            raise ValueError(f"This data module does not support {stage}.")
            
    # DataLoaders: these retrieve the data and feed it to the model. They are called by the Pytorch Lightning Trainer
                                            
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            sampler=None,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):

        if(self.test_dataset is None):
            return None
        else:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                sampler=None,
                num_workers=self.num_workers,
                pin_memory=True
            )

    def predict_dataloader(self):

        mosaic_dl = DataLoader(
            self.predict_dataset,
            batch_size = self.batch_size,
            sampler = None,
            num_workers = self.num_workers,
            pin_memory=True)

        return mosaic_dl
        
    def get_empty_mask(self):
    
        if(self.empty_mask is None):
            raise RuntimeError("The data module `empty_mask` attribute is not defined.")
        else:
            return self.empty_mask.copy()
        
    def fill_mask_mosaic(self, prediction_dl, check_crater_FID=None):
    
        """
        Get the predicted masks for individual tiles and write them to a global mask.
        """

        if(self.setup_predict_crater_FID is None):
            raise RuntimeError("No mask available. Run EjectaAzDataModule.predict_crater(PREDICT_CRATER_FID) and then run EjectaAzDataModule.setup('predict').")
        
        if(check_crater_FID is not None and check_crater_FID!=self.setup_predict_crater_FID):
            raise RuntimeError(f"The crater currently set up is {self.setup_predict_crater_FID}, but you are checking for crater {check_crater_FID}.")

        mosaic_mask = torch.tensor(self.empty_mask.values).float()
        mask_counts = torch.zeros(mosaic_mask.shape)

        if(torch.sum(mosaic_mask)>0):
            raise RuntimeError(f"The empty mask should have all pixels set to 0, but its sum is {np.sum(mosaic_mask.numpy())}.")
        
        # Write the tiles to the global mask in the CPU (less efficient) 
        '''
        mask_counts = np.zeros(mosaic_mask.shape, dtype=int)
        
        for item_dl in prediction_dl:

            (mask_batch,_), batch_pos = item_dl

            mask_batch = mask_batch.float().numpy().astype(np.float32)
            batch_pos = batch_pos.numpy()

            if(mask_batch.shape[1]!=mask_batch.shape[2]):
                raise ValueError(f"The batch should contain square images, instead its shape is {mask_batch.shape}.")
            
            for i in range(0, batch_pos.shape[0]):

                y_pos = batch_pos[i,0]
                x_pos = batch_pos[i,1]
                height = batch_pos[i,2]
                width = batch_pos[i,3]
                mosaic_mask[0,y_pos:y_pos+height, x_pos:x_pos+width] += mask_batch[i,:,:]
                mask_counts[0,y_pos:y_pos+height, x_pos:x_pos+width] += 1
        '''
        
        # Write the tiles to the global mask in the GPU, using Tensors
        
        I = 0
        for item_dl in prediction_dl:
            (mask_batch, _), batch_pos = item_dl

            # Ensure everything is float32 torch tensors
            mask_batch = mask_batch.float()  # shape: (B, H, W)
            batch_pos = batch_pos.long()     # shape: (B, 4) -> (y, x, h, w)

            B, H, W = mask_batch.shape
            device = mask_batch.device
            
            if(I==0):
                mosaic_mask = mosaic_mask.to(device)
                
            I+=1

            # Build coordinates for one sub-array
            yy, xx = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing="ij"
            )  # shape (H, W)

            # Broadcast to batch: (B, H, W)
            yy = yy.unsqueeze(0).expand(B, -1, -1)
            xx = xx.unsqueeze(0).expand(B, -1, -1)

            # Shift by batch positions
            yy = yy + batch_pos[:, 0].view(-1, 1, 1)  # y_pos
            xx = xx + batch_pos[:, 1].view(-1, 1, 1)  # x_pos

            # Compute linear indices for scatter
            H_big, W_big = mosaic_mask.shape[1:]  # (height, width) of large mask
            idx = yy * W_big + xx  # shape (B, H, W)
            idx = idx.reshape(-1)
            
            mask_batch = mask_batch.reshape(-1)
            ones_batch = torch.ones(mask_batch.shape)

            # Scatter-add into mosaic_mask
            mosaic_mask.view(-1).index_add_(
                0,
                idx,
                mask_batch
            )
            
            mask_counts.view(-1).index_add_(
                0,
                idx,
                ones_batch
            )
        
        mask_counts[mask_counts==0] = 1
        mosaic_mask = torch.ceil(mosaic_mask/mask_counts-0.5)
        
        final_mask = self.empty_mask.copy()
        final_mask[:,:,:]  = mosaic_mask.numpy().astype(np.uint8)
        
        return final_mask

