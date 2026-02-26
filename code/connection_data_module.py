"""
Filename: connection_data_module.py
Author: Michele Lissoni
Date: 2026-02-23
"""

"""

Define the data module class that organizes the data fed to the Connection model.

"""

import os
import numpy as np
import pandas as pd

import copy

from sklearn.model_selection import StratifiedShuffleSplit

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from lightning.pytorch import LightningDataModule

from planet_constants import Planet_Properties
PLANETARY_RADIUS = Planet_Properties.RADIUS

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

def geomAugmenter(image):

    """
    
    Random flipping augmentation of an image (no rotation because the Connection model uses rectangular images).
    
    Arguments:
    - image: a 3D Numpy array, the first axis indicates the bands. 
    
    """

    image = np.swapaxes(image, 0, 2)
    
    flipud_augmenter = iaa.flip.Flipud(0.5) # Flipping, with a 50% probability
    fliplr_augmenter = iaa.flip.Fliplr(0.5)
    
    augmenter = iaa.meta.Sequential([
        flipud_augmenter,
        fliplr_augmenter,
    ], random_order=False)

    image_aug = augmenter(image=image)
    
    image_aug = np.swapaxes(image_aug, 0, 2)

    return image_aug

class ConnectionDataset(Dataset):

    """
    
    A Pytorch Dataset child class, handles the labeled data for the training, validation, testing and prediction of the Connection model.
    
    """

    def __init__(
        self,
        data_root,
        position_df,
        predict = False,
        img_augment = False,
        img_norm = ([0,0,0], [255.,255.,255.]),
        att_norm = (0.0,2.0),
        **kwargs
    ):
    
        """
        
        Constructor
        
        Arguments:
        - data_root (str): the path of the data root folder.
        - position_df (pd.DataFrame): the list of all tile positions
        
        Keywords:
        - predict (bool): if True, retrieves evaluation data. If False, retrieves TVT data.
        - img_augment (bool): implement on-the-fly image augmentation
        - img_norm (tuple): normalization offset and denominator for the image bands, default ([0,0,0], [255.,255.,255.])
        - att_norm (tuple): normalization offset and denominator for the attention band (positions of crater and tile), default (0,2)
        - **kwargs: keyword arguments for the parent class
        
        """
        
        super().__init__(**kwargs)
       
        self.data_root = data_root
        # Directories for TVT and evaluation data, depending on `predict`
        image_dir = 'eval_images' if predict else 'images'
        att_dir = 'eval_attention' if predict else 'attention'

        self.img_norm = (np.array(img_norm[0]), np.array(img_norm[1]))
        self.att_norm = att_norm
        
        rng = np.random.default_rng()
        random_args = np.arange(len(position_df), dtype=int)
        rng.shuffle(random_args)
        
        position_df = position_df.iloc[random_args,:]
        
        pos_index_list = position_df['pos_index'].values
        crat_FID_list = position_df['crat_FID'].values
        data_codes = position_df['data_code'].values
        
        if(not(predict)):
            distances = position_df['distance'].values
            connected = (position_df['num_1'].values > 0).astype(int)
            split_classes = position_df['split_class'].values
        else:
            distances = None
            connected = None
            split_classes = None
            
        # Store data in attributes
        
        self.pos_index_list = pos_index_list
        self.crat_FID_list = crat_FID_list
        self.data_codes = data_codes
        self.connected = connected
        self.distances = distances
        self.split_classes = split_classes
        
        self.img_augment = img_augment
        
        self.predict = predict
        
    def __len__(self):
        return len(self.pos_index_list)

    def weights(self, connection = True, connection_1_factor = 1, distance = True):
    
        """
        Create weights.
        
        Keywords:
        - connection (bool): if True, a weight correcting for the disproportion between connected and non-connected tiles is applied.
        - connection_1_factor (float): a factor that connected tiles should be multiplied by.
        - distance (bool): if True, a weight proportional to the distance is applied to the connected tiles and inversely proportional to the non-connected tiles,
                           to emphasize areas without ejecta close to the crater and areas with ejecta far from it. 
        
        """
        
        weights = np.ones(len(self))
        
        if(connection):
            connection_weights = np.zeros(len(self))
            connection_weights[self.connected==1] = connection_1_factor/np.sum(self.connected==1)
            connection_weights[self.connected==0] = 1/np.sum(self.connected==0)
            connection_weights = connection_weights/np.sum(connection_weights)
        else:
            connection_weights = np.ones(len(self))
            
        if(distance):
            distance_weights_1 = self.distances[self.connected==1]
            distance_weights_1 = distance_weights_1/np.sum(distance_weights_1)
            distance_weights_0 = np.reciprocal(self.distances[self.connected==0])
            distance_weights_0 = distance_weights_0/np.sum(distance_weights_0)
            
            distance_weights = np.ones(len(self))
            distance_weights[self.connected==1] = distance_weights_1
            distance_weights[self.connected==0] = distance_weights_0
            distance_weights = distance_weights/np.sum(distance_weights)
        else:
            distance_weights = np.ones(len(self))
            
        weights = weights*connection_weights*distance_weights
        weights = weights/np.sum(weights)

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
        
        # Split the data

        splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size)
        train_indices, val_indices = next(splitter.split(np.zeros(len(split_classes)), split_classes))
        
        train_indices = np.append(morethanone_indices[train_indices], one_indices)
        val_indices = morethanone_indices[val_indices]
        
        # Fill the two new Datasets

        train_dataset.pos_index_list = self.pos_index_list[train_indices]
        train_dataset.crat_FID_list = self.crat_FID_list[train_indices]
        train_dataset.distances = self.distances[train_indices]
        train_dataset.connected = self.connected[train_indices]
        train_dataset.split_classes = self.split_classes[train_indices]
        train_dataset.data_codes = self.data_codes[train_indices]

        val_dataset.pos_index_list = self.pos_index_list[val_indices]
        val_dataset.crat_FID_list = self.crat_FID_list[val_indices]
        val_dataset.distances = self.distances[val_indices]
        val_dataset.connected = self.connected[val_indices]
        val_dataset.split_classes = self.split_classes[val_indices]
        val_dataset.data_codes = self.data_codes[val_indices]

        return train_dataset, val_dataset

    def __getitem__(self, index):
    
        """
        
        This method controls how a data sample (the rectangular image and attention channel) is retrieved from memory.
        The index is the data sample index.
        
        """    
    
        pos_index = self.pos_index_list[index]
        crat_FID = self.crat_FID_list[index]
        data_code = self.data_codes[index]
        
        filesuffix = str(pos_index)+'_crater_'+str(crat_FID)+'.npy'
        
        if(self.predict):
            image_path = os.path.join(self.data_root, 'eval_images', 'eval_image_'+filesuffix)
            att_path = os.path.join(self.data_root, 'eval_attention', 'eval_att_'+filesuffix)
        else:
            image_path = os.path.join(self.data_root, 'images', 'image_'+filesuffix)
            att_path = os.path.join(self.data_root, 'attention', 'att_'+filesuffix)
            connected = self.connected[index]
                
        image = np.load(image_path) # Get image
        
        if(self.img_augment):
            image = colorAugmenter(image) # On-the-fly color augmentation
            
        image = (image - self.img_norm[0][:,np.newaxis,np.newaxis])/self.img_norm[1][:,np.newaxis,np.newaxis] # Normalize image
        attention = (np.load(att_path) - self.att_norm[0])/self.att_norm[1] # Get and normalize attention
        
        image = np.concatenate((image, attention), axis=0) # Bundle image and attention
        
        if(self.img_augment):
            image = geomAugmenter(image) # On-the-fly flip augmentation
            
        image_dev = torch.tensor(image.copy(), dtype=torch.float32)
        
        metadata = torch.tensor([pos_index, crat_FID, data_code])

        if(self.predict):
            sample = {
                "image": image_dev,
                "filename": metadata
            } # These keys are read by the Terratorch Segmentation Task.
              # "filename" in this case can be used to store any metadata
                
        else:
            sample = {
                "image": image_dev,
                "mask": torch.tensor([connected], dtype=torch.float32), # In this case, "mask" contains only a single value
                "filename": metadata
            }
        
        return sample


class ConnectionDataModule(LightningDataModule):

    """
    
    A LightningDataModule child class, constructs the Datasets and Dataloaders to feed data to the model.
    
    """    

    def __init__(
        self,
        data_root,
        test_set = None,
        connection_weights = True,
        distance_weights = True,
        norm_from_train = False,
        img_augment = False,
        img_norm = ([0,0,0], [255.,255.,255.]),
        att_norm = (0.0,2.0),
        val_frac = 0.1,
        test_frac = 0.2,
        batch_size = 4,
        num_workers=0
    ):
    
        """
        
        Constructor
        
        Arguments:
        - data_root (str): the path of the segmentation data root folder.

        Keywords:
        - test_set (int or None): the identifying number of the TV-Test split. If None, a random TV-Test split is carried out.
        - connection_weights (bool): if True, a weight correcting for the disproportion between connected and non-connected tiles is applied.
        - distance_weights (bool): if True, a weight proportional to the distance is applied to the connected tiles and inversely proportional to the non-connected tiles,
                                   to emphasize areas without ejecta close to the crater and areas with ejecta far from it. 
        - norm_from_train (bool): compute normalization parameters from TV data.
        - img_augment (bool): perform on-the-fly image augmentation. 
        - img_norm (tuple): normalization offset and denominator for the image bands, default ([0,0,0], [255.,255.,255.])
        - att_norm (tuple): normalization offset and denominator for the distance band, default (0.0,2.0)
        - val_frac (float): fraction of TV data that goes into Validation
        - test_frac (float): fraction of TVT data that goes into the Test set (only applied if `test_set` is None).
        - batch_size (int): the batch size
        - num_workers (int): the number of subprocesses used by the dataloaders
                
        """

        super().__init__()
        
        position_df = pd.read_csv(os.path.join(data_root,'az_positions_all.csv')) # List of all azimuthal TVT tiles for all craters
        
        self.tt_crater_FIDs = np.unique(position_df['crat_FID']) # TVT craters
        
        if(test_set is None): # Carry out a random TV-Test split if the split number is not specified 
        
            valid_indices = np.flatnonzero(position_df['split_class']>=0)
            
            if(test_frac==0):
                train_indices = valid_indices
                test_indices = np.zeros(0, dtype=int)
            else:
                test_size = int(test_frac*len(valid_indices))
            
                splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
                train_indices, test_indices = next(splitter.split(np.zeros(len(valid_indices)), position_df['split_class'].values[valid_indices]))
            
                train_indices = valid_indices[train_indices]
                test_indices = valid_indices[test_indices]

        elif(test_set is not None):
            train_indices = np.flatnonzero(position_df['test_'+str(test_set)]==0)
            test_indices = np.flatnonzero(position_df['test_'+str(test_set)]==1)
        
        test_cols = position_df.columns.values[np.char.startswith(position_df.columns.values.astype(str), 'test_')]
        
        position_df = position_df.drop(test_cols, axis=1)
        
        # Split the tiles in TV and Test
        
        train_df = position_df.iloc[train_indices,:]
        test_df = position_df.iloc[test_indices,:]

        # Compute normalizatiion parameters from TV data

        if(norm_from_train):
        
            for i in range(0,len(train_df)):
            
                pos_index = train_df['pos_index'].iloc[i]
                crat_FID = train_df['crat_FID'].iloc[i]

                filesuffix = str(pos_index)+'_crater_'+str(crat_FID)+'.npy'

                image_path = os.path.join(data_root, 'images', 'image_'+filesuffix)
               
                image = np.load(image_path)
                
                if(i==0):
                    subimage_area = image.shape[1]*image.shape[2]
                    image_bands = np.zeros((image.shape[0], subimage_area*len(train_df)))
                
                image_bands[:, i*subimage_area:(i+1)*subimage_area] = image.reshape(image.shape[0], subimage_area)
                
            mean_bands = np.mean(image_bands, axis=1)
            std_bands = np.std(image_bands, axis=1, ddof=1)
            
            img_norm = (mean_bands, std_bands)
            
            self.img_norm = img_norm
            self.att_norm = (1.5, 0.5)
            
        else:

            self.img_norm = img_norm 
            
            self.att_norm = att_norm
        
        # Store the data and parameters in attributes
        
        self.data_root = data_root
        
        self.train_df = train_df
        self.test_df = test_df
        
        self.pred_df = None
        self.predict_crater_FID = None
        self.setup_predict_crater_FID = None
        self.predict_tt = False
        
        self.img_augment = img_augment
        
        self.connection_weights = connection_weights
        self.distance_weights = distance_weights

        self.val_frac = val_frac

        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def get_norms(self):
    
        norm_means = np.append(np.array(self.img_norm[0]), self.att_norm[0])
        norm_stds = np.append(np.array(self.img_norm[1]), self.att_norm[1])
        
        norm_arr = np.vstack([norm_means[np.newaxis,:], norm_stds[np.newaxis,:]])
    
        return norm_arr
        
    def predict_train_test(self):
    
        """
        Predict only the TVT data: used for quantitative evaluation
        """
    
        self.predict_tt = True
        self.predict_crater_FID = None
        self.setup_predict_crater_FID = None
        
        train_df = self.train_df.copy()
        train_df['data_code'] = 1
        test_df = self.test_df.copy()
        test_df['data_code'] = 2
        
        self.pred_df = pd.concat((train_df, test_df), axis=0, ignore_index=True)
        
    def predict_crater(self, crater_FID, only_tt = True):
    
        """
        Set the crater whose connected tiles
        """
    
        if(only_tt and crater_FID not in self.tt_crater_FIDs):
            return False

        self.predict_crater_FID = crater_FID
        
        pred_df = pd.read_csv(os.path.join(self.data_root,'eval_positions','eval_positions_crater_'+str(crater_FID)+'.csv'))
        pred_df.loc[:,'pos_index'] = pred_df['eval_pos_index'].values
        pred_df['crat_FID'] = crater_FID
        pred_df.drop('eval_pos_index', axis=1)
        pred_df['data_code'] = 0
        
        self.pred_df = pred_df
        self.predict_tt = False
        
        return True

    def setup(self, stage):
    
        """
        This function is called by a Pytorch Lightning Trainer 
        to prepare a 'fit', 'test' or 'predict' operation.
        
        It prepares the relevant Datasets.
        """

        if(stage == 'fit'):
        
            train_df = self.train_df.copy()
            train_df['data_code'] = 1
            
            # TV data

            train_dataset = ConnectionDataset(self.data_root,
                                              train_df,
                                              predict = False,
                                              img_augment = self.img_augment,
                                              img_norm = self.img_norm,
                                              att_norm = self.att_norm)

            # Train and val datasets

            self.train_dataset, self.val_dataset = train_dataset.train_val_split(self.val_frac)

            # Train weights

            train_weights = self.train_dataset.weights(connection = self.connection_weights,
                                                 distance = self.distance_weights)
                                                 
            self.train_sampler = WeightedRandomSampler(train_weights, 
                                                       num_samples = len(self.train_dataset), 
                                                       replacement=True)
                                                       
        elif(stage == 'test'):

            test_df = self.test_df.copy()
            test_df['data_code'] = 2
            
            if(len(test_df)==0):
                self.test_dataset = None
            else:
            
                # Test dataset
            
                self.test_dataset = ConnectionDataset(self.data_root,
                                                      test_df,
                                                      predict = False,
                                                      img_augment = False,
                                                      img_norm = self.img_norm,
                                                      att_norm = self.att_norm)
                                                  
        elif(stage == 'predict'):

            if(self.predict_crater_FID is None and not(self.predict_tt)):
                 raise ValueError("Run ConnectionDataModule.predict_crater(PREDICT_CRATER_FID) first, with PREDICT_CRATER_FID the FID of the crater you want to predict. Or if you want to predict the train and test sets, run ConnectionDataModule.predict_train_test().")   
                         
            # Dataset containing data to predict
                         
            self.predict_dataset = ConnectionDataset(self.data_root,
                                                      self.pred_df,
                                                      predict = not(self.predict_tt), # TVT data?
                                                      img_augment = False,
                                                      img_norm = self.img_norm,
                                                      att_norm = self.att_norm)
                                                      
            self.setup_predict_crater_FID = self.predict_crater_FID
            
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

        return DataLoader(
            self.predict_dataset,
            batch_size = self.batch_size,
            sampler = None,
            num_workers = self.num_workers,
            pin_memory=True)

    def predict_positions(self, prediction_dl, predict_tt = False):
    
        """
        Get the predicted connection values for individual tiles and write them to a DataFrame
        
        Arguments:
        - prediction_dl : the dataloader containing the predictions
        - predict_tt (bool): if True, returns separate DataFrames for TV and Test tiles
        """

        connected_pos_indices = np.zeros(0, dtype=int)

        for item_dl in prediction_dl: # Iterate through the dataloader containing the predictions

            (pred_batch,_), batch_pos = item_dl

            pred_batch = pred_batch.float().numpy().astype(int)
            pos_indices = batch_pos.numpy()[:,0]
            
            connected_pos_indices = np.append(connected_pos_indices, pos_indices[pred_batch[:,0]==1])

        connected_pos_df = self.pred_df.loc[np.isin(self.pred_df['pos_index'], connected_pos_indices),:].copy()
        data_codes = connected_pos_df['data_code'].values
        connected_pos_df.drop('data_code', axis=1, inplace=True)
        
        if(predict_tt): # Distinguish TV and Test tiles
            connected_train_df = connected_pos_df.iloc[np.flatnonzero(data_codes==1),:]
            connected_test_df = connected_pos_df.iloc[np.flatnonzero(data_codes==2),:]
            
            return connected_train_df, connected_test_df
        
        else:
        
            if(np.any(data_codes!=0)):
                raise RuntimeError("If you're not predicting the train and test sets, the data_code should always be 0.")

            connected_pos_df.loc[:,'eval_pos_index'] = connected_pos_df['pos_index'].values
            connected_pos_df.drop(['pos_index', 'crat_FID'], axis=1, inplace=True)
        
        return connected_pos_df

