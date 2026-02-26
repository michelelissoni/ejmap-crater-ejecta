"""
Filename: connection_data_preparation.py
Author: Michele Lissoni
Date: 2026-02-22
"""

"""

Functions to handle the pre- and post-processing of the Connection model
and the preparation of the TV-Test splits

"""

import os
import sys
code_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, code_dir)
import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray

import itertools
import gc
import time


import sklearn.metrics as skm
from sklearn.model_selection import StratifiedShuffleSplit

import torch.nn as nn

import timm

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from connection_data_module import ConnectionDataModule
from terratorch_custom_tasks import BinarySemanticSegmentationTask
from terratorch.models.model import ModelOutput

from az_processing import reprojectBasemap, emptyAz

from planet_constants import Planet_Properties
PLANETARY_RADIUS = Planet_Properties.RADIUS

def getSplitClasses(connected, distances, distance_intervals = 5, distance_min = None, distance_max = None):

    """
    Split tiles into classes according to distance and presence of ejecta. Use a equal intervals scheme.
    
    Arguments:
    - connected (np.array): set to 1 if the tile contains ejecta, 0 if it doesn't.
    - distances (np.array): the tile distance from the crater center.
    
    Keywords:
    - distance_intervals (int): the number of distance classes (default: 5)
    - distance_min, distance_max (floats): the distances between which the class intervals should be computed. 
                                           By default, the minimum and maximum of `distances` will be used.
                                           
    Returns: 
    - split_classes (np.array): the class labels
    """

    distance_min = np.amin(distances) if distance_min is None else distance_min
    distance_max = np.amax(distances) if distance_max is None else distance_max

    distance_bins = np.linspace(np.amin(distances), np.amax(distances), distance_intervals+1)
    distance_classes = np.digitize(distances, distance_bins, right=True)-1
    
    distance_classes[distance_classes==-1] = 0
    distance_classes[distance_classes==distance_intervals] = distance_intervals-1
    
    split_classes = connected.astype(int)*distance_intervals+distance_classes

    return split_classes

# getSplitClasses version that uses geometric intervals

'''
def getSplitClasses(connected, distances, right_edge = 1, left_edge = 0, right_step = 0.5, coef = 0.5, bin_num = 5, planetary_radius = PLANETARY_RADIUS):

    """
    Split tiles into classes according to distance and presence of ejecta. Use a geometric intervals scheme.
    
    Arguments:
    - connected (np.array): set to 1 if the tile contains ejecta, 0 if it doesn't.
    - distances (np.array): the tile distance from the crater center.
    
    Keywords:
    - right_edge (float): the right edge of the distance range, to be multipied by pi*planetary_radius
    - left_efge (float): the left edge of the distance range, to be multipied by pi*planetary_radius
    - right_step (float): the first distance interval to the right, to be multipied by pi*planetary_radius
    - coef (float): the factor by which the distance intervals shrink as the distance reduces.
    - bin_num (int): the number of intervals.
    - planetary_radius (float): the radius of the planet
                                               
    Returns: 
    - split_classes (np.array): the class labels
    """

    bins = np.array([right_edge - right_step*np.sum(coef**np.arange(0,i+1)) for i in range(0,bin_num-1)])
    bins = np.sort(np.concatenate(([right_edge], bins, [left_edge])))

    distance_bins = bins*np.pi*planetary_radius
    distance_classes = np.digitize(distances, distance_bins, right=True)-1
    
    distance_classes[distance_classes==-1] = 0
    distance_classes[distance_classes==bin_num] = bin_num-1
    
    split_classes = connected.astype(int)*bin_num+distance_classes

    return split_classes
'''

def positionTestSplit(position_df, test_frac, max_h_circum_default = 0.25, special_max_h_circums = None, planetary_radius = PLANETARY_RADIUS):

    """
    Split tiles into train and test sets, and discard invalid tiles. 
    
    Arguments:
    - position_df (pd.DataFrame): the list of tiles.
    - test_frac (float): the fraction of the dataset that should go into the test set (from 0 to 1).
    
    Keywords:
    - max_h_circum_default (float): the maximum distance from the crater, beyond which a tile is discarded. Multiplied by pi*planetary_radius.
    - special_max_h_circums (dict): a dictionary containing the maximum distances for select craters.
    - planetary_radius (float): the radius of the planet.
    
    Returns: 
    - train_indices, test_indices, null_indices: the row numbers of the train, test and invalid tiles.
    """

    
    distances = position_df['distance'].values # Tile distances
    crater_FIDs = position_df['crat_FID'].values # Tile crater
    split_classes = position_df['split_class'].values # Split classes
    
    max_h_circums = np.repeat(max_h_circum_default, len(position_df))
    
    if(special_max_h_circums is not None):
        for crat_FID in list(special_max_h_circums.keys()):
            max_h_circums[np.flatnonzero(crater_FIDs==crat_FID)] = special_max_h_circums[crat_FID]
            
    # Discard tiles beyond a certain distance        
            
    valid_indices = np.flatnonzero(np.less(distances, max_h_circums*planetary_radius*np.pi))
    null_indices = np.flatnonzero(np.greater_equal(distances, max_h_circums*planetary_radius*np.pi))
    
    if(len(null_indices)==len(position_df)):
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.arange(len(position_df), dtype=int)
        
    # Get split classes
        
    split_classes = split_classes[valid_indices]
    
    split_uniq, split_index, split_counts = np.unique(split_classes, return_index=True, return_counts=True)
    
    one_indices = split_index[split_counts==1] # If a class has only one element, it will go into the train set.
    morethanone_indices = np.delete(np.arange(len(split_classes), dtype=int), one_indices)
    split_classes = np.delete(split_classes, one_indices)
    
    if(test_frac==0):
        train_indices = np.arange(len(morethanone_indices), dtype=int)
        test_indices = np.zeros(0, dtype=int)
    else:
        # Stratified random split into test and train
    
        test_size = int(test_frac*len(split_classes))
        test_size = len(np.unique(split_classes)) if test_size<len(np.unique(split_classes)) else test_size
        
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
        train_indices, test_indices = next(splitter.split(np.zeros(len(split_classes)), split_classes))

    train_indices = np.append(morethanone_indices[train_indices], one_indices)
    test_indices = morethanone_indices[test_indices]
    
    train_indices = valid_indices[train_indices]
    test_indices = valid_indices[test_indices]

    return train_indices, test_indices, null_indices
       
def getSpecObjKwargs(hparam_series, spec_obj, obj_name):

    """
    Retrieve parameters for the specific choice of scheduler, optimizer, smp model and loss.
    
    Arguments:
    - hparams_series (pd.Series): the series containing the hyperparameters for the version.
    - spec_obj: 'scheduler', 'optimizer', 'model_name' or 'loss'
    - obj_name: the chosen class (e.g. StepLR, AdamW...)
    
    Returns: 
    - obj_kwargs: a dictionary containing the keywords to be fed to the class.
    """

    hparam_names = hparam_series.index.values.astype(str)
    
    specific_objects = { 'scheduler': 'LR_',
                         'optimizer': 'OPT_',
                         'model_name': 'MDL_',
                         'loss': 'LS_'
                       }

    prefix = specific_objects[spec_obj]+obj_name+'_'
    
    obj_hparams = hparam_names[np.char.startswith(hparam_names, prefix)]

    if(len(obj_hparams)==0):
        return dict()
    
    obj_hparam_keys = np.array([obj_hparam.removeprefix(prefix) for obj_hparam in obj_hparams])
    obj_kwargs = { obj_hparam_keys[i]: hparam_series.loc[obj_hparams[i]] for i in range(0,len(obj_hparams)) }

    return obj_kwargs
    
def getDataModule(hparams_series, data_root, test_set_number = None, num_workers = 4, img_norm = None, att_norm = None): 

    """
    Prepare the ConnectionDataModule object (see connection_data_module.py), which creates the Pytorch Datasets and Dataloaders.
    
    Arguments:
    - hparams_series (pd.Series): the series containing the hyperparameters for the version.
    - data_root (str): the directory containing the connection model data.
    
    Keywords:
    - test_set_number (int): the train-test split used.
    - num_workers (int): the number of subprocesses used in data loading.
    - img_norm (tuple): the offset and denominator for normalization of the image channels. If None, ([0.,0.,0.],[255.,255.,255.]) 
    - att_norm (tuple): the offset and denominator for the normalization of the attention channel. If None, (0.0,2.0).
    
    Returns: 
    - data_module : the ConnectionDataModule object.
    """

    connection_weights = hparams_series.loc['connection_weights']
    distance_weights = hparams_series.loc['distance_weights']
    img_augment = hparams_series.loc['img_augment']
    val_frac = hparams_series.loc['val_frac']
    batch_size = int(hparams_series.loc['batch_size'])
    
    norm_from_train = False 
    
    if(img_norm is None):
        img_norm = ([0.,0.,0.],[255.,255.,255.])
        norm_from_train = hparams_series.loc['norm_from_train']
    att_norm = (0.0,2.0) if att_norm is None else att_norm 
    
    data_module = ConnectionDataModule(data_root,
                                       test_set = test_set_number,
                                       img_augment = img_augment,
                                       connection_weights = connection_weights,
                                       distance_weights = distance_weights,
                                       norm_from_train = norm_from_train,
                                       img_norm = img_norm,
                                       att_norm = att_norm,
                                       val_frac = val_frac,
                                       batch_size = batch_size,
                                       num_workers = num_workers)

    return data_module
    
class timmModel(nn.Module):

    """
    Neural network architecture, drawn from the timm library
    """

    def __init__(self, model_name, pretrained=False, in_chans=4, num_classes=1):
    
        super().__init__()
        self.full_model = timm.create_model(model_name, pretrained=pretrained, in_chans=in_chans, num_classes=num_classes)
        
    def forward(self, x):
        
        output = self.full_model(x)
        
        return ModelOutput(output)
        
def getSegTask(hparams_series):

    """
    Prepare the BinarySemanticSegmentationTask (see terratorch_custom_tasks.py) that runs the neural network
    by retrieving the hyperparameters.
    Even though conceived for semantic segmentation, the object can also be used for binary classification.
    
    Arguments:
    - hparams_series (pd.Series): the series containing the hyperparameters for the version.
    
    Returns: 
    - seg_task : the BinarySemanticSegmentationTask object.
    """

    # Neural network architecture

    model_name = hparams_series.loc["model_name"]
    pretrained = hparams_series.loc["pretrained"]
    
    model = timmModel(model_name, pretrained=pretrained, in_chans=4, num_classes=1)
    
    # Loss
    
    loss = hparams_series.loc["loss"]
    loss_kwargs = getSpecObjKwargs(hparams_series, 'loss', loss) # Get parameters specific to the loss
    focal_alpha = loss_kwargs["alpha"] if "alpha" in list(loss_kwargs.keys()) and loss == "focal" else None
    
    # Learning rate scheduler
    
    lr = hparams_series.loc["lr"]
    
    scheduler = hparams_series.loc["scheduler"]
    scheduler_hparams = getSpecObjKwargs(hparams_series, "scheduler", scheduler) # Get parameters specific to the scheduler
    scheduler_hparams = None if len(scheduler_hparams)==0 else scheduler_hparams
    
    # Optimizer
    
    optimizer = hparams_series.loc["optimizer"]
    optimizer_hparams = getSpecObjKwargs(hparams_series, "optimizer", optimizer) # Get parameters specific to the optimizer
    optimizer_hparams = None if len(optimizer_hparams)==0 else optimizer_hparams
    
    # Wrap architecture and hyperparameters in a BinarySemanticSegmentationTask
    
    seg_task = BinarySemanticSegmentationTask(
        model = model,
        loss = loss,
        focal_alpha = focal_alpha,
        lr = lr,
        optimizer = optimizer,
        optimizer_hparams = optimizer_hparams,
        scheduler = scheduler,
        scheduler_hparams = scheduler_hparams,
        plot_on_val = False,
        output_on_inference = "prediction",
        path_to_record_metrics = None
    )

    return seg_task

def getTrainer(hparams_series, version, test_set_number, iteration_num, log_dir, enable_progress_bar = False, enable_model_summary = False):

    """
    Get the Pytorch Lightning Trainer that runs the training.
    
    Arguments:
    - hparams_series (pd.Series): the series containing the hyperparameters for the version.
    - version (int): the version number
    - test_set_number (int): the number of the train/test split
    - iteration_num (int): the number of the iteration
    - log_dir (str): the directory to save the temporary files during training.
    
    Keywords:
    - enable_progress_bar (bool): if True, a progress bar appears during training.
    - enable_model_summary (bool): if True, the model summary is printed before training.
    
    Returns: 
    - trainer : the Trainer object.
    """
    
    val_metric = hparams_series.loc['val_metric']
    
    # Checkpoint callback: saves the model weights
    checkpoint_callback = ModelCheckpoint(
        monitor="val/"+val_metric,
        mode="max",
        dirpath=os.path.join(log_dir, "version_"+str(version), "checkpoints"),
        filename=f"best-checkpoint_test-set-{test_set_number}_iteration-{iteration_num}", # Checkpoint file name
        save_top_k=1,
        verbose = False,
        every_n_epochs=1
    )
    
    callbacks = [checkpoint_callback]
    
    # Early stopping
    
    early_stop_patience = hparams_series.loc['early_stop_patience']
    
    if(early_stop_patience is not None):
        
        checkpoint_early_stopping = EarlyStopping(
            monitor = "val/"+val_metric,
            mode = "max",
            patience = early_stop_patience,
            verbose = False
        )
        callbacks.append(checkpoint_early_stopping)
    
    max_epochs = int(hparams_series.loc['max_epochs'])
    
    # Trainer
    trainer = Trainer(
        accelerator="gpu",
        devices = 1,
        strategy="auto",
        precision = "16-mixed",
        num_nodes = 1,
        logger = False,
        max_epochs= max_epochs,
        check_val_every_n_epoch = 1,
        log_every_n_steps = 10,
        enable_checkpointing = True,
        enable_progress_bar = enable_progress_bar,
        enable_model_summary = enable_model_summary,
        callbacks = callbacks
    )

    return trainer
    
def evaluateConnMap(conn_version, 
                      test_set, 
                      conn_result_dir,
                      conn_data_root,
                      ejc_data_root,
                      cyl_data_root,
                      crater_FIDs = None,
                      num_workers = 8):
                      
    """
    Compute the evaluation metrics that would appear if a "perfect" Segmentation model were applied to the results
    of a real Connection model.
    
    Arguments:
    
    - conn_version: version of the Connection model
    - test_set: TV-Test split
    - conn_result_dir: results directory of the Connection model
    - conn_data_root: Connection model data directory
    - ejc_data_root: Segmentation model data directory
    - cyl_data_root: data directory of the manual masks
    
    Keywords:
    - crater_FIDs: if not None, the evaluation is performed only on the provided craters.
    - num_workers: the number of subprocesses
    
    Returns:
    - metrics_dict: a dictionary containing the metric values
    """

    fresh_craters_gdf = gpd.read_file(os.path.join(ejc_data_root, 'fresh_craters.shp')).set_index('crat_FID')
    fresh_craters_gdf = fresh_craters_gdf.loc[fresh_craters_gdf['train']==1,:]
    
    if(crater_FIDs is None):
        crater_FIDs = np.sort(fresh_craters_gdf.index.values)
    else:
        crater_FIDs = np.unique(np.array(crater_FIDs).astype(int))
    
    conn_version_dir = os.path.join(conn_result_dir, 'version_'+str(conn_version))
    
    all_positions = pd.read_csv(os.path.join(conn_data_root, "az_positions_all.csv"))
    
    all_positions = all_positions.loc[(all_positions['test_'+str(test_set)]>=0) & np.isin(all_positions['crat_FID'], crater_FIDs),:] # Get TVT positions
    
    # Get data size
    
    all_connected_len = 0 # Number of tiles
    test_all_connected_len = 0 # Number of tiles in the Test set
    all_values_len = 0 # Number of pixels
    test_all_values_len = 0 # Number of pixels in the test set
        
    for i, crater_FID in enumerate(crater_FIDs):
    
        valid_positions = all_positions.loc[all_positions['crat_FID']==crater_FID,:]
        pos_indices = valid_positions['pos_index'].values # Number of tiles
        test_pos_indices = pos_indices[valid_positions['test_'+str(test_set)]==1] # Number of tiles in the test set
         
        cyl_positions = pd.read_csv(os.path.join(cyl_data_root, 'positions_crater_'+str(crater_FID)+'.csv'), index_col='pos_index')
        
        all_values_len += np.sum(np.prod(cyl_positions.loc[pos_indices, ['height', 'width']], axis=1)) # Number of pixels
        test_all_values_len += np.sum(np.prod(cyl_positions.loc[test_pos_indices, ['height', 'width']], axis=1)) # Number of pixels in the test set
        all_connected_len += len(pos_indices)
        test_all_connected_len += len(test_pos_indices)

    all_true_values = np.zeros(all_values_len, dtype=bool) # True TVT pixel values
    test_all_true_values = np.zeros(test_all_values_len, dtype=bool) # True Test pixel values
    all_pred_values = np.zeros(all_values_len, dtype=bool) # Predicted TVT pixel values
    test_all_pred_values = np.zeros(test_all_values_len, dtype=bool) # Predicted Test pixel values

    all_true_connected = np.zeros(all_connected_len, dtype=bool) # True TVT tile connection
    test_all_true_connected = np.zeros(test_all_connected_len, dtype=bool) # True Test tile connection
    all_pred_connected = np.zeros(all_connected_len, dtype=bool) # Predicted TVT tile connection
    test_all_pred_connected = np.zeros(test_all_connected_len, dtype=bool) # Predicted Test tile connection
    
    I_values = 0
    test_I_values = 0
    I_conn = 0
    test_I_conn = 0
    
    # Loop over craters
    
    for i in range(0,len(crater_FIDs)):
    
        time_start = time.time()
        
        crater_FID = crater_FIDs[i]
        center_lon = fresh_craters_gdf.loc[crater_FID,'center_lon']
        center_lat = fresh_craters_gdf.loc[crater_FID,'center_lat']
        
        valid_positions = all_positions.loc[all_positions['crat_FID']==crater_FID,:]
        pos_indices = valid_positions['pos_index'].values
        test_pos_indices = pos_indices[valid_positions['test_'+str(test_set)]==1]
         
        cyl_positions = pd.read_csv(os.path.join(cyl_data_root, 'positions_crater_'+str(crater_FID)+'.csv'), index_col='pos_index')
        true_mask = rioxarray.open_rasterio(os.path.join(cyl_data_root, 'mask_crater_'+str(crater_FID)+'.tif')).astype(np.uint8)

        pred_df = pd.read_csv(os.path.join(conn_version_dir, 'conn_positions', 'conn-pos_test-set-'+str(test_set)+'_crater-'+str(crater_FID)+'.csv'), index_col = 'eval_pos_index')
        pred_len0 = len(pred_df)
        pred_df = pred_df.loc[pred_df['train_intersect']==1,:] # Eval tiles that intersect the TVT tiles
        pred_df['crat_FID'] = crater_FID

        # Create `conn_mask`, an azimuthal mask set to 1 in the area occupied by the azimuthal eval tiles that intersect the TVT tiles
        # The areas set to 1 are those that we would map to test an ejecta mask produced by the Segmentation model

        conn_mask = emptyAz((center_lon, center_lat))

        for eval_pos_index, row in pred_df.iterrows():

            x_pos = row.loc['x']
            y_pos = row.loc['y']
            height = row.loc['height']
            width = row.loc['width']

            conn_mask[0, y_pos:y_pos+height, x_pos:x_pos+width] = 1

        # Shrink `conn_mask` to the size of the are actually occupied by 1-pixels

        mask_half_size = int(conn_mask.shape[1]/2)

        one_indices_x = np.flatnonzero(np.any(conn_mask[0,:,:], axis=0))
        one_indices_y = np.flatnonzero(np.any(conn_mask[0,:,:], axis=1))
        
        if(len(one_indices_x)>0 and len(one_indices_y)>0):
            bound_min_x = np.amin(one_indices_x)
            bound_max_x = np.amax(one_indices_x)+1
            bound_min_y = np.amin(one_indices_y)
            bound_max_y = np.amax(one_indices_y)+1

            mask_half_size_new = np.amax(np.abs(np.array([bound_min_x, bound_max_x, bound_min_y, bound_max_y])-mask_half_size))
            
            conn_mask = conn_mask[:, mask_half_size-mask_half_size_new:mask_half_size+mask_half_size_new, mask_half_size-mask_half_size_new:mask_half_size+mask_half_size_new]

        # Simulate a perfect Segmentation model by multiplying the manual mask by `conn_mask`, isolating the areas that
        # would be mapped to evaluate the model
        pred_mask = true_mask.values*reprojectBasemap(conn_mask, num_threads = num_workers).values        

        # Get pixel and tile connection values in the TVT and Test cylindrical tiles

        for pos_index in pos_indices:

            x_pos = cyl_positions.loc[pos_index, 'x']
            y_pos = cyl_positions.loc[pos_index, 'y']
            height = cyl_positions.loc[pos_index, 'height']
            width = cyl_positions.loc[pos_index, 'width']

            true_values = true_mask[0, y_pos:y_pos+height, x_pos:x_pos+width].values.flatten()
            pred_values = pred_mask[0, y_pos:y_pos+height, x_pos:x_pos+width].flatten()

            true_connected = int(np.any(true_values == 1))
            pred_connected = int(np.any(pred_values == 1))
            
            all_true_values[I_values:I_values+height*width] = true_values
            all_pred_values[I_values:I_values+height*width] = pred_values
            all_true_connected[I_conn] = true_connected
            all_pred_connected[I_conn] = pred_connected
            
            I_values += height*width
            I_conn += 1 
            
            if(pos_index in test_pos_indices):
            
                test_all_true_values[test_I_values:test_I_values+height*width] = true_values
                test_all_pred_values[test_I_values:test_I_values+height*width] = pred_values
                test_all_true_connected[test_I_conn] = true_connected
                test_all_pred_connected[test_I_conn] = pred_connected
                
                test_I_values += height*width
                test_I_conn += 1 
                            
        print(crater_FID, len(pred_df), pred_len0, time.time()-time_start)
        time_start = time.time()
        
        del true_values
        del pred_values
        del true_mask
        del conn_mask
        del pred_mask
        gc.collect()

    # Compute evaluation metrics

    metrics_dict = {'Accuracy': skm.accuracy_score(all_true_values, all_pred_values),
                    'F1_Score': skm.f1_score(all_true_values, all_pred_values),
                    'IoU': skm.jaccard_score(all_true_values, all_pred_values),
                    'Precision': skm.precision_score(all_true_values, all_pred_values),
                    'Recall': skm.recall_score(all_true_values, all_pred_values),
                    'conn_Accuracy': skm.accuracy_score(all_true_connected, all_pred_connected),
                    'conn_F1_Score': skm.f1_score(all_true_connected, all_pred_connected),
                    'conn_IoU': skm.jaccard_score(all_true_connected, all_pred_connected),
                    'conn_Precision': skm.precision_score(all_true_connected, all_pred_connected),
                    'conn_Recall': skm.recall_score(all_true_connected, all_pred_connected)
                    }
                    
    if(len(test_all_true_values)>0):
        
        metrics_dict['test_Accuracy'] = skm.accuracy_score(test_all_true_values, test_all_pred_values)
        metrics_dict['test_F1_Score'] = skm.f1_score(test_all_true_values, test_all_pred_values)
        metrics_dict['test_IoU'] = skm.jaccard_score(test_all_true_values, test_all_pred_values)
        metrics_dict['test_Precision'] = skm.precision_score(test_all_true_values, test_all_pred_values)
        metrics_dict['test_Recall'] = skm.recall_score(test_all_true_values, test_all_pred_values)

        metrics_dict['conn_test_Accuracy'] = skm.accuracy_score(test_all_true_connected, test_all_pred_connected)
        metrics_dict['conn_test_F1_Score'] = skm.f1_score(test_all_true_connected, test_all_pred_connected)
        metrics_dict['conn_test_IoU'] = skm.jaccard_score(test_all_true_connected, test_all_pred_connected)
        metrics_dict['conn_test_Precision'] = skm.precision_score(test_all_true_connected, test_all_pred_connected)
        metrics_dict['conn_test_Recall'] = skm.recall_score(test_all_true_connected, test_all_pred_connected)   
        
    return metrics_dict

# Function to add versions with new hyperparameters to the list. Not used because it is easier to do it manually.     
'''
def addVersionHparams(input_dict, version_df, default_dict, data_root, previous_values = None, drop_duplicate = False):

    input_dict = input_dict.copy()

    specific_objects = { 'scheduler': 'LR_',
                         'optimizer': 'OPT_',
                         'model_name': 'MDL_',
                         'loss' : 'LS_'}

    hparams = version_df.drop('version_run', axis=1).columns.values
    previous_versions = version_df.index.values
    
    hparams_input = np.array(list(input_dict.keys()))
    
    for spec_obj, prefix in specific_objects.items():

        if(spec_obj not in hparams_input):
            continue

        hparams_input_obj = hparams_input[np.char.startswith(hparams_input, prefix)]
        if(len(hparams_input_obj)==0):
            continue

        input_objs = np.array([splitstr[1] for splitstr in np.char.split(hparams_input_obj, '_')])

        obj_options = input_dict[spec_obj]

        hparams_input_obj2 = hparams_input_obj[np.isin(input_objs, obj_options)]
        invalid_hparams = hparams_input[~np.isin(hparams_input, hparams_input_obj2) & np.char.startswith(hparams_input, prefix)]
        hparams_input = hparams_input[np.isin(hparams_input, hparams_input_obj2) | ~np.char.startswith(hparams_input, prefix)]

        for invalid_hparam in invalid_hparams:
            input_dict.pop(invalid_hparam)
        

        if(len(hparams_input_obj2)>0 and spec_obj not in hparams_input):
            
            hparams_input = np.append(hparams_input, spec_obj)
            input_dict[spec_obj] = list(obj_options)  


    hparams_input_new = hparams_input[~np.isin(hparams_input, hparams)]

    previous_values_err = f"There are some new hparams: {hparams_input_new}. But you must specify previous values for all of them using the `previous_values` keyword."
    
    if(previous_values is not None and len(hparams_input_new)>0):
        
        hparams_input_new_previous = np.array(list(previous_values.keys()))
        
        if( not( np.all( np.isin(hparams_input_new, hparams_input_new_previous) ) ) ):
            raise ValueError(previous_values_err)
    
        for hparam_new in hparams_input_new:
            previous_value = previous_values[hparam_new]
            
            if(isinstance(previous_value, list) and isinstance(previous_value[0], str) and isinstance(previous_value[1], dict)):
                hparam_rule = previous_value[0]
                rule_dict = previous_value[1]

                version_df[hparam_new] = list(rule_dict.values())[0]
                for rule_value, value in rule_dict.items():
                    version_df.loc[version_df[hparam_rule]==rule_value,hparam_new] = value
                    
                version_df.loc[~np.isin(version_df[hparam_rule], list(rule_dict.values())),hparam_new] = np.nan
  
            else:
                version_df[hparam_new] = previous_value

            hparams = version_df.drop('version_run', axis=1).columns.values
    
    elif(previous_values is None and len(hparams_input_new)>0):
        raise ValueError(previous_values_err)


    hparam_input_combinations = list(itertools.product(*list(input_dict.values())))
    input_version_df = pd.DataFrame(hparam_input_combinations, columns = hparams_input)

    for hparam in version_df.drop('version_run', axis=1).columns.values:
        
        if(hparam not in input_version_df.columns.values):

            if(hparam in list(default_dict.keys())):
                input_version_df[hparam] = default_dict[hparam]
            else:
                raise ValueError(f"'{hparam}' is not a new hyperparameter, but nor is it in the default values dictionary.")

    hparams_after = input_version_df.columns.values.astype(str)
    
    for spec_obj, prefix in specific_objects.items():

        hparams_after_obj = hparams_after[np.char.startswith(hparams_after, prefix)]
        after_objs = np.array([splitstr[1] for splitstr in np.char.split(hparams_after_obj, '_')])

        for after_obj in np.unique(after_objs):

            input_version_df.loc[input_version_df[spec_obj]!=after_obj, hparams_after_obj[after_objs==after_obj]] = np.nan

    input_version_df.drop_duplicates(inplace=True)

    input_version_df['version'] = np.arange(len(input_version_df), dtype=int) + np.amax(previous_versions) + 1
    input_version_df['version_run'] = False
    input_version_df.set_index('version', inplace=True)

    version_df2 = pd.concat((version_df, input_version_df), axis=0)
    version_df2.fillna(np.nan, inplace=True)

    if(drop_duplicate):
        version_df2['version_run'] = False
        version_df2.drop_duplicates(inplace = True, ignore_index = False)

        input_version_df = version_df2.loc[version_df2.index.values > np.amax(previous_versions),:]

        input_version_df['version'] = np.arange(len(input_version_df), dtype=int) + np.amax(previous_versions) + 1
        input_version_df.set_index('version', inplace=True)        
    
        version_df = pd.concat((version_df, input_version_df), axis=0)

    else:
        version_df = version_df2.copy()

    return version_df
'''
