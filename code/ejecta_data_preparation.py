"""
Filename: ejecta_data_preparation.py
Author: Michele Lissoni
Date: 2026-02-22
"""

"""

Functions to handle the pre- and post-processing of the Segmentation model

"""

import os
import sys
code_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, code_dir)
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.colors as mcolors
import rasterio
import rioxarray
import xarray as xr
import gc
import time
import warnings
import numexpr as ne

from psutil import virtual_memory
from torch.cuda import memory_reserved as cuda_memory_reserved

import itertools

import sklearn.metrics as skm

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from az_processing import reprojectBasemap
from pos_processing import fillCyl, emptyPosMask
from ejecta_data_module import EjectaAzDataModule
from terratorch_custom_tasks import BinarySemanticSegmentationTask
from smp_custom_models import SMP_midinput

from planet_constants import Planet_CRS, Planet_Basemap
BASEMAP_RESOLUTION = Planet_Basemap.BASEMAP_RESOLUTION
    
def getSpecObjKwargs(hparam_series, spec_obj, obj_name):

    """
    Retrieve parameters for the specific choice of scheduler, optimizer, smp model and loss.
    
    Arguments:
    - hparams_series (pd.Series): the series containing the hyperparameters for the version.
    - spec_obj: 'scheduler', 'optimizer', 'model_name' or 'loss'
    - obj_name: the chosen class (e.g. StepLR, AdamW, Unet...)
    
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
    
def getDataModule(hparams_series, data_root, predict=False, only_train = True, train_df = None, test_df = None, img_norm = None, dist_norm = None, num_workers = 4): 

    """
    Prepare the EjectaAzDataModule object (see ejecta_data_module.py), which creates the Pytorch Datasets and Dataloaders.
    
    Arguments:
    - hparams_series (pd.Series): the series containing the hyperparameters for the version.
    - data_root (str): the directory containing the segmentation model data.
    
    Keywords:
    - predict (bool): set to True to use the module for prediction, to False to use it for training.
    - only_train (bool): import only training data (not the craters that are only used in the final mapping)
    - train_df (pd.DataFrame): the training and validation data (leave as None if only prediction)
    - test_df (pd.DataFrame): the test data.
    - img_norm (tuple): the offset and denominator for normalization of the image channels. If None, ([0.,0.,0.],[255.,255.,255.]) 
    - dist_norm (tuple): the offset and denominator for normalization of the distance channel. If None, (0.0,1.0).
    - num_workers (int): the number of subprocesses used in data loading.
    
    Returns: 
    - data_module : the EjectaAzDataModule object.
    """
     
    buffer_bounds = eval(hparams_series.loc['buffer_bounds'])
    distance_mode = str(hparams_series.loc['distance_mode'])
    img_augment = hparams_series.loc['img_augment']
    log_dist = hparams_series.loc['log_dist']
    num_1_factor = hparams_series.loc['num_1_factor']
    num_1_point95 = hparams_series.loc['num_1_point95']
    norm_from_train = hparams_series.loc['norm_from_train']
    val_frac = hparams_series.loc['val_frac']
    batch_size = int(hparams_series.loc['batch_size'])
    
    norm_from_train = hparams_series.loc['norm_from_train'] and (img_norm is None and dist_norm is None) 
    
    img_norm = ([0.,0.,0.],[255.,255.,255.]) if img_norm is None else img_norm
    dist_norm = (0.0,1.0) if dist_norm is None else dist_norm
    
    data_module = EjectaAzDataModule(data_root,
                                     predict = predict,
                                     only_train = only_train or not(predict),
                                     train_df = train_df,
                                     test_df = test_df,
                                     buffer_bounds = buffer_bounds,
                                     distance_mode = distance_mode,
                                     img_augment = img_augment,
                                     log_dist = log_dist,
                                     num_1_factor = num_1_factor,
                                     num_1_point95 = num_1_point95,
                                     norm_from_train = norm_from_train,
                                     img_norm = img_norm,
                                     dist_norm = dist_norm,
                                     val_frac = val_frac,
                                     batch_size = batch_size,
                                     num_workers = num_workers)

    return data_module

def getSegTask(hparams_series):

    """
    Prepare the BinarySemanticSegmentationTask (see terratorch_custom_tasks.py) that runs the neural network
    by retrieving the hyperparameters.
    
    Arguments:
    - hparams_series (pd.Series): the series containing the hyperparameters for the version.
    
    Returns: 
    - seg_task : the BinarySemanticSegmentationTask object.
    """

    # Neural network architecture

    distance_mode = str(hparams_series.loc["distance_mode"])
    model_name = hparams_series.loc["model_name"]
    accept_midinput = hparams_series.loc["accept_midinput"]
    if(distance_mode=='absolute'):
        midinput_dim = 4
    else:
        midinput_dim = 3
    
    model_kwargs = getSpecObjKwargs(hparams_series, 'model_name', model_name) # Get parameters specific to the architecture
    
    # Loss
    
    loss = hparams_series.loc["loss"]
    loss_kwargs = getSpecObjKwargs(hparams_series, 'loss', loss) # Get parameters specific to the loss
    focal_alpha = loss_kwargs["alpha"] if "alpha" in list(loss_kwargs.keys()) and loss == "focal" else None
    
    # Learning rate scheduler
    
    lr = hparams_series.loc["lr"]
    
    scheduler = hparams_series.loc["scheduler"]
    
    if(scheduler is not None):
        warmup = scheduler.endswith("_Warmup")
        scheduler = scheduler.removesuffix("_Warmup")
        
        scheduler_hparams = getSpecObjKwargs(hparams_series, "scheduler", scheduler) # Get parameters specific to the scheduler
        scheduler_hparams = dict() if len(scheduler_hparams)==0 else scheduler_hparams
        
        if(warmup): # Add warmup to scheduler
            warmup_milestone = hparams_series.loc["LR_Warmup_milestone"]
        
            scheduler_hparams = {"schedulers": {"LinearLR": {"start_factor": 0.01, "end_factor": 1.0, "total_iters": warmup_milestone},
                                                scheduler: scheduler_hparams},
                                 "milestones": [warmup_milestone]}
            scheduler = "SequentialLR"
    else:
        scheduler_hparams = dict()
    
    # Optimizer
    
    optimizer = hparams_series.loc["optimizer"]
    optimizer_hparams = getSpecObjKwargs(hparams_series, "optimizer", optimizer) # Get parameters specific to the optimizer
    optimizer_hparams = None if len(optimizer_hparams)==0 else optimizer_hparams
    
    model = SMP_midinput(model_name, midinput_dim = midinput_dim, accept_midinput = accept_midinput, **model_kwargs) # Neural network architecture
    
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
        output_on_inference = "probabilities",
        path_to_record_metrics = None
    )

    return seg_task

def getTrainer(hparams_series, version, run_info, log_dir, enable_progress_bar = False, enable_model_summary = False):

    """
    Get the Pytorch Lightning Trainer that runs the training.
    
    Arguments:
    - hparams_series (pd.Series): the series containing the hyperparameters for the version.
    - version (int): the version number
    - run_info (dict): the information used to create the checkpoint file name ('test-set', 'iteration')
    - log_dir (str): the directory to save the temporary files during training.
    
    Keywords:
    - enable_progress_bar (bool): if True, a progress bar appears during training.
    - enable_model_summary (bool): if True, the model summary is printed before training.
    
    Returns: 
    - trainer : the Trainer object.
    """
    
    val_metric = hparams_series.loc['val_metric']
    
    run_string = '_'.join([key + '-' + str(value) for key,value in run_info.items()])
    
    # Checkpoint callback: saves the model weights
    checkpoint_callback = ModelCheckpoint(
        monitor="val/"+val_metric,
        mode="max",
        dirpath=os.path.join(log_dir, "version_"+str(version), "checkpoints"),
        filename="best-checkpoint_"+run_string, # The name of the checkpoint file
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
    
def predictCraterEjecta(crater_FID, pred_df, model, data_module, trainer, predict_resolution = BASEMAP_RESOLUTION, reproject_basemap = False, full_size = False, num_threads = 1, fail_on_empty_df=True):
    
    """
    Create the global ejecta mask of one crater.
    
    Arguments:
    - crater_FID (int): the FID of the crater.
    - pred_df (pd.DataFrame): the tile positions dataframe for the evaluation. Derived from the Connection model results.
    - model (BinarySemanticSegmentationTask): the trained neural network.
    - data_module (EjectaAzDataModule): the data module.
    - trainer (Pytorch Lightning Trainer): the trainer to run the model.
    
    Keywords:
    - predict_resolution (float): the resolution of the output mask.
    - reproject_basemap (bool): if True, outputs the equidistant cylindrical version of the mask.
    - full_size (bool): if False, only the portion of the azimuthal mask (square, centered on the crater) containing ejecta is returned.
    - num_threads (int): the number of worker threads for reprojection.
    - fail_on_empty_df (bool): if True and `pred_df` is empty, throws an error. If False, returns an empty mask.
    
    Returns: 
    - mosaic_mask (xr.DataArray): azimuthal mask.
    - mask_for_basemap (xr.DataArray): cylindrical mask (if `reproject_basemap` is True)
    """
    
    if(len(pred_df)==0 and fail_on_empty_df):
        raise ValueError("`pred_df` is empty.")
        
    elif(len(pred_df)==0 and not(fail_on_empty_df)):
        mosaic_mask = data_module.get_empty_mask()
        
    else:
        data_module.predict_crater(crater_FID, pred_df) # Set the crater that will be predicted
        
        predicted_dl = trainer.predict(model, datamodule=data_module) # Predict the masks in the evaluation tiles

        mosaic_mask = data_module.fill_mask_mosaic(predicted_dl, check_crater_FID=crater_FID).astype(np.uint8) # 
    
    # Keep only the part of the mask actually containing ejecta
    
    mask_half_size = int(mosaic_mask.shape[1]/2)

    if(not(full_size)):
        one_indices_x = np.flatnonzero(np.any(mosaic_mask[0,:,:], axis=0))
        one_indices_y = np.flatnonzero(np.any(mosaic_mask[0,:,:], axis=1))
        
        if(len(one_indices_x)>0 and len(one_indices_y)>0):
            bound_min_x = np.amin(one_indices_x)
            bound_max_x = np.amax(one_indices_x)+1
            bound_min_y = np.amin(one_indices_y)
            bound_max_y = np.amax(one_indices_y)+1

            mask_half_size_new = np.amax(np.abs(np.array([bound_min_x, bound_max_x, bound_min_y, bound_max_y])-mask_half_size))
            
            mosaic_mask = mosaic_mask[:, mask_half_size-mask_half_size_new:mask_half_size+mask_half_size_new, mask_half_size-mask_half_size_new:mask_half_size+mask_half_size_new]
    
    # Reproject to cylindrical
    
    if(reproject_basemap):
    
        if(np.any(mosaic_mask)):
            mask_for_basemap = reprojectBasemap(mosaic_mask, num_threads = num_threads)
        else:
            mask_for_basemap, _ = emptyPosMask()
            
        return mosaic_mask, mask_for_basemap
        
    else:
        return mosaic_mask
        
def createEjectaMap(mask_paths = None, eval_positions_df = None, model_path = None, data_module = None, alpha = 0.5, save_masks_dir = None, num_threads = 1):

    """
    Create the color composite ejecta map and the individual ejecta masks.
    
    Keywords:
    - mask_paths (list): if set, the ejecta masks are not computed with the neural network, but imported from these paths.
                         If this is set, the other keywords do not need to be set.
                         
    - eval_positions_df (pd.DataFrame): the tile positions for all evaluation tiles of all craters, derived from the Connection model results.
    - model_path (str or list): the path of the model's checkpoint file. Multiple paths can also be given if an ensemble approach is to be used.
    - data_module (EjectaAzDataModule): the data module.
    - alpha (float): the opacity ejecta from different craters should have in overlap areas.
    - save_masks_dir (str): the directory to save individual ejecta masks. If None, the masks are not saved.
    - num_threads (int): the number of worker threads for reprojection.
    
    Returns: 
    - composite (xr.DataArray): the color composite ejecta map.
    """
    
    # Colors for some given craters. The others are chosen at random.
    color_dict = {0: 'lime',
                  3: 'cyan',
                  5: 'lightsteelblue',
                  8: 'orange',
                  10: 'magenta', 
                  12: 'white',
                  13: 'pink',
                  14: 'green',
                  17: 'gold',
                  24: 'darkgreen',
                  25: 'brown',
                  31: 'blue',
                  34: 'red',
                  26: 'purple',
                  45: 'darkorange',
                  57: 'wheat', 
                  58: 'olive',
                  65: 'snow',
                  84: 'grey',
                  90: 'midnightblue',
                  99: 'darksalmon', 
                  124: 'forestgreen'}
    
    
    color_dict_FIDs = np.array(list(color_dict.keys()), dtype=int)
    color_dict_colors = np.array(list(color_dict.values()))
    
    # Shuffle the color list to choose the other colors
    colorlist = np.loadtxt(os.path.join(code_dir,'colorlist.txt'), dtype=str)
    colorlist = colorlist[~np.isin(colorlist, color_dict_colors)]
    colorlist = colorlist[(np.char.find(colorlist,'grey')==-1) & (np.char.find(colorlist,'gray')==-1)] 
    rng = np.random.default_rng()
    rng.shuffle(colorlist)
    
    composite = np.zeros((3, Planet_Basemap.BASEMAP_HEIGHT, Planet_Basemap.BASEMAP_WIDTH), dtype=np.float32) # The composite: the 3-bands are necessary to render colors 
    
    predict_masks = eval_positions_df is not None and model_path is not None and data_module is not None

    if(not(predict_masks) and mask_paths is None):
        raise ValueError("If `mask_paths` is None, then `eval_positions_df`, `model_path` and `data_module` must be specified.")  
    
    if(mask_paths is not None):
        available_crater_FIDs = np.sort([int(os.path.basename(mask_path).removeprefix('cylindrical_mask_crater_').removesuffix('.tif')) for mask_path in mask_paths])
    else:
        mask_paths = []
        available_crater_FIDs = np.zeros(0, dtype=int)
    
    if(predict_masks): # If the masks must be plotted by the neural network
        if(isinstance(model_path, str)):
            model_paths = [model_path]
        elif(len(model_path)>0):
            model_paths = model_path

        predict_crater_FIDs = np.unique(eval_positions_df['crat_FID'].values)
        predict_crater_FIDs = predict_crater_FIDs[~np.isin(predict_crater_FIDs, available_crater_FIDs)]
        available_crater_FIDs = np.append(available_crater_FIDs, predict_crater_FIDs) # Crater FIDs to map
    
        trainer = Trainer( # Runs the neural network
            accelerator="gpu",
            devices = 1,
            strategy="auto",
            precision = "16-mixed",
            num_nodes = 1,
            logger = False,
            enable_progress_bar = False,
            enable_model_summary = False
        )
    
    time_start = time.time() 
    
    # Iterate over the craters
    
    for i in range(0,len(available_crater_FIDs)):
    
        crater_FID = available_crater_FIDs[i]
        color_str = color_dict[crater_FID] if crater_FID in color_dict_FIDs else colorlist[i % len(colorlist)]
        color = mcolors.to_rgb(color_str)
        
        load_mask_path = mask_paths[i] if i < len(mask_paths) else None
        
        if(load_mask_path is not None and os.path.exists(load_mask_path)): # Load mask if provided in the `mask_paths`
        
            mask_for_basemap = rioxarray.open_rasterio(load_mask_path).astype(np.uint8)
            mask_for_basemap = fillCyl(mask_for_basemap) # Full size of the basemap 
            
        elif(predict_masks): # Otherwise, predict it
        
            eval_df = eval_positions_df.loc[eval_positions_df['crat_FID']==crater_FID,:]
            eval_df.set_index('eval_pos_index', inplace=True)
            
            if(len(eval_df)==0):
                warnings.warn(f"There are no positions for crater {crater_FID}.")
                
            # Predict the mask using each model in `model_paths`, then average the masks.
                
            for j, ckpt_path in enumerate(model_paths):
                seg_task = BinarySemanticSegmentationTask.load_from_checkpoint(ckpt_path)
                    
                _, mask_for_basemap_ckpt = predictCraterEjecta(crater_FID, eval_df, seg_task, data_module, trainer, reproject_basemap = True, full_size = False, num_threads = num_threads, fail_on_empty_df = False) # Create the global mask
                
                if(j==0):
                    mask_for_basemap = mask_for_basemap_ckpt.copy()
                else:
                    mask_for_basemap = mask_for_basemap + mask_for_basemap_ckpt.values
                    
            mask_for_basemap[0,:,:] = np.floor(mask_for_basemap[0,:,:].values*1.0/len(model_paths)+0.5).astype(np.uint8)
    
        else:
            raise RuntimeError("The mask doesn't exist and `eval_positions_df`, `model_path` and `data_module` are not specified.")
    
        mask_values = mask_for_basemap.values[0,:,:]
        
        mask_empty = not(np.all(np.isin([0,1], mask_values)))
        if(mask_empty):
            warnings.warn(f"The crater {crater_FID} basemap is empty, no ejecta have been segmented.")
        
        if(i==0):
            y_coords = mask_for_basemap.coords['y'].values
            x_coords = mask_for_basemap.coords['x'].values
        
        # Add crater mask to color composite
        
        composite_mask_1 = (np.sum(composite, axis=0)>0)*mask_values
        composite_0 = np.sum(composite, axis=0)==0

        for c in range(3):
            color255 = color[c] * 255
            comp = composite[c]
            composite[c] = ne.evaluate(
        "comp * (1 - mask_values) + "
        "mask_values * composite_0 * color255 + "
        "mask_values * composite_mask_1 * color255 * alpha + "
        "comp * composite_mask_1 * (1 - alpha)"
    )
                         
        # Save individual masks
                         
        if(save_masks_dir is not None and os.path.isdir(save_masks_dir)):
        
            if(not(mask_empty)):
            
                # Shrink the mask to the area containing ejecta
            
                one_indices_x = np.flatnonzero(np.any(mask_values, axis=0))
                one_indices_y = np.flatnonzero(np.any(mask_values, axis=1))
            
                bound_min_x = np.amin(one_indices_x)
                bound_max_x = np.amax(one_indices_x)+1
                bound_min_y = np.amin(one_indices_y)
                bound_max_y = np.amax(one_indices_y)+1
                
                mask_for_basemap = mask_for_basemap[:, bound_min_y:bound_max_y, bound_min_x:bound_max_x]
            
            mask_for_basemap.rio.to_raster(os.path.join(save_masks_dir, 'cylindrical_mask_crater_'+str(crater_FID)+'.tif')) # Save mask

        del mask_for_basemap
        del mask_values      
            
        gc.collect()
        print(crater_FID, color_str, time.time()-time_start, virtual_memory().percent, cuda_memory_reserved()/1024**2)
        time_start = time.time()
    
    composite = np.clip(composite, 0, 255).astype(np.uint8)
    alpha_band = np.zeros((1, Planet_Basemap.BASEMAP_HEIGHT, Planet_Basemap.BASEMAP_WIDTH), dtype=np.uint8)
    
    # Composite transparent in the absence of ejecta
    composite_nonzero = np.nonzero(np.sum(composite,axis=0))
    alpha_band[:,composite_nonzero[0], composite_nonzero[1]] = 255
    
    # Create composite DataArray
    
    composite = xr.DataArray(data=np.concatenate((composite,alpha_band), axis=0), 
                             dims=('band','y','x'), 
                             coords = (['red','green','blue', 'alpha'], y_coords, x_coords))
    
    composite.rio.write_crs(Planet_CRS.CYL_CRS, inplace=True)
    
    return composite
    
def evaluateEjectaMap(conn_version, 
                      test_set, 
                      ejc_model_path,
                      ejc_norm_path, 
                      ejc_version_series,
                      conn_result_dir,
                      conn_data_root,  
                      ejc_data_root,
                      cyl_data_root,
                      return_conn_matrix = False,
                      crater_FIDs = None, 
                      num_workers = 8,
                      accelerator = "gpu"):
                      
    """
    Test the output of the neural networks after training by creating a global ejecta mask and comparing it with the manual mask.
    To save time, only the areas intersecting with a TVT tile are actually segmented.
    
    Arguments:
    - conn_version (int): the version of the connection model.
    - test_set (int): the train-test split.
    - ejc_model_path (str): the path of the segmentation model's checkpoint file.
    - ejc_norm_path (str): the path of the segmentation model's normalization file.
    - ejc_version_series (pd.Series): the segmentation model hyperparameters.
    - conn_result_dir (str): the results directory of the connection model.
    - conn_data_root (str): the data directory for the connection model.
    - ejc_data_root (str): the data directory for the segmentation model.
    - cyl_data_root (str): the data directory for the cylindrical, manual masks.
    
    Keywords:
                         
    - return_conn_matrix (bool): if True, return a matrix indicating whether, for each crater, a given cylindrical training tile is connected. Default: False
    - crater_FIDs (np.array): if the evaluation should be carried out only on some craters, specify them. Otherwise, all the training craters are used.
    - num_workers: the processes for data loading and threads for reprojection.
    - accelerator: "gpu" or "cpu".
    
    Returns: 
    - metrics_dict (dict): a dictionary containing the performance metrics
    - connection_matrix_df (pd.DataFrame): the connection matrix, if `return_conn_matrix` is True.
    """
                      
    # Get crater FIDs
                      
    if(crater_FIDs is None):
        fresh_craters_gdf = gpd.read_file(os.path.join(ejc_data_root, 'fresh_craters.shp'))
        fresh_craters_gdf = fresh_craters_gdf.loc[fresh_craters_gdf['train']==1,:]    
        crater_FIDs = np.sort(fresh_craters_gdf['crat_FID'].values)
    else:
        crater_FIDs = np.unique(np.array(crater_FIDs).astype(int))
    
    # Get segmentation model normalization
    ejc_norm = np.load(ejc_norm_path)
    ejc_img_norm = (ejc_norm[0,0:3], ejc_norm[1,0:3])
    ejc_dist_norm = (ejc_norm[0,3], ejc_norm[1,3])
    
    conn_version_dir = os.path.join(conn_result_dir, 'version_'+str(conn_version))

    # Get EjectaAzDataModule for prediciton
    ejc_data_module = getDataModule(ejc_version_series, ejc_data_root, predict=True, img_norm = ejc_img_norm, dist_norm = ejc_dist_norm, num_workers = num_workers)
    
    # Trainer to run the neural network
    trainer = Trainer(
        accelerator = accelerator,
        devices = 1,
        strategy="auto",
        precision = "16-mixed",
        num_nodes = 1,
        logger = False,
        enable_progress_bar = False,
        enable_model_summary = False
    )
    
    seg_task = BinarySemanticSegmentationTask.load_from_checkpoint(ejc_model_path) # Load model from checkpoint
    
    # All training tile positions
    all_positions = pd.read_csv(os.path.join(conn_data_root, "az_positions_all.csv")) 
    
    all_positions = all_positions.loc[(all_positions['test_'+str(test_set)]>=0) & np.isin(all_positions['crat_FID'], crater_FIDs),:]
    all_connected_len = 0
    test_all_connected_len = 0
    all_values_len = 0
    test_all_values_len = 0
        
    # Retrieve the coordinates for the tile positions        
    for i, crater_FID in enumerate(crater_FIDs):
    
        valid_positions = all_positions.loc[all_positions['crat_FID']==crater_FID,:]
        pos_indices = valid_positions['pos_index'].values
        test_pos_indices = pos_indices[valid_positions['test_'+str(test_set)]==1]
         
        # The coordinates are stored in CSV files in `cyl_data_root` 
        cyl_positions = pd.read_csv(os.path.join(cyl_data_root, 'positions_crater_'+str(crater_FID)+'.csv'), index_col='pos_index')
        
        all_values_len += np.sum(np.prod(cyl_positions.loc[pos_indices, ['height', 'width']], axis=1))
        test_all_values_len += np.sum(np.prod(cyl_positions.loc[test_pos_indices, ['height', 'width']], axis=1))
        all_connected_len += len(pos_indices)
        test_all_connected_len += len(test_pos_indices)

    # Arrays for the pixel-wise metrics: the ejecta maps will be flattened and compared

    all_true_values = np.zeros(all_values_len, dtype=bool) # True pixel values for TVT tiles
    test_all_true_values = np.zeros(test_all_values_len, dtype=bool) # True pixel values for Test tiles
    all_pred_values = np.zeros(all_values_len, dtype=bool) # Predicted pixel values for TVT tiles
    test_all_pred_values = np.zeros(test_all_values_len, dtype=bool) # Predicted pixel values for Test tiles

    # Arrays for the tile-wise metrics, set to 1 or 0 for each tile 

    all_true_connected = np.zeros(all_connected_len, dtype=bool) # True tile connection value, TVT tiles
    test_all_true_connected = np.zeros(test_all_connected_len, dtype=bool) # True tile connection value, Test tiles
    all_pred_connected = np.zeros(all_connected_len, dtype=bool) # Predicted tile connection value, TVT tiles
    test_all_pred_connected = np.zeros(test_all_connected_len, dtype=bool) # Predicted tile connection value, Test tiles
    
    I_values = 0
    test_I_values = 0
    I_conn = 0
    test_I_conn = 0
    
    connection_matrix_df = pd.DataFrame(index=np.zeros(0,dtype=int), columns = crater_FIDs)
    
    # Iterate over craters
    
    for i in range(0,len(crater_FIDs)):
    
        time_start = time.time()
        
        # Get the connected evaluation tiles, use only those that intersect a training tile
        
        crater_FID = crater_FIDs[i]
        pred_df = pd.read_csv(os.path.join(conn_version_dir, 'conn_positions', 'conn-pos_test-set-'+str(test_set)+'_crater-'+str(crater_FID)+'.csv'), index_col = 'eval_pos_index')
        pred_len0 = len(pred_df)
        pred_df = pred_df.loc[pred_df['train_intersect']==1,:]
        pred_df['crat_FID'] = crater_FID
        
        # Create the predicted ejecta mask
        
        mosaic_mask, pred_mask = predictCraterEjecta(crater_FID, pred_df, seg_task, ejc_data_module, trainer, reproject_basemap = True, full_size = False, num_threads = num_workers, fail_on_empty_df = False)
        
        # Get the training tiles and the manual mask
        
        valid_positions = all_positions.loc[all_positions['crat_FID']==crater_FID,:]
        pos_indices = valid_positions['pos_index'].values
        test_pos_indices = pos_indices[valid_positions['test_'+str(test_set)]==1]
        
        cyl_positions = pd.read_csv(os.path.join(cyl_data_root, 'positions_crater_'+str(crater_FID)+'.csv'), index_col='pos_index') # Training tiles
        true_mask = rioxarray.open_rasterio(os.path.join(cyl_data_root, 'mask_crater_'+str(crater_FID)+'.tif')).astype(np.uint8) # Manual mask

        if(pred_mask.shape!=true_mask.shape):
            raise RuntimeError(f"The shape of the masks should be the same, instead we have {pred_mask.shape} and {true_mask.shape}.")
            
        # For each training tile, compare the manual (true) and predicted mask
            
        for pos_index in pos_indices:

            x_pos = cyl_positions.loc[pos_index, 'x']
            y_pos = cyl_positions.loc[pos_index, 'y']
            height = cyl_positions.loc[pos_index, 'height']
            width = cyl_positions.loc[pos_index, 'width']

            true_values = true_mask[0, y_pos:y_pos+height, x_pos:x_pos+width].values.flatten()
            pred_values = pred_mask[0, y_pos:y_pos+height, x_pos:x_pos+width].values.flatten()

            true_connected = int(np.any(true_values == 1))
            pred_connected = int(np.any(pred_values == 1))
            
            all_true_values[I_values:I_values+height*width] = true_values
            all_pred_values[I_values:I_values+height*width] = pred_values
            all_true_connected[I_conn] = true_connected
            all_pred_connected[I_conn] = pred_connected
            
            connection_matrix_df.loc[pos_index, crater_FID] = int(true_connected + 2*pred_connected)
            
            I_values += height*width
            I_conn += 1 
            
            if(pos_index in test_pos_indices): # If the tile is in the Test set
            
                test_all_true_values[test_I_values:test_I_values+height*width] = true_values
                test_all_pred_values[test_I_values:test_I_values+height*width] = pred_values
                test_all_true_connected[test_I_conn] = true_connected
                test_all_pred_connected[test_I_conn] = pred_connected
                
                test_I_values += height*width
                test_I_conn += 1 
                            
        print(crater_FID, len(pred_df), pred_len0, time.time()-time_start)
        time_start = time.time()
        
        if(len(pos_indices)>0):
            del true_values
            del pred_values
        del true_mask
        del pred_mask
        del mosaic_mask
        gc.collect()

    # Dictionary containing all the metrics

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
        
    if(return_conn_matrix):
        
        return metrics_dict, connection_matrix_df
        
    else:

        return metrics_dict

# Function to add versions with new hyperparameters to the list. Not used because it is easier to do it manually.     
'''        
def addVersionHparams(input_dict, version_df, default_dict, data_root, previous_values = None, drop_duplicate = False, set_num = 100):

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
