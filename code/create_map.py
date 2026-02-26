"""
Filename: create_map.py
Author: Michele Lissoni
Date: 2026-02-24
"""

"""

Create the global ejecta map: the color composite and the individual ejecta masks.

Arguments:
- load_masks_dir: if set, the ejecta masks are not computed with the neural network, but imported from this directory.

- ejc_vers: the Segmentation model version
- conn_vers: the Connection model version
- test_set: the TV-Test split
- crat_FIDs: if set, only these craters are mapped
- use_eval_crats: if 1, the non-TVT craters are also mapped
- ejc_data_root: the Segmentation data directory
- conn_data_root: the Connection data directory
- ejc_result_dir: the Segmentation result directory
- use_iter_ckpt: if 1, and the iteration checkpoints are saved in the results directory for the chosen Segmentation version (this operation needs to be done manually), 
                 then an ensemble approach with these models is used.
- conn_result_dir: the Connection results directory
- save_masks: if 1, the individual ejecta masks are saved
- save_masks_dir: the directory where to save the individual ejecta masks
- num_workers: the number of subprocesses
- ignore_connections: if 1, the Segmentation model is applied to all the tiles of a crater.

"""

import os
import sys
code_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, code_dir)
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import argparse

import ejecta_data_preparation as edp
import connection_data_preparation as cdp
from terratorch_custom_tasks import BinarySemanticSegmentationTask
from lightning.pytorch import Trainer

import warnings

if __name__ == "__main__":

    # In-line arguments

    parser = argparse.ArgumentParser()
    parser.add_argument("--load_masks_dir", type=str, required=False)
    parser.add_argument("--ejc_vers", type=int, required=False)
    parser.add_argument("--conn_vers", type=int, required=False)
    parser.add_argument("--test_set", type=int, required=False)
    parser.add_argument("--crat_FIDs", nargs='+', required=False)
    parser.add_argument("--use_eval_crats", type=int, required=False)
    parser.add_argument("--ejc_data_root", type=str, required=True)
    parser.add_argument("--conn_data_root", type=str, required=False)
    parser.add_argument("--ejc_result_dir", type=str, required=False)
    parser.add_argument("--use_iter_ckpt", type=int, required=False)
    parser.add_argument("--conn_result_dir", type=str, required=False)
    parser.add_argument("--save_masks", type=int, required=False)
    parser.add_argument("--save_masks_dir", type=str, required=False)
    parser.add_argument("--num_workers", type=int, required=False)
    parser.add_argument("--ignore_connections", action = "store_true")
    args = parser.parse_args()
    
    print(args)

    num_workers = args.num_workers
    
    predict_masks = args.ejc_vers is not None and args.conn_vers is not None and args.test_set is not None and args.load_masks_dir is None
    if(args.load_masks_dir is None and not(predict_masks)):
        raise ValueError("Either specify `--load_masks_dir` or `--ejc_vers`, `--conn_vers` and `--test_set`.")
        
    ejc_data_root = args.ejc_data_root
    
    # Retrieve crater FIDs
    
    if(args.crat_FIDs is None):
        fresh_craters_gdf = gpd.read_file(os.path.join(ejc_data_root, 'fresh_craters.shp'))
        if(not(args.use_eval_crats==1)):
            fresh_craters_gdf = fresh_craters_gdf.loc[fresh_craters_gdf['train']==1,:]    
        crater_FIDs = np.sort(fresh_craters_gdf['crat_FID'].values)
    else:
        crater_FIDs = np.unique(np.array(args.crat_FIDs).astype(int))
        
    if(predict_masks): # If masks must be predicted and not loaded...
    
        conn_data_root = args.conn_data_root
        ejc_result_dir = args.ejc_result_dir
        conn_result_dir = args.conn_result_dir
        use_iter_ckpt = args.use_iter_ckpt
        
        if(conn_data_root is None or ejc_result_dir is None or conn_result_dir is None or use_iter_ckpt is None):
            raise ValueError("Specify `--conn_data_root`, `ejc_result_dir`, `conn_result_dir`, `use_iter_ckpt`.")
            
        save_masks_directory = args.save_masks_dir if args.save_masks==1 else None
        if(save_masks_directory is None and args.save_masks==1):
            raise ValueError("Specify `--save_masks_dir`.")
    
        load_mask_paths = None
        
        ejc_version = args.ejc_vers
        ejc_version_dir = os.path.join(ejc_result_dir, 'version_'+str(ejc_version))
        
        test_set = args.test_set
        test_set_str = str(test_set)

        if(args.use_iter_ckpt==1): # Retrieve iteration checkpoints
            available_ckpt = glob.glob(os.path.join(ejc_version_dir, 'best-checkpoint_test-set-'+test_set_str+'_iteration-*.ckpt'))        
        else:
            available_ckpt = glob.glob(os.path.join(ejc_version_dir, 'best-checkpoint_test-set-'+test_set_str+'.ckpt')) # Use best test checkpoint
        
        if(len(available_ckpt)==0):
            raise ValueError("The specified test sets are not available.")
            
        ejc_model_path = available_ckpt
        
        model_path_split = os.path.splitext(os.path.basename(available_ckpt[0]))[0].split('_')
        conn_version = args.conn_vers
        
        print('Ejecta version:', ejc_version, 'Connection version:', conn_version, 'Test set:', test_set)
        
        ejc_norm = np.load(os.path.join(ejc_version_dir, 'best-norm_test-set-'+str(test_set)+'.npy')) # Load normalization
        ejc_img_norm = (ejc_norm[0,0:3], ejc_norm[1,0:3])
        ejc_dist_norm = (ejc_norm[0,3], ejc_norm[1,3])
        
        conn_version_dir = os.path.join(conn_result_dir, 'version_'+str(conn_version))
        
        # Get Segmentation hyperparameters        
        ejc_version_series = pd.read_csv(os.path.join(code_dir, 'ejecta_version_hparams.csv'), index_col = 'version').loc[ejc_version,:]
        
        # Get Segmentation data module
        ejc_data_module = edp.getDataModule(ejc_version_series, ejc_data_root, predict=True, only_train = not(args.use_eval_crats==1), img_norm = ejc_img_norm, dist_norm = ejc_dist_norm, num_workers = num_workers)
        
        trainer = Trainer( # Trainer, runs the Segmentation model
            accelerator="gpu",
            devices = 1,
            strategy="auto",
            precision = "16-mixed",
            num_nodes = 1,
            logger = False,
            enable_progress_bar = False,
            enable_model_summary = False
        )
        
        if(args.use_eval_crats==1): # If non-TVT craters are used, the connected positions might not have been calculated yet
        
            conn_version_dir = os.path.join(conn_result_dir, 'version_'+str(conn_version))
        
            conn_ckpts = glob.glob(os.path.join(conn_version_dir, 'best-checkpoint_test-set-'+test_set_str+'_iteration-*.ckpt')) # Get Connection iteration checkpoints 
                                                                                                                                 # (will fail if these are not saved in
                                                                                                                                 # the results folder)
            
            conn_norm = np.load(os.path.join(conn_version_dir, 'best-norm_test-set-'+str(test_set)+'.npy')) # Get normalization (this is the same for every iteration for a given
                                                                                                            # TV-Test split, so only one is needed)
            conn_img_norm = (conn_norm[0,0:3], conn_norm[1,0:3])
            conn_att_norm = (conn_norm[0,3], conn_norm[1,3])
            
            conn_version_series = pd.read_csv(os.path.join(code_dir, 'connection_version_hparams.csv'), index_col = 'version').loc[conn_version,:] # Connection model hyperparameters
            
            iter_threshold = conn_version_series.loc['iter_threshold']
            
            conn_data_module = cdp.getDataModule(conn_version_series, conn_data_root, test_set_number = test_set, img_norm = conn_img_norm, att_norm = conn_att_norm, num_workers = num_workers) # Connection data module, handles the data
        
        # Iterate over craters
        
        for i in range(0,len(crater_FIDs)):
            
            crater_FID = crater_FIDs[i] 
            
            conn_pos_path = os.path.join(conn_version_dir, 'conn_positions', 'conn-pos_test-set-'+str(test_set)+'_crater-'+str(crater_FID)+'.csv')
            
            if(args.ignore_connections):
                pred_df = pd.read_csv(os.path.join(conn_data_root, 'eval_positions', 'eval_positions_crater_'+str(crater_FID)+'.csv'))
            else:
            
                # If the positions for a given crater have not yet been computed, do so now
                if(not(os.path.exists(conn_pos_path))):
                    pos_df = pd.read_csv(os.path.join(conn_data_root, 'eval_positions', 'eval_positions_crater_'+str(crater_FID)+'.csv'), index_col='eval_pos_index')
                    pos_df['iters'] = 0
                        
                    # Merge results of different iterations                        
                    for conn_ckpt in conn_ckpts:
                        conn_seg_task = BinarySemanticSegmentationTask.load_from_checkpoint(conn_ckpt)
                        
                        conn_data_module.predict_crater(crater_FID, only_tt = False)

                        pred_dl = trainer.predict(conn_seg_task, datamodule = conn_data_module)

                        prediction_df = conn_data_module.predict_positions(pred_dl, predict_tt = False)
                            
                        pos_df.loc[prediction_df['eval_pos_index'].values, 'iters'] += 1
                        
                    pred_df = pos_df.loc[pos_df['iters']*1.0/len(conn_ckpts) >= iter_threshold, :]
                    pred_df = pred_df.drop('iters', axis=1)
                    
                    pred_df.to_csv(conn_pos_path, index=True, index_label='eval_pos_index') # Save the connected positions, so that the operation won't need to be repeated for this crater
                
                    print(crater_FID, len(pred_df))
                
                # Get connected tile positions for the crater                
                pred_df = pd.read_csv(conn_pos_path)
                
            pred_df['crat_FID'] = crater_FID
           
            if(i==0):
                eval_positions_df = pred_df.copy()
            else:
                eval_positions_df = pd.concat((eval_positions_df, pred_df), ignore_index=True)
                
            print(crater_FID, len(eval_positions_df), np.unique(eval_positions_df['crat_FID']))
        
        save_map_path = os.path.join(ejc_version_dir, 'ejecta-map_ejc-version-'+str(ejc_version)+'_conn-version-'+str(conn_version)+'_test-set-'+str(test_set)+'.tif')
        
    else: # If the individual ejecta masks already exist and just need to be loaded...    
    
        load_mask_paths = []
        for crat_FID in crater_FIDs:
            mask_path = os.path.join(args.load_masks_dir, 'cylindrical_mask_crater_'+str(crat_FID)+'.tif')
            if(not(os.path.exists(mask_path))):
                warnings.warn("The mask for crater "+str(crat_FID)+" is not in the directory. The map will ignore this crater.")
            else:
                load_mask_paths.append(mask_path)
                
        load_mask_paths = np.array(load_mask_paths, dtype=str)
        save_map_path = os.path.join(args.load_masks_dir, 'ejecta-map.tif')
        eval_positions_df = None
        ejc_model_path = None
        ejc_data_module = None
        
        save_masks_directory = None
    
    ejecta_map = edp.createEjectaMap(mask_paths = load_mask_paths, eval_positions_df = eval_positions_df, model_path = ejc_model_path, data_module = ejc_data_module, save_masks_dir = save_masks_directory, num_threads = num_workers) # Create color composite and individual masks
    
    ejecta_map.rio.to_raster(save_map_path) # Save color composite: to the version results folder if it is being run for the first time,
                                            # to the folder of the individual masks if these already exist and are being loaded.
    
