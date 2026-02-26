"""
Filename: connection_train.py
Author: Michele Lissoni
Date: 2026-02-24
"""

"""

Run the Connection model (EJCONN) training

Arguments:

- index: index of the process running (necessary for embarassingly training runs)
- log_dir: log directory
- data_root: data directory
- num_workers: number of subprocesses (recommended: at least 8)

"""

import os
import sys
code_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, code_dir)
import numpy as np
import pandas as pd
import glob
import json
import argparse
import time
import warnings
import torch

from lightning.pytorch import seed_everything as seedEverything

from connection_data_preparation import getDataModule, getSegTask, getTrainer

from terratorch_custom_tasks import BinarySemanticSegmentationTask

def convert_numpy_ints(x):
    return int(x) if isinstance(x, (np.integer,)) else x

if __name__ == "__main__":

    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(device_id)
    else:
        raise RuntimeError("This training should take place on the GPU.")

    time_start = time.time()

    # In-line arguments

    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--num_workers", type=int, required=True)
    args = parser.parse_args()
    
    training_index = args.index
    log_dir = args.log_dir
    data_root = args.data_root
    num_workers = args.num_workers
    
    repeated_versions = np.loadtxt(os.path.join(log_dir, 'repeated_versions_to_run.txt'), dtype=int) # Retrieve parameters of the training run
    if(len(repeated_versions.shape)==2):
        version = repeated_versions[training_index,0] # Version number (hyperparameters)
        test_set_number = repeated_versions[training_index,1] # TV-Test split
        iteration_num = repeated_versions[training_index,2] # Iteration (corresponds to a random Train-Val split)
        seed_index = np.flatnonzero(np.flatnonzero(repeated_versions[:,0]==version)==training_index)[0] # Index for the random seed
    elif(len(repeated_versions.shape)==1):
        version = repeated_versions[0]
        test_set_number = repeated_versions[1]
        iteration_num = repeated_versions[2]
        seed_index = 0
    else:
        raise RuntimeError(f"The repeated versions array is {len(repeated_versions.shape)}-dimensional, when it should be 2-dimensional.")
        
    version_log_dir = os.path.join(log_dir, 'version_'+str(version))
        
    seed = np.load(os.path.join(version_log_dir, 'seeds.npy'))[seed_index]
    seedEverything(seed, workers=True)
    
    # Retrieve hyperparameters
    
    version_series = pd.read_csv(os.path.join(code_dir, 'connection_version_hparams.csv'), index_col = 'version', dtype_backend="numpy_nullable").loc[version,:]
    version_series = version_series.map(convert_numpy_ints)
    version_series.loc[version_series.isna()] = None
    
    data_module = getDataModule(version_series, data_root, test_set_number, num_workers = num_workers) # Data module (manages the data)

    # Save normalization
    norm_arr = data_module.get_norms()
    norm_file = f"norm_test-set-{test_set_number}_iteration-{iteration_num}.npy"
    np.save(os.path.join(version_log_dir, 'norms', norm_file), norm_arr)
    
    seg_task = getSegTask(version_series) # Get neural network
    # Trainer: runs the model
    trainer = getTrainer(version_series, version, test_set_number, iteration_num, log_dir, enable_progress_bar = True, enable_model_summary = True) 
    
    ### TRAIN ###
    
    data_module.setup('fit') # Setup the training data
    
    time_beginning = time.time()-time_start
    time_start2 = time.time()
    
    trainer.fit(seg_task, datamodule=data_module) # Train the model
    epochs_run = trainer.current_epoch
    
    time_train = time.time()-time_start2
    time_start2 = time.time()
    
    ### TEST ###
    
    best_model_file = f"best-checkpoint_test-set-{test_set_number}_iteration-{iteration_num}.ckpt" # Weights saved in this checkpoint file
    seg_task_best = BinarySemanticSegmentationTask.load_from_checkpoint(os.path.join(version_log_dir, "checkpoints", best_model_file))
    
    time_test_beginning = time.time()-time_start2
    time_start2 = time.time()
    
    if(len(data_module.test_df)==0):
    
        test_dict = {"test/loss": np.nan, 
                     "test/Accuracy": np.nan, 
                     "test/F1_Score": np.nan, 
                     "test/IoU": np.nan, 
                     "test/Precision": np.nan,
                     "test/Recall": np.nan}
                     
        warnings.warn("No test set, skipping test.")
        
    else:
    
        test_results = trainer.test(seg_task_best, datamodule = data_module, verbose = False) # Test the model
        test_dict = test_results[0] # Test metrics (not used for model evaluation)
    
    time_test = time.time()-time_start2
    time_start2 = time.time()
    
    # Run info
    test_dict['epochs_run'] = epochs_run
    test_dict['runtime'] = time.time() - time_start
    test_dict['time_beginning'] = time_beginning
    test_dict['time_train'] = time_train
    test_dict['time_test_beginning'] = time_test_beginning
    test_dict['time_test'] = time_test
    test_dict['gpu_name'] = gpu_name
    
    test_file = os.path.join(version_log_dir, "tests", f"results_test-set-{test_set_number}_iteration-{iteration_num}.json")
    
    with open(test_file, "w") as outfile:
        json.dump(test_dict, outfile)
        
    ### PREDICT ###
    
    # Predict the connection value of all the eval tiles
        
    eval_position_filelist = glob.glob(os.path.join(data_root, 'eval_positions', 'eval_positions_crater_*.csv')) # Eval positions for the various craters
    predict_crater_FIDs = np.array([int(os.path.basename(filename).removeprefix('eval_positions_crater_').removesuffix('.csv')) for filename in eval_position_filelist], dtype=int) # Available craters
    
    I = 0
    for predict_crater_FID in predict_crater_FIDs:
    
        valid_crat_FID = data_module.predict_crater(predict_crater_FID) # Set crater to predict
        
        if(not(valid_crat_FID)):
            continue
            
        pred_dl = trainer.predict(seg_task_best, datamodule = data_module) # Predict

        pred_df = data_module.predict_positions(pred_dl, predict_tt = False) # Store the results
        
        pred_df = pred_df.loc[:,['eval_pos_index']]
        pred_df['crat_FID'] = predict_crater_FID
        
        if(I==0):
            predicted_positions = pred_df.copy()
        else:
            predicted_positions = pd.concat((predicted_positions, pred_df), axis=0, ignore_index=True)
            
        I+=1
    
    # Save connected tiles to log dir (they will later be combined)
    predicted_positions.to_csv(os.path.join(version_log_dir, "positions", f"eval-pos_test-set-{test_set_number}_iteration-{iteration_num}.csv"), index=False)
