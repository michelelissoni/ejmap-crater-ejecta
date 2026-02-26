"""
Filename: ejecta_train.py
Author: Michele Lissoni
Date: 2026-02-24
"""

"""

Run the Segmentation model (EJSEG) training

Arguments:

- index: index of the process running (necessary for embarassingly training runs)
- log_dir: log directory
- data_root: data directory
- conn_data_root: data directory of the Connection model
- num_workers: number of subprocesses (recommended: at least 8)
- eval_conn_vers: version of the Connection model to use for evaluation
- cyl_data_root: data directory of the manual masks (cylindrical projection)
- conn_result_dir: directory with the results of the Connection models

"""

import os
import sys
code_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, code_dir)
import numpy as np
import pandas as pd
import json
import argparse
import time
import torch

import warnings

from lightning.pytorch import seed_everything as seedEverything

from ejecta_data_preparation import getDataModule, getSegTask, getTrainer, evaluateEjectaMap

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
    parser.add_argument("--conn_data_root", type=str, required=True)
    parser.add_argument("--num_workers", type=int, required=True)
    parser.add_argument("--eval_conn_vers", type=int, required=False)
    parser.add_argument("--cyl_data_root", type=str, required=False)
    parser.add_argument("--conn_result_dir", type=str, required=False)
    args = parser.parse_args()
    
    training_index = args.index
    log_dir = args.log_dir
    conn_data_root = args.conn_data_root
    data_root = args.data_root
    num_workers = args.num_workers
    
    evaluate_map = args.eval_conn_vers is not None and args.cyl_data_root is not None and args.conn_result_dir is not None
    
    if(not(evaluate_map) and (args.eval_conn_vers is not None or args.cyl_data_root is not None or args.conn_result_dir is not None)):
        warnings.warn("If `--eval_conn_vers`, `--cyl_data_root` and `--conn_result_dir` are not all specified, map evaluation will not occur.")
    
    repeated_versions = np.loadtxt(os.path.join(log_dir, 'repeated_versions_to_run.txt'), dtype=int) # Retrieve parameters of the training run
    if(len(repeated_versions.shape)==2):
        version = repeated_versions[training_index,0] # Version number (hyperparameters)
        test_set = repeated_versions[training_index,1] # TV-Test split
        iteration_num = repeated_versions[training_index,2] # Iteration (corresponds to a random Train-Val split)
        seed_index = np.flatnonzero(np.flatnonzero(repeated_versions[:,0]==version)==training_index)[0] # Index for the random seed
    elif(len(repeated_versions.shape)==1):
        version = repeated_versions[0]
        test_set = repeated_versions[1]
        iteration_num = repeated_versions[2]
        seed_index = 0
    else:
        raise RuntimeError(f"The repeated versions array is {len(repeated_versions.shape)}-dimensional, when it should be 2-dimensional.")
        
    run_info = {'test-set': test_set,
                'iteration': iteration_num}
                
    run_string = '_'.join([key + '-' + str(value) for key,value in run_info.items()]) 
        
    version_log_dir = os.path.join(log_dir, 'version_'+str(version))
        
    seed = np.load(os.path.join(version_log_dir,'seeds.npy'))[seed_index]
    seedEverything(seed, workers=True)
    
    # Retrieve hyperparameters
    
    version_series = pd.read_csv(os.path.join(code_dir, 'ejecta_version_hparams.csv'), index_col = 'version', dtype_backend="numpy_nullable").loc[version,:]
    version_series = version_series.map(convert_numpy_ints)
    version_series.loc[version_series.isna()] = None
    
    # Retrieve tile positions
    
    all_positions_df = pd.read_csv(os.path.join(conn_data_root, 'az_positions_all.csv'))
    test_cols = all_positions_df.columns.values[np.char.startswith(all_positions_df.columns.values.astype(str), 'test_')]
    test_cols = np.delete(test_cols, np.flatnonzero(test_cols=='test_'+str(test_set)))
    all_positions_df.drop(test_cols, axis=1, inplace=True)
    
    train_df = all_positions_df.loc[(all_positions_df['test_'+str(test_set)]==0) & (all_positions_df['num_1']>0),:] # TV set
    test_df = all_positions_df.loc[(all_positions_df['test_'+str(test_set)]==1) & (all_positions_df['num_1']>0),:] # Test set
    
    print(len(all_positions_df), np.sum(all_positions_df['test_'+str(test_set)]==0), np.sum((all_positions_df['test_'+str(test_set)]==0) & (all_positions_df['num_1']>0)), np.sum(all_positions_df['test_'+str(test_set)]==1), np.sum((all_positions_df['test_'+str(test_set)]==1) & (all_positions_df['num_1']>0)))
    
    if(np.any(train_df['test_'+str(test_set)]!=0) or np.any(test_df['test_'+str(test_set)]!=1)):
        raise RuntimeError(f"The test and train positions do not all conform to the chosen test set: {test_set}.") 
    
    data_module = getDataModule(version_series, data_root, train_df = train_df, test_df = test_df, num_workers = num_workers) # Data module (manages the data)
    
    # Save normalization
    norm_arr = data_module.get_norms()
    norm_path = os.path.join(version_log_dir, 'norms', "norm_"+run_string+".npy")
    np.save(norm_path, norm_arr)
    
    seg_task = getSegTask(version_series) # Get neural network
    # Trainer: runs the model
    trainer = getTrainer(version_series, version, run_info, log_dir, enable_progress_bar = True, enable_model_summary = True)
    
    ### TRAIN ###
    
    time_beginning = time.time()-time_start
    time_start2 = time.time()
    
    trainer.fit(seg_task, datamodule=data_module) # Train the model
    epochs_run = trainer.current_epoch
    
    time_train = time.time()-time_start2
    time_start2 = time.time()
    
    if torch.cuda.is_available():
        max_memory = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Peak GPU memory: {max_memory:.2f} GB")    
    
    ### TEST ###
    
    best_model_path = os.path.join(version_log_dir, "checkpoints", "best-checkpoint_"+run_string+".ckpt")
    seg_task_best = BinarySemanticSegmentationTask.load_from_checkpoint(best_model_path)
    
    time_test_beginning = time.time()-time_start2
    time_start2 = time.time()
    
    if(len(test_df)==0):
        
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
    
    test_file = os.path.join(version_log_dir, "tests", "results_"+run_string+".json")
    
    with open(test_file, "w") as outfile:
        json.dump(test_dict, outfile)
        
    ### PREDICT ###
    
    if(evaluate_map):
        
        # Evaluate both the Connection and Segmentation model by
        # creating the global masks for all the TVT craters (but
        # actually segmenting only the zones that intersect with the
        # TVT tiles, to save time) 
        
        eval_dict = evaluateEjectaMap(args.eval_conn_vers,
                              test_set,
                              best_model_path,
                              norm_path,
                              version_series,
                              args.conn_result_dir,
                              conn_data_root,
                              data_root,
                              args.cyl_data_root,
                              num_workers = num_workers)
                              
        eval_file = os.path.join(version_log_dir, "tests", "eval_conn-version-"+str(args.eval_conn_vers)+"_"+run_string+".json")
        
        with open(eval_file, "w") as outfile:
            json.dump(eval_dict, outfile) # Metrics saved in this json
