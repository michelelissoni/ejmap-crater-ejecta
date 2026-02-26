"""
Filename: connection_test_summarise.py
Author: Michele Lissoni
Date: 2026-02-24
"""

"""

Takes the results of each training run of the Connection model and saves them
to the version folder in the results directory. Combines the results of the various iterations
to predict the connected tiles.

Arguments:

- data_root: the data directory for the Connection model
- result_dir: the results directory for the Connection model
- log_dir: log directory
- versions: the model versions that were trained
- rm_log: if 1, the log data are removed (useful to save space, but can remove individual checkpoint files)
- save_ckpt: if 1, the best checkpoint for each TV-Test split is saved (TO DO: change so that the checkpoints for all iterations are saved?)
- eval_ejc_vers: the Segmentation model version for evaluation. If not set, the evaluation is not performed.

TO DO: change so that the separate prediction for each iteration can be used for evaluation

"""

import os
import sys
code_dir = os.path.dirname(os.path.abspath(__file__))
import shutil
import glob
import numpy as np
import pandas as pd
import json
import argparse

def int_or_none(value):
    if value == "" or value is None:
        return None
    return int(value)

def convert_numpy_ints_columnwise(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()  # keep original untouched
    for col in df.columns:
        # Create python list preserving pd.NA as-is
        new_list = [int(x) if isinstance(x, np.integer) else x for x in df[col].tolist()]
        out[col] = pd.Series(new_list, dtype=object)
    return out

if __name__ == '__main__':

    # In-line arguments

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--versions", nargs='+')
    parser.add_argument('--rm_log', type=int, required=True)
    parser.add_argument('--save_ckpt', type=int, required=True)
    parser.add_argument('--eval_ejc_vers', nargs="?", type=int_or_none, required=False, default=None)
    args = parser.parse_args()

    data_root = args.data_root
    result_dir = args.result_dir
    log_dir = args.log_dir
    rm_log = bool(args.rm_log)
    save_ckpt = bool(args.save_ckpt)
    versions = np.array(args.versions).astype(int)
    
    args = parser.parse_args()
    
    result_dir = args.result_dir
    log_dir = args.log_dir
    rm_log = bool(args.rm_log)
    versions = np.array(args.versions).astype(int)
    
    # Hyperparameters
    
    version_df = pd.read_csv(os.path.join(code_dir, 'connection_version_hparams.csv'), index_col = 'version', dtype_backend="numpy_nullable")
    version_df = convert_numpy_ints_columnwise(version_df)
    
    val_metrics = version_df.loc[versions, 'val_metric'].values # Metric used the choose best versions
    iter_thresholds = version_df.loc[versions, 'iter_threshold'].values # Fraction of iterations that say the tile is connected necessary for it to be connected in the final results
    
    saved_checkpoints = np.zeros((0,2), dtype=str)
    
    eval_position_filelist = glob.glob(os.path.join(data_root, 'eval_positions', 'eval_positions_crater_*.csv'))
    crater_FIDs = np.array([int(os.path.basename(filename).removeprefix('eval_positions_crater_').removesuffix('.csv')) for filename in eval_position_filelist], dtype=int)
    
    eval_repeats = np.zeros((0,2), dtype=int)
    
    # Loop over versions
    
    for j in range(0,len(versions)):
    
        version = versions[j]
        val_metric = val_metrics[j]
        iter_threshold = iter_thresholds[j] if np.isfinite(iter_thresholds[j]) else None
        
        version_log_dir = os.path.join(log_dir, 'version_'+str(version))
        checkpoint_log_dir = os.path.join(version_log_dir, 'checkpoints')
        test_log_dir = os.path.join(version_log_dir, 'tests')
        position_log_dir = os.path.join(version_log_dir, 'positions')
        norm_log_dir = os.path.join(version_log_dir, 'norms')
        
        test_files = glob.glob(os.path.join(test_log_dir, 'results_test-*_iteration-*.json')) # Metrics of each iteration
        
        test_sets = np.zeros(len(test_files), dtype=int)
        
        for i in range(0,len(test_files)): # Iterate over all iterations and TV-Test splits
        
            test_file = test_files[i]
            test_file_split = os.path.splitext(os.path.basename(test_file))[0].split('_')
            test_set_num = int(test_file_split[1].removeprefix('test-set-'))
            iteration_num = int(test_file_split[2].removeprefix('iteration-'))
            
            test_sets[i] = test_set_num
            
            with open(test_file) as infile:
                result_dict = json.load(infile)
                
            eval_positions = pd.read_csv(os.path.join(position_log_dir, f"eval-pos_test-set-{test_set_num}_iteration-{iteration_num}.csv")) # Connected tiles of each iteration
            eval_positions[str(test_set_num)+'_'+str(iteration_num)] = 1
                
            if(i==0):
                
                result_dict_keys = list(result_dict.keys())
                metrics = [key.removeprefix('test/') for key in result_dict_keys]
                version_results_df = pd.DataFrame(columns = metrics, index = np.arange(len(test_files), dtype=int)) # Per-tile test metrics
                version_results_df['test-set'] = np.zeros(len(version_results_df), dtype=int)
                version_results_df['iteration'] = np.zeros(len(version_results_df), dtype=int)
                version_results_df = version_results_df.loc[:,['test-set', 'iteration'] + metrics]
                
                all_eval_positions = eval_positions.copy()
                
            else:
                # Assemble all the eval positions for all craters in a single dataframe
                # Now, each column [TEST SET]_[ITERATION] will show whether a given position is connected according to the model trained in that iteration
                all_eval_positions = pd.merge(all_eval_positions, eval_positions, on=['crat_FID','eval_pos_index'], how='outer')
                
            version_results_df.loc[i, 'iteration'] = iteration_num
            version_results_df.loc[i , 'test-set'] = test_set_num
            
            for k in range(0,len(metrics)):
            
                metric = metrics[k]
                key = result_dict_keys[k]
                version_results_df.loc[i, metric] = result_dict[key]
                
        version_dir = os.path.join(result_dir, 'version_'+str(version)) # Version folder in the results directory
        position_dir = os.path.join(version_dir, 'conn_positions') # The connected tile positions lists for each crater are saved here
        
        if(not(os.path.isdir(version_dir))):
            os.mkdir(version_dir)
            os.mkdir(position_dir)
            
        else:
            position_files = glob.glob(os.path.join(position_dir,'*'))
            for position_file in position_files:
                os.remove(position_file)
            version_files = glob.glob(os.path.join(version_dir,'*'))
            for version_file in version_files:
                if(version_file!=position_dir):
                    os.remove(version_file)
 
         
        version_results_df.to_csv(os.path.join(version_dir, 'test_metrics.csv'), index=False) # Save per-tile test metrics (not used in model evaluation)
        
        test_sets, test_counts = np.unique(test_sets, return_counts=True)
        
        if(np.any(test_counts!=test_counts[0])):
            raise RuntimeError(f"The trainings per test should be the same for every test set, but instead they are: {test_sets} ; {test_counts}.")
            
        trainings_per_test = test_counts[0]
        
        version_val_metric = version_results_df[val_metric]
        best_idx = version_val_metric.idxmax() if np.any(np.isfinite(version_val_metric.values.astype(float))) else version_results_df.index.values[0]
        test_best = version_results_df.loc[best_idx, 'test-set']
        
        # Test set and versions for the evaluation
        # TO DO: add iterations, to evaluate each iteration separately
        
        eval_repeats = np.vstack([eval_repeats,np.hstack([np.repeat(version, len(test_sets))[:,np.newaxis], test_sets[:,np.newaxis]])])  
            
        # Save the connected positions and the best checkpoints for each TV-Test split
            
        for test_set in test_sets:
            test_results_df = version_results_df.loc[version_results_df['test-set']==test_set,:]
            test_val_metric = test_results_df[val_metric]
            
            best_idx = test_val_metric.idxmax() if np.any(np.isfinite(test_val_metric.values.astype(float))) else test_results_df.index.values[0]
        
            iteration_max = test_results_df.loc[best_idx, 'iteration']
            
            # Columns showing whether the eval positions are connected
            pos_columns = all_eval_positions.columns.values.astype(str)
            pos_columns = pos_columns[np.char.startswith(pos_columns, str(test_set)+'_')]
            pos_frac_predicted = np.nansum(all_eval_positions.loc[:,pos_columns], axis=1)*1.0/len(pos_columns)
            
            if(iter_threshold is not None):
                test_eval_positions = all_eval_positions.loc[pos_frac_predicted >= iter_threshold, ['crat_FID', 'eval_pos_index']] # Final result: which tiles are connected
                                                                                                                                   # depends on whether they were judged to be so
                                                                                                                                   # in enough iterations
                
                # Get the tile coordinates of the eval positions that were judged connected
                
                for crater_FID in crater_FIDs:
                    pos_df = pd.read_csv(os.path.join(data_root, 'eval_positions', 'eval_positions_crater_'+str(crater_FID)+'.csv'), index_col='eval_pos_index')
                    pos_df = pos_df.loc[test_eval_positions.loc[test_eval_positions['crat_FID']==crater_FID, 'eval_pos_index'].values,:]
                    pos_df.to_csv(os.path.join(position_dir, 'conn-pos_test-set-'+str(test_set)+'_crater-'+str(crater_FID)+'.csv'), index=True, index_label='eval_pos_index')
            
            if(not(save_ckpt) and test_set!=test_best):
                continue
                
            # Save best checkpoint and norm
            # TO DO: revise this, it is not very useful
                
            max_ckpt = os.path.join(checkpoint_log_dir, f"best-checkpoint_test-set-{test_set}_iteration-{iteration_max}.ckpt")
            dst_ckpt = f"best-checkpoint_test-set-{test_set}.ckpt"
            shutil.copy(max_ckpt, os.path.join(version_dir, dst_ckpt))
            
            max_norm = os.path.join(norm_log_dir, f"norm_test-set-{test_set}_iteration-{iteration_max}.npy")
            dst_norm = f"best-norm_test-set-{test_set}.npy"
            shutil.copy(max_norm, os.path.join(version_dir, dst_norm))
            
            saved_checkpoints = np.vstack([saved_checkpoints, np.array([[str(version), dst_ckpt]])])
        
        version_df.loc[version, 'version_run'] = True
        version_df.loc[version, 'test-set'] = str(test_sets.tolist())
        version_df.loc[version, 'trainings_per_test'] = int(trainings_per_test)
            
        # Remove log data
            
        if(rm_log):
            
            checkpoint_files = glob.glob(os.path.join(checkpoint_log_dir,'*'))
            for checkpoint_file in checkpoint_files:
                os.remove(checkpoint_file)
                
            test_files = glob.glob(os.path.join(test_log_dir,'*'))
            for test_file in test_files:
                os.remove(test_file)
                
            # TO DO: do not remove the position files until evaluation, in case each iteration is evaluated separately
                
            position_files = glob.glob(os.path.join(position_log_dir,'*'))
            for position_file in position_files:
                os.remove(position_file)
                
            norm_files = glob.glob(os.path.join(norm_log_dir,'*'))
            for norm_file in norm_files:
                os.remove(norm_file)
                
            os.rmdir(checkpoint_log_dir)
            os.rmdir(test_log_dir)
            os.rmdir(position_log_dir)
            os.rmdir(norm_log_dir)

            version_files = glob.glob(os.path.join(version_log_dir,'*'))
            for version_file in version_files:
                os.remove(version_file)
            
            os.rmdir(version_log_dir)
            
    # If evaluation is desired
            
    if(args.eval_ejc_vers is not None):
    
        eval_repeats = np.hstack([eval_repeats, np.tile(args.eval_ejc_vers, (eval_repeats.shape[0],1))]).astype(int)
    
        np.savetxt(os.path.join(log_dir, 'repeated_versions_to_eval.txt'), eval_repeats, fmt='%s') # This file will be read when the evaluation is run
        
        version_df.loc[version, 'eval_ejc_version'] = args.eval_ejc_vers
        
    else:
    
        version_df.loc[version, 'eval_ejc_version'] = np.nan        
            
    version_df.to_csv(os.path.join(code_dir, 'connection_version_hparams.csv'), index=True, index_label='version') # Save the hyperparameter list, which now shows also which
                                                                                                                   # Segmentation model version was used for evaluation, if any
    
    np.savetxt(os.path.join(result_dir,'saved_checkpoints.txt'), saved_checkpoints, fmt="%s")
                
