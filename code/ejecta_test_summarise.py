"""
Filename: ejecta_test_summarise.py
Author: Michele Lissoni
Date: 2026-02-24
"""

"""

Takes the results of each training run of the Segmentation model and saves them
to the version folder in the results directory.

Arguments:

- result_dir: the results directory for the Segmentation model
- log_dir: log directory
- versions: the model versions that were trained
- rm_log: if 1, the log data are removed (useful to save space, but can remove individual checkpoint files)
- save_ckpt: if 1, the best checkpoint for each TV-Test split is saved (TO DO: change so that the checkpoints for all iterations are saved?)
- eval_conn_vers: the Connection model version for evaluation. If not set, the evaluation is not performed.

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
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--versions", nargs='+')
    parser.add_argument('--rm_log', type=int, required=True)
    parser.add_argument('--save_ckpt', type=int, required=True)
    parser.add_argument('--eval_conn_vers', type=int, required=False)
    args = parser.parse_args()
    
    run_keys = ['test-set', 
                'iteration'] # Keys identifying a training run
    group_keys = ['test-set'] 
    
    result_dir = args.result_dir
    log_dir = args.log_dir
    rm_log = bool(args.rm_log)
    save_ckpt = bool(args.save_ckpt)
    
    evaluate = args.eval_conn_vers is not None

    versions = np.array(args.versions).astype(int)
    
    best_runs = pd.DataFrame(index=versions, columns = group_keys)
    
    # Hyperparameters
    
    version_df = pd.read_csv(os.path.join(code_dir, 'ejecta_version_hparams.csv'), index_col = 'version', dtype_backend="numpy_nullable")
    version_df = convert_numpy_ints_columnwise(version_df)
    
    val_metrics = version_df.loc[versions, 'val_metric'].values
    
    saved_checkpoints = np.zeros((0,2), dtype=str)
    
    for j in range(0,len(versions)): # Loop over versions
    
        version = versions[j]
        val_metric = val_metrics[j]
        
        version_log_dir = os.path.join(log_dir, 'version_'+str(version))
        checkpoint_log_dir = os.path.join(version_log_dir, 'checkpoints')
        test_log_dir = os.path.join(version_log_dir, 'tests')
        norm_log_dir = os.path.join(version_log_dir, 'norms')
        
        # Evaluation metrics of each TV-Test split and iteration
        test_files = glob.glob(os.path.join(test_log_dir, 'results_'+'_'.join([run_key+'-*' for run_key in run_keys])+'.json'))
        
        for i in range(0,len(test_files)):
        
            test_file = test_files[i]
            test_file_split = np.array(os.path.splitext(os.path.basename(test_file))[0].split('_'))
            
            run_string = "_".join(test_file_split[1:])
            
            with open(test_file) as infile:
                result_dict = json.load(infile)
                
            if(evaluate):
                eval_file = os.path.join(test_log_dir, 'eval_conn-version-'+str(args.eval_conn_vers)+'_'+run_string+'.json')
                
                with open(eval_file) as infile:
                    eval_dict = json.load(infile) # Evaluation metrics for each iteration
                    
            # Gather the evaluation metrics into a single dataframe `version_eval_df`
                
            if(i==0):
                
                result_dict_keys = list(result_dict.keys())
                metrics = [key.removeprefix('test/') for key in result_dict_keys]
                version_results_df = pd.DataFrame(columns = run_keys + metrics, index = np.arange(len(test_files), dtype=int))
                
                if(evaluate):
                    eval_metrics = list(eval_dict.keys())
                    version_eval_df = pd.DataFrame(columns = run_keys + eval_metrics, index = np.arange(len(test_files), dtype=int))
            
            for k in range(0,len(run_keys)):
                run_key = run_keys[k]
                version_results_df.loc[i, run_key] = int(test_file_split[k+1].removeprefix(run_key+'-'))
                
                if(evaluate):
                    version_eval_df.loc[i, run_key] = int(test_file_split[k+1].removeprefix(run_key+'-'))
            
            for k in range(0,len(metrics)):
            
                metric = metrics[k]
                key = result_dict_keys[k]
                version_results_df.loc[i, metric] = result_dict[key]
                
            if(evaluate):
                    
                for k in range(0,len(eval_metrics)):
                
                    metric = eval_metrics[k]
                    version_eval_df.loc[i, metric] = eval_dict[metric]
                
        version_dir = os.path.join(result_dir, 'version_'+str(version))
        
        if(not(os.path.isdir(version_dir))):
            os.mkdir(version_dir)
            
        else:
            version_files = glob.glob(os.path.join(version_dir,'*'))
            for version_file in version_files:
                os.remove(version_file)
        
        version_results_df.to_csv(os.path.join(version_dir, 'test_metrics.csv'), index=False)
        
        if(evaluate):
            version_eval_df.to_csv(os.path.join(version_dir, 'eval_metrics.csv'), index=False) # Save evaluation metrics
            
            version_results_df = version_eval_df # Decide which checkpoints to save based in eval metrics
        
        # Save checkpoint and normalization files of the best training run
        
        version_val_metric = version_results_df[val_metric]
        val_metric_finite = np.any(np.isfinite(version_val_metric.values.astype(float)))
        best_idx = version_val_metric.idxmax() if val_metric_finite else version_results_df.index.values[0]
        
        best_run = version_results_df.loc[best_idx, group_keys]
        best_runs.loc[version, group_keys] = best_run.values
        
        idx_max_groups = version_results_df.groupby(group_keys)[val_metric].idxmax().values if val_metric_finite else version_results_df.groupby(group_keys).head(1).index.values
        
        if(len(version_results_df) % len(idx_max_groups) != 0):
            raise RuntimeError(f"The trainings per test should be the same for every group, but instead the total trainings for version {version} are {len(version_results_df)} and the groups are {len(idx_max_groups)}.")
            
        trainings_per_test = int(len(version_results_df)/len(idx_max_groups))
        
        for idx_group in idx_max_groups:
            max_run = version_results_df.loc[idx_group, run_keys]
            is_best_run = np.all(np.equal(max_run.loc[group_keys].values, best_run.values))
            
            if(not(save_ckpt) and not(is_best_run)):
                continue
            
            max_ckpt = "best-checkpoint_"+"_".join([run_key+'-'+str(max_run.loc[run_key]) for run_key in run_keys])+'.ckpt'
            max_ckpt = os.path.join(checkpoint_log_dir, max_ckpt)
            dst_ckpt = "best-checkpoint_"+"_".join([group_key+'-'+str(max_run.loc[group_key]) for group_key in group_keys])+'.ckpt'
            shutil.copy(max_ckpt, os.path.join(version_dir, dst_ckpt))
            
            max_norm = "norm_"+"_".join([run_key+'-'+str(max_run.loc[run_key]) for run_key in run_keys])+'.npy'
            max_norm = os.path.join(norm_log_dir, max_norm)
            dst_norm = "best-norm_"+"_".join([group_key+'-'+str(max_run.loc[group_key]) for group_key in group_keys])+'.npy'
            shutil.copy(max_norm, os.path.join(version_dir, dst_norm))
            
            saved_checkpoints = np.vstack([saved_checkpoints, np.array([[str(version), dst_ckpt]])])
                                
        version_df.loc[version, 'version_run'] = True
        for group_key in group_keys:
            if(group_key not in version_df.columns.values.astype(str)):
                version_df[group_key] = np.zeros(len(version_df), dtype=int)
            
            version_df.loc[version, group_key] = str(np.unique(version_results_df[group_key]).tolist())
            
        version_df.loc[version, 'trainings_per_test'] = trainings_per_test
        
        version_df.loc[version, 'eval_conn_version'] = args.eval_conn_vers if evaluate else None
            
        if(rm_log):
            
            checkpoint_files = glob.glob(os.path.join(checkpoint_log_dir,'*'))
            for checkpoint_file in checkpoint_files:
                os.remove(checkpoint_file)
                
            test_files = glob.glob(os.path.join(test_log_dir,'*'))
            for test_file in test_files:
                os.remove(test_file)
                
            norm_files = glob.glob(os.path.join(norm_log_dir,'*'))
            for norm_file in norm_files:
                os.remove(norm_file)
                
            os.rmdir(checkpoint_log_dir)
            os.rmdir(test_log_dir)
            os.rmdir(norm_log_dir)

            version_files = glob.glob(os.path.join(version_log_dir,'*'))
            for version_file in version_files:
                os.remove(version_file)
            os.rmdir(version_log_dir)
            
    version_df.to_csv(os.path.join(code_dir, 'ejecta_version_hparams.csv'), index=True, index_label='version') # Save the hyperparameter list, which now shows also which
                                                                                                               # Connection model version was used for evaluation, if any 
    
    np.savetxt(os.path.join(result_dir,'saved_checkpoints.txt'), saved_checkpoints, fmt="%s")
    
