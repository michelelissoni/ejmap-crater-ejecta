"""
Filename: ejecta_prepare_versions.py
Author: Michele Lissoni
Date: 2026-02-24
"""

"""

Prepare the training runs of the Segmentation model (EJSEG)

Arguments:

- log_dir: log directory
- versions: the model versions that should be trained
- test_sets: the TV-Test splits over which they should be trained
- iterations: the number of iteratiions per TV-Test split

"""

import os
import sys
code_dir = os.path.dirname(os.path.abspath(__file__))
import itertools
import glob
import numpy as np
import pandas as pd
import argparse

import warnings

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--versions", nargs='+', required=True)
    parser.add_argument("--test_sets", nargs='+', required=True)
    parser.add_argument("--iterations", type=int, required=True)
    args = parser.parse_args()
    
    log_dir = args.log_dir
    versions = np.array(args.versions).astype(int)
    test_sets = np.array(args.test_sets).astype(int)
    trainings_per_test = args.iterations
    iterations = np.arange(trainings_per_test, dtype=int) # The iteration numbers
    
    test_repeats = np.array(list(itertools.product(test_sets, iterations)), dtype=int)
    test_repeat_len = test_repeats.shape[0]

    all_repeats = np.tile(test_repeats, (len(versions), 1))
    all_repeats = np.hstack([np.repeat(versions, test_repeat_len)[:,np.newaxis], all_repeats]) # All the combinations of versions, test_sets and iterations
    
    for i in range(0,len(versions)):
        version = versions[i]

        version_log_dir = os.path.join(log_dir, 'version_'+str(version))
        
        # Log directory and sub-directories: create if they don't exist

        if(not(os.path.isdir(version_log_dir))):
            os.mkdir(version_log_dir)
            os.mkdir(os.path.join(version_log_dir, 'checkpoints')) # Checkpoint files
            os.mkdir(os.path.join(version_log_dir, 'tests')) # Evaluation metrics
            os.mkdir(os.path.join(version_log_dir, 'norms')) # Normalization
            
        else: # Remove files if directories exist
        
            checkpoint_files = glob.glob(os.path.join(version_log_dir, 'checkpoints','*'))
            for checkpoint_file in checkpoint_files:
                os.remove(checkpoint_file)
            test_files = glob.glob(os.path.join(version_log_dir, 'tests','*'))
            for test_file in test_files:
                os.remove(test_file) 
            norm_files = glob.glob(os.path.join(version_log_dir, 'norms','*'))
            for norm_file in norm_files:
                os.remove(norm_file) 
        
        # Random seeds for the subsequent training run: this ensures that they are different for each training run 
        
        seed_source = np.random.default_rng()
        random_seeds = seed_source.integers(0, 2**16, test_repeat_len)
        np.save(os.path.join(version_log_dir, 'seeds.npy'), random_seeds)
    
    np.savetxt(os.path.join(log_dir, 'repeated_versions_to_run.txt'), all_repeats, fmt = '%u') # Save training run parameters to a file
                                                                                               # that will be read by the script handling each run
    
