"""
Filename: connection_prepare_versions.py
Author: Michele Lissoni
Date: 2026-02-24
"""

"""

Prepare the training runs of the Connection model (EJCONN)

Arguments:

- log_dir: log directory
- versions: the model versions that should be trained
- test_sets: the TV-Test splits over which they should be trained
- iterations: the number of iteratiions per TV-Test split

"""

import os
import sys
code_dir = os.path.dirname(os.path.abspath(__file__))
import glob
import itertools
import numpy as np
import pandas as pd
import argparse

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
            
    all_repeats = np.array(list(itertools.product(versions, test_sets, iterations)), dtype=int) # All the combinations of versions, test_sets and iterations
    
    for i in range(0,len(versions)):
        version = versions[i]

        version_log_dir = os.path.join(log_dir, 'version_'+str(version))
        
        # Log directory and sub-directories: create if they don't exist
        
        os.makedirs(version_log_dir, exist_ok=True)
        os.makedirs(os.path.join(version_log_dir, 'checkpoints'), exist_ok=True) # Checkpoint files
        os.makedirs(os.path.join(version_log_dir, 'tests'), exist_ok=True) # Evaluation metrics 
        os.makedirs(os.path.join(version_log_dir, 'positions'), exist_ok=True) # Connected tile positions
        os.makedirs(os.path.join(version_log_dir, 'norms'), exist_ok=True) # Normalization
        
        # Remove previous files
        
        checkpoint_files = glob.glob(os.path.join(version_log_dir, 'checkpoints','*'))
        for checkpoint_file in checkpoint_files:
            os.remove(checkpoint_file)
        test_files = glob.glob(os.path.join(version_log_dir, 'tests','*'))
        for test_file in test_files:
            os.remove(test_file) 
        norm_files = glob.glob(os.path.join(version_log_dir, 'norms','*'))
        for norm_file in norm_files:
            os.remove(norm_file)
        pos_files = glob.glob(os.path.join(version_log_dir, 'positions','*'))
        for pos_file in pos_files:
            os.remove(pos_file) 

        # Random seeds for the subsequent training run: this ensures that they are different for each training run 
            
        seed_source = np.random.default_rng()
        random_seeds = seed_source.integers(0, 2**16, trainings_per_test*len(test_sets))
        np.save(os.path.join(version_log_dir, 'seeds.npy'), random_seeds)
    
    np.savetxt(os.path.join(log_dir, 'repeated_versions_to_run.txt'), all_repeats, fmt = '%u') # Save training run parameters to a file
                                                                                               # that will be read by the script handling each run

