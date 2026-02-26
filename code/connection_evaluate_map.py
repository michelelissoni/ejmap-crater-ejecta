"""
Filename: connection_evaluate_map.py
Author: Michele Lissoni
Date: 2026-02-24
"""

"""

Create global ejecta maps for each crater (only with eval tiles intersecting the TVT tiles)
to evaluate a Connection and Segmentation model. Launched after a Connection training run
(the evaluation happens automatically after a Segmentation training run, but since the Connection training
is run first, evaluation is not always possible; but if it is, this script is used).

Arguments:

- index: index of the evaluation run
- log_dir: the Connection log directory
- ejc_data_root: the data directory for the Segmentation model
- data_root: the data directory for the Connection model
- cyl_data_root: the data directory for the manual masks (cylindrical projection)
- conn_result_dir: the results directory for the Connection model
- ejc_result_dir: the results directory for the Segmentation model
- num_workers: number of subprocesses

TO DO: change so that a separate prediction for each iteration can be performed for evaluation

"""

import os
import sys
code_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, code_dir)
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import argparse
import torch
import warnings

from ejecta_data_preparation import evaluateEjectaMap
from terratorch_custom_tasks import BinarySemanticSegmentationTask
from lightning.pytorch import Trainer

if __name__ == "__main__":

    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(device_id)
    else:
        warnings.warn("This evaluation should take place on the GPU.")
        gpu_name = "CPU"

    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--ejc_data_root", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--cyl_data_root", type=str, required=True)
    parser.add_argument("--conn_result_dir", type=str, required=True)
    parser.add_argument("--ejc_result_dir", type=str, required=True)
    parser.add_argument("--num_workers", type=int, required=True)
    args = parser.parse_args()
        
    eval_index = args.index
    log_dir = args.log_dir
    conn_data_root = args.data_root
    ejc_data_root = args.ejc_data_root
    cyl_data_root = args.cyl_data_root
    conn_result_dir = args.conn_result_dir
    ejc_result_dir = args.ejc_result_dir
    num_workers = args.num_workers
    
    # Retrieve parameters of evaluation run
    
    repeated_versions = np.loadtxt(os.path.join(log_dir, 'repeated_versions_to_eval.txt'), dtype=int)
    if(len(repeated_versions.shape)==2):
        conn_version = repeated_versions[eval_index,0]
        test_set = repeated_versions[eval_index, 1]
        ejc_version = repeated_versions[eval_index, 2]
    elif(len(repeated_versions.shape)==1):
        conn_version = repeated_versions[0]
        test_set = repeated_versions[1]
        ejc_version = repeated_versions[2]
    else:
        raise RuntimeError(f"The repeated versions array is {len(repeated_versions.shape)}-dimensional, when it should be 2-dimensional.")

    ejc_version_dir = os.path.join(ejc_result_dir, 'version_'+str(ejc_version))

    ejc_model_path = os.path.join(ejc_version_dir, "best-checkpoint_test-set-"+str(test_set)+".ckpt")
    ejc_norm_path = os.path.join(ejc_version_dir, "best-norm_test-set-"+str(test_set)+".npy")
    ejc_version_series = pd.read_csv(os.path.join(code_dir, 'ejecta_version_hparams.csv'), index_col = 'version').loc[ejc_version,:]
    
    # Evaluate both the Connection and Segmentation model by
    # creating the global masks for all the TVT craters (but
    # actually segmenting only the zones that intersect with the
    # TVT tiles, to save time) 
    
    eval_dict, conn_matrix_df = evaluateEjectaMap(conn_version, 
                                  test_set,
                                  ejc_model_path,
                                  ejc_norm_path,
                                  ejc_version_series,
                                  conn_result_dir,
                                  conn_data_root,
                                  ejc_data_root,
                                  cyl_data_root,
                                  return_conn_matrix = True,
                                  num_workers = num_workers)
                                  
    eval_dict["gpu_name"] = gpu_name
    
    eval_file = os.path.join(conn_result_dir, "version_"+str(conn_version), "eval_ejc-version-"+str(ejc_version)+"_test-set"+str(test_set)+".json") # Save evaluation metrics
    
    with open(eval_file, "w") as outfile:
        json.dump(eval_dict, outfile)
        
    conn_matrix_df.to_csv(os.path.join(conn_result_dir, "version_"+str(conn_version), "connmatrix_ejc-version-"+str(ejc_version)+"_test-set"+str(test_set)+".csv")) # Save connection
                                                                                                                                                                    # metrics, showing
                                                                                                                                                                    # which TVT positions
                                                                                                                                                                    # are connected for 
                                                                                                                                                                    # which craters
