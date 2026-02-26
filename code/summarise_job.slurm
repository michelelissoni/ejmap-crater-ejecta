#!/bin/bash

#SBATCH --time=00:04:59
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --partition=maup
#SBATCH --mem=250mb
#SBATCH --output=slurm-summarise-%A.out

module load python/3.11.0
source train_run_config.cfg

if [ $CONN_TRAIN -eq 1 ]; then

    srun python3 connection_test_summarise.py --rm_log $RM_LOG --data_root $CONN_DATA_ROOT --log_dir $CONN_LOG_DIR --result_dir $CONN_RESULT_DIR --versions ${CONN_VERSIONS_TO_RUN[@]} --save_ckpt $SAVE_CKPT --eval_ejc_vers $EVAL_EJC_VERSION
    
else

    srun python3 ejecta_test_summarise.py --rm_log $RM_LOG --log_dir $EJC_LOG_DIR --result_dir $EJC_RESULT_DIR --versions ${EJC_VERSIONS_TO_RUN[@]} --save_ckpt $SAVE_CKPT --eval_conn_vers $EVAL_CONN_VERSION
    
fi
