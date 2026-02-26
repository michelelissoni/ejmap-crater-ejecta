#!/bin/bash

#SBATCH --time=00:04:59
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --partition=maup
#SBATCH --mem=250mb
#SBATCH --output=slurm-evalstart-%A.out

source train_run_config.cfg

EVAL_LENGTH=$(cat ${CONN_LOG_DIR}repeated_versions_to_eval.txt|wc -l)

sbatch --array=0-$((EVAL_LENGTH-1)) --output=slurm-evaluate-%A_%a.out --tmp=$((EVAL_LENGTH*300)) --cpus-per-task=$NUM_WORKERS evaluate_job.slurm

