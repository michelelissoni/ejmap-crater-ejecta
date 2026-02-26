#!/bin/bash

module load python/3.11.0
source train_run_config.cfg

if [ $CONN_TRAIN -eq 1 ]; then

    mkdir -p $CONN_LOG_DIR

    python3 connection_prepare_versions.py --log_dir $CONN_LOG_DIR --versions ${CONN_VERSIONS_TO_RUN[@]} --test_sets ${TEST_SETS[@]} --iterations $CONN_TRAININGS_PER_TEST
    
    ARRAY_LENGTH=$(cat ${CONN_LOG_DIR}repeated_versions_to_run.txt|wc -l)
    TRAIN_NAME=training_conn
    SUMMARISE_NAME=summarise_conn
    
else

    mkdir -p $EJC_LOG_DIR

    python3 ejecta_prepare_versions.py --log_dir $EJC_LOG_DIR --versions ${EJC_VERSIONS_TO_RUN[@]} --test_sets ${TEST_SETS[@]} --iterations $EJC_TRAININGS_PER_TEST
    
    ARRAY_LENGTH=$(cat ${EJC_LOG_DIR}repeated_versions_to_run.txt|wc -l)
    TRAIN_NAME=training_ejc
    SUMMARISE_NAME=summarise_ejc
    
fi
    
echo "Versions prepared"

train_job_output=$(sbatch --array=0-$((ARRAY_LENGTH-1)) --output=slurm-training-%A_%a.out --tmp=$((ARRAY_LENGTH*300)) --cpus-per-task=$NUM_WORKERS --job-name=$TRAIN_NAME training_job.slurm)
train_job_id=$(echo "$train_job_output" | awk '{print $4}')

echo "Submitted train job with ID: $train_job_id"

summarise_job_output=$(sbatch --job-name=$SUMMARISE_NAME --dependency=afterok:${train_job_id} summarise_job.slurm) ####
summarise_job_id=$(echo "$summarise_job_output" | awk '{print $4}')

echo "Submitted summarise job with ID: $summarise_job_id"

if [ $CONN_TRAIN -eq 1 ] && [ -n "$EVAL_EJC_VERSION" ]; then

    sbatch --dependency=afterok:${summarise_job_id} evaluate_start.slurm
    
fi
