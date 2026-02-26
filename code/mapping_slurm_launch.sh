#!/bin/bash

module load python/3.11.0
source mapping_run_config.cfg

ARRAY_LENGTH=${#TEST_SETS[@]}

if [ ${#EJC_VERSIONS[@]} -ne $ARRAY_LENGTH ] | [ ${#CONN_VERSIONS[@]} -ne $ARRAY_LENGTH ]; then
    echo "The EJC_VERSIONS, CONN_VERSIONS and TEST_SETS arrays must be of equal length, but they are:" ${#EJC_VERSIONS[@]} ${#CONN_VERSIONS[@]} $ARRAY_LENGTH
    exit 1
fi
    
if [ $LOAD_MASKS -eq 1 ] && [ -n "$LOAD_MASKS_DIR" ]; then

    mapping_job_output=$(sbatch --output=slurm-mapping-%A.out --tmp=300 --cpus-per-task=$NUM_WORKERS mapping_job.slurm)
    
elif [ $LOAD_MASKS -eq 1 ]; then
    
    echo "LOAD_MASKS is set to 1, but no LOAD_MASKS_DIR indicated."
    exit 1
    
else

    if [ $RAPID_NODE_ONLY -eq 1 ]; then

        mapping_job_output=$(sbatch --array=0-$((ARRAY_LENGTH-1)) --output=slurm-mapping-%A_%a.out --tmp=$((ARRAY_LENGTH*300)) --cpus-per-task=$NUM_WORKERS --gres=gpu:1 --exclude=tycho[57] mapping_job.slurm) 
        
    else   

        mapping_job_output=$(sbatch --array=0-$((ARRAY_LENGTH-1)) --output=slurm-mapping-%A_%a.out --tmp=$((ARRAY_LENGTH*300)) --cpus-per-task=$NUM_WORKERS --gres=gpu:1 mapping_job.slurm) 

    fi
    
fi

mapping_job_id=$(echo "$mapping_job_output" | awk '{print $4}')

echo "Submitted mapping job with ID: $mapping_job_id"

