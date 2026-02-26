#!/bin/bash

#SBATCH --job-name=conn_evaluate
#SBATCH --time=00:30:00
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=short
#SBATCH --mem=10gb

module load python/3.11.0
source train_run_config.cfg

get_free_port() {
    while :; do
        PORT=$(shuf -i 10000-65535 -n 1)
        if ! lsof -i :$PORT >/dev/null; then
            echo $PORT
            return
        fi
    done
}

# 2. Get a free port
MASTER_PORT=$(get_free_port)

# 3. Optional: define a master address (usually localhost for single-node)
MASTER_ADDR="127.0.0.1"

# 4. Export environment variables
export MASTER_ADDR
export MASTER_PORT
export WORLD_SIZE=1
export RANK=0

torchrun \
    --nproc-per-node=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    connection_evaluate_map.py \
        --index $SLURM_ARRAY_TASK_ID \
        --log_dir $CONN_LOG_DIR \
        --data_root $CONN_DATA_ROOT \
        --ejc_data_root $EJC_DATA_ROOT \
        --cyl_data_root $CYL_DATA_ROOT \
        --conn_result_dir $CONN_RESULT_DIR \
        --ejc_result_dir $EJC_RESULT_DIR \
        --num_workers $NUM_WORKERS

