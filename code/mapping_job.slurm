#!/bin/bash

#SBATCH --time=00:59:59
#SBATCH --job-name=mapping
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --partition=short
#SBATCH --mem=10gb

module load python/3.11.0
source mapping_run_config.cfg

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

if [ $LOAD_MASKS -eq 1 ] && [ -n $LOAD_MASKS_DIR ]; then
     
    torchrun \
        --nproc-per-node=1 \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        create_map.py \
            --load_masks_dir $LOAD_MASKS_DIR \
            --use_eval_crats \
            --ejc_data_root $EJC_DATA_ROOT \
            --num_workers $NUM_WORKERS
            
else

    CONN_VERS=${CONN_VERSIONS[$SLURM_ARRAY_TASK_ID]}
    EJC_VERS=${EJC_VERSIONS[$SLURM_ARRAY_TASK_ID]}
    TEST_SET=${TEST_SETS[$SLURM_ARRAY_TASK_ID]}

    torchrun \
        --nproc-per-node=1 \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        create_map.py \
            --ejc_vers $EJC_VERS \
            --conn_vers $CONN_VERS \
            --test_set $TEST_SET \
            --use_eval_crats $EVAL_CRATS \
            --save_masks $SAVE_MASKS \
            --save_masks_dir $SAVE_MASKS_DIR \
            --ejc_data_root $EJC_DATA_ROOT \
            --conn_data_root $CONN_DATA_ROOT \
            --ejc_result_dir $EJC_RESULT_DIR \
            --use_iter_ckpt $USE_ITER_CKPT \
            --conn_result_dir $CONN_RESULT_DIR \
            --num_workers $NUM_WORKERS        
fi
