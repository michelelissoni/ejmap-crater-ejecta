#!/bin/bash

source train_run_config.cfg

mkdir -p $CONN_LOG_DIR
mkdir -p $EJC_LOG_DIR

if [ $CONN_TRAIN -eq 1 ]; then
    
    python3 connection_prepare_versions.py --log_dir $CONN_LOG_DIR --versions ${CONN_VERSIONS_TO_RUN[@]} --test_sets ${TEST_SETS[@]} --iterations $CONN_TRAININGS_PER_TEST

    ARRAY_LENGTH=$(cat ${CONN_LOG_DIR}repeated_versions_to_run.txt|wc -l)

    for ((i = 0 ; i < $ARRAY_LENGTH ; i++ ))
    do 
	    python3 connection_train.py --index $i --log_dir $CONN_LOG_DIR --data_root $CONN_DATA_ROOT --num_workers $NUM_WORKERS
    done

    python3 connection_test_summarise.py --rm_log $RM_LOG --data_root $CONN_DATA_ROOT --log_dir $CONN_LOG_DIR --result_dir $CONN_RESULT_DIR --versions ${CONN_VERSIONS_TO_RUN[@]} --save_ckpt $SAVE_CKPT --eval_ejc_vers $EVAL_EJC_VERSION
    
    if [ -n "$EVAL_EJC_VERSION" ]; then
    
        EVAL_LENGTH=$(cat ${CONN_LOG_DIR}repeated_versions_to_eval.txt|wc -l)
        
        for ((i = 0 ; i < $EVAL_LENGTH ; i++ ))
        do 
	        python3 connection_evaluate_map.py --index $i --log_dir $CONN_LOG_DIR --data_root $CONN_DATA_ROOT --ejc_data_root $EJC_DATA_ROOT --cyl_data_root $CYL_DATA_ROOT --conn_result_dir $CONN_RESULT_DIR --ejc_result_dir $EJC_RESULT_DIR --num_workers $NUM_WORKERS
        done
        
    fi
    
else

    python3 ejecta_prepare_versions.py --log_dir $EJC_LOG_DIR --versions ${EJC_VERSIONS_TO_RUN[@]} --test_sets ${TEST_SETS[@]} --iterations $EJC_TRAININGS_PER_TEST

    ARRAY_LENGTH=$(cat ${EJC_LOG_DIR}repeated_versions_to_run.txt|wc -l)

    for ((i = 0 ; i < $ARRAY_LENGTH ; i++ ))
    do 
	    python3 ejecta_train.py --index $i --log_dir $EJC_LOG_DIR --data_root $EJC_DATA_ROOT --conn_data_root $CONN_DATA_ROOT --num_workers $NUM_WORKERS --eval_conn_vers $EVAL_CONN_VERSION --cyl_data_root $CYL_DATA_ROOT --conn_result_dir $CONN_RESULT_DIR
    done

    python3 ejecta_test_summarise.py --rm_log $RM_LOG --log_dir $EJC_LOG_DIR --result_dir $EJC_RESULT_DIR --versions ${EJC_VERSIONS_TO_RUN[@]} --save_ckpt $SAVE_CKPT --eval_conn_vers $EVAL_CONN_VERSION
    
fi
