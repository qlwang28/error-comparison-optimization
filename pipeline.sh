#!/bin/bash

export CUDA_VISIBLE_DEVICES="5"

MODEL_PATH=../../pre-trained_model/bert_uncased_L-12_H-768_A-12

declare -A Round1=( ["TASK_DATA_PATH"]="./data/raw_data/laptop/"  ["max_seq_length"]=256 )
declare -A Round2=( ["TASK_DATA_PATH"]="./data/raw_data/rest/" ["max_seq_length"]=256 )
declare -A Round3=( ["TASK_DATA_PATH"]="./data/raw_data/device/" ["max_seq_length"]=256 )
declare -A Round4=( ["TASK_DATA_PATH"]="./data/raw_data/service/" ["max_seq_length"]=256 )
declare -a Round=("Round1"  "Round2"  "Round3"  "Round4")

for item in "${Round[@]}"; do
    declare -n dict="$item"
    TASK_DATA_PATH=${dict["TASK_DATA_PATH"]}
    max_seq_length=${dict["max_seq_length"]}

    python3 -B run_sequence_label.py --do_train --do_eval --pretrained_model_name bert \
            --data_dir ${TASK_DATA_PATH} --pretrained_params ${MODEL_PATH} \
            --batch_size 32 --learning_rate 3e-5 --num_train_epochs 8 \
            --do_lower_case --max_seq_length ${max_seq_length} --schedule WarmupLinearSchedule \
            --pretrained_vocab ${MODEL_PATH}/vocab.txt  --logging_global_step 300
done
