#!/bin/bash

export CUDA_VISIBLE_DEVICES="0"

MODEL_NAME_8B="NousResearch-Meta-Llama-3-8B-Instruct"
MODEL_NAME_13B="h2oai-llama2-13b-chat"

DEFAULT_SHOT2COLUMN_8B="0:3,4:4,8:5,16:6"
DEFAULT_SHOT2COLUMN_13B="0:7,4:8,8:9,16:10"

DEFAULT_INDEXES_triplet="(5,5), (5,13), (5,21), (5,37)"
DEFAULT_INDEXES_quadruple="(6,6), (6,14), (6,22), (6,38)"

declare -A Round1=(
    ["DATA_PATH"]="./data/Aspect Sentiment Triplet Extraction/V1"
    ["TASK"]="triplet"
    ["DOMAIN"]="14lap 14res 15res 16res"
)
declare -A Round2=(
    ["DATA_PATH"]="./data/Aspect Sentiment Triplet Extraction/V2"
    ["TASK"]="triplet"
    ["DOMAIN"]="14lap 14res 15res 16res"
)
declare -A Round3=(
    ["DATA_PATH"]="./data/Aspect Sentiment Quad Prediction/t5/acos"
    ["TASK"]="quadruple"
    ["DOMAIN"]="laptop16 rest16"
)
declare -A Round4=(
    ["DATA_PATH"]="./data/Aspect Sentiment Quad Prediction/t5/asqp"
    ["TASK"]="quadruple"
    ["DOMAIN"]="rest15 rest16"
)

declare -a Rounds=("Round1" "Round2" "Round3" "Round4")
declare -a Models=("$MODEL_NAME_8B" "$MODEL_NAME_13B")

for model in "${Models[@]}"; do
    if [[ $model == "$MODEL_NAME_8B" ]]; then
        default_shot2column=$DEFAULT_SHOT2COLUMN_8B
    elif [[ $model == "$MODEL_NAME_13B" ]]; then
        default_shot2column=$DEFAULT_SHOT2COLUMN_13B
    fi

    for item in "${Rounds[@]}"; do
        declare -n dict="$item"
        data_path=${dict["DATA_PATH"]}
        task=${dict["TASK"]}
        domain=${dict["DOMAIN"]}

        if [[ $task == "triplet" ]]; then
            default_indexes=$DEFAULT_INDEXES_triplet
        elif [[ $task == "quadruple" ]]; then
            default_indexes=$DEFAULT_INDEXES_quadruple
        fi

        for subdomain in $domain; do
            python3 -B LLM_in_context_learning.py --MODEL_NAME "$model" --data_dir "$data_path" --task "$task" --domain "$subdomain" --indexs "$default_indexes" --shot2column "$default_shot2column"
            echo
            echo
        done
    done
done
