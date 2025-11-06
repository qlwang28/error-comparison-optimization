#!/bin/bash
export CUDA_VISIBLE_DEVICES="2,4,7"

MODEL_NAME_8B="NousResearch-Meta-Llama-3-8B-Instruct"
MODEL_NAME_13B="h2oai-llama2-13b-chat"

DEFAULT_TRAINCOLUMN_8B="3"
DEFAULT_TRAINCOLUMN_13B="4"
DEFAULT_TESTCOLUMN_8B=("11" "12" "13" "14" "15")
DEFAULT_TESTCOLUMN_13B=("16" "17" "18" "19" "20")

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

declare -a Rounds=("Round3" "Round4")
declare -a Models=("$MODEL_NAME_13B")
declare -a Types=("PPO" "DPO" "CPO" "OUR") # "SFT"  

for model in "${Models[@]}"; do
    for item in "${Rounds[@]}"; do
        declare -n dict="$item"
        data_path=${dict["DATA_PATH"]}
        task=${dict["TASK"]}
        domain=${dict["DOMAIN"]}

        if [ "$model" == "$MODEL_NAME_8B" ]; then
            train_column=$DEFAULT_TRAINCOLUMN_8B
            test_columns=("${DEFAULT_TESTCOLUMN_8B[@]}")
        elif [ "$model" == "$MODEL_NAME_13B" ]; then
            train_column=$DEFAULT_TRAINCOLUMN_13B
            test_columns=("${DEFAULT_TESTCOLUMN_13B[@]}")
        fi

        for subdomain in $domain; do
            for type in "${Types[@]}"; do
                case $type in
                    "SFT") test_column=${test_columns[0]} ;;
                    "PPO") test_column=${test_columns[1]} ;;
                    "DPO") test_column=${test_columns[2]} ;;
                    "CPO") test_column=${test_columns[3]} ;;
                    "OUR") test_column=${test_columns[4]} ;;
                    *) echo "Unknown preference type: $type"; exit 1 ;;
                esac

                python3 -B LLM_predictive_preference.py --MODEL_NAME "$model" --do_train --data_dir "$data_path" --task "$task" --domain "$subdomain" --preference_type "$type"
                echo
                python3 -B LLM_predictive_preference.py --MODEL_NAME "$model" --do_eval --data_dir "$data_path" --task "$task" --domain "$subdomain" --preference_type "$type" --train_column "$train_column" --test_column "$test_column"
                echo
            done
        done
    done
done