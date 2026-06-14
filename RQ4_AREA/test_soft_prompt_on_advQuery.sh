#!/bin/bash

MAX_JOBS=${MAX_JOBS:-1}
GPU_ID=${GPU_ID:-0}
MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}

for i in {1..50}; do
    while (( $(jobs -r | wc -l) >= MAX_JOBS )); do
        sleep 1
    done

    echo ">>> Starting id_$i on GPU ${GPU_ID}"
    CUDA_VISIBLE_DEVICES=${GPU_ID} python -u test_soft_prompt.py \
        --run-dir "result/id_$i" \
        --model "${MODEL}" \
        --prompts-json "../LeakBench/Adversarial_Subset/llama3_adversarial_query.json" \
        --compare-original \
        --compare-baseline \
        --device "cuda:0" \
        --csv-out "attack_result.csv" &

done

wait
echo "All jobs finished!"
