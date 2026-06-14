#!/bin/bash
# nohup bash eval_origin_output.sh > eval_origin_output.log 2>&1 &
NUM_GPUS=6
TASKS_PER_GPU=5
TOTAL_TASKS=50
GPU_START_ID=1

mkdir -p logs_eval

run_task() {
    local id=$1
    local gpu_index=$2
    local gpu_id=$((GPU_START_ID + gpu_index))

    echo "Starting eval task obfuscate_truthQA_${id} on GPU ${gpu_id}"

    CUDA_VISIBLE_DEVICES=${gpu_id} nohup python -u eval_origin_output.py \
        --csv_path "./parallel_obfuscation/obfuscate_truthQA_${id}/obf_sys_output.csv" \
        > logs_eval/eval_truthQA_${id}.log 2>&1 &

    echo "Task obfuscate_truthQA_${id} started with PID $! on GPU ${gpu_id}"
}

current_id=1

while [ $current_id -le $TOTAL_TASKS ]; do
    echo "Launching a new batch of tasks..."

    for ((gpu=0; gpu<NUM_GPUS && current_id<=TOTAL_TASKS; gpu++)); do
        for ((slot=0; slot<TASKS_PER_GPU && current_id<=TOTAL_TASKS; slot++)); do
            run_task $current_id $gpu
            ((current_id++))
        done
    done

    echo "This batch of tasks launched. Waiting for all to finish..."
    wait
    echo "One batch completed."
done

echo "All eval tasks completed!"
echo "Check logs in ./logs_eval/ directory"
