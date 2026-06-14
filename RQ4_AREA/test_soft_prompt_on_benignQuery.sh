#!/bin/bash


NUM_GPUS=${NUM_GPUS:-2}
TASKS_PER_GPU=${TASKS_PER_GPU:-1}
TOTAL_TASKS=${TOTAL_TASKS:-50}
MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}


mkdir -p logs

run_task() {
    local id=$1
    local gpu=$2

    echo "Starting task id_${id} on GPU ${gpu}"

    CUDA_VISIBLE_DEVICES=${gpu} nohup python -u test_soft_prompt.py \
        --run-dir "result/id_${id}" \
        --model "${MODEL}" \
        --prompts-json "../LeakBench/Benign_Subset/sys_id_${id}.json" \
        --compare-baseline \
        --compare-original \
        --device "cuda:0" \
        --csv-out "benign_result.csv" \
        > logs/run_id_${id}.log 2>&1 &

    echo "Task id_${id} started with PID $! on GPU ${gpu}"
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

    echo "This batch launched. Waiting for all tasks in batch to finish..."
    wait
    echo "One batch completed."
done

echo "All tasks completed!"
echo "Check logs in ./logs/ directory"
