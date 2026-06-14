#!/bin/bash


MAX_JOBS=5

function wait_for_jobs() {
    while [ $(jobs | wc -l) -ge $MAX_JOBS ]; do
        sleep 1
    done
}


for i in $(seq 1 50); do
    results_dir="results/parallel_obfuscation/obfuscate_truthQA_${i}"
    test_file="data/benign_data/sys_id_${i}.json"

    echo "Running task $i ..."

    wait_for_jobs

    python simple_evaluation_yy.py \
        --results_dir "${results_dir}" \
        --test_file "${test_file}" \
        --eval_batch_size 64 \
        --max_new_tokens 512 \
        --temperature 0.7 \
        --top_p 0.9 \
        --top_k 100 \
        --num_return_sequences 1 &

done

wait

echo "All tasks completed!"
