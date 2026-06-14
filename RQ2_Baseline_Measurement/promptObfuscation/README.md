
# Prompt Obfuscation Experiments

This repository contains scripts and instructions for reproducing the prompt obfuscation experiments used in our paper.

## 1. Environment Setup

Please first refer to **`README_promptObfuscation_version.md`** to set up the environment required for running **promptObfuscation**.

In our paper, we follow the **default configuration of promptObfuscation**.

## 2. Generating Obfuscated Soft Prompts

After the environment is properly configured, run the following command to generate obfuscated soft prompts on the TruthfulQA dataset:


```bash
nohup python -u obfuscate_truthQA_multi.py --model_name meta-llama/Llama-3.1-8B-Instruct --systems_csv ./data/system_prompts.csv --num_processes 8 --gpu_ids 0 1 2 3 --quantize_4bit --obfuscation_method soft --batch_size 4 --dataset_name truthfulqa --obf_sys_prompt_len 10 --output_token_count 15 --window_size 5 --optimizer_iter 10 --lr 0.001 --ce_weight 1.0 --kl_weight 1.0 --seed 42  > logs/obfuscate_truthQA$(date +%Y%m%d-%H%M%S).log 2>&1 &
```
This step produces obfuscated soft prompts, which will be used in the subsequent evaluation.

## 3. Generating Model Outputs with Obfuscated Prompts

Once the obfuscated soft prompts are generated, run:
```bash
bash simple_evaluation.sh
```
This script generates model outputs using the obfuscated soft prompts.

## 4. Generating Baseline Outputs with Original System Prompts
Next, move to the results directory and run:
```bash
cd results
bash eval_origin_output.sh
```
This script generates model outputs using the original (non-obfuscated) system prompts.

The outputs generated in the above steps are used for subsequent effectiveness and usability evaluations, as reported in the paper.