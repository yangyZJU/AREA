This repository contains the code to reproduce the results in this paper: **"Prompt Obfuscation for Large Language Models"**.
It contains the code to reproduce the results presented in the paper. The code allows users to perform and evaluate prompt obfuscation as an alternative method to traditional system prompting for large language models. Furthermore, different deobfuscation methods are included.

## Project Structure
```
prompt_obfuscation
├── README.md
├── compare_output.py
├── compare_sys_prompts.py
├── data
│   ├── __init__.py
│   ├── config.py
│   ├── loader.py
│   └── utils.py
├── evaluate_finetuning.py
├── evaluate_fluency_deobfuscation.py
├── evaluate_obfuscation.py
├── evaluate_prompt_extraction.py
├── extraction_prompts
│   └── gpt4_generated.json
├── finetune.py
├── fluency_deobfuscation.py
├── generate_output.py
├── obfuscate.py
├── projection.py
├── prompt_extraction.py
├── requirements.txt
└── src
    ├── __init__.py
    ├── finetuning_utils.py
    ├── logging_config.py
    ├── model.py
    ├── output_generation.py
    ├── output_similarity.py
    ├── prompt_utils.py
    ├── style_prompts.py
    ├── sys_prompt_similarity.py
    └── utils.py
```

The `data/` directory handles dataset loading and processing. The `src/` directory contains core logic for models, generation, and evaluation. The `extraction_prompts/` directory contains extraction prompts for the prompt extraction attack. The python scripts in the root directory are used to run the experiments.

## Setup
A GPU is highly recommended for reasonable computation times.
1. Create a Python 3.12.7 environment (e.g., using conda):
```
conda create -n prompt_obfuscation python=3.12.7
conda activate prompt_obfuscation
```
2. Install the required packages:
```
pip install -r requirements.txt
```
3. Hugging Face Access: The main model used (Llama-3.1-8B) requires a Hugging Face account with access granted to the model. Log in via the command line after requesting access on the model's page:
```
huggingface-cli login
```

**Note**: All datasets and models used are hosted on [huggingface](https://huggingface.co/) and are automatically downloaded to `~/.cache`. To change the default directory, you can run `export HF_HOME="/new/path"` for huggingface and `export SENTENCE_TRANSFORMERS_HOME="/new/path"` for sentence_transformers.

# Obfuscation
The `obfuscate.py` script is used to obfuscate system prompts and can be used to reproduce the results in Table 1, 3, 4, and 5 of the paper.

## Arguments
| Argument | Type | Default Value | Description |
|----------|------|---------------|-------------|
| ```--help``` | - | - | Show help message for command line arguments |
| ```--model_name``` | <b>str</b> | ```meta-llama/Meta-Llama-3.1-8B-Instruct``` | Huggingface model name to use for obfuscation |
| ```--quantize_4bit``` | <b>bool</b> | ```False``` | Enable 4-bit quantization for the model. (Mutually exclusive with --quantize_8bit) |
| ```--quantize_8bit``` | <b>bool</b> | ```False``` | Enable 8-bit quantization for the model. (Mutually exclusive with --quantize_4bit) |
| ```--system_prompt``` | <b>str</b> | ```None``` | Specify a custom system prompt directly as a string (Mutually exclusive with --style). |
| ```--style``` | <b>str</b> | ```None``` | Specify a predefined style for the system prompt. The available styles are defined in 'src/style_prompts.py' (Mutually exclusive with --system_prompt). |
| ```--obfuscation_method``` | <b>str</b> | ```soft``` | Method for obfuscating the system prompt (choices: "soft", "hard") |
| ```--batch_size``` | <b>int</b> | ```4``` | Batch size for optimization |
| ```--dataset_size``` | <b>int</b> | ```100``` | Dataset size for optimization (80:20 split) |
| ```--dataset_name``` | <b>str</b> | ```truthfulqa``` | Dataset to use for optimization (choices: "truthfulqa", "triviaqa", "cnn_dailymail", "samsum") |
| ```--task_hints``` | <b>bool</b> | ```False``` | Whether to use task hints |
| ```--obf_sys_prompt_len``` | <b>int</b> | ```10``` | Length of the randomly initialized obfuscated system prompt in tokens. |
| ```--output_token_count``` | <b>int</b> | ```15``` | Number of output tokens to optimize over |
| ```--window_size``` | <b>int</b> | ```5``` | Number of tokens in the context window to consider for gradient calculation |
| ```--optimizer_iter``` | <b>int</b> | ```10``` | Number of optimization iterations |
| ```--lr``` | <b>float</b> | ```0.001``` | Learning rate for optimization (only used for soft prompt obfuscation) |
| ```--topk``` | <b>int</b> | ```3``` | topk value for GCG (only used for hard prompt obfuscation) |
| ```--search_width``` | <b>int</b> | ```10``` | search_width value for GCG (only used for hard prompt obfuscation) |
| ```--n_replace``` | <b>int</b> | ```1``` | n_replace value for GCG (only used for hard prompt obfuscation) |
| ```--ce_weight``` | <b>float</b> | ```1.0``` | Weight for cross-entropy loss |
| ```--kl_weight``` | <b>float</b> | ```1.0``` | Weight for KL divergence loss |
| ```--seed``` | <b>int</b> | ```42``` | Seed for reproducibility |
| ```--output_dir``` | <b>str</b> | ```results/obfuscation``` | Output directory for obfuscation results |

## Example Usage
This command is used to soft prompt obfuscate the `pirate` style system prompt using the TruthfulQA dataset with the same hyperparameters used in the paper. 
```
python3 obfuscate.py --model_name meta-llama/Meta-Llama-3.1-8B-Instruct --quantize_4bit --style pirate --obfuscation_method soft --batch_size 4 --dataset_size 800 --dataset_name truthfulqa --obf_sys_prompt_len 10 --output_token_count 15 --window_size 5 --optimizer_iter 10 --lr 0.001 --ce_weight 1.0 --kl_weight 1.0 --seed 42 --output_dir results/obfuscation
```

This command obfuscates the `pirate` style system prompt using hard prompt obfuscation with the same hyperparameters.
```
python3 obfuscate.py --model_name meta-llama/Meta-Llama-3.1-8B-Instruct --quantize_4bit --style pirate --obfuscation_method hard --batch_size 4 --dataset_size 800 --dataset_name truthfulqa --obf_sys_prompt_len 10 --output_token_count 15 --window_size 5 --optimizer_iter 10 --topk 3 --search_width 10 --n_replace 1 --ce_weight 1.0 --kl_weight 1.0 --seed 42 --output_dir results/obfuscation
```

`obfuscate.py` creates a result directory specified by `--output_dir`. It saves the processed training and test datasets in `prepared_data/`, the training loss per iteration in `train_loss.npy`, a list of obfuscated system prompts after each iteration in `obfuscated_system_prompt_list.pt`, and all hyperparameters in `params.json`.


# Obfuscation Evaluation
The `evaluate_obfuscation.py` script takes the result directory of `obfuscate.py` and evaluates the list of obfuscated system prompts on the test data using specified metrics. It reproduces the numbers in Table 1, 3, 4, and 5 of the paper.
It saves the best performing obfuscated system prompts in `best_candidate.pt`, the output of the best performing obfuscated system prompt on the test set in `best_candidate_output.json`, the metric scores of the best performing obfuscated system prompt in `best_candidate_scores.json`, all metric scores for all obfuscated system prompts in `all_scores.json`, the generation parameters in `generation_config.json`, and the output using the conventional system prompt in `conventional_output.json`. All files are saved in the same directory as the results of `obfuscate.py`.

## Arguments
| Argument | Type | Default Value | Description |
|----------|------|---------------|-------------|
| ```--help``` | - | - | Show help message for command line arguments |
| ```--results_dir``` | <b>str</b> | Required | Path to the directory where obfuscate.py saved its results. |
| ```--metrics``` | <b>list</b> | ```["sacrebleu", "rouge1", "rouge2", "rougeL", "rougeLsum", "meteor", "bertscore", "cer", "nist_mt", "chrf", "cosine_similarity"]``` | List of metrics to use for evaluation. (choices: "sacrebleu", "rouge1", "rouge2", "rougeL", "rougeLsum", "meteor", "bertscore", "cer", "nist_mt", "chrf", "cosine_similarity") |
| ```--eval_batch_size``` | <b>int</b> | ```32``` | Batch size for generating model outputs during evaluation. |
| ```--max_new_tokens``` | <b>int</b> | ```125``` | Max new tokens for generation during evaluation. |
| ```--temperature``` | <b>float</b> | ```0.7``` | Temperature for sampling. |
| ```--top_p``` | <b>float</b> | ```0.9``` | Top-p for sampling. |
| ```--top_k``` | <b>int</b> | ```100``` | Top-k for sampling. |
| ```--num_return_sequences``` | <b>int</b> | ```5``` | Number of sequences to return per prompt. |


## Example Usage
This command is used to evaluate obfuscated system prompts saved in `results/obfuscation` with the same hyperparameters used in the paper:
```
python3 evaluate_obfuscation.py --results_dir results/obfuscation --metrics "sacrebleu" "rouge1" "rouge2" "rougeL" "rougeLsum" "meteor" "bertscore" "cer" "nist_mt" "chrf" "cosine_similarity" --eval_batch_size 64 --max_new_tokens 125 --temperature 0.7 --top_p 0.9 --top_k 100 --num_return_sequences 5
```


# Finetuning
The `finetune.py` script is used to finetune LoRa adapters with outputs generated by the conventional system prompt. This can be used to reproduce the results in Table 6 of the paper.

## Arguments
| Argument | Type | Default Value | Description |
|----------|------|---------------|-------------|
| ```--help``` | - | - | Show help message for command line arguments |
| ```--model_name``` | <b>str</b> | ```meta-llama/Meta-Llama-3.1-8B-Instruct``` | Huggingface model name to use for finetuning |
| ```--quantize_4bit``` | <b>bool</b> | ```False``` | Enable 4-bit quantization for the model. (Mutually exclusive with --quantize_8bit) |
| ```--quantize_8bit``` | <b>bool</b> | ```False``` | Enable 8-bit quantization for the model. (Mutually exclusive with --quantize_4bit) |
| ```--system_prompt``` | <b>str</b> | ```None``` | Specify a custom system prompt directly as a string. (Mutually exclusive with --style) |
| ```--style``` | <b>str</b> | ```None``` | Specify a predefined style for the system prompt. The available styles are defined in 'style_prompts.py'. (Mutually exclusive with --system_prompt) |
| ```--dataset_size``` | <b>int</b> | ```100``` | Dataset size for optimization (80:20 split) |
| ```--dataset_name``` | <b>str</b> | ```truthfulqa``` | Dataset to use for optimization (choices: "truthfulqa", "triviaqa", "cnn_dailymail", "samsum") |
| ```--task_hints``` | <b>bool</b> | ```False``` | Whether to use task hints |
| ```--batch_size``` | <b>int</b> | ```4``` | Batch size for optimization |
| ```--output_token_count``` | <b>int</b> | ```15``` | Number of output tokens to optimize over |
| ```--optimizer_iter``` | <b>int</b> | ```10``` | Number of optimization iterations |
| ```--lr``` | <b>float</b> | ```0.0002``` | Learning rate for finetuning |
| ```--lora_r``` | <b>int</b> | ```8``` | LoRA rank for finetuning |
| ```--lora_alpha``` | <b>int</b> | ```16``` | LoRA alpha for finetuning |
| ```--seed``` | <b>int</b> | ```42``` | Seed for reproducibility |
| ```--output_dir``` | <b>str</b> | ```results/finetuning``` | Output directory for finetuning results |

## Example Usage
This command is used to finetune LoRa adapters for the `pirate` style system prompt using the TruthfulQA dataset with the same hyperparameters used in the paper.
```
python3 finetune.py --model_name meta-llama/Meta-Llama-3.1-8B-Instruct --quantize_4bit --style pirate --dataset_size 800 --dataset_name truthfulqa --batch_size 4 --output_token_count 15 --optimizer_iter 10 --lr 0.0002 --lora_r 8 --lora_alpha 16 --seed 42 --output_dir results/finetuning
```

The script will create a result directory specified by `--output_dir`. It saves the processed training and test datasets in `prepared_data/`, adapter directories for each iteration named `epoch_x/` in `lora_adapters/`, and all hyperparameters in `params.json`.

# Finetuning Evaluation
The `evaluate_finetuning.py` script takes the result directory of `finetune.py` and evaluates all adapters specified in `lora_adapters/` on the test data using specified metrics. It reproduces the numbers in Table 6 of the paper.

## Arguments
| Argument | Type | Default Value | Description |
|----------|------|---------------|-------------|
| ```--help``` | - | - | Show help message for command line arguments |
| ```--results_dir``` | <b>str</b> | Required | Path to the directory where finetune.py saved its results. |
| ```--metrics``` | <b>list</b> | ```["sacrebleu", "rouge1", "rouge2", "rougeL", "rougeLsum", "meteor", "bertscore", "cer", "nist_mt", "chrf", "cosine_similarity"]``` | List of metrics to use for evaluation. (choices: "sacrebleu", "rouge1", "rouge2", "rougeL", "rougeLsum", "meteor", "bertscore", "cer", "nist_mt", "chrf", "cosine_similarity") |
| ```--eval_batch_size``` | <b>int</b> | ```32``` | Batch size for generating model outputs during evaluation. |
| ```--max_new_tokens``` | <b>int</b> | ```125``` | Max new tokens for generation during evaluation. |
| ```--temperature``` | <b>float</b> | ```0.7``` | Temperature for sampling. |
| ```--top_p``` | <b>float</b> | ```0.9``` | Top-p for sampling. |
| ```--top_k``` | <b>int</b> | ```100``` | Top-k for sampling. |
| ```--num_return_sequences``` | <b>int</b> | ```5``` | Number of sequences to return per prompt. |

## Example Usage
This command is used to evaluate all LoRa adapters saved in `results/finetuning/lora_adapters` with the same hyperparameters used in the paper:

```
python3 evaluate_finetuning.py --results_dir results/finetuning --metrics "sacrebleu" "rouge1" "rouge2" "rougeL" "rougeLsum" "meteor" "bertscore" "cer" "nist_mt" "chrf" "cosine_similarity" --eval_batch_size 64 --max_new_tokens 125 --temperature 0.7 --top_p 0.9 --top_k 100 --num_return_sequences 5
```

It saves the best performing LoRa adapter in `results/finetuning/best_adapter`, the output of the best performing adapter on the test set in `best_adapter_output.json`, the metric scores of the best performing adapter in `best_adapter_scores.json`, all metric scores for all adapters in `all_scores.json`, the generation parameters in `generation_config.json`, and the output using the conventional system prompt in `conventional_output.json`.

# Prompt Extraction
The `prompt_extraction.py` script is used to run a prompt extraction attack against a model. This can be used to reproduce the results in Table 7.

## Arguments

| Argument | Type | Default Value | Description |
|----------|------|---------------|-------------|
| ```--help``` | - | - | Show help message for command line arguments |
| ```--results_dir``` | <b>str</b> | Required | Path to the directory where obfuscate.py saved its results. |
| ```--extraction_prompts_file``` | <b>str</b> | Required | Path to the file containing extraction prompts. |
| ```--batch_size``` | <b>int</b> | ```32``` | Batch size for generating model outputs during attack. |
| ```--output_filename``` | <b>str</b> | ```prompt_extraction_output.json``` | Filename for the output file. |
| ```--conventional``` | <b>bool</b> | ```False``` | Use the conventional system prompt in params.json. (One of --conventional, --system_prompt, --tensor_file, or --blank is required) |
| ```--system_prompt``` | <b>str</b> | ```None``` | Specify a custom system prompt directly as a string. (One of --conventional, --system_prompt, --tensor_file, or --blank is required) |
| ```--tensor_file``` | <b>str</b> | ```None``` | Specify a tensor file to load the system prompt from. (One of --conventional, --system_prompt, --tensor_file, or --blank is required) |
| ```--blank``` | <b>bool</b> | ```False``` | Use a blank system prompt. (One of --conventional, --system_prompt, --tensor_file, or --blank is required) |

## Example Usage
This command is used to run a prompt extraction attack on the best obfuscated system prompt saved in `results/obfuscation`:
```
python3 prompt_extraction.py --results_dir results/obfuscation --extraction_prompts_file extraction_prompts/gpt4_generated.json --batch_size 64 --output_filename prompt_extraction_output_obfuscated.json --tensor_file results/obfuscation/best_candidate.pt
```

It saves the generated output of the model in `results/obfuscation/prompt_extraction_output_obfuscated.json`.

# Prompt Extraction Evaluation
The `evaluate_prompt_extraction.py` analyzes the output generated by `prompt_extraction.py` to find successful prompt extractions. This can be used to reproduce the results in Table 7.

## Arguments
| Argument | Type | Default Value | Description |
|----------|------|---------------|-------------|
| ```--help``` | - | - | Show help message for command line arguments |
| ```--results_dir``` | <b>str</b> | Required | Path to the directory where obfuscate.py saved its results. |
| ```--extraction_output_file``` | <b>str</b> | Required | Path to the file containing extraction output. |
| ```--rouge_recall_threshold``` | <b>float</b> | ```0.9``` | Rouge recall threshold to use for approximate-match evaluation. |
| ```--successful_outputs_filename``` | <b>str</b> | ```prompt_extraction_successful_outputs.json``` | Filename for the output file containing successful outputs. |

## Example Usage
This command is used to evaluate the output saved in `results/obfuscation/prompt_extraction_output_obfuscated.json`:
```
python3 evaluate_prompt_extraction.py --results_dir results/obfuscation/ --extraction_output_file results/obfuscation/prompt_extraction_output_obfuscated.json --rouge_recall_threshold 0.9 --successful_outputs_filename prompt_extraction_successful_outputs_obfuscated.json
```

It saves the successful extraction outputs in `results/obfuscation/prompt_extraction_successful_outputs_obfuscated.json`

# Projection
The `projection.py` script is used to project an embedded (soft) prompt back to the token space. This can be used to reproduce the results in Table 8.

## Arguments
| Argument | Type | Default Value | Description |
|----------|------|---------------|-------------|
| ```--help``` | - | - | Show help message for command line arguments |
| ```--results_dir``` | <b>str</b> | Required | Path to the directory where obfuscate.py saved its results. |
| ```--embedding_file``` | <b>str</b> | Required | Path to the tensor file containing embeddings. |
| ```--euclidean``` | <b>bool</b> | ```False``` | Use euclidean projection. |
| ```--cosine``` | <b>bool</b> | ```False``` | Use cosine projection. |
| ```--projected_ids_filename``` | <b>str</b> | ```projected_ids.pt``` | Filename for the output file containing projected ids. |

## Example Usage
This command is used to project the best obfuscated soft system prompt saved in `results/obfuscation/` back to token space using euclidean distance:
```
python3 projection.py --results_dir results/obfuscation/ --embedding_file results/obfuscation/best_candidate.pt --euclidean --projected_ids_filename best_candidate_euclidean_projection.pt
```

It saves the projected token IDs in `results/obfuscation/best_candidate_euclidean_projection.pt`.

# Fluency Deobfuscation
The `fluency_deobfuscation.py` script is used to deobfuscate an embedded (soft) system prompt back into a more readable form using optimization. This can be used to reproduce the results in Table 9 and 10.

## Arguments 
| Argument | Type | Default Value | Description |
|----------|------|---------------|-------------|
| ```--help``` | - | - | Show help message for command line arguments |
| ```--results_dir``` | <b>str</b> | Required | Path to the directory where obfuscate.py saved its results. |
| ```--embedding_file``` | <b>str</b> | Required | Path to the tensor file containing embeddings. |
| ```--deobfuscation_method``` | <b>str</b> | ```soft``` | Method for deobfuscating the embedded system prompt (choices: "soft", "hard") |
| ```--batch_size``` | <b>int</b> | ```4``` | Batch size for optimization |
| ```--dataset_size``` | <b>int</b> | ```100``` | Dataset size for optimization (80:20 split) |
| ```--output_token_count``` | <b>int</b> | ```15``` | Number of output tokens to optimize over |
| ```--window_size``` | <b>int</b> | ```5``` | Number of tokens in the context window to consider for gradient calculation |
| ```--optimizer_iter``` | <b>int</b> | ```10``` | Number of optimization iterations |
| ```--lr``` | <b>float</b> | ```1e-3``` | Learning rate for optimization (only used for soft prompt obfuscation) |
| ```--topk``` | <b>int</b> | ```3``` | topk value for GCG (only used for hard prompt obfuscation) |
| ```--search_width``` | <b>int</b> | ```10``` | search_width value for GCG (only used for hard prompt obfuscation) |
| ```--n_replace``` | <b>int</b> | ```1``` | n_replace value for GCG (only used for hard prompt obfuscation) |
| ```--ce_weight``` | <b>float</b> | ```1.0``` | Weight for cross-entropy loss |
| ```--kl_weight``` | <b>float</b> | ```1.0``` | Weight for KL divergence loss |
| ```--consistency_loss_weight``` | <b>float</b> | ```1.0``` | Weight for consistency loss |
| ```--fluency_loss_weight``` | <b>float</b> | ```1.0``` | Weight for fluency loss |
| ```--deobfuscated_sys_prompts_filename``` | <b>str</b> | ```deobfuscated_sys_prompt_list.pt``` | Filename for the output file containing deobfuscated system prompts. |

## Example Usage
This command deobfuscates the best obfuscated soft system prompt saved in `results/obfuscation/` using hard prompt fluency optimization:
```
python3 fluency_deobfuscation.py --results_dir results/obfuscation/ --embedding_file results/obfuscation/best_candidate.pt --deobfuscation_method hard --batch_size 4 --dataset_size 800 --output_token_count 15 --window_size 5 --optimizer_iter 10 --topk 3 --search_width 10 --n_replace 1 --ce_weight 1.0 --kl_weight 1.0 --consistency_loss_weight 1.0 --fluency_loss_weight 1.0 --deobfuscated_sys_prompts_filename deobfuscated_sys_prompt_list_hard.pt
```

It saves a list of deobfuscated system prompts after each iteration in `results/obfuscation/deobfuscated_sys_prompt_list_hard.pt`.

# Fluency Deobfuscation Evaluation
The `evaluate_fluency_deobfuscation.py` script finds the most similar deobfuscated system prompt to the conventional system prompt using specified metrics. This can be used to reproduce the results in Table 9 and 10.

## Arguments
| Argument | Type | Default Value | Description |
|----------|------|---------------|-------------|
| ```--help``` | - | - | Show help message for command line arguments |
| ```--results_dir``` | <b>str</b> | Required | Path to the directory where obfuscate.py saved its results. |
| ```--sys_prompt_list_file``` | <b>str</b> | Required | Path to the file containing system prompts. |
| ```--metrics``` | <b>list</b> | ```["levenshtein", "jaccard", "lcs", "cosine_similarity"]``` | List of metrics to use for evaluation. (choices: "levenshtein", "jaccard", "lcs", "cosine_similarity") |
| ```--best_candidate_filename``` | <b>str</b> | ```best_sys_prompt_candidate.pt``` | Filename for the output file containing best sys prompt. |
| ```--best_candidate_scores_filename``` | <b>str</b> | ```best_sys_prompt_candidate_scores.json``` | Filename for the output file containing best sys prompt scores. |

## Example Usage
This command finds the most similar deobfuscated system prompt in `results/obfuscation/deobfuscated_sys_prompt_list_hard.pt` to the conventional system prompt:
```
python3 evaluate_fluency_deobfuscation.py --results_dir results/obfuscation/ --sys_prompt_list_file results/obfuscation/deobfuscated_sys_prompt_list_hard.pt --metrics "levenshtein" "jaccard" "lcs" "cosine_similarity" --best_candidate_filename best_sys_prompt_candidate_hard.pt --best_candidate_scores_filename best_sys_prompt_candidate_scores_hard.json
```

It saves the most similar deobfuscated system prompt in `results/obfuscation/best_sys_prompt_candidate_hard.pt` and the scores in `results/obfuscation/best_sys_prompt_candidate_scores_hard.json`

# Helper Scripts

## `compare_output.py` Arguments
| Argument | Type | Default Value | Description |
|----------|------|---------------|-------------|
| ```--help``` | - | - | Show help message for command line arguments |
| ```--output_file_1``` | <b>str</b> | Required | Path to the file containing reference outputs. |
| ```--output_file_2``` | <b>str</b> | Required | Path to the file containing candidate outputs. |
| ```--metrics``` | <b>list</b> | ```["sacrebleu", "rouge1", "rouge2", "rougeL", "rougeLsum", "meteor", "bertscore", "cer", "nist_mt", "chrf", "cosine_similarity"]``` | List of metrics to use for evaluation. (choices: "sacrebleu", "rouge1", "rouge2", "rougeL", "rougeLsum", "meteor", "bertscore", "cer", "nist_mt", "chrf", "cosine_similarity") |
| ```--output_dir``` | <b>str</b> | Required | Path to the directory where the scores will be saved. |
| ```--scores_filename``` | <b>str</b> | ```scores.json``` | Filename for the output score file. |
| ```--seed``` | <b>int</b> | ```42``` | Seed for reproducibility. |

## `generate_output.py` Arguments
| Argument | Type | Default Value | Description |
|----------|------|---------------|-------------|
| ```--help``` | - | - | Show help message for command line arguments |
| ```--results_dir``` | <b>str</b> | Required | Path to the directory where obfuscate.py saved its results. |
| ```--dataset_file``` | <b>str</b> | Required | Path to the file containing the dataset. |
| ```--batch_size``` | <b>int</b> | ```32``` | Batch size for generating model outputs. |
| ```--output_filename``` | <b>str</b> | ```prompt_extraction_output.json``` | Filename for the output file. |
| ```--seed``` | <b>int</b> | ```None``` | Seed for reproducibility. If not set, use seed from params.json |
| ```--conventional``` | <b>bool</b> | ```False``` | Use the conventional system prompt in params.json. (One of --conventional, --system_prompt, --tensor_file, or --blank is required) |
| ```--system_prompt``` | <b>str</b> | ```None``` | Specify a custom system prompt directly as a string. (One of --conventional, --system_prompt, --tensor_file, or --blank is required) |
| ```--tensor_file``` | <b>str</b> | ```None``` | Specify a tensor file to load the system prompt from. (One of --conventional, --system_prompt, --tensor_file, or --blank is required) |
| ```--blank``` | <b>bool</b> | ```False``` | Use a blank system prompt. (One of --conventional, --system_prompt, --tensor_file, or --blank is required) |

## `compare_sys_prompts.py` Arguments
| Argument | Type | Default Value | Description |
|----------|------|---------------|-------------|
| ```--help``` | - | - | Show help message for command line arguments |
| ```--results_dir``` | <b>str</b> | Required | Path to the directory where obfuscate.py saved its results. |
| ```--metrics``` | <b>list</b> | ```["levenshtein", "jaccard", "lcs", "cosine_similarity"]``` | List of metrics to use for evaluation. (choices: "levenshtein", "jaccard", "lcs", "cosine_similarity") |
| ```--sys_prompt_1_conventional``` | <b>bool</b> | ```False``` | Use the conventional system prompt from params.json. (One of --sys_prompt_1_conventional, --sys_prompt_1_file, --sys_prompt_1_string, or --sys_prompt_1_random is required) |
| ```--sys_prompt_1_file``` | <b>str</b> | ```None``` | Path to tensor ID file containing the first system prompt. (One of --sys_prompt_1_conventional, --sys_prompt_1_file, --sys_prompt_1_string, or --sys_prompt_1_random is required) |
| ```--sys_prompt_1_string``` | <b>str</b> | ```None``` | String containing the first system prompt. (One of --sys_prompt_1_conventional, --sys_prompt_1_file, --sys_prompt_1_string, or --sys_prompt_1_random is required) |
| ```--sys_prompt_1_random``` | <b>bool</b> | ```False``` | Generate a random system prompt. (One of --sys_prompt_1_conventional, --sys_prompt_1_file, --sys_prompt_1_string, or --sys_prompt_1_random is required) |
| ```--sys_prompt_2_conventional``` | <b>bool</b> | ```False``` | Use the conventional system prompt from params.json. (One of --sys_prompt_2_conventional, --sys_prompt_2_file, --sys_prompt_2_string, or --sys_prompt_2_random is required) |
| ```--sys_prompt_2_file``` | <b>str</b> | ```None``` | Path to tensor ID file containing the second system prompt. (One of --sys_prompt_2_conventional, --sys_prompt_2_file, --sys_prompt_2_string, or --sys_prompt_2_random is required) |
| ```--sys_prompt_2_string``` | <b>str</b> | ```None``` | String containing the second system prompt. (One of --sys_prompt_2_conventional, --sys_prompt_2_file, --sys_prompt_2_string, or --sys_prompt_2_random is required) |
| ```--sys_prompt_2_random``` | <b>bool</b> | ```False``` | Generate a random system prompt. (One of --sys_prompt_2_conventional, --sys_prompt_2_file, --sys_prompt_2_string, or --sys_prompt_2_random is required) |
| ```--output_dir``` | <b>str</b> | Required | Path to the directory where the scores will be saved. |
| ```--scores_filename``` | <b>str</b> | ```scores.json``` | Filename for the output score file. |
| ```--seed``` | <b>int</b> | ```42``` | Seed for reproducibility. |