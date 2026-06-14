import gc
import json
import logging
import numpy as np
import pandas as pd
import shutil
import sys
import torch

from argparse import ArgumentParser, Namespace, RawTextHelpFormatter
from pathlib import Path
from peft import PeftModel
from rich.console import Console
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Tuple, Dict

from data.utils import TextDataset, create_collate_fn
from src.logging_config import setup_logging
from src.output_similarity import compute_similarity_scores, HIGHER_IS_BETTER, AVAILABLE_METRICS, DERIVED_METRICS_SOURCES
from src.utils import set_seed, get_gpu_utilization

logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('filelock').setLevel(logging.WARNING)
logging.getLogger('accelerate').setLevel(logging.WARNING)
logging.getLogger('bitsandbytes').setLevel(logging.WARNING)
console = Console()

def get_args() -> Namespace:
    parser = ArgumentParser(
        description="Script for evaluating finetuned adapters.",
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path to the directory where finetune.py saved its results."
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        choices=list(HIGHER_IS_BETTER.keys()),
        default=["sacrebleu", "rouge1", "rouge2", "rougeL", "rougeLsum", "meteor", "bertscore", "cer", "nist_mt", "chrf", "cosine_similarity"],
        help="List of metrics to use for evaluation."
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=4,
        help="Batch size for generating model outputs during evaluation."
    )
    # Generation parameters for evaluation
    parser.add_argument("--max_new_tokens", type=int, default=125, help="Max new tokens for generation during evaluation.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p for sampling.")
    parser.add_argument("--top_k", type=int, default=100, help="Top-k for sampling.")
    parser.add_argument("--num_return_sequences", type=int, default=5, help="Number of sequences to return per prompt.")

    args = parser.parse_args()
    
    # Validate metrics
    valid_metrics = list(AVAILABLE_METRICS.keys()) + list(DERIVED_METRICS_SOURCES.keys())
    for metric in args.metrics:
        if metric not in valid_metrics:
            parser.error(f"Invalid metric: {metric}. Choices are: {valid_metrics}")
    return args

def find_best_candidate(
    candidate_scores: List[Dict[str, float]],
    metric_list: List[str],
    higher_is_better_map: Dict[str, bool]
) -> Tuple[int, Dict[str, float]]:
    """
    Finds the best candidate based on summed ranks of specified metrics.
    """
    if not candidate_scores:
        raise ValueError("candidate_scores list is empty.")
    if not metric_list:
        raise ValueError("metric_list is empty.")

    df = pd.DataFrame(candidate_scores)
    
    rank_metrics = [m for m in metric_list if m in df.columns]
    if not rank_metrics:
        logger.warning("None of the requested metrics for ranking are present in the scores. Cannot rank.")
        return -1, {}

    rank_df = pd.DataFrame(index=df.index, columns=metric_list, dtype=float)

    for metric in rank_metrics:
        if metric not in higher_is_better_map:
            logger.warning(f"Don't know if higher is better for metric '{metric}'. Assuming True.")
            ascending = False
        else:
            ascending = not higher_is_better_map[metric]
        
        ranks = df[metric].rank(method='min', ascending=ascending, na_option='bottom')
        rank_df[metric + '_rank'] = ranks
    
    df['rank_sum'] = rank_df[[m + '_rank' for m in rank_metrics]].sum(axis=1)
    
    best_idx = df['rank_sum'].idxmin()
    
    best_scores_dict = df.loc[best_idx, metric_list].to_dict()
    
    return int(best_idx), best_scores_dict

def generate_model_responses_manual(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    dataloader: DataLoader,
    generation_args: dict,
):
    outputs = []
    gpu_memory_used = []
    device = model.device
    for batch_data in tqdm(dataloader, desc="Generating responses"):
        input_ids_batch = batch_data['input_ids']
        attention_mask_batch = batch_data['attention_mask']

        with torch.no_grad():
            output_sequences = model.generate(
                input_ids=input_ids_batch.to(device, non_blocking=True),
                attention_mask=attention_mask_batch.to(device, non_blocking=True),
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                remove_invalid_values=True,
                **generation_args
            ).cpu()

        num_return_sequences = generation_args.get('num_return_sequences', 1)
        current_batch_size = output_sequences.shape[0] // num_return_sequences
        model_output = output_sequences.view(current_batch_size, num_return_sequences, -1)

        response_start_index = input_ids_batch.shape[1]
        decoded_outputs = []
        for output_list in model_output:
            output_list = output_list[:, response_start_index:]
            decoded_outputs.append(tokenizer.batch_decode(output_list, skip_special_tokens=True))
        
        outputs.extend(decoded_outputs)

        if torch.cuda.is_available():
            gpu_memory_used.append(get_gpu_utilization())
        
    if gpu_memory_used:
        logger.info(f'Max GPU memory occupied during output generation: {np.max(gpu_memory_used)//1024**2} MB')
    else:
        logger.debug('GPU memory usage not tracked for output generation (no CUDA).')
    return outputs

def main(
    results_dir: str,
    metrics: List[str],
    eval_batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    num_return_sequences: int,
) -> None:
    """
    Main function for evaluating finetuned adapters.

    Args:
        results_dir (str) - Path to the directory where finetune.py saved its results.
        metrics (List[str]) - List of metrics to use for evaluation.
        eval_batch_size (int) - Batch size for generating model outputs during evaluation.
        max_new_tokens (int) - Max new tokens for generation during evaluation.
        temperature (float) - Temperature for sampling.
        top_p (float) - Top-p for sampling.
        top_k (int) - Top-k for sampling.
        num_return_sequences (int) - Number of sequences to return per prompt.
    """
    logger = logging.getLogger(__name__)
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        logger.error(f"Results directory not found: {results_dir}")
        sys.exit(1)

    logger.info(f"Starting evaluation for results in: {results_dir}")
    params_file = results_dir / "params.json"
    if not params_file.exists():
        logger.error(f"params.json not found in {results_dir}")
        sys.exit(1)
    with open(params_file, "r") as f:
        params = json.load(f)
    logger.info(f"Loaded finetuning parameters: {json.dumps(params, indent=2)}")

    set_seed(params.get("seed", 42))

    quantization_mode = None
    if params.get("quantize_4bit", False):
        quantization_mode = "4bit"
    elif params.get("quantize_8bit", False):
        quantization_mode = "8bit"
    
    lora_adapters_path = results_dir / "lora_adapters"
    if not lora_adapters_path.is_dir():
        logger.error(f"lora_adapters directory not found in {results_dir}")
        sys.exit(1)

    # Have to load model manually otherwise we can not delete it later
    model_name = params["model_name"]
    logger.debug(f"Loading model: {model_name} with quantization: {quantization_mode or 'None'}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        padding_side="left",
    )
    new_pad_token = "<|pad|>"
    tokenizer.add_special_tokens({"pad_token": new_pad_token})

    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            logger.debug("CUDA capability >= 8.0, using bfloat16 for compute.")
            compute_dtype = torch.bfloat16
        else:
            logger.debug("CUDA capability < 8.0 or CUDA not available, using float16 for compute.")
            compute_dtype = torch.float16
    
    if quantization_mode == "4bit":
        logger.debug("Configuring for 4-bit quantization.")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=compute_dtype,
        )
    elif quantization_mode == "8bit":
        logger.debug("Configuring for 8-bit quantization.")
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
    else:
        logger.debug("No quantization requested or unsupported mode.")
        bnb_config = None

    model_kwargs = {
        "device_map": "cpu",
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if bnb_config:
            model_kwargs["quantization_config"] = bnb_config
    else:
        model_kwargs["torch_dtype"] = compute_dtype
    

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )
    model.to("cuda")
    model.eval()
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    model.config.pad_token_id = tokenizer.pad_token_id

    logger.info(f"Loaded model: {model_name}")

    test_data_file = results_dir / "prepared_data" / "test_data.json"
    if not test_data_file.exists():
        logger.error(f"test_data.json not found in {results_dir / 'prepared_data'}")
        sys.exit(1)
    with open(test_data_file, "r") as f:
        test_user_prompts = json.load(f)

    logger.info(f"Loaded test data with {len(test_user_prompts)} prompts.")

    pad_token_string = tokenizer.pad_token
    system_prompt = params.get("system_prompt", f"{pad_token_string}{pad_token_string}")

    generation_config_eval = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_new_tokens": max_new_tokens,
        "num_return_sequences": num_return_sequences,
    }

    logger.info("Generating reference outputs on test data using the conventional system prompt...")
    test_dataset = TextDataset(test_user_prompts)

    conventional_collate_fn = create_collate_fn(
        tokenizer=tokenizer,
        system_prompt=system_prompt,
    )

    test_dataloader_conventional = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        collate_fn=conventional_collate_fn,
        shuffle=False
    )

    conventional_sys_output = generate_model_responses_manual(
        model=model,
        tokenizer=tokenizer,
        dataloader=test_dataloader_conventional,
        generation_args=generation_config_eval
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()

    test_dataset = TextDataset(test_user_prompts)

    finetuning_collate_fn = create_collate_fn(
        tokenizer=tokenizer,
        system_prompt="",
    )

    test_dataloader_finetuning = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        collate_fn=finetuning_collate_fn,
        shuffle=False
    )

    scores_list = []
    logger.debug(f"Finding best finetuned adapter...")

    for adapter_idx, adapter_subdir in enumerate(sorted(lora_adapters_path.iterdir())):
        if adapter_subdir.is_dir():
            if adapter_idx == 3:
                break
            adapter_path = adapter_subdir
            adapter_name = adapter_subdir.name

            logger.info(f"Processing adapter: {adapter_name}")

            # Since peft is super annoying, we first have to load the full precision model, merge the adapter and then save it again
            # Then we can load it quantized and evaluate
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                torch_dtype=compute_dtype
            )
            model.to("cuda")
            model.eval()
            model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
            model.config.pad_token_id = tokenizer.pad_token_id

            peft_model = PeftModel.from_pretrained(
                model,
                str(adapter_path),
                is_trainable=False,
                device=model.device
            )

            peft_model = peft_model.merge_and_unload()
            peft_model.save_pretrained(f"{adapter_path}/peft_model")

            del model, peft_model
            gc.collect()
            torch.cuda.empty_cache()

            peft_model = AutoModelForCausalLM.from_pretrained(
                f"{adapter_path}/peft_model",
                device_map="cpu",
                quantization_config=bnb_config,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            peft_model.to("cuda")
            peft_model.eval()

            shutil.rmtree(f"{adapter_path}/peft_model")

            finetuning_output = generate_model_responses_manual(
                model=peft_model,
                tokenizer=tokenizer,
                dataloader=test_dataloader_finetuning,
                generation_args=generation_config_eval
            )

            del peft_model
            gc.collect()
            torch.cuda.empty_cache()

            logger.debug(f"Calculating similarity scores...")
            scores = compute_similarity_scores(
                predictions=finetuning_output,
                references=conventional_sys_output,
                metric_list=metrics
            )
            logger.info(f"Similarity scores: {scores}")
            scores_list.append(scores)


    best_idx, best_scores_dict = find_best_candidate(
        candidate_scores=scores_list,
        metric_list=metrics,
        higher_is_better_map=HIGHER_IS_BETTER
    )
    best_adapter = sorted(lora_adapters_path.iterdir())[best_idx].name
    best_adapter_path = sorted(lora_adapters_path.iterdir())[best_idx]
    params['best_candidate_idx'] = best_idx
    logger.info(f"Best adapter: {best_adapter} with scores: {best_scores_dict}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=compute_dtype
    )
    model.to("cuda")
    model.eval()
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    model.config.pad_token_id = tokenizer.pad_token_id

    peft_model = PeftModel.from_pretrained(
        model,
        str(best_adapter_path),
        is_trainable=False,
        device=model.device
    )
    peft_model = peft_model.merge_and_unload()
    peft_model.save_pretrained(f"{best_adapter_path}/peft_model")

    del model, peft_model
    gc.collect()
    torch.cuda.empty_cache()

    peft_model = AutoModelForCausalLM.from_pretrained(
        f"{best_adapter_path}/peft_model",
        device_map="cpu",
        quantization_config=bnb_config,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    peft_model.to("cuda")
    peft_model.eval()

    shutil.rmtree(f"{best_adapter_path}/peft_model")

    best_finetuning_output = generate_model_responses_manual(
        model=peft_model,
        tokenizer=tokenizer,
        dataloader=test_dataloader_finetuning,
        generation_args=generation_config_eval
    )

    del peft_model
    gc.collect()
    torch.cuda.empty_cache()

    conventional_output_dict = {
        'output': conventional_sys_output,
        'input': test_user_prompts,
        'generation_config': generation_config_eval,
        'seed': params['seed']
    }

    best_finetuning_output_dict = {
        'output': best_finetuning_output,
        'input': test_user_prompts,
        'generation_config': generation_config_eval,
        'seed': params['seed']
    }

    logger.debug(f"Saving best adapter...")
    shutil.copytree(str(best_adapter_path), str(results_dir / "best_adapter"))

    with open(results_dir / "best_adapter_scores.json", "w") as f:
        json.dump(best_scores_dict, f, indent=4)

    with open(results_dir / "best_adapter_output.json", "w") as f:
        json.dump(best_finetuning_output_dict, f, indent=4)

    with open(results_dir / "conventional_output.json", "w") as f:
        json.dump(conventional_output_dict, f, indent=4)

    with open(results_dir / "params.json", "w") as f:
        json.dump(params, f, indent=4)

    with open(results_dir / "all_scores.json", "w") as f:
        json.dump(scores_list, f, indent=4)

    with open(results_dir / "generation_config.json", "w") as f:
        json.dump(generation_config_eval, f, indent=4) 
    

    



if __name__ == "__main__":
    setup_logging('evaluate_finetuning.log', 'DEBUG')
    logger = logging.getLogger(__name__)

    logger.debug("Parsing command line arguments...")
    try:
        args = get_args()
        logger.info(f"Command line arguments received: {json.dumps(vars(args), indent=2)}")
        main(**vars(args))
    except SystemExit:
        logger.warning("Exiting due to argument parsing issue (e.g., --help or invalid arguments).")
    except FileNotFoundError as e:
        logger.error(f"A required file was not found: {e}")
    except Exception as e:
        logger.exception(f"An critical error occurred: {e}")
    finally:
        logger.info("Done.")