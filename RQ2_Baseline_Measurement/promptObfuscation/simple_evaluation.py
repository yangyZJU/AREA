import json
import logging
import numpy as np
import pandas as pd
import sys
import torch
import csv

from argparse import ArgumentParser, Namespace, RawTextHelpFormatter
from math import ceil
from pathlib import Path
from rich.console import Console
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple


from data.utils import TextDataset, create_collate_fn
from src.logging_config import setup_logging
from src.model import Model
from src.output_generation import generate_model_responses_replace
from src.utils import set_seed



console = Console()


def get_args() -> Namespace:
    parser = ArgumentParser(
        description="Script for evaluating obfuscated system prompts.",
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path to the directory where obfuscate.py saved its results."
    )

    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
        help="Batch size for generating model outputs during evaluation."
    )
    # Generation parameters for evaluation
    parser.add_argument("--max_new_tokens", type=int, default=125, help="Max new tokens for generation during evaluation.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p for sampling.")
    parser.add_argument("--top_k", type=int, default=100, help="Top-k for sampling.")
    parser.add_argument("--num_return_sequences", type=int, default=5, help="Number of sequences to return per prompt.")

    parser.add_argument("--test_file", type=str, required=True, help="benign data file")
    args = parser.parse_args()
    
    
    return args




def main(
    results_dir: str,
    eval_batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    num_return_sequences: int,
    test_file: str,
) -> None:
    """
    Main function for evaluating obfuscated system prompts.

    Args:
        results_dir (str) - Path to the directory where obfuscate.py saved its results.
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
    logger.info(f"Loaded obfuscation parameters: {json.dumps(params, indent=2)}")


    set_seed(params.get("seed", 42))

    quantization_mode = None
    if params.get("quantize_4bit", False):
        quantization_mode = "4bit"
    elif params.get("quantize_8bit", False):
        quantization_mode = "8bit"
    
    try:
        model_wrapper = Model(params["model_name"], quantization_mode)
    except Exception as e:
        logger.exception(f"Failed to load model '{params['model_name']}'. Error: {e}")
        return
    
    logger.info(f"Loaded model: {params['model_name']}")

    obf_prompts_file = results_dir / "obfuscated_system_prompt_list.pt"
    if not obf_prompts_file.exists():
        logger.error(f"obfuscated_system_prompt_list.pt not found in {results_dir}")
        sys.exit(1)
    obfuscated_system_prompt_list = torch.load(obf_prompts_file, map_location='cpu', weights_only=True)
    logger.info(f"Loaded {len(obfuscated_system_prompt_list)} obfuscated system prompts.")
    
    #test_data_file = results_dir / "prepared_data" / "test2.json"
    test_data_file = test_file

    with open(test_data_file, "r") as f:
        test_user_prompts = json.load(f)

    logger.info(f"Loaded test data with {len(test_user_prompts)} prompts.")

    is_soft_prompt_method = (params["obfuscation_method"] == "soft")

    pad_token_string = model_wrapper.tokenizer.pad_token
    system_prompt = params.get("system_prompt", f"{pad_token_string}{pad_token_string}")

    original_system_prompt = params.get("system_prompt", f"{pad_token_string}{pad_token_string}")
    original_system_prompt = original_system_prompt.replace(pad_token_string, "")
    logger.info(f"*************original_system_prompt************** {original_system_prompt}.")

    system_prompt_ids = model_wrapper.tokenizer(
        system_prompt, 
        return_tensors="pt", 
        add_special_tokens=False
    ).input_ids[0]


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
        tokenizer=model_wrapper.tokenizer,
        system_prompt=system_prompt,
    )

    test_dataloader_conventional = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        collate_fn=conventional_collate_fn,
        shuffle=False
    )


    output_token_count = params["output_token_count"]
    window_size = params["window_size"]
    optimizer_iter = params["optimizer_iter"]
    obf_sys_prompt_len = params["obf_sys_prompt_len"]
    array_shape = (ceil(output_token_count / window_size), optimizer_iter)

    scores_list = []
    logger.debug(f"Finding best obfuscated system prompt...")
    for obf_sys_prompt_idx, obf_sys_prompt in enumerate(obfuscated_system_prompt_list):
        if obf_sys_prompt_idx > 1:
            break
        token_window, iter = np.unravel_index(obf_sys_prompt_idx, array_shape)
        console.rule(f"[bold cyan]Evaluation of obfuscated system prompt at Token Window: {token_window+1}, Iteration: {iter+1}", align="center")
        logger.debug(f"Generating output...")

    

    best_idx=0
    best_candidate = obfuscated_system_prompt_list[best_idx]
    params['best_candidate_idx'] = best_idx

    logger.info(f"Regenerating output for best obfuscated system prompt...")
    best_obf_sys_output = generate_model_responses_replace(
        model_wrapper,
        test_dataloader_conventional,
        generation_config_eval,
        best_candidate,
        system_prompt_ids,
        is_soft_prompt_method,
        obf_sys_prompt_len,
        model_wrapper.tokenizer.pad_token_id
    )

    best_obf_output_dict = {
        'output': best_obf_sys_output,
        'input': test_user_prompts,
        'generation_config': generation_config_eval,
        'seed': params['seed']
    }

    
    csv_path = results_dir / "obf_sys_output.csv"
    with open(csv_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["original_system_prompt", "test_user_prompt", "obf_sys_output"])

        for user_prompt, model_output in zip(test_user_prompts, best_obf_sys_output):
            writer.writerow([original_system_prompt, user_prompt, model_output[0]])

    logger.info(f"Saved best candidate outputs to CSV: {csv_path}")



    

if __name__ == "__main__":
    setup_logging('evaluate_obfuscation.log', 'DEBUG')
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