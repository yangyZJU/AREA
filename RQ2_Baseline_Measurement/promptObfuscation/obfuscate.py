import json
import logging
import numpy as np
import sys
from typing import Optional

from argparse import ArgumentParser, Namespace, RawTextHelpFormatter
from itertools import combinations, product
from math import comb
from pathlib import Path
from rich.console import Console
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

#from data.loader import load_and_prepare_dataset
import json
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from data.utils import TextDataset, create_collate_fn
from src.logging_config import setup_logging
from src.utils import *
from src.prompt_utils import *
from src.model import Model
from src.output_generation import precompute_model_outputs
from src.style_prompts import get_style_prompt



console = Console()

def get_args() -> Namespace:
    parser = ArgumentParser(
        description="Script for system prompt obfuscation.",
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Huggingface model name to use for optimization"
    )
    quantization_group = parser.add_mutually_exclusive_group(required=False)
    quantization_group.add_argument(
        "--quantize_4bit",
        action="store_true",
        help="Enable 4-bit quantization for the model. (Cannot be used with --quantize_8bit)"
    )
    quantization_group.add_argument(
        "--quantize_8bit",
        action="store_true",
        help="Enable 8-bit quantization for the model. (Cannot be used with --quantize_4bit)"
    )
    parser.add_argument(
        "--obfuscation_method",
        type=str,
        default="soft",
        choices=["soft", "hard"],
        help="Method for obfuscating the system prompt"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=4,
        help="Batch size for optimization"
    )
    parser.add_argument(
        "--dataset_size",
        type=int,
        default=100,
        help="Dataset size for optimization (80:20 split)"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="truthfulqa",
        choices = ["truthfulqa", "triviaqa", "cnn_dailymail", "samsum"],
        help="Dataset to use for optimization"    
    )
    parser.add_argument(
        "--task_hints",
        default=False,
        action="store_true",
        help="Whether to use task hints"
    )
    parser.add_argument(
        "--obf_sys_prompt_len",
        type=int,
        default=10,
        help="Length of the randomly initialized obfuscated system prompt in tokens."
    )
    parser.add_argument(
        "--output_token_count",
        type=int,
        default=15,
        help="Number of output tokens to optimize over"
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=5,
        help="Number of tokens in the context window to consider for gradient calculation"
    )
    parser.add_argument(
        "--optimizer_iter",
        type=int,
        default=10,
        help="Number of optimization iterations"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for optimization (only used for soft prompt obfuscation)"
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        help="topk value for GCG (only used for hard prompt obfuscation)"
    )
    parser.add_argument(
        "--search_width",
        type=int,
        default=10,
        help="search_width value for GCG (only used for hard prompt obfuscation)"
    )
    parser.add_argument(
        "--n_replace",
        type=int,
        default=1,
        help="n_replace value for GCG (only used for hard prompt obfuscation)"
    )
    parser.add_argument(
        "--ce_weight",
        type=float,
        default=1.0,
        help="Weight for cross-entropy loss"
    )
    parser.add_argument(
        "--kl_weight",
        type=float,
        default=1.0,
        help="Weight for KL divergence loss"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Seed for reproducibility"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/obfuscation",
        help="Output directory for obfuscation results"
    )
    args = parser.parse_args()

    return args

def obfuscate_soft_prompt(
    model_wrapper: Model,
    precomputed_probs: torch.Tensor,
    precomputed_ids: torch.Tensor,
    train_dataloader_conventional: DataLoader,
    sys_prompt_obf: torch.Tensor,
    original_sys_prompt_ids: torch.Tensor,
    obf_sys_prompt_len: int,
    lr: float,
    optimizer_iter: int,
    token_windows: List[List[int]],
    ce_weight: float,
    kl_weight: float
) -> Tuple[List[torch.Tensor], List[float]]:
    """
    Obfuscate the system prompt using an embedded soft prompt.

    Args:
        model_wrapper (Model) - Model wrapper
        precomputed_probs (torch.Tensor) - Precomputed probabilities for the dataset
        precomputed_ids (torch.Tensor) - Precomputed token IDs for the dataset
        train_dataloader_conventional (DataLoader) - DataLoader for the dataset
        sys_prompt_obf (torch.Tensor) - The initial obfuscated system prompt
        original_sys_prompt_ids (torch.Tensor) - The original system prompt token IDs
        obf_sys_prompt_len (int) - Length of the obfuscated system prompt
        lr (float) - Learning rate for optimization
        optimizer_iter (int) - Number of optimization iterations
        token_windows (List[List[int]]) - List of token window indices
        ce_weight (float) - Weight for cross-entropy loss
        kl_weight (float) - Weight for KL divergence loss

    Returns:
        List[torch.Tensor] - List of obfuscated system prompts
        List[float] - List of losses
    """
    sys_prompt_obf_emb = model_wrapper.get_embeddings(sys_prompt_obf).detach()
    sys_prompt_obf_emb = sys_prompt_obf_emb.requires_grad_(True)
    optimizer = torch.optim.Adam([sys_prompt_obf_emb], eps=1e-3, lr=lr)

    history_train_loss_per_iteration = []
    pad_token_id = model_wrapper.tokenizer.pad_token_id
    sys_prompt_emb_list = []

    total_train_samples = precomputed_ids.shape[1]
    # To store the cumulative true token IDs from previous completed windows for each sample
    cumulative_true_ids_for_dataset = torch.empty((total_train_samples, 0), dtype=precomputed_ids.dtype)

    cumulative_tokens_offset = 0 # Keeps track of how many tokens have been processed from previous windows

    for token_window_idx, token_window_indices in enumerate(token_windows):
        console.rule(f"[bold cyan]Token Window {token_window_idx + 1}/{len(token_windows)} (Output Tokens {min(token_window_indices)+1}-{max(token_window_indices)+1})", align="center")
        # Get the target probabilities and IDs for the current window
        current_window_target_probs_full = precomputed_probs[token_window_indices, :, :]
        current_window_target_ids_full = precomputed_ids[token_window_indices, :]
        num_tokens_in_window = len(token_window_indices)

        for iteration in range(optimizer_iter):
            logger.info(f'Token Window {token_window_idx + 1}, Iteration: {iteration + 1}/{optimizer_iter}')

            iteration_accumulated_loss_scalar = 0.0
            num_batches_processed = 0
            current_sample_offset_in_dataset = 0
            gpu_memory_used = []

            for batch_idx, data_batch in tqdm(enumerate(train_dataloader_conventional), 
                                              desc=f"Win {token_window_idx+1} Iter {iteration+1}", 
                                              total=len(train_dataloader_conventional)):
                optimizer.zero_grad()

                input_ids_batch = data_batch['input_ids']
                attention_mask_batch = data_batch['attention_mask']
                current_batch_actual_size = input_ids_batch.shape[0]

                # Get the indices of the system prompt in the current batch
                sys_prompt_indices_batch = find_sys_prompt_indices_batch(
                    input_ids_batch, original_sys_prompt_ids, 
                    pad_token_id, model_wrapper.name_or_path
                )
                # Replace the system prompt with the obfuscated version
                base_embedded_input_ids = model_wrapper.get_embeddings(input_ids_batch)
                base_embedded_input_ids = replace_sys_prompt_batch(
                    sys_prompt_obf_emb, base_embedded_input_ids, sys_prompt_indices_batch
                )
                # Update the attention mask to fit the obfuscated system prompt
                base_attention_mask = update_attention_mask_batch(
                    obf_sys_prompt_len, attention_mask_batch, sys_prompt_indices_batch
                )

                #Append cumulative true tokens from PREVIOUS windows
                cumulative_true_ids_for_batch = cumulative_true_ids_for_dataset[
                    current_sample_offset_in_dataset : current_sample_offset_in_dataset + current_batch_actual_size, :
                ]

                current_embedded_input_ids_batch = base_embedded_input_ids
                current_attention_mask_batch = base_attention_mask

                #If there are previous tokens
                if cumulative_true_ids_for_batch.shape[1] > 0:
                    # Append cumulative true tokens from PREVIOUS windows
                    embedded_cumulative_tokens = model_wrapper.get_embeddings(cumulative_true_ids_for_batch)
                    current_embedded_input_ids_batch = torch.cat(
                        [current_embedded_input_ids_batch, embedded_cumulative_tokens], dim=1
                    )
                    # Append cumulative attention mask
                    attention_for_cumulative_tokens = torch.ones_like(embedded_cumulative_tokens[..., 0], dtype=current_attention_mask_batch.dtype)
                    current_attention_mask_batch = torch.cat(
                        [current_attention_mask_batch, attention_for_cumulative_tokens], dim=1
                    )

                accumulated_loss_for_window_batch_tensor = torch.tensor(0.0, requires_grad=False)
                batch_target_probs_for_window_slice = current_window_target_probs_full[:, current_sample_offset_in_dataset : current_sample_offset_in_dataset + current_batch_actual_size, :]
                batch_target_ids_for_window_slice = current_window_target_ids_full[:, current_sample_offset_in_dataset : current_sample_offset_in_dataset + current_batch_actual_size]

                temp_embedded_inputs = current_embedded_input_ids_batch.clone()
                temp_attention_mask = current_attention_mask_batch.clone()

                logits_last = None
                next_token_logits_last = None
                next_token_log_probs_last = None

                # Calculate the loss for each token in the window
                for token_step_idx in range(num_tokens_in_window):
                    logits_last  = model_wrapper.model(
                        inputs_embeds=temp_embedded_inputs.to(model_wrapper.device, non_blocking=True),
                        attention_mask=temp_attention_mask.to(model_wrapper.device, non_blocking=True),
                    ).logits.float().cpu()

                    next_token_logits_last = logits_last[:, -1, :]
                    next_token_log_probs_last = F.log_softmax(next_token_logits_last, dim=-1)

                    true_log_probs_for_token = batch_target_probs_for_window_slice[token_step_idx, :, :]
                    true_ids_for_token = batch_target_ids_for_window_slice[token_step_idx, :]

                    loss_for_token_step = loss_function_with_padding_mask(
                        pred_logits=next_token_logits_last, pred_log_probs=next_token_log_probs_last,
                        true_log_probs=true_log_probs_for_token, true_ids=true_ids_for_token,
                        kl_weight=kl_weight, ce_weight=ce_weight, pad_token_id=pad_token_id
                    )
                    accumulated_loss_for_window_batch_tensor += loss_for_token_step

                    # Update the input and attention mask for the next token
                    true_next_token_embeddings = model_wrapper.get_embeddings(true_ids_for_token)
                    temp_embedded_inputs = torch.cat(
                        [temp_embedded_inputs, true_next_token_embeddings.unsqueeze(1)], dim=1
                    )
                    attention_for_new_token = torch.ones(
                        (current_batch_actual_size, 1), dtype=temp_attention_mask.dtype
                    )
                    temp_attention_mask = torch.cat([temp_attention_mask, attention_for_new_token], dim=1)

                avg_loss_for_batch_window_tensor = accumulated_loss_for_window_batch_tensor / num_tokens_in_window

                avg_loss_for_batch_window_tensor.backward()
                optimizer.step()

                iteration_accumulated_loss_scalar += avg_loss_for_batch_window_tensor.item()
                num_batches_processed += 1
                current_sample_offset_in_dataset += current_batch_actual_size

                gpu_memory_used.append(get_gpu_utilization())

                del input_ids_batch, attention_mask_batch, sys_prompt_indices_batch
                del base_embedded_input_ids, base_attention_mask
                del cumulative_true_ids_for_batch
                if 'embedded_cumulative_tokens' in locals(): del embedded_cumulative_tokens
                if 'attention_for_cumulative_tokens' in locals(): del attention_for_cumulative_tokens
                del current_embedded_input_ids_batch, current_attention_mask_batch
                del temp_embedded_inputs, temp_attention_mask
                del batch_target_probs_for_window_slice, batch_target_ids_for_window_slice
                del accumulated_loss_for_window_batch_tensor, avg_loss_for_batch_window_tensor
                if logits_last is not None: del logits_last
                if next_token_logits_last is not None: del next_token_logits_last
                if next_token_log_probs_last is not None: del next_token_log_probs_last
            

            avg_iteration_loss = iteration_accumulated_loss_scalar / num_batches_processed if num_batches_processed > 0 else 0.0
            logger.info(f'Token Window {token_window_idx + 1}, Iteration {iteration + 1} Avg Loss: {avg_iteration_loss:.4f}')
            history_train_loss_per_iteration.append(avg_iteration_loss)


            logger.debug(f"Max GPU Utilization: {np.max(gpu_memory_used)//1024**2} MB")

            sys_prompt_emb_list.append(sys_prompt_obf_emb.clone().detach().cpu())
                

        # Append the true token IDs for the current window to the cumulative true token IDs
        true_ids_this_window_transposed = current_window_target_ids_full.transpose(0, 1)

        cumulative_true_ids_for_dataset = torch.cat(
            [cumulative_true_ids_for_dataset, true_ids_this_window_transposed], dim=1
        )
        cumulative_tokens_offset += num_tokens_in_window
        logger.info(f"Finished token window {token_window_idx + 1}. Cumulative true tokens appended: {num_tokens_in_window}. Total cumulative: {cumulative_tokens_offset}")
        del current_window_target_probs_full, current_window_target_ids_full, true_ids_this_window_transposed

    logger.info("Soft prompt obfuscation finished.")
            
    return sys_prompt_emb_list, history_train_loss_per_iteration



def calculate_max_candidates(ids_length: int, topk: int, n_replace: int):
    """Calculates the maximum number of candidates possible based on topk and n_replace."""
    if n_replace > ids_length:
        return 0
    position_combinations = comb(ids_length, n_replace)
    
    token_combinations = topk ** n_replace
    
    return position_combinations * token_combinations

def get_all_possible_candidates(
    ids: torch.Tensor,
    topk_ids: torch.Tensor,
    topk: int,
    n_replace: int
) -> torch.Tensor:
    """Generates all possible candidates for a given set of token IDs."""
    ids_length = len(ids)
    position_combinations = list(combinations(range(ids_length), n_replace))
    token_combinations = list(product(range(topk), repeat=n_replace))

    all_candidates = []
    for pos_comb in position_combinations:
        for token_comb in token_combinations:
            new_candidate = ids.clone()
            for pos, token_index in zip(pos_comb, token_comb):
                new_candidate[pos] = topk_ids[pos, token_index]
            all_candidates.append(new_candidate)

    all_candidates_tensor = torch.stack(all_candidates, dim=0)
    
    return all_candidates_tensor

def sample_candidates(
    ids: torch.Tensor,
    topk_ids: torch.Tensor,
    search_width: int,
    topk: int,
    n_replace: int
) -> torch.Tensor:
    """Samples unique candidates for a given set of token IDs."""
    candidates = ids.repeat(search_width, 1)
    seen_candidates = set()
    ids_length = len(ids)
    seen_candidates = set()

    for i in range(search_width):
        while True:
            new_candidate = ids.clone()
            unique_positions = torch.randperm(ids_length)[:n_replace]
            for pos in unique_positions:
                replacement_token_index = torch.randint(0, topk, (1,)).item()
                replacement_token = topk_ids[pos, replacement_token_index]
                new_candidate[pos] = replacement_token
            
            candidate_tuple = tuple(new_candidate.tolist())
            if(candidate_tuple not in seen_candidates):
                seen_candidates.add(candidate_tuple)
                candidates[i] = new_candidate
                break
    return candidates


def get_candidates(
    ids: torch.Tensor,
    grad: torch.Tensor,
    search_width: int,
    topk: int,
    n_replace: int
) -> torch.Tensor:
    ids_length = len(ids)
    topk_ids = (-grad).topk(topk, dim=1).indices
    max_candidates = calculate_max_candidates(ids_length, topk, n_replace)
    
    if(search_width >= max_candidates):
        candidates = get_all_possible_candidates(ids, topk_ids, topk, n_replace)
    else:
        candidates = sample_candidates(ids, topk_ids, search_width, topk, n_replace)

    return candidates



def load_and_prepare_dataset(
    json_path: str,
    split_ratio: float = 0.5,
) -> Tuple[List[str], List[str]]:
    """
    Loads a local JSON dataset, shuffles, and splits it into train/test sets.

    Args:
        json_path: Path to the JSON file (list of strings or list of dicts with 'text' key).
        split_ratio: Ratio of data to use for training (e.g., 0.8 for 80% train / 20% test).

    Returns:
        (train_texts, test_texts): two lists of strings.
    """
    # 1. Load JSON file
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2. Handle different possible structures
    if isinstance(data, dict) and "data" in data:
        data = data["data"]

    if not isinstance(data, list):
        raise ValueError("JSON file must contain a list of samples or a dict with key 'data'.")

    # 3. Extract text
    if isinstance(data[0], dict):
        if "text" not in data[0]:
            raise ValueError("Each JSON object must contain a 'text' field.")
        texts = [item["text"] for item in data]
    else:
        texts = [str(item) for item in data]

    # 4. Split into train/test
    train_texts, test_texts = train_test_split(texts, train_size=split_ratio, random_state=42)

    return train_texts, test_texts

def main(
    model_name: str,
    quantize_4bit: bool,
    quantize_8bit: bool,
    obfuscation_method: str,
    batch_size: int,
    dataset_size: int,
    dataset_name: str,
    task_hints: bool,
    obf_sys_prompt_len: int,
    output_token_count: int,
    window_size: int,
    optimizer_iter: int,
    lr: float,
    topk: int,
    search_width: int,
    n_replace: int,
    ce_weight: float,
    kl_weight: float,
    seed: int,
    output_dir: str,
) -> None:
    """
    Main function to obfuscate system prompts.

    Args:
        model_name (str) - Huggingface model name or path to use for optimization.
        quantize_4bit (bool) - If True, 4-bit quantization is enabled for the model.
        quantize_8bit (bool) - If True, 8-bit quantization is enabled for the model.
        system_prompt (str | None) - A custom system prompt string.
        style (str | None) - The name of a predefined style for the system prompt, defined in 'style_prompts.py'.
        obfuscation_method (str) - The method for obfuscating the system prompt, either "soft" or "hard".
        batch_size (int) - Batch size to use during optimization steps.
        dataset_size (int) - Total number of samples to use from the dataset.
        dataset_name (str) - Name of the dataset to use for optimization (e.g., "truthfulqa").
        task_hints (bool) - If True, task-specific hints are used.
        obf_sys_prompt_len (int) - Length of the obfuscated system prompt in tokens.
        output_token_count (int) - The number of output tokens to obfuscate over.
        window_size (int) - The size of the context window to consider for gradient calculation.
        optimizer_iter (int) - The number of optimization iterations.
        lr (float) - Learning rate for optimization (soft prompt obfuscation).
        topk (int) - topk value for GCG (hard prompt obfuscation).
        search_width (int) - search width for GCG (hard prompt obfuscation).
        n_replace (int) - n_replace value for GCG (hard prompt obfuscation).
        ce_weight (float) - Weight for the cross-entropy loss component.
        kl_weight (float) - Weight for the KL divergence loss component.
        seed (int) - Seed for random number generators to ensure reproducibility.
        output_dir (str) - Directory where obfuscation results will be saved.

    Returns:
        None
    """
    logger = logging.getLogger(__name__)

    if quantize_4bit:
        quantization_mode = "4bit"
    elif quantize_8bit:
        quantization_mode = "8bit"
    else:
        quantization_mode = None
    
    set_seed(seed)
    logger.info("Loading tokenizer and model...")
    try:
        model_wrapper = Model(model_name, quantization_mode)
    except Exception as e:
        logger.exception(f"Failed to load model '{model_name}'. Error: {e}")
        return

    logger.info("Loading dataset...")
    try:
        train_samples, test_samples = load_and_prepare_dataset(
            json_path="data_yy/Automobile_Troubleshooter.json",
            split_ratio=0.5
        )
        task_system_prompt ="You are an automobile expert specialized in troubleshooting. Diagnose problems and errors in vehicles, both visually and within engine parts, and suggest replacements or fixes. Record details such as fuel consumption and error symptoms."
    except ValueError as e:
        logger.error(f"Error during dataset preparation: {e}")
        return
    except Exception as e:
        logger.exception(f"An unexpected error occurred during dataset preparation for {dataset_name}.")
        return
    
    logger.debug(f"Example training sample: {train_samples[0]}")

    logger.info("Constructing system prompt for obfuscation...")

    
    pad_token_string = model_wrapper.tokenizer.pad_token

    conventional_sys_prompt = f"{pad_token_string}{task_system_prompt}{pad_token_string}"

    conventional_sys_prompt_ids = model_wrapper.tokenizer(
        conventional_sys_prompt, 
        return_tensors="pt", 
        add_special_tokens=False
    ).input_ids[0]
    
    logger.debug(f"Constructed system prompt for obfuscation: '{conventional_sys_prompt}'")

    logger.info(f"Initializing obfuscated system prompt with length: {obf_sys_prompt_len} tokens.")
    sys_prompt_obf = generate_random_token_sequence(obf_sys_prompt_len, model_wrapper.vocab_size)
    decoded_initial_obf = model_wrapper.tokenizer.decode(sys_prompt_obf, skip_special_tokens=False)
    logger.debug(f'Initial obfuscated system prompt IDs: {sys_prompt_obf.tolist()}')
    logger.debug(f'Decoded initial obfuscated system prompt: {decoded_initial_obf}')

   
    train_dataset = TextDataset(train_samples)

    conventional_collate_fn = create_collate_fn(
        tokenizer=model_wrapper.tokenizer,
        system_prompt=conventional_sys_prompt
    )

    train_dataloader_conventional = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=conventional_collate_fn,
        shuffle=False
    )


    logger.info("Precomputing model outputs (probs and IDs) using conventional system prompt...")
    precomputed_probs, precomputed_ids, max_generated_length = precompute_model_outputs(
        model_wrapper=model_wrapper,
        dataloader=train_dataloader_conventional,
        max_new_tokens=output_token_count
    )

    output_token_count = max_generated_length
    if window_size >= output_token_count:
        window_size = output_token_count
    
    token_windows = create_non_overlapping_windows(output_token_count, window_size)
    logger.debug(f"Number of output token windows: {len(token_windows)}")
    
    if obfuscation_method == "soft":
        logger.info("Applying soft prompt obfuscation...")
        sys_prompt_obf_list, train_loss_per_iteration = obfuscate_soft_prompt(
            model_wrapper,
            precomputed_probs,
            precomputed_ids,
            train_dataloader_conventional,
            sys_prompt_obf,
            conventional_sys_prompt_ids,
            obf_sys_prompt_len,
            lr,
            optimizer_iter,
            token_windows,
            ce_weight,
            kl_weight
        )
    
    
    params = {
        "model_name": model_name,
        "quantize_4bit": quantize_4bit,
        "quantize_8bit": quantize_8bit,
        "system_prompt": conventional_sys_prompt,
        "obfuscation_method": obfuscation_method,
        "batch_size": batch_size,
        "dataset_name": dataset_name,
        "dataset_size": dataset_size,
        "task_hints": task_hints,
        "obf_sys_prompt_len": obf_sys_prompt_len,
        "output_token_count": output_token_count,
        "window_size": window_size,
        "lr": lr,
        "optimizer_iter": optimizer_iter,
        "topk": topk,
        "search_width": search_width,
        "n_replace": n_replace,
        "ce_weight": ce_weight,
        "kl_weight": kl_weight,
        "seed": seed
    }

    output_dir_path = Path(output_dir)

    output_dir_path.mkdir(parents=True, exist_ok=True)

    with open(output_dir_path / "params.json", "w") as f:
        json.dump(params, f, indent=4)

    torch.save(sys_prompt_obf_list, output_dir_path / "obfuscated_system_prompt_list.pt")


    np.save(output_dir_path / "train_loss.npy", np.asarray(train_loss_per_iteration))


    # For prepared_data, also use the Path object
    prepared_data_dir = output_dir_path / "prepared_data"
    prepared_data_dir.mkdir(parents=True, exist_ok=True)

    train_texts_file = prepared_data_dir / "train_data.json"
    test_texts_file = prepared_data_dir / "test_data.json"

    with open(train_texts_file, "w") as f:
        json.dump(train_samples, f, indent=4)

    with open(test_texts_file, "w") as f:
        json.dump(test_samples, f, indent=4)

    logger.info(f"Results saved to {output_dir_path.resolve()}")

if __name__ == "__main__":
    setup_logging('obfuscate.log', 'DEBUG')
    logger = logging.getLogger(__name__)
    
    logger.debug("Parsing command line arguments...")
    try:
        args = get_args()
        logger.info(f"Command line arguments received: {json.dumps(vars(args), indent=2)}")
        main(**vars(args))
    except SystemExit:
        logger.warning("Exiting due to argument parsing issue (e.g., --help or invalid arguments).")
    except Exception as e:
        logger.exception(f"An critical error occurred: {e}")
    finally:
        logger.info("Done.")