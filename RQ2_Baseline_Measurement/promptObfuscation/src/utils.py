import os
import numpy as np
import random
import torch

from math import ceil
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from tqdm import tqdm
from transformers import set_seed as huggingface_set_seed
from typing import Optional, List


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    tqdm.write(f"GPU memory occupied: {info.used//1024**2} MB.")

def get_gpu_utilization() -> int:
    """
    Gets the memory utilization of the current GPU device
    
    Returns:
        int - Memory utilization of the current GPU device in bytes
    """
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    return info.used


def set_seed(seed: Optional[int] = None) -> None:
    """
    Set all seeds to make results reproducible (deterministic mode).
    When seed is None, disables deterministic mode.
    Parameters:
        seed: Optional[int] - seed to set
    
    Returns:
        None
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        huggingface_set_seed(seed)

def create_non_overlapping_windows(total_amount, window_size) -> List[List[int]]:
    if(window_size <= 0):
        raise ValueError("Window size must be a positive integer.")
    if(total_amount < 0):
        raise ValueError("Total amount must be a non-negative integer.")

    num_windows = ceil(total_amount / window_size)
    windows = []
    for i in range(num_windows):
        start_index = i * window_size
        end_index = min(start_index + window_size, total_amount)
        window = list(range(start_index, end_index))
        windows.append(window)

    return windows


def loss_function_with_padding_mask(
    pred_logits: torch.Tensor,
    pred_log_probs: torch.Tensor,
    true_log_probs: torch.Tensor,
    true_ids: torch.Tensor,         
    kl_weight: float,
    ce_weight: float,
    pad_token_id: int
) -> torch.Tensor:
    """
    Calculates a combined Cross-Entropy and KL Divergence loss,
    ignoring positions where true_ids are pad_token_id.

    Args:
        pred_logits: Predicted logits by the model for the current token.
        pred_log_probs: Log probabilities from the model's prediction.
        true_log_probs: Target log probabilities (precomputed).
        true_ids: Target token IDs (precomputed).
        kl_weight: Weight for KL divergence loss.
        ce_weight: Weight for Cross-Entropy loss.
        pad_token_id: ID of the padding token.

    Returns:
        Combined scalar loss tensor.
    """
    non_pad_mask = (true_ids != pad_token_id)

    if not non_pad_mask.any():
        return torch.tensor(0.0, dtype=pred_logits.dtype)
    
    total_loss_val = torch.tensor(0.0, dtype=pred_logits.dtype)

    if ce_weight > 0.0:
        ce_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction='mean')
        ce_loss = ce_loss_fn(pred_logits, true_ids)
        total_loss_val += ce_weight * ce_loss

    if kl_weight > 0:
        kl_loss_fn_elementwise = torch.nn.KLDivLoss(reduction='none', log_target=True)
        elementwise_kl_div = kl_loss_fn_elementwise(pred_log_probs, true_log_probs)
        sum_kl_div_per_sample = elementwise_kl_div.sum(dim=-1)
        masked_kl_div_sum_values = sum_kl_div_per_sample[non_pad_mask]
        if masked_kl_div_sum_values.numel() > 0:
            kl_loss = masked_kl_div_sum_values.mean()
            total_loss_val += kl_weight * kl_loss
    
    return total_loss_val.float()