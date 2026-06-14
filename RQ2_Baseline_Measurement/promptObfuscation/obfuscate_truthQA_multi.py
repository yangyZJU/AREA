#!/usr/bin/env python3
"""
Multi-process parallel runner for obfuscation tasks.
Runs multiple obfuscation jobs in parallel with different system prompts.



nohup python -u obfuscate_truthQA_multi.py --model_name meta-llama/Llama-3.1-8B-Instruct --systems_csv ./data/system_prompts.csv --num_processes 8 --gpu_ids 0 1 2 3 --quantize_4bit --obfuscation_method soft --batch_size 4 --dataset_name truthfulqa --obf_sys_prompt_len 10 --output_token_count 15 --window_size 5 --optimizer_iter 10 --lr 0.001 --ce_weight 1.0 --kl_weight 1.0 --seed 42  > logs/obfuscate_truthQA$(date +%Y%m%d-%H%M%S).log 2>&1 &



"""

import argparse
import logging
import multiprocessing as mp
import os
import pandas as pd
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def setup_logging(log_file: str = 'parallel_runner.log'):
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def load_system_ids(csv_path: str) -> List[str]:
    """
    Load system IDs from CSV file.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        List of system IDs
    """
    df = pd.read_csv(csv_path)
    
    if 'id' not in df.columns or 'system_prompt' not in df.columns:
        raise ValueError("CSV file must contain 'id' and 'system_prompt' columns")
    
    return [str(row['id']) for _, row in df.iterrows()]


def run_obfuscation_task(
    system_id: str,
    systems_csv: str,
    model_name: str,
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
    output_base_dir: str,
    quantize_4bit: bool,
    quantize_8bit: bool,
    gpu_id: int
) -> Tuple[str, int]:
    """
    Run a single obfuscation task.
    
    Args:
        system_id: System prompt ID to process
        systems_csv: Path to CSV file with system prompts
        model_name: Name of the model
        obfuscation_method: Obfuscation method (soft/hard)
        ... (other parameters)
        gpu_id: GPU device ID to use
        
    Returns:
        Tuple of (system_id, return_code)
    """
    logger = logging.getLogger(__name__)
    
    # Create output directory for this system ID
    output_dir = Path(output_base_dir) / f"obfuscate_truthQA_{system_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build command
    cmd = [
        sys.executable, "obfuscate_truthQA_single.py",
        "--systems_csv", systems_csv,
        "--system_id", system_id,
        "--model_name", model_name,
        "--obfuscation_method", obfuscation_method,
        "--batch_size", str(batch_size),
        "--dataset_size", str(dataset_size),
        "--dataset_name", dataset_name,
        "--obf_sys_prompt_len", str(obf_sys_prompt_len),
        "--output_token_count", str(output_token_count),
        "--window_size", str(window_size),
        "--optimizer_iter", str(optimizer_iter),
        "--lr", str(lr),
        "--topk", str(topk),
        "--search_width", str(search_width),
        "--n_replace", str(n_replace),
        "--ce_weight", str(ce_weight),
        "--kl_weight", str(kl_weight),
        "--seed", str(seed),
        "--output_dir", str(output_dir)
    ]
    
    if task_hints:
        cmd.append("--task_hints")
    
    if quantize_4bit:
        cmd.append("--quantize_4bit")
    elif quantize_8bit:
        cmd.append("--quantize_8bit")
    
    # Set GPU device
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    logger.info(f"Starting obfuscation for system_id={system_id} on GPU {gpu_id}")
    logger.debug(f"Command: {' '.join(cmd)}")
    
    try:
        # Run the process
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True
        )
        
        # Save stdout and stderr to log files
        log_dir = output_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        with open(log_dir / "stdout.log", "w") as f:
            f.write(result.stdout)
        
        with open(log_dir / "stderr.log", "w") as f:
            f.write(result.stderr)
        
        if result.returncode == 0:
            logger.info(f"Successfully completed obfuscation for system_id={system_id}")
        else:
            logger.error(f"Failed obfuscation for system_id={system_id} with return code {result.returncode}")
            logger.error(f"stderr: {result.stderr[:500]}")  # Log first 500 chars of stderr
        
        return (system_id, result.returncode)
        
    except Exception as e:
        logger.exception(f"Exception occurred while processing system_id={system_id}: {e}")
        return (system_id, -1)


def main():
    parser = argparse.ArgumentParser(
        description="Run obfuscation tasks in parallel across multiple GPUs",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--systems_csv",
        type=str,
        required=True,
        help="Path to CSV file containing system prompts with 'id' and 'system_prompt' columns"
    )
    parser.add_argument(
        "--system_ids",
        type=str,
        nargs="+",
        help="Specific system IDs to process (if not provided, processes all IDs from CSV)"
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=3,
        help="Number of parallel processes to run (default: 3)"
    )
    parser.add_argument(
        "--gpu_ids",
        type=int,
        nargs="+",
        help="List of GPU IDs to use (e.g., --gpu_ids 0 1 2). If not specified, uses 0 to num_processes-1"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Huggingface model name"
    )
    quantization_group = parser.add_mutually_exclusive_group()
    quantization_group.add_argument("--quantize_4bit", action="store_true")
    quantization_group.add_argument("--quantize_8bit", action="store_true")
    
    # Obfuscation arguments
    parser.add_argument("--obfuscation_method", type=str, default="soft", choices=["soft", "hard"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--dataset_size", type=int, default=100)
    parser.add_argument("--dataset_name", type=str, default="truthfulqa",
                       choices=["truthfulqa", "triviaqa", "cnn_dailymail", "samsum"])
    parser.add_argument("--task_hints", action="store_true")
    parser.add_argument("--obf_sys_prompt_len", type=int, default=10)
    parser.add_argument("--output_token_count", type=int, default=15)
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--optimizer_iter", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--search_width", type=int, default=10)
    parser.add_argument("--n_replace", type=int, default=1)
    parser.add_argument("--ce_weight", type=float, default=1.0)
    parser.add_argument("--kl_weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_base_dir", type=str, default="results/parallel_obfuscation")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging('parallel_runner.log')
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("Starting parallel obfuscation runner")
    logger.info("=" * 80)
    
    # Load system IDs
    try:
        if args.system_ids:
            system_ids = args.system_ids
            logger.info(f"Using specified system IDs: {system_ids}")
        else:
            system_ids = load_system_ids(args.systems_csv)
            logger.info(f"Loaded {len(system_ids)} system IDs from CSV")
    except Exception as e:
        logger.exception(f"Failed to load system IDs: {e}")
        return
    
    # Setup GPU IDs
    if args.gpu_ids:
        gpu_ids = args.gpu_ids
    else:
        gpu_ids = list(range(args.num_processes))
    
    logger.info(f"Using {args.num_processes} parallel processes with GPU IDs: {gpu_ids}")
    
    if len(gpu_ids) < args.num_processes:
        logger.warning(f"Number of GPUs ({len(gpu_ids)}) is less than number of processes ({args.num_processes})")
        logger.warning(f"Some processes will share GPUs")
    
    # Create output base directory
    Path(args.output_base_dir).mkdir(parents=True, exist_ok=True)
    
    # Prepare task arguments
    task_args_list = []
    for i, system_id in enumerate(system_ids):
        gpu_id = gpu_ids[i % len(gpu_ids)]  # Cycle through available GPUs
        task_args = (
            system_id,
            args.systems_csv,
            args.model_name,
            args.obfuscation_method,
            args.batch_size,
            args.dataset_size,
            args.dataset_name,
            args.task_hints,
            args.obf_sys_prompt_len,
            args.output_token_count,
            args.window_size,
            args.optimizer_iter,
            args.lr,
            args.topk,
            args.search_width,
            args.n_replace,
            args.ce_weight,
            args.kl_weight,
            args.seed,
            args.output_base_dir,
            args.quantize_4bit,
            args.quantize_8bit,
            gpu_id
        )
        task_args_list.append(task_args)
    
    # Run tasks in parallel
    logger.info(f"Starting parallel execution of {len(task_args_list)} tasks...")
    
    with mp.Pool(processes=args.num_processes) as pool:
        results = pool.starmap(run_obfuscation_task, task_args_list)
    
    # Report results
    logger.info("=" * 80)
    logger.info("Parallel execution completed")
    logger.info("=" * 80)
    
    successful = [r for r in results if r[1] == 0]
    failed = [r for r in results if r[1] != 0]
    
    logger.info(f"Successful: {len(successful)}/{len(results)}")
    logger.info(f"Failed: {len(failed)}/{len(results)}")
    
    if failed:
        logger.info("Failed system IDs:")
        for system_id, return_code in failed:
            logger.info(f"  - {system_id} (return code: {return_code})")
    
    logger.info(f"Results saved to: {args.output_base_dir}")
    logger.info("Done!")


if __name__ == "__main__":
    main()