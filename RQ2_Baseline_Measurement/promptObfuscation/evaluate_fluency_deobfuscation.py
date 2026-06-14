import json
import logging
import pandas as pd
import sys
import torch

from argparse import ArgumentParser, Namespace, RawTextHelpFormatter
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple

from src.logging_config import setup_logging
from src.model import Model
from src.sys_prompt_similarity import compute_sys_prompt_similarity
from src.utils import set_seed

logging.getLogger('sentence_transformers.SentenceTransformer').setLevel(logging.WARNING)

def get_args() -> Namespace:
    parser = ArgumentParser(
        description="Script for evaluating system prompt deobfuscation.",
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path to the directory where obfuscate.py saved its results."
    )
    parser.add_argument(
        "--sys_prompt_list_file",
        type=str,
        required=True,
        help="Path to the file containing system prompts."
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        choices=["levenshtein", "jaccard", "lcs", "cosine_similarity"],
        default=["levenshtein", "jaccard", "lcs", "cosine_similarity"],
        help="List of metrics to use for evaluation."
    )
    parser.add_argument(
        "--best_candidate_filename",
        type=str,
        default="best_sys_prompt_candidate.pt",
        help="Filename for the output file containing best sys prompt."
    )
    parser.add_argument(
        "--best_candidate_scores_filename",
        type=str,
        default="best_sys_prompt_candidate_scores.json",
        help="Filename for the output file containing best sys prompt scores."
    )
    args = parser.parse_args()
    return args


HIGHER_IS_BETTER = {
    "levenshtein": True,
    "jaccard": True,
    "lcs": True,
    "cosine_similarity": True,
}


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


def main(
    results_dir: str,
    sys_prompt_list_file: str,
    metrics: list[str],
    best_candidate_filename: str,
    best_candidate_scores_filename: str
):
    """
    Main function for evaluating deobfuscated system prompts
    
    Args:
        results_dir (str) - Path to the directory where obfuscate.py saved its results.
        sys_prompt_list_file (str) - Path to the file containing system prompts.
        metrics (list[str]) - List of metrics to use for evaluation.
        best_candidate_filename (str) - Filename for the output file containing best sys prompt.
        best_candidate_scores_filename (str) - Filename for the output file containing best sys prompt scores.
    """
    logger = logging.getLogger(__name__)
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        logger.error(f"Results directory not found: {results_dir}")
        sys.exit(1)

    params_file = results_dir / "params.json"
    if not params_file.exists():
        logger.error(f"params.json not found in {results_dir}")
        sys.exit(1)
    with open(params_file, "r") as f:
        params = json.load(f)
    logger.info(f"Loaded obfuscation parameters: {json.dumps(params, indent=2)}")

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

    set_seed(params["seed"])

    conventional_sys_prompt = params['system_prompt']
    pad_token_str = model_wrapper.tokenizer.pad_token
    conventional_sys_prompt = conventional_sys_prompt.replace(pad_token_str, "")

    sys_prompt_list = torch.load(sys_prompt_list_file, weights_only=True)

    scores_list = []
    logger.info(f"Finding best system prompt...")
    for sys_prompt in tqdm(sys_prompt_list):
        sys_prompt_str = model_wrapper.tokenizer.decode(sys_prompt)
        logger.info(f"Testing system prompt: {sys_prompt_str}")
        scores = compute_sys_prompt_similarity(
            sys_prompt_1=conventional_sys_prompt,
            sys_prompt_2=sys_prompt_str,
            metric_list=metrics
        )
        scores_list.append(scores)
        logger.info(f"Sys prompt similarity scores: {scores}")

    best_idx, best_scores_dict = find_best_candidate(
        candidate_scores=scores_list,
        metric_list=metrics,
        higher_is_better_map=HIGHER_IS_BETTER
    )
    best_sys_prompt = sys_prompt_list[best_idx]
    logger.info(f"Best system promp str: {model_wrapper.tokenizer.decode(best_sys_prompt)}")

    logger.info(f"Saving best system prompt to {results_dir / best_candidate_filename}")
    torch.save(best_sys_prompt, results_dir / best_candidate_filename)

    logger.info(f"Saving best system prompt scores to {results_dir / best_candidate_scores_filename}")
    with open(results_dir / best_candidate_scores_filename, "w") as f:
        json.dump(best_scores_dict, f, indent=2)


if __name__ == "__main__":
    setup_logging('evaluate_fluency_deobfuscation.log', 'DEBUG')
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