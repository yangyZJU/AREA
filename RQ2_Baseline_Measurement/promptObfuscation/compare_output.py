import json
import logging

from argparse import ArgumentParser, Namespace, RawTextHelpFormatter
from pathlib import Path

from src.logging_config import setup_logging
from src.output_similarity import compute_similarity_scores, HIGHER_IS_BETTER, AVAILABLE_METRICS, DERIVED_METRICS_SOURCES
from src.utils import set_seed


def get_args() -> Namespace:
    parser = ArgumentParser(
        description="Script for comparing outputs.",
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument(
        "--output_file_1",
        type=str,
        required=True,
        help="Path to the file containing reference outputs."
    )
    parser.add_argument(
        "--output_file_2",
        type=str,
        required=True,
        help="Path to the file containing candidate outputs."
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        choices=list(HIGHER_IS_BETTER.keys()),
        default=["sacrebleu", "rouge1", "rouge2", "rougeL", "rougeLsum", "meteor", "bertscore", "cer", "nist_mt", "chrf", "cosine_similarity"],
        help="List of metrics to use for evaluation."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the directory where the scores will be saved."
    )
    parser.add_argument(
        "--scores_filename",
        type=str,
        default="scores.json",
        help="Filename for the output score file."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for reproducibility."
    )
    args = parser.parse_args()

    valid_metrics = list(AVAILABLE_METRICS.keys()) + list(DERIVED_METRICS_SOURCES.keys())
    for metric in args.metrics:
        if metric not in valid_metrics:
            parser.error(f"Invalid metric: {metric}. Choices are: {valid_metrics}")
    return args
    

def main(
    output_file_1: str,
    output_file_2: str,
    metrics: list[str],
    output_dir: str,
    scores_filename: str,
    seed: int
) -> None:
    """
    Main function for comparing outputs.

    Args: 
        output_file_1 (str): Path to the file containing reference outputs.
        output_file_2 (str): Path to the file containing candidate outputs.
        metrics (list[str]): List of metrics to use for evaluation.
        output_dir (str): Path to the directory where the scores will be saved.
        scores_filename (str): Filename for the output score file.
        seed (int): Seed for reproducibility.
    """
    logger = logging.getLogger(__name__)

    with open(output_file_1, "r") as f:
        ref_outputs = json.load(f)
    with open(output_file_2, "r") as f:
        cand_outputs = json.load(f)

    set_seed(seed)
    scores = compute_similarity_scores(
        predictions=cand_outputs['output'],
        references=ref_outputs['output'],
        metric_list=metrics
    )
    logger.info(f"Scores: {scores}")
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True)
    output_file = output_dir / scores_filename
    with open(output_file, "w") as f:
        json.dump(scores, f, indent=4)
    
    logger.info(f"Scores saved to {output_file}")

if __name__ == "__main__":
    setup_logging('compare_output.log', 'DEBUG')
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