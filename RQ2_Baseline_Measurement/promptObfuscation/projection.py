import json
import logging
import sys
import torch
import torch.nn.functional as F


from argparse import ArgumentParser, Namespace, RawTextHelpFormatter
from pathlib import Path

from src.logging_config import setup_logging
from src.model import Model
from src.utils import set_seed


def get_args() -> Namespace:
    parser = ArgumentParser(
        description="Script for projecting embeddings.",
        formatter_class=RawTextHelpFormatter
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path to the directory where obfuscate.py saved its results."
    )
    parser.add_argument(
        "--embedding_file",
        type=str,
        required=True,
        help="Path to the tensor file containing embeddings."
    )
    parser.add_argument(
        "--euclidean",
        action="store_true",
        help="Use euclidean projection."
    )
    parser.add_argument(
        "--cosine",
        action="store_true",
        help="Use cosine projection."
    )
    parser.add_argument(
        "--projected_ids_filename",
        type=str,
        default="projected_ids.pt",
        help="Filename for the output file containing projected ids."
    )
    args = parser.parse_args()
    return args


def euclidean_projection(
    embedding_layer: torch.nn.modules.sparse.Embedding, 
    prompt_embedding: torch.Tensor
) -> torch.Tensor:
    """
    Calculates the euclidean distance between the prompt embedding and the embedding layer weights,
    and returns the nearest token ids.
    """
    distances = torch.cdist(prompt_embedding.double(), embedding_layer.weight.double(), p=2)
    nearest_token_ids = torch.argmin(distances, dim=1)
    return nearest_token_ids


def cosine_projection(
    embedding_layer: torch.nn.modules.sparse.Embedding, 
    prompt_embedding: torch.Tensor
) -> torch.Tensor:
    """
    Calculates the cosine similarity between the prompt embedding and the embedding layer weights,
    and returns the nearest token ids.
    """
    cosine_similarities = F.normalize(prompt_embedding.double()) @ F.normalize(embedding_layer.weight.double()).t()
    nearest_token_ids = torch.argmin(1.0-cosine_similarities, dim=1)
    return nearest_token_ids


def main(
    results_dir: str,
    embedding_file: str,
    euclidean: bool,
    cosine: bool,
    projected_ids_filename: str
):
    """
    Main function for projecting embeddings.

    Args:
        results_dir (str) - Path to the directory where obfuscate.py saved its results.
        embedding_file (str) - Path to the tensor file containing embeddings.
        euclidean (bool) - Use euclidean projection.
        cosine (bool) - Use cosine projection.
        projected_ids_filename (str) - Filename for the output file containing projected ids.
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

    set_seed(params["seed"])

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

    embedding_tensor = torch.load(embedding_file, weights_only=True)
    embedding_layer = model_wrapper.word_embedding_layer.cpu()

    if euclidean:
        logger.info("Using euclidean projection.")
        projected_ids = euclidean_projection(embedding_layer, embedding_tensor)
    elif cosine:
        logger.info("Using cosine projection.")
        projected_ids = cosine_projection(embedding_layer, embedding_tensor)

    decoded_ids = model_wrapper.tokenizer.decode(projected_ids)
    logger.info(f"Projected ids: {projected_ids}")
    logger.info(f"Decoded ids: {decoded_ids}")

    projected_ids_path = results_dir / projected_ids_filename
    torch.save(projected_ids, projected_ids_path)
    logger.info(f"Saved projected ids to: {projected_ids_path}")

if __name__ == "__main__":
    setup_logging('projection.log', 'DEBUG')
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