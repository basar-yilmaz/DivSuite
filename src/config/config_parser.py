"""
Config Parser

This file is used to parse the config file and the command line arguments.
It also contains the mapping of diversifier names to their class names.
"""

import argparse
import yaml
from typing import Dict, Any
import json
import os

DIVERSIFIER_MAP = {
    "motley": "MotleyDiversifier",
    "mmr": "MMRDiversifier",
    "bswap": "BSwapDiversifier",
    "clt": "CLTDiversifier",
    "msd": "MaxSumDiversifier",
    "swap": "SwapDiversifier",
    "sy": "SYDiversifier",
    "gmc": "GMCDiversifier",
    "gne": "GNEDiversifier",
}

PARAM_NAME_MAP = {
    "motley": "theta_",
    "mmr": "lambda_",
    "bswap": "theta_",
    "clt": "lambda_",
    "msd": "lambda_",
    "swap": "theta_",
    "sy": "lambda_",
    "gmc": "lambda_",
    "gne": "lambda_",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diversification Experiment Configuration"
    )

    # Data paths
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="Path to config file"
    )
    parser.add_argument("--data_path", type=str, help="Base path for data")
    parser.add_argument("--item_mappings", type=str, help="Path to item mappings file")
    parser.add_argument("--test_samples", type=str, help="Path to test samples file")
    parser.add_argument("--topk_list", type=str, help="Path to top-k list file")
    parser.add_argument("--topk_scores", type=str, help="Path to top-k scores file")

    # Embedder settings
    parser.add_argument("--model_name", type=str, help="Name of the embedding model")
    parser.add_argument("--device", type=str, help="Device to run embedder on")
    parser.add_argument("--batch_size", type=int, help="Batch size for embedder")
    parser.add_argument(
        "--use_precomputed_embeddings",
        action="store_true",
        help="Whether to use precomputed embeddings",
    )
    parser.add_argument(
        "--precomputed_embeddings_path",
        type=str,
        help="Path to precomputed embeddings file",
    )

    # Experiment parameters
    parser.add_argument(
        "--diversifier",
        type=str,
        choices=list(DIVERSIFIER_MAP.keys()),
        help="Diversifier algorithm",
    )
    parser.add_argument("--param_name", type=str, help="Parameter name to vary")
    parser.add_argument("--param_start", type=float, help="Start value for parameter")
    parser.add_argument("--param_end", type=float, help="End value for parameter")
    parser.add_argument("--param_step", type=float, help="Step size for parameter")
    parser.add_argument("--threshold_drop", type=float, help="NDCG drop threshold")
    parser.add_argument("--top_k", type=int, help="Number of top items to consider")

    # Similarity score settings
    parser.add_argument(
        "--use_similarity_scores",
        action="store_true",
        help="Whether to use precomputed similarity scores",
    )
    parser.add_argument(
        "--similarity_scores_path",
        type=str,
        help="Path to the precomputed similarity scores file",
    )

    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def merge_configs(
    yaml_config: Dict[str, Any], args: argparse.Namespace
) -> Dict[str, Any]:
    """Merge YAML config with command line arguments. Arguments override YAML values."""
    config = yaml_config.copy()

    # Update data paths if provided in args
    if args.data_path:
        config["data"]["base_path"] = args.data_path
    if args.item_mappings:
        config["data"]["item_mappings"] = args.item_mappings
    if args.test_samples:
        config["data"]["test_samples"] = args.test_samples
    if args.topk_list:
        config["data"]["topk_list"] = args.topk_list
    if args.topk_scores:
        config["data"]["topk_scores"] = args.topk_scores

    # Update embedder settings if provided in args
    if args.model_name:
        config["embedder"]["model_name"] = args.model_name
    if args.device:
        config["embedder"]["device"] = args.device
    if args.batch_size:
        config["embedder"]["batch_size"] = args.batch_size
    if args.use_precomputed_embeddings:
        config["embedder"]["use_precomputed_embeddings"] = (
            args.use_precomputed_embeddings
        )
    if args.precomputed_embeddings_path:
        config["embedder"]["precomputed_embeddings_path"] = (
            args.precomputed_embeddings_path
        )

    # Update experiment parameters if provided in args
    if args.diversifier:
        config["experiment"]["diversifier"] = args.diversifier
        # Auto-set param_name if not explicitly provided
        if not args.param_name:
            config["experiment"]["param_name"] = PARAM_NAME_MAP[args.diversifier]
    if args.param_name:
        config["experiment"]["param_name"] = args.param_name
    if args.param_start is not None:
        config["experiment"]["param_start"] = args.param_start
    if args.param_end is not None:
        config["experiment"]["param_end"] = args.param_end
    if args.param_step is not None:
        config["experiment"]["param_step"] = args.param_step
    if args.threshold_drop is not None:
        config["experiment"]["threshold_drop"] = args.threshold_drop
    if args.top_k is not None:
        config["experiment"]["top_k"] = args.top_k

    # Update similarity scores settings if provided in args
    if "similarity" not in config:
        config["similarity"] = {}

    if args.use_similarity_scores:
        config["similarity"]["use_similarity_scores"] = args.use_similarity_scores
    if args.similarity_scores_path:
        config["similarity"]["similarity_scores_path"] = args.similarity_scores_path

    return config


def print_config(config: Dict[str, Any]) -> None:
    """Print configuration in a formatted manner."""
    print("\n" + "=" * 50)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 50)
    print(json.dumps(config, indent=2))
    print("=" * 50 + "\n")


def get_config() -> Dict[str, Any]:
    """Get final configuration by merging YAML and command line arguments."""
    args = parse_args()
    yaml_config = load_config(args.config)
    final_config = merge_configs(yaml_config, args)

    # Convert diversifier name to class name
    final_config["experiment"]["diversifier"] = DIVERSIFIER_MAP[
        final_config["experiment"]["diversifier"]
    ]

    # Print the configuration
    print_config(final_config)

    # --- Start Added Path Checks ---
    # Check essential data paths
    base_path = final_config["data"].get(
        "base_path", "."
    )  # Use base_path if available, else current dir
    essential_data_keys = ["item_mappings", "test_samples", "topk_list", "topk_scores"]
    for key in essential_data_keys:
        path_key = final_config["data"].get(key)
        if not path_key:
            raise ValueError(
                f"Configuration error: Missing essential data path key '{key}' in config['data']."
            )
        full_path = os.path.join(base_path, path_key)
        if not os.path.exists(full_path):
            raise FileNotFoundError(
                f"Configuration error: File not found for '{key}': {full_path}. "
                f"Please check the path in your config file ('{args.config}') or provide the correct path via command line arguments."
            )

    # Check similarity scores path if needed
    if final_config.get("similarity", {}).get("use_similarity_scores"):
        similarity_path_key = final_config.get("similarity", {}).get(
            "similarity_scores_path"
        )
        if not similarity_path_key:
            raise ValueError(
                "Configuration error: 'use_similarity_scores' is true but 'similarity_scores_path' is not provided in config['similarity']."
            )
        full_similarity_path = os.path.join(
            base_path, similarity_path_key
        )  # Assume relative to base_path if provided
        if not os.path.exists(full_similarity_path):
            raise FileNotFoundError(
                f"Configuration error: Similarity scores file not found: {full_similarity_path}. "
                f"'use_similarity_scores' is set to true, but the specified file does not exist. "
                f"Please check the path in your config file ('{args.config}') or provide the correct path via command line arguments."
            )
    if final_config.get("embedder", {}).get("use_precomputed_embeddings"):
        precomputed_embeddings_path_key = final_config.get("embedder", {}).get(
            "precomputed_embeddings_path"
        )
        if not precomputed_embeddings_path_key:
            raise ValueError(
                "Configuration error: 'use_precomputed_embeddings' is true but 'precomputed_embeddings_path' is not provided in config['embedder']."
            )
        full_precomputed_embeddings_path = os.path.join(
            base_path, precomputed_embeddings_path_key
        )
        if not os.path.exists(full_precomputed_embeddings_path):
            raise FileNotFoundError(
                f"Configuration error: Precomputed embeddings file not found: {full_precomputed_embeddings_path}. "
                f"'use_precomputed_embeddings' is set to true, but the specified file does not exist. "
                f"Please check the path in your config file ('{args.config}') or provide the correct path via command line arguments."
            )
    # --- End Added Path Checks ---

    # Use similarity and use precomputed embeddings flag cannot be true at the same time.
    if final_config.get("similarity", {}).get(
        "use_similarity_scores"
    ) and final_config.get("embedder", {}).get("use_precomputed_embeddings"):
        raise ValueError(
            "Configuration error: 'use_similarity_scores' and 'use_precomputed_embeddings' cannot be true at the same time."
        )
    # --- End Added Flag Checks ---

    return final_config
