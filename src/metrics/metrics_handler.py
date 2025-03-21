"""Module for handling metric computation and diversification."""

import numpy as np
from typing import Dict, List, Tuple, Any

from src.data.data_utils import create_relevance_lists
from src.metrics.metrics_utils import (
    compute_average_ild_batched,
    compute_average_category_ild_batched,
    evaluate_recommendation_metrics,
)
from src.utils.utils import precompute_title_embeddings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_baseline_metrics(
    rankings: Dict[int, Tuple[List[str], List[float]]],
    pos_items: List[str],
    embedder: Any,
    top_k: int,
    use_category_ild: bool,
    categories_data: Any,
) -> Dict[str, Any]:
    """
    Compute baseline metrics for non-diversified rankings.

    Args:
        rankings: User rankings dictionary.
        pos_items: List of positive items.
        embedder: Embedder instance.
        top_k: Number of top items to consider.
        use_category_ild: Whether to compute category-based ILD.
        categories_data: Category information if use_category_ild is True.

    Returns:
        dict: Baseline metrics including NDCG, MRR, ILD, and precomputed embeddings.
    """
    baseline_relevance = create_relevance_lists(rankings, pos_items)
    baseline_prec, baseline_rec, baseline_hit, baseline_ndcg, baseline_mrr = (
        evaluate_recommendation_metrics(baseline_relevance, top_k)
    )

    precomputed_embeddings = precompute_title_embeddings(rankings, embedder)
    baseline_emb_ild = compute_average_ild_batched(
        rankings,
        embedder,
        topk=top_k,
        precomputed_embeddings=precomputed_embeddings,
    )

    baseline_cat_ild = None
    if use_category_ild:
        baseline_cat_ild = compute_average_category_ild_batched(
            rankings,
            categories_data,
            topk=top_k,
        )

    _log_baseline_metrics(
        baseline_ndcg,
        baseline_mrr,
        baseline_emb_ild,
        baseline_cat_ild,
        top_k,
        use_category_ild,
    )

    return {
        "ndcg": baseline_ndcg,
        "mrr": baseline_mrr,
        "emb_ild": baseline_emb_ild,
        "cat_ild": baseline_cat_ild,
        "precomputed_embeddings": precomputed_embeddings,
    }


def run_diversification_loop(
    rankings: Dict[int, Tuple[List[str], List[float]]],
    pos_items: List[str],
    embedder: Any,
    experiment_params: dict,
    baseline_metrics: dict,
    categories_data: Any,
) -> List[dict]:
    """
    Run the main diversification loop with varying parameters.

    Args:
        rankings: User rankings dictionary.
        pos_items: List of positive items.
        embedder: Embedder instance.
        experiment_params: Experiment parameters.
        baseline_metrics: Baseline metrics for comparison.
        categories_data: Category information if needed.

    Returns:
        list: List of result dictionaries for each parameter value.
    """
    results = []
    param_values = np.arange(
        experiment_params["param_start"],
        experiment_params["param_end"] + experiment_params["param_step"] / 2,
        experiment_params["param_step"],
    )

    for param_value in param_values:
        logger.info(
            "Running diversification with %s=%.2f",
            experiment_params["diversifier_param_name"],
            param_value,
        )

        diversifier = experiment_params["diversifier_cls"](
            embedder=embedder,
            **{experiment_params["diversifier_param_name"]: param_value},
        )

        diversified_results = _run_diversification(
            rankings,
            diversifier,
            top_k=experiment_params["top_k"],
            precomputed_embeddings=baseline_metrics["precomputed_embeddings"],
        )

        metrics = _compute_and_log_metrics(
            diversified_results,
            pos_items,
            diversifier,
            experiment_params,
            baseline_metrics,
            categories_data,
            param_value,
        )

        results.append(metrics)

        if metrics["ndcg_drop"] / 100 > experiment_params["threshold_drop"]:
            _log_early_stopping_message(
                metrics["ndcg_drop"],
                results,
                baseline_metrics,
                experiment_params["use_category_ild"],
            )
            break

    return results


def _run_diversification(
    rankings: Dict[int, Tuple[List[str], List[float]]],
    diversifier: Any,
    top_k: int = 10,
    precomputed_embeddings: Dict[str, np.ndarray] = None,
) -> Dict[int, Tuple[List[str], List[float]]]:
    """
    Apply diversification to recommendation lists.

    Args:
        rankings: User rankings dictionary.
        diversifier: Diversifier instance.
        top_k: Number of items to keep.
        precomputed_embeddings: Pre-computed embeddings.

    Returns:
        dict: Diversified rankings dictionary.
    """
    diversified_dict = {}
    title2embedding = (
        precomputed_embeddings
        if precomputed_embeddings is not None
        else precompute_title_embeddings(rankings, diversifier.embedder)
    )
    # # Only keep random 100 users for testing
    # import random

    # rankings = {
    #     k: v
    #     for k, v in rankings.items()
    #     if k in random.sample(list(rankings.keys()), 10)
    # }
    from tqdm import tqdm
    # Process each user sequentially with progress bar
    for user_id, (titles, relevance_scores) in tqdm(
        rankings.items(), desc="Diversifying users"
    ):
        if len(titles) != len(relevance_scores):
            raise ValueError(
                f"User {user_id}: Number of titles and relevance scores do not match."
            )
        items = np.array(
            [
                [i, title, float(score)]
                for i, (title, score) in enumerate(zip(titles, relevance_scores))
            ],
            dtype=object,
        )
        diversified_items = diversifier.diversify(
            items, top_k=top_k, title2embedding=title2embedding
        )

        diversified_dict[user_id] = (
            diversified_items[:, 1].tolist(),
            [float(x) for x in diversified_items[:, 2]],
        )

    return diversified_dict


def _compute_and_log_metrics(
    diversified_results: Dict[int, Tuple[List[str], List[float]]],
    pos_items: List[str],
    diversifier: Any,
    experiment_params: dict,
    baseline_metrics: dict,
    categories_data: Any,
    param_value: float,
) -> dict:
    """Compute and log metrics for the current diversification run."""
    relevance_lists_div = create_relevance_lists(diversified_results, pos_items)
    prec, rec, hit, ndcg, mrr = evaluate_recommendation_metrics(
        relevance_lists_div, experiment_params["top_k"]
    )

    emb_ild = compute_average_ild_batched(
        diversified_results,
        diversifier.embedder,
        topk=experiment_params["top_k"],
        precomputed_embeddings=baseline_metrics["precomputed_embeddings"],
    )

    cat_ild = None
    if experiment_params["use_category_ild"]:
        cat_ild = compute_average_category_ild_batched(
            diversified_results,
            categories_data,
            topk=experiment_params["top_k"],
        )

    ndcg_decrease_pct = (baseline_metrics["ndcg"] - ndcg) / baseline_metrics["ndcg"]

    _log_diversification_metrics(
        experiment_params["diversifier_param_name"],
        param_value,
        experiment_params["top_k"],
        ndcg,
        ndcg_decrease_pct,
        mrr,
        emb_ild,
        cat_ild,
        experiment_params["use_category_ild"],
    )

    result_dict = {
        experiment_params["diversifier_param_name"]: param_value,
        "ndcg": round(ndcg, 4),
        "ndcg_drop": round(ndcg_decrease_pct * 100, 4),
        "mrr": round(mrr, 4),
        "emb_ild": round(emb_ild, 4),
        "hit": round(hit, 4),
        "recall": round(rec, 4),
        "precision": round(prec, 4),
    }
    if experiment_params["use_category_ild"]:
        result_dict["cat_ild"] = round(cat_ild, 4)

    return result_dict


def _log_baseline_metrics(
    baseline_ndcg: float,
    baseline_mrr: float,
    baseline_emb_ild: float,
    baseline_cat_ild: float,
    top_k: int,
    use_category_ild: bool,
) -> None:
    """Log baseline metrics for non-diversified rankings."""
    if use_category_ild:
        logger.info(
            "Baseline (Non-diversified): NDCG@%d=%.4f, MRR@%d=%.4f, Emb-ILD@%d=%.4f, Cat-ILD@%d=%.4f",
            top_k,
            baseline_ndcg,
            top_k,
            baseline_mrr,
            top_k,
            baseline_emb_ild,
            top_k,
            baseline_cat_ild,
        )
    else:
        logger.info(
            "Baseline (Non-diversified): NDCG@%d=%.4f, MRR@%d=%.4f, Emb-ILD@%d=%.4f",
            top_k,
            baseline_ndcg,
            top_k,
            baseline_mrr,
            top_k,
            baseline_emb_ild,
        )


def _log_diversification_metrics(
    param_name: str,
    param_value: float,
    top_k: int,
    ndcg: float,
    ndcg_decrease_pct: float,
    mrr: float,
    emb_ild: float,
    cat_ild: float,
    use_category_ild: bool,
) -> None:
    """Log metrics for the current diversification run."""
    if use_category_ild:
        logger.info(
            "%s=%.2f: NDCG@%d=%.4f (%.2f%% decrease), MRR@%d=%.4f, Emb-ILD@%d=%.4f, Cat-ILD@%d=%.4f",
            param_name,
            param_value,
            top_k,
            ndcg,
            ndcg_decrease_pct * 100,
            top_k,
            mrr,
            top_k,
            emb_ild,
            top_k,
            cat_ild,
        )
    else:
        logger.info(
            "%s=%.2f: NDCG@%d=%.4f (%.2f%% decrease), MRR@%d=%.4f, Emb-ILD@%d=%.4f",
            param_name,
            param_value,
            top_k,
            ndcg,
            ndcg_decrease_pct * 100,
            top_k,
            mrr,
            top_k,
            emb_ild,
        )


def _log_early_stopping_message(
    ndcg_drop: float,
    results: List[dict],
    baseline_metrics: dict,
    use_category_ild: bool,
) -> None:
    """Log message when stopping early due to NDCG drop."""
    if use_category_ild:
        logger.warning(
            "Stopping: NDCG dropped by %.2f%%. Best MRR: %.4f, Best Emb-ILD before drop: %.4f, Best Cat-ILD before drop: %.4f",
            ndcg_drop,
            max([r["mrr"] for r in results]) if results else baseline_metrics["mrr"],
            max([r["emb_ild"] for r in results])
            if results
            else baseline_metrics["emb_ild"],
            max([r["cat_ild"] for r in results])
            if results
            else baseline_metrics["cat_ild"],
        )
    else:
        logger.warning(
            "Stopping: NDCG dropped by %.2f%%. Best MRR: %.4f, Best Emb-ILD before drop: %.4f",
            ndcg_drop,
            max([r["mrr"] for r in results]) if results else baseline_metrics["mrr"],
            max([r["emb_ild"] for r in results])
            if results
            else baseline_metrics["emb_ild"],
        )
