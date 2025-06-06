"""Module for handling metric computation and diversification."""

import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.metrics.pairwise import cosine_similarity

from src.data.data_utils import create_relevance_lists
from src.metrics.metrics_utils import (
    compute_average_ild_batched,
    compute_average_ild_from_scores,
    compute_average_category_ild_batched,
    evaluate_recommendation_metrics,
    precompute_title_embeddings,
    load_similarity_scores,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_baseline_metrics(
    rankings: Dict[int, Tuple[List[str], List[float]]],
    pos_items: List[str],
    embedder: Any = None,
    top_k: int = 10,
    use_category_ild: bool = False,
    categories_data: Any = None,
    use_similarity_scores: bool = False,
    similarity_scores_path: str = None,
    item_id_mapping: Dict[str, int] = None,
    embedding_params: dict = None,
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
        use_similarity_scores: Whether to use precomputed similarity scores for ILD calculation.
        similarity_scores_path: Path to precomputed similarity scores.
        item_id_mapping: Mapping from item ID to title if use_similarity_scores is True.
        embedding_params: Embedding parameters (use_precomputed_embeddings: bool, precomputed_embeddings_path: str).

    Returns:
        dict: Baseline metrics including NDCG, MRR, ILD, and precomputed embeddings.
    """
    baseline_relevance = create_relevance_lists(rankings, pos_items)
    _, _, _, baseline_ndcg, baseline_mrr = evaluate_recommendation_metrics(
        baseline_relevance, top_k
    )

    precomputed_embeddings = None
    title_to_id_mapping = None

    if use_similarity_scores:
        if not item_id_mapping or not similarity_scores_path:
            raise ValueError(
                "item_id_mapping and similarity_scores_path must be provided when use_similarity_scores is True."
            )
        title_to_id_mapping = {v: k for k, v in item_id_mapping.items()}
        baseline_emb_ild = compute_average_ild_from_scores(
            rankings,
            title_to_id_mapping,
            similarity_scores_path=similarity_scores_path,
            topk=top_k,
        )
    else:
        precomputed_embeddings = precompute_title_embeddings(
            rankings, embedder, embedding_params
        )
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
        "use_similarity_scores": use_similarity_scores,
        "title_to_id_mapping": title_to_id_mapping,
        "similarity_scores_path": similarity_scores_path,
    }


def _precompute_similarity_matrices(
    rankings: Dict[int, Tuple[List[str], List[float]]],
    embedder: Any,
    precomputed_embeddings: Dict[str, np.ndarray] = None,
    use_similarity_scores: bool = False,
    title_to_id_mapping: Dict[str, int] = None,
    similarity_scores_path: str = None,
) -> Dict[int, np.ndarray]:
    """
    Precompute similarity matrices for all users to avoid redundant calculations.

    Args:
        rankings: User rankings dictionary.
        embedder: Embedder instance.
        precomputed_embeddings: Pre-computed embeddings.
        use_similarity_scores: Whether to use precomputed similarity scores.
        title_to_id_mapping: Mapping from title to ID for similarity lookup.
        similarity_scores_path: Path to similarity scores file.

    Returns:
        dict: Dictionary mapping user IDs to their precomputed similarity matrices.
    """
    sim_scores_dict = None
    if use_similarity_scores:
        if not title_to_id_mapping or not similarity_scores_path:
            raise ValueError(
                "title_to_id_mapping and similarity_scores_path must be provided "
                "when use_similarity_scores is True."
            )
        sim_scores_dict = load_similarity_scores(similarity_scores_path)

    similarity_matrices = {}

    for user_id, (titles, _) in rankings.items():
        if use_similarity_scores:
            # Use preloaded similarity scores to build similarity matrix
            sim_matrix = np.zeros((len(titles), len(titles)))
            item_ids = [title_to_id_mapping.get(title) for title in titles]

            for i in range(len(item_ids)):
                for j in range(len(item_ids)):
                    if i == j:
                        sim_matrix[i, j] = 1.0  # Self-similarity is 1
                    else:
                        id1, id2 = item_ids[i], item_ids[j]
                        if id1 is None or id2 is None:
                            sim_matrix[i, j] = 0.0
                            continue

                        # Look up similarity score in both directions
                        sim = sim_scores_dict.get((id1, id2))
                        if sim is None:
                            sim = sim_scores_dict.get((id2, id1))

                        sim_matrix[i, j] = sim if sim is not None else 0.0
        else:
            # Use embeddings-based similarity
            if precomputed_embeddings is not None:
                # Use precomputed embeddings
                try:
                    embeddings = np.stack(
                        [precomputed_embeddings[title] for title in titles]
                    )
                except KeyError as e:
                    raise ValueError(f"Missing embedding for title: {e}")
            else:
                # Compute embeddings on the fly
                if embedder is None:
                    raise ValueError(
                        "Embedder must be provided when title2embedding is None"
                    )
                embeddings = embedder.encode_batch(titles)

            sim_matrix = cosine_similarity(embeddings)

        similarity_matrices[user_id] = sim_matrix

    return similarity_matrices


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

    title_to_id_mapping = baseline_metrics.get("title_to_id_mapping")

    # Precompute similarity matrices for all users to avoid redundant calculations
    use_similarity_scores = baseline_metrics.get("use_similarity_scores", False)
    similarity_matrices = _precompute_similarity_matrices(
        rankings=rankings,
        embedder=embedder,
        precomputed_embeddings=baseline_metrics["precomputed_embeddings"],
        use_similarity_scores=use_similarity_scores,
        title_to_id_mapping=title_to_id_mapping,
        similarity_scores_path=baseline_metrics.get("similarity_scores_path"),
    )

    for param_value in param_values:
        logger.info(
            "Running diversification with %s=%.4f",
            experiment_params["diversifier_param_name"],
            param_value,
        )

        diversifier_kwargs = {experiment_params["diversifier_param_name"]: param_value}

        if baseline_metrics.get("use_similarity_scores", False):
            # Pass necessary info for similarity lookup if needed by the diversifier itself
            # even though matrix is precomputed
            if title_to_id_mapping is None:
                raise ValueError(
                    "title_to_id_mapping is missing in baseline_metrics when use_similarity_scores is True."
                )
            diversifier_kwargs.update(
                {
                    "item_id_mapping": title_to_id_mapping,
                    "similarity_scores_path": baseline_metrics[
                        "similarity_scores_path"
                    ],
                    "use_similarity_scores": True,
                }
            )
        # Initialize the diversifier *with* the use_similarity_scores flag etc.,
        # but it will use the precomputed matrix passed to diversify()
        diversifier = experiment_params["diversifier_cls"](
            embedder=embedder, **diversifier_kwargs
        )

        diversified_results = _run_diversification(
            rankings,
            diversifier,
            top_k=experiment_params["top_k"],
            precomputed_embeddings=baseline_metrics["precomputed_embeddings"],
            precomputed_similarity_matrices=similarity_matrices,
        )

        metrics = _compute_and_log_metrics(
            diversified_results,
            pos_items,
            diversifier,
            experiment_params,
            baseline_metrics,
            categories_data,
            param_value,
            title_to_id_mapping=title_to_id_mapping,
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
    precomputed_similarity_matrices: Dict[int, np.ndarray] = None,
) -> Dict[int, Tuple[List[str], List[float]]]:
    """
    Apply diversification to recommendation lists.

    Args:
        rankings: User rankings dictionary.
        diversifier: Diversifier instance.
        top_k: Number of items to keep.
        precomputed_embeddings: Pre-computed embeddings.
        precomputed_similarity_matrices: Pre-computed similarity matrices per user.

    Returns:
        dict: Diversified rankings dictionary.
    """
    diversified_dict = {}

    # Check if the diversifier expects precomputed embeddings or uses similarity scores
    # This affects what we pass to its diversify method
    use_similarity_scores = getattr(diversifier, "use_similarity_scores", False)

    title2embedding = None
    if not use_similarity_scores:
        title2embedding = precomputed_embeddings

    for user_id, (titles, relevance_scores) in rankings.items():
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

        # Get precomputed similarity matrix for this user
        sim_matrix = (
            precomputed_similarity_matrices.get(user_id)
            if precomputed_similarity_matrices
            else None
        )
        if sim_matrix is None:
            # This shouldn't happen if precomputation was done correctly, but handle defensively
            logger.warning(f"Missing precomputed similarity matrix for user {user_id}")
            # Fallback or raise error? For now, let diversify handle it (it might recompute)
            pass

        # Pass the precomputed matrix and potentially embeddings to the diversify method
        diversify_kwargs = {
            "top_k": top_k,
            "precomputed_sim_matrix": sim_matrix,
        }
        if not use_similarity_scores:
            diversify_kwargs["title2embedding"] = title2embedding

        diversified_items = diversifier.diversify(items, **diversify_kwargs)

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
    title_to_id_mapping: Dict[str, int] = None,
) -> dict:
    """Compute and log metrics for the current diversification run."""
    relevance_lists_div = create_relevance_lists(diversified_results, pos_items)
    prec, rec, hit, ndcg, mrr = evaluate_recommendation_metrics(
        relevance_lists_div, experiment_params["top_k"]
    )

    if baseline_metrics.get("use_similarity_scores", False):
        if title_to_id_mapping is None:
            raise ValueError(
                "title_to_id_mapping must be provided when use_similarity_scores is True."
            )
        emb_ild = compute_average_ild_from_scores(
            diversified_results,
            title_to_id_mapping,
            similarity_scores_path=baseline_metrics["similarity_scores_path"],
            topk=experiment_params["top_k"],
        )
    else:
        if not hasattr(diversifier, "embedder"):
            raise ValueError(
                "Diversifier instance must have an 'embedder' attribute when not using similarity scores."
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
        baseline_metrics["emb_ild"],
        baseline_metrics.get("cat_ild"),
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
    cat_ild: float | None,
    use_category_ild: bool,
    baseline_emb_ild: float,
    baseline_cat_ild: float | None,
) -> None:
    """Log metrics for the current diversification run."""
    emb_ild_change_pct = 0.0
    if baseline_emb_ild != 0:
        emb_ild_change_pct = (emb_ild - baseline_emb_ild) / baseline_emb_ild

    cat_ild_change_pct = 0.0
    if (
        use_category_ild
        and cat_ild is not None
        and baseline_cat_ild is not None
        and baseline_cat_ild != 0
    ):
        cat_ild_change_pct = (cat_ild - baseline_cat_ild) / baseline_cat_ild

    if use_category_ild:
        logger.info(
            "%s=%.4f: NDCG@%d=%.4f (%.2f%%), MRR@%d=%.4f, Emb-ILD@%d=%.4f (%.2f%%), Cat-ILD@%d=%.4f (%.2f%%)",
            param_name,
            param_value,
            top_k,
            ndcg,
            ndcg_decrease_pct * 100,
            top_k,
            mrr,
            top_k,
            emb_ild,
            emb_ild_change_pct * 100,
            top_k,
            cat_ild
            if cat_ild is not None
            else 0.0,  # Handle potential None for logging
            cat_ild_change_pct * 100,
        )
    else:
        logger.info(
            "%s=%.4f: NDCG@%d=%.4f (%.2f%%), MRR@%d=%.4f, Emb-ILD@%d=%.4f (%.2f%%)",
            param_name,
            param_value,
            top_k,
            ndcg,
            ndcg_decrease_pct * 100,
            top_k,
            mrr,
            top_k,
            emb_ild,
            emb_ild_change_pct * 100,
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
            "Stopping: NDCG dropped by %.4f%%. Best MRR: %.4f, Best Emb-ILD before drop: %.4f, Best Cat-ILD before drop: %.4f",
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
            "Stopping: NDCG dropped by %.4f%%. Best MRR: %.4f, Best Emb-ILD before drop: %.4f",
            ndcg_drop,
            max([r["mrr"] for r in results]) if results else baseline_metrics["mrr"],
            max([r["emb_ild"] for r in results])
            if results
            else baseline_metrics["emb_ild"],
        )
