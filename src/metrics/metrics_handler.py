"""Module for handling metric computation and diversification."""

from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.data.data_utils import create_relevance_lists
from src.metrics.metrics_utils import (
    compute_average_category_ild_batched,
    compute_average_ild_batched,
    compute_average_ild_from_scores,
    evaluate_recommendation_metrics,
    load_similarity_scores,
    precompute_title_embeddings,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_baseline_metrics(
    rankings: dict[int, tuple[list[str], list[float]]],
    pos_items: list[str],
    embedder: Any = None,
    top_k: int = 10,
    use_category_ild: bool = False,
    categories_data: Any = None,
    use_similarity_scores: bool = False,
    similarity_scores_path: str | None = None,
    item_id_mapping: dict[str, int] | None = None,
    embedding_params: dict | None = None,
) -> dict[str, Any]:
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
    rankings: dict[int, tuple[list[str], list[float]]],
    embedder: Any,
    precomputed_embeddings: dict[str, np.ndarray] | None = None,
    use_similarity_scores: bool = False,
    title_to_id_mapping: dict[str, int] | None = None,
    similarity_scores_path: str | None = None,
) -> dict[int, np.ndarray]:
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


def _process_diversification_step(
    param_value: float,
    rankings: dict[int, tuple[list[str], list[float]]],
    pos_items: list[str],
    embedder: Any,
    experiment_params: dict,
    baseline_metrics: dict,
    categories_data: Any,
    similarity_matrices: dict[int, np.ndarray],
    title_to_id_mapping: dict[str, int],
    use_similarity_scores_globally: bool,
) -> dict[str, Any]:
    """
    Process a single diversification step: create diversifier, run diversification, and compute metrics.
    """
    diversifier_kwargs = {experiment_params["diversifier_param_name"]: param_value}
    if use_similarity_scores_globally:
        if title_to_id_mapping is None:
            raise ValueError(
                "title_to_id_mapping is missing in baseline_metrics when use_similarity_scores is True."
            )
        diversifier_kwargs.update(
            {
                "item_id_mapping": title_to_id_mapping,
                "similarity_scores_path": baseline_metrics["similarity_scores_path"],
                "use_similarity_scores": True,
            }
        )

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

    current_metrics = _compute_and_log_metrics(
        diversified_results,
        pos_items,
        diversifier,
        experiment_params,
        baseline_metrics,
        categories_data,
        param_value,
        title_to_id_mapping=title_to_id_mapping,
    )
    return current_metrics


def _refined_search_strategy(
    last_good_metrics: dict[str, Any],
    param_that_failed: float,
    rankings: dict[int, tuple[list[str], list[float]]],
    pos_items: list[str],
    embedder: Any,
    experiment_params: dict,
    baseline_metrics: dict,
    categories_data: Any,
    similarity_matrices: dict[int, np.ndarray],
    title_to_id_mapping: dict[str, int],
    use_similarity_scores_globally: bool,
    num_settings_tested_so_far: int,
) -> tuple[dict[str, Any], int]:
    """
    Execute the refined search strategy when NDCG drop threshold is exceeded.
    Returns the best metric found during refined search and the number of additional tests performed.
    """
    param_of_last_good = last_good_metrics[experiment_params["diversifier_param_name"]]
    original_param_step = experiment_params["param_step"]
    refined_param_step = original_param_step * 0.01

    logger.info(
        f"NDCG drop threshold exceeded at {experiment_params['diversifier_param_name']}={param_that_failed:.4f}. "
        f"Rolling back to {experiment_params['diversifier_param_name']}={param_of_last_good:.4f} and starting refined search "
        f"towards {param_that_failed:.4f} with step {refined_param_step:.6f}."
    )

    best_metric_from_refined_search = last_good_metrics
    candidate_param_for_refined_search = param_of_last_good
    additional_tests_in_refined_search = 0

    while True:
        candidate_param_for_refined_search += refined_param_step
        if (
            candidate_param_for_refined_search >= param_that_failed - 1e-9
        ):  # float comparison
            break

        additional_tests_in_refined_search += 1
        current_total_tests = (
            num_settings_tested_so_far + additional_tests_in_refined_search
        )
        logger.info(
            "Running refined diversification with %s=%.4f (Total settings tested: %d)",
            experiment_params["diversifier_param_name"],
            candidate_param_for_refined_search,
            current_total_tests,
        )

        refined_loop_metrics = _process_diversification_step(
            param_value=candidate_param_for_refined_search,
            rankings=rankings,
            pos_items=pos_items,
            embedder=embedder,
            experiment_params=experiment_params,
            baseline_metrics=baseline_metrics,
            categories_data=categories_data,
            similarity_matrices=similarity_matrices,
            title_to_id_mapping=title_to_id_mapping,
            use_similarity_scores_globally=use_similarity_scores_globally,
        )

        if (refined_loop_metrics["ndcg_drop"] / 100) > experiment_params[
            "threshold_drop"
        ]:
            break  # This refined step failed; the previous best_metric_from_refined_search is our choice
        else:
            best_metric_from_refined_search = (
                refined_loop_metrics  # This refined step is good
            )

    return best_metric_from_refined_search, additional_tests_in_refined_search


def run_diversification_loop(
    rankings: dict[int, tuple[list[str], list[float]]],
    pos_items: list[str],
    embedder: Any,
    experiment_params: dict,
    baseline_metrics: dict,
    categories_data: Any,
) -> tuple[list[dict], int]:
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
        tuple: A tuple containing:
            - list: List of result dictionaries for each parameter value that was kept.
            - int: Total number of parameter settings actually tested.
    """
    actual_results = []  # Stores the final list of metrics
    num_settings_tested = 0  # Counter for all tested settings
    param_values = np.arange(
        experiment_params["param_start"],
        experiment_params["param_end"] + experiment_params["param_step"] / 2,
        experiment_params["param_step"],
    )

    title_to_id_mapping = baseline_metrics.get("title_to_id_mapping")
    use_similarity_scores_globally = baseline_metrics.get(
        "use_similarity_scores", False
    )
    similarity_scores_path = baseline_metrics.get("similarity_scores_path")

    # Precompute similarity matrices for all users to avoid redundant calculations
    similarity_matrices = _precompute_similarity_matrices(
        rankings=rankings,
        embedder=embedder,
        precomputed_embeddings=baseline_metrics["precomputed_embeddings"],
        use_similarity_scores=use_similarity_scores_globally,
        title_to_id_mapping=title_to_id_mapping,
        similarity_scores_path=similarity_scores_path,
    )

    for param_idx, param_value in enumerate(param_values):
        num_settings_tested += 1
        logger.info(
            "Running diversification with %s=%.4f (Main loop %d/%d, Total settings tested: %d)",
            experiment_params["diversifier_param_name"],
            param_value,
            param_idx + 1,
            len(param_values),
            num_settings_tested,
        )

        current_metrics = _process_diversification_step(
            param_value=param_value,
            rankings=rankings,
            pos_items=pos_items,
            embedder=embedder,
            experiment_params=experiment_params,
            baseline_metrics=baseline_metrics,
            categories_data=categories_data,
            similarity_matrices=similarity_matrices,
            title_to_id_mapping=title_to_id_mapping,
            use_similarity_scores_globally=use_similarity_scores_globally,
        )

        ndcg_drop_exceeded = (current_metrics["ndcg_drop"] / 100) > experiment_params[
            "threshold_drop"
        ]

        if ndcg_drop_exceeded:
            if param_idx > 0 and actual_results:  # We have a previous "good" result
                best_refined_metric, additional_tests = _refined_search_strategy(
                    last_good_metrics=actual_results[-1],
                    param_that_failed=param_value,
                    rankings=rankings,
                    pos_items=pos_items,
                    embedder=embedder,
                    experiment_params=experiment_params,
                    baseline_metrics=baseline_metrics,
                    categories_data=categories_data,
                    similarity_matrices=similarity_matrices,
                    title_to_id_mapping=title_to_id_mapping,
                    use_similarity_scores_globally=use_similarity_scores_globally,
                    num_settings_tested_so_far=num_settings_tested,
                )
                num_settings_tested += additional_tests
                actual_results[-1] = best_refined_metric

                _log_early_stopping_message(
                    best_refined_metric[
                        "ndcg_drop"
                    ],  # Use the ndcg_drop from the best refined metric
                    actual_results,  # actual_results already updated
                    baseline_metrics,
                    experiment_params["use_category_ild"],
                )
                break  # Break the main param_value loop
            else:
                # First iteration (param_idx == 0) or no prior actual_results, and it exceeded threshold
                actual_results.append(current_metrics)  # Add the failing metric
                _log_early_stopping_message(
                    current_metrics["ndcg_drop"],
                    actual_results,  # Will contain just the one failing metric
                    baseline_metrics,
                    experiment_params["use_category_ild"],
                )
                break  # Break main loop
        else:
            # NDCG drop is acceptable
            actual_results.append(current_metrics)

    return actual_results, num_settings_tested


def _run_diversification(
    rankings: dict[int, tuple[list[str], list[float]]],
    diversifier: Any,
    top_k: int = 10,
    precomputed_embeddings: dict[str, np.ndarray] | None = None,
    precomputed_similarity_matrices: dict[int, np.ndarray] | None = None,
) -> dict[int, tuple[list[str], list[float]]]:
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
                for i, (title, score) in enumerate(
                    zip(titles, relevance_scores, strict=False)
                )
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
    diversified_results: dict[int, tuple[list[str], list[float]]],
    pos_items: list[str],
    diversifier: Any,
    experiment_params: dict,
    baseline_metrics: dict,
    categories_data: Any,
    param_value: float,
    title_to_id_mapping: dict[str, int] | None = None,
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
            "Baseline (Non-diversified): NDCG@%d=%.4f, MRR@%d=%.4f, ILD-sem@%d=%.4f, ILD-genre@%d=%.4f",
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
            "Baseline (Non-diversified): NDCG@%d=%.4f, MRR@%d=%.4f, ILD-sem@%d=%.4f",
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
            "%s=%.4f: NDCG@%d=%.4f (%.2f%%), MRR@%d=%.4f, ILD-sem@%d=%.4f (%.2f%%), ILD-genre@%d=%.4f (%.2f%%)",
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
            "%s=%.4f: NDCG@%d=%.4f (%.2f%%), MRR@%d=%.4f, ILD-sem@%d=%.4f (%.2f%%)",
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
    results: list[dict],
    baseline_metrics: dict,
    use_category_ild: bool,
) -> None:
    """Log message when stopping early due to NDCG drop."""
    if not results:
        return

    last_result = results[-1]
    if use_category_ild:
        logger.warning(
            "Stopping: NDCG dropped by %.4f%%. NDCG: %.4f, MRR before drop: %.4f, ILD-sem before drop: %.4f, ILD-genre before drop: %.4f",
            ndcg_drop,
            last_result["ndcg"],
            last_result["mrr"],
            last_result["emb_ild"],
            last_result["cat_ild"],
        )
    else:
        logger.warning(
            "Stopping: NDCG dropped by %.4f%%. MRR before drop: %.4f, ILD-sem before drop: %.4f",
            ndcg_drop,
            last_result["mrr"],
            last_result["emb_ild"],
        )
