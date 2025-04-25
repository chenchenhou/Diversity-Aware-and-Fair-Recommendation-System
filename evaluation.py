import pandas as pd
from svd_recommender import generate_svd_candidates
from dl_recommender import generate_dl_candidates
from gnn_recommender import generate_gnn_candidates
from reranking import exhaustive_rerank


def evaluate_svd_recs(
    model,
    trainset,
    item_factors,
    test_df: pd.DataFrame,
    users_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    N: int = 5,
    candidate_pool_size: int = 10,
    w_rel: float = 0.6,
    w_div: float = 0.2,
    w_fair: float = 0.2,
):
    count = 0
    for user in test_df["UserID"].unique():
        print(f"SVD Evaluation for user: {user}")
        count = count + 1
        if count > 10:  # only output first 10 persons
            break
        pool = generate_svd_candidates(model, trainset, user, movies_df, candidate_pool_size)
        best_seq, best_total_rel, best_diversity, best_fair, best_genres = exhaustive_rerank(pool, item_factors, movies_df, N, w_rel, w_div, w_fair)
        print("Best Sequence:", best_seq)
        print("Best Total Relevance:", best_total_rel)
        print("Best Diversity:", best_diversity)
        print("Best Fairness:", best_fair)
        print("Best Genres:", best_genres)


def evaluate_dl_recs(
    model,
    item_factors,
    test_df: pd.DataFrame,
    users_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    N: int = 5,
    candidate_pool_size: int = 10,
    w_rel: float = 0.6,
    w_div: float = 0.2,
    w_fair: float = 0.2,
):
    count = 0
    for user in test_df["UserID"].unique():
        print(f"DL Evaluation for user: {user}")
        count = count + 1
        if count > 10:  # only output first 10 persons
            break

        # Get seen movies for this user
        seen_movies = set(test_df[test_df["UserID"] == user]["MovieID"])

        # Generate candidates using DL model
        pool = generate_dl_candidates(model, user, movies_df, seen_movies, candidate_pool_size)

        # Rerank with the same exhaustive method
        best_seq, best_total_rel, best_diversity, best_fair, best_genres = exhaustive_rerank(pool, item_factors, movies_df, N, w_rel, w_div, w_fair)
        print("Best Sequence:", best_seq)
        print("Best Total Relevance:", best_total_rel)
        print("Best Diversity:", best_diversity)
        print("Best Fairness:", best_fair)
        print("Best Genres:", best_genres)


def evaluate_gnn_recs(
    model,
    data,
    item_factors,
    test_df: pd.DataFrame,
    users_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    N: int = 5,
    candidate_pool_size: int = 10,
    w_rel: float = 0.6,
    w_div: float = 0.2,
    w_fair: float = 0.2,
):
    count = 0
    for user in test_df["UserID"].unique():
        print(f"GNN Evaluation for user: {user}")
        count = count + 1
        if count > 10:  # only output first 10 persons
            break

        # Get seen movies for this user
        seen_movies = set(test_df[test_df["UserID"] == user]["MovieID"])

        # Generate candidates using GNN model
        pool = generate_gnn_candidates(model, data, user, movies_df, seen_movies, candidate_pool_size)

        # Rerank with the same exhaustive method
        best_seq, best_total_rel, best_diversity, best_fair, best_genres = exhaustive_rerank(pool, item_factors, movies_df, N, w_rel, w_div, w_fair)
        print("Best Sequence:", best_seq)
        print("Best Total Relevance:", best_total_rel)
        print("Best Diversity:", best_diversity)
        print("Best Fairness:", best_fair)
        print("Best Genres:", best_genres)


def compare_methods(svd_results, dl_results, gnn_results):
    """
    Compare the performance of different recommendation methods.

    Args:
        svd_results: List of metrics for SVD method [(user_id, relevance, diversity, fairness, genres)]
        dl_results: List of metrics for DL method
        gnn_results: List of metrics for GNN method
        metrics: List of metrics to compare

    Returns:
        DataFrame with comparative statistics
    """
    # Calculate average metrics per method
    methods = ["SVD", "DL", "GNN"]
    avg_metrics = {
        "SVD": {
            "relevance": sum(r[1] for r in svd_results) / len(svd_results),
            "diversity": sum(r[2] for r in svd_results) / len(svd_results),
            "fairness": sum(r[3] for r in svd_results) / len(svd_results),
            "genres": sum(r[4] for r in svd_results) / len(svd_results),
        },
        "DL": {
            "relevance": sum(r[1] for r in dl_results) / len(dl_results),
            "diversity": sum(r[2] for r in dl_results) / len(dl_results),
            "fairness": sum(r[3] for r in dl_results) / len(dl_results),
            "genres": sum(r[4] for r in dl_results) / len(dl_results),
        },
        "GNN": {
            "relevance": sum(r[1] for r in gnn_results) / len(gnn_results),
            "diversity": sum(r[2] for r in gnn_results) / len(gnn_results),
            "fairness": sum(r[3] for r in gnn_results) / len(gnn_results),
            "genres": sum(r[4] for r in gnn_results) / len(gnn_results),
        },
    }

    # Create DataFrame for comparison
    comparison_df = pd.DataFrame(
        {
            "Method": methods,
            "Avg Relevance": [avg_metrics[m]["relevance"] for m in methods],
            "Avg Diversity": [avg_metrics[m]["diversity"] for m in methods],
            "Avg Fairness": [avg_metrics[m]["fairness"] for m in methods],
            "Avg Genres": [avg_metrics[m]["genres"] for m in methods],
        }
    )

    # Print results
    print("Method Comparison:")
    print(comparison_df.to_string(index=False))

    # Determine best method per metric
    best_methods = {}
    for metric in ["relevance", "diversity", "fairness", "genres"]:
        best_method = max(methods, key=lambda m: avg_metrics[m][metric])
        best_methods[metric] = best_method

    print("\nBest method per metric:")
    for metric, method in best_methods.items():
        print(f"- {metric.capitalize()}: {method}")

    return comparison_df
