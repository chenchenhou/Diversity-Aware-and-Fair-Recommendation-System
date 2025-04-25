import pandas as pd
from svd_recommender import generate_svd_candidates
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
    return_recs: bool = False,
    user_id = None,
    return_metrics: bool = False
):
    if return_recs and user_id is not None:
        # Generate recommendations for a specific user
        pool = generate_svd_candidates(model, trainset, user_id, movies_df, candidate_pool_size)
        best_seq, best_total_rel, best_diversity, best_fair, _ = exhaustive_rerank(
            pool, item_factors, movies_df, N, w_rel, w_div, w_fair
        )
        
        if return_metrics:
            return best_seq, best_total_rel, best_diversity, best_fair
        else:
            return best_seq
    
    # Original evaluation code
    precisions, recalls, ilds = [], [], []
    count = 0
    for user in test_df["UserID"].unique():
        print(user)
        count = count + 1
        if count > 10:  # only output first 10 persons
            break
        pool = generate_svd_candidates(model, trainset, user, movies_df, candidate_pool_size)
        best_seq, best_total_rel, best_diversity, best_fair, best_genres = exhaustive_rerank(
            pool, item_factors, movies_df, N, w_rel, w_div, w_fair
        )
        print("Best Sequence:", best_seq)
        print("Best Total Relevance:", best_total_rel)
        print("Best Diversity:", best_diversity)
        print("Best Fairness:", best_fair)
        print("Best Genres:", best_genres)