import numpy as np
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity


def exhaustive_rerank(pool: list, item_factors: dict, movies_df, N: int = 5, w_rel: float = 0.6, w_div: float = 0.2, w_fair: float = 0.2):
    """
    Evaluate all possible combinations of length N from the pool
    and return the combination with maximum total utility.
    Utility of combination is:
      w_rel * sum(rel_i) + w_div * diversity + w_fair * fairness
    where diversity = 1 - avg pairwise cosine similarity among items,
    fairness = coverage_ratio of genres in the combination.
    """
    all_genres = set(g for genres in movies_df["Genres"] for g in genres)
    best_combo, best_score, best_total_rel, best_diversity, best_fair, best_genres = None, -np.inf, -np.inf, -np.inf, -np.inf, 0

    # Generate all possible combinations
    for combo in combinations(pool, N):
        mids = [mid for mid, _ in combo]
        ests = [est for _, est in combo]

        # Calculate overall relevance
        total_rel = sum((est - 1) / 4.0 for est in ests) / N

        # Calculating diversity: average pairwise cosine similarity
        vectors = [item_factors.get(mid) for mid in mids if mid in item_factors]
        if len(vectors) >= 2:
            stack = np.vstack(vectors)
            sim_matrix = cosine_similarity(stack)
            # Get the upper triangular matrix (excluding the diagonal)
            upper_tri = sim_matrix[np.triu_indices(len(vectors), k=1)]
            avg_sim = np.mean(upper_tri) if len(upper_tri) > 0 else 0
            diversity = 1 - avg_sim
        else:
            diversity = 0.0

        # Computational Fairness: Coverage
        genres_list = [set(movies_df[movies_df["MovieID"] == mid]["Genres"].iloc[0]) for mid in mids]
        covered = set().union(*genres_list)
        fair = len(covered) / len(all_genres) if all_genres else 0.0
        genres = len(covered)

        # Calculating overall utility
        total_utility = w_rel * total_rel + w_div * diversity + w_fair * fair

        # Update best combination
        if total_utility > best_score:
            best_score = total_utility
            best_combo = combo
            best_total_rel = total_rel
            best_diversity = diversity
            best_fair = fair
            best_genres = genres

    # If the best combination is found, sort it in descending order of relevance score and return it
    if best_combo is not None:
        best_seq = [mid for mid, _ in sorted(best_combo, key=lambda x: x[1], reverse=True)]
        return best_seq, best_total_rel, best_diversity, best_fair, best_genres
    else:
        return None
