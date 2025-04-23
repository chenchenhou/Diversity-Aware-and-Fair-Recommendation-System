import pandas as pd
from surprise import Dataset, Reader, SVD


# ------------------ Model Training ---------------------
def train_svd_recommender(train_df: pd.DataFrame):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(train_df[["UserID", "MovieID", "Rating"]], reader)
    trainset = data.build_full_trainset()
    algo = SVD(random_state=42)
    algo.fit(trainset)
    # Build item latent factor matrix: raw MovieID -> latent vector
    item_factors = {int(trainset.to_raw_iid(iid)): algo.qi[iid] for iid in trainset.all_items()}
    return algo, trainset, item_factors


# --------------- Initial Recommendations ---------------
def generate_svd_candidates(model, trainset, user_id: int, movies_df: pd.DataFrame, candidate_pool_size: int = 10):
    all_ids = set(movies_df["MovieID"])
    try:
        inner_uid = trainset.to_inner_uid(user_id)
        seen_raw = {int(trainset.to_raw_iid(i)) for i, _ in trainset.ur[inner_uid]}
    except ValueError:
        seen_raw = set()
    pool = []
    for mid in all_ids - seen_raw:
        est = model.predict(user_id, mid).est
        pool.append((mid, est))
    pool.sort(key=lambda x: x[1], reverse=True)
    return pool[:candidate_pool_size]
