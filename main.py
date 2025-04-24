import pandas as pd
import numpy as np
import argparse

from data_loading import load_data, preprocess_data, split_data
from svd_recommender import train_svd_recommender
from dl_recommender import train_dl_recommender
from evaluation import evaluate_svd_recs, evaluate_dl_recs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Movie Recommender System")
    parser.add_argument("--method", type=str, default="svd", help="Recommendation method to use")
    args = parser.parse_args()

    # 1. Load the data
    ratings_df, users_df, movies_df = load_data("ml-1m/ratings.dat", "ml-1m/users.dat", "ml-1m/movies.dat")

    # 2. Preprocess the data
    ratings_df, users_df, movies_df = preprocess_data(ratings_df, users_df, movies_df)

    # 3. Split into training and test sets for evaluation
    train_df, test_df = split_data(ratings_df, test_ratio=0.2)

    # 4. Train the recommendation model based on selected method
    if args.method == "svd":
        model, trainset, item_factors = train_svd_recommender(train_df)
        # 5. Evaluate with exhaustive reranking
        evaluate_svd_recs(
            model,
            trainset,
            item_factors,
            test_df,
            users_df,
            movies_df,
            N=5,
            candidate_pool_size=20,
            w_rel=0.6,
            w_div=0.2,
            w_fair=0.2,
        )
    elif args.method == "dl":
        model, item_factors = train_dl_recommender(train_df, epochs=5)
        # 5. Evaluate with exhaustive reranking
        evaluate_dl_recs(
            model,
            item_factors,
            test_df,
            users_df,
            movies_df,
            N=5,
            candidate_pool_size=20,
            w_rel=0.6,
            w_div=0.2,
            w_fair=0.2,
        )
    # Add elif blocks for additional methods as you implement them
    else:
        print(f"Method {args.method} not implemented yet")
