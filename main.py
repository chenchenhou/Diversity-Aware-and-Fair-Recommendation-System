import pandas as pd
import numpy as np
import argparse

from data_loading import load_data, preprocess_data, split_data
from svd_recommender import train_svd_recommender
from dl_recommender import train_dl_recommender
from gnn_recommender import train_gnn_recommender
from evaluation import evaluate_svd_recs, evaluate_dl_recs, evaluate_gnn_recs, compare_methods
from svd_recommender import generate_svd_candidates
from dl_recommender import generate_dl_candidates
from gnn_recommender import generate_gnn_candidates
from reranking import exhaustive_rerank


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
    elif args.method == "gnn":
        model, item_factors, graph_data = train_gnn_recommender(train_df, users_df, movies_df, epochs=5)
        # 5. Evaluate with exhaustive reranking
        evaluate_gnn_recs(
            model,
            graph_data,
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
    elif args.method == "all":
        print("Training and evaluating all recommendation methods...")

        # Train SVD model
        print("\n--- Training SVD Model ---")
        svd_model, trainset, svd_item_factors = train_svd_recommender(train_df)

        # Train DL model
        print("\n--- Training Deep Learning Model ---")
        dl_model, dl_item_factors = train_dl_recommender(train_df, epochs=5)

        # Train GNN model
        print("\n--- Training Graph Neural Network Model ---")
        gnn_model, gnn_item_factors, graph_data = train_gnn_recommender(train_df, users_df, movies_df, epochs=5)

        # Evaluate all models on a sample of users
        sample_users = test_df["UserID"].unique()[:5]  # First 5 users

        # Collect results for each method
        svd_results = []
        dl_results = []
        gnn_results = []

        print("\n=== Collecting metrics for all methods ===")

        for user in sample_users:
            print(f"Processing user {user}...")

            # SVD evaluation
            svd_pool = generate_svd_candidates(svd_model, trainset, user, movies_df, 20)
            svd_seq, svd_rel, svd_div, svd_fair, svd_genres = exhaustive_rerank(svd_pool, svd_item_factors, movies_df, 5, 0.6, 0.2, 0.2)
            svd_results.append((user, svd_rel, svd_div, svd_fair, svd_genres))

            # DL evaluation
            seen_movies = set(test_df[test_df["UserID"] == user]["MovieID"])
            dl_pool = generate_dl_candidates(dl_model, user, movies_df, seen_movies, 20)
            dl_seq, dl_rel, dl_div, dl_fair, dl_genres = exhaustive_rerank(dl_pool, dl_item_factors, movies_df, 5, 0.6, 0.2, 0.2)
            dl_results.append((user, dl_rel, dl_div, dl_fair, dl_genres))

            # GNN evaluation
            gnn_pool = generate_gnn_candidates(gnn_model, graph_data, user, movies_df, seen_movies, 20)
            gnn_seq, gnn_rel, gnn_div, gnn_fair, gnn_genres = exhaustive_rerank(gnn_pool, gnn_item_factors, movies_df, 5, 0.6, 0.2, 0.2)
            gnn_results.append((user, gnn_rel, gnn_div, gnn_fair, gnn_genres))

        # Use the compare_methods function to generate comparison stats
        print("\n=== Method Comparison Results ===")
        comparison_df = compare_methods(svd_results, dl_results, gnn_results)

    else:
        print(f"Method {args.method} not implemented yet")
