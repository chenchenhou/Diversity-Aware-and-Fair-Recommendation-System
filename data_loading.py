import pandas as pd
import numpy as np


def load_data(ratings_path: str, users_path: str, movies_path: str):
    """
    Load MovieLens 1M dataset from given file paths.
    The .dat files use '::' as separator and have no header row.
    Returns Pandas DataFrames: ratings_df, users_df, movies_df.
    """
    # Define column names for each file as per the dataset description
    ratings_cols = ["UserID", "MovieID", "Rating", "Timestamp"]
    users_cols = ["UserID", "Gender", "Age", "Occupation", "Zip-code"]
    movies_cols = ["MovieID", "Title", "Genres"]
    # Load the ratings data (UserID::MovieID::Rating::Timestamp)
    ratings_df = pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        names=ratings_cols,
        dtype={"UserID": int, "MovieID": int, "Rating": int, "Timestamp": int},
    )
    # Load the users data (UserID::Gender::Age::Occupation::Zip-code)
    users_df = pd.read_csv(
        users_path,
        sep="::",
        engine="python",
        names=users_cols,
        dtype={"UserID": int, "Gender": str, "Age": int, "Occupation": int, "Zip-code": str},
    )
    # Load the movies data (MovieID::Title::Genres)
    movies_df = pd.read_csv(
        movies_path,
        sep="::",
        engine="python",
        names=movies_cols,
        encoding="latin1",
        dtype={"MovieID": int, "Title": str, "Genres": str},
    )
    return ratings_df, users_df, movies_df


def preprocess_data(ratings_df: pd.DataFrame, users_df: pd.DataFrame, movies_df: pd.DataFrame):
    """
    Preprocess the MovieLens data:
    - Handle missing values (if any) by removing incomplete entries.
    - Normalize ratings from 1-5 to 0-1 scale.
    - Parse the genre strings into lists of genres for each movie.
    Returns the cleaned and modified DataFrames.
    """
    # 1. Remove any missing values
    ratings_df = ratings_df.dropna()
    users_df = users_df.dropna()
    movies_df = movies_df.dropna()
    # 2. Normalize ratings to [0, 1] range (1 -> 0.0, 5 -> 1.0)
    ratings_df["Rating_norm"] = (ratings_df["Rating"] - 1) / 4.0
    # 3. Parse genres into a list for each movie (split by '|')
    movies_df["Genres"] = movies_df["Genres"].apply(lambda x: x.split("|") if isinstance(x, str) else [])
    return ratings_df, users_df, movies_df


def split_data(ratings_df: pd.DataFrame, test_ratio: float = 0.2):
    """
    Split ratings data into training and test sets for evaluation.
    Ensures each user has at least one rating in train and one in test (if possible).
    Returns (train_df, test_df).
    """
    # Group by user and split each user's ratings
    train_list = []
    test_list = []
    for user_id, group in ratings_df.groupby("UserID"):
        # Shuffle the user's ratings to randomize selection for test
        user_ratings = group.sample(frac=1.0, random_state=42)
        # Determine number of test ratings for this user
        test_count = max(1, int(len(user_ratings) * test_ratio))
        if len(user_ratings) <= 1:
            # If the user has only 1 rating, keep it in training (no test data for this user)
            train_ratings = user_ratings
            test_ratings = pd.DataFrame(columns=ratings_df.columns)
        else:
            # Split the user's ratings into test and train portions
            test_ratings = user_ratings.iloc[:test_count]
            train_ratings = user_ratings.iloc[test_count:]
            # Ensure train is not empty (if test_ratio rounds up to all ratings)
            if train_ratings.empty:
                train_ratings = test_ratings.iloc[:1]
                test_ratings = test_ratings.iloc[1:]
        train_list.append(train_ratings)
        test_list.append(test_ratings)
    # Concatenate all users' splits and reset index
    train_df = pd.concat(train_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)
    return train_df, test_df
