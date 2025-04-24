import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class MovieLensDataset(Dataset):
    """PyTorch Dataset for MovieLens"""

    def __init__(self, ratings_df, n_users, n_items):
        self.user_ids = ratings_df["UserID"].values
        self.movie_ids = ratings_df["MovieID"].values
        self.ratings = ratings_df["Rating_norm"].values  # Use normalized ratings
        self.n_users = n_users
        self.n_items = n_items

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return (
            self.user_ids[idx],
            self.movie_ids[idx],
            torch.tensor(self.ratings[idx], dtype=torch.float),
        )


class NCF(nn.Module):
    """Neural Collaborative Filtering model"""

    def __init__(self, n_users, n_items, emb_dim=64, layers=[128, 64, 32]):
        super(NCF, self).__init__()

        # Embedding layers for user and item
        self.user_embedding = nn.Embedding(n_users + 1, emb_dim)  # +1 for padding
        self.item_embedding = nn.Embedding(n_items + 1, emb_dim)  # +1 for padding

        # MLP layers
        self.fc_layers = nn.ModuleList()
        layer_dims = [2 * emb_dim] + layers
        for i in range(len(layer_dims) - 1):
            self.fc_layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))

        # Output layer
        self.output_layer = nn.Linear(layer_dims[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embeds = self.user_embedding(user_indices)
        item_embeds = self.item_embedding(item_indices)

        # Concatenate user and item embeddings
        x = torch.cat([user_embeds, item_embeds], dim=1)

        # Apply MLP layers
        for layer in self.fc_layers:
            x = nn.ReLU()(layer(x))

        # Output prediction
        x = self.output_layer(x)
        return self.sigmoid(x).squeeze()


def train_dl_recommender(train_df, epochs=10, batch_size=256, lr=0.001):
    """
    Train a Neural Collaborative Filtering model on the training data.
    Returns the trained model and item factors dictionary.
    """
    # Get dimensions
    n_users = train_df["UserID"].max()
    n_items = train_df["MovieID"].max()

    # Create dataset and dataloader
    dataset = MovieLensDataset(train_df, n_users, n_items)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NCF(n_users, n_items).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (user_ids, movie_ids, ratings) in enumerate(dataloader):
            user_ids = user_ids.to(device)
            movie_ids = movie_ids.to(device)
            ratings = ratings.to(device)

            # Forward pass
            outputs = model(user_ids, movie_ids)
            loss = criterion(outputs, ratings)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {running_loss/100:.4f}")
                running_loss = 0.0

    # Extract item embeddings for diversity computation
    model.eval()
    item_factors = {}
    with torch.no_grad():
        for item_id in range(1, n_items + 1):
            item_tensor = torch.tensor([item_id]).to(device)
            item_embed = model.item_embedding(item_tensor).cpu().numpy()[0]
            item_factors[item_id] = item_embed

    return model, item_factors


def generate_dl_candidates(model, user_id: int, movies_df: pd.DataFrame, seen_movies=None, candidate_pool_size: int = 10):
    """
    Generate candidate recommendations using the trained deep learning model.
    Returns top-N unseen movies for the user as (movie_id, estimated_rating) tuples.
    """
    if seen_movies is None:
        seen_movies = set()

    # Get all movie IDs that the user hasn't seen
    all_movie_ids = set(movies_df["MovieID"])
    unseen_movies = all_movie_ids - seen_movies

    # Device for computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Predict ratings for all unseen movies
    pool = []
    model.eval()

    with torch.no_grad():
        user_tensor = torch.tensor([user_id]).to(device)

        for movie_id in unseen_movies:
            movie_tensor = torch.tensor([movie_id]).to(device)

            # Predict rating
            pred = model(user_tensor, movie_tensor).item()

            # Convert back to 1-5 scale for consistency with SVD
            est = pred * 4.0 + 1.0

            pool.append((movie_id, est))

    # Sort by predicted rating (descending) and return top candidates
    pool.sort(key=lambda x: x[1], reverse=True)
    return pool[:candidate_pool_size]
