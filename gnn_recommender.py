import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


class MovieLensGraphDataset(Dataset):
    """PyTorch Dataset for MovieLens with graph awareness"""

    def __init__(self, ratings_df, n_users, n_items):
        self.user_ids = ratings_df["UserID"].values
        self.movie_ids = ratings_df["MovieID"].values
        self.ratings = ratings_df["Rating_norm"].values
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


class GNNRecommender(nn.Module):
    """Graph Neural Network for movie recommendations"""

    def __init__(self, n_users, n_items, emb_dim=64, hidden_dim=128):
        super(GNNRecommender, self).__init__()

        # Embedding layers for user and item
        self.user_embedding = nn.Embedding(n_users + 1, emb_dim)  # +1 for padding
        self.item_embedding = nn.Embedding(n_items + 1, emb_dim)  # +1 for padding

        # Initialize embeddings with Xavier uniform initialization for better convergence
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # Graph convolutional layers
        self.conv1 = GCNConv(emb_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, emb_dim)

        # Prediction layers with expanded architecture
        self.fc_layers = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, user_indices, item_indices, edge_index=None, x=None):
        """
        Forward pass for either training (with graph) or inference
        """
        # Get embeddings
        user_emb = self.user_embedding(user_indices)
        item_emb = self.item_embedding(item_indices)

        # If training with graph structure
        if edge_index is not None and x is not None:
            # Graph convolution
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
            x = self.conv2(x, edge_index)

            # Update embeddings
            n_users = self.user_embedding.num_embeddings - 1
            user_emb = x[:n_users][user_indices - 1]
            item_emb = x[n_users:][item_indices - 1]

        # Concatenate user and item embeddings
        concat_emb = torch.cat([user_emb, item_emb], dim=1)

        # Prediction through MLP layers
        output = self.fc_layers(concat_emb)
        return torch.sigmoid(output).squeeze()


def build_graph_data(train_df, n_users, n_items):
    """
    Build a simple graph from the ratings data.
    Returns a PyTorch Geometric Data object.
    """
    # Create node features
    n_nodes = n_users + n_items
    x = torch.zeros((n_nodes, 64), dtype=torch.float)

    # Create edges from user->item for each rating
    edge_list = []
    edge_attr = []

    for _, row in train_df.iterrows():
        # Convert to 0-indexed
        user_idx = int(row["UserID"] - 1)
        item_idx = int(row["MovieID"] - 1) + n_users

        # Add edges in both directions
        edge_list.append([user_idx, item_idx])
        edge_list.append([item_idx, user_idx])

        # Add edge attributes (normalized ratings)
        rating = float(row["Rating_norm"])
        edge_attr.append(rating)
        edge_attr.append(rating)

    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # Create data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data


def train_gnn_recommender(train_df, users_df, movies_df, epochs=10, batch_size=1024, lr=0.001, emb_dim=64):
    """
    Train a Graph Neural Network model on the training data.
    Returns the trained model, item factors dictionary, and graph data.
    Modified to use batch training similar to NCF approach.
    """
    # Get dimensions
    n_users = train_df["UserID"].max()
    n_items = train_df["MovieID"].max()

    print(f"Building graph with {n_users} users and {n_items} items...")

    # Build graph data
    data = build_graph_data(train_df, n_users, n_items)

    # Create dataset and dataloader (similar to NCF)
    dataset = MovieLensGraphDataset(train_df, n_users, n_items)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GNNRecommender(n_users, n_items, emb_dim=emb_dim).to(device)

    # Move graph to device
    data = data.to(device)

    # Optimizer and loss - similar to NCF configuration
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Training loop - batch-based like NCF
    print(f"Starting training with {len(dataloader)} batches per epoch...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (user_ids, movie_ids, ratings) in enumerate(dataloader):
            # Move to device
            user_ids = user_ids.to(device)
            movie_ids = movie_ids.to(device)
            ratings = ratings.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(user_ids, movie_ids, data.edge_index, data.x)
            loss = criterion(outputs, ratings)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Print progress similar to NCF implementation
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {running_loss/100:.4f}")
                running_loss = 0.0

    # Extract item factors for diversity computation
    model.eval()
    item_factors = {}
    with torch.no_grad():
        # For each item, get its embedding (enhanced like NCF)
        for item_id in range(1, n_items + 1):
            item_tensor = torch.tensor([item_id], dtype=torch.long).to(device)
            item_emb = model.item_embedding(item_tensor).cpu().numpy()[0]
            item_factors[item_id] = item_emb

    return model, item_factors, data


def generate_gnn_candidates(model, data, user_id: int, movies_df: pd.DataFrame, seen_movies=None, candidate_pool_size: int = 10):
    """
    Generate candidate recommendations using the trained GNN model.
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
        user_tensor = torch.tensor([user_id], dtype=torch.long).to(device)

        for movie_id in unseen_movies:
            movie_tensor = torch.tensor([movie_id], dtype=torch.long).to(device)

            # Predict rating
            pred = model(user_tensor, movie_tensor)

            # Convert to 1-5 scale
            est = pred.item() * 4.0 + 1.0

            pool.append((movie_id, est))

    # Sort by predicted rating (descending) and return top candidates
    pool.sort(key=lambda x: x[1], reverse=True)
    return pool[:candidate_pool_size]
