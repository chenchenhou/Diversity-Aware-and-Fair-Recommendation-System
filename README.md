# Diversity-Aware and Fair Recommendation System

A movie recommendation system that balances accuracy, diversity, and fairness. Final project for CMU 18-786 Spring 2025.

## Overview

This project implements a recommendation system that optimizes for multiple objectives:
- **Relevance**: Accuracy of recommendations
- **Diversity**: Variety in recommended content
- **Fairness**: Balanced representation of different movie genres

## Models

Three recommendation approaches are implemented and compared:

1. **SVD**: Matrix factorization using Singular Value Decomposition
2. **NCF**: Neural Collaborative Filtering with deep learning
3. **GNN**: Graph Neural Network modeling user-item interactions as a graph

Each model generates candidate recommendations, which are then re-ranked using an exhaustive algorithm that optimizes for diversity and fairness while maintaining relevance.

## Implementation

- Uses MovieLens 1M dataset (1 million ratings, 6,000 users, 4,000 movies)
- Two-stage recommendation: candidate generation followed by diversity-aware re-ranking
- Diversity measured by embedding dissimilarity between items
- Fairness computed as genre coverage in recommendations

## Usage

```bash
# Install requirements
pip install -r requirements.txt

# Run individual methods
python main.py --method svd
python main.py --method dl
python main.py --method gnn

# Compare all methods
python main.py --method all