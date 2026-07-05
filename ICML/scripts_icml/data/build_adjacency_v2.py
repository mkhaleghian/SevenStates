"""
Build Meaningful Adjacency Matrices for EV Charging Station GNN

Better relationships than "same provider":
1. Geographic distance (existing)
2. Temporal pattern similarity (correlation-based)
3. Similar operational age
4. Similar demand level
5. K-nearest neighbors (combined features)

Usage:
    python scripts/data/build_adjacency.py

Author: EV_GNN Research Project
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

PROJECT_ROOT = os.environ.get('EV_GNN_ROOT', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in km between two coordinates."""
    from math import radians, sin, cos, sqrt, atan2
    R = 6371  # Earth radius in km
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1-a))


def build_distance_adjacency(meta, threshold_km=15.0, sigma=5.0):
    """
    A_distance: Connect stations within threshold_km.
    Weight decreases with distance (Gaussian kernel).
    """
    print("\n📍 Building Distance-Based Adjacency...")
    
    N = len(meta)
    coords = meta[['latitude', 'longitude']].values
    valid = ~np.isnan(coords).any(axis=1)
    
    A = np.zeros((N, N), dtype=np.float32)
    
    for i in range(N):
        for j in range(i+1, N):
            if valid[i] and valid[j]:
                d = haversine_distance(coords[i,0], coords[i,1], 
                                       coords[j,0], coords[j,1])
                if d < threshold_km:
                    weight = np.exp(-d**2 / (2 * sigma**2))
                    A[i, j] = weight
                    A[j, i] = weight
    
    print(f"   Stations with coordinates: {valid.sum()}/{N}")
    print(f"   Edges: {(A > 0).sum()}")
    print(f"   Threshold: {threshold_km} km, Sigma: {sigma} km")
    
    return A


def build_age_similarity_adjacency(meta, threshold_years=0.5):
    """
    A_age: Connect stations with similar operational age.
    
    Idea: Stations of similar age have:
    - Similar data availability
    - Similar "maturity" of patterns
    - Can learn from each other
    """
    print("\n📅 Building Age-Similarity Adjacency...")
    
    N = len(meta)
    ages = meta['operational_years'].values
    
    A = np.zeros((N, N), dtype=np.float32)
    
    for i in range(N):
        for j in range(i+1, N):
            age_diff = abs(ages[i] - ages[j])
            if age_diff < threshold_years:
                # Weight: closer ages = stronger connection
                weight = 1.0 - (age_diff / threshold_years)
                A[i, j] = weight
                A[j, i] = weight
    
    print(f"   Age range: {ages.min():.2f} - {ages.max():.2f} years")
    print(f"   Threshold: {threshold_years} years")
    print(f"   Edges: {(A > 0).sum()}")
    
    return A


def build_demand_correlation_adjacency(Y, M, min_overlap=100, corr_threshold=0.3):
    """
    A_correlation: Connect stations with correlated demand patterns.
    
    Idea: Stations with similar hourly patterns should share information.
    e.g., Both busy at 8am and 6pm = commuter stations
    """
    print("\n📈 Building Demand-Correlation Adjacency...")
    
    T, N = Y.shape
    A = np.zeros((N, N), dtype=np.float32)
    
    # For each pair of stations, compute correlation where both have data
    computed = 0
    significant = 0
    
    for i in tqdm(range(N), desc="   Computing correlations"):
        for j in range(i+1, N):
            # Find time points where both stations have data
            overlap_mask = (M[:, i] > 0) & (M[:, j] > 0)
            overlap_count = overlap_mask.sum()
            
            if overlap_count >= min_overlap:
                computed += 1
                y_i = Y[overlap_mask, i]
                y_j = Y[overlap_mask, j]
                
                # Pearson correlation
                if y_i.std() > 0 and y_j.std() > 0:
                    corr, _ = pearsonr(y_i, y_j)
                    
                    if corr > corr_threshold:
                        significant += 1
                        A[i, j] = corr
                        A[j, i] = corr
    
    print(f"   Pairs with sufficient overlap: {computed}")
    print(f"   Pairs with correlation > {corr_threshold}: {significant}")
    print(f"   Edges: {(A > 0).sum()}")
    
    return A


def build_demand_level_adjacency(meta, Y, M, n_bins=4):
    """
    A_demand_level: Connect stations with similar average demand.
    
    Idea: High-volume stations behave differently than low-volume.
    Connect stations in same "tier".
    """
    print("\n📊 Building Demand-Level Adjacency...")
    
    N = len(meta)
    
    # Compute average hourly demand for each station
    avg_demand = np.zeros(N)
    for i in range(N):
        valid_mask = M[:, i] > 0
        if valid_mask.sum() > 0:
            avg_demand[i] = Y[valid_mask, i].mean()
    
    # Bin stations by demand level
    percentiles = np.percentile(avg_demand[avg_demand > 0], 
                                np.linspace(0, 100, n_bins + 1))
    
    bins = np.digitize(avg_demand, percentiles[:-1])
    
    A = np.zeros((N, N), dtype=np.float32)
    
    for i in range(N):
        for j in range(i+1, N):
            if bins[i] == bins[j] and avg_demand[i] > 0 and avg_demand[j] > 0:
                # Same demand tier = connected
                # Weight by how close their actual demands are
                diff = abs(avg_demand[i] - avg_demand[j])
                max_diff = avg_demand.max() - avg_demand.min()
                weight = 1.0 - (diff / max_diff) if max_diff > 0 else 1.0
                A[i, j] = weight
                A[j, i] = weight
    
    print(f"   Demand bins: {n_bins}")
    print(f"   Avg demand range: {avg_demand.min():.2f} - {avg_demand.max():.2f}")
    print(f"   Edges: {(A > 0).sum()}")
    
    return A


def build_temporal_pattern_adjacency(Y, M, min_hours=500):
    """
    A_temporal: Connect stations with similar hourly patterns.
    
    Computes average demand by hour-of-day, then connects stations
    with similar daily profiles.
    """
    print("\n🕐 Building Temporal-Pattern Adjacency...")
    
    T, N = Y.shape
    
    # Create time index to get hour of day
    # Assuming data starts at some time and is hourly
    hours = np.arange(T) % 24  # Hour of day (0-23)
    
    # Compute average demand by hour for each station
    hourly_profiles = np.zeros((N, 24))
    
    for i in range(N):
        valid = M[:, i] > 0
        if valid.sum() >= min_hours:
            for h in range(24):
                hour_mask = (hours == h) & valid
                if hour_mask.sum() > 0:
                    hourly_profiles[i, h] = Y[hour_mask, i].mean()
    
    # Normalize profiles (0-1 scale)
    for i in range(N):
        if hourly_profiles[i].max() > 0:
            hourly_profiles[i] /= hourly_profiles[i].max()
    
    # Compute similarity (cosine similarity)
    A = np.zeros((N, N), dtype=np.float32)
    
    for i in range(N):
        for j in range(i+1, N):
            if hourly_profiles[i].sum() > 0 and hourly_profiles[j].sum() > 0:
                # Cosine similarity
                dot = np.dot(hourly_profiles[i], hourly_profiles[j])
                norm_i = np.linalg.norm(hourly_profiles[i])
                norm_j = np.linalg.norm(hourly_profiles[j])
                
                if norm_i > 0 and norm_j > 0:
                    sim = dot / (norm_i * norm_j)
                    if sim > 0.7:  # Only strong similarities
                        A[i, j] = sim
                        A[j, i] = sim
    
    print(f"   Stations with enough data: {(hourly_profiles.sum(axis=1) > 0).sum()}")
    print(f"   Edges (similarity > 0.7): {(A > 0).sum()}")
    
    return A


def build_knn_adjacency(meta, Y, M, k=5):
    """
    A_knn: K-nearest neighbors based on multiple features.
    
    Features:
    - Latitude, Longitude (location)
    - Operational age
    - Average demand
    - Number of EVSEs
    """
    print(f"\n🔗 Building KNN Adjacency (k={k})...")
    
    N = len(meta)
    
    # Build feature matrix
    features = []
    
    # Location (normalized)
    lat = meta['latitude'].fillna(meta['latitude'].median()).values
    lon = meta['longitude'].fillna(meta['longitude'].median()).values
    
    # Age
    age = meta['operational_years'].values
    
    # Average demand
    avg_demand = np.zeros(N)
    for i in range(N):
        valid = M[:, i] > 0
        if valid.sum() > 0:
            avg_demand[i] = Y[valid, i].mean()
    
    # Number of EVSEs
    n_evses = meta['num_evses'].values if 'num_evses' in meta.columns else np.ones(N)
    
    # Stack features
    X = np.stack([lat, lon, age, avg_demand, n_evses], axis=1)
    
    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # KNN
    knn = NearestNeighbors(n_neighbors=k+1, metric='euclidean')  # +1 because includes self
    knn.fit(X_scaled)
    
    distances, indices = knn.kneighbors(X_scaled)
    
    # Build adjacency
    A = np.zeros((N, N), dtype=np.float32)
    
    for i in range(N):
        for j_idx in range(1, k+1):  # Skip self (index 0)
            j = indices[i, j_idx]
            dist = distances[i, j_idx]
            weight = np.exp(-dist)  # Closer = higher weight
            A[i, j] = max(A[i, j], weight)
            A[j, i] = max(A[j, i], weight)  # Symmetric
    
    print(f"   Features used: location, age, demand, n_evses")
    print(f"   Edges: {(A > 0).sum()}")
    
    return A


def normalize_adjacency(A, add_self_loops=True):
    """Row-normalize adjacency matrix."""
    if add_self_loops:
        A = A + np.eye(A.shape[0])
    
    row_sum = A.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    return A / row_sum


def main():
    print("=" * 60)
    print("   Building Meaningful Adjacency Matrices")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    data = torch.load(f"{PROJECT_ROOT}/data/processed/tensors/data.pt", weights_only=False)
    Y = data['Y'].numpy()
    M = data['M'].numpy()
    
    meta = pd.read_csv(f"{PROJECT_ROOT}/data/processed/unified/station_metadata.csv")
    
    T, N = Y.shape
    print(f"Stations: {N}")
    print(f"Time steps: {T}")
    
    # Build different adjacency matrices
    adjacencies = {}
    
    # 1. Distance-based (existing, improved)
    adjacencies['A_distance'] = build_distance_adjacency(meta, threshold_km=20.0, sigma=8.0)
    
    # 2. Age similarity (NEW)
    adjacencies['A_age'] = build_age_similarity_adjacency(meta, threshold_years=0.5)
    
    # 3. Demand correlation (NEW)
    adjacencies['A_correlation'] = build_demand_correlation_adjacency(Y, M, min_overlap=100, corr_threshold=0.3)
    
    # 4. Demand level (NEW)
    adjacencies['A_demand_level'] = build_demand_level_adjacency(meta, Y, M, n_bins=4)
    
    # 5. Temporal pattern (NEW)
    adjacencies['A_temporal'] = build_temporal_pattern_adjacency(Y, M, min_hours=500)
    
    # 6. KNN-based (NEW)
    adjacencies['A_knn'] = build_knn_adjacency(meta, Y, M, k=5)
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary of Adjacency Matrices")
    print("=" * 60)
    print(f"\n{'Matrix':<20} {'Edges':<10} {'Meaning'}")
    print("-" * 60)
    
    for name, A in adjacencies.items():
        edges = (A > 0).sum()
        meanings = {
            'A_distance': 'Geographic proximity (<20km)',
            'A_age': 'Similar operational age',
            'A_correlation': 'Correlated demand patterns',
            'A_demand_level': 'Similar avg demand volume',
            'A_temporal': 'Similar hourly patterns',
            'A_knn': 'K-nearest (multi-feature)',
        }
        print(f"{name:<20} {edges:<10} {meanings.get(name, '')}")
    
    # Create combined adjacency (recommended for training)
    print("\n📊 Creating Combined Adjacency Matrix...")
    
    # Option 1: Simple combination
    A_combined = (
        0.3 * normalize_adjacency(adjacencies['A_distance'], add_self_loops=False) +
        0.2 * normalize_adjacency(adjacencies['A_age'], add_self_loops=False) +
        0.2 * normalize_adjacency(adjacencies['A_correlation'], add_self_loops=False) +
        0.15 * normalize_adjacency(adjacencies['A_temporal'], add_self_loops=False) +
        0.15 * normalize_adjacency(adjacencies['A_knn'], add_self_loops=False)
    )
    
    # Normalize and add self-loops
    A_combined = normalize_adjacency(A_combined, add_self_loops=True)
    adjacencies['A_combined'] = A_combined
    
    print(f"   Combined edges: {(A_combined > 0).sum()}")
    
    # Save
    output_dir = f"{PROJECT_ROOT}/data/graphs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as numpy
    np.savez(f"{output_dir}/adjacency_v2.npz", **adjacencies)
    
    # Save as torch
    torch.save(
        {k: torch.from_numpy(v) for k, v in adjacencies.items()},
        f"{output_dir}/adjacency_v2.pt"
    )
    
    print(f"\n✅ Saved to {output_dir}/adjacency_v2.npz and adjacency_v2.pt")
    
    # Recommendation
    print("\n" + "=" * 60)
    print("Recommendation for Training")
    print("=" * 60)
    print("""
Use A_combined for best results:
    adj = torch.load('adjacency_v2.pt')['A_combined']

Or experiment with individual matrices:
    - A_distance:    Best if stations are geographically clustered
    - A_correlation: Best for capturing demand patterns
    - A_knn:         Good general-purpose option
    - A_age:         Helps cold-start learn from similar-age stations
    """)
    
    return adjacencies


if __name__ == "__main__":
    adjacencies = main()
