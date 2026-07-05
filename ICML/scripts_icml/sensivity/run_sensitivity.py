#!/usr/bin/env python3
"""
TA-GNN Sensitivity Analysis for ICML 2026 Submission
=====================================================

This script runs comprehensive sensitivity experiments:
1. Cold-start window duration (Δ ∈ {7, 14, 21, 30} days) - NO retraining
2. Adjacency hyperparameters (σ, δ combinations) - requires retraining
3. Lookback window length (L ∈ {72, 168, 336} hours) - requires retraining

Usage:
    python run_sensitivity.py --experiment all
    python run_sensitivity.py --experiment coldstart  # Fast, no retraining
    python run_sensitivity.py --experiment adjacency
    python run_sensitivity.py --experiment lookback

Author: TA-GNN Team
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Default configuration - adjust paths as needed."""
    # Paths (anchored to the project root; override with EV_GNN_ROOT env var)
    PROJECT_ROOT = os.environ.get(
        'EV_GNN_ROOT',
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    DATA_PATH = os.path.join(PROJECT_ROOT, "data/processed/tensors/data_v4.pt")
    METADATA_PATH = os.path.join(PROJECT_ROOT, "data/processed/unified/station_metadata_v4.csv")
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints/sensitivity")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results/sensitivity")
    
    # Default model hyperparameters
    HIDDEN_DIM = 64
    NUM_GRU_LAYERS = 2
    NUM_ATTENTION_HEADS = 4
    NUM_ATTENTION_LAYERS = 2
    AGE_EMBED_DIM = 32
    PROVIDER_EMBED_DIM = 16
    DROPOUT = 0.2
    
    # Default training hyperparameters
    LOOKBACK = 168  # 7 days
    HORIZON = 5
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    MAX_EPOCHS = 100
    PATIENCE = 15
    POSITIVE_WEIGHT = 16.0
    
    # Default adjacency hyperparameters
    SIGMA = 5.0  # km
    DELTA = 15.0  # km
    
    # Cold-start window
    COLD_START_DAYS = 14
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Random seeds for reproducibility
    SEEDS = [42, 123, 456]


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_data(config):
    """Load the processed data tensors."""
    print(f"Loading data from {config.DATA_PATH}...")
    data = torch.load(config.DATA_PATH, map_location='cpu', weights_only=False)
    
    # Extract components
    Y = data['Y'].float()  # [T, N] demand
    M = data['M'].float()  # [T, N] mask
    time_features = data['time_features'].float()  # [T, d_f]
    station_features = data['station_features'].float()  # [N, d_s]
    provider_ids = data['provider_ids'].long()  # [N]
    station_open_times = data['station_open_times']  # [N] in hours from start
    
    # Convert station_open_times to tensor if needed
    if isinstance(station_open_times, np.ndarray):
        station_open_times = torch.from_numpy(station_open_times).float()
    elif not isinstance(station_open_times, torch.Tensor):
        station_open_times = torch.tensor(station_open_times).float()
    
    T, N = Y.shape
    print(f"  Loaded: T={T}, N={N}")
    print(f"  Observed hours: {M.sum().item():,.0f}")
    print(f"  Positive hours: {((Y > 0) & (M == 1)).sum().item():,.0f}")
    
    return {
        'Y': Y,
        'M': M,
        'time_features': time_features,
        'station_features': station_features,
        'provider_ids': provider_ids,
        'station_open_times': station_open_times,
        'T': T,
        'N': N,
        'num_providers': len(data['provider_list']),
        'd_f': time_features.shape[1],
        'd_s': station_features.shape[1],
    }


def compute_adjacency_matrix(station_features, sigma, delta, has_coords_col=2):
    """
    Compute distance-based adjacency matrix.
    
    Args:
        station_features: [N, d_s] tensor with lat/lon in columns 0,1
        sigma: Gaussian kernel bandwidth (km)
        delta: Distance cutoff (km)
        has_coords_col: Column index for has_coordinates indicator
    
    Returns:
        A: [N, N] adjacency matrix (row-normalized with self-loops)
    """
    N = station_features.shape[0]
    
    # Extract lat/lon (assuming columns 0, 1)
    lat = station_features[:, 0].numpy()
    lon = station_features[:, 1].numpy()
    
    # Check for has_coordinates indicator
    if station_features.shape[1] > has_coords_col:
        has_coords = station_features[:, has_coords_col].numpy() > 0.5
    else:
        has_coords = (lat != 0) | (lon != 0)
    
    # Compute pairwise distances (Haversine approximation)
    # For small distances, we can use Euclidean approximation with lat/lon scaling
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    
    # Average latitude for longitude scaling
    avg_lat = np.mean(lat_rad[has_coords]) if has_coords.any() else 0
    
    # Convert to approximate km
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(avg_lat)
    
    lat_km = lat * km_per_deg_lat
    lon_km = lon * km_per_deg_lon
    
    # Pairwise Euclidean distance in km
    dist = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if has_coords[i] and has_coords[j]:
                dist[i, j] = np.sqrt((lat_km[i] - lat_km[j])**2 + (lon_km[i] - lon_km[j])**2)
            else:
                dist[i, j] = np.inf  # No connection if missing coords
    
    # Gaussian kernel with cutoff
    A = np.exp(-dist**2 / (2 * sigma**2))
    A[dist > delta] = 0
    np.fill_diagonal(A, 0)  # Remove self-loops temporarily
    
    # Add self-loops and row-normalize
    A = A + np.eye(N)
    row_sum = A.sum(axis=1, keepdims=True)
    A = A / np.maximum(row_sum, 1e-8)
    
    return torch.from_numpy(A).float()


def compute_station_ages(T, station_open_times):
    """
    Compute station ages at each timestep.
    
    Args:
        T: Number of timesteps
        station_open_times: [N] tensor of opening times (in hours from t=0)
    
    Returns:
        ages: [T, N] tensor of ages in years
    """
    N = len(station_open_times)
    t_grid = torch.arange(T).float().unsqueeze(1)  # [T, 1]
    open_times = station_open_times.unsqueeze(0)  # [1, N]
    
    # Age in hours
    ages_hours = torch.clamp(t_grid - open_times, min=0)
    
    # Convert to years
    ages_years = ages_hours / (24 * 365.25)
    
    return ages_years


class EVChargingDataset(Dataset):
    """Dataset for EV charging demand forecasting."""
    
    def __init__(self, Y, M, time_features, station_features, provider_ids,
                 station_open_times, lookback, horizon, start_idx, end_idx):
        """
        Args:
            Y: [T, N] demand tensor
            M: [T, N] mask tensor
            time_features: [T, d_f] temporal features
            station_features: [N, d_s] static features
            provider_ids: [N] provider indices
            station_open_times: [N] opening times in hours
            lookback: Lookback window length L
            horizon: Forecast horizon H
            start_idx: Start index for this split
            end_idx: End index for this split
        """
        self.Y = Y
        self.M = M
        self.time_features = time_features
        self.station_features = station_features
        self.provider_ids = provider_ids
        self.station_open_times = station_open_times
        self.lookback = lookback
        self.horizon = horizon
        
        # Valid forecast origins
        self.origins = list(range(max(start_idx, lookback), min(end_idx, len(Y) - horizon)))
        
    def __len__(self):
        return len(self.origins)
    
    def __getitem__(self, idx):
        t = self.origins[idx]
        L = self.lookback
        H = self.horizon
        
        # History: [L, N]
        y_hist = self.Y[t-L:t]
        m_hist = self.M[t-L:t]
        
        # Temporal features for history: [L, d_f]
        x_hist = self.time_features[t-L:t]
        
        # Targets: [H, N]
        y_target = self.Y[t:t+H]
        m_target = self.M[t:t+H]
        
        # Station ages at forecast origin (in years)
        ages = torch.clamp(t - self.station_open_times, min=0) / (24 * 365.25)
        
        return {
            'y_hist': y_hist,
            'm_hist': m_hist,
            'x_hist': x_hist,
            'y_target': y_target,
            'm_target': m_target,
            'ages': ages,
            'origin': t,
        }


def create_data_splits(data, config, lookback=None):
    """Create train/val/test datasets."""
    if lookback is None:
        lookback = config.LOOKBACK
    
    T = data['T']
    train_end = int(0.70 * T)
    val_end = int(0.85 * T)
    
    common_args = {
        'Y': data['Y'],
        'M': data['M'],
        'time_features': data['time_features'],
        'station_features': data['station_features'],
        'provider_ids': data['provider_ids'],
        'station_open_times': data['station_open_times'],
        'lookback': lookback,
        'horizon': config.HORIZON,
    }
    
    train_dataset = EVChargingDataset(**common_args, start_idx=0, end_idx=train_end)
    val_dataset = EVChargingDataset(**common_args, start_idx=train_end, end_idx=val_end)
    test_dataset = EVChargingDataset(**common_args, start_idx=val_end, end_idx=T)
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset


# ============================================================================
# MODEL DEFINITION
# ============================================================================

class SinusoidalAgeEmbedding(nn.Module):
    """Sinusoidal embedding for station age."""
    
    def __init__(self, dim, max_age=5.0):
        super().__init__()
        self.dim = dim
        self.max_age = max_age
        
        # Fixed frequency bands
        freqs = torch.exp(torch.arange(0, dim // 2) * (-np.log(10000.0) / (dim // 2)))
        self.register_buffer('freqs', freqs)
    
    def forward(self, ages):
        """
        Args:
            ages: [...] tensor of ages in years
        Returns:
            embeddings: [..., dim] tensor
        """
        # Normalize age
        ages_norm = ages.unsqueeze(-1) / self.max_age  # [..., 1]
        
        # Compute sinusoidal embeddings
        angles = ages_norm * self.freqs * np.pi  # [..., dim//2]
        embeddings = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        
        return embeddings


class AgeModulatedAttention(nn.Module):
    """Multi-head attention with age-directional bias and proximity prior."""
    
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        # Learnable bias scales per head
        self.gamma = nn.Parameter(torch.ones(num_heads) * 2.0)  # Proximity prior
        self.delta = nn.Parameter(torch.ones(num_heads) * 0.5)  # Age-directional bias
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim)
    
    def forward(self, x, adjacency, ages, operational_mask):
        """
        Args:
            x: [B, N, D] node features
            adjacency: [N, N] proximity prior
            ages: [B, N] station ages in years
            operational_mask: [B, N] operational status (1=operational)
        
        Returns:
            out: [B, N, D] updated features
        """
        B, N, D = x.shape
        H = self.num_heads
        
        # Project to Q, K, V
        q = self.q_proj(x).view(B, N, H, -1).transpose(1, 2)  # [B, H, N, d]
        k = self.k_proj(x).view(B, N, H, -1).transpose(1, 2)
        v = self.v_proj(x).view(B, N, H, -1).transpose(1, 2)
        
        # Content attention
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, N, N]
        
        # Proximity prior: [N, N] -> [1, 1, N, N]
        prox_bias = self.gamma.view(1, H, 1, 1) * adjacency.view(1, 1, N, N)
        attn_logits = attn_logits + prox_bias
        
        # Age-directional bias: tanh(age_j - age_i)
        age_diff = ages.unsqueeze(2) - ages.unsqueeze(1)  # [B, N, N]: [i, j] = age_j - age_i
        age_bias = self.delta.view(1, H, 1, 1) * torch.tanh(age_diff).unsqueeze(1)
        attn_logits = attn_logits + age_bias
        
        # Operational mask: prevent attending to non-operational stations
        op_mask = operational_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]
        attn_logits = attn_logits.masked_fill(op_mask == 0, float('-inf'))
        
        # Softmax and apply
        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Aggregate
        out = torch.matmul(attn_weights, v)  # [B, H, N, d]
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)
        
        # Residual and layer norm
        out = self.layer_norm(x + out)
        
        return out


class TAGNN(nn.Module):
    """Temporal-Adaptive Graph Neural Network for cold-start forecasting."""
    
    def __init__(self, config, num_providers, d_f, d_s):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_proj = nn.Linear(2 + d_f, config.HIDDEN_DIM)  # demand + mask + time_features
        
        # Station encoder
        self.station_encoder = nn.Sequential(
            nn.Linear(d_s + config.PROVIDER_EMBED_DIM, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM)
        )
        self.provider_embed = nn.Embedding(num_providers, config.PROVIDER_EMBED_DIM)
        
        # Age embedding
        self.age_embed = SinusoidalAgeEmbedding(config.AGE_EMBED_DIM)
        
        # Feature fusion
        fusion_input_dim = config.HIDDEN_DIM + config.HIDDEN_DIM + config.AGE_EMBED_DIM
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, config.HIDDEN_DIM),
            nn.ReLU()
        )
        
        # Temporal pathway (GRU)
        self.temporal_gru = nn.GRU(
            input_size=config.HIDDEN_DIM,
            hidden_size=config.HIDDEN_DIM,
            num_layers=config.NUM_GRU_LAYERS,
            batch_first=True,
            dropout=config.DROPOUT if config.NUM_GRU_LAYERS > 1 else 0
        )
        
        # Spatial pathway (attention layers)
        self.spatial_layers = nn.ModuleList([
            AgeModulatedAttention(config.HIDDEN_DIM, config.NUM_ATTENTION_HEADS, config.DROPOUT)
            for _ in range(config.NUM_ATTENTION_LAYERS)
        ])
        
        # Age-gated fusion
        self.gate_alpha = nn.Parameter(torch.tensor(0.7))  # Will be softplus'd
        self.gate_beta = nn.Parameter(torch.tensor(0.5))  # In years
        
        # Hurdle heads
        self.detection_head = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, 32),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(32, config.HORIZON)
        )
        
        self.intensity_head = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, 32),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(32, config.HORIZON)
        )
    
    def forward(self, y_hist, m_hist, x_hist, ages, station_features, provider_ids, adjacency):
        """
        Args:
            y_hist: [B, L, N] demand history
            m_hist: [B, L, N] mask history
            x_hist: [B, L, d_f] temporal features
            ages: [B, N] station ages in years
            station_features: [N, d_s] static features
            provider_ids: [N] provider indices
            adjacency: [N, N] proximity matrix
        
        Returns:
            p: [B, N, H] detection probabilities
            mu: [B, N, H] intensity predictions
            y_hat: [B, N, H] expected demand (p * mu)
        """
        B, L, N = y_hist.shape
        device = y_hist.device
        
        # Station encoding (shared across batch)
        provider_emb = self.provider_embed(provider_ids)  # [N, d_e]
        station_enc = self.station_encoder(
            torch.cat([station_features, provider_emb], dim=-1)
        )  # [N, d]
        
        # Age embedding
        age_emb = self.age_embed(ages)  # [B, N, d_a]
        
        # Process each station's sequence
        # Input: [B, L, N, 2+d_f] -> per-station sequences
        x_in = torch.cat([
            y_hist.unsqueeze(-1),  # [B, L, N, 1]
            m_hist.unsqueeze(-1),  # [B, L, N, 1]
            x_hist.unsqueeze(2).expand(-1, -1, N, -1)  # [B, L, N, d_f]
        ], dim=-1)  # [B, L, N, 2+d_f]
        
        # Project inputs
        h = self.input_proj(x_in)  # [B, L, N, d]
        
        # Fuse with station and age embeddings
        station_enc_expanded = station_enc.unsqueeze(0).unsqueeze(1).expand(B, L, -1, -1)
        age_emb_expanded = age_emb.unsqueeze(1).expand(-1, L, -1, -1)
        
        h = self.fusion(torch.cat([h, station_enc_expanded, age_emb_expanded], dim=-1))
        
        # Temporal pathway: process each station's sequence with GRU
        # Reshape: [B, L, N, d] -> [B*N, L, d]
        h_temporal = h.permute(0, 2, 1, 3).reshape(B * N, L, -1)
        _, h_T = self.temporal_gru(h_temporal)  # h_T: [num_layers, B*N, d]
        h_T = h_T[-1].view(B, N, -1)  # [B, N, d]
        
        # Spatial pathway: use last timestep features
        h_S = h[:, -1, :, :]  # [B, N, d]
        operational_mask = m_hist[:, -1, :]  # [B, N]
        
        for attn_layer in self.spatial_layers:
            h_S = attn_layer(h_S, adjacency, ages, operational_mask)
        
        # Age-gated fusion
        alpha = F.softplus(self.gate_alpha) + 1e-3
        gate = torch.sigmoid(alpha * (ages - self.gate_beta))  # [B, N]
        gate = gate.unsqueeze(-1)  # [B, N, 1]
        
        h_fused = (1 - gate) * h_S + gate * h_T  # [B, N, d]
        
        # Hurdle heads
        logits = self.detection_head(h_fused)  # [B, N, H]
        p = torch.sigmoid(logits)
        
        mu = F.softplus(self.intensity_head(h_fused))  # [B, N, H]
        
        y_hat = p * mu
        
        return p, mu, y_hat, logits, gate.squeeze(-1)


# ============================================================================
# LOSS FUNCTION
# ============================================================================

class HurdleLoss(nn.Module):
    """Imbalance-aware hurdle loss for zero-inflated demand."""
    
    def __init__(self, positive_weight=16.0, lambda_cls=0.5, rho_reg=0.1):
        super().__init__()
        self.positive_weight = positive_weight
        self.lambda_cls = lambda_cls
        self.rho_reg = rho_reg
    
    def forward(self, p, mu, logits, y_target, m_target):
        """
        Args:
            p: [B, N, H] detection probabilities
            mu: [B, N, H] intensity predictions
            logits: [B, N, H] raw detection logits
            y_target: [B, H, N] targets (note: different order!)
            m_target: [B, H, N] target masks
        
        Returns:
            total_loss, loss_dict
        """
        # Transpose targets to match predictions: [B, H, N] -> [B, N, H]
        y = y_target.permute(0, 2, 1)
        m = m_target.permute(0, 2, 1)
        
        # Binary indicator
        z = (y > 0).float()
        
        # Detection loss (weighted BCE)
        pos_weight = torch.tensor([self.positive_weight], device=logits.device)
        bce = F.binary_cross_entropy_with_logits(
            logits, z, reduction='none', pos_weight=pos_weight
        )
        bce_masked = (bce * m).sum() / m.sum().clamp(min=1)
        
        # Intensity loss (positive-only MSE)
        pos_mask = m * z
        mse = ((mu - y) ** 2 * pos_mask).sum() / pos_mask.sum().clamp(min=1)
        
        # Zero regularization
        zero_mask = m * (1 - z)
        reg = (mu ** 2 * zero_mask).sum() / zero_mask.sum().clamp(min=1)
        
        # Total loss
        total = self.lambda_cls * bce_masked + (1 - self.lambda_cls) * mse + self.rho_reg * reg
        
        return total, {
            'total': total.item(),
            'bce': bce_masked.item(),
            'mse': mse.item(),
            'reg': reg.item()
        }


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion, adjacency, 
                station_features, provider_ids, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        # Move to device
        y_hist = batch['y_hist'].to(device)
        m_hist = batch['m_hist'].to(device)
        x_hist = batch['x_hist'].to(device)
        y_target = batch['y_target'].to(device)
        m_target = batch['m_target'].to(device)
        ages = batch['ages'].to(device)
        
        # Forward pass
        p, mu, y_hat, logits, gate = model(
            y_hist, m_hist, x_hist, ages,
            station_features.to(device),
            provider_ids.to(device),
            adjacency.to(device)
        )
        
        # Compute loss
        loss, _ = criterion(p, mu, logits, y_target, m_target)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def evaluate(model, dataloader, criterion, adjacency, station_features, 
             provider_ids, station_open_times, device, cold_start_days=14):
    """Evaluate model and compute all metrics."""
    model.eval()
    
    all_preds = []
    all_probs = []
    all_targets = []
    all_masks = []
    all_ages = []
    all_gates = []
    
    with torch.no_grad():
        for batch in dataloader:
            y_hist = batch['y_hist'].to(device)
            m_hist = batch['m_hist'].to(device)
            x_hist = batch['x_hist'].to(device)
            y_target = batch['y_target'].to(device)
            m_target = batch['m_target'].to(device)
            ages = batch['ages'].to(device)
            
            p, mu, y_hat, logits, gate = model(
                y_hist, m_hist, x_hist, ages,
                station_features.to(device),
                provider_ids.to(device),
                adjacency.to(device)
            )
            
            # Transpose targets
            y = y_target.permute(0, 2, 1)  # [B, N, H]
            m = m_target.permute(0, 2, 1)
            
            all_preds.append(y_hat.cpu())
            all_probs.append(p.cpu())
            all_targets.append(y.cpu())
            all_masks.append(m.cpu())
            all_ages.append(ages.cpu())
            all_gates.append(gate.cpu())
    
    # Concatenate
    preds = torch.cat(all_preds, dim=0).numpy()
    probs = torch.cat(all_probs, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()
    masks = torch.cat(all_masks, dim=0).numpy()
    ages = torch.cat(all_ages, dim=0).numpy()
    gates = torch.cat(all_gates, dim=0).numpy()
    
    # Flatten for metrics
    # Shapes: [num_samples, N, H] -> [num_samples * N * H]
    preds_flat = preds.reshape(-1)
    probs_flat = probs.reshape(-1)
    targets_flat = targets.reshape(-1)
    masks_flat = masks.reshape(-1)
    
    # Ages need to be expanded to [num_samples, N, H]
    ages_expanded = np.repeat(ages[:, :, np.newaxis], preds.shape[2], axis=2)
    ages_flat = ages_expanded.reshape(-1)
    
    # Valid mask (observed)
    valid = masks_flat == 1
    
    # Positive mask
    positive = (targets_flat > 0) & valid
    
    # Cold-start mask
    cold_start_years = cold_start_days / 365.25
    cold_start = (ages_flat < cold_start_years) & valid
    cold_start_positive = cold_start & positive
    
    # Compute metrics
    metrics = {}
    
    # Overall MAE
    if valid.sum() > 0:
        metrics['overall_mae'] = np.abs(preds_flat[valid] - targets_flat[valid]).mean()
    else:
        metrics['overall_mae'] = np.nan
    
    # Positive MAE
    if positive.sum() > 0:
        metrics['pos_mae'] = np.abs(preds_flat[positive] - targets_flat[positive]).mean()
    else:
        metrics['pos_mae'] = np.nan
    
    # Cold-Start Positive MAE
    if cold_start_positive.sum() > 0:
        metrics['cs_pos_mae'] = np.abs(preds_flat[cold_start_positive] - targets_flat[cold_start_positive]).mean()
        metrics['cs_pos_samples'] = cold_start_positive.sum()
    else:
        metrics['cs_pos_mae'] = np.nan
        metrics['cs_pos_samples'] = 0
    
    # AUPRC
    if valid.sum() > 0:
        binary_targets = (targets_flat[valid] > 0).astype(int)
        if binary_targets.sum() > 0 and binary_targets.sum() < len(binary_targets):
            metrics['auprc'] = average_precision_score(binary_targets, probs_flat[valid])
            metrics['auroc'] = roc_auc_score(binary_targets, probs_flat[valid])
        else:
            metrics['auprc'] = np.nan
            metrics['auroc'] = np.nan
    
    # Gate statistics
    metrics['avg_gate_coldstart'] = gates[ages < cold_start_years].mean() if (ages < cold_start_years).sum() > 0 else np.nan
    metrics['avg_gate_mature'] = gates[ages >= 2.0].mean() if (ages >= 2.0).sum() > 0 else np.nan
    
    return metrics


def train_model(config, data, adjacency, seed=42, lookback=None, verbose=True):
    """Full training loop."""
    if lookback is None:
        lookback = config.LOOKBACK
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create datasets
    train_dataset, val_dataset, test_dataset = create_data_splits(data, config, lookback)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)
    
    # Initialize model
    model = TAGNN(
        config, 
        num_providers=data['num_providers'],
        d_f=data['d_f'],
        d_s=data['d_s']
    ).to(config.DEVICE)
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    criterion = HurdleLoss(positive_weight=config.POSITIVE_WEIGHT)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    best_val_cs_pos = float('inf')
    patience_counter = 0
    best_state = None
    
    for epoch in range(config.MAX_EPOCHS):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, adjacency,
            data['station_features'], data['provider_ids'], config.DEVICE
        )
        
        # Validate
        val_metrics = evaluate(
            model, val_loader, criterion, adjacency,
            data['station_features'], data['provider_ids'],
            data['station_open_times'], config.DEVICE, config.COLD_START_DAYS
        )
        
        scheduler.step(val_metrics['cs_pos_mae'])
        
        # Early stopping
        if val_metrics['cs_pos_mae'] < best_val_cs_pos:
            best_val_cs_pos = val_metrics['cs_pos_mae']
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if verbose and (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}: Loss={train_loss:.4f}, Val CS-Pos={val_metrics['cs_pos_mae']:.4f}")
        
        if patience_counter >= config.PATIENCE:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break
    
    # Load best model and evaluate on test set
    model.load_state_dict(best_state)
    test_metrics = evaluate(
        model, test_loader, criterion, adjacency,
        data['station_features'], data['provider_ids'],
        data['station_open_times'], config.DEVICE, config.COLD_START_DAYS
    )
    
    return model, test_metrics


# ============================================================================
# SENSITIVITY EXPERIMENTS
# ============================================================================

def run_coldstart_window_sensitivity(config, data, model_path=None):
    """
    Experiment 1: Cold-start window duration.
    NO retraining needed - just re-evaluate with different Δ values.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: Cold-Start Window Duration Sensitivity")
    print("="*70)
    
    delta_values = [7, 14, 21, 30]  # days
    results = []
    
    # Train a single model (or load pre-trained)
    adjacency = compute_adjacency_matrix(
        data['station_features'], config.SIGMA, config.DELTA
    )
    
    if model_path and os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}")
        model = TAGNN(config, data['num_providers'], data['d_f'], data['d_s'])
        model.load_state_dict(torch.load(model_path, map_location=config.DEVICE, weights_only=False))
        model.to(config.DEVICE)
    else:
        print("Training model with default settings...")
        model, _ = train_model(config, data, adjacency, seed=42)
    
    # Evaluate with different cold-start windows
    _, _, test_dataset = create_data_splits(data, config)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)
    criterion = HurdleLoss(positive_weight=config.POSITIVE_WEIGHT)
    
    for delta in delta_values:
        print(f"\nEvaluating with Δ = {delta} days...")
        
        metrics = evaluate(
            model, test_loader, criterion, adjacency,
            data['station_features'], data['provider_ids'],
            data['station_open_times'], config.DEVICE, 
            cold_start_days=delta
        )
        
        results.append({
            'delta_days': delta,
            'cs_pos_mae': metrics['cs_pos_mae'],
            'cs_pos_samples': metrics['cs_pos_samples'],
            'avg_gate': metrics['avg_gate_coldstart'],
            'pos_mae': metrics['pos_mae'],
            'auprc': metrics['auprc']
        })
        
        print(f"  CS-Pos MAE: {metrics['cs_pos_mae']:.4f} (n={metrics['cs_pos_samples']})")
    
    return pd.DataFrame(results)


def run_adjacency_sensitivity(config, data):
    """
    Experiment 2: Adjacency hyperparameters.
    Requires retraining with different (σ, δ) combinations.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: Adjacency Hyperparameters Sensitivity")
    print("="*70)
    
    # Grid of (sigma, delta) combinations
    sigma_values = [3, 5, 10]
    delta_values = [10, 15, 25, 50]
    
    # Special configurations
    configs_to_test = [
        # Varying sigma (fixed delta=15)
        (3, 15), (5, 15), (10, 15),
        # Varying delta (fixed sigma=5)
        (5, 10), (5, 15), (5, 25), (5, 50),
        # Extreme: no spatial
        (None, 0),
        # Extreme: dense
        (10, 100),
    ]
    
    results = []
    
    for sigma, delta in configs_to_test:
        print(f"\nTraining with σ={sigma}, δ={delta}...")
        
        # Compute adjacency
        if delta == 0:
            # No spatial connections (identity only)
            N = data['N']
            adjacency = torch.eye(N)
        else:
            adjacency = compute_adjacency_matrix(
                data['station_features'], sigma, delta
            )
        
        # Compute average degree
        avg_degree = (adjacency > 0).float().sum(dim=1).mean().item()
        
        # Train with multiple seeds
        seed_results = []
        for seed in config.SEEDS:
            print(f"  Seed {seed}...")
            _, metrics = train_model(config, data, adjacency, seed=seed, verbose=False)
            seed_results.append(metrics)
        
        # Aggregate across seeds
        mean_cs_pos = np.mean([r['cs_pos_mae'] for r in seed_results])
        std_cs_pos = np.std([r['cs_pos_mae'] for r in seed_results])
        mean_pos = np.mean([r['pos_mae'] for r in seed_results])
        mean_auprc = np.mean([r['auprc'] for r in seed_results])
        
        results.append({
            'sigma': sigma,
            'delta': delta,
            'cs_pos_mae': mean_cs_pos,
            'cs_pos_std': std_cs_pos,
            'pos_mae': mean_pos,
            'auprc': mean_auprc,
            'avg_degree': avg_degree
        })
        
        print(f"  CS-Pos MAE: {mean_cs_pos:.4f} ± {std_cs_pos:.4f}")
    
    return pd.DataFrame(results)


def run_lookback_sensitivity(config, data):
    """
    Experiment 3: Lookback window length.
    Requires retraining with different L values.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3: Lookback Window Length Sensitivity")
    print("="*70)
    
    lookback_values = [72, 168, 336]  # 3, 7, 14 days
    
    results = []
    
    # Use default adjacency
    adjacency = compute_adjacency_matrix(
        data['station_features'], config.SIGMA, config.DELTA
    )
    
    for L in lookback_values:
        print(f"\nTraining with L={L} hours ({L//24} days)...")
        
        # Train with multiple seeds
        seed_results = []
        training_times = []
        
        for seed in config.SEEDS:
            print(f"  Seed {seed}...")
            start_time = time.time()
            _, metrics = train_model(config, data, adjacency, seed=seed, lookback=L, verbose=False)
            elapsed = time.time() - start_time
            
            seed_results.append(metrics)
            training_times.append(elapsed)
        
        # Aggregate
        mean_cs_pos = np.mean([r['cs_pos_mae'] for r in seed_results])
        std_cs_pos = np.std([r['cs_pos_mae'] for r in seed_results])
        mean_pos = np.mean([r['pos_mae'] for r in seed_results])
        mean_auprc = np.mean([r['auprc'] for r in seed_results])
        avg_time = np.mean(training_times)
        
        results.append({
            'lookback_hours': L,
            'lookback_days': L // 24,
            'cs_pos_mae': mean_cs_pos,
            'cs_pos_std': std_cs_pos,
            'pos_mae': mean_pos,
            'auprc': mean_auprc,
            'train_time_sec': avg_time
        })
        
        print(f"  CS-Pos MAE: {mean_cs_pos:.4f} ± {std_cs_pos:.4f}")
        print(f"  Training time: {avg_time:.1f}s")
    
    return pd.DataFrame(results)


# ============================================================================
# LATEX TABLE GENERATION
# ============================================================================

def generate_latex_tables(coldstart_results, adjacency_results, lookback_results, output_dir):
    """Generate LaTeX tables for the sensitivity analysis appendix."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Table 1: Cold-start window
    latex_coldstart = r"""
\begin{table}[h]
\centering
\caption{Sensitivity to cold-start window duration $\Delta$.}
\label{tab:sensitivity_delta}
\small
\begin{tabular}{@{}ccccc@{}}
\toprule
$\Delta$ (days) & \textbf{CS-Pos MAE}$\downarrow$ & \textbf{Samples} & \textbf{Avg Gate} & \textbf{Pos-MAE}$\downarrow$ \\
\midrule
"""
    for _, row in coldstart_results.iterrows():
        latex_coldstart += f"{int(row['delta_days'])} & {row['cs_pos_mae']:.3f} & {int(row['cs_pos_samples']):,} & {row['avg_gate']:.2f} & {row['pos_mae']:.3f} \\\\\n"
    latex_coldstart += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    # Table 2: Adjacency
    latex_adjacency = r"""
\begin{table}[h]
\centering
\caption{Sensitivity to adjacency hyperparameters.}
\label{tab:sensitivity_adjacency}
\small
\begin{tabular}{@{}cccccc@{}}
\toprule
$\sigma$ (km) & $\delta$ (km) & \textbf{CS-Pos}$\downarrow$ & \textbf{Pos-MAE}$\downarrow$ & \textbf{AUPRC}$\uparrow$ & \textbf{Avg Deg} \\
\midrule
"""
    for _, row in adjacency_results.iterrows():
        sigma_str = f"{row['sigma']}" if row['sigma'] is not None else "--"
        latex_adjacency += f"{sigma_str} & {int(row['delta'])} & {row['cs_pos_mae']:.3f} $\\pm$ {row['cs_pos_std']:.3f} & {row['pos_mae']:.3f} & {row['auprc']:.3f} & {row['avg_degree']:.1f} \\\\\n"
    latex_adjacency += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    # Table 3: Lookback
    latex_lookback = r"""
\begin{table}[h]
\centering
\caption{Sensitivity to lookback window length $L$.}
\label{tab:sensitivity_lookback}
\small
\begin{tabular}{@{}ccccccc@{}}
\toprule
$L$ (hours) & Days & \textbf{CS-Pos}$\downarrow$ & \textbf{Pos-MAE}$\downarrow$ & \textbf{AUPRC}$\uparrow$ & \textbf{Time/Epoch} \\
\midrule
"""
    for _, row in lookback_results.iterrows():
        latex_lookback += f"{int(row['lookback_hours'])} & {int(row['lookback_days'])} & {row['cs_pos_mae']:.3f} $\\pm$ {row['cs_pos_std']:.3f} & {row['pos_mae']:.3f} & {row['auprc']:.3f} & {row['train_time_sec']:.0f}s \\\\\n"
    latex_lookback += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    # Save tables
    with open(os.path.join(output_dir, 'table_coldstart.tex'), 'w') as f:
        f.write(latex_coldstart)
    
    with open(os.path.join(output_dir, 'table_adjacency.tex'), 'w') as f:
        f.write(latex_adjacency)
    
    with open(os.path.join(output_dir, 'table_lookback.tex'), 'w') as f:
        f.write(latex_lookback)
    
    print(f"\nLaTeX tables saved to {output_dir}/")
    
    return latex_coldstart, latex_adjacency, latex_lookback


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='TA-GNN Sensitivity Analysis')
    parser.add_argument('--experiment', type=str, default='all',
                        choices=['all', 'coldstart', 'adjacency', 'lookback'],
                        help='Which experiment to run')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to data_v4.pt')
    parser.add_argument('--output_dir', type=str, default='results/sensitivity',
                        help='Output directory for results')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to pre-trained model (for coldstart experiment)')
    args = parser.parse_args()
    
    # Initialize config
    config = Config()
    if args.data_path:
        config.DATA_PATH = args.data_path
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print(f"Device: {config.DEVICE}")
    data = load_data(config)
    
    # Run experiments
    coldstart_results = None
    adjacency_results = None
    lookback_results = None
    
    if args.experiment in ['all', 'coldstart']:
        coldstart_results = run_coldstart_window_sensitivity(config, data, args.model_path)
        coldstart_results.to_csv(os.path.join(args.output_dir, 'coldstart_sensitivity.csv'), index=False)
        print("\nCold-start results:")
        print(coldstart_results.to_string(index=False))
    
    if args.experiment in ['all', 'adjacency']:
        adjacency_results = run_adjacency_sensitivity(config, data)
        adjacency_results.to_csv(os.path.join(args.output_dir, 'adjacency_sensitivity.csv'), index=False)
        print("\nAdjacency results:")
        print(adjacency_results.to_string(index=False))
    
    if args.experiment in ['all', 'lookback']:
        lookback_results = run_lookback_sensitivity(config, data)
        lookback_results.to_csv(os.path.join(args.output_dir, 'lookback_sensitivity.csv'), index=False)
        print("\nLookback results:")
        print(lookback_results.to_string(index=False))
    
    # Generate LaTeX tables if all experiments completed
    if args.experiment == 'all':
        generate_latex_tables(coldstart_results, adjacency_results, lookback_results, args.output_dir)
    
    print("\n" + "="*70)
    print("SENSITIVITY ANALYSIS COMPLETE")
    print("="*70)
    print(f"Results saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
