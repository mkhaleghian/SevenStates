#!/usr/bin/env python3
"""
TA-GNN Full Sensitivity Analysis for ICML 2026
==============================================

This script runs comprehensive sensitivity experiments:
1. Cold-start window duration (Δ ∈ {7, 14, 21, 30} days) - NO retraining
2. Adjacency hyperparameters (σ, δ combinations) - requires retraining
3. Lookback window length (L ∈ {72, 168, 336} hours) - requires retraining

Uses exact model architecture from train_tagnn_dynamic_v4.py

Usage:
    cd ~/GNN
    python scripts/sensitivity/run_sensitivity_full.py --experiment all
    python scripts/sensitivity/run_sensitivity_full.py --experiment coldstart
    python scripts/sensitivity/run_sensitivity_full.py --experiment adjacency
    python scripts/sensitivity/run_sensitivity_full.py --experiment lookback

Estimated time:
    - coldstart: ~5 minutes (no retraining)
    - adjacency: ~1.5-2 hours (9 configs × 3 seeds)
    - lookback: ~45 minutes (3 configs × 3 seeds)
    - all: ~2.5-3 hours
"""

import os
import sys
import math
import time
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = os.environ.get('EV_GNN_ROOT', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_PATH = f"{PROJECT_ROOT}/data/processed/tensors/data_v4.pt"
ADJ_PATH = f"{PROJECT_ROOT}/data/graphs/adjacency_v4.pt"
CHECKPOINT_PATH = f"{PROJECT_ROOT}/checkpoints/tagnn_dynamic_v4/best_model_cs_pos.pt"
OUTPUT_DIR = f"{PROJECT_ROOT}/results/sensitivity"

# Default hyperparameters
HIDDEN_DIM = 64
NUM_GNN_LAYERS = 2
NUM_TEMPORAL_LAYERS = 2
NUM_HEADS = 4
DROPOUT = 0.2
LOOKBACK = 168
HORIZON = 5
MAX_AGE = 5.0

BATCH_SIZE = 32
LEARNING_RATE = 0.001
MAX_EPOCHS = 100
PATIENCE = 25
POS_WEIGHT = 16.0
CLASSIFICATION_WEIGHT = 0.5
COUNT_REG_WEIGHT = 0.01

# Adjacency defaults
DEFAULT_SIGMA = 5.0  # km
DEFAULT_DELTA = 15.0  # km

# Seeds for reproducibility
SEEDS = [42, 123, 456]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# MODEL COMPONENTS (exact copy from train_tagnn_dynamic_v4.py)
# ============================================================================

class SinusoidalAgeEmbedding(nn.Module):
    def __init__(self, embed_dim, max_age=5.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_age = max_age
        freqs = torch.exp(torch.linspace(0, math.log(max_age + 1), embed_dim // 2))
        self.register_buffer('freq_bands', freqs)
    
    def forward(self, age):
        if age.dim() == 1:
            age = age.unsqueeze(-1)
        else:
            age = age.unsqueeze(-1)
        age_norm = age / self.max_age
        angles = age_norm * self.freq_bands * math.pi
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class StationFeatureEncoder(nn.Module):
    def __init__(self, num_providers, hidden_dim, dropout=0.1):
        super().__init__()
        self.continuous_proj = nn.Sequential(
            nn.Linear(4, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.provider_embed = nn.Embedding(num_providers, hidden_dim // 2)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, station_features, provider_ids):
        cont_embed = self.continuous_proj(station_features)
        prov_embed = self.provider_embed(provider_ids)
        combined = torch.cat([cont_embed, prov_embed], dim=-1)
        return self.output_proj(combined)


class TrulySoftGraphAttentionV4(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1, 
                 adj_bias_scale=2.0, age_bias_scale=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.adj_bias = nn.Parameter(torch.ones(num_heads) * adj_bias_scale)
        self.age_bias = nn.Parameter(torch.ones(num_heads) * age_bias_scale)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, ages, static_adj=None, station_open_mask=None):
        B, N, D = x.shape
        H = self.num_heads
        head_dim = self.head_dim
        
        Q = self.W_q(x).view(B, N, H, head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, N, H, head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, N, H, head_dim).transpose(1, 2)
        
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if static_adj is not None:
            adj_bias = static_adj.unsqueeze(0).unsqueeze(0)
            adj_bias = adj_bias * self.adj_bias.view(1, H, 1, 1)
            attn_logits = attn_logits + adj_bias
        
        age_i = ages.unsqueeze(-1)
        age_j = ages.unsqueeze(-2)
        relative_age = age_j - age_i
        age_direction_bias = torch.tanh(relative_age).unsqueeze(1)
        age_direction_bias = age_direction_bias * self.age_bias.view(1, H, 1, 1)
        attn_logits = attn_logits + age_direction_bias
        
        if station_open_mask is not None:
            key_mask = station_open_mask.unsqueeze(1).unsqueeze(2)
            attn_logits = attn_logits.masked_fill(key_mask == 0, -1e9)
        
        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)
        out = self.layer_norm(x + out)
        
        return out, attn_weights


class MonotonicAgeGate(nn.Module):
    def __init__(self, hidden_dim, init_steepness=2.0, init_midpoint_years=0.5):
        super().__init__()
        self.a_raw = nn.Parameter(torch.tensor(init_steepness))
        self.b = nn.Parameter(torch.tensor(init_midpoint_years))
    
    def forward(self, h_spatial, h_temporal, ages):
        a = F.softplus(self.a_raw) + 0.1
        g = torch.sigmoid(a * (ages - self.b))
        g_expanded = g.unsqueeze(-1)
        h_fused = (1 - g_expanded) * h_spatial + g_expanded * h_temporal
        return h_fused, g


class TemporalEncoder(nn.Module):
    def __init__(self, hidden_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
    
    def forward(self, x):
        B, N, T, D = x.shape
        x = x.view(B * N, T, D)
        output, _ = self.gru(x)
        return output[:, -1, :].view(B, N, D)


class TAGNNDynamicV4(nn.Module):
    def __init__(self, num_nodes, num_providers, time_feat_dim=7, 
                 hidden_dim=64, num_gnn_layers=2, num_temporal_layers=2,
                 num_heads=4, dropout=0.2, horizon=5, max_age=5.0):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.age_embed_dim = 32
        
        self.input_proj = nn.Linear(2 + time_feat_dim, hidden_dim)
        self.station_encoder = StationFeatureEncoder(num_providers, hidden_dim, dropout)
        self.age_embedding = SinusoidalAgeEmbedding(self.age_embed_dim, max_age)
        self.feature_fusion = nn.Linear(hidden_dim + hidden_dim + self.age_embed_dim, hidden_dim)
        
        self.gnn_layers = nn.ModuleList([
            TrulySoftGraphAttentionV4(hidden_dim, num_heads, dropout)
            for _ in range(num_gnn_layers)
        ])
        
        self.temporal_encoder = TemporalEncoder(hidden_dim, num_temporal_layers, dropout)
        self.age_gate = MonotonicAgeGate(hidden_dim)
        
        self.count_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon),
        )
        
        self.prob_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, horizon),
        )
    
    def forward(self, y_hist, m_hist, time_features, ages, 
                station_features, provider_ids, static_adj=None):
        B, N, T = y_hist.shape
        
        y = y_hist.unsqueeze(-1)
        m = m_hist.unsqueeze(-1)
        
        if time_features.dim() == 2:
            time_features = time_features.unsqueeze(0).expand(B, -1, -1)
        time_exp = time_features.unsqueeze(1).expand(-1, N, -1, -1)
        
        h = torch.cat([y, m, time_exp], dim=-1)
        h = self.input_proj(h)
        
        station_embed = self.station_encoder(station_features, provider_ids)
        station_embed = station_embed.unsqueeze(0).unsqueeze(2).expand(B, -1, T, -1)
        
        age_embed = self.age_embedding(ages)
        age_embed_exp = age_embed.unsqueeze(2).expand(-1, -1, T, -1)
        
        h = torch.cat([h, station_embed, age_embed_exp], dim=-1)
        h = self.feature_fusion(h)
        h = F.relu(h)
        
        station_open_mask = (m_hist[:, :, -1] > 0).float()
        
        h_spatial = h[:, :, -1, :]
        for gnn_layer in self.gnn_layers:
            h_spatial, _ = gnn_layer(h_spatial, ages, static_adj, station_open_mask)
        
        h_temporal = self.temporal_encoder(h)
        h_fused, gate_values = self.age_gate(h_spatial, h_temporal, ages)
        
        count_pred = F.softplus(self.count_head(h_fused))
        prob_logits = self.prob_head(h_fused)
        prob = torch.sigmoid(prob_logits)
        
        return {
            'count': count_pred,
            'prob_logits': prob_logits,
            'prob': prob,
            'expected': prob * count_pred,
            'gate': gate_values,
        }


# ============================================================================
# LOSS FUNCTION
# ============================================================================

class TrueHurdleLoss(nn.Module):
    def __init__(self, pos_weight=16.0, classification_weight=0.5, count_reg_weight=0.01):
        super().__init__()
        self.pos_weight = pos_weight
        self.classification_weight = classification_weight
        self.count_reg_weight = count_reg_weight
    
    def forward(self, output, target, mask):
        prob_logits = output['prob_logits']
        count_pred = output['count']
        
        eps = 1e-8
        binary_target = (target > 0).float()
        
        pos_weight = torch.tensor([self.pos_weight], device=prob_logits.device)
        bce = F.binary_cross_entropy_with_logits(
            prob_logits, binary_target, pos_weight=pos_weight, reduction='none'
        )
        bce_loss = (bce * mask).sum() / (mask.sum() + eps)
        
        positive_mask = mask * binary_target
        mse = (count_pred - target) ** 2
        count_loss = (mse * positive_mask).sum() / (positive_mask.sum() + eps)
        
        zero_mask = mask * (1 - binary_target)
        count_reg = (count_pred ** 2 * zero_mask).sum() / (zero_mask.sum() + eps)
        
        total = (self.classification_weight * bce_loss + 
                 (1 - self.classification_weight) * count_loss +
                 self.count_reg_weight * count_reg)
        
        return total


# ============================================================================
# DATASET
# ============================================================================

class EVChargingDatasetV4(Dataset):
    def __init__(self, Y, M, time_features, time_index, station_open_times,
                 lookback=168, horizon=5, mode='train', 
                 train_ratio=0.7, val_ratio=0.15,
                 cold_start_augment_prob=0.0, cold_start_max_age_days=14):
        
        self.Y = torch.FloatTensor(Y)
        self.M = torch.FloatTensor(M)
        self.time_features = torch.FloatTensor(time_features)
        self.lookback = lookback
        self.horizon = horizon
        self.mode = mode
        self.cold_start_augment_prob = cold_start_augment_prob if mode == 'train' else 0.0
        self.cold_start_max_age_days = cold_start_max_age_days
        self.time_index = time_index
        self.station_open_times = station_open_times
        
        T, N = Y.shape
        self.T, self.N = T, N
        
        train_end = int(T * train_ratio)
        val_end = int(T * (train_ratio + val_ratio))
        
        if mode == 'train':
            self.time_start = lookback
            self.time_end = train_end - horizon
        elif mode == 'val':
            self.time_start = train_end
            self.time_end = val_end - horizon
        else:
            self.time_start = val_end
            self.time_end = T - horizon
        
        self.valid_times = list(range(self.time_start, self.time_end))
    
    def __len__(self):
        return len(self.valid_times)
    
    def __getitem__(self, idx):
        t = self.valid_times[idx]
        
        current_time = self.time_index[t]
        real_ages = (current_time - self.station_open_times).astype('timedelta64[h]').astype(float) / (24 * 365.25)
        real_ages = np.maximum(real_ages, 0)
        ages = torch.FloatTensor(real_ages.copy())
        
        y_hist = self.Y[t-self.lookback:t, :].T.clone()
        m_hist = self.M[t-self.lookback:t, :].T.clone()
        
        # Cold-start augmentation
        if self.cold_start_augment_prob > 0 and self.mode == 'train':
            augment_stations = torch.rand(self.N) < self.cold_start_augment_prob
            for i in range(self.N):
                if augment_stations[i] and ages[i] > 0:
                    y_hist[i, :] = 0
                    fake_age_days = torch.rand(1).item() * self.cold_start_max_age_days
                    ages[i] = fake_age_days / 365.25
                    hours_open = max(1, min(int(fake_age_days * 24), self.lookback))
                    m_hist[i, :-hours_open] = 0
                    m_hist[i, -hours_open:] = 1
        
        return {
            'y_hist': y_hist,
            'm_hist': m_hist,
            'time_hist': self.time_features[t-self.lookback:t],
            'y_target': self.Y[t:t+self.horizon, :].T,
            'm_target': self.M[t:t+self.horizon, :].T,
            'ages': ages,
            't_idx': t,
        }


# ============================================================================
# ADJACENCY COMPUTATION
# ============================================================================

def compute_adjacency_gaussian(lat, lon, sigma_km=5.0, delta_km=15.0):
    """Compute Gaussian kernel adjacency matrix."""
    N = len(lat)
    
    # Haversine approximation
    km_per_deg_lat = 111.0
    valid_mask = (lat != 0) | (lon != 0)
    avg_lat_rad = np.radians(np.mean(lat[valid_mask])) if valid_mask.any() else 0
    km_per_deg_lon = 111.0 * np.cos(avg_lat_rad)
    
    lat_km = lat * km_per_deg_lat
    lon_km = lon * km_per_deg_lon
    
    # Pairwise distance
    dist = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if valid_mask[i] and valid_mask[j]:
                dist[i, j] = np.sqrt((lat_km[i] - lat_km[j])**2 + (lon_km[i] - lon_km[j])**2)
            else:
                dist[i, j] = np.inf
    
    # Gaussian kernel with cutoff
    A = np.exp(-dist**2 / (2 * sigma_km**2))
    A[dist > delta_km] = 0
    np.fill_diagonal(A, 0)
    
    # Row-normalize with self-loop
    A = A + np.eye(N)
    row_sum = A.sum(axis=1, keepdims=True)
    A = A / np.maximum(row_sum, 1e-8)
    
    return torch.FloatTensor(A)


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

def train_epoch(model, loader, optimizer, criterion, device, 
                static_adj, station_features, provider_ids):
    model.train()
    total_loss = 0
    
    for batch in loader:
        y_hist = batch['y_hist'].to(device)
        m_hist = batch['m_hist'].to(device)
        time_hist = batch['time_hist'].to(device)
        y_target = batch['y_target'].to(device)
        m_target = batch['m_target'].to(device)
        ages = batch['ages'].to(device)
        
        optimizer.zero_grad()
        output = model(y_hist, m_hist, time_hist, ages,
                      station_features, provider_ids, static_adj)
        loss = criterion(output, y_target, m_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def evaluate(model, loader, device, static_adj, station_features, 
             provider_ids, cold_start_days=14):
    model.eval()
    window_hours = cold_start_days * 24
    
    all_data = {'expected': [], 'prob': [], 'target': [], 'mask': [], 'ages_hours': [], 'gate': []}
    
    with torch.no_grad():
        for batch in loader:
            y_hist = batch['y_hist'].to(device)
            m_hist = batch['m_hist'].to(device)
            time_hist = batch['time_hist'].to(device)
            y_target = batch['y_target'].to(device)
            m_target = batch['m_target'].to(device)
            ages = batch['ages'].to(device)
            
            output = model(y_hist, m_hist, time_hist, ages,
                          station_features, provider_ids, static_adj)
            
            all_data['expected'].append(output['expected'].cpu())
            all_data['prob'].append(output['prob'].cpu())
            all_data['target'].append(y_target.cpu())
            all_data['mask'].append(m_target.cpu())
            all_data['ages_hours'].append((ages * 24 * 365.25).cpu())
            all_data['gate'].append(output['gate'].cpu())
    
    for k in all_data:
        all_data[k] = torch.cat(all_data[k], dim=0)
    
    B, N, H = all_data['target'].shape
    expected = all_data['expected'].reshape(-1)
    prob = all_data['prob'].reshape(-1)
    target = all_data['target'].reshape(-1)
    mask = all_data['mask'].reshape(-1)
    ages_expanded = all_data['ages_hours'].unsqueeze(-1).expand(B, N, H).reshape(-1)
    
    valid = mask > 0
    positive = (target > 0) & valid
    cold_start_positive = (ages_expanded < window_hours) & (target > 0) & valid
    
    metrics = {}
    
    if valid.sum() > 0:
        metrics['overall_mae'] = (expected[valid] - target[valid]).abs().mean().item()
    
    if positive.sum() > 0:
        metrics['pos_mae'] = (expected[positive] - target[positive]).abs().mean().item()
    
    if cold_start_positive.sum() > 0:
        metrics['cs_pos_mae'] = (expected[cold_start_positive] - target[cold_start_positive]).abs().mean().item()
        metrics['cs_pos_samples'] = int(cold_start_positive.sum().item())
    else:
        metrics['cs_pos_mae'] = float('inf')
        metrics['cs_pos_samples'] = 0
    
    if valid.sum() > 0:
        try:
            binary_target = (target[valid] > 0).float().numpy()
            binary_pred = prob[valid].numpy()
            metrics['auprc'] = average_precision_score(binary_target, binary_pred)
        except:
            metrics['auprc'] = float('nan')
    
    return metrics


def train_model(data, static_adj, station_features, provider_ids, 
                seed=42, lookback=LOOKBACK, verbose=True):
    """Full training loop with early stopping on CS-Pos MAE."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    Y = data['Y']
    M = data['M']
    time_features = data['time_features']
    time_index = data['time_index']
    station_open_times = data['station_open_times']
    N = data['N']
    num_providers = data['num_providers']
    
    # Create datasets
    train_ds = EVChargingDatasetV4(
        Y, M, time_features, time_index, station_open_times,
        lookback=lookback, mode='train', cold_start_augment_prob=0.2
    )
    val_ds = EVChargingDatasetV4(
        Y, M, time_features, time_index, station_open_times,
        lookback=lookback, mode='val'
    )
    test_ds = EVChargingDatasetV4(
        Y, M, time_features, time_index, station_open_times,
        lookback=lookback, mode='test'
    )
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    # Create model
    model = TAGNNDynamicV4(
        num_nodes=N,
        num_providers=num_providers,
        time_feat_dim=time_features.shape[1],
        hidden_dim=HIDDEN_DIM,
        num_gnn_layers=NUM_GNN_LAYERS,
        num_temporal_layers=NUM_TEMPORAL_LAYERS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        horizon=HORIZON,
        max_age=MAX_AGE,
    ).to(DEVICE)
    
    criterion = TrueHurdleLoss(
        pos_weight=POS_WEIGHT,
        classification_weight=CLASSIFICATION_WEIGHT,
        count_reg_weight=COUNT_REG_WEIGHT
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)
    
    # Training loop
    best_cs_pos_mae = float('inf')
    best_state = None
    patience_counter = 0
    
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, DEVICE,
            static_adj, station_features, provider_ids
        )
        
        val_metrics = evaluate(
            model, val_loader, DEVICE, static_adj, 
            station_features, provider_ids
        )
        
        scheduler.step()
        
        if val_metrics['cs_pos_mae'] < best_cs_pos_mae - 0.001:
            best_cs_pos_mae = val_metrics['cs_pos_mae']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if verbose and epoch % 10 == 0:
            print(f"    Epoch {epoch}: Train={train_loss:.4f}, CS-Pos={val_metrics['cs_pos_mae']:.4f}")
        
        if patience_counter >= PATIENCE:
            if verbose:
                print(f"    Early stopping at epoch {epoch}")
            break
    
    # Load best and evaluate on test
    model.load_state_dict(best_state)
    model.to(DEVICE)
    
    test_metrics = evaluate(
        model, test_loader, DEVICE, static_adj,
        station_features, provider_ids
    )
    
    return test_metrics


# ============================================================================
# SENSITIVITY EXPERIMENTS
# ============================================================================

def run_coldstart_sensitivity(data, static_adj, station_features, provider_ids, 
                               model_path=None):
    """Experiment 1: Cold-start window duration (no retraining)."""
    print("\n" + "="*70)
    print("EXPERIMENT 1: Cold-Start Window Duration")
    print("="*70)
    
    delta_values = [7, 14, 21, 30]
    
    # Load pre-trained model
    N = data['N']
    num_providers = data['num_providers']
    time_feat_dim = data['time_features'].shape[1]
    
    model = TAGNNDynamicV4(
        num_nodes=N, num_providers=num_providers, time_feat_dim=time_feat_dim,
        hidden_dim=HIDDEN_DIM, num_gnn_layers=NUM_GNN_LAYERS,
        num_temporal_layers=NUM_TEMPORAL_LAYERS, num_heads=NUM_HEADS,
        dropout=DROPOUT, horizon=HORIZON, max_age=MAX_AGE
    ).to(DEVICE)
    
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint)
    model.eval()
    
    # Create test dataset
    test_ds = EVChargingDatasetV4(
        data['Y'], data['M'], data['time_features'],
        data['time_index'], data['station_open_times'], mode='test'
    )
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    results = []
    for delta in delta_values:
        print(f"\nΔ = {delta} days...")
        metrics = evaluate(model, test_loader, DEVICE, static_adj,
                          station_features, provider_ids, cold_start_days=delta)
        
        # Get gate values for this window
        all_gates = []
        all_ages = []
        with torch.no_grad():
            for batch in test_loader:
                ages = batch['ages'].to(DEVICE)
                output = model(
                    batch['y_hist'].to(DEVICE), batch['m_hist'].to(DEVICE),
                    batch['time_hist'].to(DEVICE), ages,
                    station_features, provider_ids, static_adj
                )
                all_gates.append(output['gate'].cpu())
                all_ages.append(ages.cpu())
        
        gates = torch.cat(all_gates).numpy().flatten()
        ages = torch.cat(all_ages).numpy().flatten()
        cold_mask = ages < (delta / 365.25)
        avg_gate = gates[cold_mask].mean() if cold_mask.sum() > 0 else float('nan')
        
        results.append({
            'delta_days': delta,
            'cs_pos_mae': metrics['cs_pos_mae'],
            'cs_pos_samples': metrics['cs_pos_samples'],
            'avg_gate': avg_gate,
            'pos_mae': metrics['pos_mae'],
            'auprc': metrics.get('auprc', float('nan'))
        })
        print(f"  CS-Pos MAE: {metrics['cs_pos_mae']:.4f}, Samples: {metrics['cs_pos_samples']}")
    
    return pd.DataFrame(results)


def run_adjacency_sensitivity(data, station_features_raw, station_features, provider_ids):
    """Experiment 2: Adjacency hyperparameters (requires retraining)."""
    print("\n" + "="*70)
    print("EXPERIMENT 2: Adjacency Hyperparameters")
    print("="*70)
    
    lat = station_features_raw[:, 1]
    lon = station_features_raw[:, 2]
    
    configs = [
        (3, 15), (5, 15), (10, 15),  # Varying sigma
        (5, 10), (5, 25), (5, 50),    # Varying delta
        (None, 0),                     # No spatial
        (10, 100),                     # Dense
    ]
    
    results = []
    
    for sigma, delta in configs:
        print(f"\nσ={sigma}, δ={delta}...")
        
        if delta == 0:
            adj = torch.eye(data['N']).to(DEVICE)
        else:
            adj = compute_adjacency_gaussian(lat, lon, sigma, delta).to(DEVICE)
        
        avg_degree = (adj > 0).float().sum(dim=1).mean().item()
        
        seed_results = []
        for seed in SEEDS:
            print(f"  Seed {seed}...")
            metrics = train_model(data, adj, station_features, provider_ids,
                                 seed=seed, verbose=False)
            seed_results.append(metrics)
        
        mean_cs_pos = np.mean([r['cs_pos_mae'] for r in seed_results])
        std_cs_pos = np.std([r['cs_pos_mae'] for r in seed_results])
        mean_pos = np.mean([r['pos_mae'] for r in seed_results])
        mean_auprc = np.mean([r.get('auprc', 0) for r in seed_results])
        
        results.append({
            'sigma': sigma,
            'delta': delta,
            'cs_pos_mae': mean_cs_pos,
            'cs_pos_std': std_cs_pos,
            'pos_mae': mean_pos,
            'auprc': mean_auprc,
            'avg_degree': avg_degree
        })
        print(f"  CS-Pos: {mean_cs_pos:.4f} ± {std_cs_pos:.4f}")
    
    return pd.DataFrame(results)


def run_lookback_sensitivity(data, static_adj, station_features, provider_ids):
    """Experiment 3: Lookback window length (requires retraining)."""
    print("\n" + "="*70)
    print("EXPERIMENT 3: Lookback Window Length")
    print("="*70)
    
    lookback_values = [72, 168, 336]  # 3, 7, 14 days
    
    results = []
    
    for L in lookback_values:
        print(f"\nL={L} hours ({L//24} days)...")
        
        seed_results = []
        train_times = []
        
        for seed in SEEDS:
            print(f"  Seed {seed}...")
            start = time.time()
            metrics = train_model(data, static_adj, station_features, provider_ids,
                                 seed=seed, lookback=L, verbose=False)
            elapsed = time.time() - start
            
            seed_results.append(metrics)
            train_times.append(elapsed)
        
        mean_cs_pos = np.mean([r['cs_pos_mae'] for r in seed_results])
        std_cs_pos = np.std([r['cs_pos_mae'] for r in seed_results])
        mean_pos = np.mean([r['pos_mae'] for r in seed_results])
        mean_auprc = np.mean([r.get('auprc', 0) for r in seed_results])
        avg_time = np.mean(train_times)
        
        results.append({
            'lookback_hours': L,
            'lookback_days': L // 24,
            'cs_pos_mae': mean_cs_pos,
            'cs_pos_std': std_cs_pos,
            'pos_mae': mean_pos,
            'auprc': mean_auprc,
            'train_time_sec': avg_time
        })
        print(f"  CS-Pos: {mean_cs_pos:.4f} ± {std_cs_pos:.4f}, Time: {avg_time:.1f}s")
    
    return pd.DataFrame(results)


# ============================================================================
# LATEX TABLE GENERATION
# ============================================================================

def generate_latex_tables(coldstart_df, adjacency_df, lookback_df, output_dir):
    """Generate LaTeX tables."""
    
    # Cold-start table
    if coldstart_df is not None:
        latex = r"""\begin{table}[h]
\centering
\caption{Sensitivity to cold-start window duration $\Delta$.}
\label{tab:sensitivity_delta}
\small
\begin{tabular}{@{}ccccc@{}}
\toprule
$\Delta$ (days) & \textbf{CS-Pos MAE}$\downarrow$ & \textbf{Samples} & \textbf{Avg Gate} & \textbf{Pos-MAE}$\downarrow$ \\
\midrule
"""
        for _, row in coldstart_df.iterrows():
            latex += f"{int(row['delta_days'])} & {row['cs_pos_mae']:.3f} & {int(row['cs_pos_samples']):,} & {row['avg_gate']:.2f} & {row['pos_mae']:.3f} \\\\\n"
        latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
        with open(os.path.join(output_dir, 'table_coldstart.tex'), 'w') as f:
            f.write(latex)
    
    # Adjacency table
    if adjacency_df is not None:
        latex = r"""\begin{table}[h]
\centering
\caption{Sensitivity to adjacency hyperparameters.}
\label{tab:sensitivity_adjacency}
\small
\begin{tabular}{@{}cccccc@{}}
\toprule
$\sigma$ (km) & $\delta$ (km) & \textbf{CS-Pos}$\downarrow$ & \textbf{Pos-MAE}$\downarrow$ & \textbf{AUPRC}$\uparrow$ & \textbf{Avg Deg} \\
\midrule
"""
        for _, row in adjacency_df.iterrows():
            sigma_str = f"{int(row['sigma'])}" if row['sigma'] is not None else "--"
            latex += f"{sigma_str} & {int(row['delta'])} & {row['cs_pos_mae']:.3f} $\\pm$ {row['cs_pos_std']:.3f} & {row['pos_mae']:.3f} & {row['auprc']:.3f} & {row['avg_degree']:.1f} \\\\\n"
        latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
        with open(os.path.join(output_dir, 'table_adjacency.tex'), 'w') as f:
            f.write(latex)
    
    # Lookback table
    if lookback_df is not None:
        latex = r"""\begin{table}[h]
\centering
\caption{Sensitivity to lookback window length $L$.}
\label{tab:sensitivity_lookback}
\small
\begin{tabular}{@{}cccccc@{}}
\toprule
$L$ (hours) & Days & \textbf{CS-Pos}$\downarrow$ & \textbf{Pos-MAE}$\downarrow$ & \textbf{AUPRC}$\uparrow$ & \textbf{Time} \\
\midrule
"""
        for _, row in lookback_df.iterrows():
            latex += f"{int(row['lookback_hours'])} & {int(row['lookback_days'])} & {row['cs_pos_mae']:.3f} $\\pm$ {row['cs_pos_std']:.3f} & {row['pos_mae']:.3f} & {row['auprc']:.3f} & {row['train_time_sec']:.0f}s \\\\\n"
        latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
        with open(os.path.join(output_dir, 'table_lookback.tex'), 'w') as f:
            f.write(latex)
    
    print(f"\nLaTeX tables saved to {output_dir}/")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='all',
                        choices=['all', 'coldstart', 'adjacency', 'lookback'])
    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT_PATH)
    parser.add_argument('--output', type=str, default=OUTPUT_DIR)
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    print("="*70)
    print("TA-GNN SENSITIVITY ANALYSIS")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Experiment: {args.experiment}")
    
    # Load data
    print(f"\nLoading data...")
    raw_data = torch.load(DATA_PATH, map_location='cpu', weights_only=False)
    
    Y = raw_data['Y'].numpy()
    M = raw_data['M'].numpy()
    time_features = raw_data['time_features'].numpy()
    time_index = raw_data['time_index']
    station_open_times = raw_data['station_open_times']
    station_features_raw = raw_data['station_features'].numpy()
    provider_ids = raw_data['provider_ids'].long().to(DEVICE)
    provider_list = raw_data['provider_list']
    
    T, N = Y.shape
    num_providers = len(provider_list)
    print(f"  T={T:,}, N={N}, Providers={num_providers}")
    
    # Normalize station features
    has_coords = (station_features_raw[:, 1] != 0) | (station_features_raw[:, 2] != 0)
    sf_normalized = np.zeros((N, 4), dtype=np.float32)
    sf_normalized[:, 0] = (station_features_raw[:, 0] - station_features_raw[:, 0].mean()) / (station_features_raw[:, 0].std() + 1e-8)
    valid_lat = station_features_raw[has_coords, 1]
    valid_lon = station_features_raw[has_coords, 2]
    if len(valid_lat) > 0:
        sf_normalized[has_coords, 1] = (station_features_raw[has_coords, 1] - valid_lat.mean()) / (valid_lat.std() + 1e-8)
        sf_normalized[has_coords, 2] = (station_features_raw[has_coords, 2] - valid_lon.mean()) / (valid_lon.std() + 1e-8)
    sf_normalized[:, 3] = has_coords.astype(float)
    station_features = torch.FloatTensor(sf_normalized).to(DEVICE)
    
    # Load adjacency
    if os.path.exists(ADJ_PATH):
        adj_data = torch.load(ADJ_PATH, weights_only=False)
        static_adj = adj_data['A_distance'].float().to(DEVICE)
    else:
        static_adj = torch.eye(N).to(DEVICE)
    
    # Package data
    data = {
        'Y': Y, 'M': M, 'time_features': time_features,
        'time_index': time_index, 'station_open_times': station_open_times,
        'N': N, 'num_providers': num_providers
    }
    
    # Run experiments
    coldstart_df = None
    adjacency_df = None
    lookback_df = None
    
    if args.experiment in ['all', 'coldstart']:
        coldstart_df = run_coldstart_sensitivity(
            data, static_adj, station_features, provider_ids, args.checkpoint
        )
        coldstart_df.to_csv(os.path.join(args.output, 'coldstart_sensitivity.csv'), index=False)
        print("\nCold-start results:")
        print(coldstart_df.to_string(index=False))
    
    if args.experiment in ['all', 'adjacency']:
        adjacency_df = run_adjacency_sensitivity(
            data, station_features_raw, station_features, provider_ids
        )
        adjacency_df.to_csv(os.path.join(args.output, 'adjacency_sensitivity.csv'), index=False)
        print("\nAdjacency results:")
        print(adjacency_df.to_string(index=False))
    
    if args.experiment in ['all', 'lookback']:
        lookback_df = run_lookback_sensitivity(
            data, static_adj, station_features, provider_ids
        )
        lookback_df.to_csv(os.path.join(args.output, 'lookback_sensitivity.csv'), index=False)
        print("\nLookback results:")
        print(lookback_df.to_string(index=False))
    
    # Generate LaTeX tables
    generate_latex_tables(coldstart_df, adjacency_df, lookback_df, args.output)
    
    print("\n" + "="*70)
    print("SENSITIVITY ANALYSIS COMPLETE")
    print("="*70)
    print(f"Results saved to: {args.output}/")


if __name__ == '__main__':
    main()
