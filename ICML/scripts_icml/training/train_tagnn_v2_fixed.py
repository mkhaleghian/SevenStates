"""
TA-GNN Dynamic v2 FIXED Training Script

FIXES from v2:
1. HURDLE FIX: Evaluation uses prob × count for expected value
2. MSE OPTION: Alternative simple MSE loss for fair comparison
3. M_HIST INPUT: Uses mask history to distinguish pre-open vs zero-demand
4. TRULY SOFT PRIOR: No extreme log penalty for non-edges
5. AGE-GATED FUSION: Young → spatial, Old → temporal
6. NORMALIZED FEATURES: Z-score normalization with missing indicator
7. BETTER METRICS: AUPRC, positive-only MAE, overall MAE

Usage:
    # Hurdle loss (probabilistic)
    python train_tagnn_v2_fixed.py --loss hurdle --epochs 100
    
    # MSE loss (comparable to baselines)
    python train_tagnn_v2_fixed.py --loss mse --epochs 100

Author: EV_GNN Research Project
"""

import os
import sys
import json
import argparse
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = os.environ.get('EV_GNN_ROOT', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# =============================================================================
# Dataset with Time-Varying Age
# =============================================================================

class EVChargingDatasetV2(Dataset):
    def __init__(self, Y, M, time_features, time_index, station_open_times,
                 lookback=168, horizon=5, mode='train', 
                 train_ratio=0.7, val_ratio=0.15):
        
        self.Y = torch.FloatTensor(Y)
        self.M = torch.FloatTensor(M)
        self.time_features = torch.FloatTensor(time_features)
        self.lookback = lookback
        self.horizon = horizon
        self.time_index = time_index
        self.station_open_times = station_open_times
        
        T, N = Y.shape
        train_end = int(T * train_ratio)
        val_end = int(T * (train_ratio + val_ratio))
        
        if mode == 'train':
            self.time_start, self.time_end = lookback, train_end - horizon
        elif mode == 'val':
            self.time_start, self.time_end = train_end, val_end - horizon
        else:
            self.time_start, self.time_end = val_end, T - horizon
        
        self.valid_times = list(range(self.time_start, self.time_end))
    
    def __len__(self):
        return len(self.valid_times)
    
    def __getitem__(self, idx):
        t = self.valid_times[idx]
        current_time = self.time_index[t]
        
        # Time-varying age (in years)
        ages = (current_time - self.station_open_times).astype('timedelta64[h]').astype(float) / (24 * 365.25)
        ages = np.maximum(ages, 0)
        
        return {
            'y_hist': self.Y[t-self.lookback:t, :].T,      # (N, lookback)
            'm_hist': self.M[t-self.lookback:t, :].T,      # (N, lookback) - NOW USED!
            'time_hist': self.time_features[t-self.lookback:t],
            'y_target': self.Y[t:t+self.horizon, :].T,
            'm_target': self.M[t:t+self.horizon, :].T,
            'ages': torch.FloatTensor(ages),
            't_idx': t,
        }


# =============================================================================
# Model Components
# =============================================================================

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
    """Encode static station features with proper normalization."""
    
    def __init__(self, num_providers, hidden_dim, dropout=0.1):
        super().__init__()
        
        # Input: [num_evses_norm, lat_norm, lon_norm, has_coords]
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


class DynamicGraphAttentionV2Fixed(nn.Module):
    """
    Graph Attention with TRULY SOFT adjacency prior.
    
    Fixes:
    - Non-edges get bias=0, not log(1e-6)=-13.8
    - Allows cross-provider and long-range transfer
    """
    
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
        
        # Learnable scales
        self.adj_bias = nn.Parameter(torch.ones(num_heads) * adj_bias_scale)
        self.age_bias = nn.Parameter(torch.ones(num_heads) * age_bias_scale)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, ages, static_adj=None):
        B, N, D = x.shape
        H = self.num_heads
        head_dim = self.head_dim
        
        Q = self.W_q(x).view(B, N, H, head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, N, H, head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, N, H, head_dim).transpose(1, 2)
        
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # =================================================================
        # TRULY SOFT ADJACENCY PRIOR (FIX!)
        # Non-edges get 0 bias, not extreme negative
        # =================================================================
        if static_adj is not None:
            # Option: adj_bias = adj * scale (linear, no log)
            # This gives bonus to edges but doesn't kill non-edges
            adj_bias = static_adj.unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)
            adj_bias = adj_bias * self.adj_bias.view(1, H, 1, 1)
            attn_logits = attn_logits + adj_bias
        
        # =================================================================
        # DIRECTIONAL AGE BIAS
        # =================================================================
        age_i = ages.unsqueeze(-1)
        age_j = ages.unsqueeze(-2)
        relative_age = age_j - age_i
        age_direction_bias = F.relu(relative_age).unsqueeze(1)
        age_direction_bias = age_direction_bias * self.age_bias.view(1, H, 1, 1)
        attn_logits = attn_logits + age_direction_bias
        
        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)
        out = self.layer_norm(x + out)
        
        return out, attn_weights


class AgeGatedFusion(nn.Module):
    """
    Age-dependent gate for fusing spatial and temporal representations.
    
    Young stations → rely more on spatial (neighbors)
    Old stations → rely more on temporal (self-history)
    
    g = sigmoid(W @ age_embed)
    h = g * h_spatial + (1-g) * h_temporal
    """
    
    def __init__(self, hidden_dim, age_embed_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(age_embed_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid(),
        )
    
    def forward(self, h_spatial, h_temporal, age_embed):
        """
        h_spatial: (B, N, hidden)
        h_temporal: (B, N, hidden)
        age_embed: (B, N, age_embed_dim)
        """
        g = self.gate(age_embed)  # (B, N, hidden)
        
        # Young (small age) → g should be HIGH (more spatial)
        # But sigmoid(small) → small, so we invert:
        # Actually: young → small age_embed values → small g
        # We want young → more spatial, so: h = (1-g)*spatial + g*temporal
        # This way: young (small g) → more spatial, old (large g) → more temporal
        
        h = (1 - g) * h_spatial + g * h_temporal
        return h, g


class TemporalEncoder(nn.Module):
    def __init__(self, hidden_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers, 
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
    
    def forward(self, x):
        B, N, T, D = x.shape
        x = x.view(B * N, T, D)
        output, _ = self.gru(x)
        return output[:, -1, :].view(B, N, D)


# =============================================================================
# Main Model
# =============================================================================

class TAGNNv2Fixed(nn.Module):
    """
    TA-GNN with all fixes:
    1. Uses m_hist as input
    2. Truly soft adjacency prior
    3. Age-gated fusion
    4. Proper hurdle output
    """
    
    def __init__(self, num_nodes, num_providers, time_feat_dim=7, 
                 hidden_dim=64, num_gnn_layers=2, num_temporal_layers=2,
                 num_heads=4, dropout=0.2, horizon=5, max_age=5.0):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.age_embed_dim = 32
        
        # Input: y + mask + time features (FIX: includes m_hist!)
        self.input_proj = nn.Linear(2 + time_feat_dim, hidden_dim)
        
        # Station features
        self.station_encoder = StationFeatureEncoder(num_providers, hidden_dim, dropout)
        
        # Age embedding
        self.age_embedding = SinusoidalAgeEmbedding(self.age_embed_dim, max_age)
        
        # Feature fusion
        self.feature_fusion = nn.Linear(hidden_dim + hidden_dim + self.age_embed_dim, hidden_dim)
        
        # Graph attention layers
        self.gnn_layers = nn.ModuleList([
            DynamicGraphAttentionV2Fixed(hidden_dim, num_heads, dropout)
            for _ in range(num_gnn_layers)
        ])
        
        # Temporal encoder
        self.temporal_encoder = TemporalEncoder(hidden_dim, num_temporal_layers, dropout)
        
        # Age-gated fusion (NEW!)
        self.age_gated_fusion = AgeGatedFusion(hidden_dim, self.age_embed_dim)
        
        # Output heads
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
        """
        y_hist: (B, N, T) historical demand
        m_hist: (B, N, T) historical mask - NOW USED!
        """
        B, N, T = y_hist.shape
        
        # =================================================================
        # Input: y + mask + time (FIX: includes m_hist!)
        # =================================================================
        y = y_hist.unsqueeze(-1)  # (B, N, T, 1)
        m = m_hist.unsqueeze(-1)  # (B, N, T, 1)
        
        if time_features.dim() == 2:
            time_features = time_features.unsqueeze(0).expand(B, -1, -1)
        time_exp = time_features.unsqueeze(1).expand(-1, N, -1, -1)
        
        # Concatenate y, m, time
        h = torch.cat([y, m, time_exp], dim=-1)  # (B, N, T, 2+feat)
        h = self.input_proj(h)
        
        # =================================================================
        # Station Features
        # =================================================================
        station_embed = self.station_encoder(station_features, provider_ids)
        station_embed = station_embed.unsqueeze(0).unsqueeze(2).expand(B, -1, T, -1)
        
        # =================================================================
        # Age Embedding
        # =================================================================
        age_embed = self.age_embedding(ages)  # (B, N, age_embed_dim)
        age_embed_exp = age_embed.unsqueeze(2).expand(-1, -1, T, -1)
        
        # =================================================================
        # Feature Fusion
        # =================================================================
        h = torch.cat([h, station_embed, age_embed_exp], dim=-1)
        h = self.feature_fusion(h)
        h = F.relu(h)
        
        # =================================================================
        # Spatial: Dynamic Graph Attention
        # =================================================================
        h_spatial = h[:, :, -1, :]
        
        attn_weights_list = []
        for gnn_layer in self.gnn_layers:
            h_spatial, attn_weights = gnn_layer(h_spatial, ages, static_adj)
            attn_weights_list.append(attn_weights)
        
        # =================================================================
        # Temporal: GRU
        # =================================================================
        h_temporal = self.temporal_encoder(h)
        
        # =================================================================
        # Age-Gated Fusion (NEW!)
        # =================================================================
        h_fused, gate_values = self.age_gated_fusion(h_spatial, h_temporal, age_embed)
        
        # =================================================================
        # Output
        # =================================================================
        count_pred = self.count_head(h_fused)
        count_pred = F.softplus(count_pred)
        
        prob_pred = self.prob_head(h_fused)
        prob_pred = torch.sigmoid(prob_pred)
        
        return {
            'count': count_pred,      # E[Y | Y > 0]
            'prob': prob_pred,        # P(Y > 0)
            'expected': prob_pred * count_pred,  # E[Y] = P(Y>0) * E[Y|Y>0]
            'gate': gate_values,      # For visualization
            'attn': attn_weights_list,
        }


# =============================================================================
# Loss Functions
# =============================================================================

class HurdleLoss(nn.Module):
    """Hurdle model: classification + count regression."""
    
    def __init__(self, alpha=0.3):
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCELoss(reduction='none')
    
    def forward(self, output, target, mask):
        prob_pred = output['prob']
        count_pred = output['count']
        
        eps = 1e-8
        binary_target = (target > 0).float()
        
        # Classification loss
        bce_loss = self.bce(prob_pred, binary_target)
        bce_loss = (bce_loss * mask).sum() / (mask.sum() + eps)
        
        # Count loss (only on positives)
        positive_mask = mask * binary_target
        mse_loss = (count_pred - target) ** 2
        count_loss = (mse_loss * positive_mask).sum() / (positive_mask.sum() + eps)
        
        return self.alpha * bce_loss + (1 - self.alpha) * count_loss


class MSELoss(nn.Module):
    """Simple MSE loss for fair comparison with baselines."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, output, target, mask):
        # Use expected value (prob × count)
        pred = output['expected']
        eps = 1e-8
        mse = (pred - target) ** 2
        return (mse * mask).sum() / (mask.sum() + eps)


# =============================================================================
# Comprehensive Evaluation
# =============================================================================

def comprehensive_evaluate(model, loader, device, static_adj, station_features, 
                           provider_ids, cold_start_days=14, use_expected=True):
    """
    Comprehensive evaluation with multiple metrics.
    
    Returns:
    - Overall MAE/RMSE
    - Cold-start window MAE (first W days)
    - Positive-only MAE (when Y > 0)
    - AUPRC for non-zero detection
    """
    model.eval()
    
    window_hours = cold_start_days * 24
    
    all_results = {
        'overall': {'pred': [], 'target': [], 'mask': [], 'prob': []},
        'cold_start': {'pred': [], 'target': [], 'mask': [], 'prob': []},
        'positive': {'pred': [], 'target': [], 'mask': []},
    }
    
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
            
            # Use expected value (prob × count) for evaluation
            if use_expected:
                pred = output['expected']
            else:
                pred = output['count']
            prob = output['prob']
            
            # Ages in hours
            ages_hours = ages * 24 * 365.25
            cs_mask = (ages_hours < window_hours).unsqueeze(-1).expand_as(m_target)
            pos_mask = (y_target > 0)
            
            # Store results
            all_results['overall']['pred'].append(pred.cpu())
            all_results['overall']['target'].append(y_target.cpu())
            all_results['overall']['mask'].append(m_target.cpu())
            all_results['overall']['prob'].append(prob.cpu())
            
            all_results['cold_start']['pred'].append(pred.cpu())
            all_results['cold_start']['target'].append(y_target.cpu())
            all_results['cold_start']['mask'].append((m_target * cs_mask.float()).cpu())
            all_results['cold_start']['prob'].append(prob.cpu())
            
            all_results['positive']['pred'].append(pred.cpu())
            all_results['positive']['target'].append(y_target.cpu())
            all_results['positive']['mask'].append((m_target * pos_mask.float()).cpu())
    
    # Compute metrics
    metrics = {}
    
    for key in ['overall', 'cold_start', 'positive']:
        pred = torch.cat(all_results[key]['pred']).reshape(-1)
        target = torch.cat(all_results[key]['target']).reshape(-1)
        mask = torch.cat(all_results[key]['mask']).reshape(-1)
        
        valid = mask > 0
        if valid.sum() > 0:
            metrics[f'{key}_mae'] = float((pred[valid] - target[valid]).abs().mean())
            metrics[f'{key}_rmse'] = float(((pred[valid] - target[valid]) ** 2).mean().sqrt())
            metrics[f'{key}_count'] = int(valid.sum())
    
    # AUPRC for non-zero detection
    prob_all = torch.cat(all_results['overall']['prob']).reshape(-1)
    target_all = torch.cat(all_results['overall']['target']).reshape(-1)
    mask_all = torch.cat(all_results['overall']['mask']).reshape(-1)
    
    valid = mask_all > 0
    if valid.sum() > 0:
        binary_target = (target_all[valid] > 0).numpy()
        prob_pred = prob_all[valid].numpy()
        
        try:
            metrics['auprc'] = float(average_precision_score(binary_target, prob_pred))
            metrics['auroc'] = float(roc_auc_score(binary_target, prob_pred))
        except:
            metrics['auprc'] = 0.0
            metrics['auroc'] = 0.5
    
    return metrics


# =============================================================================
# Training
# =============================================================================

def train_epoch(model, loader, optimizer, criterion, device, 
                static_adj, station_features, provider_ids):
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
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
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(loader)


# =============================================================================
# Preprocessing: Normalize Station Features
# =============================================================================

def normalize_station_features(station_features):
    """
    Normalize station features and add has_coords indicator.
    
    Input: (N, 3) - [num_evses, lat, lon]
    Output: (N, 4) - [num_evses_norm, lat_norm, lon_norm, has_coords]
    """
    N = station_features.shape[0]
    
    num_evses = station_features[:, 0]
    lat = station_features[:, 1]
    lon = station_features[:, 2]
    
    # Detect missing coords (set to 0 in preprocessing)
    has_coords = ((lat != 0) | (lon != 0)).float()
    
    # Z-score normalize
    def zscore(x, mask=None):
        if mask is not None:
            valid = mask > 0
            if valid.sum() > 0:
                mean = x[valid].mean()
                std = x[valid].std() + 1e-8
            else:
                mean, std = 0, 1
        else:
            mean = x.mean()
            std = x.std() + 1e-8
        return (x - mean) / std
    
    num_evses_norm = zscore(num_evses)
    lat_norm = zscore(lat, has_coords)
    lon_norm = zscore(lon, has_coords)
    
    # Set missing to 0 after normalization
    lat_norm = lat_norm * has_coords
    lon_norm = lon_norm * has_coords
    
    return torch.stack([num_evses_norm, lat_norm, lon_norm, has_coords], dim=1)


# =============================================================================
# Main
# =============================================================================

def main(args):
    print("=" * 70)
    print("   TA-GNN v2 FIXED Training")
    print(f"   Loss: {args.loss.upper()}")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load data
    print("\nLoading data...")
    data_path = f"{PROJECT_ROOT}/data/processed/tensors/data_v4.pt"
    if not os.path.exists(data_path):
        print(f"❌ Data not found: {data_path}")
        print("   Run preprocess_v4.py first!")
        sys.exit(1)
    
    data = torch.load(data_path, weights_only=False)
    
    Y = data['Y'].numpy()
    M = data['M'].numpy()
    time_features = data['time_features'].numpy()
    time_index = data['time_index']
    station_open_times = data['station_open_times']
    station_features_raw = data['station_features'].float()
    provider_ids = data['provider_ids'].long().to(device)
    provider_list = data['provider_list']
    
    # Normalize station features (FIX!)
    station_features = normalize_station_features(station_features_raw).to(device)
    
    T, N = Y.shape
    num_providers = len(provider_list)
    
    print(f"Data: T={T:,}, N={N}")
    print(f"Mask coverage: {M.mean()*100:.2f}%")
    print(f"Positive hours: {(Y > 0).sum():,} ({(Y > 0).sum() / M.sum() * 100:.2f}% of observed)")
    
    # Load adjacency
    adj_path = f"{PROJECT_ROOT}/data/graphs/adjacency_v4.pt"
    if os.path.exists(adj_path):
        adj_data = torch.load(adj_path, weights_only=False)
        static_adj = adj_data['A_distance'].float().to(device)
    else:
        static_adj = torch.eye(N).to(device)
    
    # Create datasets
    print("\nCreating datasets...")
    train_ds = EVChargingDatasetV2(Y, M, time_features, time_index, station_open_times, mode='train')
    val_ds = EVChargingDatasetV2(Y, M, time_features, time_index, station_open_times, mode='val')
    test_ds = EVChargingDatasetV2(Y, M, time_features, time_index, station_open_times, mode='test')
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    # Create model
    print("\nCreating model...")
    model = TAGNNv2Fixed(
        num_nodes=N,
        num_providers=num_providers,
        time_feat_dim=time_features.shape[1],
        hidden_dim=args.hidden_dim,
        num_gnn_layers=args.num_gnn_layers,
        num_temporal_layers=args.num_temporal_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        horizon=args.horizon,
        max_age=5.0,
    ).to(device)
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function
    if args.loss == 'hurdle':
        criterion = HurdleLoss(alpha=0.3)
        use_expected = True
    else:
        criterion = MSELoss()
        use_expected = True
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Training
    print("\n" + "=" * 70)
    print("Training...")
    print("=" * 70)
    
    best_mae = float('inf')
    patience_counter = 0
    history = {'train': [], 'val_mae': []}
    
    checkpoint_dir = f"{PROJECT_ROOT}/checkpoints/tagnn_v2_fixed_{args.loss}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device,
                                  static_adj, station_features, provider_ids)
        
        val_metrics = comprehensive_evaluate(model, val_loader, device, static_adj,
                                              station_features, provider_ids,
                                              use_expected=use_expected)
        
        scheduler.step()
        
        history['train'].append(float(train_loss))
        history['val_mae'].append(float(val_metrics['overall_mae']))
        
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train: {train_loss:.4f} | "
              f"Val MAE: {val_metrics['overall_mae']:.4f} | "
              f"Pos MAE: {val_metrics.get('positive_mae', 0):.4f} | "
              f"LR: {lr:.6f}")
        
        if val_metrics['overall_mae'] < best_mae - 0.001:
            best_mae = val_metrics['overall_mae']
            patience_counter = 0
            torch.save(model.state_dict(), f"{checkpoint_dir}/best_model.pt")
            print(f"  ✓ Saved (MAE: {best_mae:.4f})")
        else:
            patience_counter += 1
        
        if patience_counter >= args.patience:
            print(f"\n⚠️ Early stopping at epoch {epoch}")
            break
    
    # Final evaluation
    print("\n" + "=" * 70)
    print("Final Evaluation")
    print("=" * 70)
    
    model.load_state_dict(torch.load(f"{checkpoint_dir}/best_model.pt", weights_only=True))
    
    test_metrics = comprehensive_evaluate(model, test_loader, device, static_adj,
                                           station_features, provider_ids,
                                           cold_start_days=args.cold_start_days,
                                           use_expected=use_expected)
    
    print(f"\n📊 Test Results:")
    print(f"   Overall MAE:      {test_metrics['overall_mae']:.4f}")
    print(f"   Overall RMSE:     {test_metrics['overall_rmse']:.4f}")
    print(f"   Cold-Start MAE:   {test_metrics.get('cold_start_mae', 'N/A')}")
    print(f"   Positive-Only MAE: {test_metrics.get('positive_mae', 'N/A')}")
    print(f"   AUPRC:            {test_metrics.get('auprc', 'N/A')}")
    print(f"   AUROC:            {test_metrics.get('auroc', 'N/A')}")
    
    # Save results
    results = {
        'test': test_metrics,
        'history': history,
        'config': vars(args),
    }
    
    with open(f"{checkpoint_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {checkpoint_dir}/")
    
    return model, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss', type=str, default='mse', choices=['hurdle', 'mse'],
                        help='Loss function: hurdle or mse')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--num_gnn_layers', type=int, default=2)
    parser.add_argument('--num_temporal_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--horizon', type=int, default=5)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--cold_start_days', type=int, default=14)
    
    args = parser.parse_args()
    main(args)
