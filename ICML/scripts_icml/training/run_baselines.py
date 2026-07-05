"""
Baseline Comparison for TA-GNN Paper

Runs multiple baselines to compare against TA-GNN Dynamic:
1. Historical Average (naive baseline)
2. Last Value (persistence)
3. LSTM-only (no graph)
4. Static GCN (fixed graph convolution)
5. Static GAT (graph attention, no age modulation)
6. TA-GNN Static (our model without dynamic attention)

Usage:
    python scripts/training/run_baselines.py

Author: EV_GNN Research Project
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math
from datetime import datetime

PROJECT_ROOT = os.environ.get('EV_GNN_ROOT', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# =============================================================================
# Dataset (same as main training)
# =============================================================================

class EVChargingDataset(Dataset):
    def __init__(self, Y, M, time_features, lookback=168, horizon=5, 
                 mode='train', train_ratio=0.7, val_ratio=0.15):
        self.Y = torch.FloatTensor(Y)
        self.M = torch.FloatTensor(M)
        self.time_features = torch.FloatTensor(time_features)
        self.lookback, self.horizon = lookback, horizon
        
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
        return {
            'y_hist': self.Y[t-self.lookback:t, :].T,
            'time_hist': self.time_features[t-self.lookback:t],
            'y_target': self.Y[t:t+self.horizon, :].T,
            'm_target': self.M[t:t+self.horizon, :].T,
            't_idx': t,
        }


# =============================================================================
# Baseline 1: Historical Average
# =============================================================================

class HistoricalAverage:
    """Predicts the mean of historical values."""
    
    def __init__(self):
        self.name = "Historical Average"
    
    def predict(self, y_hist, mask_hist=None):
        """
        y_hist: (B, N, T) historical values
        Returns: (B, N, horizon) predictions
        """
        B, N, T = y_hist.shape
        # Mean over time dimension
        mean_val = y_hist.mean(dim=2, keepdim=True)  # (B, N, 1)
        return mean_val.expand(B, N, 5)  # Repeat for horizon


# =============================================================================
# Baseline 2: Last Value (Persistence)
# =============================================================================

class LastValue:
    """Predicts the last observed value."""
    
    def __init__(self):
        self.name = "Last Value (Persistence)"
    
    def predict(self, y_hist, mask_hist=None):
        """
        y_hist: (B, N, T) historical values
        Returns: (B, N, horizon) predictions
        """
        B, N, T = y_hist.shape
        last_val = y_hist[:, :, -1:]  # (B, N, 1)
        return last_val.expand(B, N, 5)


# =============================================================================
# Baseline 3: LSTM-only (No Graph)
# =============================================================================

class LSTMOnly(nn.Module):
    """Pure LSTM without any graph structure."""
    
    def __init__(self, num_nodes, time_feat_dim=7, hidden_dim=64, 
                 num_layers=2, dropout=0.2, horizon=5):
        super().__init__()
        self.name = "LSTM-only"
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        
        # Input projection
        self.input_proj = nn.Linear(1 + time_feat_dim, hidden_dim)
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # Output
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, horizon),
        )
    
    def forward(self, x, time_features, adj=None, ages=None):
        """
        x: (B, T, N) historical counts
        time_features: (T, feat_dim)
        """
        B, T, N = x.shape
        
        if time_features.dim() == 2:
            time_features = time_features.unsqueeze(0).expand(B, -1, -1)
        
        # Process each node independently
        x = x.unsqueeze(-1)  # (B, T, N, 1)
        time_exp = time_features.unsqueeze(2).expand(-1, -1, N, -1)
        h = torch.cat([x, time_exp], dim=-1)  # (B, T, N, 1+feat)
        h = self.input_proj(h)  # (B, T, N, hidden)
        
        # Reshape for LSTM: (B*N, T, hidden)
        h = h.permute(0, 2, 1, 3).reshape(B * N, T, self.hidden_dim)
        
        # LSTM
        h, _ = self.lstm(h)
        h = h[:, -1, :]  # (B*N, hidden)
        
        # Output
        pred = self.output_proj(h)  # (B*N, horizon)
        pred = pred.view(B, N, self.horizon)
        pred = F.softplus(pred)
        
        return pred


# =============================================================================
# Baseline 4: Static GCN
# =============================================================================

class StaticGCN(nn.Module):
    """Standard GCN with fixed adjacency matrix."""
    
    def __init__(self, num_nodes, time_feat_dim=7, hidden_dim=64,
                 num_gnn_layers=2, num_temporal_layers=2, dropout=0.2, horizon=5):
        super().__init__()
        self.name = "Static GCN"
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        
        # Input projection
        self.input_proj = nn.Linear(1 + time_feat_dim, hidden_dim)
        
        # GCN layers
        self.gcn_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_gnn_layers)
        ])
        self.gcn_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_gnn_layers)
        ])
        
        # Temporal GRU
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_temporal_layers,
            batch_first=True,
            dropout=dropout if num_temporal_layers > 1 else 0,
        )
        
        # Output
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon),
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, time_features, adj, ages=None):
        B, T, N = x.shape
        
        if time_features.dim() == 2:
            time_features = time_features.unsqueeze(0).expand(B, -1, -1)
        
        # Input embedding
        x = x.unsqueeze(-1)
        time_exp = time_features.unsqueeze(2).expand(-1, -1, N, -1)
        h = torch.cat([x, time_exp], dim=-1)
        h = self.input_proj(h)  # (B, T, N, hidden)
        
        # GCN on last time step
        h_spatial = h[:, -1, :, :]  # (B, N, hidden)
        for gcn, norm in zip(self.gcn_layers, self.gcn_norms):
            # Graph convolution: h' = A @ h @ W
            h_agg = torch.matmul(adj, h_spatial)  # (B, N, hidden)
            h_agg = gcn(h_agg)
            h_agg = F.relu(h_agg)
            h_agg = self.dropout(h_agg)
            h_spatial = norm(h_spatial + h_agg)  # Residual
        
        # Temporal GRU
        h_temp = h.permute(0, 2, 1, 3).reshape(B * N, T, self.hidden_dim)
        h_temp, _ = self.gru(h_temp)
        h_temp = h_temp[:, -1, :].view(B, N, self.hidden_dim)
        
        # Output
        h_out = torch.cat([h_spatial, h_temp], dim=-1)
        pred = self.output_proj(h_out)
        pred = F.softplus(pred)
        
        return pred


# =============================================================================
# Baseline 5: Static GAT (no age modulation)
# =============================================================================

class GraphAttentionLayer(nn.Module):
    """Standard GAT layer without age modulation."""
    
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, adj):
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        
        B, N, _ = x.shape
        H, D = self.num_heads, self.head_dim
        
        Q = self.W_q(x).view(B, N, H, D).transpose(1, 2)
        K = self.W_k(x).view(B, N, H, D).transpose(1, 2)
        V = self.W_v(x).view(B, N, H, D).transpose(1, 2)
        
        # Standard attention
        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)
        
        # Mask non-edges
        if adj is not None:
            mask = (adj == 0).unsqueeze(0).unsqueeze(0)
            attn = attn.masked_fill(mask, -1e9)
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, N, self.hidden_dim)
        out = self.out_proj(out)
        out = self.layer_norm(x + out)
        
        if squeeze:
            out = out.squeeze(0)
        
        return out


class StaticGAT(nn.Module):
    """GAT without age modulation."""
    
    def __init__(self, num_nodes, time_feat_dim=7, hidden_dim=64,
                 num_gnn_layers=2, num_temporal_layers=2, num_heads=4,
                 dropout=0.2, horizon=5):
        super().__init__()
        self.name = "Static GAT"
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        
        self.input_proj = nn.Linear(1 + time_feat_dim, hidden_dim)
        
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_gnn_layers)
        ])
        
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_temporal_layers,
            batch_first=True,
            dropout=dropout if num_temporal_layers > 1 else 0,
        )
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon),
        )
    
    def forward(self, x, time_features, adj, ages=None):
        B, T, N = x.shape
        
        if time_features.dim() == 2:
            time_features = time_features.unsqueeze(0).expand(B, -1, -1)
        
        x = x.unsqueeze(-1)
        time_exp = time_features.unsqueeze(2).expand(-1, -1, N, -1)
        h = torch.cat([x, time_exp], dim=-1)
        h = self.input_proj(h)
        
        # GAT on last time step
        h_spatial = h[:, -1, :, :]
        for gat in self.gat_layers:
            h_spatial = gat(h_spatial, adj)
        
        # Temporal
        h_temp = h.permute(0, 2, 1, 3).reshape(B * N, T, self.hidden_dim)
        h_temp, _ = self.gru(h_temp)
        h_temp = h_temp[:, -1, :].view(B, N, self.hidden_dim)
        
        # Output
        h_out = torch.cat([h_spatial, h_temp], dim=-1)
        pred = self.output_proj(h_out)
        pred = F.softplus(pred)
        
        return pred


# =============================================================================
# Baseline 6: TA-GNN Static (our model with fixed attention)
# =============================================================================

class HistoryEmbedding(nn.Module):
    def __init__(self, embed_dim, max_age=5.0):
        super().__init__()
        self.embed_dim, self.max_age = embed_dim, max_age
        freqs = torch.exp(torch.linspace(0, np.log(max_age + 1), embed_dim // 2))
        self.register_buffer('freq_bands', freqs)
    
    def forward(self, age):
        age_norm = (age / self.max_age).unsqueeze(-1)
        angles = age_norm * self.freq_bands.unsqueeze(0) * math.pi
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class TAGNNStatic(nn.Module):
    """TA-GNN with history embedding but WITHOUT dynamic attention."""
    
    def __init__(self, num_nodes, time_feat_dim=7, hidden_dim=64,
                 num_gnn_layers=2, num_temporal_layers=2, num_heads=4,
                 dropout=0.2, horizon=5, max_age=5.0):
        super().__init__()
        self.name = "TA-GNN Static"
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.history_embed_dim = 32
        
        self.input_proj = nn.Linear(1 + time_feat_dim, hidden_dim)
        self.history_embedding = HistoryEmbedding(32, max_age)
        self.history_proj = nn.Linear(hidden_dim + 32, hidden_dim)
        
        # Static GAT (no age modulation)
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_gnn_layers)
        ])
        
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_temporal_layers,
            batch_first=True,
            dropout=dropout if num_temporal_layers > 1 else 0,
        )
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon),
        )
    
    def forward(self, x, time_features, adj, ages):
        B, T, N = x.shape
        
        if time_features.dim() == 2:
            time_features = time_features.unsqueeze(0).expand(B, -1, -1)
        
        x = x.unsqueeze(-1)
        time_exp = time_features.unsqueeze(2).expand(-1, -1, N, -1)
        h = torch.cat([x, time_exp], dim=-1)
        h = self.input_proj(h)
        
        # Add history embedding
        age_embed = self.history_embedding(ages)
        age_exp = age_embed.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
        h = torch.cat([h, age_exp], dim=-1)
        h = self.history_proj(h)
        h = F.relu(h)
        
        # Static GAT
        h_spatial = h[:, -1, :, :]
        for gat in self.gat_layers:
            h_spatial = gat(h_spatial, adj)
        
        # Temporal
        h_temp = h.permute(0, 2, 1, 3).reshape(B * N, T, self.hidden_dim)
        h_temp, _ = self.gru(h_temp)
        h_temp = h_temp[:, -1, :].view(B, N, self.hidden_dim)
        
        # Output
        h_out = torch.cat([h_spatial, h_temp], dim=-1)
        pred = self.output_proj(h_out)
        pred = F.softplus(pred)
        
        return pred


# =============================================================================
# Training and Evaluation Functions
# =============================================================================

def train_model(model, train_loader, val_loader, adj, ages, device, 
                epochs=50, lr=0.001, patience=15):
    """Train a neural network model."""
    
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_mae = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for batch in train_loader:
            y_hist = batch['y_hist'].to(device).transpose(1, 2)
            time_hist = batch['time_hist'].to(device)
            y_target = batch['y_target'].to(device)
            m_target = batch['m_target'].to(device)
            
            optimizer.zero_grad()
            pred = model(y_hist, time_hist, adj, ages)
            
            loss = ((pred - y_target) ** 2 * m_target).sum() / (m_target.sum() + 1e-8)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_preds, val_targets, val_masks = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                y_hist = batch['y_hist'].to(device).transpose(1, 2)
                time_hist = batch['time_hist'].to(device)
                y_target = batch['y_target'].to(device)
                m_target = batch['m_target'].to(device)
                
                pred = model(y_hist, time_hist, adj, ages)
                val_preds.append(pred.cpu())
                val_targets.append(y_target.cpu())
                val_masks.append(m_target.cpu())
        
        preds = torch.cat(val_preds)
        targets = torch.cat(val_targets)
        masks = torch.cat(val_masks)
        
        mae = (preds[masks > 0] - targets[masks > 0]).abs().mean().item()
        scheduler.step(mae)
        
        if mae < best_mae - 0.001:
            best_mae = mae
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    model.load_state_dict(best_state)
    return model, best_mae


def evaluate_model(model, test_loader, adj, ages, device, ages_np):
    """Evaluate model and return metrics by age group."""
    
    model.eval() if hasattr(model, 'eval') else None
    
    bins = {
        'cold_start': (0, 0.25),
        'new': (0.25, 1.0),
        'established': (1.0, 2.0),
        'mature': (2.0, 100),
    }
    
    results = {k: {'p': [], 't': [], 'm': []} for k in bins}
    all_preds, all_targets, all_masks = [], [], []
    
    with torch.no_grad():
        for batch in test_loader:
            y_hist = batch['y_hist'].to(device).transpose(1, 2)  # (B, T, N) for models
            time_hist = batch['time_hist'].to(device)
            y_target = batch['y_target'].to(device)  # (B, N, horizon)
            m_target = batch['m_target'].to(device)  # (B, N, horizon)
            
            pred = model(y_hist, time_hist, adj, ages)  # (B, N, horizon)
            
            all_preds.append(pred.cpu())
            all_targets.append(y_target.cpu())
            all_masks.append(m_target.cpu())
            
            for name, (lo, hi) in bins.items():
                idx = np.where((ages_np >= lo) & (ages_np < hi))[0]
                if len(idx) > 0:
                    results[name]['p'].append(pred[:, idx, :].cpu())
                    results[name]['t'].append(y_target[:, idx, :].cpu())
                    results[name]['m'].append(m_target[:, idx, :].cpu())
    
    # Overall metrics
    preds = torch.cat(all_preds, dim=0).reshape(-1)
    targets = torch.cat(all_targets, dim=0).reshape(-1)
    masks = torch.cat(all_masks, dim=0).reshape(-1)
    
    valid = masks > 0
    overall_mae = (preds[valid] - targets[valid]).abs().mean().item()
    overall_rmse = ((preds[valid] - targets[valid]) ** 2).mean().sqrt().item()
    
    # Age-stratified metrics
    age_metrics = {}
    for name, data in results.items():
        if data['p']:
            p = torch.cat(data['p'], dim=0).reshape(-1)
            t = torch.cat(data['t'], dim=0).reshape(-1)
            m = torch.cat(data['m'], dim=0).reshape(-1)
            valid = m > 0
            if valid.sum() > 0:
                age_metrics[f'{name}_mae'] = (p[valid] - t[valid]).abs().mean().item()
                age_metrics[f'{name}_rmse'] = ((p[valid] - t[valid]) ** 2).mean().sqrt().item()
                age_metrics[f'{name}_count'] = int(valid.sum().item())
    
    return {
        'overall_mae': overall_mae,
        'overall_rmse': overall_rmse,
        **age_metrics
    }


def evaluate_simple_baseline(baseline, test_loader, device, ages_np):
    """Evaluate a simple (non-trainable) baseline."""
    
    bins = {
        'cold_start': (0, 0.25),
        'new': (0.25, 1.0),
        'established': (1.0, 2.0),
        'mature': (2.0, 100),
    }
    
    results = {k: {'p': [], 't': [], 'm': []} for k in bins}
    all_preds, all_targets, all_masks = [], [], []
    
    for batch in test_loader:
        # y_hist from dataset is (B, N, lookback) - DON'T transpose for simple baselines
        y_hist = batch['y_hist'].to(device)  # (B, N, T)
        y_target = batch['y_target'].to(device)  # (B, N, horizon)
        m_target = batch['m_target'].to(device)  # (B, N, horizon)
        
        pred = baseline.predict(y_hist)  # (B, N, horizon)
        
        all_preds.append(pred.cpu())
        all_targets.append(y_target.cpu())
        all_masks.append(m_target.cpu())
        
        for name, (lo, hi) in bins.items():
            idx = np.where((ages_np >= lo) & (ages_np < hi))[0]
            if len(idx) > 0:
                results[name]['p'].append(pred[:, idx, :].cpu())
                results[name]['t'].append(y_target[:, idx, :].cpu())
                results[name]['m'].append(m_target[:, idx, :].cpu())
    
    preds = torch.cat(all_preds, dim=0)      # (total_B, N, horizon)
    targets = torch.cat(all_targets, dim=0)
    masks = torch.cat(all_masks, dim=0)
    
    # Flatten for metrics
    preds_flat = preds.reshape(-1)
    targets_flat = targets.reshape(-1)
    masks_flat = masks.reshape(-1)
    
    valid = masks_flat > 0
    overall_mae = (preds_flat[valid] - targets_flat[valid]).abs().mean().item()
    overall_rmse = ((preds_flat[valid] - targets_flat[valid]) ** 2).mean().sqrt().item()
    
    age_metrics = {}
    for name, data in results.items():
        if data['p']:
            p = torch.cat(data['p'], dim=0).reshape(-1)
            t = torch.cat(data['t'], dim=0).reshape(-1)
            m = torch.cat(data['m'], dim=0).reshape(-1)
            valid = m > 0
            if valid.sum() > 0:
                age_metrics[f'{name}_mae'] = (p[valid] - t[valid]).abs().mean().item()
                age_metrics[f'{name}_rmse'] = ((p[valid] - t[valid]) ** 2).mean().sqrt().item()
                age_metrics[f'{name}_count'] = int(valid.sum().item())
    
    return {
        'overall_mae': overall_mae,
        'overall_rmse': overall_rmse,
        **age_metrics
    }


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("   BASELINE COMPARISON FOR TA-GNN PAPER")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load data
    print("\nLoading data...")
    data = torch.load(f"{PROJECT_ROOT}/data/processed/tensors/data.pt", weights_only=False)
    Y = data['Y'].numpy()
    M = data['M'].numpy()
    time_features = data['time_features'].numpy()
    
    adj_data = torch.load(f"{PROJECT_ROOT}/data/graphs/adjacency_v2.pt", weights_only=False)
    adj = adj_data['A_combined'].float().to(device)
    
    meta = pd.read_csv(f"{PROJECT_ROOT}/data/processed/unified/station_metadata.csv")
    ages = torch.FloatTensor(meta['operational_years'].values).to(device)
    ages_np = ages.cpu().numpy()
    
    T, N = Y.shape
    print(f"Data: T={T:,}, N={N}")
    
    # Create datasets
    print("\nCreating datasets...")
    train_ds = EVChargingDataset(Y, M, time_features, mode='train')
    val_ds = EVChargingDataset(Y, M, time_features, mode='val')
    test_ds = EVChargingDataset(Y, M, time_features, mode='test')
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    test_loader = DataLoader(test_ds, batch_size=32)
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    # Store all results
    all_results = {}
    
    # =========================================================================
    # Baseline 1: Historical Average
    # =========================================================================
    print("\n" + "=" * 70)
    print("Running Baseline 1: Historical Average")
    print("=" * 70)
    
    baseline_ha = HistoricalAverage()
    results_ha = evaluate_simple_baseline(baseline_ha, test_loader, device, ages_np)
    all_results['Historical Average'] = results_ha
    print(f"  Overall MAE: {results_ha['overall_mae']:.4f}")
    print(f"  Cold-start MAE: {results_ha.get('cold_start_mae', 'N/A'):.4f}")
    
    # =========================================================================
    # Baseline 2: Last Value
    # =========================================================================
    print("\n" + "=" * 70)
    print("Running Baseline 2: Last Value (Persistence)")
    print("=" * 70)
    
    baseline_lv = LastValue()
    results_lv = evaluate_simple_baseline(baseline_lv, test_loader, device, ages_np)
    all_results['Last Value'] = results_lv
    print(f"  Overall MAE: {results_lv['overall_mae']:.4f}")
    print(f"  Cold-start MAE: {results_lv.get('cold_start_mae', 'N/A'):.4f}")
    
    # =========================================================================
    # Baseline 3: LSTM-only
    # =========================================================================
    print("\n" + "=" * 70)
    print("Running Baseline 3: LSTM-only (no graph)")
    print("=" * 70)
    
    model_lstm = LSTMOnly(num_nodes=N, time_feat_dim=time_features.shape[1])
    print(f"  Parameters: {sum(p.numel() for p in model_lstm.parameters()):,}")
    
    model_lstm, val_mae = train_model(model_lstm, train_loader, val_loader, adj, ages, device)
    results_lstm = evaluate_model(model_lstm, test_loader, adj, ages, device, ages_np)
    all_results['LSTM-only'] = results_lstm
    print(f"  Overall MAE: {results_lstm['overall_mae']:.4f}")
    print(f"  Cold-start MAE: {results_lstm.get('cold_start_mae', 'N/A'):.4f}")
    
    # =========================================================================
    # Baseline 4: Static GCN
    # =========================================================================
    print("\n" + "=" * 70)
    print("Running Baseline 4: Static GCN")
    print("=" * 70)
    
    model_gcn = StaticGCN(num_nodes=N, time_feat_dim=time_features.shape[1])
    print(f"  Parameters: {sum(p.numel() for p in model_gcn.parameters()):,}")
    
    model_gcn, val_mae = train_model(model_gcn, train_loader, val_loader, adj, ages, device)
    results_gcn = evaluate_model(model_gcn, test_loader, adj, ages, device, ages_np)
    all_results['Static GCN'] = results_gcn
    print(f"  Overall MAE: {results_gcn['overall_mae']:.4f}")
    print(f"  Cold-start MAE: {results_gcn.get('cold_start_mae', 'N/A'):.4f}")
    
    # =========================================================================
    # Baseline 5: Static GAT
    # =========================================================================
    print("\n" + "=" * 70)
    print("Running Baseline 5: Static GAT (no age modulation)")
    print("=" * 70)
    
    model_gat = StaticGAT(num_nodes=N, time_feat_dim=time_features.shape[1])
    print(f"  Parameters: {sum(p.numel() for p in model_gat.parameters()):,}")
    
    model_gat, val_mae = train_model(model_gat, train_loader, val_loader, adj, ages, device)
    results_gat = evaluate_model(model_gat, test_loader, adj, ages, device, ages_np)
    all_results['Static GAT'] = results_gat
    print(f"  Overall MAE: {results_gat['overall_mae']:.4f}")
    print(f"  Cold-start MAE: {results_gat.get('cold_start_mae', 'N/A'):.4f}")
    
    # =========================================================================
    # Baseline 6: TA-GNN Static
    # =========================================================================
    print("\n" + "=" * 70)
    print("Running Baseline 6: TA-GNN Static (history embed, no dynamic attention)")
    print("=" * 70)
    
    model_tagnn_static = TAGNNStatic(num_nodes=N, time_feat_dim=time_features.shape[1],
                                      max_age=ages.max().item() + 0.5)
    print(f"  Parameters: {sum(p.numel() for p in model_tagnn_static.parameters()):,}")
    
    model_tagnn_static, val_mae = train_model(model_tagnn_static, train_loader, val_loader, 
                                               adj, ages, device)
    results_tagnn_static = evaluate_model(model_tagnn_static, test_loader, adj, ages, device, ages_np)
    all_results['TA-GNN Static'] = results_tagnn_static
    print(f"  Overall MAE: {results_tagnn_static['overall_mae']:.4f}")
    print(f"  Cold-start MAE: {results_tagnn_static.get('cold_start_mae', 'N/A'):.4f}")
    
    # =========================================================================
    # Load TA-GNN Dynamic Results
    # =========================================================================
    print("\n" + "=" * 70)
    print("Loading TA-GNN Dynamic (our method) results")
    print("=" * 70)
    
    tagnn_results_path = f"{PROJECT_ROOT}/checkpoints/tagnn_dynamic/results.json"
    if os.path.exists(tagnn_results_path):
        with open(tagnn_results_path, 'r') as f:
            tagnn_results = json.load(f)
        
        all_results['TA-GNN Dynamic (Ours)'] = {
            'overall_mae': tagnn_results['test']['mae'],
            'overall_rmse': tagnn_results['test']['rmse'],
            **{k: v for k, v in tagnn_results['age_metrics'].items()}
        }
        print(f"  Overall MAE: {tagnn_results['test']['mae']:.4f}")
        print(f"  Cold-start MAE: {tagnn_results['age_metrics'].get('cold_start_mae', 'N/A'):.4f}")
    else:
        print("  ⚠️ TA-GNN Dynamic results not found!")
    
    # =========================================================================
    # Create Comparison Table
    # =========================================================================
    print("\n" + "=" * 70)
    print("   COMPARISON TABLE")
    print("=" * 70)
    
    print(f"\n{'Method':<25} {'Overall':<10} {'Cold-Start':<12} {'New':<10} {'Established':<12} {'Mature':<10}")
    print("-" * 85)
    
    for method, results in all_results.items():
        overall = results.get('overall_mae', float('nan'))
        cold = results.get('cold_start_mae', float('nan'))
        new = results.get('new_mae', float('nan'))
        est = results.get('established_mae', float('nan'))
        mature = results.get('mature_mae', float('nan'))
        
        print(f"{method:<25} {overall:<10.4f} {cold:<12.4f} {new:<10.4f} {est:<12.4f} {mature:<10.4f}")
    
    # =========================================================================
    # Calculate Improvements
    # =========================================================================
    print("\n" + "=" * 70)
    print("   IMPROVEMENT OVER BASELINES (Cold-Start MAE)")
    print("=" * 70)
    
    if 'TA-GNN Dynamic (Ours)' in all_results:
        our_cold_start = all_results['TA-GNN Dynamic (Ours)'].get('cold_start_mae', float('nan'))
        
        print(f"\nTA-GNN Dynamic Cold-Start MAE: {our_cold_start:.4f}")
        print()
        
        for method, results in all_results.items():
            if method == 'TA-GNN Dynamic (Ours)':
                continue
            baseline_cold = results.get('cold_start_mae', float('nan'))
            if not np.isnan(baseline_cold) and baseline_cold > 0:
                improvement = (baseline_cold - our_cold_start) / baseline_cold * 100
                print(f"  vs {method:<20}: {improvement:>6.1f}% improvement")
    
    # =========================================================================
    # Save Results
    # =========================================================================
    output_path = f"{PROJECT_ROOT}/results/baseline_comparison.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert numpy types to Python types for JSON
    def convert_to_python(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_python(v) for k, v in obj.items()}
        return obj
    
    all_results_clean = convert_to_python(all_results)
    
    with open(output_path, 'w') as f:
        json.dump(all_results_clean, f, indent=2)
    
    print(f"\n✅ Results saved to {output_path}")
    
    # =========================================================================
    # Generate LaTeX Table
    # =========================================================================
    print("\n" + "=" * 70)
    print("   LATEX TABLE FOR PAPER")
    print("=" * 70)
    
    print("""
\\begin{table}[h]
\\centering
\\caption{Comparison of TA-GNN with baseline methods. MAE (sessions/hour) reported for different station age groups.}
\\label{tab:comparison}
\\begin{tabular}{lcccccc}
\\toprule
Method & Overall & Cold-Start & New & Established & Mature \\\\
\\midrule""")
    
    for method, results in all_results.items():
        overall = results.get('overall_mae', float('nan'))
        cold = results.get('cold_start_mae', float('nan'))
        new = results.get('new_mae', float('nan'))
        est = results.get('established_mae', float('nan'))
        mature = results.get('mature_mae', float('nan'))
        
        # Bold the best method
        if method == 'TA-GNN Dynamic (Ours)':
            print(f"\\textbf{{{method}}} & \\textbf{{{overall:.4f}}} & \\textbf{{{cold:.4f}}} & \\textbf{{{new:.4f}}} & \\textbf{{{est:.4f}}} & \\textbf{{{mature:.4f}}} \\\\")
        else:
            print(f"{method} & {overall:.4f} & {cold:.4f} & {new:.4f} & {est:.4f} & {mature:.4f} \\\\")
    
    print("""\\bottomrule
\\end{tabular}
\\end{table}
""")
    
    print("\n" + "=" * 70)
    print("   ✅ BASELINE COMPARISON COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
