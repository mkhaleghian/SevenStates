#!/usr/bin/env python3
"""
Quick Cold-Start Window Sensitivity Evaluation
==============================================

Evaluates your EXISTING trained model with different cold-start window durations.
NO RETRAINING REQUIRED - runs in ~5 minutes.

This script matches your exact model architecture from train_tagnn_dynamic_v4.py

Usage:
    cd ~/GNN
    python scripts/sensitivity/quick_coldstart_eval.py

Output:
    - results/sensitivity/coldstart_sensitivity.csv
    - results/sensitivity/table_coldstart.tex (LaTeX table for paper)
"""

import os
import sys
import math
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = os.environ.get('EV_GNN_ROOT', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_PATH = f"{PROJECT_ROOT}/data/processed/tensors/data_v4.pt"
ADJ_PATH = f"{PROJECT_ROOT}/data/graphs/adjacency_v4.pt"
CHECKPOINT_PATH = f"{PROJECT_ROOT}/checkpoints/tagnn_dynamic_v4/best_model_cs_pos.pt"
OUTPUT_DIR = f"{PROJECT_ROOT}/results/sensitivity"

# Model hyperparameters (must match your trained model)
HIDDEN_DIM = 64
NUM_GNN_LAYERS = 2
NUM_TEMPORAL_LAYERS = 2
NUM_HEADS = 4
DROPOUT = 0.2
LOOKBACK = 168
HORIZON = 5
MAX_AGE = 5.0

BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# MODEL COMPONENTS (exact copy from train_tagnn_dynamic_v4.py)
# ============================================================================

class SinusoidalAgeEmbedding(nn.Module):
    """Sinusoidal encoding for station operational age."""
    
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
    """Graph Attention with soft adjacency prior and directional age bias."""
    
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
    """Fixed Age-Gated Fusion: cold→spatial, mature→temporal."""
    
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
    """GRU-based temporal encoder."""
    
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
    """TA-GNN v4 - exact architecture from train_tagnn_dynamic_v4.py"""
    
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
# DATASET
# ============================================================================

class EVChargingDatasetV4(Dataset):
    """Dataset with time-varying age."""
    
    def __init__(self, Y, M, time_features, time_index, station_open_times,
                 lookback=168, horizon=5, mode='test', 
                 train_ratio=0.7, val_ratio=0.15):
        
        self.Y = torch.FloatTensor(Y)
        self.M = torch.FloatTensor(M)
        self.time_features = torch.FloatTensor(time_features)
        self.lookback = lookback
        self.horizon = horizon
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
# EVALUATION
# ============================================================================

def evaluate_with_cold_start_window(model, loader, device, static_adj, 
                                     station_features, provider_ids, 
                                     cold_start_days):
    """Evaluate model with specific cold-start window."""
    model.eval()
    window_hours = cold_start_days * 24
    
    all_data = {
        'expected': [], 'prob': [], 'target': [], 
        'mask': [], 'ages_hours': [], 'gate': []
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
    cold_start = (ages_expanded < window_hours) & valid
    cold_start_positive = cold_start & (target > 0)
    
    metrics = {'delta_days': cold_start_days}
    
    # Overall MAE
    if valid.sum() > 0:
        metrics['overall_mae'] = (expected[valid] - target[valid]).abs().mean().item()
    
    # Positive-only MAE (Pos-MAE)
    if positive.sum() > 0:
        metrics['pos_mae'] = (expected[positive] - target[positive]).abs().mean().item()
    
    # Cold-start positive MAE (CS-Pos MAE) - KEY METRIC
    if cold_start_positive.sum() > 0:
        metrics['cs_pos_mae'] = (expected[cold_start_positive] - target[cold_start_positive]).abs().mean().item()
        metrics['cs_pos_samples'] = int(cold_start_positive.sum().item())
    else:
        metrics['cs_pos_mae'] = float('nan')
        metrics['cs_pos_samples'] = 0
    
    # AUPRC
    if valid.sum() > 0:
        try:
            binary_target = (target[valid] > 0).float().numpy()
            binary_pred = prob[valid].numpy()
            metrics['auprc'] = average_precision_score(binary_target, binary_pred)
        except:
            metrics['auprc'] = float('nan')
    
    # Gate statistics within cold-start window
    gate_flat = all_data['gate'].reshape(-1)
    ages_flat = all_data['ages_hours'].reshape(-1)
    cold_gate_mask = ages_flat < window_hours
    
    if cold_gate_mask.sum() > 0:
        metrics['avg_gate'] = gate_flat[cold_gate_mask].mean().item()
    else:
        metrics['avg_gate'] = float('nan')
    
    return metrics


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT_PATH)
    parser.add_argument('--data', type=str, default=DATA_PATH)
    parser.add_argument('--adj', type=str, default=ADJ_PATH)
    parser.add_argument('--output', type=str, default=OUTPUT_DIR)
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    print("="*70)
    print("TA-GNN COLD-START WINDOW SENSITIVITY ANALYSIS")
    print("="*70)
    print(f"Device: {DEVICE}")
    
    # Load data
    print(f"\nLoading data from {args.data}...")
    data = torch.load(args.data, map_location='cpu', weights_only=False)
    
    Y = data['Y'].numpy()
    M = data['M'].numpy()
    time_features = data['time_features'].numpy()
    time_index = data['time_index']
    station_open_times = data['station_open_times']
    station_features_raw = data['station_features'].numpy()
    provider_ids = data['provider_ids'].long().to(DEVICE)
    provider_list = data['provider_list']
    
    T, N = Y.shape
    num_providers = len(provider_list)
    print(f"  T={T:,}, N={N}, Providers={num_providers}")
    
    # Normalize station features (same as training)
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
    print(f"Loading adjacency from {args.adj}...")
    if os.path.exists(args.adj):
        adj_data = torch.load(args.adj, weights_only=False)
        static_adj = adj_data['A_distance'].float().to(DEVICE)
    else:
        print("  Warning: Adjacency not found, using identity")
        static_adj = torch.eye(N).to(DEVICE)
    
    # Create test dataset
    print("\nCreating test dataset...")
    test_ds = EVChargingDatasetV4(
        Y, M, time_features, time_index, station_open_times, 
        mode='test'
    )
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    print(f"  Test samples: {len(test_ds)}")
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
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
    
    checkpoint = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint)
    model.eval()
    
    # Print learned gate parameters
    a = (F.softplus(model.age_gate.a_raw) + 0.1).item()
    b = model.age_gate.b.item()
    print(f"  Learned gate: a={a:.3f}, b={b:.3f} years ({b*365.25:.0f} days)")
    
    # Evaluate with different cold-start windows
    print("\n" + "="*70)
    print("EVALUATING WITH DIFFERENT COLD-START WINDOWS")
    print("="*70)
    
    delta_values = [7, 14, 21, 30]  # days
    results = []
    
    for delta in delta_values:
        print(f"\nΔ = {delta} days...")
        metrics = evaluate_with_cold_start_window(
            model, test_loader, DEVICE, static_adj,
            station_features, provider_ids, delta
        )
        results.append(metrics)
        print(f"  CS-Pos MAE: {metrics['cs_pos_mae']:.4f} (n={metrics['cs_pos_samples']:,})")
        print(f"  Pos-MAE: {metrics['pos_mae']:.4f}")
        print(f"  Avg Gate: {metrics['avg_gate']:.3f}")
    
    # Save results
    df = pd.DataFrame(results)
    csv_path = os.path.join(args.output, 'coldstart_sensitivity.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Generate LaTeX table
    latex = r"""\begin{table}[h]
\centering
\caption{Sensitivity to cold-start window duration $\Delta$. CS-Pos MAE is computed on positive-demand hours within the first $\Delta$ days of each station's operation.}
\label{tab:sensitivity_delta}
\small
\begin{tabular}{@{}ccccc@{}}
\toprule
$\Delta$ (days) & \textbf{CS-Pos MAE}$\downarrow$ & \textbf{Samples} & \textbf{Avg Gate $\bar{g}$} & \textbf{Pos-MAE}$\downarrow$ \\
\midrule
"""
    for _, row in df.iterrows():
        latex += f"{int(row['delta_days'])} & {row['cs_pos_mae']:.3f} & {int(row['cs_pos_samples']):,} & {row['avg_gate']:.2f} & {row['pos_mae']:.3f} \\\\\n"
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    latex_path = os.path.join(args.output, 'table_coldstart.tex')
    with open(latex_path, 'w') as f:
        f.write(latex)
    print(f"LaTeX table saved to: {latex_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(df.to_string(index=False))
    
    print("\n" + "="*70)
    print("LATEX TABLE (copy to Appendix D)")
    print("="*70)
    print(latex)


if __name__ == '__main__':
    main()
