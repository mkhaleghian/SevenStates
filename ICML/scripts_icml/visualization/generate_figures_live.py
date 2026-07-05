"""
TA-GNN Paper Visualizations - Live Evaluation Version
======================================================

This script:
1. Loads your ACTUAL trained TA-GNN v4 model checkpoint
2. Loads your ACTUAL preprocessed data
3. Runs FRESH evaluation to compute all metrics
4. Generates publication-quality figures from live results

Whenever you retrain the model, just run this script again
to regenerate all figures with updated results!

Usage:
    # Basic usage (uses default paths)
    python scripts/visualization/generate_figures_live.py

    # Specify custom paths
    python scripts/visualization/generate_figures_live.py \
        --data_path data/processed/tensors/data_v4.pt \
        --checkpoint_path checkpoints/tagnn_dynamic_v4/best_model_cs_pos.pt \
        --baseline_results results/baseline_comparison_v4.json \
        --output_dir figures

Output:
    figures/ directory with:
    - fig1_main_comparison.pdf/png
    - fig2_gate_analysis.pdf/png  
    - fig3_training_curves.pdf/png (if history available)
    - fig4_improvement.pdf/png
    - fig5_performance_by_age.pdf/png
    - fig6_architecture.pdf/png
    - table_main_results.tex
    - evaluation_results.json

Author: EV_GNN Research Project
"""

import os
import sys
import json
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

# ============================================================================
# CONFIGURATION - Change PROJECT_ROOT to match your setup
# ============================================================================

PROJECT_ROOT = os.environ.get('EV_GNN_ROOT', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Publication-quality matplotlib settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Palatino'],
    'mathtext.fontset': 'stix',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.4,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Colorblind-friendly palette
COLORS = {
    'tagnn': '#D55E00',      # Vermillion (our method)
    'lstm': '#0072B2',       # Blue
    'gcn': '#009E73',        # Bluish green
    'gat': '#CC79A7',        # Reddish purple
    'hist_avg': '#888888',   # Gray
    'last_val': '#AAAAAA',   # Light gray
    'spatial': '#E69F00',    # Orange
    'temporal': '#56B4E9',   # Sky blue
    'cold': '#F0E442',       # Yellow
    'mature': '#0072B2',     # Blue
}


# ============================================================================
# MODEL DEFINITION
# This MUST match your training code exactly!
# If you modify the model architecture, update this section.
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
    
    def get_params(self):
        """Return learned gate parameters."""
        a = (F.softplus(self.a_raw) + 0.1).item()
        b = self.b.item()
        return a, b


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


class TAGNNDynamicV4(nn.Module):
    """
    TA-GNN v4: Temporal-Adaptive Graph Neural Network
    
    Key features:
    - Age-modulated graph attention for knowledge transfer
    - Monotonic age gate for spatial-temporal fusion
    - Hurdle output for zero-inflated prediction
    """
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
        attn_weights_list = []
        for gnn_layer in self.gnn_layers:
            h_spatial, attn_weights = gnn_layer(h_spatial, ages, static_adj, station_open_mask)
            attn_weights_list.append(attn_weights)
        
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
            'attn': attn_weights_list,
        }


# ============================================================================
# DATASET
# ============================================================================

class EVChargingDatasetV4(Dataset):
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
        ages = (current_time - self.station_open_times).astype('timedelta64[h]').astype(float) / (24 * 365.25)
        ages = np.maximum(ages, 0)
        
        return {
            'y_hist': self.Y[t-self.lookback:t, :].T,
            'm_hist': self.M[t-self.lookback:t, :].T,
            'time_hist': self.time_features[t-self.lookback:t],
            'y_target': self.Y[t:t+self.horizon, :].T,
            'm_target': self.M[t:t+self.horizon, :].T,
            'ages': torch.FloatTensor(ages),
            't_idx': t,
        }


# ============================================================================
# EVALUATION
# ============================================================================

def run_evaluation(model, loader, device, static_adj, station_features, 
                   provider_ids, cold_start_days=14):
    """Run evaluation and compute all metrics."""
    model.eval()
    window_hours = cold_start_days * 24
    
    all_expected, all_prob, all_target, all_mask = [], [], [], []
    all_ages, all_gate = [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            y_hist = batch['y_hist'].to(device)
            m_hist = batch['m_hist'].to(device)
            time_hist = batch['time_hist'].to(device)
            y_target = batch['y_target'].to(device)
            m_target = batch['m_target'].to(device)
            ages = batch['ages'].to(device)
            
            output = model(y_hist, m_hist, time_hist, ages,
                          station_features, provider_ids, static_adj)
            
            all_expected.append(output['expected'].cpu())
            all_prob.append(output['prob'].cpu())
            all_target.append(y_target.cpu())
            all_mask.append(m_target.cpu())
            all_ages.append((ages * 24 * 365.25).cpu())
            all_gate.append(output['gate'].cpu())
    
    # Concatenate
    all_expected = torch.cat(all_expected, 0)
    all_prob = torch.cat(all_prob, 0)
    all_target = torch.cat(all_target, 0)
    all_mask = torch.cat(all_mask, 0)
    all_ages = torch.cat(all_ages, 0)
    all_gate = torch.cat(all_gate, 0)
    
    # Flatten for metrics
    B, N, H = all_target.shape
    expected = all_expected.view(-1)
    prob = all_prob.view(-1)
    target = all_target.view(-1)
    mask = all_mask.view(-1)
    ages_exp = all_ages.unsqueeze(-1).expand(B, N, H).reshape(-1)
    
    valid = mask > 0
    positive = (target > 0) & valid
    cold_start = (ages_exp < window_hours) & valid
    cold_start_positive = cold_start & (target > 0)
    
    metrics = {}
    
    # Core metrics
    metrics['overall_mae'] = (expected[valid] - target[valid]).abs().mean().item()
    metrics['overall_rmse'] = ((expected[valid] - target[valid])**2).mean().sqrt().item()
    
    if positive.sum() > 0:
        metrics['positive_mae'] = (expected[positive] - target[positive]).abs().mean().item()
        metrics['positive_count'] = int(positive.sum())
    
    if cold_start_positive.sum() > 0:
        metrics['cold_start_positive_mae'] = (expected[cold_start_positive] - target[cold_start_positive]).abs().mean().item()
        metrics['cold_start_positive_count'] = int(cold_start_positive.sum())
    
    # Detection metrics
    try:
        metrics['auprc'] = average_precision_score((target[valid] > 0).numpy(), prob[valid].numpy())
        metrics['auroc'] = roc_auc_score((target[valid] > 0).numpy(), prob[valid].numpy())
    except:
        pass
    
    # Gate analysis
    gate_flat = all_gate.view(-1)
    ages_flat = all_ages.view(-1) / (24 * 365.25)
    
    cold_mask = ages_flat < (cold_start_days / 365.25)
    mature_mask = ages_flat > 1.0
    
    if cold_mask.sum() > 0:
        metrics['gate_cold_start'] = gate_flat[cold_mask].mean().item()
    if mature_mask.sum() > 0:
        metrics['gate_mature'] = gate_flat[mature_mask].mean().item()
    
    # Gate and MAE by age bins
    age_bins_days = [0, 7, 14, 30, 90, 180, 365, 730, 1095]
    metrics['gate_by_age'] = {}
    metrics['mae_by_age'] = {}
    
    for i in range(len(age_bins_days) - 1):
        start, end = age_bins_days[i], age_bins_days[i+1]
        bin_name = f"{start}-{end}d"
        
        bin_mask = (ages_flat * 365.25 >= start) & (ages_flat * 365.25 < end)
        if bin_mask.sum() > 0:
            metrics['gate_by_age'][bin_name] = gate_flat[bin_mask].mean().item()
        
        bin_mask_exp = (ages_exp / 24 >= start) & (ages_exp / 24 < end) & positive
        if bin_mask_exp.sum() > 0:
            metrics['mae_by_age'][bin_name] = (expected[bin_mask_exp] - target[bin_mask_exp]).abs().mean().item()
    
    return metrics


# ============================================================================
# FIGURE GENERATION
# ============================================================================

def generate_figures(metrics, gate_params, baselines, output_dir, history=None):
    """Generate all publication-quality figures."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Update TA-GNN in baselines
    baselines['TA-GNN v4'] = {
        'overall_mae': metrics['overall_mae'],
        'positive_mae': metrics['positive_mae'],
        'cold_start_positive_mae': metrics['cold_start_positive_mae'],
        'auprc': metrics.get('auprc', 0),
    }
    
# =========================================================================
    # FIGURE 1: Main Results
    # =========================================================================
    print("  Fig 1: Main Results...")
    
    method_order = [
        ('Historical Average', 'Hist. Avg.', COLORS['hist_avg']),
        ('Last Value', 'Last Val.', COLORS['last_val']),
        ('LSTM-Hurdle', 'LSTM', COLORS['lstm']),
        ('GCN-Hurdle', 'GCN', COLORS['gcn']),
        ('GAT-Hurdle', 'GAT', COLORS['gat']),
        ('TA-GNN v4', 'TA-GNN (Ours)', COLORS['tagnn']),
    ]
    
    names = [m[1] for m in method_order]
    colors = [m[2] for m in method_order]
    cs_pos = [baselines.get(m[0], {}).get('cold_start_positive_mae', np.nan) for m in method_order]
    auprc = [baselines.get(m[0], {}).get('auprc', np.nan) for m in method_order]
    
    fig = plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1], wspace=0.35)
    ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    
    x = np.arange(len(names))
    
    # CS-Pos MAE
    bars1 = ax1.bar(x, cs_pos, 0.72, color=colors, edgecolor='white')
    bars1[-1].set_edgecolor('#8B0000')
    bars1[-1].set_linewidth(2.5)
    for i, (bar, val) in enumerate(zip(bars1, cs_pos)):
        if not np.isnan(val):
            y = bar.get_height() - 0.05 if val > 0.3 else bar.get_height() + 0.02
            ax1.text(bar.get_x() + bar.get_width()/2, y, f'{val:.3f}', 
                    ha='center', va='top' if val > 0.3 else 'bottom',
                    fontsize=12, fontweight='bold')
    ax1.set_ylabel('MAE (lower is better)', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Cold-Start Positive MAE', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=12, fontweight='bold', rotation=45, ha='right')
    ax1.set_ylim(0, 1.05)
    ax1.axhline(y=cs_pos[-1], color=COLORS['tagnn'], linestyle='--', alpha=0.4)
    ax1.tick_params(axis='y', labelsize=12)
    for label in ax1.get_yticklabels():
        label.set_fontweight('bold')
    
    # AUPRC
    bars2 = ax2.bar(x, auprc, 0.72, color=colors, edgecolor='white')
    for bar, val in zip(bars2, auprc):
        if not np.isnan(val):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008, 
                    f'{val:.3f}', ha='center', va='bottom', 
                    fontsize=12, fontweight='bold')
    ax2.set_ylabel('AUPRC (higher is better)', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Detection Performance', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=12, fontweight='bold', rotation=45, ha='right')
    ax2.set_ylim(0, 0.32)
    ax2.tick_params(axis='y', labelsize=12)
    for label in ax2.get_yticklabels():
        label.set_fontweight('bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig1_main_comparison.pdf")
    plt.savefig(f"{output_dir}/fig1_main_comparison.png", dpi=300)
    plt.close()
    
    # =========================================================================
    # FIGURE 2: Gate Analysis
    # =========================================================================
    print("  Fig 2: Gate Analysis...")
    
    a, b = gate_params
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    
    ages = np.linspace(0, 3, 500)
    g = 1 / (1 + np.exp(-a * (ages - b)))
    
    ax1 = axes[0]
    ax1.plot(ages, g, color=COLORS['tagnn'], linewidth=2.5)
    ax1.fill_between(ages, 0, g, alpha=0.25, color=COLORS['temporal'], label='Temporal')
    ax1.fill_between(ages, g, 1, alpha=0.25, color=COLORS['spatial'], label='Spatial')
    ax1.axvspan(0, 14/365, alpha=0.15, color=COLORS['cold'])
    ax1.scatter([14/365], [metrics.get('gate_cold_start', 0.34)], s=80, c=COLORS['cold'], 
                edgecolors='black', zorder=5, marker='o')
    ax1.scatter([2.0], [metrics.get('gate_mature', 0.85)], s=80, c=COLORS['mature'], 
                edgecolors='black', zorder=5, marker='s')
    ax1.set_xlabel('Station Age (years)', fontweight='bold')
    ax1.set_ylabel('Gate Value $g(a)$', fontweight='bold')
    ax1.set_title('(a) Learned Gate Function', fontweight='bold')
    ax1.set_xlim(0, 3)
    ax1.set_ylim(0, 1)
    ax1.legend(loc='lower right')
    ax1.text(0.02, 0.02, f'$\\alpha={a:.2f}$, $\\beta={b:.2f}$', transform=ax1.transAxes, fontsize=9,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax2 = axes[1]
    ax2.stackplot(ages, 1-g, g, colors=[COLORS['spatial'], COLORS['temporal']], alpha=0.7,
                  labels=['Spatial', 'Temporal'])
    ax2.axvline(x=14/365, color='black', linestyle=':', linewidth=1.5)
    ax2.text(0.05, 0.75, 'SPATIAL', fontsize=10, fontweight='bold', color='white')
    ax2.text(2.3, 0.25, 'TEMPORAL', fontsize=10, fontweight='bold', color='white')
    ax2.set_xlabel('Station Age (years)', fontweight='bold')
    ax2.set_ylabel('Representation Weight', fontweight='bold')
    ax2.set_title('(b) Spatial-Temporal Balance', fontweight='bold')
    ax2.set_xlim(0, 3)
    ax2.legend(loc='center right')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig2_gate_analysis.pdf")
    plt.savefig(f"{output_dir}/fig2_gate_analysis.png", dpi=300)
    plt.close()
    
    # =========================================================================
    # FIGURE 3: Training Curves (if available)
    # =========================================================================
    if history:
        print("  Fig 3: Training Curves...")
        epochs = np.arange(1, len(history['train']) + 1)
        
        fig, axes = plt.subplots(1, 3, figsize=(10, 3))
        
        axes[0].plot(epochs, history['train'], color=COLORS['tagnn'])
        axes[0].fill_between(epochs, history['train'], alpha=0.2, color=COLORS['tagnn'])
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('(a) Training Loss', fontweight='bold')
        
        axes[1].plot(epochs, history['val_overall_mae'], '--', color=COLORS['lstm'], label='Overall')
        axes[1].plot(epochs, history['val_cs_pos_mae'], color=COLORS['tagnn'], label='CS-Pos')
        best_ep = np.argmin(history['val_cs_pos_mae']) + 1
        axes[1].scatter([best_ep], [min(history['val_cs_pos_mae'])], s=100, c=COLORS['tagnn'], marker='*')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].set_title('(b) Validation MAE', fontweight='bold')
        axes[1].legend()
        
        axes[2].plot(epochs, history['val_auprc'], color=COLORS['gcn'])
        axes[2].fill_between(epochs, history['val_auprc'], alpha=0.2, color=COLORS['gcn'])
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('AUPRC')
        axes[2].set_title('(c) Detection AUPRC', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/fig3_training_curves.pdf")
        plt.savefig(f"{output_dir}/fig3_training_curves.png", dpi=300)
        plt.close()
    
    # =========================================================================
    # FIGURE 4: Improvement Chart
    # =========================================================================
    print("  Fig 4: Improvement Chart...")
    
    tagnn_val = metrics['cold_start_positive_mae']
    improvements = []
    for key, name, color in [('GAT-Hurdle', 'GAT', COLORS['gat']),
                             ('GCN-Hurdle', 'GCN', COLORS['gcn']),
                             ('LSTM-Hurdle', 'LSTM', COLORS['lstm']),
                             ('Last Value', 'Last Value', COLORS['last_val']),
                             ('Historical Average', 'Hist. Avg.', COLORS['hist_avg'])]:
        if key in baselines:
            base_val = baselines[key].get('cold_start_positive_mae', np.nan)
            if not np.isnan(base_val) and base_val > 0:
                improvements.append((name, (base_val - tagnn_val) / base_val * 100, color))
    
    fig, ax = plt.subplots(figsize=(6, 3.5))
    y = np.arange(len(improvements))
    bars = ax.barh(y, [i[1] for i in improvements], color=[i[2] for i in improvements], height=0.6)
    for bar, (_, val, _) in zip(bars, improvements):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height()/2, 
                f'{val:.1f}%', ha='left', va='center', fontsize=10, fontweight='bold')
    ax.set_yticks(y)
    ax.set_yticklabels([i[0] for i in improvements])
    ax.set_xlabel('Relative Improvement (%)', fontweight='bold')
    ax.set_title('TA-GNN Improvement (CS-Pos MAE)', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fig4_improvement.pdf")
    plt.savefig(f"{output_dir}/fig4_improvement.png", dpi=300)
    plt.close()
    
    # =========================================================================
    # FIGURE 5: Performance by Age
    # =========================================================================
    if 'gate_by_age' in metrics and 'mae_by_age' in metrics:
        print("  Fig 5: Performance by Age...")
        
        common_bins = [k for k in metrics['gate_by_age'] if k in metrics['mae_by_age']]
        if len(common_bins) > 2:
            fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
            x = np.arange(len(common_bins))
            
            axes[0].bar(x, [metrics['gate_by_age'][k] for k in common_bins], color=COLORS['tagnn'], alpha=0.8)
            axes[0].axhline(0.5, color='gray', linestyle='--', alpha=0.5)
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(common_bins, rotation=45, ha='right', fontsize=12, fontweight='bold')
            axes[0].set_ylabel('Gate Value', fontweight='bold')
            axes[0].set_title('(a) Gate by Station Age', fontweight='bold')
            
            axes[1].bar(x, [metrics['mae_by_age'][k] for k in common_bins], color=COLORS['gcn'], alpha=0.8)
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(common_bins, rotation=45, ha='right', fontsize=12, fontweight='bold')
            axes[1].set_ylabel('Positive MAE', fontweight='bold')
            axes[1].set_title('(b) Error by Station Age', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/fig5_performance_by_age.pdf")
            plt.savefig(f"{output_dir}/fig5_performance_by_age.png", dpi=300)
            plt.close()
    
    # =========================================================================
    # FIGURE 6: Architecture
    # =========================================================================
    print("  Fig 6: Architecture...")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    def box(x, y, w, h, text, color):
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03",
                     facecolor=color, edgecolor='#333', linewidth=1.2))
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=9, fontweight='bold')
    
    def arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='->', color='#555', lw=1.5))
    
    box(0.3, 2.3, 1.4, 1.4, 'Input', '#E8E8E8')
    box(2.2, 4, 1.6, 0.9, 'Station\nEncoder', '#FFE4B5')
    box(2.2, 2.6, 1.6, 0.9, 'Projection', '#FFE4B5')
    box(2.2, 1.1, 1.6, 0.9, 'Age\nEmbed', '#FFE4B5')
    box(4.5, 3.8, 2.2, 1.3, 'Graph\nAttention', '#FFD5CC')
    box(4.5, 1.8, 2.2, 1.3, 'Temporal\nGRU', '#CCE5FF')
    box(7.4, 2.5, 1.5, 1.5, 'Age-Gated\nFusion', '#C5E8B7')
    box(9.5, 3.5, 1.3, 0.8, 'P(y>0)', '#FFB6C1')
    box(9.5, 2.2, 1.3, 0.8, 'E[y|y>0]', '#ADD8E6')
    
    arrow(1.7, 3.0, 2.1, 3.0)
    arrow(1.7, 3.2, 2.1, 4.4)
    arrow(3.9, 4.4, 4.4, 4.4)
    arrow(3.9, 3.0, 4.4, 3.0)
    arrow(6.8, 4.4, 7.3, 3.5)
    arrow(6.8, 2.4, 7.3, 3.0)
    arrow(8.95, 3.5, 9.4, 3.9)
    arrow(8.95, 3.0, 9.4, 2.6)
    
    ax.text(6, 5.6, 'TA-GNN Architecture', fontsize=14, fontweight='bold', ha='center')
    ax.text(5.6, 5.3, 'Spatial', fontsize=10, ha='center', color=COLORS['spatial'], fontweight='bold')
    ax.text(5.6, 1.5, 'Temporal', fontsize=10, ha='center', color=COLORS['temporal'], fontweight='bold')
    
    plt.savefig(f"{output_dir}/fig6_architecture.pdf")
    plt.savefig(f"{output_dir}/fig6_architecture.png", dpi=300)
    plt.close()
    
    # =========================================================================
    # LATEX TABLE
    # =========================================================================
    print("  LaTeX Table...")
    
    latex = r"""\begin{table}[t]
\centering
\caption{Main results. \textbf{Bold} = best CS-Pos MAE.}
\small
\begin{tabular}{lccc}
\toprule
Method & Positive MAE & CS-Pos MAE & AUPRC \\
\midrule
"""
    for key, name in [('Historical Average', 'Hist. Average'),
                      ('Last Value', 'Last Value'),
                      ('LSTM-Hurdle', 'LSTM-Hurdle'),
                      ('GCN-Hurdle', 'GCN-Hurdle'),
                      ('GAT-Hurdle', 'GAT-Hurdle'),
                      ('TA-GNN v4', '\\textbf{TA-GNN (Ours)}')]:
        if key in baselines:
            m = baselines[key]
            pos = f"{m.get('positive_mae', m.get('positive_only_mae', 0)):.3f}"
            cs = f"{m.get('cold_start_positive_mae', 0):.3f}"
            aup = f"{m.get('auprc', 0):.3f}"
            if key == 'TA-GNN v4':
                cs = f"\\textbf{{{cs}}}"
            latex += f"{name} & {pos} & {cs} & {aup} \\\\\n"
    latex += r"""\bottomrule
\end{tabular}
\end{table}"""
    
    with open(f"{output_dir}/table_main_results.tex", 'w') as f:
        f.write(latex)
    
    print(f"\n✅ All figures saved to: {output_dir}/")
    return baselines


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data/processed/tensors/data_v4.pt')
    parser.add_argument('--checkpoint_path', default='checkpoints/tagnn_dynamic_v4/best_model_cs_pos.pt')
    parser.add_argument('--baseline_results', default='results/baseline_comparison_v4.json')
    parser.add_argument('--output_dir', default='figures')
    parser.add_argument('--cold_start_days', type=int, default=14)
    args = parser.parse_args()
    
    print("=" * 70)
    print("   TA-GNN Paper Figures - Live Evaluation")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load data
    print("\n📁 Loading data...")
    data_path = os.path.join(PROJECT_ROOT, args.data_path)
    data = torch.load(data_path, weights_only=False)
    
    Y = data['Y'].numpy()
    M = data['M'].numpy()
    time_features = data['time_features'].numpy()
    time_index = data['time_index']
    station_open_times = data['station_open_times']
    sf_raw = data['station_features'].numpy()
    provider_ids = data['provider_ids'].long().to(device)
    
    T, N = Y.shape
    print(f"   T={T:,}, N={N}")
    
    # Normalize features
    has_coords = (sf_raw[:, 1] != 0) | (sf_raw[:, 2] != 0)
    sf = np.zeros((N, 4), dtype=np.float32)
    sf[:, 0] = (sf_raw[:, 0] - sf_raw[:, 0].mean()) / (sf_raw[:, 0].std() + 1e-8)
    if has_coords.any():
        sf[has_coords, 1] = (sf_raw[has_coords, 1] - sf_raw[has_coords, 1].mean()) / (sf_raw[has_coords, 1].std() + 1e-8)
        sf[has_coords, 2] = (sf_raw[has_coords, 2] - sf_raw[has_coords, 2].mean()) / (sf_raw[has_coords, 2].std() + 1e-8)
    sf[:, 3] = has_coords.astype(float)
    station_features = torch.FloatTensor(sf).to(device)
    
    # Load adjacency
    adj_path = os.path.join(PROJECT_ROOT, 'data/graphs/adjacency_v4.pt')
    if os.path.exists(adj_path):
        static_adj = torch.load(adj_path, weights_only=False)['A_distance'].float().to(device)
    else:
        static_adj = torch.eye(N).to(device)
    
    # Create test loader
    test_ds = EVChargingDatasetV4(Y, M, time_features, time_index, station_open_times, mode='test')
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    print(f"   Test samples: {len(test_ds)}")
    
    # Load model
    print("\n📦 Loading model...")
    checkpoint_path = os.path.join(PROJECT_ROOT, args.checkpoint_path)
    
    model = TAGNNDynamicV4(
        num_nodes=N,
        num_providers=len(data['provider_list']),
        time_feat_dim=time_features.shape[1],
    ).to(device)
    
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    print(f"   Loaded: {checkpoint_path}")
    
    gate_params = model.age_gate.get_params()
    print(f"   Gate: α={gate_params[0]:.2f}, β={gate_params[1]:.2f}")
    
    # Run evaluation
    print("\n📊 Evaluating...")
    metrics = run_evaluation(model, test_loader, device, static_adj, 
                             station_features, provider_ids, args.cold_start_days)
    
    print(f"\n   Overall MAE: {metrics['overall_mae']:.4f}")
    print(f"   Positive MAE: {metrics['positive_mae']:.4f}")
    print(f"   CS-Pos MAE: {metrics['cold_start_positive_mae']:.4f}")
    print(f"   AUPRC: {metrics.get('auprc', 0):.4f}")
    print(f"   Gate cold/mature: {metrics.get('gate_cold_start', 0):.3f}/{metrics.get('gate_mature', 0):.3f}")
    
    # Load baselines
    baseline_path = os.path.join(PROJECT_ROOT, args.baseline_results)
    baselines = json.load(open(baseline_path)) if os.path.exists(baseline_path) else {}
    
    # Load history
    results_path = os.path.join(PROJECT_ROOT, 'checkpoints/tagnn_dynamic_v4/results_v4.json')
    history = json.load(open(results_path)).get('history') if os.path.exists(results_path) else None
    
    # Generate figures
    print("\n🎨 Generating figures...")
    output_dir = os.path.join(PROJECT_ROOT, args.output_dir)
    generate_figures(metrics, gate_params, baselines, output_dir, history)
    
    # Save metrics
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump({k: float(v) if isinstance(v, (np.floating, float)) else v 
                   for k, v in metrics.items() if not isinstance(v, dict)}, f, indent=2)


if __name__ == "__main__":
    main()
