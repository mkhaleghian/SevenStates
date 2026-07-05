"""
TA-GNN Dynamic Training Script

This trains the DYNAMIC Graph Attention model where:
- Connections are learned during training (not fixed!)
- Cold-start stations can find their own "teachers"
- Attention weights adapt based on node states and ages

This is the KEY CONTRIBUTION for the ICML paper!

Usage:
    python scripts/training/train_tagnn_dynamic.py --epochs 100

Author: EV_GNN Research Project
"""

import os
import sys
import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math

PROJECT_ROOT = os.environ.get('EV_GNN_ROOT', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)


# =============================================================================
# Dataset
# =============================================================================

class EVChargingBatchDataset(Dataset):
    """Dataset returning all stations for each time step."""
    
    def __init__(self, Y, M, time_features, station_ages,
                 lookback=168, horizon=5, mode='train',
                 train_ratio=0.7, val_ratio=0.15):
        
        self.Y = torch.FloatTensor(Y)
        self.M = torch.FloatTensor(M)
        self.time_features = torch.FloatTensor(time_features)
        self.station_ages = torch.FloatTensor(station_ages)
        self.lookback = lookback
        self.horizon = horizon
        
        T, N = Y.shape
        train_end = int(T * train_ratio)
        val_end = int(T * (train_ratio + val_ratio))
        
        if mode == 'train':
            self.time_start, self.time_end = lookback, train_end - horizon
        elif mode == 'val':
            self.time_start, self.time_end = train_end, val_end - horizon
        else:  # test
            self.time_start, self.time_end = val_end, T - horizon
        
        self.valid_times = list(range(self.time_start, self.time_end))
    
    def __len__(self):
        return len(self.valid_times)
    
    def __getitem__(self, idx):
        t = self.valid_times[idx]
        return {
            'y_hist': self.Y[t - self.lookback:t, :].T,      # (N, lookback)
            'm_hist': self.M[t - self.lookback:t, :].T,      # (N, lookback)
            'time_hist': self.time_features[t - self.lookback:t],  # (lookback, feat)
            'y_target': self.Y[t:t + self.horizon, :].T,     # (N, horizon)
            'm_target': self.M[t:t + self.horizon, :].T,     # (N, horizon)
            'time_idx': t,
        }


# =============================================================================
# Model Components
# =============================================================================

class HistoryEmbedding(nn.Module):
    """Sinusoidal encoding for station age."""
    
    def __init__(self, embed_dim, max_age_years=5.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_age_years = max_age_years
        freqs = torch.exp(torch.linspace(0, np.log(max_age_years + 1), embed_dim // 2))
        self.register_buffer('freq_bands', freqs)
    
    def forward(self, age_years):
        age_norm = (age_years / self.max_age_years).unsqueeze(-1)
        angles = age_norm * self.freq_bands.unsqueeze(0) * math.pi
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class DynamicGraphAttention(nn.Module):
    """
    Dynamic Graph Attention - THE KEY INNOVATION!
    
    Unlike static GNN:
    - Attention weights are LEARNED based on node states
    - Age modulation: young stations attend MORE to old stations
    - Static adjacency used as prior (optional)
    """
    
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1, 
                 use_static_adj=True, age_modulation=True, age_embed_dim=32):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_static_adj = use_static_adj
        self.age_modulation = age_modulation
        
        # Query, Key, Value
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        
        # Age modulation
        if age_modulation:
            self.age_query = nn.Linear(age_embed_dim, num_heads)
            self.age_key = nn.Linear(age_embed_dim, num_heads)
            self.age_gate = nn.Parameter(torch.tensor(0.5))
        
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, static_adj=None, age_embed=None):
        """
        Args:
            x: (B, N, hidden) or (N, hidden)
            static_adj: (N, N) static adjacency as prior
            age_embed: (N, embed_dim) age embeddings
        
        Returns:
            out: Updated node features
            attention: Attention weights (for visualization)
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        
        B, N, _ = x.shape
        H, D = self.num_heads, self.head_dim
        
        # Q, K, V
        Q = self.W_q(x).view(B, N, H, D).transpose(1, 2)  # (B, H, N, D)
        K = self.W_k(x).view(B, N, H, D).transpose(1, 2)
        V = self.W_v(x).view(B, N, H, D).transpose(1, 2)
        
        # Content-based attention
        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D)  # (B, H, N, N)
        
        # Age modulation: young nodes attend more to old nodes
        if self.age_modulation and age_embed is not None:
            age_q = self.age_query(age_embed)  # (N, H)
            age_k = self.age_key(age_embed)    # (N, H)
            age_attn = torch.matmul(age_q, age_k.T).unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)
            attn = attn + self.age_gate * age_attn
        
        # Use static adjacency as prior (soft mask)
        if self.use_static_adj and static_adj is not None:
            # Reduce attention to non-neighbors (but don't fully block)
            mask = (static_adj == 0).unsqueeze(0).unsqueeze(0)
            attn = attn.masked_fill(mask, -1e4)
        
        # Softmax + dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Aggregate
        out = torch.matmul(attn, V)  # (B, H, N, D)
        out = out.transpose(1, 2).contiguous().view(B, N, self.hidden_dim)
        out = self.out_proj(out)
        
        # Residual + LayerNorm
        out = self.layer_norm(x + out)
        
        if squeeze:
            out = out.squeeze(0)
            attn = attn.squeeze(0)
        
        return out, attn


class TAGNN_Dynamic(nn.Module):
    """
    TA-GNN with Dynamic Graph Attention
    
    Architecture:
    1. Input embedding
    2. History (age) embedding  
    3. Dynamic Graph Attention (connections LEARNED!)
    4. Temporal GRU
    5. Output
    """
    
    def __init__(self, num_nodes, input_dim=1, time_feat_dim=7,
                 hidden_dim=64, num_gnn_layers=2, num_temporal_layers=2,
                 num_heads=4, dropout=0.2, horizon=5,
                 history_embed_dim=32, max_age_years=5.0,
                 use_static_adj=True):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.history_embed_dim = history_embed_dim
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim + time_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # History embedding
        self.history_embedding = HistoryEmbedding(history_embed_dim, max_age_years)
        self.history_proj = nn.Sequential(
            nn.Linear(hidden_dim + history_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Dynamic Graph Attention layers
        self.gnn_layers = nn.ModuleList([
            DynamicGraphAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_static_adj=use_static_adj,
                age_modulation=True,
                age_embed_dim=history_embed_dim
            )
            for _ in range(num_gnn_layers)
        ])
        
        # Temporal GRU
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_temporal_layers,
            batch_first=True,
            dropout=dropout if num_temporal_layers > 1 else 0,
        )
        self.temporal_norm = nn.LayerNorm(hidden_dim)
        
        # Output
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, horizon)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, time_features, adj, age_years, return_attention=False):
        """
        Args:
            x: (B, T, N) historical counts
            time_features: (T, feat_dim)
            adj: (N, N) static adjacency as prior
            age_years: (N,) station ages
            return_attention: return attention weights for visualization
        """
        B, T, N = x.shape
        
        if time_features.dim() == 2:
            time_features = time_features.unsqueeze(0).expand(B, -1, -1)
        
        # Input embedding
        x = x.unsqueeze(-1)
        time_exp = time_features.unsqueeze(2).expand(-1, -1, N, -1)
        h = torch.cat([x, time_exp], dim=-1)
        h = self.input_proj(h)  # (B, T, N, hidden)
        
        # History embedding
        age_embed = self.history_embedding(age_years)  # (N, embed_dim)
        age_exp = age_embed.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
        h = torch.cat([h, age_exp], dim=-1)
        h = self.history_proj(h)
        
        # Dynamic Graph Attention on last time step
        h_spatial = h[:, -1, :, :]  # (B, N, hidden)
        attention = None
        for gnn in self.gnn_layers:
            h_spatial, attention = gnn(h_spatial, adj, age_embed)
        
        # Temporal GRU
        h_temporal = h.permute(0, 2, 1, 3).reshape(B * N, T, self.hidden_dim)
        h_temporal, _ = self.gru(h_temporal)
        h_temporal = h_temporal[:, -1, :].view(B, N, self.hidden_dim)
        h_temporal = self.temporal_norm(h_temporal)
        
        # Output
        h_out = torch.cat([h_spatial, h_temporal], dim=-1)
        pred = self.output_proj(h_out)
        pred = F.softplus(pred)
        
        if return_attention:
            return pred, attention
        return pred


# =============================================================================
# Loss
# =============================================================================

class MaskedLoss(nn.Module):
    def __init__(self, alpha=0.3):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, pred, target, mask):
        mae = torch.abs(pred - target)
        mse = (pred - target) ** 2
        loss = self.alpha * mae + (1 - self.alpha) * mse
        return (loss * mask).sum() / (mask.sum() + 1e-8)


# =============================================================================
# Training Functions
# =============================================================================

def train_epoch(model, loader, optimizer, criterion, adj, ages, device, max_grad=1.0):
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        y_hist = batch['y_hist'].to(device).transpose(1, 2)  # (B, T, N)
        time_hist = batch['time_hist'].to(device)
        y_target = batch['y_target'].to(device)
        m_target = batch['m_target'].to(device)
        
        optimizer.zero_grad()
        pred = model(y_hist, time_hist, adj, ages)
        loss = criterion(pred, y_target, m_target)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(loader)


def validate(model, loader, criterion, adj, ages, device):
    model.eval()
    total_loss = 0
    all_preds, all_targets, all_masks = [], [], []
    
    with torch.no_grad():
        for batch in loader:
            y_hist = batch['y_hist'].to(device).transpose(1, 2)
            time_hist = batch['time_hist'].to(device)
            y_target = batch['y_target'].to(device)
            m_target = batch['m_target'].to(device)
            
            pred = model(y_hist, time_hist, adj, ages)
            loss = criterion(pred, y_target, m_target)
            total_loss += loss.item()
            
            all_preds.append(pred.cpu())
            all_targets.append(y_target.cpu())
            all_masks.append(m_target.cpu())
    
    preds = torch.cat(all_preds, 0)
    targets = torch.cat(all_targets, 0)
    masks = torch.cat(all_masks, 0)
    
    p, t = preds[masks > 0].numpy(), targets[masks > 0].numpy()
    mae = np.mean(np.abs(p - t))
    rmse = np.sqrt(np.mean((p - t) ** 2))
    
    return {'loss': total_loss / len(loader), 'mae': mae, 'rmse': rmse}


def evaluate_by_age(model, loader, adj, ages, device):
    """Evaluate by station age group."""
    model.eval()
    
    bins = {
        'cold_start': (0, 0.25),
        'new': (0.25, 1.0),
        'established': (1.0, 2.0),
        'mature': (2.0, 100),
    }
    
    ages_np = ages.cpu().numpy()
    results = {k: {'p': [], 't': [], 'm': []} for k in bins}
    
    with torch.no_grad():
        for batch in loader:
            y_hist = batch['y_hist'].to(device).transpose(1, 2)
            time_hist = batch['time_hist'].to(device)
            y_target = batch['y_target'].to(device)
            m_target = batch['m_target'].to(device)
            
            pred = model(y_hist, time_hist, adj, ages)
            
            for name, (lo, hi) in bins.items():
                idx = np.where((ages_np >= lo) & (ages_np < hi))[0]
                if len(idx) > 0:
                    results[name]['p'].append(pred[:, idx, :].cpu())
                    results[name]['t'].append(y_target[:, idx, :].cpu())
                    results[name]['m'].append(m_target[:, idx, :].cpu())
    
    metrics = {}
    for name, data in results.items():
        if len(data['p']) == 0:
            continue
        p = torch.cat(data['p'], 0)
        t = torch.cat(data['t'], 0)
        m = torch.cat(data['m'], 0)
        
        pf, tf = p[m > 0].numpy(), t[m > 0].numpy()
        if len(pf) > 0:
            metrics[f'{name}_mae'] = float(np.mean(np.abs(pf - tf)))
            metrics[f'{name}_rmse'] = float(np.sqrt(np.mean((pf - tf) ** 2)))
            metrics[f'{name}_count'] = len(pf)
    
    return metrics


def visualize_attention(model, loader, adj, ages, device, save_path=None):
    """Visualize dynamic attention patterns."""
    import matplotlib.pyplot as plt
    
    model.eval()
    batch = next(iter(loader))
    
    with torch.no_grad():
        y_hist = batch['y_hist'].to(device).transpose(1, 2)
        time_hist = batch['time_hist'].to(device)
        
        _, attention = model(y_hist, time_hist, adj, ages, return_attention=True)
    
    # Average over batch and heads
    attn = attention.mean(dim=(0, 1)).cpu().numpy()  # (N, N)
    ages_np = ages.cpu().numpy()
    
    # Sort by age
    order = np.argsort(ages_np)
    attn_sorted = attn[order][:, order]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Attention matrix
    im = axes[0].imshow(attn_sorted, cmap='Blues', aspect='auto')
    axes[0].set_xlabel('Target Station (by age →)')
    axes[0].set_ylabel('Source Station (by age →)')
    axes[0].set_title('Dynamic Attention Weights\n(Younger left, Older right)')
    plt.colorbar(im, ax=axes[0])
    
    # Attention from youngest stations
    n_young = min(5, len(ages_np))
    young_idx = order[:n_young]
    
    for idx in young_idx:
        axes[1].plot(ages_np[order], attn[idx][order], 
                     label=f'Age={ages_np[idx]:.2f}yr', alpha=0.7)
    
    axes[1].set_xlabel('Target Station Age (years)')
    axes[1].set_ylabel('Attention Weight')
    axes[1].set_title('Cold-Start Stations: Who Do They Attend To?')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved attention visualization to {save_path}")
    
    return fig


# =============================================================================
# Main
# =============================================================================

def main(args):
    print("=" * 60)
    print("   TA-GNN DYNAMIC Training")
    print("   (Connections are LEARNED, not fixed!)")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Paths
    data_path = f"{PROJECT_ROOT}/data/processed/tensors/data.pt"
    adj_path = f"{PROJECT_ROOT}/data/graphs/adjacency_v2.pt"
    meta_path = f"{PROJECT_ROOT}/data/processed/unified/station_metadata.csv"
    ckpt_dir = f"{PROJECT_ROOT}/checkpoints/tagnn_dynamic"
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    data = torch.load(data_path, weights_only=False)
    Y, M = data['Y'].numpy(), data['M'].numpy()
    time_features = data['time_features'].numpy()
    
    adj_data = torch.load(adj_path, weights_only=False)
    
    # Use combined adjacency as PRIOR (dynamic attention will learn the rest)
    A = adj_data['A_combined'].float().to(device)
    print(f"Static adjacency (prior): {(A > 0).sum().item():,} edges")
    
    meta = pd.read_csv(meta_path)
    ages = torch.FloatTensor(meta['operational_years'].values).to(device)
    
    T, N = Y.shape
    print(f"Data: T={T:,}, N={N}")
    print(f"Ages: {ages.min():.2f} - {ages.max():.2f} years")
    
    # Datasets
    train_data = EVChargingBatchDataset(Y, M, time_features, ages.cpu().numpy(),
                                         args.lookback, args.horizon, 'train')
    val_data = EVChargingBatchDataset(Y, M, time_features, ages.cpu().numpy(),
                                       args.lookback, args.horizon, 'val')
    test_data = EVChargingBatchDataset(Y, M, time_features, ages.cpu().numpy(),
                                        args.lookback, args.horizon, 'test')
    
    train_loader = DataLoader(train_data, args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, args.batch_size)
    test_loader = DataLoader(test_data, args.batch_size)
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Model
    print("\nCreating DYNAMIC TA-GNN model...")
    model = TAGNN_Dynamic(
        num_nodes=N,
        time_feat_dim=time_features.shape[1],
        hidden_dim=args.hidden_dim,
        num_gnn_layers=args.num_gnn_layers,
        num_temporal_layers=args.num_temporal_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        horizon=args.horizon,
        history_embed_dim=args.history_embed_dim,
        max_age_years=ages.max().item() + 0.5,
        use_static_adj=args.use_static_adj,
    ).to(device)
    
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {params:,}")
    print(f"Dynamic attention: ENABLED")
    print(f"Static adjacency as prior: {args.use_static_adj}")
    
    # Training setup
    criterion = MaskedLoss(alpha=0.3)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    def lr_lambda(epoch):
        warmup = 10
        if epoch < warmup:
            return (epoch + 1) / warmup
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup) / (args.epochs - warmup)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training
    print("\n" + "=" * 60)
    print("Training...")
    print("=" * 60)
    
    best_mae = float('inf')
    patience_counter = 0
    history = {'train': [], 'val_mae': [], 'val_rmse': []}
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, A, ages, device)
        val_metrics = validate(model, val_loader, criterion, A, ages, device)
        scheduler.step()
        
        lr = optimizer.param_groups[0]['lr']
        history['train'].append(train_loss)
        history['val_mae'].append(val_metrics['mae'])
        history['val_rmse'].append(val_metrics['rmse'])
        
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train: {train_loss:.4f} | "
              f"Val MAE: {val_metrics['mae']:.4f} | "
              f"RMSE: {val_metrics['rmse']:.4f} | "
              f"LR: {lr:.6f}")
        
        if val_metrics['mae'] < best_mae - 0.001:
            best_mae = val_metrics['mae']
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'mae': best_mae,
            }, f"{ckpt_dir}/best_model.pt")
            print(f"  ✓ Saved (MAE: {best_mae:.4f})")
        else:
            patience_counter += 1
        
        if patience_counter >= args.patience:
            print(f"\n⚠️ Early stopping at epoch {epoch+1}")
            break
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    
    ckpt = torch.load(f"{ckpt_dir}/best_model.pt", weights_only=False)
    model.load_state_dict(ckpt['model'])
    
    test_metrics = validate(model, test_loader, criterion, A, ages, device)
    print(f"\n📊 Test Results:")
    print(f"   MAE:  {test_metrics['mae']:.4f}")
    print(f"   RMSE: {test_metrics['rmse']:.4f}")
    
    print("\n📊 Age-Stratified Results:")
    age_metrics = evaluate_by_age(model, test_loader, A, ages, device)
    
    print(f"\n{'Group':<15} {'MAE':<10} {'RMSE':<10} {'Count':<10}")
    print("-" * 45)
    for g in ['cold_start', 'new', 'established', 'mature']:
        if f'{g}_mae' in age_metrics:
            print(f"{g:<15} {age_metrics[f'{g}_mae']:<10.4f} "
                  f"{age_metrics[f'{g}_rmse']:<10.4f} {age_metrics[f'{g}_count']:<10}")
    
    # Visualize attention
    print("\n📊 Visualizing Dynamic Attention...")
    fig = visualize_attention(model, test_loader, A, ages, device,
                              save_path=f"{ckpt_dir}/attention_visualization.png")
    
    # Save results
    results = {
        'test': {k: float(v) for k, v in test_metrics.items()},
        'age_metrics': age_metrics,
        'history': history,
        'args': vars(args),
    }
    with open(f"{ckpt_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Done! Results saved to {ckpt_dir}/")
    
    return model, results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--lookback', type=int, default=168)
    p.add_argument('--horizon', type=int, default=5)
    p.add_argument('--hidden_dim', type=int, default=64)
    p.add_argument('--num_gnn_layers', type=int, default=2)
    p.add_argument('--num_temporal_layers', type=int, default=2)
    p.add_argument('--num_heads', type=int, default=4)
    p.add_argument('--history_embed_dim', type=int, default=32)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=0.0005)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--patience', type=int, default=25)
    p.add_argument('--use_static_adj', action='store_true', default=True,
                   help='Use static adjacency as prior')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    model, results = main(args)
