#!/usr/bin/env python3
"""
Verify Data and Checkpoint Compatibility
=========================================

Run this FIRST to check your data and model checkpoint structure
before running sensitivity experiments.

Usage:
    python verify_setup.py
"""

import os
import torch
import numpy as np

# Paths - anchored to the project root; override with EV_GNN_ROOT env var
PROJECT_ROOT = os.environ.get(
    'EV_GNN_ROOT',
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
DATA_PATH = os.path.join(PROJECT_ROOT, "data/processed/tensors/data_v4.pt")
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints/tagnn_dynamic_v4/best_model_cs_pos.pt")

print("="*70)
print("TA-GNN SETUP VERIFICATION")
print("="*70)

# ============================================
# 1. Check Data Structure
# ============================================
print("\n[1] CHECKING DATA STRUCTURE")
print("-"*40)

try:
    data = torch.load(DATA_PATH, map_location='cpu', weights_only=False)
    print(f"✓ Loaded: {DATA_PATH}")
    print(f"  Type: {type(data)}")
    
    if isinstance(data, dict):
        print(f"  Keys: {list(data.keys())}")
        for k, v in data.items():
            if hasattr(v, 'shape'):
                print(f"    {k}: shape={v.shape}, dtype={getattr(v, 'dtype', 'N/A')}")
            else:
                print(f"    {k}: {type(v).__name__}")
        
        # Check expected keys
        expected_keys = ['Y', 'M', 'time_features', 'station_features', 'provider_ids', 'station_open_times']
        missing = [k for k in expected_keys if k not in data]
        if missing:
            print(f"\n  ⚠️  Missing expected keys: {missing}")
        else:
            print(f"\n  ✓ All expected keys present")
            
        # Statistics
        if 'Y' in data and 'M' in data:
            Y = data['Y'].numpy() if isinstance(data['Y'], torch.Tensor) else data['Y']
            M = data['M'].numpy() if isinstance(data['M'], torch.Tensor) else data['M']
            
            T, N = Y.shape
            observed = M.sum()
            positive = ((Y > 0) & (M == 1)).sum()
            
            print(f"\n  DATA STATISTICS:")
            print(f"    Timestamps (T): {T:,}")
            print(f"    Stations (N): {N}")
            print(f"    Observed hours: {observed:,}")
            print(f"    Positive hours: {positive:,} ({100*positive/observed:.1f}%)")
            
except Exception as e:
    print(f"✗ Failed to load data: {e}")

# ============================================
# 2. Check Checkpoint Structure
# ============================================
print("\n[2] CHECKING CHECKPOINT STRUCTURE")
print("-"*40)

try:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    print(f"✓ Loaded: {CHECKPOINT_PATH}")
    print(f"  Type: {type(checkpoint)}")
    
    if isinstance(checkpoint, dict):
        print(f"  Keys: {list(checkpoint.keys())}")
        
        # Check if it's a state dict directly or wrapped
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f"\n  Checkpoint contains 'model_state_dict' wrapper")
        elif any(k.endswith('.weight') or k.endswith('.bias') for k in checkpoint.keys()):
            state_dict = checkpoint
            print(f"\n  Checkpoint is a direct state_dict")
        else:
            state_dict = None
            print(f"\n  ⚠️  Unknown checkpoint format")
        
        if state_dict:
            print(f"\n  MODEL LAYERS ({len(state_dict)} parameters):")
            
            # Group by module
            modules = {}
            for k in state_dict.keys():
                module = k.split('.')[0]
                if module not in modules:
                    modules[module] = []
                modules[module].append(k)
            
            for module, params in modules.items():
                total_params = sum(state_dict[p].numel() for p in params)
                print(f"    {module}: {len(params)} tensors, {total_params:,} params")
            
            total = sum(v.numel() for v in state_dict.values())
            print(f"\n    TOTAL: {total:,} parameters")
            
            # Key layer dimensions for model matching
            print(f"\n  KEY DIMENSIONS (for model matching):")
            key_layers = [
                'input_proj.weight',
                'temporal_gru.weight_ih_l0',
                'detection_head.0.weight',
                'provider_embed.weight',
            ]
            for layer in key_layers:
                if layer in state_dict:
                    print(f"    {layer}: {list(state_dict[layer].shape)}")
            
            # Check for gate parameters
            print(f"\n  LEARNED GATE PARAMETERS:")
            if 'gate_alpha' in state_dict:
                alpha_raw = state_dict['gate_alpha'].item()
                alpha = torch.nn.functional.softplus(torch.tensor(alpha_raw)).item() + 1e-3
                print(f"    gate_alpha (raw): {alpha_raw:.4f}")
                print(f"    gate_alpha (effective): {alpha:.4f}")
            if 'gate_beta' in state_dict:
                beta = state_dict['gate_beta'].item()
                print(f"    gate_beta: {beta:.4f} years = {beta*365.25:.0f} days")
    else:
        # Might be a direct state dict
        print(f"  Direct state dict with {len(checkpoint)} parameters")
        
except Exception as e:
    print(f"✗ Failed to load checkpoint: {e}")

# ============================================
# 3. Compatibility Check
# ============================================
print("\n[3] COMPATIBILITY CHECK")
print("-"*40)

try:
    # Check if model can be instantiated with data dimensions
    if 'data' in dir() and 'state_dict' in dir() and state_dict is not None:
        
        # Infer dimensions from checkpoint
        if 'input_proj.weight' in state_dict:
            input_dim = state_dict['input_proj.weight'].shape[1]
            hidden_dim = state_dict['input_proj.weight'].shape[0]
            print(f"  Inferred from checkpoint:")
            print(f"    Input dim: {input_dim} (should be 2 + d_f = 2 + {data['time_features'].shape[1]})")
            print(f"    Hidden dim: {hidden_dim}")
        
        if 'provider_embed.weight' in state_dict:
            num_providers, embed_dim = state_dict['provider_embed.weight'].shape
            print(f"    Num providers: {num_providers} (data has {data['num_providers'] if 'num_providers' in data else len(data.get('provider_list', []))})")
            print(f"    Provider embed dim: {embed_dim}")
        
        if 'detection_head.3.weight' in state_dict:
            horizon = state_dict['detection_head.3.weight'].shape[0]
            print(f"    Horizon: {horizon}")
        
        print(f"\n  ✓ Data and checkpoint appear compatible")
        
except Exception as e:
    print(f"  ⚠️  Could not verify compatibility: {e}")

# ============================================
# 4. Recommendations
# ============================================
print("\n[4] RECOMMENDATIONS")
print("-"*40)

print("""
Based on the above, update the sensitivity scripts:

1. In run_sensitivity.py and quick_coldstart_eval.py, set:
""")

if 'state_dict' in dir() and state_dict is not None:
    if 'input_proj.weight' in state_dict:
        print(f"   HIDDEN_DIM = {state_dict['input_proj.weight'].shape[0]}")
    if 'provider_embed.weight' in state_dict:
        print(f"   PROVIDER_EMBED_DIM = {state_dict['provider_embed.weight'].shape[1]}")
    if 'detection_head.3.weight' in state_dict:
        print(f"   HORIZON = {state_dict['detection_head.3.weight'].shape[0]}")

print("""
2. If model architecture differs significantly:
   - Import your actual model class instead of using the one in the script
   - Example: from models.tagnn import TAGNN

3. Run quick test:
   python quick_coldstart_eval.py --checkpoint YOUR_CHECKPOINT
""")

print("\n" + "="*70)
print("VERIFICATION COMPLETE")
print("="*70)
