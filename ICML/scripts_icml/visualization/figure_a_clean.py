"""
Figure A: Data Pathology - Clean Version
=========================================
Redesigned for clarity and publication quality.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

# Configuration
PROJECT_ROOT = os.environ.get('EV_GNN_ROOT', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'axes.linewidth': 0.8,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'zero': '#D0D0D0',
    'positive': '#D55E00',
    'cold_window': '#FFF3CD',
    'cold_border': '#856404',
    'obs_density': '#0072B2',
    'pos_rate': '#D55E00',
}


def create_clean_figure_a(Y, M, time_index, station_open_times, output_dir):
    """
    Clean version of Figure A: Data Pathology
    
    Panel (a): Zero-inflation pie/bar with separate histogram below
    Panel (b): Station lifecycle - simplified single-axis version
    """
    
    T, N = Y.shape
    
    # Create figure with 3 panels: (a) bars, (a-inset) histogram, (b) lifecycle
    fig = plt.figure(figsize=(11, 4.5))
    
    # Use GridSpec for precise layout
    gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1.5], width_ratios=[1, 1.4], 
                           hspace=0.4, wspace=0.35)
    
    # =========================================================================
    # Panel (a) TOP: Zero vs Positive bar chart
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Compute statistics
    valid_mask = M > 0
    valid_obs = Y[valid_mask]
    
    n_zeros = (valid_obs == 0).sum()
    n_positive = (valid_obs > 0).sum()
    total = len(valid_obs)
    
    pct_zeros = n_zeros / total * 100
    pct_positive = n_positive / total * 100
    ratio = n_zeros / n_positive if n_positive > 0 else float('inf')
    
    # Simple bar chart
    bars = ax1.bar([0, 1], [pct_zeros, pct_positive], 
                   color=[COLORS['zero'], COLORS['positive']],
                   edgecolor=['#888888', '#8B0000'], linewidth=1.5, width=0.6)
    
    # Labels on bars
    ax1.text(0, pct_zeros + 2, f'{pct_zeros:.1f}%', ha='center', va='bottom',
             fontsize=14, fontweight='bold', color='#333333')
    ax1.text(1, pct_positive + 2, f'{pct_positive:.1f}%', ha='center', va='bottom',
             fontsize=14, fontweight='bold', color=COLORS['positive'])
    
    # Ratio annotation - cleaner box
    ax1.annotate(f'{ratio:.0f}:1\nzero-to-positive\nratio', 
                 xy=(0.5, 55), ha='center', va='center', fontsize=10,
                 bbox=dict(boxstyle='round,pad=0.4', facecolor='#F8F8F8', 
                          edgecolor='#CCCCCC', linewidth=1))
    
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['Zero Demand\n(y = 0)', 'Positive Demand\n(y > 0)'], fontsize=10)
    ax1.set_ylabel('Percentage of Observations', fontweight='bold')
    ax1.set_ylim(0, 105)
    ax1.set_xlim(-0.5, 1.5)
    ax1.set_title('(a) Extreme Zero-Inflation', fontweight='bold', pad=8)
    
    # Remove top/right spines
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # =========================================================================
    # Panel (a) BOTTOM: Histogram of positive values
    # =========================================================================
    ax1_hist = fig.add_subplot(gs[1, 0])
    
    positive_vals = valid_obs[valid_obs > 0]
    
    ax1_hist.hist(positive_vals, bins=40, color=COLORS['positive'], 
                  edgecolor='white', alpha=0.85, linewidth=0.5)
    
    ax1_hist.set_xlabel('Charging Sessions per Hour (y | y > 0)', fontweight='bold')
    ax1_hist.set_ylabel('Frequency', fontweight='bold')
    ax1_hist.set_title('Distribution of Positive Demand', fontsize=10, style='italic')
    
    # Add statistics
    mean_pos = positive_vals.mean()
    median_pos = np.median(positive_vals)
    ax1_hist.axvline(mean_pos, color='#333333', linestyle='--', linewidth=1.5, label=f'Mean: {mean_pos:.2f}')
    ax1_hist.axvline(median_pos, color='#666666', linestyle=':', linewidth=1.5, label=f'Median: {median_pos:.2f}')
    ax1_hist.legend(loc='upper right', fontsize=8)
    
    ax1_hist.spines['top'].set_visible(False)
    ax1_hist.spines['right'].set_visible(False)
    
    # =========================================================================
    # Panel (b): Station Lifecycle - FULL HEIGHT, SIMPLIFIED
    # =========================================================================
    ax2 = fig.add_subplot(gs[:, 1])  # Span both rows
    
    # Compute observations by station age
    max_age_days = 365
    
    observations_by_day = np.zeros(max_age_days)
    positive_by_day = np.zeros(max_age_days)
    
    for i in range(N):
        station_open = station_open_times[i]
        for t in range(T):
            current_time = time_index[t]
            age_hours = (current_time - station_open).astype('timedelta64[h]').astype(float)
            age_days = int(age_hours / 24)
            
            if 0 <= age_days < max_age_days and M[t, i] > 0:
                observations_by_day[age_days] += 1
                if Y[t, i] > 0:
                    positive_by_day[age_days] += 1
    
    # Normalize observation density
    obs_normalized = observations_by_day / observations_by_day.max()
    
    # Smooth with rolling window
    window = 7
    obs_smooth = np.convolve(obs_normalized, np.ones(window)/window, mode='same')
    
    days = np.arange(max_age_days)
    
    # Plot observation density as filled area
    ax2.fill_between(days, 0, obs_smooth, alpha=0.4, color=COLORS['obs_density'],
                     label='Observation density')
    ax2.plot(days, obs_smooth, color=COLORS['obs_density'], linewidth=1.5)
    
    # Highlight cold-start window - PROMINENT
    cold_start_days = 14
    ax2.axvspan(0, cold_start_days, alpha=0.4, color=COLORS['cold_window'],
                edgecolor=COLORS['cold_border'], linewidth=2, linestyle='--')
    
    # Cold-start annotation - CLEAN
    ax2.annotate('', xy=(cold_start_days, 0.85), xytext=(0, 0.85),
                 arrowprops=dict(arrowstyle='<->', color=COLORS['cold_border'], lw=2))
    ax2.text(cold_start_days/2, 0.92, 'Cold-Start Window\n(14 days)', 
             ha='center', va='bottom', fontsize=12, fontweight='bold',
             color=COLORS['cold_border'])
    
    # Vertical line at cold-start boundary
    ax2.axvline(x=cold_start_days, color=COLORS['cold_border'], 
                linestyle='--', linewidth=2, alpha=0.8)
    
    # Arrow showing history accumulation
    ax2.annotate('Stations accumulate\nhistorical data', 
                 xy=(180, 0.5), xytext=(80, 0.25),
                 fontsize=9, ha='center', color='#555555',
                 arrowprops=dict(arrowstyle='->', color='#888888', lw=1.5))
    
    # Key insight annotation
    ax2.text(280, 0.75, 'Challenge:\nNew stations have\nlimited history for\ntemporal learning',
             fontsize=9, ha='left', va='center', style='italic', color='#444444',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                      edgecolor='#CCCCCC', alpha=0.9))
    
    ax2.set_xlabel('Days Since Station Opening', fontweight='bold')
    ax2.set_ylabel('Relative Observation Density', fontweight='bold')
    ax2.set_title('(b) Station Lifecycle and Cold-Start Evaluation Window', 
                  fontweight='bold', pad=8)
    ax2.set_xlim(0, max_age_days)
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc='upper right', fontsize=9)
    
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Light grid
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # =========================================================================
    # Save
    # =========================================================================
    plt.savefig(f"{output_dir}/figA_data_pathology_v2.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/figA_data_pathology_v2.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved figA_data_pathology_v2.pdf/png")
    print(f"  Zero-inflation: {pct_zeros:.1f}% zeros, {ratio:.1f}:1 ratio")
    
    return {
        'pct_zeros': pct_zeros,
        'pct_positive': pct_positive,
        'ratio': ratio,
        'mean_positive': mean_pos,
        'median_positive': median_pos,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data/processed/tensors/data_v4.pt')
    parser.add_argument('--output_dir', default='figures')
    args = parser.parse_args()
    
    print("="*60)
    print("Figure A: Data Pathology (Clean Version)")
    print("="*60)
    
    # Load data
    data_path = os.path.join(PROJECT_ROOT, args.data_path)
    print(f"\nLoading: {data_path}")
    
    data = torch.load(data_path, weights_only=False)
    
    Y = data['Y'].numpy()
    M = data['M'].numpy()
    time_index = data['time_index']
    station_open_times = data['station_open_times']
    
    T, N = Y.shape
    print(f"Data: T={T:,}, N={N}")
    
    # Output directory
    output_dir = os.path.join(PROJECT_ROOT, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate figure
    stats = create_clean_figure_a(Y, M, time_index, station_open_times, output_dir)
    
    print("\n" + "="*60)
    print("Done!")


if __name__ == "__main__":
    main()
