"""
EV_GNN Data Preprocessing Pipeline v4

FIXES from v3:
1. MASK FIX: M=1 for ALL hours after station opens (not just positive demand)
2. Saves open_time for each station (for time-varying age calculation)
3. Saves time_index timestamps (for proper temporal features)

Usage:
    # PROJECT_ROOT is auto-detected from this file's location.
    # Optionally override it:  export EV_GNN_ROOT=/path/to/ICML
    python scripts_icml/data/preprocess_v4.py
"""

import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
import torch

# =============================================================================
# Configuration
# =============================================================================
PROJECT_ROOT = os.environ.get('EV_GNN_ROOT', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
EXCEL_FILE = f"{PROJECT_ROOT}/data/raw/Session Details.xlsx"
PROCESSED_DIR = f"{PROJECT_ROOT}/data/processed"
GRAPHS_DIR = f"{PROJECT_ROOT}/data/graphs"

SHEET_NAMES = {
    'ChargePoint': 'ChargePoint',
    'ZEFNET': 'ZEFNET',
    'EVConnect': 'EV Connect',
    'ElectricEra': 'Electric Era',
    'Kempower': 'Kempower',
}

MIN_SESSIONS = 20
TARGET_TZ = "America/New_York"
DC_FAST_KEYWORDS = ['DC', 'DCFC', 'Fast', 'CCS', 'CHAdeMO']

print(f"=" * 60)
print(f"   EV_GNN Data Preprocessing Pipeline v4")
print(f"=" * 60)
print(f"\n🔧 Key Fixes in v4:")
print(f"   • MASK FIX: M=1 for ALL hours after station opens")
print(f"   • Saves station open_time for time-varying age")
print(f"   • Saves time_index for proper hour alignment")
print(f"\nProject root: {PROJECT_ROOT}")

# =============================================================================
# Timezone Handling (same as v3)
# =============================================================================

TZ_MAPPING = {
    "EST": "America/New_York", "EDT": "America/New_York",
    "CST": "America/Chicago", "CDT": "America/Chicago",
    "MST": "America/Denver", "MDT": "America/Denver",
    "PST": "America/Los_Angeles", "PDT": "America/Los_Angeles",
    "ET": "America/New_York", "CT": "America/Chicago",
    "MT": "America/Denver", "PT": "America/Los_Angeles",
}

STATE_TZ_MAPPING = {
    'CT': 'America/New_York', 'DE': 'America/New_York', 'FL': 'America/New_York',
    'GA': 'America/New_York', 'IN': 'America/New_York', 'KY': 'America/New_York',
    'ME': 'America/New_York', 'MD': 'America/New_York', 'MA': 'America/New_York',
    'MI': 'America/New_York', 'NH': 'America/New_York', 'NJ': 'America/New_York',
    'NY': 'America/New_York', 'NC': 'America/New_York', 'OH': 'America/New_York',
    'PA': 'America/New_York', 'RI': 'America/New_York', 'SC': 'America/New_York',
    'VT': 'America/New_York', 'VA': 'America/New_York', 'WV': 'America/New_York',
    'DC': 'America/New_York',
    'AL': 'America/Chicago', 'AR': 'America/Chicago', 'IL': 'America/Chicago',
    'IA': 'America/Chicago', 'KS': 'America/Chicago', 'LA': 'America/Chicago',
    'MN': 'America/Chicago', 'MS': 'America/Chicago', 'MO': 'America/Chicago',
    'NE': 'America/Chicago', 'ND': 'America/Chicago', 'OK': 'America/Chicago',
    'SD': 'America/Chicago', 'TN': 'America/Chicago', 'TX': 'America/Chicago',
    'WI': 'America/Chicago',
    'AZ': 'America/Phoenix', 'CO': 'America/Denver', 'MT': 'America/Denver',
    'NM': 'America/Denver', 'UT': 'America/Denver', 'WY': 'America/Denver',
    'ID': 'America/Denver',
    'CA': 'America/Los_Angeles', 'NV': 'America/Los_Angeles',
    'OR': 'America/Los_Angeles', 'WA': 'America/Los_Angeles',
    'AK': 'America/Anchorage', 'HI': 'Pacific/Honolulu',
    'TENNESSEE': 'America/Chicago',
}


def setup_pytz():
    try:
        import pytz
        return pytz
    except ImportError:
        import subprocess
        subprocess.check_call(['pip', 'install', '-q', 'pytz'])
        import pytz
        return pytz


# =============================================================================
# Provider Processing Functions (same as v3)
# =============================================================================

def process_chargepoint(df):
    """Process ChargePoint data - filter to DC Fast, use Station Name."""
    pytz = setup_pytz()
    target_tz = pytz.timezone(TARGET_TZ)
    
    print("  Processing ChargePoint...")
    
    # Filter to DC Fast
    original_count = len(df)
    if 'Port Type' in df.columns:
        dc_mask = df['Port Type'].str.contains('|'.join(DC_FAST_KEYWORDS), case=False, na=False)
        df = df[dc_mask].copy()
        print(f"    Filtered to DC Fast: {len(df):,} / {original_count:,}")
    
    if len(df) == 0:
        return pd.DataFrame()
    
    result = pd.DataFrame()
    station_names = df['Station Name'].fillna('Unknown').astype(str).str.strip()
    result['station_id'] = 'CP_' + station_names.str.replace(r'[^\w\s-]', '', regex=True).str.replace(r'\s+', '_', regex=True)
    result['station_name'] = df['Station Name']
    result['evse_id'] = df['EVSE ID'].astype(str)
    
    # Parse times
    start_dt = pd.to_datetime(df['Start Date'], errors='coerce')
    start_tz_col = df.get('Start Time Zone', pd.Series(['EST'] * len(df)))
    
    start_est = []
    for dt, tz_abbr in zip(start_dt, start_tz_col):
        if pd.isna(dt):
            start_est.append(pd.NaT)
            continue
        try:
            tz_name = TZ_MAPPING.get(str(tz_abbr).strip().upper(), 'America/New_York')
            source_tz = pytz.timezone(tz_name)
            dt_aware = source_tz.localize(dt)
            dt_target = dt_aware.astimezone(target_tz)
            start_est.append(dt_target.replace(tzinfo=None))
        except:
            start_est.append(dt)
    
    result['start_time'] = pd.Series(start_est)
    result['energy_kwh'] = pd.to_numeric(df['Energy (kWh)'], errors='coerce')
    result['latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    result['longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    result['port_type'] = df['Port Type'].fillna('DC Fast')
    result['city'] = df['City']
    result['state'] = df['State/Province']
    result['provider'] = 'ChargePoint'
    
    return result


def process_zefnet(df):
    """Process ZEFNET data."""
    pytz = setup_pytz()
    target_tz = pytz.timezone(TARGET_TZ)
    
    print("  Processing ZEFNET...")
    
    result = pd.DataFrame()
    station_names = df['name'].fillna('Unknown').astype(str).str.strip()
    result['station_id'] = 'ZEF_' + station_names.str.replace(r'[^\w\s-]', '', regex=True).str.replace(r'\s+', '_', regex=True)
    result['station_name'] = df['name']
    result['evse_id'] = df['serial'].astype(str)
    
    start_dt = pd.to_datetime(df['start'], errors='coerce', utc=False)
    
    start_est = []
    for dt in start_dt:
        if pd.isna(dt):
            start_est.append(pd.NaT)
            continue
        try:
            if dt.tzinfo is not None:
                dt_target = dt.astimezone(target_tz)
                start_est.append(dt_target.replace(tzinfo=None))
            else:
                source_tz = pytz.timezone('America/Chicago')
                dt_aware = source_tz.localize(dt)
                dt_target = dt_aware.astimezone(target_tz)
                start_est.append(dt_target.replace(tzinfo=None))
        except:
            start_est.append(pd.NaT)
    
    result['start_time'] = pd.Series(start_est)
    result['energy_kwh'] = pd.to_numeric(df['energy'], errors='coerce')
    result['latitude'] = np.nan
    result['longitude'] = np.nan
    result['port_type'] = 'DC Fast'
    result['city'] = df['city']
    result['state'] = df['state']
    result['provider'] = 'ZEFNET'
    
    return result


def process_evconnect(df):
    """Process EV Connect data."""
    pytz = setup_pytz()
    target_tz = pytz.timezone(TARGET_TZ)
    
    print("  Processing EV Connect...")
    
    result = pd.DataFrame()
    station_names = df['Station Name'].fillna(df['Charge Box ID']).astype(str).str.strip()
    result['station_id'] = 'EVC_' + station_names.str.replace(r'[^\w\s-]', '', regex=True).str.replace(r'\s+', '_', regex=True)
    result['station_name'] = df['Station Name']
    result['evse_id'] = df['Charge Box ID'].astype(str).str.replace('"', '')
    
    date_col = pd.to_datetime(df['Date'], errors='coerce')
    
    start_est = []
    for i, row in df.iterrows():
        date_val = date_col.loc[i] if i in date_col.index else None
        if pd.isna(date_val):
            start_est.append(pd.NaT)
            continue
        try:
            state = str(row.get('State', 'Tennessee')).strip().upper()
            if len(state) > 2:
                state = state[:2] if state not in STATE_TZ_MAPPING else state
            tz_name = STATE_TZ_MAPPING.get(state, 'America/Chicago')
            source_tz = pytz.timezone(tz_name)
            
            date_str = date_val.strftime('%Y-%m-%d')
            connected_str = str(row.get('Connected Time', ''))
            
            start_dt = pd.to_datetime(f"{date_str} {connected_str}", errors='coerce')
            if pd.notna(start_dt):
                start_aware = source_tz.localize(start_dt)
                start_target = start_aware.astimezone(target_tz)
                start_est.append(start_target.replace(tzinfo=None))
            else:
                start_est.append(pd.NaT)
        except:
            start_est.append(pd.NaT)
    
    result['start_time'] = pd.Series(start_est)
    result['energy_kwh'] = pd.to_numeric(df['Energy Provided (kWh)'], errors='coerce')
    result['latitude'] = np.nan
    result['longitude'] = np.nan
    result['port_type'] = 'DC Fast'
    result['city'] = df['City']
    result['state'] = df['State']
    result['provider'] = 'EVConnect'
    
    return result


def process_electricera(df):
    """Process Electric Era data."""
    pytz = setup_pytz()
    target_tz = pytz.timezone(TARGET_TZ)
    
    print("  Processing Electric Era...")
    
    result = pd.DataFrame()
    addresses = df['address'].fillna('Unknown').astype(str).str.strip()
    cities = df['city'].fillna('Unknown').astype(str).str.strip()
    station_names = addresses + ', ' + cities
    result['station_id'] = 'EE_' + station_names.str.replace(r'[^\w\s,-]', '', regex=True).str.replace(r'\s+', '_', regex=True)
    result['station_name'] = station_names
    result['evse_id'] = df['chargerId'].astype(str)
    
    start_dt = pd.to_datetime(df['sessionStart'], errors='coerce', utc=True)
    
    start_est = []
    for dt in start_dt:
        if pd.isna(dt):
            start_est.append(pd.NaT)
        else:
            try:
                dt_target = dt.astimezone(target_tz)
                start_est.append(dt_target.replace(tzinfo=None))
            except:
                start_est.append(pd.NaT)
    
    result['start_time'] = pd.Series(start_est)
    result['energy_kwh'] = pd.to_numeric(df['energyDeliveredKwh'], errors='coerce')
    result['latitude'] = np.nan
    result['longitude'] = np.nan
    result['port_type'] = df.get('portType', pd.Series(['DC Fast'] * len(df)))
    result['city'] = df['city']
    result['state'] = df['state']
    result['provider'] = 'ElectricEra'
    
    return result


def process_kempower(df):
    """Process Kempower data."""
    pytz = setup_pytz()
    target_tz = pytz.timezone(TARGET_TZ)
    
    print("  Processing Kempower...")
    
    result = pd.DataFrame()
    if 'stationName' in df.columns and df['stationName'].notna().any():
        station_names = df['stationName'].fillna(df['Address'] + ', ' + df['city']).astype(str).str.strip()
    else:
        addresses = df['Address'].fillna('Unknown').astype(str).str.strip()
        cities = df['city'].fillna('Unknown').astype(str).str.strip()
        station_names = addresses + ', ' + cities
    
    result['station_id'] = 'KP_' + station_names.str.replace(r'[^\w\s,-]', '', regex=True).str.replace(r'\s+', '_', regex=True)
    result['station_name'] = station_names
    result['evse_id'] = df['evseId'].astype(str)
    
    start_dt = pd.to_datetime(df['startTime'], errors='coerce', utc=True)
    
    start_est = []
    for dt in start_dt:
        if pd.isna(dt):
            start_est.append(pd.NaT)
        else:
            try:
                dt_target = dt.astimezone(target_tz)
                start_est.append(dt_target.replace(tzinfo=None))
            except:
                start_est.append(pd.NaT)
    
    result['start_time'] = pd.Series(start_est)
    result['energy_kwh'] = pd.to_numeric(df['chargedEnergy kwh'], errors='coerce')
    result['latitude'] = np.nan
    result['longitude'] = np.nan
    result['port_type'] = df.get('evseType', pd.Series(['DC Fast'] * len(df)))
    result['city'] = df['city']
    result['state'] = df['state']
    result['provider'] = 'Kempower'
    
    return result


PROCESSORS = {
    'ChargePoint': process_chargepoint,
    'ZEFNET': process_zefnet,
    'EVConnect': process_evconnect,
    'ElectricEra': process_electricera,
    'Kempower': process_kempower,
}


# =============================================================================
# Main Processing Functions
# =============================================================================

def load_and_process_all_providers(excel_path):
    """Load and process all provider sheets."""
    print("\n" + "=" * 60)
    print("Loading and Processing All Providers")
    print("=" * 60)
    
    all_sessions = []
    xl = pd.ExcelFile(excel_path)
    
    for provider_key, sheet_name in SHEET_NAMES.items():
        if sheet_name in xl.sheet_names:
            df = pd.read_excel(xl, sheet_name=sheet_name)
            print(f"\n📂 {provider_key}: {len(df):,} rows")
            
            processor = PROCESSORS[provider_key]
            processed = processor(df)
            
            if len(processed) == 0:
                continue
            
            processed = processed.dropna(subset=['station_id', 'start_time'])
            print(f"    ✓ Valid: {len(processed):,} sessions")
            
            all_sessions.append(processed)
    
    combined = pd.concat(all_sessions, ignore_index=True)
    print(f"\n✓ Total: {len(combined):,} sessions, {combined['station_id'].nunique()} stations")
    
    return combined


def compute_station_metadata(df):
    """Compute station metadata including open_time."""
    print("\n" + "=" * 60)
    print("Computing Station Metadata")
    print("=" * 60)
    
    meta = df.groupby('station_id').agg({
        'start_time': ['min', 'max', 'count'],
        'energy_kwh': 'sum',
        'evse_id': 'nunique',
        'provider': 'first',
        'latitude': 'first',
        'longitude': 'first',
        'city': 'first',
        'state': 'first',
        'station_name': 'first',
    }).reset_index()
    
    meta.columns = ['station_id', 'open_time', 'last_time', 'total_sessions',
                    'total_energy_kwh', 'num_evses', 'provider', 'latitude', 'longitude',
                    'city', 'state', 'station_name']
    
    # Filter by minimum sessions
    meta = meta[meta['total_sessions'] >= MIN_SESSIONS].reset_index(drop=True)
    
    # Compute operational years (for reference, NOT for model use)
    meta['operational_days'] = (meta['last_time'] - meta['open_time']).dt.days
    meta['operational_years'] = meta['operational_days'] / 365.25
    
    print(f"✓ {len(meta)} stations (min {MIN_SESSIONS} sessions)")
    print(f"  Open time range: {meta['open_time'].min()} to {meta['open_time'].max()}")
    
    return meta


def build_tensors_v4(df, meta):
    """
    Build tensors with CORRECTED MASK.
    
    KEY FIX: M[t,i] = 1 for ALL hours after station i opened,
             not just hours with positive demand.
    """
    print("\n" + "=" * 60)
    print("Building Tensors (v4 - FIXED MASK)")
    print("=" * 60)
    
    # Filter sessions to valid stations
    valid_stations = set(meta['station_id'])
    df = df[df['station_id'].isin(valid_stations)].copy()
    
    # Create station index
    stations = sorted(meta['station_id'].unique())
    station_to_idx = {s: i for i, s in enumerate(stations)}
    N = len(stations)
    
    # Create time index
    df['hour'] = df['start_time'].dt.floor('H')
    time_min = df['hour'].min()
    time_max = df['hour'].max()
    time_index = pd.date_range(start=time_min, end=time_max, freq='H')
    time_to_idx = {t: i for i, t in enumerate(time_index)}
    T = len(time_index)
    
    print(f"  Stations (N): {N}")
    print(f"  Time steps (T): {T:,}")
    print(f"  Time range: {time_min} to {time_max}")
    
    # Initialize tensors
    Y = np.zeros((T, N), dtype=np.float32)  # Demand (session count)
    M = np.zeros((T, N), dtype=np.float32)  # Observation mask
    
    # =================================================================
    # STEP 1: Fill Y with session counts
    # =================================================================
    print("\n  Step 1: Filling demand tensor Y...")
    hourly = df.groupby(['station_id', 'hour']).size().reset_index(name='count')
    
    for _, row in tqdm(hourly.iterrows(), total=len(hourly), desc="    "):
        sid = row['station_id']
        hour = row['hour']
        if sid in station_to_idx and hour in time_to_idx:
            i = station_to_idx[sid]
            t = time_to_idx[hour]
            Y[t, i] = row['count']
    
    # =================================================================
    # STEP 2: Set M=1 for ALL hours AFTER station opened (KEY FIX!)
    # =================================================================
    print("\n  Step 2: Setting observation mask M (FIXED)...")
    print("    M=1 for ALL hours after station opened (including zeros)")
    
    # Create station open times array
    station_open_times = np.zeros(N, dtype='datetime64[ns]')
    
    for _, station in meta.iterrows():
        sid = station['station_id']
        if sid not in station_to_idx:
            continue
        i = station_to_idx[sid]
        open_time = station['open_time']
        station_open_times[i] = np.datetime64(open_time)
        
        if pd.isna(open_time):
            continue
        
        # Find the hour when station opened
        open_hour = pd.Timestamp(open_time).floor('H')
        if open_hour in time_to_idx:
            open_idx = time_to_idx[open_hour]
            # KEY FIX: M=1 from opening ONWARD (not just when sessions occur)
            M[open_idx:, i] = 1.0
        else:
            # Station opened before our time range
            M[:, i] = 1.0
    
    # =================================================================
    # STEP 3: Create time features using REAL timestamps
    # =================================================================
    print("\n  Step 3: Creating time features (using real timestamps)...")
    
    # Convert time_index to arrays
    time_index_np = time_index.to_numpy()
    hour_of_day = time_index.hour.values
    day_of_week = time_index.dayofweek.values
    month = time_index.month.values
    
    time_features = np.stack([
        np.sin(2 * np.pi * hour_of_day / 24),
        np.cos(2 * np.pi * hour_of_day / 24),
        np.sin(2 * np.pi * day_of_week / 7),
        np.cos(2 * np.pi * day_of_week / 7),
        (day_of_week >= 5).astype(float),  # is_weekend
        np.sin(2 * np.pi * month / 12),
        np.cos(2 * np.pi * month / 12),
    ], axis=1).astype(np.float32)
    
    # =================================================================
    # STEP 4: Create station features
    # =================================================================
    print("\n  Step 4: Creating station features...")
    
    station_features = []
    for sid in stations:
        station_row = meta[meta['station_id'] == sid].iloc[0]
        features = [
            station_row['num_evses'],
            station_row['latitude'] if pd.notna(station_row['latitude']) else 0,
            station_row['longitude'] if pd.notna(station_row['longitude']) else 0,
        ]
        station_features.append(features)
    station_features = np.array(station_features, dtype=np.float32)
    
    # =================================================================
    # STEP 5: Create provider encoding
    # =================================================================
    print("\n  Step 5: Creating provider encoding...")
    
    providers = meta.set_index('station_id').loc[stations, 'provider'].values
    provider_list = sorted(set(providers))
    provider_to_idx = {p: i for i, p in enumerate(provider_list)}
    provider_ids = np.array([provider_to_idx[p] for p in providers], dtype=np.int64)
    
    print(f"    Providers: {provider_list}")
    
    # =================================================================
    # Summary Statistics
    # =================================================================
    total_observed = M.sum()
    total_cells = M.size
    positive_demand = (Y > 0).sum()
    zero_demand_observed = ((Y == 0) & (M == 1)).sum()
    
    print(f"\n" + "=" * 60)
    print(f"  TENSOR SUMMARY (v4)")
    print(f"=" * 60)
    print(f"  Y shape: {Y.shape} (T × N)")
    print(f"  M shape: {M.shape}")
    print(f"  Mask coverage: {total_observed / total_cells * 100:.2f}%")
    print(f"  Hours with demand (Y>0): {positive_demand:,}")
    print(f"  Hours with zero demand but observed (Y=0, M=1): {zero_demand_observed:,}")
    print(f"  Ratio zeros/positive: {zero_demand_observed / positive_demand:.2f}x")
    print(f"  Time features shape: {time_features.shape}")
    print(f"  Station features shape: {station_features.shape}")
    
    return {
        'Y': Y,
        'M': M,
        'time_features': time_features,
        'station_features': station_features,
        'stations': stations,
        'station_to_idx': station_to_idx,
        'time_index': time_index_np,
        'time_min': time_min,
        'station_open_times': station_open_times,
        'provider_ids': provider_ids,
        'provider_list': provider_list,
    }


def build_basic_adjacency(meta, threshold_km=15.0, sigma=5.0):
    """Build basic distance-based adjacency (no demand features to avoid leakage)."""
    print("\n" + "=" * 60)
    print("Building Basic Adjacency (metadata only, no leakage)")
    print("=" * 60)
    
    from math import radians, sin, cos, sqrt, atan2
    
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        return 2 * R * atan2(sqrt(a), sqrt(1-a))
    
    N = len(meta)
    coords = meta[['latitude', 'longitude']].values
    valid_coords = ~np.isnan(coords).any(axis=1)
    
    print(f"  Stations with coordinates: {valid_coords.sum()} / {N}")
    
    # Distance-based adjacency
    A_dist = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            if i != j and valid_coords[i] and valid_coords[j]:
                d = haversine(coords[i,0], coords[i,1], coords[j,0], coords[j,1])
                if d < threshold_km:
                    A_dist[i, j] = np.exp(-d**2 / (2 * sigma**2))
    
    # Add self-loops and normalize
    A_dist = A_dist + np.eye(N)
    row_sum = A_dist.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    A_dist = A_dist / row_sum
    
    print(f"  Distance edges: {(A_dist > 0).sum() - N:,}")
    
    return {'A_distance': A_dist}


# =============================================================================
# Main
# =============================================================================

def main():
    setup_pytz()
    
    # Create directories
    os.makedirs(f"{PROCESSED_DIR}/unified", exist_ok=True)
    os.makedirs(f"{PROCESSED_DIR}/tensors", exist_ok=True)
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    
    # 1. Load and process
    sessions_df = load_and_process_all_providers(EXCEL_FILE)
    
    # 2. Metadata
    meta = compute_station_metadata(sessions_df)
    meta.to_csv(f"{PROCESSED_DIR}/unified/station_metadata_v4.csv", index=False)
    print(f"\n✓ Saved: station_metadata_v4.csv")
    
    # 3. Tensors (with FIXED mask)
    tensors = build_tensors_v4(sessions_df, meta)
    
    # Save as PyTorch tensors
    torch.save({
        'Y': torch.from_numpy(tensors['Y']),
        'M': torch.from_numpy(tensors['M']),
        'time_features': torch.from_numpy(tensors['time_features']),
        'station_features': torch.from_numpy(tensors['station_features']),
        'stations': tensors['stations'],
        'station_to_idx': tensors['station_to_idx'],
        'time_index': tensors['time_index'],
        'time_min': tensors['time_min'],
        'station_open_times': tensors['station_open_times'],
        'provider_ids': torch.from_numpy(tensors['provider_ids']),
        'provider_list': tensors['provider_list'],
    }, f"{PROCESSED_DIR}/tensors/data_v4.pt")
    
    print(f"✓ Saved: data_v4.pt")
    
    # 4. Basic adjacency (no leakage)
    adj = build_basic_adjacency(meta)
    torch.save({k: torch.from_numpy(v) for k, v in adj.items()}, 
               f"{GRAPHS_DIR}/adjacency_v4.pt")
    print(f"✓ Saved: adjacency_v4.pt")
    
    # Summary
    print("\n" + "=" * 60)
    print("   ✅ Preprocessing v4 Complete!")
    print("=" * 60)
    print(f"\n📊 Summary:")
    print(f"   • Stations: {len(meta)}")
    print(f"   • Time steps: {tensors['Y'].shape[0]:,}")
    print(f"   • Mask coverage: {tensors['M'].mean()*100:.2f}% (was ~1.9% in v3)")
    print(f"   • Zero-demand hours now OBSERVED ✓")
    print(f"   • Station open times saved ✓")
    print(f"   • Time index saved ✓")
    print(f"   • No demand-based adjacency (no leakage) ✓")


if __name__ == "__main__":
    main()
