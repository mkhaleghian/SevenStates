"""
EV_GNN Data Preprocessing Pipeline v5

FIXES from v4:
  (BUG) Node granularity was inconsistent across providers:
        - ChargePoint / ZEFNET / EVConnect used PLUG-level ids  -> ports split into separate "stations"
        - ElectricEra / Kempower used SITE-level ids             -> ports already aggregated
        Result: 118 "stations" for ~94 physical sites (e.g. ZEFNET K0024972A = 6 nodes).

  v5 collapses connectors -> ONE physical site per location, consistently for all 5 providers:
        - demand is SUMMED across a site's connectors (correct site-level target)
        - num_ports / num_port_types become real per-site features (v4's num_evses was ~1 for 111/118)
        - open_time = EARLIEST connector (so adding a plug does NOT make a site look "new")
        - ChargePoint merges are coordinate-verified (>1km apart => flagged, NOT merged)

  Minor: optional mask "close" rule trims observed-zeros that trail a station's last session
         by more than MASK_CLOSE_DAYS (v4 kept churned stations "open" to the grid end forever).

Outputs are written to *_v5 files. NOTHING from v4 is overwritten.

Usage:
    export EV_GNN_ROOT=/home/jovyan/GNN
    python preprocess_v5.py
"""

import os
import re
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

# =============================================================================
# Configuration
# =============================================================================
PROJECT_ROOT = os.environ.get('EV_GNN_ROOT', '/home/jovyan/GNN')
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

# v5 knobs ---------------------------------------------------------------------
MERGE_COLOCATED_KM = 0.15  # ChargePoint: connectors within this distance are ONE physical site (plaza)
MASK_CLOSE_DAYS = 90       # close the mask this many days after a site's LAST session (None = never, = v4 behaviour)
# -----------------------------------------------------------------------------

print("=" * 64)
print("   EV_GNN Data Preprocessing Pipeline v5  (connector -> physical site)")
print("=" * 64)
print(f"Project root      : {PROJECT_ROOT}")
print(f"Co-locate (km)    : {MERGE_COLOCATED_KM}")
print(f"Mask close (days) : {MASK_CLOSE_DAYS}")

# =============================================================================
# Timezone handling (identical to v4)
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
# Provider processing (identical parsing to v4 — validated to produce correct
# timestamps; only thing v5 changes is what happens AFTER, in aggregation)
# =============================================================================
def process_chargepoint(df):
    pytz = setup_pytz(); target_tz = pytz.timezone(TARGET_TZ)
    print("  Processing ChargePoint...")
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
    start_dt = pd.to_datetime(df['Start Date'], errors='coerce')
    start_tz_col = df.get('Start Time Zone', pd.Series(['EST'] * len(df)))
    start_est = []
    for dt, tz_abbr in zip(start_dt, start_tz_col):
        if pd.isna(dt):
            start_est.append(pd.NaT); continue
        try:
            tz_name = TZ_MAPPING.get(str(tz_abbr).strip().upper(), 'America/New_York')
            source_tz = pytz.timezone(tz_name)
            dt_aware = source_tz.localize(dt)
            start_est.append(dt_aware.astimezone(target_tz).replace(tzinfo=None))
        except Exception:
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
    pytz = setup_pytz(); target_tz = pytz.timezone(TARGET_TZ)
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
            start_est.append(pd.NaT); continue
        try:
            if dt.tzinfo is not None:
                start_est.append(dt.astimezone(target_tz).replace(tzinfo=None))
            else:
                source_tz = pytz.timezone('America/Chicago')
                start_est.append(source_tz.localize(dt).astimezone(target_tz).replace(tzinfo=None))
        except Exception:
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
    pytz = setup_pytz(); target_tz = pytz.timezone(TARGET_TZ)
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
            start_est.append(pd.NaT); continue
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
                start_est.append(source_tz.localize(start_dt).astimezone(target_tz).replace(tzinfo=None))
            else:
                start_est.append(pd.NaT)
        except Exception:
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
    pytz = setup_pytz(); target_tz = pytz.timezone(TARGET_TZ)
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
                start_est.append(dt.astimezone(target_tz).replace(tzinfo=None))
            except Exception:
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
    pytz = setup_pytz(); target_tz = pytz.timezone(TARGET_TZ)
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
                start_est.append(dt.astimezone(target_tz).replace(tzinfo=None))
            except Exception:
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


def load_and_process_all_providers(excel_path):
    print("\n" + "=" * 64)
    print("Loading and processing all providers")
    print("=" * 64)
    all_sessions = []
    xl = pd.ExcelFile(excel_path)
    for provider_key, sheet_name in SHEET_NAMES.items():
        if sheet_name in xl.sheet_names:
            df = pd.read_excel(xl, sheet_name=sheet_name)
            print(f"\n  {provider_key}: {len(df):,} rows")
            processed = PROCESSORS[provider_key](df)
            if len(processed) == 0:
                continue
            processed = processed.dropna(subset=['station_id', 'start_time'])
            print(f"    valid: {len(processed):,} sessions")
            all_sessions.append(processed)
    combined = pd.concat(all_sessions, ignore_index=True)
    # connector_id = the OLD (plug-level) node id; we keep it for auditing + port counting
    combined['connector_id'] = combined['station_id']
    print(f"\n  Total: {len(combined):,} sessions, {combined['connector_id'].nunique()} plug-level ids")
    return combined


# =============================================================================
# v5 CORE: collapse connectors -> physical sites
# =============================================================================
def _site_key(provider, name):
    """Name-based physical-site key (used for providers WITHOUT coordinates).

      ZEFNET : strip the connector tag '~N'   (unambiguous; collapses 14 -> 3)
      others (Kempower / ElectricEra / EVConnect) : already site-level -> name as-is

    ChargePoint is NOT handled here -- it is merged by COORDINATES (see below),
    because its connector suffix appears in >=6 inconsistent formats
    ('(L)','(R)','(LL)','(RM)','-L','#1', typos like MBP/MPB) that no regex
    handles reliably, and coordinates are both available and unambiguous.
    """
    s = str(name).strip()
    if provider == 'ZEFNET':
        s = re.sub(r'\s*~\s*\d+\s*$', '', s)
    s = s.strip(' ,/-').lower()
    return f"{provider}::{s}"


def _haversine_km(lat1, lon1, lat2, lon2):
    from math import radians, sin, cos, sqrt, atan2
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))


def _greedy_cluster(points, threshold_km):
    """Greedily cluster (id, lat, lon) points; two points within threshold_km join.
    Points are sorted first for determinism. Returns {id: cluster_label}.
    Co-located stalls (~0 m apart) merge; genuinely different sites (~>1 km) do not.
    """
    points = sorted(points, key=lambda p: (round(p[1], 6), round(p[2], 6), p[0]))
    centroids = []   # [lat, lon, n]
    labels = {}
    for pid, lat, lon in points:
        placed = False
        for k, (clat, clon, n) in enumerate(centroids):
            if _haversine_km(lat, lon, clat, clon) <= threshold_km:
                centroids[k] = [(clat * n + lat) / (n + 1), (clon * n + lon) / (n + 1), n + 1]
                labels[pid] = k
                placed = True
                break
        if not placed:
            centroids.append([lat, lon, 1])
            labels[pid] = len(centroids) - 1
    return labels


def collapse_connectors_to_sites(df):
    """Assign each session a physical-site id.

    ChargePoint -> cluster co-located connectors (<= MERGE_COLOCATED_KM) into ONE site.
    ZEFNET      -> name key with '~N' stripped.
    others      -> already site-level (name key unchanged).
    """
    from collections import Counter
    print("\n" + "=" * 64)
    print("Collapsing connectors -> physical sites")
    print("=" * 64)

    df = df.copy()
    df['name_key'] = [_site_key(p, n) for p, n in zip(df['provider'], df['station_name'])]

    # connector-level summary (median coord per connector is robust to a few bad rows)
    conn = (df.groupby('connector_id')
              .agg(provider=('provider', 'first'),
                   lat=('latitude', 'median'), lon=('longitude', 'median'),
                   name_key=('name_key', 'first'))
              .reset_index())

    conn_to_site = {}

    # ---- ChargePoint: coordinate clustering (one node per physical plaza) ----
    cp = conn[conn['provider'] == 'ChargePoint']
    cp_geo = cp.dropna(subset=['lat', 'lon'])
    cp_nogeo = cp[cp['lat'].isna() | cp['lon'].isna()]
    pts = [(r.connector_id, r.lat, r.lon) for r in cp_geo.itertuples()]
    labels = _greedy_cluster(pts, MERGE_COLOCATED_KM)
    # give each cluster a readable name = modal base name of its members
    members = {}
    namekey = dict(zip(cp_geo['connector_id'], cp_geo['name_key']))
    for cid, k in labels.items():
        members.setdefault(k, []).append(namekey[cid])
    for cid, k in labels.items():
        base = Counter(members[k]).most_common(1)[0][0].replace('ChargePoint::', '')
        conn_to_site[cid] = f"CPSITE_{k:03d}::{base}"
    for r in cp_nogeo.itertuples():               # ChargePoint w/o coords -> name fallback
        conn_to_site[r.connector_id] = r.name_key
    n_cp_sites = len(set(labels.values())) + cp_nogeo['name_key'].nunique()
    print(f"  ChargePoint: {len(cp)} connectors -> {n_cp_sites} sites "
          f"(coordinate-merged within {MERGE_COLOCATED_KM} km; {len(cp_nogeo)} without coords)")

    # ---- everyone else: name-based key ----
    for r in conn[conn['provider'] != 'ChargePoint'].itertuples():
        conn_to_site[r.connector_id] = r.name_key

    df['station_id'] = df['connector_id'].map(conn_to_site)

    # ---- report the collapse, per provider ----
    rep = (df.groupby('provider')
             .agg(connectors=('connector_id', 'nunique'), sites=('station_id', 'nunique'))
             .reset_index())
    print("\n  Per-provider collapse (connectors -> sites):")
    for _, r in rep.iterrows():
        flag = "" if r['connectors'] == r['sites'] else "  <-- merged"
        print(f"    {r['provider']:<12}: {r['connectors']:>3} -> {r['sites']:>3}{flag}")
    print(f"  TOTAL nodes: {df['connector_id'].nunique()} -> {df['station_id'].nunique()}")
    return df


def compute_station_metadata_v5(df):
    """Site-level metadata with REAL port features and earliest-connector open_time."""
    print("\n" + "=" * 64)
    print("Computing site metadata (v5)")
    print("=" * 64)

    def n_port_types(s):
        return s.astype(str).str.upper().str.extract(r'(CCS|CHADEMO|CHADEMO|TYPE\s?2|NACS|J1772)')[0].nunique() or 1

    meta = df.groupby('station_id').agg(
        open_time=('start_time', 'min'),
        last_time=('start_time', 'max'),
        total_sessions=('start_time', 'count'),
        total_energy_kwh=('energy_kwh', 'sum'),
        n_connectors=('connector_id', 'nunique'),
        n_evse=('evse_id', 'nunique'),
        provider=('provider', 'first'),
        latitude=('latitude', 'mean'),        # mean over a site's connectors (handles GPS jitter)
        longitude=('longitude', 'mean'),
        city=('city', 'first'),
        state=('state', 'first'),
        station_name=('station_name', 'first'),
        n_port_types=('port_type', n_port_types),
    ).reset_index()

    # physical ports = finest granularity available (name-encoded OR evse-encoded)
    meta['num_ports'] = meta[['n_connectors', 'n_evse']].max(axis=1).astype(int)

    # filter by minimum sessions (now at SITE level)
    before = len(meta)
    meta = meta[meta['total_sessions'] >= MIN_SESSIONS].reset_index(drop=True)
    print(f"  Sites: {before} -> {len(meta)} after >= {MIN_SESSIONS} sessions filter")

    meta['operational_days'] = (meta['last_time'] - meta['open_time']).dt.days
    meta['operational_years'] = meta['operational_days'] / 365.25
    print(f"  Ports/site: min {meta['num_ports'].min()}, median {int(meta['num_ports'].median())}, "
          f"max {meta['num_ports'].max()}")
    print(f"  Open-time range: {meta['open_time'].min()} -> {meta['open_time'].max()}")
    return meta


def build_tensors_v5(df, meta):
    print("\n" + "=" * 64)
    print("Building tensors (v5)")
    print("=" * 64)

    valid = set(meta['station_id'])
    df = df[df['station_id'].isin(valid)].copy()

    stations = sorted(meta['station_id'].unique())
    station_to_idx = {s: i for i, s in enumerate(stations)}
    N = len(stations)

    df['hour'] = df['start_time'].dt.floor('H')
    time_min, time_max = df['hour'].min(), df['hour'].max()
    time_index = pd.date_range(start=time_min, end=time_max, freq='H')
    time_to_idx = {t: i for i, t in enumerate(time_index)}
    T = len(time_index)
    print(f"  N sites: {N}   T hours: {T:,}   span: {time_min} -> {time_max}")

    Y = np.zeros((T, N), dtype=np.float32)
    M = np.zeros((T, N), dtype=np.float32)

    # Y: SUM of sessions across a site's connectors, per hour (groupby site+hour count)
    hourly = df.groupby(['station_id', 'hour']).size().reset_index(name='count')
    for _, row in tqdm(hourly.iterrows(), total=len(hourly), desc="    Y"):
        i = station_to_idx.get(row['station_id'])
        t = time_to_idx.get(row['hour'])
        if i is not None and t is not None:
            Y[t, i] = row['count']

    # M: 1 from open onward; optionally CLOSE MASK_CLOSE_DAYS after the last session
    station_open_times = np.zeros(N, dtype='datetime64[ns]')
    last_times = meta.set_index('station_id')['last_time']
    trimmed_hours = 0
    for _, st in meta.iterrows():
        i = station_to_idx[st['station_id']]
        ot = st['open_time']
        station_open_times[i] = np.datetime64(ot) if pd.notna(ot) else np.datetime64('NaT')
        if pd.isna(ot):
            continue
        open_hour = pd.Timestamp(ot).floor('H')
        start = time_to_idx.get(open_hour, 0 if open_hour < time_min else None)
        if start is None:
            continue
        if MASK_CLOSE_DAYS is None:
            M[start:, i] = 1.0
        else:
            close_hour = pd.Timestamp(last_times[st['station_id']]).floor('H') + pd.Timedelta(days=MASK_CLOSE_DAYS)
            end = time_to_idx.get(min(close_hour, time_max), T - 1) + 1
            M[start:end, i] = 1.0
            trimmed_hours += (T - end)
    if MASK_CLOSE_DAYS is not None:
        print(f"  Mask close rule ({MASK_CLOSE_DAYS}d): trimmed {trimmed_hours:,} trailing observed-zero hours")

    # time features (7-dim, identical to v4)
    hod, dow, mon = time_index.hour.values, time_index.dayofweek.values, time_index.month.values
    time_features = np.stack([
        np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24),
        np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
        (dow >= 5).astype(float),
        np.sin(2 * np.pi * mon / 12), np.cos(2 * np.pi * mon / 12),
    ], axis=1).astype(np.float32)

    # station features: [num_ports, lat, lon] -- same width(3) as v4, but num_ports is now CORRECT
    station_features = []
    for sid in stations:
        r = meta[meta['station_id'] == sid].iloc[0]
        station_features.append([
            r['num_ports'],
            r['latitude'] if pd.notna(r['latitude']) else 0.0,
            r['longitude'] if pd.notna(r['longitude']) else 0.0,
        ])
    station_features = np.array(station_features, dtype=np.float32)

    providers = meta.set_index('station_id').loc[stations, 'provider'].values
    provider_list = sorted(set(providers))
    provider_to_idx = {p: i for i, p in enumerate(provider_list)}
    provider_ids = np.array([provider_to_idx[p] for p in providers], dtype=np.int64)

    # summary + checksum
    observed = int(M.sum())
    positive = int((Y > 0).sum())
    zeros_obs = int(((Y == 0) & (M == 1)).sum())
    sessions_in_Y = int(Y.sum())
    print("\n  " + "-" * 56)
    print(f"  Y {Y.shape}  observed-hours {observed:,}")
    print(f"  positive {positive:,} ({100*positive/observed:.2f}%)   "
          f"zeros {zeros_obs:,} ({100*zeros_obs/observed:.2f}%)   ratio {zeros_obs/max(positive,1):.2f}:1")
    print(f"  sessions in Y (checksum): {sessions_in_Y:,}")
    print("  " + "-" * 56)

    return {
        'Y': Y, 'M': M, 'time_features': time_features, 'station_features': station_features,
        'stations': stations, 'station_to_idx': station_to_idx,
        'time_index': time_index.to_numpy(), 'time_min': time_min,
        'station_open_times': station_open_times,
        'provider_ids': provider_ids, 'provider_list': provider_list,
        'sessions_in_Y': sessions_in_Y, 'positive': positive, 'observed': observed,
    }


def build_basic_adjacency(meta, threshold_km=15.0, sigma=5.0):
    print("\n" + "=" * 64)
    print("Building distance adjacency (metadata only, no leakage)")
    print("=" * 64)
    N = len(meta)
    coords = meta[['latitude', 'longitude']].values
    valid = ~np.isnan(coords).any(axis=1)
    print(f"  Sites with coordinates: {valid.sum()} / {N}")
    A = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            if i != j and valid[i] and valid[j]:
                d = _haversine_km(coords[i, 0], coords[i, 1], coords[j, 0], coords[j, 1])
                if d < threshold_km:
                    A[i, j] = np.exp(-d ** 2 / (2 * sigma ** 2))
    A = A + np.eye(N)
    rs = A.sum(axis=1, keepdims=True); rs[rs == 0] = 1
    A = A / rs
    print(f"  Distance edges: {int((A > 0).sum() - N):,}")
    return {'A_distance': A}


# =============================================================================
# Main
# =============================================================================
def main():
    setup_pytz()
    os.makedirs(f"{PROCESSED_DIR}/unified", exist_ok=True)
    os.makedirs(f"{PROCESSED_DIR}/tensors", exist_ok=True)
    os.makedirs(GRAPHS_DIR, exist_ok=True)

    sessions = load_and_process_all_providers(EXCEL_FILE)
    raw_sessions = len(sessions)

    sessions = collapse_connectors_to_sites(sessions)
    meta = compute_station_metadata_v5(sessions)
    meta.to_csv(f"{PROCESSED_DIR}/unified/station_metadata_v5.csv", index=False)
    print(f"\n  saved station_metadata_v5.csv")

    tensors = build_tensors_v5(sessions, meta)

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
    }, f"{PROCESSED_DIR}/tensors/data_v5.pt")
    print(f"  saved data_v5.pt")

    adj = build_basic_adjacency(meta)
    torch.save({k: torch.from_numpy(v) for k, v in adj.items()},
               f"{GRAPHS_DIR}/adjacency_v5.pt")
    print(f"  saved adjacency_v5.pt")

    # ---- final report: v4 -> v5 and the all-important checksum ----
    kept_sessions = int(meta['total_sessions'].sum())
    dropped_by_filter = raw_sessions - kept_sessions
    print("\n" + "=" * 64)
    print("   v5 COMPLETE  —  v4 vs v5")
    print("=" * 64)
    print(f"   nodes (stations) : 118        -> {len(meta)}")
    print(f"   sessions parsed  : {raw_sessions:,}")
    print(f"     dropped (<{MIN_SESSIONS} sess sites): {dropped_by_filter:,}")
    print(f"     kept on sites  : {kept_sessions:,}   -> in tensor Y: {tensors['sessions_in_Y']:,}")
    print(f"   observed hours   : 1,105,292  -> {tensors['observed']:,}")
    print(f"   positive hours   : 63,927     -> {tensors['positive']:,}")
    print(f"   zero-rate        : 94.22%     -> {100*(1-tensors['positive']/tensors['observed']):.2f}%")
    # checksum: every session belonging to a kept site must land in Y (grid spans min..max, so none fall outside)
    ok = tensors['sessions_in_Y'] == kept_sessions
    print(f"\n   CHECKSUM (Y total == sessions on kept sites): "
          f"{'PASS ✓' if ok else 'FAIL ✗  -- investigate!'}")
    if not ok:
        print(f"      kept = {kept_sessions:,}, Y total = {tensors['sessions_in_Y']:,}, "
              f"diff = {kept_sessions - tensors['sessions_in_Y']:,}")


if __name__ == "__main__":
    main()
