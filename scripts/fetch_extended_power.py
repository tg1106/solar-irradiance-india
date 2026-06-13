"""
fetch_extended_power.py
Phase B — Script 8.
Re-fetch NASA POWER data for all 8 stations with two additional
parameters needed for the wind-energy physics conversion:
  PS    = Surface pressure (kPa)
  WS50M = Wind speed at 50 metres (m/s)

Saves: data/processed/nasa_power_extended.csv
"""

from __future__ import annotations

import multiprocessing
import time
from pathlib import Path

import pandas as pd
import requests

# ── Constants ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

STATIONS: dict[str, tuple[float, float]] = {
    'Jodhpur':   (26.30, 73.02),
    'Ahmedabad': (23.03, 72.58),
    'Mumbai':    (19.08, 72.88),
    'Chennai':   (13.08, 80.27),
    'Hyderabad': (17.38, 78.48),
    'Bhopal':    (23.25, 77.40),
    'Kolkata':   (22.57, 88.36),
    'Bengaluru': (12.97, 77.59),
}

API_BASE = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMETERS = "ALLSKY_SFC_SW_DWN,T2M,RH2M,WS10M,CLOUD_AMT,PS,WS50M"
COMMUNITY = "RE"
START_DATE = "20170401"
END_DATE   = "20241231"
FILL_VALUE = -999

RENAME_MAP: dict[str, str] = {
    'ALLSKY_SFC_SW_DWN': 'ghi',
    'T2M':               'temperature',
    'RH2M':              'humidity',
    'WS10M':             'wind_speed_10m',
    'CLOUD_AMT':         'cloud_cover',
    'PS':                'surface_pressure',
    'WS50M':             'wind_speed_50m',
}

MAX_WORKERS = 4
TIMEOUT_SEC = 60
OUTPUT_PATH = ROOT / 'data' / 'processed' / 'nasa_power_extended.csv'

COL_ORDER = [
    'date', 'station', 'ghi', 'temperature', 'humidity',
    'wind_speed_10m', 'cloud_cover', 'surface_pressure', 'wind_speed_50m',
]


# ── Workers ──────────────────────────────────────────────────────────────────

def build_url(lat: float, lon: float) -> str:
    """Build the NASA POWER API URL for a single point.

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.

    Returns:
        Fully-formed query URL string.
    """
    return (
        f"{API_BASE}"
        f"?parameters={PARAMETERS}"
        f"&community={COMMUNITY}"
        f"&longitude={lon}"
        f"&latitude={lat}"
        f"&start={START_DATE}"
        f"&end={END_DATE}"
        f"&format=JSON"
    )


def fetch_station(args: tuple[str, float, float]) -> pd.DataFrame:
    """Fetch extended NASA POWER data for one station.

    Args:
        args: Tuple of (station_name, latitude, longitude).

    Returns:
        DataFrame with COL_ORDER columns. Empty DataFrame on error.
    """
    station_name, lat, lon = args
    url = build_url(lat, lon)
    print(f"  [START] {station_name}  lat={lat}  lon={lon}")

    try:
        resp = requests.get(url, timeout=TIMEOUT_SEC, verify=True)
        resp.raise_for_status()
        payload = resp.json()

        param_data: dict = payload['properties']['parameter']

        df = pd.DataFrame(param_data)
        df.index = pd.to_datetime(df.index, format='%Y%m%d')
        df.index.name = 'date'
        df = df.rename(columns=RENAME_MAP)
        df = df.replace(FILL_VALUE, float('nan'))
        df['station'] = station_name
        df = df.reset_index()

        print(f"  [OK]    {station_name}  rows={len(df)}")
        return df

    except Exception as exc:  # noqa: BLE001
        print(f"  [ERROR] {station_name}: {exc}")
        return pd.DataFrame()


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    """Orchestrate parallel fetching, combine results, and save CSV."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    worker_args = [
        (name, lat, lon)
        for name, (lat, lon) in STATIONS.items()
    ]

    print(f"\n{'='*60}")
    print("fetch_extended_power.py — NASA POWER extended fetch")
    print(f"Stations : {list(STATIONS.keys())}")
    print(f"Params   : {PARAMETERS}")
    print(f"Date range: {START_DATE} → {END_DATE}")
    print(f"Workers  : {MAX_WORKERS}")
    print(f"{'='*60}\n")

    t0 = time.time()
    with multiprocessing.Pool(processes=MAX_WORKERS) as pool:
        results = pool.map(fetch_station, worker_args)
    elapsed = time.time() - t0
    print(f"\nAll requests complete in {elapsed:.1f}s")

    non_empty = [df for df in results if not df.empty]
    if not non_empty:
        raise RuntimeError("All API requests failed. Check network or API endpoint.")

    combined = pd.concat(non_empty, ignore_index=True)
    combined = combined.sort_values(['station', 'date']).reset_index(drop=True)
    combined = combined[COL_ORDER]

    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved → {OUTPUT_PATH}")

    # ── SELF CHECK ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SELF CHECK")
    print(f"{'='*60}")

    print(f"\nTotal rows          : {len(combined):,}")
    print(f"Unique stations     : {combined['station'].nunique()}")
    assert combined['station'].nunique() == 8, "Expected 8 unique stations!"

    print(f"Date range          : {combined['date'].min().date()} → {combined['date'].max().date()}")

    print("\nMissing values per column per station:")
    data_cols = ['ghi', 'temperature', 'humidity', 'wind_speed_10m',
                  'cloud_cover', 'surface_pressure', 'wind_speed_50m']
    missing = combined.groupby('station')[data_cols].apply(lambda g: g.isna().sum())
    print(missing.to_string())

    n_missing_ghi   = int(combined['ghi'].isna().sum())
    n_missing_ws50  = int(combined['wind_speed_50m'].isna().sum())
    n_missing_ps    = int(combined['surface_pressure'].isna().sum())
    print(f"\nZero missing in ghi          : {n_missing_ghi == 0}")
    print(f"Zero missing in wind_speed_50m: {n_missing_ws50 == 0}")
    print(f"Zero missing in surface_pressure: {n_missing_ps == 0}")

    print("\nwind_speed_50m statistics per station (m/s):")
    ws_stats = combined.groupby('station')['wind_speed_50m'].agg(['mean', 'min', 'max'])
    print(ws_stats.round(3).to_string())

    print("\nsurface_pressure statistics per station (kPa):")
    ps_stats = combined.groupby('station')['surface_pressure'].agg(['mean', 'min', 'max'])
    print(ps_stats.round(3).to_string())

    print("\nFirst 5 rows:")
    print(combined.head(5).to_string(index=False))
    print(f"\n{'='*60}")
    print("Script 8 complete.")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
