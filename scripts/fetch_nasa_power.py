"""
fetch_nasa_power.py
Fetch daily GHI + meteorological variables for all 8 stations
from the NASA POWER API (no API key required).
Saves: data/processed/nasa_power_all_stations.csv
"""

from __future__ import annotations

import multiprocessing
import time
from pathlib import Path

import pandas as pd
import requests

# ── Constants ─────────────────────────────────────────────────────────────────
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
PARAMETERS = "ALLSKY_SFC_SW_DWN,T2M,RH2M,WS10M,CLOUD_AMT"
COMMUNITY = "RE"
START_DATE = "20170401"
END_DATE   = "20241231"
FILL_VALUE = -999

RENAME_MAP: dict[str, str] = {
    'ALLSKY_SFC_SW_DWN': 'ghi',
    'T2M':               'temperature',
    'RH2M':              'humidity',
    'WS10M':             'wind_speed',
    'CLOUD_AMT':         'cloud_cover',
}

MAX_WORKERS = 4
TIMEOUT_SEC = 60
OUTPUT_PATH = ROOT / 'data' / 'processed' / 'nasa_power_all_stations.csv'


# ── Workers ────────────────────────────────────────────────────────────────────

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
    """Fetch NASA POWER data for one station and return a clean DataFrame.

    Args:
        args: Tuple of (station_name, latitude, longitude).

    Returns:
        DataFrame with columns [date, station, ghi, temperature,
        humidity, wind_speed, cloud_cover].  Returns empty DataFrame
        on any error.
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


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """Orchestrate parallel fetching, combine results, and save CSV."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    worker_args = [
        (name, lat, lon)
        for name, (lat, lon) in STATIONS.items()
    ]

    print(f"\n{'='*60}")
    print("fetch_nasa_power.py — NASA POWER API fetch")
    print(f"Stations : {list(STATIONS.keys())}")
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

    col_order = ['date', 'station', 'ghi', 'temperature', 'humidity',
                 'wind_speed', 'cloud_cover']
    combined = combined[col_order]

    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved → {OUTPUT_PATH}")

    # ── SELF CHECK ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SELF CHECK")
    print(f"{'='*60}")

    print(f"\nTotal rows          : {len(combined):,}")
    print(f"Unique stations     : {combined['station'].nunique()}")
    assert combined['station'].nunique() == 8, "Expected 8 unique stations!"

    print(f"Date range          : {combined['date'].min().date()} → {combined['date'].max().date()}")

    print("\nMissing values per column per station:")
    missing = combined.groupby('station')[
        ['ghi', 'temperature', 'humidity', 'wind_speed', 'cloud_cover']
    ].apply(lambda g: g.isna().sum())
    print(missing.to_string())

    print("\nGHI statistics per station (MJ/m²/day):")
    ghi_stats = combined.groupby('station')['ghi'].agg(['mean', 'min', 'max'])
    ghi_stats.columns = ['mean', 'min', 'max']
    print(ghi_stats.round(3).to_string())

    print("\nFirst 5 rows:")
    print(combined.head(5).to_string(index=False))
    print(f"\n{'='*60}")
    print("Script 1 complete.")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
