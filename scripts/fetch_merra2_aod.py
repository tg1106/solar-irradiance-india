"""
MERRA-2 AOD Extractor
Reads OPeNDAP URLs from subset txt file, extracts daily mean AOD
at 8 station coordinates, outputs clean CSV.

Setup:
    1. Configure ~/.netrc with NASA Earthdata credentials
       (see startup check in main() for exact format)
    2. pip install requests netCDF4 numpy pandas
    3. Place nasa-merra2_aod.txt in data/raw/
    4. Run from anywhere: python scripts/fetch_merra2_aod.py
"""

from __future__ import annotations

import io
import re
import sys
import time
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd
import requests

# ── PATH ANCHORS — works from any directory ───────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "processed"
LINKS_FILE = BASE_DIR / "data" / "raw" / "nasa-merra2_aod.txt"

# ── CONSTANTS ────────────────────────────────────────────────────
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

AOD_VARS: list[str] = [
    'TOTEXTTAU',
    'DUEXTTAU',
    'BCEXTTAU',
    'SSEXTTAU',
    'SUEXTTAU',
]

FILL_THRESHOLD: float = 1e15   # MERRA-2 official fill value
CHECKPOINT_EVERY: int  = 100   # save progress every N days
REQUEST_TIMEOUT:  int  = 180   # seconds — increased for OPeNDAP latency
POLITE_DELAY:     float = 2.0  # seconds between requests — OPeNDAP rate limit
MAX_RETRIES:      int  = 3
BACKOFF:          list = [5, 15, 30]  # seconds to wait between retries
SESSION_REFRESH:  int  = 50   # refresh session every N requests


# ── HELPERS ──────────────────────────────────────────────────────

def parse_date_from_url(url: str) -> str | None:
    """
    Extract YYYYMMDD string from OPeNDAP URL filename.

    Args:
        url: Full OPeNDAP URL string
    Returns:
        Date string YYYYMMDD or None if not found
    """
    match = re.search(r'(\d{8})\.nc4', url)
    return match.group(1) if match else None


def find_nearest_idx(arr: np.ndarray, val: float) -> int:
    """
    Find index of nearest value in a numpy array.

    Args:
        arr: 1D numpy array of coordinates
        val: Target value to find
    Returns:
        Integer index of nearest element
    """
    return int(np.argmin(np.abs(arr - val)))


def fetch_and_extract(
    url: str,
    session: requests.Session
) -> dict | None:
    """
    Download one OPeNDAP URL and extract station AOD values.

    Downloads the pre-clipped India NetCDF4 file, finds nearest
    grid pixel for each station, averages 24 hourly values to
    produce a daily mean per variable per station.

    Args:
        url:     OPeNDAP URL for one day
        session: Authenticated requests.Session
    Returns:
        Dict with 'date' + station_variable columns, or None on failure
    """
    date_str = parse_date_from_url(url)
    if not date_str:
        print("  Could not parse date from URL — skipping")
        return None

    date = pd.to_datetime(date_str, format="%Y%m%d")

    try:
        for attempt in range(MAX_RETRIES):
            try:
                response = session.get(url, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()
                break  # success
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    wait = BACKOFF[attempt]
                    print(f"  Retry {attempt+1}/{MAX_RETRIES} "
                          f"after {wait}s — {e}")
                    time.sleep(wait)
                else:
                    raise  # final attempt failed, let caller handle

        # Load NetCDF from bytes — no temp file needed
        nc_bytes = io.BytesIO(response.content)
        ds = nc.Dataset("in-memory.nc", memory=nc_bytes.read())

        lats = ds.variables['lat'][:]
        lons = ds.variables['lon'][:]

        row: dict = {'date': date.strftime('%Y-%m-%d')}

        for station, (slat, slon) in STATIONS.items():
            lat_idx = find_nearest_idx(lats, slat)
            lon_idx = find_nearest_idx(lons, slon)

            for var in AOD_VARS:
                if var not in ds.variables:
                    row[f"{station}_{var}"] = float('nan')
                    continue
                data = ds.variables[var][:, lat_idx, lon_idx]
                if hasattr(data, 'mask'):
                    data = data.filled(np.nan)
                data = data.astype(float)
                data[data > FILL_THRESHOLD] = np.nan
                row[f"{station}_{var}"] = round(float(np.nanmean(data)), 6)

        ds.close()
        return row

    except Exception as e:
        print(f"  FAILED {date_str}: {e}")
        return None


# ── MAIN ─────────────────────────────────────────────────────────

def main() -> None:
    """
    Main pipeline — reads links file, fetches all days,
    extracts station AOD values, saves to CSV with checkpoints.
    """
    # ── Validate setup ───────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not LINKS_FILE.exists():
        print(f"ERROR: Links file not found at {LINKS_FILE}")
        print("Place nasa-merra2_aod.txt in data/raw/")
        return

    # ── Read URLs ────────────────────────────────────────────────
    with open(LINKS_FILE, 'r') as f:
        all_lines = [line.strip() for line in f.readlines()]

    urls = [
        line for line in all_lines
        if 'opendap.earthdata.nasa.gov' in line and '.nc4' in line
    ]

    if not urls:
        print("ERROR: No OPeNDAP URLs found in links file")
        return

    print("=" * 60)
    print("MERRA-2 AOD Extractor")
    print("=" * 60)
    print(f"Links file:  {LINKS_FILE}")
    print(f"URLs found:  {len(urls)}")
    print(f"Date range:  {parse_date_from_url(urls[0])} → "
          f"{parse_date_from_url(urls[-1])}")
    print(f"Stations:    {list(STATIONS.keys())}")
    print(f"Variables:   {AOD_VARS}")
    print("=" * 60)

    # ── Authenticated session — uses ~/.netrc ─────────────────────
    session = requests.Session()
    session.trust_env = True  # uses ~/.netrc automatically

    netrc_path = Path.home() / ".netrc"
    if not netrc_path.exists():
        print("ERROR: ~/.netrc not found.")
        print("Create it with:")
        print("  machine urs.earthdata.nasa.gov")
        print("      login YOUR_USERNAME")
        print("      password YOUR_PASSWORD")
        print("  machine opendap.earthdata.nasa.gov")
        print("      login YOUR_USERNAME")
        print("      password YOUR_PASSWORD")
        print("Then: chmod 0600 ~/.netrc")
        sys.exit(1)

    cookie_file = Path.home() / ".urs_cookies"
    cookie_file.touch(exist_ok=True)

    results: list[dict] = []
    failed:  list[str]  = []

    # ── Fetch loop ────────────────────────────────────────────────
    for i, url in enumerate(urls, 1):
        date_str = parse_date_from_url(url)
        print(f"[{i:04d}/{len(urls)}] {date_str}", end=" ... ")

        row = fetch_and_extract(url, session)

        if row:
            results.append(row)
            # Safe print — val is float or missing
            val = row.get('Jodhpur_TOTEXTTAU')
            aod_str = f"{val:.4f}" if val is not None and not np.isnan(val) \
                      else "N/A"
            print(f"Jodhpur TOTEXTTAU={aod_str}")
        else:
            failed.append(url)
            print("FAILED")

        # Checkpoint every N days
        if i % CHECKPOINT_EVERY == 0 and results:
            checkpoint_path = OUTPUT_DIR / "merra2_aod_checkpoint.csv"
            pd.DataFrame(results).to_csv(checkpoint_path, index=False)
            print(f"  >>> Checkpoint saved — {len(results)} rows "
                  f"({len(failed)} failed so far)")

        # Refresh session every 50 requests to reset cookies
        if i % SESSION_REFRESH == 0:
            session.close()
            session = requests.Session()
            session.trust_env = True
            print(f"  Session refreshed at request {i}")

        time.sleep(POLITE_DELAY)

    # ── Save final output ─────────────────────────────────────────
    if not results:
        print("\nERROR: No data fetched — check credentials and network")
        return

    df = pd.DataFrame(results)
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    out_path = OUTPUT_DIR / "merra2_aod_stations.csv"
    df.to_csv(out_path, index=False)

    # ── Self check ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SELF CHECK")
    print("=" * 60)
    print(f"Total rows:      {len(df):,}")
    print(f"Date range:      {df['date'].min()} → {df['date'].max()}")
    print(f"Total columns:   {len(df.columns)}")
    print(f"Successful days: {len(results):,}")
    print(f"Failed days:     {len(failed):,}")

    if failed:
        print(f"\nFirst 5 failed URLs:")
        for u in failed[:5]:
            print(f"  {parse_date_from_url(u)}")

    print(f"\nTOTEXTTAU stats per station:")
    for station in STATIONS:
        col = f"{station}_TOTEXTTAU"
        if col in df.columns:
            mean = df[col].mean()
            mn   = df[col].min()
            mx   = df[col].max()
            nulls = df[col].isna().sum()
            print(f"  {station:<12} mean={mean:.4f}  "
                  f"min={mn:.4f}  max={mx:.4f}  "
                  f"missing={nulls}")

    print(f"\nMissing values per column (total):")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("  None — all values present")
    else:
        print(missing.to_string())

    print(f"\nSample (first 3 rows):")
    print(df[['date', 'Jodhpur_TOTEXTTAU', 'Chennai_TOTEXTTAU',
              'Kolkata_TOTEXTTAU']].head(3).to_string())

    print(f"\nSaved to: {out_path}")
    print("=" * 60)

    if len(failed) == 0:
        print("STATUS: All days fetched successfully.")
    elif len(failed) < 10:
        print(f"STATUS: {len(failed)} days failed — acceptable, "
              f"re-run to retry.")
    else:
        print(f"STATUS: {len(failed)} days failed — check credentials "
              f"and re-run.")


if __name__ == "__main__":
    main()