"""
merge_dataset.py
Merge NASA POWER data with monsoon labels, add feature engineering
columns (lag features + cyclical time encoding), and produce the
training-ready CSV.
Saves: data/processed/dataset_final.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

POWER_CSV   = ROOT / 'data' / 'processed' / 'nasa_power_all_stations.csv'
MONSOON_CSV = ROOT / 'data' / 'processed' / 'monsoon_phases.csv'
OUTPUT_PATH = ROOT / 'data' / 'processed' / 'dataset_final.csv'

FINAL_COLUMNS = [
    'date', 'station', 'ghi', 'temperature', 'humidity',
    'wind_speed', 'cloud_cover', 'monsoon_phase', 'phase_name',
    'day_of_year', 'month', 'year',
    'ghi_lag1', 'ghi_lag7', 'ghi_rolling7',
    'doy_sin', 'doy_cos', 'month_sin', 'month_cos',
]


# ── Feature engineering ────────────────────────────────────────────────────────

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add GHI lag and rolling-mean features per station.

    Lags are computed independently per station so that the first
    day of one station never borrows from the last day of another.

    Args:
        df: DataFrame sorted by station then date with a 'ghi' column.

    Returns:
        DataFrame with three new columns: ghi_lag1, ghi_lag7,
        ghi_rolling7.
    """
    df = df.copy()
    df['ghi_lag1']     = df.groupby('station')['ghi'].transform(lambda s: s.shift(1))
    df['ghi_lag7']     = df.groupby('station')['ghi'].transform(lambda s: s.shift(7))
    df['ghi_rolling7'] = df.groupby('station')['ghi'].transform(
        lambda s: s.rolling(7, min_periods=1).mean()
    )
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar and cyclical time features.

    Cyclical encoding follows the exact formula in AGENTS.md:
        doy_sin/cos  — N=366
        month_sin/cos — N=12

    Args:
        df: DataFrame with a datetime 'date' column.

    Returns:
        DataFrame with day_of_year, month, year, doy_sin, doy_cos,
        month_sin, month_cos columns added.
    """
    df = df.copy()
    df['day_of_year'] = df['date'].dt.day_of_year.astype(int)
    df['month']       = df['date'].dt.month.astype(int)
    df['year']        = df['date'].dt.year.astype(int)

    df['doy_sin']   = np.sin(2 * np.pi * df['day_of_year'] / 366)
    df['doy_cos']   = np.cos(2 * np.pi * df['day_of_year'] / 366)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    return df


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """Load, merge, engineer features, and save final dataset."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("merge_dataset.py — Build Training Dataset")
    print(f"{'='*60}\n")

    # 1. Load inputs
    if not POWER_CSV.exists():
        raise FileNotFoundError(
            f"NASA POWER CSV not found: {POWER_CSV}\n"
            "Run scripts/fetch_nasa_power.py first."
        )
    if not MONSOON_CSV.exists():
        raise FileNotFoundError(
            f"Monsoon labels CSV not found: {MONSOON_CSV}\n"
            "Run scripts/monsoon_labels.py first."
        )

    print(f"Loading NASA POWER data from  : {POWER_CSV}")
    power = pd.read_csv(POWER_CSV, parse_dates=['date'])
    print(f"  Rows loaded: {len(power):,}")

    print(f"Loading monsoon labels from   : {MONSOON_CSV}")
    monsoon = pd.read_csv(MONSOON_CSV, parse_dates=['date'])
    print(f"  Rows loaded: {len(monsoon):,}")

    # 2. Merge — left join so all POWER rows are kept
    print("\nMerging on (date, station) — left join ...")
    df = pd.merge(power, monsoon, on=['date', 'station'], how='left')
    print(f"  Rows after merge: {len(df):,}")
    if len(df) != len(power):
        print(
            f"  WARNING: merge produced {len(df)} rows but POWER had {len(power)} rows. "
            "Check for duplicate (date, station) keys in either source."
        )

    # 3. Sort
    df = df.sort_values(['station', 'date']).reset_index(drop=True)

    # 4. Lag + rolling features (per station)
    print("Adding lag features (ghi_lag1, ghi_lag7, ghi_rolling7) ...")
    df = add_lag_features(df)

    # 5. Time features
    print("Adding time features (day_of_year, month, year, cyclical) ...")
    df = add_time_features(df)

    # 6. Drop rows where ghi is NaN
    rows_before = len(df)
    nan_per_station = df[df['ghi'].isna()].groupby('station').size()
    print(f"\nRows before GHI NaN drop: {rows_before:,}")
    if len(nan_per_station):
        print("GHI NaN rows per station:")
        print(nan_per_station.to_string())
    else:
        print("  No NaN ghi rows found.")
    df = df.dropna(subset=['ghi']).reset_index(drop=True)
    rows_after = len(df)
    print(f"Rows after  GHI NaN drop: {rows_after:,}  (dropped {rows_before - rows_after})")

    # 7. Reorder columns to contract
    df = df[FINAL_COLUMNS]

    # 8. Save
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved → {OUTPUT_PATH}")

    # ── SELF CHECK ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SELF CHECK")
    print(f"{'='*60}")

    print(f"\nTotal rows      : {len(df):,}")
    print(f"Unique stations : {df['station'].nunique()}")
    print(f"Date range      : {df['date'].min().date()} → {df['date'].max().date()}")

    print("\nMissing values per column:")
    missing = df.isna().sum()
    print(missing[missing > 0].to_string() if missing.any() else "  None")

    print("\nGHI stats per station (mean / std / min / max):")
    ghi_stats = df.groupby('station')['ghi'].agg(['mean', 'std', 'min', 'max']).round(3)
    print(ghi_stats.to_string())

    print("\nPhase distribution across full dataset:")
    phase_dist = df.groupby('monsoon_phase').size()
    for p, cnt in phase_dist.items():
        print(f"  Phase {p} ({['pre_monsoon','active_monsoon','post_monsoon'][p]}): {cnt:,} rows")

    assert df['ghi'].isna().sum() == 0, "FAIL: NaN values remain in ghi column!"
    print("\nConfirm: no NaN in ghi column — PASS")

    print(f"\n{'='*60}")
    print("Script 3 complete.")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
