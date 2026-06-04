"""
baseline.py
Persistence baseline: predict tomorrow = today (ghi_lag1).
Establishes the performance floor that every model must beat.
Saves: outputs/metrics_persistence.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

DATASET_CSV = ROOT / 'data' / 'processed' / 'dataset_final.csv'
OUTPUT_PATH = ROOT / 'outputs' / 'metrics_persistence.json'

TRAIN_START = "2017-04-01"
TRAIN_END   = "2022-12-31"
VAL_START   = "2023-01-01"
VAL_END     = "2023-12-31"
TEST_START  = "2024-01-01"
TEST_END    = "2024-12-31"

SPLITS: dict[str, tuple[str, str]] = {
    'train': (TRAIN_START, TRAIN_END),
    'val':   (VAL_START,   VAL_END),
    'test':  (TEST_START,  TEST_END),
}


# ── Metric helpers ─────────────────────────────────────────────────────────────

def compute_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    clim_rmse: float,
) -> dict[str, float]:
    """Compute MAE, RMSE, R², and climatological skill score.

    Args:
        actual:    Ground-truth GHI values (no NaNs).
        predicted: Predicted GHI values (no NaNs).
        clim_rmse: RMSE of predicting the training climatology mean
                   for every day (used for skill score denominator).

    Returns:
        Dict with keys: mae, rmse, r2, skill.
    """
    errors   = actual - predicted
    mae      = float(np.mean(np.abs(errors)))
    rmse     = float(np.sqrt(np.mean(errors ** 2)))
    ss_res   = float(np.sum(errors ** 2))
    ss_tot   = float(np.sum((actual - actual.mean()) ** 2))
    r2       = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    skill    = 1.0 - (rmse / clim_rmse) if clim_rmse > 0 else float('nan')
    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'skill': skill}


def print_metrics_table(
    station_metrics: dict[str, dict[str, float]],
    split_name: str,
) -> None:
    """Print a formatted metrics table for one data split.

    Args:
        station_metrics: Nested dict keyed by station name.
        split_name: Label for the split ('train', 'val', or 'test').
    """
    header = f"{'Station':<12} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'Skill':>8}"
    print(f"\n--- Persistence Baseline: {split_name.upper()} ---")
    print(header)
    print('-' * len(header))
    for station, m in sorted(station_metrics.items()):
        print(
            f"{station:<12} "
            f"{m['mae']:>8.4f} "
            f"{m['rmse']:>8.4f} "
            f"{m['r2']:>8.4f} "
            f"{m['skill']:>8.4f}"
        )


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """Compute persistence metrics for all splits and stations."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("baseline.py — Persistence Baseline Evaluation")
    print(f"{'='*60}\n")

    if not DATASET_CSV.exists():
        raise FileNotFoundError(
            f"Final dataset not found: {DATASET_CSV}\n"
            "Run scripts/merge_dataset.py first."
        )
    df = pd.read_csv(DATASET_CSV, parse_dates=['date'])
    print(f"Dataset loaded: {len(df):,} rows")

    # Compute train mean GHI per station — used for climatological RMSE
    train_mask = (df['date'] >= TRAIN_START) & (df['date'] <= TRAIN_END)
    train_clim: dict[str, float] = (
        df[train_mask].groupby('station')['ghi'].mean().to_dict()
    )

    all_metrics: dict[str, Any] = {}

    for split_name, (start, end) in SPLITS.items():
        mask = (df['date'] >= start) & (df['date'] <= end)
        split_df = df[mask].copy()
        print(f"\nSplit '{split_name}': {len(split_df):,} rows  ({start} → {end})")

        station_metrics: dict[str, dict[str, float]] = {}

        for station in sorted(split_df['station'].unique()):
            sdf = split_df[split_df['station'] == station][['ghi', 'ghi_lag1']].dropna()
            if sdf.empty:
                print(f"  WARNING: no valid rows for {station} in {split_name}")
                continue

            actual    = sdf['ghi'].to_numpy()
            predicted = sdf['ghi_lag1'].to_numpy()

            # Climatological RMSE: predict train mean GHI every day
            clim_mean = train_clim.get(station, actual.mean())
            clim_pred = np.full_like(actual, clim_mean)
            clim_rmse = float(np.sqrt(np.mean((actual - clim_pred) ** 2)))

            metrics = compute_metrics(actual, predicted, clim_rmse)
            station_metrics[station] = metrics

        # Overall metrics (all stations pooled)
        all_actual    = split_df.dropna(subset=['ghi', 'ghi_lag1'])['ghi'].to_numpy()
        all_predicted = split_df.dropna(subset=['ghi', 'ghi_lag1'])['ghi_lag1'].to_numpy()
        clim_mean_all = np.mean([train_clim[s] for s in train_clim])
        clim_rmse_all = float(np.sqrt(np.mean((all_actual - clim_mean_all) ** 2)))
        overall = compute_metrics(all_actual, all_predicted, clim_rmse_all)
        station_metrics['OVERALL'] = overall

        print_metrics_table(station_metrics, split_name)
        all_metrics[split_name] = station_metrics

    # Save JSON
    with open(OUTPUT_PATH, 'w') as fh:
        json.dump(all_metrics, fh, indent=2)
    print(f"\nSaved → {OUTPUT_PATH}")

    # ── SELF CHECK ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SELF CHECK")
    print(f"{'='*60}")

    test_metrics = all_metrics.get('test', {})
    print("\nRMSE values on TEST set (all 8 stations):")
    for station, m in sorted(test_metrics.items()):
        if station == 'OVERALL':
            continue
        print(f"  {station:<12}: RMSE = {m['rmse']:.4f}")

    print(f"\n{'─'*50}")
    print("TARGET — your LSTM must beat these RMSE values:")
    print(f"{'─'*50}")
    for station, m in sorted(test_metrics.items()):
        if station == 'OVERALL':
            continue
        print(f"  {station:<12}: {m['rmse']:.4f} MJ/m²/day")

    print(f"\n{'='*60}")
    print("Script 4 complete.")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
