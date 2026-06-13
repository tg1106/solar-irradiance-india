"""
forecast_generation.py
Phase B — Script 11.
Use the already-trained SolarTransformer (transformer_best.pt) to
forecast next-day GHI on the 2024 test set, convert forecasted and
actual GHI/wind speed to Solar RGI / Wind RGI, and compute the hybrid
combined RGI using each station's optimal solar:wind mix from
optimize_mix.py.

Wind speed is forecast via persistence (wind_speed_50m.shift(1)) —
wind forecasting is out of scope for this pipeline; noted as a
limitation in the paper.

Inputs:  models/transformer_best.pt
         models/scaler_transformer.pkl
         data/processed/dataset_final.csv
         data/processed/merra2_aod_stations.csv
         data/processed/rgi_dataset.csv
         outputs/optimal_mix.csv
Outputs: outputs/generation_forecasts.csv
         outputs/generation_trends.png
         outputs/metrics_generation.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# ── Path anchors ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))

from transformer_monsoon import (  # noqa: E402
    AOD_VARS, FEATURE_COLS, FILL_THRESHOLD, MONSOON_COL, SEQ_LEN, STATIONS_LIST,
    TARGET_COL, TEST_END, TEST_START, TRAIN_END, TRAIN_START, VAL_END, VAL_START,
    SolarTransformer, D_MODEL, N_HEADS, N_LAYERS, DIM_FF, DROPOUT, INPUT_SIZE,
)

DATASET_CSV  = ROOT / 'data'    / 'processed' / 'dataset_final.csv'
AOD_CSV      = ROOT / 'data'    / 'processed' / 'merra2_aod_stations.csv'
RGI_CSV      = ROOT / 'data'    / 'processed' / 'rgi_dataset.csv'
OPTIMAL_CSV  = ROOT / 'outputs' / 'optimal_mix.csv'
MODEL_PATH   = ROOT / 'models'  / 'transformer_best.pt'
SCALER_PATH  = ROOT / 'models'  / 'scaler_transformer.pkl'

FORECASTS_CSV = ROOT / 'outputs' / 'generation_forecasts.csv'
TRENDS_PNG    = ROOT / 'outputs' / 'generation_trends.png'
METRICS_JSON  = ROOT / 'outputs' / 'metrics_generation.json'

PLOT_STATIONS = ['Jodhpur', 'Chennai', 'Kolkata', 'Bengaluru']

# ── Physics constants — AGENTS.md, do not change (mirrors compute_rgi.py) ───────
PANEL_EFFICIENCY = 0.18
TEMP_COEFFICIENT = 0.004
T_REF            = 25.0
GHI_MAX          = 8.0

R_GAS    = 287.05
CUT_IN   = 3.0
RATED    = 12.0
CUT_OUT  = 25.0
CP       = 0.4

# ── Device selection — AGENTS.md ─────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')
print(f"Using device: {DEVICE}")


# ── Physics functions — mirrors compute_rgi.py ───────────────────────────────────

def solar_rgi(ghi: float, temperature: float) -> float:
    """Convert GHI to Solar RGI [0, 1]. See compute_rgi.py for derivation."""
    efficiency = PANEL_EFFICIENCY * (1 - TEMP_COEFFICIENT * (temperature - T_REF))
    raw = efficiency * ghi / (PANEL_EFFICIENCY * GHI_MAX)
    return float(np.clip(raw, 0, 1))


def air_density(pressure_kpa: float, temperature_c: float) -> float:
    """Compute air density from surface pressure and temperature. See compute_rgi.py."""
    P = pressure_kpa * 1000
    T = temperature_c + 273.15
    return P / (R_GAS * T)


def wind_rgi(wind_speed_50m: float, pressure_kpa: float, temperature_c: float) -> float:
    """Convert wind speed at 50m to Wind RGI [0, 1]. See compute_rgi.py."""
    v = wind_speed_50m
    rho = air_density(pressure_kpa, temperature_c)
    rho_std = 1.225

    if v < CUT_IN or v > CUT_OUT:
        return 0.0
    if v >= RATED:
        power_ratio = 1.0
    else:
        power_ratio = (rho / rho_std) * (v ** 3) / (RATED ** 3)
    return float(np.clip(power_ratio, 0, 1))


# ── Data preparation — mirrors transformer_monsoon.py Steps 1-5 ─────────────────

def build_scaled_test_split() -> tuple[pd.DataFrame, pd.DataFrame, object]:
    """Rebuild the scaled+raw 2024 test split exactly as in transformer_monsoon.py.

    Returns:
        Tuple of (df_test_scaled, df_test_raw, scaler). df_test_raw retains
        the unscaled 'ghi' and 'temperature' columns aligned by (date, station).
    """
    df  = pd.read_csv(DATASET_CSV, parse_dates=['date'])
    aod = pd.read_csv(AOD_CSV,     parse_dates=['date'])

    aod_dfs: list[pd.DataFrame] = []
    for station in STATIONS_LIST:
        rename    = {f"{station}_{var}": var for var in AOD_VARS}
        available = {k: v for k, v in rename.items() if k in aod.columns}
        if not available:
            continue
        sdf = aod[['date'] + list(available.keys())].rename(columns=available).copy()
        sdf['station'] = station
        for var in AOD_VARS:
            if var in sdf.columns:
                sdf[var] = sdf[var].where(sdf[var] < FILL_THRESHOLD, other=float('nan'))
        aod_dfs.append(sdf)

    aod_long = pd.concat(aod_dfs, ignore_index=True)
    df = pd.merge(df, aod_long, on=['date', 'station'], how='left')

    for var in AOD_VARS:
        if var not in df.columns:
            continue
        df[var] = df.groupby('station')[var].transform(lambda s: s.fillna(s.mean()))
        df[var] = df.groupby('station')[var].transform(lambda s: s.ffill().bfill())

    df = df.sort_values(['station', 'date']).reset_index(drop=True)
    df[TARGET_COL] = df.groupby('station')['ghi'].transform(lambda s: s.shift(-1))

    test_mask = (df['date'] >= TEST_START) & (df['date'] <= TEST_END)
    df_test_raw = df[test_mask].copy().reset_index(drop=True)

    all_needed = FEATURE_COLS + [MONSOON_COL, TARGET_COL]
    df_test_raw.dropna(subset=all_needed, inplace=True)
    df_test_raw.reset_index(drop=True, inplace=True)

    scaler = joblib.load(SCALER_PATH)
    df_test_scaled = df_test_raw.copy()
    df_test_scaled[FEATURE_COLS] = scaler.transform(df_test_scaled[FEATURE_COLS])

    ghi_idx   = FEATURE_COLS.index('ghi')
    ghi_min   = float(scaler.data_min_[ghi_idx])
    ghi_range = float(scaler.data_max_[ghi_idx]) - ghi_min
    df_test_scaled[TARGET_COL] = (
        (df_test_scaled[TARGET_COL] - ghi_min) / ghi_range if ghi_range > 0
        else df_test_scaled[TARGET_COL]
    )

    return df_test_scaled, df_test_raw, scaler


def forecast_ghi_per_station(
    df_test_scaled: pd.DataFrame, df_test_raw: pd.DataFrame, scaler,
) -> pd.DataFrame:
    """Run the trained SolarTransformer on the 2024 test set (CPU, MPS fix).

    Args:
        df_test_scaled: Scaled test DataFrame (FEATURE_COLS scaled to [0,1]).
        df_test_raw:    Unscaled test DataFrame (same row order per station).
        scaler:         Fitted MinMaxScaler used to inverse-transform GHI.

    Returns:
        DataFrame with columns [date, station, actual_ghi, forecasted_ghi,
        temperature].
    """
    model = SolarTransformer(
        input_size=INPUT_SIZE, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, dim_ff=DIM_FF, dropout=DROPOUT,
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    eval_device = torch.device('cpu')
    model = model.to(eval_device)

    ghi_idx = FEATURE_COLS.index('ghi')
    rows: list[dict] = []

    for station, grp_scaled in df_test_scaled.groupby('station', sort=True):
        grp_scaled = grp_scaled.sort_values('date').reset_index(drop=True)
        grp_raw    = (
            df_test_raw[df_test_raw['station'] == station]
            .sort_values('date').reset_index(drop=True)
        )

        feats    = grp_scaled[FEATURE_COLS].to_numpy(dtype=np.float32)
        moon_arr = grp_scaled[MONSOON_COL].fillna(0).astype('int64').to_numpy()
        dates    = grp_scaled['date'].to_numpy()
        n        = len(grp_scaled)

        with torch.no_grad():
            for i in range(n - SEQ_LEN):
                x_feat = feats[i: i + SEQ_LEN]
                x_moon = moon_arr[i: i + SEQ_LEN]
                if np.isnan(x_feat).any():
                    continue

                xf_t = torch.tensor(x_feat, dtype=torch.float32).unsqueeze(0).to(eval_device)
                xm_t = torch.tensor(x_moon, dtype=torch.long).unsqueeze(0).to(eval_device)
                yp   = model(xf_t, xm_t).numpy().flatten()[0]

                dummy = np.zeros((1, len(FEATURE_COLS)), dtype=np.float32)
                dummy[0, ghi_idx] = yp
                forecasted_ghi = float(scaler.inverse_transform(dummy)[0, ghi_idx])

                target_idx  = i + SEQ_LEN - 1
                target_date = pd.Timestamp(dates[target_idx])
                actual_ghi  = float(grp_raw.loc[target_idx, TARGET_COL])
                temperature = float(grp_raw.loc[target_idx, 'temperature'])

                rows.append({
                    'date': target_date,
                    'station': station,
                    'actual_ghi': actual_ghi,
                    'forecasted_ghi': forecasted_ghi,
                    'temperature': temperature,
                })

    model.to(DEVICE)
    return pd.DataFrame(rows)


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_generation_trends(df: pd.DataFrame, optimal_mix: pd.DataFrame) -> None:
    """Save a 2x2 grid of actual vs forecasted combined RGI for PLOT_STATIONS.

    Args:
        df: generation_forecasts DataFrame (one row per test-set window).
        optimal_mix: optimal_mix DataFrame (station, optimal_solar_pct).
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=False)
    by_station = optimal_mix.set_index('station')

    for ax, station in zip(axes.flat, PLOT_STATIONS):
        sub = df[df['station'] == station].sort_values('date')
        ax.plot(sub['date'], sub['actual_combined_rgi'], color='steelblue',
                linestyle='-', linewidth=1.2, label='Actual Combined RGI')
        ax.plot(sub['date'], sub['forecasted_combined_rgi'], color='darkorange',
                linestyle='--', linewidth=1.2, label='Forecasted Combined RGI')

        opt_pct = by_station.loc[station, 'optimal_solar_pct'] * 100
        ax.text(
            0.02, 0.95, f"Optimal mix: {opt_pct:.0f}% solar / {100-opt_pct:.0f}% wind",
            transform=ax.transAxes, va='top', ha='left', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        )
        ax.set_title(station)
        ax.set_ylabel('Combined RGI')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Hybrid Solar-Wind Generation Index — Test Year 2024', fontsize=14)
    plt.tight_layout()
    plt.savefig(TRENDS_PNG, dpi=150)
    plt.close(fig)
    print(f"  Saved → {TRENDS_PNG}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    """Forecast hybrid generation index for the 2024 test year per station."""
    print(f"\n{'='*60}")
    print("forecast_generation.py — Phase B Step 4: Generation forecasts")
    print(f"{'='*60}\n")

    for path, desc in [
        (MODEL_PATH,  'transformer_best.pt'),
        (SCALER_PATH, 'scaler_transformer.pkl'),
        (RGI_CSV,     'rgi_dataset.csv (run compute_rgi.py)'),
        (OPTIMAL_CSV, 'optimal_mix.csv (run optimize_mix.py)'),
    ]:
        if not path.exists():
            print(f"ERROR: {desc} not found: {path}")
            sys.exit(1)

    print("Step 1 — Rebuilding scaled 2024 test split ...")
    df_test_scaled, df_test_raw, scaler = build_scaled_test_split()
    print(f"  Test rows (post-dropna): {len(df_test_raw):,}")

    print("\nStep 2 — Forecasting GHI with SolarTransformer (CPU, MPS fix) ...")
    fc = forecast_ghi_per_station(df_test_scaled, df_test_raw, scaler)
    print(f"  Forecast windows: {len(fc):,}")

    print("\nStep 3 — Loading rgi_dataset.csv and optimal_mix.csv ...")
    rgi = pd.read_csv(RGI_CSV, parse_dates=['date'])
    optimal_mix = pd.read_csv(OPTIMAL_CSV)

    print("\nStep 4 — Computing wind_speed_50m persistence (lag-1) ...")
    rgi = rgi.sort_values(['station', 'date']).reset_index(drop=True)
    rgi['wind_speed_50m_lag1'] = rgi.groupby('station')['wind_speed_50m'].shift(1)

    print("\nStep 5 — Merging forecasts with RGI dataset ...")
    merged = pd.merge(
        fc,
        rgi[[
            'date', 'station', 'surface_pressure', 'wind_speed_50m',
            'wind_speed_50m_lag1', 'solar_rgi', 'wind_rgi', 'monsoon_phase',
        ]],
        on=['date', 'station'], how='inner',
    )
    merged.rename(columns={'solar_rgi': 'actual_solar_rgi', 'wind_rgi': 'actual_wind_rgi'}, inplace=True)
    print(f"  Merged rows: {len(merged):,}")
    merged = merged.dropna(subset=['wind_speed_50m_lag1']).reset_index(drop=True)
    print(f"  Rows after dropping missing lag-1 wind: {len(merged):,}")

    print("\nStep 6 — Converting forecasted GHI and persistence wind to RGI ...")
    merged['forecasted_solar_rgi'] = merged.apply(
        lambda r: solar_rgi(r['forecasted_ghi'], r['temperature']), axis=1
    )
    merged['wind_rgi_persistence'] = merged.apply(
        lambda r: wind_rgi(r['wind_speed_50m_lag1'], r['surface_pressure'], r['temperature']), axis=1
    )

    print("\nStep 7 — Combining with each station's optimal mix ratio ...")
    pct_map = optimal_mix.set_index('station')['optimal_solar_pct'].to_dict()
    merged['optimal_solar_pct'] = merged['station'].map(pct_map)
    merged['actual_combined_rgi'] = (
        merged['optimal_solar_pct'] * merged['actual_solar_rgi']
        + (1 - merged['optimal_solar_pct']) * merged['actual_wind_rgi']
    )
    merged['forecasted_combined_rgi'] = (
        merged['optimal_solar_pct'] * merged['forecasted_solar_rgi']
        + (1 - merged['optimal_solar_pct']) * merged['wind_rgi_persistence']
    )

    out_cols = [
        'date', 'station',
        'actual_solar_rgi', 'forecasted_solar_rgi',
        'actual_wind_rgi', 'wind_rgi_persistence',
        'optimal_solar_pct',
        'actual_combined_rgi', 'forecasted_combined_rgi',
    ]
    out_df = merged[out_cols].sort_values(['station', 'date']).reset_index(drop=True)

    FORECASTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(FORECASTS_CSV, index=False)
    print(f"\nSaved → {FORECASTS_CSV}")

    print("\nStep 8 — Per-station MAE / RMSE on combined RGI ...")
    metrics: dict[str, dict[str, float]] = {}
    for station, grp in out_df.groupby('station', sort=True):
        err = grp['actual_combined_rgi'] - grp['forecasted_combined_rgi']
        mae  = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        metrics[station] = {'mae': mae, 'rmse': rmse}

    print(f"\n  {'Station':<12} {'MAE':>8} {'RMSE':>8}")
    print('  ' + '-' * 30)
    for s, m in sorted(metrics.items()):
        print(f"  {s:<12} {m['mae']:>8.4f} {m['rmse']:>8.4f}")

    METRICS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_JSON, 'w') as fh:
        json.dump(metrics, fh, indent=2)
    print(f"\nSaved → {METRICS_JSON}")

    print("\nStep 9 — Generation trends plot ...")
    plot_generation_trends(out_df, optimal_mix)

    # ── SELF CHECK ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SELF CHECK")
    print(f"{'='*60}")

    combined_range_ok = bool(
        out_df['actual_combined_rgi'].between(0, 1).all()
        and out_df['forecasted_combined_rgi'].between(0, 1).all()
    )
    print(f"\nCombined RGI values in [0,1]: {combined_range_ok}")
    assert combined_range_ok, "Combined RGI out of [0,1] range!"

    jodhpur = out_df[out_df['station'] == 'Jodhpur'].copy()
    jodhpur['month'] = jodhpur['date'].dt.month
    winter = jodhpur[jodhpur['month'].isin([11, 12, 1, 2])]['actual_combined_rgi'].mean()
    summer = jodhpur[~jodhpur['month'].isin([11, 12, 1, 2])]['actual_combined_rgi'].mean()
    print(f"\nJodhpur combined RGI — winter (clear sky) mean: {winter:.4f}, "
          f"rest-of-year mean: {summer:.4f}  (winter higher: {winter > summer})")

    chennai_std = out_df[out_df['station'] == 'Chennai']['actual_combined_rgi'].std()
    jodhpur_std = out_df[out_df['station'] == 'Jodhpur']['actual_combined_rgi'].std()
    print(f"\nChennai combined RGI std: {chennai_std:.4f}, "
          f"Jodhpur combined RGI std: {jodhpur_std:.4f} "
          f"(Chennai more stable: {chennai_std < jodhpur_std})")

    print("\nPer-station combined RGI mean on test set:")
    for s in sorted(out_df['station'].unique()):
        mean_actual = out_df[out_df['station'] == s]['actual_combined_rgi'].mean()
        print(f"  {s:<12} {mean_actual:.4f}")

    print("\nPhase B complete.")

    print(f"\n{'='*60}")
    print("Script 11 complete.")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
