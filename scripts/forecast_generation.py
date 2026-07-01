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

FIX (2026-06):
- Removed fragile sys.path import from transformer_monsoon. All
  constants and the SolarTransformer class are now self-contained.
- RATED wind speed changed from 12 m/s to 7 m/s (IEC Class III).
- FILL_THRESHOLD corrected to 1e15 (MERRA-2 official spec).

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
import math
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ── Path anchors ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

DATASET_CSV  = ROOT / 'data'    / 'processed' / 'dataset_final.csv'
AOD_CSV      = ROOT / 'data'    / 'processed' / 'merra2_aod_stations.csv'
RGI_CSV      = ROOT / 'data'    / 'processed' / 'rgi_dataset.csv'
OPTIMAL_CSV  = ROOT / 'outputs' / 'optimal_mix.csv'
MODEL_PATH   = ROOT / 'models'  / 'transformer_best.pt'
SCALER_PATH  = ROOT / 'models'  / 'scaler_transformer.pkl'

FORECASTS_CSV = ROOT / 'outputs' / 'generation_forecasts.csv'
TRENDS_PNG    = ROOT / 'outputs' / 'generation_trends.png'
METRICS_JSON  = ROOT / 'outputs' / 'metrics_generation.json'

# ── Self-contained constants (FIX: no import from transformer_monsoon) ────────
STATIONS_LIST: list[str] = [
    'Jodhpur', 'Ahmedabad', 'Mumbai', 'Chennai',
    'Hyderabad', 'Bhopal', 'Kolkata', 'Bengaluru',
]

FEATURE_COLS: list[str] = [
    'ghi', 'temperature', 'humidity', 'wind_speed',
    'cloud_cover',
    'doy_sin', 'doy_cos', 'month_sin', 'month_cos',
    'TOTEXTTAU', 'DUEXTTAU', 'BCEXTTAU', 'SSEXTTAU', 'SUEXTTAU',
]
MONSOON_COL = 'monsoon_phase'
TARGET_COL  = 'ghi_target'

AOD_VARS: list[str] = ['TOTEXTTAU', 'DUEXTTAU', 'BCEXTTAU', 'SSEXTTAU', 'SUEXTTAU']
FILL_THRESHOLD: float = 1e15   # FIX: was 1e14, now matches MERRA-2 official spec

TRAIN_START = "2017-04-01"
TRAIN_END   = "2022-12-31"
VAL_START   = "2023-01-01"
VAL_END     = "2023-12-31"
TEST_START  = "2024-01-01"
TEST_END    = "2024-12-31"

# SolarTransformer architecture — must match transformer_best.pt exactly
INPUT_SIZE = len(FEATURE_COLS)  # 14
D_MODEL    = 64
N_HEADS    = 4
N_LAYERS   = 2
DIM_FF     = 128
DROPOUT    = 0.1
SEQ_LEN    = 7

PLOT_STATIONS = ['Jodhpur', 'Chennai', 'Kolkata', 'Bengaluru']

# ── Physics constants ─────────────────────────────────────────────────────────
PANEL_EFFICIENCY = 0.18
TEMP_COEFFICIENT = 0.004
T_REF            = 25.0
GHI_MAX          = 8.0

R_GAS   = 287.05
CUT_IN  = 3.0
# FIX: RATED=7 m/s for IEC Class III low-wind-speed turbines.
# India's mean 50m wind speed ~5 m/s. With RATED=12, Wind RGI ≈ 7%
# everywhere making solar always dominant — optimisation was trivial.
# RATED=7 gives ~36% Wind RGI at 5 m/s — physically meaningful.
RATED   = 7.0
CUT_OUT = 25.0
CP      = 0.4

# ── Device selection ──────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')
print(f"Using device: {DEVICE}")


# ── SolarTransformer (FIX: self-contained, no sys.path import) ───────────────

class SolarTransformer(nn.Module):
    """Transformer forecaster with monsoon embedding.

    Duplicated from transformer_monsoon.py to remove sys.path dependency.
    Architecture must stay in sync with transformer_best.pt.
    """

    def __init__(
        self, input_size: int, d_model: int, n_heads: int,
        n_layers: int, dim_ff: int, dropout: float,
    ) -> None:
        super().__init__()
        self.input_proj        = nn.Linear(input_size, d_model)
        self.monsoon_embedding = nn.Embedding(3, d_model)
        self.register_buffer('pos_enc',
                             self._build_pos_enc(seq_len=100, d_model=d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=dim_ff, dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)
        self.output  = nn.Linear(d_model, 1)

    @staticmethod
    def _build_pos_enc(seq_len: int, d_model: int) -> torch.Tensor:
        pe  = torch.zeros(seq_len, d_model)
        pos = torch.arange(0, seq_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor, monsoon: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x) + self.monsoon_embedding(monsoon)
        x = x + self.pos_enc[:, :x.size(1), :]
        x = self.dropout(x)
        x = self.transformer(x)
        return self.output(x[:, -1, :])


# ── Physics functions ─────────────────────────────────────────────────────────

def solar_rgi(ghi: float, temperature: float) -> float:
    """Convert GHI to Solar RGI [0, 1]."""
    efficiency = PANEL_EFFICIENCY * (1 - TEMP_COEFFICIENT * (temperature - T_REF))
    return float(np.clip(efficiency * ghi / (PANEL_EFFICIENCY * GHI_MAX), 0, 1))


def air_density(pressure_kpa: float, temperature_c: float) -> float:
    """Compute air density (kg/m³) from surface pressure and temperature."""
    return (pressure_kpa * 1000) / (R_GAS * (temperature_c + 273.15))


def wind_rgi(wind_speed_50m: float, pressure_kpa: float, temperature_c: float) -> float:
    """Convert wind speed at 50m to Wind RGI [0, 1].

    RATED=7 m/s appropriate for India's IEC Class III low-wind sites.
    """
    v = wind_speed_50m
    if v < CUT_IN or v > CUT_OUT:
        return 0.0
    rho = air_density(pressure_kpa, temperature_c)
    power_ratio = 1.0 if v >= RATED else (rho / 1.225) * (v ** 3) / (RATED ** 3)
    return float(np.clip(power_ratio, 0, 1))


# ── Data preparation ──────────────────────────────────────────────────────────

def build_scaled_test_split():
    """Rebuild the scaled 2024 test split matching transformer_monsoon.py."""
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

    test_mask   = (df['date'] >= TEST_START) & (df['date'] <= TEST_END)
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
        (df_test_scaled[TARGET_COL] - ghi_min) / ghi_range
        if ghi_range > 0 else df_test_scaled[TARGET_COL]
    )

    return df_test_scaled, df_test_raw, scaler


def forecast_ghi_per_station(df_test_scaled, df_test_raw, scaler) -> pd.DataFrame:
    """Run SolarTransformer on 2024 test set (CPU, MPS fix)."""
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
        grp_raw    = (df_test_raw[df_test_raw['station'] == station]
                      .sort_values('date').reset_index(drop=True))

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
                rows.append({
                    'date':           pd.Timestamp(dates[target_idx]),
                    'station':        station,
                    'actual_ghi':     float(grp_raw.loc[target_idx, TARGET_COL]),
                    'forecasted_ghi': forecasted_ghi,
                    'temperature':    float(grp_raw.loc[target_idx, 'temperature']),
                })

    model.to(DEVICE)
    return pd.DataFrame(rows)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_generation_trends(df: pd.DataFrame, optimal_mix: pd.DataFrame) -> None:
    """Save 2x2 grid of actual vs forecasted combined RGI."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    by_station = optimal_mix.set_index('station')

    for ax, station in zip(axes.flat, PLOT_STATIONS):
        sub = df[df['station'] == station].sort_values('date')
        ax.plot(sub['date'], sub['actual_combined_rgi'],
                color='steelblue', linestyle='-', linewidth=1.2,
                label='Actual Combined RGI')
        ax.plot(sub['date'], sub['forecasted_combined_rgi'],
                color='darkorange', linestyle='--', linewidth=1.2,
                label='Forecasted Combined RGI')
        opt_pct = by_station.loc[station, 'optimal_solar_pct'] * 100
        ax.text(0.02, 0.95,
                f"Optimal mix: {opt_pct:.0f}% solar / {100-opt_pct:.0f}% wind",
                transform=ax.transAxes, va='top', ha='left', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
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
    print(f"RATED wind speed: {RATED} m/s (IEC Class III, India low-wind sites)")
    print(f"{'='*60}\n")

    for path, desc in [
        (MODEL_PATH,  'transformer_best.pt'),
        (SCALER_PATH, 'scaler_transformer.pkl'),
        (RGI_CSV,     'rgi_dataset.csv — run compute_rgi.py first'),
        (OPTIMAL_CSV, 'optimal_mix.csv — run optimize_mix.py first'),
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
        rgi[['date', 'station', 'surface_pressure', 'wind_speed_50m',
             'wind_speed_50m_lag1', 'solar_rgi', 'wind_rgi', 'monsoon_phase']],
        on=['date', 'station'], how='inner',
    )
    merged.rename(columns={'solar_rgi': 'actual_solar_rgi',
                            'wind_rgi':  'actual_wind_rgi'}, inplace=True)
    print(f"  Merged rows: {len(merged):,}")
    merged = merged.dropna(subset=['wind_speed_50m_lag1']).reset_index(drop=True)
    print(f"  Rows after dropping missing lag-1 wind: {len(merged):,}")

    print("\nStep 6 — Converting forecasted GHI and persistence wind to RGI ...")
    merged['forecasted_solar_rgi'] = merged.apply(
        lambda r: solar_rgi(r['forecasted_ghi'], r['temperature']), axis=1
    )
    merged['wind_rgi_persistence'] = merged.apply(
        lambda r: wind_rgi(r['wind_speed_50m_lag1'],
                           r['surface_pressure'], r['temperature']), axis=1
    )

    print("\nStep 7 — Combining with each station's optimal mix ratio ...")
    pct_map = optimal_mix.set_index('station')['optimal_solar_pct'].to_dict()
    merged['optimal_solar_pct']      = merged['station'].map(pct_map)
    merged['actual_combined_rgi']    = (
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
    out_df = (merged[out_cols]
              .sort_values(['station', 'date'])
              .reset_index(drop=True))

    FORECASTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(FORECASTS_CSV, index=False)
    print(f"\nSaved → {FORECASTS_CSV}")

    print("\nStep 8 — Per-station MAE / RMSE on combined RGI ...")
    metrics: dict[str, dict[str, float]] = {}
    for station, grp in out_df.groupby('station', sort=True):
        err  = grp['actual_combined_rgi'] - grp['forecasted_combined_rgi']
        metrics[station] = {
            'mae':  float(np.mean(np.abs(err))),
            'rmse': float(np.sqrt(np.mean(err ** 2))),
        }

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

    # ── SELF CHECK ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SELF CHECK")
    print(f"{'='*60}")

    combined_ok = bool(
        out_df['actual_combined_rgi'].between(0, 1).all()
        and out_df['forecasted_combined_rgi'].between(0, 1).all()
    )
    print(f"\nCombined RGI values in [0,1]: {combined_ok}")
    assert combined_ok, "Combined RGI out of [0,1] range!"

    print("\nPer-station mean actual combined RGI on test set:")
    for s in sorted(out_df['station'].unique()):
        mean_rgi = out_df[out_df['station'] == s]['actual_combined_rgi'].mean()
        mix_pct  = pct_map.get(s, float('nan')) * 100
        print(f"  {s:<12} mean_RGI={mean_rgi:.4f}  optimal_solar={mix_pct:.0f}%")

    print("\nPhase B complete.")
    print(f"\n{'='*60}")
    print("Script 11 complete.")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
