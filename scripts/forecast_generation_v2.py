"""
forecast_generation_v2.py
Phase B — Script 13.
Updated hybrid generation pipeline that replaces the naive wind persistence
forecast with a trained WindTransformer, mirroring the SolarTransformer
already used for the GHI leg.

Changes from forecast_generation.py (Script 11):
  1. WindTransformer class added (self-contained, no cross-file import).
  2. Wind leg now uses wind_transformer_best.pt + scaler_wind.pkl for
     actual next-day wind speed prediction (not wind_speed_50m.shift(1)).
  3. wind_rgi_persistence renamed to wind_rgi_forecasted throughout.
  4. Generation trends plot title updated to reflect both legs forecasted.
  5. Outputs saved as _v2 variants; v1 vs v2 RMSE comparison printed.

Inputs:  models/transformer_best.pt       (solar GHI forecaster)
         models/scaler_transformer.pkl
         models/wind_transformer_best.pt  (wind speed forecaster — NEW)
         models/scaler_wind.pkl           (NEW)
         data/processed/dataset_final.csv
         data/processed/merra2_aod_stations.csv
         data/processed/nasa_power_extended.csv
         data/processed/monsoon_phases.csv
         data/processed/rgi_dataset.csv
         outputs/optimal_mix.csv
         outputs/metrics_generation.json   (v1 baseline for comparison)
Outputs: outputs/generation_forecasts_v2.csv
         outputs/generation_trends_v2.png
         outputs/metrics_generation_v2.json
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

# Solar inputs (same as forecast_generation.py)
DATASET_CSV  = ROOT / 'data'    / 'processed' / 'dataset_final.csv'
AOD_CSV      = ROOT / 'data'    / 'processed' / 'merra2_aod_stations.csv'
MODEL_PATH   = ROOT / 'models'  / 'transformer_best.pt'
SCALER_PATH  = ROOT / 'models'  / 'scaler_transformer.pkl'

# Wind inputs (new for v2)
EXTENDED_CSV       = ROOT / 'data'   / 'processed' / 'nasa_power_extended.csv'
MONSOON_CSV        = ROOT / 'data'   / 'processed' / 'monsoon_phases.csv'
WIND_MODEL_PATH    = ROOT / 'models' / 'wind_transformer_best.pt'
WIND_SCALER_PATH   = ROOT / 'models' / 'scaler_wind.pkl'

# Shared inputs
RGI_CSV      = ROOT / 'data'    / 'processed' / 'rgi_dataset.csv'
OPTIMAL_CSV  = ROOT / 'outputs' / 'optimal_mix.csv'
V1_METRICS   = ROOT / 'outputs' / 'metrics_generation.json'

# v2 outputs
FORECASTS_V2_CSV = ROOT / 'outputs' / 'generation_forecasts_v2.csv'
TRENDS_V2_PNG    = ROOT / 'outputs' / 'generation_trends_v2.png'
METRICS_V2_JSON  = ROOT / 'outputs' / 'metrics_generation_v2.json'

# ── Solar Transformer constants (must match transformer_best.pt) ───────────────
STATIONS_LIST: list[str] = [
    'Jodhpur', 'Ahmedabad', 'Mumbai', 'Chennai',
    'Hyderabad', 'Bhopal', 'Kolkata', 'Bengaluru',
]

SOLAR_FEATURE_COLS: list[str] = [
    'ghi', 'temperature', 'humidity', 'wind_speed',
    'cloud_cover',
    'doy_sin', 'doy_cos', 'month_sin', 'month_cos',
    'TOTEXTTAU', 'DUEXTTAU', 'BCEXTTAU', 'SSEXTTAU', 'SUEXTTAU',
]
SOLAR_MONSOON_COL = 'monsoon_phase'
SOLAR_TARGET_COL  = 'ghi_target'

AOD_VARS: list[str] = ['TOTEXTTAU', 'DUEXTTAU', 'BCEXTTAU', 'SSEXTTAU', 'SUEXTTAU']
FILL_THRESHOLD: float = 1e15

SOLAR_INPUT_SIZE = len(SOLAR_FEATURE_COLS)   # 14
SOLAR_D_MODEL    = 64
SOLAR_N_HEADS    = 4
SOLAR_N_LAYERS   = 2
SOLAR_DIM_FF     = 128
SOLAR_DROPOUT    = 0.1
SOLAR_SEQ_LEN    = 7

# ── Wind Transformer constants (must match wind_transformer_best.pt) ───────────
WIND_FEATURE_COLS: list[str] = [
    'wind_speed_50m', 'surface_pressure', 'temperature',
    'doy_sin', 'doy_cos', 'month_sin', 'month_cos',
]
WIND_MONSOON_COL = 'monsoon_phase'
WIND_TARGET_COL  = 'wind_speed_50m_target'

WIND_INPUT_SIZE = len(WIND_FEATURE_COLS)   # 7
WIND_D_MODEL    = 64
WIND_N_HEADS    = 4
WIND_N_LAYERS   = 2
WIND_DIM_FF     = 128
WIND_DROPOUT    = 0.1
WIND_SEQ_LEN    = 7

# ── Split dates ────────────────────────────────────────────────────────────────
TRAIN_START = "2017-04-01"
TRAIN_END   = "2022-12-31"
TEST_START  = "2024-01-01"
TEST_END    = "2024-12-31"

PLOT_STATIONS = ['Jodhpur', 'Chennai', 'Kolkata', 'Bengaluru']

# ── Physics constants — AGENTS.md / matching compute_rgi.py ───────────────────
PANEL_EFFICIENCY = 0.18
TEMP_COEFFICIENT = 0.004
T_REF            = 25.0
GHI_MAX          = 8.0

R_GAS   = 287.05
CUT_IN  = 3.0
RATED   = 7.0      # IEC Class III — matches compute_rgi.py and forecast_generation.py
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
    """Solar GHI forecaster — duplicated from forecast_generation.py.

    Must stay architecturally in sync with transformer_best.pt.
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


# ── WindTransformer (FIX: self-contained, same architecture) ──────────────────

class WindTransformer(nn.Module):
    """Wind speed forecaster — must stay in sync with wind_transformer_best.pt.

    Architecturally identical to SolarTransformer; operates on wind features.
    Self-contained per AGENTS.md fix pattern — no import from wind_transformer.py.
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

    RATED=7 m/s — IEC Class III, consistent with compute_rgi.py.
    """
    v = wind_speed_50m
    if v < CUT_IN or v > CUT_OUT:
        return 0.0
    rho = air_density(pressure_kpa, temperature_c)
    power_ratio = 1.0 if v >= RATED else (rho / 1.225) * (v ** 3) / (RATED ** 3)
    return float(np.clip(power_ratio, 0, 1))


# ── Solar data preparation — identical to forecast_generation.py ──────────────

def build_solar_scaled_test_split():
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
    df[SOLAR_TARGET_COL] = df.groupby('station')['ghi'].transform(lambda s: s.shift(-1))

    test_mask   = (df['date'] >= TEST_START) & (df['date'] <= TEST_END)
    df_test_raw = df[test_mask].copy().reset_index(drop=True)

    all_needed = SOLAR_FEATURE_COLS + [SOLAR_MONSOON_COL, SOLAR_TARGET_COL]
    df_test_raw.dropna(subset=all_needed, inplace=True)
    df_test_raw.reset_index(drop=True, inplace=True)

    scaler = joblib.load(SCALER_PATH)
    df_test_scaled = df_test_raw.copy()
    df_test_scaled[SOLAR_FEATURE_COLS] = scaler.transform(df_test_scaled[SOLAR_FEATURE_COLS])

    ghi_idx   = SOLAR_FEATURE_COLS.index('ghi')
    ghi_min   = float(scaler.data_min_[ghi_idx])
    ghi_range = float(scaler.data_max_[ghi_idx]) - ghi_min
    df_test_scaled[SOLAR_TARGET_COL] = (
        (df_test_scaled[SOLAR_TARGET_COL] - ghi_min) / ghi_range
        if ghi_range > 0 else df_test_scaled[SOLAR_TARGET_COL]
    )

    return df_test_scaled, df_test_raw, scaler


def forecast_ghi_per_station(
    df_test_scaled: pd.DataFrame,
    df_test_raw:    pd.DataFrame,
    scaler,
) -> pd.DataFrame:
    """Run SolarTransformer on 2024 test set (CPU, MPS fix)."""
    model = SolarTransformer(
        input_size=SOLAR_INPUT_SIZE, d_model=SOLAR_D_MODEL,
        n_heads=SOLAR_N_HEADS, n_layers=SOLAR_N_LAYERS,
        dim_ff=SOLAR_DIM_FF, dropout=SOLAR_DROPOUT,
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    eval_device = torch.device('cpu')
    model = model.to(eval_device)

    ghi_idx = SOLAR_FEATURE_COLS.index('ghi')
    rows: list[dict] = []

    for station, grp_scaled in df_test_scaled.groupby('station', sort=True):
        grp_scaled = grp_scaled.sort_values('date').reset_index(drop=True)
        grp_raw    = (df_test_raw[df_test_raw['station'] == station]
                      .sort_values('date').reset_index(drop=True))

        feats    = grp_scaled[SOLAR_FEATURE_COLS].to_numpy(dtype=np.float32)
        moon_arr = grp_scaled[SOLAR_MONSOON_COL].fillna(0).astype('int64').to_numpy()
        dates    = grp_scaled['date'].to_numpy()
        n        = len(grp_scaled)

        with torch.no_grad():
            for i in range(n - SOLAR_SEQ_LEN):
                x_feat = feats[i: i + SOLAR_SEQ_LEN]
                x_moon = moon_arr[i: i + SOLAR_SEQ_LEN]
                if np.isnan(x_feat).any():
                    continue
                xf_t = torch.tensor(x_feat, dtype=torch.float32).unsqueeze(0).to(eval_device)
                xm_t = torch.tensor(x_moon, dtype=torch.long).unsqueeze(0).to(eval_device)
                yp   = model(xf_t, xm_t).numpy().flatten()[0]

                dummy = np.zeros((1, len(SOLAR_FEATURE_COLS)), dtype=np.float32)
                dummy[0, ghi_idx] = yp
                forecasted_ghi = float(scaler.inverse_transform(dummy)[0, ghi_idx])

                target_idx = i + SOLAR_SEQ_LEN - 1
                rows.append({
                    'date':           pd.Timestamp(dates[target_idx]),
                    'station':        station,
                    'actual_ghi':     float(grp_raw.loc[target_idx, SOLAR_TARGET_COL]),
                    'forecasted_ghi': forecasted_ghi,
                    'temperature':    float(grp_raw.loc[target_idx, 'temperature']),
                })

    return pd.DataFrame(rows)


# ── Wind data preparation and forecasting (new for v2) ────────────────────────

def build_wind_scaled_test_split():
    """Prepare the scaled 2024 test split for wind speed forecasting.

    Mirrors the solar approach: load extended CSV, merge monsoon,
    compute cyclical date features, create shift(-1) target, scale.
    """
    df      = pd.read_csv(EXTENDED_CSV, parse_dates=['date'])
    monsoon = pd.read_csv(MONSOON_CSV,  parse_dates=['date'])

    # Replace NASA POWER fill value
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].replace(-999, np.nan)

    # Merge monsoon phase
    df = pd.merge(
        df,
        monsoon[['date', 'station', 'monsoon_phase']],
        on=['date', 'station'],
        how='left',
    )

    # Cyclical date features
    df['doy']       = df['date'].dt.dayofyear
    df['month']     = df['date'].dt.month
    df['doy_sin']   = np.sin(2 * np.pi * df['doy']   / 365.25)
    df['doy_cos']   = np.cos(2 * np.pi * df['doy']   / 365.25)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)

    df = df.sort_values(['station', 'date']).reset_index(drop=True)
    df[WIND_TARGET_COL] = df.groupby('station')['wind_speed_50m'].transform(
        lambda s: s.shift(-1)
    )

    test_mask   = (df['date'] >= TEST_START) & (df['date'] <= TEST_END)
    df_test_raw = df[test_mask].copy().reset_index(drop=True)

    all_needed = WIND_FEATURE_COLS + [WIND_MONSOON_COL, WIND_TARGET_COL]
    df_test_raw.dropna(subset=all_needed, inplace=True)
    df_test_raw.reset_index(drop=True, inplace=True)

    scaler_wind = joblib.load(WIND_SCALER_PATH)
    df_test_scaled = df_test_raw.copy()
    df_test_scaled[WIND_FEATURE_COLS] = scaler_wind.transform(
        df_test_scaled[WIND_FEATURE_COLS]
    )

    wind_idx   = WIND_FEATURE_COLS.index('wind_speed_50m')
    wind_min   = float(scaler_wind.data_min_[wind_idx])
    wind_range = float(scaler_wind.data_max_[wind_idx]) - wind_min
    df_test_scaled[WIND_TARGET_COL] = (
        (df_test_scaled[WIND_TARGET_COL] - wind_min) / wind_range
        if wind_range > 0 else df_test_scaled[WIND_TARGET_COL]
    )

    return df_test_scaled, df_test_raw, scaler_wind


def forecast_wind_per_station(
    df_test_scaled: pd.DataFrame,
    df_test_raw:    pd.DataFrame,
    scaler_wind,
) -> pd.DataFrame:
    """Run WindTransformer on 2024 test set (CPU, MPS fix).

    Returns DataFrame with date, station, forecasted_wind_speed (m/s).
    Date is the last day of each input window (prediction made on that day,
    for the following day — same date attribution as solar leg).
    """
    model = WindTransformer(
        input_size=WIND_INPUT_SIZE, d_model=WIND_D_MODEL,
        n_heads=WIND_N_HEADS, n_layers=WIND_N_LAYERS,
        dim_ff=WIND_DIM_FF, dropout=WIND_DROPOUT,
    )
    model.load_state_dict(torch.load(WIND_MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    eval_device = torch.device('cpu')
    model = model.to(eval_device)

    wind_idx = WIND_FEATURE_COLS.index('wind_speed_50m')
    rows: list[dict] = []

    for station, grp_scaled in df_test_scaled.groupby('station', sort=True):
        grp_scaled = grp_scaled.sort_values('date').reset_index(drop=True)
        grp_raw    = (df_test_raw[df_test_raw['station'] == station]
                      .sort_values('date').reset_index(drop=True))

        feats    = grp_scaled[WIND_FEATURE_COLS].to_numpy(dtype=np.float32)
        moon_arr = grp_scaled[WIND_MONSOON_COL].fillna(0).astype('int64').to_numpy()
        dates    = grp_scaled['date'].to_numpy()
        n        = len(grp_scaled)

        with torch.no_grad():
            for i in range(n - WIND_SEQ_LEN):
                x_feat = feats[i: i + WIND_SEQ_LEN]
                x_moon = moon_arr[i: i + WIND_SEQ_LEN]
                if np.isnan(x_feat).any():
                    continue
                xf_t = torch.tensor(x_feat, dtype=torch.float32).unsqueeze(0).to(eval_device)
                xm_t = torch.tensor(x_moon, dtype=torch.long).unsqueeze(0).to(eval_device)
                yp   = model(xf_t, xm_t).numpy().flatten()[0]

                dummy = np.zeros((1, len(WIND_FEATURE_COLS)), dtype=np.float32)
                dummy[0, wind_idx] = yp
                forecasted_wind_ms = float(scaler_wind.inverse_transform(dummy)[0, wind_idx])

                target_idx = i + WIND_SEQ_LEN - 1
                rows.append({
                    'date':                   pd.Timestamp(dates[target_idx]),
                    'station':                station,
                    'forecasted_wind_speed':  forecasted_wind_ms,
                })

    return pd.DataFrame(rows)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_generation_trends_v2(df: pd.DataFrame, optimal_mix: pd.DataFrame) -> None:
    """Save 2x2 grid of actual vs forecasted combined RGI (v2: both legs forecasted)."""
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

    fig.suptitle(
        'Hybrid Solar-Wind Generation Index — Both Legs Forecasted — 2024',
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(TRENDS_V2_PNG, dpi=150)
    plt.close(fig)
    print(f"  Saved → {TRENDS_V2_PNG}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    """v2 hybrid generation pipeline: SolarTransformer + WindTransformer."""
    print(f"\n{'='*65}")
    print("forecast_generation_v2.py — Script 13: Both legs Transformer-forecasted")
    print(f"Solar model  : {MODEL_PATH.name}")
    print(f"Wind model   : {WIND_MODEL_PATH.name}")
    print(f"RATED wind   : {RATED} m/s (IEC Class III)")
    print(f"{'='*65}\n")

    required = [
        (MODEL_PATH,       'transformer_best.pt — run transformer_monsoon.py first'),
        (SCALER_PATH,      'scaler_transformer.pkl — run transformer_monsoon.py first'),
        (WIND_MODEL_PATH,  'wind_transformer_best.pt — run wind_transformer.py first'),
        (WIND_SCALER_PATH, 'scaler_wind.pkl — run wind_transformer.py first'),
        (RGI_CSV,          'rgi_dataset.csv — run compute_rgi.py first'),
        (OPTIMAL_CSV,      'optimal_mix.csv — run optimize_mix.py first'),
    ]
    for path, desc in required:
        if not path.exists():
            print(f"ERROR: {desc} not found: {path}")
            sys.exit(1)

    # ── Step 1: Build solar scaled test split ─────────────────────────────────
    print("Step 1 — Rebuilding scaled 2024 solar test split ...")
    df_solar_scaled, df_solar_raw, solar_scaler = build_solar_scaled_test_split()
    print(f"  Solar test rows (post-dropna): {len(df_solar_raw):,}")

    # ── Step 2: Forecast GHI with SolarTransformer (CPU, MPS fix) ────────────
    print("\nStep 2 — Forecasting GHI with SolarTransformer (CPU, MPS fix) ...")
    ghi_fc = forecast_ghi_per_station(df_solar_scaled, df_solar_raw, solar_scaler)
    print(f"  GHI forecast windows: {len(ghi_fc):,}")

    # ── Step 3: Build wind scaled test split (new) ────────────────────────────
    print("\nStep 3 — Building scaled 2024 wind test split ...")
    if not EXTENDED_CSV.exists():
        print(f"ERROR: nasa_power_extended.csv not found: {EXTENDED_CSV}")
        sys.exit(1)
    if not MONSOON_CSV.exists():
        print(f"ERROR: monsoon_phases.csv not found: {MONSOON_CSV}")
        sys.exit(1)
    df_wind_scaled, df_wind_raw, wind_scaler = build_wind_scaled_test_split()
    print(f"  Wind test rows (post-dropna): {len(df_wind_raw):,}")

    # ── Step 4: Forecast wind speed with WindTransformer (CPU, MPS fix) ──────
    print("\nStep 4 — Forecasting wind speed with WindTransformer (CPU, MPS fix) ...")
    wind_fc = forecast_wind_per_station(df_wind_scaled, df_wind_raw, wind_scaler)
    print(f"  Wind forecast windows: {len(wind_fc):,}")

    # ── Step 5: Load RGI dataset and optimal mix ──────────────────────────────
    print("\nStep 5 — Loading rgi_dataset.csv and optimal_mix.csv ...")
    rgi         = pd.read_csv(RGI_CSV,     parse_dates=['date'])
    optimal_mix = pd.read_csv(OPTIMAL_CSV)
    print(f"  RGI dataset rows   : {len(rgi):,}")
    print(f"  Optimal mix rows   : {len(optimal_mix)}")

    rgi = rgi.sort_values(['station', 'date']).reset_index(drop=True)

    # ── Step 6: Merge GHI forecasts with RGI and wind forecasts ──────────────
    print("\nStep 6 — Merging GHI forecasts with RGI dataset ...")
    merged = pd.merge(
        ghi_fc,
        rgi[['date', 'station', 'surface_pressure', 'wind_speed_50m',
             'solar_rgi', 'wind_rgi', 'monsoon_phase']],
        on=['date', 'station'], how='inner',
    )
    merged.rename(columns={
        'solar_rgi': 'actual_solar_rgi',
        'wind_rgi':  'actual_wind_rgi',
    }, inplace=True)
    print(f"  Rows after solar+RGI merge: {len(merged):,}")

    print("\nStep 7 — Merging wind Transformer forecasts ...")
    merged = pd.merge(
        merged,
        wind_fc[['date', 'station', 'forecasted_wind_speed']],
        on=['date', 'station'], how='left',
    )
    n_missing_wind = merged['forecasted_wind_speed'].isna().sum()
    if n_missing_wind > 0:
        print(f"  WARNING: {n_missing_wind} rows missing Transformer wind forecast "
              "(falling back to persistence for those rows)")
        # Fallback: use wind_speed_50m (current day) as persistence for missing predictions
        rgi_lag = rgi.sort_values(['station', 'date']).copy()
        rgi_lag['wind_speed_50m_lag1'] = rgi_lag.groupby('station')['wind_speed_50m'].shift(1)
        lag_map = rgi_lag.set_index(['date', 'station'])['wind_speed_50m_lag1'].to_dict()
        merged['_persist_fallback'] = merged.apply(
            lambda r: lag_map.get((r['date'], r['station']), np.nan), axis=1
        )
        merged['forecasted_wind_speed'] = merged['forecasted_wind_speed'].fillna(
            merged['_persist_fallback']
        )
        merged.drop(columns=['_persist_fallback'], inplace=True)

    merged = merged.dropna(subset=['forecasted_wind_speed']).reset_index(drop=True)
    print(f"  Rows after wind merge (post-dropna): {len(merged):,}")

    # ── Step 8: Convert forecasted GHI and wind to RGI ───────────────────────
    print("\nStep 8 — Converting forecasted GHI and Transformer wind to RGI ...")
    merged['forecasted_solar_rgi'] = merged.apply(
        lambda r: solar_rgi(r['forecasted_ghi'], r['temperature']), axis=1
    )
    # NEW: wind_rgi_forecasted (replaces wind_rgi_persistence from v1)
    merged['wind_rgi_forecasted'] = merged.apply(
        lambda r: wind_rgi(
            r['forecasted_wind_speed'],
            r['surface_pressure'],
            r['temperature'],
        ), axis=1
    )

    # ── Step 9: Combine with optimal mix ratio ────────────────────────────────
    print("\nStep 9 — Combining with each station's optimal mix ratio ...")
    pct_map = optimal_mix.set_index('station')['optimal_solar_pct'].to_dict()
    merged['optimal_solar_pct']       = merged['station'].map(pct_map)
    merged['actual_combined_rgi']     = (
        merged['optimal_solar_pct'] * merged['actual_solar_rgi']
        + (1 - merged['optimal_solar_pct']) * merged['actual_wind_rgi']
    )
    merged['forecasted_combined_rgi'] = (
        merged['optimal_solar_pct'] * merged['forecasted_solar_rgi']
        + (1 - merged['optimal_solar_pct']) * merged['wind_rgi_forecasted']
    )

    out_cols = [
        'date', 'station',
        'actual_solar_rgi', 'forecasted_solar_rgi',
        'actual_wind_rgi', 'wind_rgi_forecasted',
        'optimal_solar_pct',
        'actual_combined_rgi', 'forecasted_combined_rgi',
    ]
    out_df = (
        merged[out_cols]
        .sort_values(['station', 'date'])
        .reset_index(drop=True)
    )

    FORECASTS_V2_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(FORECASTS_V2_CSV, index=False)
    print(f"\n  Saved → {FORECASTS_V2_CSV}")

    # ── Step 10: Per-station MAE / RMSE on combined RGI ───────────────────────
    print("\nStep 10 — Per-station MAE / RMSE on combined RGI (v2: wind Transformer) ...")
    v2_metrics: dict[str, dict[str, float]] = {}
    for station, grp in out_df.groupby('station', sort=True):
        err  = grp['actual_combined_rgi'] - grp['forecasted_combined_rgi']
        v2_metrics[station] = {
            'mae':  float(np.mean(np.abs(err))),
            'rmse': float(np.sqrt(np.mean(err ** 2))),
        }

    print(f"\n  {'Station':<12} {'MAE':>8} {'RMSE':>8}")
    print('  ' + '-' * 30)
    for s, m in sorted(v2_metrics.items()):
        print(f"  {s:<12} {m['mae']:>8.4f} {m['rmse']:>8.4f}")

    # Overall RMSE across all stations combined
    all_err = (
        out_df['actual_combined_rgi'] - out_df['forecasted_combined_rgi']
    ).to_numpy()
    v2_overall_rmse = float(np.sqrt(np.mean(all_err ** 2)))
    v2_metrics['OVERALL'] = {
        'mae':  float(np.mean(np.abs(all_err))),
        'rmse': v2_overall_rmse,
    }
    print(f"\n  OVERALL (all stations): MAE={v2_metrics['OVERALL']['mae']:.4f}  "
          f"RMSE={v2_metrics['OVERALL']['rmse']:.4f}")

    METRICS_V2_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_V2_JSON, 'w') as fh:
        json.dump(v2_metrics, fh, indent=2)
    print(f"\n  Saved → {METRICS_V2_JSON}")

    # ── Step 11: Generation trends v2 plot ────────────────────────────────────
    print("\nStep 11 — Generation trends v2 plot ...")
    plot_generation_trends_v2(out_df, optimal_mix)

    # ── Step 12: v1 vs v2 comparison ──────────────────────────────────────────
    print(f"\n{'='*65}")
    print("v1 (wind persistence) vs v2 (wind Transformer) comparison")
    print(f"{'='*65}")

    v1_overall_rmse = float('nan')
    if V1_METRICS.exists():
        with open(V1_METRICS) as fh:
            v1_data = json.load(fh)
        # v1 metrics_generation.json is per-station; compute overall from all errors
        # Try loading OVERALL if present, otherwise compute from per-station average
        if 'OVERALL' in v1_data:
            v1_overall_rmse = float(v1_data['OVERALL']['rmse'])
        else:
            # Load v1 forecasts CSV to compute overall if available
            v1_csv = ROOT / 'outputs' / 'generation_forecasts.csv'
            if v1_csv.exists():
                v1_df = pd.read_csv(v1_csv)
                v1_err = (
                    v1_df['actual_combined_rgi'] - v1_df['forecasted_combined_rgi']
                ).to_numpy()
                v1_overall_rmse = float(np.sqrt(np.mean(v1_err ** 2)))
            else:
                # Fall back to mean of per-station RMSEs
                station_rmses = [
                    m['rmse'] for s, m in v1_data.items() if s != 'OVERALL'
                ]
                if station_rmses:
                    v1_overall_rmse = float(np.mean(station_rmses))
    else:
        print(f"  WARNING: {V1_METRICS.name} not found — cannot compute v1 baseline.")

    improvement = (
        (v1_overall_rmse - v2_overall_rmse) / v1_overall_rmse * 100
        if not np.isnan(v1_overall_rmse) and v1_overall_rmse > 0 else float('nan')
    )

    if not np.isnan(v1_overall_rmse):
        print(f"\nv1 (wind persistence) overall RMSE : {v1_overall_rmse:.6f}")
    else:
        print(f"\nv1 (wind persistence) overall RMSE : N/A (metrics_generation.json not found)")
    print(f"v2 (wind Transformer) overall RMSE : {v2_overall_rmse:.6f}")
    if not np.isnan(improvement):
        print(f"Improvement                         : {improvement:+.2f}%")
    else:
        print(f"Improvement                         : N/A")

    # ── SELF CHECK ────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("SELF CHECK")
    print(f"{'='*65}")

    combined_ok = bool(
        out_df['actual_combined_rgi'].between(0, 1).all()
        and out_df['forecasted_combined_rgi'].between(0, 1).all()
    )
    print(f"\nCombined RGI values in [0,1]: {combined_ok}")
    assert combined_ok, "Combined RGI out of [0,1] range!"

    wind_rgi_fc_ok = bool(out_df['wind_rgi_forecasted'].between(0, 1).all())
    print(f"wind_rgi_forecasted in [0,1] : {wind_rgi_fc_ok}")
    assert wind_rgi_fc_ok, "wind_rgi_forecasted out of [0,1] range!"

    print(f"\nColumn 'wind_rgi_persistence' present: {'wind_rgi_persistence' in out_df.columns}")
    print(f"Column 'wind_rgi_forecasted' present : {'wind_rgi_forecasted' in out_df.columns}")

    print("\nPer-station mean actual combined RGI (v2):")
    for s in sorted(out_df['station'].unique()):
        mean_rgi = out_df[out_df['station'] == s]['actual_combined_rgi'].mean()
        mix_pct  = pct_map.get(s, float('nan')) * 100
        print(f"  {s:<12} mean_RGI={mean_rgi:.4f}  optimal_solar={mix_pct:.0f}%")

    print(f"\nSaved outputs:")
    print(f"  {FORECASTS_V2_CSV}")
    print(f"  {TRENDS_V2_PNG}")
    print(f"  {METRICS_V2_JSON}")

    print(f"\n{'='*65}")
    print("Script 13 complete.")
    print(f"{'='*65}\n")


if __name__ == '__main__':
    main()
