"""
wind_transformer.py
Script 12 — Wind Speed Transformer with Monsoon Embedding.
Symmetry-of-rigor with Phase A solar pipeline: solar got
Persistence→LSTM→LSTM+AOD→Transformer+Monsoon; wind gets
Persistence→Transformer+Monsoon directly (AOD has no physical
relevance to wind — go straight to the best architecture).

Inputs:  data/processed/nasa_power_extended.csv
         data/processed/monsoon_phases.csv
         data/processed/rgi_dataset.csv  (for Wind RGI ground truth)
Outputs: models/wind_transformer_best.pt
         models/scaler_wind.pkl
         outputs/metrics_wind_persistence.json
         outputs/metrics_wind_transformer.json
         outputs/loss_curve_wind.png
         outputs/comparison_wind.png
         outputs/predictions_wind_jodhpur.png
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
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

# ── Path anchors ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

EXTENDED_CSV              = ROOT / 'data'    / 'processed' / 'nasa_power_extended.csv'
MONSOON_CSV               = ROOT / 'data'    / 'processed' / 'monsoon_phases.csv'
RGI_CSV                   = ROOT / 'data'    / 'processed' / 'rgi_dataset.csv'
MODEL_PATH                = ROOT / 'models'  / 'wind_transformer_best.pt'
SCALER_PATH               = ROOT / 'models'  / 'scaler_wind.pkl'
METRICS_PERSISTENCE_JSON  = ROOT / 'outputs' / 'metrics_wind_persistence.json'
METRICS_TRANSFORMER_JSON  = ROOT / 'outputs' / 'metrics_wind_transformer.json'
LOSS_CURVE_PNG            = ROOT / 'outputs' / 'loss_curve_wind.png'
COMPARISON_PNG            = ROOT / 'outputs' / 'comparison_wind.png'
PREDICTIONS_PNG           = ROOT / 'outputs' / 'predictions_wind_jodhpur.png'

# ── Data constants ─────────────────────────────────────────────────────────────
TRAIN_START = "2017-04-01"
TRAIN_END   = "2022-12-31"
VAL_START   = "2023-01-01"
VAL_END     = "2023-12-31"
TEST_START  = "2024-01-01"
TEST_END    = "2024-12-31"

STATIONS_LIST: list[str] = [
    'Jodhpur', 'Ahmedabad', 'Mumbai', 'Chennai',
    'Hyderabad', 'Bhopal', 'Kolkata', 'Bengaluru',
]

# monsoon_phase is EXCLUDED from FEATURE_COLS — handled by Embedding separately
# 7 continuous features total; INPUT_SIZE=7
FEATURE_COLS: list[str] = [
    'wind_speed_50m', 'surface_pressure', 'temperature',
    'doy_sin', 'doy_cos', 'month_sin', 'month_cos',
]
MONSOON_COL = 'monsoon_phase'
TARGET_COL  = 'wind_speed_50m_target'

# ── Model hyperparameters — identical to transformer_monsoon.py ────────────────
INPUT_SIZE  = len(FEATURE_COLS)   # 7
D_MODEL     = 64
N_HEADS     = 4
N_LAYERS    = 2
DIM_FF      = 128
DROPOUT     = 0.1
SEQ_LEN     = 7
BATCH_SIZE  = 32
LR          = 0.0005
EPOCHS      = 50
ES_PATIENCE = 10
GRAD_CLIP   = 1.0
SCHEDULER_PATIENCE = 5
SCHEDULER_FACTOR   = 0.5

# ── Physics constants — AGENTS.md / matching compute_rgi.py ───────────────────
R_GAS   = 287.05   # J/(kg·K) specific gas constant for dry air
CUT_IN  = 3.0      # m/s — turbine starts generating
RATED   = 7.0      # m/s — IEC Class III low-wind-speed turbines (India ~5 m/s mean)
CUT_OUT = 25.0     # m/s — turbine shuts down for safety
CP      = 0.4      # Betz efficiency coefficient

# ── Device selection — exact AGENTS.md pattern ────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')
print(f"Using device: {DEVICE}")


# ── Physics ────────────────────────────────────────────────────────────────────

def air_density(pressure_kpa: float, temperature_c: float) -> float:
    """Compute air density (kg/m³): ρ = P / (R·T)."""
    return (pressure_kpa * 1000) / (R_GAS * (temperature_c + 273.15))


def wind_rgi(wind_speed_50m: float, pressure_kpa: float, temperature_c: float) -> float:
    """Convert wind speed at 50m to Wind RGI [0, 1].

    Duplicated from compute_rgi.py — RATED=7 m/s, same constants.
    """
    v = wind_speed_50m
    if v < CUT_IN or v > CUT_OUT:
        return 0.0
    rho = air_density(pressure_kpa, temperature_c)
    power_ratio = 1.0 if v >= RATED else (rho / 1.225) * (v ** 3) / (RATED ** 3)
    return float(np.clip(power_ratio, 0, 1))


# ── Model ──────────────────────────────────────────────────────────────────────

class WindTransformer(nn.Module):
    """Transformer-based wind speed forecaster with learned monsoon embedding.

    Architecturally identical to SolarTransformer in transformer_monsoon.py;
    renamed for clarity and adapted to wind input features.

    Architecture:
      1. Input projection: Linear(input_size → d_model)
      2. Monsoon embedding: Embedding(3, d_model) — added elementwise
      3. Sinusoidal positional encoding
      4. TransformerEncoder: n_layers × (MultiHeadAttention + FFN)
      5. Last timestep → Linear(d_model → 1)
    """

    def __init__(
        self,
        input_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dim_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_proj        = nn.Linear(input_size, d_model)
        self.monsoon_embedding = nn.Embedding(3, d_model)
        self.register_buffer(
            'pos_enc',
            self._build_pos_enc(seq_len=100, d_model=d_model)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
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
        return pe.unsqueeze(0)   # (1, seq_len, d_model)

    def forward(self, x: torch.Tensor, monsoon: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x) + self.monsoon_embedding(monsoon)
        x = x + self.pos_enc[:, :x.size(1), :]
        x = self.dropout(x)
        x = self.transformer(x)
        return self.output(x[:, -1, :])


# ── Dataset ────────────────────────────────────────────────────────────────────

class SlidingWindowDataset(Dataset):
    """Sliding window dataset — exact pattern from transformer_monsoon.py.

    Target index: i + seq_len - 1 (AGENTS.md canonical form).
    Never i + seq_len — off-by-one was a critical Phase A bug.
    Windows never cross station or split boundaries.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int,
        feature_cols: list[str],
        monsoon_col: str,
        target_col: str,
    ) -> None:
        self.seq_len = seq_len
        self.data: list[tuple[np.ndarray, np.ndarray, float]] = []
        self._build(df, feature_cols, monsoon_col, target_col)

    def _build(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        monsoon_col: str,
        target_col: str,
    ) -> None:
        for _, grp in df.groupby('station', sort=True):
            grp   = grp.sort_values('date').reset_index(drop=True)
            feats = grp[feature_cols].to_numpy(dtype=np.float32)
            moon  = grp[monsoon_col].fillna(0).astype('int64').to_numpy()
            tgts  = grp[target_col].to_numpy(dtype=np.float32)
            n     = len(grp)
            for i in range(n - self.seq_len):
                x_feat = feats[i: i + self.seq_len]
                x_moon = moon[i:  i + self.seq_len]
                # AGENTS.md: targets[i + seq_len - 1] — never i + seq_len
                y      = tgts[i + self.seq_len - 1]
                if not (np.isnan(x_feat).any() or np.isnan(y)):
                    self.data.append((x_feat, x_moon, y))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_feat, x_moon, y = self.data[idx]
        return (
            torch.tensor(x_feat, dtype=torch.float32),
            torch.tensor(x_moon, dtype=torch.long),
            torch.tensor([y],    dtype=torch.float32),
        )


# ── Metric helpers ─────────────────────────────────────────────────────────────

def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, and R²."""
    err  = actual - predicted
    mae  = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_r = float(np.sum(err ** 2))
    ss_t = float(np.sum((actual - actual.mean()) ** 2))
    r2   = 1.0 - ss_r / ss_t if ss_t > 0 else float('nan')
    return {'mae': mae, 'rmse': rmse, 'r2': r2}


# ── Training helpers ───────────────────────────────────────────────────────────

def run_epoch(
    model:     WindTransformer,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Run one training epoch on DEVICE with gradient clipping."""
    model.train()
    total_loss = 0.0
    n_batches  = 0
    for xb_feat, xb_moon, yb in loader:
        xb_feat = xb_feat.to(DEVICE)
        xb_moon = xb_moon.to(DEVICE)
        yb      = yb.to(DEVICE)
        pred    = model(xb_feat, xb_moon)
        loss    = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item()
        n_batches  += 1
    return total_loss / max(n_batches, 1)


def run_eval_epoch(
    model:     WindTransformer,
    loader:    DataLoader,
    criterion: nn.Module,
) -> float:
    """Evaluate on CPU — exact AGENTS.md MPS inference fix pattern."""
    model.eval()
    eval_device = torch.device('cpu')
    model_cpu   = model.to(eval_device)
    total_loss  = 0.0
    with torch.no_grad():
        for xb_feat, xb_moon, yb in loader:
            xb_feat = xb_feat.to(eval_device)
            xb_moon = xb_moon.to(eval_device)
            yb      = yb.to(eval_device)
            pred    = model_cpu(xb_feat, xb_moon)
            total_loss += criterion(pred, yb).item() * len(xb_feat)
    val_loss = total_loss / len(loader.dataset)
    model.to(DEVICE)
    return val_loss


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_loss_curve(train_losses: list[float], val_losses: list[float]) -> None:
    """Save wind training loss curve."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label='Train Loss')
    ax.plot(val_losses,   label='Val Loss (CPU, MPS-corrected)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('WindTransformer Training: Train vs Val Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(LOSS_CURVE_PNG, dpi=150)
    plt.close(fig)
    print(f"  Saved → {LOSS_CURVE_PNG}")


def plot_comparison_wind(
    stations:        list[str],
    persist_rmse:    dict[str, float],
    transformer_rmse: dict[str, float],
) -> None:
    """Save two-bar comparison chart: Wind Persistence vs WindTransformer RMSE."""
    x = np.arange(len(stations))
    w = 0.35
    fig, ax = plt.subplots(figsize=(14, 6))

    def _vals(d: dict[str, float]) -> list[float]:
        return [d.get(s, float('nan')) for s in stations]

    b_p = ax.bar(x - w/2, _vals(persist_rmse),    w, label='Wind Persistence', color='steelblue',   alpha=0.85)
    b_t = ax.bar(x + w/2, _vals(transformer_rmse), w, label='Wind Transformer', color='mediumpurple', alpha=0.85)

    for bars in (b_p, b_t):
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f'{h:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Station')
    ax.set_ylabel('RMSE (m/s)')
    ax.set_title('Wind Forecast Test RMSE: Persistence vs WindTransformer+Monsoon')
    ax.set_xticks(x)
    ax.set_xticklabels(stations, rotation=30, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(COMPARISON_PNG, dpi=150)
    plt.close(fig)
    print(f"  Saved → {COMPARISON_PNG}")


def plot_predictions_jodhpur(
    df_test_scaled: pd.DataFrame,
    df_test_raw:    pd.DataFrame,
    scaler:         MinMaxScaler,
) -> None:
    """Save actual vs predicted wind speed for Jodhpur test 2024."""
    print("\nStep 13 (plot) — Prediction sample plot for Jodhpur 2024 ...")

    model_plot = WindTransformer(INPUT_SIZE, D_MODEL, N_HEADS, N_LAYERS, DIM_FF, DROPOUT)
    model_plot.load_state_dict(
        torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    )
    model_plot.eval()
    eval_device = torch.device('cpu')
    model_plot  = model_plot.to(eval_device)

    jodhpur_s = (
        df_test_scaled[df_test_scaled['station'] == 'Jodhpur']
        .sort_values('date').reset_index(drop=True)
    )
    jodhpur_r = (
        df_test_raw[df_test_raw['station'] == 'Jodhpur']
        .sort_values('date').reset_index(drop=True)
    )
    print(f"  Jodhpur test rows: {len(jodhpur_s)}")

    feats    = jodhpur_s[FEATURE_COLS].to_numpy(dtype=np.float32)
    moon_arr = jodhpur_s[MONSOON_COL].fillna(0).astype('int64').to_numpy()
    tgts     = jodhpur_s[TARGET_COL].to_numpy(dtype=np.float32)
    dates    = jodhpur_s['date'].to_numpy()
    wind_idx = FEATURE_COLS.index('wind_speed_50m')
    n        = len(jodhpur_s)

    pred_dates:  list[pd.Timestamp] = []
    actuals:     list[float]        = []
    predictions: list[float]        = []

    with torch.no_grad():
        for i in range(n - SEQ_LEN):
            x_feat = feats[i: i + SEQ_LEN]
            x_moon = moon_arr[i: i + SEQ_LEN]
            y_true = tgts[i + SEQ_LEN - 1]
            if np.isnan(x_feat).any() or np.isnan(y_true):
                continue

            xf_t = torch.tensor(x_feat, dtype=torch.float32).unsqueeze(0).to(eval_device)
            xm_t = torch.tensor(x_moon, dtype=torch.long).unsqueeze(0).to(eval_device)
            yp   = model_plot(xf_t, xm_t).numpy().flatten()[0]

            dummy_p = np.zeros((1, len(FEATURE_COLS)), dtype=np.float32)
            dummy_p[0, wind_idx] = yp
            y_pred_ms = scaler.inverse_transform(dummy_p)[0, wind_idx]

            dummy_a = np.zeros((1, len(FEATURE_COLS)), dtype=np.float32)
            dummy_a[0, wind_idx] = y_true
            y_actual_ms = scaler.inverse_transform(dummy_a)[0, wind_idx]

            pred_dates.append(pd.Timestamp(dates[i + SEQ_LEN - 1]))
            actuals.append(y_actual_ms)
            predictions.append(y_pred_ms)

    print(f"  Valid inference windows: {len(pred_dates)}")
    if not pred_dates:
        print("  WARNING: no valid Jodhpur windows — skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(pred_dates, actuals,     color='steelblue',  linestyle='-',
            linewidth=1.2, label='Actual Wind Speed')
    ax.plot(pred_dates, predictions, color='darkorange', linestyle='--',
            linewidth=1.2, label='Predicted Wind Speed')
    ax.set_title('WindTransformer Predictions vs Actual — Jodhpur Test 2024')
    ax.set_xlabel('Date')
    ax.set_ylabel('Wind Speed 50m (m/s)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PREDICTIONS_PNG, dpi=150)
    plt.close(fig)
    print(f"  Saved → {PREDICTIONS_PNG}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """Script 12 — WindTransformer with monsoon embedding training and evaluation."""
    ROOT.joinpath('models').mkdir(parents=True, exist_ok=True)
    ROOT.joinpath('outputs').mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print("wind_transformer.py — Script 12: Wind Transformer + Monsoon Embedding")
    print(f"{'='*65}\n")

    # ── Step 1: Load and validate inputs ──────────────────────────────────────
    print("Step 1 — Loading inputs ...")
    if not EXTENDED_CSV.exists():
        print(f"ERROR: nasa_power_extended.csv not found: {EXTENDED_CSV}")
        print("Run scripts/fetch_extended_power.py first.")
        sys.exit(1)
    if not MONSOON_CSV.exists():
        print(f"ERROR: monsoon_phases.csv not found: {MONSOON_CSV}")
        print("Run scripts/monsoon_labels.py first.")
        sys.exit(1)

    df      = pd.read_csv(EXTENDED_CSV, parse_dates=['date'])
    monsoon = pd.read_csv(MONSOON_CSV,  parse_dates=['date'])
    print(f"  nasa_power_extended rows : {len(df):,}")
    print(f"  monsoon_phases rows      : {len(monsoon):,}")

    # Replace NASA POWER fill value
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].replace(-999, np.nan)

    print(f"  Stations in extended CSV: {sorted(df['station'].unique())}")

    # ── Step 2: Merge monsoon_phase ───────────────────────────────────────────
    print("\nStep 2 — Merging monsoon_phase into extended dataset ...")
    rows_before = len(df)
    df = pd.merge(
        df,
        monsoon[['date', 'station', 'monsoon_phase']],
        on=['date', 'station'],
        how='left',
    )
    print(f"  Rows before merge: {rows_before:,}  after: {len(df):,}")
    print(f"  monsoon_phase missing: {df['monsoon_phase'].isna().sum()}")

    # Compute cyclical date features (not in nasa_power_extended.csv)
    df['doy']       = df['date'].dt.dayofyear
    df['month']     = df['date'].dt.month
    df['doy_sin']   = np.sin(2 * np.pi * df['doy']   / 365.25)
    df['doy_cos']   = np.cos(2 * np.pi * df['doy']   / 365.25)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)

    df = df.sort_values(['station', 'date']).reset_index(drop=True)

    # ── Step 3: Build target; verify shift correctness ────────────────────────
    print("\nStep 3 — Building target column (wind_speed_50m.shift(-1) per station) ...")
    df[TARGET_COL] = df.groupby('station')['wind_speed_50m'].transform(
        lambda s: s.shift(-1)
    )
    print(f"  Target NaN count (last row per station): {df[TARGET_COL].isna().sum()}")

    # Verification: target[i] == wind_speed_50m[i+1] for Jodhpur
    print("\n  Verification (first 20 rows, Jodhpur): target[i] == wind_speed_50m[i+1]")
    jdh = df[df['station'] == 'Jodhpur'].sort_values('date').reset_index(drop=True)
    print(f"  {'Row':>4} | {'Date':>12} | {'wind[i]':>10} | {'target[i]':>10} | "
          f"{'wind[i+1]':>10} | {'Match':>6}")
    print(f"  {'-'*70}")
    mismatches = 0
    for i in range(min(20, len(jdh) - 1)):
        t_val = jdh.loc[i, TARGET_COL]
        w_next = jdh.loc[i + 1, 'wind_speed_50m']
        match  = np.isnan(t_val) and np.isnan(w_next) or (
            not np.isnan(t_val) and not np.isnan(w_next) and abs(t_val - w_next) < 1e-6
        )
        if not match:
            mismatches += 1
        m_str = 'YES' if match else '*** NO ***'
        print(f"  {i:>4} | {str(jdh.loc[i,'date'].date()):>12} | "
              f"{jdh.loc[i,'wind_speed_50m']:>10.4f} | "
              f"{t_val if not np.isnan(t_val) else float('nan'):>10.4f} | "
              f"{w_next if not np.isnan(w_next) else float('nan'):>10.4f} | "
              f"{m_str:>6}")
    if mismatches > 0:
        raise ValueError(
            f"Target shift verification FAILED: {mismatches} mismatches in "
            "first 20 Jodhpur rows. Check wind_speed_50m.shift(-1) logic."
        )
    print("  Verification PASSED — target[i] == wind_speed_50m[i+1] confirmed.")

    # ── Step 4: Time-based split — exact AGENTS.md dates ─────────────────────
    print("\nStep 4 — Time-based split ...")
    train_mask = (df['date'] >= TRAIN_START) & (df['date'] <= TRAIN_END)
    val_mask   = (df['date'] >= VAL_START)   & (df['date'] <= VAL_END)
    test_mask  = (df['date'] >= TEST_START)  & (df['date'] <= TEST_END)

    df_train_raw = df[train_mask].copy().reset_index(drop=True)
    df_val_raw   = df[val_mask].copy().reset_index(drop=True)
    df_test_raw  = df[test_mask].copy().reset_index(drop=True)

    all_needed = FEATURE_COLS + [MONSOON_COL, TARGET_COL]
    for name, split_df in [
        ('train', df_train_raw), ('val', df_val_raw), ('test', df_test_raw)
    ]:
        before = len(split_df)
        split_df.dropna(subset=all_needed, inplace=True)
        split_df.reset_index(drop=True, inplace=True)
        print(f"  {name:>5}: {before:,} rows → {len(split_df):,} after dropna "
              f"(dropped {before - len(split_df)})")

    # ── Step 5: Wind persistence baseline (before scaling, on raw data) ───────
    print("\nStep 5 — Wind persistence baseline on test set ...")
    print("  Persistence: predicted wind = yesterday's wind = wind_speed_50m (no shift)")
    print("  Actual:      wind_speed_50m_target = next-day wind speed")

    persist_metrics: dict[str, dict[str, float]] = {}
    persist_rmse_by_station: dict[str, float]    = {}

    all_persist_actual: list[np.ndarray] = []
    all_persist_pred:   list[np.ndarray] = []

    for station, grp in df_test_raw.groupby('station', sort=True):
        grp = grp.sort_values('date').reset_index(drop=True)
        actual = grp[TARGET_COL].to_numpy(dtype=np.float64)
        pred   = grp['wind_speed_50m'].to_numpy(dtype=np.float64)
        valid  = ~(np.isnan(actual) | np.isnan(pred))
        if valid.sum() == 0:
            print(f"  WARNING: no valid persistence windows for {station}")
            continue
        a = actual[valid]
        p = pred[valid]
        m = compute_metrics(a, p)
        persist_metrics[station]            = m
        persist_rmse_by_station[station]    = m['rmse']
        all_persist_actual.append(a)
        all_persist_pred.append(p)

    if all_persist_actual:
        ca = np.concatenate(all_persist_actual)
        cp = np.concatenate(all_persist_pred)
        persist_metrics['OVERALL'] = compute_metrics(ca, cp)
        persist_rmse_by_station['OVERALL'] = persist_metrics['OVERALL']['rmse']

    print(f"\n  Wind Persistence baseline — test set per station:")
    print(f"  {'Station':<12} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    print('  ' + '-' * 42)
    for s, m in sorted(persist_metrics.items()):
        print(f"  {s:<12} {m['mae']:>8.4f} {m['rmse']:>8.4f} {m['r2']:>8.4f}")

    METRICS_PERSISTENCE_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PERSISTENCE_JSON, 'w') as fh:
        json.dump({'test': persist_metrics}, fh, indent=2)
    print(f"\n  Saved → {METRICS_PERSISTENCE_JSON}")

    # ── Step 6: Fit MinMaxScaler on TRAIN continuous features only ────────────
    print("\nStep 6 — Fitting MinMaxScaler on TRAIN (continuous features only) ...")
    scaler = MinMaxScaler()
    df_train_scaled = df_train_raw.copy()
    df_val_scaled   = df_val_raw.copy()
    df_test_scaled  = df_test_raw.copy()

    df_train_scaled[FEATURE_COLS] = scaler.fit_transform(df_train_scaled[FEATURE_COLS])
    df_val_scaled[FEATURE_COLS]   = scaler.transform(df_val_scaled[FEATURE_COLS])
    df_test_scaled[FEATURE_COLS]  = scaler.transform(df_test_scaled[FEATURE_COLS])

    SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    print(f"  Scaler saved → {SCALER_PATH}")
    print(f"  monsoon_phase NOT scaled — stays integer for Embedding layer")

    wind_idx   = FEATURE_COLS.index('wind_speed_50m')
    wind_min   = float(scaler.data_min_[wind_idx])
    wind_range = float(scaler.data_max_[wind_idx]) - wind_min
    print(f"  wind_speed_50m train range: [{wind_min:.4f}, {wind_min+wind_range:.4f}] m/s")

    def scale_target(s: pd.Series) -> pd.Series:
        return (s - wind_min) / wind_range if wind_range > 0 else s

    df_train_scaled[TARGET_COL] = scale_target(df_train_scaled[TARGET_COL])
    df_val_scaled[TARGET_COL]   = scale_target(df_val_scaled[TARGET_COL])
    df_test_scaled[TARGET_COL]  = scale_target(df_test_scaled[TARGET_COL])

    # ── Step 7: SlidingWindowDatasets ─────────────────────────────────────────
    print(f"\nStep 7 — Building SlidingWindowDatasets (SEQ_LEN={SEQ_LEN}) ...")
    train_ds = SlidingWindowDataset(
        df_train_scaled, SEQ_LEN, FEATURE_COLS, MONSOON_COL, TARGET_COL
    )
    val_ds = SlidingWindowDataset(
        df_val_scaled, SEQ_LEN, FEATURE_COLS, MONSOON_COL, TARGET_COL
    )
    test_ds = SlidingWindowDataset(
        df_test_scaled, SEQ_LEN, FEATURE_COLS, MONSOON_COL, TARGET_COL
    )
    print(f"  Train windows: {len(train_ds):,}")
    print(f"  Val   windows: {len(val_ds):,}")
    print(f"  Test  windows: {len(test_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Step 8: Device ─────────────────────────────────────────────────────────
    print(f"\nStep 8 — Device: {DEVICE}")

    # ── Step 9: Train WindTransformer ─────────────────────────────────────────
    print(f"\nStep 9 — Initialising WindTransformer ...")
    print(f"  input_size={INPUT_SIZE}  d_model={D_MODEL}  n_heads={N_HEADS}"
          f"  n_layers={N_LAYERS}  dim_ff={DIM_FF}")
    model = WindTransformer(
        input_size=INPUT_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        dim_ff=DIM_FF,
        dropout=DROPOUT,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"  monsoon_embedding: {model.monsoon_embedding}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min',
        patience=SCHEDULER_PATIENCE,
        factor=SCHEDULER_FACTOR,
    )

    print(f"\n  Training for up to {EPOCHS} epochs (LR={LR}) ...")
    print("  Val loss on CPU — MPS inference artifact fix active")

    train_losses:  list[float] = []
    val_losses:    list[float] = []
    best_val_loss  = float('inf')
    epochs_no_impr = 0

    for epoch in range(1, EPOCHS + 1):
        try:
            train_loss = run_epoch(model, train_loader, criterion, optimizer)
            val_loss   = run_eval_epoch(model, val_loader, criterion)
        except RuntimeError as exc:
            if 'out of memory' in str(exc).lower():
                print(f"\n  CUDA OUT OF MEMORY at epoch {epoch}.")
                raise
            raise

        if np.isnan(train_loss) or np.isnan(val_loss):
            print(f"\n  WARNING: NaN loss at epoch {epoch}. Stopping.")
            break

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)
            epochs_no_impr = 0
        else:
            epochs_no_impr += 1

        if epoch % 5 == 0:
            print(f"  Epoch {epoch:>3}/{EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f}")

        if epochs_no_impr >= ES_PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"({ES_PATIENCE} epochs without improvement).")
            break

    print(f"\n  Best val loss: {best_val_loss:.4f}  →  {MODEL_PATH}")
    plot_loss_curve(train_losses, val_losses)

    # ── Step 10: Test evaluation (CPU, MPS fix) ───────────────────────────────
    print("\nStep 10 — Test evaluation (CPU, MPS fix) ...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    eval_device = torch.device('cpu')
    model_cpu   = model.to(eval_device)

    transformer_metrics:        dict[str, dict[str, float]] = {}
    transformer_rmse_by_station: dict[str, float]           = {}

    # Also collect arrays for RGI comparison (Step 12)
    all_actual_wind:  list[np.ndarray] = []
    all_pred_wind:    list[np.ndarray] = []
    all_persist_wind: list[np.ndarray] = []   # persistence = last window day's wind
    all_pressure:     list[np.ndarray] = []
    all_temp:         list[np.ndarray] = []

    for station in STATIONS_LIST:
        grp_s = (
            df_test_scaled[df_test_scaled['station'] == station]
            .sort_values('date').reset_index(drop=True)
        )
        grp_r = (
            df_test_raw[df_test_raw['station'] == station]
            .sort_values('date').reset_index(drop=True)
        )
        if len(grp_s) == 0:
            print(f"  WARNING: no test data for {station}")
            continue

        feats    = grp_s[FEATURE_COLS].to_numpy(dtype=np.float32)
        moon_arr = grp_s[MONSOON_COL].fillna(0).astype('int64').to_numpy()
        tgts     = grp_s[TARGET_COL].to_numpy(dtype=np.float32)
        wind_raw = grp_r['wind_speed_50m'].to_numpy(dtype=np.float64)
        pres_raw = grp_r['surface_pressure'].to_numpy(dtype=np.float64)
        temp_raw = grp_r['temperature'].to_numpy(dtype=np.float64)
        tgt_raw  = grp_r[TARGET_COL].to_numpy(dtype=np.float64)
        n        = len(grp_s)

        acts:         list[float] = []
        preds:        list[float] = []
        persists:     list[float] = []
        pressures:    list[float] = []
        temps:        list[float] = []

        with torch.no_grad():
            for i in range(n - SEQ_LEN):
                x_feat = feats[i: i + SEQ_LEN]
                x_moon = moon_arr[i: i + SEQ_LEN]
                y_true = tgts[i + SEQ_LEN - 1]
                if np.isnan(x_feat).any() or np.isnan(y_true):
                    continue

                xf_t = torch.tensor(x_feat, dtype=torch.float32).unsqueeze(0).to(eval_device)
                xm_t = torch.tensor(x_moon, dtype=torch.long).unsqueeze(0).to(eval_device)
                yp   = model_cpu(xf_t, xm_t).numpy().flatten()[0]

                # Inverse transform prediction
                dummy = np.zeros((1, len(FEATURE_COLS)), dtype=np.float32)
                dummy[0, wind_idx] = yp
                y_pred_ms = scaler.inverse_transform(dummy)[0, wind_idx]

                # Actual next-day wind (from raw target)
                y_actual_ms = tgt_raw[i + SEQ_LEN - 1]
                if np.isnan(y_actual_ms):
                    continue

                # Persistence = last window day's wind = "current" wind
                persist_ms = wind_raw[i + SEQ_LEN - 1]

                # Pressure and temp from last window day (closest available approx)
                pressure_day = pres_raw[i + SEQ_LEN - 1]
                temp_day     = temp_raw[i + SEQ_LEN - 1]

                acts.append(y_actual_ms)
                preds.append(y_pred_ms)
                persists.append(persist_ms)
                pressures.append(pressure_day)
                temps.append(temp_day)

        if not acts:
            print(f"  WARNING: no valid test windows for {station}")
            continue

        a = np.array(acts)
        p = np.array(preds)
        m = compute_metrics(a, p)
        transformer_metrics[station]            = m
        transformer_rmse_by_station[station]    = m['rmse']
        all_actual_wind.append(a)
        all_pred_wind.append(p)
        all_persist_wind.append(np.array(persists))
        all_pressure.append(np.array(pressures))
        all_temp.append(np.array(temps))

    model.to(DEVICE)

    if all_actual_wind:
        ca = np.concatenate(all_actual_wind)
        cp = np.concatenate(all_pred_wind)
        transformer_metrics['OVERALL'] = compute_metrics(ca, cp)
        transformer_rmse_by_station['OVERALL'] = transformer_metrics['OVERALL']['rmse']

    print(f"\n  WindTransformer test metrics per station:")
    print(f"  {'Station':<12} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    print('  ' + '-' * 42)
    for s, m in sorted(transformer_metrics.items()):
        print(f"  {s:<12} {m['mae']:>8.4f} {m['rmse']:>8.4f} {m['r2']:>8.4f}")

    with open(METRICS_TRANSFORMER_JSON, 'w') as fh:
        json.dump({'test': transformer_metrics}, fh, indent=2)
    print(f"\n  Saved → {METRICS_TRANSFORMER_JSON}")

    # ── Step 11: Comparison table: Persistence vs Transformer ─────────────────
    print("\nStep 11 — Comparison table: Wind Persistence vs WindTransformer")

    stations_sorted = [s for s in STATIONS_LIST if s in transformer_metrics]
    sep = '-' * 68

    print(f"\n  {sep}")
    print(f"  {'Station':<12} | {'Persist RMSE':>14} | {'Transformer RMSE':>18} | {'Gain%':>8}")
    print(f"  {sep}")

    beats = 0
    for s in stations_sorted:
        t_r = transformer_metrics[s]['rmse']
        p_r = persist_metrics.get(s, {}).get('rmse', float('nan'))
        gain = (p_r - t_r) / p_r * 100 if not np.isnan(p_r) else float('nan')
        if not np.isnan(p_r) and t_r < p_r:
            beats += 1
        g_s   = f"{gain:+.2f}%" if not np.isnan(gain) else "N/A"
        beat_m = '  ← BEAT' if (not np.isnan(p_r) and t_r < p_r) else ''
        print(f"  {s:<12} | {p_r:>14.4f} | {t_r:>18.4f} | {g_s:>8}{beat_m}")

    ov_t = transformer_metrics.get('OVERALL', {}).get('rmse', float('nan'))
    ov_p = persist_metrics.get('OVERALL', {}).get('rmse', float('nan'))
    ov_g = (ov_p - ov_t) / ov_p * 100 if not np.isnan(ov_p) else float('nan')
    print(f"  {sep}")
    print(f"  {'OVERALL':<12} | {ov_p:>14.4f} | {ov_t:>18.4f} | {ov_g:>+7.2f}%")
    print(f"  {sep}")

    print(f"\n  Wind Transformer beats persistence on {beats}/{len(stations_sorted)} stations")

    plot_comparison_wind(
        stations_sorted,
        {s: persist_metrics[s]['rmse']      for s in stations_sorted if s in persist_metrics},
        {s: transformer_metrics[s]['rmse']  for s in stations_sorted if s in transformer_metrics},
    )

    # ── Step 12: Wind RGI comparison ──────────────────────────────────────────
    print("\nStep 12 — Wind RGI comparison (justifies replacing persistence in Phase B) ...")
    print(f"  RATED={RATED} m/s  CUT_IN={CUT_IN} m/s  CUT_OUT={CUT_OUT} m/s  (same as compute_rgi.py)")

    if not all_actual_wind:
        print("  Skipping: no test windows available.")
    else:
        all_actual_arr  = np.concatenate(all_actual_wind)
        all_pred_arr    = np.concatenate(all_pred_wind)
        all_persist_arr = np.concatenate(all_persist_wind)
        all_pressure_arr = np.concatenate(all_pressure)
        all_temp_arr    = np.concatenate(all_temp)

        actual_rgi_arr   = np.array([
            wind_rgi(float(w), float(p), float(t))
            for w, p, t in zip(all_actual_arr,  all_pressure_arr, all_temp_arr)
        ])
        persist_rgi_arr  = np.array([
            wind_rgi(float(w), float(p), float(t))
            for w, p, t in zip(all_persist_arr, all_pressure_arr, all_temp_arr)
        ])
        transformer_rgi_arr = np.array([
            wind_rgi(float(w), float(p), float(t))
            for w, p, t in zip(all_pred_arr,    all_pressure_arr, all_temp_arr)
        ])

        persist_rgi_metrics     = compute_metrics(actual_rgi_arr, persist_rgi_arr)
        transformer_rgi_metrics = compute_metrics(actual_rgi_arr, transformer_rgi_arr)

        rgi_mae_improvement = (
            (persist_rgi_metrics['mae'] - transformer_rgi_metrics['mae'])
            / persist_rgi_metrics['mae'] * 100
            if persist_rgi_metrics['mae'] > 0 else float('nan')
        )
        rgi_rmse_improvement = (
            (persist_rgi_metrics['rmse'] - transformer_rgi_metrics['rmse'])
            / persist_rgi_metrics['rmse'] * 100
            if persist_rgi_metrics['rmse'] > 0 else float('nan')
        )

        print(f"\n  Wind RGI accuracy: persistence vs Transformer wind forecast")
        print(f"  {'Method':<25} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
        print('  ' + '-' * 55)
        m = persist_rgi_metrics
        print(f"  {'Wind Persistence':<25} {m['mae']:>8.4f} {m['rmse']:>8.4f} {m['r2']:>8.4f}")
        m = transformer_rgi_metrics
        print(f"  {'Wind Transformer':<25} {m['mae']:>8.4f} {m['rmse']:>8.4f} {m['r2']:>8.4f}")

        print(f"\n  Wind RGI MAE improvement  : {rgi_mae_improvement:+.2f}%")
        print(f"  Wind RGI RMSE improvement : {rgi_rmse_improvement:+.2f}%")
        print(f"  Mean actual Wind RGI      : {actual_rgi_arr.mean():.4f}")
        print(f"  Mean persistence Wind RGI : {persist_rgi_arr.mean():.4f}")
        print(f"  Mean Transformer Wind RGI : {transformer_rgi_arr.mean():.4f}")

    # ── Step 13: Jodhpur prediction plot ──────────────────────────────────────
    plot_predictions_jodhpur(df_test_scaled, df_test_raw, scaler)

    # ── SELF CHECK ──────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("SELF CHECK")
    print(f"{'='*65}")

    print(f"\nD_MODEL={D_MODEL}  N_HEADS={N_HEADS}  N_LAYERS={N_LAYERS}  — confirmed")
    print(f"INPUT_SIZE={INPUT_SIZE}  (7 wind features, monsoon_phase via Embedding)")
    print(f"monsoon_embedding: {model.monsoon_embedding}")
    print(f"scaler_wind.pkl saved           : {SCALER_PATH.exists()}")
    print(f"wind_transformer_best.pt saved  : {MODEL_PATH.exists()}")
    print(f"metrics_wind_persistence.json   : {METRICS_PERSISTENCE_JSON.exists()}")
    print(f"metrics_wind_transformer.json   : {METRICS_TRANSFORMER_JSON.exists()}")
    print(f"loss_curve_wind.png             : {LOSS_CURVE_PNG.exists()}")
    print(f"comparison_wind.png             : {COMPARISON_PNG.exists()}")
    print(f"predictions_wind_jodhpur.png    : {PREDICTIONS_PNG.exists()}")

    print(f"\nBest val loss (CPU)          : {best_val_loss:.4f}")
    print(f"Overall test RMSE Transformer: {ov_t:.4f}")
    print(f"Overall test RMSE Persistence: {ov_p:.4f}")

    print(f"\nFinal per-station RMSE:")
    for s in stations_sorted:
        t_r = transformer_metrics[s]['rmse']
        p_r = persist_metrics.get(s, {}).get('rmse', float('nan'))
        flag = 'BEAT' if (not np.isnan(p_r) and t_r < p_r) else '----'
        print(f"  {s:<12}: Transformer={t_r:.4f}  Persistence={p_r:.4f}  [{flag}]")

    print(f"\nWind Transformer beats persistence on {beats}/{len(stations_sorted)} stations")
    if not all_actual_wind:
        print("Wind RGI MAE improvement: N/A (no test windows)")
    else:
        print(f"Wind RGI MAE improvement: {rgi_mae_improvement:+.2f}%")

    print(f"\nScript 12 complete.")
    print(f"{'='*65}\n")


if __name__ == '__main__':
    main()
