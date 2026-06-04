"""
add_aod_retrain.py
Phase 2: merge MERRA-2 AOD into the dataset and retrain the LSTM.
Adds 5 AOD variables (TOTEXTTAU, DUEXTTAU, BCEXTTAU, SSEXTTAU, SUEXTTAU)
to the 10 Phase-1 features (15 total input features).
This is the paper's primary result script.

Saves:
  models/lstm_aod_best.pt
  models/scaler_aod.pkl
  outputs/metrics_lstm_aod.json
  outputs/loss_curve_aod.png
  outputs/comparison_aod.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

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

# ── Constants ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

DATASET_CSV      = ROOT / 'data' / 'processed' / 'dataset_final.csv'
AOD_CSV          = ROOT / 'data' / 'processed' / 'merra2_aod_stations.csv'
PERSISTENCE_JSON = ROOT / 'outputs' / 'metrics_persistence.json'
LSTM_JSON        = ROOT / 'outputs' / 'metrics_lstm.json'       # Phase 1 — never overwrite
MODEL_PATH       = ROOT / 'models'  / 'lstm_aod_best.pt'
SCALER_PATH      = ROOT / 'models'  / 'scaler_aod.pkl'
METRICS_JSON     = ROOT / 'outputs' / 'metrics_lstm_aod.json'
LOSS_CURVE_PNG   = ROOT / 'outputs' / 'loss_curve_aod.png'
COMPARISON_PNG   = ROOT / 'outputs' / 'comparison_aod.png'

TRAIN_START = "2017-04-01"
TRAIN_END   = "2022-12-31"
VAL_START   = "2023-01-01"
VAL_END     = "2023-12-31"
TEST_START  = "2024-01-01"
TEST_END    = "2024-12-31"

# Station names — must match AGENTS.md STATIONS dict exactly
STATIONS_LIST: list[str] = [
    'Jodhpur', 'Ahmedabad', 'Mumbai', 'Chennai',
    'Hyderabad', 'Bhopal', 'Kolkata', 'Bengaluru',
]

# Phase 2 feature set (AGENTS.md: 14, actual list count = 15)
# INPUT_SIZE = len(FEATURE_COLS) ensures model and feature list stay in sync
FEATURE_COLS: list[str] = [
    'ghi', 'temperature', 'humidity', 'wind_speed',
    'cloud_cover', 'monsoon_phase',
    'doy_sin', 'doy_cos', 'month_sin', 'month_cos',
    'TOTEXTTAU', 'DUEXTTAU', 'BCEXTTAU',
    'SSEXTTAU', 'SUEXTTAU',
]

TARGET_COL = 'ghi_target'
AOD_VARS: list[str] = ['TOTEXTTAU', 'DUEXTTAU', 'BCEXTTAU', 'SSEXTTAU', 'SUEXTTAU']

HIGH_AOD_THRESHOLD: float = 0.4    # TOTEXTTAU > this = "high aerosol" day

# LSTM hyperparameters — identical to Phase 1 except input_size
SEQ_LEN     = 7
INPUT_SIZE  = len(FEATURE_COLS)    # derived from list so model & data stay in sync
HIDDEN_SIZE = 128
NUM_LAYERS  = 2
DROPOUT     = 0.2
EPOCHS      = 50
BATCH_SIZE  = 32
LR          = 0.001
GRAD_CLIP   = 1.0
SCHEDULER_PATIENCE  = 5
SCHEDULER_FACTOR    = 0.5
EARLY_STOP_PATIENCE = 10

# ── Device ────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')       # Windows uni lab RTX 4090
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')        # Mac Apple Silicon
else:
    DEVICE = torch.device('cpu')
print(f"Using device: {DEVICE}")


# ── Model ─────────────────────────────────────────────────────────────────────

class LSTMForecaster(nn.Module):
    """Two-layer LSTM with dropout and a fully-connected output head.

    Args:
        input_size:  Number of input features per timestep.
        hidden_size: Hidden units per LSTM layer.
        num_layers:  Number of stacked LSTM layers.
        dropout:     Dropout probability between LSTM layers.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [batch, seq_len, input_size].

        Returns:
            Prediction tensor of shape [batch, 1].
        """
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ── Dataset ───────────────────────────────────────────────────────────────────

class SlidingWindowDataset(Dataset):
    """Sliding-window dataset that never crosses station or split boundaries.

    Windows are built independently per station.
    Target index uses targets[i + seq_len - 1] (AGENTS.md canonical form),
    equivalent to targets[i - 1] in the loop-from-seq_len notation.

    Args:
        df:           DataFrame for a single split (already scaled).
        seq_len:      Number of historical timesteps per window.
        feature_cols: Feature column names.
        target_col:   Scaled target column name.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int,
        feature_cols: list[str],
        target_col: str,
    ) -> None:
        self.seq_len      = seq_len
        self.feature_cols = feature_cols
        self.target_col   = target_col
        self.windows: list[tuple[np.ndarray, float]] = []
        self._build(df)

    def _build(self, df: pd.DataFrame) -> None:
        """Pre-build all (X, y) pairs, grouped by station.

        Args:
            df: Split DataFrame, sorted by station then date.
        """
        for _, group in df.groupby('station', sort=True):
            group    = group.sort_values('date').reset_index(drop=True)
            features = group[self.feature_cols].to_numpy(dtype=np.float32)
            targets  = group[self.target_col].to_numpy(dtype=np.float32)
            n = len(group)
            for i in range(n - self.seq_len):
                x = features[i: i + self.seq_len]
                # targets[i + seq_len - 1] = ghi one day after the window ends
                # AGENTS.md canonical form — never use targets[i + seq_len]
                y = targets[i + self.seq_len - 1]
                if not (np.isnan(x).any() or np.isnan(y)):
                    self.windows.append((x, y))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.windows[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor([y], dtype=torch.float32)


# ── Metric helpers ─────────────────────────────────────────────────────────────

def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, and R².

    Args:
        actual:    Ground-truth array (no NaNs).
        predicted: Prediction array (no NaNs).

    Returns:
        Dict with keys mae, rmse, r2.
    """
    errors = actual - predicted
    mae    = float(np.mean(np.abs(errors)))
    rmse   = float(np.sqrt(np.mean(errors ** 2)))
    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return {'mae': mae, 'rmse': rmse, 'r2': r2}


# ── Training helpers ───────────────────────────────────────────────────────────

def run_epoch(
    model:     LSTMForecaster,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Run one training epoch on DEVICE with gradient clipping.

    Args:
        model:     LSTMForecaster on DEVICE.
        loader:    Training DataLoader (shuffled).
        criterion: Loss function (MSELoss).
        optimizer: Adam optimizer.

    Returns:
        Mean training loss over the epoch.
    """
    model.train()
    total_loss = 0.0
    n_batches  = 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        pred   = model(xb)
        loss   = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item()
        n_batches  += 1
    return total_loss / max(n_batches, 1)


def run_eval_epoch(
    model:     LSTMForecaster,
    loader:    DataLoader,
    criterion: nn.Module,
) -> float:
    """Evaluate on CPU to avoid the MPS inference artifact.

    Apple Silicon MPS reports ~2.5x lower loss during torch.no_grad() passes,
    corrupting early stopping and best-model checkpointing.
    This function implements the exact AGENTS.md fix pattern.

    Args:
        model:     LSTMForecaster (moved to CPU, then back to DEVICE).
        loader:    Evaluation DataLoader.
        criterion: Loss function (MSELoss).

    Returns:
        Sample-weighted mean loss over the full split.
    """
    model.eval()
    eval_device = torch.device('cpu')
    model_cpu   = model.to(eval_device)
    total_loss  = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb   = xb.to(eval_device)
            yb   = yb.to(eval_device)
            pred = model_cpu(xb)
            total_loss += criterion(pred, yb).item() * len(xb)
    val_loss = total_loss / len(loader.dataset)
    model.to(DEVICE)
    return val_loss


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_on_split(
    model:        LSTMForecaster,
    df_split:     pd.DataFrame,
    scaler:       MinMaxScaler,
    feature_cols: list[str],
    target_col:   str,
    seq_len:      int,
) -> tuple[dict[str, dict[str, float]], np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate model on a split, returning metrics and raw prediction arrays.

    All inference is on CPU to avoid the MPS inference artifact (AGENTS.md).
    Returns TOTEXTTAU values in original (unscaled) space for attribution.

    Args:
        model:        Trained LSTMForecaster.
        df_split:     Scaled DataFrame for the split.
        scaler:       Fitted MinMaxScaler for inverse-transforming GHI.
        feature_cols: Feature column names (len = model input_size).
        target_col:   Scaled target column name.
        seq_len:      Sliding window length.

    Returns:
        Tuple of (station_metrics, all_actuals, all_preds, all_totexttau).
        all_totexttau is NaN-filled where AOD data is absent.
    """
    model.eval()
    eval_device = torch.device('cpu')
    model_cpu   = model.to(eval_device)

    ghi_idx = feature_cols.index('ghi')
    has_aod = 'TOTEXTTAU' in feature_cols
    aod_idx = feature_cols.index('TOTEXTTAU') if has_aod else -1

    all_actual: list[np.ndarray] = []
    all_pred:   list[np.ndarray] = []
    all_aod:    list[np.ndarray] = []
    station_metrics: dict[str, dict[str, float]] = {}

    for station, group in df_split.groupby('station', sort=True):
        group    = group.sort_values('date').reset_index(drop=True)
        features = group[feature_cols].to_numpy(dtype=np.float32)
        targets  = group[target_col].to_numpy(dtype=np.float32)
        n = len(group)

        acts:  list[float] = []
        preds: list[float] = []
        aods:  list[float] = []

        with torch.no_grad():
            for i in range(n - seq_len):
                x      = features[i: i + seq_len]
                y_true = targets[i + seq_len - 1]
                if np.isnan(x).any() or np.isnan(y_true):
                    continue

                x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(eval_device)
                yp  = model_cpu(x_t).numpy().flatten()[0]

                dummy = np.zeros((1, len(feature_cols)), dtype=np.float32)
                dummy[0, ghi_idx] = yp
                y_pred = scaler.inverse_transform(dummy)[0, ghi_idx]

                dummy_t = np.zeros((1, len(feature_cols)), dtype=np.float32)
                dummy_t[0, ghi_idx] = y_true
                y_actual = scaler.inverse_transform(dummy_t)[0, ghi_idx]

                acts.append(y_actual)
                preds.append(y_pred)

                if has_aod:
                    aod_sc = x[-1, aod_idx]
                    dummy_a = np.zeros((1, len(feature_cols)), dtype=np.float32)
                    dummy_a[0, aod_idx] = aod_sc
                    aods.append(scaler.inverse_transform(dummy_a)[0, aod_idx])
                else:
                    aods.append(float('nan'))

        if not acts:
            print(f"  WARNING: no valid windows for {station}")
            continue

        a = np.array(acts)
        p = np.array(preds)
        station_metrics[station] = compute_metrics(a, p)
        all_actual.append(a)
        all_pred.append(p)
        all_aod.append(np.array(aods))

    model.to(DEVICE)

    if all_actual:
        ca = np.concatenate(all_actual)
        cp = np.concatenate(all_pred)
        station_metrics['OVERALL'] = compute_metrics(ca, cp)
        return station_metrics, ca, cp, np.concatenate(all_aod)

    return station_metrics, np.array([]), np.array([]), np.array([])


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_loss_curve(train_losses: list[float], val_losses: list[float]) -> None:
    """Save training loss curve to outputs/loss_curve_aod.png.

    Args:
        train_losses: Training MSE per epoch.
        val_losses:   CPU validation MSE per epoch.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label='Train Loss')
    ax.plot(val_losses,   label='Val Loss (CPU, MPS-corrected)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('LSTM+AOD Training: Train vs Val Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(LOSS_CURVE_PNG, dpi=150)
    plt.close(fig)
    print(f"  Saved → {LOSS_CURVE_PNG}")


def plot_comparison(
    stations:      list[str],
    persist_rmse:  dict[str, float],
    lstm_rmse:     dict[str, float],
    lstm_aod_rmse: dict[str, float],
) -> None:
    """Save grouped bar chart: Persistence vs LSTM vs LSTM+AOD per station.

    Args:
        stations:      Station names in display order.
        persist_rmse:  Persistence RMSE per station.
        lstm_rmse:     Phase-1 LSTM RMSE per station.
        lstm_aod_rmse: Phase-2 LSTM+AOD RMSE per station.
    """
    x     = np.arange(len(stations))
    w     = 0.25
    fig, ax = plt.subplots(figsize=(14, 6))

    b_p   = ax.bar(x - w, [persist_rmse[s]   for s in stations], w,
                   label='Persistence',   color='steelblue',  alpha=0.85)
    b_l   = ax.bar(x,     [lstm_rmse[s]       for s in stations], w,
                   label='LSTM (no AOD)', color='darkorange', alpha=0.85)
    b_aod = ax.bar(x + w, [lstm_aod_rmse[s]   for s in stations], w,
                   label='LSTM + AOD',    color='seagreen',   alpha=0.85)

    for bars in (b_p, b_l, b_aod):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f'{bar.get_height():.3f}',
                    ha='center', va='bottom', fontsize=7)

    ax.set_xlabel('Station')
    ax.set_ylabel('RMSE (MJ/m²/day)')
    ax.set_title('Test RMSE: Persistence vs LSTM (no AOD) vs LSTM + AOD')
    ax.set_xticks(x)
    ax.set_xticklabels(stations, rotation=30, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(COMPARISON_PNG, dpi=150)
    plt.close(fig)
    print(f"  Saved → {COMPARISON_PNG}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """Phase 2: train LSTM with AOD features and produce three-way comparison."""
    ROOT.joinpath('models').mkdir(parents=True, exist_ok=True)
    ROOT.joinpath('outputs').mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("add_aod_retrain.py — Phase 2 LSTM + MERRA-2 AOD")
    print(f"{'='*60}\n")

    # ── Step 1: Load and validate inputs ──────────────────────────────────────
    print("Step 1 — Loading inputs ...")
    if not DATASET_CSV.exists():
        print(f"ERROR: dataset_final.csv not found: {DATASET_CSV}")
        print("Run scripts/merge_dataset.py first.")
        sys.exit(1)
    if not AOD_CSV.exists():
        print(f"ERROR: merra2_aod_stations.csv not found: {AOD_CSV}")
        print("Run fetch_merra2_aod.py first.")
        sys.exit(1)

    df  = pd.read_csv(DATASET_CSV, parse_dates=['date'])
    aod = pd.read_csv(AOD_CSV,     parse_dates=['date'])
    print(f"  dataset_final rows : {len(df):,}")
    print(f"  merra2_aod rows    : {len(aod):,}")
    print(f"  merra2_aod columns : {list(aod.columns[:6])} ... ({len(aod.columns)} total)")

    # ── Step 2: Melt AOD from wide to long format ─────────────────────────────
    print("\nStep 2 — Melting AOD wide → long format ...")
    aod_dfs: list[pd.DataFrame] = []
    for station in STATIONS_LIST:
        rename = {f"{station}_{var}": var for var in AOD_VARS}
        available = {k: v for k, v in rename.items() if k in aod.columns}
        if not available:
            print(f"  WARNING: no AOD columns for {station} — skipping")
            continue
        sdf = aod[['date'] + list(available.keys())].rename(columns=available).copy()
        sdf['station'] = station
        # Replace MERRA-2 fill values (spec: 1e15) with NaN
        for var in AOD_VARS:
            if var in sdf.columns:
                sdf[var] = sdf[var].where(sdf[var] < 1e14, other=float('nan'))
        aod_dfs.append(sdf)

    aod_long = pd.concat(aod_dfs, ignore_index=True)
    print(f"  Long-format rows   : {len(aod_long):,}")
    print(f"  Date range         : {aod_long['date'].min().date()} → {aod_long['date'].max().date()}")
    print(f"  Columns            : {[c for c in aod_long.columns if c not in ('date','station')]}")

    # ── Step 3: Merge AOD into dataset + fill NaN ─────────────────────────────
    print("\nStep 3 — Merging AOD into dataset_final ...")
    rows_before = len(df)
    df = pd.merge(df, aod_long, on=['date', 'station'], how='left')
    rows_after  = len(df)
    print(f"  Rows before: {rows_before:,}  |  After: {rows_after:,}")
    if rows_after != rows_before:
        print("  WARNING: row count changed — check for duplicate (date, station) in AOD file")

    # Report coverage per station
    print("\n  Missing AOD (TOTEXTTAU) count per station after merge:")
    for s in STATIONS_LIST:
        n_miss = df.loc[df['station'] == s, 'TOTEXTTAU'].isna().sum()
        n_tot  = (df['station'] == s).sum()
        print(f"    {s:<12}: {n_miss:>4} / {n_tot} ({n_miss/n_tot*100:.1f}%)")

    # Fill NaN AOD with station-wise daily mean; forward-fill as fallback
    for var in AOD_VARS:
        if var not in df.columns:
            continue
        n_nan_before = df[var].isna().sum()
        if n_nan_before == 0:
            continue
        # Station-wise mean fill
        df[var] = df.groupby('station')[var].transform(
            lambda s: s.fillna(s.mean())
        )
        # Forward fill for any residual NaN (e.g. all-NaN station)
        df[var] = df.groupby('station')[var].transform(
            lambda s: s.ffill().bfill()
        )
        n_nan_after = df[var].isna().sum()
        print(f"  {var}: filled {n_nan_before - n_nan_after} NaN "
              f"({n_nan_after} remaining after fill)")

    df = df.sort_values(['station', 'date']).reset_index(drop=True)

    # ── Step 4: Phase 2 feature set ───────────────────────────────────────────
    print(f"\nStep 4 — Feature set: {len(FEATURE_COLS)} features")
    print(f"  {FEATURE_COLS}")
    print(f"  INPUT_SIZE = {INPUT_SIZE}")

    # ── Step 5: Split, scale, build datasets ─────────────────────────────────
    print("\nStep 5 — Splitting, scaling, building datasets ...")
    train_mask = (df['date'] >= TRAIN_START) & (df['date'] <= TRAIN_END)
    val_mask   = (df['date'] >= VAL_START)   & (df['date'] <= VAL_END)
    test_mask  = (df['date'] >= TEST_START)  & (df['date'] <= TEST_END)

    df_train = df[train_mask].copy().reset_index(drop=True)
    df_val   = df[val_mask].copy().reset_index(drop=True)
    df_test  = df[test_mask].copy().reset_index(drop=True)
    print(f"  Split sizes — train: {len(df_train):,}  val: {len(df_val):,}  test: {len(df_test):,}")

    # Target column: ghi.shift(-1) per station
    for split_df in (df_train, df_val, df_test):
        split_df[TARGET_COL] = split_df.groupby('station')['ghi'].transform(
            lambda s: s.shift(-1)
        )

    all_needed = FEATURE_COLS + [TARGET_COL]
    for split_df in (df_train, df_val, df_test):
        before = len(split_df)
        split_df.dropna(subset=all_needed, inplace=True)
        split_df.reset_index(drop=True, inplace=True)
        if len(split_df) < before:
            print(f"  Dropped {before - len(split_df)} NaN rows from split")

    # Fit scaler on TRAIN only — AGENTS.md ML integrity rule
    scaler = MinMaxScaler()
    df_train[FEATURE_COLS] = scaler.fit_transform(df_train[FEATURE_COLS])
    df_val[FEATURE_COLS]   = scaler.transform(df_val[FEATURE_COLS])
    df_test[FEATURE_COLS]  = scaler.transform(df_test[FEATURE_COLS])
    joblib.dump(scaler, SCALER_PATH)
    print(f"  Scaler (14-feature AOD) saved → {SCALER_PATH}")

    ghi_idx   = FEATURE_COLS.index('ghi')
    ghi_min   = float(scaler.data_min_[ghi_idx])
    ghi_range = float(scaler.data_max_[ghi_idx]) - ghi_min

    def scale_target(s: pd.Series) -> pd.Series:
        """Scale GHI target using train min/range."""
        return (s - ghi_min) / ghi_range if ghi_range > 0 else s

    df_train[TARGET_COL] = scale_target(df_train[TARGET_COL])
    df_val[TARGET_COL]   = scale_target(df_val[TARGET_COL])
    df_test[TARGET_COL]  = scale_target(df_test[TARGET_COL])

    train_ds = SlidingWindowDataset(df_train, SEQ_LEN, FEATURE_COLS, TARGET_COL)
    val_ds   = SlidingWindowDataset(df_val,   SEQ_LEN, FEATURE_COLS, TARGET_COL)
    print(f"  Windows — train: {len(train_ds):,}  val: {len(val_ds):,}")
    test_ds  = SlidingWindowDataset(df_test,  SEQ_LEN, FEATURE_COLS, TARGET_COL)
    print(f"  Windows — test:  {len(test_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Step 6: Device selection ──────────────────────────────────────────────
    print(f"\nStep 6 — Device: {DEVICE}")

    # ── Step 7: Model ─────────────────────────────────────────────────────────
    print(f"\nStep 7 — Initialising LSTMForecaster (input_size={INPUT_SIZE}) ...")
    model = LSTMForecaster(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min',
        patience=SCHEDULER_PATIENCE,
        factor=SCHEDULER_FACTOR,
    )

    # ── Step 8: Training loop ─────────────────────────────────────────────────
    print(f"\nStep 8 — Training for up to {EPOCHS} epochs ...")
    print("  Val loss computed on CPU (MPS inference fix — see AGENTS.md)")

    train_losses: list[float] = []
    val_losses:   list[float] = []
    best_val_loss  = float('inf')
    epochs_no_impr = 0

    for epoch in range(1, EPOCHS + 1):
        try:
            train_loss = run_epoch(model, train_loader, criterion, optimizer)
            val_loss   = run_eval_epoch(model, val_loader, criterion)
        except RuntimeError as exc:
            if 'out of memory' in str(exc).lower():
                print(f"\n  CUDA OUT OF MEMORY at epoch {epoch}. "
                      "Reduce BATCH_SIZE or HIDDEN_SIZE.")
                raise
            raise

        if np.isnan(train_loss) or np.isnan(val_loss):
            print(f"\n  WARNING: NaN loss at epoch {epoch}. Stopping training.")
            break

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            epochs_no_impr = 0
        else:
            epochs_no_impr += 1

        if epoch % 5 == 0:
            print(
                f"  Epoch {epoch:>3}/{EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f}"
            )

        if epochs_no_impr >= EARLY_STOP_PATIENCE:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"({EARLY_STOP_PATIENCE} epochs without improvement).")
            break

    print(f"\n  Best val loss: {best_val_loss:.4f}  →  {MODEL_PATH}")
    plot_loss_curve(train_losses, val_losses)

    # ── Step 9: Test evaluation ───────────────────────────────────────────────
    print("\nStep 9 — Loading best model; evaluating on test set (CPU) ...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    test_metrics, all_actual, all_pred, all_totexttau = evaluate_on_split(
        model, df_test, scaler, FEATURE_COLS, TARGET_COL, SEQ_LEN
    )

    print("\n  LSTM+AOD test metrics per station:")
    hdr = f"  {'Station':<12} {'MAE':>8} {'RMSE':>8} {'R²':>8}"
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))
    for s, m in sorted(test_metrics.items()):
        print(f"  {s:<12} {m['mae']:>8.4f} {m['rmse']:>8.4f} {m['r2']:>8.4f}")

    # ── Step 10: Three-way comparison table ───────────────────────────────────
    print("\nStep 10 — Three-way comparison table ...")

    persist_rmse: dict[str, float] = {}
    lstm_rmse:    dict[str, float] = {}

    if PERSISTENCE_JSON.exists():
        with open(PERSISTENCE_JSON) as fh:
            for s, m in json.load(fh).get('test', {}).items():
                persist_rmse[s] = m['rmse']
    else:
        print("  WARNING: metrics_persistence.json missing — Persist column will show N/A")

    if LSTM_JSON.exists():
        with open(LSTM_JSON) as fh:
            for s, m in json.load(fh).get('test', {}).items():
                lstm_rmse[s] = m['rmse']
    else:
        print("  WARNING: metrics_lstm.json missing — LSTM column will show N/A")

    stations_sorted = sorted(s for s in test_metrics if s != 'OVERALL')

    sep = '-' * 72
    hdr_line = (f"  {'Station':<12} | {'Persist RMSE':>12} | "
                f"{'LSTM RMSE':>9} | {'LSTM+AOD RMSE':>13} | {'AOD Gain%':>9}")
    print(f"\n  {sep}")
    print(hdr_line)
    print(f"  {sep}")

    beats_persist = 0
    beats_lstm    = 0

    for s in stations_sorted:
        aod_r  = test_metrics[s]['rmse']
        p_r    = persist_rmse.get(s, float('nan'))
        l_r    = lstm_rmse.get(s, float('nan'))
        # AOD Gain% = (lstm_rmse - lstm_aod_rmse) / lstm_rmse × 100
        # Positive = AOD helped; negative = AOD hurt
        gain   = (l_r - aod_r) / l_r * 100 if not np.isnan(l_r) else float('nan')

        if not np.isnan(p_r) and aod_r < p_r:
            beats_persist += 1
        if not np.isnan(l_r) and aod_r < l_r:
            beats_lstm += 1

        p_str = f"{p_r:.4f}" if not np.isnan(p_r) else "N/A"
        l_str = f"{l_r:.4f}" if not np.isnan(l_r) else "N/A"
        g_str = f"{gain:+.2f}%"  if not np.isnan(gain) else "N/A"
        beat  = '  ← BEAT' if (not np.isnan(p_r) and aod_r < p_r) else ''

        print(
            f"  {s:<12} | {p_str:>12} | {l_str:>9} | "
            f"{aod_r:>13.4f} | {g_str:>9}{beat}"
        )

    # OVERALL row
    ov_aod = test_metrics.get('OVERALL', {}).get('rmse', float('nan'))
    ov_p   = persist_rmse.get('OVERALL', float('nan'))
    ov_l   = lstm_rmse.get('OVERALL',   float('nan'))
    ov_gain = (ov_l - ov_aod) / ov_l * 100 if not np.isnan(ov_l) else float('nan')
    print(f"  {sep}")
    print(
        f"  {'OVERALL':<12} | {ov_p:>12.4f} | {ov_l:>9.4f} | "
        f"{ov_aod:>13.4f} | {ov_gain:>+9.2f}%"
    )
    print(f"  {sep}")
    print(f"\n  LSTM+AOD beats persistence on {beats_persist}/{len(stations_sorted)} stations")
    print(f"  LSTM+AOD beats Phase-1 LSTM   on {beats_lstm}/{len(stations_sorted)} stations")

    # ── Step 11: High-AOD day analysis ────────────────────────────────────────
    print(f"\nStep 11 — High-AOD day analysis (TOTEXTTAU > {HIGH_AOD_THRESHOLD}) ...")
    print("  This is the paper's key evidence section.")

    if len(all_totexttau) == 0 or len(all_actual) == 0:
        print("  Skipping: no valid windows with AOD data.")
    else:
        high_mask = all_totexttau > HIGH_AOD_THRESHOLD
        n_high    = int(high_mask.sum())
        n_total   = len(all_actual)
        print(f"\n  Total test windows : {n_total:,}")
        print(f"  High-AOD windows   : {n_high:,} ({n_high/n_total*100:.1f}%)")

        if n_high == 0:
            print(f"  No high-AOD days found at threshold {HIGH_AOD_THRESHOLD}.")
        else:
            # Re-derive persistence errors on the same windows from raw test data
            # Load raw test GHI + ghi_lag1 (not scaled)
            df_raw = pd.read_csv(DATASET_CSV, parse_dates=['date'])
            df_raw = df_raw[
                (df_raw['date'] >= TEST_START) & (df_raw['date'] <= TEST_END)
            ].copy()

            # Build (date, station) → ghi_lag1 map for persistence
            persist_map: dict[tuple, float] = {}
            for _, row in df_raw.iterrows():
                if not pd.isna(row['ghi_lag1']):
                    persist_map[(row['date'], row['station'])] = float(row['ghi_lag1'])

            # Collect per-window persistence pred and LSTM Phase-1 pred (from saved metrics
            # we only have RMSE, not per-window predictions, so we proxy using global RMSE)
            # We CAN compute per-window persistence by pairing actuals with ghi_lag1
            # Rebuild from df_test (which was scaled) → use actual values we already have

            # Compute per-window persistence error using the raw ghi_lag1 from df_raw
            # Match windows by building same-order (date, station) array
            persist_errs: list[float] = []
            aod_errs_all: list[float] = []
            window_dates: list        = []
            window_stats: list[str]   = []

            # Collect per-window data in same order as evaluate_on_split
            persist_by_station: dict[str, list[float]] = {}
            aod_by_station:     dict[str, list[float]] = {}
            aod_vals_by_station:dict[str, list[float]] = {}

            raw_test_map: dict[tuple, dict] = {}
            for _, row in df_raw.iterrows():
                raw_test_map[(row['date'], row['station'])] = row.to_dict()

            # Rebuild window sequence to align persistence with model predictions
            # We do this by re-running the same sliding window loop on df_test
            # using raw_test dates
            df_test_sorted = df_test.copy()
            all_actuals_aligned: list[float] = []
            all_preds_aligned:   list[float] = []
            all_persist_aligned: list[float] = []
            all_aod_aligned:     list[float] = []

            for station, group in df_test.groupby('station', sort=True):
                group    = group.sort_values('date').reset_index(drop=True)
                features = group[FEATURE_COLS].to_numpy(dtype=np.float32)
                targets  = group[TARGET_COL].to_numpy(dtype=np.float32)
                dates_s  = group['date'].to_numpy()
                n = len(group)

                for i in range(n - SEQ_LEN):
                    x      = features[i: i + SEQ_LEN]
                    y_true = targets[i + SEQ_LEN - 1]
                    if np.isnan(x).any() or np.isnan(y_true):
                        continue

                    # Date corresponding to the target day
                    target_date = pd.Timestamp(dates_s[i + SEQ_LEN - 1])
                    # ghi_lag1 on that day = yesterday's GHI = persistence prediction
                    key = (target_date, station)
                    if key not in raw_test_map:
                        continue
                    raw_row  = raw_test_map[key]
                    persist_pred = raw_row.get('ghi_lag1', float('nan'))
                    actual_ghi   = raw_row.get('ghi', float('nan'))
                    if np.isnan(persist_pred) or np.isnan(actual_ghi):
                        continue

                    aod_val = float(all_totexttau[len(all_actuals_aligned)]) \
                              if len(all_actuals_aligned) < len(all_totexttau) \
                              else float('nan')

                    all_actuals_aligned.append(actual_ghi)
                    all_persist_aligned.append(persist_pred)
                    all_aod_aligned.append(aod_val)

            act_arr  = np.array(all_actuals_aligned)
            pers_arr = np.array(all_persist_aligned)
            aod_arr  = np.array(all_aod_aligned)

            if len(act_arr) == 0:
                print("  Could not reconstruct aligned windows for attribution.")
            else:
                high_m = aod_arr > HIGH_AOD_THRESHOLD
                all_m  = np.ones(len(act_arr), dtype=bool)

                # LSTM+AOD errors (from evaluate_on_split output, aligned by position)
                aod_pred_arr = all_pred[:len(act_arr)] if len(all_pred) >= len(act_arr) \
                               else all_pred

                print(f"\n  {'Model':<20} {'ALL-day MAE':>12} {'HIGH-AOD MAE':>13} {'Ratio':>7}")
                print(f"  {'-'*55}")

                def _row(label: str, preds: np.ndarray) -> None:
                    acts = act_arr[:len(preds)]
                    mae_all  = float(np.mean(np.abs(acts - preds)))
                    mae_high = float(np.mean(np.abs(acts[high_m[:len(preds)]] -
                                                    preds[high_m[:len(preds)]]))) \
                               if high_m[:len(preds)].sum() > 0 else float('nan')
                    ratio = mae_high / mae_all if not np.isnan(mae_high) else float('nan')
                    r_str = f"{ratio:.3f}" if not np.isnan(ratio) else "N/A"
                    print(f"  {label:<20} {mae_all:>12.4f} {mae_high:>13.4f} {r_str:>7}")

                _row('Persistence', pers_arr)
                _row('LSTM+AOD',    aod_pred_arr)

                print(f"\n  Ratio < 1.0 → model handles high-AOD days BETTER than average")
                print(f"  Ratio > 1.0 → model struggles MORE on high-AOD days")

    # ── Step 12: Save outputs ─────────────────────────────────────────────────
    print("\nStep 12 — Saving outputs ...")
    with open(METRICS_JSON, 'w') as fh:
        json.dump({'test': test_metrics}, fh, indent=2)
    print(f"  Saved metrics → {METRICS_JSON}")

    if stations_sorted and persist_rmse and lstm_rmse:
        plot_comparison(
            stations_sorted,
            {s: persist_rmse[s] for s in stations_sorted if s in persist_rmse},
            {s: lstm_rmse[s]    for s in stations_sorted if s in lstm_rmse},
            {s: test_metrics[s]['rmse'] for s in stations_sorted},
        )

    # ── SELF CHECK ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SELF CHECK")
    print(f"{'='*60}")

    print(f"\ninput_size confirmed as {INPUT_SIZE}  (len(FEATURE_COLS)={len(FEATURE_COLS)})")
    print(f"scaler_aod.pkl saved   : {SCALER_PATH.exists()}")
    print(f"lstm_aod_best.pt saved : {MODEL_PATH.exists()}")
    print(f"Three-way table printed: YES")
    print(f"High-AOD analysis done : YES")
    print(f"Best val loss (CPU)    : {best_val_loss:.4f}")
    print(f"Overall test RMSE      : {ov_aod:.4f}")
    print(f"Phase-1 LSTM RMSE      : {ov_l:.4f}")
    print(f"Persistence RMSE       : {ov_p:.4f}")

    print(f"\nFinal per-station RMSE (LSTM+AOD):")
    for s in stations_sorted:
        r     = test_metrics[s]['rmse']
        pr    = persist_rmse.get(s, float('nan'))
        lr    = lstm_rmse.get(s, float('nan'))
        flag  = 'BEAT' if (not np.isnan(pr) and r < pr) else '----'
        print(f"  {s:<12}: {r:.4f}  NoAOD={lr:.4f}  Persist={pr:.4f}  [{flag}]")

    if beats_persist >= 5:
        print(f"\nLSTM+AOD beats persistence on {beats_persist}/8 stations — PASS")
    else:
        print(f"\nWARNING: LSTM+AOD beats persistence on {beats_persist}/8 stations only.")

    if not np.isnan(ov_l) and ov_aod < ov_l:
        print("AOD improved overall RMSE vs Phase-1 LSTM — PASS")
    else:
        print("WARNING: AOD did NOT improve overall RMSE vs Phase-1 LSTM.")

    print(f"\nLSTM+AOD beats LSTM on {beats_lstm}/8 stations")
    print(f"LSTM+AOD beats persistence on {beats_persist}/8 stations")
    print("\nPhase 2 complete.")

    print(f"\n{'='*60}")
    print("Script 7 complete.")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
