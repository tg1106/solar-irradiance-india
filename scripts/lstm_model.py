"""
lstm_model.py
Train a 2-layer LSTM to forecast next-day GHI using meteorological
variables, cyclical time features, and monsoon phase labels.
Model A (Phase 1) — no AOD; AOD is added in Phase 2 as Model B.
Saves: models/lstm_best.pt, models/scaler.pkl,
       outputs/metrics_lstm.json, outputs/loss_curve.png,
       outputs/predictions_sample.png
"""

from __future__ import annotations

import json
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

DATASET_CSV       = ROOT / 'data' / 'processed' / 'dataset_final.csv'
PERSISTENCE_JSON  = ROOT / 'outputs' / 'metrics_persistence.json'
MODEL_PATH        = ROOT / 'models' / 'lstm_best.pt'
SCALER_PATH       = ROOT / 'models' / 'scaler.pkl'
METRICS_JSON      = ROOT / 'outputs' / 'metrics_lstm.json'
LOSS_CURVE_PNG    = ROOT / 'outputs' / 'loss_curve.png'
PREDICTIONS_PNG   = ROOT / 'outputs' / 'predictions_sample.png'

TRAIN_START = "2017-04-01"
TRAIN_END   = "2022-12-31"
VAL_START   = "2023-01-01"
VAL_END     = "2023-12-31"
TEST_START  = "2024-01-01"
TEST_END    = "2024-12-31"

FEATURE_COLS = [
    'ghi',
    'temperature',
    'humidity',
    'wind_speed',
    'cloud_cover',
    'monsoon_phase',
    'doy_sin',
    'doy_cos',
    'month_sin',
    'month_cos',
]
TARGET_COL = 'ghi_target'

SEQ_LEN     = 7     # Fix 1: 30→7, GHI autocorrelation is 1-3 day, not 30-day
INPUT_SIZE  = len(FEATURE_COLS)   # 10
HIDDEN_SIZE = 128   # 128: smaller model converges faster with this dataset size
NUM_LAYERS  = 2
DROPOUT     = 0.2
EPOCHS      = 50
BATCH_SIZE  = 32
LR          = 0.001   # 0.001: reverted — LR=0.0003 was too slow to converge in 50 epochs
GRAD_CLIP   = 1.0     # Fix 4: gradient clipping norm
SCHEDULER_PATIENCE = 5
SCHEDULER_FACTOR   = 0.5
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
            x: Input tensor of shape [batch, seq_len, features].

        Returns:
            Prediction tensor of shape [batch, 1].
        """
        out, _ = self.lstm(x)
        pred = self.fc(out[:, -1, :])   # use last timestep
        return pred


# ── Dataset ───────────────────────────────────────────────────────────────────

class SlidingWindowDataset(Dataset):
    """Sliding-window dataset that never crosses station or split boundaries.

    For each station the windows are built independently, guaranteeing
    that no window can straddle two stations or leak across train/val/test.

    Args:
        df:           DataFrame for a single split (already filtered).
        seq_len:      Number of historical timesteps per window.
        feature_cols: Column names used as input features.
        target_col:   Column name for the regression target.
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
        """Pre-build all (X, y) windows from the dataframe.

        Args:
            df: Split DataFrame, already sorted by station then date.
        """
        for station, group in df.groupby('station', sort=True):
            group = group.sort_values('date').reset_index(drop=True)
            features = group[self.feature_cols].to_numpy(dtype=np.float32)
            targets  = group[self.target_col].to_numpy(dtype=np.float32)
            n = len(group)
            for i in range(self.seq_len, n):
                x = features[i - self.seq_len: i]
                # Fix 2: targets[i-1] = ghi[i] = tomorrow of window's last day (i-1).
                # Original targets[i] = ghi[i+1] was forecasting 2 days ahead, not 1.
                y = targets[i - 1]
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
    model: LSTMForecaster,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    train: bool,
) -> float:
    """Run one full epoch (training or evaluation).

    Args:
        model:     LSTMForecaster instance.
        loader:    DataLoader for the split.
        criterion: Loss function (MSELoss).
        optimizer: Adam optimizer (None during eval).
        train:     Whether to update weights.

    Returns:
        Mean loss over the epoch.
    """
    model.train(train)
    total_loss = 0.0
    n_batches  = 0
    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)
            loss = criterion(pred, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)  # Fix 4
                optimizer.step()
            total_loss += loss.item()
            n_batches  += 1
    return total_loss / max(n_batches, 1)


# ── CPU evaluation — MPS inference fix ────────────────────────────────────────

def run_eval_epoch(
    model: LSTMForecaster,
    loader: DataLoader,
    criterion: nn.Module,
) -> float:
    """Evaluate on CPU to avoid the MPS inference artifact.

    Apple Silicon MPS reports ~2.5x lower loss than the true value during
    torch.no_grad() passes, corrupting early stopping and checkpointing.
    Moving the model to CPU for evaluation gives the accurate number.
    The model is moved back to DEVICE after evaluation.

    Uses sample-weighted mean (not mean-of-batch-means) to handle the
    smaller last batch correctly.

    Args:
        model:     LSTMForecaster instance (will be moved CPU→DEVICE).
        loader:    DataLoader for the evaluation split.
        criterion: Loss function (MSELoss).

    Returns:
        Sample-weighted mean loss over the full split.
    """
    # Exact AGENTS.md MPS inference fix pattern
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
    model: LSTMForecaster,
    df_split: pd.DataFrame,
    scaler: MinMaxScaler,
) -> dict[str, dict[str, float]]:
    """Evaluate model on a split and return per-station + overall metrics.

    Predictions are inverse-transformed back to MJ/m²/day before
    computing metrics.

    Args:
        model:    Trained LSTMForecaster (best checkpoint loaded).
        df_split: DataFrame for the split containing scaled features
                  and the target column.
        scaler:   Fitted MinMaxScaler for inverse-transforming GHI.

    Returns:
        Dict keyed by station name (plus 'OVERALL') with mae/rmse/r2.
    """
    # Exact AGENTS.md MPS inference fix — move to CPU for all forward passes
    model.eval()
    eval_device = torch.device('cpu')
    model_cpu   = model.to(eval_device)

    all_actual: list[np.ndarray]  = []
    all_pred:   list[np.ndarray]  = []
    station_metrics: dict[str, dict[str, float]] = {}

    ghi_col_idx = FEATURE_COLS.index('ghi')

    for station, group in df_split.groupby('station', sort=True):
        group = group.sort_values('date').reset_index(drop=True)
        features = group[FEATURE_COLS].to_numpy(dtype=np.float32)
        targets  = group[TARGET_COL].to_numpy(dtype=np.float32)
        n = len(group)

        actuals_list: list[float] = []
        preds_list:   list[float] = []

        with torch.no_grad():
            for i in range(SEQ_LEN, n):
                x = features[i - SEQ_LEN: i]
                # targets[i-1] ≡ targets[i + SEQ_LEN - 1] in 0-based notation
                # = ghi[i]: 1 day ahead of window end.  See AGENTS.md.
                y_true = targets[i - 1]
                if np.isnan(x).any() or np.isnan(y_true):
                    continue
                x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(eval_device)
                y_pred_scaled = model_cpu(x_t).numpy().flatten()[0]

                # Inverse transform: rebuild a dummy row with GHI in its slot
                dummy = np.zeros((1, len(FEATURE_COLS)), dtype=np.float32)
                dummy[0, ghi_col_idx] = y_pred_scaled
                y_pred = scaler.inverse_transform(dummy)[0, ghi_col_idx]

                dummy_true = np.zeros((1, len(FEATURE_COLS)), dtype=np.float32)
                dummy_true[0, ghi_col_idx] = y_true
                y_actual = scaler.inverse_transform(dummy_true)[0, ghi_col_idx]

                actuals_list.append(y_actual)
                preds_list.append(y_pred)

        if not actuals_list:
            print(f"  WARNING: no valid windows for {station}")
            continue

        actual_arr = np.array(actuals_list)
        pred_arr   = np.array(preds_list)
        station_metrics[station] = compute_metrics(actual_arr, pred_arr)
        all_actual.append(actual_arr)
        all_pred.append(pred_arr)

    if all_actual:
        combined_actual = np.concatenate(all_actual)
        combined_pred   = np.concatenate(all_pred)
        station_metrics['OVERALL'] = compute_metrics(combined_actual, combined_pred)

    model.to(DEVICE)
    return station_metrics


# ── Plotting helpers ───────────────────────────────────────────────────────────

def plot_loss_curve(
    train_losses: list[float],
    val_losses:   list[float],
) -> None:
    """Save train vs val loss per epoch to outputs/loss_curve.png.

    Args:
        train_losses: List of training loss values per epoch.
        val_losses:   List of validation loss values per epoch.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label='Train Loss')
    ax.plot(val_losses,   label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('LSTM Training: Train vs Val Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(LOSS_CURVE_PNG, dpi=150)
    plt.close(fig)
    print(f"  Saved loss curve → {LOSS_CURVE_PNG}")


def plot_predictions_sample(
    model:    LSTMForecaster,
    df_test:  pd.DataFrame,
    scaler:   MinMaxScaler,
) -> None:
    """Save actual vs predicted GHI for Chennai in 2024 test year.

    Args:
        model:   Trained LSTMForecaster.
        df_test: Test set DataFrame with scaled features.
        scaler:  Fitted MinMaxScaler.
    """
    # Exact AGENTS.md MPS inference fix — move to CPU for all forward passes
    model.eval()
    eval_device = torch.device('cpu')
    model_cpu   = model.to(eval_device)
    ghi_col_idx = FEATURE_COLS.index('ghi')

    station = 'Chennai'
    group = df_test[df_test['station'] == station].sort_values('date').reset_index(drop=True)
    features = group[FEATURE_COLS].to_numpy(dtype=np.float32)
    targets  = group[TARGET_COL].to_numpy(dtype=np.float32)
    dates    = group['date'].to_numpy()
    n = len(group)

    plot_dates:   list = []
    actuals_list: list[float] = []
    preds_list:   list[float] = []

    with torch.no_grad():
        for i in range(SEQ_LEN, n):
            x = features[i - SEQ_LEN: i]
            y_true = targets[i - 1]
            if np.isnan(x).any() or np.isnan(y_true):
                continue
            x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(eval_device)
            y_pred_scaled = model_cpu(x_t).numpy().flatten()[0]

            dummy = np.zeros((1, len(FEATURE_COLS)), dtype=np.float32)
            dummy[0, ghi_col_idx] = y_pred_scaled
            y_pred = scaler.inverse_transform(dummy)[0, ghi_col_idx]

            dummy_true = np.zeros((1, len(FEATURE_COLS)), dtype=np.float32)
            dummy_true[0, ghi_col_idx] = y_true
            y_actual = scaler.inverse_transform(dummy_true)[0, ghi_col_idx]

            plot_dates.append(dates[i])
            actuals_list.append(y_actual)
            preds_list.append(y_pred)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(plot_dates, actuals_list, label='Actual GHI',    alpha=0.8, linewidth=1.2)
    ax.plot(plot_dates, preds_list,   label='Predicted GHI', alpha=0.8, linewidth=1.2,
            linestyle='--')
    ax.set_xlabel('Date')
    ax.set_ylabel('GHI (MJ/m²/day)')
    ax.set_title(f'LSTM Predictions vs Actual — {station} Test 2024')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PREDICTIONS_PNG, dpi=150)
    plt.close(fig)
    model.to(DEVICE)
    print(f"  Saved prediction plot → {PREDICTIONS_PNG}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """Full training + evaluation pipeline for LSTMForecaster (Phase 1)."""
    ROOT.joinpath('models').mkdir(parents=True, exist_ok=True)
    ROOT.joinpath('outputs').mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("lstm_model.py — Phase 1 LSTM Forecaster")
    print(f"{'='*60}\n")

    # ── Step 1: Load and split ─────────────────────────────────────────────────
    print("Step 1 — Loading dataset ...")
    if not DATASET_CSV.exists():
        raise FileNotFoundError(
            f"Final dataset not found: {DATASET_CSV}\n"
            "Run scripts/merge_dataset.py first."
        )
    df = pd.read_csv(DATASET_CSV, parse_dates=['date'])
    print(f"  Total rows: {len(df):,}")

    train_mask = (df['date'] >= TRAIN_START) & (df['date'] <= TRAIN_END)
    val_mask   = (df['date'] >= VAL_START)   & (df['date'] <= VAL_END)
    test_mask  = (df['date'] >= TEST_START)  & (df['date'] <= TEST_END)

    df_train = df[train_mask].copy().reset_index(drop=True)
    df_val   = df[val_mask].copy().reset_index(drop=True)
    df_test  = df[test_mask].copy().reset_index(drop=True)
    print(f"  Train rows: {len(df_train):,}")
    print(f"  Val   rows: {len(df_val):,}")
    print(f"  Test  rows: {len(df_test):,}")

    # ── Step 2: Target column ──────────────────────────────────────────────────
    print("\nStep 2 — Building target column (ghi shifted -1 per station) ...")
    for split_df in (df_train, df_val, df_test):
        split_df[TARGET_COL] = split_df.groupby('station')['ghi'].transform(
            lambda s: s.shift(-1)
        )

    # Drop last row per station (no target)
    df_train = df_train.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    df_val   = df_val.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    df_test  = df_test.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    # ── Fix 2 — Target verification ───────────────────────────────────────────
    print("\nFix 2 — Target column verification (first 20 rows, Jodhpur train):")
    print(f"  Confirm: ghi_target[i] = ghi[i+1] (tomorrow, not today, not day-after-tomorrow)")
    sdf_check = df_train[df_train['station'] == 'Jodhpur'].sort_values('date').reset_index(drop=True)
    ghi_vals    = sdf_check['ghi'].to_numpy()
    target_vals = sdf_check[TARGET_COL].to_numpy()
    print(f"  {'row':>4}  {'date':>12}  {'ghi[i]':>10}  {'target[i]':>10}  {'ghi[i+1]':>10}  {'ok?':>5}")
    print(f"  {'-'*60}")
    for idx in range(min(20, len(sdf_check) - 1)):
        d       = sdf_check['date'].iloc[idx].date()
        g       = ghi_vals[idx]
        t       = target_vals[idx]
        nxt     = ghi_vals[idx + 1]
        ok      = 'YES' if abs(t - nxt) < 1e-6 else 'MISMATCH'
        print(f"  {idx:>4}  {str(d):>12}  {g:>10.4f}  {t:>10.4f}  {nxt:>10.4f}  {ok:>5}")
    # Raise immediately if target column is corrupted
    mismatches = sum(
        1 for i in range(len(sdf_check) - 1)
        if abs(target_vals[i] - ghi_vals[i + 1]) > 1e-6
    )
    if mismatches:
        raise ValueError(f"Target column mismatch in {mismatches} rows — shift logic is broken!")
    print(f"  Target column verified: ghi_target == ghi.shift(-1) across all rows — PASS")

    # ── Step 3: Drop NaN rows ──────────────────────────────────────────────────
    print("\nStep 3 — Dropping NaN rows in features or target ...")
    all_needed = FEATURE_COLS + [TARGET_COL]
    for name, split_df in [('train', df_train), ('val', df_val), ('test', df_test)]:
        before = len(split_df)
        dropped = split_df[split_df[all_needed].isna().any(axis=1)]
        if len(dropped):
            print(f"  {name}: dropped {len(dropped)} NaN rows per station:")
            print(dropped.groupby('station').size().to_string())

    df_train = df_train.dropna(subset=all_needed).reset_index(drop=True)
    df_val   = df_val.dropna(subset=all_needed).reset_index(drop=True)
    df_test  = df_test.dropna(subset=all_needed).reset_index(drop=True)

    # ── Step 4: Scale features ─────────────────────────────────────────────────
    print("\nStep 4 — Fitting MinMaxScaler on TRAIN set only ...")
    scaler = MinMaxScaler()
    df_train[FEATURE_COLS] = scaler.fit_transform(df_train[FEATURE_COLS])
    df_val[FEATURE_COLS]   = scaler.transform(df_val[FEATURE_COLS])
    df_test[FEATURE_COLS]  = scaler.transform(df_test[FEATURE_COLS])
    joblib.dump(scaler, SCALER_PATH)
    print(f"  Scaler saved → {SCALER_PATH}")

    # Also scale the target for training (using ghi column index)
    ghi_col_idx = FEATURE_COLS.index('ghi')
    ghi_min = scaler.data_min_[ghi_col_idx]
    ghi_max = scaler.data_max_[ghi_col_idx]
    ghi_range = ghi_max - ghi_min

    def scale_target(series: pd.Series) -> pd.Series:
        return (series - ghi_min) / ghi_range if ghi_range > 0 else series

    df_train[TARGET_COL] = scale_target(df_train[TARGET_COL])
    df_val[TARGET_COL]   = scale_target(df_val[TARGET_COL])
    df_test[TARGET_COL]  = scale_target(df_test[TARGET_COL])

    # ── Step 5: Datasets and DataLoaders ──────────────────────────────────────
    print("\nStep 5 — Building SlidingWindowDatasets ...")
    train_ds = SlidingWindowDataset(df_train, SEQ_LEN, FEATURE_COLS, TARGET_COL)
    val_ds   = SlidingWindowDataset(df_val,   SEQ_LEN, FEATURE_COLS, TARGET_COL)
    test_ds  = SlidingWindowDataset(df_test,  SEQ_LEN, FEATURE_COLS, TARGET_COL)
    print(f"  Train windows: {len(train_ds):,}")
    print(f"  Val   windows: {len(val_ds):,}")
    print(f"  Test  windows: {len(test_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Step 7: Model ─────────────────────────────────────────────────────────
    print("\nStep 7 — Initialising LSTMForecaster ...")
    model = LSTMForecaster(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min',
        patience=SCHEDULER_PATIENCE,
        factor=SCHEDULER_FACTOR,
    )

    # ── Step 8: Training loop ─────────────────────────────────────────────────
    print(f"\nStep 8 — Training for up to {EPOCHS} epochs ...")
    train_losses: list[float] = []
    val_losses:   list[float] = []
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        try:
            train_loss = run_epoch(model, train_loader, criterion, optimizer, train=True)
            val_loss   = run_eval_epoch(model, val_loader, criterion)
        except RuntimeError as exc:
            if 'out of memory' in str(exc).lower():
                print(
                    f"\n  CUDA OUT OF MEMORY at epoch {epoch}. "
                    "Reduce BATCH_SIZE or HIDDEN_SIZE and restart."
                )
                raise
            raise

        if np.isnan(train_loss) or np.isnan(val_loss):
            print(
                f"\n  WARNING: NaN loss detected at epoch {epoch} "
                f"(train={train_loss}, val={val_loss}). "
                "Check learning rate and input scaling. Stopping training."
            )
            break

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 5 == 0:
            print(
                f"  Epoch {epoch:>3}/{EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f}"
            )

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\n  Early stopping triggered at epoch {epoch} "
                  f"(no improvement for {EARLY_STOP_PATIENCE} epochs).")
            break

    print(f"\n  Best val loss: {best_val_loss:.4f}  (model saved → {MODEL_PATH})")
    plot_loss_curve(train_losses, val_losses)

    # ── Step 9: Evaluation ────────────────────────────────────────────────────
    print("\nStep 9 — Loading best model and evaluating on test set ...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    test_metrics = evaluate_on_split(model, df_test, scaler)

    print("\nTest set metrics per station:")
    header = f"  {'Station':<12} {'MAE':>8} {'RMSE':>8} {'R²':>8}"
    print(header)
    print('  ' + '-' * (len(header) - 2))
    for station, m in sorted(test_metrics.items()):
        print(
            f"  {station:<12} "
            f"{m['mae']:>8.4f} "
            f"{m['rmse']:>8.4f} "
            f"{m['r2']:>8.4f}"
        )

    # ── Step 10: Comparison with persistence ──────────────────────────────────
    print("\nStep 10 — Comparing against persistence baseline ...")
    if PERSISTENCE_JSON.exists():
        with open(PERSISTENCE_JSON) as fh:
            persistence = json.load(fh)
        persist_test = persistence.get('test', {})

        print(f"\n  {'Station':<12} {'Pers. RMSE':>12} {'LSTM RMSE':>12} {'Improvement%':>14}")
        print('  ' + '-' * 54)
        beats = 0
        for station in sorted(test_metrics.keys()):
            if station == 'OVERALL':
                continue
            lstm_rmse = test_metrics[station]['rmse']
            if station in persist_test:
                p_rmse = persist_test[station]['rmse']
                improvement = (p_rmse - lstm_rmse) / p_rmse * 100
                symbol = '+' if improvement > 0 else ''
                if improvement > 0:
                    beats += 1
                print(
                    f"  {station:<12} "
                    f"{p_rmse:>12.4f} "
                    f"{lstm_rmse:>12.4f} "
                    f"{symbol}{improvement:>13.2f}%"
                )
            else:
                print(f"  {station:<12} {'N/A':>12} {lstm_rmse:>12.4f} {'N/A':>14}")
    else:
        print("  metrics_persistence.json not found — skipping comparison.")
        beats = 0

    # ── Step 11: Save metrics and plots ───────────────────────────────────────
    print("\nStep 11 — Saving outputs ...")
    with open(METRICS_JSON, 'w') as fh:
        json.dump({'test': test_metrics}, fh, indent=2)
    print(f"  Saved metrics → {METRICS_JSON}")

    plot_predictions_sample(model, df_test, scaler)

    # ── SELF CHECK ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SELF CHECK")
    print(f"{'='*60}")

    n_stations = sum(1 for k in test_metrics if k != 'OVERALL')
    print(f"\nFinal test RMSE per station vs persistence RMSE:")
    if PERSISTENCE_JSON.exists():
        persist_test_metrics = persistence.get('test', {})
        for station in sorted(test_metrics.keys()):
            if station == 'OVERALL':
                continue
            lstm_r = test_metrics[station]['rmse']
            p_r    = persist_test_metrics.get(station, {}).get('rmse', float('nan'))
            flag = 'BEAT' if lstm_r < p_r else '----'
            print(f"  {station:<12}: LSTM={lstm_r:.4f}  Persistence={p_r:.4f}  [{flag}]")

    if beats >= 5:
        print(f"\nLSTM beats persistence on {beats}/{n_stations} stations — PASS")
    else:
        print(
            f"\nWARNING — model underperforms persistence on "
            f"{n_stations - beats}/{n_stations} stations."
        )
        # ── SEQ_LEN scan: try 3 and 14 to find better context window ──────────
        print("\nRunning SEQ_LEN scan (15 epochs each) to find best context window ...")
        scan_results: dict[int, float] = {SEQ_LEN: best_val_loss}

        for trial_seq in [3, 14]:
            print(f"\n  --- SEQ_LEN={trial_seq} ---")
            trial_ds_train = SlidingWindowDataset(df_train, trial_seq, FEATURE_COLS, TARGET_COL)
            trial_ds_val   = SlidingWindowDataset(df_val,   trial_seq, FEATURE_COLS, TARGET_COL)
            trial_loader_t = DataLoader(trial_ds_train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
            trial_loader_v = DataLoader(trial_ds_val,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

            trial_model = LSTMForecaster(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(DEVICE)
            trial_opt   = torch.optim.Adam(trial_model.parameters(), lr=LR)
            trial_best  = float('inf')
            for ep in range(1, 16):
                tl = run_epoch(trial_model, trial_loader_t, criterion, trial_opt, train=True)
                vl = run_eval_epoch(trial_model, trial_loader_v, criterion)
                if vl < trial_best:
                    trial_best = vl
            scan_results[trial_seq] = trial_best
            print(f"  SEQ_LEN={trial_seq}: best val loss (15ep) = {trial_best:.6f}")

        best_seq = min(scan_results, key=scan_results.get)
        print(f"\nSEQ_LEN scan results (15-epoch val loss):")
        for seq, vl in sorted(scan_results.items()):
            marker = '  <-- BEST' if seq == best_seq else ''
            print(f"  SEQ_LEN={seq:>2}: val_loss={vl:.6f}{marker}")
        print(f"\nRecommendation: retrain with SEQ_LEN={best_seq} for full {EPOCHS} epochs.")

    print("\nPhase 1 complete. Ready for Phase 2 — AOD feature.")

    print(f"\n{'='*60}")
    print("Script 5 complete.")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
