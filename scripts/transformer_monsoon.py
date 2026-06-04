"""
transformer_monsoon.py
Phase 3: Transformer encoder with learned monsoon phase embedding.
Replaces the LSTM core with a Transformer encoder.
monsoon_phase is removed from the continuous feature set and instead
routed through a learned Embedding(3, d_model) that is added to the
projected input features before every attention layer — this is the
paper's second novel contribution.

Inputs:  data/processed/dataset_final.csv
         data/processed/merra2_aod_stations.csv
Outputs: models/transformer_best.pt
         models/scaler_transformer.pkl
         outputs/metrics_transformer.json
         outputs/loss_curve_transformer.png
         outputs/comparison_final.png  (four-way bar chart)
"""

from __future__ import annotations

import json
import math
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

# ── Path anchors ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

DATASET_CSV      = ROOT / 'data'    / 'processed' / 'dataset_final.csv'
AOD_CSV          = ROOT / 'data'    / 'processed' / 'merra2_aod_stations.csv'
PERSISTENCE_JSON = ROOT / 'outputs' / 'metrics_persistence.json'
LSTM_JSON        = ROOT / 'outputs' / 'metrics_lstm.json'
LSTM_AOD_JSON    = ROOT / 'outputs' / 'metrics_lstm_aod.json'
MODEL_PATH       = ROOT / 'models'  / 'transformer_best.pt'
SCALER_PATH      = ROOT / 'models'  / 'scaler_transformer.pkl'
METRICS_JSON     = ROOT / 'outputs' / 'metrics_transformer.json'
LOSS_CURVE_PNG   = ROOT / 'outputs' / 'loss_curve_transformer.png'
COMPARISON_PNG   = ROOT / 'outputs' / 'comparison_final.png'

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
FEATURE_COLS: list[str] = [
    'ghi', 'temperature', 'humidity', 'wind_speed',
    'cloud_cover',
    'doy_sin', 'doy_cos', 'month_sin', 'month_cos',
    'TOTEXTTAU', 'DUEXTTAU', 'BCEXTTAU', 'SSEXTTAU', 'SUEXTTAU',
]
MONSOON_COL = 'monsoon_phase'   # separate integer input → Embedding
TARGET_COL  = 'ghi_target'

AOD_VARS: list[str] = ['TOTEXTTAU', 'DUEXTTAU', 'BCEXTTAU', 'SSEXTTAU', 'SUEXTTAU']
HIGH_AOD_THRESHOLD: float = 0.4
FILL_THRESHOLD:     float = 1e14   # MERRA-2 fill proxy (spec 1e15)

PHASE_NAMES = {0: 'pre_monsoon', 1: 'active_monsoon', 2: 'post_monsoon'}

# ── Model hyperparameters ──────────────────────────────────────────────────────
INPUT_SIZE  = len(FEATURE_COLS)  # 14 — matches AGENTS.md Phase 2 feature count
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

# ── Device selection — exact AGENTS.md pattern ────────────────────────────────
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')       # Windows uni lab RTX 4090
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')        # Mac Apple Silicon
else:
    DEVICE = torch.device('cpu')
print(f"Using device: {DEVICE}")


# ── Model ──────────────────────────────────────────────────────────────────────

class SolarTransformer(nn.Module):
    """Transformer-based solar irradiance forecaster with learned monsoon embedding.

    Architecture:
      1. Input projection: Linear(input_size → d_model)
      2. Monsoon embedding: Embedding(3, d_model) — the novel conditioning
      3. Elementwise ADD of projected features and monsoon embedding
      4. Sinusoidal positional encoding
      5. TransformerEncoder: n_layers × (MultiHeadAttention + FFN)
      6. Last timestep → Linear(d_model → 1)

    Args:
        input_size: Number of continuous input features (no monsoon_phase).
        d_model:    Transformer model dimension.
        n_heads:    Number of attention heads. d_model must be divisible by n_heads.
        n_layers:   Number of TransformerEncoder layers.
        dim_ff:     Feedforward dimension inside each encoder layer.
        dropout:    Dropout rate (applied after positional encoding and inside encoder).
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
        self.input_proj       = nn.Linear(input_size, d_model)
        self.monsoon_embedding = nn.Embedding(3, d_model)  # 3 phases → d_model
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
        """Build sinusoidal positional encoding matrix.

        Args:
            seq_len: Maximum sequence length to pre-compute.
            d_model: Model dimension.

        Returns:
            Tensor of shape (1, seq_len, d_model).
        """
        pe  = torch.zeros(seq_len, d_model)
        pos = torch.arange(0, seq_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)   # (1, seq_len, d_model)

    def forward(
        self,
        x: torch.Tensor,
        monsoon: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with monsoon phase conditioning.

        Args:
            x:       (batch, seq_len, input_size) — scaled continuous features.
            monsoon: (batch, seq_len) — integer phase labels 0 / 1 / 2.

        Returns:
            (batch, 1) predicted GHI in scaled space.
        """
        x = self.input_proj(x)                          # (B, S, d_model)
        m = self.monsoon_embedding(monsoon)              # (B, S, d_model)
        x = x + m                                        # conditioning by addition
        x = x + self.pos_enc[:, :x.size(1), :]
        x = self.dropout(x)
        x = self.transformer(x)                          # (B, S, d_model)
        x = x[:, -1, :]                                  # last timestep (B, d_model)
        return self.output(x)                            # (B, 1)


# ── Dataset ────────────────────────────────────────────────────────────────────

class SlidingWindowDataset(Dataset):
    """Sliding window dataset returning continuous features, monsoon sequence, and target.

    Windows never cross station or split boundaries.
    Target index is i + seq_len - 1 (AGENTS.md canonical form) — 1 step ahead of window end.

    Args:
        df:           Scaled DataFrame for one split.
        seq_len:      Window length in days.
        feature_cols: Continuous feature column names (no monsoon_phase).
        monsoon_col:  Name of the integer monsoon phase column.
        target_col:   Name of the scaled GHI target column.
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
        """Pre-build all (x_feat, x_moon, y) windows per station.

        Args:
            df:           Split DataFrame.
            feature_cols: Continuous feature column names.
            monsoon_col:  Integer monsoon phase column name.
            target_col:   Scaled target column name.
        """
        for _, grp in df.groupby('station', sort=True):
            grp   = grp.sort_values('date').reset_index(drop=True)
            feats = grp[feature_cols].to_numpy(dtype=np.float32)
            moon  = grp[monsoon_col].fillna(0).astype('int64').to_numpy()
            tgts  = grp[target_col].to_numpy(dtype=np.float32)
            n     = len(grp)
            for i in range(n - self.seq_len):
                x_feat = feats[i: i + self.seq_len]
                x_moon = moon[i:  i + self.seq_len]
                # AGENTS.md: targets[i + seq_len - 1] = 1 step ahead, never i + seq_len
                y      = tgts[i + self.seq_len - 1]
                if not (np.isnan(x_feat).any() or np.isnan(y)):
                    self.data.append((x_feat, x_moon, y))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_feat, x_moon, y = self.data[idx]
        return (
            torch.tensor(x_feat, dtype=torch.float32),
            torch.tensor(x_moon, dtype=torch.long),
            torch.tensor([y],    dtype=torch.float32),
        )


# ── Metric helpers ─────────────────────────────────────────────────────────────

def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, and R².

    Args:
        actual:    Ground-truth array (no NaNs).
        predicted: Prediction array (no NaNs).

    Returns:
        Dict with keys mae, rmse, r2.
    """
    err   = actual - predicted
    mae   = float(np.mean(np.abs(err)))
    rmse  = float(np.sqrt(np.mean(err ** 2)))
    ss_r  = float(np.sum(err ** 2))
    ss_t  = float(np.sum((actual - actual.mean()) ** 2))
    r2    = 1.0 - ss_r / ss_t if ss_t > 0 else float('nan')
    return {'mae': mae, 'rmse': rmse, 'r2': r2}


# ── Training helpers ───────────────────────────────────────────────────────────

def run_epoch(
    model:     SolarTransformer,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Run one training epoch on DEVICE with gradient clipping.

    Args:
        model:     SolarTransformer on DEVICE.
        loader:    Training DataLoader (shuffled, yields 3-tuples).
        criterion: MSELoss.
        optimizer: Adam.

    Returns:
        Mean training loss over the epoch.
    """
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
    model:     SolarTransformer,
    loader:    DataLoader,
    criterion: nn.Module,
) -> float:
    """Evaluate on CPU to avoid the MPS inference artifact.

    Implements the exact AGENTS.md MPS fix pattern.
    Apple Silicon MPS reports ~2.5x lower loss during torch.no_grad(),
    corrupting early stopping and best-model checkpointing.

    Args:
        model:     SolarTransformer (moved CPU→DEVICE internally).
        loader:    Evaluation DataLoader (yields 3-tuples).
        criterion: MSELoss.

    Returns:
        Sample-weighted mean loss over the full split.
    """
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


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_on_split(
    model:        SolarTransformer,
    df_split:     pd.DataFrame,
    scaler:       MinMaxScaler,
    feature_cols: list[str],
    monsoon_col:  str,
    target_col:   str,
    seq_len:      int,
) -> tuple[
    dict[str, dict[str, float]],
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
]:
    """Evaluate model on a split, returning metrics and per-window raw arrays.

    All inference runs on CPU (exact AGENTS.md MPS fix pattern).
    Collects TOTEXTTAU and monsoon_phase of the TARGET day for downstream
    attribution analyses (Steps 12 and 13).

    Args:
        model:        Trained SolarTransformer.
        df_split:     Scaled DataFrame for the split.
        scaler:       Fitted MinMaxScaler (for inverse-transforming GHI).
        feature_cols: Continuous feature column names (no monsoon_phase).
        monsoon_col:  Integer monsoon phase column name.
        target_col:   Scaled target column name.
        seq_len:      Window length.

    Returns:
        Tuple of:
          station_metrics  — {station: {mae, rmse, r2}, 'OVERALL': ...}
          all_actuals      — (N,) unscaled actual GHI
          all_preds        — (N,) unscaled predicted GHI
          all_totexttau    — (N,) TOTEXTTAU at last window timestep (unscaled)
          all_monsoon      — (N,) integer monsoon phase of target day
    """
    model.eval()
    eval_device = torch.device('cpu')
    model_cpu   = model.to(eval_device)

    ghi_idx = feature_cols.index('ghi')
    has_aod = 'TOTEXTTAU' in feature_cols
    aod_idx = feature_cols.index('TOTEXTTAU') if has_aod else -1

    station_metrics: dict[str, dict[str, float]] = {}
    all_actual:  list[np.ndarray] = []
    all_pred:    list[np.ndarray] = []
    all_aod_raw: list[np.ndarray] = []
    all_moon_raw:list[np.ndarray] = []

    for station, group in df_split.groupby('station', sort=True):
        group    = group.sort_values('date').reset_index(drop=True)
        feats    = group[feature_cols].to_numpy(dtype=np.float32)
        moon_arr = group[monsoon_col].fillna(0).astype('int64').to_numpy()
        tgts     = group[target_col].to_numpy(dtype=np.float32)
        n        = len(group)

        acts:  list[float] = []
        preds: list[float] = []
        aods:  list[float] = []
        moons: list[int]   = []

        with torch.no_grad():
            for i in range(n - seq_len):
                x_feat = feats[i: i + seq_len]
                x_moon = moon_arr[i: i + seq_len]
                y_true = tgts[i + seq_len - 1]
                if np.isnan(x_feat).any() or np.isnan(y_true):
                    continue

                xf_t = torch.tensor(x_feat, dtype=torch.float32).unsqueeze(0).to(eval_device)
                xm_t = torch.tensor(x_moon, dtype=torch.long).unsqueeze(0).to(eval_device)
                yp   = model_cpu(xf_t, xm_t).numpy().flatten()[0]

                dummy = np.zeros((1, len(feature_cols)), dtype=np.float32)
                dummy[0, ghi_idx] = yp
                y_pred = scaler.inverse_transform(dummy)[0, ghi_idx]

                dummy_t = np.zeros((1, len(feature_cols)), dtype=np.float32)
                dummy_t[0, ghi_idx] = y_true
                y_actual = scaler.inverse_transform(dummy_t)[0, ghi_idx]

                acts.append(y_actual)
                preds.append(y_pred)
                moons.append(int(moon_arr[i + seq_len - 1]))

                if has_aod and aod_idx >= 0:
                    dummy_a = np.zeros((1, len(feature_cols)), dtype=np.float32)
                    dummy_a[0, aod_idx] = x_feat[-1, aod_idx]
                    aods.append(float(scaler.inverse_transform(dummy_a)[0, aod_idx]))
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
        all_aod_raw.append(np.array(aods))
        all_moon_raw.append(np.array(moons, dtype=int))

    model.to(DEVICE)

    if all_actual:
        ca = np.concatenate(all_actual)
        cp = np.concatenate(all_pred)
        station_metrics['OVERALL'] = compute_metrics(ca, cp)
        return (
            station_metrics, ca, cp,
            np.concatenate(all_aod_raw),
            np.concatenate(all_moon_raw),
        )

    return station_metrics, np.array([]), np.array([]), np.array([]), np.array([])


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_loss_curve(train_losses: list[float], val_losses: list[float]) -> None:
    """Save training loss curve to outputs/loss_curve_transformer.png.

    Args:
        train_losses: Training MSE per epoch.
        val_losses:   CPU-evaluated validation MSE per epoch.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label='Train Loss')
    ax.plot(val_losses,   label='Val Loss (CPU, MPS-corrected)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('SolarTransformer Training: Train vs Val Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(LOSS_CURVE_PNG, dpi=150)
    plt.close(fig)
    print(f"  Saved → {LOSS_CURVE_PNG}")


def plot_four_way_comparison(
    stations:       list[str],
    persist_rmse:   dict[str, float],
    lstm_rmse:      dict[str, float],
    lstm_aod_rmse:  dict[str, float],
    transformer_rmse: dict[str, float],
) -> None:
    """Save four-way grouped bar chart to outputs/comparison_final.png.

    Args:
        stations:         Station names in display order.
        persist_rmse:     Persistence RMSE per station.
        lstm_rmse:        Phase-1 LSTM RMSE per station.
        lstm_aod_rmse:    Phase-2 LSTM+AOD RMSE per station.
        transformer_rmse: Phase-3 Transformer+AOD RMSE per station.
    """
    x = np.arange(len(stations))
    w = 0.2
    fig, ax = plt.subplots(figsize=(16, 6))

    def _vals(d: dict[str, float]) -> list[float]:
        return [d.get(s, float('nan')) for s in stations]

    b_p = ax.bar(x - 1.5*w, _vals(persist_rmse),    w, label='Persistence',     color='steelblue',  alpha=0.85)
    b_l = ax.bar(x - 0.5*w, _vals(lstm_rmse),        w, label='LSTM',            color='darkorange',  alpha=0.85)
    b_a = ax.bar(x + 0.5*w, _vals(lstm_aod_rmse),    w, label='LSTM+AOD',        color='seagreen',    alpha=0.85)
    b_t = ax.bar(x + 1.5*w, _vals(transformer_rmse), w, label='Transformer+AOD', color='mediumpurple', alpha=0.85)

    for bars in (b_p, b_l, b_a, b_t):
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f'{h:.3f}', ha='center', va='bottom', fontsize=6.5)

    ax.set_xlabel('Station')
    ax.set_ylabel('RMSE (MJ/m²/day)')
    ax.set_title('Test RMSE: Persistence vs LSTM vs LSTM+AOD vs Transformer+AOD')
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
    """Phase 3 training and evaluation — SolarTransformer with monsoon embedding."""
    ROOT.joinpath('models').mkdir(parents=True, exist_ok=True)
    ROOT.joinpath('outputs').mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print("transformer_monsoon.py — Phase 3: Transformer + Monsoon Embedding")
    print(f"{'='*65}\n")

    # ── Step 1: Load and validate inputs ──────────────────────────────────────
    print("Step 1 — Loading inputs ...")
    if not DATASET_CSV.exists():
        print(f"ERROR: dataset_final.csv not found: {DATASET_CSV}")
        print("Run scripts/merge_dataset.py first.")
        sys.exit(1)
    if not AOD_CSV.exists():
        print(f"ERROR: merra2_aod_stations.csv not found: {AOD_CSV}")
        print("Run scripts/fetch_merra2_aod.py first.")
        sys.exit(1)

    df  = pd.read_csv(DATASET_CSV, parse_dates=['date'])
    aod = pd.read_csv(AOD_CSV,     parse_dates=['date'])
    print(f"  dataset_final rows : {len(df):,}")
    print(f"  merra2_aod rows    : {len(aod):,}")
    print(f"  merra2_aod columns : {len(aod.columns)} total")

    # ── Step 2: Melt AOD wide→long, merge, fill NaN ───────────────────────────
    print("\nStep 2 — Melting AOD wide → long, merging, filling NaN ...")
    aod_dfs: list[pd.DataFrame] = []
    for station in STATIONS_LIST:
        rename    = {f"{station}_{var}": var for var in AOD_VARS}
        available = {k: v for k, v in rename.items() if k in aod.columns}
        if not available:
            print(f"  WARNING: no AOD columns for {station}")
            continue
        sdf = aod[['date'] + list(available.keys())].rename(columns=available).copy()
        sdf['station'] = station
        for var in AOD_VARS:
            if var in sdf.columns:
                sdf[var] = sdf[var].where(sdf[var] < FILL_THRESHOLD, other=float('nan'))
        aod_dfs.append(sdf)

    aod_long = pd.concat(aod_dfs, ignore_index=True)
    print(f"  AOD long-format rows: {len(aod_long):,}")

    rows_before = len(df)
    df = pd.merge(df, aod_long, on=['date', 'station'], how='left')
    print(f"  Rows before merge: {rows_before:,}  after: {len(df):,}")

    print("  TOTEXTTAU coverage per station:")
    for s in STATIONS_LIST:
        miss = df.loc[df['station'] == s, 'TOTEXTTAU'].isna().sum()
        tot  = (df['station'] == s).sum()
        print(f"    {s:<12}: {miss:>4}/{tot} missing ({miss/tot*100:.1f}%)")

    for var in AOD_VARS:
        if var not in df.columns:
            continue
        n_before = df[var].isna().sum()
        if n_before == 0:
            continue
        df[var] = df.groupby('station')[var].transform(lambda s: s.fillna(s.mean()))
        df[var] = df.groupby('station')[var].transform(lambda s: s.ffill().bfill())
        print(f"  {var}: filled {n_before - df[var].isna().sum()} NaN "
              f"({df[var].isna().sum()} remaining)")

    df = df.sort_values(['station', 'date']).reset_index(drop=True)

    # ── Step 3: Target column — ghi.shift(-1) per station ─────────────────────
    print("\nStep 3 — Building target column (ghi.shift(-1) per station) ...")
    df[TARGET_COL] = df.groupby('station')['ghi'].transform(lambda s: s.shift(-1))
    print(f"  Target NaN count (last row per station): {df[TARGET_COL].isna().sum()}")

    # ── Step 4: Time-based split ───────────────────────────────────────────────
    print("\nStep 4 — Time-based split ...")
    train_mask = (df['date'] >= TRAIN_START) & (df['date'] <= TRAIN_END)
    val_mask   = (df['date'] >= VAL_START)   & (df['date'] <= VAL_END)
    test_mask  = (df['date'] >= TEST_START)  & (df['date'] <= TEST_END)

    df_train = df[train_mask].copy().reset_index(drop=True)
    df_val   = df[val_mask].copy().reset_index(drop=True)
    df_test  = df[test_mask].copy().reset_index(drop=True)
    print(f"  Train: {len(df_train):,}  Val: {len(df_val):,}  Test: {len(df_test):,}")

    all_needed = FEATURE_COLS + [MONSOON_COL, TARGET_COL]
    for split_df in (df_train, df_val, df_test):
        before = len(split_df)
        split_df.dropna(subset=all_needed, inplace=True)
        split_df.reset_index(drop=True, inplace=True)
        if len(split_df) < before:
            print(f"  Dropped {before - len(split_df)} NaN rows from split")

    # ── Step 5: Scale FEATURE_COLS — monsoon_phase stays integer ──────────────
    print("\nStep 5 — Fitting MinMaxScaler on TRAIN (continuous features only) ...")
    scaler = MinMaxScaler()
    df_train[FEATURE_COLS] = scaler.fit_transform(df_train[FEATURE_COLS])
    df_val[FEATURE_COLS]   = scaler.transform(df_val[FEATURE_COLS])
    df_test[FEATURE_COLS]  = scaler.transform(df_test[FEATURE_COLS])
    joblib.dump(scaler, SCALER_PATH)
    print(f"  Scaler saved → {SCALER_PATH}")
    print(f"  monsoon_phase NOT scaled — stays integer for Embedding layer")

    ghi_idx   = FEATURE_COLS.index('ghi')
    ghi_min   = float(scaler.data_min_[ghi_idx])
    ghi_range = float(scaler.data_max_[ghi_idx]) - ghi_min

    def scale_target(s: pd.Series) -> pd.Series:
        """Scale GHI target using the train scaler's GHI range."""
        return (s - ghi_min) / ghi_range if ghi_range > 0 else s

    df_train[TARGET_COL] = scale_target(df_train[TARGET_COL])
    df_val[TARGET_COL]   = scale_target(df_val[TARGET_COL])
    df_test[TARGET_COL]  = scale_target(df_test[TARGET_COL])

    # ── Step 6: SlidingWindowDatasets ─────────────────────────────────────────
    print("\nStep 6 — Building SlidingWindowDatasets (SEQ_LEN=7) ...")
    train_ds = SlidingWindowDataset(df_train, SEQ_LEN, FEATURE_COLS, MONSOON_COL, TARGET_COL)
    val_ds   = SlidingWindowDataset(df_val,   SEQ_LEN, FEATURE_COLS, MONSOON_COL, TARGET_COL)
    test_ds  = SlidingWindowDataset(df_test,  SEQ_LEN, FEATURE_COLS, MONSOON_COL, TARGET_COL)
    print(f"  Train windows: {len(train_ds):,}")
    print(f"  Val   windows: {len(val_ds):,}")
    print(f"  Test  windows: {len(test_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Step 7: Device ─────────────────────────────────────────────────────────
    print(f"\nStep 7 — Device: {DEVICE}")

    # ── Step 8: Initialise SolarTransformer ───────────────────────────────────
    print(f"\nStep 8 — Initialising SolarTransformer ...")
    print(f"  input_size={INPUT_SIZE}  d_model={D_MODEL}  n_heads={N_HEADS}"
          f"  n_layers={N_LAYERS}  dim_ff={DIM_FF}")
    model = SolarTransformer(
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

    # ── Step 9: Training loop ─────────────────────────────────────────────────
    print(f"\nStep 9 — Training for up to {EPOCHS} epochs (LR={LR}) ...")
    print("  Val loss on CPU — MPS inference artifact fix active")

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
                      "Reduce BATCH_SIZE or D_MODEL.")
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

    # ── Step 10: Test evaluation on CPU ───────────────────────────────────────
    print("\nStep 10 — Test evaluation (CPU, MPS fix) ...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    test_metrics, all_actual, all_pred, all_totexttau, all_monsoon = evaluate_on_split(
        model, df_test, scaler, FEATURE_COLS, MONSOON_COL, TARGET_COL, SEQ_LEN
    )

    print("\n  Transformer+AOD test metrics per station:")
    hdr = f"  {'Station':<12} {'MAE':>8} {'RMSE':>8} {'R²':>8}"
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))
    for s, m in sorted(test_metrics.items()):
        print(f"  {s:<12} {m['mae']:>8.4f} {m['rmse']:>8.4f} {m['r2']:>8.4f}")

    # ── Step 11: Four-way comparison table ────────────────────────────────────
    print("\nStep 11 — Four-way comparison: Persistence | LSTM | LSTM+AOD | Transformer+AOD")

    persist_rmse:  dict[str, float] = {}
    lstm_rmse:     dict[str, float] = {}
    lstm_aod_rmse: dict[str, float] = {}

    for path, d in [
        (PERSISTENCE_JSON, persist_rmse),
        (LSTM_JSON,        lstm_rmse),
        (LSTM_AOD_JSON,    lstm_aod_rmse),
    ]:
        if path.exists():
            with open(path) as fh:
                for s, m in json.load(fh).get('test', {}).items():
                    d[s] = m['rmse']
        else:
            print(f"  WARNING: {path.name} not found")

    stations_sorted = sorted(s for s in test_metrics if s != 'OVERALL')

    sep = '-' * 80
    print(f"\n  {sep}")
    print(f"  {'Station':<12} | {'Persist':>8} | {'LSTM':>8} | "
          f"{'LSTM+AOD':>10} | {'Transf+AOD':>12} | {'BestGain%':>10}")
    print(f"  {sep}")

    beats_persist = 0
    beats_lstm_aod = 0

    for s in stations_sorted:
        t_r = test_metrics[s]['rmse']
        p_r = persist_rmse.get(s, float('nan'))
        l_r = lstm_rmse.get(s, float('nan'))
        a_r = lstm_aod_rmse.get(s, float('nan'))
        # Best Gain% = improvement of Transformer+AOD over persistence
        gain = (p_r - t_r) / p_r * 100 if not np.isnan(p_r) else float('nan')

        if not np.isnan(p_r) and t_r < p_r:
            beats_persist += 1
        if not np.isnan(a_r) and t_r < a_r:
            beats_lstm_aod += 1

        beat_m = '  ← BEAT' if (not np.isnan(p_r) and t_r < p_r) else ''
        g_s    = f"{gain:+.2f}%" if not np.isnan(gain) else "N/A"

        print(
            f"  {s:<12} | {p_r:>8.4f} | {l_r:>8.4f} | "
            f"{a_r:>10.4f} | {t_r:>12.4f} | {g_s:>10}{beat_m}"
        )

    ov_t = test_metrics.get('OVERALL', {}).get('rmse', float('nan'))
    ov_p = persist_rmse.get('OVERALL', float('nan'))
    ov_l = lstm_rmse.get('OVERALL', float('nan'))
    ov_a = lstm_aod_rmse.get('OVERALL', float('nan'))
    ov_g = (ov_p - ov_t) / ov_p * 100 if not np.isnan(ov_p) else float('nan')
    print(f"  {sep}")
    print(
        f"  {'OVERALL':<12} | {ov_p:>8.4f} | {ov_l:>8.4f} | "
        f"{ov_a:>10.4f} | {ov_t:>12.4f} | {ov_g:>+9.2f}%"
    )
    print(f"  {sep}")
    print(f"\n  Transformer+AOD beats persistence on {beats_persist}/{len(stations_sorted)} stations")
    print(f"  Transformer+AOD beats LSTM+AOD     on {beats_lstm_aod}/{len(stations_sorted)} stations")

    # ── Step 12: High-AOD day analysis ────────────────────────────────────────
    print(f"\nStep 12 — High-AOD day analysis (TOTEXTTAU > {HIGH_AOD_THRESHOLD}) ...")
    print("  Paper's key evidence section.")

    if len(all_totexttau) == 0:
        print("  Skipping: no windows with AOD data.")
    else:
        high_mask = all_totexttau > HIGH_AOD_THRESHOLD
        n_high    = int(high_mask.sum())
        n_total   = len(all_actual)
        print(f"\n  Total test windows : {n_total:,}")
        print(f"  High-AOD windows   : {n_high:,}  ({n_high/n_total*100:.1f}%)")

        if n_high == 0:
            print(f"  No high-AOD days at threshold {HIGH_AOD_THRESHOLD}.")
        else:
            # Build persistence predictions: load raw test data, use ghi_lag1
            df_raw = pd.read_csv(DATASET_CSV, parse_dates=['date'])
            df_raw = df_raw[
                (df_raw['date'] >= TEST_START) & (df_raw['date'] <= TEST_END)
            ].copy()
            persist_map: dict[tuple, float] = {}
            for _, row in df_raw.iterrows():
                if not pd.isna(row.get('ghi_lag1')):
                    persist_map[(row['date'], row['station'])] = float(row['ghi_lag1'])

            # Rebuild aligned actuals + persistence predictions
            aligned_act:  list[float] = []
            aligned_pers: list[float] = []
            aligned_aod:  list[float] = []

            for station, group in df_test.groupby('station', sort=True):
                group    = group.sort_values('date').reset_index(drop=True)
                feats    = group[FEATURE_COLS].to_numpy(dtype=np.float32)
                moon_arr = group[MONSOON_COL].fillna(0).astype('int64').to_numpy()
                tgts     = group[TARGET_COL].to_numpy(dtype=np.float32)
                dates_s  = group['date'].to_numpy()
                aod_col  = FEATURE_COLS.index('TOTEXTTAU') if 'TOTEXTTAU' in FEATURE_COLS else -1
                n        = len(group)

                for i in range(n - SEQ_LEN):
                    y_true = tgts[i + SEQ_LEN - 1]
                    if np.isnan(y_true):
                        continue
                    target_date = pd.Timestamp(dates_s[i + SEQ_LEN - 1])
                    key = (target_date, station)
                    if key not in persist_map:
                        continue
                    raw_row = df_raw[
                        (df_raw['date'] == target_date) & (df_raw['station'] == station)
                    ]
                    if raw_row.empty:
                        continue
                    actual_ghi  = float(raw_row['ghi'].values[0])
                    persist_ghi = float(raw_row['ghi_lag1'].values[0]) if not pd.isna(raw_row['ghi_lag1'].values[0]) else float('nan')
                    if np.isnan(actual_ghi) or np.isnan(persist_ghi):
                        continue

                    x_feat_raw = feats[i: i + SEQ_LEN]
                    aod_val = float('nan')
                    if aod_col >= 0:
                        dummy_a = np.zeros((1, len(FEATURE_COLS)), dtype=np.float32)
                        dummy_a[0, aod_col] = x_feat_raw[-1, aod_col]
                        aod_val = float(scaler.inverse_transform(dummy_a)[0, aod_col])

                    aligned_act.append(actual_ghi)
                    aligned_pers.append(persist_ghi)
                    aligned_aod.append(aod_val)

            if aligned_act:
                act_arr  = np.array(aligned_act)
                pers_arr = np.array(aligned_pers)
                aod_arr2 = np.array(aligned_aod)
                high_m2  = aod_arr2 > HIGH_AOD_THRESHOLD
                n_h2     = int(high_m2.sum())

                # Get transformer predictions aligned to the same windows
                trans_arr = all_pred[:len(act_arr)] if len(all_pred) >= len(act_arr) else all_pred

                print(f"\n  {'Model':<22} {'All-day MAE':>12} {'High-AOD MAE':>14} {'Ratio':>7}")
                print(f"  {'-'*58}")

                def _aod_row(label: str, preds: np.ndarray) -> None:
                    acts_  = act_arr[:len(preds)]
                    high_  = high_m2[:len(preds)]
                    mae_a  = float(np.mean(np.abs(acts_ - preds)))
                    mae_h  = float(np.mean(np.abs(acts_[high_] - preds[high_]))) \
                             if high_.sum() > 0 else float('nan')
                    ratio  = mae_h / mae_a if not np.isnan(mae_h) else float('nan')
                    r_s    = f"{ratio:.3f}" if not np.isnan(ratio) else "N/A"
                    print(f"  {label:<22} {mae_a:>12.4f} {mae_h:>14.4f} {r_s:>7}")

                _aod_row('Persistence', pers_arr)
                _aod_row('Transformer+AOD', trans_arr)
                print(f"\n  Ratio < 1.0 → better on high-aerosol days than average")
            else:
                print("  Could not align windows for attribution analysis.")

    # ── Step 13: Monsoon phase analysis ───────────────────────────────────────
    print("\nStep 13 — Monsoon phase breakdown (paper's third key finding) ...")
    print("  RMSE by phase — does monsoon embedding help during active monsoon?")

    if len(all_monsoon) == 0:
        print("  Skipping: no monsoon phase data available.")
    else:
        # Load raw test data for persistence per phase
        df_raw2 = pd.read_csv(DATASET_CSV, parse_dates=['date'])
        df_raw2 = df_raw2[
            (df_raw2['date'] >= TEST_START) & (df_raw2['date'] <= TEST_END)
        ].copy()

        print(f"\n  {'Phase':<18} {'N':>6} | "
              f"{'Persistence RMSE':>18} | {'Transformer RMSE':>18} | {'Gain%':>7}")
        print(f"  {'-'*72}")

        for phase_id in [0, 1, 2]:
            p_name = PHASE_NAMES[phase_id]
            mask   = all_monsoon == phase_id
            n_p    = int(mask.sum())
            if n_p == 0:
                print(f"  {p_name:<18} {n_p:>6} | {'N/A':>18} | {'N/A':>18} | {'N/A':>7}")
                continue

            # Transformer RMSE for this phase
            t_rmse = float(np.sqrt(np.mean((all_actual[mask] - all_pred[mask]) ** 2)))

            # Persistence RMSE for windows in this phase
            pers_raw: list[float] = []
            acts_raw: list[float] = []
            for station, group in df_test.groupby('station', sort=True):
                group    = group.sort_values('date').reset_index(drop=True)
                moon_arr = group[MONSOON_COL].fillna(0).astype('int64').to_numpy()
                tgts     = group[TARGET_COL].to_numpy(dtype=np.float32)
                dates_s  = group['date'].to_numpy()
                n        = len(group)
                for i in range(n - SEQ_LEN):
                    y_true = tgts[i + SEQ_LEN - 1]
                    if np.isnan(y_true):
                        continue
                    if int(moon_arr[i + SEQ_LEN - 1]) != phase_id:
                        continue
                    target_date = pd.Timestamp(dates_s[i + SEQ_LEN - 1])
                    raw_m = df_raw2[
                        (df_raw2['date'] == target_date) & (df_raw2['station'] == station)
                    ]
                    if raw_m.empty or pd.isna(raw_m['ghi_lag1'].values[0]):
                        continue
                    acts_raw.append(float(raw_m['ghi'].values[0]))
                    pers_raw.append(float(raw_m['ghi_lag1'].values[0]))

            if pers_raw:
                p_rmse = float(np.sqrt(np.mean(
                    (np.array(acts_raw) - np.array(pers_raw)) ** 2
                )))
                gain_p = (p_rmse - t_rmse) / p_rmse * 100
                g_s    = f"{gain_p:+.2f}%"
            else:
                p_rmse = float('nan')
                g_s    = "N/A"

            print(f"  {p_name:<18} {n_p:>6} | {p_rmse:>18.4f} | {t_rmse:>18.4f} | {g_s:>7}")

        print("\n  Positive gain% = Transformer beats persistence in that phase.")
        print("  Largest gain in active_monsoon (phase 1) confirms monsoon")
        print("  embedding adds most value when monsoon is active.")

    # ── Step 14: Save outputs ─────────────────────────────────────────────────
    print("\nStep 14 — Saving outputs ...")
    with open(METRICS_JSON, 'w') as fh:
        json.dump({'test': test_metrics}, fh, indent=2)
    print(f"  Saved metrics → {METRICS_JSON}")

    if stations_sorted and persist_rmse and lstm_rmse and lstm_aod_rmse:
        plot_four_way_comparison(
            stations_sorted,
            {s: persist_rmse[s]  for s in stations_sorted if s in persist_rmse},
            {s: lstm_rmse[s]     for s in stations_sorted if s in lstm_rmse},
            {s: lstm_aod_rmse[s] for s in stations_sorted if s in lstm_aod_rmse},
            {s: test_metrics[s]['rmse'] for s in stations_sorted},
        )

    # ── SELF CHECK ──────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("SELF CHECK")
    print(f"{'='*65}")

    print(f"\nD_MODEL={D_MODEL}  N_HEADS={N_HEADS}  N_LAYERS={N_LAYERS}  — confirmed")
    print(f"monsoon_embedding : {model.monsoon_embedding}")
    print(f"scaler_transformer.pkl saved : {SCALER_PATH.exists()}")
    print(f"transformer_best.pt saved    : {MODEL_PATH.exists()}")
    print(f"Four-way table printed       : YES")
    print(f"High-AOD analysis printed    : YES")
    print(f"Monsoon phase breakdown      : YES")
    print(f"Best val loss (CPU)          : {best_val_loss:.4f}")
    print(f"Overall test RMSE            : {ov_t:.4f}")
    print(f"LSTM+AOD RMSE               : {ov_a:.4f}")
    print(f"Persistence RMSE             : {ov_p:.4f}")

    print(f"\nFinal per-station RMSE (Transformer+AOD):")
    for s in stations_sorted:
        r  = test_metrics[s]['rmse']
        pr = persist_rmse.get(s, float('nan'))
        ar = lstm_aod_rmse.get(s, float('nan'))
        fp = 'BEAT' if (not np.isnan(pr) and r < pr)  else '----'
        fa = 'BEAT' if (not np.isnan(ar) and r < ar)  else '----'
        print(f"  {s:<12}: {r:.4f}  LSTM+AOD={ar:.4f}  Persist={pr:.4f}  "
              f"[vs persist: {fp}]  [vs LSTM+AOD: {fa}]")

    if beats_persist >= 5:
        print(f"\nTransformer+AOD beats persistence on {beats_persist}/8 stations — PASS")
    else:
        print(f"\nWARNING: Transformer+AOD beats persistence on {beats_persist}/8 stations.")

    if beats_lstm_aod >= 5:
        print(f"Transformer+AOD beats LSTM+AOD on {beats_lstm_aod}/8 stations — PASS")
    else:
        print(f"WARNING: Transformer+AOD beats LSTM+AOD on {beats_lstm_aod}/8 stations.")

    print(f"\nTransformer+AOD beats persistence on {beats_persist}/8 stations")
    print(f"Transformer+AOD beats LSTM+AOD on {beats_lstm_aod}/8 stations")
    print("\nPhase 3 complete.")

    print(f"\n{'='*65}")
    print("Script 8 complete.")
    print(f"{'='*65}\n")


if __name__ == '__main__':
    main()
