# AGENTS.md — Solar Irradiance Forecasting — India Multi-Zone
# READ THIS FULLY BEFORE WRITING ANY CODE. NO EXCEPTIONS.

## Project Summary
Forecasting daily solar irradiance across 8 Indian stations
spanning 4 climate zones using deep learning.

Phase 1 (COMPLETE): NASA POWER data only — GHI + met variables
                    LSTM baseline established. Overall test RMSE = 1.1944
                    1/8 stations beat persistence (Kolkata +0.23%)
                    R² positive on all 8 stations (0.12–0.37)

Phase 2 (NOW):      Add MERRA-2 AOD as novel input feature
                    Expected to close gap on high-aerosol days
                    where persistence fails most severely

Design all code to be extensible — AOD slots in as extra
columns without restructuring the pipeline.

---

## Absolute Rules — Never Break These

### Cross-platform
- ALWAYS use pathlib.Path — never os.path, never / or \
- Use Path(__file__).resolve().parent.parent to anchor all paths
- Code must run identically on macOS and Windows

### Data integrity
- NASA POWER fill value:  -999  → replace with NaN immediately
- MERRA-2 fill value:     1e15  → replace with NaN immediately
- Always print row counts before and after any drop/filter
- Corrupt or failed API calls: log, skip, never crash
- Never modify raw data — always save processed to new file

### ML integrity — CRITICAL, NEVER VIOLATE
- Train/val/test split: STRICTLY TIME-BASED, never random
- Fit ALL scalers on TRAIN SET ONLY — never on full dataset
- Apply (transform only) fitted scaler to val and test
- Sliding windows must NOT cross station boundaries
- Sliding windows must NOT cross train/val/test boundaries
- No future data can leak into any past window
- Target column = ghi.shift(-1) per station
  (tomorrow's GHI — 1 step ahead, NOT 2 steps)
  CRITICAL: use targets[i-1] not targets[i] in SlidingWindowDataset
  Off-by-one here was the critical bug in Phase 1 — never repeat

### MPS Inference Fix — MANDATORY in all training scripts
Apple Silicon MPS has an inference artifact during
torch.set_grad_enabled(False) that causes val loss to read
~2.5x lower than true value. This corrupts early stopping
and best-model checkpointing.

Fix — use this exact pattern for ALL evaluation loops:

```python
# Move to CPU for evaluation — avoids MPS inference artifact
model.eval()
eval_device = torch.device('cpu')
model_cpu = model.to(eval_device)

total_loss = 0.0
with torch.no_grad():
    for xb, yb in loader:
        xb = xb.to(eval_device)
        yb = yb.to(eval_device)
        pred = model_cpu(xb)
        total_loss += criterion(pred, yb).item() * len(xb)

val_loss = total_loss / len(loader.dataset)

# Move back to training device
model.to(DEVICE)
```

This applies to: validation loop, test evaluation, any
inference step. Training forward+backward pass stays on DEVICE.

### Code quality
- Type hints on every function
- Docstring on every function: purpose, args, returns
- All constants at top of file — zero magic numbers inline
- Verbose print statements showing progress throughout
- Every script saves output before exiting
- Every script prints a SELF CHECK block at the end

---

## Hardware — Device Selection
Use this exact block in every training script:

```python
import torch
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')      # Windows uni lab RTX 4090
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')       # Mac Apple Silicon
else:
    DEVICE = torch.device('cpu')
print(f"Using device: {DEVICE}")
```

---

## Station Coordinates — DO NOT MODIFY EVER
```python
STATIONS = {
    'Jodhpur':   (26.30, 73.02),
    'Ahmedabad': (23.03, 72.58),
    'Mumbai':    (19.08, 72.88),
    'Chennai':   (13.08, 80.27),
    'Hyderabad': (17.38, 78.48),
    'Bhopal':    (23.25, 77.40),
    'Kolkata':   (22.57, 88.36),
    'Bengaluru': (12.97, 77.59),
}
```

---

## Data Splits — FIXED, NEVER CHANGE
```python
TRAIN_START = "2017-04-01"
TRAIN_END   = "2022-12-31"
VAL_START   = "2023-01-01"
VAL_END     = "2023-12-31"
TEST_START  = "2024-01-01"
TEST_END    = "2024-12-31"
```

---

## LSTM Hyperparameters — Phase 1 Confirmed Best
```python
SEQ_LEN     = 7      # confirmed best via scan (3=0.009002, 7=0.008954, 14=0.009175)
HIDDEN_SIZE = 128    # 256 caused underdispersion — do not increase
NUM_LAYERS  = 2
DROPOUT     = 0.2
BATCH_SIZE  = 32
LR          = 0.001  # 0.0003 caused underdispersion — keep at 0.001
GRAD_CLIP   = 1.0    # clip_grad_norm_ applied before optimizer.step()
EPOCHS      = 50
ES_PATIENCE = 10     # early stopping patience
```

---

## Phase 1 Baseline Results — Reference Numbers
These are the numbers Phase 2 must improve upon.
DO NOT overwrite metrics_lstm.json — load it for comparison.

```
Overall test RMSE (no AOD): 1.1944 MJ/m²/day
Overall persistence RMSE:   0.8956 MJ/m²/day

Per-station LSTM test RMSE (no AOD):
  Ahmedabad  : 1.2497
  Bengaluru  : 1.2294
  Bhopal     : 1.1253
  Chennai    : 1.3112
  Hyderabad  : 1.0725
  Jodhpur    : 1.2268
  Kolkata    : 1.0803
  Mumbai     : 1.2377
```

---

## NASA POWER API Specification
- Base URL: https://power.larc.nasa.gov/api/temporal/daily/point
- Parameters: ALLSKY_SFC_SW_DWN,T2M,RH2M,WS10M,CLOUD_AMT
- Community: RE
- Format: JSON
- Max concurrent requests: 4
- Fill value: -999
- No API key or registration required
- Column rename map:
    ALLSKY_SFC_SW_DWN → ghi          (MJ/m²/day)
    T2M               → temperature  (°C)
    RH2M              → humidity     (%)
    WS10M             → wind_speed   (m/s)
    CLOUD_AMT         → cloud_cover  (%)

---

## MERRA-2 AOD Specification
- Product:    M2T1NXAER (tavg1_2d_aer_Nx)
- Resolution: 0.625° lon × 0.5° lat, hourly
- Daily mean: average all 24 hourly values per day
- Fill value: 1e15 → NaN immediately on load
- Access:     NASA Earthdata login required
- Links file: data/raw/nasa-merra2_aod.txt
              (OPeNDAP URLs, one per day, pre-clipped to India)
- Credentials: stored in .env only — never hardcoded

AOD variables (all 5 used):
  TOTEXTTAU = Total extinction AOD at 550nm  ← PRIMARY feature
  DUEXTTAU  = Dust extinction AOD            ← Jodhpur/Rajasthan signal
  BCEXTTAU  = Black carbon extinction AOD    ← Kolkata/Mumbai signal
  SSEXTTAU  = Sea salt extinction AOD        ← Chennai/Mumbai coastal
  SUEXTTAU  = Sulfate extinction AOD         ← Industrial all-India

Output column naming convention in merra2_aod_stations.csv:
  date, Jodhpur_TOTEXTTAU, Jodhpur_DUEXTTAU, ..., Bengaluru_SUEXTTAU
  (40 columns total: date + 8 stations × 5 variables)

---

## Phase 2 Feature Set
Phase 1 features (9):
  ghi, temperature, humidity, wind_speed, cloud_cover,
  monsoon_phase, doy_sin, doy_cos, month_sin, month_cos

Phase 2 features (14) — add these 5:
  TOTEXTTAU, DUEXTTAU, BCEXTTAU, SSEXTTAU, SUEXTTAU

input_size changes from 9 → 14
All other hyperparameters remain identical to Phase 1.
New scaler saved to models/scaler_aod.pkl (separate from Phase 1)
New model saved to models/lstm_aod_best.pt

---

## Monsoon Phase Rules
- 0 = pre_monsoon    (before onset date)
- 1 = active_monsoon (onset to withdrawal inclusive)
- 2 = post_monsoon   (after withdrawal date)
- Labels are per-station per-day — NOT national averages
- All date ranges resolved to midpoint single date

---

## Cyclical Encoding — Use Exactly This
```python
import numpy as np
# day_of_year (N=366), month (N=12)
df['doy_sin']   = np.sin(2 * np.pi * df['day_of_year'] / 366)
df['doy_cos']   = np.cos(2 * np.pi * df['day_of_year'] / 366)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
```

---

## SlidingWindowDataset — Critical Implementation Note
The off-by-one bug from Phase 1 must never recur.
Correct implementation:

```python
# When building (X, y) pairs:
# X = features[i : i + seq_len]       shape: (seq_len, n_features)
# y = targets[i + seq_len - 1]        ← THIS IS CORRECT
#                                        = ghi of day after window ends
#
# targets array = ghi.shift(-1) values
# So targets[i + seq_len - 1] = ghi one day after the window
#
# WRONG (old bug): targets[i + seq_len]  ← 2 days ahead
# CORRECT:         targets[i + seq_len - 1]  ← 1 day ahead
```

Verify with this check before training:
```python
# Window ending on date D should predict GHI on date D+1
# Print and manually verify for first 3 windows
```

---

## Project Folder Structure
```
solar_forecast/
├── data/
│   ├── raw/
│   │   └── nasa-merra2_aod.txt     ← OPeNDAP links file
│   └── processed/
│       ├── nasa_power_all_stations.csv
│       ├── monsoon_phases.csv
│       ├── dataset_final.csv
│       └── merra2_aod_stations.csv ← Phase 2
├── scripts/
│   ├── fetch_nasa_power.py         ← Script 1: GHI + met data
│   ├── monsoon_labels.py           ← Script 2: phase labels
│   ├── merge_dataset.py            ← Script 3: merge + features
│   ├── baseline.py                 ← Script 4: persistence baseline
│   ├── lstm_model.py               ← Script 5: LSTM Phase 1
│   ├── fetch_merra2_aod.py         ← Script 6: AOD download
│   └── add_aod_retrain.py          ← Script 7: LSTM Phase 2
├── models/
│   ├── scaler.pkl                  ← Phase 1 scaler (9 features)
│   ├── lstm_best.pt                ← Phase 1 best model
│   ├── scaler_aod.pkl              ← Phase 2 scaler (14 features)
│   └── lstm_aod_best.pt            ← Phase 2 best model
├── outputs/
│   ├── metrics_persistence.json
│   ├── metrics_lstm.json           ← Phase 1 — DO NOT OVERWRITE
│   ├── metrics_lstm_aod.json       ← Phase 2
│   ├── loss_curve.png              ← Phase 1
│   ├── loss_curve_aod.png          ← Phase 2
│   ├── predictions_sample.png      ← Phase 1
│   └── comparison_aod.png          ← Phase 2 three-way comparison
├── AGENTS.md
├── .env                            ← credentials, never commit
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Script Execution Order
```
Phase 1 (complete):
1. fetch_nasa_power.py   → nasa_power_all_stations.csv   ✓ DONE
2. monsoon_labels.py     → monsoon_phases.csv             ✓ DONE
3. merge_dataset.py      → dataset_final.csv              ✓ DONE
4. baseline.py           → metrics_persistence.json       ✓ DONE
5. lstm_model.py         → lstm_best.pt + metrics_lstm.json  ✓ DONE

Phase 2 (now):
6. fetch_merra2_aod.py   → merra2_aod_stations.csv
7. add_aod_retrain.py    → lstm_aod_best.pt + metrics_lstm_aod.json
```

Each script is standalone and fully re-runnable.
Each script verifies its own output in a SELF CHECK block.

---

## Key Lessons From Phase 1 — Do Not Repeat
1. Off-by-one in SlidingWindowDataset target index
   was the critical bug — model forecast 2 steps ahead
   while persistence forecasts 1 step. Never use targets[i+seq_len].
   Always use targets[i+seq_len-1].

2. MPS inference artifact inflates reported val loss by ~2.5x.
   Always evaluate on CPU. See MPS Inference Fix section above.

3. SEQ_LEN=7 is confirmed best for this dataset.
   Do not change without a full scan.

4. LR=0.001 and HIDDEN=128 are confirmed best.
   LR=0.0003 and HIDDEN=256 both caused underdispersion.

5. LSTM is underdispersed at 39% of actual variance.
   AOD is the expected fix — it provides the atmospheric
   signal needed to predict sharp irradiance drops.