# Outputs Guide

What each generated file in `outputs/`, `data/processed/`, and `models/` actually
shows, grouped by phase. Click any filename (or its preview image) to open it.

---

## Phase 1 — LSTM Baseline

[![loss_curve.png](outputs/loss_curve.png)](outputs/loss_curve.png)

**[`outputs/loss_curve.png`](outputs/loss_curve.png)**
Training vs CPU-evaluated validation MSE loss per epoch for the Phase 1 LSTM. Use
this to check for overfitting (val loss diverging upward) or under-training
(both curves still falling at the early-stopping point).

[![predictions_sample.png](outputs/predictions_sample.png)](outputs/predictions_sample.png)

**[`outputs/predictions_sample.png`](outputs/predictions_sample.png)**
Actual vs predicted GHI for Chennai, test year 2024, Phase 1 LSTM. Shows how
closely the LSTM tracks day-to-day irradiance swings (and where it lags during
monsoon cloud cover).

**[`outputs/metrics_persistence.json`](outputs/metrics_persistence.json)**
The "predict tomorrow = today" baseline: MAE, RMSE, R², and skill score per
station + overall. Every model is judged against this floor — if a model
can't beat persistence, it isn't adding value.

**[`outputs/metrics_lstm.json`](outputs/metrics_lstm.json)**
Phase 1 LSTM test-set MAE / RMSE / R² per station + overall (`OVERALL` key).
Do not overwrite — Phase 3's comparison chart reads this file.

---

## Phase 2 — LSTM + MERRA-2 AOD

[![loss_curve_aod.png](outputs/loss_curve_aod.png)](outputs/loss_curve_aod.png)

**[`outputs/loss_curve_aod.png`](outputs/loss_curve_aod.png)**
Train vs val loss curve for the AOD-augmented LSTM (15 input features instead
of 10).

[![comparison_aod.png](outputs/comparison_aod.png)](outputs/comparison_aod.png)

**[`outputs/comparison_aod.png`](outputs/comparison_aod.png)**
Three-way grouped bar chart of test RMSE per station: **Persistence vs LSTM
vs LSTM+AOD**. Shows whether adding aerosol optical depth (TOTEXTTAU, etc.)
improves on the Phase 1 LSTM.

**[`outputs/metrics_lstm_aod.json`](outputs/metrics_lstm_aod.json)**
Phase 2 LSTM+AOD test-set MAE / RMSE / R² per station + overall.

---

## Phase 3 — Transformer + Monsoon Embedding

[![loss_curve_transformer.png](outputs/loss_curve_transformer.png)](outputs/loss_curve_transformer.png)

**[`outputs/loss_curve_transformer.png`](outputs/loss_curve_transformer.png)**
Train vs val loss curve for the SolarTransformer (monsoon-phase embedding +
attention).

[![predictions_transformer_chennai.png](outputs/predictions_transformer_chennai.png)](outputs/predictions_transformer_chennai.png)

**[`outputs/predictions_transformer_chennai.png`](outputs/predictions_transformer_chennai.png)**
Actual vs predicted GHI for Chennai, test year 2024, Transformer+AOD. Direct
visual comparison to `predictions_sample.png` (Phase 1) on the same station
and period.

[![comparison_final.png](outputs/comparison_final.png)](outputs/comparison_final.png)

**[`outputs/comparison_final.png`](outputs/comparison_final.png)**
The headline result chart. Four-way grouped bar chart of test RMSE per
station: **Persistence vs LSTM vs LSTM+AOD vs Transformer+AOD**. Lower bars
are better. This is the chart that shows the Transformer beats persistence on
8/8 stations (overall RMSE 0.7841 vs 0.8956 MJ/m²/day).

**[`outputs/metrics_transformer.json`](outputs/metrics_transformer.json)**
Phase 3 Transformer+AOD test-set MAE / RMSE / R² per station + overall — the
numbers behind `comparison_final.png`.

---

## Phase B — Hybrid Solar-Wind Generation

[![mix_chart.png](outputs/mix_chart.png)](outputs/mix_chart.png)

**[`outputs/mix_chart.png`](outputs/mix_chart.png)**
Bar chart of the optimal solar % vs wind % capacity split per station
(sorted by solar fraction). Jodhpur is 100% solar; Mumbai is the most
wind-balanced at 50/50.

[![mix_sweep_curves.png](outputs/mix_sweep_curves.png)](outputs/mix_sweep_curves.png)

**[`outputs/mix_sweep_curves.png`](outputs/mix_sweep_curves.png)**
For every station, the generation/stability ratio (mean combined RGI ÷ std
combined RGI) as the solar fraction is swept 0→100%. The marked dot on each
curve is that station's optimal mix — shows how sharply peaked (or flat) the
optimum is.

**[`outputs/optimal_mix.csv`](outputs/optimal_mix.csv)**
Per-station table: optimal mix (max generation/stability ratio), max-generation
mix, min-variance mix, and the mix recomputed separately for active-monsoon
vs non-monsoon days — i.e. whether the ideal solar:wind ratio shifts
seasonally.

[![generation_trends.png](outputs/generation_trends.png)](outputs/generation_trends.png)

**[`outputs/generation_trends.png`](outputs/generation_trends.png)**
2×2 grid (Jodhpur, Chennai, Kolkata, Bengaluru) of actual vs forecasted
combined RGI over the 2024 test year, using each station's optimal mix. Solar
RGI comes from the Transformer's GHI forecast; wind RGI uses a 1-day
persistence forecast of 50m wind speed.

**[`outputs/generation_forecasts.csv`](outputs/generation_forecasts.csv)**
Row-level data behind `generation_trends.png`: actual/forecasted solar RGI,
actual/persistence wind RGI, optimal mix %, and actual/forecasted combined
RGI for every test-set day and station.

**[`outputs/metrics_generation.json`](outputs/metrics_generation.json)**
Per-station MAE / RMSE between actual and forecasted combined RGI — how
accurate the hybrid generation forecast is once solar (Transformer) and wind
(persistence) are blended.

---

## Processed Datasets (`data/processed/`)

- **[`dataset_final.csv`](data/processed/dataset_final.csv)** — the training-ready table for Phases 1-3: GHI, met variables, monsoon phase, lag/rolling/cyclical features, 8 stations × ~2,832 days.
- **[`rgi_dataset.csv`](data/processed/rgi_dataset.csv)** — Phase B's physics-converted table: Solar RGI, Wind RGI, combined RGI at 50/50, 70/30 and 30/70 reference ratios, plus air density.
- **[`nasa_power_extended.csv`](data/processed/nasa_power_extended.csv)** — raw NASA POWER pull including the extra surface-pressure and 50m-wind-speed fields needed for Phase B.
- **[`merra2_aod_stations.csv`](data/processed/merra2_aod_stations.csv)** — daily MERRA-2 aerosol optical depth (5 variables) per station, used by Phases 2-3.
- **[`monsoon_phases.csv`](data/processed/monsoon_phases.csv)** — per-station per-day monsoon phase label (0 = pre, 1 = active, 2 = post), from IMD onset/withdrawal dates.

## Saved Models (`models/`)

- **[`lstm_best.pt`](models/lstm_best.pt)** / **[`scaler.pkl`](models/scaler.pkl)** — Phase 1 LSTM weights + feature scaler.
- **[`lstm_aod_best.pt`](models/lstm_aod_best.pt)** / **[`scaler_aod.pkl`](models/scaler_aod.pkl)** — Phase 2 LSTM+AOD weights + scaler.
- **[`transformer_best.pt`](models/transformer_best.pt)** / **[`scaler_transformer.pkl`](models/scaler_transformer.pkl)** — Phase 3 Transformer weights + scaler. Reused as-is by Phase B (no retraining).
