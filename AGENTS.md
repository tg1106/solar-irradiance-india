```markdown
# AGENTS.md — Solar-Wind Hybrid Energy Generation Forecasting
# India Multi-Zone — READ THIS FULLY BEFORE ANY CODE

## Project Summary
Two-phase research project:

PHASE A (COMPLETE): Solar irradiance forecasting
- LSTM baseline, LSTM+AOD, SolarTransformer+MonsoonEmbedding
- 8 stations, 4 climate zones, Apr 2017–Dec 2024
- Best result: Transformer beats persistence 8/8 stations
- Overall RMSE: 0.7841 MJ/m²/day vs 0.8956 persistence

PHASE B (NOW): Hybrid Solar-Wind Energy Generation
- Step 1: Forecast GHI + wind speed (Transformer already trained)
- Step 2: Convert to Solar RGI + Wind RGI using physics
- Step 3: Find optimal solar:wind ratio per station
- Output: per-station generation forecasts + optimal mix ratios

---

## Absolute Rules — Never Break

### Cross-platform
- ALWAYS use pathlib.Path — never os.path, never / or \
- Use Path(__file__).resolve().parent.parent for all paths
- Code must run identically on macOS and Windows

### Data integrity
- NASA POWER fill value: -999 → NaN immediately
- MERRA-2 fill value: 1e15 → NaN immediately
- Never modify raw data — save processed to new file
- Always print row counts before and after any filter

### ML integrity — CRITICAL
- Train/val/test split: STRICTLY TIME-BASED, never random
- Train: 2017-04-01 to 2022-12-31
- Val:   2023-01-01 to 2023-12-31
- Test:  2024-01-01 to 2024-12-31
- Fit ALL scalers on TRAIN ONLY
- Target = ghi.shift(-1) per station — 1 step ahead
- SlidingWindowDataset target index: targets[i+seq_len-1]
  NEVER targets[i+seq_len] — off-by-one was critical Phase A bug

### MPS Inference Fix — MANDATORY
All evaluation loops must run on CPU:
```python
model.eval()
eval_device = torch.device('cpu')
model_cpu = model.to(eval_device)
with torch.no_grad():
for xb, yb in loader:
        xb, yb = xb.to(eval_device), yb.to(eval_device)
        pred = model_cpu(xb)
model.to(DEVICE)
```

### Device Selection

```python
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')
print(f"Using device: {DEVICE}")
```

---

## Station Coordinates — DO NOT MODIFY

```python
STATIONS ={
'Jodhpur':(26.30,73.02),
'Ahmedabad':(23.03,72.58),
'Mumbai':(19.08,72.88),
'Chennai':(13.08,80.27),
'Hyderabad':(17.38,78.48),
'Bhopal':(23.25,77.40),
'Kolkata':(22.57,88.36),
'Bengaluru':(12.97,77.59),
}
```

---

## Phase A Results — DO NOT OVERWRITE

```




Persistence overall test RMSE:    0.8956 MJ/m²/day
LSTM Phase 1 overall test RMSE:   1.0948
LSTM+AOD Phase 2 overall test RMSE: 1.0369
Transformer Phase 3 overall test RMSE: 0.7841  R²=0.6881


Monsoon phase breakdown (Transformer):
pre_monsoon:    +11.63% over persistence
active_monsoon: +14.35% over persistence  ← largest gain
post_monsoon:   +9.79%  over persistence








```

---

## Phase B — New Data Requirements

### Additional NASA POWER Parameters Needed

Re-fetch with these additional parameters for all 8 stations:
  PS    = Surface pressure (kPa) — needed for air density
  WS50M = Wind speed at 50m (m/s) — turbine hub height

Existing parameters already downloaded:
  ALLSKY_SFC_SW_DWN → ghi (MJ/m²/day)
  T2M               → temperature (°C)
  RH2M              → humidity (%)
  WS10M             → wind_speed_10m (m/s)
  CLOUD_AMT         → cloud_cover (%)

New fetch output: data/processed/nasa_power_extended.csv
Columns: date, station, ghi, temperature, humidity,
         wind_speed_10m, cloud_cover,
         surface_pressure, wind_speed_50m

---

## Physics Conversion Formulas — Phase B Step 2

### Solar Relative Generation Index (Solar RGI)

```python
# Standard PV generation model
PANEL_EFFICIENCY  = 0.18    # 18% — typical monocrystalline
TEMP_COEFFICIENT  = 0.004   # 0.4%/°C power loss above T_ref
T_REF             = 25.0    # Standard test condition temperature

def solar_rgi(ghi, temperature):
    """
    Convert GHI to Solar RGI [0, 1].
    Accounts for temperature-dependent efficiency loss.
    """
    efficiency = PANEL_EFFICIENCY * (
        1 - TEMP_COEFFICIENT * (temperature - T_REF)
    )
    # GHI in MJ/m²/day → Wh/m²/day × efficiency = Wh generated
    # Normalise by max possible (clear sky peak ~8 MJ/m²/day)
    GHI_MAX = 8.0  # MJ/m²/day — empirical max from dataset
    raw = efficiency * ghi / (PANEL_EFFICIENCY * GHI_MAX)
    return float(np.clip(raw, 0, 1))
```

### Wind Relative Generation Index (Wind RGI)

```python
# Standard wind turbine power curve model
R_GAS    = 287.05   # J/(kg·K) specific gas constant for dry air
CUT_IN   = 3.0      # m/s — turbine starts generating
RATED    = 12.0     # m/s — turbine at full power
CUT_OUT  = 25.0     # m/s — turbine shuts down for safety
CP       = 0.4      # Betz efficiency coefficient

def air_density(pressure_kpa, temperature_c):
    """
    Compute air density from surface pressure and temperature.
    ρ = P / (R × T)
    """
    P = pressure_kpa * 1000   # kPa → Pa
    T = temperature_c + 273.15  # °C → K
    return P / (R_GAS * T)

def wind_rgi(wind_speed_50m, pressure_kpa, temperature_c):
    """
    Convert wind speed at 50m to Wind RGI [0, 1].
    Uses standard turbine power curve with air density correction.
    """
    v = wind_speed_50m
    rho = air_density(pressure_kpa, temperature_c)
    rho_std = 1.225  # kg/m³ standard air density at sea level

    if v < CUT_IN or v > CUT_OUT:
        return 0.0

    # Power output proportional to ρv³
    # Normalised to rated power at v=RATED, rho=rho_std
    if v >= RATED:
        power_ratio = 1.0
    else:
        power_ratio = (rho / rho_std) * (v**3) / (RATED**3)

    return float(np.clip(power_ratio, 0, 1))
```

### Optimal Mix Optimisation

```python
# For each station find solar:wind ratio maximising
# combined RGI while minimising variance (stability)

def find_optimal_mix(solar_rgi_series, wind_rgi_series):
    """
    Sweep solar fractions 0% to 100% in 1% steps.
    For each fraction compute:
      - Mean combined RGI (maximise this)
      - Std combined RGI (minimise this — stability)
      - Sharpe-like ratio: mean / std (maximise this)
    Return the fraction maximising mean/std ratio.
    """
    best_ratio = -1
    best_solar_pct = 0.5

    for solar_frac in np.arange(0, 1.01, 0.01):
        wind_frac = 1 - solar_frac
        combined = solar_frac * solar_rgi_series + \
                   wind_frac * wind_rgi_series
        mean_gen = combined.mean()
        std_gen  = combined.std()
        if std_gen > 0:
            ratio = mean_gen / std_gen
        else:
            ratio = mean_gen
        if ratio > best_ratio:
            best_ratio = ratio
            best_solar_pct = solar_frac

    return best_solar_pct, 1 - best_solar_pct
```

---

## Phase B Script Execution Order

```




Phase B (added — complete):
8.  fetch_extended_power.py  → nasa_power_extended.csv          [DONE]
9.  compute_rgi.py           → rgi_dataset.csv                  [DONE]
10. optimize_mix.py          → optimal_mix.csv + mix_chart.png
                                + mix_sweep_curves.png           [DONE]
11. forecast_generation.py   → generation_forecasts.csv
                                + generation_trends.png
                                + metrics_generation.json        [DONE]








```

Phase A scripts (1-7) remain unchanged.
Phase B builds on top of Phase A outputs.

---

## Phase B Output Files

| File                      | Location        | Description                                | Size  | Rows   |
| ------------------------- | ---------------- | ------------------------------------------- | ----- | ------ |
| nasa_power_extended.csv   | data/processed/ | GHI + met + PS + WS50M                       | 1.3M  | 22,656 |
| rgi_dataset.csv           | data/processed/ | Daily Solar RGI + Wind RGI per station       | 3.4M  | 22,656 |
| optimal_mix.csv           | outputs/        | Optimal solar:wind % per station             | 4.0K  | 8      |
| generation_forecasts.csv  | outputs/        | Forecasted RGI per station test year         | 368K  | 2,864  |
| mix_chart.png             | outputs/        | Bar chart: optimal ratios all stations       | 60K   | --     |
| mix_sweep_curves.png      | outputs/        | Generation/stability sweep per station       | 200K  | --     |
| generation_trends.png     | outputs/        | 2x2 actual vs forecasted combined RGI        | 676K  | --     |
| metrics_generation.json   | outputs/        | Per-station MAE/RMSE on combined RGI         | 4.0K  | --     |

### Phase B Key Result -- Optimal Solar:Wind Mix (Sharpe-style mean/std)

| Station   | Optimal Solar% | Optimal Wind% | Mean RGI | Std RGI |
| --------- | --------------: | --------------: | -------: | ------: |
| Jodhpur   | 100.0% | 0.0%  | 0.6675 | 0.1576 |
| Hyderabad | 67.0%  | 33.0% | 0.4575 | 0.0955 |
| Kolkata   | 66.0%  | 34.0% | 0.3855 | 0.1037 |
| Bengaluru | 65.0%  | 35.0% | 0.4722 | 0.0929 |
| Bhopal    | 62.0%  | 38.0% | 0.4166 | 0.1168 |
| Ahmedabad | 60.0%  | 40.0% | 0.4297 | 0.1029 |
| Chennai   | 52.0%  | 48.0% | 0.3707 | 0.0815 |
| Mumbai    | 50.0%  | 50.0% | 0.3598 | 0.0781 |

Note: with the AGENTS.md turbine constants (RATED=12 m/s) against typical
50m wind speeds (~5 m/s mean across all 8 stations), Wind RGI is small
relative to Solar RGI everywhere, so the optimal mix is solar-dominant
at every station. The monsoon-vs-non-monsoon split still shows the
expected complementarity shift (more wind-favourable during active
monsoon for most stations, e.g. Bhopal 51% solar in monsoon vs 100%
non-monsoon).

---

## Confirmed Phase A Hyperparameters — DO NOT CHANGE

```python
SEQ_LEN     = 7
HIDDEN_SIZE = 128
D_MODEL     = 64
N_HEADS     = 4
N_LAYERS    = 2
DIM_FF      = 128
DROPOUT     = 0.1
LR_TRANSFORMER = 0.0005
GRAD_CLIP   = 1.0
ES_PATIENCE = 10
```

---

## Key Lessons From Phase A — Do Not Repeat

1. Target index: targets[i+seq_len-1] not targets[i+seq_len]
2. MPS inference artifact — always evaluate on CPU
3. SEQ_LEN=7 confirmed best — do not change
4. MOSDAC had massive gaps — use NASA POWER as primary
5. netrc authentication required for MERRA-2 OPeNDAP

```

```
