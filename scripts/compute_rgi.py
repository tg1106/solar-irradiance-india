"""
compute_rgi.py
Phase B — Script 9.
Convert raw GHI and 50m wind speed into Solar RGI and Wind RGI
("Relative Generation Index", both in [0, 1]) using the physics
formulas defined in AGENTS.md, and pre-compute combined RGI at
three reference solar:wind mix ratios.

Input:  data/processed/nasa_power_extended.csv
        data/processed/monsoon_phases.csv
Output: data/processed/rgi_dataset.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Path anchors ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

EXTENDED_CSV = ROOT / 'data' / 'processed' / 'nasa_power_extended.csv'
MONSOON_CSV  = ROOT / 'data' / 'processed' / 'monsoon_phases.csv'
OUTPUT_CSV   = ROOT / 'data' / 'processed' / 'rgi_dataset.csv'

# ── Physics constants — AGENTS.md, do not change ────────────────────────────────
PANEL_EFFICIENCY = 0.18    # 18% — typical monocrystalline
TEMP_COEFFICIENT = 0.004   # 0.4%/°C power loss above T_ref
T_REF            = 25.0    # Standard test condition temperature
GHI_MAX          = 8.0     # MJ/m²/day — empirical max from dataset

R_GAS    = 287.05   # J/(kg·K) specific gas constant for dry air
CUT_IN   = 3.0      # m/s — turbine starts generating
RATED    = 12.0     # m/s — turbine at full power
CUT_OUT  = 25.0     # m/s — turbine shuts down for safety
CP       = 0.4      # Betz efficiency coefficient


# ── Physics functions — AGENTS.md, implemented exactly ──────────────────────────

def solar_rgi(ghi: float, temperature: float) -> float:
    """Convert GHI to Solar RGI [0, 1].

    Accounts for temperature-dependent efficiency loss.

    Args:
        ghi: Daily global horizontal irradiance (MJ/m²/day).
        temperature: Daily mean air temperature (°C).

    Returns:
        Solar RGI in [0, 1].
    """
    efficiency = PANEL_EFFICIENCY * (
        1 - TEMP_COEFFICIENT * (temperature - T_REF)
    )
    # GHI in MJ/m²/day → Wh/m²/day × efficiency = Wh generated
    # Normalise by max possible (clear sky peak ~8 MJ/m²/day)
    raw = efficiency * ghi / (PANEL_EFFICIENCY * GHI_MAX)
    return float(np.clip(raw, 0, 1))


def air_density(pressure_kpa: float, temperature_c: float) -> float:
    """Compute air density from surface pressure and temperature.

    ρ = P / (R × T)

    Args:
        pressure_kpa: Surface pressure (kPa).
        temperature_c: Air temperature (°C).

    Returns:
        Air density (kg/m³).
    """
    P = pressure_kpa * 1000      # kPa → Pa
    T = temperature_c + 273.15   # °C → K
    return P / (R_GAS * T)


def wind_rgi(wind_speed_50m: float, pressure_kpa: float, temperature_c: float) -> float:
    """Convert wind speed at 50m to Wind RGI [0, 1].

    Uses a standard turbine power curve with air-density correction.

    Args:
        wind_speed_50m: Wind speed at 50m hub height (m/s).
        pressure_kpa: Surface pressure (kPa).
        temperature_c: Air temperature (°C).

    Returns:
        Wind RGI in [0, 1].
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
        power_ratio = (rho / rho_std) * (v ** 3) / (RATED ** 3)

    return float(np.clip(power_ratio, 0, 1))


def combined_rgi(solar_frac: float, solar_rgi_val: float, wind_rgi_val: float) -> float:
    """Blend Solar RGI and Wind RGI at a given solar fraction.

    Args:
        solar_frac: Fraction of capacity allocated to solar, in [0, 1].
        solar_rgi_val: Solar RGI value [0, 1].
        wind_rgi_val: Wind RGI value [0, 1].

    Returns:
        Combined RGI in [0, 1].
    """
    return solar_frac * solar_rgi_val + (1 - solar_frac) * wind_rgi_val


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    """Apply physics conversions to the extended NASA POWER dataset."""
    print(f"\n{'='*60}")
    print("compute_rgi.py — Phase B Step 2: Physics conversion")
    print(f"{'='*60}\n")

    if not EXTENDED_CSV.exists():
        print(f"ERROR: nasa_power_extended.csv not found: {EXTENDED_CSV}")
        print("Run scripts/fetch_extended_power.py first.")
        sys.exit(1)
    if not MONSOON_CSV.exists():
        print(f"ERROR: monsoon_phases.csv not found: {MONSOON_CSV}")
        print("Run scripts/monsoon_labels.py first.")
        sys.exit(1)

    print("Step 1 — Loading inputs ...")
    df = pd.read_csv(EXTENDED_CSV, parse_dates=['date'])
    monsoon = pd.read_csv(MONSOON_CSV, parse_dates=['date'])
    print(f"  nasa_power_extended rows : {len(df):,}")
    print(f"  monsoon_phases rows      : {len(monsoon):,}")

    print("\nStep 2 — Merging monsoon_phase ...")
    rows_before = len(df)
    df = pd.merge(
        df,
        monsoon[['date', 'station', 'monsoon_phase', 'phase_name']],
        on=['date', 'station'],
        how='left',
    )
    print(f"  Rows before merge: {rows_before:,}  after: {len(df):,}")
    print(f"  monsoon_phase missing: {df['monsoon_phase'].isna().sum()}")

    print("\nStep 3 — Computing air_density, solar_rgi, wind_rgi (row-wise) ...")
    df['air_density'] = df.apply(
        lambda r: air_density(r['surface_pressure'], r['temperature']), axis=1
    )
    df['solar_rgi'] = df.apply(
        lambda r: solar_rgi(r['ghi'], r['temperature']), axis=1
    )
    df['wind_rgi'] = df.apply(
        lambda r: wind_rgi(r['wind_speed_50m'], r['surface_pressure'], r['temperature']), axis=1
    )

    print("\nStep 4 — Computing combined RGI at reference ratios (50/50, 70/30, 30/70) ...")
    df['combined_rgi_50_50'] = df.apply(
        lambda r: combined_rgi(0.5, r['solar_rgi'], r['wind_rgi']), axis=1
    )
    df['combined_rgi_70_30'] = df.apply(
        lambda r: combined_rgi(0.7, r['solar_rgi'], r['wind_rgi']), axis=1
    )
    df['combined_rgi_30_70'] = df.apply(
        lambda r: combined_rgi(0.3, r['solar_rgi'], r['wind_rgi']), axis=1
    )

    print("\nStep 5 — Selecting output columns ...")
    out_cols = [
        'date', 'station', 'ghi', 'temperature',
        'wind_speed_50m', 'surface_pressure',
        'solar_rgi', 'wind_rgi',
        'combined_rgi_50_50', 'combined_rgi_70_30', 'combined_rgi_30_70',
        'monsoon_phase', 'air_density',
    ]
    df_out = df[out_cols].sort_values(['station', 'date']).reset_index(drop=True)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved → {OUTPUT_CSV}")

    # ── SELF CHECK ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SELF CHECK")
    print(f"{'='*60}")

    print(f"\nTotal rows: {len(df_out):,}")

    solar_range_ok = bool(df_out['solar_rgi'].between(0, 1).all())
    wind_range_ok  = bool(df_out['wind_rgi'].between(0, 1).all())
    print(f"\nSolar RGI in [0,1] for all rows: {solar_range_ok}")
    print(f"  range: [{df_out['solar_rgi'].min():.4f}, {df_out['solar_rgi'].max():.4f}]")
    print(f"Wind RGI in [0,1] for all rows : {wind_range_ok}")
    print(f"  range: [{df_out['wind_rgi'].min():.4f}, {df_out['wind_rgi'].max():.4f}]")
    assert solar_range_ok, "Solar RGI out of [0,1] range!"
    assert wind_range_ok,  "Wind RGI out of [0,1] range!"

    print("\nPer-station mean Solar RGI / Wind RGI:")
    means = df_out.groupby('station')[['solar_rgi', 'wind_rgi']].mean()
    print(means.round(4).to_string())

    jodhpur_solar = means.loc['Jodhpur', 'solar_rgi']
    kolkata_solar = means.loc['Kolkata', 'solar_rgi']
    print(f"\nPhysical sanity — Jodhpur solar_rgi ({jodhpur_solar:.4f}) "
          f"> Kolkata solar_rgi ({kolkata_solar:.4f}): {jodhpur_solar > kolkata_solar}")

    coastal  = ['Mumbai', 'Chennai']
    inland   = ['Jodhpur', 'Bhopal']
    coastal_wind = means.loc[coastal, 'wind_rgi'].mean()
    inland_wind  = means.loc[inland, 'wind_rgi'].mean()
    print(f"Physical sanity — coastal mean wind_rgi ({coastal_wind:.4f}) "
          f"> inland mean wind_rgi ({inland_wind:.4f}): {coastal_wind > inland_wind}")

    print(f"\n{'='*60}")
    print("Script 9 complete.")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
