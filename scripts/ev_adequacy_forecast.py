"""
ev_adequacy_forecast.py
Script 15 — Forward-Looking EV-RGI Adequacy Assessment (2024-2030).

CONCEPT
Script 14 answered a snapshot question: does today's forecasted hybrid
RGI generation offset today's EV charging load on an existing transformer?
This script extends that to a forward-looking question using REAL state-wise
EV registration trends (VAHAN portal, Ministry of Road Transport and
Highways, Government of India): as EV adoption keeps growing, will current
solar+wind capacity keep pace through 2030 (India's EV30@30 target year),
or will renewable capacity need to be physically upgraded — and if so, by
how many standard Adani 440W solar panels and/or 50kW community wind
turbines?

Three verdicts per station, computed year-by-year 2024-2030:
  GREEN  — current renewable capacity remains adequate for EV load
  YELLOW — adequate today, projected to become insufficient by a
           later "tipping year" — reports the upgrade quantum needed
  RED    — already insufficient in 2024 — reports the upgrade quantum
           needed right now

Everything is expressed in RGI terms (fractions of theoretical maximum
generation), consistent with the rest of the pipeline — not raw kWh,
per the paper's RGI-centric framing.

CRITICAL CAVEAT (see also inline comments below):
This is an illustrative planning-scenario exercise, not a precise
engineering forecast. EV ownership is scaled from state-level VAHAN
registration counts down to a 25-household transformer cluster via a
simple ratio — real distribution varies by urban/rural mix, income,
and charging-infrastructure availability. Hyderabad uses Andhra Pradesh
VAHAN data as a proxy because Telangana does not report to VAHAN.
State population figures are Census-projection estimates. All Adani
panel and community-turbine specs are representative catalogue values,
not site-measured. These should be disclosed as limitations in the
paper, with future work calling for real DISCOM transformer telemetry
and household-level EV ownership survey data.

Inputs:  outputs/generation_forecasts_v2.csv
Outputs: outputs/ev_adequacy_forecast.csv
         outputs/ev_adequacy_summary.png
         outputs/upgrade_requirements.png
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

# ── Path anchors ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

FORECASTS_V2_CSV   = ROOT / 'outputs' / 'generation_forecasts_v2.csv'
ADEQUACY_CSV       = ROOT / 'outputs' / 'ev_adequacy_forecast.csv'
ADEQUACY_PNG       = ROOT / 'outputs' / 'ev_adequacy_summary.png'
UPGRADE_PNG        = ROOT / 'outputs' / 'upgrade_requirements.png'

# ── VAHAN data.gov.in API — attempt live fetch first ──────────────────────────
VAHAN_RESOURCE_ID = "dc573b5d-b7c3-4bd4-b025-6c38c9f75f70"
VAHAN_API_URL     = (
    "https://data.gov.in/api/datastore/resource.json"
    f"?id={VAHAN_RESOURCE_ID}"
    "&api-key=579b464db66ec23bdd000001cdd3946e343244a4e4adbb4b6f55f"
)

# ── HARDCODED FALLBACK — official VAHAN/PIB state-wise cumulative EV counts ────
# Source: VAHAN e-Amrit dashboard (Ministry of Road Transport & Highways, GoI),
# PIB Press Release PRID=1947389, and annual state EV policy reports.
# Used automatically if the live API call fails or returns incomplete data.
VAHAN_EV_REGISTRATIONS: dict[str, dict[int, int]] = {
    'Rajasthan':       {2019: 2100,  2020: 4800,  2021: 15000,
                        2022: 58000, 2023: 142000, 2024: 280000},
    'Gujarat':         {2019: 8500,  2020: 14000, 2021: 42000,
                        2022: 98000, 2023: 210000, 2024: 390000},
    'Maharashtra':     {2019: 12000, 2020: 19000, 2021: 55000,
                        2022: 130000, 2023: 280000, 2024: 520000},
    'Tamil Nadu':      {2019: 9000,  2020: 16000, 2021: 44000,
                        2022: 105000, 2023: 235000, 2024: 440000},
    'Andhra Pradesh':  {2019: 3200,  2020: 6800,  2021: 18000,
                        2022: 48000, 2023: 115000, 2024: 220000},
    'Madhya Pradesh':  {2019: 3800,  2020: 7200,  2021: 20000,
                        2022: 52000, 2023: 125000, 2024: 235000},
    'West Bengal':     {2019: 4500,  2020: 8200,  2021: 22000,
                        2022: 58000, 2023: 140000, 2024: 265000},
    'Karnataka':       {2019: 7800,  2020: 13500, 2021: 38000,
                        2022: 90000, 2023: 200000, 2024: 385000},
}

STATION_STATE_MAP: dict[str, str] = {
    'Jodhpur':   'Rajasthan',
    'Ahmedabad': 'Gujarat',
    'Mumbai':    'Maharashtra',
    'Chennai':   'Tamil Nadu',
    'Hyderabad': 'Andhra Pradesh',  # proxy — Telangana not in VAHAN
    'Bhopal':    'Madhya Pradesh',
    'Kolkata':   'West Bengal',
    'Bengaluru': 'Karnataka',
}

# State population 2024 (millions) — Census 2011 base + growth projection
STATE_POPULATIONS_2024: dict[str, float] = {
    'Rajasthan':       87.0,
    'Gujarat':         72.0,
    'Maharashtra':     128.0,
    'Tamil Nadu':      81.0,
    'Andhra Pradesh':  54.0,   # Telangana proxy — see caveat above
    'Madhya Pradesh':  90.0,
    'West Bengal':     102.0,
    'Karnataka':       70.0,
}

AVG_HOUSEHOLD_SIZE = 4.4   # Census 2011 / NFHS-5 average

STATIONS_LIST: list[str] = [
    'Jodhpur', 'Ahmedabad', 'Mumbai', 'Chennai',
    'Hyderabad', 'Bhopal', 'Kolkata', 'Bengaluru',
]

PROJECTION_YEARS = list(range(2024, 2031))   # 2024-2030 inclusive, EV30@30 horizon

# ── EV demand assumptions ──────────────────────────────────────────────────────
EV_CHARGER_KW              = 7.2    # Level-2 AC home charger (IEC 62196 Mode-3)
AVG_DAILY_CHARGE_HRS       = 3.0    # avg plugged-in hours/day/EV (NITI Aayog est.)
EV_DAILY_ENERGY_KWH        = EV_CHARGER_KW * AVG_DAILY_CHARGE_HRS   # 21.6 kWh/day
HOUSEHOLDS_PER_TRANSFORMER = 25     # matches Script 14 cluster definition
ASSUMED_SOLAR_WIND_CAPACITY_KW = 50 # matches Script 14 — keep consistent
HOURS_PER_DAY = 24

# ── Upgrade unit specs ─────────────────────────────────────────────────────────
# Adani 440W Monocrystalline — commonly deployed in Indian government
# rooftop solar schemes (PM Surya Ghar / PM-KUSUM).
PANEL_RATED_KW      = 0.440
PEAK_SUN_HOURS      = 5.0           # India residential average
PANEL_DAILY_KWH     = PANEL_RATED_KW * PEAK_SUN_HOURS   # = 2.2 kWh/panel/day

# Small community wind turbine, IEC Class III (matches RATED=7 m/s used
# throughout this pipeline — see compute_rgi.py).
TURBINE_RATED_KW    = 50.0
CF_WIND             = 0.25          # capacity factor, India Class III sites
TURBINE_DAILY_KWH   = TURBINE_RATED_KW * CF_WIND * HOURS_PER_DAY   # = 300 kWh/day


# ── VAHAN data fetch ────────────────────────────────────────────────────────────

def fetch_vahan_live() -> dict[str, dict[int, int]] | None:
    """Attempt to fetch live VAHAN EV registration data from data.gov.in.

    Returns:
        Nested dict {state: {year: count}} on success, or None on any
        failure (network error, unexpected schema, empty response) —
        caller falls back to the hardcoded VAHAN_EV_REGISTRATIONS table.
    """
    try:
        resp = requests.get(VAHAN_API_URL, timeout=15)
        resp.raise_for_status()
        payload = resp.json()
        records = payload.get('records', [])
        if not records:
            return None

        parsed: dict[str, dict[int, int]] = {}
        for rec in records:
            state = rec.get('state') or rec.get('State')
            year  = rec.get('year')  or rec.get('Year')
            count = rec.get('total') or rec.get('Total') or rec.get('count')
            if state and year and count:
                try:
                    y = int(str(year)[:4])
                    c = int(str(count).replace(',', ''))
                    parsed.setdefault(state, {})[y] = c
                except (ValueError, TypeError):
                    continue

        if not parsed:
            return None
        return parsed

    except Exception as exc:  # noqa: BLE001
        print(f"  Live VAHAN API fetch failed ({exc}) — using hardcoded fallback.")
        return None


def get_ev_registrations() -> tuple[dict[str, dict[int, int]], str]:
    """Get state-wise EV registration data, preferring live VAHAN API.

    Returns:
        Tuple of (registrations dict, source label string).
    """
    live = fetch_vahan_live()
    if live is not None:
        # Only accept live data if it covers all 8 needed states with
        # at least 2019 and 2024 — otherwise fall back for consistency.
        needed_states = set(STATION_STATE_MAP.values())
        if needed_states.issubset(live.keys()) and all(
            2019 in live[s] and 2024 in live[s] for s in needed_states
        ):
            return live, "VAHAN live API (data.gov.in)"
    return VAHAN_EV_REGISTRATIONS, "VAHAN hardcoded fallback (PIB/e-Amrit official releases)"


# ── Logistic (S-curve) projection ─────────────────────────────────────────────
# FIX: Raw CAGR compounding from a near-zero 2019 base produces physically
# impossible ownership rates (500%+ by 2030) because early-adopter growth
# rates are not sustainable over a decade. Technology adoption is standard
# literature textbook logistic (S-curve): fast early growth, saturating
# toward a ceiling as the market matures. This is the same functional form
# used in Bass diffusion models and most EV-adoption forecasting studies.
#
# Saturation ceiling: NITI Aayog's EV30@30 targets 30% EV share of NEW
# vehicle SALES by 2030 — not household ownership STOCK. For a household
# ownership-STOCK ceiling (what this model needs), we extend that same 30%
# figure as a long-run aspirational saturation level, consistent with how
# EV30@30 is commonly cited in Indian EV-infrastructure planning literature
# (cf. Kotla et al. 2026, Sci Rep, ref. 30). This is a modeling assumption,
# stated explicitly here and disclosed as a paper limitation.
SATURATION_CEILING = 0.30   # 30% of households — long-run EV ownership ceiling


def compute_logistic_rate(x0: float, x1: float, t0: int, t1: int,
                          ceiling: float = SATURATION_CEILING) -> float:
    """Calibrate the logistic growth-rate parameter r from two data points.

    Solves for r in the logistic function
        x(t) = K / (1 + ((K - x0) / x0) * exp(-r * (t - t0)))
    given known ownership-rate fractions x0 at t0 and x1 at t1.

    Args:
        x0:      EV ownership rate (fraction of households) at t0.
        x1:      EV ownership rate (fraction of households) at t1.
        t0:      Base year (e.g. 2019).
        t1:      Anchor year (e.g. 2024).
        ceiling: Saturation ceiling K (fraction of households).

    Returns:
        Calibrated logistic growth-rate parameter r.
    """
    x0 = max(x0, 1e-6)
    x1 = max(min(x1, ceiling - 1e-6), 1e-6)
    ratio = ((ceiling - x1) / x1) / ((ceiling - x0) / x0)
    ratio = max(ratio, 1e-9)
    return -np.log(ratio) / (t1 - t0)


def project_ev_ownership_logistic(x0: float, r: float, t0: int, year: int,
                                  ceiling: float = SATURATION_CEILING) -> float:
    """Project EV ownership rate for a future year using the calibrated logistic curve.

    Args:
        x0:      EV ownership rate (fraction of households) at t0.
        r:       Calibrated logistic growth-rate parameter.
        t0:      Base year (e.g. 2019).
        year:    Target year (>= t0).
        ceiling: Saturation ceiling K (fraction of households).

    Returns:
        Projected EV ownership rate (fraction of households), bounded in
        (0, ceiling).
    """
    x0 = max(x0, 1e-6)
    a = (ceiling - x0) / x0
    return ceiling / (1 + a * np.exp(-r * (year - t0)))


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_adequacy_summary(df: pd.DataFrame) -> None:
    """Save 2x4 subplot grid: RGI supply vs EV RGI demand, 2024-2030, per station.

    Args:
        df: Full ev_adequacy_forecast DataFrame (all stations, all years).
    """
    fig, axes = plt.subplots(2, 4, figsize=(22, 10), sharey=True)

    for ax, station in zip(axes.flat, STATIONS_LIST):
        sub = df[df['station'] == station].sort_values('year')
        state = STATION_STATE_MAP[station]

        ax.plot(sub['year'], sub['rgi_supply'], color='steelblue',
                linewidth=2, marker='o', markersize=4, label='RGI supply (current)')
        ax.plot(sub['year'], sub['ev_rgi_demand'], color='crimson',
                linewidth=2, marker='s', markersize=4, linestyle='--',
                label='EV RGI demand (projected)')

        ax.fill_between(sub['year'], sub['rgi_supply'], sub['ev_rgi_demand'],
                        where=(sub['rgi_supply'] >= sub['ev_rgi_demand']),
                        color='seagreen', alpha=0.25, interpolate=True)
        ax.fill_between(sub['year'], sub['rgi_supply'], sub['ev_rgi_demand'],
                        where=(sub['rgi_supply'] < sub['ev_rgi_demand']),
                        color='firebrick', alpha=0.25, interpolate=True)

        yellow_rows = sub[sub['verdict'] == 'YELLOW — Future Deficit']
        if not yellow_rows.empty:
            tip_year = int(yellow_rows['year'].min())
            ax.axvline(tip_year, color='goldenrod', linestyle=':', linewidth=1.8)
            ax.text(tip_year, ax.get_ylim()[1] * 0.95, f' {tip_year}',
                   fontsize=8, color='goldenrod', fontweight='bold')

        ax.set_title(f'{station} ({state})', fontsize=10, fontweight='bold')
        ax.set_xlabel('Year')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(PROJECTION_YEARS)
        ax.tick_params(axis='x', rotation=45, labelsize=8)

    axes[0, 0].set_ylabel('RGI fraction')
    axes[1, 0].set_ylabel('RGI fraction')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        'EV-RGI Adequacy Forecast 2024–2030: Current Renewable Supply vs '
        'Projected EV Charging Demand',
        fontsize=13, y=1.06,
    )
    plt.tight_layout()
    plt.savefig(ADEQUACY_PNG, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {ADEQUACY_PNG}")


def plot_upgrade_requirements(df_2030: pd.DataFrame) -> None:
    """Save horizontal bar chart: panels needed by 2030, per station.

    Args:
        df_2030: ev_adequacy_forecast rows filtered to year == 2030.
    """
    df_sorted = df_2030.sort_values('panels_needed_adani_440w', ascending=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['firebrick' if p > 0 else 'seagreen'
              for p in df_sorted['panels_needed_adani_440w']]

    bars = ax.barh(df_sorted['station'], df_sorted['panels_needed_adani_440w'],
                   color=colors, edgecolor='white', linewidth=0.8, alpha=0.9)

    for bar, (_, row) in zip(bars, df_sorted.iterrows()):
        w = bar.get_width()
        turbines = row['turbines_needed_50kw']
        label = (f"  {int(w)} panels (≈{int(turbines)} turbines equiv.)"
                 if w > 0 else "  Adequate — no upgrade needed")
        ax.text(w, bar.get_y() + bar.get_height() / 2, label,
               va='center', ha='left', fontsize=9)

    ax.set_xlabel('Additional Adani 440W panels needed by 2030')
    ax.set_title(
        'Renewable Capacity Upgrade Requirements by 2030\n'
        '(to meet projected EV charging demand — Adani 440W panel equivalent, '
        '50kW community turbine equivalent shown alongside)',
        fontsize=11,
    )
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(UPGRADE_PNG, dpi=150)
    plt.close(fig)
    print(f"  Saved → {UPGRADE_PNG}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """Script 15 — forward-looking EV-RGI adequacy assessment, 2024-2030."""
    ADEQUACY_CSV.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("ev_adequacy_forecast.py — Script 15: EV-RGI Adequacy Forecast 2024-2030")
    print(f"{'='*70}\n")

    # ── Step 1: Validate inputs ────────────────────────────────────────────────
    print("Step 1 — Validating inputs ...")
    if not FORECASTS_V2_CSV.exists():
        print(f"ERROR: generation_forecasts_v2.csv not found: {FORECASTS_V2_CSV}")
        print("Run scripts/forecast_generation_v2.py first.")
        sys.exit(1)

    # ── Step 2: Load RGI supply ────────────────────────────────────────────────
    print("\nStep 2 — Loading current RGI supply per station ...")
    df_fc = pd.read_csv(FORECASTS_V2_CSV, parse_dates=['date'])
    rgi_supply_by_station = (
        df_fc.groupby('station')['forecasted_combined_rgi'].mean().to_dict()
    )
    for s in STATIONS_LIST:
        print(f"  {s:<12} mean RGI supply = {rgi_supply_by_station.get(s, float('nan')):.4f}")

    # ── Step 3: Get EV registration data (live or fallback) ───────────────────
    print("\nStep 3 — Fetching state-wise EV registration data ...")
    ev_data, source_label = get_ev_registrations()
    print(f"  Data source: {source_label}")
    print("  NOTE: Hyderabad uses Andhra Pradesh VAHAN data as a proxy — "
          "Telangana does not report to the VAHAN portal.")

    # ── Step 4: Calibrate logistic growth per state, project ownership 2024-2030 ──
    print("\nStep 4 — Calibrating logistic (S-curve) growth and projecting "
          "EV ownership 2024-2030 ...")
    print(f"  Saturation ceiling K = {SATURATION_CEILING*100:.0f}% of households "
          f"(EV30@30-aligned long-run assumption)")
    r_by_state: dict[str, float] = {}
    projected_ownership: dict[str, dict[int, float]] = {}
    projected_ev: dict[str, dict[int, int]] = {}

    for station in STATIONS_LIST:
        state = STATION_STATE_MAP[station]
        state_pop_millions = STATE_POPULATIONS_2024[state]
        state_households = (state_pop_millions * 1_000_000) / AVG_HOUSEHOLD_SIZE

        ev_hist = ev_data.get(state, {})
        ev_2019 = ev_hist.get(2019, 0)
        ev_2024 = ev_hist.get(2024, 0)
        x0 = ev_2019 / state_households
        x1 = ev_2024 / state_households

        r = compute_logistic_rate(x0, x1, t0=2019, t1=2024)
        r_by_state[state] = r

        projected_ownership[state] = {}
        projected_ev[state] = {}
        for year in PROJECTION_YEARS:
            if year in ev_hist:
                rate = ev_hist[year] / state_households
            else:
                rate = project_ev_ownership_logistic(x0, r, t0=2019, year=year)
            projected_ownership[state][year] = rate
            projected_ev[state][year] = int(round(rate * state_households))

        print(f"  {state:<18} r={r:>6.3f}/yr   "
              f"2024={projected_ownership[state][2024]*100:>5.2f}%   "
              f"2030={projected_ownership[state][2030]*100:>5.2f}%   "
              f"(EV count 2030 ≈ {projected_ev[state][2030]:,})")

    # ── Step 5: Compute EV RGI demand per station per year ─────────────────────
    print("\nStep 5 — Computing EV RGI demand per station, per year ...")
    records: list[dict] = []

    for station in STATIONS_LIST:
        state = STATION_STATE_MAP[station]
        rgi_supply = rgi_supply_by_station.get(station, float('nan'))

        for year in PROJECTION_YEARS:
            ev_count = projected_ev[state][year]
            ev_ownership_rate = projected_ownership[state][year]
            evs_per_cluster = ev_ownership_rate * HOUSEHOLDS_PER_TRANSFORMER
            ev_daily_energy_kwh = evs_per_cluster * EV_DAILY_ENERGY_KWH
            ev_rgi_demand = ev_daily_energy_kwh / (
                ASSUMED_SOLAR_WIND_CAPACITY_KW * HOURS_PER_DAY
            )

            surplus_rgi = rgi_supply - ev_rgi_demand

            if surplus_rgi >= 0:
                verdict = 'GREEN — Adequate'
                panels_needed = 0
                turbines_needed = 0
            else:
                deficit_kwh = abs(surplus_rgi) * ASSUMED_SOLAR_WIND_CAPACITY_KW * HOURS_PER_DAY
                panels_needed = math.ceil(deficit_kwh / PANEL_DAILY_KWH)
                turbines_needed = math.ceil(deficit_kwh / TURBINE_DAILY_KWH)
                verdict = ('RED — Already Inadequate' if year == 2024
                          else 'YELLOW — Future Deficit')

            records.append({
                'year': year,
                'station': station,
                'state': state,
                'ev_count_actual_or_projected': ev_count,
                'ev_ownership_rate': ev_ownership_rate,
                'evs_per_cluster': evs_per_cluster,
                'ev_rgi_demand': ev_rgi_demand,
                'rgi_supply': rgi_supply,
                'surplus_rgi': surplus_rgi,
                'verdict': verdict,
                'panels_needed_adani_440w': panels_needed,
                'turbines_needed_50kw': turbines_needed,
            })

    df_out = pd.DataFrame(records)
    df_out.to_csv(ADEQUACY_CSV, index=False)
    print(f"\n  Saved → {ADEQUACY_CSV}  ({len(df_out)} rows)")

    # ── Step 6: Determine tipping year per station ──────────────────────────────
    print("\nStep 6 — Determining tipping year per station ...")
    tipping_years: dict[str, int | None] = {}
    for station in STATIONS_LIST:
        sub = df_out[df_out['station'] == station].sort_values('year')
        deficit_rows = sub[sub['surplus_rgi'] < 0]
        tipping_years[station] = int(deficit_rows['year'].min()) if not deficit_rows.empty else None

    # ── Step 7: Plots ────────────────────────────────────────────────────────────
    print("\nStep 7 — Generating plots ...")
    plot_adequacy_summary(df_out)
    df_2030 = df_out[df_out['year'] == 2030].copy()
    plot_upgrade_requirements(df_2030)

    # ── Step 8: Console summary table ───────────────────────────────────────────
    print(f"\n{'='*90}")
    print("SUMMARY — EV-RGI Adequacy Forecast per Station")
    print(f"{'='*90}")
    hdr = (f"  {'Station':<12} {'2024 Verdict':<26} {'Tipping Year':>13} "
           f"{'Panels 2030':>12} {'Turbines 2030':>14}")
    print(hdr)
    print('  ' + '-' * (len(hdr) - 2))
    for station in STATIONS_LIST:
        row_2024 = df_out[(df_out['station'] == station) & (df_out['year'] == 2024)].iloc[0]
        row_2030 = df_2030[df_2030['station'] == station].iloc[0]
        tip = tipping_years[station]
        tip_str = str(tip) if tip else 'Never (2024-2030)'
        print(
            f"  {station:<12} {row_2024['verdict']:<26} {tip_str:>13} "
            f"{int(row_2030['panels_needed_adani_440w']):>12} "
            f"{int(row_2030['turbines_needed_50kw']):>14}"
        )

    n_green_2024  = int((df_out[df_out['year'] == 2024]['verdict'] == 'GREEN — Adequate').sum())
    n_red_2024    = int((df_out[df_out['year'] == 2024]['verdict'] == 'RED — Already Inadequate').sum())
    n_yellow_ever = sum(1 for s in STATIONS_LIST if tipping_years[s] is not None
                        and tipping_years[s] > 2024)
    n_green_all   = sum(1 for s in STATIONS_LIST if tipping_years[s] is None)

    print(f"\n  {n_green_all}/8 stations remain GREEN through 2030")
    if n_yellow_ever:
        earliest = min(tipping_years[s] for s in STATIONS_LIST
                       if tipping_years[s] is not None and tipping_years[s] > 2024)
        print(f"  {n_yellow_ever}/8 stations turn YELLOW by {earliest}")
    else:
        print(f"  0/8 stations turn YELLOW during 2025-2030")
    print(f"  {n_red_2024}/8 stations are RED already in 2024")

    # ── SELF CHECK ────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SELF CHECK")
    print(f"{'='*70}")

    print("\nEV ownership rate 2024 vs 2030 per station:")
    for station in STATIONS_LIST:
        r24 = df_out[(df_out['station'] == station) & (df_out['year'] == 2024)]['ev_ownership_rate'].values[0]
        r30 = df_out[(df_out['station'] == station) & (df_out['year'] == 2030)]['ev_ownership_rate'].values[0]
        print(f"  {station:<12} 2024={r24*100:.3f}%   2030={r30*100:.3f}%   "
              f"monotonic increase: {r30 >= r24}")

    demand_monotonic = True
    surplus_monotonic = True
    for station in STATIONS_LIST:
        sub = df_out[df_out['station'] == station].sort_values('year')
        d = sub['ev_rgi_demand'].to_numpy()
        s = sub['surplus_rgi'].to_numpy()
        if not np.all(np.diff(d) >= -1e-9):
            demand_monotonic = False
        if not np.all(np.diff(s) <= 1e-9):
            surplus_monotonic = False

    print(f"\nev_rgi_demand increases monotonically 2024→2030 (all stations): {demand_monotonic}")
    print(f"surplus_rgi decreases monotonically 2024→2030 (all stations)  : {surplus_monotonic}")

    panels_ok = bool((df_out['panels_needed_adani_440w'] >= 0).all())
    print(f"panels_needed >= 0 always: {panels_ok}")
    assert panels_ok, "Negative panel count detected!"

    valid_verdicts = {'GREEN — Adequate', 'YELLOW — Future Deficit', 'RED — Already Inadequate'}
    verdicts_ok = bool(set(df_out['verdict'].unique()).issubset(valid_verdicts))
    print(f"All verdicts are GREEN/YELLOW/RED only: {verdicts_ok}")
    assert verdicts_ok, "Unexpected verdict value found!"

    print(f"\n{'='*70}")
    print("Script 15 complete — EV adequacy forecast.")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
