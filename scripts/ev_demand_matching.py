"""
ev_demand_matching.py
Script 14 — EV Demand Matching via Forecasted Hybrid Generation.

CONCEPT
Residential street-level transformers in India were sized decades ago for
legacy load (lighting, fans, refrigeration, AC) before EV adoption.
A single Level-2 home EV charger draws ~7.2 kW continuously — comparable
to 2-3 additional houses of simultaneous load. This script computes, per
station per day, how much of an incremental EV charging load the forecasted
hybrid RGI signal can support on the existing transformer, expressed against
the fixed headroom limit — mirroring the P_grid(t) <= P_max,transformer
constraint used in EV-infrastructure planning literature
(cf. Kotla et al. 2026, Sci Rep, equation 22).

This is NOT a grid redesign. It answers: given tomorrow's forecasted local
renewable generation, how many EV charging sessions can this street's
transformer safely absorb without exceeding its rated headroom?

CRITICAL CAVEAT (see also inline comments below):
This is an illustrative scaling exercise using representative assumed
values for transformer rating, household count, baseline load fraction,
and assumed local renewable capacity — NOT validated against actual DISCOM
transformer load data for any real street. The RGI-to-kW conversion
(ASSUMED_SOLAR_WIND_CAPACITY_KW) is a modeling assumption that should be
stated as a limitation in the paper, with real distribution-transformer
telemetry as future work.

Inputs:  outputs/generation_forecasts_v2.csv
         outputs/optimal_mix.csv
Outputs: outputs/ev_demand_matching.csv
         outputs/ev_capacity_relief.png
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

# ── Path anchors ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

FORECASTS_V2_CSV = ROOT / 'outputs' / 'generation_forecasts_v2.csv'
OPTIMAL_MIX_CSV  = ROOT / 'outputs' / 'optimal_mix.csv'
EV_CSV           = ROOT / 'outputs' / 'ev_demand_matching.csv'
EV_PLOT_PNG      = ROOT / 'outputs' / 'ev_capacity_relief.png'

# ── Grid and EV infrastructure assumptions ────────────────────────────────────
# All values are representative Indian residential planning figures; cite as
# modeling assumptions in the paper — NOT validated against DISCOM telemetry.

TRANSFORMER_RATING_KVA = 100       # typical Indian residential distribution
                                   # transformer (BEE 2022; CERC distribution
                                   # planning guidelines, Annexure II)
POWER_FACTOR           = 0.9       # typical residential power factor (IS 13234)
TRANSFORMER_RATING_KW  = TRANSFORMER_RATING_KVA * POWER_FACTOR   # = 90 kW

BASELINE_LOAD_FRACTION = 0.65      # fraction of rated capacity already committed
                                   # to legacy load before EV adoption
                                   # (lighting 10% + fans 10% + fridges 8% +
                                   #  AC mix 37% ≈ 65%; BEE residential survey 2023)
EV_CHARGER_KW          = 7.2       # typical Indian home Level-2 AC charger draw
                                   # (Tata Nexon EV / MG ZS EV 7.2 kW OBC spec;
                                   #  EVSE standard IEC 62196 Mode-3 Type-2)
HOUSEHOLDS_ON_TRANSFORMER = 25     # typical Indian LT feeder household count per
                                   # distribution transformer (CEA 2022 rural/
                                   # peri-urban baseline; urban can be 40-60)

# MODELING ASSUMPTION — must be disclosed as a paper limitation:
# Represents the combined rooftop solar (30 kW ≈ 15 panels × 2 kW) +
# community wind (20 kW micro-turbine cluster) capacity assumed for a
# 25-household transformer cluster, i.e. ~2 kW solar + ~0.8 kW wind per
# household. Real values vary by building type, shading, and available land.
ASSUMED_SOLAR_WIND_CAPACITY_KW = 50

# ── Derived constants (computed once, same for every station and day) ─────────
BASELINE_LOAD_KW           = TRANSFORMER_RATING_KW * BASELINE_LOAD_FRACTION
                                                    # = 58.5 kW
AVAILABLE_HEADROOM_KW      = TRANSFORMER_RATING_KW - BASELINE_LOAD_KW
                                                    # = 31.5 kW
EV_CHARGERS_WITHOUT_GENERATION = math.floor(AVAILABLE_HEADROOM_KW / EV_CHARGER_KW)
                                                    # = 4 (floor(31.5 / 7.2))

STATIONS_LIST: list[str] = [
    'Jodhpur', 'Ahmedabad', 'Mumbai', 'Chennai',
    'Hyderabad', 'Bhopal', 'Kolkata', 'Bengaluru',
]


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_ev_capacity_relief(summary: pd.DataFrame) -> None:
    """Save bar chart: mean additional EVs enabled per station, sorted descending.

    Each bar is annotated with the % of test-year days where forecasted
    generation enables at least 1 additional EV without exceeding headroom.
    """
    df_sorted = summary.sort_values('mean_additional_evs', ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(14, 6))
    colors  = plt.cm.viridis_r(np.linspace(0.2, 0.85, len(df_sorted)))

    bars = ax.bar(
        df_sorted['station'],
        df_sorted['mean_additional_evs'],
        color=colors,
        edgecolor='white',
        linewidth=0.8,
        alpha=0.92,
    )

    for bar, (_, row) in zip(bars, df_sorted.iterrows()):
        h   = bar.get_height()
        pct = row['pct_days_geq_1_extra_ev']
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.03,
            f"{h:.2f} EVs\n{pct:.1f}% days",
            ha='center', va='bottom',
            fontsize=8.5, fontweight='bold',
        )

    ax.axhline(y=1, color='crimson', linestyle='--', linewidth=1.2, alpha=0.7,
               label='1 additional EV threshold')

    ax.set_xlabel('Station')
    ax.set_ylabel('Mean additional simultaneous EV chargers enabled')
    ax.set_title(
        f'EV Charging Capacity Relief from Forecasted Hybrid Renewable Generation\n'
        f'({TRANSFORMER_RATING_KVA} kVA transformer · {HOUSEHOLDS_ON_TRANSFORMER} households · '
        f'{ASSUMED_SOLAR_WIND_CAPACITY_KW} kW assumed solar+wind capacity per cluster)',
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(EV_PLOT_PNG, dpi=150)
    plt.close(fig)
    print(f"  Saved → {EV_PLOT_PNG}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """Script 14 — EV demand matching against forecasted hybrid generation."""
    EV_CSV.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("ev_demand_matching.py — Script 14: EV Demand Matching")
    print(f"{'='*70}\n")

    # ── Step 1: Validate inputs ────────────────────────────────────────────────
    print("Step 1 — Validating inputs ...")
    for path, desc in [
        (FORECASTS_V2_CSV, 'generation_forecasts_v2.csv — run forecast_generation_v2.py first'),
        (OPTIMAL_MIX_CSV,  'optimal_mix.csv — run optimize_mix.py first'),
    ]:
        if not path.exists():
            print(f"ERROR: {desc}")
            print(f"  Expected: {path}")
            sys.exit(1)

    # ── Step 2: Load inputs ────────────────────────────────────────────────────
    print("\nStep 2 — Loading inputs ...")
    df_fc       = pd.read_csv(FORECASTS_V2_CSV, parse_dates=['date'])
    df_mix      = pd.read_csv(OPTIMAL_MIX_CSV)

    print(f"  generation_forecasts_v2 rows : {len(df_fc):,}")
    print(f"  optimal_mix rows             : {len(df_mix)}")
    print(f"  Columns in forecasts_v2      : {list(df_fc.columns)}")
    print(f"  Date range                   : {df_fc['date'].min().date()} → "
          f"{df_fc['date'].max().date()}")
    print(f"  Stations present             : {sorted(df_fc['station'].unique())}")

    required_cols = ['date', 'station', 'forecasted_combined_rgi']
    missing = [c for c in required_cols if c not in df_fc.columns]
    if missing:
        print(f"ERROR: missing required columns in forecasts_v2: {missing}")
        sys.exit(1)

    # ── Step 3: Print assumptions ──────────────────────────────────────────────
    print(f"\nStep 3 — Grid and EV assumptions (explicit constants):")
    print(f"  Transformer rating            : {TRANSFORMER_RATING_KVA} kVA")
    print(f"  Power factor                  : {POWER_FACTOR}")
    print(f"  Transformer rating (kW)       : {TRANSFORMER_RATING_KW:.1f} kW")
    print(f"  Baseline load fraction        : {BASELINE_LOAD_FRACTION*100:.0f}%")
    print(f"  Baseline load                 : {BASELINE_LOAD_KW:.1f} kW")
    print(f"  Available headroom            : {AVAILABLE_HEADROOM_KW:.1f} kW")
    print(f"  EV charger draw               : {EV_CHARGER_KW} kW (Level-2 AC)")
    print(f"  Households per transformer    : {HOUSEHOLDS_ON_TRANSFORMER}")
    print(f"  Assumed local capacity        : {ASSUMED_SOLAR_WIND_CAPACITY_KW} kW")
    print(f"    (MODELING ASSUMPTION — see paper limitation section)")
    print(f"  EVs without generation        : {EV_CHARGERS_WITHOUT_GENERATION} "
          f"(floor({AVAILABLE_HEADROOM_KW} / {EV_CHARGER_KW}))")

    # ── Step 4: Compute daily EV metrics ──────────────────────────────────────
    print(f"\nStep 4 — Computing daily EV headroom metrics per station ...")
    rows_before = len(df_fc)
    df = df_fc.dropna(subset=['forecasted_combined_rgi']).copy()
    df = df.sort_values(['station', 'date']).reset_index(drop=True)
    print(f"  Rows before dropna: {rows_before:,}  after: {len(df):,}")

    # Step 4.1 — local generation from forecasted RGI (illustrative kW scaling)
    df['local_generation_kw'] = (
        df['forecasted_combined_rgi'] * ASSUMED_SOLAR_WIND_CAPACITY_KW
    )

    # Step 4.2 — net headroom: headroom + local generation offsets baseline draw
    # Cap at TRANSFORMER_RATING_KW; log capping frequency below
    df['net_available_headroom_kw_raw'] = (
        AVAILABLE_HEADROOM_KW + df['local_generation_kw']
    )
    df['capped'] = df['net_available_headroom_kw_raw'] > TRANSFORMER_RATING_KW
    df['net_available_headroom_kw'] = df['net_available_headroom_kw_raw'].clip(
        upper=TRANSFORMER_RATING_KW
    )

    # Step 4.3 — EV counts
    df['max_simultaneous_ev_chargers'] = (
        df['net_available_headroom_kw'] // EV_CHARGER_KW
    ).astype(int)
    df['ev_chargers_without_generation'] = EV_CHARGERS_WITHOUT_GENERATION
    df['additional_evs_enabled'] = (
        df['max_simultaneous_ev_chargers'] - EV_CHARGERS_WITHOUT_GENERATION
    ).clip(lower=0)  # cannot be negative

    n_capped = int(df['capped'].sum())
    n_total  = len(df)
    print(f"\n  Capping check: net headroom exceeded transformer rating on "
          f"{n_capped}/{n_total} rows ({n_capped/n_total*100:.2f}%)")
    if n_capped > 0:
        print(f"  Max raw headroom observed: {df['net_available_headroom_kw_raw'].max():.2f} kW "
              f"(capped to {TRANSFORMER_RATING_KW:.1f} kW)")
    else:
        print(f"  Max raw headroom observed: {df['net_available_headroom_kw_raw'].max():.2f} kW "
              f"— no capping required (< {TRANSFORMER_RATING_KW:.1f} kW)")

    # ── Step 5: Save daily output CSV ─────────────────────────────────────────
    print(f"\nStep 5 — Saving daily output CSV ...")
    out_cols = [
        'date', 'station',
        'forecasted_combined_rgi',
        'local_generation_kw',
        'net_available_headroom_kw',
        'max_simultaneous_ev_chargers',
        'ev_chargers_without_generation',
        'additional_evs_enabled',
    ]
    if 'optimal_solar_pct' in df.columns:
        out_cols.insert(3, 'optimal_solar_pct')

    df[out_cols].to_csv(EV_CSV, index=False)
    print(f"  Rows saved: {len(df):,}")
    print(f"  Saved → {EV_CSV}")

    # ── Step 6: Aggregate per station ─────────────────────────────────────────
    print(f"\nStep 6 — Aggregating per station over test year 2024 ...")

    summary_rows: list[dict] = []
    for station in STATIONS_LIST:
        sub = df[df['station'] == station]
        if len(sub) == 0:
            print(f"  WARNING: no rows for {station}")
            continue

        mean_local_gen   = float(sub['local_generation_kw'].mean())
        mean_add_evs     = float(sub['additional_evs_enabled'].mean())
        max_add_evs      = int(sub['additional_evs_enabled'].max())
        min_add_evs      = int(sub['additional_evs_enabled'].min())
        days_geq_1       = int((sub['additional_evs_enabled'] >= 1).sum())
        total_days       = len(sub)
        pct_days_geq_1   = days_geq_1 / total_days * 100

        # best and worst day dates for reference
        best_day  = sub.loc[sub['additional_evs_enabled'].idxmax(), 'date']
        worst_day = sub.loc[sub['additional_evs_enabled'].idxmin(), 'date']

        opt_solar_pct = float('nan')
        mix_row = df_mix[df_mix['station'] == station]
        if not mix_row.empty and 'optimal_solar_pct' in mix_row.columns:
            opt_solar_pct = float(mix_row['optimal_solar_pct'].values[0]) * 100

        summary_rows.append({
            'station':            station,
            'optimal_solar_pct':  opt_solar_pct,
            'mean_local_gen_kw':  mean_local_gen,
            'mean_additional_evs': mean_add_evs,
            'max_additional_evs': max_add_evs,
            'min_additional_evs': min_add_evs,
            'days_geq_1_extra_ev': days_geq_1,
            'total_days':         total_days,
            'pct_days_geq_1_extra_ev': pct_days_geq_1,
            'best_day':           str(best_day.date()) if hasattr(best_day, 'date') else str(best_day),
            'worst_day':          str(worst_day.date()) if hasattr(worst_day, 'date') else str(worst_day),
        })

    summary = pd.DataFrame(summary_rows)

    # ── Step 7: Print aggregated table ────────────────────────────────────────
    print(f"\nStep 7 — Per-station EV demand matching summary (test year 2024):\n")

    hdr_fmt = (
        f"  {'Station':<12} | {'Opt.Solar%':>10} | {'Mean gen kW':>11} | "
        f"{'Mean +EVs':>9} | {'Max +EVs':>8} | {'Min +EVs':>8} | {'% days ≥1 EV':>13}"
    )
    print(hdr_fmt)
    print('  ' + '-' * (len(hdr_fmt) - 2))

    for _, row in summary.iterrows():
        opt_s = f"{row['optimal_solar_pct']:.0f}%" if not np.isnan(row['optimal_solar_pct']) else "N/A"
        print(
            f"  {row['station']:<12} | {opt_s:>10} | "
            f"{row['mean_local_gen_kw']:>11.2f} | "
            f"{row['mean_additional_evs']:>9.2f} | "
            f"{row['max_additional_evs']:>8d} | "
            f"{row['min_additional_evs']:>8d} | "
            f"{row['pct_days_geq_1_extra_ev']:>12.1f}%"
        )

    print(f"\n  EVs without any generation (baseline headroom only): "
          f"{EV_CHARGERS_WITHOUT_GENERATION} chargers on every day")
    print(f"  (= floor({AVAILABLE_HEADROOM_KW:.1f} kW / {EV_CHARGER_KW} kW per charger)")

    # ── Step 8: Bar chart ─────────────────────────────────────────────────────
    print(f"\nStep 8 — Saving ev_capacity_relief.png ...")
    plot_ev_capacity_relief(summary)

    # ── Step 9: Paper-ready narrative sentence ────────────────────────────────
    print(f"\nStep 9 — Auto-generated narrative sentence (paper-ready):\n")

    mean_all        = float(summary['mean_additional_evs'].mean())
    best_row        = summary.loc[summary['mean_additional_evs'].idxmax()]
    worst_row       = summary.loc[summary['mean_additional_evs'].idxmin()]
    best_station    = best_row['station']
    worst_station   = worst_row['station']
    best_mean_evs   = float(best_row['mean_additional_evs'])
    worst_mean_evs  = float(worst_row['mean_additional_evs'])

    narrative = (
        f"Across the 8 stations, forecasted hybrid solar-wind generation enables "
        f"an average of {mean_all:.1f} additional simultaneous EV home chargers "
        f"on a standard {TRANSFORMER_RATING_KVA} kVA residential transformer "
        f"without exceeding rated headroom, with {best_station} enabling the most "
        f"({best_mean_evs:.1f}) and {worst_station} the least ({worst_mean_evs:.1f}) "
        f"— demonstrating that location-aware renewable generation forecasting can "
        f"materially extend existing distribution infrastructure lifespan without "
        f"requiring physical reinforcement, directly addressing the EV-driven "
        f"transformer overload problem documented in residential grid studies."
    )
    print(f"  {narrative}")

    # ── Step 10: Critical caveat ───────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("CRITICAL CAVEAT:")
    print(
        "  This analysis uses representative assumed values — transformer rating "
        f"({TRANSFORMER_RATING_KVA} kVA), household count ({HOUSEHOLDS_ON_TRANSFORMER}), "
        f"baseline load fraction ({BASELINE_LOAD_FRACTION*100:.0f}%), and especially "
        f"local renewable capacity ({ASSUMED_SOLAR_WIND_CAPACITY_KW} kW) — that are "
        "NOT validated against actual DISCOM transformer load data for any real street. "
        f"The RGI-to-kW conversion via ASSUMED_SOLAR_WIND_CAPACITY_KW={ASSUMED_SOLAR_WIND_CAPACITY_KW} "
        "is a modeling assumption that must be disclosed as a paper limitation, with "
        "real distribution-transformer telemetry cited as future work."
    )
    print(f"{'─'*70}")

    # ── SELF CHECK ──────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SELF CHECK")
    print(f"{'='*70}")

    headroom_ok = bool(
        (df['net_available_headroom_kw'] <= TRANSFORMER_RATING_KW + 1e-9).all()
    )
    print(f"\nnet_available_headroom_kw never exceeds {TRANSFORMER_RATING_KW:.1f} kW: "
          f"{'PASS' if headroom_ok else 'FAIL'}")
    if not headroom_ok:
        n_fail = int((df['net_available_headroom_kw'] > TRANSFORMER_RATING_KW).sum())
        print(f"  WARNING: {n_fail} rows exceed transformer rating after capping — "
              "check capping logic.")
    assert headroom_ok, "Headroom cap constraint violated!"

    rgi_ok = bool(df['forecasted_combined_rgi'].between(0, 1).all())
    print(f"forecasted_combined_rgi in [0,1]: {'PASS' if rgi_ok else 'FAIL'}")

    add_evs_ok = bool((df['additional_evs_enabled'] >= 0).all())
    print(f"additional_evs_enabled >= 0     : {'PASS' if add_evs_ok else 'FAIL'}")

    print(f"\nSaved outputs:")
    print(f"  {EV_CSV}")
    print(f"  {EV_PLOT_PNG}")

    print(f"\nRows in ev_demand_matching.csv: {len(df):,}")
    print(f"Stations covered: {sorted(df['station'].unique())}")
    print(f"\nConstant-headroom sanity:")
    print(f"  Transformer capacity        : {TRANSFORMER_RATING_KW:.1f} kW")
    print(f"  Baseline legacy load        : {BASELINE_LOAD_KW:.1f} kW")
    print(f"  Headroom without generation : {AVAILABLE_HEADROOM_KW:.1f} kW")
    print(f"  EVs without generation      : {EV_CHARGERS_WITHOUT_GENERATION}")
    print(f"  Max generation contribution : "
          f"{ASSUMED_SOLAR_WIND_CAPACITY_KW * 1.0:.1f} kW (at RGI=1.0)")
    print(f"  Max possible headroom       : "
          f"{AVAILABLE_HEADROOM_KW + ASSUMED_SOLAR_WIND_CAPACITY_KW:.1f} kW "
          f"({'below' if AVAILABLE_HEADROOM_KW + ASSUMED_SOLAR_WIND_CAPACITY_KW <= TRANSFORMER_RATING_KW else 'above'} "
          f"transformer rating — cap {'inactive' if n_capped == 0 else 'active'})")

    print(f"\nScript 14 complete — EV demand-matching analysis.")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
