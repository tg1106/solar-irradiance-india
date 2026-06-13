"""
optimize_mix.py
Phase B — Script 10.
Find the optimal solar:wind capacity ratio per station that
maximises energy generation while minimising variance (stability).
This is the paper's primary novel finding for Phase B.

Input:  data/processed/rgi_dataset.csv
Output: outputs/optimal_mix.csv
        outputs/mix_chart.png
        outputs/mix_sweep_curves.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Path anchors ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

RGI_CSV       = ROOT / 'data'    / 'processed' / 'rgi_dataset.csv'
OPTIMAL_CSV   = ROOT / 'outputs' / 'optimal_mix.csv'
MIX_CHART_PNG = ROOT / 'outputs' / 'mix_chart.png'
SWEEP_PNG     = ROOT / 'outputs' / 'mix_sweep_curves.png'

SOLAR_FRACS = np.round(np.arange(0.0, 1.01, 0.01), 2)

CLIMATE_ZONES: dict[str, str] = {
    'Jodhpur':   'Arid',
    'Ahmedabad': 'Semi-arid',
    'Mumbai':    'Tropical coastal W',
    'Chennai':   'Tropical coastal E',
    'Hyderabad': 'Semi-arid Deccan',
    'Bhopal':    'Semi-arid interior',
    'Kolkata':   'Humid sub-tropical',
    'Bengaluru': 'Elevated plateau',
}


# ── Optimisation — AGENTS.md find_optimal_mix, implemented exactly ──────────────

def find_optimal_mix(
    solar_rgi_series: pd.Series, wind_rgi_series: pd.Series
) -> tuple[float, float]:
    """Sweep solar fractions 0% to 100% in 1% steps.

    For each fraction compute:
      - Mean combined RGI (maximise this)
      - Std combined RGI (minimise this — stability)
      - Sharpe-like ratio: mean / std (maximise this)
    Return the fraction maximising mean/std ratio.

    Args:
        solar_rgi_series: Solar RGI values [0, 1].
        wind_rgi_series:  Wind RGI values [0, 1].

    Returns:
        Tuple of (optimal_solar_pct, optimal_wind_pct).
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


def sweep_mix(solar_series: pd.Series, wind_series: pd.Series) -> pd.DataFrame:
    """Sweep solar_frac 0..1 step 0.01 and compute mean/std/ratio of combined RGI.

    Args:
        solar_series: Solar RGI values [0, 1].
        wind_series:  Wind RGI values [0, 1].

    Returns:
        DataFrame indexed by solar_frac with columns mean, std, ratio.
    """
    rows = []
    for solar_frac in SOLAR_FRACS:
        wind_frac = 1 - solar_frac
        combined = solar_frac * solar_series + wind_frac * wind_series
        mean_gen = combined.mean()
        std_gen  = combined.std()
        ratio = mean_gen / std_gen if std_gen > 0 else mean_gen
        rows.append({'solar_frac': solar_frac, 'mean': mean_gen, 'std': std_gen, 'ratio': ratio})
    return pd.DataFrame(rows).set_index('solar_frac')


def best_frac(sweep: pd.DataFrame, column: str, maximise: bool) -> float:
    """Return the solar_frac that maximises (or minimises) a sweep column.

    Args:
        sweep: Output of sweep_mix().
        column: Column name to optimise ('mean', 'std', or 'ratio').
        maximise: If True, return the argmax; otherwise the argmin.

    Returns:
        The optimal solar_frac.
    """
    return float(sweep[column].idxmax() if maximise else sweep[column].idxmin())


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_mix_chart(df: pd.DataFrame) -> None:
    """Save grouped bar chart of optimal solar% vs wind% per station.

    Args:
        df: optimal_mix DataFrame, sorted by optimal_solar_pct descending.
    """
    stations = df['station'].tolist()
    x = np.arange(len(stations))
    w = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - w/2, df['optimal_solar_pct'] * 100, w, label='Solar %', color='orange')
    ax.bar(x + w/2, df['optimal_wind_pct'] * 100,  w, label='Wind %',  color='steelblue')
    ax.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    ax.set_xlabel('Station')
    ax.set_ylabel('Capacity fraction (%)')
    ax.set_title('Optimal Solar-Wind Capacity Mix by Station')
    ax.set_xticks(x)
    ax.set_xticklabels(stations, rotation=30, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(MIX_CHART_PNG, dpi=150)
    plt.close(fig)
    print(f"  Saved → {MIX_CHART_PNG}")


def plot_sweep_curves(sweeps: dict[str, pd.DataFrame], optimal: dict[str, float]) -> None:
    """Save mean/std ratio vs solar fraction for all stations.

    Args:
        sweeps:  Per-station sweep DataFrames (output of sweep_mix).
        optimal: Per-station optimal solar fraction.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    for station, sweep in sweeps.items():
        line, = ax.plot(sweep.index, sweep['ratio'], label=station, linewidth=1.5)
        opt = optimal[station]
        ax.plot(opt, sweep.loc[opt, 'ratio'], 'o', color=line.get_color(), markersize=7)

    ax.set_xlabel('Solar fraction')
    ax.set_ylabel('Mean / Std (generation-stability ratio)')
    ax.set_title('Generation-Stability Tradeoff: Solar Fraction Sweep')
    ax.legend(ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(SWEEP_PNG, dpi=150)
    plt.close(fig)
    print(f"  Saved → {SWEEP_PNG}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    """Compute optimal solar:wind mix per station and save outputs."""
    print(f"\n{'='*60}")
    print("optimize_mix.py — Phase B Step 3: Optimal solar:wind mix")
    print(f"{'='*60}\n")

    if not RGI_CSV.exists():
        print(f"ERROR: rgi_dataset.csv not found: {RGI_CSV}")
        print("Run scripts/compute_rgi.py first.")
        sys.exit(1)

    print("Step 1 — Loading rgi_dataset.csv ...")
    df = pd.read_csv(RGI_CSV, parse_dates=['date'])
    print(f"  rows: {len(df):,}")

    monsoon_mask    = df['monsoon_phase'] == 1
    nonmonsoon_mask = df['monsoon_phase'].isin([0, 2])
    print(f"  active_monsoon rows: {monsoon_mask.sum():,}")
    print(f"  non-monsoon rows   : {nonmonsoon_mask.sum():,}")

    print("\nStep 2 — Sweeping solar fractions 0..100% per station ...")

    records: list[dict] = []
    sweeps:  dict[str, pd.DataFrame] = {}
    optimal_fracs: dict[str, float] = {}

    for station, grp in df.groupby('station', sort=True):
        solar = grp['solar_rgi']
        wind  = grp['wind_rgi']

        sweep = sweep_mix(solar, wind)
        sweeps[station] = sweep

        # 1. Optimal mix — maximise mean/std ratio
        optimal_solar_pct, optimal_wind_pct = find_optimal_mix(solar, wind)
        optimal_fracs[station] = optimal_solar_pct

        # 2. Max generation mix — maximise mean only
        max_gen_solar_pct = best_frac(sweep, 'mean', maximise=True)

        # 3. Min variance mix — minimise std only
        min_var_solar_pct = best_frac(sweep, 'std', maximise=False)

        # 4. Monsoon-season optimal (active_monsoon rows only)
        mon_grp = grp[monsoon_mask.loc[grp.index]]
        if len(mon_grp) > 0:
            monsoon_solar_pct, _ = find_optimal_mix(mon_grp['solar_rgi'], mon_grp['wind_rgi'])
        else:
            monsoon_solar_pct = float('nan')

        # 5. Non-monsoon optimal (pre + post monsoon rows only)
        nonmon_grp = grp[nonmonsoon_mask.loc[grp.index]]
        if len(nonmon_grp) > 0:
            nonmonsoon_solar_pct, _ = find_optimal_mix(nonmon_grp['solar_rgi'], nonmon_grp['wind_rgi'])
        else:
            nonmonsoon_solar_pct = float('nan')

        mean_at_opt = sweep.loc[optimal_solar_pct, 'mean']
        std_at_opt  = sweep.loc[optimal_solar_pct, 'std']

        records.append({
            'station':              station,
            'climate_zone':         CLIMATE_ZONES.get(station, 'Unknown'),
            'optimal_solar_pct':    optimal_solar_pct,
            'optimal_wind_pct':     optimal_wind_pct,
            'max_gen_solar_pct':    max_gen_solar_pct,
            'max_gen_wind_pct':     1 - max_gen_solar_pct,
            'min_var_solar_pct':    min_var_solar_pct,
            'min_var_wind_pct':     1 - min_var_solar_pct,
            'monsoon_solar_pct':    monsoon_solar_pct,
            'monsoon_wind_pct':     1 - monsoon_solar_pct if not np.isnan(monsoon_solar_pct) else float('nan'),
            'nonmonsoon_solar_pct': nonmonsoon_solar_pct,
            'nonmonsoon_wind_pct':  1 - nonmonsoon_solar_pct if not np.isnan(nonmonsoon_solar_pct) else float('nan'),
            'mean_combined_rgi_at_optimal': mean_at_opt,
            'std_combined_rgi_at_optimal':  std_at_opt,
            'mean_solar_only_rgi':  solar.mean(),
            'mean_wind_only_rgi':   wind.mean(),
        })

    out_df = pd.DataFrame(records)
    out_df = out_df.sort_values('optimal_solar_pct', ascending=False).reset_index(drop=True)

    OPTIMAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OPTIMAL_CSV, index=False)
    print(f"\nSaved → {OPTIMAL_CSV}")

    print("\nStep 3 — Generating plots ...")
    plot_mix_chart(out_df)
    plot_sweep_curves(sweeps, optimal_fracs)

    # ── Result table ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("RESULT TABLE — Optimal Solar:Wind Mix (Phase B key finding)")
    print(f"{'='*70}")
    hdr = f"{'Station':<12} {'Optimal Solar%':>15} {'Optimal Wind%':>14} {'Mean RGI':>9} {'Std RGI':>8}"
    print(hdr)
    print('-' * len(hdr))
    for _, r in out_df.iterrows():
        print(
            f"{r['station']:<12} "
            f"{r['optimal_solar_pct']*100:>14.1f}% "
            f"{r['optimal_wind_pct']*100:>13.1f}% "
            f"{r['mean_combined_rgi_at_optimal']:>9.4f} "
            f"{r['std_combined_rgi_at_optimal']:>8.4f}"
        )

    # ── SELF CHECK ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SELF CHECK")
    print(f"{'='*60}")

    by_station = out_df.set_index('station')

    for s in ['Jodhpur', 'Ahmedabad']:
        pct = by_station.loc[s, 'optimal_solar_pct']
        print(f"{s} optimal_solar_pct = {pct*100:.1f}%  (>60% expected): {pct > 0.6}")

    kolkata_pct = by_station.loc['Kolkata', 'optimal_solar_pct']
    print(f"Kolkata optimal_solar_pct = {kolkata_pct*100:.1f}%  (<50% expected): {kolkata_pct < 0.5}")

    coastal = ['Mumbai', 'Chennai']
    inland  = ['Jodhpur', 'Bhopal']
    coastal_wind = by_station.loc[coastal, 'optimal_wind_pct'].mean()
    inland_wind  = by_station.loc[inland, 'optimal_wind_pct'].mean()
    print(f"Coastal mean optimal_wind_pct ({coastal_wind*100:.1f}%) "
          f"> inland mean ({inland_wind*100:.1f}%): {coastal_wind > inland_wind}")

    print("\nMonsoon vs non-monsoon optimal solar% per station:")
    diffs = (out_df['nonmonsoon_solar_pct'] - out_df['monsoon_solar_pct']).abs()
    for _, r in out_df.iterrows():
        print(f"  {r['station']:<12} monsoon={r['monsoon_solar_pct']*100:>5.1f}%  "
              f"non-monsoon={r['nonmonsoon_solar_pct']*100:>5.1f}%  "
              f"diff={abs(r['nonmonsoon_solar_pct']-r['monsoon_solar_pct'])*100:.1f}pp")
    print(f"\nAny station with differing monsoon/non-monsoon optimal (>1pp): "
          f"{bool((diffs > 0.01).any())}")

    print(f"\n{'='*60}")
    print("Script 10 complete.")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
