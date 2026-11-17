"""make_paper_figures.py -- print-legible redraws of paper Figs. 8-11.

Same data as the original Phase B plots (read from outputs/*.csv), but with
bold, larger fonts and print-sized canvases. Writes only to
outputs/paper_outputs/; the original outputs/*.png are untouched.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'outputs'
DEST = OUT / 'paper_outputs'

plt.rcParams.update({
    'font.size': 9, 'font.weight': 'bold',
    'axes.labelweight': 'bold', 'axes.titleweight': 'bold',
    'axes.labelsize': 9, 'axes.titlesize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'axes.linewidth': 1.0,
})

TREND_STATIONS = ['Jodhpur', 'Chennai', 'Kolkata', 'Bengaluru']
ALL_STATIONS = ['Jodhpur', 'Ahmedabad', 'Mumbai', 'Chennai',
                'Hyderabad', 'Bhopal', 'Kolkata', 'Bengaluru']


def _bold_ticks(ax) -> None:
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight('bold')


def fig_trends() -> None:
    df = pd.read_csv(OUT / 'generation_forecasts_v2.csv', parse_dates=['date'])
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), sharex=True)
    for ax, st in zip(axes.flat, TREND_STATIONS):
        g = df[df['station'] == st].sort_values('date')
        sol = int(round(g['optimal_solar_pct'].iloc[0] * 100))
        ax.plot(g['date'], g['actual_combined_rgi'], color='tab:blue', lw=1.1, label='Actual')
        ax.plot(g['date'], g['forecasted_combined_rgi'], color='tab:orange', lw=1.1, label='Forecast')
        ax.set_title(f'{st} ({sol}% solar / {100 - sol}% wind)')
        ax.set_ylabel('Combined RGI')
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b'))
        ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(interval=2))
        _bold_ticks(ax)
    axes[0, 0].legend(loc='lower left', ncol=2, frameon=False)
    for ax in axes[1]:
        ax.set_xlabel('Month, 2024')
    fig.tight_layout()
    fig.savefig(DEST / 'generation_trends_v2.png', dpi=300)
    plt.close(fig)


def fig_relief() -> None:
    df = pd.read_csv(OUT / 'ev_demand_matching.csv')
    m = df.groupby('station')['additional_evs_enabled'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    bars = ax.bar(range(len(m)), m.values, color=plt.cm.viridis(np.linspace(0.8, 0.15, len(m))))
    for b, v in zip(bars, m.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f'{v:.2f}', ha='center', va='bottom', fontsize=7)
    ax.axhline(1, color='crimson', ls='--', lw=1.0, label='1 extra EV')
    ax.set_xticks(range(len(m)))
    ax.set_xticklabels(m.index, rotation=45, ha='right')
    ax.set_ylabel('Mean additional EV chargers')
    ax.set_ylim(0, m.max() * 1.18)
    ax.legend(frameon=False, loc='lower left')
    _bold_ticks(ax)
    fig.tight_layout()
    fig.savefig(DEST / 'ev_capacity_relief.png', dpi=300)
    plt.close(fig)


def fig_adequacy() -> None:
    df = pd.read_csv(OUT / 'ev_adequacy_forecast.csv')
    fig, axes = plt.subplots(4, 2, figsize=(7.2, 6.4), sharex=True, sharey=True)
    for ax, st in zip(axes.flat, ALL_STATIONS):
        g = df[df['station'] == st].sort_values('year')
        ax.plot(g['year'], g['rgi_supply'], color='steelblue', lw=1.6, marker='o', ms=3, label='RGI supply')
        ax.plot(g['year'], g['ev_rgi_demand'], color='crimson', lw=1.6, ls='--', marker='s', ms=3,
                label='EV RGI demand')
        ax.fill_between(g['year'], g['rgi_supply'], g['ev_rgi_demand'],
                        where=g['rgi_supply'] >= g['ev_rgi_demand'], color='seagreen', alpha=0.25)
        ax.set_title(f"{st} ({g['state'].iloc[0]})", fontsize=8.5)
        ax.grid(alpha=0.3)
        ax.set_xticks(range(2024, 2031))
        _bold_ticks(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel('RGI fraction')
    for ax in axes[-1]:
        ax.set_xlabel('Year')
    axes[0, 0].legend(loc='center left', frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(DEST / 'ev_adequacy_summary.png', dpi=300)
    plt.close(fig)


def fig_upgrade() -> None:
    df = pd.read_csv(OUT / 'ev_adequacy_forecast.csv')
    d = df[df['year'] == 2030].sort_values('panels_needed_adani_440w')
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    ax.barh(d['station'], d['panels_needed_adani_440w'],
            color=['firebrick' if p > 0 else 'seagreen' for p in d['panels_needed_adani_440w']])
    for i, p in enumerate(d['panels_needed_adani_440w']):
        ax.text(p, i, '  Adequate' if p == 0 else f'  {int(p)} panels', va='center', fontsize=7.5)
    ax.set_xlim(-0.5, 8)
    ax.set_xlabel('Extra 440 W panels needed by 2030')
    ax.grid(axis='x', alpha=0.3)
    _bold_ticks(ax)
    fig.tight_layout()
    fig.savefig(DEST / 'upgrade_requirements.png', dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    DEST.mkdir(parents=True, exist_ok=True)
    for f in (fig_trends, fig_relief, fig_adequacy, fig_upgrade):
        f()
        print(f'{f.__name__} done')
