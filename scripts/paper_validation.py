"""paper_validation.py -- numbers for the three placeholder tables in the paper.

  1. MPS vs CPU validation loss, same checkpoint (Table: tab:mps_validation)
  2. Diebold-Mariano tests, trained model vs persistence (tab:significance)
  3. EV-adequacy sensitivity to planning constants (tab:sensitivity)

Reads existing checkpoints/data only; trains nothing, overwrites nothing
except outputs/paper_validation.json.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))

import transformer_monsoon as tm  # noqa: E402
import wind_transformer as wt  # noqa: E402

OUT_JSON = ROOT / 'outputs' / 'paper_validation.json'
CPU = torch.device('cpu')
MPS = torch.device('mps') if torch.backends.mps.is_available() else None


# -- data prep (mirrors the two training scripts exactly) ----------------------

def _split(df: pd.DataFrame, need: list[str]):
    out = []
    for lo, hi in ((tm.TRAIN_START, tm.TRAIN_END), (tm.VAL_START, tm.VAL_END),
                   (tm.TEST_START, tm.TEST_END)):
        part = df[(df['date'] >= lo) & (df['date'] <= hi)].dropna(subset=need)
        out.append(part.reset_index(drop=True))
        print(f"    split {lo}..{hi}: {len(out[-1]):,} rows")
    return out


def prep_solar():
    df = pd.read_csv(tm.DATASET_CSV, parse_dates=['date'])
    aod = pd.read_csv(tm.AOD_CSV, parse_dates=['date'])
    parts = []
    for s in tm.STATIONS_LIST:
        ren = {f"{s}_{v}": v for v in tm.AOD_VARS if f"{s}_{v}" in aod.columns}
        sdf = aod[['date'] + list(ren)].rename(columns=ren).copy()
        sdf['station'] = s
        for v in tm.AOD_VARS:
            sdf[v] = sdf[v].where(sdf[v] < tm.FILL_THRESHOLD, np.nan)
        parts.append(sdf)
    df = df.merge(pd.concat(parts, ignore_index=True), on=['date', 'station'], how='left')
    for v in tm.AOD_VARS:
        df[v] = df.groupby('station')[v].transform(lambda s: s.fillna(s.mean()))
        df[v] = df.groupby('station')[v].transform(lambda s: s.ffill().bfill())
    df = df.sort_values(['station', 'date']).reset_index(drop=True)
    df[tm.TARGET_COL] = df.groupby('station')['ghi'].transform(lambda s: s.shift(-1))
    tr, va, te = _split(df, tm.FEATURE_COLS + [tm.MONSOON_COL, tm.TARGET_COL])
    sc = MinMaxScaler().fit(tr[tm.FEATURE_COLS])          # TRAIN only
    lo, rng = sc.data_min_[0], sc.data_max_[0] - sc.data_min_[0]
    for d in (va, te):
        d[tm.FEATURE_COLS] = sc.transform(d[tm.FEATURE_COLS])
        d[tm.TARGET_COL] = (d[tm.TARGET_COL] - lo) / rng
    return va, te, lo, rng, 0   # ghi is feature 0


def prep_wind():
    df = pd.read_csv(wt.EXTENDED_CSV, parse_dates=['date'])
    mon = pd.read_csv(wt.MONSOON_CSV, parse_dates=['date'])
    for c in df.select_dtypes(include=[np.number]).columns:
        df[c] = df[c].replace(-999, np.nan)
    df = df.merge(mon[['date', 'station', 'monsoon_phase']], on=['date', 'station'], how='left')
    df['doy_sin'] = np.sin(2 * np.pi * df['date'].dt.dayofyear / 365.25)
    df['doy_cos'] = np.cos(2 * np.pi * df['date'].dt.dayofyear / 365.25)
    df['month_sin'] = np.sin(2 * np.pi * df['date'].dt.month / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['date'].dt.month / 12.0)
    df = df.sort_values(['station', 'date']).reset_index(drop=True)
    df[wt.TARGET_COL] = df.groupby('station')['wind_speed_50m'].transform(lambda s: s.shift(-1))
    tr, va, te = _split(df, wt.FEATURE_COLS + [wt.MONSOON_COL, wt.TARGET_COL])
    sc = MinMaxScaler().fit(tr[wt.FEATURE_COLS])          # TRAIN only
    lo, rng = sc.data_min_[0], sc.data_max_[0] - sc.data_min_[0]
    for d in (va, te):
        d[wt.FEATURE_COLS] = sc.transform(d[wt.FEATURE_COLS])
        d[wt.TARGET_COL] = (d[wt.TARGET_COL] - lo) / rng
    return va, te, lo, rng, 0   # wind_speed_50m is feature 0


def load_models():
    s = tm.SolarTransformer(tm.INPUT_SIZE, tm.D_MODEL, tm.N_HEADS, tm.N_LAYERS, tm.DIM_FF, tm.DROPOUT)
    s.load_state_dict(torch.load(tm.MODEL_PATH, map_location=CPU))
    w = wt.WindTransformer(wt.INPUT_SIZE, wt.D_MODEL, wt.N_HEADS, wt.N_LAYERS, wt.DIM_FF, wt.DROPOUT)
    w.load_state_dict(torch.load(wt.MODEL_PATH, map_location=CPU))
    return s.eval(), w.eval()


# -- 1. MPS vs CPU validation loss --------------------------------------------

def val_loss(model, ds, device) -> float:
    model = model.to(device).eval()
    tot = 0.0
    with torch.no_grad():
        for xf, xm, y in DataLoader(ds, batch_size=32, shuffle=False):
            p = model(xf.to(device), xm.to(device))
            tot += torch.nn.functional.mse_loss(p, y.to(device), reduction='sum').item()
    return tot / len(ds)


# -- 2. Diebold-Mariano --------------------------------------------------------

def windows(df, model, feats, moon, tgt, lo, rng, fi):
    """Per-station (actual, model, persistence) in raw units, test windows."""
    res = {}
    for st, g in df.groupby('station', sort=True):
        g = g.sort_values('date').reset_index(drop=True)
        F = g[feats].to_numpy(np.float32)
        M = g[moon].astype('int64').to_numpy()
        T = g[tgt].to_numpy(np.float32)
        a, p, q = [], [], []
        with torch.no_grad():
            for i in range(len(g) - tm.SEQ_LEN):
                if np.isnan(F[i:i + tm.SEQ_LEN]).any() or np.isnan(T[i + tm.SEQ_LEN - 1]):
                    continue
                yp = model(torch.tensor(F[i:i + tm.SEQ_LEN])[None], torch.tensor(M[i:i + tm.SEQ_LEN])[None])
                a.append(T[i + tm.SEQ_LEN - 1] * rng + lo)
                p.append(float(yp[0, 0]) * rng + lo)
                q.append(F[i + tm.SEQ_LEN - 1, fi] * rng + lo)   # persistence: last observed day
        res[st] = tuple(np.array(x, dtype=np.float64) for x in (a, p, q))
    return res


def dm_test(a, p, q):
    """DM, squared-error loss, h=1, Harvey-Leybourne-Newbold correction, two-sided t(n-1).
    Positive statistic = trained model more accurate than persistence."""
    d = (a - q) ** 2 - (a - p) ** 2
    n = len(d)
    stat = d.mean() / math.sqrt(d.var(ddof=1) / n) * math.sqrt((n - 1) / n)
    return float(stat), float(2 * stats.t.sf(abs(stat), n - 1))


# -- 3. EV sensitivity ---------------------------------------------------------

def sensitivity():
    ev = pd.read_csv(ROOT / 'outputs' / 'ev_adequacy_forecast.csv')
    fc = pd.read_csv(ROOT / 'outputs' / 'generation_forecasts_v2.csv')
    cap0 = 50.0
    # verdict demand = evs_per_cluster * 21.6 kWh / (cap * 24); supply = mean forecast RGI
    ev['need_kwh'] = ev['evs_per_cluster'] * 21.6
    out = {'cluster_capacity': {}, 'transformer_rating': {}, 'legacy_load': {}}
    for f in (-0.3, -0.2, 0.2, 0.3):
        cap = cap0 * (1 + f)
        flips = ev[ev['need_kwh'] / (cap * 24) > ev['rgi_supply']]['station'].nunique()
        out['cluster_capacity'][f] = int(flips)
        # transformer rating / legacy load never enter the verdict; they only move
        # the bare-transformer charger count and the snapshot relief
        for key, kva, leg in (('transformer_rating', 100 * (1 + f), 0.65), ('legacy_load', 100, 0.65 * (1 + f))):
            headroom = kva * 0.9 * (1 - leg)
            base = math.floor(headroom / 7.2)
            gen = fc['forecasted_combined_rgi'] * cap0
            extra = np.floor((headroom + gen) / 7.2) - base
            share = (extra >= 1).groupby(fc['station']).mean().min() * 100
            out[key][f] = {'verdict_flips': 0, 'bare_chargers': int(base),
                           'min_pct_days_extra_ev': float(share)}
    ratio = (ev['need_kwh'] / 24 / ev['rgi_supply']).max()   # kW of capacity at the flip point
    out['breakeven_cluster_kw'] = float(ratio)
    return out


def main() -> None:
    print(f"MPS available: {MPS is not None}")
    solar, wind = load_models()
    print("Preparing solar data ...")
    s_va, s_te, s_lo, s_rng, s_fi = prep_solar()
    print("Preparing wind data ...")
    w_va, w_te, w_lo, w_rng, w_fi = prep_wind()
    res: dict = {'mps': {}, 'dm': {}, 'check': {}}

    for name, model, va, mon, tgt, feats in (
        ('SolarTransformer', solar, s_va, tm.MONSOON_COL, tm.TARGET_COL, tm.FEATURE_COLS),
        ('WindTransformer', wind, w_va, wt.MONSOON_COL, wt.TARGET_COL, wt.FEATURE_COLS),
    ):
        ds = (tm if name.startswith('Solar') else wt).SlidingWindowDataset(va, tm.SEQ_LEN, feats, mon, tgt)
        c = val_loss(model, ds, CPU)
        m = val_loss(model, ds, MPS) if MPS else float('nan')
        res['mps'][name] = {'cpu': c, 'mps': m, 'cpu_over_mps': c / m}
        print(f"  {name}: CPU {c:.6f}  MPS {m:.6f}  CPU/MPS {c / m:.3f}")
        model.to(CPU)

    for name, model, te, mon, tgt, feats, lo, rng, fi in (
        ('solar', solar, s_te, tm.MONSOON_COL, tm.TARGET_COL, tm.FEATURE_COLS, s_lo, s_rng, s_fi),
        ('wind', wind, w_te, wt.MONSOON_COL, wt.TARGET_COL, wt.FEATURE_COLS, w_lo, w_rng, w_fi),
    ):
        per = windows(te, model, feats, mon, tgt, lo, rng, fi)
        res['dm'][name] = {st: dict(zip(('stat', 'p'), dm_test(*v))) | {'n': len(v[0])} for st, v in per.items()}
        A, P, Q = (np.concatenate([v[i] for v in per.values()]) for i in range(3))
        res['check'][name] = {'model_rmse': float(np.sqrt(np.mean((A - P) ** 2))),
                              'persistence_rmse': float(np.sqrt(np.mean((A - Q) ** 2)))}
        print(f"  {name} overall RMSE model {res['check'][name]['model_rmse']:.4f} "
              f"persistence {res['check'][name]['persistence_rmse']:.4f}")
        for st, r in res['dm'][name].items():
            print(f"    {st:<10} DM {r['stat']:+.3f}  p {r['p']:.4g}  n {r['n']}")

    res['sensitivity'] = sensitivity()
    print(json.dumps(res['sensitivity'], indent=1))
    OUT_JSON.write_text(json.dumps(res, indent=2))
    print(f"Saved -> {OUT_JSON}")


if __name__ == '__main__':
    main()
