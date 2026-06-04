"""
monsoon_labels.py
Generate per-station per-day monsoon phase labels covering
2017-04-01 to 2024-12-31 based on IMD official onset/withdrawal dates.
Saves: data/processed/monsoon_phases.csv
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT / 'data' / 'processed' / 'monsoon_phases.csv'

DATE_START = "2017-04-01"
DATE_END   = "2024-12-31"

PHASE_NAMES: dict[int, str] = {
    0: 'pre_monsoon',
    1: 'active_monsoon',
    2: 'post_monsoon',
}

# IMD official monsoon onset dates — DO NOT MODIFY
ONSET_DATES: dict[str, dict[int, str]] = {
    'Jodhpur':   {2016:'8 Jul',  2017:'15 Jul', 2018:'29 Jun',
                  2019:'18 Jul', 2020:'25 Jun', 2021:'15 Jul',
                  2022:'2 Jul',  2023:'30 Jun', 2024:'2 Jul'},
    'Ahmedabad': {2016:'30 Jun', 2017:'25 Jun', 2018:'26 Jun',
                  2019:'1 Jul',  2020:'23 Jun', 2021:'25 Jun',
                  2022:'29 Jun', 2023:'25 Jun', 2024:'26 Jun'},
    'Mumbai':    {2016:'21 Jun', 2017:'11 Jun', 2018:'16 Jun',
                  2019:'24 Jun', 2020:'14 Jun', 2021:'10 Jun',
                  2022:'12 Jun', 2023:'10 Jun', 2024:'9 Jun'},
    'Chennai':   {2016:'5 Jun',  2017:'7 Jun',  2018:'5 Jun',
                  2019:'5 Jun',  2020:'2 Jun',  2021:'3 Jun',
                  2022:'5 Jun',  2023:'1 Jun',  2024:'2 Jun'},
    'Hyderabad': {2016:'10 Jun', 2017:'9 Jun',  2018:'5 Jun',
                  2019:'10 Jun', 2020:'13 Jun', 2021:'15 Jun',
                  2022:'10 Jun', 2023:'10 Jun', 2024:'10 Jun'},
    'Bhopal':    {2016:'21 Jun', 2017:'22 Jun', 2018:'26 Jun',
                  2019:'22 Jun', 2020:'19 Jun', 2021:'15 Jul',
                  2022:'23 Jun', 2023:'22 Jun', 2024:'19 Jun'},
    'Kolkata':   {2016:'12 Jun', 2017:'10 Jun', 2018:'10 Jun',
                  2019:'12 Jun', 2020:'8 Jun',  2021:'4 Jun',
                  2022:'9 Jun',  2023:'9 Jun',  2024:'28 May'},
    'Bengaluru': {2016:'5 Jun',  2017:'7 Jun',  2018:'5 Jun',
                  2019:'5 Jun',  2020:'2 Jun',  2021:'3 Jun',
                  2022:'30 May', 2023:'1 Jun',  2024:'2 Jun'},
}

# IMD official monsoon withdrawal dates — DO NOT MODIFY
WITHDRAWAL_DATES: dict[str, dict[int, str]] = {
    'Jodhpur':   {2016:'27 Sep', 2017:'28 Sep', 2018:'30 Sep',
                  2019:'9 Oct',  2020:'29 Sep', 2021:'7 Oct',
                  2022:'24 Sep', 2023:'27 Sep', 2024:'28 Sep'},
    'Ahmedabad': {2016:'13 Sep', 2017:'28 Sep', 2018:'30 Sep',
                  2019:'11 Oct', 2020:'1 Oct',  2021:'7 Oct',
                  2022:'26 Sep', 2023:'28 Sep', 2024:'28 Sep'},
    'Mumbai':    {2016:'15 Oct', 2017:'15 Oct', 2018:'3 Oct',
                  2019:'14 Oct', 2020:'5 Oct',  2021:'9 Oct',
                  2022:'8 Oct',  2023:'4 Oct',  2024:'10 Oct'},
    'Chennai':   {2016:'23 Oct', 2017:'24 Oct', 2018:'13 Oct',
                  2019:'15 Oct', 2020:'14 Oct', 2021:'18 Oct',
                  2022:'18 Oct', 2023:'14 Oct', 2024:'14 Oct'},
    'Hyderabad': {2016:'17 Oct', 2017:'20 Oct', 2018:'13 Oct',
                  2019:'15 Oct', 2020:'14 Oct', 2021:'10 Oct',
                  2022:'20 Oct', 2023:'14 Oct', 2024:'5 Oct'},
    'Bhopal':    {2016:'10 Oct', 2017:'8 Oct',  2018:'2 Oct',
                  2019:'13 Oct', 2020:'1 Oct',  2021:'8 Oct',
                  2022:'13 Oct', 2023:'7 Oct',  2024:'3 Oct'},
    'Kolkata':   {2016:'14 Oct', 2017:'15 Oct', 2018:'4 Oct',
                  2019:'11 Oct', 2020:'24 Oct', 2021:'10 Oct',
                  2022:'15 Oct', 2023:'11 Oct', 2024:'12 Oct'},
    'Bengaluru': {2016:'23 Oct', 2017:'20 Oct', 2018:'13 Oct',
                  2019:'15 Oct', 2020:'16 Oct', 2021:'18 Oct',
                  2022:'18 Oct', 2023:'14 Oct', 2024:'10 Oct'},
}


# ── Core logic ────────────────────────────────────────────────────────────────

def parse_imd_date(date_str: str, year: int) -> datetime:
    """Parse an IMD date string like '15 Jul' with an explicit year.

    Args:
        date_str: Day + abbreviated month, e.g. '15 Jul'.
        year: Four-digit year integer.

    Returns:
        datetime object.
    """
    return datetime.strptime(f"{date_str} {year}", "%d %b %Y")


def assign_phase(date: pd.Timestamp, station: str) -> tuple[int, str]:
    """Assign monsoon phase integer and name for a given station + date.

    Looks up onset/withdrawal from the hardcoded IMD dicts.  Raises a
    clear ValueError if the year is not covered (valid range 2017–2024).

    Args:
        date: The calendar date to classify.
        station: Station name matching ONSET_DATES keys.

    Returns:
        Tuple of (phase_int, phase_name_str).

    Raises:
        ValueError: If year or station is not in the IMD lookup dicts.
    """
    year = date.year
    if year not in ONSET_DATES.get(station, {}):
        raise ValueError(
            f"No IMD onset date for station='{station}', year={year}. "
            f"Covered years: {sorted(next(iter(ONSET_DATES.values())).keys())}"
        )
    onset_str      = ONSET_DATES[station][year]
    withdrawal_str = WITHDRAWAL_DATES[station][year]

    onset      = parse_imd_date(onset_str, year)
    withdrawal = parse_imd_date(withdrawal_str, year)

    d = date.date()
    if d < onset.date():
        phase = 0
    elif onset.date() <= d <= withdrawal.date():
        phase = 1
    else:
        phase = 2

    return phase, PHASE_NAMES[phase]


def build_labels() -> pd.DataFrame:
    """Build the full station × date monsoon label DataFrame.

    Returns:
        DataFrame with columns: date, station, monsoon_phase, phase_name.
    """
    date_range = pd.date_range(start=DATE_START, end=DATE_END, freq='D')
    stations = list(ONSET_DATES.keys())

    records: list[dict] = []
    for station in stations:
        print(f"  Labelling {station} ...")
        for date in date_range:
            phase, name = assign_phase(date, station)
            records.append({
                'date':          date,
                'station':       station,
                'monsoon_phase': phase,
                'phase_name':    name,
            })

    return pd.DataFrame(records)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """Generate monsoon labels and save CSV with SELF CHECK."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("monsoon_labels.py — IMD Monsoon Phase Labelling")
    print(f"Date range: {DATE_START} → {DATE_END}")
    print(f"Stations  : {list(ONSET_DATES.keys())}")
    print(f"{'='*60}\n")

    df = build_labels()
    df = df.sort_values(['station', 'date']).reset_index(drop=True)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved → {OUTPUT_PATH}")

    # ── SELF CHECK ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SELF CHECK")
    print(f"{'='*60}")

    print(f"\nTotal rows     : {len(df):,}")
    print(f"Unique stations: {df['station'].nunique()}")

    print("\nPhase distribution per station (count of 0, 1, 2):")
    dist = df.groupby(['station', 'monsoon_phase']).size().unstack(fill_value=0)
    dist.columns = [PHASE_NAMES[c] for c in dist.columns]
    print(dist.to_string())

    print("\nJodhpur active monsoon days per year:")
    jodhpur = df[(df['station'] == 'Jodhpur') & (df['monsoon_phase'] == 1)].copy()
    jodhpur['year'] = jodhpur['date'].dt.year
    print(jodhpur.groupby('year').size().to_string())

    print("\nSample — Jodhpur July–August 2019 (should be active_monsoon):")
    sample_active = df[
        (df['station'] == 'Jodhpur') &
        (df['date'] >= '2019-07-01') &
        (df['date'] <= '2019-08-31')
    ][['date', 'station', 'monsoon_phase', 'phase_name']].head(10)
    print(sample_active.to_string(index=False))

    print("\nSample — Jodhpur January 2020 (should be pre_monsoon):")
    sample_pre = df[
        (df['station'] == 'Jodhpur') &
        (df['date'] >= '2020-01-01') &
        (df['date'] <= '2020-01-31')
    ][['date', 'station', 'monsoon_phase', 'phase_name']].head(10)
    print(sample_pre.to_string(index=False))

    print(f"\n{'='*60}")
    print("Script 2 complete.")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
