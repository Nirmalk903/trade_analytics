#!/usr/bin/env python3
"""
Unified economic calendar fetcher + filter + CLI wrapper.

This module provides a simple programmatic API to fetch calendars
using the existing `economic_calendar_fetcher.EconomicCalendarFetcher` and
utilities to save combined and filtered CSVs for use by the Streamlit app.
"""
from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime, timedelta
import logging
import pandas as pd
import pendulum as pm

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("results/economic_calendar")


def _ensure_output_dir(path: Path | str):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def fetch_and_save(from_date: str | None = None, to_date: str | None = None,
                   regions: list | None = None,
                   trading_econ_key: str | None = None,
                   fred_key: str | None = None,
                   output_dir: str | Path = DEFAULT_OUTPUT_DIR,
                   save_per_region: bool = True) -> dict:
    """Fetch economic calendar using available fetchers and save CSVs.

    Returns a dict of region -> DataFrame.
    """
    _ensure_output_dir(output_dir)
    if regions is None:
        regions = ['Americas', 'Europe', 'India', 'China', 'Japan']

    if from_date is None or to_date is None:
        now = pm.now()
        from_date = (now.subtract(days=1)).to_date_string() if from_date is None else from_date
        to_date = now.add(days=7).to_date_string() if to_date is None else to_date

    # Import the robust fetcher if available, else fall back to market_calendar functions
    try:
        from economic_calendar_fetcher import EconomicCalendarFetcher
        fetcher = EconomicCalendarFetcher(trading_econ_key, fred_key)
        calendars = fetcher.fetch(from_date, to_date, regions)
    except Exception:
        # Best-effort fallback to market_calendar.economic_calendar_with_fallback
        try:
            from market_calendar import economic_calendar_with_fallback
            calendars = economic_calendar_with_fallback(regions=regions, days_back=1, days_forward=7)
        except Exception:
            logger.exception("No calendar fetcher available")
            return {}

    # Save per-region CSVs and a combined CSV
    timestamp = pm.now().format('YYYY-MM-DD_HHmmss')
    out = Path(output_dir)
    saved = []
    combined_frames = []
    for region, df in (calendars or {}).items():
        if df is None or df.empty:
            continue
        if save_per_region:
            fname = out / f"{region}_calendar_{timestamp}.csv"
            df.to_csv(fname, index=False)
            saved.append(str(fname))
        df_copy = df.copy()
        df_copy['Region'] = region
        combined_frames.append(df_copy)

    if combined_frames:
        combined = pd.concat(combined_frames, ignore_index=True)
        combined_fname = out / f"economic_calendar_{timestamp}.csv"
        combined.to_csv(combined_fname, index=False)
        # also save a stable filename without timestamp for easy loading
        combined_stable = out / "economic_calendar.csv"
        combined.to_csv(combined_stable, index=False)
        saved.append(str(combined_fname))
        saved.append(str(combined_stable))

    logger.info(f"Saved calendar files: {saved}")
    return calendars or {}


def filter_and_save(input_path: str | Path | None = None,
                    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
                    output_name: str = "economic_calendar_filtered.csv") -> Path | None:
    """Load a combined calendar CSV (or latest in output_dir), filter by importance
    (high-only for most regions, keep all for India), and save a single filtered CSV.
    Returns the Path to the saved filtered CSV or None if nothing to save.
    """
    out = _ensure_output_dir(output_dir)
    p_input = None

    if input_path:
        p_input = Path(input_path)
    else:
        # find most recent combined file
        candidates = sorted(out.glob('economic_calendar_*.csv')) + sorted(out.glob('economic_calendar.csv'))
        p_input = candidates[-1] if candidates else None

    if not p_input or not p_input.exists():
        logger.warning("No combined calendar CSV found to filter")
        return None

    df = pd.read_csv(p_input)
    if df.empty:
        logger.warning("Combined calendar CSV is empty")
        return None

    # Ensure Region column exists
    if 'Region' not in df.columns:
        # simple inference like other utilities
        region_mapping = {
            'Americas': ['USD', 'United States'],
            'Europe': ['EUR', 'GBP', 'Eurozone', 'Germany', 'France', 'UK'],
            'India': ['INR', 'India'],
            'China': ['CNY', 'CNH', 'China'],
            'Japan': ['JPY', 'Japan'],
        }

        def infer_region(row):
            country = str(row.get('Country', '')).strip()
            currency = str(row.get('Currency', '')).strip()
            for region, tokens in region_mapping.items():
                if country in tokens or currency in tokens:
                    return region
            return 'Unknown'

        df['Region'] = df.apply(infer_region, axis=1)

    regions_high_only = ['Americas', 'Europe', 'Japan', 'China']
    regions_keep_all = ['India']

    def should_keep_row(row):
        region = row.get('Region')
        if region in regions_high_only:
            return str(row.get('Importance')) == 'High'
        if region in regions_keep_all:
            return True
        return True

    df_filtered = df[df.apply(should_keep_row, axis=1)].copy()
    out_file = out / output_name
    df_filtered.to_csv(out_file, index=False)
    logger.info(f"Saved filtered calendar to {out_file}")
    return out_file


def main():
    """CLI wrapper to fetch and optionally save/filter calendars."""
    import argparse

    parser = argparse.ArgumentParser(description="Fetch and save economic calendar with fallback")
    parser.add_argument('--days', type=int, default=7, help='Days forward to fetch')
    parser.add_argument('--regions', nargs='+', default=['Americas','Europe','India','China','Japan'])
    parser.add_argument('--save', action='store_true', help='Save CSV files')
    parser.add_argument('--filter', action='store_true', help='Apply filtering rules and save filtered CSV')
    args = parser.parse_args()

    now = pm.now()
    from_date = now.subtract(days=1).to_date_string()
    to_date = now.add(days=args.days).to_date_string()

    calendars = fetch_and_save(from_date=from_date, to_date=to_date, regions=args.regions)
    if args.filter:
        filter_and_save()

    # Print summary
    total = sum(len(df) for df in calendars.values() if df is not None)
    print(f"Fetched {total} events across {len(calendars)} regions")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
