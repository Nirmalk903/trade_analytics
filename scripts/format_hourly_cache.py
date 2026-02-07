#!/usr/bin/env python3
"""Reformat hourly cache JSON files to include dd-mmm-yyyy date format."""

from pathlib import Path
import pandas as pd

folder = Path(__file__).resolve().parents[1] / "Underlying_data_kite_hourly"
if not folder.exists():
    print("Hourly data folder not found")
    raise SystemExit(0)

files = sorted(folder.glob("*.json"))
if not files:
    print("No hourly json files found")
    raise SystemExit(0)

for f in files:
    try:
        df = pd.read_json(f, orient="records")
        if df.empty:
            print(f"Skipped empty: {f.name}")
            continue

        if "date_time" in df.columns:
            df["date_time"] = pd.to_datetime(df["date_time"])
        elif "date" in df.columns and "time" in df.columns:
            df["date_time"] = pd.to_datetime(
                df["date"].astype(str) + " " + df["time"].astype(str),
                format="%d-%b-%Y %H:%M",
            )
        elif "date" in df.columns:
            df["date_time"] = pd.to_datetime(df["date"])
        else:
            print(f"Skipped (no date): {f.name}")
            continue

        df["date"] = df["date_time"].dt.strftime("%d-%b-%Y")
        df["time"] = df["date_time"].dt.strftime("%H:%M")

        df.to_json(f, orient="records", indent=2, default_handler=str)
        print(f"Updated: {f.name}")
    except Exception as exc:
        print(f"Failed: {f.name} -> {exc}")
