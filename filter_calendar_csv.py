#!/usr/bin/env python3
"""
Filter existing economic calendar CSV files based on importance levels.
- Americas, Europe, Japan, China: Keep only High importance
- India: Keep all importance levels
"""

import pandas as pd
from pathlib import Path
import pendulum as pm

# Define which regions keep all vs only high
regions_high_only = ['Americas', 'Europe', 'Japan', 'China']
regions_keep_all = ['India']

calendar_dir = 'results/economic_calendar'

# Create directory if it doesn't exist
Path(calendar_dir).mkdir(parents=True, exist_ok=True)

print("\n" + "="*80)
print("FILTERING ECONOMIC CALENDAR CSV FILES".center(80))
print("="*80 + "\n")

# Process each region CSV
for region in regions_high_only + regions_keep_all:
    # Find the CSV file for this region (with or without timestamp)
    files_with_timestamp = sorted(Path(calendar_dir).glob(f'{region}_calendar_*.csv'))
    files_without_timestamp = sorted(Path(calendar_dir).glob(f'{region}_calendar.csv'))
    
    # Use the most recent timestamped file, or the non-timestamped file
    if files_with_timestamp:
        file_path = files_with_timestamp[-1]
    elif files_without_timestamp:
        file_path = files_without_timestamp[0]
    else:
        print(f"⚠ No CSV file found for {region}")
        continue
    
    df = pd.read_csv(file_path)
    
    if region in regions_high_only:
        # Keep only High importance
        df_filtered = df[df['Importance'] == 'High'].copy()
        print(f"✓ {region}: {len(df)} → {len(df_filtered)} events (High importance only)")
    else:
        # Keep all
        df_filtered = df.copy()
        print(f"✓ {region}: {len(df)} events (All importance levels)")
    
    # Save without timestamp
    new_filename = f"{calendar_dir}/{region}_calendar.csv"
    df_filtered.to_csv(new_filename, index=False)
    print(f"  → Saved: {Path(new_filename).name}\n")

print("="*80)
print("✓ FILTERING COMPLETE".center(80))
print("="*80 + "\n")
