#!/usr/bin/env python3
"""
Filter existing economic calendar CSV file based on importance levels.
- Americas, Europe, Japan, China: Keep only High importance
- India: Keep all importance levels
Saves a single combined filtered file.
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
print("FILTERING ECONOMIC CALENDAR CSV".center(80))
print("="*80 + "\n")

# Find the combined calendar file (with or without timestamp)
files_with_timestamp = sorted(Path(calendar_dir).glob('economic_calendar_*.csv'))
files_without_timestamp = sorted(Path(calendar_dir).glob('economic_calendar.csv'))
legacy_all_regions = sorted(Path(calendar_dir).glob('All_Regions_calendar_*.csv'))

file_path = None
if files_with_timestamp:
    file_path = files_with_timestamp[-1]
elif files_without_timestamp:
    file_path = files_without_timestamp[0]
elif legacy_all_regions:
    file_path = legacy_all_regions[-1]

if file_path:
    df = pd.read_csv(file_path)
    if 'Region' not in df.columns:
        region_mapping = {
            'Americas': ['USD', 'CAD', 'BRL', 'MXN', 'United States', 'Canada', 'Brazil', 'Mexico'],
            'Europe': ['EUR', 'GBP', 'CHF', 'SEK', 'NOK', 'DKK', 'Eurozone', 'Germany', 'France', 'UK'],
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
else:
    # Fallback: build a combined calendar from per-region files
    combined_frames = []
    for region in regions_high_only + regions_keep_all:
        files_with_timestamp = sorted(Path(calendar_dir).glob(f'{region}_calendar_*.csv'))
        files_without_timestamp = sorted(Path(calendar_dir).glob(f'{region}_calendar.csv'))
        if files_with_timestamp:
            region_file = files_with_timestamp[-1]
        elif files_without_timestamp:
            region_file = files_without_timestamp[0]
        else:
            continue

        region_df = pd.read_csv(region_file)
        region_df['Region'] = region
        combined_frames.append(region_df)

    if not combined_frames:
        raise FileNotFoundError("No combined economic calendar CSV found and no per-region files available.")

    df = pd.concat(combined_frames, ignore_index=True)

def should_keep_row(row):
    region = row.get('Region')
    if region in regions_high_only:
        return row.get('Importance') == 'High'
    if region in regions_keep_all:
        return True
    return True

df_filtered = df[df.apply(should_keep_row, axis=1)].copy()

for region in sorted(df['Region'].dropna().unique()):
    region_df = df[df['Region'] == region]
    filtered_region_df = df_filtered[df_filtered['Region'] == region]
    if region in regions_high_only:
        print(f"✓ {region}: {len(region_df)} → {len(filtered_region_df)} events (High importance only)")
    else:
        print(f"✓ {region}: {len(region_df)} → {len(filtered_region_df)} events (All importance levels)")

# Save single filtered calendar
output_file = f"{calendar_dir}/economic_calendar_filtered.csv"
df_filtered.to_csv(output_file, index=False)
print(f"\n→ Saved: {Path(output_file).name}\n")

print("="*80)
print("✓ FILTERING COMPLETE".center(80))
print("="*80 + "\n")
