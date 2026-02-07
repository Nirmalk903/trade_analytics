#!/usr/bin/env python3
"""Create mock 2-year hourly data for testing the stock filter agent."""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Create directory
KITE_HOURLY_DATA_DIR = Path("Underlying_data_kite_hourly")
KITE_HOURLY_DATA_DIR.mkdir(exist_ok=True)

# Create 2 years of hourly data for multiple stocks
symbols = ["INFY", "TCS", "RELIANCE"]
for symbol in symbols:
    # Generate 2 years of hourly data (8760 hours)
    dates = pd.date_range(end=datetime.now(), periods=8760, freq='H')
    
    # Generate realistic price movement
    base_price = {"INFY": 1500, "TCS": 3000, "RELIANCE": 1400}[symbol]
    prices = base_price + np.cumsum(np.random.randn(len(dates)) * 2)
    
    data = {
        'date': dates,
        'open': prices + np.random.randn(len(dates)) * 0.5,
        'high': prices + np.abs(np.random.randn(len(dates)) * 1),
        'low': prices - np.abs(np.random.randn(len(dates)) * 1),
        'close': prices,
        'volume': np.random.randint(1000, 50000, len(dates))
    }
    
    df = pd.DataFrame(data)
    output_file = KITE_HOURLY_DATA_DIR / f"{symbol}_1h_data.json"
    df.to_json(output_file, orient='records', indent=2)
    
    print(f"✓ Created {symbol}: {output_file}")
    print(f"  Records: {len(df)}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Price range: {df['close'].min():.2f} - {df['close'].max():.2f}\n")
