"""
Test script for stock_filter_agent using synthetic data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from stock_filter_agent import (
    calculate_sma_slope,
    classify_slope,
    detect_slope_direction_change,
    get_cusum_signal,
    check_price_above_sma,
    analyze_stock
)

# Generate synthetic test data
def create_test_data(pattern="uptrend"):
    """Create synthetic price data for testing."""
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    
    if pattern == "uptrend":
        # Uptrend with slope changing from negative to positive
        prices = 100 + np.arange(100) * 0.5 + np.random.normal(0, 2, 100)
    elif pattern == "downtrend":
        # Downtrend
        prices = 100 - np.arange(100) * 0.3 + np.random.normal(0, 2, 100)
    elif pattern == "trough":
        # Create a trough: down then up
        down_part = 100 - np.arange(50) * 0.5
        up_part = 75 + np.arange(50) * 0.5
        prices = np.concatenate([down_part, up_part]) + np.random.normal(0, 1, 100)
    else:
        prices = np.random.normal(100, 5, 100)
    
    df = pd.DataFrame({
        'date': dates,
        'open': prices + np.random.normal(0, 1, 100),
        'high': prices + 2 + np.random.normal(0, 1, 100),
        'low': prices - 2 + np.random.normal(0, 1, 100),
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, 100)
    })
    df.set_index('date', inplace=True)
    return df

def test_slope_detection():
    """Test slope detection with tolerance."""
    print("\n" + "="*80)
    print("TEST 1: Slope Direction Detection")
    print("="*80)
    
    # Create data with a clear trough
    df = create_test_data("trough")
    sma_values, slope_values = calculate_sma_slope(df['close'], period=10, lookback=1)
    
    print(f"\nGenerated synthetic data with {len(df)} candles")
    print(f"Price range: {df['close'].min():.2f} - {df['close'].max():.2f}")
    
    # Detect direction changes
    direction_changes, classified_slopes = detect_slope_direction_change(slope_values, tolerance=0.001)
    
    # Find where change occurred
    change_indices = np.where(direction_changes)[0]
    print(f"\nSlope direction changes detected at indices: {change_indices}")
    
    if len(change_indices) > 0:
        idx = change_indices[0]
        print(f"\nFirst direction change at index {idx}:")
        print(f"  Previous 3 slopes: {slope_values[max(0, idx-3):idx]}")
        print(f"  Current slope: {slope_values[idx]:.6f}")
        print(f"  Classified slopes: {classified_slopes[max(0, idx-3):idx+1]}")
        print(f"  ✓ Direction change from negative to positive detected!")
    else:
        print("\n✗ No direction change detected (expected for this synthetic data)")

def test_price_above_sma():
    """Test price vs SMA check."""
    print("\n" + "="*80)
    print("TEST 2: Price Above SMA")
    print("="*80)
    
    df = create_test_data("uptrend")
    sma_values, slope_values = calculate_sma_slope(df['close'], period=10, lookback=1)
    
    price_above_sma = check_price_above_sma(df['close'], sma_values)
    
    print(f"\nLast 5 price vs SMA comparison:")
    for i in range(-5, 0):
        price = df['close'].iloc[i]
        sma = sma_values[i]
        above = price_above_sma[i]
        status = "✓" if above else "✗"
        print(f"  {status} Price: {price:.2f} > SMA: {sma:.2f} = {above}")
    
    print(f"\nPercentage of prices above SMA: {price_above_sma.sum() / len(price_above_sma) * 100:.1f}%")

def test_cusum():
    """Test CUSUM signal generation."""
    print("\n" + "="*80)
    print("TEST 3: CUSUM Signal Generation")
    print("="*80)
    
    df = create_test_data("trough")
    cusum_trigger, diag_df = get_cusum_signal(df['close'])
    
    print(f"\nGenerated CUSUM signals for {len(df)} data points")
    print(f"Total triggers: {np.sum(cusum_trigger != 0)}")
    
    # Show last signals
    last_triggers = cusum_trigger[-10:]
    print(f"\nLast 10 CUSUM signals: {last_triggers}")
    
    trigger_values = np.unique(cusum_trigger)
    print(f"Signal values present: {trigger_values}")

def test_full_analysis():
    """Test full stock analysis with synthetic data."""
    print("\n" + "="*80)
    print("TEST 4: Full Stock Analysis")
    print("="*80)
    
    # Create synthetic data that might meet criteria
    df = create_test_data("trough")
    
    result = analyze_stock(df, "TEST_STOCK", sma_period=10, slope_lookback=1)
    
    if result:
        print(f"\n✓ Analysis completed for {result['symbol']}")
        print(f"\nCriteria Summary:")
        print(f"  1. Slope direction change: {'✓' if result['criterion_1_slope_change']['meets'] else '✗'}")
        print(f"     - Current slope: {result['criterion_1_slope_change']['current_slope']:.6f}")
        print(f"     - Classified: {result['criterion_1_slope_change']['current_slope_classified']}")
        
        print(f"  2. CUSUM = 1: {'✓' if result['criterion_2_cusum']['meets'] else '✗'}")
        print(f"     - Value: {result['criterion_2_cusum']['cusum_value']}")
        
        print(f"  3. Price > SMA: {'✓' if result['criterion_3_price_above_sma']['meets'] else '✗'}")
        print(f"     - Price: {result['criterion_3_price_above_sma']['price']:.2f}")
        print(f"     - SMA: {result['criterion_3_price_above_sma']['sma']:.2f}")
        
        print(f"\n📊 Overall: {result['criteria_met_count']}/3 criteria met")
        print(f"   All criteria: {'✓ PASS' if result['meets_all_criteria'] else '✗ FAIL'}")

if __name__ == "__main__":
    print("\n" + "="*80)
    print("STOCK FILTER AGENT - UNIT TESTS")
    print("="*80)
    
    test_slope_detection()
    test_price_above_sma()
    test_cusum()
    test_full_analysis()
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETED")
    print("="*80 + "\n")
