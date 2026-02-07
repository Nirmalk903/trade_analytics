"""
Stock Filter Agent - Optimized Version
Filters stocks based on three technical criteria:
  1) SMA slope changing from negative to positive
  2) CUSUM filter value = 1
  3) Current price above SMA
Uses engineered data by default for fast analysis.
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import pandas as pd
import numpy as np
from dotenv import load_dotenv

from algorithms.cusum_filter import getTEvents

# Optional imports
try:
    from kiteconnect import KiteConnect
except ImportError:
    KiteConnect = None

try:
    from data_download_vbt import get_symbols, get_dates_from_most_active_files
except ImportError:
    get_symbols = None
    get_dates_from_most_active_files = None

# Configuration
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results" / "stock_filter"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
ENGINEERED_DATA_DIR = Path(__file__).parent / "Engineered_data"
KITE_HOURLY_DATA_DIR = Path(__file__).parent / "Underlying_data_kite_hourly"
KITE_HOURLY_DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_engineered_data(symbol: str) -> Optional[pd.DataFrame]:
    """Load pre-engineered data from local files."""
    try:
        file_path = ENGINEERED_DATA_DIR / f"{symbol}_1d_features.json"
        
        if not file_path.exists():
            return None
        
        df = pd.read_json(file_path, lines=True, orient='records')
        
        if df.empty:
            return None
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        
        return df.sort_index()
    
    except Exception as e:
        logger.error(f"Error loading engineered data for {symbol}: {e}")
        return None


def get_all_stocks(top_n: Optional[int] = None) -> List[str]:
    """Load stocks from most active underlying list."""
    if not get_symbols or not get_dates_from_most_active_files:
        logger.error("Cannot load symbols: data_download_vbt not available")
        return []
    
    try:
        dates = get_dates_from_most_active_files()
        if dates is None or len(dates) == 0:
            logger.error("No most active underlying files found")
            return []
        
        latest_date = dates[-1]
        symbols, _ = get_symbols(latest_date, top_n=top_n or 500)
        logger.info(f"Loaded {len(symbols)} stocks from most active list")
        return symbols
    
    except Exception as e:
        logger.error(f"Error loading stocks: {e}")
        return []


# ============================================================================
# TECHNICAL ANALYSIS
# ============================================================================

def calculate_sma_slope(prices: pd.Series, period: int = 10, lookback: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate Simple Moving Average and its slope.
    
    Args:
        prices: Series of prices
        period: Period for SMA calculation (default 10)
        lookback: Number of periods to look back for slope calculation (default 1)
                  lookback=1: slope[i] = sma[i] - sma[i-1]
                  lookback=2: slope[i] = sma[i] - sma[i-2]
                  etc.
    
    Returns:
        Tuple of (SMA values, Slope of SMA based on lookback period)
    """
    sma = prices.rolling(window=period).mean()
    
    # Calculate slope based on lookback period
    if lookback <= 0:
        raise ValueError("lookback must be positive integer")
    
    slope = sma.diff(periods=lookback)
    return sma.values, slope.values


def classify_slope(slope: np.ndarray, tolerance: float = 0.001) -> np.ndarray:
    """Classify slope as -1 (down), 0 (flat), or +1 (up)."""
    classified = np.zeros(len(slope), dtype=int)
    classified[slope > tolerance] = 1
    classified[slope < -tolerance] = -1
    return classified


def detect_slope_direction_change(slope: np.ndarray, tolerance: float = 0.001) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect if slope changed from negative to positive, capturing local troughs.
    
    Uses classified slopes to identify direction changes while ignoring flat regions.
    A trough is detected when: previous slope was negative and current is positive,
    potentially with flat region(s) in between.
    
    Args:
        slope: Array of slope values
        tolerance: Tolerance level for classifying slope as flat
    
    Returns:
        Tuple of (direction_change array, classified_slope array)
        - direction_change: Boolean array indicating where trough/direction change occurred
        - classified_slope: Array with slope classifications (-1, 0, +1)
    """
    if len(slope) < 2:
        classified = classify_slope(slope, tolerance)
        return np.array([False] * len(slope)), classified
    
    classified = classify_slope(slope, tolerance)
    direction_changes = np.zeros(len(slope), dtype=bool)
    
    # Track the last non-zero slope direction
    last_negative_idx = -1
    
    for i in range(1, len(classified)):
        # If we encounter a positive slope
        if classified[i] > 0:
            # Check if there was a negative slope before (possibly with flats in between)
            if last_negative_idx >= 0:
                # Mark the current point as direction change (local trough detected)
                direction_changes[i] = True
                last_negative_idx = -1
        
        # Update last negative slope index
        if classified[i] < 0:
            last_negative_idx = i
    
    return direction_changes, classified


def get_cusum_signal(prices: pd.Series, h: Optional[float] = None) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Calculate CUSUM filter signal.
    
    Args:
        prices: Series of prices
        h: CUSUM threshold (if None, calculated as 10% of median absolute price)
    
    Returns:
        Tuple of (trigger array, diagnostic dataframe)
    """
    if h is None:
        h = max(1e-6, float(abs(prices).median() * 0.1))
    
    t_events, diag_df = getTEvents(prices, h)
    
    # Create trigger array
    trigger = np.zeros(len(prices))
    
    # Map diagnostic dataframe indices to position in original series
    if not diag_df.empty:
        # Get the position of each index in the original series
        for idx in diag_df.index:
            # Find the position of this index in the original prices series
            pos = prices.index.get_loc(idx)
            if isinstance(pos, (int, np.integer)):
                trigger[pos] = diag_df.loc[idx, 'trigger']
    
    return trigger, diag_df


def check_price_above_sma(prices: pd.Series, sma: np.ndarray) -> np.ndarray:
    """
    Check if current price is above SMA.
    
    Args:
        prices: Series of prices
        sma: SMA values
    
    Returns:
        Boolean array where True means price > SMA
    """
    return prices.values > sma


def fetch_kite_data(kite: KiteConnect, symbol: str, days: int = 30, interval: str = "day") -> Optional[pd.DataFrame]:
    """Fetch historical data from Kite API (optional fallback)."""
    if not KiteConnect:
        logger.error("KiteConnect not installed")
        return None
    
    try:
        instruments = kite.instruments(exchange="NSE")
        token = None
        for inst in instruments:
            if inst.get("tradingsymbol", "").upper() == symbol.upper() and inst.get("segment", "") == "NSE":
                token = inst.get("instrument_token")
                break
        
        if token is None:
            return None
        
        from datetime import timedelta
        end = datetime.now()
        start = end - timedelta(days=days)
        data = kite.historical_data(token, start, end, interval=interval)
        
        if not data:
            return None
        
        df = pd.DataFrame(data)
        df["date_time"] = pd.to_datetime(df["date"])
        df.set_index("date_time", inplace=True)
        
        return df.sort_index()
    
    except Exception as e:
        logger.error(f"Error fetching Kite data for {symbol}: {e}")
        return None


def load_or_fetch_kite_hourly(kite: Optional[KiteConnect], symbol: str, force_refresh: bool = False) -> Optional[pd.DataFrame]:
    """Load hourly data from cache or fetch from Kite (2 years of data)."""
    cache_file = KITE_HOURLY_DATA_DIR / f"{symbol}_1h_data.json"
    
    # Try loading from cache if not forcing refresh
    if cache_file.exists() and not force_refresh:
        try:
            df = pd.read_json(cache_file, orient='records')
            if not df.empty:
                if 'date_time' in df.columns:
                    df['date_time'] = pd.to_datetime(df['date_time'])
                    df.set_index('date_time', inplace=True)
                elif 'date' in df.columns and 'time' in df.columns:
                    df['date_time'] = pd.to_datetime(
                        df['date'].astype(str) + ' ' + df['time'].astype(str),
                        format='%d-%b-%Y %H:%M'
                    )
                    df.set_index('date_time', inplace=True)
                elif 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'], format='%d-%b-%Y')
                    df.set_index('date', inplace=True)
                logger.info(f"Loaded {symbol} hourly data from cache ({len(df)} records)")
                return df.sort_index()
        except Exception as e:
            logger.warning(f"Cache load failed for {symbol}: {e}. Fetching fresh data...")
    
    # Fetch fresh data from Kite (2 years = ~730 days)
    if not kite:
        logger.error("Kite client not available for fetching hourly data")
        return None
    
    try:
        logger.info(f"Fetching 2 years of hourly data for {symbol}...")
        df = fetch_kite_data(kite, symbol, days=730, interval="60")
        
        if df is not None and not df.empty:
            # Save to cache with formatted date
            df_cache = df.reset_index()
            if 'date_time' in df_cache.columns:
                df_cache['date'] = df_cache['date_time'].dt.strftime('%d-%b-%Y')
                df_cache['time'] = df_cache['date_time'].dt.strftime('%H:%M')
            df_cache.to_json(cache_file, orient='records', indent=2, default_handler=str)
            logger.info(f"Saved {symbol} hourly data to cache ({len(df)} records)")
            return df
        else:
            logger.warning(f"No hourly data fetched for {symbol}")
            return None
    
    except Exception as e:
        logger.error(f"Error fetching hourly data for {symbol}: {e}")
        return None


def analyze_stock(df: pd.DataFrame, symbol: str, sma_period: int = 10, 
                  slope_lookback: int = 1) -> Optional[Dict]:
    """Analyze stock against three criteria."""
    if df is None or len(df) < 15:
        return None
    
    try:
        # Use 'Close' or 'close' column
        close_col = 'Close' if 'Close' in df.columns else 'close'
        close_prices = df[close_col]
        
        # Calculate indicators
        sma_values, slope_values = calculate_sma_slope(close_prices, period=sma_period, lookback=slope_lookback)
        slope_changes, classified_slopes = detect_slope_direction_change(slope_values, tolerance=0.001)
        cusum_trigger, _ = get_cusum_signal(close_prices)
        price_above_sma = check_price_above_sma(close_prices, sma_values)
        
        # Current values
        idx = -1
        current_price = float(close_prices.iloc[idx])
        current_sma = float(sma_values[idx])
        last_slope_changed = bool(slope_changes[idx])
        last_cusum = int(cusum_trigger[idx])
        current_price_above_sma = bool(price_above_sma[idx])
        current_slope = float(slope_values[idx])
        current_slope_classified = int(classified_slopes[idx])
        
        # Evaluate criteria
        meets_all = last_slope_changed and (last_cusum == 1) and current_price_above_sma
        criteria_count = sum([last_slope_changed, last_cusum == 1, current_price_above_sma])
        
        return {
            "symbol": symbol,
            "timestamp": str(df.index[-1]),
            "current_price": current_price,
            "sma_10": current_sma,
            "criterion_1_slope_change": {
                "meets": last_slope_changed,
                "current_slope": current_slope,
                "current_slope_classified": current_slope_classified,
                "previous_slope": float(slope_values[-2]) if len(slope_values) > 1 else None,
                "previous_slope_classified": int(classified_slopes[-2]) if len(classified_slopes) > 1 else None,
            },
            "criterion_2_cusum": {
                "meets": last_cusum == 1,
                "cusum_value": last_cusum,
            },
            "criterion_3_price_above_sma": {
                "meets": current_price_above_sma,
                "price": current_price,
                "sma": current_sma,
            },
            "meets_all_criteria": meets_all,
            "criteria_met_count": criteria_count
        }
    
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        return None


def filter_stocks(symbols: List[str], data_source: str = "engineered", 
                  sma_period: int = 10, slope_lookback: int = 1,
                  dry_run: bool = False, force_refresh: bool = False) -> Dict:
    """Filter stocks based on three criteria."""
    logger.info(f"Filtering {len(symbols)} stocks using {data_source} data")
    logger.info(f"SMA period={sma_period}, Slope lookback={slope_lookback}")
    
    all_results = []
    filtered_stocks = []
    kite = None
    
    for idx, symbol in enumerate(symbols, 1):
        logger.info(f"[{idx}/{len(symbols)}] Analyzing {symbol}...")
        
        # Load data
        if data_source == "engineered":
            df = load_engineered_data(symbol)
        elif data_source in ("kite", "kite-hourly") and KiteConnect:
            if not kite:
                from os import environ
                api_key = environ.get("KITE_API_KEY")
                access_token = environ.get("KITE_ACCESS_TOKEN")
                if api_key and access_token:
                    kite = KiteConnect(api_key=api_key)
                    kite.set_access_token(access_token)
            
            if data_source == "kite-hourly":
                df = load_or_fetch_kite_hourly(kite, symbol, force_refresh=force_refresh)
            else:
                df = fetch_kite_data(kite, symbol) if kite else None
        else:
            all_results.append({
                "symbol": symbol,
                "error": f"Data source '{data_source}' not available",
                "meets_all_criteria": False,
                "criteria_met_count": 0
            })
            continue
        
        if df is None:
            all_results.append({
                "symbol": symbol,
                "error": "No data available",
                "meets_all_criteria": False,
                "criteria_met_count": 0
            })
            continue
        
        # Analyze
        analysis = analyze_stock(df, symbol, sma_period=sma_period, slope_lookback=slope_lookback)
        
        if analysis is None:
            all_results.append({
                "symbol": symbol,
                "error": "Analysis failed",
                "meets_all_criteria": False,
                "criteria_met_count": 0
            })
            continue
        
        all_results.append(analysis)
        
        if analysis["meets_all_criteria"]:
            filtered_stocks.append(symbol)
            logger.info(f"  ✓ PASS")
        else:
            logger.debug(f"  ✗ ({analysis['criteria_met_count']}/3)")
    
    # Summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "data_source": data_source,
        "parameters": {"sma_period": sma_period, "slope_lookback": slope_lookback},
        "symbols_analyzed": len(symbols),
        "stocks_meeting_all_criteria": len(filtered_stocks),
        "filtered_symbols": filtered_stocks,
        "details": all_results,
        "pass_rate": f"{len(filtered_stocks) / len(symbols) * 100:.2f}%" if symbols else "0%"
    }
    
    # Save
    if not dry_run:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = RESULTS_DIR / f"filtered_stocks_{timestamp}.json"
        try:
            with open(output_file, "w") as f:
                json.dump(summary, f, indent=2, default=str)
            logger.info(f"Results saved: {output_file.name}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    return summary


def print_summary(summary: Dict):
    """Pretty print the filter summary."""
    print("\n" + "="*80)
    print("STOCK FILTER RESULTS".center(80))
    print("="*80)
    print(f"\nTimestamp: {summary['timestamp']}")
    print(f"Data Source: {summary['data_source']}")
    print(f"Stocks Analyzed: {summary['symbols_analyzed']}")
    print(f"Stocks Meeting All Criteria: {summary['stocks_meeting_all_criteria']}")
    print(f"Pass Rate: {summary['pass_rate']}")
    
    if summary['filtered_symbols']:
        print(f"\n✓ FILTERED STOCKS (Meeting all 3 criteria):")
        for symbol in summary['filtered_symbols']:
            print(f"  • {symbol}")
    else:
        print(f"\nNo stocks met all criteria.")
    
    print("\n" + "-"*80)
    print("DETAILED ANALYSIS".center(80))
    print("-"*80)
    
    slope_labels = {-1: "↓ NEG", 0: "→ FLAT", 1: "↑ POS"}
    
    for result in summary['details']:
        symbol = result['symbol']
        meets_all = "✓" if result.get('meets_all_criteria', False) else "✗"
        
        print(f"\n{meets_all} {symbol}")
        
        # Check if this is an error result
        if 'error' in result:
            print(f"   Status: ⚠ {result['error']}")
        else:
            price = result['current_price']
            sma = result['sma_10']
            criteria_met = result['criteria_met_count']
            
            print(f"   Price: {price:.2f} | SMA(10): {sma:.2f}")
            print(f"   Criteria Met: {criteria_met}/3")
            
            # Show criterion details
            c1 = "✓" if result['criterion_1_slope_change']['meets'] else "✗"
            c2 = "✓" if result['criterion_2_cusum']['meets'] else "✗"
            c3 = "✓" if result['criterion_3_price_above_sma']['meets'] else "✗"
            
            current_slope = result['criterion_1_slope_change'].get('current_slope', 0)
            current_slope_class = result['criterion_1_slope_change'].get('current_slope_classified', 0)
            prev_slope_class = result['criterion_1_slope_change'].get('previous_slope_classified', 0)
            
            slope_class_label = slope_labels.get(current_slope_class, "?")
            prev_slope_label = slope_labels.get(prev_slope_class, "?") if prev_slope_class else "?"
            
            print(f"   {c1} Slope change (-ve to +ve): {prev_slope_label} → {slope_class_label} (value: {current_slope:.6f})")
            print(f"   {c2} CUSUM signal: {result['criterion_2_cusum']['cusum_value']}")
            print(f"   {c3} Price > SMA: {result['criterion_3_price_above_sma']['price']:.2f} > {result['criterion_3_price_above_sma']['sma']:.2f}")
    
    print("\n" + "="*80 + "\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Stock Filter Agent")
    parser.add_argument("--symbols", type=str, help="Comma-separated symbols (e.g., INFY,TCS)")
    parser.add_argument("--top-n", type=int, help="Limit to top N stocks")
    parser.add_argument("--data-source", default="engineered", choices=["engineered", "kite", "kite-hourly"],
                        help="Data source: engineered (daily), kite (daily), or kite-hourly (hourly, 2yr)")
    parser.add_argument("--sma-period", type=int, default=10, help="SMA period (default: 10)")
    parser.add_argument("--slope-lookback", type=int, default=1, help="Slope lookback (default: 1)")
    parser.add_argument("--force-refresh", action="store_true", help="Refresh hourly data cache")
    parser.add_argument("--dry-run", action="store_true", help="Skip saving results")
    
    args = parser.parse_args()
    
    # Get symbols
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        symbols = get_all_stocks(top_n=args.top_n)
        if not symbols:
            logger.error("Failed to load stocks")
            return
    
    # Filter
    summary = filter_stocks(
        symbols=symbols,
        data_source=args.data_source,
        sma_period=args.sma_period,
        slope_lookback=args.slope_lookback,
        dry_run=args.dry_run,
        force_refresh=args.force_refresh
    )
    
    print_summary(summary)


if __name__ == "__main__":
    main()
