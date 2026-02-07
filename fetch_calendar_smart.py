#!/usr/bin/env python3
"""
Fetch economic calendar with automatic fallback to multiple sources.

Usage:
    python3 fetch_calendar_smart.py
    python3 fetch_calendar_smart.py --regions Americas Europe India
    python3 fetch_calendar_smart.py --days 14 --save
"""

import logging
import argparse
from market_calendar import economic_calendar_with_fallback, print_regional_calendar

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Fetch economic calendar with intelligent fallback')
    parser.add_argument('--regions', nargs='+', 
                       default=['Americas', 'Europe', 'India', 'China', 'Japan'],
                       help='Regions to fetch')
    parser.add_argument('--days', type=int, default=7,
                       help='Number of days to look forward')
    parser.add_argument('--save', action='store_true',
                       help='Save results to CSV files')
    
    args = parser.parse_args()
    
    print("\n" + "="*100)
    print("ECONOMIC CALENDAR - MULTI-SOURCE FALLBACK".center(100))
    print("="*100)
    print(f"Regions: {', '.join(args.regions)}")
    print(f"Period: Next {args.days} days")
    print("="*100 + "\n")
    
    # Fetch with fallback
    logger.info("Starting economic calendar fetch with automatic fallback...")
    calendars = economic_calendar_with_fallback(
        regions=args.regions,
        days_forward=args.days
    )
    
    if not calendars:
        print("✗ Failed to fetch economic calendar from all sources")
        print("\nTo enable additional sources, configure:")
        print("  • Trading Economics API: export TRADING_ECONOMICS_KEY='your_key'")
        print("  • FRED API: export FRED_API_KEY='your_key'")
        print("  • Investing.com: No key needed (web scraping)")
        return
    
    # Display results
    total_events = sum(len(df) for df in calendars.values() if not df.empty)
    print(f"✓ Fetched {total_events} economic events from available sources\n")
    
    # Show details by region
    for region, df in calendars.items():
        if not df.empty:
            print(f"  • {region}: {len(df)} events")
    
    # Save if requested
    if args.save:
        from pathlib import Path
        import pandas as pd
        import pendulum as pm
        
        output_dir = 'results/economic_calendar'
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for region, df in calendars.items():
            if not df.empty:
                filename = f"{output_dir}/{region}_calendar.csv"
                df.to_csv(filename, index=False)
                logger.info(f"Saved {region} to {filename}")
    
    print("\n" + "="*100 + "\n")


if __name__ == "__main__":
    main()
