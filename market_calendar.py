import investpy #used for importing global economic calendar
import pandas as pd
import numpy as np
import pendulum as pm
import json
import yfinance as yf
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Regional mapping for economic calendar
REGIONS = {
    'Americas': ['USD', 'CAD', 'BRL', 'MXN'],
    'Europe': ['EUR', 'GBP', 'CHF', 'SEK', 'NOK', 'DKK'],
    'Asia': {
        'India': ['INR'],
        'China': ['CNY', 'CNH'],
        'Japan': ['JPY']
    }
}

# Region-specific importance level defaults
REGION_IMPORTANCE_DEFAULTS = {
    'Americas': ['high'],
    'Europe': ['high'],
    'India': ['high', 'medium', 'low'],  # Keep all importance levels for India
    'China': ['high'],
    'Japan': ['high']
}


def economic_calendar_by_region(regions=['Americas', 'Europe', 'India', 'China', 'Japan'], 
                                 importance_levels=None, 
                                 days_back=1, 
                                 days_forward=7):
    """
    Fetch economic calendar for multiple regions with region-specific importance defaults.
    
    Parameters:
    -----------
    regions : list
        List of regions/countries: 'Americas', 'Europe', 'India', 'China', 'Japan'
    importance_levels : dict or list, optional
        If dict: {'Americas': ['high'], 'India': ['high', 'medium'], ...}
        If list: Applied to all regions
        If None: Uses region-specific defaults
    days_back : int
        Days to look back from today
    days_forward : int
        Days to look forward from today
        
    Returns:
    --------
    dict : Dictionary with region-wise DataFrames
    """
    from_date = pm.now().subtract(days=days_back).strftime('%d/%m/%Y')
    to_date = pm.now().add(days=days_forward).strftime('%d/%m/%Y')
    
    # Set importance levels: use region-specific defaults if not provided
    if importance_levels is None:
        region_importance = REGION_IMPORTANCE_DEFAULTS.copy()
    elif isinstance(importance_levels, dict):
        # Merge with defaults for regions not specified
        region_importance = REGION_IMPORTANCE_DEFAULTS.copy()
        region_importance.update(importance_levels)
    else:
        # Apply same importance levels to all regions
        region_importance = {region: importance_levels for region in regions}
    
    logger.info(f"Fetching economic calendar from {from_date} to {to_date}")
    logger.info(f"Region-specific importance levels: {region_importance}")
    
    try:
        # Fetch full calendar with retry
        max_retries = 2
        calendar = None
        for attempt in range(max_retries):
            try:
                calendar = investpy.news.economic_calendar(time_zone=None, from_date=from_date, to_date=to_date)
                break
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)[:100]}")
                if attempt == max_retries - 1:
                    logger.error(f"Failed to fetch economic calendar after {max_retries} retries")
                    raise
                import time
                time.sleep(1)  # Wait before retry
        
        if calendar is None or calendar.empty:
            logger.warning("No economic calendar data returned")
            return {}
        calendar = calendar.drop(columns=['id', 'time'], errors='ignore')
        
        # Collect all currencies for the selected regions
        all_currencies = []
        for region in regions:
            if region == 'Americas':
                all_currencies.extend(REGIONS['Americas'])
            elif region == 'Europe':
                all_currencies.extend(REGIONS['Europe'])
            elif region == 'India':
                all_currencies.extend(REGIONS['Asia']['India'])
            elif region == 'China':
                all_currencies.extend(REGIONS['Asia']['China'])
            elif region == 'Japan':
                all_currencies.extend(REGIONS['Asia']['Japan'])
        
        all_currencies = list(set(all_currencies))  # Remove duplicates
        
        # Collect all importance levels needed
        all_importance_levels = list(set([item for sublist in region_importance.values() for item in sublist]))
        
        # Filter by importance and currencies (handle None importance values)
        df_filtered = calendar.copy()
        df_filtered = df_filtered[df_filtered['importance'].notna()]  # Remove None values
        df_filtered = df_filtered[df_filtered['currency'].notna()]  # Remove None currencies
        df = df_filtered.query(
            "importance in @all_importance_levels & (currency in @all_currencies or zone=='india')"
        ).reset_index(drop=True)
        
        # Format the dataframe - but keep lowercase for query operations
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.title()
        df['currency'] = df['currency'].str.upper()
        
        # Split by region with region-specific importance filtering
        region_dfs = {}
        for region in regions:
            region_importance_list = region_importance.get(region, ['high'])
            # Convert to title case for comparison
            region_importance_list_title = [x.title() for x in region_importance_list]
            
            if region == 'Americas':
                currencies = REGIONS['Americas']
                region_df = df[df['currency'].isin(currencies) & df['importance'].isin(region_importance_list_title)].copy()
            elif region == 'Europe':
                currencies = REGIONS['Europe']
                region_df = df[df['currency'].isin(currencies) & df['importance'].isin(region_importance_list_title)].copy()
            elif region == 'India':
                currencies = REGIONS['Asia']['India']
                region_df = df[((df['currency'].isin(currencies) | df['zone'].str.title() == 'India')) & df['importance'].isin(region_importance_list_title)].copy()
            elif region == 'China':
                currencies = REGIONS['Asia']['China']
                region_df = df[df['currency'].isin(currencies) & df['importance'].isin(region_importance_list_title)].copy()
            elif region == 'Japan':
                currencies = REGIONS['Asia']['Japan']
                region_df = df[df['currency'].isin(currencies) & df['importance'].isin(region_importance_list_title)].copy()
            else:
                region_df = pd.DataFrame()
            
            region_dfs[region] = region_df.sort_values('date').reset_index(drop=True)
        
        # Format column names for output
        for region_df in region_dfs.values():
            if not region_df.empty:
                region_df.columns = [col.title() for col in region_df.columns]
        
        # Create a combined view with only high importance events for consistency
        combined_importance = ['high']
        region_dfs['All_Regions'] = df.query("Importance in @combined_importance").sort_values('Date').reset_index(drop=True)
        
        logger.info(f"Successfully fetched calendar for {len(regions)} regions")
        return region_dfs
        
    except Exception as e:
        logger.error(f"Error fetching economic calendar: {e}")
        return {}


def economic_calendar():
    """Legacy function for backward compatibility"""
    from_date = pm.now().subtract(days=1).strftime('%d/%m/%Y')
    to_date = pm.now().add(days=3).strftime('%d/%m/%Y')
    
    try:
        calendar = investpy.news.economic_calendar(time_zone=None, from_date=from_date, to_date=to_date)
        calendar = calendar.drop(columns=['id', 'time'], errors='ignore')
        xccy = ['USD', 'GBP', 'EUR', 'INR', 'JPY', 'CNH']
        importance_level = ['high']
        df = calendar.query("importance in @importance_level & currency in @xccy or zone=='india'").reset_index(drop=True)
        
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.title()
            df['currency'] = df['currency'].str.upper()
        df.columns = [col.title() for col in df.columns]
        
        return df
    except Exception as e:
        logger.error(f"Error in economic_calendar: {e}")
        return pd.DataFrame()


def print_regional_calendar(regions=['Americas', 'Europe', 'India', 'China', 'Japan'], 
                           importance_levels=None, 
                           days_forward=7):
    """
    Fetch and print economic calendar for specified regions.
    Uses region-specific importance defaults if not specified.
    """
    calendars = economic_calendar_by_region(
        regions=regions, 
        importance_levels=importance_levels,
        days_forward=days_forward
    )
    
    if not calendars:
        print("Failed to fetch economic calendars")
        return
    
    print("\n" + "="*100)
    print("GLOBAL ECONOMIC CALENDAR".center(100))
    print("="*100)
    
    # Get actual importance levels used
    if importance_levels is None:
        actual_importance = REGION_IMPORTANCE_DEFAULTS
    elif isinstance(importance_levels, dict):
        actual_importance = REGION_IMPORTANCE_DEFAULTS.copy()
        actual_importance.update(importance_levels)
    else:
        actual_importance = {region: importance_levels for region in regions}
    
    for region, df in calendars.items():
        if region == 'All_Regions':
            continue
            
        region_importance = actual_importance.get(region, ['high'])
        print(f"\n{'─'*100}")
        print(f"📊 {region.upper()}".center(100))
        print(f"Importance: {', '.join(region_importance).title()}".center(100))
        print(f"{'─'*100}")
        
        if df.empty:
            print(f"  No {region_importance} importance events scheduled for {region}")
        else:
            print(f"\n{len(df)} events scheduled:\n")
            # Display key columns
            display_cols = ['Date', 'Zone', 'Currency', 'Event', 'Importance', 'Actual', 'Forecast', 'Previous']
            available_cols = [col for col in display_cols if col in df.columns]
            print(df[available_cols].to_string(index=False))
    
    print("\n" + "="*100 + "\n")
    return calendars


def save_calendar_to_csv(calendars, output_dir='results/economic_calendar'):
    """
    Save regional calendars to CSV files.
    
    Parameters:
    -----------
    calendars : dict
        Dictionary of DataFrames from economic_calendar_by_region()
    output_dir : str
        Directory to save CSV files
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = pm.now().format('YYYY-MM-DD_HHmmss')
    
    saved_files = []
    for region, df in calendars.items():
        if not df.empty:
            filename = f"{output_dir}/{region}_calendar_{timestamp}.csv"
            df.to_csv(filename, index=False)
            saved_files.append(filename)
            logger.info(f"Saved {region} calendar to {filename}")
    
    return saved_files
# The below function fetches the earnings calendar for a list of stock tickers using the yfinance library.


def stock_earnings_calendar(tickers):
    from_date = pm.now().subtract(days=1)  # .strftime('%d/%m/%Y')
    to_date = pm.now().add(days=120)  # .strftime('%d/%m/%Y')
    yf_tickers = ['^NSEI' if ticker == 'NIFTY' else '^NSEBANK' if ticker == 'BANKNIFTY' else f'{ticker}.NS' for ticker in tickers]

    ls = []
    for ticker in yf_tickers:
        # print(f"Fetching earnings calendar for {ticker} from {from_date} to {to_date}")
        try:
            # Fetch the stock data
            stock = yf.Ticker(ticker)
            earnings = stock.earnings_dates

            if earnings.empty:
                print(f"No earnings data available for {ticker}.")
                continue  # Skip to the next ticker

            # Format and clean up the DataFrame
            earnings.reset_index(inplace=True)
            earnings.columns = ['Date', 'EPS Estimate', 'Reported EPS', 'Surprise (%)',"A"]
            earnings['Date'] = pd.to_datetime(earnings['Date'], errors='coerce')
            earnings = earnings.query("Date >= @from_date & Date <= @to_date").reset_index(drop=True)
            earnings = earnings.sort_values(by='Date').reset_index(drop=True)
            earnings['Ticker'] = ticker  # Add the ticker symbol to the DataFrame
            ls.append(earnings)
        except Exception as e:
            print(f"Error fetching earnings calendar for {ticker}: {e}")
            continue  # Skip to the next ticker

    if not ls:
        print("No earnings data found for the provided tickers.")
        return pd.DataFrame()

    df = pd.concat(ls, ignore_index=True)
    df.set_index('Ticker', inplace=True)
    df.sort_values(by='Date', inplace=True)
    
    return df

# Commented out - call explicitly when needed
# stock_earnings_calendar(['NIFTY', 'BANKNIFTY', 'RELIANCE', 'TCS', 'INFY'])

# The below code fetches the latest news articles related to Nifty50 stocks using the NewsAPI.
# It requires an API key from NewsAPI or a similar service. The code fetches the latest news articles and prints the title, source, publication date, and URL of each article.


import requests
import pandas as pd

def fetch_nifty50_news():
    # Define the query parameters
    api_key = "1b8de5003ddd4249bb3173cc8413a5dc"
    from_date = pd.to_datetime(pm.now().subtract(days=1)) # .strftime('%d/%m/%Y')
    to_date = pd.to_datetime(pm.now())
    url = "https://newsapi.org/v2/everything"
    query = "Nifty50 stocks OR Indian stock market OR Nifty50"
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "apiKey": api_key
    }

    # Make the API request
    response = requests.get(url, params=params)

    if response.status_code == 200:
        news_data = response.json()
        articles = news_data.get("articles", [])
        if articles:
            
            df = pd.DataFrame(articles)
            df = df[['title', 'publishedAt', 'source']]
            df['source'] = df['source'].apply(lambda x: x['name'])
            df['title'] = df['title'].str.replace('Stock market update:', '', regex=False).str.strip()
            df['title'] = df['title'].str.ljust(50)
            df['publishedAt'] = pd.to_datetime(df['publishedAt'], errors='coerce')
            df = df[0:10]  # Fetch top 5 articles
            
        else:
            print("No news articles found.")
    else:
        print(f"Failed to fetch news. HTTP Status Code: {response.status_code}")
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch economic calendar for major regions')
    parser.add_argument('--regions', nargs='+', 
                       default=['Americas', 'Europe', 'India', 'China', 'Japan'],
                       help='Regions to fetch: Americas, Europe, India, China, Japan')
    parser.add_argument('--importance', nargs='+', 
                       default=['high'],
                       choices=['high', 'medium', 'low'],
                       help='Event importance levels to include')
    parser.add_argument('--days', type=int, default=7,
                       help='Number of days to look forward')
    parser.add_argument('--save', action='store_true',
                       help='Save results to CSV files')
    parser.add_argument('--output-dir', default='results/economic_calendar',
                       help='Output directory for CSV files')
    
    args = parser.parse_args()
    
    # Fetch and display calendar
    print("\n" + "="*100)
    print("UPCOMING ECONOMIC EVENTS - MAJOR REGIONS".center(100))
    print("="*100)
    print(f"Regions: {', '.join(args.regions)}")
    print(f"Importance: {', '.join(args.importance)}")
    print(f"Period: Next {args.days} days")
    print("="*100)
    
    calendars = print_regional_calendar(
        regions=args.regions,
        importance_levels=args.importance,
        days_forward=args.days
    )
    
    # Save to CSV if requested
    if args.save and calendars:
        saved_files = save_calendar_to_csv(calendars, output_dir=args.output_dir)
        print(f"\n✓ Saved {len(saved_files)} calendar files:")
        for file in saved_files:
            print(f"  • {file}")
    
    # Usage examples in comments:
    # python market_calendar.py
    # python market_calendar.py --regions Americas Europe India
    # python market_calendar.py --importance high medium --days 14
    # python market_calendar.py --save --output-dir my_calendar_data