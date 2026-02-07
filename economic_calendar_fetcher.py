#!/usr/bin/env python3
"""
Multi-source fallback approach for economic calendar data.
Tries multiple sources in order of preference for maximum reliability.

Sources (in priority order):
1. Trading Economics API (most comprehensive)
2. Direct web scraping from Investing.com
3. FRED API (US economic data)
4. Cached/local data fallback
"""

import pandas as pd
import logging
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import time

logger = logging.getLogger(__name__)

class EconomicCalendarFetcher:
    """Multi-source fallback fetcher for economic calendar data."""
    
    def __init__(self, trading_econ_key=None, fred_key=None):
        """
        Initialize with optional API keys.
        
        Parameters:
        -----------
        trading_econ_key : str
            Trading Economics API key (from env: TRADING_ECONOMICS_KEY)
        fred_key : str
            FRED API key (from env: FRED_API_KEY)
        """
        import os
        self.trading_econ_key = trading_econ_key or os.getenv('TRADING_ECONOMICS_KEY')
        self.fred_key = fred_key or os.getenv('FRED_API_KEY')
        
    def fetch(self, from_date=None, to_date=None, regions=None):
        """
        Fetch economic calendar with automatic fallback.
        
        Parameters:
        -----------
        from_date : str
            Start date (YYYY-MM-DD)
        to_date : str
            End date (YYYY-MM-DD)
        regions : list
            List of regions to fetch (Americas, Europe, India, China, Japan)
            
        Returns:
        --------
        dict : Dictionary with region-wise DataFrames or empty dict if all fail
        """
        if regions is None:
            regions = ['Americas', 'Europe', 'India', 'China', 'Japan']
        
        if from_date is None:
            from_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        if to_date is None:
            to_date = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
        
        logger.info(f"Attempting to fetch economic calendar from {from_date} to {to_date}")
        
        # Try sources in order
        sources = [
            ('Trading Economics API', self._fetch_trading_economics),
            ('Investing.com Web Scrape', self._fetch_investing_com),
            ('FRED API (US only)', self._fetch_fred),
        ]
        
        for source_name, fetch_func in sources:
            try:
                logger.info(f"Attempting: {source_name}")
                data = fetch_func(from_date, to_date, regions)
                if data and any(df is not None and not df.empty for df in data.values()):
                    logger.info(f"✓ Successfully fetched from {source_name}")
                    return data
                else:
                    logger.warning(f"No data returned from {source_name}")
            except Exception as e:
                logger.warning(f"Failed to fetch from {source_name}: {str(e)[:100]}")
                continue
        
        logger.error("All sources failed to fetch economic calendar data")
        return {}
    
    def _fetch_trading_economics(self, from_date, to_date, regions):
        """
        Fetch from Trading Economics API.
        Requires API key from https://tradingeconomics.com/api/
        """
        if not self.trading_econ_key:
            logger.warning("Trading Economics API key not configured")
            return None
        
        try:
            url = "https://api.tradingeconomics.com/calendar"
            params = {
                'c': self.trading_econ_key,
                'format': 'json'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if not isinstance(data, list):
                return None
            
            df = pd.DataFrame(data)
            
            # Filter by date
            df['Date'] = pd.to_datetime(df.get('Date', []), errors='coerce')
            df = df[(df['Date'] >= from_date) & (df['Date'] <= to_date)]
            
            # Organize by region
            region_dfs = self._organize_by_region(df, regions)
            return region_dfs if region_dfs else None
            
        except Exception as e:
            logger.debug(f"Trading Economics API error: {e}")
            return None
    
    def _fetch_investing_com(self, from_date, to_date, regions):
        """
        Direct web scraping from Investing.com economic calendar.
        More reliable than investpy library.
        """
        try:
            url = "https://www.investing.com/economic-calendar/"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find economic calendar table
            calendar_data = []
            rows = soup.find_all('tr', {'data-eventid': True})
            
            for row in rows:
                try:
                    cells = row.find_all('td')
                    if len(cells) >= 5:
                        event_data = {
                            'Date': cells[0].get_text(strip=True),
                            'Time': cells[1].get_text(strip=True),
                            'Country': cells[2].get_text(strip=True),
                            'Event': cells[3].get_text(strip=True),
                            'Importance': cells[4].get_text(strip=True),
                            'Forecast': cells[5].get_text(strip=True) if len(cells) > 5 else '',
                            'Previous': cells[6].get_text(strip=True) if len(cells) > 6 else '',
                        }
                        calendar_data.append(event_data)
                except:
                    continue
            
            if not calendar_data:
                return None
            
            df = pd.DataFrame(calendar_data)
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df[(df['Date'] >= from_date) & (df['Date'] <= to_date)]
            
            region_dfs = self._organize_by_region(df, regions)
            return region_dfs if region_dfs else None
            
        except Exception as e:
            logger.debug(f"Investing.com scraping error: {e}")
            return None
    
    def _fetch_fred(self, from_date, to_date, regions):
        """
        Fetch US economic data from FRED API.
        Excellent for US indicators. Requires API key from https://fred.stlouisfed.org/
        """
        if not self.fred_key or 'Americas' not in regions:
            return None
        
        try:
            import fredapi
            fred = fredapi.Fred(api_key=self.fred_key)
            
            # Key US economic indicators
            us_indicators = {
                'UNRATE': {'name': 'Unemployment Rate', 'importance': 'High'},
                'PAYEMS': {'name': 'Nonfarm Payrolls', 'importance': 'High'},
                'CPIAUCSL': {'name': 'CPI (All Urban Consumers)', 'importance': 'High'},
                'CPILFESL': {'name': 'Core CPI', 'importance': 'High'},
                'DCOILWTICO': {'name': 'Crude Oil Prices', 'importance': 'Medium'},
                'DEXUSEU': {'name': 'USD/EUR Exchange Rate', 'importance': 'Medium'},
            }
            
            from_dt = pd.to_datetime(from_date)
            to_dt = pd.to_datetime(to_date)
            
            calendar_data = []
            for series_id, info in us_indicators.items():
                try:
                    data = fred.get(series_id, observation_start=from_date, observation_end=to_date)
                    if not data.empty:
                        for date, value in data.items():
                            calendar_data.append({
                                'Date': date,
                                'Country': 'United States',
                                'Currency': 'USD',
                                'Event': info['name'],
                                'Importance': info['importance'],
                                'Actual': value,
                            })
                except:
                    continue
            
            if not calendar_data:
                return None
            
            df = pd.DataFrame(calendar_data)
            region_dfs = {'Americas': df}
            return region_dfs
            
        except ImportError:
            logger.warning("fredapi not installed. Install with: pip install fredapi")
            return None
        except Exception as e:
            logger.debug(f"FRED API error: {e}")
            return None
    
    def _organize_by_region(self, df, regions):
        """Organize DataFrame by region."""
        if df.empty:
            return None
        
        region_mapping = {
            'Americas': ['USD', 'CAD', 'BRL', 'MXN', 'United States', 'Canada', 'Brazil', 'Mexico'],
            'Europe': ['EUR', 'GBP', 'CHF', 'SEK', 'NOK', 'DKK', 'Eurozone', 'Germany', 'France', 'UK'],
            'India': ['INR', 'India'],
            'China': ['CNY', 'CNH', 'China'],
            'Japan': ['JPY', 'Japan'],
        }
        
        region_dfs = {}
        for region in regions:
            currencies = region_mapping.get(region, [])
            mask = df['Currency'].isin(currencies) | df['Country'].isin(currencies)
            region_df = df[mask].copy()
            region_dfs[region] = region_df if not region_df.empty else pd.DataFrame()
        
        return region_dfs


def get_economic_calendar(from_date=None, to_date=None, regions=None, 
                         trading_econ_key=None, fred_key=None):
    """
    Convenience function to fetch economic calendar with fallback.
    
    Parameters:
    -----------
    from_date : str, optional
        Start date (YYYY-MM-DD)
    to_date : str, optional
        End date (YYYY-MM-DD)
    regions : list, optional
        List of regions to fetch
    trading_econ_key : str, optional
        Trading Economics API key
    fred_key : str, optional
        FRED API key
        
    Returns:
    --------
    dict : Dictionary with region-wise DataFrames
    """
    fetcher = EconomicCalendarFetcher(trading_econ_key, fred_key)
    return fetcher.fetch(from_date, to_date, regions)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    data = get_economic_calendar()
    for region, df in data.items():
        if not df.empty:
            print(f"\n{region}: {len(df)} events")
            print(df.head())
