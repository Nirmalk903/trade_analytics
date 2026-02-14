import pandas as pd
import numpy as np
import requests
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from functools import lru_cache
from datetime import datetime as dt_time
from datetime import timedelta
from Options_Utility import atm_strike, tau
from Options_Utility import highlight_rows  # Placeholder for future implementation
from black_scholes_functions import *
from json import JSONDecodeError
import json
import time
import os
from quantlib_black_scholes import calculate_greeks
from pathlib import Path
import random

# Short list of realistic User-Agent strings to rotate
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:116.0) Gecko/20100101 Firefox/116.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1"
]

# Function to fetch options data from NSE website with retry logic

@lru_cache()
@retry(stop=stop_after_attempt(3), wait=wait_fixed(1), retry=retry_if_exception_type((requests.RequestException, ValueError)))
def fetch_options_data(symbol, timeout=10):
    """Fetch raw JSON from NSE option-chain endpoint and return a Python dict."""
    symbol = symbol.upper()
    symbol_type = 'indices' if symbol in ['NIFTY', 'BANKNIFTY', 'MIDCPNIFTY', 'FINNIFTY'] else 'equities'
    url = f"https://www.nseindia.com/api/option-chain-{symbol_type}?symbol={symbol}"
    ua = random.choice(USER_AGENTS)
    headers = {
        'User-Agent': ua,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.nseindia.com',
        'Sec-Fetch-Site': 'same-origin',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Dest': 'document',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache',
        'sec-ch-ua': '"Chromium";v="116", "Not)A;Brand";v="24"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"'
    }
    session = requests.Session()
    session.headers.update(headers)
    # First request to populate cookies / prime server
    resp = session.get(url, timeout=timeout)
    resp.raise_for_status()
    # short delay to reduce server-side rate-limit triggers
    time.sleep(0.5)
    # Use cookies from initial request
    resp2 = session.get(url, cookies=resp.cookies.get_dict(), timeout=timeout)
    # Save raw response for debugging
    try:
        raw_dir = Path('./OptionChainRaw')
        raw_dir.mkdir(parents=True, exist_ok=True)
        ts = dt_time.now().strftime('%Y%m%d_%H%M%S')
        status = resp2.status_code
        raw_file = raw_dir / f"{symbol}_resp_{ts}_{status}.json"
        raw_file.write_text(resp2.text)
    except Exception:
        pass
    resp2.raise_for_status()
    try:
        return resp2.json()
    except ValueError as e:
        raise ValueError(f"Invalid JSON received from {url}: {e}")


# Alternative function to fetch options data from NSE Website

@lru_cache()
@retry(stop=stop_after_attempt(3), wait=wait_fixed(1), retry=retry_if_exception_type((requests.RequestException, ValueError)))
def fetch_live_options_data(symbol, timeout=10):
    """Fetch and return raw JSON dict from NSE with limited retries and timeouts."""
    symbol = symbol.upper()
    symbol_type = 'indices' if symbol in ['NIFTY', 'BANKNIFTY', 'MIDCPNIFTY', 'FINNIFTY'] else 'equities'
    url = f"https://www.nseindia.com/api/option-chain-{symbol_type}?symbol={symbol}"

    ua = random.choice(USER_AGENTS)
    headers = {
        'User-Agent': ua,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.nseindia.com',
        'Sec-Fetch-Site': 'same-origin',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Dest': 'document',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache',
        'sec-ch-ua': '"Chromium";v="116", "Not)A;Brand";v="24"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"'
    }

    session = requests.Session()
    session.headers.update(headers)
    # make an initial request to get cookies, then request the JSON
    resp1 = session.get(url, timeout=timeout)
    resp1.raise_for_status()
    # small pause to avoid immediate rate-limit rejection
    time.sleep(0.5)
    resp2 = session.get(url, cookies=resp1.cookies.get_dict(), timeout=timeout)
    # Save raw response for debugging
    try:
        raw_dir = Path('./OptionChainRaw')
        raw_dir.mkdir(parents=True, exist_ok=True)
        ts = dt_time.now().strftime('%Y%m%d_%H%M%S')
        status = resp2.status_code
        raw_file = raw_dir / f"{symbol}_resp_{ts}_{status}.json"
        raw_file.write_text(resp2.text)
    except Exception:
        pass
    resp2.raise_for_status()
    try:
        return resp2.json()
    except ValueError as e:
        raise ValueError(f"Invalid JSON received from {url}: {e}")


def fetch_and_save_options_chain(symbol):
    symbol = symbol.upper()
    print(f'printing option chain for {symbol}')
    try:
        data = fetch_live_options_data(symbol)
        if not data or not isinstance(data, dict):
            print(f"Warning: No data returned for {symbol}. Skipping.")
            return f"No data for {symbol}"

        records = data.get('records', {})
        expiry_dates = records.get('expiryDates', [])
        raw_data = records.get('data', [])

        if not expiry_dates or not isinstance(expiry_dates, (list, tuple)):
            print(f"Warning: No expiryDates found for {symbol}. Skipping.")
            return f"Invalid data for {symbol}"

        dates = pd.to_datetime(expiry_dates, format='%d-%b-%Y', errors='coerce')
        if dates.empty:
            print(f"Warning: Could not parse expiryDates for {symbol}. Skipping.")
            return f"Invalid data for {symbol}"

        max_expiry = dates.iloc[0] + timedelta(days=90)
        expiry_list = [d.strftime('%d-%b-%Y') for d in dates if dates.iloc[0] <= d <= max_expiry]

        ls = []
        spot_price = records.get('underlyingValue', None)

        for dt in expiry_list:
            # filter raw_data (list of dicts) for matching expiry
            filtered = [r for r in raw_data if r.get('expiryDate') == dt]
            for row in filtered:
                call = row.get('CE') or {}
                put = row.get('PE') or {}

                stp = row.get('strikePrice')

                calloi = call.get('openInterest', 0) if isinstance(call, dict) else 0
                callcoi = call.get('changeinOpenInterest', 0) if isinstance(call, dict) else 0
                cltp = call.get('lastPrice', 0) if isinstance(call, dict) else 0

                putoi = put.get('openInterest', 0) if isinstance(put, dict) else 0
                putcoi = put.get('changeinOpenInterest', 0) if isinstance(put, dict) else 0
                pltp = put.get('lastPrice', 0) if isinstance(put, dict) else 0

                optdata = {
                    'Expiry': dt,
                    'call_oi': calloi,
                    'call_change_oi': callcoi,
                    'call_ltp': cltp,
                    'strike_price': stp,
                    'put_ltp': pltp,
                    'put_oi': putoi,
                    'put_change_oi': putcoi,
                    'spot_price': spot_price
                }

                ls.append(optdata)

        if not ls:
            print(f"No option rows extracted for {symbol}.")
            return f"No option rows for {symbol}"

        OptionChain = pd.DataFrame(ls)
        new_dir = f'./OptionChainJSON'
        os.makedirs(new_dir, exist_ok=True)
        file_path = os.path.join(new_dir, f'{symbol}_OptionChain.json')
        OptionChain.to_json(file_path, orient='records')
        return f'Option Chain Saved'
    except Exception as e:
        print(f"Error fetching/saving option chain for {symbol}: {e}")
        return f"Error for {symbol}"



def apply_greeks(row, option_type='call'):
    option_price = row['call_ltp'] if option_type == 'call' else row['put_ltp']
    if pd.isna(option_price) or option_price == 0:
        return {
        'delta': 0,
        'gamma': 0,
        'vega': 0,
        'theta': 0,
        'rho': 0,
        'IV': 0}
    try:
        return calculate_greeks(
            option_price=option_price,
            spot_price=row['spot_price'],
            strike_price=row['strike_price'],
            risk_free_rate=row['rate'],
            time_to_expiry=row['tau'],
            option_type=option_type
        )  
    except Exception as e:
        # print(f"Error calculating IV for row {row}: {e}")
        return {
        'delta': 0,
        'gamma': 0,
        'vega': 0,
        'theta': 0,
        'rho': 0,
        'IV': 0}  # Return 0 if there's an error in calculation
            


# Function to enrich option chain with additional data

def enrich_option_chain(symbol):
    symbol = symbol.upper()
    print(f'Enriching option chain for {symbol}')
    file_name = f'{symbol}_OptionChain.json'
    file_path = os.path.join('./OptionChainJSON', file_name)
    chain = pd.read_json(file_path, orient='records')
    chain['Expiry'] = pd.to_datetime(chain['Expiry'], format='%d-%b-%Y', errors='coerce')
    chain['tau'] = chain['Expiry'].apply(lambda x: tau(x))
    chain['Expiry'] = chain['Expiry'].dt.strftime('%d-%b-%Y')
    chain['expiry_days'] = chain['tau'].apply(lambda x: int(x * 365))
    chain['rate'] = 0.1
    atm_strike_price = atm_strike(chain['spot_price'].iloc[0], chain)
    chain['atm_strike_price'] = atm_strike_price
    chain['is_atm_strike'] = chain['strike_price'].apply(lambda x: "Y" if x == atm_strike_price else "N")
    chain['call_iv'] = chain.apply(lambda row: apply_greeks(row, option_type='call').get('IV'), axis=1)
    chain['put_iv'] = chain.apply(lambda row: apply_greeks(row, option_type='put').get('IV'), axis=1)
    chain['call_delta'] = chain.apply(lambda row: apply_greeks(row, option_type='call').get('delta'), axis=1)
    chain['put_delta'] = chain.apply(lambda row: apply_greeks(row, option_type='put').get('delta'), axis=1)
    chain['gamma'] = chain.apply(lambda row: apply_greeks(row, option_type='call').get('gamma'), axis=1)
    chain['vega'] = chain.apply(lambda row: apply_greeks(row, option_type='call').get('vega'), axis=1)
    
    # Writing enriched option chain to JSON file
    new_dir = f'./OptionChainJSON_Enriched'
    os.makedirs(new_dir, exist_ok=True)
    file_path = os.path.join(new_dir, f'{symbol}_OptionChain_Enriched.json')
    chain.to_json(file_path,orient='records')
    
    # Writing ATM data to JSON file
    atm_dir = f'./ATM_OptionChainJSON'
    os.makedirs(atm_dir, exist_ok=True)
    atm_file_path = os.path.join(atm_dir, f'{symbol}_ATM_OptionChain.json')
    chain_v = chain.copy()
    chain_v['timestamp'] = dt_time.now().strftime('%Y-%m-%d %H:%M')
    if os.path.exists(atm_file_path):
        existing_data = pd.read_json(atm_file_path, orient='records')
    else:
        existing_data = pd.DataFrame(columns=chain_v.columns)
    new_data = chain_v.query('is_atm_strike == "Y"')
    updated_data = pd.concat([existing_data, new_data]).drop_duplicates(keep='last',inplace = False)
    updated_data = updated_data[~updated_data.index.duplicated(keep='last')]
    updated_data.to_json(atm_file_path, orient='records')
    print(f"ATM data for {symbol} saved successfully.")
    
    return None


def load_enriched_option_chain(symbol):        
    symbol = symbol.upper()
    print(f'Loading option chain for {symbol}')
    file_name = f'{symbol}_OptionChain_Enriched.json'
    file_path = os.path.join('./OptionChainJSON_Enriched', file_name)
    chain = pd.read_json(file_path, orient='records')
    return chain


def load_atm_chain(symbol):
    symbol = symbol.upper()
    print(f'Loading ATM data for {symbol}')
    file_name = f'{symbol}_ATM_OptionChain.json'
    file_path = os.path.join('./ATM_OptionChainJSON', file_name)
    chain = pd.read_json(file_path, orient='records')
    print(f"ATM data for {symbol} loaded successfully.")
    return chain
