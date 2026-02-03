import pandas as pd
import numpy as np
import requests
from tenacity import retry, stop_after_attempt, stop_after_delay, wait_fixed, wait_random, retry
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

# Function to fetch options data from NSE website with retry logic

@lru_cache()
@retry(wait=wait_random(min=0.1, max=1))
def fetch_options_data(symbol):
    symbol = symbol.upper()
    symbol_type = 'indices' if symbol in ['NIFTY','BANKNIFTY','MIDCPNIFTY','FINNIFTY'] else 'equities'
    url = f"https://www.nseindia.com/api/option-chain-{symbol_type}?symbol={symbol}"
    headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36',
                'accept-encoding': 'gzip, deflate, br, zstd',
                'accept-language': 'en-US,en;q=0.9',
                'accept': '*/*'}
    session = requests.Session()
    request = session.get(url, headers=headers)
    print(request.status_code)
    response = session.get(url, headers=headers, cookies=dict(request.cookies))
    
    return pd.DataFrame(response.json())


# Alternative function to fetch options data from NSE Website

@lru_cache()
def fetch_live_options_data(symbol):
    symbol = symbol.upper()
    symbol_type = 'indices' if symbol in ['NIFTY', 'BANKNIFTY', 'MIDCPNIFTY', 'FINNIFTY'] else 'equities'
    url = f"https://www.nseindia.com/api/option-chain-{symbol_type}?symbol={symbol}"
    
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36',
        'accept-encoding': 'gzip, deflate, br, zstd',
        'accept-language': 'en-US,en;q=0.9',
        'accept': '*/*'
    }
                
    session = requests.Session()
    while True:
        request = session.get(url, headers=headers)
        if request.status_code == 200:
            cookies = dict(request.cookies)
            break
        else:
            time.sleep(0.5)  # Wait for 0.5 seconds before retrying

    while True:
        response = session.get(url, headers=headers, cookies=cookies)
        if response.status_code == 200:
            print(f"Success! Status code 200 received. {symbol} saved")
            break
        else:
            time.sleep(0.5)

    df = pd.DataFrame(response.json())
    return df


def fetch_and_save_options_chain(symbol):
    symbol = symbol.upper()
    print(f'printing option chain for {symbol}')
    try:
        data = fetch_live_options_data(symbol)
        # Check if data is empty or not a DataFrame
        if data is None or data.empty:
            print(f"Warning: No data returned for {symbol}. Skipping.")
            return f"No data for {symbol}"
        # Check for required keys before proceeding
        if 'records' not in data or 'expiryDates' not in data.loc[:, 'records']:
            print(f"Warning: Invalid data structure for {symbol}. Skipping.")
            return f"Invalid data for {symbol}"

        dates = pd.to_datetime(data.loc['expiryDates', 'records'], format='%d-%b-%Y')
        max_expiry = dates[0] + timedelta(days=90)
        expiry = [i.strftime('%d-%b-%Y') for i in dates if dates[0] <= i <= max_expiry]

        ls = []
        for dt in expiry:
            rawoptions = pd.DataFrame(data['records']['data']).fillna(0)
            rawoptions = rawoptions[rawoptions['expiryDate'] == dt].reset_index(drop=True)

            for i in range(len(rawoptions)):
                calloi = callcoi = cltp = putoi = putcoi = pltp = iv = 0
                stp = rawoptions['strikePrice'][i]
                if rawoptions['CE'][i] == 0:
                    calloi = callcoi = 0
                else:
                    calloi = rawoptions['CE'][i]['openInterest']
                    callcoi = rawoptions['CE'][i]['changeinOpenInterest']
                    cltp = rawoptions['CE'][i]['lastPrice']
                    civ = rawoptions['CE'][i]['impliedVolatility']

                if rawoptions['PE'][i] == 0:
                    putoi = putcoi = 0
                else:
                    putoi = rawoptions['PE'][i]['openInterest']
                    putcoi = rawoptions['PE'][i]['changeinOpenInterest']
                    pltp = rawoptions['PE'][i]['lastPrice']
                    piv = rawoptions['PE'][i]['impliedVolatility']
                    expiry = rawoptions['PE'][i]['expiryDate']

                optdata = {
                    'Expiry': expiry,
                    'call_oi': calloi,
                    'call_change_oi': callcoi,
                    'call_ltp': cltp,
                    'strike_price': stp,
                    'put_ltp': pltp,
                    'put_oi': putoi,
                    'put_change_oi': putcoi,
                    'spot_price': data.loc['underlyingValue', 'records']
                }

                ls.append(optdata)
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
