import pandas as pd
import numpy as np
from Options_Utility import highlight_rows  # Placeholder for future implementation
from black_scholes_functions import *
import os
from get_options_data import fetch_and_save_options_chain, enrich_option_chain,load_enriched_option_chain, load_atm_chain
from data_download_vbt import getdata_vbt, get_underlying_data_vbt, get_symbols,  get_dates_from_most_active_files
from feature_engineering import add_features
import matplotlib.pyplot as plt
from plotting import plot_garch_vs_rsi, plot_garch_vs_avg_iv
import time

# dt = datetime.now()


symbols = get_symbols(get_dates_from_most_active_files()[-1],top_n=17)[0]
prev_symbols = get_symbols(get_dates_from_most_active_files()[-2],top_n=17)[0]
new_symbols = set(symbols) - set(prev_symbols)
symbol_excluded = set(prev_symbols) - set(symbols)

# symbols = ['M&M']

for _ in new_symbols:
    print(f"New symbol added: {_}") 
    
for _ in symbol_excluded:
    print(f"Symbol excluded: {_}")



def create_analytics(symbols):
    for symbol in symbols:
        try:
            get_underlying_data_vbt([symbol], period='10y', interval='1d')
            add_features([symbol])
            # Fetch and save options chain data with timeout
            start_time = time.time()
            try:
                fetch_and_save_options_chain(symbol)
            except Exception as e:
                print(f"Error fetching option chain for {symbol}: {e}")
            elapsed = time.time() - start_time
            timeout = 120  # seconds
            if elapsed > timeout:  # 120 seconds timeout
                print(f"Skipping {symbol} option chain: took more than {timeout} seconds.")
                continue
            
            enrich_option_chain(symbol)
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")

    return "Analytics created successfully for all symbols."

create_analytics(symbols)

# PLOTS *************************************
plot_garch_vs_rsi(symbols)
plot_garch_vs_avg_iv([s for s in symbols if s != 'ETERNAL'], options_expiry='26-Jun-2025')

symbols
