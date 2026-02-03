"""
Automated Trading Agent using CUSUM Filter
- Fetches hourly price data from Kite
- Runs CUSUM filter to generate signals
- Enters Bull Call Spread on +1, Bear Put Spread on -1
- Uses option_strategy_agent to execute trades
"""

import os
import logging
from datetime import datetime, timedelta
from kiteconnect import KiteConnect
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import json
from pathlib import Path
from algorithms.cusum_filter import getTEvents
from option_strategy_agent import run_agent

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SESSION_PATH = os.environ.get("KITE_SESSION_PATH", os.path.expanduser("~/.kite_session.json"))
API_KEY = os.environ.get("KITE_API_KEY")

# --- Kite Helpers ---
def load_access_token():
    token = os.environ.get("KITE_ACCESS_TOKEN")
    if token:
        return token
    if not os.path.exists(SESSION_PATH):
        logger.error(f"Session file not found: {SESSION_PATH}")
        return None
    try:
        with open(SESSION_PATH, "r", encoding="utf8") as f:
            data = json.load(f)
        return data.get("access_token")
    except Exception as e:
        logger.error(f"Error reading session file: {e}")
        return None

def init_kite_client():
    api_key = API_KEY
    access_token = load_access_token()
    if not api_key or not access_token:
        raise SystemExit("KITE_API_KEY and valid access token required (run kite.py first)")
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite

def fetch_hourly_data(kite, symbol, days=5):
    # Get instrument token
    instruments = kite.instruments(exchange="NSE")
    token = None
    for inst in instruments:
        if inst.get("tradingsymbol", "").upper() == symbol.upper() and inst.get("segment", "") == "NSE":
            token = inst.get("instrument_token")
            break
    if token is None:
        logger.error(f"Instrument token not found for {symbol}")
        return None
    end = datetime.now()
    start = end - timedelta(days=days)
    data = kite.historical_data(token, start, end, interval="hour")
    df = pd.DataFrame(data)
    df["date_time"] = pd.to_datetime(df["date"])
    df.set_index("date_time", inplace=True)
    return df

def run_cusum_strategy(symbol, expiry, lot_size=50, dry_run=True):
    kite = init_kite_client()
    df = fetch_hourly_data(kite, symbol)
    if df is None or df.empty:
        logger.error("No hourly data fetched.")
        return
    close_series = df["close"]
    # Use robust threshold
    h = max(1e-6, float(abs(close_series).median() * 0.1))
    t_events, diag = getTEvents(close_series, h)
    # Only use last signal
    last_trigger = diag["trigger"].iloc[-1] if not diag.empty else 0
    logger.info(f"CUSUM last trigger: {last_trigger}")
    instructions = None
    if last_trigger == 1:
        # Bull Call Spread
        instructions = [
            {"type": "BUY", "option_type": "CE", "strike": "ATM", "qty": lot_size},
            {"type": "SELL", "option_type": "CE", "strike": "ATM+200", "qty": lot_size}
        ]
    elif last_trigger == -1:
        # Bear Put Spread
        instructions = [
            {"type": "BUY", "option_type": "PE", "strike": "ATM", "qty": lot_size},
            {"type": "SELL", "option_type": "PE", "strike": "ATM-200", "qty": lot_size}
        ]
    else:
        logger.info("No actionable CUSUM signal.")
        return
    logger.info(f"Executing strategy: {instructions}")
    run_agent(symbol, instructions, expiry=expiry, monitor=True, dry_run=dry_run)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Automated CUSUM Trading Agent")
    parser.add_argument("--symbol", required=True, help="Underlying symbol (e.g. NIFTY)")
    parser.add_argument("--expiry", required=True, help="Expiry (e.g. 09JAN2026)")
    parser.add_argument("--lot-size", type=int, default=50, help="Lot size")
    parser.add_argument("--dry-run", action="store_true", help="Simulate orders without placing them")
    args = parser.parse_args()
    run_cusum_strategy(args.symbol, args.expiry, lot_size=args.lot_size, dry_run=args.dry_run)
