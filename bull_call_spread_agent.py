"""
Agent to execute a Bull Call Spread and manage risk as per strategy_position_manager.py logic.
- Places a Bull Call Spread (buy lower strike call, sell higher strike call)
- Sets targets and stop-loss as % of max profit/loss and total funds
- Uses KiteConnect for order placement and monitoring
- Logs results in results/kite/strategy_manager/
"""

import os
import logging
from pathlib import Path
from kiteconnect import KiteConnect
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
from Options_Utility import atm_strike
import json

# --- Config ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
def get_session_path():
    # 1. Check environment variable
    env_path = os.environ.get("KITE_SESSION_PATH")
    if env_path and os.path.exists(env_path):
        print(f"Using session file from KITE_SESSION_PATH: {env_path}")
        return env_path
    # 2. Try home directory
    home_path = os.path.expanduser("~/.kite_session.json")
    if os.path.exists(home_path):
        print(f"Using session file from home directory: {home_path}")
        return home_path
    # 3. Try current directory
    cwd_path = os.path.abspath(".kite_session.json")
    if os.path.exists(cwd_path):
        print(f"Using session file from current directory: {cwd_path}")
        return cwd_path
    print("Session file not found in any known location.")
    return home_path  # fallback for error message

SESSION_PATH = get_session_path()
RESULTS_DIR = Path(__file__).parent / "results" / "kite" / "strategy_manager"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Utility Functions ---

def load_access_token():
    token = os.environ.get("KITE_ACCESS_TOKEN")
    if token:
        return token
    print(f"Looking for session file at: {SESSION_PATH}")
    if not os.path.exists(SESSION_PATH):
        print(f"Session file not found: {SESSION_PATH}")
        return None
    try:
        with open(SESSION_PATH, "r", encoding="utf8") as f:
            data = json.load(f)
        access_token = data.get("access_token")
        if not access_token:
            print("No access_token found in session file. Please re-run kite.py and complete login.")
        return access_token
    except Exception as e:
        print(f"Error reading session file: {e}")
        return None

def init_kite_client():
    api_key = os.environ.get("KITE_API_KEY")
    access_token = load_access_token()
    if not api_key or not access_token:
        raise SystemExit("KITE_API_KEY and valid access token required (run kite.py first)")
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite

def get_account_funds(kite):
    try:
        margins = kite.margins()
        equity_margins = margins.get("equity", {})
        return equity_margins.get("available", {}).get("live_balance", 0)
    except Exception as e:
        logger.error(f"Failed to fetch margins: {e}")
        return 0

def place_bull_call_spread(kite, underlying, lower_strike, higher_strike, expiry, qty, profit_pct, stop_loss_pct, max_fund_risk, dry_run=False):
    # Construct option symbols
    ce_buy = f"{underlying}{expiry}{int(lower_strike)}CE"
    ce_sell = f"{underlying}{expiry}{int(higher_strike)}CE"
    logger.info(f"Placing Bull Call Spread: BUY {ce_buy}, SELL {ce_sell}, Qty: {qty}")
    if dry_run:
        logger.info("[DRY RUN] Orders not placed.")
        return {"buy": ce_buy, "sell": ce_sell, "qty": qty}
    # Place buy order (lower strike)
    kite.place_order(
        variety=kite.VARIETY_REGULAR,
        exchange=kite.EXCHANGE_NFO,
        tradingsymbol=ce_buy,
        transaction_type=kite.TRANSACTION_TYPE_BUY,
        quantity=qty,
        order_type=kite.ORDER_TYPE_MARKET,
        product=kite.PRODUCT_MIS
    )
    # Place sell order (higher strike)
    kite.place_order(
        variety=kite.VARIETY_REGULAR,
        exchange=kite.EXCHANGE_NFO,
        tradingsymbol=ce_sell,
        transaction_type=kite.TRANSACTION_TYPE_SELL,
        quantity=qty,
        order_type=kite.ORDER_TYPE_MARKET,
        product=kite.PRODUCT_MIS
    )
    logger.info("Orders placed.")
    # Risk management (simplified)
    # In production, fetch option prices and calculate max profit/loss
    # Here, just log the intended targets
    logger.info(f"Target profit: {profit_pct}% | Stop loss: {stop_loss_pct}% | Max fund risk: {max_fund_risk}%")
    return {"buy": ce_buy, "sell": ce_sell, "qty": qty}

def main():

    import argparse
    parser = argparse.ArgumentParser(description="Bull Call Spread Agent")
    parser.add_argument("--underlying", required=True)
    # No need for user to specify higher strike; it will be set as lower_strike + 200
    parser.add_argument("--expiry", required=False, help="Expiry (default: next Tuesday)")
    parser.add_argument("--qty", type=int, default=50)
    parser.add_argument("--profit-pct", type=float, default=30)
    parser.add_argument("--stop-loss-pct", type=float, default=30)
    parser.add_argument("--max-fund-risk", type=float, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()


    # Determine next Tuesday expiry if not provided
    from datetime import datetime, timedelta
    today = datetime.now()
    days_ahead = (1 - today.weekday() + 7) % 7  # 1 = Tuesday
    if days_ahead == 0:
        days_ahead = 7
    next_tuesday = today + timedelta(days=days_ahead)
    expiry_str = next_tuesday.strftime('%d%b').upper()
    expiry = args.expiry if args.expiry else expiry_str
    logger.info(f"Using expiry: {expiry}")

    kite = init_kite_client()
    funds = get_account_funds(kite)
    logger.info(f"Available funds: ₹{funds:,.2f}")

    # Fetch live spot price using Kite WebSocket
    from kiteconnect import KiteTicker
    import threading
    spot = None
    spot_ready = threading.Event()
    # Get instrument token for underlying
    try:
        nse_instruments = kite.instruments(exchange="NSE")
        instrument_token = None
        for inst in nse_instruments:
            if inst.get("tradingsymbol", "").upper() == args.underlying.upper() and inst.get("segment", "") == "NSE":
                instrument_token = inst.get("instrument_token")
                break
        if not instrument_token:
            logger.error(f"Instrument token not found for {args.underlying} in NSE instruments.")
            return
    except Exception as e:
        logger.error(f"Failed to fetch instrument token: {e}")
        return

    def on_ticks(ws, ticks):
        nonlocal spot
        if ticks and "last_price" in ticks[0]:
            spot = float(ticks[0]["last_price"])
            logger.info(f"Live spot price for {args.underlying}: {spot}")
            spot_ready.set()

    def on_connect(ws, response):
        ws.subscribe([instrument_token])

    kws = KiteTicker(api_key, access_token)
    kws.on_ticks = on_ticks
    kws.on_connect = on_connect
    # Start WebSocket in a thread
    ws_thread = threading.Thread(target=kws.connect, kwargs={"threaded": True})
    ws_thread.daemon = True
    ws_thread.start()
    # Wait for live spot price
    logger.info("Waiting for live spot price from WebSocket...")
    spot_ready.wait(timeout=10)
    if spot is None:
        logger.error("Did not receive live spot price within timeout.")
        return

    # Fetch live option chain from Kite
    try:
        instruments = kite.instruments(exchange="NFO")
        # Print sample NIFTY option contracts and expiry formats
        nifty_opts = [inst for inst in instruments if inst.get("name", "").upper().startswith("NIFTY") and inst.get("instrument_type", "") == "OPT"]
        logger.info(f"Sample NIFTY option contracts (first 5): {[{'name': x.get('name'), 'expiry': x.get('expiry'), 'tradingsymbol': x.get('tradingsymbol')} for x in nifty_opts[:5]]}")
        logger.info(f"Unique expiry values for NIFTY options: {sorted(set(str(x.get('expiry')) for x in nifty_opts))}")
        # Filter for this underlying and expiry
        expiry_upper = expiry.upper()
        option_chain = [
            inst for inst in instruments
            if inst.get("name", "").upper() == args.underlying.upper()
            and inst.get("instrument_type", "") == "OPT"
            and expiry_upper in str(inst.get("expiry", "")).upper()
        ]
        if not option_chain:
            logger.error(f"No option contracts found for {args.underlying} and expiry {expiry}")
            return
        option_chain_df = pd.DataFrame(option_chain)
        logger.info(f"Fetched {len(option_chain_df)} option contracts for {args.underlying} {expiry}")
    except Exception as e:
        logger.error(f"Failed to fetch live option chain: {e}")
        return

    # Determine ATM strike
    lower_strike = atm_strike(spot, option_chain_df)
    logger.info(f"ATM strike (used as lower strike): {lower_strike}")

    higher_strike = lower_strike + 200
    logger.info(f"Higher strike (200 points above ATM): {higher_strike}")

    # Fetch lot size for the underlying from instruments
    lot_size = 50  # fallback default
    try:
        instruments = kite.instruments(exchange="NFO")
        for inst in instruments:
            if inst.get("name", "").upper() == args.underlying.upper() and inst.get("segment", "").startswith("NFO-OPT"):
                lot_size = int(inst.get("lot_size", 50))
                break
        logger.info(f"Lot size for {args.underlying}: {lot_size}")
    except Exception as e:
        logger.warning(f"Could not fetch lot size, using default 50: {e}")

    result = place_bull_call_spread(
        kite,
        args.underlying,
        lower_strike,
        higher_strike,
        expiry,
        lot_size,
        args.profit_pct,
        args.stop_loss_pct,
        args.max_fund_risk,
        args.dry_run
    )
    # Log result
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"bull_call_spread_{now}.json"
    with open(out_path, "w", encoding="utf8") as f:
        import json
        json.dump(result, f, indent=2)
    logger.info(f"Result saved to {out_path}")

if __name__ == "__main__":
    main()
