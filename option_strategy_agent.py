"""
General Option Strategy Agent
- Builds option strategies based on instructions for a symbol
- Fetches option prices from Kite
- Calculates and plots payoff diagram, max/min profit
- Places limit orders and monitors position as per strategy_position_manager.py logic
"""


import os
import logging
from pathlib import Path
from kiteconnect import KiteConnect
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from Options_Utility import atm_strike
from strategy_position_manager import calculate_strategy_metrics

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SESSION_PATH = os.environ.get("KITE_SESSION_PATH", os.path.expanduser("~/.kite_session.json"))
RESULTS_DIR = Path(__file__).parent / "results" / "kite" / "strategy_manager"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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
    api_key = os.environ.get("KITE_API_KEY")
    access_token = load_access_token()
    if not api_key or not access_token:
        raise SystemExit("KITE_API_KEY and valid access token required (run kite.py first)")
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite

def fetch_option_chain(kite, underlying, expiry):
    instruments = kite.instruments(exchange="NFO")
    expiry_upper = expiry.upper()
    option_chain = [
        inst for inst in instruments
        if inst.get("name", "").upper() == underlying.upper()
        and inst.get("instrument_type", "") == "OPT"
        and expiry_upper in str(inst.get("expiry", "")).upper()
    ]
    return pd.DataFrame(option_chain)

# --- Strategy Builder ---
def build_strategy_from_instructions(instructions, spot, option_chain_df):
    """
    instructions: list of dicts, e.g. [{"type": "BUY", "option_type": "CE", "strike": 18000, "qty": 50}]
    Returns: list of legs (dict)
    """
    legs = []
    for instr in instructions:
        strike = instr.get("strike")
        if strike == "ATM":
            strike = atm_strike(spot, option_chain_df)
        # Find option symbol
        row = option_chain_df[option_chain_df['strike_price'] == strike]
        if instr["option_type"] == "CE":
            row = row[row['instrument_type'] == 'CE']
        else:
            row = row[row['instrument_type'] == 'PE']
        if row.empty:
            logger.warning(f"No contract found for {instr}")
            continue
        symbol = row.iloc[0]['tradingsymbol']
        ltp = row.iloc[0].get('last_price', 0)
        legs.append({
            "action": instr["type"],
            "option_type": instr["option_type"],
            "strike": strike,
            "qty": instr["qty"],
            "symbol": symbol,
            "ltp": ltp
        })
    return legs

# --- Payoff Calculation ---
def payoff_profile(legs, spot_range):
    payoff = np.zeros_like(spot_range, dtype=float)
    for leg in legs:
        if leg["option_type"] == "CE":
            intrinsic = np.maximum(spot_range - leg["strike"], 0)
        else:
            intrinsic = np.maximum(leg["strike"] - spot_range, 0)
        if leg["action"] == "BUY":
            payoff += (intrinsic - leg["ltp"]) * (leg["qty"] // abs(leg["qty"]))
        else:
            payoff -= (intrinsic - leg["ltp"]) * (leg["qty"] // abs(leg["qty"]))
    return payoff

def plot_payoff(spot_range, payoff, underlying, out_path):
    plt.figure(figsize=(8,5))
    plt.plot(spot_range, payoff, label="Payoff")
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel("Spot Price at Expiry")
    plt.ylabel("P&L")
    plt.title(f"Payoff Diagram: {underlying}")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path)
    plt.close()

# --- Order Placement ---
def place_limit_orders(kite, legs):
    order_ids = []
    for leg in legs:
        order_type = kite.TRANSACTION_TYPE_BUY if leg["action"] == "BUY" else kite.TRANSACTION_TYPE_SELL
        try:
            order_id = kite.place_order(
                variety=kite.VARIETY_REGULAR,
                exchange=kite.EXCHANGE_NFO,
                tradingsymbol=leg["symbol"],
                transaction_type=order_type,
                quantity=leg["qty"],
                order_type=kite.ORDER_TYPE_LIMIT,
                price=leg["ltp"],
                product=kite.PRODUCT_MIS
            )
            order_ids.append(order_id)
            logger.info(f"Order placed: {leg['action']} {leg['symbol']} @ {leg['ltp']}")
        except Exception as e:
            logger.error(f"Order failed for {leg['symbol']}: {e}")
    return order_ids

# --- Monitoring (delegated to strategy_position_manager) ---
def monitor_position(strategy_dict):
    # This function can call strategy_position_manager logic for monitoring
    # For now, just log
    logger.info(f"Monitoring position: {strategy_dict}")
    # You can import and use monitor logic from strategy_position_manager.py

# --- Main Agent ---
def run_agent(underlying, instructions, expiry, monitor=False, dry_run=True):
    kite = init_kite_client()
    # Get spot price
    instruments = kite.instruments(exchange="NSE")
    spot = None
    for inst in instruments:
        if inst.get("tradingsymbol", "").upper() == underlying.upper() and inst.get("segment", "") == "NSE":
            spot = float(inst.get("last_price", 0))
            break
    if spot is None:
        logger.error(f"Spot price not found for {underlying}")
        return
    logger.info(f"Spot price for {underlying}: {spot}")
    # Fetch option chain
    option_chain_df = fetch_option_chain(kite, underlying, expiry)
    # Build strategy
    legs = build_strategy_from_instructions(instructions, spot, option_chain_df)
    logger.info(f"Strategy legs: {legs}")
    # Payoff
    spot_range = np.arange(spot * 0.8, spot * 1.2, 10)
    payoff = payoff_profile(legs, spot_range)
    # Max/min profit
    max_profit = np.max(payoff)
    min_profit = np.min(payoff)
    logger.info(f"Max profit: {max_profit:.2f}, Min profit: {min_profit:.2f}")
    # Plot
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"payoff_{underlying}_{now}.png"
    plot_payoff(spot_range, payoff, underlying, out_path)
    logger.info(f"Payoff diagram saved to {out_path}")
    # Place orders
    if not dry_run:
        order_ids = place_limit_orders(kite, legs)
        logger.info(f"Order IDs: {order_ids}")
    # Monitor
    if monitor:
        monitor_position({"underlying": underlying, "legs": legs, "expiry": expiry})
    # Save summary
    summary = {
        "underlying": underlying,
        "legs": legs,
        "max_profit": max_profit,
        "min_profit": min_profit,
        "payoff_plot": str(out_path)
    }
    out_json = RESULTS_DIR / f"strategy_{underlying}_{now}.json"
    with open(out_json, "w", encoding="utf8") as f:
        import json
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {out_json}")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="General Option Strategy Agent")
    parser.add_argument("--symbol", required=True, help="Underlying symbol (e.g. NIFTY)")
    parser.add_argument("--expiry", required=True, help="Expiry (e.g. 09JAN2026)")
    parser.add_argument(
        "--instructions",
        required=True,
        help='Strategy instructions as JSON string. Example: "[{\"type\": \"BUY\", \"option_type\": \"CE\", \"strike\": \"ATM\", \"qty\": 50}, {\"type\": \"SELL\", \"option_type\": \"CE\", \"strike\": \"ATM+200\", \"qty\": 50}]"'
    )
    parser.add_argument("--monitor", action="store_true", help="Monitor the position after entry")
    parser.add_argument("--dry-run", action="store_true", help="Simulate orders without placing them")
    args = parser.parse_args()

    import json
    try:
        instructions = json.loads(args.instructions)
    except Exception as e:
        logger.error(f"Failed to parse instructions: {e}")
        exit(1)

    run_agent(args.symbol, instructions, expiry=args.expiry, monitor=args.monitor, dry_run=args.dry_run)
