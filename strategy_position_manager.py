"""
Strategy-Based Position Manager - Bull Call Spread and Other Strategies

Features:
- Detects option strategies (bull call spread, bear put spread, etc.)
- Calculates max profit/loss for strategies
- Sets targets based on % of max profit
- Caps stop loss at % of total funds
- Places protective orders for entire strategy

Usage:
  python strategy_position_manager.py --profit-pct 30 --stop-loss-pct 30 --max-fund-risk 2
  python strategy_position_manager.py --profit-pct 25 --stop-loss-pct 25 --max-fund-risk 1.5 --dry-run
  python strategy_position_manager.py --profit-pct 30 --stop-loss-pct 30 --max-fund-risk 2 --monitor
  # Monitor mode with custom interval (default is 30 seconds)
python strategy_position_manager.py --profit-pct 30 --stop-loss-pct 30 --max-fund-risk 2 --monitor --interval 60

# Monitor mode with dry-run (simulates without placing actual orders)
python strategy_position_manager.py --profit-pct 30 --stop-loss-pct 30 --max-fund-risk 2 --monitor --dry-run
"""
import os
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from kiteconnect import KiteConnect
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SESSION_PATH = os.path.expanduser("~/.kite_session.json")
RESULTS_DIR = Path(__file__).parent / "results" / "kite" / "strategy_manager"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_access_token():
    """Load access token from session file or environment"""
    token = os.environ.get("KITE_ACCESS_TOKEN")
    if token:
        return token
    
    try:
        with open(SESSION_PATH, "r", encoding="utf8") as f:
            data = json.load(f)
        return data.get("access_token")
    except Exception:
        return None


def init_kite_client():
    """Initialize authenticated Kite client"""
    api_key = os.environ.get("KITE_API_KEY")
    access_token = load_access_token()
    
    if not api_key or not access_token:
        raise SystemExit("KITE_API_KEY and valid access token required (run kite.py first)")
    
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    
    # Verify connection
    try:
        profile = kite.profile()
        logger.info(f"Connected as: {profile.get('user_name', 'N/A')} ({profile.get('user_id', 'N/A')})")
    except Exception as e:
        raise SystemExit(f"Authentication failed: {e}")
    
    return kite


def get_account_funds(kite: KiteConnect) -> float:
    """Get total available funds in account"""
    try:
        margins = kite.margins()
        equity_margins = margins.get("equity", {})
        available = equity_margins.get("available", {}).get("live_balance", 0)
        logger.info(f"Total available funds: ₹{available:,.2f}")
        return available
    except Exception as e:
        logger.error(f"Failed to fetch margins: {e}")
        return 0


def get_positions(kite: KiteConnect) -> List[Dict]:
    """Fetch current positions (net positions)"""
    try:
        positions_data = kite.positions()
        net_positions = positions_data.get("net", [])
        # Filter only positions with non-zero quantity
        open_positions = [p for p in net_positions if p.get("quantity", 0) != 0]
        logger.info(f"Found {len(open_positions)} open positions")
        return open_positions
    except Exception as e:
        logger.error(f"Failed to fetch positions: {e}")
        return []


def parse_option_symbol(symbol: str) -> Optional[Dict]:
    """
    Parse option symbol to extract underlying, strike, and type
    Example: NIFTY24JAN24000CE -> underlying=NIFTY, strike=24000, option_type=CE
    """
    symbol = symbol.upper()
    
    # Try to extract option type (CE/PE)
    if symbol.endswith("CE"):
        option_type = "CE"
        symbol_base = symbol[:-2]
    elif symbol.endswith("PE"):
        option_type = "PE"
        symbol_base = symbol[:-2]
    else:
        return None  # Not an option
    
    # Extract strike price (last digits before CE/PE)
    strike_str = ""
    for i in range(len(symbol_base) - 1, -1, -1):
        if symbol_base[i].isdigit():
            strike_str = symbol_base[i] + strike_str
        else:
            break
    
    if not strike_str:
        return None
    
    strike = float(strike_str)
    underlying = symbol_base[:-len(strike_str)]
    
    return {
        "underlying": underlying,
        "strike": strike,
        "option_type": option_type,
        "full_symbol": symbol
    }


def detect_strategies(positions: List[Dict]) -> List[Dict]:
    """
    Detect option strategies from positions
    Returns list of detected strategies with their positions
    """
    # Group positions by underlying
    by_underlying = defaultdict(list)
    
    for pos in positions:
        symbol = pos.get("tradingsymbol", "")
        parsed = parse_option_symbol(symbol)
        
        if parsed:
            pos["parsed_option"] = parsed
            by_underlying[parsed["underlying"]].append(pos)
        else:
            # Non-option position - treat as standalone
            by_underlying[symbol].append(pos)
    
    strategies = []
    
    for underlying, positions_list in by_underlying.items():
        option_positions = [p for p in positions_list if "parsed_option" in p]
        
        if len(option_positions) < 2:
            # Single position or non-option - treat each as individual
            for pos in positions_list:
                strategies.append({
                    "type": "single",
                    "underlying": underlying,
                    "positions": [pos],
                    "name": pos.get("tradingsymbol")
                })
            continue
        
        # Sort by strike
        option_positions.sort(key=lambda x: x["parsed_option"]["strike"])
        
        # Detect bull call spread: Long lower strike call + Short higher strike call
        calls = [p for p in option_positions if p["parsed_option"]["option_type"] == "CE"]
        puts = [p for p in option_positions if p["parsed_option"]["option_type"] == "PE"]
        
        if len(calls) == 2:
            long_call = [p for p in calls if p.get("quantity", 0) > 0]
            short_call = [p for p in calls if p.get("quantity", 0) < 0]
            
            if long_call and short_call:
                if long_call[0]["parsed_option"]["strike"] < short_call[0]["parsed_option"]["strike"]:
                    strategies.append({
                        "type": "bull_call_spread",
                        "underlying": underlying,
                        "positions": [long_call[0], short_call[0]],
                        "name": f"{underlying} Bull Call Spread",
                        "long_strike": long_call[0]["parsed_option"]["strike"],
                        "short_strike": short_call[0]["parsed_option"]["strike"]
                    })
                    continue
                elif long_call[0]["parsed_option"]["strike"] > short_call[0]["parsed_option"]["strike"]:
                    # Bear call spread
                    strategies.append({
                        "type": "bear_call_spread",
                        "underlying": underlying,
                        "positions": [short_call[0], long_call[0]],
                        "name": f"{underlying} Bear Call Spread",
                        "short_strike": short_call[0]["parsed_option"]["strike"],
                        "long_strike": long_call[0]["parsed_option"]["strike"]
                    })
                    continue
        
        # Detect bull put spread: Short higher strike put + Long lower strike put
        if len(puts) == 2:
            long_put = [p for p in puts if p.get("quantity", 0) > 0]
            short_put = [p for p in puts if p.get("quantity", 0) < 0]
            
            if long_put and short_put:
                if short_put[0]["parsed_option"]["strike"] > long_put[0]["parsed_option"]["strike"]:
                    strategies.append({
                        "type": "bull_put_spread",
                        "underlying": underlying,
                        "positions": [short_put[0], long_put[0]],
                        "name": f"{underlying} Bull Put Spread",
                        "short_strike": short_put[0]["parsed_option"]["strike"],
                        "long_strike": long_put[0]["parsed_option"]["strike"]
                    })
                    continue
        
        # If not matched to known strategy, treat as collection of individual positions
        for pos in option_positions:
            strategies.append({
                "type": "single",
                "underlying": underlying,
                "positions": [pos],
                "name": pos.get("tradingsymbol")
            })
    
    return strategies


def calculate_strategy_metrics(strategy: Dict) -> Dict:
    """
    Calculate max profit, max loss, and current P&L for a strategy
    """
    positions = strategy["positions"]
    strategy_type = strategy["type"]
    
    # Calculate total current P&L
    total_pnl = sum(p.get("pnl", 0) for p in positions)
    total_quantity = sum(abs(p.get("quantity", 0)) for p in positions)
    
    if strategy_type == "bull_call_spread":
        # Max Profit = (Short Strike - Long Strike) * Lot Size - Net Premium Paid
        long_pos = [p for p in positions if p.get("quantity", 0) > 0][0]
        short_pos = [p for p in positions if p.get("quantity", 0) < 0][0]
        
        long_qty = abs(long_pos.get("quantity", 0))
        long_avg = long_pos.get("average_price", 0)
        short_avg = short_pos.get("average_price", 0)
        
        net_premium_paid = (long_avg - short_avg) * long_qty
        strike_width = strategy["short_strike"] - strategy["long_strike"]
        max_profit = (strike_width * long_qty) - net_premium_paid
        max_loss = net_premium_paid
        
    elif strategy_type == "bear_call_spread":
        # Max Profit = Net Premium Received
        # Max Loss = (Long Strike - Short Strike) * Lot Size - Net Premium Received
        short_pos = [p for p in positions if p.get("quantity", 0) < 0][0]
        long_pos = [p for p in positions if p.get("quantity", 0) > 0][0]
        
        short_qty = abs(short_pos.get("quantity", 0))
        short_avg = short_pos.get("average_price", 0)
        long_avg = long_pos.get("average_price", 0)
        
        net_premium_received = (short_avg - long_avg) * short_qty
        strike_width = strategy["long_strike"] - strategy["short_strike"]
        max_profit = net_premium_received
        max_loss = (strike_width * short_qty) - net_premium_received
        
    elif strategy_type == "bull_put_spread":
        # Max Profit = Net Premium Received
        # Max Loss = (Short Strike - Long Strike) * Lot Size - Net Premium Received
        short_pos = [p for p in positions if p.get("quantity", 0) < 0][0]
        long_pos = [p for p in positions if p.get("quantity", 0) > 0][0]
        
        short_qty = abs(short_pos.get("quantity", 0))
        short_avg = short_pos.get("average_price", 0)
        long_avg = long_pos.get("average_price", 0)
        
        net_premium_received = (short_avg - long_avg) * short_qty
        strike_width = strategy["short_strike"] - strategy["long_strike"]
        max_profit = net_premium_received
        max_loss = (strike_width * short_qty) - net_premium_received
        
    else:  # single position
        # For single positions, estimate based on current metrics
        pos = positions[0]
        qty = abs(pos.get("quantity", 0))
        avg_price = pos.get("average_price", 0)
        
        # Simple heuristic: max loss = invested amount for long, max profit = 2x for long
        if pos.get("quantity", 0) > 0:
            max_loss = avg_price * qty
            max_profit = avg_price * qty * 2  # arbitrary multiplier
        else:
            max_profit = avg_price * qty
            max_loss = avg_price * qty * 3  # arbitrary multiplier for short
    
    return {
        "current_pnl": total_pnl,
        "max_profit": max_profit,
        "max_loss": max_loss,
        "profit_pct": (total_pnl / max_profit * 100) if max_profit > 0 else 0,
        "loss_pct": (abs(total_pnl) / max_loss * 100) if max_loss > 0 and total_pnl < 0 else 0
    }


def calculate_exit_levels(strategy: Dict, metrics: Dict, profit_pct: float, 
                          stop_loss_pct: float, max_fund_risk_pct: float, 
                          total_funds: float) -> Dict:
    """
    Calculate target and stop loss levels based on max profit percentage
    and fund risk limits
    """
    max_profit = metrics["max_profit"]
    max_loss = metrics["max_loss"]
    current_pnl = metrics["current_pnl"]
    
    # Target: profit_pct% of max profit
    target_pnl = max_profit * (profit_pct / 100)
    
    # Stop loss: stop_loss_pct% of max profit (as loss)
    stop_loss_pnl = -max_profit * (stop_loss_pct / 100)
    
    # Cap stop loss at max_fund_risk_pct% of total funds
    max_allowed_loss = total_funds * (max_fund_risk_pct / 100)
    if abs(stop_loss_pnl) > max_allowed_loss:
        stop_loss_pnl = -max_allowed_loss
        logger.info(f"  ⚠️  Stop loss capped at {max_fund_risk_pct}% of funds: ₹{max_allowed_loss:,.2f}")
    
    # Calculate required price movements for strategy exit
    # For spread strategies, we need to calculate what combined P&L we need
    
    return {
        "target_pnl": target_pnl,
        "stop_loss_pnl": stop_loss_pnl,
        "current_pnl": current_pnl,
        "target_profit_pct": profit_pct,
        "stop_loss_pct": stop_loss_pct,
        "distance_to_target": target_pnl - current_pnl,
        "distance_to_sl": current_pnl - stop_loss_pnl
    }


def place_strategy_exit_orders(kite: KiteConnect, strategy: Dict, exit_levels: Dict, 
                               dry_run: bool = False) -> List[str]:
    """
    Place exit orders for entire strategy when target or SL is hit
    For now, using simple market orders for each leg
    """
    order_ids = []
    positions = strategy["positions"]
    
    logger.info(f"\n  Exit Levels for {strategy['name']}:")
    logger.info(f"    Target P&L: ₹{exit_levels['target_pnl']:,.2f} ({exit_levels['target_profit_pct']}% of max profit)")
    logger.info(f"    Stop Loss P&L: ₹{exit_levels['stop_loss_pnl']:,.2f} ({exit_levels['stop_loss_pct']}% of max profit)")
    logger.info(f"    Current P&L: ₹{exit_levels['current_pnl']:,.2f}")
    logger.info(f"    Distance to target: ₹{exit_levels['distance_to_target']:,.2f}")
    logger.info(f"    Distance to SL: ₹{exit_levels['distance_to_sl']:,.2f}")
    
    # Note: Kite doesn't support strategy-level orders directly
    # We would need to monitor P&L and place market orders when levels are hit
    # For now, log the exit levels - manual monitoring or separate monitoring script needed
    
    if dry_run:
        logger.info(f"  [DRY RUN] Would monitor strategy and exit all positions when P&L reaches levels")
    else:
        logger.info(f"  ⚠️  Manual monitoring required: Exit all positions when strategy P&L reaches target or SL")
        logger.info(f"  💡 Consider using --monitor mode for automated monitoring")
    
    return order_ids


def monitor_and_exit_strategies(kite: KiteConnect, strategies: List[Dict], 
                                exit_levels_map: Dict, dry_run: bool = False):
    """
    Monitor strategies and execute exits when levels are hit
    """
    for strategy in strategies:
        strategy_key = strategy["name"]
        exit_levels = exit_levels_map.get(strategy_key)
        
        if not exit_levels:
            continue
        
        current_pnl = exit_levels["current_pnl"]
        target_pnl = exit_levels["target_pnl"]
        stop_loss_pnl = exit_levels["stop_loss_pnl"]
        
        # Check if target hit
        if current_pnl >= target_pnl:
            logger.info(f"\n🎯 TARGET HIT for {strategy_key}! P&L: ₹{current_pnl:,.2f} >= Target: ₹{target_pnl:,.2f}")
            logger.info(f"Exiting all positions...")
            
            for pos in strategy["positions"]:
                # Place market order to exit
                exit_qty = abs(pos.get("quantity", 0))
                exit_type = "SELL" if pos.get("quantity", 0) > 0 else "BUY"
                
                try:
                    if not dry_run:
                        order_id = kite.place_order(
                            variety=kite.VARIETY_REGULAR,
                            exchange=pos.get("exchange"),
                            tradingsymbol=pos.get("tradingsymbol"),
                            transaction_type=exit_type,
                            quantity=exit_qty,
                            order_type="MARKET",
                            product=pos.get("product"),
                            validity="DAY"
                        )
                        logger.info(f"  ✓ Exited {pos.get('tradingsymbol')}: Order {order_id}")
                    else:
                        logger.info(f"  [DRY RUN] Would exit {pos.get('tradingsymbol')}: {exit_type} {exit_qty}")
                except Exception as e:
                    logger.error(f"  Failed to exit {pos.get('tradingsymbol')}: {e}")
        
        # Check if stop loss hit
        elif current_pnl <= stop_loss_pnl:
            logger.info(f"\n🛑 STOP LOSS HIT for {strategy_key}! P&L: ₹{current_pnl:,.2f} <= SL: ₹{stop_loss_pnl:,.2f}")
            logger.info(f"Exiting all positions...")
            
            for pos in strategy["positions"]:
                # Place market order to exit
                exit_qty = abs(pos.get("quantity", 0))
                exit_type = "SELL" if pos.get("quantity", 0) > 0 else "BUY"
                
                try:
                    if not dry_run:
                        order_id = kite.place_order(
                            variety=kite.VARIETY_REGULAR,
                            exchange=pos.get("exchange"),
                            tradingsymbol=pos.get("tradingsymbol"),
                            transaction_type=exit_type,
                            quantity=exit_qty,
                            order_type="MARKET",
                            product=pos.get("product"),
                            validity="DAY"
                        )
                        logger.info(f"  ✓ Exited {pos.get('tradingsymbol')}: Order {order_id}")
                    else:
                        logger.info(f"  [DRY RUN] Would exit {pos.get('tradingsymbol')}: {exit_type} {exit_qty}")
                except Exception as e:
                    logger.error(f"  Failed to exit {pos.get('tradingsymbol')}: {e}")


def save_strategy_report(strategies: List[Dict], exit_levels_map: Dict):
    """Save strategy analysis report"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = RESULTS_DIR / f"strategy_report_{timestamp}.json"
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "strategies": strategies,
        "exit_levels": exit_levels_map
    }
    
    try:
        with open(report_file, "w", encoding="utf8") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Report saved to {report_file}")
    except Exception as e:
        logger.error(f"Failed to save report: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Strategy-Based Position Manager - Max Profit Based Targets"
    )
    parser.add_argument("--profit-pct", type=float, default=30,
                        help="Target profit as %% of max profit (default: 30)")
    parser.add_argument("--stop-loss-pct", type=float, default=30,
                        help="Stop loss as %% of max profit (default: 30)")
    parser.add_argument("--max-fund-risk", type=float, default=2,
                        help="Maximum stop loss as %% of total funds (default: 2)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate without placing orders")
    parser.add_argument("--monitor", action="store_true",
                        help="Continuously monitor and auto-exit when levels hit")
    parser.add_argument("--interval", type=int, default=30,
                        help="Monitoring interval in seconds (default: 30)")
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("Strategy-Based Position Manager Starting")
    logger.info(f"Target: {args.profit_pct}% of max profit | SL: {args.stop_loss_pct}% of max profit")
    logger.info(f"Max Fund Risk: {args.max_fund_risk}% | Dry Run: {args.dry_run}")
    logger.info("="*80)
    
    # Initialize Kite client
    kite = init_kite_client()

    # Get account funds
    total_funds = get_account_funds(kite)
    if total_funds == 0:
        logger.warning("Could not fetch account funds, using default risk limits")
        total_funds = 100000  # Default fallback

    # --- WebSocket for live spot price ---
    from kiteconnect import KiteTicker
    import threading
    live_spot = {}
    spot_ready = threading.Event()

    # Get instrument tokens for all underlyings in open positions
    def get_underlying_tokens(kite, positions):
        tokens = {}
        try:
            nse_instruments = kite.instruments(exchange="NSE")
            for pos in positions:
                underlying = pos.get("tradingsymbol", "").split()[0].upper()
                for inst in nse_instruments:
                    if inst.get("tradingsymbol", "").upper() == underlying and inst.get("segment", "") == "NSE":
                        tokens[underlying] = inst.get("instrument_token")
                        break
        except Exception as e:
            logger.error(f"Failed to fetch instrument tokens: {e}")
        return tokens

    def on_ticks(ws, ticks):
        for tick in ticks:
            underlying = tick.get("tradingsymbol", "").split()[0].upper()
            live_spot[underlying] = tick["last_price"]
        spot_ready.set()

    def on_connect(ws, response):
        ws.subscribe(list(underlying_tokens.values()))

    # Main loop
    try:
        while True:
            positions = get_positions(kite)
            if not positions:
                logger.info("No open positions found")
                if not args.monitor:
                    break
                time.sleep(args.interval)
                continue

            # Detect strategies
            strategies = detect_strategies(positions)

            logger.info("\n" + "="*80)
            logger.info(f"DETECTED {len(strategies)} STRATEGIES:")
            logger.info("="*80)

            exit_levels_map = {}

            # Get instrument tokens for all underlyings
            underlying_tokens = get_underlying_tokens(kite, positions)
            if underlying_tokens:
                api_key = os.environ.get("KITE_API_KEY")
                access_token = load_access_token()
                kws = KiteTicker(api_key, access_token)
                kws.on_ticks = on_ticks
                kws.on_connect = on_connect
                ws_thread = threading.Thread(target=kws.connect, kwargs={"threaded": True})
                ws_thread.daemon = True
                ws_thread.start()
                logger.info("Waiting for live spot prices from WebSocket...")
                spot_ready.wait(timeout=10)
            else:
                logger.warning("No instrument tokens found for live spot prices.")

            for strategy in strategies:
                logger.info(f"\n📊 {strategy['name']} ({strategy['type'].replace('_', ' ').title()})")
                logger.info(f"   Positions: {len(strategy['positions'])}")
                underlying = strategy.get('underlying')
                live_price = live_spot.get(underlying)
                if live_price:
                    logger.info(f"   [LIVE] Spot price for {underlying}: {live_price}")
                else:
                    logger.warning(f"   [LIVE] Spot price for {underlying} not available yet.")
                # Use live_price in calculations if available
                for pos in strategy['positions']:
                    symbol = pos.get("tradingsymbol")
                    qty = pos.get("quantity")
                    # Example: attach live price to position for downstream logic
                    pos['live_spot_price'] = live_price
                    logger.info(f"     • {symbol}: Qty={qty}, Live Spot={live_price}")
                    avg = pos.get("average_price", 0)
                    ltp = pos.get("last_price", avg)
                    pnl = pos.get("pnl", 0)
                    logger.info(f"     • {symbol}: Qty={qty}, Avg={avg:.2f}, LTP={ltp:.2f}, P&L=₹{pnl:,.2f}")
                
                # Calculate strategy metrics
                metrics = calculate_strategy_metrics(strategy)
                logger.info(f"\n   Strategy Metrics:")
                logger.info(f"     Max Profit: ₹{metrics['max_profit']:,.2f}")
                logger.info(f"     Max Loss: ₹{metrics['max_loss']:,.2f}")
                logger.info(f"     Current P&L: ₹{metrics['current_pnl']:,.2f} ({metrics['profit_pct']:.1f}% of max profit)")
                
                # Calculate exit levels
                exit_levels = calculate_exit_levels(
                    strategy, metrics, 
                    args.profit_pct, args.stop_loss_pct, 
                    args.max_fund_risk, total_funds
                )
                
                exit_levels_map[strategy['name']] = exit_levels
                
                # Place/show exit orders
                place_strategy_exit_orders(kite, strategy, exit_levels, dry_run=args.dry_run)
            
            # Monitor and auto-exit if monitoring enabled
            if args.monitor:
                logger.info("\n" + "="*80)
                logger.info("MONITORING MODE: Checking for exit conditions...")
                logger.info("="*80)
                monitor_and_exit_strategies(kite, strategies, exit_levels_map, dry_run=args.dry_run)
            
            # Save report
            save_strategy_report(strategies, exit_levels_map)
            
            # Break if not monitoring continuously
            if not args.monitor:
                break
            
            logger.info(f"\nNext check in {args.interval} seconds... (Press Ctrl+C to stop)")
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        logger.info("\n\nStopped by user")
    except Exception as e:
        logger.error(f"Error in main loop: {e}", exc_info=True)
    
    logger.info("\nStrategy Position Manager Stopped")


if __name__ == "__main__":
    main()
