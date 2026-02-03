"""
Position Manager Agent - Automated Stop Loss and Profit Target Management

Features:
- Fetches current open positions from Kite
- Calculates stop loss and profit target levels based on configurable rules
- Places protective orders (stop loss and limit orders)
- Monitors and updates orders based on position changes
- Supports multiple strategies: percentage-based, ATR-based, fixed points

Usage:
  python position_manager.py --stop-loss 2.0 --target 5.0 --mode percent
  python position_manager.py --stop-loss 50 --target 100 --mode points --dry-run
  python position_manager.py --config position_config.json

Configuration:
  - Set KITE_API_KEY and have valid session or KITE_ACCESS_TOKEN
  - Default rules: 2% stop loss, 5% profit target
"""
import os
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from kiteconnect import KiteConnect
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SESSION_PATH = os.path.expanduser("~/.kite_session.json")
RESULTS_DIR = Path(__file__).parent / "results" / "kite" / "position_manager"
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


def calculate_stop_loss_target(position: Dict, stop_loss_pct: float, target_pct: float, 
                                mode: str = "percent") -> Dict:
    """
    Calculate stop loss and profit target levels for a position
    
    Args:
        position: Position dict from Kite API
        stop_loss_pct: Stop loss percentage or points
        target_pct: Profit target percentage or points
        mode: "percent" or "points"
    
    Returns:
        Dict with stop_loss_price, target_price, order_type (BUY/SELL)
    """
    quantity = position.get("quantity", 0)
    avg_price = position.get("average_price", 0)
    last_price = position.get("last_price") or position.get("close_price") or avg_price
    
    if quantity == 0 or avg_price == 0:
        return None
    
    # Determine if this is a long or short position
    is_long = quantity > 0
    
    if mode == "percent":
        # Percentage-based calculation
        if is_long:
            stop_loss_price = avg_price * (1 - stop_loss_pct / 100)
            target_price = avg_price * (1 + target_pct / 100)
            sl_order_type = "SELL"  # Sell to exit long
        else:
            stop_loss_price = avg_price * (1 + stop_loss_pct / 100)
            target_price = avg_price * (1 - target_pct / 100)
            sl_order_type = "BUY"  # Buy to cover short
    else:  # points mode
        if is_long:
            stop_loss_price = avg_price - stop_loss_pct
            target_price = avg_price + target_pct
            sl_order_type = "SELL"
        else:
            stop_loss_price = avg_price + stop_loss_pct
            target_price = avg_price - target_pct
            sl_order_type = "BUY"
    
    return {
        "symbol": position.get("tradingsymbol"),
        "exchange": position.get("exchange"),
        "quantity": abs(quantity),
        "avg_price": avg_price,
        "last_price": last_price,
        "is_long": is_long,
        "stop_loss_price": round(stop_loss_price, 2),
        "target_price": round(target_price, 2),
        "sl_order_type": sl_order_type,
        "product": position.get("product", "CNC"),
        "instrument_token": position.get("instrument_token")
    }


def place_stop_loss_order(kite: KiteConnect, params: Dict, dry_run: bool = False) -> Optional[str]:
    """
    Place a stop loss order
    
    Returns order_id or None if failed
    """
    try:
        order_params = {
            "exchange": params["exchange"],
            "tradingsymbol": params["symbol"],
            "transaction_type": params["sl_order_type"],
            "quantity": params["quantity"],
            "order_type": "SL",  # Stop Loss order
            "product": params["product"],
            "validity": "DAY",
            "price": 0,  # For SL order, price is 0
            "trigger_price": params["stop_loss_price"]
        }
        
        if dry_run:
            logger.info(f"[DRY RUN] Would place SL order: {order_params}")
            return "DRY_RUN_SL_ORDER"
        
        order_id = kite.place_order(variety=kite.VARIETY_REGULAR, **order_params)
        logger.info(f"✓ Placed SL order {order_id} for {params['symbol']} @ trigger={params['stop_loss_price']}")
        return order_id
    except Exception as e:
        logger.error(f"Failed to place SL order for {params['symbol']}: {e}")
        return None


def place_target_order(kite: KiteConnect, params: Dict, dry_run: bool = False) -> Optional[str]:
    """
    Place a profit target limit order
    
    Returns order_id or None if failed
    """
    try:
        order_params = {
            "exchange": params["exchange"],
            "tradingsymbol": params["symbol"],
            "transaction_type": params["sl_order_type"],  # Same side as SL (exit order)
            "quantity": params["quantity"],
            "order_type": "LIMIT",
            "product": params["product"],
            "validity": "DAY",
            "price": params["target_price"]
        }
        
        if dry_run:
            logger.info(f"[DRY RUN] Would place target order: {order_params}")
            return "DRY_RUN_TARGET_ORDER"
        
        order_id = kite.place_order(variety=kite.VARIETY_REGULAR, **order_params)
        logger.info(f"✓ Placed target order {order_id} for {params['symbol']} @ price={params['target_price']}")
        return order_id
    except Exception as e:
        logger.error(f"Failed to place target order for {params['symbol']}: {e}")
        return None


def get_pending_orders(kite: KiteConnect) -> List[Dict]:
    """Get all pending orders"""
    try:
        orders = kite.orders()
        # Filter only open/pending orders
        pending = [o for o in orders if o.get("status") in ["TRIGGER PENDING", "OPEN", "PENDING"]]
        return pending
    except Exception as e:
        logger.error(f"Failed to fetch orders: {e}")
        return []


def display_orders(kite: KiteConnect, show_all: bool = False):
    """Display current orders in formatted view"""
    try:
        orders = kite.orders()
        
        if not show_all:
            # Filter only pending/open orders
            orders = [o for o in orders if o.get("status") in ["TRIGGER PENDING", "OPEN", "PENDING"]]
        
        if not orders:
            logger.info("No pending orders found")
            return
        
        logger.info("\n" + "="*100)
        logger.info(f"{'ORDER ID':<12} {'SYMBOL':<20} {'TYPE':<6} {'QTY':<6} {'PRICE':<10} {'TRIGGER':<10} {'STATUS':<15} {'TIME':<10}")
        logger.info("="*100)
        
        for order in orders:
            order_id = order.get("order_id", "-")[:10]
            symbol = order.get("tradingsymbol", "-")[:18]
            trans_type = order.get("transaction_type", "-")[:4]
            qty = order.get("quantity", 0)
            price = order.get("price", 0)
            trigger = order.get("trigger_price", 0)
            status = order.get("status", "-")[:13]
            order_time = order.get("order_timestamp", "-")
            if isinstance(order_time, str) and len(order_time) > 10:
                order_time = order_time[11:19]  # Extract HH:MM:SS
            
            price_str = f"{price:.2f}" if price else "-"
            trigger_str = f"{trigger:.2f}" if trigger else "-"
            
            logger.info(f"{order_id:<12} {symbol:<20} {trans_type:<6} {qty:<6} {price_str:<10} {trigger_str:<10} {status:<15} {order_time:<10}")
        
        logger.info("="*100)
        logger.info(f"Total: {len(orders)} orders")
        
    except Exception as e:
        logger.error(f"Failed to display orders: {e}")


def cancel_order(kite: KiteConnect, order_id: str, variety: str = "regular") -> bool:
    """Cancel an order"""
    try:
        kite.cancel_order(variety=variety, order_id=order_id)
        logger.info(f"✓ Cancelled order {order_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to cancel order {order_id}: {e}")
        return False


def save_position_report(positions: List[Dict], orders_placed: Dict):
    """Save position and order report to file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = RESULTS_DIR / f"position_report_{timestamp}.json"
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "positions": positions,
        "orders_placed": orders_placed
    }
    
    try:
        with open(report_file, "w", encoding="utf8") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Report saved to {report_file}")
    except Exception as e:
        logger.error(f"Failed to save report: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Position Manager - Automated Stop Loss and Profit Target Agent"
    )
    parser.add_argument("--stop-loss", type=float, default=2.0,
                        help="Stop loss value (default: 2.0)")
    parser.add_argument("--target", type=float, default=5.0,
                        help="Profit target value (default: 5.0)")
    parser.add_argument("--mode", choices=["percent", "points"], default="percent",
                        help="Calculation mode: percent or points (default: percent)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate order placement without actually placing orders")
    parser.add_argument("--cancel-existing", action="store_true",
                        help="Cancel existing pending orders before placing new ones")
    parser.add_argument("--monitor", action="store_true",
                        help="Keep monitoring and updating orders (runs continuously)")
    parser.add_argument("--interval", type=int, default=60,
                        help="Monitoring interval in seconds (default: 60)")
    parser.add_argument("--show-orders", action="store_true",
                        help="Display current pending orders and exit (no new orders placed)")
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("Position Manager Agent Starting")
    logger.info(f"Mode: {args.mode} | Stop Loss: {args.stop_loss} | Target: {args.target}")
    logger.info(f"Dry Run: {args.dry_run}")
    logger.info("="*60)
    
    # Initialize Kite client
    kite = init_kite_client()
    
    # If user just wants to see orders, show and exit
    if args.show_orders:
        logger.info("\nFetching current orders...")
        display_orders(kite, show_all=False)
        return
    
    try:
        while True:
            # Fetch current positions
            positions = get_positions(kite)
            
            if not positions:
                logger.info("No open positions found")
                if not args.monitor:
                    break
                time.sleep(args.interval)
                continue
            
            # Display positions
            logger.info("\n" + "="*60)
            logger.info("CURRENT POSITIONS:")
            logger.info("="*60)
            for pos in positions:
                symbol = pos.get("tradingsymbol")
                qty = pos.get("quantity")
                avg_price = pos.get("average_price")
                last_price = pos.get("last_price", avg_price)
                pnl = pos.get("pnl", 0)
                logger.info(f"{symbol}: Qty={qty}, Avg={avg_price:.2f}, LTP={last_price:.2f}, P&L={pnl:.2f}")
            
            # Cancel existing orders if requested
            if args.cancel_existing:
                pending_orders = get_pending_orders(kite)
                logger.info(f"\nCancelling {len(pending_orders)} pending orders...")
                for order in pending_orders:
                    if not args.dry_run:
                        cancel_order(kite, order["order_id"], variety=order.get("variety", "regular"))
            
            # Calculate and place orders for each position
            orders_placed = {"stop_loss": [], "target": []}
            
            logger.info("\n" + "="*60)
            logger.info("PLACING PROTECTIVE ORDERS:")
            logger.info("="*60)
            
            for pos in positions:
                params = calculate_stop_loss_target(
                    pos, 
                    args.stop_loss, 
                    args.target, 
                    mode=args.mode
                )
                
                if not params:
                    continue
                
                logger.info(f"\n{params['symbol']} ({'LONG' if params['is_long'] else 'SHORT'}):")
                logger.info(f"  Entry: {params['avg_price']:.2f} | Current: {params['last_price']:.2f}")
                logger.info(f"  Stop Loss: {params['stop_loss_price']:.2f}")
                logger.info(f"  Target: {params['target_price']:.2f}")
                
                # Place stop loss order
                sl_order_id = place_stop_loss_order(kite, params, dry_run=args.dry_run)
                if sl_order_id:
                    orders_placed["stop_loss"].append({
                        "symbol": params["symbol"],
                        "order_id": sl_order_id,
                        "price": params["stop_loss_price"]
                    })
                
                # Place target order
                target_order_id = place_target_order(kite, params, dry_run=args.dry_run)
                if target_order_id:
                    orders_placed["target"].append({
                        "symbol": params["symbol"],
                        "order_id": target_order_id,
                        "price": params["target_price"]
                    })
            
            # Save report
            save_position_report(positions, orders_placed)
            
            logger.info("\n" + "="*60)
            logger.info(f"SUMMARY: Placed {len(orders_placed['stop_loss'])} SL orders, "
                       f"{len(orders_placed['target'])} target orders")
            logger.info("="*60)
            
            # Display current orders
            if not args.dry_run:
                logger.info("\nCurrent pending orders:")
                display_orders(kite, show_all=False)
            
            # Break if not monitoring continuously
            if not args.monitor:
                break
            
            logger.info(f"\nMonitoring enabled. Next check in {args.interval} seconds...")
            logger.info("Press Ctrl+C to stop")
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        logger.info("\n\nStopped by user")
    except Exception as e:
        logger.error(f"Error in main loop: {e}", exc_info=True)
    
    logger.info("\nPosition Manager Agent Stopped")


if __name__ == "__main__":
    main()
