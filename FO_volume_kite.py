"""
Download combined daily F&O volume (FUT + OPT) for all symbols and save JSON/CSV.

Usage examples (PowerShell):
$env:KITE_API_KEY="APIKEY"; $env:KITE_ACCESS_TOKEN="ACCTOKEN"
python download_fo_volume.py --days 7

Or:
python download_fo_volume.py --api-key KEY --access-token TOKEN --days 1

Notes:
- Requires kiteconnect (pip install kiteconnect).
- Be mindful of Kite rate limits; this script sleeps between historical calls.
"""
from __future__ import annotations

import os
import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional

# load .env from project root (if python-dotenv installed)
try:
    from dotenv import load_dotenv  # type: ignore
    ROOT = Path(__file__).resolve().parents[1]
    env_path = ROOT / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=str(env_path))
        logging.getLogger("fo_volume").info("Loaded .env from %s", env_path)
    else:
        # fallback to any .env on PYTHONPATH / cwd
        load_dotenv()
except Exception:
    # dotenv not installed or failed -> continue, _load_kite_credentials also attempts dotenv if available
    pass

# logging
logger = logging.getLogger("fo_volume")
if not logger.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

# optional imports
try:
    from kiteconnect import KiteConnect  # type: ignore
except Exception:
    KiteConnect = None  # type: ignore

# try to import helper symbol discovery used elsewhere
try:
    from data_download_vbt import get_symbols, get_dates_from_most_active_files  # type: ignore
except Exception:
    get_symbols = None
    get_dates_from_most_active_files = None

# helpers
def safe_mkdir(p: Path) -> Path:
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.warning("mkdir failed %s: %s", p, e)
    return p


def _load_kite_credentials(cli_key: Optional[str], cli_token: Optional[str]) -> (Optional[str], Optional[str]):
    """
    Resolve Kite credentials using the same robust order commonly used elsewhere:
      1. CLI args
      2. Environment variables KITE_API_KEY / KITE_ACCESS_TOKEN
      3. .env (if python-dotenv available)
      4. project kite_credentials.json or ~/.kite_credentials.json
    Returns (api_key, access_token) or (None, None).
    """
    # 1) CLI args
    if cli_key and cli_token:
        return cli_key, cli_token

    # 2) environment
    env_key = os.environ.get("KITE_API_KEY") or os.environ.get("KITE_KEY")
    env_token = os.environ.get("KITE_ACCESS_TOKEN") or os.environ.get("KITE_TOKEN")
    if env_key and env_token:
        return env_key, env_token

    # 3) dotenv
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()  # loads .env in cwd
        env_key = os.environ.get("KITE_API_KEY") or os.environ.get("KITE_KEY")
        env_token = os.environ.get("KITE_ACCESS_TOKEN") or os.environ.get("KITE_TOKEN")
        if env_key and env_token:
            return env_key, env_token
    except Exception:
        pass

    # 4) credentials file in project root or home
    root = Path(__file__).resolve().parents[1]
    candidates = [
        root / "kite_credentials.json",
        root / ".kite_credentials.json",
        Path.home() / ".kite_credentials.json"
    ]
    for p in candidates:
        try:
            if p.exists():
                data = json.loads(p.read_text(encoding="utf8"))
                k = data.get("api_key") or data.get("KITE_API_KEY")
                t = data.get("access_token") or data.get("KITE_ACCESS_TOKEN")
                if k and t:
                    return k, t
        except Exception:
            continue

    return None, None

def last_n_days_range(days: int) -> (datetime, datetime):
    end = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days - 1)
    return start.date(), end.date()

def call_historical(kite, token: int, start_date: str, end_date: str, interval: str = "day", retry: int = 2) -> Optional[List[Dict[str, Any]]]:
    """Try kite.historical_data() then kite.historical() with retries."""
    for attempt in range(retry):
        try:
            if hasattr(kite, "historical_data"):
                return kite.historical_data(token, start_date.isoformat() if hasattr(start_date, "isoformat") else start_date,
                                            end_date.isoformat() if hasattr(end_date, "isoformat") else end_date, interval)
            # fallback
            if hasattr(kite, "historical"):
                return kite.historical(token, start_date, end_date, interval)
            return None
        except Exception as e:
            logger.debug("historical call failed token=%s attempt=%d: %s", token, attempt + 1, e)
            time.sleep(0.25 * (attempt + 1))
    return None

def fetch_and_aggregate(kite, instruments: List[Dict[str, Any]], start_date: datetime.date, end_date: datetime.date,
                        instrument_lot_map: Dict[int, int],
                        pause: float = 0.35, max_tokens: Optional[int] = None,
                        restrict_fut_to_nearest_expiry: bool = True) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Aggregate per-underlying per-day metrics for FUT and OPT separately and combined.
    Output structure:
      { underlying: {
           'YYYY-MM-DD': {
               'fut_contracts', 'opt_contracts', 'fo_contracts',
               'fut_shares', 'opt_shares', 'fo_shares',
               'fut_value_inr', 'opt_value_inr', 'fo_value_inr',
               'avg_price_per_share', 'avg_price_per_contract'
           }
       } }
    Only instruments with instrument_type FUT or OPT are considered.
    """
    by_underlying: Dict[str, List[Dict[str, Any]]] = {}
    for inst in instruments:
        try:
            itype = (inst.get("instrument_type") or "").upper()
            if itype not in {"FUT", "OPT"}:
                continue
            name = (inst.get("name") or inst.get("tradingsymbol") or "").upper()
            if not name:
                continue
            tok = inst.get("instrument_token") or inst.get("instrumentToken") or inst.get("token")
            if not tok:
                continue
            expiry = inst.get("expiry")  # may be None for some instruments
            by_underlying.setdefault(name, []).append({"token": int(tok), "type": itype, "expiry": expiry})
        except Exception:
            continue

    logger.info("Found %d underlyings with F&O instruments", len(by_underlying))
    aggregated: Dict[str, Dict[str, Dict[str, float]]] = {}

    total_tokens_processed = 0
    for underlying, token_infos in by_underlying.items():
        if max_tokens and total_tokens_processed >= max_tokens:
            break
        daily_map: Dict[str, Dict[str, float]] = {}
        # dedupe tokens
        seen_tokens = set()

        # determine nearest FUT expiry for this underlying (if requested)
        fut_target_expiry = None
        if restrict_fut_to_nearest_expiry:
            fut_dates = []
            for ti in token_infos:
                if ti["type"] != "FUT":
                    continue
                exp = ti.get("expiry")
                try:
                    if exp:
                        fut_dates.append(datetime.fromisoformat(str(exp)).date())
                except Exception:
                    continue
            if fut_dates:
                # pick expiry >= start_date if possible else the nearest (min)
                fut_dates_sorted = sorted(fut_dates)
                fut_target_expiry = next((d for d in fut_dates_sorted if d >= start_date), fut_dates_sorted[0])
                fut_target_expiry = fut_target_expiry.isoformat()
                logger.debug("[%s] selected FUT expiry %s", underlying, fut_target_expiry)

        for info in token_infos:
            if max_tokens and total_tokens_processed >= max_tokens:
                break
            tok = info["token"]
            if tok in seen_tokens:
                continue
            seen_tokens.add(tok)
            typ = info["type"]  # 'FUT' or 'OPT'
            # if restricting FUT to nearest expiry, skip FUT tokens not matching target
            if typ == "FUT" and restrict_fut_to_nearest_expiry:
                exp = info.get("expiry")
                if fut_target_expiry and (not exp or str(exp)[:10] != fut_target_expiry):
                    # skip this FUT contract because it's not the selected expiry
                    continue
            bars = call_historical(kite, tok, start_date, end_date, "day")
            total_tokens_processed += 1
            time.sleep(pause)
            if not bars:
                continue
            lot = int(instrument_lot_map.get(int(tok), 1))
            for bar in bars:
                dt = bar.get("date") or bar.get("timestamp") or bar.get("datetime")
                if isinstance(dt, str):
                    try:
                        dstr = dt.split("T")[0]
                    except Exception:
                        dstr = dt
                else:
                    try:
                        dstr = str(pd.Timestamp(dt).date())
                    except Exception:
                        dstr = str(dt)

                vol_contracts = int(bar.get("volume") or 0)
                close_price = bar.get("close") or bar.get("last_price") or bar.get("close_price") or 0.0
                try:
                    price_f = float(close_price or 0.0)
                except Exception:
                    price_f = 0.0

                shares = vol_contracts * lot
                notional = shares * price_f

                entry = daily_map.get(dstr)
                if not entry:
                    entry = {
                        "fut_contracts": 0.0, "opt_contracts": 0.0, "fo_contracts": 0.0,
                        "fut_shares": 0.0, "opt_shares": 0.0, "fo_shares": 0.0,
                        "fut_value_inr": 0.0, "opt_value_inr": 0.0, "fo_value_inr": 0.0
                    }
                    daily_map[dstr] = entry

                if typ == "FUT":
                    entry["fut_contracts"] += vol_contracts
                    entry["fut_shares"] += shares
                    entry["fut_value_inr"] += notional
                else:
                    entry["opt_contracts"] += vol_contracts
                    entry["opt_shares"] += shares
                    entry["opt_value_inr"] += notional

                # combined
                entry["fo_contracts"] = entry["fut_contracts"] + entry["opt_contracts"]
                entry["fo_shares"] = entry["fut_shares"] + entry["opt_shares"]
                entry["fo_value_inr"] = entry["fut_value_inr"] + entry["opt_value_inr"]

        # post-process compute averages for combined FO
        for d, metrics in list(daily_map.items()):
            shares = float(metrics.get("fo_shares", 0.0)) or 0.0
            contracts = float(metrics.get("fo_contracts", 0.0)) or 0.0
            value = float(metrics.get("fo_value_inr", 0.0)) or 0.0
            metrics["avg_price_per_share"] = (value / shares) if shares > 0 else 0.0
            metrics["avg_price_per_contract"] = (value / contracts) if contracts > 0 else 0.0
        if daily_map:
            aggregated[underlying] = daily_map
    return aggregated

def save_outputs(out_dir: Path, aggregated: Dict[str, Dict[str, Dict[str, float]]], start_date: datetime.date, end_date: datetime.date):
    safe_mkdir(out_dir)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    json_path = out_dir / f"fo_combined_volume_{start_date.isoformat()}_{end_date.isoformat()}_{stamp}.json"
    csv_path = out_dir / f"fo_combined_volume_{start_date.isoformat()}_{end_date.isoformat()}_{stamp}.csv"
    # write JSON
    try:
        with json_path.open("w", encoding="utf8") as fh:
            json.dump({"start": start_date.isoformat(), "end": end_date.isoformat(), "generated_at": stamp, "data": aggregated}, fh, indent=2)
        logger.info("Saved JSON %s", json_path)
    except Exception as e:
        logger.error("Failed saving JSON: %s", e)
    # write CSV: rows = underlying, date, volume
    try:
        # CSV columns: underlying,date,fut_contracts,opt_contracts,fo_contracts,fut_shares,opt_shares,fo_shares,fut_value_inr,opt_value_inr,fo_value_inr,avg_price_per_share,avg_price_per_contract
        with csv_path.open("w", encoding="utf8") as fh:
            fh.write("underlying,date,fut_contracts,opt_contracts,fo_contracts,fut_shares,opt_shares,fo_shares,fut_value_inr,opt_value_inr,fo_value_inr,avg_price_per_share,avg_price_per_contract\n")
            for ug, daymap in aggregated.items():
                for d, metrics in sorted(daymap.items()):
                    fc = int(metrics.get("fut_contracts", 0))
                    oc = int(metrics.get("opt_contracts", 0))
                    foc = int(metrics.get("fo_contracts", 0))
                    fs = int(metrics.get("fut_shares", 0))
                    os_ = int(metrics.get("opt_shares", 0))
                    fos = int(metrics.get("fo_shares", 0))
                    fv = float(metrics.get("fut_value_inr", 0.0))
                    ov = float(metrics.get("opt_value_inr", 0.0))
                    fov = float(metrics.get("fo_value_inr", 0.0))
                    avg_pps = float(metrics.get("avg_price_per_share", 0.0))
                    avg_ppc = float(metrics.get("avg_price_per_contract", 0.0))
                    fh.write(f"{ug},{d},{fc},{oc},{foc},{fs},{os_},{fos},{fv:.2f},{ov:.2f},{fov:.2f},{avg_pps:.4f},{avg_ppc:.4f}\n")
        logger.info("Saved CSV %s", csv_path)
    except Exception as e:
        logger.error("Failed saving CSV: %s", e)

# CLI
if __name__ == "__main__":
    import argparse
    try:
        import pandas as pd  # optional, for timestamp parsing fallback
    except Exception:
        pd = None

    parser = argparse.ArgumentParser(description="Download combined F&O daily volume")
    parser.add_argument("--api-key", help="Kite API key (or set KITE_API_KEY)")
    parser.add_argument("--access-token", help="Kite access token (or set KITE_ACCESS_TOKEN)")
    parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parents[1] / "results" / "fo_volume"), help="output folder")
    parser.add_argument("--days", type=int, default=1, help="number of days (including today) to fetch")
    parser.add_argument("--pause", type=float, default=0.35, help="pause between historical calls (seconds)")
    parser.add_argument("--max-tokens", type=int, default=0, help="limit tokens processed (0 = no limit)")
    parser.add_argument("--top-n", type=int, default=50, help="number of top symbols from helper to process")
    args = parser.parse_args()

    from kiteconnect import KiteConnect
    import os

    SESSION_PATH = os.path.expanduser("~/.kite_session.json")
    API_KEY = os.getenv("KITE_API_KEY")

    if not API_KEY:
        logger.error("KITE_API_KEY not set in environment or .env")
        raise SystemExit(1)
    if not os.path.exists(SESSION_PATH):
        logger.error("Saved session not found at %s. Run kite.py to authenticate first.", SESSION_PATH)
        raise SystemExit(1)

    sd = json.load(open(SESSION_PATH, "r"))
    token = sd.get("access_token")
    if not token:
        logger.error("access_token missing in saved session %s", SESSION_PATH)
        raise SystemExit(1)

    kite = KiteConnect(api_key=API_KEY)
    kite.set_access_token(token)
    try:
        kite.profile()
        logger.info("Authenticated kite client OK")
    except Exception as e:
        logger.exception("Saved token invalid: %s", e)
        raise SystemExit(1)

    # compute date range
    start_date, end_date = last_n_days_range(max(1, args.days))

    # fetch full NFO instruments once
    try:
        instruments = kite.instruments("NFO")
    except Exception as e:
        logger.error("Failed to fetch instruments('NFO'): %s", e)
        raise SystemExit(1)

    # if helper available, restrict to symbols returned by get_symbols()
    symbols_set = None
    if get_symbols and get_dates_from_most_active_files:
        try:
            dates = get_dates_from_most_active_files()
            if dates:
                syms, _ = get_symbols(dates[-1], top_n=args.top_n)
                if syms:
                    symbols_set = set(s.upper() for s in syms)
                    logger.info("Restricting to %d symbols from get_symbols()", len(symbols_set))
        except Exception:
            symbols_set = None

    if symbols_set:
        # filter instruments to those matching the helper symbols (by 'name' field)
        instruments = [inst for inst in instruments if (inst.get("name") or "").upper() in symbols_set]
        logger.info("Filtered instruments count: %d", len(instruments))

    # build instrument -> lot size map (fallback to 1 if not present)
    instrument_lot_map: Dict[int, int] = {}
    for inst in instruments:
        try:
            tok = inst.get("instrument_token") or inst.get("instrumentToken") or inst.get("token")
            if not tok:
                continue
            # common keys for lot size used in various dumps
            lot = inst.get("lot_size") or inst.get("lot") or inst.get("lotsize") or inst.get("lotSize")
            lot_int = int(lot) if lot is not None else 1
            instrument_lot_map[int(tok)] = lot_int
        except Exception:
            continue

    aggregated = fetch_and_aggregate(kite, instruments, start_date, end_date, instrument_lot_map,
                                     pause=args.pause, max_tokens=(args.max_tokens or None))
    outp = Path(args.out_dir)
    save_outputs(outp, aggregated, start_date, end_date)