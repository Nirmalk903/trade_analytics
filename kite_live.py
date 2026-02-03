"""
Simple Kite websocket (KiteTicker) example to receive live ticks.

Usage:
  - Set env vars KITE_API_KEY and either KITE_ACCESS_TOKEN or have a saved session at ~/.kite_session.json
  - pip install kiteconnect
  - python kite_live.py --tokens 738561 256265

Notes:
  - Use instrument tokens (integers) for subscribing. Use existing helpers in the repo to map symbol->token if needed.
  - This script prints incoming ticks and can optionally write to a file.
"""
import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from kiteconnect import KiteTicker, KiteConnect
from dotenv import load_dotenv
import datetime
from data_download_vbt import get_symbols, get_dates_from_most_active_files
import subprocess

load_dotenv()

SESSION_PATH = os.path.expanduser("~/.kite_session.json")
CACHE_PATH = os.path.join(os.path.dirname(__file__), "results", "kite", "instruments_cache.json")


def _load_access_token_from_session(path=SESSION_PATH):
    try:
        with open(path, "r", encoding="utf8") as f:
            j = json.load(f)
        return j.get("access_token")
    except Exception:
        return None


def _load_instruments_cache(path=CACHE_PATH):
    try:
        with open(path, "r", encoding="utf8") as f:
            j = json.load(f)
        j["cached_at"] = datetime.datetime.fromisoformat(j["cached_at"])
        return j
    except Exception:
        return None


def _save_instruments_cache(instruments, path=CACHE_PATH):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        j = {"cached_at": datetime.datetime.utcnow().isoformat(), "instruments": instruments}
        with open(path, "w", encoding="utf8") as f:
            json.dump(j, f, default=str)
        return True
    except Exception:
        return False


def _get_all_instruments(api_key: str, access_token: str, exchanges, cache_path=CACHE_PATH, cache_ttl_minutes: int = 1440):
    # Try cache
    if cache_ttl_minutes and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf8") as f:
                j = json.load(f)
            cached_at = datetime.datetime.fromisoformat(j.get("cached_at"))
            age_min = (datetime.datetime.utcnow() - cached_at).total_seconds() / 60.0
            if age_min <= cache_ttl_minutes:
                instruments = j.get("instruments", [])
                print(f"✓ Using cached instruments ({len(instruments)} instruments, age: {int(age_min)}m)")
                return instruments
            else:
                print(f"Cache expired (age: {int(age_min)}m > TTL: {cache_ttl_minutes}m), fetching fresh data...")
        except Exception as e:
            print(f"Cache load failed ({e}), fetching fresh data...")

    print(f"Fetching instruments from Kite API for exchanges: {', '.join(exchanges)}")
    kc = KiteConnect(api_key=api_key)
    kc.set_access_token(access_token)
    instruments = []
    for ex in exchanges:
        try:
            insts = kc.instruments(ex)
            instruments.extend(insts)
            print(f"  ✓ {ex}: {len(insts)} instruments")
        except Exception as e:
            print(f"  ✗ {ex}: failed ({e})")
            continue

    try:
        _save_instruments_cache(instruments, cache_path)
        print(f"✓ Cached {len(instruments)} instruments to {cache_path}")
    except Exception as e:
        print(f"Failed to save cache: {e}")
    return instruments


def resolve_symbols_to_tokens(api_key: str, access_token: str, symbols, exchanges=None, cache_path=CACHE_PATH, cache_ttl_minutes: int = 1440):
    """Resolve a list of symbols to instrument tokens using Kite instruments().

    Uses a local JSON cache to avoid repeated / repeated HTTP calls.

    Returns (resolved_tokens:list[int], not_found:list[str])
    """
    exchanges = exchanges or ["NSE", "NFO", "BSE", "NSE_INDICES"]

    instruments = _get_all_instruments(api_key, access_token, exchanges, cache_path=cache_path, cache_ttl_minutes=cache_ttl_minutes)

    token_map = {}  # maps lowercase symbol/name/tradingsymbol -> instrument_token
    for inst in instruments:
        tok = inst.get("instrument_token")
        if not tok:
            continue
        for key in (inst.get("tradingsymbol"), inst.get("name"), inst.get("symbol")):
            if key:
                token_map[key.lower()] = tok

    resolved = []
    not_found = []
    for s in symbols:
        k = s.lower()
        if k in token_map:
            resolved.append(token_map[k])
            continue
        # try prefix match
        matches = [v for kk, v in token_map.items() if kk.startswith(k)]
        if len(matches) == 1:
            resolved.append(matches[0])
        elif len(matches) > 1:
            print(f"Multiple matches for {s}; using first match")
            resolved.append(matches[0])
        else:
            not_found.append(s)
    return resolved, not_found


def main():
    parser = argparse.ArgumentParser(description="Kite live ticker example")
    parser.add_argument("--api-key", help="Kite API key (or set KITE_API_KEY env var)")
    parser.add_argument("--access-token", help="Kite access token (or set KITE_ACCESS_TOKEN env var)")
    parser.add_argument("--tokens", nargs="+", type=int, help="Instrument tokens to subscribe")
    parser.add_argument("--symbols", nargs="+", help="Instrument symbols to resolve to tokens (e.g., RELIANCE, NIFTY23JANFUT)")
    parser.add_argument("--exchange", nargs="+", default=["NSE","NFO","BSE","NSE_INDICES"], help="Exchanges to search (defaults to NSE,NFO,BSE,NSE_INDICES)")
    parser.add_argument("--cache-ttl", type=int, default=1440, help="Cache TTL in minutes (default 1440 = 1 day). Set 0 to always refresh.")
    parser.add_argument("--outfile", help="Path to jsonl file to append ticks")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("KITE_API_KEY")
    access_token = args.access_token or os.environ.get("KITE_ACCESS_TOKEN") or _load_access_token_from_session()

    if not api_key or not access_token:
        raise SystemExit("KITE_API_KEY and KITE_ACCESS_TOKEN (or saved session) required")

    tokens = args.tokens or []
    # Resolve symbols if provided and extend tokens
    if args.symbols:
        print("Resolving symbols to tokens:", args.symbols)
        resolved, not_found = resolve_symbols_to_tokens(api_key, access_token, args.symbols, exchanges=args.exchange, cache_path=CACHE_PATH, cache_ttl_minutes=args.cache_ttl)
        if resolved:
            tokens = list(tokens) + resolved
            print(f"Resolved tokens: {resolved}")
        if not_found:
            print(f"Symbols not found: {not_found}")

    if not tokens:
        raise SystemExit("No tokens to subscribe. Provide --tokens and/or --symbols that resolve to tokens.")

    outfile = args.outfile
    if outfile:
        Path(outfile).parent.mkdir(parents=True, exist_ok=True)

    print("Connecting KiteTicker...")
    kws = KiteTicker(api_key, access_token)

    def on_ticks(ws, ticks):
        # ticks is a list of tick dicts
        for t in ticks:
            print(t)
            if outfile:
                with open(outfile, "a", encoding="utf8") as f:
                    f.write(json.dumps(t, default=str) + "\n")

    def on_connect(ws, response):
        print("Connected: subscribing to tokens:", tokens)
        ws.subscribe(tokens)
        # MODE_FULL gives all fields, MODE_LTP only ltp. Change as needed.
        try:
            ws.set_mode(ws.MODE_FULL, tokens)
        except Exception:
            pass

    def on_error(ws, code, reason):
        print("Error:", code, reason)

    def on_close(ws, code, reason):
        print("Closed:", code, reason)

    def on_reconnect(ws, attempts):
        print(f"Reconnecting... attempts={attempts}")

    kws.on_ticks = on_ticks
    kws.on_connect = on_connect
    kws.on_error = on_error
    kws.on_close = on_close
    kws.on_reconnect = on_reconnect

    # run in threaded mode so Ctrl+C works in console
    kws.connect(threaded=True)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Interrupted: stopping")
        kws.disconnect()
    latest_date = get_dates_from_most_active_files()[-1]
    symbols = get_symbols(latest_date)[0]
    subprocess.run(['python', 'kite_live.py', '--symbols'] + symbols)


if __name__ == "__main__":
    main()

from kiteconnect import KiteConnect
import os, json
api_key = os.environ["KITE_API_KEY"]
access_token = os.environ["KITE_ACCESS_TOKEN"]
kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)
instruments = kite.instruments("NSE")
print([i["tradingsymbol"] for i in instruments if "NIFTY" in i["tradingsymbol"]])
