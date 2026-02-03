"""
Download option chains from Kite and save per-symbol JSON files under
trading_app/OptionChainJSON_Kite.

Features / robustness improvements:
- Defensive imports and clear error messages if kiteconnect missing.
- Safe path handling and directory creation.
- Retries for network calls with backoff.
- Chunked LTP requests to avoid oversized calls.
- Prefer symbols from data_download_vbt.get_symbols, fallback to scanning folders.
- Select expiry: last Tuesday of current month (fallback to same month expiry).
- Clear logging and CLI.
"""
from __future__ import annotations

import os
import sys
import json
import time
import calendar
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone, timedelta

# Configure logger (simple, safe to reuse)
logger = logging.getLogger("optionchain_kite")
if not logger.handlers:
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)

# Optional helper imports (best-effort)
try:
    from kiteconnect import KiteConnect  # type: ignore
except Exception:
    KiteConnect = None  # type: ignore

try:
    from data_download_vbt import get_symbols, get_dates_from_most_active_files  # type: ignore
except Exception:
    get_symbols = None
    get_dates_from_most_active_files = None


# ----------------------
# Utilities
# ----------------------
def last_weekday_of_month(year: int, month: int, weekday: int) -> datetime:
    """Return UTC datetime of the last given weekday (0=Mon..6=Sun) for the month."""
    last_day = calendar.monthrange(year, month)[1]
    dt = datetime(year, month, last_day, tzinfo=timezone.utc)
    while dt.weekday() != weekday:
        dt -= timedelta(days=1)
    return dt


def safe_mkdir(p: Path) -> Path:
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.warning("Could not create directory %s: %s", p, e)
    return p


def discover_symbols(top_n: int = 50) -> List[str]:
    """Try get_symbols() helper, else scan Underlying_data_kite/Engineered_data for files."""
    syms: List[str] = []
    if get_symbols and get_dates_from_most_active_files:
        try:
            dates = get_dates_from_most_active_files()
            if dates:
                syms, _ = get_symbols(dates[-1], top_n=top_n)
                if syms:
                    return [s.upper() for s in syms]
        except Exception:
            logger.debug("get_symbols() helper failed", exc_info=True)
    # fallback scanning folders
    root = Path(__file__).resolve().parents[1]
    candidates = [root / "Underlying_data_kite", root / "Engineered_data"]
    names = set()
    for d in candidates:
        try:
            if d.exists() and d.is_dir():
                for f in d.iterdir():
                    if f.suffix.lower() in {".csv", ".json"}:
                        base = f.stem
                        sym = base.split("_")[0] if "_" in base else base
                        names.add(sym.upper())
        except Exception:
            continue
    return sorted(names)


# ----------------------
# Kite / instrument helpers
# ----------------------
def instruments_nfo(kite: Any) -> List[Dict[str, Any]]:
    """Fetch instruments('NFO') with retries."""
    attempts = 3
    for i in range(attempts):
        try:
            return kite.instruments("NFO")
        except Exception as e:
            logger.warning("instruments('NFO') failed (attempt %d/%d): %s", i + 1, attempts, e)
            time.sleep(0.5 * (i + 1))
    return []


def group_option_chain_by_expiry(instruments: List[Dict[str, Any]], underlying: str) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """
    Group option instruments by expiry (ISO date string) then by strike:
      { 'YYYY-MM-DD': { strike: { 'CE': {...}, 'PE': {...} } } }
    Only includes instruments whose 'name' matches underlying (case-insensitive).
    """
    out: Dict[str, Dict[int, Dict[str, Any]]] = {}
    ux = (underlying or "").upper()
    for inst in instruments:
        try:
            if (inst.get("instrument_type") or "").upper() != "OPT":
                continue
            name = (inst.get("name") or "").upper()
            if name != ux:
                continue
            expiry = inst.get("expiry")
            if not expiry:
                continue
            strike = int(float(inst.get("strike", 0)))
            opt_type = (inst.get("option_type") or "").upper()
            if expiry not in out:
                out[expiry] = {}
            if strike not in out[expiry]:
                out[expiry][strike] = {}
            out[expiry][strike][opt_type] = {
                "instrument_token": inst.get("instrument_token"),
                "tradingsymbol": inst.get("tradingsymbol"),
                "expiry": expiry,
                "strike": strike,
                "option_type": opt_type,
            }
        except Exception:
            continue
    return out


def chunks(lst: List[Any], n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def fetch_ltp_map(kite: Any, tokens: List[int], chunk_size: int = 100, pause: float = 0.2) -> Dict[int, Dict[str, Any]]:
    """Fetch LTP/oi/volume for tokens in batches. Returns map token->info."""
    out: Dict[int, Dict[str, Any]] = {}
    if not tokens:
        return out
    for batch in chunks(tokens, chunk_size):
        attempts = 2
        for attempt in range(attempts):
            try:
                resp = kite.ltp(batch)
                for k, v in resp.items():
                    try:
                        tok = int(str(k).split(":")[-1])
                    except Exception:
                        continue
                    out[tok] = {
                        "last_price": v.get("last_price"),
                        "oi": v.get("oi"),
                        "volume": v.get("volume"),
                        "depth": v.get("depth"),
                    }
                break
            except Exception as e:
                logger.debug("ltp() failed for batch (attempt %d/%d): %s", attempt + 1, attempts, e)
                time.sleep(0.25 * (attempt + 1))
        time.sleep(pause)
    return out


# ----------------------
# Output
# ----------------------
def save_option_chain_json(out_dir: Path, underlying: str, expiry: str, chain: Dict[int, Dict[str, Any]], ltp_map: Dict[int, Dict[str, Any]]):
    out_dir = safe_mkdir(out_dir)
    rows = []
    for strike in sorted(chain.keys()):
        rec = chain[strike]
        ce = rec.get("CE")
        pe = rec.get("PE")
        ce_tok = int(ce["instrument_token"]) if ce and ce.get("instrument_token") else None
        pe_tok = int(pe["instrument_token"]) if pe and pe.get("instrument_token") else None
        rows.append({
            "strike": strike,
            "ce": {
                "tradingsymbol": ce.get("tradingsymbol") if ce else None,
                "token": ce_tok,
                "ltp": ltp_map.get(ce_tok, {}).get("last_price") if ce_tok else None,
                "oi": ltp_map.get(ce_tok, {}).get("oi") if ce_tok else None,
                "volume": ltp_map.get(ce_tok, {}).get("volume") if ce_tok else None,
            },
            "pe": {
                "tradingsymbol": pe.get("tradingsymbol") if pe else None,
                "token": pe_tok,
                "ltp": ltp_map.get(pe_tok, {}).get("last_price") if pe_tok else None,
                "oi": ltp_map.get(pe_tok, {}).get("oi") if pe_tok else None,
                "volume": ltp_map.get(pe_tok, {}).get("volume") if pe_tok else None,
            }
        })
    payload = {
        "underlying": underlying,
        "expiry": expiry,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
    }
    fname = f"{underlying}_{expiry}_OptionChain.json".replace(":", "-")
    out_path = out_dir / fname
    try:
        with out_path.open("w", encoding="utf8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        logger.info("Saved option chain: %s", out_path)
    except Exception as e:
        logger.error("Failed to write JSON %s: %s", out_path, e)


# ----------------------
# Main flow
# ----------------------
def main(api_key: Optional[str] = None,
         access_token: Optional[str] = None,
         out_base: Optional[Path] = None,
         top_n: int = 50):
    if KiteConnect is None:
        logger.error("kiteconnect not installed. Install via: pip install kiteconnect")
        return

    api_key = api_key or os.environ.get("KITE_API_KEY")
    access_token = access_token or os.environ.get("KITE_ACCESS_TOKEN")
    if not api_key or not access_token:
        logger.error("API key and access token required (args or env KITE_API_KEY / KITE_ACCESS_TOKEN)")
        return

    kite = KiteConnect(api_key=api_key)
    try:
        kite.set_access_token(access_token)
    except Exception as e:
        logger.error("Failed to set access token: %s", e)
        return

    symbols = discover_symbols(top_n=top_n)
    if not symbols:
        logger.error("No symbols discovered; aborting")
        return

    instruments = instruments_nfo(kite)
    if not instruments:
        logger.error("Failed to fetch NFO instruments; aborting")
        return

    now = datetime.now(timezone.utc)
    last_tuesday_date = last_weekday_of_month(now.year, now.month, weekday=1).date()

    out_base = out_base or (Path(__file__).resolve().parents[1] / "OptionChainJSON_Kite")
    out_base = out_base.resolve()
    safe_mkdir(out_base)

    logger.info("Downloading option chains for %d symbols; saving to %s", len(symbols), out_base)
    for sym in symbols:
        try:
            logger.info("Processing %s", sym)
            grouped = group_option_chain_by_expiry(instruments, sym)
            if not grouped:
                logger.debug("No option instruments for %s", sym)
                continue

            target_expiry = str(last_tuesday_date.isoformat())
            if target_expiry not in grouped:
                # pick any expiry in same month/year
                for e in grouped.keys():
                    try:
                        d = datetime.fromisoformat(e).date()
                        if d.year == now.year and d.month == now.month:
                            target_expiry = e
                            break
                    except Exception:
                        continue
            if target_expiry not in grouped:
                logger.warning("[%s] no suitable expiry found for current month; skipping", sym)
                continue

            chain = grouped[target_expiry]
            tokens: List[int] = []
            for st, rec in chain.items():
                for ot in ("CE", "PE"):
                    v = rec.get(ot)
                    if v and v.get("instrument_token"):
                        try:
                            tokens.append(int(v["instrument_token"]))
                        except Exception:
                            continue
            tokens = sorted(set(tokens))
            ltp_map = fetch_ltp_map(kite, tokens, chunk_size=100, pause=0.25)
            save_option_chain_json(out_base, sym, target_expiry, chain, ltp_map)
            time.sleep(0.15)
        except Exception as e:
            logger.exception("Failed symbol %s: %s", sym, e)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download option chains from Kite and save JSON files")
    parser.add_argument("--api-key", help="Kite API key (or set KITE_API_KEY)")
    parser.add_argument("--access-token", help="Kite access token (or set KITE_ACCESS_TOKEN)")
    parser.add_argument("--out-dir", help="Output folder (defaults to trading_app/OptionChainJSON_Kite)")
    parser.add_argument("--top-n", type=int, default=50, help="Number of symbols to fetch from helper")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else None
    main(api_key=args.api_key, access_token=args.access_token, out_base=out_dir, top_n=args.top_n)