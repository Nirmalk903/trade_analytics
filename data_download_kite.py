import os
import sys
import json
import time
import glob
import logging
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional
import json as _json

import pandas as pd
from dotenv import load_dotenv

# allow local imports (project root)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_download_vbt import get_symbols, get_dates_from_most_active_files  # noqa: E402

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _build_instrument_map_from_kite(kite_client, exchanges=None) -> dict:
    # include NSE_INDICES so index tokens (NIFTY/BANKNIFTY) are discovered
    exchanges = exchanges or ["NSE", "BSE", "NFO", "MCX", "NSE_INDICES"]
    inst_map = {}
    for ex in exchanges:
        try:
            instruments = kite_client.instruments(ex)
        except Exception:
            continue
        for inst in instruments:
            ts = inst.get("tradingsymbol") or inst.get("instrumentName") or inst.get("symbol")
            token = inst.get("instrument_token") or inst.get("token")
            if not ts or not token:
                continue
            # normalize symbol keys for robust lookup
            try:
                tok_int = int(token)
            except Exception:
                tok_int = token
            norm_variants = set()
            t_up = str(ts).upper().strip()
            norm_variants.add(t_up)                       # "NIFTY 50"
            norm_variants.add(t_up.replace(" ", ""))     # "NIFTY50"
            norm_variants.add(t_up.replace(".", "").replace("-", ""))  # remove punctuation
            # also map simple alphanumeric only
            norm_variants.add("".join(ch for ch in t_up if ch.isalnum()))
            # store all variants pointing to same token
            for k in norm_variants:
                if k:
                    inst_map[k] = tok_int
            # keep original-case key too (for backwards compatibility)
            inst_map[str(ts)] = tok_int
    return inst_map


def _normalize_historical_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # If API returned a nested 'ohlc' object, expand it into columns
    if "ohlc" in df.columns:
        try:
            ohlc_expanded = pd.json_normalize(df["ohlc"]).applymap(lambda v: v if pd.notna(v) else None)
            # ensure lowercase names
            ohlc_expanded.columns = [c.lower() for c in ohlc_expanded.columns]
            # keep only typical keys if present
            keep = [c for c in ["open", "high", "low", "close"] if c in ohlc_expanded.columns]
            if keep:
                df = pd.concat([df.drop(columns=["ohlc"]), ohlc_expanded[keep]], axis=1)
            else:
                df = df.drop(columns=["ohlc"])
        except Exception:
            # if parsing fails, drop the column to avoid later errors
            df = df.drop(columns=["ohlc"])

    # unify timestamp
    if "date" in df.columns and "date_time" not in df.columns:
        df = df.rename(columns={"date": "date_time"})
    if "date_time" in df.columns:
        df["date_time"] = pd.to_datetime(df["date_time"])
    else:
        for c in ("tradable_timestamp", "timestamp", "time"):
            if c in df.columns:
                df = df.rename(columns={c: "date_time"})
                df["date_time"] = pd.to_datetime(df["date_time"])
                break

    # map OHLC/price/volume variants (handle casing)
    if "close" in df.columns and "price" not in df.columns:
        df = df.rename(columns={"close": "price"})
    if "volume_traded" in df.columns and "volume" not in df.columns:
        df = df.rename(columns={"volume_traded": "volume"})
    for col in ("open", "high", "low"):
        if col not in df.columns:
            if col.capitalize() in df.columns:
                df = df.rename(columns={col.capitalize(): col})
            elif col.upper() in df.columns:
                df = df.rename(columns={col.upper(): col})

    # if OHLC available but price missing, use close/open as fallback
    if "price" not in df.columns and all(c in df.columns for c in ("open", "high", "low", "close")):
        df["price"] = df["close"]
    elif "price" not in df.columns and all(c in df.columns for c in ("open", "high", "low")):
        df["price"] = df["open"]

    # keep only useful columns in a consistent order
    candidate = ["date_time", "open", "high", "low", "price", "volume", "oi", "close"]
    keep = [c for c in candidate if c in df.columns]
    return df.loc[:, keep]


def _load_index_tokens_from_file():
    """Load saved index tokens if present (tries matches.json or nifty_banknifty_tokens.json)."""
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "results", "kite", "kite_tokens"))
    candidates = [
        os.path.join(base, "matches.json"),
        os.path.join(base, "nifty_banknifty_tokens.json"),
        os.path.join(base, "matches.csv"),  # fallback if JSON not present
    ]
    p = None
    for c in candidates:
        if os.path.exists(c):
            p = c
            break
    if p is None:
        return {}
    try:
        with open(p, "r", encoding="utf8") as f:
            try:
                j = _json.load(f)
            except Exception:
                # if CSV or badly-formed JSON, return empty
                return {}
        out = {}
        # prefer top-level mapped token dicts if present
        idx = j.get("index_tokens") or {}
        # index_tokens may be {"nifty50": {"NIFTY 50": token, ...}, "banknifty": {...}}
        for kname, val in idx.items():
            if isinstance(val, dict):
                # take first token value
                for nm, tok in val.items():
                    out[kname.upper().replace(" ", "")] = int(tok)
                    break
            elif isinstance(val, (int, str)):
                try:
                    out[kname.upper().replace(" ", "")] = int(val)
                except Exception:
                    pass
        # also accept constituents mapped_tokens area
        const = j.get("constituents") or {}
        if "nifty50" in const and isinstance(const["nifty50"].get("mapped_tokens"), dict):
            # prefer explicit mapped token for index constituents list (not used here but load if needed)
            pass
        return out
    except Exception:
        logger.debug("Failed to load index tokens file %s", p, exc_info=True)
        return {}

# load index tokens once per run
_INDEX_TOKENS = _load_index_tokens_from_file()


def _load_matches_map():
    """Load matches.json -> mapping normalised_tradingsymbol -> token."""
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "results", "kite", "kite_tokens"))
    p = os.path.join(base, "matches.json")
    if not os.path.exists(p):
        return {}
    try:
        with open(p, "r", encoding="utf8") as f:
            j = _json.load(f)
        entries = j if isinstance(j, list) else (j.get("matches") or j.get("data") or [])
        # entries may also be list of {"tradingsymbol":..., "instrument_token":...}
        out = {}
        if isinstance(entries, list):
            for it in entries:
                try:
                    ts = (it.get("tradingsymbol") if isinstance(it, dict) else None) or (it[0] if isinstance(it, (list, tuple)) and len(it) > 0 else None)
                    tok = (it.get("instrument_token") if isinstance(it, dict) else None) or (it[1] if isinstance(it, (list, tuple)) and len(it) > 1 else None)
                    if ts and tok:
                        key = "".join(ch for ch in str(ts).upper() if ch.isalnum())
                        out[key] = int(tok)
                except Exception:
                    continue
        return out
    except Exception:
        logger.debug("Failed to load matches.json at %s", p, exc_info=True)
        return {}

# load matches map for direct index lookups
_MATCHES_MAP = _load_matches_map()


def _to_aware_utc(dt):
    """Return a timezone-aware UTC datetime for comparison/fetching."""
    if dt is None:
        return None
    # pandas Timestamp
    try:
        if isinstance(dt, pd.Timestamp):
            dt = dt.to_pydatetime()
    except Exception:
        pass
    # parse strings or other objects via pandas for robustness
    if not isinstance(dt, datetime):
        try:
            dt = pd.to_datetime(dt).to_pydatetime()
        except Exception:
            # last resort: return None
            return None
    if dt.tzinfo is None:
        # assume naive times are local/IST was observed earlier; treat as UTC for safety
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def fetch_raw_price_volume_from_kite(
    kite_client,
    symbols: Iterable[str],
    out_root: Optional[str] = None,
    interval: str = "day",
    lookback_days: int = 365,
    rate_limit_sleep: float = 0.35,
    refetch_full_history: bool = False,
) -> None:
    out_root = out_root or os.path.abspath(r"C:\Users\nirma\OneDrive\MyProjects\trading_app\Underlying_data_kite\daily")
    os.makedirs(out_root, exist_ok=True)

    logger.info("Building instrument map via Kite instruments() ...")
    inst_map = _build_instrument_map_from_kite(kite_client)
    logger.info("Instrument map size: %d", len(inst_map))

    # use timezone-aware UTC for all comparisons
    now = datetime.now(timezone.utc)

    for sym in symbols:
        try:
            # resolve token robustly:
            token = None
            if isinstance(sym, int):
                token = sym
            else:
                # numeric string -> use as token
                try:
                    token = int(str(sym))
                except Exception:
                    # normalized key (alphanumeric uppercase) for matches.json lookup
                    norm_key = "".join(ch for ch in str(sym).upper() if ch.isalnum())
                    if norm_key and _MATCHES_MAP and norm_key in _MATCHES_MAP:
                        token = _MATCHES_MAP[norm_key]
                    else:
                        # try instrument map lookups (several normalized variants)
                        key_raw = str(sym)
                        key_up = key_raw.upper()
                        key_nospace = key_up.replace(" ", "")
                        key_alnum = "".join(ch for ch in key_up if ch.isalnum())
                        token = inst_map.get(key_raw) or inst_map.get(key_up) or inst_map.get(key_nospace) or inst_map.get(key_alnum)
                        # fallback to index tokens file if still unresolved
                        if token is None:
                            if key_alnum in _INDEX_TOKENS:
                                token = _INDEX_TOKENS[key_alnum]
                            else:
                                if "NIFTY" in key_alnum and "50" in key_alnum:
                                    token = _INDEX_TOKENS.get("NIFTY50") or _INDEX_TOKENS.get("NIFTY")
                                if token is None and ("BANK" in key_alnum and "NIFTY" in key_alnum):
                                    token = _INDEX_TOKENS.get("BANKNIFTY") or _INDEX_TOKENS.get("BANK")

            if token is None:
                logger.warning("[%s] instrument token not found; skipping", sym)
                continue

            # ensure token is int
            try:
                token = int(token)
            except Exception:
                logger.warning("[%s] token not integer: %r; skipping", sym, token)
                continue

            # save using convention: {symbol}_1d.csv for daily ("day") interval, otherwise use interval as suffix
            clean_sym = str(sym).replace(" ", "_")
            suffix = "1d" if str(interval).lower() in ("day", "1d") else str(interval)
            out_path = os.path.join(out_root, f"{clean_sym}_{suffix}.csv")

            if os.path.exists(out_path):
                try:
                    prev = pd.read_csv(out_path, parse_dates=["date_time"])
                    last_dt = prev["date_time"].max()
                    # add one second and convert to aware UTC
                    start = pd.to_datetime(last_dt) + pd.Timedelta(seconds=1)
                    start = _to_aware_utc(start)
                except Exception:
                    start = _to_aware_utc(now - timedelta(days=lookback_days))
            else:
                start = _to_aware_utc(now - timedelta(days=lookback_days))

            end = _to_aware_utc(now)
            logger.info("[%s] fetching historical %s -> %s (interval=%s)", sym, start, end, interval)

            # ensure token and dates
            try:
                token = int(token)
            except Exception:
                logger.warning("[%s] instrument token is not integer: %r", sym, token)

            # normalize any remaining start/end representations to timezone-aware UTC
            start = _to_aware_utc(start)
            end = _to_aware_utc(end)

            rows = []
            # some providers reject huge spans — chunk into 1-year pieces
            max_days_per_call = 365
            cur_start = start
            while cur_start is not None and end is not None and cur_start < end:
                cur_end = min(end, cur_start + timedelta(days=max_days_per_call))
                logger.debug("[%s] fetching chunk %s -> %s", sym, cur_start, cur_end)
                try:
                    # try keyword signature first
                    chunk = kite_client.historical_data(instrument_token=token, from_date=cur_start, to_date=cur_end, interval=interval)
                except TypeError:
                    try:
                        chunk = kite_client.historical_data(token, cur_start, cur_end, interval)
                    except Exception as e:
                        logger.exception("[%s] kite.historical_data (positional) failed: %s", sym, e)
                        chunk = []
                except Exception as e:
                    # log full exception to help debugging (requests/KiteException include response text)
                    logger.exception("[%s] kite.historical_data failed: %s", sym, e)
                    chunk = []

                if chunk:
                    # some calls may return dict with 'data' key or list directly
                    if isinstance(chunk, dict) and "data" in chunk:
                        chunk_rows = chunk["data"]
                    else:
                        chunk_rows = list(chunk)
                    rows.extend(chunk_rows)

                # advance start for next chunk (keep timezone-aware)
                cur_start = _to_aware_utc(cur_end + timedelta(seconds=1))
                # small polite sleep to avoid rate limits
                time.sleep(rate_limit_sleep)
            # dedupe rows by timestamp in case of overlap
            if rows:
                try:
                    temp = pd.DataFrame(rows)
                    if "date" in temp.columns and "date_time" not in temp.columns:
                        temp = temp.rename(columns={"date": "date_time"})
                    if "date_time" in temp.columns:
                        temp["date_time"] = pd.to_datetime(temp["date_time"])
                        temp.drop_duplicates(subset=["date_time"], keep="last", inplace=True)
                        rows = temp.to_dict(orient="records")
                except Exception:
                    # if normalization fails, leave rows as-is
                    pass

            if not rows:
                logger.info("[%s] no new rows returned", sym)
                time.sleep(rate_limit_sleep)
                continue

            df_new = pd.DataFrame(rows)
            df_new = _normalize_historical_frame(df_new)

            if df_new.empty or "date_time" not in df_new.columns:
                logger.warning("[%s] fetched frame empty or missing date_time -> skipping", sym)
                time.sleep(rate_limit_sleep)
                continue

            df_new.sort_values("date_time", inplace=True)
            df_new.set_index("date_time", inplace=True)

            # append/merge with smarter update: prefer df_new values for matching timestamps,
            # and append rows that didn't exist previously. Optionally re-fetch full history to backfill.
            if os.path.exists(out_path) and not refetch_full_history:
                try:
                    prev = pd.read_csv(out_path, parse_dates=["date_time"]).set_index("date_time")
                    # Update existing rows with non-null values from df_new
                    prev.update(df_new)
                    # find rows that are new (present in df_new but not in prev)
                    new_idx = df_new.index.difference(prev.index)
                    if len(new_idx):
                        appended = df_new.loc[new_idx]
                        combined = pd.concat([prev, appended], axis=0)
                    else:
                        combined = prev
                    combined = combined.sort_index()
                    combined.to_csv(out_path, index=True)
                    # write with index as date_time column
                    # ensure file has column name 'date_time' not index name
                    # (re-save to have date_time as column)
                    combined.reset_index().to_csv(out_path, index=False)
                    logger.info("[%s] merged/updated %d rows -> %s (total %d)", sym, len(df_new), out_path, len(combined))
                except Exception as e:
                    logger.exception("[%s] failed to merge/update historical file: %s; falling back to overwrite", sym, e)
                    try:
                        df_new.reset_index().to_csv(out_path, index=False)
                        logger.info("[%s] overwrote historical file with %d rows -> %s", sym, len(df_new), out_path)
                    except Exception as e2:
                        logger.exception("[%s] failed to write historical file: %s", sym, e2)
            else:
                # either no existing file, or user requested full re-fetch (refetch_full_history=True)
                try:
                    # if refetch_full_history is True and file exists, we overwrite with df_new (which should be full range)
                    df_new.reset_index().to_csv(out_path, index=False)
                    logger.info("[%s] wrote historical file %d rows -> %s", sym, len(df_new), out_path)
                except Exception as e:
                    logger.exception("[%s] failed to write historical file: %s", sym, e)

            time.sleep(rate_limit_sleep)

        except Exception as exc:
            logger.exception("Unexpected error processing %s: %s", sym, exc)
            time.sleep(rate_limit_sleep)

    logger.info("Historical fetch complete. Files saved under %s", out_root)


if __name__ == "__main__":
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

    # pick date from most-active files if available, otherwise try latest file or today
    
    dates = get_dates_from_most_active_files() or []
    symbols, meta = get_symbols(dates[-1] if dates else None, top_n=50)
    

    symbols, meta = get_symbols(sd_arg, top_n=50)
    logger.info("get_symbols returned %d symbols (date=%s)", len(symbols), sd_arg)
    if not symbols:
        logger.error("No symbols returned by get_symbols; nothing to fetch")
        raise SystemExit(1)

    # include index keys so these get resolved via saved matches.json
    symbols = list(symbols) + ["NIFTY50", "NIFTYBANK"]

    out_root = os.path.abspath(r"C:\Users\nirma\OneDrive\MyProjects\trading_app\Underlying_data_kite\daily")
    fetch_raw_price_volume_from_kite(kite, symbols, out_root=out_root, interval="day", lookback_days=3650)
    logger.info("Done.")