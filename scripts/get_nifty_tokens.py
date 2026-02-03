import os
import sys
import json
import logging
from typing import Dict, List
import io

import pandas as pd
import requests
from dotenv import load_dotenv

# ensure project root on path to reuse helpers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_download_kite import _build_instrument_map_from_kite  # noqa: E402

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SESSION_PATH = os.path.expanduser("~/.kite_session.json")
API_KEY = os.getenv("KITE_API_KEY")
if not API_KEY or not os.path.exists(SESSION_PATH):
    raise SystemExit("Set KITE_API_KEY and ensure saved session exists (~/.kite_session.json)")

sd = json.load(open(SESSION_PATH, "r"))
ACCESS_TOKEN = sd.get("access_token")
if not ACCESS_TOKEN:
    raise SystemExit("access_token missing in saved session")

from kiteconnect import KiteConnect  # import after env check

kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(ACCESS_TOKEN)

inst_map = _build_instrument_map_from_kite(kite, exchanges=["NSE_INDICES", "NSE", "NFO", "BSE"])
# normalize keys for lookup
inst_map_norm = {k.upper(): v for k, v in inst_map.items()}

out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "kite", "kite_tokens"))
os.makedirs(out_dir, exist_ok=True)

def find_index_tokens(keywords: List[str]) -> Dict[str, int]:
    """Robustly find index instrument tokens by ensuring all keyword fragments appear in the tradingsymbol."""
    found = {}
    kws = [w.upper() for w in keywords]
    for name, token in inst_map.items():
        k = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in name).upper()
        if all(w in k for w in kws):
            found[name] = token
    return found

def fetch_nse_constituents(index_short: str) -> List[str]:
    """
    Try to download constituency CSV from NSE for index_short:
    index_short examples: 'nifty50' -> ind_nifty50list.csv, 'niftybank' -> ind_niftybanklist.csv
    Returns list of symbols (upper-cased).
    """
    candidates = [
        f"https://archives.nseindia.com/content/indices/ind_{index_short}list.csv",
        f"https://www1.nseindia.com/content/indices/ind_{index_short}list.csv",
        f"https://www.nseindia.com/content/indices/ind_{index_short}list.csv",
    ]
    headers = {"User-Agent": "python-requests/2.x", "Accept": "text/csv, */*"}
    for url in candidates:
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code != 200:
                continue
            txt = resp.text
            df = pd.read_csv(io.StringIO(txt))
            # find column that looks like symbol
            col = None
            for c in df.columns:
                if c.strip().lower() in ("symbol", "symbol " , "scrip code", "code"):
                    col = c
                    break
                if "symbol" in c.lower():
                    col = c
                    break
            if col is None:
                # try first column fallback
                col = df.columns[0]
            syms = df[col].astype(str).str.upper().str.strip().tolist()
            return syms
        except Exception as e:
            logger.debug("failed to fetch %s: %s", url, e)
    return []

# find index tokens
nifty_tokens = find_index_tokens(["NIFTY", "50"]) or find_index_tokens(["NIFTY50", "NIFTY 50"])
bank_tokens = find_index_tokens(["BANK", "NIFTY"]) or find_index_tokens(["BANKNIFTY"])

# fetch constituents
nifty50_symbols = fetch_nse_constituents("nifty50")
banknifty_symbols = fetch_nse_constituents("banknifty")

# Always save constituent symbol lists (even if empty) and mappings
def write_list(lst, fname):
    p = os.path.join(out_dir, fname)
    try:
        with open(p, "w", encoding="utf8") as f:
            for s in lst:
                f.write(s + "\n")
    except Exception:
        logger.exception("Failed to write %s", p)
    return p

# Save raw constituent lists
csv_const_nifty = write_list(nifty50_symbols, "nifty50_constituents.txt")
csv_const_bank = write_list(banknifty_symbols, "banknifty_constituents.txt")

def map_symbols_to_tokens(symbols: List[str]) -> Dict[str, int]:
    mapped = {}
    for s in symbols:
        # try exact match in inst_map (tradingsymbol often equals company symbol)
        t = inst_map_norm.get(s.upper())
        if t is None:
            # try suffix variants (e.g. 'RELIANCE' vs 'RELIANCE.NS' keys)
            for k, v in inst_map_norm.items():
                if k.startswith(s.upper()) or s.upper() in k:
                    t = v
                    break
        if t:
            mapped[s] = t
    return mapped

nifty50_tokens = map_symbols_to_tokens(nifty50_symbols) if nifty50_symbols else {}
banknifty_tokens = map_symbols_to_tokens(banknifty_symbols) if banknifty_symbols else {}

# Save outputs
result = {
    "index_tokens": {
        "nifty50": nifty_tokens,
        "banknifty": bank_tokens
    },
    "constituents": {
        "nifty50": {"symbols": nifty50_symbols, "mapped_tokens": nifty50_tokens},
        "banknifty": {"symbols": banknifty_symbols, "mapped_tokens": banknifty_tokens}
    }
}

json_path = os.path.join(out_dir, "nifty_banknifty_tokens.json")
with open(json_path, "w", encoding="utf8") as f:
    json.dump(result, f, indent=2)

# also write simple CSVs for mapped tokens
def write_csv(mapping: Dict[str, int], fname: str):
    import csv
    p = os.path.join(out_dir, fname)
    with open(p, "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["symbol", "instrument_token"])
        for s, t in sorted(mapping.items()):
            w.writerow([s, t])
    return p

csv_nifty = write_csv(nifty50_tokens, "nifty50_tokens.csv")
csv_bank = write_csv(banknifty_tokens, "banknifty_tokens.csv")
csv_index = write_csv({k: v for k, v in list(nifty_tokens.items()) + list(bank_tokens.items())}, "index_tokens.csv")

print("Saved JSON:", json_path)
print("Saved NIFTY50 tokens CSV:", csv_nifty)
print("Saved BANKNIFTY tokens CSV:", csv_bank)
print("Saved index tokens CSV:", csv_index)
print("Saved NIFTY50 constituents list:", csv_const_nifty)
print("Saved BANKNIFTY constituents list:", csv_const_bank)
print("Folder:", out_dir)