import os, json, sys
from dotenv import load_dotenv

# ensure project root is on sys.path so local modules can be imported when running the script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from kiteconnect import KiteConnect
from data_download_kite import _build_instrument_map_from_kite  # reuse helper
load_dotenv()

SESSION_PATH = os.path.expanduser("~/.kite_session.json")
API_KEY = os.getenv("KITE_API_KEY") or os.getenv("KITE_KEY")
print(f"DEBUG: SESSION_PATH={SESSION_PATH}")
print(f"DEBUG: KITE_API_KEY present: {bool(API_KEY)}")
if not API_KEY:
    raise SystemExit("Set KITE_API_KEY in environment or .env (KITE_API_KEY).")
if not os.path.exists(SESSION_PATH):
    raise SystemExit(f"Saved session not found at {SESSION_PATH}. Run kite.py to authenticate first.")

sd = json.load(open(SESSION_PATH))
token = sd.get("access_token")
kite = KiteConnect(api_key=API_KEY)
kite.set_access_token(token)

# build map and search
# inst_map = _build_instrument_map_from_kite(kite, exchanges=["NSE","NFO","BSE","MCX","NSE_INDICES"])
inst_map = _build_instrument_map_from_kite(kite, exchanges=["NSE"])
search_terms = ["banknifty", "nifty 50",]
matches = []
for tradingsymbol, instrument_token in inst_map.items():
    key = tradingsymbol.lower()
    if any(term in key for term in search_terms):
        matches.append((tradingsymbol, instrument_token))

if not matches:
    print("No matches found. Try inspecting a subset of instruments (kite.instruments) or adjust search terms.")
else:
    print("Matches (tradingsymbol, instrument_token):")
    for s, t in sorted(matches):
        print(s, t)

    # Save outputs to results/kite/kite_tokens (project-root/results/kite/kite_tokens)
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results", "kite", "kite_tokens"))
    print("DEBUG: Saving matches to:", out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # JSON
    json_path = os.path.join(out_dir, "matches.json")
    with open(json_path, "w", encoding="utf8") as f:
        json.dump([{"tradingsymbol": s, "instrument_token": t} for s, t in sorted(matches)], f, indent=2)
    # CSV
    csv_path = os.path.join(out_dir, "matches.csv")
    try:
        import csv
        with open(csv_path, "w", newline="", encoding="utf8") as f:
            w = csv.writer(f)
            w.writerow(["tradingsymbol", "instrument_token"])
            for s, t in sorted(matches):
                w.writerow([s, t])
    except Exception:
        pass

    print("Saved matches to:", json_path, "and", csv_path)