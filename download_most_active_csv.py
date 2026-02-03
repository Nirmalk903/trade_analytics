import os
import requests
import time
from datetime import datetime
from typing import Optional

# Output directory
OUT_DIR = os.path.join(os.path.dirname(__file__), 'Most_Active_Underlying')
os.makedirs(OUT_DIR, exist_ok=True)

# NSE endpoints
BASE_URL = "https://www.nseindia.com"
PAGE_URL = "https://www.nseindia.com/market-data/most-active-underlying"
API_URL = "https://www.nseindia.com/api/live-analysis-most-active-underlying"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Referer": PAGE_URL,
    "Origin": BASE_URL,
    "Connection": "keep-alive",
}

def download_most_active_csv(timeout: float = 15.0) -> Optional[str]:
    session = requests.Session()
    session.headers.update(HEADERS)

    # Preflight: get cookies by visiting the main page
    try:
        session.get(BASE_URL, timeout=timeout)
        time.sleep(0.5)
        session.get(PAGE_URL, timeout=timeout)
        time.sleep(0.5)
    except Exception as e:
        print(f"[ERROR] Could not preflight cookies: {e}")
        return None

    # Try API endpoint
    try:
        resp = session.get(API_URL, timeout=timeout)
        if resp.status_code == 200 and resp.headers.get('Content-Type', '').startswith('application/json'):
            data = resp.json()
            if isinstance(data, dict) and 'data' in data and isinstance(data['data'], list):
                import pandas as pd
                df = pd.DataFrame(data['data'])
                # Set required column names explicitly
                required_columns = [
                    "Symbol",
                    "Volume (Contracts) - Futures",
                    "Volume (Contracts) - Options",
                    "Volume (Contracts) - Total",
                    "Value (₹ Lakhs) - Futures",
                    "Value (₹ Lakhs) - Options (Premium)",
                    "Value (₹ Lakhs) - Total",
                    "Open Interest (Contracts)",
                    "Underlying"
                ]
                # If the number of columns matches, set them directly; else, print warning
                if len(df.columns) == len(required_columns):
                    df.columns = required_columns
                else:
                    print(f"[WARNING] Column count mismatch. Data columns: {list(df.columns)}")
                    # Try to align columns by name if possible
                    for i, col in enumerate(required_columns):
                        if i < len(df.columns):
                            df.rename(columns={df.columns[i]: col}, inplace=True)
                # Format: LA-MOST-ACTIVE-UNDERLYING-14-Jan-2026
                today = datetime.now().strftime('%d-%b-%Y')
                fname = f"LA-MOST-ACTIVE-UNDERLYING-{today}.csv"
                out_path = os.path.join(OUT_DIR, fname)
                df.to_csv(out_path, index=False)
                print(f"[SUCCESS] Saved: {out_path}")
                return out_path
            else:
                print("[ERROR] Unexpected JSON structure from API.")
        else:
            print(f"[ERROR] API request failed: status={resp.status_code}")
    except Exception as e:
        print(f"[ERROR] Exception during API request: {e}")
    print("[FAIL] Could not download most active underlying data.")
    return None

if __name__ == "__main__":
    download_most_active_csv()
