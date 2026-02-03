import os
import json
import logging
import datetime
import webbrowser
import threading
import queue
import urllib.parse
import http.server
import socketserver
import requests
import time
from typing import Optional
from kiteconnect import KiteConnect
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
logging.basicConfig(level=logging.INFO)

API_KEY = os.getenv("KITE_API_KEY")
API_SECRET = os.getenv("KITE_API_SECRET")
if not API_KEY or not API_SECRET:
    raise SystemExit("Set KITE_API_KEY and KITE_API_SECRET in environment or .env (do not hardcode secrets).")

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "results", "kite"))
os.makedirs(RESULTS_DIR, exist_ok=True)
SESSION_PATH = os.path.expanduser("~/.kite_session.json")


def _json_serial(obj):
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not JSON serializable")


def _save_session(session_data, path=SESSION_PATH):
    try:
        with open(path, "w") as f:
            json.dump(session_data, f, default=_json_serial, indent=2)
        return True
    except Exception as e:
        logging.exception("Failed to save session: %s", e)
        return False


def _load_saved_session(path=SESSION_PATH):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _token_is_valid(kite_client) -> bool:
    try:
        kite_client.profile()
        return True
    except Exception:
        return False


def _capture_request_token_from_local_redirect(login_url: str, port: int = 8080, timeout: int = 180) -> Optional[str]:
    q = queue.Queue()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            params = urllib.parse.parse_qs(parsed.query)
            token = params.get("request_token", [None])[0]
            q.put(token)
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body><h3>Login complete. You can close this window.</h3></body></html>")
            threading.Thread(target=self.server.shutdown, daemon=True).start()

        def log_message(self, format, *args):
            return

    try:
        server = socketserver.TCPServer(("127.0.0.1", port), Handler)
    except OSError:
        return None

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        webbrowser.open(login_url)
    except Exception:
        print("Open this URL in a browser and complete login:", login_url)

    try:
        token = q.get(timeout=timeout)
    except queue.Empty:
        token = None

    try:
        server.shutdown()
    except Exception:
        pass

    return token


def ensure_authenticated(kite_client: KiteConnect, api_secret: str, auto_port: int = 8080, timeout: int = 180) -> bool:
    sd = _load_saved_session()
    if sd and sd.get("access_token"):
        kite_client.set_access_token(sd["access_token"])
        if _token_is_valid(kite_client):
            logging.info("Reused saved access token.")
            return True
        logging.info("Saved access token invalid/expired; will re-authenticate.")

    login_url = kite_client.login_url()
    print("Starting interactive authentication...")
    req_token = _capture_request_token_from_local_redirect(login_url, port=auto_port, timeout=timeout)
    if not req_token:
        print("Automatic capture failed or timed out.")
        print("Open the following URL in a browser, complete login and paste the request_token from the redirect URL:")
        print(login_url)
        req_token = input("Paste the request_token from the redirect URL here: ").strip()

    try:
        session_data = kite_client.generate_session(req_token, api_secret=api_secret)
    except Exception as e:
        logging.exception("generate_session failed: %s", e)
        return False

    access_token = session_data.get("access_token")
    if not access_token:
        logging.error("No access_token obtained during authentication: %s", session_data)
        return False

    kite_client.set_access_token(access_token)
    _save_session(session_data)
    logging.info("Authenticated and saved session.")
    return True


def _normalize_and_save(list_of_dicts, name: str, out_dir=RESULTS_DIR):
    if not list_of_dicts:
        logging.info("No rows for %s", name)
        return
    df = pd.DataFrame(list_of_dicts)
    dict_cols = [c for c in df.columns if df[c].apply(lambda x: isinstance(x, dict)).any()] if not df.empty else []
    for c in dict_cols:
        df_expanded = pd.json_normalize(df[c]).add_prefix(f"{c}_")
        df = pd.concat([df.drop(columns=[c]), df_expanded], axis=1)
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors="ignore")
        except Exception:
            pass
    out_path = os.path.join(out_dir, f"{name}.csv")
    df.to_csv(out_path, index=False)
    logging.info("Saved %s rows=%d -> %s", name, len(df), out_path)


def _mcp_subscribe(url: str, on_message, stop_event: threading.Event, reconnect_delay: float = 5.0):
    """
    Simple SSE client using requests. Calls on_message(data_str) for each SSE 'data:' event.
    Keeps reconnecting until stop_event.is_set().
    """
    headers = {"Accept": "text/event-stream"}
    while not stop_event.is_set():
        try:
            with requests.get(url, headers=headers, stream=True, timeout=60) as resp:
                if resp.status_code != 200:
                    print(f"[MCP] subscribe failed status={resp.status_code}, retrying in {reconnect_delay}s")
                    time.sleep(reconnect_delay)
                    continue

                buffer = ""
                for raw in resp.iter_lines(decode_unicode=True):
                    if stop_event.is_set():
                        break
                    if raw is None:
                        continue
                    line = raw.strip()
                    if line == "":
                        # blank line -> dispatch accumulated event
                        if buffer:
                            # collect data: lines (may be multiple data: lines)
                            data_lines = [l[len("data:"):].strip() for l in buffer.splitlines() if l.startswith("data:")]
                            data = "\n".join(data_lines)
                            try:
                                on_message(data)
                            except Exception as e:
                                print("[MCP] handler error:", e)
                            buffer = ""
                        continue
                    # accumulate
                    buffer += line + "\n"
        except Exception as exc:
            if stop_event.is_set():
                break
            print(f"[MCP] connection error: {exc}; reconnecting in {reconnect_delay}s")
            time.sleep(reconnect_delay)
    print("[MCP] subscriber stopped")


def _mcp_default_handler_factory(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "mcp_events.jsonl")

    def _handler(data_str: str):
        # try parse JSON, otherwise save raw string
        try:
            j = json.loads(data_str)
        except Exception:
            j = {"raw": data_str}
        # append as json line
        with open(out_file, "a", encoding="utf8") as f:
            f.write(json.dumps(j, default=_json_serial) + "\n")
        print("[MCP] event saved")
    return _handler


def main():
    kite = KiteConnect(api_key=API_KEY)

    # optional: start MCP subscriber if environment variable provided
    mcp_url = os.getenv("KITE_MCP_URL")
    mcp_thread = None
    mcp_stop = None
    if mcp_url:
        print(f"[MCP] starting subscriber -> {mcp_url}")
        mcp_stop = threading.Event()
        handler = _mcp_default_handler_factory(RESULTS_DIR)
        mcp_thread = threading.Thread(target=_mcp_subscribe, args=(mcp_url, handler, mcp_stop), daemon=True)
        mcp_thread.start()

    # Reuse saved session if present and valid, otherwise interactively authenticate once
    if not ensure_authenticated(kite, API_SECRET, auto_port=8080, timeout=180):
        # stop subscriber cleanly if running
        if mcp_stop:
            mcp_stop.set()
            mcp_thread.join(timeout=2)
        raise SystemExit("Authentication failed")

    # fetch and save positions
    try:
        positions = kite.positions()
        if isinstance(positions, dict):
            _normalize_and_save(positions.get("net", []), "positions_net")
            _normalize_and_save(positions.get("day", []), "positions_day")
        else:
            _normalize_and_save(positions, "positions_all")
    except Exception as e:
        logging.exception("kite.positions() failed: %s", e)

    # fetch and save holdings
    try:
        holdings = kite.holdings()
        _normalize_and_save(holdings, "holdings")
    except Exception as e:
        logging.exception("kite.holdings() failed: %s", e)

    # optional: print profile for verification
    try:
        profile = kite.profile()
        print("Account profile:", profile.get("client_id", ""), profile.get("user_name", ""))
    except Exception:
        pass

    # stop MCP subscriber if started
    if mcp_stop:
        mcp_stop.set()
        mcp_thread.join(timeout=2)


if __name__ == "__main__":
    main()