#!/usr/bin/env python3
"""
Fetch top headlines, upcoming events, and corporate actions for symbols
covered in streamlit_app.py (most active symbols).

Outputs a single CSV with one row per symbol.
"""

import argparse
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
import re
import time
import xml.etree.ElementTree as ET
import pandas as pd
import requests
import yfinance as yf

from data_download_vbt import get_symbols, get_dates_from_most_active_files

NEWS_SOURCES = {
    "Mint": "https://www.livemint.com/rss/market",
    "EconomicTimes": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "Bloomberg": "https://www.bloomberg.com/feeds/markets.xml",
    "Investing": "https://www.investing.com/rss/news_25.rss",
    "Moneycontrol": "https://www.moneycontrol.com/rss/marketreports.xml",
}

EVENT_KEYWORDS = [
    "earnings", "results", "board meeting", "agm", "egm", "investor meet",
]

ACTION_KEYWORDS = [
    "dividend", "split", "bonus", "buyback", "rights issue", "merger", "demerger",
]

DEFAULT_DAYS_BACK = 7
DEFAULT_MAX_HEADLINES = 5
NSE_EQUITY_LIST_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"


def _resolve_date(date_arg: str | None) -> pd.Timestamp:
    if date_arg:
        return pd.to_datetime(date_arg).normalize()

    dates_raw = get_dates_from_most_active_files()
    if dates_raw is None:
        return pd.Timestamp.today().normalize()

    try:
        dates_list = list(pd.to_datetime(list(dates_raw)))
        if dates_list:
            return max(dates_list).normalize()
    except Exception:
        try:
            return pd.to_datetime(dates_raw).normalize()
        except Exception:
            return pd.Timestamp.today().normalize()

    return pd.Timestamp.today().normalize()


def _safe_first(items, default=""):
    return items[0] if items else default


def _normalize_company_name(name: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", name).strip()
    tokens = {cleaned}
    tokens.add(cleaned.lower())
    tokens.add(cleaned.replace("&", "and"))
    tokens.add(cleaned.replace("&", ""))
    tokens.add(cleaned.replace("Limited", "").replace("Ltd.", "").replace("Ltd", "").strip())

    words = [w for w in re.split(r"\s+", cleaned) if w]
    if len(words) > 1:
        acronym = "".join(w[0].upper() for w in words if w[0].isalpha())
        if acronym:
            tokens.add(acronym)

    return [t for t in tokens if t]


def _load_name_map() -> dict:
    output_dir = Path("results/news_events")
    output_dir.mkdir(parents=True, exist_ok=True)

    nse_cache = output_dir / "nse_symbol_names.csv"
    name_map = {}

    # Load cached NSE list if fresh, else download
    refresh = True
    if nse_cache.exists():
        mtime = datetime.fromtimestamp(nse_cache.stat().st_mtime)
        if datetime.now() - mtime < timedelta(days=30):
            refresh = False

    if refresh:
        try:
            resp = requests.get(NSE_EQUITY_LIST_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
            resp.raise_for_status()
            nse_cache.write_bytes(resp.content)
        except Exception:
            pass

    if nse_cache.exists():
        try:
            df = pd.read_csv(nse_cache)
            if "SYMBOL" in df.columns and "NAME OF COMPANY" in df.columns:
                for _, row in df.iterrows():
                    symbol = str(row.get("SYMBOL", "")).strip()
                    name = str(row.get("NAME OF COMPANY", "")).strip()
                    if symbol and name:
                        name_map[symbol.upper()] = name
        except Exception:
            pass

    # Optional BSE mapping file (user-provided)
    bse_cache = output_dir / "bse_symbol_names.csv"
    if bse_cache.exists():
        try:
            df = pd.read_csv(bse_cache)
            for _, row in df.iterrows():
                symbol = str(row.get("SYMBOL", "")).strip()
                name = str(row.get("NAME", "")).strip()
                if symbol and name:
                    name_map[symbol.upper()] = name
        except Exception:
            pass

    return name_map


def _symbol_tokens(symbol: str, name_map: dict) -> list[str]:
    tokens = {symbol}
    tokens.add(symbol.replace("&", ""))
    tokens.add(symbol.replace("&", "and"))
    tokens.add(symbol.replace("-", ""))
    tokens.add(symbol.replace(" ", ""))
    tokens.add(symbol.lower())

    alias_map = {
        "NIFTY": ["nifty", "nifty 50"],
        "BANKNIFTY": ["bank nifty"],
        "FINNIFTY": ["fin nifty"],
    }
    for alias in alias_map.get(symbol.upper(), []):
        tokens.add(alias)

    company_name = name_map.get(symbol.upper()) if name_map else None
    if company_name:
        for name_token in _normalize_company_name(company_name):
            tokens.add(name_token)

    return [t for t in tokens if t]


def _compile_symbol_patterns(symbol: str, name_map: dict) -> list[re.Pattern]:
    tokens = _symbol_tokens(symbol, name_map)
    patterns = []
    for tok in tokens:
        tok_esc = re.escape(tok)
        # Use word boundaries where possible; fallback to plain search for symbols like M&M
        if re.fullmatch(r"[A-Za-z0-9]+", tok):
            patterns.append(re.compile(rf"\b{tok_esc}\b", re.IGNORECASE))
        else:
            patterns.append(re.compile(tok_esc, re.IGNORECASE))
    return patterns


def _google_news_rss(symbol: str) -> str:
    query = f"{symbol} stock India"
    return f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=en-IN&gl=IN&ceid=IN:en"


def _fetch_rss_feed(url: str, source: str, session: requests.Session | None = None) -> list[dict]:
    sess = session or requests.Session()
    last_err = None
    for attempt in range(3):
        try:
            resp = sess.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
            resp.raise_for_status()
            break
        except Exception as exc:
            last_err = exc
            time.sleep(0.5 * (attempt + 1))
    else:
        return []

    try:
        root = ET.fromstring(resp.content)
    except Exception:
        return []

    items = []

    # RSS 2.0
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub_date = (item.findtext("pubDate") or "").strip()
        description = (item.findtext("description") or "").strip()
        items.append({
            "source": source,
            "title": title,
            "link": link,
            "published": pub_date,
            "summary": description,
        })

    # Atom
    for entry in root.findall(".//{http://www.w3.org/2005/Atom}entry"):
        title = (entry.findtext("{http://www.w3.org/2005/Atom}title") or "").strip()
        link_el = entry.find("{http://www.w3.org/2005/Atom}link")
        link = link_el.attrib.get("href", "").strip() if link_el is not None else ""
        pub_date = (entry.findtext("{http://www.w3.org/2005/Atom}updated") or "").strip()
        summary = (entry.findtext("{http://www.w3.org/2005/Atom}summary") or "").strip()
        items.append({
            "source": source,
            "title": title,
            "link": link,
            "published": pub_date,
            "summary": summary,
        })

    return items


def _parse_item_time(text: str) -> pd.Timestamp | None:
    if not text:
        return None
    try:
        return pd.to_datetime(parsedate_to_datetime(text))
    except Exception:
        try:
            return pd.to_datetime(text, errors="coerce")
        except Exception:
            return None


def _normalize_items(items: list[dict]) -> list[dict]:
    normalized = []
    for item in items:
        published = item.get("published", "")
        ts = _parse_item_time(published)
        item["published_ts"] = ts
        normalized.append(item)
    return normalized


def _fetch_all_news_items(symbols: list[str]) -> list[dict]:
    all_items = []
    session = requests.Session()
    for source, url in NEWS_SOURCES.items():
        all_items.extend(_fetch_rss_feed(url, source, session=session))

    # Per-symbol Google News RSS to avoid missing relevant headlines
    for symbol in symbols:
        all_items.extend(_fetch_rss_feed(_google_news_rss(symbol), "GoogleNews", session=session))

    return _normalize_items(all_items)


def _match_items_for_symbol(symbol: str, items: list[dict], days_back: int, name_map: dict) -> list[dict]:
    patterns = _compile_symbol_patterns(symbol, name_map)
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=days_back)
    matched = []
    for item in items:
        text = f"{item.get('title', '')} {item.get('summary', '')}"
        if not any(p.search(text) for p in patterns):
            continue
        ts = item.get("published_ts", None)
        if ts is not None and pd.notna(ts) and ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        if ts is not None and pd.notna(ts):
            if ts < cutoff:
                continue
        matched.append(item)
    return matched


def _derive_event_action(items: list[dict]) -> tuple[str, str, str, str]:
    event = ""
    event_date = ""
    action = ""
    action_date = ""

    for item in items:
        title = f"{item.get('title', '')} {item.get('summary', '')}".lower()
        published = item.get("published", "")

        if not event and any(k in title for k in EVENT_KEYWORDS):
            event = "Event"
            event_date = published

        if not action and any(k in title for k in ACTION_KEYWORDS):
            action = "Corporate Action"
            action_date = published

        if event and action:
            break

    return event, event_date, action, action_date


def _fetch_symbol_data(symbol: str, news_items: list[dict], days_back: int, max_headlines: int, name_map: dict) -> dict:
    data = {
        "Symbol": symbol,
        "Headline": "",
        "HeadlineTime": "",
        "HeadlineSource": "",
        "HeadlineLink": "",
        "Headlines": "",
        "HeadlineSources": "",
        "HeadlineLinks": "",
        "Event": "",
        "EventDate": "",
        "CorporateAction": "",
        "CorporateActionDate": "",
        "Dividend": "",
        "DividendDate": "",
        "Split": "",
        "SplitDate": "",
    }

    matched_items = _match_items_for_symbol(symbol, news_items, days_back=days_back, name_map=name_map)
    if matched_items:
        # Sort by time desc, then by keyword relevance
        def _score(item: dict) -> tuple:
            title = f"{item.get('title', '')} {item.get('summary', '')}".lower()
            kw_score = sum(k in title for k in EVENT_KEYWORDS + ACTION_KEYWORDS)
            ts = item.get("published_ts")
            return (kw_score, ts or pd.Timestamp.min.tz_localize("UTC"))

        matched_items = sorted(matched_items, key=_score, reverse=True)
        top = matched_items[0]
        data["Headline"] = top.get("title", "")
        data["HeadlineTime"] = top.get("published", "")
        data["HeadlineSource"] = top.get("source", "")
        data["HeadlineLink"] = top.get("link", "")

        top_items = matched_items[:max_headlines]
        data["Headlines"] = " | ".join([i.get("title", "") for i in top_items if i.get("title")])
        data["HeadlineSources"] = " | ".join([i.get("source", "") for i in top_items if i.get("source")])
        data["HeadlineLinks"] = " | ".join([i.get("link", "") for i in top_items if i.get("link")])

        event, event_date, action, action_date = _derive_event_action(matched_items)
        data["Event"] = event
        data["EventDate"] = event_date
        data["CorporateAction"] = action
        data["CorporateActionDate"] = action_date

    try:
        ticker = yf.Ticker(symbol)
    except Exception:
        return data

    # Yahoo Finance fallback (headline/event/actions)
    if not data["Headline"]:
        try:
            news = ticker.news or []
            if news:
                headline = news[0].get("title", "")
                ts = news[0].get("providerPublishTime", None)
                data["Headline"] = headline
                data["HeadlineSource"] = "YahooFinance"
                if ts:
                    data["HeadlineTime"] = datetime.utcfromtimestamp(ts).isoformat()
        except Exception:
            pass

    # Events (earnings date as a proxy for upcoming events)
    try:
        cal = ticker.calendar
        if isinstance(cal, pd.DataFrame) and not cal.empty:
            if "Earnings Date" in cal.index:
                earnings = cal.loc["Earnings Date"].values
                if len(earnings) > 0:
                    data["Event"] = "Earnings"
                    data["EventDate"] = str(_safe_first(earnings))
    except Exception:
        pass

    # Corporate actions: dividends/splits (most recent)
    try:
        dividends = ticker.dividends
        if isinstance(dividends, pd.Series) and not dividends.empty:
            last_div_date = dividends.index[-1]
            data["Dividend"] = float(dividends.iloc[-1])
            data["DividendDate"] = str(pd.to_datetime(last_div_date).date())
    except Exception:
        pass

    try:
        splits = ticker.splits
        if isinstance(splits, pd.Series) and not splits.empty:
            last_split_date = splits.index[-1]
            data["Split"] = float(splits.iloc[-1])
            data["SplitDate"] = str(pd.to_datetime(last_split_date).date())
    except Exception:
        pass

    return data


def main():
    parser = argparse.ArgumentParser(description="Fetch headlines, events, and corporate actions for most active symbols")
    parser.add_argument("--date", help="Date to select most-active symbols (YYYY-MM-DD)")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top symbols")
    parser.add_argument("--output", default="results/news_events/news_events_corporate_actions.csv", help="Output CSV path")
    parser.add_argument("--days-back", type=int, default=DEFAULT_DAYS_BACK, help="Days back to search headlines")
    parser.add_argument("--max-headlines", type=int, default=DEFAULT_MAX_HEADLINES, help="Max headlines per symbol")
    args = parser.parse_args()

    selected_date = _resolve_date(args.date)
    symbols = get_symbols(selected_date, top_n=args.top_n)[0]

    name_map = _load_name_map()
    news_items = _fetch_all_news_items(symbols)

    rows = []
    for sym in symbols:
        rows.append(_fetch_symbol_data(sym, news_items, days_back=args.days_back, max_headlines=args.max_headlines, name_map=name_map))

    df = pd.DataFrame(rows)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
