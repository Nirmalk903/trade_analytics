"""
Robust CUSUM utilities and plotting for engineered Close series.

- Reads engineered files only (Close price).
- Defensive I/O and logging.
- CLI entrypoint to produce per-symbol CSVs and a combined PDF of charts.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple, List
from datetime import datetime

import logging
import math

import numpy as np
import pandas as pd

# matplotlib: prefer non-interactive backend when DISPLAY not available
try:
    import matplotlib

    if not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
except Exception:
    plt = None
    PdfPages = None

# make project root import-friendly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# optional helpers (best-effort import)
try:
    from data_download_vbt import get_symbols, get_dates_from_most_active_files  # type: ignore
except Exception:
    get_symbols = None
    get_dates_from_most_active_files = None

# logging
logger = logging.getLogger("cusum")
if not logger.handlers:
    h = logging.StreamHandler(sys.stdout)
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    h.setFormatter(logging.Formatter(fmt))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


def getTEvents(g_raw: pd.Series, h: float) -> Tuple[pd.DatetimeIndex, pd.DataFrame]:
    """Compute CUSUM trigger event timestamps and diagnostic dataframe.

    Returns (tEvents, diag_df) where diag_df indexed by diff timestamps with columns sPos,sNeg,trigger.
    """
    if g_raw is None or len(g_raw) == 0:
        return pd.DatetimeIndex([]), pd.DataFrame(columns=["sPos", "sNeg", "trigger"])

    g = pd.Series(g_raw).astype(float).copy()
    diff = g.diff().fillna(0.0)

    t_events: List[pd.Timestamp] = []
    sPos, sNeg = 0.0, 0.0
    rows = []

    for idx in diff.index[1:]:
        d = float(diff.loc[idx])
        sPos = max(0.0, sPos + d)
        sNeg = min(0.0, sNeg + d)

        trigger_val = 0
        if sNeg < -h:
            trigger_val = -1
            t_events.append(idx)
            sNeg = 0.0
        elif sPos > h:
            trigger_val = 1
            t_events.append(idx)
            sPos = 0.0

        rows.append({"sPos": sPos, "sNeg": sNeg, "trigger": trigger_val})

    diag_df = pd.DataFrame(rows, index=diff.index[1:]) if rows else pd.DataFrame(columns=["sPos", "sNeg", "trigger"])
    return pd.DatetimeIndex(t_events), diag_df


def find_dollar_imbalance_file(symbol: str,
                               engineered_dir: str,
                               results_dir: str,
                               underlying_dir: Optional[str] = None) -> Optional[str]:
    """
    Restrict input discovery to engineered data only (Close price).
    Do NOT prefer or return dollar-imbalance files.
    """
    engineered_dir = str(engineered_dir)
    candidates = [
        os.path.join(engineered_dir, f"{symbol}_1d_features.json"),
        os.path.join(engineered_dir, f"{symbol}_1d_features.csv"),
        os.path.join(engineered_dir, f"{symbol}.csv"),
    ]
    for p in candidates:
        try:
            if p and os.path.exists(p):
                return p
        except Exception:
            continue
    return None


def read_series_from_file(file_path: str) -> Tuple[Optional[pd.Series], Optional[str]]:
    """
    Read engineered file and return Close series only.
    Returns (series, "Close") or (None, None) if file is invalid / missing Close.
    """
    try:
        if file_path.lower().endswith(".json"):
            df = pd.read_json(file_path, orient="records", lines=True)
        else:
            df = pd.read_csv(file_path, parse_dates=True, infer_datetime_format=True)
    except Exception as e:
        logger.warning("Failed reading %s: %s", file_path, e)
        return None, None

    # normalize date column
    if "date_time" not in df.columns and "Date" in df.columns:
        df = df.rename(columns={"Date": "date_time"})

    # require Close (case sensitive as produced) or fallback to lowercase 'close'
    if "Close" in df.columns:
        val_col = "Close"
    elif "close" in df.columns:
        val_col = "close"
    else:
        logger.debug("Engineered file %s missing Close column", file_path)
        return None, None

    if "date_time" not in df.columns:
        logger.debug("Engineered file %s missing date_time/Date column", file_path)
        return None, None

    ser = pd.Series(df[val_col].values, index=pd.to_datetime(df["date_time"], errors="coerce"))
    ser = ser[~ser.index.duplicated(keep="first")].dropna()
    return (ser, "Close") if not ser.empty else (None, None)


def _ensure_dir(d: str) -> None:
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass


def plot_cusum_events_all(symbols: Iterable[str],
                          engineered_dir: str,
                          results_dir: str,
                          underlying_dir: Optional[str] = None,
                          lookback_years: int = 2) -> None:
    """Compute CUSUM events for each symbol, save per-symbol CSV and a combined PDF of charts."""
    results_dir = str(results_dir)
    _ensure_dir(results_dir)
    cusum_dir = os.path.join(results_dir, "cusum")
    _ensure_dir(cusum_dir)
    pdf_path = os.path.join(results_dir, "cusum_events_close_charts.pdf")

    with PdfPages(pdf_path) as pdf:
        # add a cover/title page with report date
        try:
            cover = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
            cover_text = f"CUSUM Events Report\nGenerated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"
            cover.text(0.5, 0.6, "CUSUM Events Report", ha="center", va="center", fontsize=20, weight="bold")
            cover.text(0.5, 0.45, f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}", ha="center", va="center", fontsize=10)
            cover.text(0.5, 0.3, f"Engineered dir: {engineered_dir}\nResults dir: {results_dir}", ha="center", va="center", fontsize=9)
            plt.axis("off")
            pdf.savefig(cover)
            plt.close(cover)
        except Exception:
            # non-fatal; continue generating pages
            logger.debug("Failed to add cover page to PDF", exc_info=True)
        for symbol in symbols:
            try:
                file_path = find_dollar_imbalance_file(symbol, engineered_dir, results_dir, underlying_dir)
                if not file_path:
                    logger.info("[%s] no input file found; skipping", symbol)
                    continue

                ser, label = read_series_from_file(file_path)
                if ser is None or ser.empty:
                    logger.info("[%s] series empty after read; skipping", symbol)
                    continue

                # restrict to lookback window
                end_date = ser.index.max()
                if pd.isna(end_date):
                    logger.info("[%s] invalid index; skipping", symbol)
                    continue
                start_date = end_date - pd.DateOffset(years=lookback_years)
                work_ser = ser.loc[start_date:end_date]
                if work_ser.empty:
                    work_ser = ser  # fallback

                # pick threshold h robustly
                std = float(work_ser.std()) if work_ser.size > 1 else 0.0
                if not np.isfinite(std) or std <= 0:
                    h = max(1e-6, float(abs(work_ser).median() * 0.1))
                else:
                    h = max(1e-6, std * 0.2)

                t_events, diag = getTEvents(work_ser, h)

                # build events df with 'close' column for consistency
                if len(t_events) > 0:
                    ev_vals = [work_ser.loc[t] if t in work_ser.index else np.nan for t in t_events]
                    ev_sPos = [diag.loc[t, "sPos"] if t in diag.index else np.nan for t in t_events]
                    ev_sNeg = [diag.loc[t, "sNeg"] if t in diag.index else np.nan for t in t_events]
                    ev_trig = [diag.loc[t, "trigger"] if t in diag.index else 0 for t in t_events]
                    events_df = pd.DataFrame({
                        "event_time": list(t_events),
                        "close": ev_vals,
                        "sPos": ev_sPos,
                        "sNeg": ev_sNeg,
                        "trigger": ev_trig
                    })
                else:
                    events_df = pd.DataFrame(columns=["event_time", "close", "sPos", "sNeg", "trigger"])

                csv_path = os.path.join(cusum_dir, f"{symbol}_cusum_events.csv")
                try:
                    events_df.to_csv(csv_path, index=False)
                except Exception as e:
                    logger.warning("[%s] failed to write CSV: %s", symbol, e)

                # plotting
                plt.figure(figsize=(12, 6))
                plt.plot(work_ser.index, work_ser.values, label=label, lw=0.8)
                if len(t_events) > 0:
                    triggers = [diag.loc[t, "trigger"] if t in diag.index else 0 for t in t_events]
                    pos_times = [t for t, tr in zip(t_events, triggers) if tr == 1]
                    neg_times = [t for t, tr in zip(t_events, triggers) if tr == -1]
                    if pos_times:
                        vals = [work_ser.loc[t] for t in pos_times if t in work_ser.index]
                        plt.scatter(pos_times, vals, color="green", label="CUSUM +1", zorder=5)
                    if neg_times:
                        vals = [work_ser.loc[t] for t in neg_times if t in work_ser.index]
                        plt.scatter(neg_times, vals, color="red", label="CUSUM -1", zorder=5)
                plt.title(f"{symbol} - {label} with CUSUM Events")
                plt.xlabel("Date")
                plt.ylabel(label)
                plt.legend()
                plt.tight_layout()
                pdf.savefig()
                plt.close()

            except Exception as e:
                logger.exception("Failed processing %s: %s", symbol, e)

    logger.info("All CUSUM charts saved to %s", pdf_path)
    logger.info("Saved per-symbol CUSUM CSV files to %s", cusum_dir)


def plot_garch_cusum_events_all(symbols: Iterable[str],
                                engineered_dir: str,
                                results_dir: str,
                                lookback_years: int = 2) -> None:
    """Plot CUSUM events on 'garch_vol' from engineered files (keeps original behaviour)."""
    results_dir = str(results_dir)
    _ensure_dir(results_dir)
    cusum_dir = os.path.join(results_dir, "cusum")
    _ensure_dir(cusum_dir)
    pdf_path = os.path.join(results_dir, "cusum_events_garch_vol_charts.pdf")

    with PdfPages(pdf_path) as pdf:
        # add a cover/title page with report date
        try:
            cover = plt.figure(figsize=(11.69, 8.27))
            cover.text(0.5, 0.6, "CUSUM GARCH Volatility Report", ha="center", va="center", fontsize=18, weight="bold")
            cover.text(0.5, 0.45, f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}", ha="center", va="center", fontsize=10)
            plt.axis("off")
            pdf.savefig(cover)
            plt.close(cover)
        except Exception:
            logger.debug("Failed to add GARCH cover page", exc_info=True)
        for symbol in symbols:
            try:
                file_path = os.path.join(engineered_dir, f"{symbol}_1d_features.json")
                if not os.path.exists(file_path):
                    logger.info("[%s] missing engineered file for GARCH plot; skipping", symbol)
                    continue

                df = pd.read_json(file_path, orient="records", lines=True)
                if "Date" not in df.columns or "garch_vol" not in df.columns:
                    logger.info("[%s] missing Date/garch_vol; skipping", symbol)
                    continue

                df = df.sort_values("Date")
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                df = df.set_index("Date")
                df = df[~df.index.duplicated(keep="first")]
                end_date = df.index.max()
                if pd.isna(end_date):
                    continue
                start_date = end_date - pd.DateOffset(years=lookback_years)
                work_df = df.loc[start_date:end_date].copy()
                if work_df.empty:
                    logger.info("[%s] no data in last %d years, skipping", symbol, lookback_years)
                    continue

                h = float(work_df["garch_vol"].std() * 0.25)
                if not np.isfinite(h) or h <= 0:
                    h = 1e-6

                t_events, diag = getTEvents(work_df["garch_vol"], h)

                if len(t_events) > 0:
                    ev_gv = [work_df.loc[t, "garch_vol"] if t in work_df.index else np.nan for t in t_events]
                    ev_sPos = [diag.loc[t, "sPos"] if t in diag.index else np.nan for t in t_events]
                    ev_sNeg = [diag.loc[t, "sNeg"] if t in diag.index else np.nan for t in t_events]
                    ev_trig = [diag.loc[t, "trigger"] if t in diag.index else 0 for t in t_events]
                    events_df = pd.DataFrame({
                        "event_time": list(t_events),
                        "garch_vol": ev_gv,
                        "sPos": ev_sPos,
                        "sNeg": ev_sNeg,
                        "trigger": ev_trig
                    })
                else:
                    events_df = pd.DataFrame(columns=["event_time", "garch_vol", "sPos", "sNeg", "trigger"])

                csv_path = os.path.join(cusum_dir, f"{symbol}_garch_cusum_events.csv")
                try:
                    events_df.to_csv(csv_path, index=False)
                except Exception as e:
                    logger.warning("[%s] failed to write GARCH CSV: %s", symbol, e)

                plt.figure(figsize=(12, 6))
                plt.plot(work_df.index, work_df["garch_vol"], label="GARCH Volatility", lw=0.8)
                if len(t_events) > 0:
                    triggers = [diag.loc[t, "trigger"] if t in diag.index else 0 for t in t_events]
                    pos_times = [t for t, tr in zip(t_events, triggers) if tr == 1]
                    neg_times = [t for t, tr in zip(t_events, triggers) if tr == -1]
                    if pos_times:
                        vals = [work_df.loc[t, "garch_vol"] for t in pos_times if t in work_df.index]
                        plt.scatter(pos_times, vals, color="green", label="CUSUM +1", zorder=5)
                    if neg_times:
                        vals = [work_df.loc[t, "garch_vol"] for t in neg_times if t in work_df.index]
                        plt.scatter(neg_times, vals, color="red", label="CUSUM -1", zorder=5)
                plt.title(f"{symbol} - GARCH Volatility with CUSUM Events")
                plt.xlabel("Date")
                plt.ylabel("GARCH Volatility")
                plt.legend()
                plt.tight_layout()
                pdf.savefig()
                plt.close()

            except Exception as e:
                logger.exception("GARCH plot failed for %s: %s", symbol, e)

    logger.info("All GARCH CUSUM charts saved to %s", pdf_path)
    logger.info("Saved per-symbol GARCH CUSUM CSV files to %s", cusum_dir)


def discover_symbols(prefer_n: int = 20,
                     engineered_dir: Optional[str] = None,
                     underlying_dir: Optional[str] = None) -> list:
    """Try get_symbols, else enumerate files in engineered / underlying dirs."""
    symbols = []
    try:
        if get_dates_from_most_active_files and get_symbols:
            dates = get_dates_from_most_active_files()
            if dates:
                symbols, _ = get_symbols(dates[-1], top_n=prefer_n)
    except Exception:
        symbols = []

    if symbols:
        return symbols

    # fallback by listing files
    paths = []
    if underlying_dir:
        paths.append(Path(underlying_dir))
    if engineered_dir:
        paths.append(Path(engineered_dir))
    names = set()
    for p in paths:
        try:
            if p.exists() and p.is_dir():
                for fname in p.iterdir():
                    if fname.suffix.lower() in {".csv", ".json"}:
                        base = fname.stem
                        sym = base.split("_")[0] if "_" in base else base
                        names.add(sym.upper())
        except Exception:
            continue
    return sorted(names)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CUSUM event extraction and plotting")
    parser.add_argument("--engineered", default=str(ROOT / "Engineered_data"))
    parser.add_argument("--underlying", default=str(ROOT / "Underlying_data_kite"))
    parser.add_argument("--results", default=str(ROOT / "results"))
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--garch", action="store_true", help="Also run GARCH CUSUM plots")
    args = parser.parse_args()

    _engineered = args.engineered
    _underlying = args.underlying
    _results = args.results
    _symbols = discover_symbols(prefer_n=args.top_n, engineered_dir=_engineered, underlying_dir=_underlying)

    logger.info("Processing %d symbols; engineered=%s underlying=%s results=%s", len(_symbols), _engineered, _underlying, _results)

    try:
        plot_cusum_events_all(_symbols, _engineered, _results, underlying_dir=_underlying)
    except Exception as e:
        logger.exception("plot_cusum_events_all failed: %s", e)

    if args.garch:
        try:
            plot_garch_cusum_events_all(_symbols, _engineered, _results)
        except Exception as e:
            logger.exception("plot_garch_cusum_events_all failed: %s", e)

    logger.info("Done. Check %s for CSVs and charts", _results)