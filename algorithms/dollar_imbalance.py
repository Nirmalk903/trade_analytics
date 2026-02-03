# -------------------------
# Dollar Imbalance Bars (DIB)
# -------------------------

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
from logging.handlers import RotatingFileHandler

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from data_download_vbt import getdata_vbt, get_underlying_data_vbt, get_symbols, get_dates_from_most_active_files

from dataclasses import dataclass
from typing import Optional, Tuple, Union

# -------------------------
# Logging setup
# -------------------------
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs'))
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, 'dollar_imbalance.log')

logger = logging.getLogger("dollar_imbalance")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    fh = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=3, encoding='utf-8')
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s", "%Y-%m-%d %H:%M:%S")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

@dataclass
class EMADollarImbalanceConfig:
    # tuned defaults for lower-frequency / aggregated inputs (e.g. daily)
    num_prev_bars: int = 3
    expected_imbalance_span: int = 10    # shorter span so EWMA adapts quicker
    exp_num_ticks_init: int = 5    # smaller initial bar size for sparser data
    exp_num_ticks_min: int = 1
    # exp_num_ticks_max: float = np.inf
    exp_num_ticks_max: float = 5

def _ewma_last(values, span: int) -> float:
    if len(values) == 0:
        return float('nan')
    return float(pd.Series(values).ewm(span=span, adjust=False).mean().iloc[-1])

def _detect_data_freq_seconds(dt_index) -> Optional[float]:
    """
    Return median seconds between consecutive timestamps in dt_index.
    Returns None if not enough timestamps or parsing fails.
    """
    try:
        idx = pd.DatetimeIndex(pd.to_datetime(dt_index))
    except Exception:
        return None
    if len(idx) < 2:
        return None
    deltas = (idx[1:] - idx[:-1]).total_seconds()
    # handle possible NaN/inf
    deltas = np.asarray(deltas, dtype=float)
    deltas = deltas[np.isfinite(deltas)]
    if deltas.size == 0:
        return None
    return float(np.median(deltas))

class EMADollarImbalanceBars:
    """
    Builds Dollar Imbalance Bars (DIB).
    Emits a bar when abs(cum_signed_dollar) >= E[T] * abs(E[b])
    where signed_dollar = tick_sign * (price * volume).
    """
    def __init__(self, cfg: Optional[EMADollarImbalanceConfig] = None, analyse_thresholds: bool = False):
        self.cfg = cfg or EMADollarImbalanceConfig()
        self.analyse_thresholds = analyse_thresholds
        self._reset_state()
        self.prev_price = None
        self.prev_tick_rule = 0
        self.global_tick_num = 0
        self.prev_bar_Ts = []
        self.prev_bar_avg_signed = []
        self.threshold_log = []

    def _reset_state(self):
        self.open_price = None
        self.high_price = -np.inf
        self.low_price = np.inf
        self.cum_ticks = 0
        self.cum_volume = 0.0
        self.cum_buy_volume = 0.0
        self.cum_dollar_value = 0.0
        self.cum_signed_dollar = 0.0

    def _tick_rule(self, price: float) -> int:
        if self.prev_price is None:
            s = self.prev_tick_rule or 1
        else:
            dp = price - self.prev_price
            if dp > 0:
                s = 1
            elif dp < 0:
                s = -1
            else:
                s = self.prev_tick_rule or 1
        self.prev_price = price
        self.prev_tick_rule = s
        return 1 if s >= 0 else -1

    def _expected_b(self) -> float:
        if not self.prev_bar_avg_signed:
            return float('nan')
        span = min(len(self.prev_bar_avg_signed), self.cfg.expected_imbalance_span)
        return float(_ewma_last([abs(x) for x in self.prev_bar_avg_signed], span=span))

    def _expected_T(self) -> float:
        if not self.prev_bar_Ts:
            return float(self.cfg.exp_num_ticks_init)
        span = min(len(self.prev_bar_Ts), self.cfg.num_prev_bars)
        est = float(_ewma_last(self.prev_bar_Ts, span=span))
        return float(np.clip(est, self.cfg.exp_num_ticks_min, self.cfg.exp_num_ticks_max))

    def transform(self, ticks: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info("Transform start: %d rows", len(ticks))
        req = ['date_time', 'price', 'volume']
        miss = [c for c in req if c not in ticks.columns]
        if miss:
            logger.error("Input DataFrame missing columns: %s", miss)
            raise ValueError(f"Input DataFrame missing columns: {miss}")

        df = ticks.sort_values("date_time").reset_index(drop=True).copy()
        # ensure numeric volume (some engineered files may have NaN/strings)
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)

        # quick check: if data appears to be low-frequency (one row per day),
        # lower thresholds automatically (safer defaults)
        inferred_freq = None
        if len(df) > 2:
            inferred_freq = (df["date_time"].iloc[-1] - df["date_time"].iloc[0]) / (len(df) - 1)
        if inferred_freq is not None and inferred_freq >= pd.Timedelta(days=1):
            # relax thresholds for daily data
            cfg_relaxed = True
        else:
            cfg_relaxed = False

        # Pre-compute tick sign and signed dollar
        prices = df["price"].astype(float)
        vols = df["volume"].astype(float)
        diffs = prices.diff().fillna(0)
        tick_sign = np.sign(diffs)
        tick_sign.replace(0, np.nan, inplace=True)
        tick_sign.ffill(inplace=True)
        tick_sign.fillna(1, inplace=True)
        df["tick_sign"] = tick_sign.astype(int)
        df["dollar"] = prices * vols
        df["signed_dollar"] = df["tick_sign"] * df["dollar"]

        # sensible initial expected imbalance (E[b]) from first N ticks and initial expected T
        init_window = min(len(df), 100)
        if init_window > 0:
            init_mean_abs_signed = float(df["signed_dollar"].abs().iloc[:init_window].mean() or 0.0)
            # if volumes/prices are tiny, avoid zero threshold
            expected_b = max(1e-8, init_mean_abs_signed)
        else:
            expected_b = 1e-8

        # if input is daily / sparse and initial expected_b tiny, reduce exp_num_ticks_init
        expected_T = float(self.cfg.exp_num_ticks_init if not cfg_relaxed else max(1, int(self.cfg.exp_num_ticks_init / 10)))

        bars = []
        # clear threshold log (re-use for each transform call)
        self.threshold_log = []
        # reset running state
        self._reset_state()
        self.prev_price = None
        self.prev_tick_rule = 0
        self.global_tick_num = 0
        self.prev_bar_Ts = []
        self.prev_bar_avg_signed = []

        # optional progress logging
        log_every = max(1, int(len(df) / 10))

        for idx, row in df.iterrows():
            if idx % log_every == 0:
                logger.debug("Processing row %d/%d", idx, len(df))

            ts = row['date_time']
            price = float(row['price'])
            vol = float(row['volume'])
            dollar = float(row['dollar'])
            sgn = int(row['tick_sign'])

            self.global_tick_num += 1

            # price bookkeeping
            if self.open_price is None:
                self.open_price = price
            self.high_price = max(self.high_price, price)
            self.low_price = min(self.low_price, price)

            # accumulators
            self.cum_ticks += 1
            self.cum_volume += vol
            self.cum_dollar_value += dollar
            if sgn > 0:
                self.cum_buy_volume += vol
            self.cum_signed_dollar += sgn * dollar

            # update expected_b and expected_T from histories when available
            if self.prev_bar_avg_signed:
                expected_b = float(_ewma_last([abs(x) for x in self.prev_bar_avg_signed],
                                              span=min(len(self.prev_bar_avg_signed), self.cfg.expected_imbalance_span)))
                expected_b = max(expected_b, 1e-12)
            # update expected_T
            if self.prev_bar_Ts:
                expected_T = float(_ewma_last(self.prev_bar_Ts, span=min(len(self.prev_bar_Ts), self.cfg.num_prev_bars)))
                expected_T = float(np.clip(expected_T, self.cfg.exp_num_ticks_min, self.cfg.exp_num_ticks_max))

            # safety: if expected_b is zero or NaN, set a small fallback so threshold != 0
            if not np.isfinite(expected_b) or expected_b <= 0:
                expected_b = 1e-8

            # optionally log thresholds for debugging
            if self.analyse_thresholds:
                self.threshold_log.append({
                    "date_time": ts,
                    "cum_signed_dollar": self.cum_signed_dollar,
                    "expected_T": expected_T,
                    "expected_b": expected_b,
                    "threshold": expected_T * abs(expected_b)
                })

            threshold = expected_T * abs(expected_b)
            # avoid threshold being ridiculously large for sparse inputs
            if cfg_relaxed:
                threshold = threshold * 0.5

            if abs(self.cum_signed_dollar) >= threshold:
                bar = {
                    "date_time": ts,
                    "tick_num": self.global_tick_num,
                    "open": self.open_price,
                    "high": max(self.high_price, self.open_price),
                    "low": min(self.low_price, self.open_price),
                    "close": price,
                    "volume": self.cum_volume,
                    "cum_buy_volume": self.cum_buy_volume,
                    "cum_ticks": self.cum_ticks,
                    "cum_dollar_value": self.cum_dollar_value,
                    "signed_dollar_imbalance": self.cum_signed_dollar
                }
                bars.append(bar)

                # update histories
                self.prev_bar_Ts.append(self.cum_ticks)
                avg_signed = self.cum_signed_dollar / max(1, self.cum_ticks)
                self.prev_bar_avg_signed.append(avg_signed)

                # reset running accumulators for next bar
                self._reset_state()

        # if there is a remaining partial bar, append it so output is never empty
        if self.cum_ticks > 0:
            bar = {
                "date_time": df["date_time"].iloc[-1],
                "tick_num": self.global_tick_num,
                "open": self.open_price if self.open_price is not None else df["price"].iloc[0],
                "high": max(self.high_price, self.open_price) if self.open_price is not None else df["price"].max(),
                "low": min(self.low_price, self.open_price) if self.open_price is not None else df["price"].min(),
                "close": df["price"].iloc[-1],
                "volume": self.cum_volume,
                "cum_buy_volume": self.cum_buy_volume,
                "cum_ticks": self.cum_ticks,
                "cum_dollar_value": self.cum_dollar_value,
                "signed_dollar_imbalance": self.cum_signed_dollar
            }
            bars.append(bar)

        bars_df = pd.DataFrame(bars)
        thr_df = pd.DataFrame(self.threshold_log) if self.analyse_thresholds else pd.DataFrame()
        logger.info("Transform complete: emitted %d bars (input rows=%d)", len(bars_df), len(df))
        return bars_df, thr_df

def get_ema_dollar_imbalance_bars(data: Union[str, pd.DataFrame],
                                  num_prev_bars: int = 3,
                                  expected_imbalance_span: int = 1000,
                                  exp_num_ticks_init: int = 2000,
                                  exp_num_ticks_constraints: Tuple[int, float] = (1, np.inf),
                                  analyse_thresholds: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Wrapper to produce Dollar Imbalance Bars.
    data: DataFrame or CSV path containing ['date_time','price','volume'].
    """
    if isinstance(data, str):
        df = pd.read_csv(data, parse_dates=['date_time'])
    else:
        df = data.copy()
        if 'date_time' not in df.columns:
            raise ValueError("DataFrame must include 'date_time' column")
        if not np.issubdtype(df['date_time'].dtype, np.datetime64):
            df['date_time'] = pd.to_datetime(df['date_time'])

    cfg = EMADollarImbalanceConfig(
        num_prev_bars=num_prev_bars,
        expected_imbalance_span=expected_imbalance_span,
        exp_num_ticks_init=exp_num_ticks_init,
        exp_num_ticks_min=exp_num_ticks_constraints[0],
        exp_num_ticks_max=exp_num_ticks_constraints[1],
    )
    engine = EMADollarImbalanceBars(cfg, analyse_thresholds=analyse_thresholds)
    return engine.transform(df)

# -------------------------
# Apply DIBs to all symbols and save
# -------------------------
def apply_dollar_imbalance_bars_all_symbols(engineered_dir: str = "./Engineered_data",
                                            results_dir: str = "./results/dollar_imbalance",
                                            num_prev_bars: int = 3,
                                            expected_imbalance_span: int = 5,
                                            exp_num_ticks_init: int = 5,
                                            exp_num_ticks_constraints: Tuple[int, float] = (1, 5),
                                            aggressive: bool = False):
    """
    Apply DIB to all available symbols.
    - Robustly finds symbols (most active list if available, otherwise files in engineered_dir).
    - Automatically relaxes parameters for low-frequency (daily) inputs.
    - Always writes a CSV per symbol (real bars or a one-row summary).
    """
    logger.info("Starting Dollar Imbalance Bars run. engineered_dir=%s results_dir=%s", engineered_dir, results_dir)
    os.makedirs(results_dir, exist_ok=True)

    # 1) Try to get symbols from most-active files / get_symbols. Fall back to engineered_dir listing.
    symbols = []
    try:
        dates = get_dates_from_most_active_files()
        if len(dates) > 0:
            symbols, _ = get_symbols(dates[-1], top_n=17)
            logger.info("Found %d symbols from most-active files", len(symbols))
    except Exception:
        logger.exception("Failed to obtain symbols from most-active files, falling back to engineered_dir")

    if not symbols:
        # fallback: scan engineered_dir for files like SYMBOL_1d_features.json or SYMBOL_1d.csv
        if os.path.isdir(engineered_dir):
            for fname in os.listdir(engineered_dir):
                base, ext = os.path.splitext(fname)
                # accept files named SYMBOL_1d_features.json or SYMBOL_1d.csv
                if ext.lower() in {".json", ".csv"}:
                    symbol = base.split("_")[0]
                    if symbol:
                        symbols.append(symbol)
            symbols = sorted(set(symbols))
            logger.info("Found %d symbols from engineered_dir", len(symbols))
    if not symbols:
        logger.warning("No symbols found to process (no most-active files and no engineered files).")
        return

    # 2) Process each symbol
    for symbol in symbols:
        logger.info("Processing symbol %s", symbol)
        # try engineered json first, then underlying csv as fallback
        engineered_path = os.path.join(engineered_dir, f"{symbol}_1d_features.json")
        engineered_path_csv = os.path.join(engineered_dir, f"{symbol}_1d_features.csv")
        underlying_csv = os.path.join(os.path.dirname(__file__), "..", "Underlying_data_vbt", f"{symbol}_1d.csv")
        df = None

        for path in (engineered_path, engineered_path_csv, underlying_csv):
            if path and os.path.exists(path):
                try:
                    if path.lower().endswith(".json"):
                        df = pd.read_json(path, orient='records', lines=True)
                    else:
                        df = pd.read_csv(path)
                    break
                except Exception as e:
                    logger.warning("Failed reading %s for %s: %s", path, symbol, e)
                    df = None

        if df is None or df.empty:
            logger.warning("No input file found or file empty for %s, skipping.", symbol)
            continue

        # normalize expected column names
        if 'Date' in df.columns:
            df = df.rename(columns={'Date': 'date_time'})
        if 'date_time' not in df.columns and 'Date' in df.columns:
            df['date_time'] = pd.to_datetime(df['Date'])
        if 'date_time' in df.columns:
            df['date_time'] = pd.to_datetime(df['date_time'])
        if 'Close' in df.columns and 'price' not in df.columns:
            df = df.rename(columns={'Close': 'price'})
        if 'Volume' in df.columns and 'volume' not in df.columns:
            df = df.rename(columns={'Volume': 'volume'})

        if not {'date_time', 'price', 'volume'}.issubset(df.columns):
            logger.warning("Required columns missing in %s file, skipping. Columns present: %s", symbol, list(df.columns))
            continue

        tick_df = df[['date_time', 'price', 'volume']].sort_values('date_time').reset_index(drop=True).copy()

        # infer frequency and relax params for daily/low-frequency series
        freq_relaxed = False
        if len(tick_df) > 2:
            inferred = (tick_df['date_time'].iloc[-1] - tick_df['date_time'].iloc[0]) / (len(tick_df) - 1)
            if inferred >= pd.Timedelta(days=1):
                freq_relaxed = True

        # --- determine data frequency and scale thresholds ---
        # detect sampling frequency (seconds between rows)
        med_dt = _detect_data_freq_seconds(tick_df['date_time']) if 'date_time' in tick_df.columns else None
        # If data is low-frequency (daily), scale thresholds down strongly
        if med_dt is None:
            freq_scale = 1.0
        elif med_dt >= 24*3600 - 1:   # roughly daily
            freq_scale = 0.01         # shrink thresholds by 100x for daily inputs
        elif med_dt >= 3600:         # hourly-ish
            freq_scale = 0.1
        else:
            freq_scale = 1.0

        if aggressive:
            freq_scale *= 0.2

        use_expected_span = max(1, int(max(1, expected_imbalance_span) * freq_scale))
        use_exp_ticks = max(1, int(max(1, exp_num_ticks_init) * freq_scale))

        logger.debug("[%s] med_dt=%s sec freq_scale=%s E_span=%s exp_ticks=%s",
                     symbol, getattr(med_dt, "__str__", lambda: med_dt)(), freq_scale, use_expected_span, use_exp_ticks)

        # Initialize estimator E0 (expected imbalance span) per-symbol if missing and keep it numeric
        if not hasattr(tick_df, "_E0") or tick_df._E0 is None:
            # use median of volume-weighted dollar ticks over a small window or fallback to use_expected_span
            try:
                sample_vals = (tick_df['price'] * tick_df['volume']).abs().values
                sample_med = float(np.median(sample_vals[sample_vals > 0])) if len(sample_vals) else float(use_expected_span)
                E0_init = max(1.0, sample_med if np.isfinite(sample_med) else float(use_expected_span))
            except Exception:
                E0_init = float(use_expected_span)
            # store per-symbol E0 on the DataFrame object for subsequent calls
            tick_df._E0 = float(E0_init)

        # helper inside bar generator: expected_imbalance uses E0 and p_buy
        def _expected_imbalance(E0, p_buy):
            base = float(E0) if (E0 and np.isfinite(E0) and E0 > 0) else float(use_expected_span)
            abs_term = abs(2 * p_buy - 1)
            if abs_term < 1e-6:
                return base
            return base * abs_term

        # Use EWMA updates for E0 and P_buy inside algorithm when a bar completes
        lam_E0_local = 0.9
        lam_p_local = 0.9

        # when accumulating ticks, use expected = _expected_imbalance(current_E0, p_buy)
        current_E0 = float(tick_df._E0)
        current_pbuy = 0.5

        # then inside your main tick loop (where you previously did expected_imbalance()):
        # replace calls to expected_imbalance() with:
        #    threshold = _expected_imbalance(current_E0, current_pbuy)
        # and when a bar forms, update:
        #    realized_T = len(current_bar_ticks)  # or realized dollar/volume as appropriate
        #    current_E0 = (1 - lam_E0_local) * float(realized_T) + lam_E0_local * current_E0
        #    current_pbuy = (1 - lam_p_local) * p_bar + lam_p_local * current_pbuy
        #
        # This keeps thresholds numeric, adaptive, and avoids inf/None values.

        bars_df, thr_df = get_ema_dollar_imbalance_bars(
            tick_df,
            num_prev_bars=max(1, num_prev_bars),
            expected_imbalance_span=use_expected_span,
            exp_num_ticks_init=use_exp_ticks,
            exp_num_ticks_constraints=exp_num_ticks_constraints,
            analyse_thresholds=True
        )

        out_path = os.path.join(results_dir, f"{symbol}_dollar_imbalance_bars.csv")

        # ensure something is written (bars or summary)
        if bars_df is None or bars_df.empty:
            summary = {
                "date_time": [tick_df["date_time"].iloc[-1] if len(tick_df) else pd.NaT],
                "open": [tick_df["price"].iloc[0] if len(tick_df) else np.nan],
                "high": [tick_df["price"].max() if len(tick_df) else np.nan],
                "low": [tick_df["price"].min() if len(tick_df) else np.nan],
                "close": [tick_df["price"].iloc[-1] if len(tick_df) else np.nan],
                "volume": [tick_df["volume"].sum() if len(tick_df) else 0.0],
                "cum_ticks": [len(tick_df)],
                "cum_dollar_value": [(tick_df["price"]*tick_df["volume"]).sum() if len(tick_df) else 0.0],
                "signed_dollar_imbalance": [0.0]
            }
            pd.DataFrame(summary).to_csv(out_path, index=False)
            logger.info("[%s] no bars; wrote summary to %s (freq_relaxed=%s, span=%d, exp_T=%d)", symbol, out_path, freq_relaxed, use_expected_span, use_exp_ticks)
        else:
            df_to_write = bars_df
            # Add a small diagnostics print after bars are generated for symbol:
            logger.info("[%s] wrote output to %s  produced_bars=%d  init_E0=%s  final_E0=%s  final_pbuy=%s",
                        symbol, out_path, len(df_to_write), getattr(tick_df, "_E0", None), current_E0, current_pbuy)
            bars_df.to_csv(out_path, index=False)
            logger.info("[%s] saved %d bars to %s (freq_relaxed=%s, span=%d, exp_T=%d)", symbol, len(bars_df), out_path, freq_relaxed, use_expected_span, use_exp_ticks)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Dollar Imbalance Bars for all symbols")
    parser.add_argument("--engineered", default="./Engineered_data", help="Folder with engineered files")
    parser.add_argument("--results", default="./results/dollar_imbalance", help="Folder to save results")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--num_prev_bars", type=int, default=3)
    parser.add_argument("--expected_span", type=int, default=50)
    parser.add_argument("--exp_ticks", type=int, default=20)
    parser.add_argument("--aggressive", action="store_true", help="Use aggressive (lower) thresholds for more frequent bars")
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")

    logger.info("Running dollar_imbalance as script")
    apply_dollar_imbalance_bars_all_symbols(
        engineered_dir=args.engineered,
        results_dir=args.results,
        num_prev_bars=args.num_prev_bars,
        expected_imbalance_span=args.expected_span,
        exp_num_ticks_init=args.exp_ticks,
        aggressive=args.aggressive
    )