# Core implementation (EMA Tick Imbalance Bars)

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, List, Union
import numpy as np
import pandas as pd
from data_download_vbt import get_symbols, get_dates_from_most_active_files


# utility functions
def ewma(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Simple exponentially-weighted moving average (span-like window).
    """
    if window <= 1:
        return arr.astype(float)
    alpha = 2.0 / (window + 1.0)
    out = np.empty(arr.shape[0], dtype=float)
    out[0] = float(arr[0])
    for i in range(1, len(arr)):
        out[i] = alpha * float(arr[i]) + (1 - alpha) * out[i-1]
    return out

@dataclass
class EMATickImbalanceConfig:
    # EWMA over previous bars to estimate E[T]
    num_prev_bars: int = 3
    # EWMA window over signed ticks to estimate E[θ] (θ = signed tick imbalance)
    expected_imbalance_window: int = 1000
    # Warmup / initial E[T]
    exp_num_ticks_init: int = 2000
    # Bounds for E[T]
    exp_num_ticks_min: int = 1
    exp_num_ticks_max: float = np.inf

class EMATickImbalanceBars:
    """
    EMA Tick Imbalance Bars (Lopez de Prado, Ch.2 of AFML).
    Forms a bar when |sum_{i in bar} b_i| > E[T] * |E[b]|,
    where b_i is the tick rule sign (+1 for uptick, -1 for downtick, carry over the latest sign on ties).
    """
    def __init__(self, cfg: Optional[EMATickImbalanceConfig] = None, analyse_thresholds: bool = False):
        self.cfg = cfg or EMATickImbalanceConfig()
        self.analyse_thresholds = analyse_thresholds

        # stream state
        self.prev_price: Optional[float] = None
        self.prev_tick_rule: int = 0
        self.global_tick_num: int = 0

        # across-stream stats
        self.signed_tick_hist: List[int] = []
        self.ticks_per_bar_hist: List[int] = []
        self.threshold_log: List[dict] = []

        # bar-local cache
        self._reset_bar_state()

    def _reset_bar_state(self):
        self.open_price: Optional[float] = None
        self.high_price: float = -np.inf
        self.low_price: float = np.inf
        self.cum_ticks: int = 0
        self.cum_volume: float = 0.0
        self.cum_buy_volume: float = 0.0
        self.cum_dollar_value: float = 0.0
        self.cum_theta: float = 0.0

    def _tick_rule(self, price: float) -> int:
        if self.prev_price is None:
            s = self.prev_tick_rule or 0
        else:
            dp = price - self.prev_price
            if dp > 0:
                s = 1
            elif dp < 0:
                s = -1
            else:
                s = self.prev_tick_rule or 0
        self.prev_price = price
        self.prev_tick_rule = s
        return 1 if s >= 0 else -1  # treat 0 as previous sign; defaults to +1 at start

    def _expected_imbalance(self) -> float:
        # E[b] via EWMA of recent signed ticks
        if len(self.signed_tick_hist) == 0:
            return np.nan
        w = min(len(self.signed_tick_hist), self.cfg.expected_imbalance_window)
        return float(ewma(np.asarray(self.signed_tick_hist[-w:], dtype=float), window=w)[-1])

    def _expected_num_ticks(self) -> float:
        if not self.ticks_per_bar_hist:
            return float(self.cfg.exp_num_ticks_init)
        w = min(len(self.ticks_per_bar_hist), self.cfg.num_prev_bars)
        est = float(ewma(np.asarray(self.ticks_per_bar_hist[-w:], dtype=float), window=w)[-1])
        return float(np.clip(est, self.cfg.exp_num_ticks_min, self.cfg.exp_num_ticks_max))

    def _threshold_hit(self, exp_imb: float, exp_T: float) -> bool:
        if np.isnan(exp_imb):
            return False
        return abs(self.cum_theta) > (abs(exp_imb) * exp_T)

    def _emit_bar_row(self, ts: pd.Timestamp, price: float, tick_num: int) -> tuple:
        high = max(self.high_price, self.open_price)
        low = min(self.low_price, self.open_price)
        return (
            ts, tick_num,
            self.open_price, high, low, price,
            self.cum_volume, self.cum_buy_volume,
            self.cum_ticks, self.cum_dollar_value
        )

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        df columns required: ['date_time','price','volume'].
        Returns:
          bars_df with columns:
            ['date_time','tick_num','open','high','low','close',
             'volume','cum_buy_volume','cum_ticks','cum_dollar_value']
          thresholds_df (optional diagnostic stream per tick).
        """
        req = ['date_time', 'price', 'volume']
        miss = [c for c in req if c not in df.columns]
        if miss:
            raise ValueError(f"Input DataFrame missing columns: {miss}")

        # ensure datetime dtype
        if not np.issubdtype(df['date_time'].dtype, np.datetime64):
            df = df.copy()
            df['date_time'] = pd.to_datetime(df['date_time'])

        bars = []
        exp_T = float(self.cfg.exp_num_ticks_init)

        for ts, p, v in df[['date_time', 'price', 'volume']].itertuples(index=False, name=None):
            self.global_tick_num += 1
            price = float(p)
            vol = float(v)
            dollar_val = price * vol

            # tick rule sign
            sgn = self._tick_rule(price)
            self.signed_tick_hist.append(sgn)

            # init open/high/low
            if self.open_price is None:
                self.open_price = price
            self.high_price = max(self.high_price, price)
            self.low_price = min(self.low_price, price)

            # bar accumulators
            self.cum_ticks += 1
            self.cum_volume += vol
            self.cum_dollar_value += dollar_val
            if sgn > 0:
                self.cum_buy_volume += vol
            self.cum_theta += sgn

            exp_b = self._expected_imbalance()
            if self.analyse_thresholds:
                self.threshold_log.append(
                    dict(timestamp=ts, cum_theta=self.cum_theta, exp_num_ticks=exp_T, exp_imbalance=exp_b)
                )

            if self._threshold_hit(exp_b, exp_T):
                bars.append(self._emit_bar_row(ts, price, self.global_tick_num))
                self.ticks_per_bar_hist.append(self.cum_ticks)

                # update expectations for next bar
                exp_T = self._expected_num_ticks()
                # reset for next bar
                self._reset_bar_state()

        cols = [
            'date_time','tick_num','open','high','low','close',
            'volume','cum_buy_volume','cum_ticks','cum_dollar_value'
        ]
        bars_df = pd.DataFrame(bars, columns=cols)
        thr_df = pd.DataFrame(self.threshold_log) if self.analyse_thresholds else pd.DataFrame()
        return bars_df, thr_df

# Convenience wrapper (CSV or DataFrame)
def get_ema_tick_imbalance_bars(
    data, *,
    num_prev_bars: int = 3,
    expected_imbalance_window: int = 1000,
    exp_num_ticks_init: int = 2000,
    exp_num_ticks_constraints: Tuple[int, float] = (1, np.inf),
    analyse_thresholds: bool = False
):
    """
    data: DataFrame with ['date_time','price','volume'] or a CSV path.
    Returns: (bars_df, thresholds_df)
    """
    if isinstance(data, str):
        df = pd.read_csv(data, parse_dates=['date_time'])
    else:
        df = data.copy()
        if 'date_time' not in df.columns:
            raise ValueError("DataFrame must include 'date_time'.")
        if not np.issubdtype(df['date_time'].dtype, np.datetime64):
            df['date_time'] = pd.to_datetime(df['date_time'])

    cfg = EMATickImbalanceConfig(
        num_prev_bars=num_prev_bars,
        expected_imbalance_window=expected_imbalance_window,
        exp_num_ticks_init=exp_num_ticks_init,
        exp_num_ticks_min=exp_num_ticks_constraints[0],
        exp_num_ticks_max=exp_num_ticks_constraints[1],
    )
    engine = EMATickImbalanceBars(cfg, analyse_thresholds=analyse_thresholds)
    return engine.transform(df)


# Irregular tick generator using an inhomogeneous Poisson process combined with a simple microstructure

"""
The inhomogeneous Poisson process gives a U-shaped intraday intensity, i.e. lots of activity at 
open/close, quiet mid-day and adds microstructure for prices (persistent buy/sell order flow, 
occasional zero-change prints with some multi-tick jumps) and volumes (lognormal, scaled by the 
same intraday seasonality).
"""

@dataclass
class TickGenConfig:
    # timeline
    start: Union[str, np.datetime64] = np.datetime64("2025-01-01T09:00:00")
    duration_seconds: int = 6*60*60 + 30*60  # 6.5h session
    # target count OR base intensity
    target_ticks: Optional[int] = 20000
    lambda_base: Optional[float] = None  # ticks per second, if provided overrides target_ticks
    # price process
    base_price: float = 100.0
    tick_size: float = 0.01
    p_zero: float = 0.15          # probability of "no price change" print
    p_same_sign: float = 0.6      # persistence of order flow
    p_jump: float = 0.001         # rare multi-tick jumps
    jump_geom_p: float = 0.35     # geometric tail for jump size in ticks (mean ~ (1-p)/p)
    # volume
    vol_lognorm_mean: float = 1.2
    vol_lognorm_sigma: float = 0.5
    min_volume: int = 1
    # reproducibility
    seed: int = 7

def u_shape_seasonality(u: np.ndarray) -> np.ndarray:
    """
    Intraday U-shape: high at the open/close, low mid-session.
    Range ~ [0.5, 1.5], mean ~ 2/3.
    """
    return 0.5 + 2.0*(u - 0.5)**2

def simulate_nhpp_times(cfg: TickGenConfig, seasonality_fn: Callable[[np.ndarray], np.ndarray] = u_shape_seasonality) -> np.ndarray:
    """
    Ogata thinning to sample event times (seconds from start) for NHPP with lambda(t) = lambda_base * seasonality(t/T).
    If cfg.lambda_base is None, it's set from cfg.target_ticks so that E[N] ≈ target_ticks.
    """
    rng = np.random.default_rng(cfg.seed)
    T = cfg.duration_seconds
    if cfg.lambda_base is None:
        mean_seasonality = 2.0/3.0
        lambda_base = (cfg.target_ticks or 10000) / (T * mean_seasonality)
    else:
        lambda_base = cfg.lambda_base
    
    # maximum intensity over the day
    lam_max = lambda_base * seasonality_fn(np.array([0.0, 1.0])).max() 
    
    t = 0.0
    times = []
    while t < T:
        t += rng.exponential(1.0 / lam_max)
        if t >= T:
            break
        u = t / T
        lam_t = lambda_base * float(seasonality_fn(np.array([u]))[0])
        if rng.random() < lam_t / lam_max:
            times.append(t)
    return np.array(times, dtype=float)

def generate_irregular_ticks(cfg: TickGenConfig) -> pd.DataFrame:
    """
    Produce a tick DataFrame with irregular timestamps using NHPP arrivals and simple microstructure.
    Columns: date_time, price, volume
    """
    rng = np.random.default_rng(cfg.seed)
    # event times (seconds offsets)
    arr = simulate_nhpp_times(cfg)
    # timestamps
    start = np.datetime64(cfg.start)
    timestamps = start + arr.astype("timedelta64[s]")
    
    # price path with microstructure:
    price = np.empty(len(arr), dtype=float)
    price[0] = cfg.base_price
    last_sign = rng.choice([-1, 1])
    for i in range(1, len(arr)):
        # zero-change print?
        if rng.random() < cfg.p_zero:
            dp = 0.0
        else:
            # persistent sign process
            if rng.random() < cfg.p_same_sign:
                sgn = last_sign
            else:
                sgn = -last_sign
            last_sign = sgn
            # occasional multi-tick jumps
            if rng.random() < cfg.p_jump:
                jump_ticks = 1 + rng.geometric(cfg.jump_geom_p)
            else:
                jump_ticks = 1
            dp = sgn * cfg.tick_size * jump_ticks
        # update price and clamp positive
        price[i] = max(cfg.tick_size, price[i-1] + dp)
    
    u = arr / (cfg.duration_seconds if cfg.duration_seconds else 1.0)
    season = u_shape_seasonality(u)  # ~[0.5,1.5]
    base_vol = rng.lognormal(mean=cfg.vol_lognorm_mean, sigma=cfg.vol_lognorm_sigma, size=len(arr))
    volume = np.maximum(cfg.min_volume, np.rint(base_vol * season)).astype(int)
    
    df = pd.DataFrame({"date_time": timestamps, "price": np.round(price, 2), "volume": volume})
    return df

# Generate a sample irregular tick stream & preview
cfg = TickGenConfig(
    start=np.datetime64("2025-01-01T09:00:00"),
    duration_seconds=6*60*60 + 30*60,
    target_ticks=20000,
    base_price=100.0,
    tick_size=0.01,
    p_zero=0.18,
    p_same_sign=0.62,
    p_jump=0.0015,
    jump_geom_p=0.35,
    vol_lognorm_mean=1.1,
    vol_lognorm_sigma=0.55,
    min_volume=1,
    seed=11
)

ticks_df = generate_irregular_ticks(cfg)

# quick sanity stats
import numpy as np
deltas = np.diff(ticks_df['date_time'].values.astype('datetime64[s]').astype('int64'))
print("ticks:", len(ticks_df))
print("mean interarrival (s):", float(np.mean(deltas)))
print("median interarrival (s):", float(np.median(deltas)))
print("min,max interarrival (s):", int(np.min(deltas)), int(np.max(deltas)))


# Build the bars
from IPython.display import display

bars_df, thr_df = get_ema_tick_imbalance_bars(
    ticks_df,
    num_prev_bars=3,
    expected_imbalance_window=200,
    exp_num_ticks_init=400,
    exp_num_ticks_constraints=(50, 2000),
    analyse_thresholds=True
)

print("Bars formed:", len(bars_df))
display(bars_df.head(10))
display(thr_df.tail(10))

# Candlestick chart of EMA Tick Imbalance Bars
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"

fig = go.Figure(
    data=[
        go.Candlestick(
            x=bars_df['date_time'],
            open=bars_df['open'],
            high=bars_df['high'],
            low=bars_df['low'],
            close=bars_df['close'],
            name="EMA Tick Imbalance Bars"
        )
    ]
)

fig.update_layout(
    title="EMA Tick Imbalance Bars — Candlestick",
    xaxis_title="Time",
    yaxis_title="Price",
    xaxis_rangeslider_visible=False,
    template="plotly_white"
)

fig.show()

# Applicaton

def apply_ema_tick_imbalance_bars_all_symbols(engineered_dir, results_dir):
    symbols = get_symbols(get_dates_from_most_active_files()[-1], top_n=17)[0]
    os.makedirs(results_dir, exist_ok=True)
    for symbol in symbols:
        file_path = os.path.join(engineered_dir, f"{symbol}_1d_features.json")
        if not os.path.exists(file_path):
            continue
        df = pd.read_json(file_path, orient='records', lines=True)
        if not {'Date', 'Close', 'Volume'}.issubset(df.columns):
            continue
        df = df.sort_values("Date")
        df["date_time"] = pd.to_datetime(df["Date"])
        df = df.rename(columns={"Close": "price", "Volume": "volume"})
        # Only keep required columns
        tick_df = df[["date_time", "price", "volume"]].copy()
        # Apply EMA Tick Imbalance Bars
        bars_df, thr_df = get_ema_tick_imbalance_bars(
            tick_df,
            num_prev_bars=3,
            expected_imbalance_window=200,
            exp_num_ticks_init=400,
            exp_num_ticks_constraints=(50, 2000),
            analyse_thresholds=True
        )
        # Save bars to CSV
        bars_path = os.path.join(results_dir, f"{symbol}_ema_tick_imbalance_bars.csv")
        bars_df.to_csv(bars_path, index=False)
        print(f"Saved EMA Tick Imbalance Bars for {symbol} to {bars_path}")

# Example usage:
if __name__ == "__main__":
    engineered_dir = "./Engineered_data"
    results_dir = "./results/tick_imbalance"
    apply_ema_tick_imbalance_bars_all_symbols(engineered_dir, results_dir)
                       