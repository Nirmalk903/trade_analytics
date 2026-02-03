# filepath: c:\Users\nirma\OneDrive\MyProjects\trading_app\algorithms\test.py
# ...existing code...
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    from statsmodels.graphics.tsaplots import plot_acf
    from statsmodels.stats.stattools import durbin_watson, jarque_bera
except Exception:
    raise SystemExit("Install statsmodels: pip install statsmodels")

def analyze_bars(csv_path: str, out_png: str, title_prefix: str):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path, parse_dates=['date_time'], dayfirst=False)
    if df.empty:
        print(f"File is empty: {csv_path}")
        return

    close_col = None
    for c in ("close", "Close"):
        if c in df.columns:
            close_col = c
            break
    if close_col is None:
        print(f"No close column in {csv_path}; columns: {list(df.columns)}")
        return

    df = df.sort_values("date_time").reset_index(drop=True)
    df['log_close'] = np.log(df[close_col].replace(0, np.nan))
    df['ret'] = df['log_close'].diff()
    returns = df['ret'].dropna()
    if returns.empty:
        print(f"No returns available in {csv_path}")
        return

    lag1 = returns.autocorr(lag=1)
    dw = float(durbin_watson(returns.values))

    jb_res = jarque_bera(returns.values)
    # statsmodels.jarque_bera returns (jbstat, pvalue, skew, kurtosis)
    if hasattr(jb_res, "__len__") and len(jb_res) >= 2:
        jb_stat = float(jb_res[0])
        jb_p = float(jb_res[1])
    else:
        jb_stat = float(jb_res)
        jb_p = np.nan

    print(f"{title_prefix}: bars={len(df)}, returns={len(returns)}, lag1={lag1:.6f}, DW={dw:.6f}, JB={jb_stat:.4f}, JB_p={jb_p:.4g}")

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9,4))
    plot_acf(returns, lags=40, ax=ax, zero=False)
    ax.set_title(f"{title_prefix} ACF  (Lag-1={lag1:.4f}  DW={dw:.4f}  JB={jb_stat:.2f} p={jb_p:.3g})")
    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"ACF plot saved to {out_png}")

if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    # Dollar imbalance bars
    symbol = 'RELIANCE'
    dib_csv = os.path.join(PROJECT_ROOT, "results", "dollar_imbalance", F"{symbol}_dollar_imbalance_bars.csv")
    dib_png = os.path.join(PROJECT_ROOT, "results", "dollar_imbalance", F"{symbol}_dib_acf.png")
    analyze_bars(dib_csv, dib_png, F"{symbol} Dollar-Imbalance-Bar Returns")

    # Tick imbalance bars
    tib_csv = os.path.join(PROJECT_ROOT, "results", "tick_imbalance", F"{symbol}_ema_tick_imbalance_bars.csv")
    tib_png = os.path.join(PROJECT_ROOT, "results", "tick_imbalance", F"{symbol}_tib_acf.png")
    analyze_bars(tib_csv, tib_png, F"{symbol} Tick-Imbalance-Bar Returns")