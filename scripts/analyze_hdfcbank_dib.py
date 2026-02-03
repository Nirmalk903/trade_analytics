import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    from statsmodels.graphics.tsaplots import plot_acf
    from statsmodels.stats.stattools import durbin_watson
except Exception:
    raise SystemExit("Install statsmodels: pip install statsmodels")

# Serial correlation (lag-1)
lag1 = returns.autocorr(lag=1)
print(f"HDFCBANK DIB: number of bars = {len(df)}, number of returns = {len(returns)}")
print(f"Lag-1 serial correlation (returns) = {lag1:.6f}")

# Durbin-Watson statistic
dw = float(durbin_watson(returns.values))
print(f"Durbin-Watson statistic = {dw:.6f}")

# Plot ACF
fig, ax = plt.subplots(figsize=(9,4))
plot_acf(returns, lags=40, ax=ax, zero=False)
ax.set_title(f"ACF of HDFCBANK Dollar-Imbalance-Bar Returns\nLag-1={lag1:.4f}  DW={dw:.4f}")
plt.tight_layout()
fig.savefig(OUT_PNG, dpi=150)
print(f"ACF plot saved to: {OUT_PNG}")