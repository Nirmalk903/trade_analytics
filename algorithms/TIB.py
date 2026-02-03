import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

class ImbalanceBars:
    def __init__(self, bar_type="tick", lam_E0=0.9, lam_p=0.9, init_E0=100, warmup_bars=5):
        self.bar_type = bar_type
        self.lam_E0 = lam_E0
        self.lam_p = lam_p
        self.init_E0 = init_E0
        self.warmup_bars = warmup_bars

        self.reset_estimators()
        self.reset_bar()

    def reset_estimators(self):
        # initialize E0 to the configured initial value so bars can start forming
        self.E0 = float(self.init_E0)
        self.warmup_T = []
        self.P_buy = 0.5

    def reset_bar(self):
        self.ticks = []
        self.signs = []
        self.imbalance = 0.0
        self.last_sign = 1

    def tick_rule(self, price, prev_price):
        if price > prev_price:
            return 1
        elif price < prev_price:
            return -1
        else:
            return self.last_sign

    def update_estimators(self, realized_T, buy_flags):
        # update E0 by EWMA using realized T
        try:
            self.E0 = (1 - self.lam_E0) * float(realized_T) + self.lam_E0 * float(self.E0)
        except Exception:
            # fallback: set to realized_T if something odd
            self.E0 = float(realized_T)

        if buy_flags:
            p_bar = np.mean([1 if b == 1 else 0 for b in buy_flags])
            self.P_buy = (1 - self.lam_p) * p_bar + self.lam_p * self.P_buy

    def expected_imbalance(self):
        # base expected count (use init if E0 somehow invalid)
        base = float(self.E0) if (self.E0 is not None and np.isfinite(self.E0) and self.E0 > 0) else float(self.init_E0)
        abs_term = abs(2 * self.P_buy - 1)
        # avoid zero threshold when P_buy == 0.5 -> use base
        if abs_term < 1e-6:
            return base
        return base * abs_term

    def process_tick(self, price, volume):
        if len(self.ticks) == 0:
            sign = self.last_sign
        else:
            prev_price = self.ticks[-1][0]
            sign = self.tick_rule(price, prev_price)
        self.last_sign = sign

        if self.bar_type == "tick":
            contrib = sign * 1
        elif self.bar_type == "volume":
            contrib = sign * volume
        elif self.bar_type == "dollar":
            contrib = sign * price * volume
        else:
            raise ValueError("bar_type must be 'tick', 'volume', or 'dollar'")

        self.signs.append(sign)
        self.imbalance += contrib
        self.ticks.append((price, volume))

        if abs(self.imbalance) >= self.expected_imbalance():
            bar = self.form_bar()
            realized_T = len(self.ticks)
            self.update_estimators(realized_T, self.signs)
            self.reset_bar()
            return bar
        return None

    def form_bar(self):
        prices = [p for p, v in self.ticks]
        volumes = [v for p, v in self.ticks]
        return {
            "open": prices[0],
            "high": max(prices),
            "low": min(prices),
            "close": prices[-1],
            "volume": sum(volumes),
            "ticks": len(prices),
            "imbalance": self.imbalance
        }

    def batch_run(self, prices, volumes):
        bars = []
        for p, v in zip(prices, volumes):
            bar = self.process_tick(p, v)
            if bar:
                bars.append(bar)
        return pd.DataFrame(bars)


# 🔹 Plotting Utility
def plot_bars(prices, bars_dict):
    """
    prices : list of raw tick prices
    bars_dict : dict { "TIB": df, "VIB": df, "DIB": df }
    """
    plt.figure(figsize=(12, 6))
    plt.plot(prices, color="gray", alpha=0.5, label="Tick Prices")

    colors = {"TIB": "blue", "VIB": "green", "DIB": "red"}
    for name, df in bars_dict.items():
        if not df.empty:
            plt.scatter(df.index, df["close"], label=name, color=colors[name], s=60)

    plt.title("Information-Driven Bars (TIB vs VIB vs DIB)")
    plt.xlabel("Tick index")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()


# 🔹 Example Usage
if __name__ == "__main__":
    # input and output directories
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    input_dir = os.path.join(project_root, "Underlying_data_vbt")
    out_dir = os.path.join(project_root, "results", "TIB")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(input_dir):
        print(f"Input folder not found: {input_dir}")
    else:
        files = [f for f in os.listdir(input_dir) if f.lower().endswith(".csv")]
        print(f"Found {len(files)} files in {input_dir}")

        for fname in files:
            fpath = os.path.join(input_dir, fname)
            symbol = os.path.splitext(fname)[0].split("_")[0]
            try:
                # attempt to read with header and parse first column as date
                try:
                    df = pd.read_csv(fpath, parse_dates=[0], infer_datetime_format=True)
                except Exception:
                    # fallback if headerless
                    df = pd.read_csv(fpath, header=None)
                if df.empty:
                    print(f"{symbol}: file empty, skipping")
                    continue

                # determine price and volume columns robustly
                price_col = None
                vol_col = None
                cols = [c.lower() for c in df.columns.astype(str)]

                if "price" in cols:
                    price_col = df.columns[cols.index("price")]
                elif "close" in cols:
                    price_col = df.columns[cols.index("close")]
                elif df.shape[1] >= 2:
                    # try common position (second column or fifth for OHLC)
                    if df.shape[1] >= 5:
                        price_col = df.columns[4]  # close in many OHLC files
                    else:
                        price_col = df.columns[1]

                if "volume" in cols:
                    vol_col = df.columns[cols.index("volume")]
                elif df.shape[1] >= 6:
                    vol_col = df.columns[5]
                else:
                    # fallback to ones if no volume column
                    vol_col = None

                prices = df[price_col].astype(float).values if price_col is not None else df.iloc[:, 1].astype(float).values
                if vol_col is not None:
                    volumes = df[vol_col].fillna(0).astype(float).values
                else:
                    volumes = np.ones_like(prices).astype(float)

                if len(prices) == 0:
                    print(f"{symbol}: no price data, skipping")
                    continue

                # create bar generators
                tib = ImbalanceBars(bar_type="tick", warmup_bars=5)
                vib = ImbalanceBars(bar_type="volume", warmup_bars=5)
                dib = ImbalanceBars(bar_type="dollar", warmup_bars=5)

                # run
                bars_tick = tib.batch_run(prices, volumes)
                bars_vol = vib.batch_run(prices, volumes)
                bars_dol = dib.batch_run(prices, volumes)

                # attach symbol and save
                def save_df(df_bars, suffix):
                    if df_bars is None:
                        df_bars = pd.DataFrame()
                    df_out = df_bars.copy()
                    df_out["symbol"] = symbol
                    out_path = os.path.join(out_dir, f"{symbol}_{suffix}.csv")
                    df_out.to_csv(out_path, index=False)
                    print(f"Wrote {len(df_out)} rows to {out_path}")

                save_df(bars_tick, "TIB")
                save_df(bars_vol, "VIB")
                save_df(bars_dol, "DIB")

            except Exception as e:
                print(f"Failed processing {fname}: {e}")