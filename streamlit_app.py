import streamlit as st
import pandas as pd
# from market_calendar import stock_earnings_calendar
from PIL import Image
from feature_engineering import process_symbol, create_underlying_analytics
from plotting import plot_garch_vs_rsi, plot_garch_vs_avg_iv
from data_download_vbt import getdata_vbt, get_underlying_data_vbt, get_symbols,  get_dates_from_most_active_files
import os
import time
from datetime import datetime as dt_time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplfinance as mpf
import plotly.graph_objects as go
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from algorithms.cusum_filter import getTEvents, read_series_from_file, find_dollar_imbalance_file

st.title("Trading Analytics Dashboard")

st.write("Use the controls below to select date and filter symbols, then click 'Run Analytics'.")

# Select date and top N symbols (robust handling of various return types)
dates_raw = get_dates_from_most_active_files()
# normalize into a list of pandas Timestamps safely
dates_list = []
if dates_raw is None:
    dates_list = []
else:
    try:
        # if dates_raw is iterable (Index, list, ndarray)
        dates_list = list(pd.to_datetime(list(dates_raw)))
    except Exception:
        try:
            # single value fallback
            dates_list = [pd.to_datetime(dates_raw)]
        except Exception:
            dates_list = []

if len(dates_list) > 0:
    dates_list = sorted(dates_list)
    date_strs = [d.date().isoformat() for d in dates_list]
    # show newest first in the selectbox and default-select newest
    selected_date = st.selectbox("Select Date", date_strs[::-1], index=0)
else:
    # fallback: no most-active dates available
    today_str = pd.Timestamp.today().date().isoformat()
    st.warning("No most-active dates found — defaulting to today.")
    selected_date = st.selectbox("Select Date", [today_str], index=0)

# Select number of top symbols
op_n = st.slider("Number of Top Symbols", 1, 20, 10)

# Convert selected_date string back to datetime for get_symbols
selected_date_dt = pd.to_datetime(selected_date)

all_symbols = get_symbols(selected_date_dt, top_n=op_n)[0]

# # Add filters: multi-select for symbols
st.markdown(
    """
    <style>
    .symbol-font .stMultiSelect label, .symbol-font .stMultiSelect span {
        font-size: 0.3em !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
selected_symbols = st.multiselect(
    "Filter and Select Symbols", options=all_symbols, default=all_symbols, key=f"symbol_multiselect_{op_n}", help=None
)

# --- Feature Engineering Step (parallelized and cached) ---

if st.button("Run Analytics"):
    st.info("Running feature engineering for selected symbols. Please wait...")
    progress_bar = st.progress(0, text="Starting...")

    start_time = time.time()  # Start timer

    with st.spinner("Processing features..."):
        # Remove parallel execution, use create_underlying_analytics for all selected symbols
        create_underlying_analytics(selected_symbols)
        progress_bar.progress(1.0, text=f"Processed {len(selected_symbols)}/{len(selected_symbols)} symbols")

    elapsed = time.time() - start_time  # End timer
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    st.success(f"Feature engineering completed for selected symbols! Time taken: {minutes} min {seconds} sec.")
    progress_bar.empty()
    # Refresh the app so the UI reads newly written engineered files / images
    try:
        # preferred API (works on many Streamlit versions)
        st.experimental_rerun()
    except AttributeError:
        # fallback: raise Streamlit's internal RerunException if available
        try:
            from streamlit.runtime.scriptrunner.script_runner import RerunException
            raise RerunException()
        except Exception:
            # final fallback: ask user to manually reload and stop execution
            st.warning("Refresh requested but automatic rerun is not available in this Streamlit build. Please reload the page.")
            st.stop()


# Plot GARCH vs RSI

st.subheader(f"GARCH Vol Percentile vs RSI  {selected_date}")
df1 = plot_garch_vs_rsi(selected_symbols)

image_path = os.path.join("Images", f"garch_vs_rsi_{selected_date}.png")
if os.path.exists(image_path):
    st.image(Image.open(image_path), caption="GARCH vs RSI Scatter Plot", width='stretch')
else:
    st.warning(f"Image not found: {image_path}")

# Plot Historical Chart
st.header("Plot Historical Chart")
hist_symbol = st.selectbox(
    "Select stock for historical chart", options=selected_symbols, key="hist_symbol"
)

# Select number of days to show (period)
period_options = {
    "6 Months": 126,
    "1 Year": 252,
    "2 Years": 504,
    "All": None
}
selected_period_label = st.selectbox(
    "Select period", list(period_options.keys()), index=2
)
num_days = period_options[selected_period_label]

# CUSUM controls
show_cusum = st.checkbox("Show CUSUM Events", value=True, key="show_cusum_on_chart")
cusum_threshold_mult = st.slider("CUSUM threshold (std multiplier)", 0.05, 1.0, 0.2, 0.05, key="cusum_threshold_hist") if show_cusum else 0.2

if st.button("Show Historical Chart"):
    feature_file = os.path.join("Engineered_data", f"{hist_symbol}_1d_features.json")
    if os.path.exists(feature_file):
        # read fresh, parse dates robustly
        df_hist = pd.read_json(feature_file, orient='records', lines=True)
        if "Date" not in df_hist.columns:
            # try to recover Date from index or first column
            if df_hist.index.name == "Date":
                df_hist = df_hist.reset_index()
        df_hist["Date"] = pd.to_datetime(df_hist.get("Date", None), errors="coerce")
        if df_hist["Date"].isna().all():
            st.warning(f"Feature file for {hist_symbol} is missing parsable 'Date' values.")
        else:
            df_hist = df_hist.sort_values("Date")
            df_hist["Date"] = pd.to_datetime(df_hist["Date"])
            # Calculate moving averages
            df_hist["MA_10"] = df_hist["Close"].rolling(window=10).mean()
            df_hist["MA_50"] = df_hist["Close"].rolling(window=50).mean()
            df_hist["MA_100"] = df_hist["Close"].rolling(window=100).mean()
            
            df_hist_recent = df_hist.tail(num_days) if num_days is not None else df_hist

            # --- Add summary table for latest row ---
            latest = df_hist_recent.iloc[-1]
            summary_dict = {
                "Latest Price": latest["Close"],
                "Daily Return": latest.get("Returns", None),
                "GARCH Volatility": latest.get("garch_vol", None),
                "GARCH Volatility Percentile": latest.get("garch_vol_percentile", None),
                "Daily CPR": latest.get("dCPR", None),
                "RSI": latest.get("RSI", None),
                "RSI Percentile": latest.get("RSI_percentile", None),
                "Weekly RSI": latest.get("RSI_weekly", None),
                "Weekly RSI Percentile": latest.get("RSI_percentile_weekly", None)
            }
            # Format numeric values: 'Daily Return' as percentage with one decimal, others as 0 decimals
            for k, v in summary_dict.items():
                if k == "Daily Return" and isinstance(v, (int, float)) and v is not None:
                    summary_dict[k] = f"{v*100:.1f}%"
                elif isinstance(v, (int, float)) and v is not None:
                    summary_dict[k] = f"{v:.0f}"
            summary_df = pd.DataFrame([summary_dict])

            # --- Display as tabular format with heading ---
            st.subheader(f"Stock Analysis - {selected_date}")
            st.table(summary_df)

            # --- Existing plotting code ---
            # Check for minimum data length
            min_rows = 100
            if len(df_hist_recent) < min_rows:
                st.warning(f"Not enough data to plot all indicators (need at least {min_rows} rows, got {len(df_hist_recent)}).")
            else:
                df_mpf = df_hist_recent.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]]

                apds = []
                # Only add plots if the data is not all-NaN
                if not np.all(np.isnan(df_hist_recent["wCPR"].values)):
                    apds.append(mpf.make_addplot(df_hist_recent["wCPR"].values, panel=0,
                                                 type='scatter', markersize=0.5, color='blue', marker='o', ylabel='wCPR'))
                if not np.all(np.isnan(df_hist_recent["MA_10"].values)):
                    apds.append(mpf.make_addplot(df_hist_recent["MA_10"].values, panel=0,
                                                 type='line', color='green', width=1.2, ylabel='MA 10'))
                if not np.all(np.isnan(df_hist_recent["MA_50"].values)):
                    apds.append(mpf.make_addplot(df_hist_recent["MA_50"].values, panel=0,
                                                 type='line', color='orange', width=1.2, ylabel='MA 50'))
                if not np.all(np.isnan(df_hist_recent["MA_100"].values)):
                    apds.append(mpf.make_addplot(df_hist_recent["MA_100"].values, panel=0,
                                                 type='line', color='purple', width=1.2, ylabel='MA 100'))
                
                # Add CUSUM markers to panel 0 if enabled
                if show_cusum:
                    try:
                        close_ser = pd.Series(df_hist_recent["Close"].values, index=df_hist_recent["Date"])
                        std = float(close_ser.std()) if close_ser.size > 1 else 0.0
                        if not np.isfinite(std) or std <= 0:
                            h = max(1e-6, float(abs(close_ser).median() * 0.1))
                        else:
                            h = max(1e-6, std * float(cusum_threshold_mult))
                        
                        t_events, diag = getTEvents(close_ser, h)
                        
                        if len(t_events) > 0:
                            triggers = [diag.loc[t, "trigger"] if t in diag.index else 0 for t in t_events]
                            # Create scatter arrays aligned with df_hist_recent index
                            cusum_pos = np.full(len(df_hist_recent), np.nan)
                            cusum_neg = np.full(len(df_hist_recent), np.nan)
                            
                            # Reset index to use positional indexing
                            df_temp = df_hist_recent.reset_index(drop=True)
                            
                            for t, tr in zip(t_events, triggers):
                                if t in close_ser.index:
                                    # Find positional index (iloc position) not label index
                                    mask = df_temp["Date"] == t
                                    if mask.any():
                                        pos_idx = mask.idxmax()  # Gets first True position
                                        if tr == 1:
                                            cusum_pos[pos_idx] = close_ser.loc[t]
                                        elif tr == -1:
                                            cusum_neg[pos_idx] = close_ser.loc[t]
                            
                            # Add CUSUM markers
                            apds.append(mpf.make_addplot(cusum_pos, panel=0, type='scatter', 
                                                         markersize=100, marker='^', color='lime', 
                                                         edgecolors='darkgreen', linewidths=1.5, ylabel='CUSUM+'))
                            apds.append(mpf.make_addplot(cusum_neg, panel=0, type='scatter', 
                                                         markersize=100, marker='v', color='red', 
                                                         edgecolors='darkred', linewidths=1.5, ylabel='CUSUM-'))
                    except Exception as e:
                        st.warning(f"Could not add CUSUM overlay: {e}")
                
                if not np.all(np.isnan(df_hist_recent["RSI"].values)):
                    apds.append(mpf.make_addplot(df_hist_recent["RSI"].values, panel=1,
                                                 type='line', color='grey', width=1.2, ylabel='RSI'))
                if not np.all(np.isnan(df_hist_recent["garch_vol"].values)):
                    apds.append(mpf.make_addplot(df_hist_recent["garch_vol"].values, panel=2,
                                                 type='bar', color='red', width=1.2, ylabel='Volatility'))
                if not np.all(np.isnan(df_hist_recent["garch_vol_percentile"].values)):
                    apds.append(mpf.make_addplot(df_hist_recent["garch_vol_percentile"].values, panel=3,
                                                 type='line', color='orange', width=1.2, ylabel='VolP'))

                # Add squared returns to panel 4 (5th panel, index 4) using 'Returns' column from engineered data
                if "Returns" in df_hist_recent.columns:
                    squared_returns = df_hist_recent["Returns"] ** 2
                    if not np.all(np.isnan(squared_returns.values)):
                        apds.append(mpf.make_addplot(
                            squared_returns.values,
                            panel=4,
                            type='line',
                            color='brown',
                            width=1.2,
                            ylabel='Squared Ret'
                        ))

                panel_ratios = (6, 1, 1, 1, 1)  # Add extra panel for squared returns

                fig, axlist = mpf.plot(
                    df_mpf,
                    type='line',
                    style='yahoo',
                    addplot=apds,
                    panel_ratios=panel_ratios,
                    returnfig=True,
                    figsize=(10, 10)
                )

                # Add symbol name to center top of OHLC panel
                axlist[0].text(
                    0.5, 0.98, f"{hist_symbol}",
                    transform=axlist[0].transAxes,
                    fontsize=16,
                    fontweight='bold',
                    va='top',
                    ha='center',
                    color='navy',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3')
                )

                # Add custom legend for moving averages
                ma_lines = []
                ma_labels = []
                if not np.all(np.isnan(df_hist_recent["MA_10"].values)):
                    ma_lines.append(axlist[0].plot([], [], color='green', linewidth=2)[0])
                    ma_labels.append('MA 10')
                if not np.all(np.isnan(df_hist_recent["MA_50"].values)):
                    ma_lines.append(axlist[0].plot([], [], color='orange', linewidth=2)[0])
                    ma_labels.append('MA 50')
                if not np.all(np.isnan(df_hist_recent["MA_100"].values)):
                    ma_lines.append(axlist[0].plot([], [], color='purple', linewidth=2)[0])
                    ma_labels.append('MA 100')
                
                # Add CUSUM to legend if shown
                if show_cusum:
                    ma_lines.append(axlist[0].plot([], [], marker='^', color='lime', linestyle='None', markersize=8, markeredgecolor='darkgreen')[0])
                    ma_labels.append('CUSUM +1')
                    ma_lines.append(axlist[0].plot([], [], marker='v', color='red', linestyle='None', markersize=8, markeredgecolor='darkred')[0])
                    ma_labels.append('CUSUM -1')

                if ma_lines:
                    axlist[0].legend(ma_lines, ma_labels, loc='upper left')

                st.pyplot(fig, width='stretch')
    else:
        st.warning(f"Feature file not found for {hist_symbol}")



# --- Summary table for all symbols (latest row for each) ---
summary_rows = []
for symbol in selected_symbols:
    feature_file = os.path.join("Engineered_data", f"{symbol}_1d_features.json")
    if not os.path.exists(feature_file):
        continue

    # always read fresh from disk to avoid stale cached data
    try:
        df = pd.read_json(feature_file, orient='records', lines=True)
    except Exception:
        continue

    # recover Date column if stored as index
    if "Date" not in df.columns and df.index.name == "Date":
        df = df.reset_index()

    if df.empty:
        continue

    # robust date parsing
    df["Date"] = pd.to_datetime(df.get("Date", None), errors="coerce")

    # skip if no parsable dates
    if df["Date"].isna().all():
        continue

    # find the true latest date (use max, not last row)
    max_date = df["Date"].max()
    latest_rows = df.loc[df["Date"] == max_date]

    if latest_rows.empty:
        continue

    # if multiple rows have same latest date pick the last one (most recent in file)
    latest = latest_rows.sort_values("Date").iloc[-1]

    summary_rows.append({
        "Symbol": symbol,
        "Date": pd.to_datetime(latest["Date"]).date(),
        "Latest Price": latest.get("Close", np.nan),
        "Daily Return": latest.get("Returns", None),
        "GARCH Volatility": latest.get("garch_vol", None),
        "GARCH Volatility Percentile": latest.get("garch_vol_percentile", None),
        "Vol_Change": latest.get("garch_vol_pct", None),
        "Daily CPR": latest.get("dCPR", None),
        "RSI": latest.get("RSI", None),
        "RSI Percentile": latest.get("RSI_percentile", None),
        "Weekly RSI": latest.get("RSI_weekly", None),
        "Weekly RSI Percentile": latest.get("RSI_percentile_weekly", None)
    })

# compute header latest date from collected rows (use max across symbols)
if summary_rows:
    overall_latest = max([r["Date"] for r in summary_rows])
    summary_all_df = pd.DataFrame(summary_rows)
    # Remove the 'Date' column and keep overall_latest for header
    summary_all_df = summary_all_df.drop(columns=["Date"])
    latest_date = overall_latest
else:
    summary_all_df = pd.DataFrame()
    latest_date = ""

if not summary_all_df.empty:
    # Keep a copy of the original numeric columns for sorting
    numeric_cols = ["Latest Price", "Daily Return", "GARCH Volatility", "GARCH Volatility Percentile","Vol_Change",
                    "Daily CPR", "RSI", "RSI Percentile", "Weekly RSI", "Weekly RSI Percentile"]

    # Convert numeric columns to float for sorting
    for col in numeric_cols:
        if col in summary_all_df.columns:
            summary_all_df[col] = pd.to_numeric(summary_all_df[col], errors='coerce')

    # Format columns for display
    for col in summary_all_df.columns:
        if col == "Daily Return":
            summary_all_df[col] = summary_all_df[col].apply(
                lambda x: f"{x*100:.1f}%" if pd.notnull(x) and isinstance(x, (int, float, np.floating)) else ""
            )
        elif col == "Vol_Change":
            summary_all_df[col] = summary_all_df[col].apply(
                lambda x: f"{x:.2f}%" if pd.notnull(x) and isinstance(x, (int, float, np.floating)) else ""
            )
        elif col in numeric_cols and col not in ["Daily Return", "Vol_Change"]:
            summary_all_df[col] = summary_all_df[col].apply(
                lambda x: f"{x:.0f}" if pd.notnull(x) else "")

    # Reset index to start from 1
    summary_all_df.index = summary_all_df.index + 1

    # --- Highlight Daily Return: green for positive, red for negative, center align ---
    def highlight_daily_return(val):
        try:
            num = float(val.replace('%', ''))
            color = 'green' if num > 0 else 'red' if num < 0 else 'black'
            return f'color: {color}; text-align: center;'
        except:
            return 'text-align: center;'

    styled_df = summary_all_df.style.map(highlight_daily_return, subset=['Daily Return']) \
                                    .set_properties(**{'text-align': 'center'})

    st.subheader(f"Stock Analysis - {selected_date}")
    st.dataframe(styled_df, width='stretch')

    # --- Place the Refresh button and logic here ---
    if st.button("Refresh Summary Table", key="refresh_summary_table_bottom"):
        st.info("Refreshing summary table with latest engineered data...")
        # st.experimental_rerun()

else:
    st.warning("No feature files found for the selected symbols.")


# Upcoming Earnings, Dividends & Corporate Actions

# --- Correlation Matrix for Selected Stocks ---
st.header("Correlation Matrix for Selected Stocks")

close_prices = {}
for symbol in selected_symbols:
    feature_file = os.path.join("Engineered_data", f"{symbol}_1d_features.json")
    if os.path.exists(feature_file):
        df = pd.read_json(feature_file, orient='records', lines=True)
        if "Date" not in df.columns:
            if df.index.name == "Date":
                df = df.reset_index()
        if "Date" not in df.columns or df.empty:
            continue
        df = df.sort_values("Date")
        df["Date"] = pd.to_datetime(df["Date"])
        close_prices[symbol] = df.set_index("Date")["Close"]

if close_prices:
    close_df = pd.DataFrame(close_prices)
    returns_df = close_df.pct_change(fill_method=None)
    corr_matrix = returns_df.corr()
    st.subheader("Correlation Matrix (Daily Returns)")
    st.dataframe(
        corr_matrix.style.format("{:.2f}").background_gradient(cmap='coolwarm'),
        width='stretch'
    )
else:
    st.warning("Not enough data to compute correlation matrix for selected stocks.")