import json
import time
from pathlib import Path

import mplfinance as mpf
import news_heasines as news_headlines
import numpy as np
import pandas as pd
import pendulum
import pytz
import streamlit as st

from algorithms.cusum_filter import getTEvents
from data_download_vbt import get_dates_from_most_active_files, get_symbols
from feature_engineering import create_underlying_analytics
from plotting import plot_garch_vs_rsi
from calendar_fetcher import fetch_and_save, filter_and_save


st.set_page_config(page_title="Trading Analytics Dashboard", layout="wide")

ENGINEERED_DIR = Path("Engineered_data")
IMAGES_DIR = Path("Images")
CALENDAR_DIR = Path("results/economic_calendar")
SETTINGS_FILE = Path.home() / ".algotrading_settings.json"

regions_high_only = ["Americas", "Europe", "Japan", "China"]
regions_keep_all = ["India"]


def _find_calendar_file() -> Path | None:
    candidates = []
    candidates.extend(sorted(CALENDAR_DIR.glob("economic_calendar_filtered*.csv")))
    candidates.extend(sorted(CALENDAR_DIR.glob("economic_calendar*.csv")))
    candidates.extend(sorted(CALENDAR_DIR.glob("All_Regions_calendar_*.csv")))
    if candidates:
        return candidates[-1]
    return None


def _infer_region(df: pd.DataFrame) -> pd.DataFrame:
    if "Region" in df.columns:
        return df

    region_mapping = {
        'Americas': ['USD','United States',],
        'Europe': ['EUR', 'GBP','Eurozone', 'Germany', 'France', 'UK'],
        'India': ['INR', 'India'],
        'China': ['CNY', 'CNH', 'China'],
        'Japan': ['JPY', 'Japan'],
    }

    def infer_region(row):
        country = str(row.get('Country', '')).strip()
        currency = str(row.get('Currency', '')).strip()
        for region, tokens in region_mapping.items():
            if country in tokens or currency in tokens:
                return region
        return 'Unknown'

    df = df.copy()
    df['Region'] = df.apply(infer_region, axis=1)
    return df


def _filter_calendar_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = _infer_region(df)
    region_series = df.get("Region", pd.Series(index=df.index, dtype="object"))
    importance_series = df.get("Importance", pd.Series(index=df.index, dtype="object"))

    mask_high_only = region_series.isin(regions_high_only) & (importance_series != "High")
    return df[~mask_high_only].copy()


def load_filtered_calendar(save: bool = True):
    calendar_file = _find_calendar_file()
    if not calendar_file:
        return None, "No combined economic calendar CSV found."

    df = pd.read_csv(calendar_file)
    if df.empty:
        return df, "Calendar file is empty."

    df_filtered = _filter_calendar_df(df)

    if save:
        CALENDAR_DIR.mkdir(parents=True, exist_ok=True)
        output_file = CALENDAR_DIR / "economic_calendar_filtered.csv"
        df_filtered.to_csv(output_file, index=False)

    return df_filtered, None

def _safe_to_datetime_list(dates_raw) -> list[pd.Timestamp]:
    if dates_raw is None:
        return []
    try:
        return list(pd.to_datetime(list(dates_raw)))
    except Exception:
        try:
            return [pd.to_datetime(dates_raw)]
        except Exception:
            return []


@st.cache_data(ttl=300)
def _read_feature_df(symbol: str) -> pd.DataFrame:
    feature_file = ENGINEERED_DIR / f"{symbol}_1d_features.json"
    if not feature_file.exists():
        return pd.DataFrame()

    try:
        df = pd.read_json(feature_file, orient="records", lines=True)
    except Exception:
        return pd.DataFrame()

    if "Date" not in df.columns and df.index.name == "Date":
        df = df.reset_index()

    if "Date" not in df.columns:
        return pd.DataFrame()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")
    return df


def _rerun_app():
    try:
        st.rerun()
    except AttributeError:
        try:
            st.experimental_rerun()
        except Exception:
            st.warning("Refresh requested but automatic rerun is not available in this Streamlit build. Please reload the page.")
            st.stop()


st.title("Trading Analytics Dashboard")

# Sidebar controls
with st.sidebar:
    st.header("Controls")

    # Select date and top N symbols (robust handling of various return types)
    dates_raw = get_dates_from_most_active_files()
    dates_list = _safe_to_datetime_list(dates_raw)

    if len(dates_list) > 0:
        dates_list = sorted(dates_list)
        date_strs = [d.date().isoformat() for d in dates_list]
        # show newest first in the selectbox and default-select newest
        selected_date = st.selectbox("Select Date", date_strs[::-1], index=0, key="sidebar_date")
    else:
        # fallback: no most-active dates available
        today_str = pd.Timestamp.today().date().isoformat()
        st.warning("No most-active dates found — defaulting to today.")
        selected_date = st.selectbox("Select Date", [today_str], index=0, key="sidebar_date_fallback")

    # Select number of top symbols
    op_n = st.slider("Number of Top Symbols", 1, 50, 10, key="sidebar_topn")

    # Convert selected_date string back to datetime for get_symbols
    selected_date_dt = pd.to_datetime(selected_date)

    all_symbols = get_symbols(selected_date_dt, top_n=op_n)[0]

    # symbol filter
    st.markdown("""
    <style>
    .symbol-font .stMultiSelect label, .symbol-font .stMultiSelect span {
        font-size: 0.85em !important;
    }
    </style>
    """, unsafe_allow_html=True)

    selected_symbols = st.multiselect(
        "Filter and Select Symbols",
        options=all_symbols,
        default=all_symbols,
        key=f"symbol_multiselect_{op_n}",
    )

    st.divider()

    # Historical chart controls in sidebar
    st.subheader("Chart Settings")
    hist_symbol = st.selectbox("Historical chart symbol", options=selected_symbols or all_symbols, key="sidebar_hist_symbol")
    period_options = {"6 Months": 126, "1 Year": 252, "2 Years": 504, "All": None}
    selected_period_label = st.selectbox("Select period", list(period_options.keys()), index=2, key="sidebar_period")
    num_days = period_options[selected_period_label]

    show_cusum = st.checkbox("Show CUSUM Events", value=True, key="sidebar_show_cusum")
    cusum_threshold_mult = st.slider("CUSUM threshold (std multiplier)", 0.05, 1.0, 0.2, 0.05, key="sidebar_cusum_threshold") if show_cusum else 0.2

    st.divider()

    # Run analytics button
    run_analytics_btn = st.button("Run Analytics", key="sidebar_run_analytics")

# --- Feature Engineering Step (parallelized and cached) ---

if 'run_analytics_btn' in locals() and run_analytics_btn:
    st.info("Running feature engineering for selected symbols. Please wait...")
    progress_bar = st.progress(0, text="Starting...")

    start_time = time.time()  # Start timer

    if not selected_symbols:
        st.warning("No symbols selected. Please select at least one symbol.")
        st.stop()

    with st.spinner("Processing features..."):
        try:
            create_underlying_analytics(selected_symbols)
            progress_bar.progress(1.0, text=f"Processed {len(selected_symbols)}/{len(selected_symbols)} symbols")
        except Exception as exc:
            progress_bar.empty()
            st.error(f"Feature engineering failed: {exc}")
            st.stop()

    elapsed = time.time() - start_time  # End timer
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    st.success(f"Feature engineering completed for selected symbols! Time taken: {minutes} min {seconds} sec.")
    progress_bar.empty()
    _read_feature_df.clear()
    _rerun_app()


# Plot GARCH vs RSI

st.subheader(f"GARCH Vol Percentile vs RSI  {selected_date}")
df1 = plot_garch_vs_rsi(selected_symbols)

image_path = IMAGES_DIR / f"garch_vs_rsi_{selected_date}.png"
if image_path.exists():
    st.image(str(image_path), caption="GARCH vs RSI Scatter Plot", use_container_width=True)
else:
    st.warning(f"Image not found: {image_path}")

# Plot Historical Chart
with st.expander("Plot Historical Chart", expanded=True):
    # Use sidebar-selected chart controls if present, otherwise fall back
    hist_symbol = hist_symbol if 'hist_symbol' in locals() else selected_symbols[0] if selected_symbols else None
    num_days = num_days if 'num_days' in locals() else None
    show_cusum = show_cusum if 'show_cusum' in locals() else True
    cusum_threshold_mult = cusum_threshold_mult if 'cusum_threshold_mult' in locals() else 0.2

    if st.button("Show Historical Chart"):
        df_hist = _read_feature_df(hist_symbol)
        if df_hist.empty:
            st.warning(f"Feature file for {hist_symbol} is missing or unreadable.")
        elif df_hist["Date"].isna().all():
            st.warning(f"Feature file for {hist_symbol} is missing parsable 'Date' values.")
        else:
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

                st.pyplot(fig, use_container_width=True)
    else:
        st.warning(f"Feature file not found for {hist_symbol}")



# --- Summary table for all symbols (latest row for each) ---
with st.expander("Summary Table", expanded=True):
    summary_rows = []
    for symbol in selected_symbols:
        df = _read_feature_df(symbol)
        if df.empty:
            continue

        max_date = df["Date"].max()
        latest_rows = df.loc[df["Date"] == max_date]
        if latest_rows.empty:
            continue

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
            "Weekly RSI Percentile": latest.get("RSI_percentile_weekly", None),
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
        numeric_cols = [
            "Latest Price",
            "Daily Return",
            "GARCH Volatility",
            "GARCH Volatility Percentile",
            "Vol_Change",
            "Daily CPR",
            "RSI",
            "RSI Percentile",
            "Weekly RSI",
            "Weekly RSI Percentile",
        ]

        # Convert numeric columns to float for sorting
        for col in numeric_cols:
            if col in summary_all_df.columns:
                summary_all_df[col] = pd.to_numeric(summary_all_df[col], errors="coerce")

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
                summary_all_df[col] = summary_all_df[col].apply(lambda x: f"{x:.0f}" if pd.notnull(x) else "")

        # Reset index to start from 1
        summary_all_df.index = summary_all_df.index + 1

        # --- Highlight Daily Return: green for positive, red for negative, center align ---
        def highlight_daily_return(val):
            try:
                num = float(val.replace("%", ""))
                color = "green" if num > 0 else "red" if num < 0 else "black"
                return f"color: {color}; text-align: center;"
            except Exception:
                return "text-align: center;"

        styled_df = summary_all_df.style.map(highlight_daily_return, subset=["Daily Return"]).set_properties(
            **{"text-align": "center"}
        )

        st.subheader(f"Stock Analysis - {selected_date}")
        st.dataframe(styled_df, use_container_width=True)

        # --- Place the Refresh button and logic here ---
        if st.button("Refresh Summary Table", key="refresh_summary_table_bottom"):
            st.info("Refreshing summary table with latest engineered data...")
            _read_feature_df.clear()
            _rerun_app()

    else:
        st.warning("No feature files found for the selected symbols.")


# Upcoming Earnings, Dividends & Corporate Actions

# --- Correlation Matrix for Selected Stocks ---
st.header("Correlation Matrix for Selected Stocks")

close_prices = {}
for symbol in selected_symbols:
    df = _read_feature_df(symbol)
    if df.empty or "Close" not in df.columns:
        continue
    close_prices[symbol] = df.set_index("Date")["Close"]

if close_prices:
    close_df = pd.DataFrame(close_prices)
    returns_df = close_df.pct_change(fill_method=None)
    corr_matrix = returns_df.corr()
    st.subheader("Correlation Matrix (Daily Returns)")
    st.dataframe(
        corr_matrix.style.format("{:.2f}").background_gradient(cmap="coolwarm"),
        use_container_width=True,
    )
else:
    st.warning("Not enough data to compute correlation matrix for selected stocks.")


# --- Economic Calendar (filtered, combined) ---
st.header("Economic Calendar")
with st.expander("View calendar", expanded=True):
    # Timezone selector for calendar filtering (default to system timezone)
    # Short curated timezone list for compact UI
    tz_list = [
        "UTC",
        "US/Eastern",
        "US/Central",
        "US/Pacific",
        "Europe/London",
        "Europe/Paris",
        "Asia/Kolkata",
        "Asia/Shanghai",
        "Asia/Tokyo",
        "Australia/Sydney",
    ]

    try:
        default_tz = pendulum.local_timezone().name
    except Exception:
        default_tz = "UTC"

    # Ensure user's system timezone appears first if it's not already in the short list
    if default_tz not in tz_list:
        tz_list = [default_tz] + tz_list

    tz_index = tz_list.index(default_tz) if default_tz in tz_list else 0

    # Load persisted timezone from settings file if present
    persisted_tz = None
    try:
        if SETTINGS_FILE.exists():
            cfg = json.loads(SETTINGS_FILE.read_text())
            persisted_tz = cfg.get("calendar_tz")
    except Exception:
        persisted_tz = None

    if persisted_tz and persisted_tz not in tz_list:
        tz_list = [persisted_tz] + tz_list

    selected_tz = st.selectbox("Select timezone for calendar", options=tz_list, index=tz_list.index(persisted_tz) if persisted_tz in tz_list else tz_index, key="calendar_tz_select")

    # Persist selection for future sessions
    try:
        cfg = {}
        if SETTINGS_FILE.exists():
            cfg = json.loads(SETTINGS_FILE.read_text())
        cfg["calendar_tz"] = selected_tz
        SETTINGS_FILE.write_text(json.dumps(cfg))
    except Exception:
        pass
    # Allow user to fetch latest calendar on demand
    if st.button("Fetch Latest Calendar", key="fetch_calendar_btn"):
        with st.spinner("Fetching economic calendar and saving CSVs..."):
            try:
                fetch_and_save()
                filter_and_save()
                st.success("Calendar fetched and filtered — reloading view.")
                _rerun_app()
            except Exception as e:
                st.error(f"Failed to fetch calendar: {e}")

    calendar_df, calendar_err = load_filtered_calendar(save=True)
    if calendar_err:
        st.warning(calendar_err)
    elif calendar_df is None or calendar_df.empty:
        st.info("No calendar data available.")
    else:
        if "Date" in calendar_df.columns:
            calendar_df["Date"] = pd.to_datetime(calendar_df["Date"], errors="coerce")

        sort_cols = [c for c in ["Date", "Time"] if c in calendar_df.columns]
        if sort_cols:
            calendar_df = calendar_df.sort_values(sort_cols)

        # Exclude past releases prior to the selected date (keep NaT rows)
        try:
            # Use the user's selected timezone to compute 'today' and filter past releases
            tzobj = pytz.timezone(selected_tz)
            today_in_tz = pd.Timestamp.now(tz=tzobj).date()
            if "Date" in calendar_df.columns:
                # compare by date to avoid timezone-aware vs naive mismatches
                calendar_df = calendar_df[calendar_df["Date"].isna() | (calendar_df["Date"].dt.date >= today_in_tz)]
        except Exception:
            # if something goes wrong with date parsing or timezone, skip filtering
            pass

        display_cols = [
            col for col in ["Date", "Time", "Region", "Country", "Currency", "Event", "Importance", "Forecast", "Previous", "Actual"]
            if col in calendar_df.columns
        ]

        # Reset index so the displayed table has a clean 1..N index
        calendar_df = calendar_df.reset_index(drop=True)
        calendar_df.index = range(1, len(calendar_df) + 1)

        st.dataframe(calendar_df[display_cols], use_container_width=True)


# --- News Headlines ---
st.header("News Headlines")

@st.cache_data(ttl=1800)
def _load_news_headlines(symbols: list[str], selected_date: str):
    name_map = news_headlines._load_name_map()
    news_items = news_headlines._fetch_all_news_items(symbols)

    rows = []
    for sym in symbols:
        rows.append(
            news_headlines._fetch_symbol_data(
                sym,
                news_items,
                days_back=news_headlines.DEFAULT_DAYS_BACK,
                max_headlines=news_headlines.DEFAULT_MAX_HEADLINES,
                name_map=name_map,
            )
        )

    df = pd.DataFrame(rows)
    if "HeadlineTime" in df.columns:
        df["HeadlineTime"] = df["HeadlineTime"].astype(str)
    return df


with st.expander("View headlines", expanded=True):
    news_df = _load_news_headlines(selected_symbols, selected_date)
    if news_df is None or news_df.empty:
        st.info("No headlines available.")
    else:
        display_cols = [
            col for col in [
                "Symbol", "Headline", "HeadlineTime", "HeadlineSource", "HeadlineLink",
                "Event", "EventDate", "CorporateAction", "CorporateActionDate",
                "Headlines", "HeadlineSources", "HeadlineLinks",
            ]
            if col in news_df.columns
        ]
        st.dataframe(news_df[display_cols], use_container_width=True)