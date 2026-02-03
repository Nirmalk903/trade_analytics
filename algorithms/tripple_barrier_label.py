import os
import sys

from schedule import jobs
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from data_download_vbt import getdata_vbt, get_underlying_data_vbt, get_symbols, get_dates_from_most_active_files
import multiprocessing as mp
from algorithms.cusum_filter import getTEvents
from volatility_modeling import getDailyVol


def mpPandasObj(func, pdObj, numThreads=24, **kargs):
    """
    Parallelize jobs, return a DataFrame or Series.
    pdObj[0]: name
    pdObj[1]: list of index values
    """
    # Split index values
    parts = linParts(len(pdObj[1]), numThreads)
    index_values = pdObj[1]
    for part in parts:
        job = {pdObj[0]: index_values[part.start:part.stop]}
        job.update(kargs)
        jobs.append(job)
    # Run jobs in parallel
    pool = mp.Pool(processes=numThreads)
    out = pool.map(func_mp, [(func, job) for job in jobs])
    pool.close()
    pool.join()
    # Combine results
    if isinstance(out[0], pd.DataFrame):
        df = pd.concat(out, axis=0)
    elif isinstance(out[0], pd.Series):
        df = pd.concat(out, axis=0)
    else:
        df = out
    return df

def func_mp(args):
    func, job = args
    return func(**job)

def linParts(numAtoms, numThreads):
    # Partition of atoms with a single loop
    parts = np.linspace(0, numAtoms, min(numThreads, numAtoms) + 1)
    parts = np.ceil(parts).astype(int)
    return [range(parts[i], parts[i+1]) for i in range(parts.size - 1)]


def applyPtSlOnT1(close, events, ptSl, molecule):
    events_ = events.loc[molecule]
    out = events_[['tl']].copy(deep=True)  # <-- changed from 't1' to 'tl'
    if ptSl[0] > 0:
        pt = ptSl[0] * events_['trgt']
    else:
        pt = pd.Series(index=events_.index)
    if ptSl[1] > 0:
        sl = -ptSl[1] * events_['trgt']
    else:
        sl = pd.Series(index=events_.index)
    for loc, tl in events_['tl'].fillna(close.index[-1]).items():  # <-- changed from 't1' to 'tl'
        df0 = close[loc:tl]
        df0 = (df0 / close[loc] - 1) * events_.at[loc, 'side']
        out.loc[loc, 'sl'] = df0[df0 < sl[loc]].index.min()
        out.loc[loc, 'pt'] = df0[df0 > pt[loc]].index.min()
    return out


# Getting the time for first touch
def getEvents(close, tEvents, ptSl, trgt, minRet, numThreads, tl=False, side=None):
    trgt = trgt.reindex(tEvents)
    trgt = trgt[trgt > minRet]
    if tl is False:
        tl=pd.Series(pd.NaT, index=tEvents)
    if side is None:
        side_, ptSl_ = pd.Series(1., index=trgt.index), [ptSl[0], ptSl[0]]
    else:
        side_, ptSl_ = side.loc[trgt.index], ptSl[:2]

    events = pd.concat({'tl': tl, 'trgt': trgt, 'side': side_}, axis=1).dropna(subset=['trgt'])
    df0= mpPandasObj(func = applyPtSlOnT1, pdObj = ('molecule', events.index), numThreads=numThreads,close=close,events=events,ptSl=ptSl_)

    # Remove duplicate indices from both DataFrames
    df0 = df0[~df0.index.duplicated(keep='first')]
    events = events[~events.index.duplicated(keep='first')]

    # Align indices before assignment
    events['tl'] = df0.dropna(how='all').min(axis=1).reindex(events.index)
    
    if side is None:
        events = events.drop('side', axis=1)
    return events

# t1 = close.index.searchsorted(tEvents+pd.Timedelta(days=numDays))
# t1 = t1[t1 < close.shape[0]]
# t1 = pd.Series(close.index[t1], index=tEvents[:t1.size])

# Labeling for side and size

def getBins(events, close):
    """
    Compute triple barrier method labels (bins) for events.
    Returns a DataFrame with 'ret' and 'bin' columns.
    """
    # Drop events with missing t1
    events_ = events.dropna(subset=['tl']).copy()
    # Ensure tl values are in the close index (use bfill for missing)
    px = close.reindex(events_.index.union(events_['tl']).drop_duplicates(), method='bfill')
    # Calculate returns
    out = pd.DataFrame(index=events_.index)
    out['ret'] = px.loc[events_['tl']].values / px.loc[events_.index].values - 1

    # If 'side' exists, apply it
    if 'side' in events_.columns:
        out['ret'] *= events_['side']
        out['bin'] = np.sign(out['ret'])
    else:
        out['bin'] = np.sign(out['ret'])

    # If 'size' exists, override bin for non-positive returns
    if 'size' in events_.columns:
        out.loc[out['ret'] <= 0, 'bin'] = 1

    # Ensure 'bin' is integer type (no -0.0)
    threshold = 1e-6
    out['bin'] = np.where(out['ret'] > threshold, 1, np.where(out['ret'] < -threshold, -1, 0))
    return out

# Dropping Under-Populated Labels

def dropLabels(events, minPtc=0.05):
    while True:
        df0 = events['bin'].value_counts(normalize=True)
        if df0.min() > minPtc or df0.size < 3:
            break
        events = events[events['bin'] != df0.idxmin()]
    return events

def get_vertical_barrier(close, tEvents, numDays=5):
    """
    For each tEvent, find the timestamp of the vertical barrier (numDays ahead).
    If the barrier is beyond the available data, use the last available date.
    """
    t1 = {}
    for t in tEvents:
        idx = close.index.searchsorted(t + pd.Timedelta(days=numDays))
        if idx < close.shape[0]:
            t1[t] = close.index[idx]
        else:
            t1[t] = close.index[-1]
    return pd.Series(t1)


def apply_triple_barrier_to_all_symbols(engineered_dir, results_dir, numDays=5):
    symbols = get_symbols(get_dates_from_most_active_files()[-1], top_n=17)[0]
    os.makedirs(results_dir, exist_ok=True)
    for symbol in symbols:
        file_path = os.path.join(engineered_dir, f"{symbol}_1d_features.json")
        if not os.path.exists(file_path):
            continue
        df = pd.read_json(file_path, orient='records', lines=True)
        if "Date" not in df.columns or "Close" not in df.columns:
            continue
        df = df.sort_values("Date")
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
        # Get tEvents using CUSUM
        h = df["Close"].std() * 0.05
        tEvents = getTEvents(df["Close"], h)
        # Get daily volatility
        daily_vol = getDailyVol(df["Close"])
        # Compute vertical barrier (tl)
        tl = get_vertical_barrier(df["Close"], tEvents, numDays=numDays)
        # Apply triple barrier labeling
        events = getEvents(close=df["Close"], tEvents=tEvents, ptSl=[1, 1], trgt=daily_vol, minRet=0.005, numThreads=1, tl=tl)
        # Get bins (labels)
        if events is not None and not events.empty:
            bins = getBins(events, df["Close"])
            # Drop labels with NaN
            bins = dropLabels(bins)
            # Save bins DataFrame
            out_path = os.path.join(results_dir, f"{symbol}_triple_barrier_bins.csv")
            bins.to_csv(out_path)
            print(f"Triple barrier bins for {symbol} saved to {out_path}")

# Example usage:
if __name__ == "__main__":
    engineered_dir = "./Engineered_data"
    results_dir = "./results/tripple_barrier"
    apply_triple_barrier_to_all_symbols(engineered_dir, results_dir, numDays=10)

