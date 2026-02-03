# Hamilton's Markov Regime Switching Model (MRS)
# Regime Switching GARCH Model (RGARCH)
# Hidden Markov Model (HMM)

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from tqdm import tqdm

def load_engineered_data(symbols, data_dir="Engineered_data"):
    dfs = []
    for symbol in tqdm(symbols, desc="Loading features"):
        file_path = os.path.join(data_dir, f"{symbol}_1d_features.json")
        if os.path.exists(file_path):
            df = pd.read_json(file_path, orient='records', lines=True)
            df['symbol'] = symbol
            if "Date" not in df.columns or df.empty:
                continue
            if not np.issubdtype(df['Date'].dtype, np.datetime64):
                df['Date'] = pd.to_datetime(df['Date'])
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def apply_hamilton_regime_switching(df, n_regimes=2, target_col='Returns'):
    """
    Applies Hamilton's Markov Regime Switching Model to the target column.
    Returns a DataFrame with regime probabilities, predicted regimes, and model state.
    """
    results = []
    for symbol in df['symbol'].unique():
        sdf = df[df['symbol'] == symbol].sort_values('Date').copy()
        y = sdf[['Date', target_col]].dropna()
        y = y.set_index('Date')
        y.index = pd.to_datetime(y.index)
        y = y.sort_index()
        y = y[~y.index.duplicated(keep='first')]
        y = y[target_col]
        if len(y) < 100:
            continue  # Not enough data for stable estimation
        try:
            model = MarkovRegression(y, k_regimes=n_regimes, trend='c', switching_variance=True)
            res = model.fit(disp=False)
            # --- Fix: set sdf index to Date before .loc ---
            sdf = sdf.set_index('Date')
            sdf = sdf.loc[y.index]
            sdf = sdf.reset_index()
            for i in range(n_regimes):
                sdf[f'regime_{i}_prob'] = res.smoothed_marginal_probabilities[i].values
            sdf['predicted_regime'] = res.smoothed_marginal_probabilities.idxmax(axis=1)
            # Add the most likely regime (state) from the model for each row
            sdf['hamilton_state'] = res.smoothed_marginal_probabilities.values.argmax(axis=1)
            results.append(sdf)
            print(f"{symbol}: Regime switching model fitted. LogLik={res.llf:.2f}")
        except Exception as e:
            print(f"{symbol}: Model failed - {e}")
    return pd.concat(results) if results else pd.DataFrame()

if __name__ == "__main__":
    # Example usage
    from data_download_vbt import get_symbols, get_dates_from_most_active_files

    symbols = get_symbols(get_dates_from_most_active_files()[-1], top_n=10)[0]
    df = load_engineered_data(symbols)
    if not df.empty and 'Returns' in df.columns:
        df_rs = apply_hamilton_regime_switching(df, n_regimes=2, target_col='Returns')
        if not df_rs.empty:
            # Save results
            out_path = os.path.join(os.path.dirname(__file__), "hamilton_markov_results.csv")
            df_rs.to_csv(out_path, index=False)
            print(f"\nRegime switching results saved to {out_path}")
        else:
            print("No regime switching results generated.")
    else:
        print("No data loaded or 'Returns' column missing.")

    # Read the file from the algorithms folder
    out_path = os.path.join(os.path.dirname(__file__), "regime_switching_results.csv")
    if os.path.exists(out_path):
        df = pd.read_csv(out_path)
        print(df.groupby('predicted_regime')['Returns'].agg(['mean', 'std', 'count']))
    else:
        print(f"{out_path} does not exist.")

