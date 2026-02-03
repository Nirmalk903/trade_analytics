import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.mixture import GaussianMixture
import pandas as pd
import numpy as np
import os
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

def apply_gaussian_mixture_model(df, n_components=2, target_col='Returns'):
    """
    Applies a Gaussian Mixture Model to the target column for each symbol.
    Returns a DataFrame with GMM states and state probabilities.
    """
    results = []
    for symbol in df['symbol'].unique():
        sdf = df[df['symbol'] == symbol].sort_values('Date').copy()
        y = sdf[target_col].dropna().values.reshape(-1, 1)
        if len(y) < 100:
            continue  # Not enough data for stable estimation
        try:
            gmm = GaussianMixture(n_components=n_components, covariance_type="full", n_init=10, random_state=42)
            gmm.fit(y)
            gmm_states = gmm.predict(y)
            sdf = sdf.iloc[-len(y):].copy()
            sdf['gmm_state'] = gmm_states
            # Posterior probabilities for each state
            posteriors = gmm.predict_proba(y)
            for i in range(n_components):
                sdf[f'gmm_state_{i}_prob'] = posteriors[:, i]
            results.append(sdf)
            print(f"{symbol}: GMM fitted. Score={gmm.score(y):.2f}")
        except Exception as e:
            print(f"{symbol}: GMM failed - {e}")
    return pd.concat(results) if results else pd.DataFrame()



if __name__ == "__main__":
    # Example usage
    from data_download_vbt import get_symbols, get_dates_from_most_active_files

    symbols = get_symbols(get_dates_from_most_active_files()[-1], top_n=10)[0]
    df = load_engineered_data(symbols)
    if not df.empty and 'Returns' in df.columns:
        df_gmm = apply_gaussian_mixture_model(df, n_components=2, target_col='Returns')
        if not df_gmm.empty:
            # Save results
            out_path = os.path.join(os.path.dirname(__file__), "gaussian_mixture_model_results.csv")
            df_gmm.to_csv(out_path, index=False)
            print(f"\nGMM results saved to {out_path}")
        else:
            print("No GMM results generated.")
    else:
        print("No data loaded or 'Returns' column missing.")

    # Read the file from the algorithms folder
    out_path = os.path.join(os.path.dirname(__file__), "gaussian_mixture_model_results.csv")
    if os.path.exists(out_path):
        df = pd.read_csv(out_path)
        print("\nGMM State Statistics by Stock:")
        for symbol in df['symbol'].unique():
            print(f"\nSymbol: {symbol}")
            print(df[df['symbol'] == symbol].groupby('gmm_state')['Returns'].agg(['mean', 'std', 'count']))
    else:
        print(f"{out_path} does not exist.")