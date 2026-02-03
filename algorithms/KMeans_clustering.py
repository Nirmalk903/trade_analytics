import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
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

def run_kmeans_clustering(df, n_clusters=4, feature_exclude=None):
    if feature_exclude is None:
        feature_exclude = ['Date', 'symbol', 'Close', 'High', 'Low', 'Volume']
    feature_cols = [col for col in df.columns if col not in feature_exclude and np.issubdtype(df[col], np.number)]
    df = df.dropna(subset=feature_cols, how='all').reset_index(drop=True)
    X = df[feature_cols].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20, max_iter=1000)
    labels = kmeans.fit_predict(X_scaled)
    df['kmeans_cluster'] = labels
    print(f"KMeans clustering complete. Inertia: {kmeans.inertia_:.2f}")
    return df, kmeans, feature_cols

if __name__ == "__main__":
    # Example usage
    from data_download_vbt import get_symbols, get_dates_from_most_active_files

    symbols = get_symbols(get_dates_from_most_active_files()[-1], top_n=17)[0]
    df = load_engineered_data(symbols)
    if not df.empty:
        df_clustered, kmeans, feature_cols = run_kmeans_clustering(df, n_clusters=4)
        # Save results
        out_path = os.path.join(os.path.dirname(__file__), "kmeans_clustering_results.csv")
        df_clustered.to_csv(out_path, index=False)
        print(f"KMeans clustering results saved to {out_path}")

        # Print cluster summary
        print("\nCluster summary (mean of features):")
        print(df_clustered.groupby('kmeans_cluster')[feature_cols].mean())

        # Show stock distribution per cluster
        print("\nStocks per cluster (latest data point):")
        latest_df = df_clustered.sort_values('Date').groupby('symbol').tail(1)
        stock_table = latest_df.groupby('kmeans_cluster')['symbol'].unique().reset_index()
        stock_table['symbol'] = stock_table['symbol'].apply(lambda x: ', '.join(sorted(set(x))))
        print(stock_table.to_string(index=False))
    else:
        print("No data loaded.")