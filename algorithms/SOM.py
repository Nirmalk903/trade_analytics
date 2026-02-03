import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from minisom import MiniSom
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from data_download_vbt import get_symbols, get_dates_from_most_active_files
from tqdm import tqdm
import shap
import time
import seaborn as sns

# --- PARAMETERS ---
SOM_GRID = (4, 4)
SOM_ITER = 500  # Reduce iterations for speed, increase for accuracy
N_FEATURE_CLUSTERS = 5  # For feature selection
SAVE_DIR = "SOM_Image"

def load_features(symbols, data_dir="Engineered_data"):
    dfs = []
    cutoff_date = pd.Timestamp.today() - pd.DateOffset(years=5)
    print(f"Loading features for last 5 years for {len(symbols)} symbols...")
    for symbol in tqdm(symbols, desc="Loading features"):
        file_path = os.path.join(data_dir, f"{symbol}_1d_features.json")
        if os.path.exists(file_path):
            df = pd.read_json(file_path, orient='records', lines=True)
            df['symbol'] = symbol
            if 'Date' not in df.columns or df.empty:
                print(f"  Skipped {symbol}: No 'Date' column or empty DataFrame")
                continue
            if not np.issubdtype(df['Date'].dtype, np.datetime64):
                df['Date'] = pd.to_datetime(df['Date'])
            df = df[df['Date'] >= cutoff_date]
            dfs.append(df)
            print(f"  Loaded {symbol}: {len(df)} rows")
        else:
            print(f"  File not found for {symbol}")
    print("Feature loading complete.\n")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def select_uncorrelated_features(df, feature_cols, n_clusters=N_FEATURE_CLUSTERS):
    print("Selecting uncorrelated features using KMeans clustering on correlation matrix...")
    corr = df[feature_cols].corr().abs()
    dist = 1 - corr
    kmeans = KMeans(n_clusters=min(n_clusters, len(feature_cols)), random_state=42, n_init=10)
    kmeans.fit(dist)
    labels = kmeans.labels_
    selected_features = []
    for cluster in range(kmeans.n_clusters):
        cluster_features = [feature_cols[i] for i in range(len(feature_cols)) if labels[i] == cluster]
        if len(cluster_features) == 1:
            selected_features.append(cluster_features[0])
        else:
            avg_corr = corr.loc[cluster_features, cluster_features].mean().sort_values()
            selected_features.append(avg_corr.index[0])
    print(f"Selected features: {selected_features}\n")
    return selected_features

def select_features_shap(df, feature_cols, target_col='RSI', n_features=5):
    print("Selecting features using SHAP...")
    from sklearn.ensemble import RandomForestRegressor
    X = df[feature_cols].fillna(0)
    y = df[target_col].fillna(0)
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X, y)
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    shap_importance = np.abs(shap_values.values).mean(axis=0)
    feature_importance = pd.Series(shap_importance, index=feature_cols)
    top_features = feature_importance.sort_values(ascending=False).head(n_features).index.tolist()
    print(f"Top {n_features} features by SHAP: {top_features}\n")
    return top_features

def run_som_analysis(symbols, plot=True, save_dir=SAVE_DIR, use_shap=False):
    print("Running SOM analysis...")
    df = load_features(symbols)
    feature_cols = [col for col in df.columns if col not in ['Date', 'symbol','Close','High','Low','Volume'] and np.issubdtype(df[col].dtype, np.number)]
    print(f"Feature columns considered: {feature_cols}")
    df = df.dropna(subset=feature_cols, how='all').reset_index(drop=True)
    print(f"Data shape after dropping rows with all NaNs in features: {df.shape}")

    # Feature selection
    if len(feature_cols) > 1:
        if use_shap and 'RSI' in feature_cols:
            selected_features = select_features_shap(df, feature_cols, target_col='RSI', n_features=min(5, len(feature_cols)))
        else:
            selected_features = select_uncorrelated_features(df, feature_cols, n_clusters=min(N_FEATURE_CLUSTERS, len(feature_cols)))
    else:
        selected_features = feature_cols
    print(f"Selected features for SOM: {selected_features}")
    X = df[selected_features].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Data scaled for SOM.")

    # --- SOM Training ---
    som = MiniSom(SOM_GRID[0], SOM_GRID[1], X_scaled.shape[1], sigma=1.0, learning_rate=0.5, random_seed=42)
    som.random_weights_init(X_scaled)
    print(f"Training SOM ({SOM_GRID[0]}x{SOM_GRID[1]}, {SOM_ITER} iterations)...")
    som.train_random(X_scaled, SOM_ITER)
    print("SOM training complete.")

    df['SOM_cluster'] = [som.winner(x) for x in X_scaled]
    print("Assigned SOM clusters to each row.")

    # Ensure save directory exists
    save_path = os.path.join(os.path.dirname(__file__), save_dir)
    os.makedirs(save_path, exist_ok=True)

    # Plot SOM chart (showing all points, colored by symbol)
    if plot:
        print("Saving SOM clusters plot...")
        plt.figure(figsize=(7, 7))
        plt.title(f"SOM Clusters ({SOM_GRID[0]}x{SOM_GRID[1]})")
        symbols_unique = df['symbol'].unique()
        colors = plt.colormaps['tab20']
        for i, symbol in enumerate(symbols_unique):
            idx = df['symbol'] == symbol
            color = colors(i % colors.N)
            for x in X_scaled[idx]:
                w = som.winner(x)
                plt.plot(w[0]+0.1, w[1]+0.1, 'o', markersize=6, color=color, alpha=0.5)
            last_idx = df[df['symbol'] == symbol].index[-1]
            w_last = som.winner(X_scaled[last_idx])
            plt.text(w_last[0]+0.1, w_last[1]+0.1, symbol, fontsize=10, color=color)
        plt.xlim(-0.5, SOM_GRID[0]-0.5)
        plt.ylim(-0.5, SOM_GRID[1]-0.5)
        plt.grid(True)
        plt.xlabel("SOM X")
        plt.ylabel("SOM Y")
        som_img_path = os.path.join(save_path, "som_clusters.png")
        plt.savefig(som_img_path, bbox_inches='tight')
        plt.close()
        print(f"SOM plot saved to {som_img_path}.")

    print("Calculating cluster centers (feature means per cluster)...")
    cluster_centers = {}
    for cluster in set(df['SOM_cluster']):
        idx = df['SOM_cluster'] == cluster
        cluster_centers[cluster] = df.loc[idx, selected_features].mean()
    print("Cluster centers calculated.\n")
    return df, cluster_centers, selected_features

if __name__ == "__main__":
    start_time = time.time()
    symbols = get_symbols(get_dates_from_most_active_files()[-1], top_n=17)[0]
    df, cluster_centers, selected_features = run_som_analysis(symbols, plot=True, save_dir=SAVE_DIR, use_shap=False)

    # Print all cluster centers
    print("\nAll SOM cluster centers:")
    for cluster, center in cluster_centers.items():
        print(f"\nCluster {cluster}:")
        print(center)

    # Example: Find clusters with low RSI (potentially oversold)
    print("\nChecking for clusters with low RSI (potentially oversold):")
    for cluster, center in cluster_centers.items():
        if center.get('RSI', 50) < 35:
            print(f"\nCluster {cluster} may be oversold:")
            print(center)

    # Convert cluster_centers to DataFrame for easier analysis
    centers_df = pd.DataFrame(cluster_centers).T
    print("\nCluster centers DataFrame:")
    print(centers_df)

    # --- Table of SOM clusters and associated stocks (all data) ---
    print("\nTable of SOM clusters and associated stocks (all data):")
    cluster_stock_table = df.groupby('SOM_cluster')['symbol'].unique().reset_index()
    cluster_stock_table['symbol'] = cluster_stock_table['symbol'].apply(lambda x: ', '.join(sorted(set(x))))
    print(cluster_stock_table.to_string(index=False))
    table_path = os.path.join(os.path.dirname(__file__), SAVE_DIR, "cluster_stock_table.csv")
    cluster_stock_table.to_csv(table_path, index=False)
    print(f"\nCluster-stock association table saved to {table_path}.")

    # --- Table of SOM clusters and associated stocks (latest data point) ---
    latest_df = df.sort_values('Date').groupby('symbol').tail(1)
    cluster_stock_map = latest_df.groupby('SOM_cluster')['symbol'].unique().reset_index()
    cluster_stock_map['symbol'] = cluster_stock_map['symbol'].apply(lambda x: ', '.join(sorted(set(x))))
    print("\nStocks for each SOM cluster (cell) based on latest data point:")
    print(cluster_stock_map.to_string(index=False))
    save_path = os.path.join(os.path.dirname(__file__), SAVE_DIR, "latest_cluster_stock_table.csv")
    cluster_stock_map.to_csv(save_path, index=False)
    print(f"Cluster-stock association table (latest data) saved to {save_path}.")

    # --- Plot heatmap of cluster centers ---
    print("Saving heatmap of cluster centers...")
    plt.figure(figsize=(10, 6))
    sns.heatmap(centers_df, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("SOM Cluster Centers (Feature Means)")
    heatmap_img_path = os.path.join(os.path.dirname(__file__), SAVE_DIR, "som_cluster_centers_heatmap.png")
    plt.savefig(heatmap_img_path, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved to {heatmap_img_path}.")

    end_time = time.time()
    print(f"\nSOM analysis pipeline completed in {int(end_time - start_time)} seconds.")