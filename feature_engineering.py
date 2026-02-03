import pandas as pd
import numpy as np
# import pandas_ta as ta
import os
import vectorbt as vbt
from volatility_modeling import garch_vol, getDailyVol
from scipy.stats import percentileofscore
from data_download_vbt import getdata_vbt, get_underlying_data_vbt, get_symbols,  get_dates_from_most_active_files
from algorithms.hamilton_markov_model import apply_hamilton_regime_switching
from algorithms.hidden_markov_model import apply_hidden_markov_model
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from hmmlearn.hmm import GaussianHMM
from sklearn.mixture import GaussianMixture
from algorithms.cusum_filter import getTEvents

class FeatureEngineering:
    
    def __init__(self,X):
        
        self.X = X
        self.High = X.High
        self.Low = X.Low
        self.Close = X.Close
        self.Open = X.Open
        self.Volume = X.Volume
        
    def __call__(self, X):
        self.callme(X)
        
    def dCPR(self,X): #creates daily central pivot range
        
        X['dCPR'] = (np.array(X[['High','Low','Close']].shift(1))).mean(axis=1)

        # Calculting Daily PCR to Low, High and Close

        # X['Low2dCPR'] = X['Low']/ X['dCPR']
        # X['High2dCPR'] = X['High']/ X['dCPR']
        X['Close2dCPR'] = X['Close']/ X['dCPR']
        
        return X
    
    def wCPR(self, X):  # Creates weekly pivot range
    # Ensure the index is a DatetimeIndex
        if not isinstance(X.index, pd.DatetimeIndex):
            X.set_index('Date', inplace=True)
            # X = X.asfreq("B")

        self.logic = {'High': 'max', 'Low': 'min', 'Close': 'last'}
        df1 = X.sort_index()
        df1 = df1.resample('W').apply(self.logic)
        df1.index = df1.index - pd.tseries.frequencies.to_offset("6D")
        df1.rename(columns={'High': 'wHigh', 'Low': 'wLow', 'Close': 'wClose'}, inplace=True)

        df1['wCPR'] = (np.array(df1[['wHigh', 'wLow', 'wClose']].shift(1))).mean(axis=1)

        for i in range(len(X.index)):
            for j in range(len(df1.index)):
                if (X.index[i] >= df1.index[j]) & (X.index[i] <= df1.index[j] + pd.offsets.Day(6)):
                    X.loc[X.index[i], 'wCPR'] = df1.loc[df1.index[j], 'wCPR']

        # X['Low2wCPR'] = X['Low'] / X['wCPR']
        # X['High2wCPR'] = X['High'] / X['wCPR']
        X['Close2wCPR'] = X['Close'] / X['wCPR']

        return X
    
    def cpr_vol(self,X,lag): #creates volatility columns for daily and weekly CPRs
        
        self.lag = lag
        vol_lag = range(5,lag,5)

        col = X.filter(regex='2d|2wCPR')

        for i in vol_lag:
            for j in col:                                                                                                                                                               
                X[f"{j}_{i}dVol"] = np.array(X[j].rolling(i).std()*np.sqrt(251))
        return X
    
    def price_momentum(self,X,lag):
        self.lag=lag
        moments = {f'Price_Moment_{i}': X.Close - X.Close.shift(i) for i in range(1, lag)}
        X = X.assign(**moments)
            
        return X
    
    def sma(self, X, lag):
        self.lag=lag
        for i in range(10,lag,5):
            
            X[f'SMA_{i}'] = X.Close.rolling(i).mean()
        
        return X
    
    def price_range(self,X):
        
        # Calculating intra-day price range

        X['O-C']= X['Open']-X['Close']
        X['H-L'] = X['High'] - X['Low']
        
        return X
    
    def dpo_centered(self, X, period = 20):
        
        if 'Close' not in X.columns:
            raise ValueError("The DataFrame must contain a 'Close' column.")

        # Calculate the Simple Moving Average (SMA)
        sma = X['Close'].rolling(window=period).mean()

        # Shift the SMA by (period / 2) + 1 periods
        sma_shifted = sma.shift(int((period / 2) + 1))

        # Calculate the DPO
        X[f'DPO_{period}'] = X['Close'] - sma_shifted
            
        return X
    
    # def cfo(self, X, period = 20):
    #     self.period = period
    #     X['CFO'] = ta.cfo(X['Close'], length=self.period, fillna=True)
    #     X['CFO'] = X['CFO'].replace([np.inf, -np.inf], np.nan)
    #     X['CFO'] = X['CFO'].fillna(method='bfill')
    #     return X
    
    def rolling_vol(self, X, lag):
        self.lag=lag
        for i in range(5,lag,5):
            
            X[f'Rolling_vol_{i}'] = np.log(X.Close).diff().rolling(i).std()*np.sqrt(251)*100
        
        return X
    
    def returns(self,X):
        X['Returns'] = X['Close'].pct_change(fill_method=None)
        
        return X
    
    def rsi(self, X, window=14, percentile_window=14):
        import vectorbt as vbt
        # Ensure 'Date' column exists
        if 'Date' not in X.columns:
            if X.index.name == 'Date':
                X = X.reset_index()
            else:
                raise KeyError("'Date' column is required in the DataFrame for resampling.")
        # Daily RSI
        rsi = vbt.RSI.run(X['Close'], window=window, short_name='RSI')
        X['RSI'] = rsi.rsi

        # Daily RSI percentile (long window, as before)
        rsi_percentiles = []
        for i in range(len(X)):
            if i < percentile_window:
                rsi_percentiles.append(np.nan)
            else:
                window_rsi = X['RSI'].iloc[i - percentile_window:i]
                val = X['RSI'].iloc[i]
                pct = percentileofscore(window_rsi, val)
                rsi_percentiles.append(pct)
        X['RSI_percentile'] = rsi_percentiles

        # --- Weekly RSI and percentile ---
        # Resample to weekly and calculate RSI
        X_weekly = X.set_index('Date').resample('W').last()
        rsi_weekly = vbt.RSI.run(X_weekly['Close'], window=window, short_name='RSI')
        X_weekly['RSI_weekly'] = rsi_weekly.rsi

        # Calculate weekly RSI percentile (rolling, on weekly data)
        rsi_percentiles_weekly = []
        for i in range(len(X_weekly)):
            if i < 52:  # 1 year of weekly data
                rsi_percentiles_weekly.append(np.nan)
            else:
                window_rsi = X_weekly['RSI_weekly'].iloc[i - 52:i]
                val = X_weekly['RSI_weekly'].iloc[i]
                pct = percentileofscore(window_rsi, val)
                rsi_percentiles_weekly.append(pct)
        X_weekly['RSI_percentile_weekly'] = rsi_percentiles_weekly

        # Map weekly RSI and percentile back to daily data
        X['RSI_weekly'] = X_weekly['RSI_weekly'].reindex(X.set_index('Date').index, method='ffill').values
        X['RSI_percentile_weekly'] = X_weekly['RSI_percentile_weekly'].reindex(X.set_index('Date').index, method='ffill').values

        # --- Monthly RSI and percentile ---
        X_monthly = X.set_index('Date').resample('ME').last()
        rsi_monthly = vbt.RSI.run(X_monthly['Close'], window=window, short_name='RSI')
        X_monthly['RSI_monthly'] = rsi_monthly.rsi

        rsi_percentiles_monthly = []
        for i in range(len(X_monthly)):
            if i < 12:  # 1 year of monthly data
                rsi_percentiles_monthly.append(np.nan)
            else:
                window_rsi = X_monthly['RSI_monthly'].iloc[i - 12:i]
                val = X_monthly['RSI_monthly'].iloc[i]
                pct = percentileofscore(window_rsi, val)
                rsi_percentiles_monthly.append(pct)
        X_monthly['RSI_percentile_monthly'] = rsi_percentiles_monthly

        # Map monthly RSI and percentile back to daily data
        X['RSI_monthly'] = X_monthly['RSI_monthly'].reindex(X.set_index('Date').index, method='ffill').values
        X['RSI_percentile_monthly'] = X_monthly['RSI_percentile_monthly'].reindex(X.set_index('Date').index, method='ffill').values

        return X
    
    def bollinger_bands(self,X, period=20, std=2):
        # Calculate the rolling mean and standard deviation
        rolling_mean = X['Close'].rolling(window=period).mean()
        rolling_std = X['Close'].rolling(window=period).std()

        # Calculate the upper and lower Bollinger Bands
        X['Bollinger_High'] = rolling_mean + (rolling_std * std)
        X['Bollinger_Low'] = rolling_mean - (rolling_std * std)

        return X
    
    def keltner_channel(self,X, period=20,atr_multiplier=1.5):
        # Calculate the rolling mean and average true range (ATR)
        X['Previous_Close'] = X['Close'].shift(1)
        X['True_Range'] = np.maximum(X['High'] - X['Low'], 
                                      np.maximum(abs(X['High'] - X['Previous_Close']), 
                                                 abs(X['Low'] - X['Previous_Close'])))
        X['ATR'] = X['True_Range'].rolling(window=period).mean()
        X.drop(columns=['Previous_Close', 'True_Range'], inplace=True)
        rolling_mean = X['Close'].rolling(window=period).mean()
        # Calculate the upper and lower Keltner Channels
        X['Keltner_High'] = rolling_mean + (X['ATR'] * atr_multiplier)
        X['Keltner_Low'] = rolling_mean - (X['ATR'] * atr_multiplier)

        return X
    
    def identify_volatility_squeezes(self, X, period=20):
        # Calculate the Bollinger Bands
        rolling_mean = X['Close'].rolling(window=period).mean()
        rolling_std = X['Close'].rolling(window=period).std()
        X['Bollinger_High'] = rolling_mean + (rolling_std * 2)
        X['Bollinger_Low'] = rolling_mean - (rolling_std * 2)

        # Calculate the Keltner Channels
        X['Previous_Close'] = X['Close'].shift(1)
        X['True_Range'] = np.maximum(X['High'] - X['Low'], 
                                      np.maximum(abs(X['High'] - X['Previous_Close']), 
                                                 abs(X['Low'] - X['Previous_Close'])))
        X['ATR'] = X['True_Range'].rolling(window=period).mean()
        rolling_mean_keltner = X['Close'].rolling(window=period).mean()
        X.drop(columns=['Previous_Close', 'True_Range'], inplace=True)
        X['Keltner_High'] = rolling_mean_keltner + (X['ATR'] * 1.5)
        X['Keltner_Low'] = rolling_mean_keltner - (X['ATR'] * 1.5)

        # Identify volatility squeezes
        X['Volatility_Squeeze'] = ((X['Bollinger_High'] - X['Bollinger_Low']) < 
                                    (X['Keltner_High'] - X['Keltner_Low'])).astype(int)
        
        return X
    
    def add_daily_vol(self, X):
        if 'Date' in X.columns:
            X = X.sort_values('Date')
            X['Date'] = pd.to_datetime(X['Date'])
            X = X.set_index('Date')
            X['daily_vol'] = getDailyVol(X['Close'], span0=100)
            X = X.reset_index()
        return X


    def callme(self, X):
        X = self.returns(X)
        X = self.dCPR(X)
        X = self.wCPR(X)
        # X = self.cpr_vol(X,lag=20)
        X = self.price_momentum(X,lag=20)
        X = self.sma(X,lag=20)
        X = self.price_range(X)
        X = self.dpo_centered(X)
        # X = self.dpo_non_centered(X,lag=20)
        # X = self.cfo(X,lag=20)
        X = self.rolling_vol(X,lag=25)
        X = self.rsi(X)
        X = self.bollinger_bands(X, period=20, std=2)
        X = self.keltner_channel(X, period=20, atr_multiplier=1.5)
        X = self.identify_volatility_squeezes(X, period=20)
        X = self.add_daily_vol(X)

        print('Feature Engineering Completed')
        return X
    
    


def add_features(symbols, interval='1d'):
    # --- Load NIFTY and BANKNIFTY close prices for correlation ---
    index_symbols = ['NIFTY', 'BANKNIFTY']
    index_closes = {}
    for idx_symbol in index_symbols:
        idx_path = f'./Underlying_data_vbt/{idx_symbol}_{interval}.csv'
        idx_path = os.path.join(idx_path)
        if os.path.exists(idx_path):
            idx_df = pd.read_csv(idx_path, parse_dates=True)
            idx_df['Date'] = pd.to_datetime(idx_df['Date'], format='%Y-%m-%d', errors='coerce')
            idx_df = idx_df.sort_values('Date')
            index_closes[idx_symbol] = idx_df.set_index('Date')['Close']
        else:
            print(f"Index data for {idx_symbol} not found. Rolling correlation features will be NaN.")

    for symbol in symbols:
        print(f'Loading data for {symbol}')
        data_path = f'./Underlying_data_vbt/{symbol}_{interval}.csv'
        data_path = os.path.join(data_path)
        if not os.path.exists(data_path):
            print(f"Data file for {symbol} not found. Please download the data first.")
            continue
        else:
            data = pd.read_csv(data_path, parse_dates=True)
            data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d', errors='coerce')
            data = data.sort_values('Date')

        # Apply feature engineering using callme
        feat_eng = FeatureEngineering(data)
        data = feat_eng.callme(data)

        # Add GARCH volatility
        data['garch_vol'] = garch_vol(symbol).values
        data['garch_vol'] = data['garch_vol'].bfill()
        data['garch_vol_pct'] = data['garch_vol'].diff()
        data['garch_vol_percentile'] = [np.round(percentileofscore(data['garch_vol'], i),0) for i in data['garch_vol'] ]
        data['garch_vol_percentile'] = data['garch_vol_percentile'].bfill()

        # --- Add 21-day rolling correlation with NIFTY and BANKNIFTY ---
        data = data.set_index('Date')
        # data = data.asfreq('B')
        if 'Close' in data.columns:
            for idx_symbol in index_symbols:
                if idx_symbol in index_closes:
                    # Align dates and calculate rolling correlation
                    combined = pd.DataFrame({
                        symbol: data['Close'],
                        idx_symbol: index_closes[idx_symbol]
                    }).dropna()
                    rolling_corr = combined[symbol].rolling(21).corr(combined[idx_symbol])
                    # Reindex to all dates in data
                    data[f'corr21_{idx_symbol}'] = rolling_corr.reindex(data.index)
                else:
                    data[f'corr21_{idx_symbol}'] = np.nan
        data = data.reset_index()

        # --- Add Hamilton Markov Regime Switching features internally ---
        try:
            returns = data['Returns'].dropna()
            if len(returns) >= 100:
                y = returns.copy()
                y.index = data.loc[returns.index, 'Date']
                y = y[~y.index.duplicated(keep='first')]
                model = MarkovRegression(y, k_regimes=2, trend='c', switching_variance=True)
                res = model.fit(disp=False)
                markov_idx = y.index
                for i in range(2):
                    data.loc[data['Date'].isin(markov_idx), f'hamilton_regime_{i}_prob'] = res.smoothed_marginal_probabilities[i].values
                data.loc[data['Date'].isin(markov_idx), 'hamilton_pred_regime'] = res.smoothed_marginal_probabilities.idxmax(axis=1).values
                data.loc[data['Date'].isin(markov_idx), 'hamilton_state'] = res.smoothed_marginal_probabilities.values.argmax(axis=1)
        except Exception as e:
            print(f"Hamilton Markov failed for {symbol}: {e}")

        # --- Add Hidden Markov Model features internally ---
        try:
            hmm_returns = data['Returns'].dropna().values.reshape(-1, 1)
            if len(hmm_returns) >= 100:
                model = GaussianHMM(n_components=2, covariance_type="full", n_iter=200, random_state=42)
                model.fit(hmm_returns)
                hidden_states = model.predict(hmm_returns)
                posteriors = model.predict_proba(hmm_returns)
                hmm_idx = data['Returns'].dropna().index
                data.loc[hmm_idx, 'hmm_state'] = hidden_states
                for i in range(2):
                    data.loc[hmm_idx, f'hmm_state_{i}_prob'] = posteriors[:, i]
        except Exception as e:
            print(f"Hidden Markov failed for {symbol}: {e}")

        # --- Add Gaussian Mixture Model features internally ---
        try:
            gmm_returns = data['Returns'].dropna().values.reshape(-1, 1)
            if len(gmm_returns) >= 100:
                gmm_model = GaussianMixture(n_components=2, covariance_type="full", n_init=10, random_state=42)
                gmm_model.fit(gmm_returns)
                gmm_states = gmm_model.predict(gmm_returns)
                gmm_posteriors = gmm_model.predict_proba(gmm_returns)
                gmm_idx = data['Returns'].dropna().index
                data.loc[gmm_idx, 'gmm_state'] = gmm_states
                for i in range(2):
                    data.loc[gmm_idx, f'gmm_state_{i}_prob'] = gmm_posteriors[:, i]
        except Exception as e:
            print(f"Gaussian Mixture Model failed for {symbol}: {e}")

        # Ensure 'Date' is a column, not index
        if 'Date' not in data.columns:
            data.reset_index(inplace=True)
        data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')

        # Save the engineered features to a new CSV file
        new_dir = f'./Engineered_data'
        os.makedirs(new_dir, exist_ok=True)
        file_name = f'{symbol}_{interval}_features.csv'
        file_path = os.path.join(new_dir, file_name)
        data.to_json(file_path.replace('.csv', '.json'), orient='records', lines=True)
        print(f"Feature engineered data for {symbol} saved successfully.")
    
    return None


# add_features(symbols=['NIFTY'], interval='1d')

def update_features(symbols, interval='1d'):
    for symbol in symbols:
        print(f'Loading data for {symbol}')
        
        # Load the raw data
        data_path = f'./Underlying_data_vbt/{symbol}_{interval}.csv'
        if not os.path.exists(data_path):
            print(f"Data file for {symbol} not found. Please download the data first.")
            continue

        raw_data = pd.read_csv(data_path)
        raw_data['Date'] = pd.to_datetime(raw_data['Date'], format='%Y-%m-%d', errors='coerce')
        raw_data.set_index('Date', inplace=True)
        # raw_data = raw_data.asfreq('B')  # Ensure business day frequency

        # Check if processed data already exists
        processed_path = f'./Engineered_data/{symbol}_{interval}_features.json'
        if os.path.exists(processed_path):
            processed_data = pd.read_json(processed_path, orient='records', lines=True)
            processed_data['Date'] = pd.to_datetime(processed_data['Date'], format='%Y-%m-%d', errors='coerce')
            processed_data.set_index('Date', inplace=True)
            # processed_data = processed_data.asfreq('B')  # Ensure business day frequency

            # Find new data that needs to be processed
            new_data = raw_data.loc[~raw_data.index.isin(processed_data.index)]
            if new_data.empty:
                print(f"No new data to process for {symbol}.")
                continue

            print(f"Processing {len(new_data)} new rows for {symbol}.")
        else:
            print(f"No existing processed data found for {symbol}. Processing all data.")
            new_data = raw_data

        # Apply feature engineering to the new data
        feat_eng = FeatureEngineering(new_data)
        new_data = feat_eng.callme(new_data)

        # Add GARCH volatility
        new_data['garch_vol'] = garch_vol(symbol).values[-len(new_data):]
        new_data['garch_vol'] = new_data['garch_vol'].bfill()
        new_data['garch_vol_pct_chng'] = new_data['garch_vol'].pct_change(fill_method=None)
        new_data['garch_vol_percentile'] = [
            np.round(percentileofscore(raw_data['garch_vol'], i), 0) for i in raw_data['garch_vol']
        ]

        # Combine the new data with the existing processed data
        if os.path.exists(processed_path):
            combined_data = pd.concat([processed_data, new_data])
        else:
            combined_data = new_data

        # Save the updated data to a JSON file
        combined_data.reset_index(inplace=True)
        combined_data.to_json(processed_path, orient='records', lines=True)
        print(f"Feature engineered data for {symbol} saved successfully.")

    return combined_data

def process_symbol(symbol,interval='1d'):
    print(f'Loading data for {symbol}')
    
    # Load the data
    data_path = f'./Underlying_data_vbt/{symbol}_{interval}.csv'
    data_path = os.path.join(data_path)
    if not os.path.exists(data_path):
        print(f"Data file for {symbol} not found. Please download the data first.")
        return
    else:
        data = pd.read_csv(data_path, parse_dates=True)
        data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d', errors='coerce')
    
    # Apply feature engineering using callme
    feat_eng = FeatureEngineering(data)
    data = feat_eng.callme(data)
    
    # Add GARCH volatility
    data['garch_vol'] = garch_vol(symbol).values
    data['garch_vol'] = data['garch_vol'].bfill()
    data['garch_vol_pct'] = data['garch_vol'].diff()
    # Calculate percentiles
    data['garch_vol_percentile'] = [np.round(percentileofscore(data['garch_vol'], i),0) for i in data['garch_vol'] ]
    data['garch_vol_percentile'] = data['garch_vol_percentile'].bfill()

    # Ensure 'Date' is a column, not index
    if 'Date' not in data.columns:
        data.reset_index(inplace=True)
    # Convert 'Date' to string for JSON serialization
    data['Date'] = data['Date'].dt.strftime('%Y-%m-%d')
    
    # Save the engineered features to a new CSV file
    new_dir = f'./Engineered_data'
    os.makedirs(new_dir, exist_ok=True)
    file_name = f'{symbol}_{interval}_features.csv'
    file_path = os.path.join(new_dir, file_name)
    # data.to_csv(file_path, index=True)
    data.to_json(file_path.replace('.csv', '.json'), orient='records', lines=True)
    print(f"Feature engineered data for {symbol} saved successfully.")
    
    return


def create_underlying_analytics(symbols):
    get_underlying_data_vbt(symbols, period='10y', interval='1d')
    add_features(symbols)
    # plot_garch_vs_rsi(symbols)
    return "Analytics created successfully for all symbols."
