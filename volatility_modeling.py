import pandas as pd
import numpy as np
import os
import pickle
from arch import arch_model

# symbols = ['^NSEI', '^NSEBANK', 'RELIANCE.NS', 'TATAMOTORS.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS', 'TCS.NS', 'AXISBANK.NS', 'BAJFINANCE.NS']

def egarch(symbols):
    """Load data for each symbol, fit an EGARCH model, and save the optimized model."""
    for symbol in symbols:
        print(f'Loading data for {symbol}')
        
        # Load the data
        data_path = f'./Underlying_data_vbt/{symbol}_1d.csv'
        if not os.path.exists(data_path):
            print(f"Data file for {symbol} does not exist.")
            continue
        
        data = pd.read_csv(data_path)
        
        # Ensure 'Close' column is present
        if 'Close' not in data.columns:
            print(f"'Close' column not found in {symbol} data.")
            continue
        
        # Fit the EGARCH model
        returns = np.log(data['Close']).diff().dropna() * 100  # Convert to percentage returns
        g1 = arch_model(returns, vol='EGARCH', p=1,o=0, q=1, dist='Normal', rescale=False)
        g1_fit = g1.fit(disp='off')
        
        # Save the model results to a text file
        model_results_path = f'./EGARCH_Results/{symbol}_EGARCH_results.txt'
        os.makedirs(os.path.dirname(model_results_path), exist_ok=True)
        with open(model_results_path, 'w') as f:
            f.write(str(g1_fit.summary()))
        
        # Save the optimized model using pickle
        model_pickle_path = f'./EGARCH_Models/{symbol}_EGARCH_model.pkl'
        os.makedirs(os.path.dirname(model_pickle_path), exist_ok=True)
        with open(model_pickle_path, 'wb') as f:
            pickle.dump(g1_fit, f)
        print(f"Optimized EGARCH model for {symbol} saved successfully.")

 

def garch_vol(symbol):
 
    model_pickle_path = f'./EGARCH_Models/{symbol}_EGARCH_model.pkl'

    # Check if the model file exists
    if not os.path.exists(model_pickle_path):
        print(f"Model file for {symbol} does not exist.")
        print(f"Running the EGARCH function to create the model first.")
        egarch([symbol])
        # return None

    # Load the saved model
    with open(model_pickle_path, 'rb') as f:
        saved_model = pickle.load(f)

    # Extract the model parameters
    params = saved_model.params
    
    # Load new data (e.g., returns)
    new_data_path = f'./Underlying_data_vbt/{symbol}_1d.csv'
    data = pd.read_csv(new_data_path)
    returns = np.log(data['Close']).diff().dropna() * 100  # Convert to percentage returns

    # Fit the EGARCH model to the new data
    garch_model = arch_model(returns, vol='EGARCH', p=1, o=0, q=1, dist='Normal', rescale=False)
    refitted_model = garch_model.fit(starting_values=params, disp='off')
    
    garch_vol = refitted_model.conditional_volatility * np.sqrt(251)  # Annualize the volatility
    
    index_diff = len(data) - len(garch_vol)
    if index_diff > 0:
        # Create a Series of NaN values with length index_diff
        nan_series = pd.Series([np.nan] * index_diff, index=returns.index[:index_diff])
    else:
        # If the lengths are equal, create an empty Series 
        nan_series = pd.Series(dtype=float)
        
    garch_vol = pd.concat([nan_series, pd.Series(garch_vol, index=data.index[-len(garch_vol):])])
    

    print(f"Refitted EGARCH model for {symbol}:")   
    
    return garch_vol


def getDailyVol(close, span0=100):
    df0= close.index.searchsorted(close.index - pd.Timedelta(days=1))
    df0=df0[df0>0]
    df0= pd.Series(close.index[df0-1], index=close.index[close.shape[0]-df0.shape[0]:])
    df0 = close.loc[df0.index] / close.loc[df0.values].values -1
    df0 = df0.ewm(span=span0).std()
    return df0

