import pandas as pd
import numpy as np
import pendulum as p
from datetime import datetime
from dateutil.parser import parse


# determining ATM strike price based on the underlying price and the option type

def atm_strike(spot,option_chain):
    option_chain=option_chain[option_chain['Expiry']==option_chain['Expiry'].unique()[0]]
    multiplier = int(option_chain['strike_price'].diff().unique()[-1])
    # multiplier = int(option_chain['strike_price'].diff().unique().min())    
  
    # Smaller multiple
    a = (spot // multiplier) * multiplier
    # Larger multiple
    b = a + multiplier
    # Return of closest of two
    return int((b if spot - a > b - spot else a))



# Function to determine time to expiration in years

def tau(expiry):
    # expiry = pd.to_datetime(expiry, errors='coerce')
    current_time = p.now(tz='local')
    # expiry = parse(expiry)
    expiry = p.datetime(expiry.year, expiry.month, expiry.day,15,30)
    delta = expiry.diff(current_time).in_seconds()
    delta = delta / (365 * 24 * 60 * 60)  # Convert seconds to years
    return delta


# def tau(expiry_date):
#     """
#     Calculate the time to expiry in years.
#     Args:
#         expiry_date (datetime): The expiry date.
#     Returns:
#         float: Time to expiry in years.
#     """
#     today = datetime.now()
#     delta = expiry_date - today
#     return delta.days / 365.0


# Function to highlight dataframe rows based on a condition
def highlight_rows(x):
    if x['STRIKE PRICE'] == atm_strike:
        return['background-color: pink']*x.shape[0]
    else:
        return['background-color: white']*x.shape[0]


# Placeholder for future implementation
def main():
    print("Options Utility Script")

if __name__ == "__main__":
    main()