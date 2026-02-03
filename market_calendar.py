import investpy #used for importing global economic calendar
import pandas as pd
import numpy as np
import pendulum as pm
import json
import yfinance as yf


# The below function fetches the global economic calendar using the investpy library.
# It filters the data for high-importance events and specific currencies (USD, GBP, EUR, INR, JPY, CNH).

def economic_calendar():
    from_date=pm.now().subtract(days=1).strftime('%d/%m/%Y')
    to_date=pm.now().add(days=3).strftime('%d/%m/%Y')
    calendar = investpy.news.economic_calendar(time_zone=None, from_date=from_date, to_date=to_date)
    calendar = calendar.drop(columns=['id','time'])
    xccy = ['USD','GBP','EUR','INR','JPY','CNH']
    importance_level = ['high']
    df = calendar.query("importance in @importance_level & currency in @xccy or zone=='india'").reset_index(drop=True)
    
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.title()
        df['currency'] = df['currency'].str.upper()
    df.columns = [col.title() for col in df.columns]
    df.style.format({'Date':'{:%b %Y}'})
    df.set_index('Date')
    
    return df

economic_calendar()
# The below function fetches the earnings calendar for a list of stock tickers using the yfinance library.


def stock_earnings_calendar(tickers):
    from_date = pm.now().subtract(days=1)  # .strftime('%d/%m/%Y')
    to_date = pm.now().add(days=120)  # .strftime('%d/%m/%Y')
    yf_tickers = ['^NSEI' if ticker == 'NIFTY' else '^NSEBANK' if ticker == 'BANKNIFTY' else f'{ticker}.NS' for ticker in tickers]

    ls = []
    for ticker in yf_tickers:
        # print(f"Fetching earnings calendar for {ticker} from {from_date} to {to_date}")
        try:
            # Fetch the stock data
            stock = yf.Ticker(ticker)
            earnings = stock.earnings_dates

            if earnings.empty:
                print(f"No earnings data available for {ticker}.")
                continue  # Skip to the next ticker

            # Format and clean up the DataFrame
            earnings.reset_index(inplace=True)
            earnings.columns = ['Date', 'EPS Estimate', 'Reported EPS', 'Surprise (%)',"A"]
            earnings['Date'] = pd.to_datetime(earnings['Date'], errors='coerce')
            earnings = earnings.query("Date >= @from_date & Date <= @to_date").reset_index(drop=True)
            earnings = earnings.sort_values(by='Date').reset_index(drop=True)
            earnings['Ticker'] = ticker  # Add the ticker symbol to the DataFrame
            ls.append(earnings)
        except Exception as e:
            print(f"Error fetching earnings calendar for {ticker}: {e}")
            continue  # Skip to the next ticker

    if not ls:
        print("No earnings data found for the provided tickers.")
        return pd.DataFrame()

    df = pd.concat(ls, ignore_index=True)
    df.set_index('Ticker', inplace=True)
    df.sort_values(by='Date', inplace=True)
    
    return df

stock_earnings_calendar(['NIFTY', 'BANKNIFTY', 'RELIANCE', 'TCS', 'INFY'])

# The below code fetches the latest news articles related to Nifty50 stocks using the NewsAPI.
# It requires an API key from NewsAPI or a similar service. The code fetches the latest news articles and prints the title, source, publication date, and URL of each article.


import requests
import pandas as pd

def fetch_nifty50_news():
    # Define the query parameters
    api_key = "1b8de5003ddd4249bb3173cc8413a5dc"
    from_date = pd.to_datetime(pm.now().subtract(days=1)) # .strftime('%d/%m/%Y')
    to_date = pd.to_datetime(pm.now())
    url = "https://newsapi.org/v2/everything"
    query = "Nifty50 stocks OR Indian stock market OR Nifty50"
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "apiKey": api_key
    }

    # Make the API request
    response = requests.get(url, params=params)

    if response.status_code == 200:
        news_data = response.json()
        articles = news_data.get("articles", [])
        if articles:
            
            df = pd.DataFrame(articles)
            df = df[['title', 'publishedAt', 'source']]
            df['source'] = df['source'].apply(lambda x: x['name'])
            df['title'] = df['title'].str.replace('Stock market update:', '', regex=False).str.strip()
            df['title'] = df['title'].str.ljust(50)
            df['publishedAt'] = pd.to_datetime(df['publishedAt'], errors='coerce')
            df = df[0:10]  # Fetch top 5 articles
            
        else:
            print("No news articles found.")
    else:
        print(f"Failed to fetch news. HTTP Status Code: {response.status_code}")
    
    return df