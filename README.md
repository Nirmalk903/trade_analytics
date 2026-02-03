This is a trading app

Gets the list of top 15 stocks and Nifty as well as BankNifty, in total 17 active underlyings based on total volume (Options + Futures)
For each underlying , downloads data from YFinance and creates features using VectorBT
Plots RSI vs Garch volatility percentile


Feature Engineering
- Add VIX and analytics based on it (percentage change, pecentile)
- Add FX and Dollar rate
- Add 5y/10y bond yields
- Add Inflation data as and when available, otherwise concensus
- Add global indices performance (SPX, NK225, STOXX)
- Even calendar (Macro and stock specific corporate actions) - ie. if event is due within 5 days
- Advanced technical indicators - i.e VWAP

         