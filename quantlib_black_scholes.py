import QuantLib as ql
import pandas as pd

def calculate_greeks(option_price, spot_price, strike_price, risk_free_rate, time_to_expiry, option_type):
    """
    Calculate the implied volatility of an option using QuantLib.

    Args:
        option_price (float): The market price of the option.
        spot_price (float): The current price of the underlying asset.
        strike_price (float): The strike price of the option.
        risk_free_rate (float): The risk-free interest rate (annualized).
        time_to_expiry (float): Time to expiry in years.
        option_type (str): 'call' or 'put'.

    Returns:
        float: The implied volatility.
    """
    # Define option type
    if option_type.lower() == 'call':
        option_type = ql.Option.Call
    elif option_type.lower() == 'put':
        option_type = ql.Option.Put
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    # Set up QuantLib objects
    calendar = ql.NullCalendar()
    day_count = ql.Actual365Fixed()
    settlement_date = ql.Date.todaysDate()
    maturity_date = settlement_date + int(time_to_expiry * 365)

    # Option payoff and exercise
    payoff = ql.PlainVanillaPayoff(option_type, strike_price)
    exercise = ql.EuropeanExercise(maturity_date)

    # Create the option
    european_option = ql.VanillaOption(payoff, exercise)

    # Market data
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
    rate_handle = ql.YieldTermStructureHandle(ql.FlatForward(settlement_date, risk_free_rate, day_count))
    vol_handle = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(settlement_date, calendar, 0.05, day_count))  # Initial guess for volatility

    # Black-Scholes-Merton process
    bsm_process = ql.BlackScholesMertonProcess(spot_handle, rate_handle, rate_handle, vol_handle)

    # Implied volatility calculation
    implied_vol = european_option.impliedVolatility(option_price, bsm_process)
    
    # Pricing engine
    engine = ql.AnalyticEuropeanEngine(bsm_process)
    european_option.setPricingEngine(engine)

    # Calculate Greeks
    delta = european_option.delta()
    gamma = european_option.gamma()
    vega = european_option.vega()
    theta = european_option.theta()
    rho = european_option.rho()
    
    greeks = {
        'delta': round(delta,2),
        'gamma': round(gamma,4),
        'vega': round(vega/100,3),
        'theta': round(theta,2),  
        'rho': round(rho/100,3),  # Convert to percentage rho
        'IV': round(implied_vol*100,2)}  # Convert to percentage implied volatility
    
    return greeks

