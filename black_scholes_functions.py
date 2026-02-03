import pandas as pd
import numpy as np
from scipy.stats import norm
from math import log, sqrt, exp
from datetime import datetime, timedelta, time
from scipy.optimize import minimize




# Function to price options using the Black-Scholes model

class OptionsBS():
    
    def __init__(self, strike, spot, risk_free_rate, volatility, time_to_expiry):
        
        self.strike = strike
        self.spot = spot
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility
        self.time_to_expiry = time_to_expiry
        
        
        self._d1_ = (log(self.spot/self.strike) + (self.risk_free_rate + 0.5*self.volatility**2)*self.time_to_expiry)/(self.volatility*sqrt(self.time_to_expiry))   
        
        self._d2_ = self._d1_ - self.volatility*sqrt(self.time_to_expiry)
        
    def __iter__(self):
        
        for item in [self.spot, self.risk_free_rate, self.volatility, self.time_to_expiry]:
            
            yield item
            
    def callBS(self):
        
        callBS = self.spot*norm.cdf(self._d1_) - self.strike*exp(-self.risk_free_rate*self.time_to_expiry)*norm.cdf(self._d2_)
        
        return np.round(callBS,2)
    
    def putBS(self):
        
        putBS = self.strike*exp(-self.risk_free_rate*self.time_to_expiry)*norm.cdf(-self._d2_) -self.spot*norm.cdf(-self._d1_)
        
        return np.round(putBS,2)
    
    def vegaBS(self):
        
        vega = self.spot*norm.pdf(self._d1_)*np.sqrt(self.time_to_expiry)
        
        return int(vega)
    
# function to determine Implied_Volatility using Newton-Raphson method

class Implied_Vol(OptionsBS):
    def __init__(self, strike, spot, risk_free_rate, volatility, time_to_expiry):
        
        super().__init__(strike, spot, risk_free_rate, volatility, time_to_expiry)
    
    def __iter__(self): #to make the class iterable
        
        for item in [self.strike, self.spot, self.risk_free_rate, self.volatility, self.time_to_expiry]:
            
            yield item
    f = 1
    # global f
    def newton_iv(self, callprice=None, putprice=None):
        V0 = 1 # initial guess
        h = 0.001
        tolerance = 1e-7
        epsilon = 1e-14                             # some kind of error or floor
        maxiter = 200
        global f
     
        if callprice:
            # f(x) = Black Scholes Call price - Market Price - defining the f(x) here
            f = lambda V: eval('OptionsBS')(self.strike, self.spot, self.risk_free_rate, V, self.time_to_expiry).callBS() - callprice
        if putprice:
            f = lambda V: eval('OptionsBS')(self.strike, self.spot, self.risk_free_rate, V, self.time_to_expiry).putBS() - putprice

        for i in range(maxiter):
            
            y = f(V0)
            yprime = (f(V0+h) - f(V0-h))/(2*h)
            
            if abs(yprime)<epsilon:
                break
            V1 = V0 - y/yprime
            
            if (abs(V1-V0) <= tolerance*abs(V1)):
                break
            V0 = V1
        
        return np.round(V1,3)
    
# Functions to determine Options Greeks using the Black-Scholes model

class Greeks():
    
    def __init__(self, strike, spot, risk_free_rate, volatility, time_to_expiry):
        
        self.strike = strike
        self.spot = spot
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility
        self.time_to_expiry = time_to_expiry
        
        
        self._d1_ = (log(self.spot/self.strike) + (self.risk_free_rate + 0.5*self.volatility**2)*self.time_to_expiry)/(self.volatility*sqrt(self.time_to_expiry))   
        
        self._d2_ = self._d1_ - self.volatility*sqrt(self.time_to_expiry)
        
        self._PV_= exp**-(self.risk_free_rate*self.time_to_expiry)
        
    def delta(self,option_type=None):
        
        if option_type == 'call':
            delta = norm.cdf(self._d1_)
            
        if option_type == 'put':
            delta = -norm.cdf(-self._d1_)
            
        return np.round(delta,2)
    
    def gamma(self):
        
        gamma = norm.pdf(self._d1_) / (self.spot * self.volatility*np.sqrt(self.time_to_expiry))
        
        return np.round(gamma,4)
    
    def vega(self):
        
        if self.volatility == 0 or self.time_to_expiry == 0:
            return 0.0
        else:
            return np.round(self.spot * norm.pdf(self._d1_) * self.time_to_expiry**0.5 / 100,3)

    
    def theta(self,option_type=None):
        
        if option_type=='call':
            theta = -self.spot * norm.pdf(self._d1_) * self.volatility / (2 * self.time_to_expiry**0.5) - self.risk_free_rate * self.strike * self._PV_ * norm.cdf(self._d2_)

        if option_type=='put':
            theta = -self.spot * norm.pdf(self._d1_) * self.volatility / (2 * self.time_to_expiry**0.5) + self.risk_free_rate * self.strike * self._PV_ * norm.cdf(-self._d2_)
        return np.round(theta/365,3)

    
    def rho(self,option_type=None):
    
        if option_type=='call':
            rho = self.strike * self.time_to_expiry * self._PV_ * norm.cdf(self._d2_) / 100
        
        if option_type=='put':
            rho = -self.strike * self.time_to_expiry * self._PV_ * norm.cdf(-self._d2_) / 100

        return np.round(rho,3)
    
# Placeholder for future implementation
def main():
    print("Options Black Scholes Script")

if __name__ == "__main__":
    main()



# file added to github on 19th Aprl 2025