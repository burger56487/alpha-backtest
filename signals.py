# signals.py
import pandas as pd

def momentum(prices, lookback=60, skip=5):
    """Return over the past `lookback` days, skipping the most recent `skip` days to avoid reversal."""
    return prices.shift(skip) / prices.shift(lookback) - 1

def reversal(returns, window=5):
    """Short-term reversal: negative of the cumulative return over the past `window` days."""
    return -returns.rolling(window).sum()

def low_volatility(returns, window=60):
    """Low-volatility factor: negative of the past volatility (lower volatility is preferred)."""
    return -returns.rolling(window).std()
