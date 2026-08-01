# signals.py
import pandas as pd

def momentum(prices, lookback=60, skip=5):
    """过去 lookback 天收益(跳过最近 skip 天,避开反转)"""
    return prices.shift(skip) / prices.shift(lookback) - 1

def reversal(returns, window=5):
    """短期反转:过去 window 天收益取负"""
    return -returns.rolling(window).sum()

def low_volatility(returns, window=60):
    """低波动因子:过去波动率取负(低波动更优)"""
    return -returns.rolling(window).std()
