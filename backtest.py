# backtest.py
import numpy as np
import pandas as pd

def backtest(signal, returns, quantile=0.2):
    """Signal determines positions; returns are next-period returns."""
    positions = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)
    for date in signal.index:
        s = signal.loc[date].dropna()
        if len(s) < 10:
            continue
        n = int(len(s) * quantile)
        longs = s.nlargest(n).index
        shorts = s.nsmallest(n).index
        positions.loc[date, longs] = 1.0 / n
        positions.loc[date, shorts] = -1.0 / n
    # Portfolio daily return = yesterday's positions * today's returns
    port_ret = (positions.shift(1) * returns).sum(axis=1)
    return port_ret, positions

