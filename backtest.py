# backtest.py
import numpy as np

def backtest(signal, returns, quantile=0.2):
    """signal 决定持仓,returns 是下一期收益"""
    positions = pd.DataFrame(0, index=signal.index, columns=signal.columns)
    for date in signal.index:
        s = signal.loc[date].dropna()
        if len(s) < 10:
            continue
        n = int(len(s) * quantile)
        longs = s.nlargest(n).index
        shorts = s.nsmallest(n).index
        positions.loc[date, longs] = 1.0 / n
        positions.loc[date, shorts] = -1.0 / n
    # 组合日收益 = 昨天持仓 × 今天收益
    port_ret = (positions.shift(1) * returns).sum(axis=1)
    return port_ret, positions
