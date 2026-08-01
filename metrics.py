# metrics.py
import numpy as np

def performance(port_ret):
    ann_ret = port_ret.mean() * 252
    ann_vol = port_ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol
    cum = (1 + port_ret).cumprod()
    drawdown = (cum / cum.cummax() - 1).min()
    return {
        "Annual Return": ann_ret,
        "Annual Vol": ann_vol,
        "Sharpe": sharpe,
        "Max Drawdown": drawdown,
    }
