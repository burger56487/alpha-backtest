import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from signals import momentum, reversal, low_volatility
from backtest import backtest
from metrics import performance

TICKERS = [
    "AAPL","MSFT","GOOGL","AMZN","META","NVDA","JPM","V","JNJ","WMT",
    "PG","MA","HD","BAC","XOM","CVX","KO","PEP","ABBV","MRK",
    "COST","MCD","CSCO","ADBE","CRM","NKE","INTC","T","VZ","DIS",
]

def get_data(start="2019-01-01", end="2024-01-01"):
    os.makedirs("data", exist_ok=True)
    cache = "data/prices.csv"
    if os.path.exists(cache):
        prices = pd.read_csv(cache, index_col=0, parse_dates=True)
    else:
        raw = yf.download(TICKERS, start=start, end=end)["Close"]
        prices = raw.dropna(axis=1, how="any")
        prices.to_csv(cache)
    return prices

def run_factor(signal, returns, cost_bps=0.0):
    """返回:扣成本后的组合日收益, 绩效字典"""
    port_ret, positions = backtest(signal, returns)
    turnover = positions.diff().abs().sum(axis=1)
    port_ret_net = port_ret - turnover * (cost_bps / 1e4)
    perf = performance(port_ret_net)
    perf["Avg Turnover"] = turnover.mean()
    return port_ret_net, perf

def main():
    prices = get_data()
    returns = prices.pct_change().dropna()
    print(f"Loaded {prices.shape[1]} stocks, {prices.shape[0]} days\n")

    signals = {
        "Momentum":       momentum(prices, lookback=60, skip=5),
        "Reversal":       reversal(returns, window=5),
        "Low-Volatility": low_volatility(returns, window=60),
    }

    os.makedirs("results", exist_ok=True)

    # ------------------------------------------------------------------
    # 增强 1: 无成本 vs 有成本 对比
    # ------------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    rows = []
    for name, sig in signals.items():
        sig = sig.reindex(returns.index)
        # 无成本
        _, perf_gross = run_factor(sig, returns, cost_bps=0.0)
        # 有成本 (5bp)
        port_net, perf_net = run_factor(sig, returns, cost_bps=5.0)
        rows.append({
            "Factor": name,
            "Sharpe (gross)": round(perf_gross["Sharpe"], 3),
            "Sharpe (net 5bp)": round(perf_net["Sharpe"], 3),
            "Max Drawdown": round(perf_net["Max Drawdown"], 3),
            "Avg Turnover": round(perf_net["Avg Turnover"], 3),
        })
        cum = (1 + port_net).cumprod()
        plt.plot(cum.index, cum.values, label=name)

    plt.title("Cumulative Return of Long-Short Factor Portfolios (net 5bp)")
    plt.xlabel("Date"); plt.ylabel("Cumulative Return")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig("results/cumulative_returns.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved results/cumulative_returns.png")

    summary = pd.DataFrame(rows).set_index("Factor")
    print("\n=== Performance Summary: gross vs net of 5bp cost ===")
    print(summary.to_string())

    # ------------------------------------------------------------------
    # 增强 2: momentum lookback 敏感性 + 柱状图
    # ------------------------------------------------------------------
    lookbacks = [20, 40, 60, 90, 120]
    sharpes = []
    print("\n=== Momentum lookback sensitivity (Sharpe, net 5bp) ===")
    for lb in lookbacks:
        sig = momentum(prices, lookback=lb, skip=5).reindex(returns.index)
        _, perf = run_factor(sig, returns, cost_bps=5.0)
        sharpes.append(perf["Sharpe"])
        print(f"  lookback={lb:>3d}:  Sharpe={perf['Sharpe']:.2f}")

    plt.figure(figsize=(8, 5))
    colors = ["#d9534f" if s < 0 else "#5cb85c" for s in sharpes]  # 负红正绿
    plt.bar([str(lb) for lb in lookbacks], sharpes, color=colors)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Momentum Sharpe vs Lookback Horizon (net 5bp)")
    plt.xlabel("Lookback (days)"); plt.ylabel("Sharpe Ratio")
    plt.grid(True, axis="y", alpha=0.3)
    plt.savefig("results/momentum_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved results/momentum_sensitivity.png")

if __name__ == "__main__":
    main()
