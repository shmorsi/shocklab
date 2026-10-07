"""Walk-forward monthly backtest -> results/backtest_*.

Usage: python -m scripts.backtest
"""
from __future__ import annotations

import json

from shocklab import config as C
from shocklab.backtest import run
from shocklab.data import load_panel


def main() -> None:
    metrics, returns, last = run(load_panel())
    metrics.to_csv(C.RESULTS_DIR / "backtest_metrics.csv", index=False)
    equity = (1 + returns).cumprod()
    equity.to_csv(C.RESULTS_DIR / "backtest_equity.csv")
    (C.RESULTS_DIR / "backtest_last_weights.json").write_text(json.dumps(last, indent=1))
    lines = ["| Strategy | CAGR | Volatility | Max drawdown | Worst month | Sharpe | Avg monthly turnover |",
             "|---|---|---|---|---|---|---|"]
    for _, r in metrics.iterrows():
        lines.append(f"| {r.strategy} | {r.CAGR:.2%} | {r.Volatility:.2%} | {r['Max drawdown']:.2%} | "
                     f"{r['Worst month']:.2%} | {r.Sharpe:.2f} | {r['Avg monthly turnover']:.1%} |")
    md = (f"Period {returns.index[0].date()} to {returns.index[-1].date()}, monthly rebalancing, "
          f"10 bp per unit turnover, Sharpe vs 3M T-bill.\n\n" + "\n".join(lines))
    (C.RESULTS_DIR / "backtest_metrics.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
