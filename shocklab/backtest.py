"""Walk-forward monthly backtest, 2018 onwards.

At each month-end close D we decide weights using only data observed by D, trade
at D's close paying 10 bp per unit of turnover, then let weights drift with daily
returns until the next month-end. Strategies:
  - CVaR (MDN):       min CVaR over 5,000 MDN scenarios with bootstrapped factor moves
  - CVaR (baseline):  same optimiser, scenarios from the linear factor + t model
  - Equal weight, 60/40 SPY/IEF, Minimum variance (trailing 1y daily covariance)
All long-only; the optimised ones cap each asset at 15%.
Caveat: TSX names are held in CAD (no FX conversion), a simplification.
"""
from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd

from shocklab import baseline
from shocklab import config as C
from shocklab.data import Panel
from shocklab.mdn import MDNModel, load_for_date
from shocklab.optimize import W_MAX, min_cvar
from shocklab.scenarios import base_scenarios, bootstrap_shocks, shock_history

COST_BP = 10
N_SCEN = 5_000
MIN_RET = 0.0  # expected 10-day return floor for the CVaR strategies


def month_ends(index: pd.DatetimeIndex, start: str) -> list[pd.Timestamp]:
    """Last trading day of each month; the first decision is the close before `start`."""
    s = pd.Series(index, index=index)
    ends = s.groupby([index.year, index.month]).max().tolist()
    first = [d for d in ends if d < pd.Timestamp(start)][-1]
    return [d for d in ends if d >= first]


def min_variance(daily: pd.DataFrame, w_max: float = W_MAX) -> np.ndarray:
    """min w' Sigma w  s.t. sum w = 1, 0 <= w <= w_max (a small QP)."""
    sigma = np.cov(daily.to_numpy(), rowvar=False)
    sigma = (sigma + sigma.T) / 2 + 1e-10 * np.eye(len(sigma))
    w = cp.Variable(len(sigma))
    cp.Problem(cp.Minimize(cp.quad_form(w, sigma)), [cp.sum(w) == 1, w >= 0, w <= w_max]).solve(solver=cp.CLARABEL)
    v = np.clip(w.value, 0, None)
    return v / v.sum()


def target_weights(panel: Panel, D: pd.Timestamp, daily_ret: pd.DataFrame, cache: dict[int, MDNModel],
                   seed: int) -> dict[str, pd.Series]:
    """Every strategy's target weights at decision date D (uses data <= D only)."""
    past = daily_ret.loc[:D].iloc[-252:]
    universe = [a for a in C.ASSETS if past[a].notna().all()]
    out = {
        "Equal weight": pd.Series(1 / len(universe), index=universe),
        "60/40 SPY/IEF": pd.Series({"SPY": 0.6, "IEF": 0.4}),
        "Min variance": pd.Series(min_variance(past[universe]), index=universe),
    }
    shocks = bootstrap_shocks(shock_history(panel, D), N_SCEN, seed)
    regime = panel.regime_all.loc[D].to_numpy(float)
    base = baseline.fit_before(panel, D)
    for name, model in [("CVaR (MDN)", load_for_date(D, cache)), ("CVaR (baseline)", base)]:
        assets = [a for a in universe if a in model.assets]
        idx = [model.assets.index(a) for a in assets]
        scen = np.expm1(base_scenarios(model, shocks, regime, seed)[:, idx])
        out[name] = pd.Series(min_cvar(scen, min_ret=MIN_RET).w, index=assets)
    return out


def simulate(daily_simple: pd.DataFrame, targets: dict[pd.Timestamp, pd.Series], end: pd.Timestamp) -> tuple[pd.Series, list[float]]:
    """Daily portfolio returns with drift between rebalances and costs on rebalance days."""
    dates = sorted(targets)
    days = daily_simple.loc[dates[0]:end].index[1:]
    rets, turnovers = [], []
    w = pd.Series(0.0, index=daily_simple.columns)
    pending = targets[dates[0]].reindex(daily_simple.columns, fill_value=0.0)
    tv = float(np.abs(pending - w).sum())
    w, turnovers = pending, [tv]
    cost_today = tv * COST_BP / 1e4
    for day in days:
        r = daily_simple.loc[day].fillna(0.0)
        gross = float((w * r).sum())
        rets.append(gross - cost_today)
        w = w * (1 + r) / (1 + gross)  # drift
        cost_today = 0.0
        if day in targets and day != dates[0]:
            new = targets[day].reindex(daily_simple.columns, fill_value=0.0)
            tv = float(np.abs(new - w).sum())
            turnovers.append(tv)
            # trade at the close: cost hits tomorrow's return (same compounding, simpler loop)
            cost_today = tv * COST_BP / 1e4
            w = new
    return pd.Series(rets, index=days), turnovers


def metrics(r: pd.Series, rf_daily: pd.Series, turnovers: list[float]) -> dict[str, float]:
    eq = (1 + r).cumprod()
    years = len(r) / 252
    monthly = (1 + r).groupby([r.index.year, r.index.month]).prod() - 1
    excess = r - rf_daily.reindex(r.index).fillna(0.0)
    return {
        "CAGR": float(eq.iloc[-1] ** (1 / years) - 1),
        "Volatility": float(r.std() * np.sqrt(252)),
        "Max drawdown": float((eq / eq.cummax() - 1).min()),
        "Worst month": float(monthly.min()),
        "Sharpe": float(excess.mean() / r.std() * np.sqrt(252)),
        "Avg monthly turnover": float(np.mean(turnovers[1:])) if len(turnovers) > 1 else 0.0,
    }


def run(panel: Panel, start: str = "2018-01-01") -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    daily_simple = panel.prices.pct_change(fill_method=None)
    daily_log = np.log(panel.prices).diff()
    rf_daily = panel.fred["rf"].ffill() / 100 / 252
    end = panel.prices.index[-1]
    cache: dict[int, MDNModel] = {}
    targets: dict[str, dict[pd.Timestamp, pd.Series]] = {}
    decisions = [d for d in month_ends(panel.prices.index, start) if d < end]
    for i, D in enumerate(decisions):
        for name, w in target_weights(panel, D, daily_log, cache, C.SEED + i).items():
            targets.setdefault(name, {})[D] = w
    returns, rows = {}, []
    for name, tg in targets.items():
        r, tvs = simulate(daily_simple, tg, end)
        returns[name] = r
        rows.append({"strategy": name, **metrics(r, rf_daily, tvs)})
    last = {name: {k: round(float(v), 4) for k, v in tg[max(tg)].items() if v > 1e-4} for name, tg in targets.items()}
    return pd.DataFrame(rows), pd.DataFrame(returns), last
