import numpy as np
import pandas as pd

from shocklab import risk
from shocklab.backtest import COST_BP, month_ends, simulate
from shocklab.optimize import frontier, min_cvar, minimal_change, ru_cvar


def scenarios(seed=0, S=3000, N=6):
    rng = np.random.default_rng(seed)
    mu = np.linspace(-0.002, 0.01, N)
    vol = np.linspace(0.01, 0.08, N)
    common = rng.standard_t(4, size=(S, 1)) * 0.02
    return mu + common * np.linspace(0.2, 1.5, N) + rng.standard_t(4, size=(S, N)) * vol


def test_ru_cvar_matches_tail_average():
    r = np.random.default_rng(1).normal(0, 0.03, 200_000)
    assert abs(ru_cvar(r) - risk.cvar(r)) < 1e-3


def test_min_cvar_beats_feasible_portfolios_and_respects_constraints():
    scen = scenarios()
    res = min_cvar(scen, w_max=0.4)
    assert np.isclose(res.w.sum(), 1) and (res.w >= -1e-9).all() and (res.w <= 0.4 + 1e-6).all()
    rng = np.random.default_rng(2)
    for _ in range(200):
        w = rng.dirichlet(np.ones(6))
        if w.max() <= 0.4:
            assert ru_cvar(scen @ w) >= res.cvar - 1e-6


def test_return_floor_and_turnover_limit():
    scen = scenarios()
    base = min_cvar(scen, w_max=0.4)
    floor = base.exp_ret + 0.002
    res = min_cvar(scen, w_max=0.4, min_ret=floor)
    assert res.exp_ret >= floor - 1e-7 and res.cvar >= base.cvar - 1e-9
    w0 = np.full(6, 1 / 6)
    lim = min_cvar(scen, w_max=0.4, w0=w0, turnover=0.1)
    assert lim.turnover <= 0.1 + 1e-6


def test_minimal_change_gets_80pct_with_less_turnover():
    scen = scenarios()
    w0 = np.full(6, 1 / 6)
    small, full = minimal_change(scen, w0, w_max=0.4)
    c0 = ru_cvar(scen @ w0)
    assert small.cvar <= c0 - 0.8 * (c0 - full.cvar) + 1e-6
    assert small.turnover <= np.abs(full.w - w0).sum() + 1e-9


def test_frontier_is_monotone():
    scen = scenarios()
    pts = frontier(scen, scen, n_points=6, w_max=0.4)
    rets = [p["exp_ret"] for p in pts]
    cvars = [p["cvar"] for p in pts]
    assert all(np.diff(rets) >= -1e-7) and all(np.diff(cvars) >= -1e-6)


def test_month_ends():
    idx = pd.bdate_range("2017-12-01", "2018-03-15")
    me = month_ends(idx, "2018-01-01")
    assert me[0] == pd.Timestamp("2017-12-29") and pd.Timestamp("2018-01-31") in me


def test_backtest_costs_and_drift():
    idx = pd.bdate_range("2018-01-01", periods=30)
    rets = pd.DataFrame({"A": 0.01, "B": 0.0}, index=idx)
    targets = {idx[0]: pd.Series({"A": 0.5, "B": 0.5}), idx[20]: pd.Series({"A": 0.5, "B": 0.5})}
    r, tvs = simulate(rets, targets, idx[-1])
    assert np.isclose(r.iloc[0], 0.005 - 1.0 * COST_BP / 1e4)  # initial buy: turnover 1.0
    assert np.isclose(r.iloc[1], 0.5 * 1.01 / 1.005 * 0.01)  # weight drifted toward A
    assert 0 < tvs[1] < 0.2
    # the rebalance cost is charged on the day after the rebalance
    assert r.iloc[20] < r.iloc[19]
