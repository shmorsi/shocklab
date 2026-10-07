"""Rockafellar–Uryasev CVaR minimisation over scenarios (a linear program).

For scenarios r_s (S x N simple returns), weights w, loss L_s = -r_s . w:

    CVaR_b(w) = min_a  a + 1/((1-b) S) * sum_s max(L_s - a, 0)

The optimal a is the VaR. Replacing max(., 0) by slack variables u_s >= L_s - a,
u_s >= 0 turns "minimise CVaR over w" into an LP:

    min_{w, a, u}  a + sum(u) / ((1-b) S)
    s.t. u_s >= -r_s . w - a,  u >= 0,  sum(w) = 1,  0 <= w <= w_max,
         mean_return(w) >= r_min,  ||w - w0||_1 <= turnover (optional)
"""
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from shocklab import config as C

W_MAX = 0.15


@dataclass
class OptResult:
    w: np.ndarray
    cvar: float  # on the risk scenarios, R-U definition
    exp_ret: float  # mean simple return on the return scenarios
    turnover: float  # ||w - w0||_1 (0 if no w0)
    status: str


def ru_cvar(port: np.ndarray, conf: float = C.CONF) -> float:
    """Evaluate the R-U formula at its minimiser a = VaR (the conf-quantile of losses)."""
    loss = -port
    a = np.quantile(loss, conf)
    return float(a + np.maximum(loss - a, 0).mean() / (1 - conf))


def _problem(risk_scen, ret_scen, conf, w_max, min_ret, w0, turnover):
    S, N = risk_scen.shape
    w, a, u = cp.Variable(N), cp.Variable(), cp.Variable(S, nonneg=True)
    cvar = a + cp.sum(u) / ((1 - conf) * S)
    cons = [u >= -risk_scen @ w - a, cp.sum(w) == 1, w >= 0, w <= w_max]
    if min_ret is not None:
        cons.append(ret_scen.mean(axis=0) @ w >= min_ret)
    if turnover is not None and w0 is not None:
        cons.append(cp.norm1(w - w0) <= turnover)
    return w, cvar, cons


def _result(w_val, risk_scen, ret_scen, w0, status, conf) -> OptResult:
    w_val = np.clip(w_val, 0, None)
    w_val = w_val / w_val.sum()
    tv = float(np.abs(w_val - w0).sum()) if w0 is not None else 0.0
    return OptResult(w_val, ru_cvar(risk_scen @ w_val, conf), float(ret_scen.mean(0) @ w_val), tv, status)


def min_cvar(risk_scen: np.ndarray, ret_scen: np.ndarray | None = None, conf: float = C.CONF,
             w_max: float = W_MAX, min_ret: float | None = None, w0: np.ndarray | None = None,
             turnover: float | None = None) -> OptResult:
    """Minimise CVaR on `risk_scen`; the expected-return floor is measured on `ret_scen`
    (defaults to the same scenarios). If the return floor makes it infeasible, it is
    dropped and the status says so."""
    ret_scen = risk_scen if ret_scen is None else ret_scen
    w, cvar, cons = _problem(risk_scen, ret_scen, conf, w_max, min_ret, w0, turnover)
    prob = cp.Problem(cp.Minimize(cvar), cons)
    prob.solve(solver=cp.HIGHS)
    if w.value is None and min_ret is not None:
        res = min_cvar(risk_scen, ret_scen, conf, w_max, None, w0, turnover)
        res.status = "return floor infeasible; dropped"
        return res
    return _result(w.value, risk_scen, ret_scen, w0, prob.status, conf)


def minimal_change(risk_scen: np.ndarray, w0: np.ndarray, ret_scen: np.ndarray | None = None,
                   frac: float = 0.8, conf: float = C.CONF, w_max: float = W_MAX,
                   min_ret: float | None = None) -> tuple[OptResult, OptResult]:
    """Smallest turnover that captures `frac` of the achievable CVaR reduction.

    Step 1: the unconstrained optimum gives CVaR*. Step 2: minimise ||w - w0||_1
    subject to CVaR(w) <= CVaR(w0) - frac * (CVaR(w0) - CVaR*). The R-U expression
    is an upper bound on CVaR that is tight at the optimum, so constraining it is valid.
    Returns (minimal-change result, full optimum)."""
    ret_scen = risk_scen if ret_scen is None else ret_scen
    full = min_cvar(risk_scen, ret_scen, conf, w_max, min_ret)
    c0 = ru_cvar(risk_scen @ w0, conf)
    target = c0 - frac * (c0 - full.cvar)
    w, cvar, cons = _problem(risk_scen, ret_scen, conf, w_max, min_ret if "dropped" not in full.status else None,
                             None, None)
    prob = cp.Problem(cp.Minimize(cp.norm1(w - w0)), cons + [cvar <= target + 1e-9])
    prob.solve(solver=cp.HIGHS)
    if w.value is None:
        return full, full
    return _result(w.value, risk_scen, ret_scen, w0, prob.status, conf), full


def frontier(risk_scen: np.ndarray, ret_scen: np.ndarray, n_points: int = 12, conf: float = C.CONF,
             w_max: float = W_MAX) -> list[dict[str, float]]:
    """CVaR-vs-return frontier: sweep the return floor from the min-CVaR portfolio's
    return up to the highest return achievable under the weight caps."""
    lo = min_cvar(risk_scen, ret_scen, conf, w_max).exp_ret
    mu = ret_scen.mean(axis=0)
    # max return with caps = fill the best assets up to w_max in order (greedy is optimal here)
    w, left = np.zeros_like(mu), 1.0
    for i in np.argsort(-mu):
        w[i] = min(w_max, left)
        left -= w[i]
    hi = float(mu @ w)
    pts = []
    for r in np.linspace(lo, hi, n_points):
        res = min_cvar(risk_scen, ret_scen, conf, w_max, min_ret=r - 1e-9)
        pts.append({"exp_ret": res.exp_ret, "cvar": res.cvar})
    return pts
