"""Scenario-based risk measures. Returns are portfolio returns; VaR/CVaR are reported
as positive LOSS numbers (e.g. 0.05 = a 5% loss)."""
from __future__ import annotations

import numpy as np

from shocklab import config as C


def portfolio_returns(scen_log: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Asset log returns -> simple returns, then weight (simple returns aggregate linearly)."""
    return np.expm1(scen_log) @ w


def var(r: np.ndarray, conf: float = C.CONF) -> float:
    """VaR = loss threshold exceeded with probability 1 - conf = -quantile(r, 1 - conf)."""
    return float(-np.quantile(r, 1 - conf))


def cvar(r: np.ndarray, conf: float = C.CONF) -> float:
    """CVaR (expected shortfall) = average loss in the worst (1 - conf) of scenarios."""
    tail = r <= -var(r, conf)
    return float(-r[tail].mean())


def cvar_contributions(scen_simple: np.ndarray, w: np.ndarray, conf: float = C.CONF) -> np.ndarray:
    """Component CVaR: asset i's average loss w_i * (-r_i) inside the portfolio's tail.

    Because the portfolio loss is the sum of asset losses, these add up exactly to CVaR.
    """
    port = scen_simple @ w
    tail = port <= -var(port, conf)
    return -(scen_simple[tail] * w).mean(axis=0)
