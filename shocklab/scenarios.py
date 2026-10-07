"""Scenario generation for optimisation.

Stress scenarios: fix the factor moves to the user's shock and sample returns.
Base (unconditional) scenarios: we don't know next period's macro moves, so we
bootstrap 10-day factor moves from history (before the decision date) and let the
model map each one to asset returns. Regime features stay at today's values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from shocklab import config as C
from shocklab.baseline import LinearFactorModel
from shocklab.data import Panel
from shocklab.mdn import MDNModel


def bootstrap_shocks(history: np.ndarray, n: int, seed: int = C.SEED) -> np.ndarray:
    """Resample rows of past factor moves with replacement."""
    rng = np.random.default_rng(seed)
    return history[rng.integers(0, len(history), size=n)]


def shock_history(panel: Panel, date: pd.Timestamp) -> np.ndarray:
    """All 10-day factor moves fully observed by `date`."""
    return panel.X[C.FACTORS].to_numpy(float)[panel.known_before(date)]


def base_scenarios(model, shocks: np.ndarray, regime: np.ndarray, seed: int = C.SEED) -> np.ndarray:
    """(n, assets) 10-day LOG returns under bootstrapped factor moves."""
    if isinstance(model, MDNModel):
        return model.simulate_many(shocks, regime, seed)
    assert isinstance(model, LinearFactorModel)
    resid = model.simulate(np.zeros(len(C.FACTORS)), n=len(shocks), seed=seed) - model.alpha
    return model.alpha + shocks @ model.beta.T + resid
