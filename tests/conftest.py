"""Synthetic panel so tests run offline and fast."""
import numpy as np
import pandas as pd
import pytest

from shocklab import config as C
from shocklab.data import Panel, build_panel


def make_synthetic(n_days: int = 1400, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Factor levels follow random walks; assets load linearly on daily factor moves + noise."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2012-01-02", periods=n_days)
    fd = rng.normal(0, [0.02, 5, 0.004, 0.05, 3], size=(n_days, 5))  # daily factor moves
    levels = pd.DataFrame(index=days)
    levels["oil"] = 60 * np.exp(np.cumsum(fd[:, 0]))
    levels["rates"] = 2 + np.cumsum(fd[:, 1]) / 100
    levels["usd"] = 100 * np.exp(np.cumsum(fd[:, 2]))
    levels["vix"] = 18 * np.exp(np.cumsum(fd[:, 3]) * 0.2)
    levels["credit"] = 2 + np.cumsum(fd[:, 4]) / 100
    levels["rf"] = 1.0
    n = len(C.ASSETS)
    B = rng.normal(0, 1, size=(n, 5)) * np.array([0.2, -0.0005, -0.5, -0.05, -0.0008])
    daily = fd @ B.T + rng.standard_t(5, size=(n_days, n)) * 0.008
    prices = pd.DataFrame(100 * np.exp(np.cumsum(daily, axis=0)), index=days, columns=C.ASSETS)
    prices.iloc[:300, C.ASSETS.index("JETS")] = np.nan  # a ticker with shorter history
    return prices, levels


@pytest.fixture(scope="session")
def panel() -> Panel:
    prices, levels = make_synthetic()
    return build_panel(prices, levels)
