import numpy as np
import pandas as pd
from scipy import stats

from shocklab import baseline, risk
from shocklab import config as C
from shocklab.data import align, build_panel
from tests.conftest import make_synthetic


def test_windows_are_forward_10d_log_returns(panel):
    t = panel.Y.index[100]
    i = panel.prices.index.get_loc(t)
    expected = np.log(panel.prices.iloc[i + 10] / panel.prices.iloc[i])
    assert np.allclose(panel.Y.loc[t], expected, equal_nan=True)
    assert panel.end[t] == panel.prices.index[i + 10]
    # rates factor is a bp change
    assert np.isclose(panel.X.loc[t, "rates"], 100 * (panel.fred["rates"].iloc[i + 10] - panel.fred["rates"].iloc[i]))


def test_no_lookahead_fit_ignores_future_data():
    """Scramble everything after the fit date: the fitted model must not change."""
    prices, levels = make_synthetic()
    cutoff = prices.index[1000]
    m1 = baseline.fit_before(build_panel(prices, levels), cutoff)
    rng = np.random.default_rng(1)
    p2, l2 = prices.copy(), levels.copy()
    after = prices.index > cutoff
    p2.loc[after] *= np.exp(rng.normal(0, 0.2, size=p2.loc[after].shape))
    l2.loc[after, "oil"] *= 3.0
    m2 = baseline.fit_before(build_panel(p2, l2), cutoff)
    assert m1.assets == m2.assets
    assert np.allclose(m1.beta, m2.beta) and np.allclose(m1.shape, m2.shape) and m1.nu == m2.nu


def test_fit_uses_only_windows_ended_before_date(panel):
    cutoff = panel.Y.index[800]
    rows = panel.known_before(cutoff)
    assert panel.end[rows].max() <= cutoff


def test_masks_short_history_assets(panel):
    early = baseline.fit_before(panel, panel.Y.index[400], years=1)
    assert "JETS" not in early.assets
    late = baseline.fit_before(panel, panel.Y.index[1300], years=1)
    assert "JETS" in late.assets


def test_baseline_recovers_betas(panel):
    m = baseline.fit_before(panel, panel.Y.index[-1], years=5)
    assert m.beta.shape == (len(m.assets), 5)
    # simulated mean should match the analytic mean
    s = np.array([0.1, 20, 0.0, 0.3, 10])
    sims = m.simulate(s, n=40_000)
    assert np.allclose(sims.mean(axis=0), m.mean(s), atol=3 * sims.std(axis=0).max() / 200)


def test_var_cvar_on_normal():
    r = np.random.default_rng(0).normal(0, 0.02, 1_000_000)
    z = stats.norm.ppf(0.95)
    assert abs(risk.var(r) - 0.02 * z) < 2e-4
    assert abs(risk.cvar(r) - 0.02 * stats.norm.pdf(z) / 0.05) < 2e-4


def test_cvar_contributions_sum_to_cvar():
    rng = np.random.default_rng(0)
    scen = rng.normal(0, 0.03, size=(50_000, 4))
    w = np.array([0.4, 0.3, 0.2, 0.1])
    assert np.isclose(risk.cvar_contributions(scen, w).sum(), risk.cvar(scen @ w))


def test_determinism(panel):
    m1 = baseline.fit_before(panel, panel.Y.index[1200])
    m2 = baseline.fit_before(panel, panel.Y.index[1200])
    s = np.zeros(5)
    assert np.array_equal(m1.simulate(s, 1000), m2.simulate(s, 1000))


def test_align_ffill_limit_and_negative_oil():
    days = pd.bdate_range("2020-01-01", periods=12)
    prices = pd.DataFrame({"SPY": np.arange(12.0) + 100}, index=days)
    fred = pd.DataFrame({f: 1.0 for f in C.FACTORS} | {"rf": 1.0}, index=days)
    fred.iloc[2:7, 1] = np.nan  # 5-day gap in rates: only 3 days may be filled
    fred.iloc[9, 0] = -37.0  # negative oil print -> missing -> filled from previous day
    _, f = align(prices, fred)
    assert f["rates"].iloc[2:5].notna().all()
    assert f["rates"].iloc[5:7].isna().all()
    assert f["oil"].iloc[9] == 1.0
