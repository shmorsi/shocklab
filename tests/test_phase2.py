import numpy as np
import pandas as pd
import torch
from scipy import stats

from shocklab.data import build_panel
from shocklab.evaluate import kupiec_pvalue, eval_window_starts
from shocklab.mdn import mdn_nll, train_mdn
from tests.conftest import make_synthetic

TRAIN_END, VAL_END = "2015-06-30", "2016-06-30"


def test_mdn_nll_matches_scipy_mixture():
    rng = np.random.default_rng(0)
    k, d = 3, 4
    logits, mu, log_sig = rng.normal(size=k), rng.normal(size=(k, d)), rng.normal(0, 0.3, size=(k, d))
    y = rng.normal(size=d)
    pi = np.exp(logits) / np.exp(logits).sum()
    dens = sum(pi[j] * np.prod(stats.norm.pdf(y, mu[j], np.exp(log_sig[j]))) for j in range(k))
    T = lambda a: torch.tensor(a, dtype=torch.float64)[None]
    got = mdn_nll(torch.log_softmax(T(logits), 1), T(mu), T(log_sig), T(y), torch.ones(1, d, dtype=torch.float64))
    assert np.isclose(got.item(), -np.log(dens))


def test_masking_equals_dropping_dimension():
    rng = np.random.default_rng(1)
    T = lambda a: torch.tensor(a, dtype=torch.float64)[None]
    log_pi = torch.log_softmax(T(rng.normal(size=2)), 1)
    mu, ls, y = rng.normal(size=(2, 3)), rng.normal(0, .2, size=(2, 3)), rng.normal(size=3)
    mask = np.array([1.0, 0.0, 1.0])
    full = mdn_nll(log_pi, T(mu), T(ls), T(y), T(mask))
    dropped = mdn_nll(log_pi, T(mu[:, [0, 2]]), T(ls[:, [0, 2]]), T(y[[0, 2]]), torch.ones(1, 2, dtype=torch.float64))
    assert np.isclose(full.item(), dropped.item())


def test_mdn_training_is_deterministic_and_raw_logpdf_consistent(panel):
    m1 = train_mdn(panel, TRAIN_END, VAL_END, max_epochs=3)
    m2 = train_mdn(panel, TRAIN_END, VAL_END, max_epochs=3)
    for a, b in zip(m1.net.parameters(), m2.net.parameters()):
        assert torch.equal(a, b)
    t = panel.Y.index[-1]
    s, r, y = panel.X.loc[t].to_numpy(float), panel.R.loc[t].to_numpy(float), panel.Y.loc[t].to_numpy()
    pi, mu, sd = m1.mixture(s, r)
    direct = np.log(sum(pi[j] * np.prod(stats.norm.pdf(y, mu[j], sd[j])) for j in range(len(pi))))
    assert np.isclose(m1.logpdf(y, s, r), direct, rtol=1e-4)
    assert np.array_equal(m1.simulate(s, r, 500), m1.simulate(s, r, 500))


def test_mdn_no_lookahead():
    """Scramble everything after val_end: the trained network must be identical."""
    prices, levels = make_synthetic()
    m1 = train_mdn(build_panel(prices, levels), TRAIN_END, VAL_END, max_epochs=3)
    after = prices.index > pd.Timestamp(VAL_END)
    p2 = prices.copy()
    p2.loc[after] *= 1.5 + np.random.default_rng(3).random(p2.loc[after].shape)
    m2 = train_mdn(build_panel(p2, levels), TRAIN_END, VAL_END, max_epochs=3)
    for a, b in zip(m1.net.parameters(), m2.net.parameters()):
        assert torch.equal(a, b)


def test_test_windows_do_not_overlap(panel):
    starts = eval_window_starts(panel, "2014-01-01")
    pos = panel.prices.index.get_indexer(starts)
    assert (np.diff(pos) >= 10).all()


def test_kupiec():
    assert kupiec_pvalue(5, 100) > 0.99  # exactly 5% -> cannot reject
    assert kupiec_pvalue(20, 100) < 0.001
