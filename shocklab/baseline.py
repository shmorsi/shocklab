"""Linear factor model + multivariate Student-t residuals (the benchmark).

    r_i = a_i + b_i . f + e_i,      e ~ multivariate t(0, S, nu)

Betas: OLS per asset on the trailing 3 years of 10-day windows, using only windows
that ENDED on or before the fit date. Residual tails: a shared Student-t degrees of
freedom (median of per-asset fits), scale matrix chosen so Cov(e) = sample covariance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from shocklab import config as C
from shocklab.data import Panel

MIN_COVERAGE = 0.95  # asset must have returns for 95% of the fit window, else masked


@dataclass
class LinearFactorModel:
    assets: list[str]
    alpha: np.ndarray  # (n,)
    beta: np.ndarray  # (n, k) sensitivities to each factor
    shape: np.ndarray  # (n, n) multivariate-t scale matrix S
    nu: float  # degrees of freedom
    fit_date: pd.Timestamp

    def mean(self, shock: np.ndarray) -> np.ndarray:
        """Expected 10d log return per asset given factor moves (t has mean 0 for nu > 1)."""
        return self.alpha + self.beta @ shock

    def simulate(self, shock: np.ndarray, n: int = 10_000, seed: int = C.SEED) -> np.ndarray:
        """Draw n scenarios: a multivariate t is a Gaussian divided by sqrt(chi2_nu / nu)."""
        rng = np.random.default_rng(seed)
        L = np.linalg.cholesky(self.shape)
        z = rng.standard_normal((n, len(self.assets))) @ L.T
        w = rng.chisquare(self.nu, size=(n, 1)) / self.nu
        return self.mean(shock) + z / np.sqrt(w)

    def logpdf(self, y: np.ndarray, shock: np.ndarray) -> float:
        """Joint log density of one realized return vector."""
        return float(stats.multivariate_t(self.mean(shock), self.shape, df=self.nu).logpdf(y))


def ols(X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Least squares with intercept: minimise ||y - a - Xb||^2 -> (a, b, residuals)."""
    A = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef[0], coef[1:], y - A @ coef


def fit_t_dof(resid: np.ndarray) -> float:
    """Shared tail parameter: median of per-asset Student-t MLE dof, clipped to [3, 30]."""
    dofs = [stats.t.fit(resid[:, j] / resid[:, j].std(), floc=0)[0] for j in range(resid.shape[1])]
    return float(np.clip(np.median(dofs), 3.0, 30.0))


def fit_before(panel: Panel, date: str | pd.Timestamp, years: int = 3) -> LinearFactorModel:
    """Fit on windows with start >= date - years and end <= date (no lookahead)."""
    date = pd.Timestamp(date)
    rows = panel.known_before(date) & (panel.Y.index >= date - pd.DateOffset(years=years))
    X = panel.X[C.FACTORS].to_numpy()[rows]
    Y = panel.Y[rows]
    assets = [a for a in Y.columns if Y[a].notna().mean() >= MIN_COVERAGE]
    Y = Y[assets].to_numpy()
    keep = ~np.isnan(Y).any(axis=1)  # common rows so the residual covariance is consistent
    X, Y = X[keep], Y[keep]
    if len(Y) < 100:
        raise ValueError(f"not enough history before {date.date()}")

    fits = [ols(X, Y[:, j]) for j in range(Y.shape[1])]
    alpha = np.array([f[0] for f in fits])
    beta = np.array([f[1] for f in fits])
    resid = np.column_stack([f[2] for f in fits])

    nu = fit_t_dof(resid)
    cov = np.cov(resid, rowvar=False)
    shape = cov * (nu - 2) / nu  # Cov of a t with scale S is S * nu / (nu - 2)
    return LinearFactorModel(assets, alpha, beta, shape, nu, date)
