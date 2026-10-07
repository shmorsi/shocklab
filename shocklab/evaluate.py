"""Honest out-of-sample comparison: MDN vs linear factor + Student-t baseline.

Each model is given the factor moves that actually happened over a window and must
produce a distribution for the 24 asset returns. We score:
  - NLL: -log p(realized vector), joint over all assets (lower = better)
  - VaR breach rate: share of windows where the realized return fell below -VaR95
    (target 5%); for assets (pooled) and the equal-weight portfolio, + Kupiec test
  - CVaR error: on portfolio breach windows, realized loss minus predicted CVaR
    (0 = tail severity right; >0 = losses worse than the model's tail average)
Test windows are non-overlapping (every 10th trading day from 2020-01-01).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from shocklab import baseline, risk
from shocklab import config as C
from shocklab.data import Panel
from shocklab.events import all_events
from shocklab.mdn import MDNModel, load_for_date

N_SIMS = 10_000


def eval_window_starts(panel: Panel, start: str = C.TEST_START) -> list[pd.Timestamp]:
    """Every 10th trading day, so windows (t, t+10] never overlap."""
    days = panel.prices.index[panel.prices.index >= pd.Timestamp(start)]
    return [t for t in days[:: C.HORIZON] if t in panel.Y.index]


@dataclass
class Prediction:
    logpdf: float
    var_assets: np.ndarray  # per-asset VaR95 (loss, simple return)
    port_var: float
    port_cvar: float
    port_mean: float


def predict(model, assets: list[str], shock, regime, realized: np.ndarray, w: np.ndarray) -> Prediction:
    """Score one model on one window. Baseline ignores the regime features."""
    idx = [model.assets.index(a) for a in assets]
    if isinstance(model, MDNModel):
        sims = model.simulate(shock, regime, N_SIMS)[:, idx]
        # marginal of a diagonal mixture over a subset = same mixture on those dims
        lp = _mdn_marginal_logpdf(model, realized, shock, regime, idx)
    else:
        sims = model.simulate(shock, N_SIMS)[:, idx]
        sub = baseline.LinearFactorModel(assets, model.alpha[idx], model.beta[idx],
                                         model.shape[np.ix_(idx, idx)], model.nu, model.fit_date)
        lp = sub.logpdf(realized, shock)
    simple = np.expm1(sims)
    port = simple @ w
    return Prediction(lp, -np.quantile(simple, 0.05, axis=0), risk.var(port), risk.cvar(port), float(port.mean()))


def _mdn_marginal_logpdf(m: MDNModel, y, shock, regime, idx) -> float:
    if len(idx) == len(m.assets):
        return m.logpdf(y, shock, regime)
    pi, mu, sd = m.mixture(shock, regime)
    log_n = stats.norm.logpdf(y, mu[:, idx], sd[:, idx]).sum(axis=1)
    return float(np.log(np.sum(pi * np.exp(log_n - log_n.max()))) + log_n.max())


def kupiec_pvalue(breaches: int, n: int, p: float = 1 - C.CONF) -> float:
    """Likelihood-ratio test that the breach probability equals p (chi2 with 1 dof)."""
    x = breaches
    phat = min(max(x / n, 1e-12), 1 - 1e-12)
    ll0 = (n - x) * np.log(1 - p) + x * np.log(p)
    ll1 = (n - x) * np.log(1 - phat) + x * np.log(phat)
    return float(1 - stats.chi2.cdf(-2 * (ll0 - ll1), df=1))


@dataclass
class Scorer:
    nll: list[float] = field(default_factory=list)
    asset_breach: list[np.ndarray] = field(default_factory=list)
    port_breach: list[bool] = field(default_factory=list)
    cvar_err: list[float] = field(default_factory=list)
    abs_err: list[float] = field(default_factory=list)

    def add(self, p: Prediction, realized_simple: np.ndarray, w: np.ndarray) -> None:
        port_real = float(realized_simple @ w)
        self.nll.append(-p.logpdf)
        self.asset_breach.append(realized_simple < -p.var_assets)
        breach = port_real < -p.port_var
        self.port_breach.append(breach)
        if breach:
            self.cvar_err.append(-port_real - p.port_cvar)
        self.abs_err.append(abs(port_real - p.port_mean))

    def summary(self) -> dict[str, float]:
        n, b = len(self.port_breach), int(np.sum(self.port_breach))
        return {
            "windows": n,
            "mean_nll": float(np.mean(self.nll)),
            "asset_var_breach_rate": float(np.mean(self.asset_breach)),
            "port_var_breach_rate": b / n,
            "port_breaches": b,
            "kupiec_p": kupiec_pvalue(b, n),
            "port_cvar_error": float(np.mean(self.cvar_err)) if self.cvar_err else float("nan"),
            "port_mean_abs_error": float(np.mean(self.abs_err)),
        }


def run(panel: Panel) -> tuple[pd.DataFrame, pd.DataFrame]:
    static = MDNModel.load(C.ARTIFACTS_DIR / "mdn_static.pt")
    cache: dict[int, MDNModel] = {}
    names = ["Baseline (OLS + t)", "MDN static (train<=2017)", "MDN walk-forward"]
    scorers = {n: Scorer() for n in names}

    for t in eval_window_starts(panel):
        realized_log = panel.Y.loc[t]
        base = baseline.fit_before(panel, t)
        assets = [a for a in base.assets if not np.isnan(realized_log[a])]
        w = np.full(len(assets), 1 / len(assets))
        shock, regime = panel.X.loc[t, C.FACTORS].to_numpy(float), panel.R.loc[t].to_numpy(float)
        y = realized_log[assets].to_numpy()
        for name, model in zip(names, [base, static, load_for_date(t, cache)]):
            scorers[name].add(predict(model, assets, shock, regime, y, w), np.expm1(y), w)
    test = pd.DataFrame([{"model": n, **s.summary()} for n, s in scorers.items()])

    rows = []
    for ev in all_events(panel):
        base = baseline.fit_before(panel, ev.t)
        assets = [a for a in base.assets if not np.isnan(ev.realized[a])]
        w = np.full(len(assets), 1 / len(assets))
        y = ev.realized[assets].to_numpy()
        port_real = float(np.expm1(y) @ w)
        for name, model in zip(names, [base, static, load_for_date(ev.t, cache)]):
            p = predict(model, assets, ev.shock, ev.regime, y, w)
            rows.append({
                "event": ev.name, "start": ev.t.date(), "model": name, "nll": -p.logpdf,
                "asset_breaches": int((np.expm1(y) < -p.var_assets).sum()), "n_assets": len(assets),
                "pred_port_mean": p.port_mean, "pred_var95": p.port_var, "pred_cvar95": p.port_cvar,
                "realized_port": port_real, "port_breach": port_real < -p.port_var,
                "cvar_error": -port_real - p.port_cvar,
            })
    return test, pd.DataFrame(rows)


def to_markdown(test: pd.DataFrame, events: pd.DataFrame) -> str:
    t = test.copy()
    lines = [
        "### Test period 2020+ (non-overlapping 10-day windows)", "",
        "| Model | Windows | Mean NLL ↓ | Asset VaR95 breach (target 5%) | Portfolio VaR95 breach | Kupiec p | Portfolio CVaR error | Portfolio MAE |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in t.iterrows():
        lines.append(f"| {r.model} | {r.windows} | {r.mean_nll:.2f} | {r.asset_var_breach_rate:.1%} | "
                     f"{r.port_var_breach_rate:.1%} ({r.port_breaches}) | {r.kupiec_p:.3f} | "
                     f"{r.port_cvar_error:+.2%} | {r.port_mean_abs_error:.2%} |")
    lines += ["", "### Historical events (equal-weight portfolio, realized factor moves given)", "",
              "| Event | Model | NLL ↓ | Asset breaches | Pred. mean | VaR95 | CVaR95 | Realized | Breach | CVaR error |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in events.iterrows():
        lines.append(f"| {r.event} | {r.model} | {r.nll:.1f} | {r.asset_breaches}/{r.n_assets} | "
                     f"{r.pred_port_mean:+.2%} | {r.pred_var95:.2%} | {r.pred_cvar95:.2%} | "
                     f"{r.realized_port:+.2%} | {'yes' if r.port_breach else 'no'} | {r.cvar_error:+.2%} |")
    lines += ["", "NLL is the joint negative log density of the 24-asset realized return vector "
              "(10-day log returns, raw units). CVaR error = realized loss − predicted CVaR, averaged "
              "over portfolio breach windows (test) or for the single window (events); positive means "
              "the loss was worse than the model's own tail average."]
    return "\n".join(lines)
