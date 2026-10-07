"""Stress engine behind the API: loads saved models once, answers stress/optimise queries.

Shocks arrive in human units (oil %, rates bp, USD %, VIX %, credit bp) and are
converted to model units (log changes for oil/USD/VIX, bp for rates/credit).
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from shocklab import config as C
from shocklab import risk
from shocklab.baseline import LinearFactorModel
from shocklab.mdn import MDNModel
from shocklab.optimize import frontier, min_cvar, minimal_change, ru_cvar
from shocklab.scenarios import base_scenarios, bootstrap_shocks

STATE_FILE = C.ARTIFACTS_DIR / "api_state.npz"
META_FILE = C.ARTIFACTS_DIR / "api_meta.json"
N_STRESS = 10_000
N_OPT = 4_000
N_BINS = 60
HUMAN_KEYS = {"oil": "oil_pct", "rates": "rates_bp", "usd": "usd_pct", "vix": "vix_pct", "credit": "credit_bp"}


def shock_from_human(h: dict[str, float]) -> np.ndarray:
    """% moves -> log changes; bp stay bp."""
    out = []
    for f in C.FACTORS:
        v = float(h.get(HUMAN_KEYS[f], 0.0))
        out.append(np.log1p(v / 100) if f in C.LOG_FACTORS else v)
    return np.array(out)


def human_from_shock(s: np.ndarray) -> dict[str, float]:
    return {HUMAN_KEYS[f]: float(np.expm1(v) * 100 if f in C.LOG_FACTORS else v) for f, v in zip(C.FACTORS, s)}


def histogram(samples: dict[str, np.ndarray], bins: int = N_BINS) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Shared bin edges (0.2%-99.8% of all samples) so models are comparable on one axis.
    Densities are fractions of ALL samples; the far tails beyond the edges are not drawn
    (clipping them into the edge bins would create fake spikes)."""
    pooled = np.concatenate(list(samples.values()))
    lo, hi = np.quantile(pooled, [0.002, 0.998])
    edges = np.linspace(lo, hi, bins + 1)
    return edges, {k: np.histogram(v, edges)[0] / len(v) for k, v in samples.items()}


@dataclass
class Engine:
    baseline: LinearFactorModel
    mdn: MDNModel
    regime: np.ndarray  # latest [vix_level, avg_corr]
    shock_hist: np.ndarray  # past 10-day factor moves, for unconditional scenarios
    meta: dict

    @property
    def assets(self) -> list[str]:
        return [a for a in self.mdn.assets if a in self.baseline.assets]

    @staticmethod
    def load() -> "Engine":
        z = np.load(STATE_FILE, allow_pickle=False)
        meta = json.loads(META_FILE.read_text())
        base = LinearFactorModel(meta["baseline_assets"], z["alpha"], z["beta"], z["shape"], float(z["nu"]),
                                 meta["as_of"])
        mdn = MDNModel.load(C.ARTIFACTS_DIR / meta["mdn_file"])
        return Engine(base, mdn, z["regime"], z["shock_hist"], meta)

    # ---------- scenario helpers ----------
    def _idx(self, model, assets):
        return [model.assets.index(a) for a in assets]

    def stress_scenarios(self, name: str, shock: np.ndarray, regime: np.ndarray, assets: list[str], n: int) -> np.ndarray:
        """(n, assets) simple 10-day returns with the factor moves fixed at `shock`."""
        if name == "mdn":
            sims = self.mdn.simulate(shock, regime, n)[:, self._idx(self.mdn, assets)]
        else:
            sims = self.baseline.simulate(shock, n)[:, self._idx(self.baseline, assets)]
        return np.expm1(sims)

    def base_scen(self, name: str, regime: np.ndarray, assets: list[str], n: int) -> np.ndarray:
        model = self.mdn if name == "mdn" else self.baseline
        shocks = bootstrap_shocks(self.shock_hist, n)
        return np.expm1(base_scenarios(model, shocks, regime)[:, self._idx(model, assets)])

    def mean_log(self, name: str, shock: np.ndarray, regime: np.ndarray, assets: list[str]) -> np.ndarray:
        if name == "mdn":
            return self.mdn.mean(shock, regime)[self._idx(self.mdn, assets)]
        return self.baseline.mean(shock)[self._idx(self.baseline, assets)]

    def factor_attribution(self, name: str, shock, regime, assets, w) -> tuple[np.ndarray, float, float]:
        """Contribution of each factor to the expected portfolio (log) return.

        Baseline: exactly w_i * beta_if * shock_f (the model is linear).
        MDN: one-at-a-time — move only factor f, measure the change vs no shock;
        whatever the MDN does non-additively shows up as an 'interaction' term."""
        zero = np.zeros(len(C.FACTORS))
        base_mean = self.mean_log(name, zero, regime, assets)
        total = self.mean_log(name, shock, regime, assets)
        mat = np.zeros((len(assets), len(C.FACTORS)))
        for j in range(len(C.FACTORS)):
            if name == "baseline":
                mat[:, j] = self.baseline.beta[self._idx(self.baseline, assets), j] * shock[j]
            else:
                one = zero.copy()
                one[j] = shock[j]
                mat[:, j] = self.mean_log(name, one, regime, assets) - base_mean
        mat *= w[:, None]
        interaction = float(w @ (total - base_mean) - mat.sum())
        return mat, float(w @ base_mean), interaction

    # ---------- endpoints ----------
    def stress(self, weights: dict[str, float], human_shock: dict[str, float], regime: np.ndarray | None = None) -> dict:
        assets, w = self.portfolio(weights)
        shock = shock_from_human(human_shock)
        regime = self.regime if regime is None else regime
        port, out = {}, {}
        for name in ("baseline", "mdn"):
            scen = self.stress_scenarios(name, shock, regime, assets, N_STRESS)
            p = scen @ w
            port[name] = p
            contrib = risk.cvar_contributions(scen, w)
            mat, carry, inter = self.factor_attribution(name, shock, regime, assets, w)
            out[name] = {
                "expected": float(p.mean()), "var95": risk.var(p), "cvar95": risk.cvar(p),
                "contributions": [
                    {"asset": a, "weight": float(wi), "cvar_contrib": float(c), "expected_contrib": float(wi * m)}
                    for a, wi, c, m in zip(assets, w, contrib, scen.mean(axis=0))
                ],
                "factor_attribution": [{"factor": f, "contribution": float(mat[:, j].sum())} for j, f in enumerate(C.FACTORS)]
                + [{"factor": "carry", "contribution": carry}, {"factor": "interaction", "contribution": inter}],
                "asset_factor": [{"asset": a, **{f: float(mat[i, j]) for j, f in enumerate(C.FACTORS)}}
                                 for i, a in enumerate(assets)],
            }
            out[name]["biggest_drag"] = max(out[name]["contributions"], key=lambda c: c["cvar_contrib"])["asset"]
        edges, dens = histogram(port)
        for name in out:
            out[name]["histogram"] = dens[name].tolist()
        return {"bin_edges": edges.tolist(), "models": out, "shock": human_from_shock(shock),
                "regime": {"vix_level": float(regime[0]), "avg_corr": float(regime[1])}}

    def optimize(self, weights: dict[str, float], human_shock: dict[str, float], model: str = "mdn",
                 mode: str = "min_cvar", w_max: float = 0.15, min_ret: float | None = 0.0,
                 turnover: float | None = None) -> dict:
        """Hedge the stress, keep normal-times return: CVaR is measured on stressed
        scenarios, the return floor on unconditional (bootstrapped-macro) scenarios."""
        held, w_held = self.portfolio(weights)
        assets = self.assets  # the optimiser may buy anything in the universe
        w0 = np.array([dict(zip(held, w_held)).get(a, 0.0) for a in assets])
        shock = shock_from_human(human_shock)
        risk_s = self.stress_scenarios(model, shock, self.regime, assets, N_OPT)
        ret_s = self.base_scen(model, self.regime, assets, N_OPT)
        if mode == "minimal_change":
            res, full = minimal_change(risk_s, w0, ret_s, w_max=w_max, min_ret=min_ret)
        else:
            res = full = min_cvar(risk_s, ret_s, w_max=w_max, min_ret=min_ret, w0=w0, turnover=turnover)
        before, after = risk_s @ w0, risk_s @ res.w
        edges, dens = histogram({"before": before, "after": after})
        trades = sorted(
            [{"asset": a, "from": float(x), "to": float(y), "delta": float(y - x)}
             for a, x, y in zip(assets, w0, res.w) if abs(y - x) > 5e-4],
            key=lambda t: -abs(t["delta"]))
        return {
            "status": res.status, "mode": mode, "model": model, "turnover": res.turnover,
            "weights": {a: float(x) for a, x in zip(assets, res.w) if x > 5e-4},
            "trades": trades, "bin_edges": edges.tolist(),
            "before": {"histogram": dens["before"].tolist(), "var95": risk.var(before), "cvar95": ru_cvar(before),
                       "expected_stress": float(before.mean()), "expected_base": float(ret_s.mean(0) @ w0)},
            "after": {"histogram": dens["after"].tolist(), "var95": risk.var(after), "cvar95": ru_cvar(after),
                      "expected_stress": float(after.mean()), "expected_base": res.exp_ret},
            "full_optimum_cvar95": full.cvar,
            "frontier": frontier(risk_s, ret_s, n_points=10, w_max=w_max),
        }

    def portfolio(self, weights: dict[str, float]) -> tuple[list[str], np.ndarray]:
        unknown = set(weights) - set(self.assets)
        if unknown:
            raise ValueError(f"unknown tickers: {sorted(unknown)}")
        assets = [a for a in self.assets if weights.get(a, 0) > 0]
        if not assets:
            raise ValueError("portfolio has no positive weights")
        w = np.array([weights[a] for a in assets], dtype=float)
        return assets, w / w.sum()
