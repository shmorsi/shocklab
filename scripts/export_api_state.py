"""Freeze everything the API needs into artifacts/ and results/ (no data/ at runtime).

- artifacts/api_state.npz + api_meta.json: baseline fitted on all data, latest regime,
  historical factor moves (for unconditional scenarios), which MDN file to serve.
- results/events.json: event replays for both models incl. a day-by-day path.
- results/backtest.json, results/comparison.json: for the dashboard.

Usage: python -m scripts.export_api_state
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from shocklab import baseline, risk
from shocklab import config as C
from shocklab.data import load_panel
from shocklab.events import all_events
from shocklab.mdn import load_for_date, model_path
from shocklab.stress import META_FILE, STATE_FILE, histogram, human_from_shock

N = 10_000


def band(p: np.ndarray, days: int = C.HORIZON) -> list[dict[str, float]]:
    """Day-d band from the 10-day distribution: centre scales with d/10, spread with
    sqrt(d/10) (random-walk approximation; the models only predict the 10-day horizon)."""
    m = p.mean()
    q05, q25, q75, q95 = np.quantile(p, [0.05, 0.25, 0.75, 0.95])
    out = []
    for d in range(days + 1):
        c, s = m * d / days, np.sqrt(d / days)
        out.append({"day": d, "mean": c, "q05": c + (q05 - m) * s, "q25": c + (q25 - m) * s,
                    "q75": c + (q75 - m) * s, "q95": c + (q95 - m) * s})
    return out


def events_json(panel) -> list[dict]:
    out, cache = [], {}
    for ev in all_events(panel):
        base = baseline.fit_before(panel, ev.t)
        mdn = load_for_date(ev.t, cache)
        assets = [a for a in base.assets if not np.isnan(ev.realized[a])]
        w = np.full(len(assets), 1 / len(assets))
        sims = {
            "baseline": np.expm1(base.simulate(ev.shock, N)[:, [base.assets.index(a) for a in assets]]),
            "mdn": np.expm1(mdn.simulate(ev.shock, ev.regime, N)[:, [mdn.assets.index(a) for a in assets]]),
        }
        port = {k: v @ w for k, v in sims.items()}
        edges, dens = histogram(port)
        i0 = panel.prices.index.get_loc(ev.t)
        px = panel.prices[assets].iloc[i0 : i0 + C.HORIZON + 1]
        path = (px / px.iloc[0]).mean(axis=1) - 1  # buy-and-hold equal weight
        realized_simple = np.expm1(ev.realized[assets].to_numpy())
        models = {}
        for k, s in sims.items():
            lo, hi = np.quantile(s, [0.05, 0.95], axis=0)
            models[k] = {
                "expected": float(port[k].mean()), "var95": risk.var(port[k]), "cvar95": risk.cvar(port[k]),
                "histogram": dens[k].tolist(), "band": band(port[k]),
                "assets": [{"asset": a, "pred": float(s[:, i].mean()), "lo": float(lo[i]), "hi": float(hi[i]),
                            "realized": float(realized_simple[i])} for i, a in enumerate(assets)],
            }
        out.append({
            "name": ev.name, "start": str(ev.t.date()), "end": str(ev.end.date()),
            "shock": human_from_shock(ev.shock),
            "regime": {"vix_level": float(ev.regime[0]), "avg_corr": float(ev.regime[1])},
            "realized_port": float(realized_simple @ w), "bin_edges": edges.tolist(),
            "path": [{"day": d, "date": str(dt.date()), "value": float(v)} for d, (dt, v) in enumerate(path.items())],
            "models": models,
        })
    return out


def backtest_json() -> dict:
    eq = pd.read_csv(C.RESULTS_DIR / "backtest_equity.csv", index_col=0, parse_dates=True)
    weekly = eq.resample("W-FRI").last().dropna(how="all")
    return {
        "metrics": pd.read_csv(C.RESULTS_DIR / "backtest_metrics.csv").to_dict(orient="records"),
        "dates": [str(d.date()) for d in weekly.index],
        "equity": {c: weekly[c].round(5).tolist() for c in weekly.columns},
        "last_weights": json.loads((C.RESULTS_DIR / "backtest_last_weights.json").read_text()),
    }


def main() -> None:
    panel = load_panel()
    as_of = panel.end.max()
    base = baseline.fit_before(panel, as_of)
    latest_year = max(int(p.stem.split("_")[1]) for p in C.ARTIFACTS_DIR.glob("mdn_20*.pt"))
    regime = panel.regime_all.iloc[-1].to_numpy(float)
    shock_hist = panel.X[C.FACTORS].to_numpy(float)
    np.savez_compressed(STATE_FILE, alpha=base.alpha, beta=base.beta, shape=base.shape, nu=base.nu,
                        regime=regime, shock_hist=shock_hist)
    hist_q = {f: [float(x) for x in np.quantile(panel.X[f], [0.01, 0.99])] for f in C.FACTORS}
    META_FILE.write_text(json.dumps({
        "as_of": str(as_of.date()), "regime_date": str(panel.regime_all.index[-1].date()),
        "baseline_assets": base.assets, "baseline_nu": base.nu,
        "mdn_file": model_path(latest_year).name, "mdn_trained_through": str(min(as_of, pd.Timestamp(f"{latest_year - 1}-12-31")).date()),
        "factors": C.FACTORS, "factor_1_99_pct": hist_q,
        "credit_series": "BAA10Y (Moody's Baa minus 10Y Treasury)",
    }, indent=1))
    (C.RESULTS_DIR / "events.json").write_text(json.dumps(events_json(panel)))
    (C.RESULTS_DIR / "backtest.json").write_text(json.dumps(backtest_json()))
    comp = pd.read_csv(C.RESULTS_DIR / "model_comparison.csv")
    (C.RESULTS_DIR / "comparison.json").write_text(comp.to_json(orient="records"))
    print("exported; as of", as_of.date(), "serving", model_path(latest_year).name)


if __name__ == "__main__":
    main()
