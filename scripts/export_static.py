"""Export everything the browser-only (GitHub Pages) build needs into web/public/data/.

The static site runs both models in JavaScript, so this writes the baseline's
parameters (with the Cholesky factor precomputed), the MDN's weights and scalers,
the latest regime, the historical factor moves, and the precomputed replays,
backtest and comparison.

Usage: python -m scripts.export_static
"""
from __future__ import annotations

import json

import numpy as np
import torch

from shocklab import config as C
from shocklab.mdn import K, LOG_SIG_MAX, LOG_SIG_MIN
from shocklab.stress import Engine

OUT = C.ROOT / "web" / "public" / "data"


def r(a, d: int = 7):
    """Round floats so the JSON stays small without changing results meaningfully."""
    return np.round(np.asarray(a, dtype=float), d).tolist()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    e = Engine.load()
    b, m = e.baseline, e.mdn
    sd = {k: v.detach().numpy() for k, v in m.net.state_dict().items()}
    engine = {
        "factors": C.FACTORS, "log_factors": sorted(C.LOG_FACTORS),
        "baseline": {"assets": b.assets, "alpha": r(b.alpha), "beta": r(b.beta),
                     "chol": r(np.linalg.cholesky(b.shape), 9), "nu": float(b.nu)},
        "mdn": {"assets": m.assets, "k": K, "log_sig_min": LOG_SIG_MIN, "log_sig_max": LOG_SIG_MAX,
                "x_mean": r(m.xs.mean), "x_std": r(m.xs.std), "y_mean": r(m.ys.mean), "y_std": r(m.ys.std),
                "w1": r(sd["body.0.weight"]), "b1": r(sd["body.0.bias"]),
                "w2": r(sd["body.2.weight"]), "b2": r(sd["body.2.bias"]),
                "w3": r(sd["head.weight"]), "b3": r(sd["head.bias"])},
        "regime": r(e.regime), "shock_hist": r(e.shock_hist, 5),
    }
    (OUT / "engine.json").write_text(json.dumps(engine, separators=(",", ":")))
    meta = {**e.meta, "assets": e.assets, "regime": {"vix_level": float(e.regime[0]), "avg_corr": float(e.regime[1])}}
    (OUT / "meta.json").write_text(json.dumps(meta))
    for name in ("events", "backtest", "comparison"):
        (OUT / f"{name}.json").write_text((C.RESULTS_DIR / f"{name}.json").read_text())
    print("wrote", sorted(p.name for p in OUT.iterdir()), f"engine {(OUT / 'engine.json').stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
