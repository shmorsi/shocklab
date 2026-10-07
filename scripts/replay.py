"""Replay historical events with the linear baseline: fit before the event date,
apply the realized factor moves, compare predicted vs realized 10-day returns.

Usage: python -m scripts.replay
"""
from __future__ import annotations

import numpy as np

from shocklab import baseline, risk
from shocklab.data import load_panel
from shocklab.events import all_events, describe_shock


def main() -> None:
    panel = load_panel()
    for ev in all_events(panel):
        m = baseline.fit_before(panel, ev.t)
        sims = m.simulate(ev.shock)
        lo, hi = np.quantile(sims, [0.05, 0.95], axis=0)
        realized = ev.realized[m.assets].to_numpy()
        print(f"\n=== {ev.name}: {ev.t.date()} -> {ev.end.date()} (fit {m.fit_date.date()}, t dof {m.nu:.1f})")
        print(f"realized shock: {describe_shock(ev.shock)}")
        print(f"{'asset':8s} {'pred':>8s} {'5%':>8s} {'95%':>8s} {'real':>8s}  in-band")
        for i, a in enumerate(m.assets):
            inside = lo[i] <= realized[i] <= hi[i]
            print(f"{a:8s} {np.expm1(m.mean(ev.shock)[i]):8.2%} {np.expm1(lo[i]):8.2%} "
                  f"{np.expm1(hi[i]):8.2%} {np.expm1(realized[i]):8.2%}  {'yes' if inside else 'NO'}")
        w = np.full(len(m.assets), 1 / len(m.assets))
        port = risk.portfolio_returns(sims, w)
        real_port = float(np.expm1(realized) @ w)
        print(f"{'EQ-WT':8s} {port.mean():8.2%} {np.quantile(port, .05):8.2%} {np.quantile(port, .95):8.2%} "
              f"{real_port:8.2%}  VaR95 {risk.var(port):.2%}  CVaR95 {risk.cvar(port):.2%}")


if __name__ == "__main__":
    main()
