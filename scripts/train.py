"""Train one walk-forward MDN per year (2018 .. next year) and save to artifacts/.

Model for year Y: train on windows ending <= Dec 31 of Y-3, early-stop on Y-2..Y-1
to choose the epoch count, then refit on everything up to Dec 31 of Y-1.
Also saves mdn_static.pt: the spec's split (train <=2017, early stop on 2018-2019,
no refit), evaluated untouched on 2020+. The newest walk-forward model (next
calendar year) uses all data and is what the API serves.

Usage: python -m scripts.train
"""
from __future__ import annotations

import pandas as pd

from shocklab import config as C
from shocklab.data import load_panel
from shocklab.mdn import model_path, train_mdn, walk_forward_splits


def main() -> None:
    panel = load_panel()
    C.ARTIFACTS_DIR.mkdir(exist_ok=True)
    last_year = panel.end.max().year + 1
    static = train_mdn(panel, C.TRAIN_END, C.VAL_END)
    static.save(C.ARTIFACTS_DIR / "mdn_static.pt")
    log = [{"year": "static", "train_end": C.TRAIN_END, "val_end": C.VAL_END,
            "epochs": static.epochs, "best_val_nll_std_units": round(static.best_val_nll, 3)}]
    print(log[-1])
    for year in range(2018, last_year + 1):
        train_end, val_end = walk_forward_splits(year)
        m = train_mdn(panel, train_end, val_end, refit=True)
        m.save(model_path(year))
        log.append({"year": year, "train_end": train_end, "val_end": val_end,
                    "epochs": m.epochs, "best_val_nll_std_units": round(m.best_val_nll, 3)})
        print(log[-1])
    pd.DataFrame(log).to_csv(C.RESULTS_DIR / "mdn_training_log.csv", index=False)


if __name__ == "__main__":
    main()
