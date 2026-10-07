"""Out-of-sample MDN vs baseline comparison -> results/model_comparison.{csv,md}.

Usage: python -m scripts.evaluate
"""
from __future__ import annotations

import pandas as pd

from shocklab import config as C
from shocklab.data import load_panel
from shocklab.evaluate import run, to_markdown


def main() -> None:
    test, events = run(load_panel())
    out = pd.concat([test.assign(scope="test_2020+"), events.assign(scope="event")], ignore_index=True)
    out.to_csv(C.RESULTS_DIR / "model_comparison.csv", index=False)
    md = to_markdown(test, events)
    (C.RESULTS_DIR / "model_comparison.md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
