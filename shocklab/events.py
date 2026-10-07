"""Historical event replays: fit before the event, feed in the factor moves that
actually happened, compare predicted vs realized 10-day returns."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from shocklab import config as C
from shocklab.data import Panel


@dataclass
class EventWindow:
    name: str
    t: pd.Timestamp  # window start (close of the event date)
    end: pd.Timestamp
    shock: np.ndarray  # realized factor moves over the window
    regime: np.ndarray  # regime features at t
    realized: pd.Series  # realized 10d log returns per asset


def event_window(panel: Panel, name: str, date: str) -> EventWindow:
    """Snap to the last US trading day on or before `date`."""
    t = panel.Y.index[panel.Y.index <= pd.Timestamp(date)][-1]
    return EventWindow(
        name, t, panel.end[t], panel.X.loc[t, C.FACTORS].to_numpy(float),
        panel.R.loc[t].to_numpy(float), panel.Y.loc[t],
    )


def all_events(panel: Panel) -> list[EventWindow]:
    return [event_window(panel, n, d) for n, d in C.EVENTS.items()]


def describe_shock(shock: np.ndarray) -> str:
    """Human units: % for log factors, bp for yields/spreads."""
    parts = []
    for f, v in zip(C.FACTORS, shock):
        parts.append(f"{f} {np.expm1(v):+.1%}" if f in C.LOG_FACTORS else f"{f} {v:+.0f}bp")
    return ", ".join(parts)
