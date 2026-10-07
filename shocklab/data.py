"""Download, align and cache prices + macro factors, then build 10-day windows.

Conventions (all indexed by the window START date t):
  Y[t] = log P[t+h] - log P[t]              asset 10-day log return
  X[t] = factor move over (t, t+h]           log change (oil/usd/vix) or bp change
  R[t] = regime features known AT t          VIX level, 60-day avg pairwise corr
  end[t] = date of t+h                       a window is "known" only once end[t] has passed
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from shocklab import config as C

FFILL_LIMIT = 3  # max consecutive missing days we are willing to forward-fill


def _cache(name: str) -> "pd.DataFrame | None":
    path = C.DATA_DIR / f"{name}.parquet"
    return pd.read_parquet(path) if path.exists() else None


def _save(df: pd.DataFrame, name: str) -> None:
    C.DATA_DIR.mkdir(exist_ok=True)
    df.to_parquet(C.DATA_DIR / f"{name}.parquet")


def download_prices(refresh: bool = False) -> pd.DataFrame:
    """Daily total-return closes (dividend/split adjusted) from Yahoo."""
    if not refresh and (df := _cache("prices_raw")) is not None:
        return df
    import yfinance as yf

    raw = yf.download(C.ASSETS, start=C.START, auto_adjust=True, progress=False)["Close"]
    raw.index = pd.DatetimeIndex(raw.index).tz_localize(None).normalize()
    raw = raw[C.ASSETS]
    _save(raw, "prices_raw")
    return raw


def download_factors(refresh: bool = False) -> pd.DataFrame:
    """Daily FRED series, renamed to short factor names (+ the T-bill rate)."""
    if not refresh and (df := _cache("fred_raw")) is not None:
        return df
    import pandas_datareader.data as web

    codes = list(C.FRED_SERIES) + [C.RF_SERIES]
    raw = web.DataReader(codes, "fred", C.START)
    raw = raw.rename(columns={**C.FRED_SERIES, C.RF_SERIES: "rf"})
    raw.index = pd.DatetimeIndex(raw.index).normalize()
    _save(raw, "fred_raw")
    return raw


def align(prices_raw: pd.DataFrame, fred_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Put everything on the US trading calendar (days SPY traded).

    - TSX names miss Canadian holidays -> forward-fill up to 3 days (zero return that day).
    - FRED gaps (holidays, missing prints) -> forward-fill up to 3 days.
    - WTI printed -$37 on 2020-04-20 (expiry squeeze); log changes need positive
      prices, so non-positive oil prints are treated as missing before filling.
    Leading NaNs (tickers that did not exist yet) are left as NaN = masked.
    """
    days = prices_raw["SPY"].dropna().index
    prices = prices_raw.reindex(prices_raw.index.union(days)).ffill(limit=FFILL_LIMIT).reindex(days)
    fred = fred_raw.copy()
    fred.loc[fred["oil"] <= 0, "oil"] = np.nan
    fred = fred.reindex(fred.index.union(days)).ffill(limit=FFILL_LIMIT).reindex(days)
    # Drop trailing days where the factors have not been published yet.
    last = fred[C.FACTORS].dropna().index.max()
    return prices.loc[:last], fred.loc[:last]


def avg_pairwise_corr(daily_ret: pd.DataFrame, window: int = 60) -> pd.Series:
    """Mean off-diagonal correlation of daily returns over the trailing window (uses data <= t)."""
    vals = daily_ret.to_numpy()
    out = np.full(len(vals), np.nan)
    for i in range(window - 1, len(vals)):
        block = vals[i - window + 1 : i + 1]
        cols = ~np.isnan(block).any(axis=0)  # only assets with a full window
        if cols.sum() < 3:
            continue
        c = np.corrcoef(block[:, cols], rowvar=False)
        n = c.shape[0]
        out[i] = (c.sum() - n) / (n * (n - 1))
    return pd.Series(out, index=daily_ret.index, name="avg_corr")


@dataclass
class Panel:
    prices: pd.DataFrame  # daily closes, US calendar
    fred: pd.DataFrame  # daily factor levels + rf
    Y: pd.DataFrame  # 10d forward asset log returns, index = window start t
    X: pd.DataFrame  # 10d forward factor moves
    R: pd.DataFrame  # regime features at t
    end: pd.Series  # window end date for each t

    def known_before(self, date: str | pd.Timestamp) -> np.ndarray:
        """Boolean mask of windows fully observed on or before `date` (no lookahead)."""
        return (self.end <= pd.Timestamp(date)).to_numpy()


def factor_moves(levels: pd.DataFrame, h: int) -> pd.DataFrame:
    """Forward h-day factor moves: log change for oil/usd/vix, bp change for yields/spreads."""
    out = {}
    for f in C.FACTORS:
        s = levels[f]
        if f in C.LOG_FACTORS:
            out[f] = np.log(s.shift(-h)) - np.log(s)
        else:
            out[f] = (s.shift(-h) - s) * 100.0  # percent -> basis points
    return pd.DataFrame(out)


def build_panel(prices: pd.DataFrame, fred: pd.DataFrame, h: int = C.HORIZON) -> Panel:
    logp = np.log(prices)
    Y = logp.shift(-h) - logp
    X = factor_moves(fred, h)
    daily = logp.diff()
    R = pd.DataFrame({"vix_level": fred["vix"], "avg_corr": avg_pairwise_corr(daily)})
    end = pd.Series(prices.index, index=prices.index).shift(-h)
    # Keep windows whose factors, regime and end date are all known.
    ok = X.notna().all(axis=1) & R.notna().all(axis=1) & end.notna()
    return Panel(prices, fred, Y[ok], X[ok], R[ok], end[ok])


def load_panel(refresh: bool = False) -> Panel:
    prices, fred = align(download_prices(refresh), download_factors(refresh))
    return build_panel(prices, fred)
