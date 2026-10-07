"""Single place for universe, factor and date settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
ARTIFACTS_DIR = ROOT / "artifacts"

START = "2007-01-01"
HORIZON = 10  # trading days
SEED = 7

ASSETS: list[str] = [
    "SPY", "QQQ", "IWM", "EFA", "EEM", "XLE", "XLF", "XLK", "XLU", "XLY", "XLV",
    "XLI", "XLB", "TLT", "IEF", "HYG", "GLD", "USO", "JETS",
    "XIU.TO", "RY.TO", "CNQ.TO", "CP.TO", "SU.TO",
]

# FRED series -> short factor name.
# NOTE: FRED truncated the ICE BofA HY OAS (BAMLH0A0HYM2) to the last 3 years, so
# the credit factor uses Moody's Baa minus 10Y Treasury (BAA10Y), daily since 1986.
FRED_SERIES: dict[str, str] = {
    "DCOILWTICO": "oil",
    "DGS10": "rates",
    "DTWEXBGS": "usd",
    "VIXCLS": "vix",
    "BAA10Y": "credit",
}
FACTORS: list[str] = ["oil", "rates", "usd", "vix", "credit"]
LOG_FACTORS = {"oil", "usd", "vix"}  # log changes; the others are bp level changes
RF_SERIES = "DTB3"  # 3-month T-bill, only used for Sharpe ratios

# Date split (by the END of each 10-day window, so no label leaks across a boundary).
TRAIN_END = "2017-12-31"
VAL_END = "2019-12-31"
TEST_START = "2020-01-01"

CONF = 0.95  # VaR / CVaR level

# Historical events: (name, start date). Window = next 10 US trading days.
EVENTS: dict[str, str] = {
    "COVID crash": "2020-02-21",
    "Ukraine invasion": "2022-02-23",
    "2022 rate shock": "2022-06-03",
    "SVB": "2023-03-08",
    "Yen carry unwind": "2024-07-31",
}
