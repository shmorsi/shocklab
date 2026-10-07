"""Request/response models. Shocks use human units; returns are decimals (0.05 = 5%)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Shock(BaseModel):
    oil_pct: float = Field(0.0, ge=-60, le=60, description="WTI move, %")
    rates_bp: float = Field(0.0, ge=-200, le=300, description="10Y Treasury yield change, bp")
    usd_pct: float = Field(0.0, ge=-10, le=10, description="Broad USD index move, %")
    vix_pct: float = Field(0.0, ge=-50, le=200, description="VIX move, %")
    credit_bp: float = Field(0.0, ge=-150, le=400, description="Baa-10Y credit spread change, bp")


class Regime(BaseModel):
    vix_level: float = Field(..., gt=5, lt=100)
    avg_corr: float = Field(..., gt=-0.2, lt=1)


class StressRequest(BaseModel):
    weights: dict[str, float] = Field(..., description="ticker -> weight (normalised server-side)")
    shock: Shock = Shock()
    regime: Regime | None = Field(None, description="defaults to the latest observed regime")


class Contribution(BaseModel):
    asset: str
    weight: float
    cvar_contrib: float
    expected_contrib: float


class FactorContribution(BaseModel):
    factor: str
    contribution: float


class ModelStress(BaseModel):
    expected: float
    var95: float
    cvar95: float
    biggest_drag: str
    histogram: list[float]
    contributions: list[Contribution]
    factor_attribution: list[FactorContribution]
    asset_factor: list[dict[str, float | str]]


class StressResponse(BaseModel):
    bin_edges: list[float]
    models: dict[Literal["baseline", "mdn"], ModelStress]
    shock: dict[str, float]
    regime: dict[str, float]


class OptimizeRequest(BaseModel):
    weights: dict[str, float]
    shock: Shock = Shock()
    model: Literal["mdn", "baseline"] = "mdn"
    mode: Literal["min_cvar", "minimal_change"] = "min_cvar"
    w_max: float = Field(0.15, gt=0.04, le=1.0)
    min_ret: float | None = Field(0.0, description="floor on expected 10-day return (unconditional scenarios)")
    turnover: float | None = Field(None, gt=0, le=2)


class Trade(BaseModel):
    asset: str
    from_: float = Field(..., alias="from")
    to: float
    delta: float


class PortfolioRisk(BaseModel):
    histogram: list[float]
    var95: float
    cvar95: float
    expected_stress: float
    expected_base: float


class OptimizeResponse(BaseModel):
    status: str
    mode: str
    model: str
    turnover: float
    weights: dict[str, float]
    trades: list[Trade]
    bin_edges: list[float]
    before: PortfolioRisk
    after: PortfolioRisk
    full_optimum_cvar95: float
    frontier: list[dict[str, float]]
