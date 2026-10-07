"""Shock Lab API. Models are loaded once at startup from artifacts/; replays,
backtest and model comparison are precomputed JSON in results/.

Run: uvicorn api.main:app --reload --port 8000
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import OptimizeRequest, OptimizeResponse, StressRequest, StressResponse
from shocklab import config as C
from shocklab.stress import Engine

STATE: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    STATE["engine"] = Engine.load()
    for name in ("events", "backtest", "comparison"):
        STATE[name] = json.loads((C.RESULTS_DIR / f"{name}.json").read_text())
    yield
    STATE.clear()


app = FastAPI(title="Shock Lab", version="1.0", lifespan=lifespan)
origins = ["http://localhost:3000", "http://127.0.0.1:3000"] + [
    o for o in os.environ.get("CORS_ORIGINS", "").split(",") if o
]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])


def engine() -> Engine:
    return STATE["engine"]


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/meta")
def meta() -> dict:
    e = engine()
    return {**e.meta, "assets": e.assets,
            "regime": {"vix_level": float(e.regime[0]), "avg_corr": float(e.regime[1])}}


@app.post("/stress", response_model=StressResponse)
def stress(req: StressRequest) -> dict:
    regime = None if req.regime is None else np.array([req.regime.vix_level, req.regime.avg_corr])
    try:
        return engine().stress(req.weights, req.shock.model_dump(), regime)
    except ValueError as err:
        raise HTTPException(422, str(err)) from err


@app.post("/optimize", response_model=OptimizeResponse, response_model_by_alias=True)
def optimize(req: OptimizeRequest) -> dict:
    try:
        return engine().optimize(req.weights, req.shock.model_dump(), req.model, req.mode,
                                 req.w_max, req.min_ret, req.turnover)
    except ValueError as err:
        raise HTTPException(422, str(err)) from err


@app.get("/events")
def events() -> list[dict]:
    return STATE["events"]


@app.get("/backtest")
def backtest() -> dict:
    return STATE["backtest"]


@app.get("/comparison")
def comparison() -> list[dict]:
    return STATE["comparison"]
