"""API contract tests against the saved artifacts (skipped if they were not built)."""
import pytest
from fastapi.testclient import TestClient

from shocklab.stress import STATE_FILE, human_from_shock, shock_from_human

pytestmark = pytest.mark.skipif(not STATE_FILE.exists(), reason="run `make train` first")

EW = {a: 1.0 for a in ["SPY", "QQQ", "IWM", "EFA", "TLT", "IEF", "HYG", "GLD", "XLE", "RY.TO"]}


@pytest.fixture(scope="module")
def client():
    from api.main import app
    with TestClient(app) as c:
        yield c


def test_shock_unit_roundtrip():
    h = {"oil_pct": -30.0, "rates_bp": 50.0, "usd_pct": 3.0, "vix_pct": 100.0, "credit_bp": 80.0}
    back = human_from_shock(shock_from_human(h))
    assert all(abs(back[k] - v) < 1e-9 for k, v in h.items())


def test_stress(client):
    r = client.post("/stress", json={"weights": EW, "shock": {"oil_pct": -30, "vix_pct": 120, "credit_bp": 100}})
    assert r.status_code == 200
    d = r.json()
    for m in ("baseline", "mdn"):
        res = d["models"][m]
        assert res["cvar95"] >= res["var95"]
        assert 0.98 < sum(res["histogram"]) <= 1 + 1e-9  # far tails are outside the plotted range
        assert abs(sum(c["cvar_contrib"] for c in res["contributions"]) - res["cvar95"]) < 1e-6
        assert len(res["factor_attribution"]) == 7
    assert len(d["bin_edges"]) == len(d["models"]["mdn"]["histogram"]) + 1
    # a risk-off shock should hurt an equity-heavy portfolio under both models
    assert d["models"]["baseline"]["expected"] < 0 and d["models"]["mdn"]["expected"] < 0


def test_stress_is_deterministic(client):
    body = {"weights": EW, "shock": {"rates_bp": 100}}
    assert client.post("/stress", json=body).json() == client.post("/stress", json=body).json()


def test_bad_inputs(client):
    assert client.post("/stress", json={"weights": {"NOPE": 1}}).status_code == 422
    assert client.post("/stress", json={"weights": EW, "shock": {"oil_pct": 500}}).status_code == 422


def test_optimize(client):
    r = client.post("/optimize", json={"weights": EW, "shock": {"vix_pct": 150, "credit_bp": 120}})
    assert r.status_code == 200
    d = r.json()
    assert d["after"]["cvar95"] <= d["before"]["cvar95"] + 1e-9
    assert abs(sum(d["weights"].values()) - 1) < 1e-6 and max(d["weights"].values()) <= 0.15 + 1e-6
    assert len(d["frontier"]) == 10 and "from" in d["trades"][0]
    m = client.post("/optimize", json={"weights": EW, "shock": {"vix_pct": 150}, "mode": "minimal_change"}).json()
    assert m["turnover"] <= d["turnover"] + 1e-6


def test_precomputed(client):
    ev = client.get("/events").json()
    assert [e["name"] for e in ev][0] == "COVID crash" and len(ev[0]["path"]) == 11
    assert "metrics" in client.get("/backtest").json()
    assert len(client.get("/comparison").json()) > 0
    assert "assets" in client.get("/meta").json()
