# Shock Lab. Python 3.11 venv in .venv; web/ uses npm.
PY := .venv/bin/python
.PHONY: setup data train backtest api web test screenshots all

setup:            ## create venv + install Python and web deps
	uv venv -p 3.11 .venv || python3.11 -m venv .venv
	uv pip install -p $(PY) -r requirements.txt || $(PY) -m pip install -r requirements.txt
	cd web && npm install

data:             ## download + cache prices and FRED factors to data/*.parquet
	$(PY) -m scripts.download_data

train:            ## walk-forward MDNs, then out-of-sample MDN vs baseline comparison + event replays
	$(PY) -m scripts.train
	$(PY) -m scripts.evaluate
	$(PY) -m scripts.replay > results/replay_baseline.txt

backtest:         ## monthly walk-forward backtest, then freeze API state
	$(PY) -m scripts.backtest
	$(PY) -m scripts.export_api_state

api:              ## FastAPI on :8000 (needs artifacts/ + results/ from train/backtest)
	.venv/bin/uvicorn api.main:app --reload --port 8000

web:              ## Next.js dashboard on :3000 (expects the API on :8000)
	cd web && npm run dev

test:             ## pytest + TypeScript typecheck + lint
	$(PY) -m pytest -q
	cd web && npx tsc --noEmit && npx eslint .

screenshots:      ## README screenshots (needs api + web running, Google Chrome installed)
	node docs/screenshots.mjs

all: data train backtest test
