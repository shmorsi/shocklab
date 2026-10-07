# Shock Lab API. The image serves frozen models: artifacts/ (MDN weights, baseline
# params, regime) and results/ (precomputed replays, backtest, comparison).
# Rebuild after `make train backtest` to ship new numbers.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8000
WORKDIR /app

COPY requirements-api.txt .
# CPU-only torch keeps the image ~1 GB smaller than the default CUDA build
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-api.txt

COPY shocklab/ shocklab/
COPY api/ api/
COPY artifacts/ artifacts/
COPY results/ results/

EXPOSE 8000
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
