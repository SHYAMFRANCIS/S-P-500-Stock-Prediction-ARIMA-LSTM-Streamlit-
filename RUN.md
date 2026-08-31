# Run & Deployment Guide — S&P 500 Stock Prediction

> Single source of truth for local run, tests, services and Docker deployment.

---

## 0. Prerequisites

| Requirement | Version | Check |
|---|---|---|
| Python | `>=3.12` | `python --version` |
| uv | latest | `uv --version` — install via `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"` |
| Git | any | `git --version` |
| Docker + Compose | optional for full stack | `docker --version && docker compose version` |

---

## 1. Install

```powershell
# clone (or already in place)
git clone https://github.com/<your-username>/sp500-stock-prediction.git
Set-Location "S&P 500 Stock Prediction"

# install deps + editable package (pyproject.toml uses hatchling, src/ layout)
uv sync

# verify package is importable (must print path)
uv run python -c "import sp500_stock_prediction; print(sp500_stock_prediction.__file__)"
```

> **Windows App Control note:** `optree` native DLL is blocked by policy. Project pins `dm-tree>=0.1.10` as Keras fallback. Do **not** force `optree` — tests will recurse. `uv sync` is enough.

---

## 2. Quick Smoke Test

```powershell
uv run python main.py
# -> Hello from sp500-stock-prediction!
```

---

## 3. Tests (Hermetic by default)

```powershell
uv run pytest -q
# 79 passed, 1 skipped (LSTM uses dm-tree fallback, no optree needed)

# verbose / single module
uv run pytest tests/test_arima_model.py -v
uv run pytest tests/test_lstm_model.py -v

# live-market integration (hits yfinance, gated)
RUN_NETWORK_TESTS=1 uv run pytest -q
```

Pytest config: `pyproject.toml` sets `pythonpath = ["src"]` so `uv run pytest` resolves `sp500_stock_prediction` without `PYTHONPATH`.

---

## 4. Environment Variables

Copy template and edit:

```powershell
Copy-Item .env.example .env
# then edit .env
```

| Variable | Default | Used by |
|---|---|---|
| `RETRAIN_THRESHOLD` | `0.05` | dashboard + API — relative close move that triggers retrain |
| `TICKERS` | `^GSPC,AAPL,MSFT` | API `realtime_pipeline` watchlist |
| `CACHE_DB` | `dashboard_cache.db` / `/app/data/...` | SQLite cache path |
| `MODEL_PATH` | `/app/models` (docker) / `./models` (local) | persisted `.keras` + `.json` scaler |
| `REPORTS_DIR` | `/app/reports` | HTML/Markdown exports |
| `REDIS_URL` | `redis://redis:6379/0` (docker) | API cache layer |
| `ALPHA_VANTAGE_API_KEY` | `` | optional alternative provider |
| `POLYGON_API_KEY` | `` | optional alternative provider |

---

## 5. Run — Streamlit Dashboard (`:8501`)

Six tabs: **Overview · EDA · ARIMA · LSTM · Compare · Live**

```powershell
# foreground (recommended for dev)
uv run streamlit run src/sp500_stock_prediction/dashboard.py --server.port 8501

# headless (no browser auto-open)
uv run streamlit run src/sp500_stock_prediction/dashboard.py --server.headless true --server.port 8501

# background (new window, keeps running after terminal closes)
Start-Process -NoNewWindow -FilePath "uv" -ArgumentList "run streamlit run src/sp500_stock_prediction/dashboard.py --server.port 8501"

# open
Start-Process http://localhost:8501
```

**Known Windows issue — `pyarrow` DLL blocked:**

`st.dataframe` / `st.table` both import `pyarrow.lib` which is blocked by App Control.  
Fix is already applied in `src/sp500_stock_prediction/dashboard.py:27-65`: wrapper falls back to `st.markdown(df.to_html(), unsafe_allow_html=True)`.  
If you see `DLL load failed while importing lib`, pull latest code — no action needed.

Sidebar: choose `Ticker` (default `^GSPC`) + `Period` (`1mo..5y`). Data cached 15 min (`@st.cache_data ttl=900`).

---

## 6. Run — FastAPI Real-Time API (`:8000`)

```powershell
# dev (auto-reload)
uv run uvicorn sp500_stock_prediction.realtime_pipeline:create_app --factory --host 0.0.0.0 --port 8000 --reload

# prod
uv run uvicorn sp500_stock_prediction.realtime_pipeline:create_app --factory --host 0.0.0.0 --port 8000

# health + predict
curl http://localhost:8000/health
curl http://localhost:8000/predict/AAPL
# {"ticker":"AAPL","predicted_price":212.35,"confidence_interval":[210.1,214.6],"model_used":"persistence_baseline","as_of_close":212.3,"timestamp":"..."}
curl http://localhost:8000/metrics
```

API endpoints: `GET /health` (liveness + timestamp), `GET /predict/{ticker}`, `GET /metrics` (retrain/cache/scheduler), `WS /ws/stream` (live snapshots).

---

## 7. Run — Both Services Locally (Two terminals)

```powershell
# terminal 1 — API
uv run uvicorn sp500_stock_prediction.realtime_pipeline:create_app --factory --host 0.0.0.0 --port 8000

# terminal 2 — Dashboard
uv run streamlit run src/sp500_stock_prediction/dashboard.py --server.port 8501
```

Dashboard Live tab polls local cache; API serves predictions independently.

---

## 8. Docker Deployment — Full Stack (Dashboard + API + Redis + Prometheus)

```powershell
Copy-Item .env.example .env   # optionally add keys
docker compose up --build
docker compose up --build -d  # detached
docker compose logs -f
docker compose ps
docker compose down
docker compose down -v        # also wipe volumes
```

| Service | URL | Container | Purpose |
|---|---|---|---|
| Dashboard | http://localhost:8501 | `sp500-app` | Streamlit UI |
| API | http://localhost:8000 | `sp500-api` | FastAPI + WebSocket |
| Prometheus | http://localhost:9090 | `sp500-prometheus` | scrapes `api:8000/metrics` every 30s (`docker/prometheus.yml`) |
| Redis | localhost:6379 | `sp500-redis` | cache layer, AOF enabled |

Volumes (persist across restarts): `app-data` (`/app/data`), `models` (`/app/models`), `reports` (`/app/reports`), `redis-data`, `prometheus-data`.

Docker details:
- Multi-stage `Dockerfile` on `python:3.12-slim` — builder exports `uv.lock` → wheels → runtime copies `/opt/venv`
- Runtime `PYTHONPATH=/app/src`, runs as non-root `appuser`, `HEALTHCHECK` on `/_stcore/health`
- API command overridden in `docker-compose.yml:39-46` to `uvicorn ... --port 8000`

Rebuild after dependency change:
```powershell
docker compose build --no-cache
docker compose up -d
```

---

## 9. Programmatic Usage

```python
from sp500_stock_prediction.data_collector import StockDataCollector
from sp500_stock_prediction.eda_engine import StockEDA
from sp500_stock_prediction.arima_model import ARIMAPredictor

collector = StockDataCollector(cache_db="cache.db")
df = collector.fetch_single("AAPL", period="2y")

eda = StockEDA(df=df, ticker="AAPL")
print(eda.adf_test()["is_stationary"])

model = ARIMAPredictor().fit(df["Close"])
forecast, ci = model.predict(steps=10)
print(forecast.tail())
```

---

## 10. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: sp500_stock_prediction` | editable install missing (pre-hatchling) | `uv sync` then `uv run python -c "import sp500_stock_prediction"` — `pyproject.toml` now declares `[build-system]` + `[tool.hatch.build.targets.wheel]` |
| `RecursionError` / `optree` DLL blocked | `optree._C` blocked by App Control | Already fixed: depends on `dm-tree` (`uv sync`), Keras auto-falls back to `dm_tree` |
| `ImportError: DLL load failed while importing lib` on `st.dataframe` | `pyarrow.lib` blocked | Already patched in `dashboard.py:27-65` → renders via `st.markdown(df.to_html())` |
| `Port 8501 is not available` | stale Streamlit process | `Get-Process python \| Stop-Process -Force` or `taskkill /PID <pid> /F`, then `netstat -ano \| Select-String 8501` |
| `No module named 'sp500_stock_prediction'` only in Streamlit | `PYTHONPATH` not set | Use `uv run streamlit ...` (editable install) — not `streamlit ...` directly |

---

## 11. Project Structure

```
src/sp500_stock_prediction/
  data_collector.py      # yfinance + SQLite cache + retry/backoff
  eda_engine.py          # ADF/KPSS/Jarque-Bera/Ljung-Box + indicators + HTML report
  arima_model.py         # Auto-ARIMA (AIC grid) + forecast CI + diagnostics
  lstm_model.py          # Stacked LSTM + early stopping + scaler persistence
  model_comparison.py    # MAE/MSE/RMSE/R²/MAPE + Diebold-Mariano + ensemble
  realtime_pipeline.py   # Scheduler + FastAPI factory + WebSocket
  dashboard.py           # 6-tab Streamlit app
tests/                   # hermetic suites (network mocked)
docker/prometheus.yml    # scrape api:8000/metrics
```

---

## 12. Useful One-Liners

```powershell
uv run pytest --collect-only                 # list tests
uv run ruff check .                          # lint (if configured)
uv run python -m sp500_stock_prediction.data_collector  # ad-hoc ingestion
docker compose config                        # validate compose file
docker logs sp500-app --tail 50
docker logs sp500-api --tail 50
curl http://localhost:8000/health | ConvertFrom-Json | Format-List
```
