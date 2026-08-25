# 📈 S&P 500 Stock Prediction

An end-to-end machine learning platform for **S&P 500 market analysis and forecasting** — from raw market data ingestion through statistical and deep learning models to a production-style serving layer with an interactive dashboard.

Built as a modular pipeline where every stage is independently testable:

```
┌──────────────┐   ┌─────────────┐   ┌──────────┐   ┌──────────┐
│ Data         │ → │ Exploratory │ → │ ARIMA &  │ → │ Model    │
│ Ingestion    │   │ Analysis    │   │ LSTM     │   │ Ensemble │
└──────────────┘   └─────────────┘   └──────────┘   └────┬─────┘
                                                          │
      ┌───────────────────────────────┬──────────────────┴──┐
      ▼                               ▼                     ▼
┌──────────────┐              ┌──────────────┐      ┌──────────────┐
│ Real-Time    │              │ Streamlit    │      │ Docker       │
│ API (FastAPI)│              │ Dashboard    │      │ Deployment   │
└──────────────┘              └──────────────┘      └──────────────┘
```

---

## ✨ Features

- **Automated data ingestion** — S&P 500 universe fetching (Wikipedia scrape + static fallback), retry with exponential backoff, SQLite caching with TTL
- **Statistical EDA engine** — ADF/KPSS stationarity tests, Jarque-Bera normality, Ljung-Box autocorrelation, automated HTML reports
- **Technical indicators** — SMA(20/50), EMA(12/26), RSI(14) with Wilder smoothing, MACD(12/26/9), Bollinger Bands(20,2) — fully vectorized
- **Two forecasting paradigms** — Auto-ARIMA (AIC grid search with Hessian-inversion filtering) and a regularized stacked LSTM with early stopping
- **Model comparison** — MAE/MSE/RMSE/R²/MAPE side-by-side, one-sided Diebold-Mariano significance test, RMSE-based winner selection, weighted ensembling
- **Real-time serving** — FastAPI REST endpoints + WebSocket price streaming, scheduled ingestion via background scheduler, threshold-based retrain triggers
- **Interactive dashboard** — six-tab Streamlit app with live monitoring, alerts, and report export

## 🧱 Project Structure

| Module | File | Description |
|--------|------|-------------|
| 1 · Data Collection | `src/sp500_stock_prediction/data_collector.py` | yfinance ingestion, validation, SQLite cache |
| 2 · EDA Engine | `src/sp500_stock_prediction/eda_engine.py` | Statistical tests, indicators, HTML reports |
| 3 · ARIMA Model | `src/sp500_stock_prediction/arima_model.py` | Auto order selection, forecasts with CI, diagnostics |
| 4 · LSTM Model | `src/sp500_stock_prediction/lstm_model.py` | Sequence prep, Keras training, model persistence |
| 5 · Comparison | `src/sp500_stock_prediction/model_comparison.py` | Metrics, Diebold-Mariano test, ensemble |
| 6 · Realtime Pipeline | `src/sp500_stock_prediction/realtime_pipeline.py` | Scheduler, change detection, FastAPI/WebSocket |
| 7 · Dashboard | `src/sp500_stock_prediction/dashboard.py` | Six-tab Streamlit UI |

## 🚀 Quickstart

Requires **Python 3.12+** and [uv](https://docs.astral.sh/uv/).

```bash
# Clone and install
git clone https://github.com/<your-username>/sp500-stock-prediction.git
cd sp500-stock-prediction
uv sync

# Run the test suite (hermetic by default)
uv run pytest

# Include live-market integration tests
RUN_NETWORK_TESTS=1 uv run pytest

# Launch the dashboard
uv run streamlit run src/sp500_stock_prediction/dashboard.py
```

The dashboard opens at `http://localhost:8501`.

## 🖥️ Usage

### Streamlit Dashboard (`:8501`)

| Tab | What it does |
|-----|--------------|
| **Overview** | Latest price/change/volume, key statistics, SMA overlays |
| **EDA** | ADF/KPSS/Jarque-Bera results, return distribution, correlation heatmap, indicator chart, HTML report export |
| **ARIMA** | Auto or manual `(p,d,q)` selection, forecast with 95% confidence bands, residual diagnostics |
| **LSTM** | Configurable lookback/epochs, live training progress, predicted-vs-actual evaluation |
| **Compare** | Side-by-side metrics, combined forecast chart, 🏆 winner announcement, Markdown report download |
| **Live** | Refreshable price monitor with configurable alert threshold |

### REST API (`:8000`)

```bash
uvicorn sp500_stock_prediction.realtime_pipeline:create_app \
    --factory --host 0.0.0.0 --port 8000
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness probe with timestamp |
| `/predict/{ticker}` | GET | Point forecast + confidence interval + model used |
| `/metrics` | GET | Retrain events, cache state, scheduler status |
| `/ws/stream` | WebSocket | Live price snapshots |

```bash
curl http://localhost:8000/predict/AAPL
# {"ticker":"AAPL","predicted_price":212.35,"confidence_interval":[210.1,214.6],
#  "model_used":"persistence_baseline","as_of_close":212.3,"timestamp":"..."}
```

### Programmatic Example

```python
from sp500_stock_prediction.data_collector import StockDataCollector
from sp500_stock_prediction.eda_engine import StockEDA
from sp500_stock_prediction.arima_model import ARIMAPredictor

collector = StockDataCollector(cache_db="cache.db")
df = collector.fetch_single("AAPL", period="2y")

eda = StockEDA(df=df, ticker="AAPL")
print(eda.adf_test()["is_stationary"])       # False — prices are I(1)

model = ARIMAPredictor().fit(df["Close"])   # auto-selects (p,d,q) by AIC
forecast, ci_95 = model.predict(steps=10)
print(forecast.tail())
```

## 🐳 Docker Deployment

The full stack (dashboard, API, Redis, Prometheus) deploys with one command:

```bash
cp .env.example .env        # optionally add API keys
docker compose up --build
```

| Service | URL | Purpose |
|---------|-----|---------|
| Dashboard | `http://localhost:8501` | Streamlit UI |
| API | `http://localhost:8000` | FastAPI predictions |
| Prometheus | `http://localhost:9090` | Metrics scraping |
| Redis | `localhost:6379` | Cache layer |

SQLite caches, trained models, and generated reports persist in named volumes (`app-data`, `models`, `reports`). See [`Dockerfile`](Dockerfile) — multi-stage build on `python:3.12-slim` with a prebuilt virtualenv and non-root runtime user.

## ⚙️ Configuration

Environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `RETRAIN_THRESHOLD` | `0.05` | Relative close-price move that triggers retraining |
| `TICKERS` | `^GSPC,AAPL,MSFT` | Tickers served by the realtime pipeline |
| `MODEL_PATH` | `/app/models` | Directory for persisted `.pkl` / `.keras` models |
| `CACHE_DB` | per-service | SQLite cache location |
| `ALPHA_VANTAGE_API_KEY` | *(empty)* | Optional alternative data provider |
| `POLYGON_API_KEY` | *(empty)* | Optional alternative data provider |

## 🧪 Testing

The suite is **fully hermetic** — network calls are mocked, TensorFlow runs tiny networks, and the API is exercised in-process:

```bash
uv run pytest                          # 79 passed, ~15s
RUN_NETWORK_TESTS=1 uv run pytest      # adds live yfinance integration test
```

Every module ships with its own test file under `tests/`; the LSTM suite intentionally uses a small lookback/network so full-suite runs stay under a minute.

## 📦 Tech Stack

| Layer | Tools |
|-------|-------|
| Data | `yfinance`, `pandas`, `numpy`, SQLite |
| Statistics | `statsmodels`, `scipy` |
| Deep Learning | `tensorflow` / `keras` |
| ML Utilities | `scikit-learn` |
| Visualization | `plotly`, `matplotlib`, `seaborn` |
| Serving | `fastapi`, `uvicorn`, `websockets`, `schedule` |
| UI | `streamlit`, `jinja2` |
| Tooling | `uv`, `pytest`, Docker Compose |

## 🔒 Reproducibility

- Dependency versions locked via `uv.lock`
- Seeded RNGs across `python` / `numpy` / `tensorflow` (`LSTMPredictor.set_seeds`)
- Optional deterministic TF operations (`enable_op_determinism`)

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. It is **not financial advice**. Historical performance and backtested metrics do not guarantee future results — never trade real capital based solely on outputs from this system.

## 🗺️ Roadmap

- [ ] GitHub Actions CI (lint + hermetic test suite)
- [ ] Transformer-based forecaster benchmark
- [ ] Prometheus-native `/metrics` exposition (`prometheus-client`)
- [ ] Portfolio-level forecasting across the full S&P 500 universe
- [ ] PDF report export alongside HTML/Markdown

## 📄 License

License not yet selected — see [`LICENSE`](LICENSE). All rights reserved until then.
