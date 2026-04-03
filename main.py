"""
api/main.py
-----------
FastAPI application exposing the XAU/USD prediction model as a REST API.

Endpoints:
  GET  /health          — Liveness + readiness check (used by K8s probes)
  GET  /model/info      — Model metadata and top features
  POST /predict         — Single prediction
  POST /predict/batch   — Batch predictions for date range

IB Production Notes:
  - Authentication would be added via OAuth2/API keys in production
  - Rate limiting via slowapi or cloud gateway
  - Async endpoints for I/O-bound operations (data fetching)
  - Structured logging → Splunk/Datadog in production
"""

import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    PredictRequest, PredictResponse,
    BatchPredictRequest, BatchPredictResponse,
    HealthResponse, ModelInfoResponse,
)
from api.predictor import predictor

# ─────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

START_TIME = time.time()


# ─────────────────────────────────────────────
# LIFESPAN (startup / shutdown)
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load model on startup. In production:
    - Pull model from S3/GCS artifact store (not local disk)
    - Validate model schema against current feature pipeline
    - Set up telemetry / tracing (OpenTelemetry)
    """
    logger.info("Starting XAU/USD ML Signal API...")
    success = predictor.load()
    if not success:
        logger.warning(
            "Model not loaded — API will start but /predict will return 503. "
            "Run `python model/train.py` to train the model first."
        )
    else:
        logger.info(f"Model ready: {predictor.model_version}")

    yield  # API is running

    logger.info("Shutting down XAU/USD ML Signal API...")


# ─────────────────────────────────────────────
# APP DEFINITION
# ─────────────────────────────────────────────

app = FastAPI(
    title="XAU/USD ML Signal API",
    description="""
## Gold Price Direction Prediction API

Production-grade ML pipeline for XAU/USD (Gold/USD) next-day direction signals.

### Features
- **LightGBM model** trained on 34+ features
- **Macro factors**: Real yields, DXY, VIX, CPI, Fed funds rate
- **Technical indicators**: RSI, MACD, Bollinger Bands, ATR, EMA crossovers
- **Order flow proxies**: COT positioning z-scores, CVD divergence
- **Regime detection**: HMM-based market state classification

### Validation
- Walk-forward cross-validation (5 folds)
- OOS Sharpe ratio tracked via MLflow
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — in production, restrict to internal services only
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# MIDDLEWARE — Request logging
# ─────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every request with timing. In production → structured logs to Splunk."""
    start = time.time()
    response = await call_next(request)
    duration_ms = (time.time() - start) * 1000
    logger.info(
        f"{request.method} {request.url.path} "
        f"→ {response.status_code} ({duration_ms:.1f}ms)"
    )
    return response


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Infrastructure"])
async def health():
    """
    Liveness + readiness probe.
    Returns 200 if healthy, 503 if model not loaded.
    Used by Kubernetes, load balancers, and monitoring systems.
    """
    uptime = time.time() - START_TIME
    status = "healthy" if predictor.is_loaded else "degraded"

    response = HealthResponse(
        status=status,
        model_loaded=predictor.is_loaded,
        model_version=predictor.model_version,
        uptime_seconds=round(uptime, 1),
        last_prediction=predictor._last_pred_date,
    )

    if not predictor.is_loaded:
        return JSONResponse(status_code=503, content=response.model_dump())

    return response


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    """
    Model metadata: version, features, training metrics, top features.
    Useful for model governance / audit trails (required at IBs).
    """
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return ModelInfoResponse(
        model_version=predictor.model_version,
        n_features=len(predictor.feature_cols),
        feature_names=predictor.feature_cols,
        training_metrics=predictor.metrics,
        top_features=predictor.get_feature_importance(),
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(request: PredictRequest):
    """
    Generate next-day XAU/USD direction signal.

    - **signal**: LONG | SHORT | NEUTRAL
    - **probability**: raw model output (0-1)
    - **confidence**: HIGH | MEDIUM | LOW based on probability magnitude
    - **regime**: detected market regime (BULLISH | BEARISH | NEUTRAL)

    Example request:
    ```json
    {"date": null, "threshold": 0.5}
    ```
    """
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run model/train.py first."
        )

    try:
        result = predictor.predict(
            target_date=request.date,
            threshold=request.threshold,
        )
        return PredictResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictRequest):
    """
    Batch predictions for a date range.
    Returns signal, probability, actual outcome, and correctness for each date.
    Useful for backtesting and performance attribution.
    """
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        predictions = predictor.predict_batch(
            start_date=request.start_date,
            end_date=request.end_date,
            threshold=request.threshold,
        )

        # Compute win rate on predictions with known outcomes
        correct = [p for p in predictions if p.get("correct") is True]
        known   = [p for p in predictions if p.get("correct") is not None]
        win_rate = len(correct) / len(known) if known else None

        return BatchPredictResponse(
            predictions=predictions,
            start_date=request.start_date,
            end_date=request.end_date,
            n_predictions=len(predictions),
            win_rate=round(win_rate, 4) if win_rate else None,
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", tags=["Infrastructure"])
async def root():
    return {
        "service": "XAU/USD ML Signal API",
        "version": "1.0.0",
        "docs"   : "/docs",
        "health" : "/health",
    }
