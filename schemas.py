"""
api/schemas.py
--------------
Pydantic v2 request/response models for the prediction API.
Strong typing is critical in production financial systems — 
a wrong data type in a trade signal can cause real losses.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal
from datetime import date


# ─────────────────────────────────────────────
# REQUEST MODELS
# ─────────────────────────────────────────────

class PredictRequest(BaseModel):
    """
    Request model for /predict endpoint.
    Caller can provide a specific date or get live prediction.
    """
    date: Optional[str] = Field(
        default=None,
        description="Date for prediction (YYYY-MM-DD). Defaults to latest available.",
        examples=["2024-01-15"]
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Probability threshold for LONG signal (0.5 = balanced)",
    )

    @field_validator("date")
    @classmethod
    def validate_date(cls, v):
        if v is not None:
            try:
                date.fromisoformat(v)
            except ValueError:
                raise ValueError("Date must be in YYYY-MM-DD format")
        return v


class BatchPredictRequest(BaseModel):
    """Request for batch predictions over a date range."""
    start_date: str = Field(..., description="Start date YYYY-MM-DD")
    end_date: str   = Field(..., description="End date YYYY-MM-DD")
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


# ─────────────────────────────────────────────
# RESPONSE MODELS
# ─────────────────────────────────────────────

class PredictResponse(BaseModel):
    """
    Prediction response with signal, confidence, and metadata.
    Designed to be human-readable AND machine-parseable.
    """
    # Core signal
    signal: Literal["LONG", "SHORT", "NEUTRAL"] = Field(
        description="Trading signal direction"
    )
    probability: float = Field(
        description="Model probability of UP move (0-1)",
        ge=0.0, le=1.0
    )
    confidence: Literal["HIGH", "MEDIUM", "LOW"] = Field(
        description="Confidence tier based on probability magnitude"
    )

    # Market context
    regime: Literal["BULLISH", "BEARISH", "NEUTRAL"] = Field(
        description="Detected market regime from HMM"
    )
    real_yield_signal: Optional[str] = Field(
        default=None,
        description="Real yield environment (key gold driver)"
    )

    # Model metadata
    prediction_date: str      = Field(description="Date of prediction")
    model_version: str        = Field(description="Model version identifier")
    features_used: int        = Field(description="Number of features in model")
    threshold_used: float     = Field(description="Probability threshold applied")

    # Risk
    disclaimer: str = Field(
        default="For research purposes only. Not financial advice.",
        description="Required disclaimer"
    )


class BatchPredictResponse(BaseModel):
    predictions: list[dict]
    start_date: str
    end_date: str
    n_predictions: int
    win_rate: Optional[float] = None


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    model_loaded: bool
    model_version: str
    uptime_seconds: float
    last_prediction: Optional[str] = None


class ModelInfoResponse(BaseModel):
    model_version: str
    n_features: int
    feature_names: list[str]
    training_metrics: dict
    top_features: list[dict]
