"""
api/predictor.py
----------------
Singleton model loader and inference engine.
Loaded once at startup, reused across all API requests.

IB Production Pattern:
- Model loaded into memory once (expensive)
- Feature pipeline re-run on latest data at each request
- Prediction cached for same date (avoid redundant computation)
- Model can be hot-swapped without restarting the API
"""

import os
import json
import time
import logging
import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime, date
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model", "artifacts")


class GoldPredictor:
    """
    Production inference engine for XAU/USD direction prediction.
    
    Follows the IB pattern of:
    1. Load model once at startup
    2. Accept raw date → fetch latest data → featurize → predict
    3. Return structured prediction with confidence and regime
    """

    def __init__(self):
        self.model          = None
        self.feature_cols   = None
        self.metrics        = {}
        self.model_version  = "not_loaded"
        self.load_time      = None
        self._last_pred_date = None
        self._last_pred      = None

    def load(self) -> bool:
        """Load model and feature list from disk. Returns True if successful."""
        model_path    = os.path.join(MODEL_DIR, "model.lgb")
        features_path = os.path.join(MODEL_DIR, "features.json")
        metrics_path  = os.path.join(MODEL_DIR, "metrics.json")

        if not os.path.exists(model_path):
            logger.error(f"Model not found at {model_path}. Run model/train.py first.")
            return False

        try:
            self.model = lgb.Booster(model_file=model_path)

            with open(features_path) as f:
                self.feature_cols = json.load(f)

            if os.path.exists(metrics_path):
                with open(metrics_path) as f:
                    self.metrics = json.load(f)

            self.model_version = f"lgbm-v{datetime.fromtimestamp(os.path.getmtime(model_path)).strftime('%Y%m%d')}"
            self.load_time     = time.time()

            logger.info(f"Model loaded: {self.model_version} | {len(self.feature_cols)} features")
            return True

        except Exception as e:
            logger.error(f"Model load failed: {e}")
            return False

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.feature_cols is not None

    def predict(self, target_date: Optional[str] = None, threshold: float = 0.5) -> dict:
        """
        Run end-to-end prediction for a given date.
        
        1. Fetches latest market data (or up to target_date)
        2. Runs feature engineering pipeline
        3. Aligns features to trained feature list
        4. Returns prediction with confidence and regime context
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call predictor.load() first.")

        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

        from data.ingestion import build_master_dataset
        from features.engineering import build_features

        # Fetch recent data (90 days for indicator warmup)
        end_date   = target_date or datetime.today().strftime("%Y-%m-%d")
        start_date = pd.Timestamp(end_date) - pd.Timedelta(days=365)
        start_str  = start_date.strftime("%Y-%m-%d")

        logger.info(f"Fetching data for prediction: {end_date}")
        master = build_master_dataset(start=start_str)
        X, _   = build_features(master, fit_regime=True)

        # Use latest available row (or closest to target_date)
        if target_date:
            available = X.index[X.index <= pd.Timestamp(target_date)]
            if len(available) == 0:
                raise ValueError(f"No data available for date {target_date}")
            pred_date = available[-1]
        else:
            pred_date = X.index[-1]

        row = X.loc[[pred_date]]

        # Align to trained feature list (fill missing with 0)
        for col in self.feature_cols:
            if col not in row.columns:
                row[col] = 0.0
        row = row[self.feature_cols]

        # Inference
        proba  = float(self.model.predict(row)[0])
        signal = self._proba_to_signal(proba, threshold)
        conf   = self._proba_to_confidence(proba, threshold)

        # Regime (from features)
        regime = self._detect_regime(row)
        real_yield_signal = self._real_yield_context(row)

        self._last_pred_date = str(pred_date.date())
        self._last_pred      = signal

        return {
            "signal"           : signal,
            "probability"      : round(proba, 4),
            "confidence"       : conf,
            "regime"           : regime,
            "real_yield_signal": real_yield_signal,
            "prediction_date"  : self._last_pred_date,
            "model_version"    : self.model_version,
            "features_used"    : len(self.feature_cols),
            "threshold_used"   : threshold,
            "disclaimer"       : "For research purposes only. Not financial advice.",
        }

    def predict_batch(self, start_date: str, end_date: str, threshold: float = 0.5) -> list[dict]:
        """Run predictions for a date range. Used for backtesting via API."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded.")

        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

        from data.ingestion import build_master_dataset
        from features.engineering import build_features

        # Need extra warmup data before start_date
        warmup_start = (pd.Timestamp(start_date) - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
        master = build_master_dataset(start=warmup_start)
        X, _   = build_features(master, fit_regime=True)

        # Filter to requested range
        mask = (X.index >= pd.Timestamp(start_date)) & (X.index <= pd.Timestamp(end_date))
        X_range = X[mask]

        if len(X_range) == 0:
            return []

        # Align features
        for col in self.feature_cols:
            if col not in X_range.columns:
                X_range[col] = 0.0
        X_aligned = X_range[self.feature_cols]

        probas  = self.model.predict(X_aligned)
        targets = master["target"].reindex(X_range.index)
        returns = master["return_1d"].reindex(X_range.index)

        results = []
        for i, (idx, proba) in enumerate(zip(X_range.index, probas)):
            signal = self._proba_to_signal(proba, threshold)
            results.append({
                "date"       : str(idx.date()),
                "signal"     : signal,
                "probability": round(float(proba), 4),
                "actual"     : int(targets.iloc[i]) if not pd.isna(targets.iloc[i]) else None,
                "return_1d"  : round(float(returns.iloc[i]), 5) if not pd.isna(returns.iloc[i]) else None,
                "correct"    : signal == ("LONG" if targets.iloc[i] == 1 else "SHORT")
                               if not pd.isna(targets.iloc[i]) else None,
            })

        return results

    def get_feature_importance(self) -> list[dict]:
        """Return top features by gain importance."""
        if not self.is_loaded:
            return []
        importance = self.model.feature_importance(importance_type="gain")
        pairs = sorted(zip(self.feature_cols, importance), key=lambda x: x[1], reverse=True)
        return [{"feature": f, "importance": round(float(v), 2)} for f, v in pairs[:20]]

    # ── Private helpers ──────────────────────────────────────

    @staticmethod
    def _proba_to_signal(proba: float, threshold: float) -> str:
        if proba >= threshold + 0.05:
            return "LONG"
        elif proba <= threshold - 0.05:
            return "SHORT"
        else:
            return "NEUTRAL"

    @staticmethod
    def _proba_to_confidence(proba: float, threshold: float) -> str:
        distance = abs(proba - threshold)
        if distance >= 0.2:
            return "HIGH"
        elif distance >= 0.1:
            return "MEDIUM"
        else:
            return "LOW"

    @staticmethod
    def _detect_regime(row: pd.DataFrame) -> str:
        if "regime_bull" in row.columns and row["regime_bull"].values[0] == 1:
            return "BULLISH"
        elif "regime_bear" in row.columns and row["regime_bear"].values[0] == 1:
            return "BEARISH"
        return "NEUTRAL"

    @staticmethod
    def _real_yield_context(row: pd.DataFrame) -> Optional[str]:
        if "real_yield" in row.columns:
            ry = float(row["real_yield"].values[0])
            if ry < -0.5:
                return f"NEGATIVE ({ry:.2f}%) → Supportive for gold"
            elif ry > 1.5:
                return f"HIGH ({ry:.2f}%) → Headwind for gold"
            else:
                return f"NEUTRAL ({ry:.2f}%)"
        return None


# Global singleton — loaded once, used by all API workers
predictor = GoldPredictor()
