# GOLD-PRICE-PREDICTOR
```bash
pip install -r requirements.txt
python model/train.py          # Train + log to MLflow
uvicorn api.main:app --reload  # Serve API
```

### 2. Docker
```bash
docker-compose up --build
```

### 3. API Endpoints
- `POST /predict`  — Get next-day gold direction signal
- `GET  /health`   — Health check
- `GET  /model/info` — Current model metadata

## Signal Output
```json
{
  "signal": "LONG",
  "probability": 0.73,
  "confidence": "HIGH",
  "regime": "BULLISH",
  "features_used": 34,
  "model_version": "lgbm-v1.2"
}
```

## Key Features Used
- **Macro**: DXY, US real yields (TIPS), VIX, CPI, Fed funds rate
- **Technical**: RSI, MACD, Bollinger Bands, ATR, EMA crossovers
- **Order Flow**: COT net positioning z-score, COT momentum, retail sentiment
- **Regime**: HMM-based market regime (bull/bear/neutral)

## Validation Methodology
- Walk-forward cross-validation (5 folds, no lookahead)
- Out-of-sample Sharpe ratio target: > 1.2
- Minimum 2-year OOS test period

