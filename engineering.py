"""
features/engineering.py
------------------------
Builds the full feature matrix from the master dataset.
Three feature groups:
  1. Technical indicators (price/volume based)
  2. Macro features (already ingested, further transformed here)
  3. Order flow proxies (COT-based, retail sentiment)

IB Note: Feature engineering is where most alpha lives.
         Every feature must be economically motivated — be ready
         to explain WHY each feature should predict gold returns.
"""

import pandas as pd
import numpy as np
from ta import momentum, trend, volatility, volume
import logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. TECHNICAL FEATURES
# ─────────────────────────────────────────────

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standard technical indicators. All computed on close price.
    
    Economic motivation for gold:
    - RSI: overbought/oversold in a mean-reverting asset
    - ATR: volatility regime — gold spikes in crisis
    - BB width: volatility compression before breakout
    - MACD: medium-term trend strength
    """
    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    vol   = df["volume"]

    # ── Momentum ──────────────────────────────
    df["rsi_14"]    = momentum.RSIIndicator(close, window=14).rsi()
    df["rsi_28"]    = momentum.RSIIndicator(close, window=28).rsi()
    df["roc_5"]     = momentum.ROCIndicator(close, window=5).roc()
    df["roc_20"]    = momentum.ROCIndicator(close, window=20).roc()
    df["stoch_k"]   = momentum.StochasticOscillator(high, low, close).stoch()

    # ── Trend ──────────────────────────────────
    macd_obj        = trend.MACD(close)
    df["macd"]      = macd_obj.macd()
    df["macd_sig"]  = macd_obj.macd_signal()
    df["macd_diff"] = macd_obj.macd_diff()

    df["ema_10"]    = trend.EMAIndicator(close, window=10).ema_indicator()
    df["ema_50"]    = trend.EMAIndicator(close, window=50).ema_indicator()
    df["ema_200"]   = trend.EMAIndicator(close, window=200).ema_indicator()
    df["ema_cross"] = (df["ema_10"] - df["ema_50"]) / df["ema_50"]  # normalized crossover

    # ── Volatility ─────────────────────────────
    bb              = volatility.BollingerBands(close)
    df["bb_upper"]  = bb.bollinger_hband()
    df["bb_lower"]  = bb.bollinger_lband()
    df["bb_width"]  = (df["bb_upper"] - df["bb_lower"]) / close
    df["bb_pct"]    = bb.bollinger_pband()  # where price sits within bands

    df["atr_14"]    = volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    df["atr_norm"]  = df["atr_14"] / close  # normalized ATR

    # ── Volume ─────────────────────────────────
    df["obv"]       = volume.OnBalanceVolumeIndicator(close, vol).on_balance_volume()
    df["obv_ema"]   = df["obv"].ewm(span=20).mean()
    df["obv_diff"]  = (df["obv"] - df["obv_ema"]) / (df["obv_ema"].abs() + 1)

    # ── Price structure ────────────────────────
    df["ret_1d"]    = close.pct_change(1)
    df["ret_5d"]    = close.pct_change(5)
    df["ret_20d"]   = close.pct_change(20)
    df["ret_60d"]   = close.pct_change(60)

    # Realized volatility (rolling std of returns)
    df["rvol_10d"]  = df["ret_1d"].rolling(10).std() * np.sqrt(252)
    df["rvol_30d"]  = df["ret_1d"].rolling(30).std() * np.sqrt(252)

    # High-low range (intraday volatility proxy)
    df["hl_range"]  = (high - low) / close

    logger.info(f"Technical features added: {len([c for c in df.columns if c not in ['open','high','low','close','volume']])} total cols")
    return df


# ─────────────────────────────────────────────
# 2. MACRO FEATURES
# ─────────────────────────────────────────────

def add_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw macro series into model-ready features.
    
    Key economic relationships with gold:
    - Real yields ↑ → Gold ↓ (opportunity cost of holding gold)
    - DXY ↑ → Gold ↓ (gold priced in USD)
    - VIX ↑ → Gold ↑ (safe haven demand)
    - Breakeven inflation ↑ → Gold ↑ (inflation hedge)
    """

    # Real yield regime — most important single factor for gold
    if "real_yield" in df.columns:
        df["real_yield_level"]   = df["real_yield"]
        df["real_yield_negative"] = (df["real_yield"] < 0).astype(int)  # binary: negative real rates
        df["real_yield_regime"]  = pd.cut(
            df["real_yield"],
            bins=[-99, -0.5, 0, 0.5, 1.5, 99],
            labels=[0, 1, 2, 3, 4]
        ).astype(float)

    # DXY regime
    if "dxy" in df.columns:
        df["dxy_zscore"] = _rolling_zscore(df["dxy"], 252)
        df["dxy_above_200ma"] = (df["dxy"] > df["dxy"].rolling(200).mean()).astype(int)

    # VIX — regime and momentum
    if "vix" in df.columns:
        df["vix_level"]   = df["vix"]
        df["vix_spike"]   = (df["vix"] > 30).astype(int)  # crisis threshold
        df["vix_chg_5d"]  = df["vix"].diff(5)
        df["vix_zscore"]  = _rolling_zscore(df["vix"], 252)

    # Rate environment
    if "fed_funds" in df.columns and "breakeven" in df.columns:
        df["real_rate_gap"] = df["fed_funds"] - df["breakeven"]  # real vs inflation expectations

    # Inflation regime
    if "cpi_yoy" in df.columns:
        df["inflation_high"] = (df["cpi_yoy"] > 3.0).astype(int)

    logger.info("Macro features added")
    return df


# ─────────────────────────────────────────────
# 3. ORDER FLOW FEATURES (COT-based)
# ─────────────────────────────────────────────

def add_order_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Order flow proxies from COT data + derived signals.
    
    Economic motivation:
    - MM net z-score: measures hedge fund positioning extremes
      → When z-score > 2: crowded long, mean-reversion risk
      → When z-score < -2: max pessimism, contrarian buy signal
    
    - Commercial net: hedgers (producers) position opposite to price
      → Extreme commercial short = very bullish (they're hedging high prices)
    
    - COT momentum: trend in positioning (acceleration signal)
    
    Note: In production, supplement with:
      - Real-time delta/CVD from CME Globex
      - LMAX/EBS institutional flow data
      - Options skew and put/call ratios
    """

    # Core COT z-score signals
    if "mm_net_zscore" in df.columns:
        df["cot_mm_extreme_long"]  = (df["mm_net_zscore"] >  1.5).astype(int)
        df["cot_mm_extreme_short"] = (df["mm_net_zscore"] < -1.5).astype(int)
        df["cot_mm_signal"] = np.where(
            df["mm_net_zscore"] > 1.5, -1,   # contrarian: crowded = fade
            np.where(df["mm_net_zscore"] < -1.5, 1, 0)
        )

    # Commercial positioning (smart money hedge)
    if "comm_net_zscore" in df.columns:
        df["cot_comm_extreme"] = (df["comm_net_zscore"].abs() > 1.5).astype(int)

    # COT momentum (trend of positioning change)
    if "mm_net_chg_4w" in df.columns:
        df["cot_momentum_positive"] = (df["mm_net_chg_4w"] > 0).astype(int)
        df["cot_momentum_zscore"]   = _rolling_zscore(df["mm_net_chg_4w"], 52)

    # Retail sentiment proxy (contrarian)
    # In production: use OANDA/IG retail positioning data via API
    # Here we proxy with a lagged price signal (retail chases price)
    if "ret_20d" in df.columns:
        retail_long_proxy = df["ret_20d"].rolling(4).mean()  # retail follows 1m trend
        df["retail_sentiment_zscore"] = _rolling_zscore(retail_long_proxy, 52)
        df["retail_extreme_long"]  = (df["retail_sentiment_zscore"] >  1.5).astype(int)
        df["retail_extreme_short"] = (df["retail_sentiment_zscore"] < -1.5).astype(int)

    # CVD proxy (Cumulative Volume Delta)
    # True CVD requires tick data. We proxy using OBV divergence from price
    if "obv_diff" in df.columns and "ret_5d" in df.columns:
        df["cvd_price_divergence"] = np.sign(df["ret_5d"]) != np.sign(df["obv_diff"])
        df["cvd_price_divergence"] = df["cvd_price_divergence"].astype(int)

    logger.info("Order flow features added")
    return df


# ─────────────────────────────────────────────
# 4. REGIME DETECTION (HMM-based)
# ─────────────────────────────────────────────

def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hidden Markov Model regime detection.
    Identifies latent market states: bull / bear / high-vol crisis
    
    IB Interview: Be ready to explain why regime matters —
    feature importances shift dramatically across regimes.
    A model that ignores regime will overfit to the dominant regime.
    """
    try:
        from hmmlearn.hmm import GaussianHMM

        returns = df["ret_1d"].fillna(0).values.reshape(-1, 1)
        features_for_hmm = np.column_stack([
            returns,
            df["rvol_10d"].fillna(0).values
        ])

        model = GaussianHMM(n_components=3, covariance_type="full",
                            n_iter=200, random_state=42)
        model.fit(features_for_hmm)
        regimes = model.predict(features_for_hmm)

        # Label regimes by average return (0=bear, 1=neutral, 2=bull)
        regime_returns = [df["ret_1d"].values[regimes == r].mean() for r in range(3)]
        regime_map = {r: i for i, r in enumerate(np.argsort(regime_returns))}
        df["regime"] = [regime_map[r] for r in regimes]
        df["regime_bull"]     = (df["regime"] == 2).astype(int)
        df["regime_bear"]     = (df["regime"] == 0).astype(int)
        df["regime_change"]   = (df["regime"].diff() != 0).astype(int)
        logger.info("HMM regime detection complete (3 states: bear/neutral/bull)")

    except Exception as e:
        logger.warning(f"HMM regime detection failed: {e} — using fallback")
        df["regime"]       = 1
        df["regime_bull"]  = 0
        df["regime_bear"]  = 0
        df["regime_change"] = 0

    return df


# ─────────────────────────────────────────────
# 5. MASTER FEATURE BUILDER
# ─────────────────────────────────────────────

FEATURE_COLS = None  # set after first build — used for consistent inference

def build_features(df: pd.DataFrame, fit_regime: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """
    Run the full feature engineering pipeline.
    Returns (feature_df, feature_column_names).
    
    Args:
        df: master dataset from ingestion.build_master_dataset()
        fit_regime: if True, fit HMM on full data (training only)
    
    Returns:
        X: feature DataFrame
        feature_cols: list of feature column names (for consistent inference)
    """
    logger.info("Running feature engineering pipeline...")

    df = add_technical_features(df.copy())
    df = add_macro_features(df)
    df = add_order_flow_features(df)

    if fit_regime:
        df = add_regime_features(df)

    # Drop raw OHLCV + target from features
    drop_cols = ["open", "high", "low", "close", "volume", "target",
                 "return_1d", "ret_1d", "mm_long", "mm_short",
                 "comm_long", "comm_short"]
    feature_cols = [c for c in df.columns if c not in drop_cols
                    and not df[c].isnull().all()]

    X = df[feature_cols].copy()

    # Final clean: drop any remaining all-null columns, fill residual NaNs
    X = X.dropna(axis=1, how="all")
    X = X.ffill().bfill()
    feature_cols = list(X.columns)

    logger.info(f"Feature matrix: {X.shape[0]} rows × {X.shape[1]} features")
    return X, feature_cols


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=window // 2).mean()
    std  = series.rolling(window, min_periods=window // 2).std()
    return (series - mean) / (std + 1e-8)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    from data.ingestion import build_master_dataset
    master = build_master_dataset(start="2018-01-01")
    X, cols = build_features(master)
    print(f"\nFeature sample:\n{X.tail(3)}")
    print(f"\nTotal features: {len(cols)}")
    print(f"\nFeature groups:")
    print(f"  Technical:  {len([c for c in cols if any(k in c for k in ['rsi','macd','bb','ema','atr','obv','ret','rvol','hl','roc','stoch'])])}")
    print(f"  Macro:      {len([c for c in cols if any(k in c for k in ['dxy','vix','real','cpi','fed','breakeven','inflation'])])}")
    print(f"  Order flow: {len([c for c in cols if any(k in c for k in ['cot','mm','comm','retail','cvd'])])}")
    print(f"  Regime:     {len([c for c in cols if 'regime' in c])}")
