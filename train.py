"""
model/train.py
--------------
LightGBM training pipeline with:
  - Walk-forward cross-validation (no lookahead bias)
  - MLflow experiment tracking
  - Feature importance analysis
  - Model persistence for API serving

IB Interview Note: Walk-forward validation is NON-NEGOTIABLE in finance.
Random train/test split causes catastrophic lookahead bias in time series.
Always be ready to explain this in interviews.
"""

import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score
)
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

LGBM_PARAMS = {
    "objective"       : "binary",
    "metric"          : "auc",
    "boosting_type"   : "gbdt",
    "num_leaves"      : 31,          # Keep low to prevent overfit on financial data
    "learning_rate"   : 0.05,
    "feature_fraction": 0.8,         # Feature subsampling
    "bagging_fraction": 0.8,         # Row subsampling
    "bagging_freq"    : 5,
    "min_child_samples": 20,         # Minimum data in a leaf — prevents overfit
    "lambda_l1"       : 0.1,         # L1 regularization
    "lambda_l2"       : 0.1,         # L2 regularization
    "verbose"         : -1,
    "random_state"    : 42,
}

NUM_BOOST_ROUND = 500
EARLY_STOPPING  = 50
MODEL_DIR       = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(MODEL_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# WALK-FORWARD VALIDATION
# ─────────────────────────────────────────────

def walk_forward_splits(
    n: int,
    n_folds: int = 5,
    min_train_size: float = 0.4,
    gap: int = 5  # days between train end and test start (avoids leakage)
) -> list[tuple]:
    """
    Generate walk-forward (expanding window) train/test splits.
    
    Each fold:
      - Training: all data up to cutoff
      - Gap: 5 days (avoids weekend/holiday leakage)
      - Test: next ~15% of data
    
    This is the ONLY correct validation method for financial time series.
    """
    splits = []
    test_size = int(n * (1 - min_train_size) / n_folds)

    for i in range(n_folds):
        train_end   = int(n * min_train_size) + i * test_size
        test_start  = train_end + gap
        test_end    = min(test_start + test_size, n)

        if test_end <= test_start:
            break

        splits.append((
            list(range(0, train_end)),
            list(range(test_start, test_end))
        ))

    return splits


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

def compute_metrics(y_true, y_pred_proba, threshold: float = 0.5) -> dict:
    """
    Compute classification metrics + financial metrics (Sharpe proxy).
    
    IB Note: AUC and Sharpe ratio are the two metrics that matter most.
    Accuracy alone is misleading — a model predicting all 1s gets ~52% accuracy.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    metrics = {
        "accuracy"  : accuracy_score(y_true, y_pred),
        "auc"       : roc_auc_score(y_true, y_pred_proba),
        "precision" : precision_score(y_true, y_pred, zero_division=0),
        "recall"    : recall_score(y_true, y_pred, zero_division=0),
        "f1"        : f1_score(y_true, y_pred, zero_division=0),
    }

    # Information Coefficient (IC) — standard quant metric
    # Measures rank correlation between predicted probability and actual return
    metrics["ic"] = float(np.corrcoef(y_pred_proba, y_true)[0, 1])

    return metrics


def compute_sharpe(returns: pd.Series, annualize: bool = True) -> float:
    """
    Annualized Sharpe ratio of strategy returns.
    Long when model predicts UP, flat/short otherwise.
    """
    if returns.std() == 0:
        return 0.0
    sharpe = returns.mean() / returns.std()
    if annualize:
        sharpe *= np.sqrt(252)
    return float(sharpe)


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

def train(
    X: pd.DataFrame,
    y: pd.Series,
    price_returns: pd.Series,  # actual daily returns for Sharpe calculation
    experiment_name: str = "xauusd_lgbm",
    run_name: str = None,
) -> dict:
    """
    Full training pipeline:
      1. Walk-forward cross-validation
      2. Final model fit on all data
      3. MLflow logging
      4. Model + feature list persistence
    
    Returns dict with model, metrics, feature importance.
    """
    run_name = run_name or f"lgbm-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):

        # ── Log hyperparameters ────────────────
        mlflow.log_params(LGBM_PARAMS)
        mlflow.log_param("n_features", X.shape[1])
        mlflow.log_param("n_samples",  X.shape[0])
        mlflow.log_param("date_start", str(X.index[0].date()) if hasattr(X.index[0], 'date') else str(X.index[0]))
        mlflow.log_param("date_end",   str(X.index[-1].date()) if hasattr(X.index[-1], 'date') else str(X.index[-1]))

        # ── Walk-forward CV ────────────────────
        logger.info("Running walk-forward cross-validation...")
        splits = walk_forward_splits(len(X), n_folds=5)

        fold_metrics = []
        fold_sharpes = []
        oof_proba    = np.zeros(len(X))

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test  = X.iloc[test_idx]
            y_test  = y.iloc[test_idx]
            ret_test = price_returns.iloc[test_idx]

            train_data = lgb.Dataset(X_train, label=y_train)
            val_data   = lgb.Dataset(X_test,  label=y_test, reference=train_data)

            callbacks = [
                lgb.early_stopping(EARLY_STOPPING, verbose=False),
                lgb.log_evaluation(period=-1)
            ]

            fold_model = lgb.train(
                LGBM_PARAMS,
                train_data,
                num_boost_round=NUM_BOOST_ROUND,
                valid_sets=[val_data],
                callbacks=callbacks,
            )

            proba = fold_model.predict(X_test)
            oof_proba[test_idx] = proba

            metrics = compute_metrics(y_test.values, proba)

            # Sharpe: go long when model predicts UP, flat when predicts DOWN
            strategy_returns = ret_test * (proba >= 0.5).astype(float)
            sharpe = compute_sharpe(strategy_returns)

            metrics["sharpe"] = sharpe
            fold_metrics.append(metrics)
            fold_sharpes.append(sharpe)

            logger.info(
                f"  Fold {fold_idx+1}/{len(splits)}: "
                f"AUC={metrics['auc']:.3f} | Acc={metrics['accuracy']:.3f} | Sharpe={sharpe:.2f}"
            )

        # ── Aggregate CV metrics ───────────────
        cv_metrics = {
            f"cv_{k}_mean": float(np.mean([m[k] for m in fold_metrics]))
            for k in fold_metrics[0].keys()
        }
        cv_metrics.update({
            f"cv_{k}_std": float(np.std([m[k] for m in fold_metrics]))
            for k in fold_metrics[0].keys()
        })

        logger.info(f"\nCV Results:")
        logger.info(f"  AUC:    {cv_metrics['cv_auc_mean']:.3f} ± {cv_metrics['cv_auc_std']:.3f}")
        logger.info(f"  Acc:    {cv_metrics['cv_accuracy_mean']:.3f} ± {cv_metrics['cv_accuracy_std']:.3f}")
        logger.info(f"  Sharpe: {cv_metrics['cv_sharpe_mean']:.2f} ± {cv_metrics['cv_sharpe_std']:.2f}")

        mlflow.log_metrics(cv_metrics)

        # ── Final model on full data ───────────
        logger.info("\nTraining final model on full dataset...")
        full_train = lgb.Dataset(X, label=y)

        final_model = lgb.train(
            LGBM_PARAMS,
            full_train,
            num_boost_round=int(NUM_BOOST_ROUND * 0.9),  # slightly fewer rounds on full data
        )

        # ── Feature importance ─────────────────
        importance_df = pd.DataFrame({
            "feature"   : X.columns,
            "importance": final_model.feature_importance(importance_type="gain"),
        }).sort_values("importance", ascending=False)

        logger.info(f"\nTop 10 Features:")
        for _, row in importance_df.head(10).iterrows():
            logger.info(f"  {row['feature']:35s} {row['importance']:>10.1f}")

        # ── Save artifacts ─────────────────────
        model_path   = os.path.join(MODEL_DIR, "model.lgb")
        features_path = os.path.join(MODEL_DIR, "features.json")
        metrics_path = os.path.join(MODEL_DIR, "metrics.json")

        final_model.save_model(model_path)

        feature_list = list(X.columns)
        with open(features_path, "w") as f:
            json.dump(feature_list, f)

        all_metrics = {**cv_metrics, "n_features": len(feature_list)}
        with open(metrics_path, "w") as f:
            json.dump(all_metrics, f, indent=2)

        # ── MLflow artifacts ───────────────────
        mlflow.log_artifact(model_path)
        mlflow.log_artifact(features_path)
        mlflow.log_artifact(metrics_path)

        importance_path = os.path.join(MODEL_DIR, "feature_importance.csv")
        importance_df.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path)

        run_id = mlflow.active_run().info.run_id
        logger.info(f"\nMLflow Run ID: {run_id}")
        logger.info(f"Artifacts saved to: {MODEL_DIR}")

        return {
            "model"      : final_model,
            "run_id"     : run_id,
            "metrics"    : cv_metrics,
            "importance" : importance_df,
            "feature_cols": feature_list,
        }


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    from data.ingestion import build_master_dataset
    from features.engineering import build_features

    logger.info("Loading data...")
    master = build_master_dataset(start="2015-01-01")

    logger.info("Engineering features...")
    X, feature_cols = build_features(master)

    y      = master["target"].loc[X.index]
    ret_1d = master["return_1d"].loc[X.index]

    results = train(X, y, ret_1d)

    print(f"\n{'='*50}")
    print(f"Training complete!")
    print(f"CV AUC:    {results['metrics']['cv_auc_mean']:.3f}")
    print(f"CV Sharpe: {results['metrics']['cv_sharpe_mean']:.2f}")
    print(f"Features:  {len(results['feature_cols'])}")
