"""
model/trainer.py
Treinamento, validação temporal e persistência do modelo preditivo.

Usa TimeSeriesSplit para garantir que o modelo nunca "veja o futuro"
durante a validação cruzada — essencial para dados esportivos.
"""

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from config.settings import (
    CV_N_SPLITS,
    MODEL_TYPE,
    MODELS_DIR,
    RANDOM_FOREST_PARAMS,
    XGBOOST_PARAMS,
)
from model.features import FEATURE_COLS, TARGET_COL

logger = logging.getLogger(__name__)

MODEL_PATH   = MODELS_DIR / "cartola_model.pkl"
SCALER_PATH  = MODELS_DIR / "scaler.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"


# ── Fábrica de modelos ────────────────────────────────────────────────────────

def _criar_modelo(tipo: str):
    if tipo == "xgboost":
        try:
            from xgboost import XGBRegressor
            return XGBRegressor(**XGBOOST_PARAMS, verbosity=0)
        except ImportError:
            logger.warning("XGBoost não instalado. Usando RandomForest.")
            return RandomForestRegressor(**RANDOM_FOREST_PARAMS)
    elif tipo == "random_forest":
        return RandomForestRegressor(**RANDOM_FOREST_PARAMS)
    else:
        return Ridge(alpha=1.0)


# ── Validação temporal ────────────────────────────────────────────────────────

def validar_temporal(
    df: pd.DataFrame,
    n_splits: int = CV_N_SPLITS,
) -> dict:
    """
    Executa TimeSeriesSplit e retorna métricas médias por fold.
    Os folds são ordenados por rodada para evitar data leakage.
    """
    df_sorted = df.sort_values("rodada").reset_index(drop=True)
    features  = [c for c in FEATURE_COLS if c in df_sorted.columns]
    X = df_sorted[features].fillna(0).values
    y = df_sorted[TARGET_COL].values

    tscv   = TimeSeriesSplit(n_splits=n_splits)
    scaler = StandardScaler()

    metricas_folds = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        modelo = _criar_modelo(MODEL_TYPE)
        modelo.fit(X_train_sc, y_train)
        y_pred = modelo.predict(X_test_sc)

        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)

        metricas_folds.append({"fold": fold, "mae": mae, "rmse": rmse, "r2": r2})
        logger.info("Fold %d → MAE=%.2f  RMSE=%.2f  R²=%.3f", fold, mae, rmse, r2)

    medias = {
        "mae_medio":  float(np.mean([m["mae"]  for m in metricas_folds])),
        "rmse_medio": float(np.mean([m["rmse"] for m in metricas_folds])),
        "r2_medio":   float(np.mean([m["r2"]   for m in metricas_folds])),
        "folds":      metricas_folds,
    }
    logger.info(
        "Validação temporal → MAE médio=%.2f  RMSE médio=%.2f  R²médio=%.3f",
        medias["mae_medio"], medias["rmse_medio"], medias["r2_medio"],
    )
    return medias


# ── Treino final ──────────────────────────────────────────────────────────────

def treinar(df: pd.DataFrame) -> tuple:
    """
    Treina o modelo no dataset completo (após validação temporal),
    persiste modelo + scaler + métricas em disco e retorna ambos.

    Retorna
    -------
    modelo, scaler, metricas
    """
    if df.empty:
        raise ValueError("DataFrame de treino está vazio.")

    logger.info("Iniciando treinamento com modelo '%s'...", MODEL_TYPE)

    features = [c for c in FEATURE_COLS if c in df.columns]
    df_clean = df[features + [TARGET_COL, "rodada"]].dropna(subset=[TARGET_COL])

    # Validação temporal antes do treino final
    metricas = validar_temporal(df_clean)

    X = df_clean[features].fillna(0).values
    y = df_clean[TARGET_COL].values

    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X)
    modelo  = _criar_modelo(MODEL_TYPE)
    modelo.fit(X_sc, y)

    # Importância de features (quando disponível)
    if hasattr(modelo, "feature_importances_"):
        importancias = dict(zip(features, modelo.feature_importances_.tolist()))
        metricas["feature_importances"] = dict(
            sorted(importancias.items(), key=lambda x: x[1], reverse=True)
        )

    # Persistência
    with open(MODEL_PATH,  "wb") as f: pickle.dump(modelo, f)
    with open(SCALER_PATH, "wb") as f: pickle.dump(scaler, f)
    with open(METRICS_PATH, "w") as f: json.dump(metricas, f, indent=2)

    logger.info("Modelo salvo em %s", MODEL_PATH)
    return modelo, scaler, metricas


# ── Carregamento ──────────────────────────────────────────────────────────────

def carregar_modelo() -> tuple:
    """Carrega modelo e scaler já treinados do disco."""
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em {MODELS_DIR}. Execute 'python main.py train' primeiro."
        )
    with open(MODEL_PATH,  "rb") as f: modelo = pickle.load(f)
    with open(SCALER_PATH, "rb") as f: scaler = pickle.load(f)
    logger.info("Modelo carregado de %s", MODEL_PATH)
    return modelo, scaler


def carregar_metricas() -> dict:
    if not METRICS_PATH.exists():
        return {}
    with open(METRICS_PATH) as f:
        return json.load(f)
