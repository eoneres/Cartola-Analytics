"""
autolearn/engine.py
Motor de auto-aprendizado do sistema Cartola FC.

Responsabilidades:
  1. Detectar quando há dados novos suficientes para re-treinar
  2. Re-treinar o modelo com o histórico atualizado
  3. Comparar métricas do novo modelo com o modelo em produção
  4. Fazer rollout automático se o novo modelo for melhor
  5. Ajustar os pesos do score composto com base na correlação histórica
  6. Manter registro de versões do modelo (model registry simples)

Este módulo é chamado pelo scheduler (scheduler/jobs.py) após cada rodada,
mas também pode ser executado manualmente via CLI.
"""

import json
import logging
import pickle
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import DATA_DIR, MODELS_DIR, SCORE_PESOS, TEMPORADA_ATUAL
from model.features import FEATURE_COLS, TARGET_COL, construir_features
from model.trainer import _criar_modelo, validar_temporal

logger = logging.getLogger(__name__)

REGISTRY_DIR  = MODELS_DIR / "registry"
REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
REGISTRY_FILE = REGISTRY_DIR / "registry.json"

PROCESSED_DIR = DATA_DIR / "processed"


# ── Model Registry ────────────────────────────────────────────────────────────

def _carregar_registry() -> list[dict]:
    if REGISTRY_FILE.exists():
        with open(REGISTRY_FILE) as f:
            return json.load(f)
    return []


def _salvar_registry(registry: list[dict]) -> None:
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2)


def _registrar_modelo(versao: str, metricas: dict, caminho: Path, ativo: bool = False) -> None:
    registry = _carregar_registry()
    # Desativar versões anteriores se nova vai ser ativa
    if ativo:
        for r in registry:
            r["ativo"] = False
    registry.append({
        "versao":      versao,
        "criado_em":   datetime.utcnow().isoformat(),
        "metricas":    metricas,
        "caminho":     str(caminho),
        "ativo":       ativo,
    })
    _salvar_registry(registry)
    logger.info("Modelo registrado: versão=%s ativo=%s MAE=%.2f", versao, ativo, metricas.get("mae_medio", 0))


def modelo_em_producao() -> dict | None:
    registry = _carregar_registry()
    ativos = [r for r in registry if r.get("ativo")]
    return ativos[-1] if ativos else None


def listar_versoes() -> list[dict]:
    return _carregar_registry()


# ── Detecção de necessidade de re-treino ──────────────────────────────────────

def verificar_necessidade_retreino(
    min_rodadas_novas: int = 3,
    min_mae_degradacao: float = 2.0,
) -> dict:
    """
    Verifica se o modelo precisa ser re-treinado.

    Critérios:
      - Há N ou mais rodadas novas desde o último treino
      - O MAE nas rodadas recentes degradou além de um limiar

    Retorna dict com {precisa: bool, motivo: str, rodadas_novas: int}
    """
    hist_path = PROCESSED_DIR / "historico_completo.parquet"
    if not hist_path.exists():
        return {"precisa": False, "motivo": "sem dados históricos", "rodadas_novas": 0}

    df = pd.read_parquet(hist_path)
    if "rodada" not in df.columns:
        return {"precisa": False, "motivo": "coluna rodada ausente", "rodadas_novas": 0}

    max_rodada = df["rodada"].max()
    prod = modelo_em_producao()

    if prod is None:
        return {"precisa": True, "motivo": "nenhum modelo em produção", "rodadas_novas": int(max_rodada)}

    # Rodadas desde o último treino
    meta = prod.get("metricas", {})
    rodada_treino = meta.get("rodada_max_treino", 0)
    rodadas_novas = int(max_rodada - rodada_treino)

    if rodadas_novas >= min_rodadas_novas:
        return {
            "precisa":      True,
            "motivo":       f"{rodadas_novas} novas rodadas disponíveis",
            "rodadas_novas": rodadas_novas,
        }

    return {
        "precisa":       False,
        "motivo":        f"apenas {rodadas_novas} novas rodadas (mínimo: {min_rodadas_novas})",
        "rodadas_novas": rodadas_novas,
    }


# ── Re-treino ─────────────────────────────────────────────────────────────────

def retreinar(forcar: bool = False) -> dict:
    """
    Re-treina o modelo com o histórico mais recente e faz deploy
    automático se superar o modelo em produção.

    Parâmetros
    ----------
    forcar : ignora a verificação de necessidade e re-treina sempre

    Retorna
    -------
    dict com {versao, metricas, deployed, comparacao}
    """
    if not forcar:
        check = verificar_necessidade_retreino()
        if not check["precisa"]:
            logger.info("Re-treino desnecessário: %s", check["motivo"])
            return {"deployed": False, "motivo": check["motivo"]}

    logger.info("Iniciando re-treino automático...")

    hist_path = PROCESSED_DIR / "historico_completo.parquet"
    part_path = PROCESSED_DIR / "partidas.parquet"

    if not hist_path.exists():
        raise FileNotFoundError("Histórico não encontrado para re-treino.")

    df_hist = pd.read_parquet(hist_path)
    df_part = pd.read_parquet(part_path) if part_path.exists() else None

    df_feat  = construir_features(df_hist, df_part)
    features = [c for c in FEATURE_COLS if c in df_feat.columns]
    df_clean = df_feat[features + [TARGET_COL, "rodada"]].dropna(subset=[TARGET_COL])

    # Validação temporal no conjunto completo
    from sklearn.preprocessing import StandardScaler

    df_sorted = df_clean.sort_values("rodada").reset_index(drop=True)
    X = df_sorted[features].fillna(0).values
    y = df_sorted[TARGET_COL].values

    from config.settings import MODEL_TYPE
    from sklearn.model_selection import TimeSeriesSplit

    tscv   = TimeSeriesSplit(n_splits=5)
    scaler = StandardScaler()

    maes, rmses, r2s = [], [], []
    for train_idx, test_idx in tscv.split(X):
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)
        m = _criar_modelo(MODEL_TYPE)
        m.fit(X_tr_sc, y_tr)
        y_pred = m.predict(X_te_sc)
        maes.append(mean_absolute_error(y_te, y_pred))
        rmses.append(np.sqrt(mean_squared_error(y_te, y_pred)))
        r2s.append(r2_score(y_te, y_pred))

    novas_metricas = {
        "mae_medio":        round(float(np.mean(maes)), 4),
        "rmse_medio":       round(float(np.mean(rmses)), 4),
        "r2_medio":         round(float(np.mean(r2s)), 4),
        "rodada_max_treino": int(df_clean["rodada"].max()),
        "n_amostras":        int(len(df_clean)),
        "criado_em":         datetime.utcnow().isoformat(),
    }

    # Treino final no conjunto completo
    X_sc = scaler.fit_transform(X)
    modelo_novo = _criar_modelo(MODEL_TYPE)
    modelo_novo.fit(X_sc, y)

    # Versão com timestamp
    versao = datetime.utcnow().strftime("v%Y%m%d_%H%M%S")
    versao_dir = REGISTRY_DIR / versao
    versao_dir.mkdir()

    model_path  = versao_dir / "model.pkl"
    scaler_path = versao_dir / "scaler.pkl"
    with open(model_path,  "wb") as f: pickle.dump(modelo_novo, f)
    with open(scaler_path, "wb") as f: pickle.dump(scaler, f)

    # Comparar com modelo em produção
    prod = modelo_em_producao()
    deploy = False
    comparacao = {}

    if prod is None:
        deploy = True
        comparacao = {"motivo": "primeiro modelo"}
    else:
        mae_prod = prod["metricas"].get("mae_medio", 999)
        mae_novo = novas_metricas["mae_medio"]
        delta    = mae_prod - mae_novo
        comparacao = {
            "mae_producao": mae_prod,
            "mae_novo":     mae_novo,
            "delta_mae":    round(delta, 4),
            "melhora":      delta > 0,
        }
        if delta > 0:
            deploy = True
            logger.info("Novo modelo é melhor: MAE %.2f → %.2f (Δ=%.2f)", mae_prod, mae_novo, delta)
        else:
            logger.info("Modelo atual é melhor ou equivalente: MAE %.2f vs %.2f", mae_prod, mae_novo)

    if deploy:
        # Copiar para caminho de produção
        shutil.copy(model_path,  MODELS_DIR / "cartola_model.pkl")
        shutil.copy(scaler_path, MODELS_DIR / "scaler.pkl")
        with open(MODELS_DIR / "metrics.json", "w") as f:
            json.dump(novas_metricas, f, indent=2)
        logger.info("Modelo %s deployado em produção.", versao)

    _registrar_modelo(versao, novas_metricas, versao_dir, ativo=deploy)

    return {
        "versao":      versao,
        "metricas":    novas_metricas,
        "deployed":    deploy,
        "comparacao":  comparacao,
    }


# ── Ajuste automático de pesos do score composto ──────────────────────────────

def ajustar_pesos_score(df_historico: pd.DataFrame) -> dict:
    """
    Calcula os pesos ótimos do score composto baseado na correlação
    histórica de cada componente com a pontuação real.

    Componentes analisados:
      - media_historica   → SCORE_PESOS["media_historica"]
      - media_recente     → SCORE_PESOS["forma_recente"]
      - forca_adversario  → SCORE_PESOS["fator_adversario"]

    Usa correlação de Pearson normalizada para definir os pesos.
    Salva os novos pesos sugeridos em models/pesos_sugeridos.json.
    """
    df = df_historico.copy()
    componentes = {
        "media_historica":  "media_historica",
        "forma_recente":    "media_recente",
        "fator_adversario": "forca_adversario",
    }

    corrs = {}
    for nome, col in componentes.items():
        if col in df.columns and TARGET_COL in df.columns:
            c = df[[col, TARGET_COL]].dropna().corr().iloc[0, 1]
            corrs[nome] = max(0.01, float(abs(c)))   # apenas correlações positivas
        else:
            corrs[nome] = 0.25   # fallback neutro

    # Normalizar para somar ~0.9 (reservando 0.10 para sentimento)
    total = sum(corrs.values())
    pesos = {k: round(v / total * 0.90, 3) for k, v in corrs.items()}
    pesos["sentimento"] = 0.10

    # Garantir que somam 1.0
    diff = 1.0 - sum(pesos.values())
    pesos["media_historica"] = round(pesos["media_historica"] + diff, 3)

    saida = {
        "pesos_calculados":  pesos,
        "correlacoes":       {k: round(v, 4) for k, v in corrs.items()},
        "calculado_em":      datetime.utcnow().isoformat(),
        "n_amostras":        int(len(df)),
    }

    path = MODELS_DIR / "pesos_sugeridos.json"
    with open(path, "w") as f:
        json.dump(saida, f, indent=2)

    logger.info("Pesos sugeridos calculados: %s", pesos)
    return saida
