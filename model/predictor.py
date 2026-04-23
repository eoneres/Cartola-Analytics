"""
model/predictor.py
Geração de previsões e otimização de escalação.

Fluxo:
  1. Carrega modelo treinado
  2. Constrói features para a próxima rodada
  3. Prevê pontuação de cada atleta
  4. Calcula score composto (ML + forma + adversário)
  5. Otimiza escalação por perfil (conservador / balanceado / agressivo)
"""

import logging

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from config.settings import (
    ESCALACAO_SLOTS,
    MAX_JOGADORES_TIME,
    ORCAMENTO_PADRAO,
    PERFIS,
    POSICOES,
    SCORE_PESOS,
)
from model.features import FEATURE_COLS, construir_features
from model.trainer import carregar_modelo

logger = logging.getLogger(__name__)


# ── Previsão ──────────────────────────────────────────────────────────────────

def prever_pontuacoes(
    df_historico: pd.DataFrame,
    df_mercado: pd.DataFrame,
    df_partidas: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Prevê a pontuação esperada de cada atleta na próxima rodada.

    Retorna DataFrame com colunas:
        atleta_id, apelido, posicao, clube_id, preco,
        pontuacao_prevista, score_composto, media_recente, ...
    """
    modelo, scaler = carregar_modelo()

    # Construir features sobre o histórico
    df_feat = construir_features(df_historico, df_partidas)
    if df_feat.empty:
        logger.error("Sem features — verifique o histórico carregado.")
        return pd.DataFrame()

    # Pegar a última rodada de cada atleta (simula a "próxima" rodada)
    ultima = (
        df_feat.sort_values("rodada")
        .groupby("atleta_id")
        .last()
        .reset_index()
    )

    features  = [c for c in FEATURE_COLS if c in ultima.columns]
    X         = ultima[features].fillna(0).values
    X_sc      = scaler.transform(X)
    previsoes = modelo.predict(X_sc)
    previsoes = np.clip(previsoes, 0, None)   # pontuação não pode ser negativa

    ultima["pontuacao_prevista"] = previsoes

    # Juntar com dados de mercado (preço atual, apelido)
    mercado_cols = ["atleta_id", "apelido", "posicao", "posicao_id",
                    "clube_id", "preco_num", "jogos_num"]
    mercado_cols = [c for c in mercado_cols if c in df_mercado.columns]
    df_result = ultima.merge(
        df_mercado[mercado_cols],
        on="atleta_id",
        how="left",
        suffixes=("", "_mercado"),
    )
    if "preco_num" in df_result.columns:
        df_result["preco"] = df_result["preco_num"]

    # Score composto
    df_result = _calcular_score_composto(df_result)

    colunas_saida = [
        "atleta_id", "apelido", "posicao", "clube_id", "preco",
        "pontuacao_prevista", "score_composto",
        "media_historica", "media_recente", "consistencia",
        "tendencia", "regularidade", "eh_mandante", "forca_adversario",
    ]
    colunas_saida = [c for c in colunas_saida if c in df_result.columns]
    df_result = df_result[colunas_saida].sort_values("score_composto", ascending=False)

    logger.info("Previsões geradas para %d atletas.", len(df_result))
    return df_result.reset_index(drop=True)


def _calcular_score_composto(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score final ponderado:
        score = w1*media_historica + w2*media_recente +
                w3*forca_adversario + w4*pontuacao_prevista
    Normalizado por posição para comparação justa.
    """
    pesos = SCORE_PESOS

    def _norm(series: pd.Series) -> pd.Series:
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series(0.5, index=series.index)
        return (series - mn) / (mx - mn)

    df["_n_hist"]   = _norm(df.get("media_historica",   pd.Series(0, index=df.index)))
    df["_n_rec"]    = _norm(df.get("media_recente",     pd.Series(0, index=df.index)))
    df["_n_adv"]    = _norm(df.get("forca_adversario",  pd.Series(0, index=df.index)))
    df["_n_ml"]     = _norm(df.get("pontuacao_prevista",pd.Series(0, index=df.index)))

    df["score_composto"] = (
        pesos["media_historica"]  * df["_n_hist"] +
        pesos["forma_recente"]    * df["_n_rec"]  +
        pesos["fator_adversario"] * df["_n_adv"]  +
        pesos["sentimento"]       * df["_n_ml"]
    )
    df.drop(columns=["_n_hist","_n_rec","_n_adv","_n_ml"], inplace=True)
    return df


# ── Otimização de escalação ───────────────────────────────────────────────────

def otimizar_escalacao(
    df_previsoes: pd.DataFrame,
    orcamento: float = ORCAMENTO_PADRAO,
    perfil: str = "balanceado",
    formacao: dict | None = None,
) -> pd.DataFrame:
    """
    Seleciona a escalação ótima usando programação linear (greedy + restrições).

    Parâmetros
    ----------
    df_previsoes : resultado de prever_pontuacoes()
    orcamento    : limite em cartoletas
    perfil       : 'conservador' | 'balanceado' | 'agressivo'
    formacao     : dict com slots por posição (padrão: ESCALACAO_SLOTS)

    Retorna
    -------
    DataFrame com os jogadores selecionados e justificativa.
    """
    if formacao is None:
        formacao = ESCALACAO_SLOTS

    config     = PERFIS.get(perfil, PERFIS["balanceado"])
    min_preco  = config["min_preco"]
    w_media    = config["peso_media"]
    w_upside   = config["peso_upside"]

    df = df_previsoes.copy()
    df = df[df["preco"].fillna(0) >= min_preco]
    df["objetivo"] = (
        w_media  * df["score_composto"] +
        w_upside * df["pontuacao_prevista"].fillna(0)
    )

    selecionados = []
    times_usados: dict[int, int] = {}

    for posicao, n_slots in formacao.items():
        candidatos = (
            df[df["posicao"] == posicao]
            .sort_values("objetivo", ascending=False)
            .copy()
        )
        count = 0
        for _, row in candidatos.iterrows():
            if count >= n_slots:
                break
            clube = row["clube_id"]
            if times_usados.get(clube, 0) >= MAX_JOGADORES_TIME:
                continue
            custo = row.get("preco", 0)
            if custo > orcamento:
                continue
            selecionados.append(row)
            orcamento -= custo
            times_usados[clube] = times_usados.get(clube, 0) + 1
            count += 1

    if not selecionados:
        logger.warning("Nenhum jogador selecionado — verifique orçamento e dados.")
        return pd.DataFrame()

    df_escal = pd.DataFrame(selecionados)
    df_escal["perfil_escolhido"] = perfil
    df_escal["justificativa"] = df_escal.apply(_justificativa, axis=1)

    logger.info(
        "Escalação otimizada (%s): %d jogadores | custo total: C$%.1f",
        perfil, len(df_escal),
        df_previsoes.loc[df_escal.index, "preco"].sum() if "preco" in df_previsoes.columns else 0,
    )
    return df_escal[["apelido","posicao","clube_id","preco",
                      "score_composto","pontuacao_prevista",
                      "perfil_escolhido","justificativa"]].reset_index(drop=True)


def _justificativa(row: pd.Series) -> str:
    partes = []
    if row.get("media_recente", 0) > 5:
        partes.append(f"Boa forma recente ({row['media_recente']:.1f} pts/jogo)")
    if row.get("eh_mandante", 0) == 1:
        partes.append("Joga em casa")
    if row.get("forca_adversario", 0) > 0.6:
        partes.append("Adversário vulnerável")
    if row.get("consistencia", 99) < 2:
        partes.append("Alta consistência")
    return " · ".join(partes) if partes else "Score composto elevado"


# ── Alertas ───────────────────────────────────────────────────────────────────

def gerar_alertas(df_previsoes: pd.DataFrame, top_n: int = 5) -> dict:
    """
    Retorna dicionário com alertas categorizados:
      - em_alta       : maiores tendências positivas
      - surpresas     : alta previsão ML vs média histórica baixa
      - riscos        : queda de tendência + inconsistência
      - zeraveis      : baixa regularidade
    """
    df = df_previsoes.copy()

    alertas = {}

    # Em alta
    if "tendencia" in df.columns:
        alertas["em_alta"] = (
            df[df["tendencia"] > 0]
            .nlargest(top_n, "tendencia")[["apelido","posicao","tendencia","score_composto"]]
            .to_dict("records")
        )

    # Surpresas (upside)
    if "pontuacao_prevista" in df.columns and "media_historica" in df.columns:
        df["_delta"] = df["pontuacao_prevista"] - df["media_historica"]
        alertas["surpresas"] = (
            df[df["_delta"] > 2]
            .nlargest(top_n, "_delta")[["apelido","posicao","pontuacao_prevista","media_historica"]]
            .to_dict("records")
        )

    # Riscos
    if "tendencia" in df.columns and "consistencia" in df.columns:
        alertas["riscos"] = (
            df[(df["tendencia"] < 0) & (df["consistencia"] > 3)]
            .nlargest(top_n, "consistencia")[["apelido","posicao","tendencia","consistencia"]]
            .to_dict("records")
        )

    # Prováveis zeráveis
    if "regularidade" in df.columns:
        alertas["zeraveis"] = (
            df[df["regularidade"] < 0.4]
            .nsmallest(top_n, "regularidade")[["apelido","posicao","regularidade"]]
            .to_dict("records")
        )

    return alertas
