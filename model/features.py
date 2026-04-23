"""
model/features.py
Feature engineering para o modelo preditivo do Cartola FC.

Transforma o histórico bruto em features usadas pelo modelo de ML:
  - Médias móveis de pontuação
  - Forma recente (últimas N rodadas)
  - Consistência (desvio padrão)
  - Fator mandante/visitante
  - Fator adversário (média de gols sofridos pelo adversário)
  - Features de preço e posição
"""

import logging

import numpy as np
import pandas as pd

from config.settings import JANELA_FORMA, POSICOES

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _media_movel(series: pd.Series, janela: int) -> pd.Series:
    return series.shift(1).rolling(janela, min_periods=1).mean()


def _desvio_movel(series: pd.Series, janela: int) -> pd.Series:
    return series.shift(1).rolling(janela, min_periods=2).std().fillna(0)


def _tendencia(series: pd.Series, janela: int) -> pd.Series:
    """Slope linear (positivo = melhora, negativo = queda)."""
    def slope(s):
        if len(s) < 2:
            return 0.0
        x = np.arange(len(s))
        return float(np.polyfit(x, s, 1)[0])

    return series.shift(1).rolling(janela, min_periods=2).apply(slope, raw=True).fillna(0)


# ── Feature principal ─────────────────────────────────────────────────────────

def construir_features(
    df_historico: pd.DataFrame,
    df_partidas: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Constrói a tabela de features a partir do histórico de pontuações.

    Parâmetros
    ----------
    df_historico : DataFrame com colunas [atleta_id, rodada, pontuacao,
                   posicao_id, clube_id, preco, scout_*]
    df_partidas  : DataFrame com [rodada, clube_mandante, clube_visitante]
                   usado para calcular fator mandante e fator adversário.

    Retorna
    -------
    DataFrame com uma linha por (atleta_id, rodada) e todas as features.
    """
    if df_historico.empty:
        logger.warning("Histórico vazio — nenhuma feature gerada.")
        return pd.DataFrame()

    df = df_historico.sort_values(["atleta_id", "rodada"]).copy()

    # ── Features por atleta (agrupadas) ──────────────────────────────────────
    grp = df.groupby("atleta_id")

    df["media_historica"]   = grp["pontuacao"].transform(lambda s: _media_movel(s, 38))
    df["media_recente"]     = grp["pontuacao"].transform(lambda s: _media_movel(s, JANELA_FORMA))
    df["consistencia"]      = grp["pontuacao"].transform(lambda s: _desvio_movel(s, JANELA_FORMA))
    df["tendencia"]         = grp["pontuacao"].transform(lambda s: _tendencia(s, JANELA_FORMA))
    df["max_recente"]       = grp["pontuacao"].transform(lambda s: s.shift(1).rolling(JANELA_FORMA, min_periods=1).max())
    df["jogos_acumulados"]  = grp["rodada"].transform("rank").astype(int)

    # Pontuação acima de zero nas últimas N rodadas (regularidade)
    df["regularidade"] = grp["pontuacao"].transform(
        lambda s: (s.shift(1).rolling(JANELA_FORMA, min_periods=1) > 0).mean()
    )

    # ── One-hot de posição ────────────────────────────────────────────────────
    for pid, pnome in POSICOES.items():
        df[f"pos_{pnome.lower()}"] = (df["posicao_id"] == pid).astype(int)

    # ── Fator mandante/visitante ──────────────────────────────────────────────
    if df_partidas is not None and not df_partidas.empty:
        df = _adicionar_fator_mandante(df, df_partidas)
    else:
        df["eh_mandante"] = 0

    # ── Fator adversário ──────────────────────────────────────────────────────
    if df_partidas is not None and not df_partidas.empty:
        df = _adicionar_fator_adversario(df, df_partidas)
    else:
        df["forca_adversario"] = 0.5

    # ── Features de preço ────────────────────────────────────────────────────
    df["preco_log"]       = np.log1p(df["preco"])
    df["custo_beneficio"] = df["media_recente"] / (df["preco"].replace(0, np.nan))
    df["custo_beneficio"] = df["custo_beneficio"].fillna(0)

    # ── Remover rodada 1 (sem histórico suficiente) ───────────────────────────
    df = df[df["rodada"] > 1].reset_index(drop=True)

    logger.info("Features construídas: %d linhas, %d colunas.", len(df), len(df.columns))
    return df


def _adicionar_fator_mandante(df: pd.DataFrame, df_partidas: pd.DataFrame) -> pd.DataFrame:
    """Adiciona coluna 'eh_mandante' (1 = joga em casa, 0 = fora)."""
    mapa = {}
    for _, row in df_partidas.iterrows():
        r = row["rodada"]
        mapa[(r, row["clube_mandante"])]  = 1
        mapa[(r, row["clube_visitante"])] = 0

    df["eh_mandante"] = df.apply(
        lambda r: mapa.get((r["rodada"], r["clube_id"]), 0), axis=1
    )
    return df


def _adicionar_fator_adversario(df: pd.DataFrame, df_partidas: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula 'forca_adversario': média de gols sofridos pelo adversário
    nas últimas rodadas (quanto maior, mais fácil é o jogo).
    """
    partidas = df_partidas.dropna(subset=["gols_mandante", "gols_visitante"]).copy()
    if partidas.empty:
        df["forca_adversario"] = 0.5
        return df

    # gols sofridos por clube em cada rodada
    mandante  = partidas[["rodada", "clube_mandante",  "gols_visitante"]].rename(
        columns={"clube_mandante": "clube_id", "gols_visitante": "gols_sofridos"}
    )
    visitante = partidas[["rodada", "clube_visitante", "gols_mandante"]].rename(
        columns={"clube_visitante": "clube_id", "gols_mandante": "gols_sofridos"}
    )
    gols = pd.concat([mandante, visitante]).sort_values(["clube_id", "rodada"])
    gols["media_gols_sofridos"] = gols.groupby("clube_id")["gols_sofridos"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    gols_map = gols.set_index(["rodada", "clube_id"])["media_gols_sofridos"].to_dict()

    # mapear adversário de cada atleta
    adv_map = {}
    for _, row in df_partidas.iterrows():
        r = row["rodada"]
        adv_map[(r, row["clube_mandante"])]  = row["clube_visitante"]
        adv_map[(r, row["clube_visitante"])] = row["clube_mandante"]

    def _forca(row):
        adv = adv_map.get((row["rodada"], row["clube_id"]))
        if adv is None:
            return 0.5
        return gols_map.get((row["rodada"], adv), 0.5)

    df["forca_adversario"] = df.apply(_forca, axis=1)
    # Normalizar 0-1
    mx = df["forca_adversario"].max()
    if mx > 0:
        df["forca_adversario"] = df["forca_adversario"] / mx
    return df


# ── Lista de features usadas no modelo ───────────────────────────────────────

FEATURE_COLS = [
    "media_historica",
    "media_recente",
    "consistencia",
    "tendencia",
    "max_recente",
    "regularidade",
    "jogos_acumulados",
    "eh_mandante",
    "forca_adversario",
    "preco_log",
    "custo_beneficio",
    "pos_goleiro",
    "pos_lateral",
    "pos_zagueiro",
    "pos_meia",
    "pos_atacante",
    "pos_técnico",
]

TARGET_COL = "pontuacao"
