"""
data_collection/pipeline.py
Pipeline ETL completo: coleta → limpeza → normalização → persistência.
"""

import logging
from pathlib import Path

import pandas as pd

from config.settings import DATA_DIR, POSICOES, SCOUTS_PESOS, TEMPORADA_ATUAL
from data_collection.cartola_api import (
    buscar_atletas_mercado,
    buscar_atletas_pontuados,
    buscar_historico_completo,
    buscar_partidas,
    buscar_status_mercado,
)

logger = logging.getLogger(__name__)

RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ── Limpeza ───────────────────────────────────────────────────────────────────

def _limpar_atletas(atletas: list[dict]) -> pd.DataFrame:
    """Converte lista bruta de atletas para DataFrame limpo."""
    df = pd.DataFrame(atletas)

    colunas_necessarias = [
        "atleta_id", "apelido", "posicao_id", "clube_id",
        "preco_num", "media_num", "pontos_num", "jogos_num",
    ]
    for col in colunas_necessarias:
        if col not in df.columns:
            df[col] = 0

    df = df[colunas_necessarias].copy()
    df["posicao"] = df["posicao_id"].map(POSICOES).fillna("Desconhecido")
    df["preco_num"]  = pd.to_numeric(df["preco_num"],  errors="coerce").fillna(0.0)
    df["media_num"]  = pd.to_numeric(df["media_num"],  errors="coerce").fillna(0.0)
    df["pontos_num"] = pd.to_numeric(df["pontos_num"], errors="coerce").fillna(0.0)
    df["jogos_num"]  = pd.to_numeric(df["jogos_num"],  errors="coerce").fillna(0).astype(int)

    df = df[df["atleta_id"].notna()].drop_duplicates("atleta_id")
    logger.info("Atletas após limpeza: %d", len(df))
    return df


def _limpar_pontuados(atletas: list[dict], rodada: int) -> pd.DataFrame:
    """
    Normaliza os dados de pontuação por rodada, expandindo os scouts
    em colunas individuais.
    """
    registros = []
    for a in atletas:
        scouts = a.get("scout", {}) or {}
        rec = {
            "atleta_id":  a.get("atleta_id"),
            "apelido":    a.get("apelido", ""),
            "posicao_id": a.get("posicao_id"),
            "clube_id":   a.get("clube_id"),
            "rodada":     rodada,
            "pontuacao":  float(a.get("pontos_num", 0) or 0),
            "preco":      float(a.get("preco_num", 0) or 0),
            "variacao":   float(a.get("variacao_num", 0) or 0),
            "media":      float(a.get("media_num", 0) or 0),
        }
        for scout, peso in SCOUTS_PESOS.items():
            rec[f"scout_{scout}"] = int(scouts.get(scout.upper(), 0) or 0)
        registros.append(rec)

    df = pd.DataFrame(registros)
    df["posicao"] = df["posicao_id"].map(POSICOES).fillna("Desconhecido")
    return df


def _limpar_partidas(partidas: list[dict], rodada: int) -> pd.DataFrame:
    """Estrutura as partidas com mandante, visitante e placar."""
    registros = []
    for p in partidas:
        registros.append({
            "rodada":          rodada,
            "clube_mandante":  p.get("clube_casa_id"),
            "clube_visitante": p.get("clube_visitante_id"),
            "gols_mandante":   p.get("placar_oficial_mandante", None),
            "gols_visitante":  p.get("placar_oficial_visitante", None),
            "valida":          p.get("valida", True),
        })
    return pd.DataFrame(registros)


# ── Persistência ──────────────────────────────────────────────────────────────

def _salvar_parquet(df: pd.DataFrame, nome: str, subdir: Path) -> Path:
    path = subdir / f"{nome}.parquet"
    df.to_parquet(path, index=False)
    logger.info("Salvo: %s (%d linhas)", path, len(df))
    return path


def _salvar_csv(df: pd.DataFrame, nome: str, subdir: Path) -> Path:
    path = subdir / f"{nome}.csv"
    df.to_csv(path, index=False)
    return path


# ── Etapas do pipeline ────────────────────────────────────────────────────────

def etapa_mercado_atual() -> pd.DataFrame:
    """Coleta e persiste snapshot do mercado atual."""
    logger.info("=== Etapa: mercado atual ===")
    status  = buscar_status_mercado()
    atletas = buscar_atletas_mercado()
    df      = _limpar_atletas(atletas)
    df["temporada"] = TEMPORADA_ATUAL
    df["rodada_ref"] = status.get("rodada_atual", 0)
    _salvar_parquet(df, "mercado_atual", PROCESSED_DIR)
    _salvar_csv(df, "mercado_atual", PROCESSED_DIR)
    return df


def etapa_historico(rodada_inicio: int, rodada_fim: int) -> pd.DataFrame:
    """Coleta histórico de pontuações e persiste por rodada e consolidado."""
    logger.info("=== Etapa: histórico rodadas %d–%d ===", rodada_inicio, rodada_fim)
    brutos = buscar_historico_completo(rodada_inicio, rodada_fim)

    frames = []
    for rodada in range(rodada_inicio, rodada_fim + 1):
        subset = [a for a in brutos if a.get("rodada") == rodada]
        if not subset:
            continue
        df_rodada = _limpar_pontuados(subset, rodada)
        _salvar_parquet(df_rodada, f"pontuados_r{rodada:02d}", RAW_DIR)
        frames.append(df_rodada)

    if not frames:
        logger.warning("Nenhum dado histórico encontrado.")
        return pd.DataFrame()

    df_total = pd.concat(frames, ignore_index=True)
    df_total["temporada"] = TEMPORADA_ATUAL
    _salvar_parquet(df_total, "historico_completo", PROCESSED_DIR)
    _salvar_csv(df_total, "historico_completo", PROCESSED_DIR)
    logger.info("Histórico consolidado: %d registros.", len(df_total))
    return df_total


def etapa_partidas(rodada_inicio: int, rodada_fim: int) -> pd.DataFrame:
    """Coleta e persiste dados de partidas para cálculo do fator adversário."""
    logger.info("=== Etapa: partidas ===")
    frames = []
    for rodada in range(rodada_inicio, rodada_fim + 1):
        try:
            partidas = buscar_partidas(rodada, usar_cache=True)
            df_p = _limpar_partidas(partidas, rodada)
            frames.append(df_p)
        except Exception as e:
            logger.warning("Erro ao buscar partidas rodada %d: %s", rodada, e)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    _salvar_parquet(df, "partidas", PROCESSED_DIR)
    return df


# ── Ponto de entrada ──────────────────────────────────────────────────────────

def executar_pipeline(rodada_inicio: int = 1, rodada_fim: int = 10) -> dict:
    """
    Executa o pipeline ETL completo e retorna dicionário com os DataFrames.

    Uso:
        dados = executar_pipeline(rodada_inicio=1, rodada_fim=14)
    """
    logger.info("Iniciando pipeline ETL — temporada %d", TEMPORADA_ATUAL)

    resultado = {}
    resultado["mercado"]  = etapa_mercado_atual()
    resultado["historico"] = etapa_historico(rodada_inicio, rodada_fim)
    resultado["partidas"]  = etapa_partidas(rodada_inicio, rodada_fim)

    logger.info("Pipeline concluído.")
    return resultado
