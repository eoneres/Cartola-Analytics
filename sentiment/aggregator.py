"""
sentiment/aggregator.py
Agrega os scores de sentimento individuais em métricas por jogador.

Para cada jogador produz:
  - score_medio       : média ponderada dos scores (-1 a +1)
  - tendencia         : slope da série temporal de scores (positivo = melhora)
  - volume            : número de textos coletados
  - distribuicao      : % positivo / neutro / negativo
  - hype_score        : indicador composto de visibilidade + sentimento positivo
  - alerta            : "em alta" | "neutro" | "em crise"

O score final é normalizado 0–1 e integrado ao score composto do modelo
de ML via config/settings.py (SCORE_PESOS["sentimento"]).
"""

import logging
from datetime import datetime

import numpy as np
import pandas as pd

from config.settings import DATA_DIR
from sentiment.analyzer import analisar_lote
from sentiment.collector import coletar_textos_jogador

logger = logging.getLogger(__name__)

SENTIMENT_DIR = DATA_DIR / "processed" / "sentiment"
SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)


# ── Pipeline por jogador ──────────────────────────────────────────────────────

def processar_jogador(
    jogador: str,
    modo_analise: str = "auto",
    incluir_twitter:  bool = True,
    incluir_reddit:   bool = True,
    incluir_noticias: bool = True,
) -> dict:
    """
    Coleta e analisa todos os textos disponíveis sobre um jogador.

    Retorna
    -------
    dict com métricas agregadas de sentimento.
    """
    logger.info("Processando sentimento para: %s", jogador)

    # 1. Coleta
    textos_raw = coletar_textos_jogador(
        jogador,
        incluir_twitter=incluir_twitter,
        incluir_reddit=incluir_reddit,
        incluir_noticias=incluir_noticias,
    )

    if not textos_raw:
        logger.warning("Nenhum texto coletado para '%s'.", jogador)
        return _resultado_vazio(jogador)

    # 2. Análise de sentimento
    textos_str = [t["texto"] for t in textos_raw]
    analises   = analisar_lote(textos_str, modo=modo_analise)

    # 3. Combinar com metadados de fonte
    registros = []
    for raw, analise in zip(textos_raw, analises):
        registros.append({
            "jogador":    jogador,
            "fonte":      raw["fonte"],
            "texto":      analise["texto"],
            "sentimento": analise["sentimento"],
            "score":      analise["score"],
            "confianca":  analise["confianca"],
            "modo":       analise["modo"],
            "timestamp":  datetime.utcnow().isoformat(),
        })

    df = pd.DataFrame(registros)

    # 4. Agregar
    resultado = _agregar(jogador, df)

    # 5. Persistir
    _salvar(jogador, df, resultado)

    return resultado


def _agregar(jogador: str, df: pd.DataFrame) -> dict:
    """Computa métricas agregadas a partir dos registros de sentimento."""
    scores = df["score"].values
    pesos  = df["confianca"].values  # textos mais confiantes têm mais peso

    # Score médio ponderado pela confiança
    score_medio = float(np.average(scores, weights=pesos)) if pesos.sum() > 0 else 0.0

    # Distribuição
    dist = df["sentimento"].value_counts(normalize=True).to_dict()

    # Tendência temporal (slope sobre os últimos N textos, em ordem de coleta)
    tendencia = 0.0
    if len(scores) >= 3:
        x = np.arange(len(scores))
        tendencia = float(np.polyfit(x, scores, 1)[0])

    # Volume por fonte
    vol_fonte = df["fonte"].value_counts().to_dict()

    # Hype score: combina volume + positividade
    # hype alto = muita menção positiva → jogador em evidência
    pct_positivo = dist.get("positivo", 0.0)
    volume_norm  = min(1.0, len(df) / 100)    # normaliza por 100 textos
    hype_score   = round(0.5 * pct_positivo + 0.5 * volume_norm, 4)

    # Alerta categórico
    if score_medio > 0.2 and tendencia >= 0:
        alerta = "em alta"
    elif score_medio < -0.2 or tendencia < -0.05:
        alerta = "em crise"
    else:
        alerta = "neutro"

    return {
        "jogador":       jogador,
        "score_medio":   round(score_medio, 4),
        "tendencia":     round(tendencia, 6),
        "volume":        len(df),
        "hype_score":    hype_score,
        "alerta":        alerta,
        "pct_positivo":  round(dist.get("positivo", 0.0), 3),
        "pct_neutro":    round(dist.get("neutro",   0.0), 3),
        "pct_negativo":  round(dist.get("negativo", 0.0), 3),
        "vol_por_fonte": vol_fonte,
        "atualizado_em": datetime.utcnow().isoformat(),
    }


def _resultado_vazio(jogador: str) -> dict:
    return {
        "jogador":       jogador,
        "score_medio":   0.0,
        "tendencia":     0.0,
        "volume":        0,
        "hype_score":    0.0,
        "alerta":        "sem dados",
        "pct_positivo":  0.0,
        "pct_neutro":    1.0,
        "pct_negativo":  0.0,
        "vol_por_fonte": {},
        "atualizado_em": datetime.utcnow().isoformat(),
    }


def _salvar(jogador: str, df: pd.DataFrame, resultado: dict) -> None:
    import json as _json
    slug = jogador.lower().replace(" ", "_")
    df.to_parquet(SENTIMENT_DIR / f"{slug}_textos.parquet", index=False)
    # vol_por_fonte é dict — serializar como string JSON para compatibilidade com PyArrow
    resultado_serializado = resultado.copy()
    if isinstance(resultado_serializado.get("vol_por_fonte"), dict):
        resultado_serializado["vol_por_fonte"] = _json.dumps(
            resultado_serializado["vol_por_fonte"], ensure_ascii=False
        )
    pd.DataFrame([resultado_serializado]).to_parquet(
        SENTIMENT_DIR / f"{slug}_score.parquet", index=False
    )
    logger.info("Sentimento salvo para '%s'.", jogador)


# ── Pipeline para lista de jogadores ─────────────────────────────────────────

def processar_lista_jogadores(
    jogadores: list[str],
    modo_analise: str = "auto",
    **kwargs,
) -> pd.DataFrame:
    """
    Processa sentimento para uma lista de jogadores e retorna DataFrame
    consolidado, pronto para merge com as previsões do modelo ML.

    Uso:
        df_sent = processar_lista_jogadores(["Endrick", "Gabi", "Arrascaeta"])
        # resultado:
        # jogador | score_medio | tendencia | hype_score | alerta | ...
    """
    resultados = []
    for jogador in jogadores:
        try:
            res = processar_jogador(jogador, modo_analise=modo_analise, **kwargs)
            resultados.append(res)
        except Exception as e:
            logger.error("Falha ao processar '%s': %s", jogador, e)
            resultados.append(_resultado_vazio(jogador))

    df = pd.DataFrame(resultados)

    # Normalizar score_medio para 0-1 (para integração com score composto)
    mn, mx = df["score_medio"].min(), df["score_medio"].max()
    if mx > mn:
        df["score_sentimento_norm"] = (df["score_medio"] - mn) / (mx - mn)
    else:
        df["score_sentimento_norm"] = 0.5

    # Salvar consolidado — serializar dicts para evitar ArrowNotImplementedError
    import json as _json
    df_save = df.copy()
    if "vol_por_fonte" in df_save.columns:
        df_save["vol_por_fonte"] = df_save["vol_por_fonte"].apply(
            lambda v: _json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else str(v or "{}")
        )
    out = SENTIMENT_DIR / "sentimento_consolidado.parquet"
    df_save.to_parquet(out, index=False)
    df_save.to_csv(SENTIMENT_DIR / "sentimento_consolidado.csv", index=False)
    logger.info("Sentimento consolidado salvo: %d jogadores.", len(df))

    return df


# ── Integração com previsões ML ───────────────────────────────────────────────

def integrar_sentimento(
    df_previsoes: pd.DataFrame,
    df_sentimento: pd.DataFrame,
    peso_sentimento: float = 0.10,
) -> pd.DataFrame:
    """
    Incorpora o score de sentimento ao score composto das previsões.

    Parâmetros
    ----------
    df_previsoes    : DataFrame de prever_pontuacoes()
    df_sentimento   : DataFrame de processar_lista_jogadores()
    peso_sentimento : quanto o sentimento contribui para o score final

    Retorna
    -------
    df_previsoes com colunas adicionais:
        score_sentimento_norm, hype_score, alerta_sentimento,
        score_composto_final (score_composto rebalanceado)
    """
    df = df_previsoes.merge(
        df_sentimento[["jogador", "score_sentimento_norm", "hype_score",
                       "alerta", "pct_positivo", "tendencia"]].rename(
            columns={
                "jogador":   "apelido",
                "alerta":    "alerta_sentimento",
                "tendencia": "tendencia_sentimento",
            }
        ),
        on="apelido",
        how="left",
    )

    # Preencher jogadores sem dados de sentimento com neutro
    df["score_sentimento_norm"] = df["score_sentimento_norm"].fillna(0.5)
    df["hype_score"]            = df["hype_score"].fillna(0.0)
    df["alerta_sentimento"]     = df["alerta_sentimento"].fillna("sem dados")

    # Recalcular score composto incluindo sentimento
    w_ant = 1.0 - peso_sentimento
    df["score_composto_final"] = (
        w_ant * df["score_composto"] +
        peso_sentimento * df["score_sentimento_norm"]
    ).round(4)

    df = df.sort_values("score_composto_final", ascending=False).reset_index(drop=True)
    logger.info("Sentimento integrado: %d atletas atualizados.", len(df))
    return df
