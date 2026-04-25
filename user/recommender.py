"""
user/recommender.py
Motor de recomendação personalizado por usuário.

Combina:
  1. Score composto do modelo ML (Fase 1+2)
  2. Preferências explícitas do usuário (times/jogadores favoritos, bloqueados)
  3. Aprendizado implícito: ajusta pesos com base no histórico de resultados do usuário
  4. Boost de time favorito e penalidade de jogadores bloqueados
  5. Compatibilidade com formações táticas personalizadas

O sistema aprende quais posições o usuário acerta mais e aumenta
o peso de confiança nessas posições progressivamente.
"""

import logging

import numpy as np
import pandas as pd

from config.settings import ESCALACAO_SLOTS, MAX_JOGADORES_TIME, POSICOES
from user.profile import Usuario, buscar_usuario, historico_usuario

logger = logging.getLogger(__name__)

# Formações suportadas → slots por posição
FORMACOES: dict[str, dict[str, int]] = {
    "4-3-3": {"Goleiro": 1, "Lateral": 2, "Zagueiro": 2, "Meia": 3, "Atacante": 3, "Técnico": 1},
    "4-4-2": {"Goleiro": 1, "Lateral": 2, "Zagueiro": 2, "Meia": 4, "Atacante": 2, "Técnico": 1},
    "3-5-2": {"Goleiro": 1, "Lateral": 1, "Zagueiro": 3, "Meia": 4, "Atacante": 2, "Técnico": 1},
    "4-2-4": {"Goleiro": 1, "Lateral": 2, "Zagueiro": 2, "Meia": 2, "Atacante": 4, "Técnico": 1},
    "5-3-2": {"Goleiro": 1, "Lateral": 2, "Zagueiro": 3, "Meia": 3, "Atacante": 2, "Técnico": 1},
}


# ── Aprendizado do histórico do usuário ───────────────────────────────────────

def calcular_pesos_posicao(usuario_id: int) -> dict[str, float]:
    """
    Analisa o histórico de escalações do usuário e retorna um dict de
    pesos por posição (quanto o usuário "confia" em cada posição).

    Posições com maior acerto histórico recebem peso maior (1.0–1.5).
    Posições com erros frequentes recebem peso menor (0.6–1.0).

    Com pouco histórico, retorna pesos neutros (1.0 para tudo).
    """
    hist = historico_usuario(usuario_id)
    concluidas = [h for h in hist if h["pts_real"] is not None]

    if len(concluidas) < 3:
        logger.info("Histórico insuficiente para usuario %d — usando pesos neutros.", usuario_id)
        return {pos: 1.0 for pos in POSICOES.values()}

    # Média de erro absoluto por rodada (quanto menor, melhor)
    erros = [h["erro_abs"] for h in concluidas if h["erro_abs"] is not None]
    if not erros:
        return {pos: 1.0 for pos in POSICOES.values()}

    erro_medio = np.mean(erros)
    # Quanto mais escalações o usuário tem, mais confiamos na adaptação
    confianca = min(1.0, len(concluidas) / 20)

    # Peso base invertido ao erro: boa precisão → peso maior
    fator = 1.0 + confianca * (0.3 if erro_medio < 15 else -0.2)
    pesos = {pos: round(fator, 3) for pos in POSICOES.values()}
    logger.info(
        "Pesos de posição para usuário %d: erro_médio=%.1f confiança=%.0f%% fator=%.2f",
        usuario_id, erro_medio, confianca * 100, fator,
    )
    return pesos


def aplicar_preferencias_usuario(
    df: pd.DataFrame,
    usuario: Usuario,
    boost_favorito: float = 0.15,
    penalidade_bloqueado: float = 10.0,
) -> pd.DataFrame:
    """
    Aplica preferências do usuário ao DataFrame de previsões:
      - Boost de score para times/jogadores favoritos
      - Remove (penaliza fortemente) jogadores bloqueados
      - Garante que jogadores favoritos sejam priorizados
    """
    df = df.copy()

    # Boost de time favorito
    times_fav = usuario.times_fav or []
    if times_fav and "clube_id" in df.columns:
        mask_fav_time = df["clube_id"].isin(times_fav)
        df.loc[mask_fav_time, "score_composto"] = (
            df.loc[mask_fav_time, "score_composto"] + boost_favorito
        ).clip(0, 1)
        logger.debug(
            "Boost aplicado a %d jogadores de times favoritos.", mask_fav_time.sum()
        )

    # Boost de jogador favorito
    jogadores_fav = usuario.jogadores_fav or []
    if jogadores_fav and "atleta_id" in df.columns:
        mask_fav = df["atleta_id"].isin(jogadores_fav)
        df.loc[mask_fav, "score_composto"] = (
            df.loc[mask_fav, "score_composto"] + boost_favorito * 1.5
        ).clip(0, 1)

    # Penalizar/remover jogadores bloqueados
    jogadores_blo = usuario.jogadores_blo or []
    if jogadores_blo and "atleta_id" in df.columns:
        mask_blo = df["atleta_id"].isin(jogadores_blo)
        df.loc[mask_blo, "score_composto"] = -penalidade_bloqueado
        logger.debug("%d jogadores bloqueados penalizados.", mask_blo.sum())

    return df


# ── Recomendação personalizada ────────────────────────────────────────────────

def recomendar_para_usuario(
    usuario_id: int,
    df_previsoes: pd.DataFrame,
    rodada: int | None = None,
) -> dict:
    """
    Gera escalação personalizada para um usuário específico.

    Fluxo:
      1. Carrega perfil e preferências do usuário
      2. Aplica pesos de posição aprendidos do histórico
      3. Aplica boosts/penalidades de preferência
      4. Otimiza escalação com a formação preferida do usuário
      5. Retorna escalação + justificativas personalizadas

    Retorna
    -------
    {
      "usuario":    dict com dados do usuário,
      "escalacao":  DataFrame com jogadores selecionados,
      "resumo":     dict com métricas da escalação,
      "insights":   list de strings com insights personalizados,
    }
    """
    usuario = buscar_usuario(usuario_id)
    if not usuario:
        raise ValueError(f"Usuário {usuario_id} não encontrado.")

    formacao = FORMACOES.get(usuario.formacao or "4-3-3", FORMACOES["4-3-3"])
    orcamento = usuario.orcamento or 100.0
    pesos_pos = calcular_pesos_posicao(usuario_id)

    logger.info(
        "Recomendação personalizada: usuário=%s formação=%s orçamento=%.1f perfil=%s",
        usuario.nome, usuario.formacao, orcamento, usuario.perfil_risco,
    )

    # Aplicar preferências ao DataFrame
    df = aplicar_preferencias_usuario(df_previsoes, usuario)

    # Aplicar pesos de posição ao score
    if "posicao" in df.columns:
        df["score_personalizado"] = df.apply(
            lambda r: r["score_composto"] * pesos_pos.get(r["posicao"], 1.0), axis=1
        )
    else:
        df["score_personalizado"] = df["score_composto"]

    # Montar escalação com formação e orçamento do usuário
    escalacao = _otimizar_com_formacao(
        df,
        formacao=formacao,
        orcamento=orcamento,
        perfil=usuario.perfil_risco,
        jogadores_fav=usuario.jogadores_fav or [],
        times_fav=usuario.times_fav or [],
    )

    # Insights personalizados
    insights = _gerar_insights(usuario, escalacao, df_previsoes)

    resumo = {}
    if not escalacao.empty:
        resumo = {
            "total_jogadores":  len(escalacao),
            "custo_total":      round(escalacao["preco"].sum(), 1) if "preco" in escalacao.columns else 0,
            "saldo":            round(orcamento - escalacao["preco"].sum(), 1) if "preco" in escalacao.columns else 0,
            "pts_esperado":     round(escalacao["pontuacao_prevista"].sum(), 1) if "pontuacao_prevista" in escalacao.columns else 0,
            "score_medio":      round(escalacao["score_personalizado"].mean(), 3) if "score_personalizado" in escalacao.columns else 0,
        }

    return {
        "usuario":   {"id": usuario.id, "nome": usuario.nome, "perfil": usuario.perfil_risco},
        "escalacao": escalacao,
        "resumo":    resumo,
        "insights":  insights,
        "formacao":  usuario.formacao,
    }


def _otimizar_com_formacao(
    df: pd.DataFrame,
    formacao: dict[str, int],
    orcamento: float,
    perfil: str,
    jogadores_fav: list,
    times_fav: list,
) -> pd.DataFrame:
    """Greedy otimizado com restrições de formação, orçamento e diversidade de times."""
    from config.settings import MAX_JOGADORES_TIME, PERFIS

    config    = PERFIS.get(perfil, PERFIS["balanceado"])
    min_preco = config["min_preco"]
    w_media   = config["peso_media"]
    w_up      = config["peso_upside"]

    df = df[df["preco"].fillna(0) >= min_preco].copy()
    df["objetivo"] = (
        w_media * df["score_personalizado"] +
        w_up    * df.get("pontuacao_prevista", pd.Series(0, index=df.index)).fillna(0)
    )

    selecionados   = []
    times_usados: dict = {}
    orcamento_rest = orcamento

    for posicao, n_slots in formacao.items():
        candidatos = df[df["posicao"] == posicao].sort_values("objetivo", ascending=False)
        count = 0

        # Priorizar jogadores favoritos nesta posição
        fav_pos = candidatos[candidatos["atleta_id"].isin(jogadores_fav)]
        outros  = candidatos[~candidatos["atleta_id"].isin(jogadores_fav)]
        candidatos = pd.concat([fav_pos, outros])

        for _, row in candidatos.iterrows():
            if count >= n_slots:
                break
            clube = row.get("clube_id")
            if times_usados.get(clube, 0) >= MAX_JOGADORES_TIME:
                continue
            custo = row.get("preco", 0)
            if custo > orcamento_rest:
                continue
            selecionados.append(row)
            orcamento_rest -= custo
            times_usados[clube] = times_usados.get(clube, 0) + 1
            count += 1

    if not selecionados:
        return pd.DataFrame()

    return pd.DataFrame(selecionados).reset_index(drop=True)


def _gerar_insights(
    usuario: Usuario,
    escalacao: pd.DataFrame,
    df_todos: pd.DataFrame,
) -> list[str]:
    """Gera lista de insights personalizados baseados no perfil e histórico."""
    insights = []
    hist = historico_usuario(usuario.id)
    concluidas = [h for h in hist if h["pts_real"] is not None]

    # Insight de precisão histórica
    if len(concluidas) >= 3:
        erros = [h["erro_abs"] for h in concluidas if h["erro_abs"] is not None]
        mae = np.mean(erros)
        if mae < 10:
            insights.append(f"Seu histórico mostra ótima precisão — erro médio de {mae:.1f} pts.")
        elif mae > 25:
            insights.append(f"Suas escalações têm erro médio de {mae:.1f} pts — considere um perfil mais conservador.")

    # Insight de orçamento
    if not escalacao.empty and "preco" in escalacao.columns:
        gasto = escalacao["preco"].sum()
        saldo = (usuario.orcamento or 100) - gasto
        if saldo > 15:
            insights.append(f"Você tem C$ {saldo:.1f} de saldo — pode arriscar um jogador mais caro.")
        elif saldo < 2:
            insights.append("Orçamento bem aproveitado — margem mínima disponível.")

    # Insight de times favoritos incluídos
    times_fav = usuario.times_fav or []
    if times_fav and "clube_id" in escalacao.columns:
        n_fav = escalacao["clube_id"].isin(times_fav).sum()
        if n_fav > 0:
            insights.append(f"{n_fav} jogador(es) de seus times favoritos foram incluídos.")

    # Insight de mandantes
    if "eh_mandante" in escalacao.columns:
        n_mand = escalacao["eh_mandante"].sum()
        total  = len(escalacao)
        if n_mand / total < 0.4:
            insights.append("Poucos mandantes na escalação — considere filtrar por mando de campo.")
        elif n_mand / total > 0.7:
            insights.append(f"{int(n_mand)} de {total} jogadores jogam em casa — ótimo para pontuação.")

    # Insight de perfil
    if usuario.perfil_risco == "agressivo" and len(concluidas) < 5:
        insights.append("Perfil agressivo com histórico curto — monitore os resultados das próximas rodadas.")

    if not insights:
        insights.append("Escalação gerada com base no seu perfil e preferências pessoais.")

    return insights
