"""
data_collection/cartola_api.py
Cliente para a API pública do Cartola FC.

A API pública do Cartola não exige autenticação para endpoints de
leitura (mercado, atletas, partidas). Este módulo encapsula todas
as chamadas com retry automático e cache local em JSON.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config.settings import (
    CARTOLA_ATLETAS_URL,
    CARTOLA_MERCADO_URL,
    CARTOLA_PARTIDAS_URL,
    CARTOLA_PONTUADOS_URL,
    DATA_DIR,
    HTTP_BACKOFF,
    HTTP_RETRIES,
    HTTP_TIMEOUT,
)

logger = logging.getLogger(__name__)

CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _build_session() -> requests.Session:
    """Cria sessão HTTP com retry automático e headers padrão."""
    session = requests.Session()
    retry = Retry(
        total=HTTP_RETRIES,
        backoff_factor=HTTP_BACKOFF,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.headers.update({
        "User-Agent": "CartolaFC-Analyzer/1.0",
        "Accept": "application/json",
    })
    return session


_SESSION = _build_session()


def _get(url: str, params: dict | None = None) -> dict | list:
    """GET com timeout e tratamento de erros."""
    try:
        resp = _SESSION.get(url, params=params, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        logger.error("HTTP %s ao acessar %s: %s", resp.status_code, url, e)
        raise
    except requests.exceptions.ConnectionError as e:
        logger.error("Falha de conexão em %s: %s", url, e)
        raise
    except requests.exceptions.Timeout:
        logger.error("Timeout ao acessar %s", url)
        raise


def _cache_path(nome: str) -> Path:
    return CACHE_DIR / f"{nome}.json"


def _salvar_cache(nome: str, dados: Any) -> None:
    path = _cache_path(nome)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)
    logger.debug("Cache salvo: %s", path)


def _carregar_cache(nome: str) -> Any | None:
    path = _cache_path(nome)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


# ── Endpoints públicos ────────────────────────────────────────────────────────

def buscar_status_mercado(usar_cache: bool = False) -> dict:
    """
    Retorna o status atual do mercado (rodada, status, fechamento).

    Exemplo de resposta:
        {
          "rodada_atual": 14,
          "status_mercado": 1,   # 1=aberto, 2=fechado, 4=em manutenção
          "fechamento": {"timestamp": 1234567890}
        }
    """
    if usar_cache:
        cached = _carregar_cache("mercado_status")
        if cached:
            return cached

    logger.info("Buscando status do mercado...")
    dados = _get(CARTOLA_MERCADO_URL)
    _salvar_cache("mercado_status", dados)
    return dados


def buscar_atletas_mercado(usar_cache: bool = False) -> list[dict]:
    """
    Retorna todos os atletas disponíveis no mercado com preço e média.

    Campos relevantes por atleta:
        atleta_id, apelido, posicao_id, clube_id,
        preco_num, media_num, pontos_num, jogos_num, status_id
    """
    if usar_cache:
        cached = _carregar_cache("atletas_mercado")
        if cached:
            return cached

    logger.info("Buscando atletas do mercado...")
    dados = _get(CARTOLA_ATLETAS_URL)
    atletas = dados.get("atletas", [])
    _salvar_cache("atletas_mercado", atletas)
    logger.info("%d atletas encontrados.", len(atletas))
    return atletas


def buscar_atletas_pontuados(rodada: int, usar_cache: bool = True) -> list[dict]:
    """
    Retorna pontuação detalhada dos atletas em uma rodada específica.

    Inclui scouts individuais (gols, assistências, desarmes, etc.).
    """
    cache_key = f"pontuados_r{rodada}"
    if usar_cache:
        cached = _carregar_cache(cache_key)
        if cached:
            return cached

    logger.info("Buscando atletas pontuados — rodada %d...", rodada)
    dados = _get(CARTOLA_PONTUADOS_URL, params={"rodada": rodada})
    atletas = list(dados.get("atletas", {}).values())
    _salvar_cache(cache_key, atletas)
    logger.info("%d atletas pontuados na rodada %d.", len(atletas), rodada)
    return atletas


def buscar_partidas(rodada: int, usar_cache: bool = True) -> list[dict]:
    """
    Retorna as partidas de uma rodada com mandante/visitante e placar.
    """
    cache_key = f"partidas_r{rodada}"
    if usar_cache:
        cached = _carregar_cache(cache_key)
        if cached:
            return cached

    logger.info("Buscando partidas — rodada %d...", rodada)
    dados = _get(f"{CARTOLA_PARTIDAS_URL}/{rodada}")
    partidas = dados.get("partidas", [])
    _salvar_cache(cache_key, partidas)
    return partidas


def buscar_historico_completo(rodada_inicio: int, rodada_fim: int) -> list[dict]:
    """
    Coleta histórico de pontuações de múltiplas rodadas com pausa entre
    requisições para não sobrecarregar a API.

    Retorna lista unificada com campo 'rodada' em cada registro.
    """
    todos = []
    for rodada in range(rodada_inicio, rodada_fim + 1):
        try:
            atletas = buscar_atletas_pontuados(rodada, usar_cache=True)
            for a in atletas:
                a["rodada"] = rodada
            todos.extend(atletas)
            logger.info("Rodada %d: %d atletas coletados.", rodada, len(atletas))
        except Exception as e:
            logger.warning("Erro na rodada %d: %s — pulando.", rodada, e)
        time.sleep(0.5)   # pausa cortês entre requisições

    logger.info("Total coletado: %d registros.", len(todos))
    return todos
