"""
sentiment/collector.py
Coleta de textos de redes sociais e portais de notícias esportivas.

Fontes suportadas:
  - Twitter/X  (via API v2 — requer Bearer Token)
  - Reddit     (via API pública JSON — sem autenticação)
  - Notícias   (web scraping de Globo Esporte e UOL Esporte)

Configuração (variáveis de ambiente ou .env):
  TWITTER_BEARER_TOKEN=...

Uso sem credenciais:
  O módulo funciona em modo degradado usando apenas Reddit + notícias.
"""

import logging
import os
import time
from datetime import datetime, timedelta

import requests
from bs4 import BeautifulSoup

from config.settings import DATA_DIR, HTTP_BACKOFF, HTTP_RETRIES, HTTP_TIMEOUT

logger = logging.getLogger(__name__)

RAW_SOCIAL_DIR = DATA_DIR / "raw" / "social"
RAW_SOCIAL_DIR.mkdir(parents=True, exist_ok=True)

# ── Sessão HTTP compartilhada ─────────────────────────────────────────────────
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def _sessao() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=HTTP_RETRIES,
        backoff_factor=HTTP_BACKOFF,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({"User-Agent": "CartolaAnalyzer/2.0 (research)"})
    return s


_SESSION = _sessao()


# ══════════════════════════════════════════════════════════════════════════════
# TWITTER / X  (API v2)
# ══════════════════════════════════════════════════════════════════════════════

TWITTER_SEARCH_URL = "https://api.twitter.com/2/tweets/search/recent"
TWITTER_BEARER    = os.getenv("TWITTER_BEARER_TOKEN", "")


def _twitter_headers() -> dict:
    if not TWITTER_BEARER:
        raise EnvironmentError(
            "TWITTER_BEARER_TOKEN não configurado. "
            "Defina a variável de ambiente ou adicione ao .env"
        )
    return {"Authorization": f"Bearer {TWITTER_BEARER}"}


def coletar_tweets(
    jogador: str,
    max_resultados: int = 100,
    horas_atras: int = 48,
) -> list[dict]:
    """
    Busca tweets recentes sobre um jogador.

    Parâmetros
    ----------
    jogador       : nome/apelido do jogador (ex: "Endrick")
    max_resultados: máximo de tweets (10–100 no plano gratuito)
    horas_atras   : janela de busca em horas

    Retorna
    -------
    Lista de dicts com {id, text, created_at, public_metrics}
    """
    if not TWITTER_BEARER:
        logger.warning("Twitter não configurado — pulando coleta de tweets.")
        return []

    inicio = (datetime.utcnow() - timedelta(hours=horas_atras)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    query = (
        f'"{jogador}" (Cartola OR Brasileirão OR Brasileirao OR futebol) '
        f"-is:retweet lang:pt"
    )

    params = {
        "query":        query,
        "max_results":  min(max_resultados, 100),
        "start_time":   inicio,
        "tweet.fields": "created_at,public_metrics,lang",
    }

    try:
        resp = _SESSION.get(
            TWITTER_SEARCH_URL,
            headers=_twitter_headers(),
            params=params,
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        tweets = data.get("data", [])
        logger.info("Twitter: %d tweets coletados para '%s'.", len(tweets), jogador)
        return tweets
    except Exception as e:
        logger.error("Erro ao coletar tweets para '%s': %s", jogador, e)
        return []


# ══════════════════════════════════════════════════════════════════════════════
# REDDIT  (API JSON pública — sem autenticação)
# ══════════════════════════════════════════════════════════════════════════════

REDDIT_SUBREDDITS = ["futebol", "cartoleiros", "brasileirao"]
REDDIT_SEARCH_URL = "https://www.reddit.com/r/{sub}/search.json"


def coletar_reddit(
    jogador: str,
    max_posts: int = 50,
    dias_atras: int = 7,
) -> list[dict]:
    """
    Busca posts e comentários no Reddit sobre um jogador.

    Retorna
    -------
    Lista de dicts com {id, title, selftext, score, created_utc, subreddit}
    """
    resultados = []
    limite_ts  = time.time() - dias_atras * 86400

    for sub in REDDIT_SUBREDDITS:
        url = REDDIT_SEARCH_URL.format(sub=sub)
        params = {
            "q":       jogador,
            "sort":    "new",
            "limit":   max_posts,
            "type":    "link",
            "restrict_sr": True,
        }
        try:
            resp = _SESSION.get(url, params=params, timeout=HTTP_TIMEOUT)
            resp.raise_for_status()
            posts = resp.json().get("data", {}).get("children", [])
            for p in posts:
                d = p.get("data", {})
                if d.get("created_utc", 0) < limite_ts:
                    continue
                resultados.append({
                    "id":          d.get("id"),
                    "title":       d.get("title", ""),
                    "selftext":    d.get("selftext", ""),
                    "score":       d.get("score", 0),
                    "created_utc": d.get("created_utc"),
                    "subreddit":   sub,
                    "fonte":       "reddit",
                })
            time.sleep(1.0)   # respeitar rate limit do Reddit
        except Exception as e:
            logger.warning("Reddit/%s — erro para '%s': %s", sub, jogador, e)

    logger.info("Reddit: %d posts coletados para '%s'.", len(resultados), jogador)
    return resultados


# ══════════════════════════════════════════════════════════════════════════════
# NOTÍCIAS  (web scraping)
# ══════════════════════════════════════════════════════════════════════════════

_FONTES_NOTICIAS = [
    {
        "nome":     "ge.globo",
        "url":      "https://ge.globo.com/busca/?q={query}&species=noticia",
        "seletor":  "div.widget--info__text-container",
        "titulo":   "h2.widget--info__title",
        "resumo":   "p.widget--info__description",
    },
    {
        "nome":     "uol.esporte",
        "url":      "https://busca.uol.com.br/result.html?q={query}&site=esporte",
        "seletor":  "li.results__item",
        "titulo":   "h2",
        "resumo":   "p",
    },
]


def coletar_noticias(jogador: str, max_noticias: int = 20) -> list[dict]:
    """
    Faz scraping de manchetes e resumos de notícias sobre o jogador.

    Retorna
    -------
    Lista de dicts com {titulo, resumo, fonte, url}
    """
    resultados = []
    query = requests.utils.quote(jogador)

    for fonte in _FONTES_NOTICIAS:
        url = fonte["url"].format(query=query)
        try:
            resp = _SESSION.get(url, timeout=HTTP_TIMEOUT)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            itens = soup.select(fonte["seletor"])[:max_noticias]
            for item in itens:
                titulo_el = item.select_one(fonte["titulo"])
                resumo_el = item.select_one(fonte["resumo"])
                titulo = titulo_el.get_text(strip=True) if titulo_el else ""
                resumo = resumo_el.get_text(strip=True) if resumo_el else ""
                if not titulo:
                    continue
                resultados.append({
                    "titulo":  titulo,
                    "resumo":  resumo,
                    "fonte":   fonte["nome"],
                    "url":     url,
                    "jogador": jogador,
                })
            time.sleep(1.5)
        except Exception as e:
            logger.warning("Notícias/%s — erro para '%s': %s", fonte["nome"], jogador, e)

    logger.info("Notícias: %d artigos coletados para '%s'.", len(resultados), jogador)
    return resultados


# ══════════════════════════════════════════════════════════════════════════════
# Interface unificada
# ══════════════════════════════════════════════════════════════════════════════

def coletar_textos_jogador(
    jogador: str,
    incluir_twitter: bool = True,
    incluir_reddit:  bool = True,
    incluir_noticias: bool = True,
) -> list[dict]:
    """
    Coleta todos os textos disponíveis sobre um jogador de todas as fontes.

    Retorna lista unificada com campo 'texto' (conteúdo principal)
    e 'fonte' (twitter | reddit | noticias).
    """
    textos = []

    if incluir_twitter:
        for t in coletar_tweets(jogador):
            textos.append({
                "jogador": jogador,
                "texto":   t.get("text", ""),
                "fonte":   "twitter",
                "meta":    t,
            })

    if incluir_reddit:
        for p in coletar_reddit(jogador):
            conteudo = f"{p.get('title','')} {p.get('selftext','')}".strip()
            textos.append({
                "jogador": jogador,
                "texto":   conteudo,
                "fonte":   "reddit",
                "meta":    p,
            })

    if incluir_noticias:
        for n in coletar_noticias(jogador):
            conteudo = f"{n.get('titulo','')} {n.get('resumo','')}".strip()
            textos.append({
                "jogador": jogador,
                "texto":   conteudo,
                "fonte":   "noticias",
                "meta":    n,
            })

    logger.info(
        "Total de textos coletados para '%s': %d (Twitter=%d, Reddit=%d, Notícias=%d)",
        jogador,
        len(textos),
        sum(1 for t in textos if t["fonte"] == "twitter"),
        sum(1 for t in textos if t["fonte"] == "reddit"),
        sum(1 for t in textos if t["fonte"] == "noticias"),
    )
    return textos
