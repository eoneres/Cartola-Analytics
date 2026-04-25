"""
sentiment/analyzer.py
Pipeline de análise de sentimento com dois modos:

  MODO 1 — VADER (baseline, sem GPU, rápido)
    Léxico baseado em regras adaptado para português via tradução.
    Ideal para volume alto de textos ou ambientes sem GPU.

  MODO 2 — BERT (avançado, recomendado)
    Usa o modelo "neuralmind/bert-base-portuguese-cased" ou
    "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
    — ambos pré-treinados em português.
    Requer: transformers, torch (ou tensorflow).

O modo é selecionado automaticamente com base na disponibilidade
dos pacotes; pode ser forçado via parâmetro `modo`.

Saída por texto:
  {
    "texto":        str,
    "sentimento":   "positivo" | "neutro" | "negativo",
    "score":        float,   # -1.0 (muito negativo) a +1.0 (muito positivo)
    "confianca":    float,   # 0.0 a 1.0
    "modo":         "vader" | "bert",
  }
"""

import logging
import re
import unicodedata
from functools import lru_cache
from typing import Literal

logger = logging.getLogger(__name__)

# ── Pré-processamento ─────────────────────────────────────────────────────────

_URL_RE      = re.compile(r"https?://\S+")
_MENTION_RE  = re.compile(r"@\w+")
_HASHTAG_RE  = re.compile(r"#(\w+)")
_MULTI_RE    = re.compile(r"\s{2,}")
_EMOJI_RE    = re.compile(
    "["
    u"\U0001F600-\U0001F64F"
    u"\U0001F300-\U0001F5FF"
    u"\U0001F680-\U0001F6FF"
    u"\U0001F1E0-\U0001F1FF"
    "]+",
    flags=re.UNICODE,
)


def preprocessar(texto: str, manter_hashtag_texto: bool = True) -> str:
    """
    Limpa o texto para análise de sentimento.

    - Remove URLs, menções, caracteres de controle
    - Expande hashtags (#BomJogo → "BomJogo")
    - Normaliza espaços
    """
    t = _URL_RE.sub(" ", texto)
    t = _MENTION_RE.sub(" ", t)
    if manter_hashtag_texto:
        t = _HASHTAG_RE.sub(r"\1 ", t)
    else:
        t = _HASHTAG_RE.sub(" ", t)
    t = _EMOJI_RE.sub(" ", t)
    t = unicodedata.normalize("NFKC", t)
    t = _MULTI_RE.sub(" ", t).strip()
    return t


# ══════════════════════════════════════════════════════════════════════════════
# MODO 1 — VADER adaptado para português
# ══════════════════════════════════════════════════════════════════════════════

# Léxico de sentimento em português (futebol/esportes)
# Estrutura: palavra → score (-1 a +1)
_LEXICO_PT: dict[str, float] = {
    # Positivos fortes
    "gol": 0.7, "golaço": 0.9, "hat-trick": 0.9, "campeão": 0.9,
    "incrível": 0.8, "sensacional": 0.9, "excelente": 0.8, "ótimo": 0.7,
    "brilhante": 0.85, "craque": 0.8, "fenomenal": 0.9, "espetacular": 0.9,
    "top": 0.6, "decisivo": 0.7, "impecável": 0.8, "perfeito": 0.85,
    "melhorou": 0.5, "evoluiu": 0.5, "artilheiro": 0.7, "boa": 0.4,
    "bom": 0.4, "bem": 0.3, "aprovado": 0.5, "contratação": 0.3,

    # Positivos moderados
    "assistência": 0.5, "passou": 0.3, "jogou": 0.2, "participou": 0.2,
    "marcou": 0.5, "contribuiu": 0.4, "recuperou": 0.4, "voltou": 0.2,

    # Negativos fortes
    "horrível": -0.9, "terrível": -0.9, "péssimo": -0.85, "lixo": -0.9,
    "inútil": -0.85, "fraco": -0.6, "decepcionante": -0.7, "vergonha": -0.8,
    "ridículo": -0.8, "acabou": -0.6, "lesão": -0.7, "lesionado": -0.7,
    "suspenso": -0.6, "expulso": -0.8, "cartão": -0.4, "falhando": -0.6,
    "errou": -0.5, "perdeu": -0.5, "zerou": -0.6, "zerou": -0.7,
    "ruim": -0.6, "mal": -0.4, "pior": -0.7, "caindo": -0.5,

    # Negativos moderados
    "desclassificado": -0.7, "rebaixado": -0.8, "crise": -0.6,
    "contundido": -0.6, "dúvida": -0.3, "reserva": -0.3, "banco": -0.3,

    # Negações (invertidas no código)
    "não": -1.0, "nunca": -1.0, "jamais": -1.0, "nem": -0.8,
}

_INTENSIFICADORES = {"muito": 1.3, "demais": 1.3, "super": 1.4, "mega": 1.4,
                     "extremamente": 1.5, "bastante": 1.2, "mais": 1.1}
_NEGACOES = {"não", "nunca", "jamais", "nem", "nada", "nenhum"}


def _vader_pt(texto: str) -> dict:
    """Análise de sentimento baseada em léxico português."""
    palavras = preprocessar(texto).lower().split()
    scores   = []
    negado   = False
    intensif = 1.0

    for i, palavra in enumerate(palavras):
        if palavra in _NEGACOES:
            negado = True
            continue
        if palavra in _INTENSIFICADORES:
            intensif = _INTENSIFICADORES[palavra]
            continue

        score = _LEXICO_PT.get(palavra, 0.0)
        if score != 0.0:
            if negado:
                score = -score * 0.75
                negado = False
            score *= intensif
            scores.append(max(-1.0, min(1.0, score)))
            intensif = 1.0
        else:
            negado   = False
            intensif = 1.0

    if not scores:
        return {"score": 0.0, "confianca": 0.3, "modo": "vader"}

    score_final = sum(scores) / len(scores)
    confianca   = min(0.85, 0.3 + 0.1 * len(scores))
    return {"score": round(score_final, 4), "confianca": round(confianca, 4), "modo": "vader"}


# ══════════════════════════════════════════════════════════════════════════════
# MODO 2 — BERT (Transformers)
# ══════════════════════════════════════════════════════════════════════════════

# Modelos recomendados em ordem de preferência
_BERT_MODELOS = [
    "lxyuan/distilbert-base-multilingual-cased-sentiments-student",  # leve, multilingual
    "neuralmind/bert-base-portuguese-cased",                          # BERTimbau PT
]


@lru_cache(maxsize=1)
def _carregar_pipeline_bert(modelo: str):
    """Carrega o pipeline de sentimento do Hugging Face (com cache)."""
    try:
        from transformers import pipeline as hf_pipeline
    except ImportError:
        raise ImportError("transformers não instalado. Use: python main.py sentiment --modo vader")

    logger.info("Carregando modelo BERT: %s", modelo)
    pipe = hf_pipeline(
        "text-classification",
        model=modelo,
        tokenizer=modelo,
        max_length=512,
        truncation=True,
        device=-1,      # CPU por padrão; use device=0 para GPU CUDA
    )
    logger.info("Modelo BERT carregado.")
    return pipe


def _bert_disponivel() -> bool:
    """Verifica disponibilidade sem importar o módulo completo (evita erros de torchvision)."""
    try:
        import importlib.util
        return (
            importlib.util.find_spec("transformers") is not None and
            importlib.util.find_spec("torch") is not None
        )
    except Exception:
        return False


_LABEL_MAP = {
    # distilbert multilingual
    "positive": 1.0, "Positive": 1.0,
    "neutral":  0.0, "Neutral":  0.0,
    "negative": -1.0, "Negative": -1.0,
    # outros modelos
    "LABEL_2": 1.0,   # positivo
    "LABEL_1": 0.0,   # neutro
    "LABEL_0": -1.0,  # negativo
    "POS": 1.0, "NEU": 0.0, "NEG": -1.0,
}


def _bert_pt(texto: str, modelo: str) -> dict:
    """Classifica sentimento via BERT."""
    pipe = _carregar_pipeline_bert(modelo)
    texto_limpo = preprocessar(texto)[:512]

    try:
        resultado = pipe(texto_limpo)[0]
        label = resultado["label"]
        score_raw = _LABEL_MAP.get(label, 0.0)

        # Se o modelo retorna apenas positivo/negativo sem neutro,
        # inferir neutro quando a confiança for baixa
        confianca = float(resultado["score"])
        if abs(score_raw) > 0 and confianca < 0.55:
            score_raw = score_raw * 0.4   # reduz magnitude

        return {
            "score":     round(score_raw * confianca, 4),
            "confianca": round(confianca, 4),
            "modo":      "bert",
            "label_raw": label,
        }
    except Exception as e:
        logger.error("Erro no BERT para texto '%s...': %s", texto[:50], e)
        return {"score": 0.0, "confianca": 0.0, "modo": "bert_erro"}


# ══════════════════════════════════════════════════════════════════════════════
# Interface pública
# ══════════════════════════════════════════════════════════════════════════════

def analisar(
    texto: str,
    modo: Literal["auto", "vader", "bert"] = "auto",
    modelo_bert: str | None = None,
) -> dict:
    """
    Analisa o sentimento de um texto.

    Parâmetros
    ----------
    texto       : texto bruto (tweet, post, notícia)
    modo        : "auto" escolhe BERT se disponível, senão VADER
    modelo_bert : nome do modelo HuggingFace (usa lista interna se None)

    Retorna
    -------
    {
      "texto":      str (preprocessado),
      "sentimento": "positivo" | "neutro" | "negativo",
      "score":      float,       # -1 a +1
      "confianca":  float,       # 0 a 1
      "modo":       "vader" | "bert",
    }
    """
    texto_limpo = preprocessar(texto)
    if not texto_limpo:
        return {
            "texto": "", "sentimento": "neutro",
            "score": 0.0, "confianca": 0.0, "modo": "vazio",
        }

    usar_bert = (modo == "bert") or (modo == "auto" and _bert_disponivel())

    if usar_bert:
        modelo = modelo_bert or _BERT_MODELOS[0]
        try:
            resultado = _bert_pt(texto_limpo, modelo)
        except Exception as e:
            logger.warning("BERT falhou (%s), caindo para VADER.", e)
            resultado = _vader_pt(texto_limpo)
    else:
        resultado = _vader_pt(texto_limpo)

    score = resultado["score"]
    if score > 0.15:
        sentimento = "positivo"
    elif score < -0.15:
        sentimento = "negativo"
    else:
        sentimento = "neutro"

    return {
        "texto":      texto_limpo,
        "sentimento": sentimento,
        "score":      score,
        "confianca":  resultado.get("confianca", 0.5),
        "modo":       resultado.get("modo", "vader"),
    }


def analisar_lote(
    textos: list[str],
    modo: Literal["auto", "vader", "bert"] = "auto",
    batch_size: int = 32,
) -> list[dict]:
    """
    Analisa uma lista de textos em lote.

    Para BERT, processa em batches para eficiência de memória.
    Para VADER, processa sequencialmente (é rápido o suficiente).
    """
    usar_bert = (modo == "bert") or (modo == "auto" and _bert_disponivel())

    if usar_bert and len(textos) > 1:
        return _bert_lote(textos, batch_size)

    return [analisar(t, modo="vader") for t in textos]


def _bert_lote(textos: list[str], batch_size: int) -> list[dict]:
    """Processa textos em batches via BERT."""
    modelo = _BERT_MODELOS[0]
    try:
        pipe = _carregar_pipeline_bert(modelo)
    except Exception:
        logger.warning("Não foi possível carregar BERT — usando VADER em lote.")
        return [analisar(t, modo="vader") for t in textos]

    resultados = []
    for i in range(0, len(textos), batch_size):
        batch   = textos[i : i + batch_size]
        limpos  = [preprocessar(t)[:512] for t in batch]

        try:
            preds = pipe(limpos)
        except Exception as e:
            logger.warning("Erro no batch BERT [%d:%d]: %s", i, i + batch_size, e)
            preds = [{"label": "neutral", "score": 0.5}] * len(batch)

        for texto_orig, pred in zip(batch, preds):
            score_raw = _LABEL_MAP.get(pred["label"], 0.0)
            confianca = float(pred["score"])
            score     = round(score_raw * confianca, 4)
            resultados.append({
                "texto":      preprocessar(texto_orig),
                "sentimento": "positivo" if score > 0.15 else ("negativo" if score < -0.15 else "neutro"),
                "score":      score,
                "confianca":  round(confianca, 4),
                "modo":       "bert",
            })

    return resultados
