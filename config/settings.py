"""
config/settings.py
Configurações centrais do sistema Cartola FC MVP.
Edite este arquivo para ajustar o comportamento do sistema.
"""

import os
from pathlib import Path

# ── Diretórios ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

# No Render: RENDER_DISK_PATH aponta para o Persistent Disk (/var/data).
# Localmente: usa data/ e models/ dentro do projeto.
DATA_DIR   = Path(os.environ.get("RENDER_DISK_PATH",   str(BASE_DIR / "data")))
MODELS_DIR = Path(os.environ.get("RENDER_MODELS_PATH", str(BASE_DIR / "models")))
LOGS_DIR   = Path(os.environ.get("RENDER_LOGS_PATH",   str(BASE_DIR / "logs")))

for d in [DATA_DIR, MODELS_DIR, LOGS_DIR, DATA_DIR / "processed", DATA_DIR / "processed" / "sentiment"]:
    d.mkdir(parents=True, exist_ok=True)

# ── Temporada ─────────────────────────────────────────────────────────────────
TEMPORADA_ATUAL = 2025
RODADA_INICIAL  = 1

# ── API do Cartola FC (endpoints públicos conhecidos) ─────────────────────────
CARTOLA_BASE_URL    = "https://api.cartola.globo.com"
CARTOLA_MERCADO_URL = f"{CARTOLA_BASE_URL}/mercado/status"
CARTOLA_ATLETAS_URL = f"{CARTOLA_BASE_URL}/atletas/mercado"
# Endpoint correto: path param  /atletas/pontuados/{rodada}
# O formato legado ?rodada=N retorna corpo vazio desde 2024
CARTOLA_PONTUADOS_URL = f"{CARTOLA_BASE_URL}/atletas/pontuados"
CARTOLA_PARTIDAS_URL  = f"{CARTOLA_BASE_URL}/partidas"

# Timeout e retries para requisições HTTP
HTTP_TIMEOUT = 15       # segundos
HTTP_RETRIES = 3
HTTP_BACKOFF  = 2.0     # fator de espera exponencial entre retries

# ── Posições do Cartola ───────────────────────────────────────────────────────
POSICOES = {
    1: "Goleiro",
    2: "Lateral",
    3: "Zagueiro",
    4: "Meia",
    5: "Atacante",
    6: "Técnico",
}

# Slots da escalação padrão (4-3-3)
# Formação padrão (usada como fallback)
ESCALACAO_SLOTS = {
    "Goleiro":  1,
    "Lateral":  2,
    "Zagueiro": 2,
    "Meia":     3,
    "Atacante": 3,
    "Técnico":  1,
}

# ── Todas as formações disponíveis no Cartola FC ──────────────────────────────
# Formato: "nome_exibição": {posição: qtd, ...}
# Regras do Cartola: 1 GOL, 1 TEC, 12 jogadores de linha no total
# Linha = Laterais + Zagueiros + Meias + Atacantes = 8 titulares + 3 reservas
# O sistema usa apenas os titulares (sem banco): 1 GOL + 8 linha + 1 TEC = 10
FORMACOES = {
    # ── 3 defensores ──────────────────────────────────────────────────────────
    "3-4-3": {
        "label":    "3-4-3",
        "descricao":"Ofensiva com 3 zagueiros, 4 meias e 3 atacantes",
        "slots": {"Goleiro": 1, "Lateral": 0, "Zagueiro": 3, "Meia": 4, "Atacante": 3, "Técnico": 1},
    },
    "3-5-2": {
        "label":    "3-5-2",
        "descricao":"Equilíbrio com 3 zagueiros, 5 meias e 2 atacantes",
        "slots": {"Goleiro": 1, "Lateral": 0, "Zagueiro": 3, "Meia": 5, "Atacante": 2, "Técnico": 1},
    },
    "3-6-1": {
        "label":    "3-6-1",
        "descricao":"Ultra ofensiva de meio com 6 meias e 1 atacante",
        "slots": {"Goleiro": 1, "Lateral": 0, "Zagueiro": 3, "Meia": 6, "Atacante": 1, "Técnico": 1},
    },
    # ── 4 defensores ──────────────────────────────────────────────────────────
    "4-3-3": {
        "label":    "4-3-3",
        "descricao":"Clássica com 2 laterais, 2 zagueiros, 3 meias e 3 atacantes",
        "slots": {"Goleiro": 1, "Lateral": 2, "Zagueiro": 2, "Meia": 3, "Atacante": 3, "Técnico": 1},
    },
    "4-4-2": {
        "label":    "4-4-2",
        "descricao":"Equilibrada com 4 defensores, 4 meias e 2 atacantes",
        "slots": {"Goleiro": 1, "Lateral": 2, "Zagueiro": 2, "Meia": 4, "Atacante": 2, "Técnico": 1},
    },
    "4-5-1": {
        "label":    "4-5-1",
        "descricao":"Defensiva com 5 meias e 1 atacante",
        "slots": {"Goleiro": 1, "Lateral": 2, "Zagueiro": 2, "Meia": 5, "Atacante": 1, "Técnico": 1},
    },
    "4-2-4": {
        "label":    "4-2-4",
        "descricao":"Super ofensiva com 4 atacantes e apenas 2 meias",
        "slots": {"Goleiro": 1, "Lateral": 2, "Zagueiro": 2, "Meia": 2, "Atacante": 4, "Técnico": 1},
    },
    # ── 5 defensores ──────────────────────────────────────────────────────────
    "5-3-2": {
        "label":    "5-3-2",
        "descricao":"Defensiva com 3 laterais, 2 zagueiros, 3 meias e 2 atacantes",
        "slots": {"Goleiro": 1, "Lateral": 3, "Zagueiro": 2, "Meia": 3, "Atacante": 2, "Técnico": 1},
    },
    "5-4-1": {
        "label":    "5-4-1",
        "descricao":"Ultra defensiva com 5 defensores, 4 meias e 1 atacante",
        "slots": {"Goleiro": 1, "Lateral": 3, "Zagueiro": 2, "Meia": 4, "Atacante": 1, "Técnico": 1},
    },
    "5-2-3": {
        "label":    "5-2-3",
        "descricao":"5 defensores, 2 meias e 3 atacantes",
        "slots": {"Goleiro": 1, "Lateral": 3, "Zagueiro": 2, "Meia": 2, "Atacante": 3, "Técnico": 1},
    },
}

# Formação padrão
FORMACAO_PADRAO = "4-3-3"

# ── Scout — pesos para pontuação manual (fallback sem API) ─────────────────────
SCOUTS_PESOS = {
    "gol":              8.0,
    "assistencia":      5.0,
    "finalizacao_trave":3.0,
    "finalizacao_out":  1.2,
    "finalizacao_def":  1.5,
    "desarme":          1.3,
    "interceptacao":    1.8,
    "falta_cometida":  -0.3,
    "cartao_amarelo":  -1.0,
    "cartao_vermelho": -3.0,
    "gol_sofrido":     -1.0,  # para goleiros/defensores
    "jogo_sem_sofrer": 5.0,   # clean sheet
    "penalti_perdido": -2.0,
    "penalti_cometido":-2.0,
}

# ── Score composto ────────────────────────────────────────────────────────────
# Soma deve ser 1.0
SCORE_PESOS = {
    "media_historica":  0.35,
    "forma_recente":    0.35,   # últimas 5 rodadas
    "fator_adversario": 0.20,   # dificuldade do próximo jogo
    "sentimento":       0.10,   # será ativado na Fase 2
}

# Janela de rodadas para "forma recente"
JANELA_FORMA = 5

# ── Modelo de Machine Learning ────────────────────────────────────────────────
MODEL_TYPE = "xgboost"   # opções: "xgboost", "random_forest", "linear"

XGBOOST_PARAMS = {
    "n_estimators":      300,
    "max_depth":         5,
    "learning_rate":     0.05,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "random_state":      42,
    "n_jobs":           -1,
}

RANDOM_FOREST_PARAMS = {
    "n_estimators":   200,
    "max_depth":        8,
    "random_state":    42,
    "n_jobs":          -1,
}

# Validação cruzada temporal (Time Series Split)
CV_N_SPLITS = 5

# ── Otimização de escalação ───────────────────────────────────────────────────
ORCAMENTO_PADRAO   = 100.0   # C$ (cartoletas)
MAX_JOGADORES_TIME = 5       # máximo de jogadores do mesmo clube

# Perfis de risco
PERFIS = {
    "conservador": {
        "peso_media":    0.7,
        "peso_upside":   0.3,
        "min_preco":     5.0,   # evitar jogadores baratos/instáveis
    },
    "balanceado": {
        "peso_media":    0.5,
        "peso_upside":   0.5,
        "min_preco":     3.0,
    },
    "agressivo": {
        "peso_media":    0.3,
        "peso_upside":   0.7,
        "min_preco":     1.0,   # aceita jogadores de baixo preço com upside
    },
}

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL  = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
