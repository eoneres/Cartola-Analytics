"""
config/settings.py
Configurações centrais do sistema Cartola FC MVP.
Edite este arquivo para ajustar o comportamento do sistema.
"""

from pathlib import Path

# ── Diretórios ────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR   = BASE_DIR / "logs"

for d in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Temporada ─────────────────────────────────────────────────────────────────
TEMPORADA_ATUAL = 2025
RODADA_INICIAL  = 1

# ── API do Cartola FC (endpoints públicos conhecidos) ─────────────────────────
CARTOLA_BASE_URL    = "https://api.cartola.globo.com"
CARTOLA_MERCADO_URL = f"{CARTOLA_BASE_URL}/mercado/status"
CARTOLA_ATLETAS_URL = f"{CARTOLA_BASE_URL}/atletas/mercado"
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
ESCALACAO_SLOTS = {
    "Goleiro":   1,
    "Lateral":   2,
    "Zagueiro":  2,
    "Meia":      3,
    "Atacante":  3,
    "Técnico":   1,
}

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
