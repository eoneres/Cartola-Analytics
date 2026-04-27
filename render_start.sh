#!/usr/bin/env bash
# =============================================================================
#  render_start.sh
#  Script executado pelo container ao iniciar no Render.
#  1. Popula dados se o Disk estiver vazio (primeiro deploy ou Disk novo)
#  2. Sobe o Streamlit na porta $PORT (injetada pelo Render)
# =============================================================================

set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }

# Porta injetada pelo Render (padrão 10000); Streamlit usa $PORT
PORT="${PORT:-10000}"

info "=== Cartola FC — Inicialização no Render ==="
info "DATA_DIR   : ${RENDER_DISK_PATH:-./data}"
info "MODELS_DIR : ${RENDER_MODELS_PATH:-./models}"
info "PORT       : $PORT"

# ── Garante que as pastas do Disk existem ─────────────────────────────────────
DISK="${RENDER_DISK_PATH:-./data}"
mkdir -p "$DISK/raw" "$DISK/processed/sentiment" "$DISK/cache"
mkdir -p "${RENDER_MODELS_PATH:-./models}/registry"
mkdir -p "${RENDER_LOGS_PATH:-./logs}"

# ── Popula dados se previsões não existirem ───────────────────────────────────
PREV="$DISK/processed/previsoes.parquet"

if [ ! -f "$PREV" ]; then
    info "Primeiro deploy detectado — populando dados (pode levar ~2 min)..."
    python scripts/popular_dashboard.py --rodadas 14 || {
        warn "popular_dashboard.py falhou — dashboard abrirá com dados limitados."
    }
else
    info "Dados encontrados no Disk — pulando população inicial."
    info "Previsões: $(python3 -c "import pandas as pd; print(len(pd.read_parquet('$PREV')), 'atletas')" 2>/dev/null || echo 'N/A')"
fi

# ── Sobe o Streamlit ──────────────────────────────────────────────────────────
info "Iniciando Streamlit na porta $PORT..."
exec streamlit run dashboard/app.py \
    --server.port        "$PORT" \
    --server.address     "0.0.0.0" \
    --server.headless    true \
    --server.enableCORS  false \
    --server.enableXsrfProtection false \
    --browser.gatherUsageStats false
