#!/usr/bin/env bash
# =============================================================================
#  render_start.sh — script de inicialização no Render (plano Free)
#  Os dados já foram gerados no docker build.
#  Este script apenas garante consistência e sobe o Streamlit.
# =============================================================================

set -euo pipefail

PORT="${PORT:-10000}"
DISK="${RENDER_DISK_PATH:-/app/data}"

echo "[INFO] Cartola FC — iniciando (porta $PORT)"
echo "[INFO] DATA_DIR: $DISK"

# Garante pastas (caso o container seja iniciado sem o build completo)
mkdir -p "$DISK/raw" "$DISK/processed/sentiment" "$DISK/cache"
mkdir -p "${RENDER_MODELS_PATH:-/app/models}/registry"
mkdir -p "${RENDER_LOGS_PATH:-/app/logs}"

# Se os dados não existirem (fallback de segurança), gera agora
if [ ! -f "$DISK/processed/previsoes.parquet" ]; then
    echo "[INFO] Dados não encontrados — gerando agora (~3 min)..."
    python scripts/popular_dashboard.py --rodadas 14 || \
        echo "[WARN] popular_dashboard falhou — dashboard abrirá com dados limitados."
else
    N=$(python3 -c "import pandas as pd; print(len(pd.read_parquet('$DISK/processed/previsoes.parquet')))" 2>/dev/null || echo "?")
    echo "[INFO] Dados prontos — $N atletas com previsão."
fi

echo "[INFO] Iniciando Streamlit na porta $PORT..."
exec streamlit run dashboard/app.py \
    --server.port                 "$PORT" \
    --server.address              "0.0.0.0" \
    --server.headless             true \
    --server.enableCORS           false \
    --server.enableXsrfProtection false \
    --browser.gatherUsageStats    false
