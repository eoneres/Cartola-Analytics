# =============================================================================
#  Dockerfile — Cartola FC Analytics
#  Plano Free do Render: sem Disk persistente.
#  Os dados são gerados no BUILD (docker build) e ficam dentro da imagem.
#  A cada novo deploy os dados são regenerados automaticamente.
# =============================================================================

FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Paths internos ao container (sobrescritos pelo render.yaml)
    RENDER_DISK_PATH=/app/data \
    RENDER_MODELS_PATH=/app/models \
    RENDER_LOGS_PATH=/app/logs

WORKDIR /app

# Dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git \
    && rm -rf /var/lib/apt/lists/*

# Instala dependências Python
COPY requirements-render.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements-render.txt

# Copia o projeto
COPY . .

# Cria estrutura de pastas
RUN mkdir -p data/raw data/processed/sentiment data/cache \
             models/registry logs

# Pré-gera todos os dados durante o build
# (mercado sintético + histórico + modelo + previsões + sentimento)
# Isso garante que o container sobe com dados prontos imediatamente.
RUN python scripts/popular_dashboard.py --rodadas 14 || \
    echo "Aviso: popular_dashboard falhou, dados serão gerados no start"

EXPOSE 8501

CMD ["bash", "render_start.sh"]
