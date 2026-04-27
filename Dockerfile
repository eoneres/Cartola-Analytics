# =============================================================================
#  Dockerfile — Cartola FC Analytics (Streamlit + FastAPI)
#  Usado pelo Render para construir a imagem do serviço.
# =============================================================================

FROM python:3.12-slim

# Evita prompts interativos durante instalação
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Diretório de trabalho
WORKDIR /app

# Dependências do sistema (mínimas)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Instala dependências Python
COPY requirements-render.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements-render.txt

# Copia o projeto inteiro
COPY . .

# Cria pastas locais como fallback (sobrescritas pelo Disk em produção)
RUN mkdir -p data/raw data/processed/sentiment data/cache models/registry logs

# Porta do Streamlit
EXPOSE 8501

# Script de inicialização: popula dados se necessário e sobe o dashboard
CMD ["bash", "render_start.sh"]
