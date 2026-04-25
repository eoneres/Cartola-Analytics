# =============================================================================
#  Makefile — Cartola FC Analytics
#  Atalhos para todas as operações do sistema
# =============================================================================
#
#  Uso rápido:
#    make setup          → instala deps + coleta + treina + dashboard
#    make setup-sintetico → igual, mas sem depender da API do Cartola
#    make dash           → só abre o dashboard (setup já feito)
#    make full           → coleta + treina + prevê (sem dashboard)
#    make treinar        → apenas re-treina o modelo
#    make prever         → apenas gera previsões
#    make api            → sobe a API REST FastAPI em :8000
#    make scheduler      → inicia o scheduler de automação
#    make status         → mostra status do modelo e jobs
#    make limpar-cache   → apaga cache de API (força nova coleta)
#    make help           → lista todos os comandos

PYTHON   ?= python3
VENV     := .venv
PIP      := $(VENV)/bin/pip
PY       := $(VENV)/bin/python
STREAM   := $(VENV)/bin/streamlit
RODADAS  ?= 14

# Detecta SO para ativar venv corretamente
ifeq ($(OS),Windows_NT)
    ACTIVATE := $(VENV)/Scripts/activate
    PY       := $(VENV)/Scripts/python
    STREAM   := $(VENV)/Scripts/streamlit
else
    ACTIVATE := $(VENV)/bin/activate
endif

.PHONY: help setup setup-sintetico dash full \
        coletar treinar prever sintetico popular \
        api scheduler autolearn status \
        limpar-cache limpar-tudo clean

# ── Ajuda ──────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  Cartola FC Analytics — Comandos disponíveis"
	@echo "  ─────────────────────────────────────────────"
	@echo "  make setup            Instalação completa + dashboard"
	@echo "  make setup-sintetico  Instalação com dados sintéticos (sem API)"
	@echo "  make dash             Abre o dashboard Streamlit"
	@echo "  make full             Coleta + treina + prevê"
	@echo "  make coletar          Coleta dados históricos da API"
	@echo "  make sintetico        Gera dados históricos sintéticos"
	@echo "  make treinar          Treina o modelo XGBoost"
	@echo "  make prever           Gera previsões da próxima rodada"
	@echo "  make api              Sobe a API REST FastAPI em :8000"
	@echo "  make scheduler        Inicia o scheduler automático"
	@echo "  make autolearn        Verifica e dispara re-treino"
	@echo "  make status           Status do modelo e jobs"
	@echo "  make limpar-cache     Remove cache de API"
	@echo "  make limpar-tudo      Remove dados + modelos + cache"
	@echo ""
	@echo "  Variáveis:"
	@echo "    RODADAS=14          Número de rodadas a coletar (padrão: 14)"
	@echo "    PYTHON=python3      Interpretador Python"
	@echo ""

# ── Ambiente virtual ──────────────────────────────────────────────────────────
$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip --quiet

venv: $(VENV)/bin/activate

deps: venv
	$(PIP) install -r requirements.txt --quiet
	@echo "✅ Dependências instaladas."

# ── Setup completo ────────────────────────────────────────────────────────────
setup: deps popular
	@echo "🚀 Abrindo dashboard..."
	$(STREAM) run dashboard/app.py \
	    --server.headless true \
	    --server.enableCORS false \
	    --server.enableXsrfProtection false

setup-sintetico: deps
	@echo "🔄 Populando dashboard com dados sintéticos..."
	$(PY) scripts/popular_dashboard.py --sem-mercado --rodadas $(RODADAS)
	@echo "🚀 Abrindo dashboard..."
	$(STREAM) run dashboard/app.py \
	    --server.headless true \
	    --server.enableCORS false \
	    --server.enableXsrfProtection false

popular: deps
	@echo "📊 Populando todos os dados do dashboard..."
	$(PY) scripts/popular_dashboard.py --rodadas $(RODADAS)
	@echo "✅ Dashboard pronto."

# ── Dashboard ─────────────────────────────────────────────────────────────────
dash:
	$(STREAM) run dashboard/app.py \
	    --server.headless true \
	    --server.enableCORS false \
	    --server.enableXsrfProtection false

# ── Pipeline de dados ─────────────────────────────────────────────────────────
full: coletar treinar prever
	@echo "✅ Pipeline completo executado."

coletar:
	$(PY) main.py collect --rodada-inicio 1 --rodada-fim $(RODADAS)

sintetico:
	$(PY) scripts/gerar_dados_sinteticos.py --rodadas $(RODADAS)

treinar:
	$(PY) main.py train

prever:
	$(PY) main.py predict

# ── Serviços ──────────────────────────────────────────────────────────────────
api:
	$(PY) main.py api --porta 8000 --reload

scheduler:
	$(PY) main.py scheduler start

autolearn:
	$(PY) main.py autolearn retreinar

# ── Status / diagnóstico ──────────────────────────────────────────────────────
status:
	@echo ""
	@echo "── Modelo em produção ──"
	$(PY) main.py autolearn status
	@echo ""
	@echo "── Jobs agendados ──"
	$(PY) main.py scheduler status
	@echo ""
	@echo "── Arquivos de dados ──"
	@ls -lh data/processed/*.parquet 2>/dev/null || echo "Nenhum dado processado encontrado."
	@ls -lh models/*.pkl 2>/dev/null || echo "Nenhum modelo treinado encontrado."

# ── Limpeza ───────────────────────────────────────────────────────────────────
limpar-cache:
	rm -rf data/cache/
	@echo "🗑️  Cache de API removido."

limpar-tudo: limpar-cache
	rm -rf data/raw/ data/processed/
	rm -rf models/*.pkl models/*.json models/registry/
	@echo "🗑️  Dados e modelos removidos. Execute 'make setup' para reiniciar."

clean: limpar-tudo
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
