#!/usr/bin/env bash
# =============================================================================
#  setup.sh — Cartola FC Analytics: instalação e inicialização automática
# =============================================================================
#
#  Uso:
#    bash setup.sh              # instala deps + coleta + treina + inicia dashboard
#    bash setup.sh --only-dash  # só inicia o dashboard (já configurado antes)
#    bash setup.sh --sintetico  # usa dados sintéticos sem chamar a API
#    bash setup.sh --api        # também sobe a API REST FastAPI em :8000
#
# =============================================================================

set -euo pipefail

# ── Cores ─────────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }
section() { echo -e "\n${GREEN}══════════════════════════════════════${NC}"; echo -e "${GREEN}  $*${NC}"; echo -e "${GREEN}══════════════════════════════════════${NC}"; }

# ── Flags ─────────────────────────────────────────────────────────────────────
ONLY_DASH=false
SINTETICO=false
START_API=false
RODADA_INICIO=1
RODADA_FIM=14

for arg in "$@"; do
  case $arg in
    --only-dash)  ONLY_DASH=true ;;
    --sintetico)  SINTETICO=true ;;
    --api)        START_API=true ;;
    --rodada-fim=*) RODADA_FIM="${arg#*=}" ;;
    --rodada-inicio=*) RODADA_INICIO="${arg#*=}" ;;
  esac
done

# ── Diretório raiz do projeto ──────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

section "Cartola FC Analytics — Setup Automático"
info "Diretório: $SCRIPT_DIR"

# ── Passo 1: Python ────────────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
  error "Python 3 não encontrado. Instale Python 3.11+ e tente novamente."
fi
PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
info "Python $PY_VER detectado."

# ── Passo 2: Ambiente virtual ──────────────────────────────────────────────────
if $ONLY_DASH; then
  info "Modo --only-dash: pulando instalação."
else
  section "Configurando ambiente virtual"

  if [ ! -d ".venv" ]; then
    info "Criando .venv..."
    python3 -m venv .venv
  else
    info ".venv já existe, reutilizando."
  fi

  # Ativar venv
  if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
  elif [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
  else
    warn "Não foi possível ativar o venv automaticamente."
  fi

  section "Instalando dependências"
  pip install --upgrade pip --quiet
  pip install -r requirements.txt --quiet
  info "Dependências instaladas."

  # ── Passos 3–6: Popular todos os dados do dashboard ──────────────────────────
  section "Populando dados do dashboard (histórico + modelo + previsões + sentimento)"

  SEM_MERC_FLAG=""
  if $SINTETICO; then
    info "Modo --sintetico: pulando coleta de API..."
    SEM_MERC_FLAG="--sem-mercado"
    # Garante que o mercado foi coletado ao menos uma vez antes
    if [ ! -f "data/processed/mercado_atual.parquet" ]; then
      info "Coletando mercado atual (necessário para dados sintéticos)..."
      python3 main.py collect --rodada-inicio 1 --rodada-fim 1 || true
    fi
  fi

  python3 scripts/popular_dashboard.py --rodadas "$RODADA_FIM" && \
    info "Todos os dados gerados com sucesso." || {
      warn "popular_dashboard.py falhou. Tentando pipeline manual..."
      python3 scripts/gerar_dados_sinteticos.py --rodadas "$RODADA_FIM"
      python3 main.py train || true
      python3 main.py predict || true
  }
fi

# ── Passo 6: API REST (opcional) ──────────────────────────────────────────────
if $START_API; then
  section "Iniciando API REST FastAPI"
  info "API disponível em http://localhost:8000 | Docs: http://localhost:8000/docs"
  python3 main.py api --porta 8000 &
  API_PID=$!
  info "API rodando (PID $API_PID)"
fi

# ── Passo 7: Dashboard ────────────────────────────────────────────────────────
section "Iniciando Dashboard Streamlit"
info "Dashboard disponível em http://localhost:8501"
echo ""
echo -e "${YELLOW}  Pressione Ctrl+C para encerrar.${NC}"
echo ""

# No GitHub Codespaces o Streamlit precisa de --server.headless
STREAMLIT_ARGS="dashboard/app.py --server.headless true --server.enableCORS false --server.enableXsrfProtection false"

# Detectar se está no Codespaces
if [ -n "${CODESPACE_NAME:-}" ]; then
  info "GitHub Codespaces detectado — configurando server.address=0.0.0.0"
  STREAMLIT_ARGS="$STREAMLIT_ARGS --server.address 0.0.0.0 --server.port 8501"
fi

streamlit run $STREAMLIT_ARGS
