"""
api/app.py
API REST do sistema Cartola FC — Fase 3.

Endpoints:
  GET  /health                        — status da API e modelo
  GET  /mercado/status                — status do mercado Cartola
  GET  /previsoes                     — previsões da próxima rodada
  GET  /previsoes/{atleta_id}         — previsão de jogador específico
  GET  /alertas                       — alertas da rodada
  POST /escalacao/otimizar            — monta escalação (sem usuário)
  POST /usuarios                      — cria usuário
  GET  /usuarios/{id}                 — dados do usuário
  PUT  /usuarios/{id}/preferencias    — atualiza preferências
  POST /usuarios/{id}/escalacao       — gera escalação personalizada
  POST /usuarios/{id}/resultado       — registra pontuação real
  GET  /usuarios/{id}/historico       — histórico de escalações
  GET  /autolearn/status              — status do model registry
  POST /autolearn/retreinar           — dispara re-treino manual

Execute com:
    uvicorn api.app:app --reload --port 8000
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── Lifespan (carrega dados na inicialização) ─────────────────────────────────

_cache: dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega previsões e mercado em memória na inicialização."""
    logger.info("Carregando dados em memória...")
    try:
        from config.settings import DATA_DIR
        processed = DATA_DIR / "processed"

        if (processed / "previsoes.parquet").exists():
            _cache["previsoes"] = pd.read_parquet(processed / "previsoes.parquet")
            logger.info("Previsões carregadas: %d atletas.", len(_cache["previsoes"]))

        if (processed / "mercado_atual.parquet").exists():
            _cache["mercado"] = pd.read_parquet(processed / "mercado_atual.parquet")

        from model.trainer import carregar_metricas
        _cache["metricas"] = carregar_metricas()

    except Exception as e:
        logger.warning("Erro ao pré-carregar dados: %s", e)

    yield
    _cache.clear()
    logger.info("API encerrada, cache limpo.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Cartola FC — Sistema Inteligente API",
    description="API REST para previsões, escalações e análise de sentimento do Cartola FC.",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # em produção, restringir para o domínio do frontend
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas Pydantic ──────────────────────────────────────────────────────────

class EscalacaoRequest(BaseModel):
    orcamento:  float       = Field(100.0, ge=10, le=300)
    perfil:     str         = Field("balanceado", pattern="^(conservador|balanceado|agressivo)$")
    formacao:   str         = Field("4-3-3")

class UsuarioCreate(BaseModel):
    nome:         str        = Field(..., min_length=2, max_length=80)
    email:        str | None = None
    perfil_risco: str        = Field("balanceado", pattern="^(conservador|balanceado|agressivo)$")
    orcamento:    float      = Field(100.0, ge=10)

class PreferenciasUpdate(BaseModel):
    perfil_risco:  str | None        = None
    orcamento:     float | None      = None
    formacao:      str | None        = None
    times_fav:     list[int] | None  = None
    jogadores_fav: list[int] | None  = None
    jogadores_blo: list[int] | None  = None

class EscalacaoUsuarioRequest(BaseModel):
    rodada: int = Field(..., ge=1, le=38)

class ResultadoRequest(BaseModel):
    escalacao_id: int
    pts_real:     float = Field(..., ge=0)
    avaliacao:    int | None = Field(None, ge=1, le=5)
    comentario:   str | None = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _df_para_lista(df: pd.DataFrame, max_rows: int = 200) -> list[dict]:
    """Converte DataFrame para lista de dicts serializável."""
    return df.head(max_rows).fillna(0).to_dict(orient="records")


def _previsoes() -> pd.DataFrame:
    if "previsoes" not in _cache:
        raise HTTPException(503, "Previsões não disponíveis. Execute 'python main.py predict'.")
    return _cache["previsoes"]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Sistema"])
def health():
    """Status da API, modelo e dados disponíveis."""
    from autolearn.engine import modelo_em_producao
    prod = modelo_em_producao()
    return {
        "status":          "ok",
        "timestamp":       datetime.utcnow().isoformat(),
        "previsoes_ok":    "previsoes" in _cache,
        "n_atletas":       len(_cache.get("previsoes", pd.DataFrame())),
        "modelo_versao":   prod.get("versao") if prod else None,
        "modelo_mae":      prod["metricas"].get("mae_medio") if prod else None,
    }


@app.get("/mercado/status", tags=["Mercado"])
def mercado_status():
    """Status atual do mercado Cartola FC."""
    try:
        from data_collection.cartola_api import buscar_status_mercado
        return buscar_status_mercado(usar_cache=True)
    except Exception as e:
        raise HTTPException(503, f"Erro ao buscar mercado: {e}")


@app.get("/previsoes", tags=["Previsões"])
def listar_previsoes(
    posicao:  str | None = Query(None, description="Filtrar por posição"),
    limite:   int        = Query(50, ge=1, le=200),
    ordenar:  str        = Query("score_composto", description="Coluna para ordenação"),
):
    """Lista previsões de pontuação ordenadas por score."""
    df = _previsoes().copy()
    if posicao:
        df = df[df["posicao"].str.lower() == posicao.lower()]
    if ordenar in df.columns:
        df = df.sort_values(ordenar, ascending=False)
    return {"total": len(df), "atletas": _df_para_lista(df, limite)}


@app.get("/previsoes/{atleta_id}", tags=["Previsões"])
def previsao_atleta(atleta_id: int):
    """Retorna previsão detalhada de um atleta específico."""
    df = _previsoes()
    if "atleta_id" not in df.columns:
        raise HTTPException(404, "Coluna atleta_id não disponível.")
    row = df[df["atleta_id"] == atleta_id]
    if row.empty:
        raise HTTPException(404, f"Atleta {atleta_id} não encontrado nas previsões.")
    return row.iloc[0].fillna(0).to_dict()


@app.get("/alertas", tags=["Previsões"])
def alertas_rodada(top_n: int = Query(5, ge=1, le=20)):
    """Retorna alertas categorizados da rodada."""
    from model.predictor import gerar_alertas
    return gerar_alertas(_previsoes(), top_n=top_n)


@app.post("/escalacao/otimizar", tags=["Escalação"])
def otimizar_escalacao(body: EscalacaoRequest):
    """Gera escalação otimizada sem perfil de usuário."""
    from model.predictor import otimizar_escalacao as _otimizar
    df_escal = _otimizar(_previsoes(), orcamento=body.orcamento, perfil=body.perfil)
    if df_escal.empty:
        raise HTTPException(422, "Não foi possível montar escalação com os parâmetros fornecidos.")
    return {
        "perfil":      body.perfil,
        "orcamento":   body.orcamento,
        "jogadores":   _df_para_lista(df_escal),
        "pts_esperado": round(df_escal["pontuacao_prevista"].sum(), 1) if "pontuacao_prevista" in df_escal.columns else None,
    }


# ── Usuários ──────────────────────────────────────────────────────────────────

@app.post("/usuarios", tags=["Usuários"], status_code=201)
def criar_usuario(body: UsuarioCreate):
    """Cria novo perfil de usuário."""
    from user.profile import criar_usuario as _criar
    u = _criar(nome=body.nome, email=body.email, perfil_risco=body.perfil_risco, orcamento=body.orcamento)
    return {"id": u.id, "nome": u.nome, "perfil_risco": u.perfil_risco, "orcamento": u.orcamento}


@app.get("/usuarios/{usuario_id}", tags=["Usuários"])
def buscar_usuario(usuario_id: int):
    """Retorna dados do perfil de um usuário."""
    from user.profile import buscar_usuario as _buscar
    u = _buscar(usuario_id)
    if not u:
        raise HTTPException(404, f"Usuário {usuario_id} não encontrado.")
    return {
        "id": u.id, "nome": u.nome, "email": u.email,
        "perfil_risco": u.perfil_risco, "orcamento": u.orcamento,
        "formacao": u.formacao, "times_fav": u.times_fav,
        "jogadores_fav": u.jogadores_fav, "jogadores_blo": u.jogadores_blo,
    }


@app.put("/usuarios/{usuario_id}/preferencias", tags=["Usuários"])
def atualizar_preferencias(usuario_id: int, body: PreferenciasUpdate):
    """Atualiza preferências do usuário."""
    from user.profile import atualizar_preferencias as _atualizar
    u = _atualizar(usuario_id, **body.model_dump(exclude_none=True))
    if not u:
        raise HTTPException(404, f"Usuário {usuario_id} não encontrado.")
    return {"ok": True, "usuario_id": usuario_id}


@app.post("/usuarios/{usuario_id}/escalacao", tags=["Usuários"])
def escalacao_personalizada(usuario_id: int, body: EscalacaoUsuarioRequest):
    """Gera escalação personalizada para o usuário com base no seu perfil e histórico."""
    from user.recommender import recomendar_para_usuario
    try:
        resultado = recomendar_para_usuario(usuario_id, _previsoes(), rodada=body.rodada)
    except ValueError as e:
        raise HTTPException(404, str(e))

    df_escal = resultado["escalacao"]
    return {
        "usuario":    resultado["usuario"],
        "formacao":   resultado["formacao"],
        "resumo":     resultado["resumo"],
        "insights":   resultado["insights"],
        "jogadores":  _df_para_lista(df_escal) if not df_escal.empty else [],
    }


@app.post("/usuarios/{usuario_id}/resultado", tags=["Usuários"])
def registrar_resultado(usuario_id: int, body: ResultadoRequest):
    """Registra a pontuação real obtida para aprendizado do modelo."""
    from user.profile import registrar_resultado as _registrar
    fb = _registrar(
        escalacao_id=body.escalacao_id,
        pts_real=body.pts_real,
        avaliacao=body.avaliacao,
        comentario=body.comentario,
    )
    if not fb:
        raise HTTPException(404, f"Escalação {body.escalacao_id} não encontrada.")
    return {"ok": True, "feedback_id": fb.id}


@app.get("/usuarios/{usuario_id}/historico", tags=["Usuários"])
def historico(usuario_id: int):
    """Retorna histórico de escalações com métricas de precisão."""
    from user.profile import historico_usuario
    return {"usuario_id": usuario_id, "historico": historico_usuario(usuario_id)}


# ── Auto-learning ─────────────────────────────────────────────────────────────

@app.get("/autolearn/status", tags=["Auto-Learning"])
def autolearn_status():
    """Status do model registry e versão em produção."""
    from autolearn.engine import listar_versoes, modelo_em_producao
    return {
        "em_producao": modelo_em_producao(),
        "versoes":     listar_versoes(),
    }


@app.post("/autolearn/retreinar", tags=["Auto-Learning"])
def retreinar(forcar: bool = Query(False)):
    """Dispara re-treino manual do modelo."""
    from autolearn.engine import retreinar as _retreinar
    try:
        resultado = _retreinar(forcar=forcar)
        return resultado
    except Exception as e:
        raise HTTPException(500, f"Erro no re-treino: {e}")
