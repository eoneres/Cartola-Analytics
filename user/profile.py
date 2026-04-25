"""
user/profile.py
Sistema de perfis de usuário com preferências, histórico de escalações
e feedback de resultados.

Cada usuário tem:
  - Perfil de risco (conservador / balanceado / agressivo)
  - Times favoritos (prioridade na escalação)
  - Jogadores favoritos / bloqueados
  - Histórico de escalações por rodada
  - Feedback de pontuação real vs esperada
  - Preferências de formação tática

Persistência: SQLite local via SQLAlchemy (sem servidor necessário).
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    JSON, Boolean, Column, DateTime, Float, ForeignKey,
    Integer, String, Text, create_engine, event,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

from config.settings import DATA_DIR

logger = logging.getLogger(__name__)

DB_PATH = DATA_DIR / "usuarios.db"
ENGINE  = create_engine(f"sqlite:///{DB_PATH}", echo=False)

# Habilita WAL mode para melhor concorrência no SQLite
@event.listens_for(ENGINE, "connect")
def _set_wal(dbapi_conn, _):
    dbapi_conn.execute("PRAGMA journal_mode=WAL")
    dbapi_conn.execute("PRAGMA foreign_keys=ON")

SessionLocal = sessionmaker(bind=ENGINE, autoflush=False, autocommit=False)


# ── Models SQLAlchemy ─────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


class Usuario(Base):
    __tablename__ = "usuarios"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    nome          = Column(String(80), nullable=False)
    email         = Column(String(120), unique=True, nullable=True)
    perfil_risco  = Column(String(20), default="balanceado")   # conservador|balanceado|agressivo
    orcamento     = Column(Float, default=100.0)
    formacao      = Column(String(10), default="4-3-3")
    times_fav     = Column(JSON, default=list)     # lista de clube_id
    jogadores_fav = Column(JSON, default=list)     # lista de atleta_id
    jogadores_blo = Column(JSON, default=list)     # lista de atleta_id bloqueados
    criado_em     = Column(DateTime, default=datetime.utcnow)
    atualizado_em = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    escalacoes = relationship("Escalacao", back_populates="usuario", cascade="all, delete-orphan")
    feedbacks  = relationship("Feedback",  back_populates="usuario", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Usuario id={self.id} nome={self.nome} perfil={self.perfil_risco}>"


class Escalacao(Base):
    __tablename__ = "escalacoes"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    usuario_id   = Column(Integer, ForeignKey("usuarios.id"), nullable=False)
    rodada       = Column(Integer, nullable=False)
    temporada    = Column(Integer, nullable=False)
    jogadores    = Column(JSON, nullable=False)   # lista de atleta_id selecionados
    formacao     = Column(String(10))
    orcamento_us = Column(Float)                  # orçamento usado
    pts_esperado = Column(Float)                  # soma das previsões
    pts_real     = Column(Float, nullable=True)   # preenchido após a rodada
    criado_em    = Column(DateTime, default=datetime.utcnow)

    usuario  = relationship("Usuario",  back_populates="escalacoes")
    feedback = relationship("Feedback", back_populates="escalacao", uselist=False)

    def erro_absoluto(self) -> float | None:
        if self.pts_real is not None:
            return abs(self.pts_real - self.pts_esperado)
        return None


class Feedback(Base):
    __tablename__ = "feedbacks"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    usuario_id   = Column(Integer, ForeignKey("usuarios.id"), nullable=False)
    escalacao_id = Column(Integer, ForeignKey("escalacoes.id"), nullable=True)
    rodada       = Column(Integer, nullable=False)
    avaliacao    = Column(Integer)             # 1-5 estrelas
    comentario   = Column(Text, nullable=True)
    pts_real     = Column(Float, nullable=True)
    criado_em    = Column(DateTime, default=datetime.utcnow)

    usuario   = relationship("Usuario",   back_populates="feedbacks")
    escalacao = relationship("Escalacao", back_populates="feedback")


# Cria tabelas se não existirem
Base.metadata.create_all(ENGINE)


# ── CRUD helpers ──────────────────────────────────────────────────────────────

def criar_usuario(
    nome: str,
    email: str | None = None,
    perfil_risco: str = "balanceado",
    orcamento: float = 100.0,
) -> Usuario:
    with SessionLocal() as db:
        u = Usuario(nome=nome, email=email, perfil_risco=perfil_risco, orcamento=orcamento)
        db.add(u)
        db.commit()
        db.refresh(u)
        logger.info("Usuário criado: %s (id=%d)", nome, u.id)
        return u


def buscar_usuario(usuario_id: int) -> Usuario | None:
    with SessionLocal() as db:
        return db.get(Usuario, usuario_id)


def buscar_usuario_por_email(email: str) -> Usuario | None:
    with SessionLocal() as db:
        return db.query(Usuario).filter(Usuario.email == email).first()


def listar_usuarios() -> list[Usuario]:
    with SessionLocal() as db:
        return db.query(Usuario).all()


def atualizar_preferencias(
    usuario_id: int,
    perfil_risco: str | None = None,
    orcamento: float | None = None,
    formacao: str | None = None,
    times_fav: list | None = None,
    jogadores_fav: list | None = None,
    jogadores_blo: list | None = None,
) -> Usuario | None:
    with SessionLocal() as db:
        u = db.get(Usuario, usuario_id)
        if not u:
            logger.warning("Usuário %d não encontrado.", usuario_id)
            return None
        if perfil_risco  is not None: u.perfil_risco  = perfil_risco
        if orcamento     is not None: u.orcamento     = orcamento
        if formacao      is not None: u.formacao      = formacao
        if times_fav     is not None: u.times_fav     = times_fav
        if jogadores_fav is not None: u.jogadores_fav = jogadores_fav
        if jogadores_blo is not None: u.jogadores_blo = jogadores_blo
        u.atualizado_em = datetime.utcnow()
        db.commit()
        db.refresh(u)
        return u


def salvar_escalacao(
    usuario_id: int,
    rodada: int,
    temporada: int,
    jogadores: list[int],
    formacao: str,
    orcamento_usado: float,
    pts_esperado: float,
) -> Escalacao:
    with SessionLocal() as db:
        e = Escalacao(
            usuario_id=usuario_id,
            rodada=rodada,
            temporada=temporada,
            jogadores=jogadores,
            formacao=formacao,
            orcamento_us=orcamento_usado,
            pts_esperado=pts_esperado,
        )
        db.add(e)
        db.commit()
        db.refresh(e)
        logger.info("Escalação salva: usuário=%d rodada=%d id=%d", usuario_id, rodada, e.id)
        return e


def registrar_resultado(
    escalacao_id: int,
    pts_real: float,
    avaliacao: int | None = None,
    comentario: str | None = None,
) -> Feedback | None:
    with SessionLocal() as db:
        esc = db.get(Escalacao, escalacao_id)
        if not esc:
            logger.warning("Escalação %d não encontrada.", escalacao_id)
            return None
        esc.pts_real = pts_real

        fb = Feedback(
            usuario_id=esc.usuario_id,
            escalacao_id=escalacao_id,
            rodada=esc.rodada,
            avaliacao=avaliacao,
            comentario=comentario,
            pts_real=pts_real,
        )
        db.add(fb)
        db.commit()
        db.refresh(fb)
        logger.info(
            "Resultado registrado: escalação=%d pts_real=%.1f erro=%.1f",
            escalacao_id, pts_real, abs(pts_real - esc.pts_esperado),
        )
        return fb


def historico_usuario(usuario_id: int) -> list[dict]:
    """Retorna histórico completo de escalações com métricas de precisão."""
    with SessionLocal() as db:
        escalacoes = (
            db.query(Escalacao)
            .filter(Escalacao.usuario_id == usuario_id)
            .order_by(Escalacao.rodada)
            .all()
        )
        return [
            {
                "escalacao_id": e.id,
                "rodada":       e.rodada,
                "temporada":    e.temporada,
                "formacao":     e.formacao,
                "pts_esperado": e.pts_esperado,
                "pts_real":     e.pts_real,
                "erro_abs":     e.erro_absoluto(),
                "orcamento_us": e.orcamento_us,
                "n_jogadores":  len(e.jogadores or []),
            }
            for e in escalacoes
        ]
