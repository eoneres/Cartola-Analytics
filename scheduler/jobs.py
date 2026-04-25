"""
scheduler/jobs.py
Agendador de tarefas automáticas do sistema Cartola FC.

Jobs configurados:
  1. coletar_rodada_atual   — a cada 6h durante a semana da rodada
  2. atualizar_sentimento   — diariamente (principais jogadores do mercado)
  3. retreinar_modelo       — após cada rodada fechada
  4. gerar_previsoes        — quando o mercado abre (véspera da rodada)
  5. ajustar_pesos          — semanalmente após consolidar resultados

Usa APScheduler com persistência em SQLite para sobreviver a reinicializações.

Uso:
    python -m scheduler.jobs          # roda o scheduler em foreground
    python main.py scheduler start    # via CLI principal
    python main.py scheduler status   # mostra jobs agendados
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from config.settings import DATA_DIR, TEMPORADA_ATUAL

logger = logging.getLogger(__name__)

SCHEDULER_DB = DATA_DIR / "scheduler.db"

# ── Configuração do APScheduler ───────────────────────────────────────────────

jobstores = {
    "default": SQLAlchemyJobStore(url=f"sqlite:///{SCHEDULER_DB}"),
}
executors = {
    "default": ThreadPoolExecutor(max_workers=3),
}
job_defaults = {
    "coalesce":       True,    # executar apenas uma vez se atrasado
    "max_instances":  1,       # não sobrepor execuções
    "misfire_grace_time": 3600,
}

scheduler = BlockingScheduler(
    jobstores=job_stores if (job_stores := jobstores) else {},
    executors=executors,
    job_defaults=job_defaults,
    timezone="America/Sao_Paulo",
)


# ── Funções dos jobs ──────────────────────────────────────────────────────────

def job_coletar_rodada():
    """Coleta dados da rodada atual e atualiza o mercado."""
    logger.info("[JOB] Iniciando coleta de dados — %s", datetime.now().isoformat())
    try:
        from data_collection.cartola_api import buscar_atletas_mercado, buscar_status_mercado
        from data_collection.pipeline import etapa_mercado_atual

        status = buscar_status_mercado(usar_cache=False)
        rodada = status.get("rodada_atual", 0)
        logger.info("[JOB] Rodada atual: %d | Status mercado: %d", rodada, status.get("status_mercado", 0))

        etapa_mercado_atual()
        logger.info("[JOB] Mercado atualizado com sucesso.")
    except Exception as e:
        logger.error("[JOB] Erro na coleta: %s", e, exc_info=True)


def job_atualizar_sentimento():
    """Atualiza análise de sentimento para os top jogadores do mercado."""
    logger.info("[JOB] Atualizando sentimento — %s", datetime.now().isoformat())
    try:
        import pandas as pd
        from sentiment.aggregator import processar_lista_jogadores

        merc_path = DATA_DIR / "processed" / "mercado_atual.parquet"
        if not merc_path.exists():
            logger.warning("[JOB] Mercado não disponível para sentimento.")
            return

        df_merc = pd.read_parquet(merc_path)
        # Top 30 jogadores por média como proxy de relevância
        col_media = "media_num" if "media_num" in df_merc.columns else "media_recente"
        top30 = (
            df_merc.nlargest(30, col_media)["apelido"]
            .dropna().tolist()
            if col_media in df_merc.columns
            else []
        )

        if top30:
            processar_lista_jogadores(top30, modo_analise="auto", incluir_twitter=False)
            logger.info("[JOB] Sentimento atualizado para %d jogadores.", len(top30))
    except Exception as e:
        logger.error("[JOB] Erro no sentimento: %s", e, exc_info=True)


def job_retreinar_modelo():
    """Re-treina o modelo se houver dados novos suficientes."""
    logger.info("[JOB] Verificando necessidade de re-treino — %s", datetime.now().isoformat())
    try:
        from autolearn.engine import retreinar, verificar_necessidade_retreino

        check = verificar_necessidade_retreino(min_rodadas_novas=3)
        logger.info("[JOB] Verificação: %s", check)

        if check["precisa"]:
            resultado = retreinar()
            if resultado.get("deployed"):
                logger.info("[JOB] Novo modelo deployado: %s", resultado.get("versao"))
            else:
                logger.info("[JOB] Modelo atual mantido: %s", resultado.get("motivo"))
        else:
            logger.info("[JOB] Re-treino desnecessário: %s", check["motivo"])
    except Exception as e:
        logger.error("[JOB] Erro no re-treino: %s", e, exc_info=True)


def job_gerar_previsoes():
    """Gera e persiste previsões para a próxima rodada."""
    logger.info("[JOB] Gerando previsões — %s", datetime.now().isoformat())
    try:
        import pandas as pd
        from model.predictor import prever_pontuacoes

        processed = DATA_DIR / "processed"
        hist_path = processed / "historico_completo.parquet"
        merc_path = processed / "mercado_atual.parquet"
        part_path = processed / "partidas.parquet"

        if not hist_path.exists() or not merc_path.exists():
            logger.warning("[JOB] Dados insuficientes para previsões.")
            return

        df_hist = pd.read_parquet(hist_path)
        df_merc = pd.read_parquet(merc_path)
        df_part = pd.read_parquet(part_path) if part_path.exists() else None

        df_prev = prever_pontuacoes(df_hist, df_merc, df_part)
        if not df_prev.empty:
            df_prev.to_parquet(processed / "previsoes.parquet", index=False)
            logger.info("[JOB] Previsões geradas: %d atletas.", len(df_prev))
    except Exception as e:
        logger.error("[JOB] Erro nas previsões: %s", e, exc_info=True)


def job_ajustar_pesos():
    """Recalcula os pesos ótimos do score composto com base no histórico acumulado."""
    logger.info("[JOB] Ajustando pesos do score — %s", datetime.now().isoformat())
    try:
        import pandas as pd
        from autolearn.engine import ajustar_pesos_score
        from model.features import construir_features

        hist_path = DATA_DIR / "processed" / "historico_completo.parquet"
        part_path = DATA_DIR / "processed" / "partidas.parquet"

        if not hist_path.exists():
            return

        df_hist = pd.read_parquet(hist_path)
        df_part = pd.read_parquet(part_path) if part_path.exists() else None
        df_feat = construir_features(df_hist, df_part)

        if not df_feat.empty:
            resultado = ajustar_pesos_score(df_feat)
            logger.info("[JOB] Pesos sugeridos: %s", resultado["pesos_calculados"])
    except Exception as e:
        logger.error("[JOB] Erro no ajuste de pesos: %s", e, exc_info=True)


# ── Registro dos jobs ─────────────────────────────────────────────────────────

def registrar_jobs():
    """Registra todos os jobs com seus gatilhos."""

    # 1. Coleta de dados — a cada 6h (Seg–Sex)
    scheduler.add_job(
        job_coletar_rodada,
        trigger=CronTrigger(day_of_week="mon-fri", hour="6,12,18,23", minute=0),
        id="coletar_rodada",
        name="Coleta de dados da rodada",
        replace_existing=True,
    )

    # 2. Sentimento — todos os dias às 7h
    scheduler.add_job(
        job_atualizar_sentimento,
        trigger=CronTrigger(hour=7, minute=0),
        id="atualizar_sentimento",
        name="Atualização de sentimento NLP",
        replace_existing=True,
    )

    # 3. Re-treino — toda Terça às 3h (após rodada do fim de semana)
    scheduler.add_job(
        job_retreinar_modelo,
        trigger=CronTrigger(day_of_week="tue", hour=3, minute=0),
        id="retreinar_modelo",
        name="Re-treino automático do modelo",
        replace_existing=True,
    )

    # 4. Previsões — toda Quinta às 8h (mercado abre Qui/Sex)
    scheduler.add_job(
        job_gerar_previsoes,
        trigger=CronTrigger(day_of_week="thu,fri", hour=8, minute=0),
        id="gerar_previsoes",
        name="Geração de previsões",
        replace_existing=True,
    )

    # 5. Ajuste de pesos — toda Quarta às 4h
    scheduler.add_job(
        job_ajustar_pesos,
        trigger=CronTrigger(day_of_week="wed", hour=4, minute=0),
        id="ajustar_pesos",
        name="Ajuste automático de pesos do score",
        replace_existing=True,
    )

    logger.info("Jobs registrados: %d", len(scheduler.get_jobs()))
    for job in scheduler.get_jobs():
        logger.info("  %-25s → próxima execução: %s", job.name, job.next_run_time)


def listar_jobs() -> list[dict]:
    """Retorna lista de jobs com status."""
    return [
        {
            "id":              job.id,
            "nome":            job.name,
            "proxima_execucao": str(job.next_run_time) if job.next_run_time else "pausado",
        }
        for job in scheduler.get_jobs()
    ]


# ── Ponto de entrada ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )
    logger.info("Iniciando scheduler — fuso: America/Sao_Paulo")
    registrar_jobs()
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler encerrado.")
