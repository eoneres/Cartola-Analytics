"""
main.py
Ponto de entrada CLI do sistema Cartola FC — Fase 2.

Uso:
    python main.py collect   [--rodada-inicio N] [--rodada-fim M]
    python main.py train
    python main.py predict   [--rodada N]
    python main.py full      [--rodada-inicio N] [--rodada-fim M]
    python main.py sentiment --jogadores Endrick Gabi [--modo auto|bert|vader]
"""

import argparse
import logging
import sys
from pathlib import Path

from config.settings import LOG_FORMAT, LOG_LEVEL, LOGS_DIR

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / "cartola.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("main")


def cmd_collect(args):
    from data_collection.pipeline import executar_pipeline
    logger.info("Coletando dados (rodadas %d–%d)...", args.rodada_inicio, args.rodada_fim)
    resultado = executar_pipeline(args.rodada_inicio, args.rodada_fim)
    for k, df in resultado.items():
        logger.info("%-12s → %d registros", k, len(df))
    logger.info("Coleta concluída.")


def cmd_train(args):
    from data_collection.pipeline import PROCESSED_DIR
    import pandas as pd
    from model.features import construir_features
    from model.trainer import treinar

    hist_path = PROCESSED_DIR / "historico_completo.parquet"
    part_path = PROCESSED_DIR / "partidas.parquet"

    if not hist_path.exists():
        logger.error("Histórico não encontrado. Execute 'python main.py collect' primeiro.")
        sys.exit(1)

    logger.info("Carregando histórico...")
    df_hist = pd.read_parquet(hist_path)
    df_part = pd.read_parquet(part_path) if part_path.exists() else None

    logger.info("Construindo features...")
    df_feat = construir_features(df_hist, df_part)

    logger.info("Treinando modelo...")
    modelo, scaler, metricas = treinar(df_feat)

    logger.info("MAE médio: %.2f pts", metricas.get("mae_medio", 0))
    logger.info("R² médio:  %.3f",     metricas.get("r2_medio",  0))
    logger.info("Treinamento concluído.")


def cmd_predict(args):
    from data_collection.pipeline import PROCESSED_DIR
    import pandas as pd
    from model.predictor import gerar_alertas, prever_pontuacoes

    hist_path = PROCESSED_DIR / "historico_completo.parquet"
    merc_path = PROCESSED_DIR / "mercado_atual.parquet"
    part_path = PROCESSED_DIR / "partidas.parquet"

    if not hist_path.exists() or not merc_path.exists():
        logger.error("Dados não encontrados. Execute 'collect' e 'train' primeiro.")
        sys.exit(1)

    df_hist = pd.read_parquet(hist_path)
    df_merc = pd.read_parquet(merc_path)
    df_part = pd.read_parquet(part_path) if part_path.exists() else None

    logger.info("Gerando previsões...")
    df_prev = prever_pontuacoes(df_hist, df_merc, df_part)

    if df_prev.empty:
        logger.error("Nenhuma previsão gerada.")
        sys.exit(1)

    out_path = PROCESSED_DIR / "previsoes.parquet"
    df_prev.to_parquet(out_path, index=False)
    df_prev.head(20).to_csv(PROCESSED_DIR / "previsoes_top20.csv", index=False)
    logger.info("Previsões salvas em %s", out_path)

    top_cols = [c for c in ["apelido","posicao","preco","pontuacao_prevista","score_composto"]
                if c in df_prev.columns]
    print("\n=== TOP 10 RECOMENDADOS ===")
    print(df_prev[top_cols].head(10).to_string(index=False))

    alertas = gerar_alertas(df_prev)
    em_alta = alertas.get("em_alta", [])
    if em_alta:
        print("\n=== EM ALTA ===")
        for j in em_alta[:5]:
            print(f"  {j.get('apelido','?')} ({j.get('posicao','?')}) — tendência {j.get('tendencia',0):.2f}")


def cmd_full(args):
    """Executa coleta + treino + previsão em sequência."""
    cmd_collect(args)
    cmd_train(args)
    cmd_predict(args)


def cmd_sentiment(args):
    """Coleta textos e calcula score de sentimento para uma lista de jogadores."""
    from sentiment.aggregator import processar_lista_jogadores

    jogadores = args.jogadores
    if not jogadores:
        logger.error(
            "Informe ao menos um jogador. "
            "Ex: python main.py sentiment --jogadores Endrick Gabi"
        )
        sys.exit(1)

    logger.info("Iniciando análise de sentimento para: %s", jogadores)
    df = processar_lista_jogadores(
        jogadores,
        modo_analise=args.modo,
        incluir_twitter=not args.sem_twitter,
        incluir_reddit=not args.sem_reddit,
        incluir_noticias=not args.sem_noticias,
    )

    print("\n=== SCORES DE SENTIMENTO ===")
    cols = [c for c in
            ["jogador", "score_medio", "hype_score", "alerta", "volume",
             "pct_positivo", "pct_negativo"]
            if c in df.columns]
    print(df[cols].to_string(index=False))


# ── Parser ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sistema Inteligente Cartola FC — Fase 2",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    sub = parser.add_subparsers(dest="comando")

    # collect
    p_collect = sub.add_parser("collect", help="Coleta e processa dados históricos")
    p_collect.add_argument("--rodada-inicio", type=int, default=1,  dest="rodada_inicio")
    p_collect.add_argument("--rodada-fim",    type=int, default=10, dest="rodada_fim")

    # train
    sub.add_parser("train", help="Treina o modelo preditivo")

    # predict
    p_pred = sub.add_parser("predict", help="Gera previsões para a próxima rodada")
    p_pred.add_argument("--rodada", type=int, default=None)

    # full
    p_full = sub.add_parser("full", help="Coleta + treina + prevê em sequência")
    p_full.add_argument("--rodada-inicio", type=int, default=1,  dest="rodada_inicio")
    p_full.add_argument("--rodada-fim",    type=int, default=10, dest="rodada_fim")

    # sentiment
    p_sent = sub.add_parser("sentiment", help="Analisa sentimento de jogadores (Fase 2)")
    p_sent.add_argument(
        "--jogadores", nargs="+", required=True,
        help="Lista de jogadores. Ex: --jogadores Endrick Gabi Arrascaeta",
    )
    p_sent.add_argument(
        "--modo", choices=["auto", "bert", "vader"], default="auto",
        help="Modelo de análise: auto (padrão), bert ou vader",
    )
    p_sent.add_argument("--sem-twitter",  action="store_true", dest="sem_twitter",
                        help="Não coleta tweets")
    p_sent.add_argument("--sem-reddit",   action="store_true", dest="sem_reddit",
                        help="Não coleta posts do Reddit")
    p_sent.add_argument("--sem-noticias", action="store_true", dest="sem_noticias",
                        help="Não faz scraping de notícias")

    # scheduler
    p_sched = sub.add_parser('scheduler', help='Gerencia jobs automáticos (Fase 3)')
    p_sched.add_argument('acao', choices=['start','status','run'], help='start | status | run')
    p_sched.add_argument('--job', default='coletar',
                         choices=['coletar','sentimento','retreinar','prever','pesos'],
                         help='Job a executar (apenas com acao=run)')

    # autolearn
    p_al = sub.add_parser('autolearn', help='Controla auto-learning do modelo (Fase 3)')
    p_al.add_argument('acao', choices=['status','retreinar','versoes'])
    p_al.add_argument('--forcar', action='store_true', dest='forcar',
                      help='Força re-treino mesmo sem dados novos')

    # api
    p_api = sub.add_parser('api', help='Inicia a API REST FastAPI (Fase 3)')
    p_api.add_argument('--porta',  type=int, default=8000)
    p_api.add_argument('--reload', action='store_true')

    args = parser.parse_args()

    if args.comando == 'collect':
        cmd_collect(args)
    elif args.comando == 'train':
        cmd_train(args)
    elif args.comando == 'predict':
        cmd_predict(args)
    elif args.comando == 'full':
        cmd_full(args)
    elif args.comando == 'sentiment':
        cmd_sentiment(args)
    elif args.comando == 'scheduler':
        cmd_scheduler(args)
    elif args.comando == 'autolearn':
        cmd_autolearn(args)
    elif args.comando == 'api':
        cmd_api(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()


def cmd_scheduler(args):
    """Gerencia o scheduler de jobs automáticos."""
    if args.acao == "start":
        from scheduler.jobs import registrar_jobs, scheduler
        logger.info("Iniciando scheduler (Ctrl+C para parar)...")
        registrar_jobs()
        scheduler.start()
    elif args.acao == "status":
        from scheduler.jobs import listar_jobs, registrar_jobs, scheduler
        registrar_jobs()
        jobs = listar_jobs()
        print("\n=== JOBS AGENDADOS ===")
        for j in jobs:
            print(f"  {j['id']:<25} | próxima: {j['proxima_execucao']}")
    elif args.acao == "run":
        from scheduler.jobs import (
            job_ajustar_pesos, job_atualizar_sentimento,
            job_coletar_rodada, job_gerar_previsoes, job_retreinar_modelo,
        )
        mapa = {
            "coletar":    job_coletar_rodada,
            "sentimento": job_atualizar_sentimento,
            "retreinar":  job_retreinar_modelo,
            "prever":     job_gerar_previsoes,
            "pesos":      job_ajustar_pesos,
        }
        job_fn = mapa.get(args.job)
        if not job_fn:
            logger.error("Job desconhecido. Opções: %s", list(mapa.keys()))
            sys.exit(1)
        logger.info("Executando job '%s' manualmente...", args.job)
        job_fn()


def cmd_autolearn(args):
    """Controla o auto-learning do modelo."""
    from autolearn.engine import (
        listar_versoes, modelo_em_producao, retreinar,
        verificar_necessidade_retreino,
    )
    if args.acao == "status":
        prod = modelo_em_producao()
        print("\n=== MODELO EM PRODUÇÃO ===")
        if prod:
            print(f"  Versão:  {prod['versao']}")
            print(f"  MAE:     {prod['metricas'].get('mae_medio','—')}")
            print(f"  R²:      {prod['metricas'].get('r2_medio','—')}")
        else:
            print("  Nenhum modelo em produção.")
        check = verificar_necessidade_retreino()
        print(f"\n  Re-treino necessário: {check['precisa']} ({check['motivo']})")

    elif args.acao == "retreinar":
        resultado = retreinar(forcar=args.forcar)
        if resultado.get("deployed"):
            print(f"✅ Novo modelo deployado: {resultado['versao']}")
            print(f"   MAE: {resultado['metricas']['mae_medio']:.2f}")
        else:
            print(f"ℹ️  Modelo atual mantido: {resultado.get('motivo','')}")

    elif args.acao == "versoes":
        versoes = listar_versoes()
        print(f"\n=== {len(versoes)} VERSÃO(ÕES) NO REGISTRY ===")
        for v in versoes:
            ativo = "✅" if v.get("ativo") else "  "
            print(f"  {ativo} {v['versao']} | MAE={v['metricas'].get('mae_medio','?')} | {v['criado_em'][:19]}")


def cmd_api(args):
    """Inicia a API REST FastAPI."""
    import uvicorn
    logger.info("Iniciando API em http://0.0.0.0:%d ...", args.porta)
    uvicorn.run("api.app:app", host="0.0.0.0", port=args.porta, reload=args.reload)
