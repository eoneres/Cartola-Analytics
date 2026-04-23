"""
main.py
Ponto de entrada CLI do sistema Cartola FC MVP.

Uso:
    python main.py collect  [--rodada-inicio N] [--rodada-fim M]
    python main.py train
    python main.py predict  [--rodada N]
    python main.py full     [--rodada-inicio N] [--rodada-fim M]
"""

import argparse
import logging
import sys
from pathlib import Path

# ── Setup de logging ──────────────────────────────────────────────────────────
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


# ── Comandos ──────────────────────────────────────────────────────────────────

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

    hist_path  = PROCESSED_DIR / "historico_completo.parquet"
    merc_path  = PROCESSED_DIR / "mercado_atual.parquet"
    part_path  = PROCESSED_DIR / "partidas.parquet"

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

    # Salvar para o dashboard
    out_path = PROCESSED_DIR / "previsoes.parquet"
    df_prev.to_parquet(out_path, index=False)
    df_prev.head(20).to_csv(PROCESSED_DIR / "previsoes_top20.csv", index=False)
    logger.info("Previsões salvas em %s", out_path)

    # Mostrar top 10
    top_cols = [c for c in ["apelido","posicao","preco","pontuacao_prevista","score_composto"]
                if c in df_prev.columns]
    print("\n=== TOP 10 RECOMENDADOS ===")
    print(df_prev[top_cols].head(10).to_string(index=False))

    # Alertas
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


# ── Parser ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sistema Inteligente Cartola FC — MVP",
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

    args = parser.parse_args()

    if args.comando == "collect":
        cmd_collect(args)
    elif args.comando == "train":
        cmd_train(args)
    elif args.comando == "predict":
        cmd_predict(args)
    elif args.comando == "full":
        cmd_full(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
