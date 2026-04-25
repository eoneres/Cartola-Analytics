"""
scripts/gerar_dados_sinteticos.py
==================================
Gera dados históricos sintéticos baseados nos atletas do mercado atual.
Usado como fallback quando a API do Cartola não retorna dados históricos
(ex: início de temporada, endpoint temporariamente indisponível, etc.).

Uso:
    python scripts/gerar_dados_sinteticos.py --rodadas 14
    python scripts/gerar_dados_sinteticos.py --rodadas 14 --seed 42
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DATA_DIR, POSICOES, SCOUTS_PESOS, TEMPORADA_ATUAL

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


# Média e desvio padrão de pontuação por posição (valores realistas Cartola FC)
PONTUACAO_STATS = {
    "Goleiro":   {"mu": 5.5,  "sigma": 4.0},
    "Lateral":   {"mu": 4.0,  "sigma": 3.5},
    "Zagueiro":  {"mu": 4.5,  "sigma": 3.8},
    "Meia":      {"mu": 6.0,  "sigma": 5.0},
    "Atacante":  {"mu": 7.0,  "sigma": 6.0},
    "Técnico":   {"mu": 5.0,  "sigma": 4.0},
    "Desconhecido": {"mu": 5.0, "sigma": 4.0},
}


def gerar_historico(rodadas: int = 14, seed: int = 42) -> pd.DataFrame:
    """
    Gera DataFrame histórico sintético baseado nos atletas do mercado atual.

    Cada atleta recebe pontuações por rodada com variação realista,
    calibradas pela média_num do mercado.
    """
    merc_path = PROCESSED_DIR / "mercado_atual.parquet"
    if not merc_path.exists():
        logger.error("mercado_atual.parquet não encontrado. Execute 'python main.py collect' primeiro.")
        sys.exit(1)

    df_merc = pd.read_parquet(merc_path)
    logger.info("Mercado carregado: %d atletas.", len(df_merc))

    rng = np.random.default_rng(seed)
    registros = []

    for _, atleta in df_merc.iterrows():
        posicao = atleta.get("posicao", "Desconhecido")
        stats = PONTUACAO_STATS.get(posicao, PONTUACAO_STATS["Desconhecido"])
        media_api = float(atleta.get("media_num", 0) or 0)

        # Usa média da API se disponível, senão usa média da posição
        mu = media_api if media_api > 0 else stats["mu"]
        sigma = stats["sigma"]

        # Jogadores com preço mais alto têm pontuações mais consistentes
        preco = float(atleta.get("preco_num", 5) or 5)
        consistencia = min(0.9, preco / 30)  # quanto mais caro, menos variação

        for rodada in range(1, rodadas + 1):
            # Probabilidade de jogar: jogadores melhores jogam mais
            prob_jogar = min(0.95, 0.5 + consistencia * 0.5)
            if rng.random() > prob_jogar:
                continue  # não jogou esta rodada

            # Pontuação com tendência temporal (simula forma)
            forma_fator = 1.0 + rng.uniform(-0.2, 0.2)
            pontuacao = max(0.0, rng.normal(mu * forma_fator, sigma * (1 - consistencia * 0.5)))

            # Scouts sintéticos proporcional à pontuação
            gols = int(pontuacao > 8 and rng.random() < 0.3)
            assists = int(pontuacao > 6 and rng.random() < 0.2)
            desarmes = int(rng.random() < 0.4) if posicao in ("Zagueiro", "Lateral", "Meia") else 0

            rec = {
                "atleta_id":  atleta["atleta_id"],
                "apelido":    atleta.get("apelido", ""),
                "posicao_id": atleta.get("posicao_id", 0),
                "posicao":    posicao,
                "clube_id":   atleta.get("clube_id", 0),
                "rodada":     rodada,
                "pontuacao":  round(pontuacao, 2),
                "preco":      float(atleta.get("preco_num", 0) or 0),
                "variacao":   round(rng.uniform(-2, 2), 2),
                "media":      round(mu, 2),
                "temporada":  TEMPORADA_ATUAL,
            }
            for scout in SCOUTS_PESOS:
                rec[f"scout_{scout}"] = 0
            rec["scout_gol"] = gols
            rec["scout_assistencia"] = assists
            rec["scout_desarme"] = desarmes

            registros.append(rec)

    df = pd.DataFrame(registros)
    logger.info("Dados sintéticos gerados: %d registros, %d rodadas.", len(df), rodadas)
    return df


def salvar(df: pd.DataFrame, rodadas: int):
    """Salva histórico consolidado e arquivos por rodada."""
    # Consolidado
    out = PROCESSED_DIR / "historico_completo.parquet"
    df.to_parquet(out, index=False)
    df.to_csv(PROCESSED_DIR / "historico_completo.csv", index=False)
    logger.info("Salvo: %s (%d linhas)", out, len(df))

    # Por rodada
    for r in range(1, rodadas + 1):
        df_r = df[df["rodada"] == r]
        if not df_r.empty:
            df_r.to_parquet(RAW_DIR / f"pontuados_r{r:02d}.parquet", index=False)

    logger.info("Arquivos por rodada salvos em %s", RAW_DIR)


def main():
    parser = argparse.ArgumentParser(description="Gera histórico sintético para o Cartola Analytics")
    parser.add_argument("--rodadas", type=int, default=14, help="Número de rodadas a simular")
    parser.add_argument("--seed", type=int, default=42, help="Semente aleatória para reprodutibilidade")
    args = parser.parse_args()

    logger.info("Gerando %d rodadas de dados sintéticos (seed=%d)...", args.rodadas, args.seed)
    df = gerar_historico(rodadas=args.rodadas, seed=args.seed)
    salvar(df, args.rodadas)
    logger.info("✅ Dados sintéticos prontos. Execute 'python main.py train' para treinar o modelo.")


if __name__ == "__main__":
    main()
