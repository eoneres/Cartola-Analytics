"""
sentiment/dashboard_page.py
Página de análise de sentimento para o dashboard Streamlit.

Importada por dashboard/app.py como uma nova aba.
Pode também ser executada de forma independente:
    streamlit run sentiment/dashboard_page.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from config.settings import DATA_DIR
from sentiment.aggregator import (
    integrar_sentimento,
    processar_lista_jogadores,
)
from sentiment.analyzer import analisar

SENTIMENT_DIR   = DATA_DIR / "processed" / "sentiment"
CONSOLIDADO_PATH = SENTIMENT_DIR / "sentimento_consolidado.parquet"
PREVISOES_PATH  = DATA_DIR / "processed" / "previsoes.parquet"


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=600)
def _carregar_sentimento() -> pd.DataFrame:
    if CONSOLIDADO_PATH.exists():
        return pd.read_parquet(CONSOLIDADO_PATH)
    return pd.DataFrame()


@st.cache_data(ttl=300)
def _carregar_previsoes() -> pd.DataFrame:
    if PREVISOES_PATH.exists():
        return pd.read_parquet(PREVISOES_PATH)
    return pd.DataFrame()


def _cor_alerta(alerta: str) -> str:
    return {"em alta": "green", "em crise": "red", "neutro": "gray"}.get(alerta, "gray")


def _badge_html(texto: str, cor: str) -> str:
    cores = {
        "green": ("#d1fae5", "#065f46"),
        "red":   ("#fee2e2", "#991b1b"),
        "gray":  ("#f3f4f6", "#374151"),
    }
    bg, fg = cores.get(cor, cores["gray"])
    return (
        f'<span style="background:{bg};color:{fg};padding:2px 10px;'
        f'border-radius:12px;font-size:12px;font-weight:500">{texto}</span>'
    )


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def render():
    st.header("Análise de Sentimento")
    st.caption("Percepção pública sobre jogadores via Twitter, Reddit e notícias esportivas.")

    tab1, tab2, tab3 = st.tabs(["Painel Geral", "Analisar Jogador", "Teste Rápido"])

    # ── TAB 1: Painel geral ───────────────────────────────────────────────────
    with tab1:
        df_sent = _carregar_sentimento()

        if df_sent.empty:
            st.info(
                "Nenhum dado de sentimento encontrado.\n\n"
                "Execute a análise na aba **Analisar Jogador** ou via CLI:\n"
                "```bash\npython main.py sentiment --jogadores Endrick Gabi\n```"
            )
        else:
            # KPIs
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Jogadores analisados", len(df_sent))
            c2.metric(
                "Em alta",
                len(df_sent[df_sent["alerta"] == "em alta"]),
                delta_color="normal",
            )
            c3.metric(
                "Em crise",
                len(df_sent[df_sent["alerta"] == "em crise"]),
                delta_color="inverse",
            )
            c4.metric(
                "Score médio geral",
                f"{df_sent['score_medio'].mean():.2f}",
            )

            st.divider()

            # Tabela com badge de alerta
            st.subheader("Ranking de sentimento")
            df_show = df_sent.sort_values("score_medio", ascending=False).reset_index(drop=True)

            cols_tab = [c for c in [
                "jogador", "score_medio", "hype_score", "tendencia",
                "pct_positivo", "pct_negativo", "volume", "alerta",
            ] if c in df_show.columns]

            st.dataframe(
                df_show[cols_tab],
                hide_index=True,
                use_container_width=True,
                column_config={
                    "jogador":      st.column_config.TextColumn("Jogador"),
                    "score_medio":  st.column_config.NumberColumn("Score", format="%.3f"),
                    "hype_score":   st.column_config.ProgressColumn("Hype", min_value=0, max_value=1),
                    "tendencia":    st.column_config.NumberColumn("Tendência", format="%.4f"),
                    "pct_positivo": st.column_config.NumberColumn("% Positivo", format="%.1%"),
                    "pct_negativo": st.column_config.NumberColumn("% Negativo", format="%.1%"),
                    "volume":       st.column_config.NumberColumn("Vol. textos"),
                    "alerta":       st.column_config.TextColumn("Status"),
                },
            )

            # Gráfico de barras: score por jogador
            if "score_medio" in df_show.columns and len(df_show) > 0:
                st.subheader("Score de sentimento por jogador")
                st.bar_chart(
                    df_show.set_index("jogador")["score_medio"].head(20),
                    use_container_width=True,
                )

            # Integração com previsões
            df_prev = _carregar_previsoes()
            if not df_prev.empty:
                st.divider()
                st.subheader("Impacto no score composto")
                df_int = integrar_sentimento(df_prev, df_sent)
                cols_int = [c for c in [
                    "apelido", "posicao", "score_composto",
                    "score_sentimento_norm", "score_composto_final",
                    "alerta_sentimento",
                ] if c in df_int.columns]
                st.dataframe(
                    df_int[cols_int].head(20),
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "score_composto":       st.column_config.NumberColumn("Score anterior", format="%.3f"),
                        "score_sentimento_norm":st.column_config.NumberColumn("Sentimento", format="%.3f"),
                        "score_composto_final": st.column_config.ProgressColumn("Score final", min_value=0, max_value=1),
                    },
                )

    # ── TAB 2: Analisar jogador ───────────────────────────────────────────────
    with tab2:
        st.subheader("Analisar jogadores agora")
        st.caption("Coleta textos em tempo real e calcula os scores de sentimento.")

        nomes_input = st.text_input(
            "Nomes dos jogadores (separados por vírgula)",
            placeholder="Ex: Endrick, Gabi, Arrascaeta",
        )

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            usar_twitter  = st.checkbox("Twitter/X",  value=bool(True))
        with col_b:
            usar_reddit   = st.checkbox("Reddit",     value=True)
        with col_c:
            usar_noticias = st.checkbox("Notícias",   value=True)

        modo = st.radio(
            "Modelo de análise",
            ["auto", "bert", "vader"],
            horizontal=True,
            help="**auto**: BERT se disponível, senão VADER  |  **bert**: Modelo de linguagem (requer `transformers`)  |  **vader**: Léxico (rápido, sem GPU)",
        )

        if st.button("Iniciar análise", type="primary"):
            if not nomes_input.strip():
                st.error("Informe ao menos um jogador.")
            else:
                jogadores = [j.strip() for j in nomes_input.split(",") if j.strip()]
                with st.spinner(f"Analisando {len(jogadores)} jogador(es)..."):
                    df_result = processar_lista_jogadores(
                        jogadores,
                        modo_analise=modo,
                        incluir_twitter=usar_twitter,
                        incluir_reddit=usar_reddit,
                        incluir_noticias=usar_noticias,
                    )
                st.success("Análise concluída!")
                st.cache_data.clear()

                for _, row in df_result.iterrows():
                    with st.expander(f"{row['jogador']} — {row['alerta'].upper()}"):
                        ca, cb, cc = st.columns(3)
                        ca.metric("Score médio", f"{row['score_medio']:.3f}")
                        cb.metric("Hype score",  f"{row['hype_score']:.3f}")
                        cc.metric("Vol. textos", int(row["volume"]))

                        dist_cols = st.columns(3)
                        dist_cols[0].metric("Positivo", f"{row['pct_positivo']:.1%}")
                        dist_cols[1].metric("Neutro",   f"{row['pct_neutro']:.1%}")
                        dist_cols[2].metric("Negativo", f"{row['pct_negativo']:.1%}")

    # ── TAB 3: Teste rápido ───────────────────────────────────────────────────
    with tab3:
        st.subheader("Teste rápido de sentimento")
        st.caption("Classifica qualquer texto manualmente.")

        texto_teste = st.text_area(
            "Digite ou cole um texto sobre um jogador",
            placeholder="Ex: Endrick foi decisivo hoje, que golaço incrível!",
            height=120,
        )
        modo_teste = st.radio("Modo", ["auto", "bert", "vader"], horizontal=True, key="modo_teste")

        if st.button("Classificar", key="btn_teste"):
            if not texto_teste.strip():
                st.warning("Digite algum texto.")
            else:
                with st.spinner("Classificando..."):
                    resultado = analisar(texto_teste, modo=modo_teste)

                score = resultado["score"]
                sent  = resultado["sentimento"]
                cor   = {"positivo": "green", "negativo": "red", "neutro": "gray"}[sent]

                st.markdown(
                    f"**Sentimento:** {_badge_html(sent.upper(), cor)}",
                    unsafe_allow_html=True,
                )
                st.metric("Score", f"{score:.3f}", help="-1 (muito negativo) a +1 (muito positivo)")
                st.metric("Confiança", f"{resultado['confianca']:.1%}")
                st.caption(f"Modo: `{resultado['modo']}` | Texto processado: _{resultado['texto']}_")


# Execução standalone
if __name__ == "__main__":
    st.set_page_config(
        page_title="Sentimento — Cartola FC",
        page_icon="💬",
        layout="wide",
    )
    render()
