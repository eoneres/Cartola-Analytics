"""
dashboard/app.py
Dashboard interativo do sistema Cartola FC MVP.
Execute com: streamlit run dashboard/app.py

No GitHub Codespaces, use:
  streamlit run dashboard/app.py \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false
Ou simplesmente execute: bash setup.sh --only-dash
"""

import os
import sys
from pathlib import Path

# Garante que o projeto raiz está no path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Corrige aviso de CORS no GitHub Codespaces (tunnel cross-origin)
# Essas variáveis são lidas pelo Streamlit antes da inicialização do servidor
os.environ.setdefault("STREAMLIT_SERVER_ENABLE_CORS", "false")
os.environ.setdefault("STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION", "false")
os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")

import pandas as pd
import streamlit as st

from config.settings import DATA_DIR, ESCALACAO_SLOTS, FORMACOES, FORMACAO_PADRAO, MODELS_DIR, ORCAMENTO_PADRAO, POSICOES
from model.predictor import gerar_alertas, otimizar_escalacao
from model.trainer import carregar_metricas
from sentiment.dashboard_page import render as render_sentimento
from dashboard.fase3_page import render as render_fase3

# ── Health check para o Render ────────────────────────────────────────────────
# O Render bate em /healthz para saber se o serviço está vivo.
# O Streamlit não expõe rotas HTTP customizadas nativamente, então usamos
# o endpoint interno /_stcore/health que o Streamlit já fornece.
# O render.yaml já aponta healthCheckPath para /_stcore/health.

# ── Configuração da página ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cartola FC — Sistema Inteligente",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROCESSED_DIR = DATA_DIR / "processed"
PREVISOES_PATH = PROCESSED_DIR / "previsoes.parquet"
HISTORICO_PATH = PROCESSED_DIR / "historico_completo.parquet"


# ── Helpers ────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def carregar_previsoes() -> pd.DataFrame:
    if PREVISOES_PATH.exists():
        return pd.read_parquet(PREVISOES_PATH)
    return pd.DataFrame()


@st.cache_data(ttl=300)
def carregar_historico() -> pd.DataFrame:
    if HISTORICO_PATH.exists():
        return pd.read_parquet(HISTORICO_PATH)
    return pd.DataFrame()


def _badge(texto: str, cor: str = "#0F6E56") -> str:
    return (
        f'<span style="background:{cor};color:#fff;padding:2px 8px;'
        f'border-radius:4px;font-size:12px;">{texto}</span>'
    )


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚽ Cartola FC")
    st.caption("Sistema Inteligente — Fase 3")
    st.divider()

    pagina = st.radio(
        "Navegação",
        ["Ranking de Jogadores", "Montar Escalação", "Alertas", "Sentimento", "Fase 3 · Personalização", "Métricas do Modelo"],
        label_visibility="collapsed",
    )

    st.divider()

    st.subheader("Filtros globais")
    posicoes_disponiveis = list(POSICOES.values())
    pos_filtro = st.multiselect(
        "Posições",
        posicoes_disponiveis,
        default=posicoes_disponiveis,
    )
    preco_min, preco_max = st.slider(
        "Faixa de preço (C$)",
        min_value=0.0, max_value=50.0,
        value=(0.0, 50.0), step=0.5,
    )

# ── Carregar dados ─────────────────────────────────────────────────────────────
df_prev = carregar_previsoes()
df_hist = carregar_historico()

SEM_DADOS = df_prev.empty

if SEM_DADOS:
    st.warning(
        "Nenhuma previsão encontrada. Execute primeiro:\n\n"
        "```bash\npython main.py collect\npython main.py train\npython main.py predict\n```"
    )

# ── Aplicar filtros globais ────────────────────────────────────────────────────
if not SEM_DADOS:
    df_filtrado = df_prev.copy()
    if "posicao" in df_filtrado.columns:
        df_filtrado = df_filtrado[df_filtrado["posicao"].isin(pos_filtro)]
    if "preco" in df_filtrado.columns:
        df_filtrado = df_filtrado[
            df_filtrado["preco"].between(preco_min, preco_max)
        ]


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 1 — Ranking de Jogadores
# ══════════════════════════════════════════════════════════════════════════════
if pagina == "Ranking de Jogadores":
    st.header("Ranking de Jogadores")

    if SEM_DADOS:
        st.stop()

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jogadores analisados", len(df_filtrado))
    c2.metric(
        "Maior score composto",
        f"{df_filtrado['score_composto'].max():.2f}" if "score_composto" in df_filtrado.columns else "—",
    )
    c3.metric(
        "Previsão máxima",
        f"{df_filtrado['pontuacao_prevista'].max():.1f} pts"
        if "pontuacao_prevista" in df_filtrado.columns else "—",
    )
    c4.metric(
        "Média do grupo",
        f"{df_filtrado['pontuacao_prevista'].mean():.1f} pts"
        if "pontuacao_prevista" in df_filtrado.columns else "—",
    )

    st.divider()

    # Ordenação
    col_ord, col_asc = st.columns([3, 1])
    with col_ord:
        ordenar_por = st.selectbox(
            "Ordenar por",
            ["score_composto", "pontuacao_prevista", "media_recente", "preco"],
        )
    with col_asc:
        ascendente = st.checkbox("Crescente", value=False)

    df_rank = df_filtrado.sort_values(ordenar_por, ascending=ascendente).head(50)

    # Tabela
    colunas_tabela = [
        c for c in
        ["apelido", "posicao", "preco", "media_recente",
         "pontuacao_prevista", "score_composto", "tendencia",
         "regularidade", "eh_mandante"]
        if c in df_rank.columns
    ]
    st.dataframe(
        df_rank[colunas_tabela].reset_index(drop=True),
        use_container_width=True,
        column_config={
            "apelido":             st.column_config.TextColumn("Jogador"),
            "posicao":             st.column_config.TextColumn("Posição"),
            "preco":               st.column_config.NumberColumn("Preço (C$)", format="%.1f"),
            "media_recente":       st.column_config.NumberColumn("Média recente", format="%.1f"),
            "pontuacao_prevista":  st.column_config.NumberColumn("Previsão ML", format="%.1f"),
            "score_composto":      st.column_config.ProgressColumn("Score", min_value=0, max_value=1),
            "tendencia":           st.column_config.NumberColumn("Tendência", format="%.2f"),
            "regularidade":        st.column_config.ProgressColumn("Regularidade", min_value=0, max_value=1),
            "eh_mandante":         st.column_config.CheckboxColumn("Mandante?"),
        },
        hide_index=True,
    )

    # Gráfico de dispersão preço × previsão
    if "preco" in df_rank.columns and "pontuacao_prevista" in df_rank.columns:
        st.subheader("Preço × Previsão de pontuação")
        st.scatter_chart(
            df_rank,
            x="preco",
            y="pontuacao_prevista",
            color="posicao",
            size="score_composto",
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 2 — Montar Escalação
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "Montar Escalação":
    st.header("Montar Escalação")

    if SEM_DADOS:
        st.stop()

    # ── Seletor de formação ───────────────────────────────────────────────────
    st.subheader("1. Escolha a formação")

    nomes_formacao = list(FORMACOES.keys())
    idx_padrao = nomes_formacao.index(FORMACAO_PADRAO) if FORMACAO_PADRAO in nomes_formacao else 0

    # Grid visual com botões de formação (5 por linha)
    COLS_POR_LINHA = 5
    linhas = [nomes_formacao[i:i+COLS_POR_LINHA] for i in range(0, len(nomes_formacao), COLS_POR_LINHA)]

    if "formacao_selecionada" not in st.session_state:
        st.session_state.formacao_selecionada = FORMACAO_PADRAO

    for linha in linhas:
        cols_form = st.columns(len(linha))
        for col, nome in zip(cols_form, linha):
            ativo = st.session_state.formacao_selecionada == nome
            label = f"**{nome}**" if ativo else nome
            if col.button(label, key=f"btn_form_{nome}", use_container_width=True,
                          type="primary" if ativo else "secondary"):
                st.session_state.formacao_selecionada = nome

    formacao_escolhida = st.session_state.formacao_selecionada
    info_form = FORMACOES[formacao_escolhida]

    # Preview da formação selecionada
    with st.container():
        slots = info_form["slots"]
        st.caption(f"📋 {info_form['descricao']}")

        # Linha de badges por posição
        icones = {"Goleiro": "🧤", "Lateral": "🔁", "Zagueiro": "🛡️", "Meia": "⚙️", "Atacante": "⚽", "Técnico": "📋"}
        badges = "  ".join(
            f"{icones.get(pos, '')} **{qtd}× {pos}**"
            for pos, qtd in slots.items() if qtd > 0
        )
        st.markdown(badges)
        total_linha = sum(v for k, v in slots.items() if k not in ("Goleiro", "Técnico"))
        total_time  = sum(slots.values())
        st.caption(f"Total: {total_time} jogadores  •  {total_linha} de linha")

    st.divider()

    # ── Parâmetros ────────────────────────────────────────────────────────────
    st.subheader("2. Configure os parâmetros")

    col1, col2, col3 = st.columns(3)
    with col1:
        perfil = st.selectbox(
            "Perfil de risco",
            ["conservador", "balanceado", "agressivo"],
            index=1,
            help=(
                "**Conservador**: prefere jogadores com alta média histórica e preço médio-alto.\n\n"
                "**Balanceado**: equilíbrio entre segurança e upside.\n\n"
                "**Agressivo**: aposta em jogadores com alta previsão ML, mesmo que com histórico curto."
            ),
        )
    with col2:
        orcamento = st.number_input(
            "Orçamento (C$)",
            min_value=10.0, max_value=200.0,
            value=float(ORCAMENTO_PADRAO), step=1.0,
        )
    with col3:
        st.write("")
        st.write("")
        gerar = st.button("⚽ Gerar escalação", type="primary", use_container_width=True)

    # ── Resultado ─────────────────────────────────────────────────────────────
    if gerar:
        with st.spinner("Otimizando escalação..."):
            df_escal = otimizar_escalacao(
                df_prev,
                orcamento=orcamento,
                perfil=perfil,
                formacao=formacao_escolhida,
            )

        if df_escal.empty:
            st.error("Não foi possível montar uma escalação com esses parâmetros. "
                     "Tente aumentar o orçamento ou mudar a formação.")
        else:
            custo_total   = df_escal["preco"].sum() if "preco" in df_escal.columns else 0
            pts_esperados = df_escal["pontuacao_prevista"].sum() if "pontuacao_prevista" in df_escal.columns else 0

            st.success(f"✅ Escalação **{formacao_escolhida}** gerada com {len(df_escal)} jogadores!")

            sa, sb, sc, sd = st.columns(4)
            sa.metric("Formação",         formacao_escolhida)
            sb.metric("Custo total",      f"C$ {custo_total:.1f}")
            sc.metric("Saldo restante",   f"C$ {orcamento - custo_total:.1f}")
            sd.metric("Pontuação esperada", f"{pts_esperados:.1f} pts")

            st.divider()
            st.subheader("3. Jogadores selecionados")

            # Ordem de exibição: Goleiro → Lateral → Zagueiro → Meia → Atacante → Técnico
            ORDEM_POSICAO = ["Goleiro", "Lateral", "Zagueiro", "Meia", "Atacante", "Técnico"]
            icones_pos = {"Goleiro": "🧤", "Lateral": "🔁", "Zagueiro": "🛡️",
                          "Meia": "⚙️", "Atacante": "⚽", "Técnico": "📋"}

            for posicao in ORDEM_POSICAO:
                subset = df_escal[df_escal["posicao"] == posicao]
                if subset.empty:
                    continue
                icone = icones_pos.get(posicao, "")
                st.markdown(f"#### {icone} {posicao}")
                cols_esc = [c for c in
                    ["apelido", "preco", "pontuacao_prevista", "score_composto", "justificativa"]
                    if c in subset.columns]
                st.dataframe(
                    subset[cols_esc].rename(columns={
                        "apelido": "Jogador", "preco": "Preço (C$)",
                        "pontuacao_prevista": "Pts previstos",
                        "score_composto": "Score",
                        "justificativa": "Por quê?",
                    }),
                    hide_index=True, use_container_width=True,
                )

            # Download
            st.divider()
            csv = df_escal.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Baixar escalação (CSV)",
                data=csv,
                file_name=f"escalacao_{formacao_escolhida}_{perfil}.csv",
                mime="text/csv",
            )


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 3 — Alertas
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "Alertas":
    st.header("Alertas da Rodada")

    if SEM_DADOS:
        st.stop()

    alertas = gerar_alertas(df_prev, top_n=5)

    def _tabela_alerta(lista: list, titulo: str, cor: str):
        st.subheader(titulo)
        if not lista:
            st.caption("Nenhum alerta nesta categoria.")
            return
        st.dataframe(pd.DataFrame(lista), hide_index=True, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        _tabela_alerta(alertas.get("em_alta", []),  "Em alta", "#0F6E56")
        _tabela_alerta(alertas.get("surpresas", []),"Possíveis surpresas", "#185FA5")
    with col_b:
        _tabela_alerta(alertas.get("riscos", []),   "Riscos / em queda", "#A32D2D")
        _tabela_alerta(alertas.get("zeraveis", []), "Prováveis zeráveis", "#854F0B")


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 4 — Métricas do Modelo
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "Métricas do Modelo":
    st.header("Métricas do Modelo")

    metricas = carregar_metricas()

    if not metricas:
        st.info("Treine o modelo primeiro: `python main.py train`")
        st.stop()

    m1, m2, m3 = st.columns(3)
    m1.metric("MAE médio",  f"{metricas.get('mae_medio',  0):.2f} pts")
    m2.metric("RMSE médio", f"{metricas.get('rmse_medio', 0):.2f} pts")
    m3.metric("R² médio",   f"{metricas.get('r2_medio',   0):.3f}")

    # Folds
    folds = metricas.get("folds", [])
    if folds:
        st.subheader("Performance por fold (validação temporal)")
        st.dataframe(pd.DataFrame(folds), hide_index=True, use_container_width=True)
        st.line_chart(pd.DataFrame(folds).set_index("fold")[["mae", "rmse"]])

    # Feature importances
    fi = metricas.get("feature_importances", {})
    if fi:
        st.subheader("Importância das features")
        df_fi = pd.DataFrame(fi.items(), columns=["feature", "importancia"]).sort_values(
            "importancia", ascending=True
        )
        st.bar_chart(df_fi.set_index("feature")["importancia"], horizontal=True)

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 5 — Análise de Sentimento (Fase 2)
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "Sentimento":
    render_sentimento()

elif pagina == "Fase 3 · Personalização":
    render_fase3()
