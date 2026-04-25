"""
dashboard/fase3_page.py
Página da Fase 3 no dashboard Streamlit.

Seções:
  - Meu Perfil      : criar/editar perfil, preferências
  - Minha Escalação : recomendação personalizada com insights
  - Meu Histórico   : evolução de pontuações + precisão do modelo
  - Auto-Learning   : status do model registry, pesos sugeridos
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from config.settings import DATA_DIR, MODELS_DIR

PROCESSED_DIR = DATA_DIR / "processed"


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def _previsoes() -> pd.DataFrame:
    p = PROCESSED_DIR / "previsoes.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def _usuario_id() -> int | None:
    return st.session_state.get("usuario_id")


def _set_usuario(uid: int):
    st.session_state["usuario_id"] = uid


# ── Render principal ──────────────────────────────────────────────────────────

def render():
    st.header("Fase 3 — Personalização & Auto-Learning")

    tab1, tab2, tab3, tab4 = st.tabs([
        "👤 Meu Perfil",
        "⚽ Minha Escalação",
        "📈 Meu Histórico",
        "🤖 Auto-Learning",
    ])

    with tab1: _tab_perfil()
    with tab2: _tab_escalacao()
    with tab3: _tab_historico()
    with tab4: _tab_autolearn()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Perfil
# ══════════════════════════════════════════════════════════════════════════════

def _tab_perfil():
    st.subheader("Perfil de usuário")

    from user.profile import (
        atualizar_preferencias,
        buscar_usuario,
        criar_usuario,
        listar_usuarios,
    )

    # Seleção de usuário existente
    usuarios = listar_usuarios()
    opcoes   = {f"{u.nome} (id={u.id})": u.id for u in usuarios}

    col_sel, col_novo = st.columns([3, 1])
    with col_sel:
        if opcoes:
            sel = st.selectbox("Selecionar usuário", ["— novo —"] + list(opcoes.keys()))
            if sel != "— novo —":
                _set_usuario(opcoes[sel])
        else:
            st.info("Nenhum usuário cadastrado. Crie um abaixo.")

    # Criar novo usuário
    with st.expander("➕ Criar novo usuário", expanded=not bool(opcoes)):
        with st.form("form_novo_usuario"):
            nome    = st.text_input("Nome")
            email   = st.text_input("E-mail (opcional)")
            perfil  = st.selectbox("Perfil de risco", ["conservador", "balanceado", "agressivo"], index=1)
            orçam   = st.number_input("Orçamento (C$)", min_value=10.0, max_value=300.0, value=100.0)
            if st.form_submit_button("Criar", type="primary"):
                if nome.strip():
                    u = criar_usuario(nome.strip(), email or None, perfil, orçam)
                    _set_usuario(u.id)
                    st.success(f"Usuário '{u.nome}' criado (id={u.id})")
                    st.rerun()
                else:
                    st.error("Informe um nome.")

    # Editar preferências do usuário selecionado
    uid = _usuario_id()
    if uid:
        u = buscar_usuario(uid)
        if u:
            st.divider()
            st.subheader(f"Preferências de {u.nome}")

            with st.form("form_prefs"):
                c1, c2, c3 = st.columns(3)
                novo_perfil  = c1.selectbox("Perfil", ["conservador","balanceado","agressivo"],
                                            index=["conservador","balanceado","agressivo"].index(u.perfil_risco or "balanceado"))
                novo_orcam   = c2.number_input("Orçamento (C$)", value=float(u.orcamento or 100))
                nova_form    = c3.selectbox("Formação", ["4-3-3","4-4-2","3-5-2","4-2-4","5-3-2"],
                                            index=["4-3-3","4-4-2","3-5-2","4-2-4","5-3-2"].index(u.formacao or "4-3-3"))

                times_fav_str = st.text_input(
                    "IDs de times favoritos (separados por vírgula)",
                    value=", ".join(map(str, u.times_fav or [])),
                    help="Ex: 262, 266 (use os clube_id do Cartola)",
                )
                jog_blo_str = st.text_input(
                    "IDs de jogadores bloqueados (separados por vírgula)",
                    value=", ".join(map(str, u.jogadores_blo or [])),
                )

                if st.form_submit_button("Salvar preferências"):
                    def _parse_ids(s): return [int(x.strip()) for x in s.split(",") if x.strip().isdigit()]
                    atualizar_preferencias(
                        uid,
                        perfil_risco=novo_perfil,
                        orcamento=novo_orcam,
                        formacao=nova_form,
                        times_fav=_parse_ids(times_fav_str),
                        jogadores_blo=_parse_ids(jog_blo_str),
                    )
                    st.success("Preferências salvas!")
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Escalação Personalizada
# ══════════════════════════════════════════════════════════════════════════════

def _tab_escalacao():
    st.subheader("Escalação personalizada")

    uid = _usuario_id()
    if not uid:
        st.warning("Selecione ou crie um usuário na aba **Meu Perfil** primeiro.")
        return

    df_prev = _previsoes()
    if df_prev.empty:
        st.info("Execute `python main.py predict` para gerar previsões.")
        return

    rodada = st.number_input("Rodada", min_value=1, max_value=38, value=1)

    if st.button("Gerar minha escalação", type="primary"):
        from user.profile import salvar_escalacao
        from user.recommender import recomendar_para_usuario

        with st.spinner("Personalizando sua escalação..."):
            resultado = recomendar_para_usuario(uid, df_prev, rodada=rodada)

        u_info   = resultado["usuario"]
        resumo   = resultado["resumo"]
        insights = resultado["insights"]
        df_escal = resultado["escalacao"]

        st.success(f"Escalação gerada para **{u_info['nome']}** (perfil: {u_info['perfil']})")

        # KPIs
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Jogadores",      resumo.get("total_jogadores", 0))
        c2.metric("Custo total",    f"C$ {resumo.get('custo_total', 0):.1f}")
        c3.metric("Saldo restante", f"C$ {resumo.get('saldo', 0):.1f}")
        c4.metric("Pts esperados",  f"{resumo.get('pts_esperado', 0):.1f}")

        # Insights
        st.subheader("💡 Insights personalizados")
        for insight in insights:
            st.info(f"• {insight}")

        st.divider()

        # Tabela por posição
        if not df_escal.empty:
            from config.settings import ESCALACAO_SLOTS
            for posicao in ESCALACAO_SLOTS:
                subset = df_escal[df_escal["posicao"] == posicao]
                if subset.empty:
                    continue
                st.subheader(posicao)
                cols = [c for c in ["apelido","preco","pontuacao_prevista","score_personalizado","eh_mandante"] if c in subset.columns]
                st.dataframe(subset[cols], hide_index=True, use_container_width=True)

            # Salvar escalação no histórico
            if st.button("💾 Salvar esta escalação no meu histórico"):
                atleta_ids = df_escal["atleta_id"].tolist() if "atleta_id" in df_escal.columns else []
                from user.profile import buscar_usuario
                u_db = buscar_usuario(uid)
                esc = salvar_escalacao(
                    usuario_id=uid,
                    rodada=rodada,
                    temporada=2025,
                    jogadores=atleta_ids,
                    formacao=resultado.get("formacao","4-3-3"),
                    orcamento_usado=resumo.get("custo_total",0),
                    pts_esperado=resumo.get("pts_esperado",0),
                )
                st.success(f"Escalação salva (id={esc.id}). Registre o resultado após a rodada!")

            # Download
            csv = df_escal.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Baixar escalação (CSV)", csv,
                               file_name=f"escalacao_usuario{uid}_r{rodada}.csv", mime="text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Histórico
# ══════════════════════════════════════════════════════════════════════════════

def _tab_historico():
    st.subheader("Meu histórico de escalações")

    uid = _usuario_id()
    if not uid:
        st.warning("Selecione um usuário na aba **Meu Perfil**.")
        return

    from user.profile import historico_usuario, registrar_resultado

    hist = historico_usuario(uid)
    if not hist:
        st.info("Nenhuma escalação salva ainda.")
        return

    df_hist = pd.DataFrame(hist)
    concluidas = df_hist[df_hist["pts_real"].notna()]

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total escalações", len(df_hist))
    c2.metric("Rodadas concluídas", len(concluidas))
    if not concluidas.empty:
        c3.metric("Média pts reais",    f"{concluidas['pts_real'].mean():.1f}")
        c4.metric("MAE médio",          f"{concluidas['erro_abs'].mean():.1f} pts")

    # Tabela
    st.divider()
    st.dataframe(df_hist, hide_index=True, use_container_width=True,
        column_config={
            "pts_esperado": st.column_config.NumberColumn("Pts esperados", format="%.1f"),
            "pts_real":     st.column_config.NumberColumn("Pts reais",     format="%.1f"),
            "erro_abs":     st.column_config.NumberColumn("Erro absoluto", format="%.1f"),
            "orcamento_us": st.column_config.NumberColumn("Custo (C$)",    format="%.1f"),
        }
    )

    # Gráfico de evolução
    if not concluidas.empty and len(concluidas) > 1:
        st.subheader("Evolução de pontuação por rodada")
        st.line_chart(concluidas.set_index("rodada")[["pts_esperado","pts_real"]])

    # Registrar resultado manualmente
    st.divider()
    st.subheader("Registrar resultado de uma rodada")
    pendentes = df_hist[df_hist["pts_real"].isna()]
    if pendentes.empty:
        st.caption("Todas as escalações já têm resultado registrado.")
    else:
        with st.form("form_resultado"):
            opcoes_esc = {f"Rodada {r['rodada']} (id={r['escalacao_id']})": r["escalacao_id"]
                          for _, r in pendentes.iterrows()}
            esc_sel    = st.selectbox("Escalação", list(opcoes_esc.keys()))
            pts        = st.number_input("Pontuação real obtida", min_value=0.0, step=0.5)
            aval       = st.slider("Avaliação (1–5 ⭐)", 1, 5, 3)
            coment     = st.text_area("Comentário (opcional)")

            if st.form_submit_button("Registrar"):
                registrar_resultado(
                    escalacao_id=opcoes_esc[esc_sel],
                    pts_real=pts,
                    avaliacao=aval,
                    comentario=coment or None,
                )
                st.success("Resultado registrado! O modelo usará isso para melhorar.")
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Auto-Learning
# ══════════════════════════════════════════════════════════════════════════════

def _tab_autolearn():
    st.subheader("Auto-Learning — Modelo em evolução")

    from autolearn.engine import (
        ajustar_pesos_score,
        listar_versoes,
        modelo_em_producao,
        retreinar,
        verificar_necessidade_retreino,
    )
    from model.trainer import carregar_metricas

    # Status atual
    prod     = modelo_em_producao()
    metricas = carregar_metricas()

    c1, c2, c3 = st.columns(3)
    c1.metric("Versão em produção", prod.get("versao","—") if prod else "—")
    c2.metric("MAE atual",  f"{metricas.get('mae_medio',0):.2f} pts" if metricas else "—")
    c3.metric("R² atual",   f"{metricas.get('r2_medio',0):.3f}"     if metricas else "—")

    # Verificação de necessidade
    check = verificar_necessidade_retreino()
    if check["precisa"]:
        st.warning(f"⚠️ Re-treino recomendado: {check['motivo']}")
    else:
        st.success(f"✅ Modelo atualizado: {check['motivo']}")

    # Ações manuais
    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        if st.button("🔁 Re-treinar agora", type="primary"):
            with st.spinner("Re-treinando... pode levar alguns minutos."):
                resultado = retreinar(forcar=True)
            if resultado.get("deployed"):
                st.success(f"✅ Novo modelo deployado: {resultado['versao']} | MAE: {resultado['metricas']['mae_medio']:.2f}")
            else:
                st.info(f"Modelo atual mantido. {resultado.get('motivo','')}")
            st.rerun()

    with col_b:
        if st.button("⚖️ Recalcular pesos do score"):
            hist_path = DATA_DIR / "processed" / "historico_completo.parquet"
            if not hist_path.exists():
                st.error("Histórico não disponível.")
            else:
                import pandas as _pd
                from model.features import construir_features
                df_hist = _pd.read_parquet(hist_path)
                df_feat = construir_features(df_hist)
                resultado = ajustar_pesos_score(df_feat)
                st.json(resultado)

    # Histórico de versões
    st.divider()
    st.subheader("Histórico de versões")
    versoes = listar_versoes()
    if versoes:
        df_v = pd.DataFrame([
            {
                "versão":    v["versao"],
                "criado_em": v["criado_em"][:19],
                "mae":       v["metricas"].get("mae_medio", "—"),
                "rmse":      v["metricas"].get("rmse_medio","—"),
                "r²":        v["metricas"].get("r2_medio",  "—"),
                "ativo":     "✅" if v.get("ativo") else "",
            }
            for v in versoes
        ])
        st.dataframe(df_v, hide_index=True, use_container_width=True)

        if len(df_v) > 1:
            st.subheader("Evolução do MAE entre versões")
            df_plot = df_v[df_v["mae"] != "—"].copy()
            df_plot["mae"] = df_plot["mae"].astype(float)
            st.line_chart(df_plot.set_index("versão")["mae"])
    else:
        st.info("Nenhuma versão registrada ainda. Treine o modelo com `python main.py train`.")

    # Pesos sugeridos
    pesos_path = MODELS_DIR / "pesos_sugeridos.json"
    if pesos_path.exists():
        import json
        with open(pesos_path) as f:
            pesos = json.load(f)
        st.divider()
        st.subheader("Pesos sugeridos pelo auto-learning")
        st.json(pesos)
        st.caption(
            "Esses pesos foram calculados automaticamente com base na correlação histórica. "
            "Para aplicá-los, atualize `SCORE_PESOS` em `config/settings.py`."
        )


# Execução standalone
if __name__ == "__main__":
    st.set_page_config(page_title="Fase 3 — Cartola FC", page_icon="🤖", layout="wide")
    render()
