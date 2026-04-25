<div align="center">

# ⚽ Cartola FC — Sistema Inteligente

**Plataforma de análise preditiva para o Cartola FC**  
Previsão por Machine Learning · Análise de Sentimento NLP · Perfis Personalizados · Automação Completa

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-orange)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 📋 Índice

- [Visão geral](#-visão-geral)
- [Funcionalidades](#-funcionalidades)
- [Arquitetura](#-arquitetura)
- [Pré-requisitos](#-pré-requisitos)
- [Instalação](#-instalação)
- [Início rápido](#-início-rápido)
- [CLI — referência de comandos](#-cli--referência-de-comandos)
- [Dashboard](#-dashboard)
- [API REST](#-api-rest)
- [Configuração](#-configuração)
- [Estrutura do projeto](#-estrutura-do-projeto)
- [Solução de problemas](#-solução-de-problemas)
- [Contribuindo](#-contribuindo)

---

## 🎯 Visão geral

O **Sistema Inteligente para o Cartola FC** é uma plataforma completa que combina coleta de dados via API oficial, Machine Learning e Processamento de Linguagem Natural para maximizar a pontuação dos usuários no Cartola FC. O sistema foi desenvolvido em três fases evolutivas:

| Fase | Módulos | Destaque |
|------|---------|----------|
| **Fase 1 — MVP** | Coleta ETL, modelo XGBoost, otimizador de escalação, dashboard | Pipeline completo da API ao dashboard |
| **Fase 2 — NLP** | Análise de sentimento (VADER + BERT), coleta Twitter/Reddit/notícias | Score de percepção pública integrado ao modelo |
| **Fase 3 — Personalização** | Perfis de usuário, API REST, scheduler automático, auto-learning | Recomendações adaptadas ao histórico individual |

---

## ✨ Funcionalidades

### 🤖 Machine Learning
- Modelo **XGBoost** treinado com validação temporal (`TimeSeriesSplit`)
- Features: média histórica, forma recente (últimas 5 rodadas), dificuldade do adversário, tendência
- **Score composto** ponderado: média histórica (35%) + forma recente (35%) + adversário (20%) + sentimento (10%)
- Fallback automático para **Random Forest** se XGBoost não estiver instalado
- **Auto-learning**: re-treino automático semanal com versionamento de modelos

### 📊 Dados & Pipeline
- Coleta automática via **API pública do Cartola FC**
- Suporte a dois formatos do endpoint histórico: path param `/pontuados/{rodada}` (atual) e query param `?rodada=N` (legado)
- **Fallback offline**: geração de dados sintéticos realistas quando a API está indisponível
- Dados persistidos em **Parquet** (rápido e compacto)

### 💬 Análise de Sentimento (Fase 2)
- Coleta de **Twitter/X**, **Reddit** e portais de notícias (Globo Esporte, UOL Esporte)
- Dois modelos: **BERT** multilingual (preciso) e **VADER** (rápido, sem GPU)
- Seleção automática do modelo disponível (`--modo auto`)

### 👤 Personalização (Fase 3)
- Perfis de usuário com times favoritos, jogadores bloqueados e formação preferida
- Histórico de escalações com métricas de precisão por rodada
- **API REST** (FastAPI) com documentação interativa (Swagger)
- **Scheduler** com jobs automáticos: coleta, sentimento, re-treino, previsões

### 📱 Dashboard Streamlit
- 6 páginas: Ranking, Escalação, Alertas, Sentimento, Fase 3 e Métricas
- Gráficos interativos, filtros por posição e faixa de preço
- Funciona no **GitHub Codespaces** sem configuração adicional

---

## 🏗 Arquitetura

```
                     ┌─────────────────────────────┐
                     │    API Cartola FC (pública)  │
                     └──────────────┬──────────────┘
                                    │ coleta ETL
                     ┌──────────────▼──────────────┐
                     │   data_collection/pipeline   │◄── fallback sintético
                     └──────────────┬──────────────┘
                                    │ Parquet
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
   │  model/features  │  │sentiment/collector│  │  user/profile    │
   │  model/trainer   │  │sentiment/analyzer │  │  user/recommender│
   │  model/predictor │  │sentiment/aggregator│ │  autolearn/engine│
   └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
            │                     │                      │
            └─────────────────────┼──────────────────────┘
                                  ▼
                     ┌─────────────────────────┐
                     │   dashboard/app.py       │  ← Streamlit :8501
                     │   api/app.py             │  ← FastAPI   :8000
                     └─────────────────────────┘
```

---

## 🛠 Pré-requisitos

| Software | Versão mínima |
|----------|--------------|
| Python | 3.11+ (recomendado: 3.12) |
| pip | 23.0+ |
| Git | qualquer versão recente |
| GPU CUDA (opcional) | CUDA 12+ para acelerar BERT |

> **Sistemas operacionais:** Windows 10/11 (recomenda-se WSL2), macOS 12+, Ubuntu 20.04+

---

## 📦 Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/eoneres/Cartola-Analytics.git
cd Cartola-Analytics
```

### 2. Crie o ambiente virtual

```bash
python -m venv .venv

# Linux / macOS / Codespaces:
source .venv/bin/activate

# Windows:
.venv\Scripts\activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

Para habilitar o modelo BERT (análise de sentimento avançada):

```bash
# CPU:
pip install transformers torch

# GPU CUDA 12:
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers
```

> Sem `transformers`/`torch`, o sistema usa o analisador **VADER** automaticamente — mais rápido, sem GPU.

### 4. Configure as credenciais (opcional)

```bash
cp .env.example .env
# Edite .env e preencha TWITTER_BEARER_TOKEN
```

> Sem Twitter: a coleta usa Reddit e scraping de notícias normalmente.

---

## 🚀 Início rápido

### Opção A — Script automático (recomendado)

Um único comando instala, popula os dados e abre o dashboard:

```bash
bash setup.sh
```

Se a API do Cartola não retornar dados históricos (início de temporada, ambiente offline):

```bash
bash setup.sh --sintetico
```

### Opção B — Make

```bash
make setup           # instalação completa + dashboard
make setup-sintetico # idem, usando dados sintéticos
make dash            # só abre o dashboard (setup já realizado)
make status          # status do modelo e jobs
```

### Opção C — Popular dados e abrir manualmente

```bash
# Popula todos os dados (funciona offline)
python scripts/popular_dashboard.py

# Abre o dashboard
streamlit run dashboard/app.py \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false
```

### Opção D — Passo a passo manual

```bash
python main.py collect --rodada-inicio 1 --rodada-fim 14
python main.py train
python main.py predict
streamlit run dashboard/app.py
```

---

## 🖥 CLI — referência de comandos

### `collect` — Coleta de dados

```bash
python main.py collect --rodada-inicio 1 --rodada-fim 14
```

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `--rodada-inicio` | `1` | Primeira rodada a coletar |
| `--rodada-fim` | `10` | Última rodada a coletar |

> Se a API retornar 0 registros, use o fallback sintético:
> ```bash
> python scripts/gerar_dados_sinteticos.py --rodadas 14
> ```

### `train` — Treino do modelo

```bash
python main.py train
```

Treina o XGBoost com validação temporal e exibe métricas:

```
MAE médio: 8.42 pts  |  R² médio: 0.614
```

### `predict` — Previsões

```bash
python main.py predict
```

Gera previsões para todos os atletas e salva em `data/processed/previsoes.parquet`.

### `full` — Pipeline completo

```bash
python main.py full --rodada-inicio 1 --rodada-fim 14
```

### `sentiment` — Análise de sentimento

```bash
python main.py sentiment --jogadores Endrick Gabigol Arrascaeta
python main.py sentiment --jogadores Pedro --modo bert
python main.py sentiment --jogadores Hulk --sem-twitter
```

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `--jogadores` | obrigatório | Lista de nomes |
| `--modo` | `auto` | `auto`, `bert` ou `vader` |
| `--sem-twitter` | — | Ignora coleta de tweets |
| `--sem-reddit` | — | Ignora posts do Reddit |
| `--sem-noticias` | — | Ignora scraping de notícias |

### `scheduler` — Jobs automáticos

```bash
python main.py scheduler start                      # inicia em foreground
python main.py scheduler status                     # lista próximas execuções
python main.py scheduler run --job retreinar        # executa job manualmente
```

Jobs disponíveis: `coletar`, `sentimento`, `retreinar`, `prever`, `pesos`

| Job | Agendamento padrão |
|-----|--------------------|
| Coleta de dados | Seg–Sex às 6h, 12h, 18h e 23h |
| Sentimento | Diariamente às 7h |
| Re-treino | Terças às 3h |
| Previsões | Qui e Sex às 8h |
| Ajuste de pesos | Quartas às 4h |

### `autolearn` — Auto-learning

```bash
python main.py autolearn status
python main.py autolearn retreinar
python main.py autolearn retreinar --forcar
python main.py autolearn versoes
```

### `api` — API REST

```bash
python main.py api --porta 8000
python main.py api --porta 8000 --reload   # modo desenvolvimento
```

### `popular_dashboard` — Populador offline

```bash
python scripts/popular_dashboard.py             # popula tudo
python scripts/popular_dashboard.py --rodadas 20
python scripts/popular_dashboard.py --forcar    # regenera tudo
```

Popula mercado, histórico, partidas, treina o modelo, gera previsões e sentimento. Funciona 100% sem internet.

---

## 📊 Dashboard

Acesse em `http://localhost:8501` após iniciar o Streamlit.

| Página | Descrição |
|--------|-----------|
| **Ranking de Jogadores** | Atletas ordenados por score composto com filtros de posição e preço |
| **Montar Escalação** | Escalação otimizada por perfil de risco (conservador / balanceado / agressivo) |
| **Alertas** | Jogadores em alta, surpresas, riscos e prováveis zeráveis da rodada |
| **Sentimento** | Painel NLP com hype score, tendência e percepção pública |
| **Fase 3 · Personalização** | Perfil, histórico de escalações e auto-learning |
| **Métricas do Modelo** | MAE, RMSE, R² e evolução por fold de validação |

### GitHub Codespaces

O arquivo `.streamlit/config.toml` já está configurado corretamente. Use:

```bash
bash setup.sh --only-dash
```

---

## 🌐 API REST

Acesse `http://localhost:8000/docs` para a documentação interativa (Swagger).

| Método | Rota | Descrição |
|--------|------|-----------|
| `GET` | `/health` | Status da API e versão do modelo |
| `GET` | `/previsoes` | Lista previsões com filtros |
| `GET` | `/previsoes/{atleta_id}` | Previsão detalhada |
| `GET` | `/alertas` | Alertas da rodada |
| `POST` | `/escalacao/otimizar` | Gera escalação sem perfil de usuário |
| `POST` | `/usuarios` | Cria perfil de usuário |
| `POST` | `/usuarios/{id}/escalacao` | Escalação personalizada |
| `POST` | `/usuarios/{id}/resultado` | Registra resultado real |
| `GET` | `/autolearn/status` | Versão em produção |

---

## ⚙️ Configuração

Edite `config/settings.py` para personalizar o comportamento:

```python
# Modelo de ML
MODEL_TYPE = "xgboost"   # ou "random_forest", "linear"

# Pesos do score composto (soma = 1.0)
SCORE_PESOS = {
    "media_historica":  0.35,
    "forma_recente":    0.35,
    "fator_adversario": 0.20,
    "sentimento":       0.10,
}

# Janela de forma recente
JANELA_FORMA = 5

# Orçamento padrão (C$)
ORCAMENTO_PADRAO = 100.0
```

---

## 📁 Estrutura do projeto

```
Cartola-Analytics/
│
├── config/settings.py               # Configurações centrais
├── data_collection/
│   ├── cartola_api.py               # Client da API do Cartola
│   └── pipeline.py                  # ETL com fallback sintético
├── model/
│   ├── features.py                  # Engenharia de features
│   ├── trainer.py                   # Treino com TimeSeriesSplit
│   └── predictor.py                 # Previsões e alertas
├── sentiment/                       # Fase 2 — NLP
│   ├── collector.py                 # Twitter/Reddit/notícias
│   ├── analyzer.py                  # BERT + VADER
│   ├── aggregator.py                # Score consolidado
│   └── dashboard_page.py
├── user/                            # Fase 3 — Personalização
│   ├── profile.py                   # CRUD de perfis (SQLite)
│   └── recommender.py
├── autolearn/engine.py              # Auto-learning e model registry
├── scheduler/jobs.py                # Jobs automáticos (APScheduler)
├── api/app.py                       # API REST FastAPI
├── dashboard/
│   ├── app.py                       # Entry point do Streamlit
│   └── fase3_page.py
├── scripts/
│   ├── popular_dashboard.py         # Popula todos os dados (offline-safe)
│   └── gerar_dados_sinteticos.py    # Histórico sintético como fallback
├── data/
│   ├── raw/                         # Dados brutos por rodada
│   ├── processed/                   # Dados processados + previsões
│   │   └── sentiment/               # Scores NLP
│   └── cache/                       # Cache de API
├── models/                          # Modelos serializados + registry
├── logs/
├── .streamlit/config.toml           # Config do servidor (CORS, tema verde)
├── .env.example                     # Template de credenciais
├── .gitignore
├── Makefile                         # Atalhos para todos os comandos
├── setup.sh                         # Script de inicialização automática
├── requirements.txt
└── main.py                          # Entry point CLI
```

---

## 🔧 Solução de problemas

| Erro | Solução |
|------|---------|
| `Nenhuma previsão encontrada` | `python scripts/popular_dashboard.py` |
| `API retorna 0 registros históricos` | `python scripts/gerar_dados_sinteticos.py --rodadas 14` |
| `ArrowNotImplementedError: vol_por_fonte` | Atualizado na v3.1 — use o ZIP mais recente |
| `SyntaxError: import pandas as pd as _pd` | Corrigido na v3.1 — use o ZIP mais recente |
| Erro de CORS no browser (Codespaces) | `bash setup.sh --only-dash` |
| `ModuleNotFoundError: xgboost` | `pip install xgboost` — RandomForest é usado como fallback |
| `fatal: unable to stat 'data/usuarios.db-shm'` | `git rm --cached data/usuarios.db-shm && git commit` |
| `Treine o modelo primeiro` | `python main.py train` |

---

## 🤝 Contribuindo

1. Faça um fork do repositório
2. Crie uma branch: `git checkout -b feature/minha-feature`
3. Commit: `git commit -m 'feat: minha feature'`
4. Push: `git push origin feature/minha-feature`
5. Abra um Pull Request

---

<div align="center">

Desenvolvido com Python · XGBoost · BERT · FastAPI · Streamlit

</div>
