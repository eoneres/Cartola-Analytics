# Cartola FC — Sistema Inteligente (MVP)

Sistema preditivo de escalação para o Cartola FC, com coleta de dados, modelo de Machine Learning e dashboard interativo.

## Estrutura do Projeto

```
cartola-mvp/
├── config/
│   └── settings.py          # Configurações centrais
├── data_collection/
│   ├── cartola_api.py        # Cliente da API do Cartola FC
│   ├── brasileirao_scraper.py# Scraper de estatísticas
│   └── pipeline.py           # Pipeline ETL completo
├── model/
│   ├── features.py           # Feature engineering
│   ├── trainer.py            # Treinamento do modelo
│   └── predictor.py          # Geração de previsões
├── dashboard/
│   └── app.py                # Dashboard Streamlit
├── requirements.txt
└── main.py                   # Ponto de entrada
```

## Instalação

```bash
pip install -r requirements.txt
```

## Uso

### 1. Coletar dados
```bash
python main.py collect
```

### 2. Treinar modelo
```bash
python main.py train
```

### 3. Gerar previsões para a próxima rodada
```bash
python main.py predict --rodada 15
```

### 4. Abrir dashboard
```bash
streamlit run dashboard/app.py
```

## Configuração

Edite `config/settings.py` para ajustar:
- Temporada atual
- Pesos do score composto
- Hiperparâmetros do modelo
- Limite de orçamento para escalação

## Métricas do MVP

- **MAE** (Mean Absolute Error) na pontuação prevista
- **Top-N accuracy**: % de vezes que o jogador real está entre os N recomendados
- **ROI médio** nas ligas simuladas
