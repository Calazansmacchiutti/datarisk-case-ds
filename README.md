# Case Técnico — Modelo de Inadimplência em Crédito Pessoa Física

## Resumo da entrega

Esta solução entrega:

- Modelo preditivo para estimar a probabilidade de inadimplência em crédito PF;
- Target principal FPD5, definido como atraso superior a 5 dias na primeira parcela;
- Pipeline reprodutível com leitura dos dados, construção do target, engenharia de features, treino, calibração, avaliação e scoring;
- Arquivo oficial `outputs/submissao_case.csv` com as colunas exigidas: `id_cliente` e `probabilidade_inadimplencia`;
- Política de crédito baseada nas probabilidades calibradas;
- Streamlit com narrativa executiva e técnica para avaliação;
- Notebooks e relatórios complementares para auditoria da abordagem.

## Validação da submissão

O arquivo `outputs/submissao_case.csv` foi gerado no formato exigido pelo desafio:

- 40.000 linhas, uma para cada cliente da `base_submissao.parquet`;
- Colunas: `id_cliente` e `probabilidade_inadimplencia`;
- Probabilidades no intervalo [0, 1];
- Sem colunas adicionais.

A saída auxiliar `outputs/decisoes_credito_simuladas.csv` contém a aplicação da política de crédito, mas não faz parte da submissão oficial.



## Guia de Avaliação

A entrega foi pensada para ser percorrida em três passos, do mais sintético ao mais detalhado:

| Ordem | Artefato | Conteúdo |
|---|---|---|
| **1** | **`app.py` (Streamlit)** | Narrativa principal da entrega — objetivo, target, modelo, avaliação e política, com métricas e figuras dinâmicas. **Comece aqui.** |
| **2** | **`notebooks/solucao_credito.ipynb`** | Notebook que executa a solução end-to-end com os dados que sustentam a narrativa do Streamlit: target FPD5, modelo LightGBM calibrado, avaliação e geração do `submissao_case.csv`. |
| **3** | **`notebooks/exploracao_decisoes.ipynb`** | Notebook de exploração: análise das bases, comparação de definições de target, diagnóstico de features e justificativas técnicas das decisões adotadas. |

> O arquivo `outputs/submissao_case.csv` com as probabilidades de inadimplência por cliente **já está gerado** e disponível no repositório — não é necessário re-executar o pipeline para avaliar a entrega.

---

## Executando o Streamlit

```bash
pip install -r requirements.txt
streamlit run app.py
```

O app abre em `http://localhost:8501` e está organizado em seis seções no menu lateral: Visão Geral, Dados e Target, Engenharia e Anti-leakage, Modelagem, Avaliação e Política de Crédito.

---

## Objetivo

Construir um modelo preditivo para estimar a probabilidade de inadimplência — definida como **First Payment Default em 5 dias corridos (FPD5)** — em contratos de crédito pessoa física, com **política de concessão de crédito** derivada das probabilidades calibradas pelo modelo.

## Resultados Principais

| Conjunto | AUC | KS | Brier Score |
|---|---|---|---|
| Validação (2024 H1) | 0,710 | 0,378 | 0,00547 |
| Teste (2024 H2) | 0,634 | 0,290 | 0,00358 |

A queda de desempenho entre validação e teste é coerente com o **split temporal estrito** e com o baixo bad rate do período de teste (~0,36%, apenas 26 bads). Os intervalos de confiança via bootstrap se sobrepõem, e o modelo mantém ordenação discriminante útil para política de crédito: os **20% clientes mais arriscados capturam 42% dos bads**.

---

## Documentação Complementar

Em ordem de relevância para aprofundamento técnico:

1. **`reports/relatorio_abordagem.md`** — documento de referência: definição do target, engenharia de features, modelagem, calibração, política de crédito, métricas e limitações conhecidas.
2. **`reports/dicionario_features.md`** — catálogo das ~85 variáveis construídas, com descrição e origem de cada feature.
3. **`reports/decis_test_fpd5.csv`** — tabela de decis no conjunto de teste.
4. **`reports/metricas_modelo.json`** — métricas globais e thresholds da política em formato estruturado.

---

## Estrutura do Código

```
src/
├── data/        leitura, validação, deduplicação de parcelas e exclusões
├── target/      FPD5 (primário) e EVER15MOB03 (secundário, para validação cruzada do target)
├── features/    variáveis cadastrais, de solicitação, de contratos e de comportamento de pagamento
├── models/      split temporal, treino (regressão logística + LightGBM), avaliação, calibração e predição
├── policy/      faixas de risco por quantil e regras paralelas de crédito
├── utils/       logger, io, métricas (KS, tabela de decis)
├── run_pipeline.py    orquestrador end-to-end do pipeline
└── experiment.py      executor de experimentos controlados
```

---

## Reprodução da Solução

### Requisitos

Python 3.11.

### Instalação de dependências

```bash
pip install -r requirements.txt
```

### Posicionamento dos dados

Antes de executar o pipeline, posicionar os arquivos fornecidos pelo case em `data/raw/`:

```
data/raw/base_cadastral.parquet
data/raw/base_submissao.parquet
data/raw/historico_emprestimos.parquet
data/raw/historico_parcelas.parquet
```

### Execução do pipeline completo

```bash
python -m src.run_pipeline --config config/config.yaml
```

Alternativamente, via Makefile:

```bash
make install   # instalação das dependências
make run       # execução do pipeline completo
make test      # execução da suíte de testes
```

Tempo médio de execução: 3 minutos.

### Artefatos gerados

| Arquivo | Descrição |
|---|---|
| `outputs/submissao_case.csv` | **Entrega oficial** — colunas `id_cliente` e `probabilidade_inadimplencia` |
| `outputs/decisoes_credito_simuladas.csv` | Saída auxiliar: faixa de risco e decisão sugerida por cliente (não integra a submissão oficial) |
| `reports/metricas_modelo.json` | Métricas globais e thresholds da política de crédito |
| `reports/decis_test_fpd5.csv` | Tabela de decis no conjunto de teste |
| `models/lightgbm.joblib` | Modelo serializado |

---

## Configuração

Todos os parâmetros relevantes estão centralizados em `config/config.yaml`:

- **target**: threshold de dias em atraso (dpd), janela de maturidade (mob) e critérios de elegibilidade do contrato
- **cohort**: status de contratos válidos e gap máximo permitido entre data de decisão e primeiro vencimento
- **split**: datas de corte para treino, validação e teste
- **model**: hiperparâmetros do LightGBM e método de calibração de probabilidades
- **policy**: quantis utilizados para a definição dos thresholds das faixas de risco
- **drop_columns**: colunas excluídas explicitamente do pipeline de modelagem

---

## Testes Automatizados

```bash
pytest tests/ -v
```

Suíte com 15 testes cobrindo: lógica de construção do target FPD5, restrições de anti-leakage temporal, comportamento esperado em clientes sem histórico de crédito (cold-start) e validações de formato e integridade da submissão.

---

## Observações

- As bases de dados originais não estão incluídas na entrega. O código reproduz todo o fluxo a partir dos arquivos fornecidos pelo case.
- A solução é determinística: `random_state=42` em todos os componentes que envolvem aleatoriedade.
- O arquivo de submissão (`outputs/submissao_case.csv`) já está disponível no repositório e pode ser consultado sem necessidade de re-execução do pipeline.


## Limitações conhecidas

- O período de teste possui baixo volume de eventos de inadimplência: 26 bads, com bad rate aproximada de 0,36%.
- Por isso, métricas pontuais como AUC no teste devem ser interpretadas com cautela.
- A avaliação foi complementada com intervalos de confiança via bootstrap e análise de decis.
- Foi observado drift entre validação e teste em variáveis importantes, reforçando a necessidade de monitoramento periódico em produção.
- O target FPD5 mede inadimplência precoce, não perda final; por isso, o modelo também foi avaliado contra o target secundário EVER15MOB03.