"""Streamlit app — Modelo de Inadimplência em Crédito Pessoa Física."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
FIGURES = ROOT / "reports" / "figures"
REPORTS = ROOT / "reports"
OUTPUTS = ROOT / "outputs"

st.set_page_config(
    page_title="Modelo de Inadimplência — Datarisk",
    page_icon="📊",
    layout="wide",
)

# ── sidebar navigation ─────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Navegação")
    secao = st.radio(
        "Seção",
        [
            "Visão Geral",
            "Dados e Target",
            "Engenharia e Anti-leakage",
            "Modelagem",
            "Avaliação",
            "Política de Crédito",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Case Técnico — Crédito PF")
    st.caption("Modelo: LightGBM + Platt calibration")
    st.caption("Target: FPD5 (atraso > 5 dias na 1ª parcela)")


# ── helpers ────────────────────────────────────────────────────────────────────
@st.cache_data
def load_metricas():
    with open(REPORTS / "metricas_modelo.json") as f:
        return json.load(f)


@st.cache_data
def load_decis():
    return pd.read_csv(REPORTS / "decis_test_fpd5.csv")


@st.cache_data
def load_submissao():
    return pd.read_csv(OUTPUTS / "submissao_case.csv")


@st.cache_data
def load_decisoes():
    return pd.read_csv(OUTPUTS / "decisoes_credito_simuladas.csv")


def fig(name: str):
    path = FIGURES / name
    if path.exists():
        st.image(str(path), use_container_width=True)
    else:
        st.warning(f"Figura não encontrada: {name}")


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 1 — VISÃO GERAL
# ══════════════════════════════════════════════════════════════════════════════
if secao == "Visão Geral":
    st.title("Modelo de Inadimplência em Crédito Pessoa Física")
    st.markdown(
        """
        Este projeto desenvolve uma solução **end-to-end** para estimar a probabilidade de inadimplência
        de clientes em novas solicitações de crédito pessoa física. A entrega final gera o arquivo
        `submissao_case.csv`, com uma probabilidade de inadimplência para cada cliente da
        `base_submissao.parquet`, e propõe uma **política de crédito** baseada nessas probabilidades.

        A inadimplência foi definida como **FPD5**, isto é, atraso superior a 5 dias na primeira parcela
        do contrato. Essa definição foi escolhida por ser observável nas bases fornecidas, adequada ao
        momento de decisão de crédito e mais estável do que FPD1. O target FPD5 apresentou
        **1.190 bads em 105.092 contratos elegíveis**, com bad rate de **1,13%**.

        O modelo final utiliza **LightGBM com calibração sigmoid/Platt**, treinado em janela temporal
        2020–2023 (62.181 contratos, 1,0% bad rate) e validado em períodos futuros. Na validação 2024 H1,
        o modelo atingiu **AUC de 0,710 com KS de 0,378**; no teste 2024 H2, **AUC de 0,634 com KS de 0,290**.
        Apesar da baixa quantidade de bads no teste (26 eventos), os decis de maior risco concentram parte
        relevante da inadimplência: os **20% clientes mais arriscados capturam 42% dos bads**.

        A política de crédito divide os clientes em **quatro faixas de risco** com base nos quantis da
        PD calibrada: verde para aprovação automática, amarela para aprovação com ajuste, laranja para
        análise manual e vermelha para rejeição. Clientes sem histórico de crédito recebem tratamento
        conservador adicional, sendo rebaixados da faixa verde para a amarela.
        """
    )

    metricas = load_metricas()
    val = metricas["calibrated_val"]
    tst = metricas["calibrated_test"]
    thr = metricas["policy_thresholds"]

    st.divider()
    st.subheader("Métricas Principais")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Validação** (2024 H1 — 12.828 contratos)")
        c1, c2, c3 = st.columns(3)
        c1.metric("AUC", f"{val['auc']:.3f}")
        c2.metric("KS", f"{val['ks']:.3f}")
        c3.metric("Brier", f"{val['brier']:.5f}")

    with col2:
        st.markdown("**Teste** (2024 H2 — 7.216 contratos)")
        c1, c2, c3 = st.columns(3)
        c1.metric("AUC", f"{tst['auc']:.3f}")
        c2.metric("KS", f"{tst['ks']:.3f}")
        c3.metric("Brier", f"{tst['brier']:.5f}")

    st.divider()
    st.subheader("Arquitetura da Solução")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            **Pipeline**
            - Dados: 4 bases (cadastral, empréstimos, parcelas, submissão)
            - Target: FPD5 — atraso > 5 dias na primeira parcela
            - Janela de treino: 2020–2023 (62.181 contratos, 1,0% bad rate)
            - Modelo: LightGBM + calibração sigmoid (Platt)
            - Anti-leakage: filtro temporal estrito por `data_decisao`

            **Features (~85 variáveis)**
            - Cadastrais: renda, idade, ocupação, moradia, posse de bens
            - Solicitação: valor, parcela/renda, tipo de contrato, LTV
            - Histórico de contratos: aprovados, recusados, cancelados, recência
            - Pagamento: pré-pagamento, atrasos, perfil disciplinado
            """
        )
    with col2:
        st.markdown(
            """
            **Política de Crédito — 4 faixas**

            | Faixa | Threshold | Decisão |
            |---|---|---|
            | 🟢 Verde | PD ≤ {t_low:.4f} (q70) | Aprovar automático |
            | 🟡 Amarela | PD ≤ {t_med:.4f} (q90) | Aprovar com ajuste |
            | 🟠 Laranja | PD ≤ {t_high:.4f} (q97) | Análise manual |
            | 🔴 Vermelha | PD > {t_high:.4f} | Rejeitar |

            **Regra adicional**
            - Cold-start (sem histórico) na faixa verde → rebaixado para amarela
            """.format(
                t_low=thr["t_low"],
                t_med=thr["t_med"],
                t_high=thr["t_high"],
            )
        )

    st.divider()
    st.subheader("Entregáveis")
    sub = load_submissao()
    dec = load_decisoes()
    col1, col2, col3 = st.columns(3)
    col1.metric("Clientes na Submissão", f"{len(sub):,}")
    col2.metric("PD Média", f"{sub['probabilidade_inadimplencia'].mean():.4f}")
    col3.metric("PD Mediana", f"{sub['probabilidade_inadimplencia'].median():.4f}")

    st.caption(
        "Arquivo de submissão: `outputs/submissao_case.csv` — colunas `id_cliente` e `probabilidade_inadimplencia`"
    )


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 2 — DADOS E TARGET
# ══════════════════════════════════════════════════════════════════════════════
elif secao == "Dados e Target":
    st.title("Dados e Definição do Target")

    st.subheader("Bases de dados")
    st.markdown(
        """
        | Base | Linhas | Período | Uso |
        |---|---|---|---|
        | `base_cadastral` | 40.000 | — | Features cadastrais |
        | `base_submissao` | 40.000 | fev/2025 | Clientes para predição |
        | `historico_emprestimos` | 186.890 | 2017–2025 | Target + features históricas |
        | `historico_parcelas` | 1.390.978 | 2017–2025 | Features de pagamento |

        94,9% dos clientes da submissão têm histórico de crédito. **5,1% são cold-start** (sem contratos anteriores).
        """
    )

    st.divider()
    st.subheader("Volume de Contratos ao Longo do Tempo")
    fig("vol_contratos.png")

    st.divider()
    st.subheader("Evolução da Bad Rate (FPD5)")
    col1, col2 = st.columns(2)
    with col1:
        fig("bad_rate_temporal.png")
    with col2:
        fig("target_evolucao.png")

    st.divider()
    st.subheader("Por que FPD5?")
    st.markdown(
        """
        **88,8% das primeiras parcelas são pré-pagas** (mediana: 11 dias antes do vencimento).
        Nesse contexto, atraso superior a 5 dias na primeira parcela é um desvio comportamental significativo.

        | Definição | Contratos | Bad rate | Bads |
        |---|---|---|---|
        | FPD1 | 105.092 | 1,13% | 1.190 — sensível demais (ruído operacional) |
        | **FPD5** | **105.092** | **1,13%** | **1.190 — escolhido** |
        | EVER15MOB03 | 102.162 | 1,60% | 1.638 — target secundário (validação cruzada) |
        | EVER30MOB03 | ~ | ~ | ~ — poucos bads para modelagem estável |

        **Target secundário (EVER15MOB03)**: o modelo treinado em FPD5 avaliado contra EVER15MOB03
        na validação entrega AUC 0,7159 / KS 0,4100 — evidência de captura de padrões reais de risco,
        não apenas comportamento específico da primeira parcela.

        **Janelas de treino/validação/teste** (split temporal estrito):
        """
    )
    fig("janelas_treino.png")

    col1, col2, col3 = st.columns(3)
    col1.markdown(
        """
        **Treino** 2020–2023
        - 62.181 contratos
        - Bad rate: 1,00%
        """
    )
    col2.markdown(
        """
        **Validação** 2024 H1
        - 12.828 contratos
        - Bad rate: 0,55%
        """
    )
    col3.markdown(
        """
        **Teste** 2024 H2+
        - 7.216 contratos
        - Bad rate: 0,36%
        """
    )


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 3 — ENGENHARIA E ANTI-LEAKAGE
# ══════════════════════════════════════════════════════════════════════════════
elif secao == "Engenharia e Anti-leakage":
    st.title("Engenharia de Atributos e Anti-leakage")

    st.markdown(
        """
        Foram criadas aproximadamente **85 variáveis explicativas** combinando informações
        cadastrais, características da solicitação, histórico de contratos e comportamento de
        pagamento. Toda a engenharia foi construída com **separação temporal estrita** para
        evitar vazamento de informação no momento da decisão de crédito.
        """
    )

    st.divider()
    st.subheader("Famílias de Features")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            **Cadastrais**
            - Renda, idade, ocupação
            - Tipo de moradia, posse de bens (carro, imóvel)
            - Atributos demográficos e socioeconômicos

            **Solicitação**
            - Valor solicitado, prazo, parcela
            - Razão parcela/renda, LTV
            - Tipo de contrato (`flag_revolving`, `flag_consumer`)
            """
        )
    with col2:
        st.markdown(
            """
            **Histórico de contratos**
            - Contratos aprovados, recusados, cancelados
            - Recência da última solicitação
            - Volume contratado acumulado

            **Comportamento de pagamento**
            - Padrão de pré-pagamento (mediana, frequência)
            - Atrasos históricos, parcelas em aberto
            - Perfil disciplinado vs. recorrente em atraso
            """
        )

    st.divider()
    st.subheader("Anti-leakage: âncora temporal em `data_decisao`")
    st.markdown(
        """
        Cada feature histórica é calculada **somente com informações disponíveis até a
        `data_decisao`** do contrato em treino, e até a `data_solicitacao` do registro em
        scoring. Isso replica a condição real do momento de concessão e impede que comportamento
        posterior contamine o sinal.

        - `data_liberacao` foi descartada como âncora por ter **96,2% de nulos** (D-003).
        - Contratos com gap `data_decisao → primeiro_vencimento > 90 dias` foram filtrados — o
          gap máximo observado era de **2.178 dias** (outlier) contra mediana de **31 dias** (D-013).
        - A `base_submissao` **não foi usada** para treinar; apenas para scoring final.
        """
    )

    st.divider()
    st.subheader("Tratamento de Cold-start (sem histórico)")
    st.markdown(
        """
        **5,1% dos clientes da submissão** não possuem contratos anteriores observáveis.
        Em vez de imputar com média/mediana — o que diluiria o sinal de novidade — adotamos:

        | Tratamento | Efeito |
        |---|---|
        | `fillna(0)` em features históricas e de pagamento | Preserva zeros como "ausência" |
        | `flag_sem_historico` e `flag_sem_historico_pagamento` | Carregam o sinal categórico |
        | Rebaixamento de faixa verde → amarela na política | Incerteza adicional na decisão |

        Essa abordagem (D-006 + D-012) permite que o LightGBM aprenda splits específicos para
        cold-start, sem perder os clientes maturados.
        """
    )

    st.divider()
    st.subheader("Cohort de modelagem")
    st.markdown(
        """
        A modelagem usa contratos **Approved com maturação suficiente** para observar FPD5
        como base de target. Contratos **Refused e Canceled** não geram target, mas alimentam
        features comportamentais ricas (recência de recusa, taxa histórica de cancelamento,
        etc.) — ver D-005.

        Também houve **deduplicação de parcelas** (D-007): mantida apenas a `versao_parcela`
        mais recente para cada `(id_contrato, numero_parcela)`. Sem isso, 78.606 parcelas
        (5,7%) com múltiplas versões poluiriam as features de pagamento.
        """
    )


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 4 — MODELAGEM
# ══════════════════════════════════════════════════════════════════════════════
elif secao == "Modelagem":
    st.title("Modelagem")

    st.subheader("Algoritmo: LightGBM")
    st.markdown(
        """
        Modelo único com `flag_revolving` e `flag_consumer` como features — LightGBM cria splits naturais
        por tipo de contrato sem necessidade de modelos separados (D-008).

        **Hiperparâmetros** (conservadores, bad rate baixa):
        | Parâmetro | Valor | Justificativa |
        |---|---|---|
        | `num_leaves` | 15 | Evita memorização |
        | `max_depth` | 5 | Evita memorização |
        | `min_data_in_leaf` | 500 | ~6–7 bads mínimos por folha |
        | `learning_rate` | 0,03 | Convergência gradual |
        | `reg_alpha / reg_lambda` | 0,5 / 1,0 | Regularização L1+L2 |
        | `n_estimators` | 1000 + early stopping (30) | Parada por validação |
        """
    )

    st.divider()
    st.subheader("Importância das Features")
    fig("feature_importance.png")

    st.divider()
    st.subheader("Calibração: Sigmoid vs Isotônica (D-014)")
    col1, col2 = st.columns([2, 1])
    with col1:
        fig("calibracao_comparativo.png")
    with col2:
        st.markdown(
            """
            **Problema com isotônica**: colapsava 7 dos 10 decis em score idêntico (0,00266),
            tornando 70% da política indistinguível.

            **Correção na sigmoid**: bug detectado onde regularização L2 padrão da `LogisticRegression`
            sob bad rate baixa produzia coeficiente negativo (correlação score bruto vs calibrado = -1,0).
            Correção: logit-link + C=1e6 (sem regularização efetiva).

            **Resultado**: 10/10 decis com scores únicos, ranqueamento granular para a política.
            """
        )

    st.divider()
    st.subheader("Seleção da Janela de Treino (D-017)")
    col1, col2 = st.columns([2, 1])
    with col1:
        fig("gap_regularizacao.png")
    with col2:
        st.markdown(
            """
            Testadas 4 janelas com bootstrap pareado (2.000 reamostragens):

            | Janela | Test AUC | P(melhor) |
            |---|---|---|
            | 2017–2023 | 0,604 | — |
            | **2020–2023** | **0,634** | 88% |
            | 2022–2023 | 0,631 | 80% |
            | 2023 só | 0,660 | 92% |

            2020–2023 escolhida: melhor val AUC, volume razoável (62k),
            evita portfólio pré-COVID com perfil distinto.
            """
        )


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 5 — AVALIAÇÃO
# ══════════════════════════════════════════════════════════════════════════════
elif secao == "Avaliação":
    st.title("Avaliação do Modelo")

    metricas = load_metricas()
    val = metricas["calibrated_val"]
    tst = metricas["calibrated_test"]

    st.subheader("Métricas Globais (LightGBM Calibrado)")
    df_met = pd.DataFrame(
        {
            "Split": ["Validação", "Teste"],
            "n": [val["n"], tst["n"]],
            "Bad rate": [f"{val['bad_rate']:.2%}", f"{tst['bad_rate']:.2%}"],
            "AUC": [f"{val['auc']:.4f}", f"{tst['auc']:.4f}"],
            "KS": [f"{val['ks']:.4f}", f"{tst['ks']:.4f}"],
            "Brier": [f"{val['brier']:.5f}", f"{tst['brier']:.5f}"],
            "Log Loss": [f"{val['log_loss']:.5f}", f"{tst['log_loss']:.5f}"],
        }
    )
    st.dataframe(df_met, hide_index=True, use_container_width=True)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Curva ROC")
        fig("roc_curve.png")

    with col2:
        st.subheader("Bootstrap + Drift de Features (D-015 e D-016)")
        fig("gap_bootstrap_drift.png")
        st.markdown(
            """
            **ICs bootstrap se sobrepõem** — gap val/teste não é estatisticamente significativo.
            - Validação (71 bads): AUC 0,710 CI95% [0,64; 0,77]
            - Teste (26 bads): AUC 0,634 CI95% [0,50; 0,76]

            **Drift**: 13/15 features mais importantes com KS p<1e-30 entre val e teste.
            Portfólio mudou entre 2024 H1 e H2 — aceito como ruído + drift estrutural (D-015).
            """
        )

    st.divider()
    st.subheader("Tabela de Decis — Conjunto de Teste")
    decis = load_decis()
    decis_fmt = decis.copy()
    decis_fmt["score_mean"] = decis_fmt["score_mean"].map("{:.4f}".format)
    decis_fmt["score_min"] = decis_fmt["score_min"].map("{:.4f}".format)
    decis_fmt["score_max"] = decis_fmt["score_max"].map("{:.4f}".format)
    decis_fmt["bad_rate"] = (decis["bad_rate"] / 100).map("{:.2%}".format)
    decis_fmt["pct_bads_acumulado"] = decis_fmt["pct_bads_acumulado"].map("{:.1f}%".format)
    decis_fmt.columns = [
        "Decil", "n", "Bads", "Score Médio", "Score Mín", "Score Máx", "Bad Rate", "% Bads Acum."
    ]
    st.dataframe(decis_fmt, hide_index=True, use_container_width=True)

    st.markdown(
        """
        Decis 9+10 (top 20% mais arriscados) capturam **42% dos bads**.
        Decil 10 tem score médio 9,4× maior que decil 1. Lift top 10% = 1,92×.
        """
    )

    st.divider()
    st.subheader("Avaliação por Segmento (Teste Calibrado)")
    df_seg = pd.DataFrame(
        {
            "Segmento": ["Cash loans", "Consumer loans", "Revolving loans"],
            "n": [2343, 4341, 532],
            "Bads": [6, 14, 6],
            "AUC": ["0,5347", "0,5938", "0,6480"],
            "KS": ["0,2394", "0,2650", "0,4861"],
            "Observação": [
                "6 bads → IC±0,18, AUC indistinguível de 0,40–0,70",
                "Volume maior, mais confiável",
                "Melhor discriminação no segmento",
            ],
        }
    )
    st.dataframe(df_seg, hide_index=True, use_container_width=True)
    st.caption(
        "Com poucos bads por segmento no teste, a métrica mais confiável de performance segmentada "
        "é a tabela de decis — não a AUC pontual."
    )

    st.divider()
    st.subheader("Robustez: Target Secundário (EVER15MOB03)")
    st.markdown(
        """
        O modelo treinado em FPD5 foi avaliado contra EVER15MOB03 na validação:

        | Métrica | Valor |
        |---|---|
        | AUC | 0,7159 |
        | KS | 0,4100 |

        Performance comparável ao target primário — evidência de captura de padrões reais de risco,
        não apenas comportamento da primeira parcela.
        """
    )


# ══════════════════════════════════════════════════════════════════════════════
# SEÇÃO 6 — POLÍTICA DE CRÉDITO
# ══════════════════════════════════════════════════════════════════════════════
elif secao == "Política de Crédito":
    st.title("Política de Crédito")

    metricas = load_metricas()
    thr = metricas["policy_thresholds"]
    dec = load_decisoes()

    st.subheader("Thresholds Derivados da Validação (D-010)")
    st.markdown(
        """
        A bad rate observada é baixa (~0,55% na validação). Thresholds absolutos pré-definidos
        (ex: 5%, 10%) classificariam quase todos como baixo risco — política inoperante.

        Thresholds derivados de **quantis empíricos da PD calibrada na validação**:
        """
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🟢 t_low (q70)", f"{thr['t_low']:.5f}", "Aprovar automático")
    col2.metric("🟡 t_med (q90)", f"{thr['t_med']:.5f}", "Aprovar com ajuste")
    col3.metric("🟠 t_high (q97)", f"{thr['t_high']:.5f}", "Análise manual")
    col4.metric("🔴 acima de t_high", "3% dos clientes", "Rejeitar")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribuição de PD com Faixas de Risco")
        fig("distribuicao_pd.png")
    with col2:
        st.subheader("Distribuição de Thresholds na Submissão")
        fig("politica_thresholds.png")

    st.divider()
    st.subheader("Resultados da Política na Submissão")
    fig("politica_credito.png")

    col1, col2 = st.columns(2)

    with col1:
        decisao_counts = dec["decisao_sugerida"].value_counts().reset_index()
        decisao_counts.columns = ["Decisão", "Clientes"]
        decisao_counts["% do Total"] = (decisao_counts["Clientes"] / len(dec) * 100).map("{:.1f}%".format)
        label_map = {
            "aprovar_automatico": "🟢 Aprovar automático",
            "aprovar_com_ajuste": "🟡 Aprovar com ajuste",
            "analise_manual": "🟠 Análise manual",
            "rejeitar": "🔴 Rejeitar",
        }
        decisao_counts["Decisão"] = decisao_counts["Decisão"].map(label_map)
        st.dataframe(decisao_counts, hide_index=True, use_container_width=True)

    with col2:
        faixa_pd = dec.groupby("faixa_risco")["probabilidade_inadimplencia"].agg(
            ["mean", "min", "max"]
        ).reset_index()
        faixa_pd.columns = ["Faixa", "PD Média", "PD Mín", "PD Máx"]
        faixa_map = {
            "verde": "🟢 Verde",
            "amarela": "🟡 Amarela",
            "laranja": "🟠 Laranja",
            "vermelha": "🔴 Vermelha",
        }
        faixa_pd["Faixa"] = faixa_pd["Faixa"].map(faixa_map)
        for col in ["PD Média", "PD Mín", "PD Máx"]:
            faixa_pd[col] = faixa_pd[col].map("{:.5f}".format)
        st.dataframe(faixa_pd, hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("Regras Paralelas")
    st.markdown(
        """
        | Regra | Efeito |
        |---|---|
        | Cold-start (sem histórico de crédito) na faixa verde | Rebaixado para amarela — aprovar com limite reduzido |
        | Comprometimento de renda > 40% | **Removida** (D-011): 99,4% dos clientes da submissão têm essa razão acima de 40% — regra seria inoperante |

        A feature de comprometimento de renda **permanece no modelo** como preditor, mas não gera regra paralela de corte.
        """
    )

    st.divider()
    st.subheader("Submissão — Exploração")
    sub = load_submissao()
    with st.expander("Visualizar amostra do arquivo de submissão"):
        st.dataframe(sub.head(20), hide_index=True, use_container_width=True)

    with st.expander("Filtrar decisões por faixa de risco"):
        faixas = ["Todas"] + sorted(dec["faixa_risco"].unique().tolist())
        faixa_sel = st.selectbox("Faixa de risco", faixas)
        if faixa_sel == "Todas":
            df_filtrado = dec
        else:
            df_filtrado = dec[dec["faixa_risco"] == faixa_sel]
        st.dataframe(df_filtrado.head(100), hide_index=True, use_container_width=True)
        st.caption(f"{len(df_filtrado):,} clientes nessa faixa.")

    st.divider()
    st.subheader("Resumo das Decisões Técnicas")

    with st.expander("Decisões Chave e Trade-offs"):
        st.markdown(
            """
            **D-008 | Modelo Único Segmentado**
            Modelo único com flags de tipo de contrato (`flag_revolving`, `flag_consumer`) em vez de modelos
            separados. O LightGBM aprende splits automáticos por segmento, economizando custo operacional
            sem perder discriminação.

            ---

            **D-010 | Thresholds Adaptativos**
            Thresholds da política derivados de **quantis empíricos da PD na validação** (q70, q90, q97),
            não valores absolutos. Com PD média de 0,55%, thresholds fixos (5%, 10%) seriam inoperantes —
            99% dos clientes cairiam em "verde". Quantis adaptam à distribuição real e são parametrizáveis
            pelo apetite de risco do banco.

            ---

            **D-014 | Calibração Sigmoid vs Isotônica**
            Escolhi **Platt scaling** porque isotônica colapsava os 10 decis em apenas 7 scores únicos
            (colapso de discriminação). Com poucos bads na validação (~71), isotônica mapeia regiões inteiras
            para o mesmo valor. O trade-off foi mínimo: queda de 0,003 no AUC, ganho de ranqueamento
            granular que a política precisa para operar.

            ---

            **D-015 | Aceitação do Gap Val/Teste com Diagnóstico**
            O gap entre validação (AUC 0,71) e teste (AUC 0,63) foi investigado profundamente.
            Conclusão: ICs bootstrap se sobrepõem (estatisticamente não significativo) e 33/73 features
            mostram drift estrutural real (p<0,001). Aceitar gap não é resignação — é diagnóstico que
            vai virar requisito de monitoramento em produção.

            ---

            **D-016 | Transparência Estatística**
            Toda métrica de performance é reportada com **IC bootstrap 95%**. Com apenas 26 bads no teste,
            o IC é [0,51; 0,75], mostrando que qualquer valor nesse range é compatível com os dados.
            Ocultar a incerteza leva a decisões precipitadas — o tamanho do IC importa mais que o ponto central.

            ---

            **D-017 | Janela de Treino 2020–2023**
            Reduzir de 2017–2023 para 2020–2023 (descartando portfólio pré-COVID). Bootstrap pareado
            (2.000 reamostras) mostrou P(2020–2023 superior) = 88% no test AUC. Descartar 27% do volume
            vale pela aderência ao portfólio atual.
            """
        )

    st.markdown(
        """
        ---

        ## 📊 Modelo em Produção

        A solução está **pronta para deployment**:
        - Modelo serializado (`models/lightgbm.joblib`)
        - Política de crédito com 4 faixas de risco
        - Scoring gerado para 40.000 clientes (`outputs/submissao_case.csv`)
        - Métricas calibradas com ICs bootstrap
        - Tratamento explícito de cold-start e regras paralelas

        **Próximos passos em produção**:
        1. Integrar scoring ao sistema de concessão
        2. Implementar monitoramento mensal de **drift de features**
        3. Avaliar performance em 90 dias (validação em portfólio novo)
        4. Recalibrar policy se drift > 10% em features críticas
        """
    )
