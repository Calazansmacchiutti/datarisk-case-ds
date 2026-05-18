"""Orquestrador de features. Funciona para treino (mode='train') e inferencia (mode='score')."""
from typing import Literal
import numpy as np
import pandas as pd

from src.features.customer_features import build_customer_features
from src.features.contract_features import build_contract_features
from src.features.payment_features import build_payment_features
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _add_application_features(apps: pd.DataFrame, ref_col: str) -> pd.DataFrame:
    """Features da solicitacao atual."""
    df = apps.copy()
    # razoes financeiras
    df["valor_credito_sobre_renda_anual"] = df["valor_credito"] / df["renda_anual"].replace(0, np.nan)
    df["valor_credito_sobre_renda_mensal"] = df["valor_credito"] / df["renda_mensal"].replace(0, np.nan)
    df["valor_parcela_sobre_renda_mensal"] = df["valor_parcela"] / df["renda_mensal"].replace(0, np.nan)
    df["ltv_estimado"] = df["valor_credito"] / df["valor_bem"].replace(0, np.nan)
    df["flag_valor_bem_missing"] = df["valor_bem"].isna().astype(int)
    df["flag_credito_maior_que_bem"] = (df["valor_credito"] > df["valor_bem"].fillna(np.inf)).astype(int)
    df["qtd_parcelas_implicita"] = df["valor_credito"] / df["valor_parcela"].replace(0, np.nan)

    # tempo
    df["periodo_dia_solicitacao"] = pd.cut(
        df["hora_solicitacao"],
        bins=[-1, 5, 11, 17, 23],
        labels=["madrugada", "manha", "tarde", "noite"],
    ).astype(str)
    df["flag_fim_de_semana"] = df["dia_semana_solicitacao"].isin(["SATURDAY", "SUNDAY"]).astype(int)

    # segmento
    df["flag_revolving"] = (df["tipo_contrato"] == "Revolving loans").astype(int)
    df["flag_consumer"] = (df["tipo_contrato"] == "Consumer loans").astype(int)

    return df


def make_features(
    applications: pd.DataFrame,
    cad: pd.DataFrame,
    emp: pd.DataFrame,
    par: pd.DataFrame,
    ref_col: str,
    mode: Literal["train", "score"],
) -> pd.DataFrame:
    """Pipeline unificado de features.

    applications: para 'train' precisa ter colunas id_cliente, id_contrato, data_decisao,
        target_fpd5, target_ever15mob03, e as colunas da solicitacao do contrato historico.
        Para 'score' precisa ter as colunas da base_submissao + id_cliente.
    ref_col: nome da coluna de referencia temporal ('data_decisao' ou 'data_solicitacao').
    """
    apps = applications.copy()
    apps["ref_date"] = pd.to_datetime(apps[ref_col])

    logger.info(f"[{mode}] features cadastrais ({len(apps):,} linhas)")
    apps = build_customer_features(cad, apps)

    logger.info(f"[{mode}] features de solicitacao")
    apps = _add_application_features(apps, ref_col)

    logger.info(f"[{mode}] features de historico de contratos")
    apps = build_contract_features(apps, emp)

    logger.info(f"[{mode}] features de pagamentos e pre-pagamento")
    apps = build_payment_features(apps, emp, par)

    logger.info(f"[{mode}] base final: shape {apps.shape}")
    return apps


def build_training_applications(emp: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Cria base de treino: contratos com target + colunas da solicitacao historica."""
    cols_emp = [
        "id_contrato", "id_cliente", "data_decisao", "tipo_contrato",
        "valor_credito", "valor_bem", "valor_parcela",
        "dia_semana_solicitacao", "hora_solicitacao",
    ]
    base = targets.merge(emp[cols_emp], on=["id_contrato", "id_cliente", "data_decisao"], how="left")
    return base


def build_scoring_applications(sub: pd.DataFrame) -> pd.DataFrame:
    """Cria base de inferencia a partir de base_submissao."""
    out = sub.copy()
    return out


SEGMENT_NUM_COLS = [
    "idade", "qtd_filhos", "qtd_membros_familia", "renda_anual", "renda_mensal",
    "dependentes", "dependentes_por_renda_mensal", "filhos_por_membro",
    "flag_possui_bens", "nota_regiao_cliente", "nota_regiao_cliente_cidade",
    "nota_regiao_diff", "flag_ocupacao_missing", "flag_idade_fora_padrao",
    "valor_credito", "valor_bem", "valor_parcela", "hora_solicitacao",
    "valor_credito_sobre_renda_anual", "valor_credito_sobre_renda_mensal",
    "valor_parcela_sobre_renda_mensal", "ltv_estimado", "flag_valor_bem_missing",
    "flag_credito_maior_que_bem", "qtd_parcelas_implicita", "flag_fim_de_semana",
    "flag_revolving", "flag_consumer",
    "qtd_contratos_previos", "qtd_aprovados_prev", "qtd_recusados_prev",
    "qtd_cancelados_prev", "qtd_unused_offer_prev", "qtd_contratos_3m",
    "qtd_contratos_6m", "qtd_contratos_12m", "dias_desde_ultimo_contrato",
    "dias_desde_primeiro_contrato", "valor_credito_medio_hist", "valor_credito_max_hist",
    "valor_parcela_medio_hist", "qtd_parcelas_media_hist", "pct_seguro_hist",
    "qtd_recusa_HC", "qtd_recusa_LIMIT", "qtd_recusa_SCO", "qtd_recusa_SCOFR",
    "qtd_recusa_VERIF", "qtd_recusa_SYSTEM",
    "taxa_aprovacao_hist", "taxa_recusa_hist", "taxa_cancelamento_hist",
    "flag_sem_historico_credito",
    "qtd_parcelas_observadas", "qtd_parcelas_validas", "max_dias_atraso", "media_dias_atraso",
    "qtd_atraso_pos", "qtd_atraso_5", "qtd_atraso_15", "qtd_atraso_30", "qtd_atraso_60",
    "qtd_pre_pagas", "media_dias_pre_pagamento", "max_dias_pre_pagamento",
    "pct_atraso_pos", "pct_atraso_15", "pct_atraso_30", "pct_pre_pagas",
    "flag_perfil_disciplinado", "flag_ja_teve_atraso_15", "flag_ja_teve_atraso_30",
    "flag_sem_historico_pagamento",
]

SEGMENT_CAT_COLS = [
    "sexo", "tipo_renda", "ocupacao", "tipo_organizacao", "nivel_educacao",
    "estado_civil", "tipo_moradia", "possui_carro", "possui_imovel",
    "tipo_contrato", "dia_semana_solicitacao", "periodo_dia_solicitacao", "faixa_idade",
    "produto_mais_freq", "canal_mais_freq", "tipo_contrato_mais_freq", "motivo_recusa_mais_freq",
]


def split_X_y(df: pd.DataFrame, target_col: str | None = None):
    """Separa preditores (numericos + categoricos) do target. Retorna (X_num, X_cat, y_or_none)."""
    num = [c for c in SEGMENT_NUM_COLS if c in df.columns]
    cat = [c for c in SEGMENT_CAT_COLS if c in df.columns]
    X = df[num + cat].copy()
    for c in cat:
        X[c] = X[c].fillna("OUTROS").astype(str)
    y = df[target_col].values if target_col and target_col in df.columns else None
    return X, num, cat, y
