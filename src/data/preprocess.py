"""Pré-processamento: deduplicação de parcelas, dias_atraso, drops."""
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def deduplicate_parcelas(par: pd.DataFrame) -> pd.DataFrame:
    """Mantem apenas a versao mais recente por (id_contrato, numero_parcela)."""
    before = len(par)
    par = par.sort_values(["id_contrato", "numero_parcela", "versao_parcela"], na_position="first")
    par = par.drop_duplicates(["id_contrato", "numero_parcela"], keep="last")
    logger.info(f"parcelas deduplicadas: {before:,} -> {len(par):,} ({before-len(par):,} removidas)")
    return par


def add_dias_atraso(par: pd.DataFrame) -> pd.DataFrame:
    """Calcula dias de atraso. Negativo = pre-pagamento."""
    par = par.copy()
    par["dias_atraso"] = (par["data_real_pagamento"] - par["data_prevista_pagamento"]).dt.days
    return par


def drop_useless_columns(emp: pd.DataFrame, drop_cols: list[str]) -> pd.DataFrame:
    cols = [c for c in drop_cols if c in emp.columns]
    if cols:
        logger.info(f"descartando colunas: {cols}")
        emp = emp.drop(columns=cols)
    return emp


def filter_xna(emp: pd.DataFrame) -> pd.DataFrame:
    before = len(emp)
    emp = emp[emp["tipo_contrato"] != "XNA"].copy()
    if len(emp) < before:
        logger.info(f"filtrados {before-len(emp)} contratos XNA")
    return emp


def preprocess(data: dict[str, pd.DataFrame], config: dict) -> dict[str, pd.DataFrame]:
    par = data["historico_parcelas"]
    par = deduplicate_parcelas(par)
    par = add_dias_atraso(par)
    data["historico_parcelas"] = par

    emp = data["historico_emprestimos"]
    emp = drop_useless_columns(emp, config.get("drop_columns", []))
    if config["cohort"].get("filtrar_xna", True):
        emp = filter_xna(emp)
    data["historico_emprestimos"] = emp
    return data
