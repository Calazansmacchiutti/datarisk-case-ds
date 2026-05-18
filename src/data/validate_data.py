"""Checks de qualidade. Loga warnings e segue, não aborta exceto em casos críticos."""
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def validate_raw_data(data: dict[str, pd.DataFrame]) -> None:
    cad = data["base_cadastral"]
    sub = data["base_submissao"]
    emp = data["historico_emprestimos"]
    par = data["historico_parcelas"]

    assert cad["id_cliente"].is_unique, "id_cliente duplicado em base_cadastral"
    assert sub["id_cliente"].is_unique, "id_cliente duplicado em base_submissao"

    cob_sub = sub["id_cliente"].isin(cad["id_cliente"]).mean()
    cob_emp = emp["id_cliente"].isin(cad["id_cliente"]).mean()
    cob_par_emp = par["id_contrato"].isin(emp["id_contrato"]).mean()
    logger.info(f"cobertura sub->cad: {cob_sub:.2%}")
    logger.info(f"cobertura emp->cad: {cob_emp:.2%}")
    logger.info(f"cobertura par->emp: {cob_par_emp:.2%}")

    if cob_sub < 1.0:
        logger.warning(f"{(1-cob_sub)*100:.2f}% dos clientes da submissao nao estao na cadastral")

    null_rates = (emp.isna().mean() * 100).round(2)
    high_null = null_rates[null_rates > 90]
    if len(high_null):
        logger.info(f"colunas em historico_emprestimos com null > 90%: {high_null.to_dict()}")

    cold_start = ~sub["id_cliente"].isin(emp["id_cliente"])
    logger.info(f"clientes cold-start na submissao: {cold_start.sum():,} ({cold_start.mean():.2%})")
