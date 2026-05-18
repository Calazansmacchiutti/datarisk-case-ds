"""Construção dos targets FPD5 (primario) e EVER15MOB03 (secundario)."""
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _eligible_contracts(emp: pd.DataFrame, status_validos: list[str], gap_max_dias: int) -> pd.DataFrame:
    """Approved + data_primeiro_vencimento valida + gap razoavel decisao->primeiro_venc."""
    elig = emp[
        emp["status_contrato"].isin(status_validos)
        & emp["data_primeiro_vencimento"].notna()
        & emp["data_decisao"].notna()
    ].copy()
    elig["gap_dias"] = (elig["data_primeiro_vencimento"] - elig["data_decisao"]).dt.days
    before = len(elig)
    elig = elig[(elig["gap_dias"] >= 0) & (elig["gap_dias"] <= gap_max_dias)]
    logger.info(f"elegíveis após filtro de gap: {before:,} -> {len(elig):,}")
    return elig


def build_target_fpd5(
    emp: pd.DataFrame, par: pd.DataFrame, config: dict
) -> pd.DataFrame:
    """FPD5: atraso > 5 dias na primeira parcela, em contratos com pelo menos 30 dias de maturação."""
    snapshot = pd.Timestamp(config["project"]["snapshot_date"])
    cfg = config["target"]["primary"]
    cohort_cfg = config["cohort"]

    elig = _eligible_contracts(
        emp, cohort_cfg["status_validos_target"], cohort_cfg["gap_max_decisao_primeiro_venc_dias"]
    )
    elig = elig[elig["data_primeiro_vencimento"] + pd.Timedelta(days=cfg["min_maturity_days"]) <= snapshot]
    logger.info(f"contratos maduros para FPD5: {len(elig):,}")

    primeira = par[par["numero_parcela"] == 1][["id_contrato", "dias_atraso"]]
    df = elig[["id_contrato", "id_cliente", "data_decisao"]].merge(primeira, on="id_contrato", how="inner")

    df["target_fpd5"] = (df["dias_atraso"].fillna(-999) > cfg["dpd_threshold"]).astype(int)
    bad = df["target_fpd5"].mean()
    logger.info(f"FPD5: cohort {len(df):,} contratos, bad rate {bad:.2%}, bads {int(df['target_fpd5'].sum()):,}")
    return df[["id_contrato", "id_cliente", "data_decisao", "target_fpd5"]]


def build_target_ever15mob03(
    emp: pd.DataFrame, par: pd.DataFrame, config: dict
) -> pd.DataFrame:
    """EVER15MOB03: atraso > 15 dias em qualquer parcela nos primeiros 3 meses, com 30d de buffer."""
    snapshot = pd.Timestamp(config["project"]["snapshot_date"])
    cfg = config["target"]["secondary"]
    cohort_cfg = config["cohort"]

    elig = _eligible_contracts(
        emp, cohort_cfg["status_validos_target"], cohort_cfg["gap_max_decisao_primeiro_venc_dias"]
    )

    janela = par[["id_contrato", "data_prevista_pagamento", "dias_atraso"]].merge(
        elig[["id_contrato", "data_primeiro_vencimento"]], on="id_contrato", how="inner"
    )
    janela["mes_offset"] = (
        (janela["data_prevista_pagamento"] - janela["data_primeiro_vencimento"]).dt.days / 30.44
    )
    janela = janela[(janela["mes_offset"] >= -0.1) & (janela["mes_offset"] <= cfg["mob_months"] + 0.1)]
    janela = janela[
        janela["data_prevista_pagamento"] + pd.Timedelta(days=cfg["min_maturity_days_after_window"]) <= snapshot
    ]
    janela["bad_parcela"] = (janela["dias_atraso"].fillna(-999) > cfg["dpd_threshold"]).astype(int)

    target = janela.groupby("id_contrato")["bad_parcela"].max().rename("target_ever15mob03").reset_index()

    elig_target = elig[elig["data_primeiro_vencimento"] + pd.Timedelta(days=cfg["mob_months"] * 31 + cfg["min_maturity_days_after_window"]) <= snapshot]
    target = target.merge(elig_target[["id_contrato", "id_cliente", "data_decisao"]], on="id_contrato", how="inner")

    bad = target["target_ever15mob03"].mean()
    logger.info(f"EVER15MOB03: cohort {len(target):,} contratos, bad rate {bad:.2%}, bads {int(target['target_ever15mob03'].sum()):,}")
    return target[["id_contrato", "id_cliente", "data_decisao", "target_ever15mob03"]]


def build_targets(emp: pd.DataFrame, par: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Retorna base com ambos os targets joinados em id_contrato."""
    primary = build_target_fpd5(emp, par, config)
    secondary = build_target_ever15mob03(emp, par, config)
    base = primary.merge(
        secondary[["id_contrato", "target_ever15mob03"]], on="id_contrato", how="left"
    )
    base["target_ever15mob03"] = base["target_ever15mob03"].fillna(-1).astype(int)
    return base
