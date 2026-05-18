"""Features de pagamento. Inclui pré-pagamento (sinal forte nesse dataset)."""
import numpy as np
import pandas as pd


def build_payment_features(
    apps: pd.DataFrame, emp: pd.DataFrame, par: pd.DataFrame
) -> pd.DataFrame:
    """Para cada (id_cliente, ref_date), agrega historico de parcelas com data_prevista_pagamento < ref_date.

    Anti-leakage: usa parcelas de contratos com data_decisao < ref_date E data_prevista < ref_date.
    """
    # parcelas + decisao do contrato
    pj = par[
        ["id_contrato", "id_cliente", "numero_parcela", "data_prevista_pagamento", "data_real_pagamento", "dias_atraso"]
    ].merge(
        emp[["id_contrato", "data_decisao"]], on="id_contrato", how="left"
    )

    # join temporal com apps por id_cliente
    join = apps[["id_cliente", "ref_date"]].drop_duplicates().merge(
        pj, on="id_cliente", how="left"
    )

    # filtros anti-leakage:
    # - contrato decidido antes da ref_date
    # - parcela com data_prevista anterior a ref_date (so olha parcelas que ja venceram antes da decisao atual)
    valid = (
        join["data_decisao"].notna()
        & (join["data_decisao"] < join["ref_date"])
        & join["data_prevista_pagamento"].notna()
        & (join["data_prevista_pagamento"] < join["ref_date"])
    )
    h = join[valid].copy()

    # flags de atraso
    h["dias_atraso"] = h["dias_atraso"].fillna(-999)  # -999 sinal para "data_real ausente"
    h["valid_atraso"] = (h["dias_atraso"] != -999).astype(int)
    h.loc[h["dias_atraso"] == -999, "dias_atraso"] = np.nan

    h["atraso_pos"] = (h["dias_atraso"] > 0).astype(float)
    h["atraso_5"] = (h["dias_atraso"] > 5).astype(float)
    h["atraso_15"] = (h["dias_atraso"] > 15).astype(float)
    h["atraso_30"] = (h["dias_atraso"] > 30).astype(float)
    h["atraso_60"] = (h["dias_atraso"] > 60).astype(float)
    h["pre_pago"] = (h["dias_atraso"] < 0).astype(float)
    h["dias_pre_pagamento"] = (-h["dias_atraso"]).clip(lower=0)

    grp = h.groupby(["id_cliente", "ref_date"], dropna=False)
    agg = grp.agg(
        qtd_parcelas_observadas=("dias_atraso", "size"),
        qtd_parcelas_validas=("valid_atraso", "sum"),
        max_dias_atraso=("dias_atraso", "max"),
        media_dias_atraso=("dias_atraso", "mean"),
        qtd_atraso_pos=("atraso_pos", "sum"),
        qtd_atraso_5=("atraso_5", "sum"),
        qtd_atraso_15=("atraso_15", "sum"),
        qtd_atraso_30=("atraso_30", "sum"),
        qtd_atraso_60=("atraso_60", "sum"),
        qtd_pre_pagas=("pre_pago", "sum"),
        media_dias_pre_pagamento=("dias_pre_pagamento", "mean"),
        max_dias_pre_pagamento=("dias_pre_pagamento", "max"),
    ).reset_index()

    n = agg["qtd_parcelas_observadas"].replace(0, np.nan)
    agg["pct_atraso_pos"] = agg["qtd_atraso_pos"] / n
    agg["pct_atraso_15"] = agg["qtd_atraso_15"] / n
    agg["pct_atraso_30"] = agg["qtd_atraso_30"] / n
    agg["pct_pre_pagas"] = agg["qtd_pre_pagas"] / n
    agg["flag_perfil_disciplinado"] = (agg["pct_pre_pagas"] > 0.8).astype(int)
    agg["flag_ja_teve_atraso_15"] = (agg["qtd_atraso_15"] > 0).astype(int)
    agg["flag_ja_teve_atraso_30"] = (agg["qtd_atraso_30"] > 0).astype(int)

    out = apps.merge(agg, on=["id_cliente", "ref_date"], how="left")
    fill_zero = [
        "qtd_parcelas_observadas", "qtd_parcelas_validas",
        "qtd_atraso_pos", "qtd_atraso_5", "qtd_atraso_15", "qtd_atraso_30", "qtd_atraso_60",
        "qtd_pre_pagas", "flag_perfil_disciplinado", "flag_ja_teve_atraso_15", "flag_ja_teve_atraso_30",
    ]
    for c in fill_zero:
        out[c] = out[c].fillna(0).astype(int)
    out["flag_sem_historico_pagamento"] = (out["qtd_parcelas_observadas"] == 0).astype(int)
    return out
