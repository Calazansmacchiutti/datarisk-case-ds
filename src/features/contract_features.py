"""Features históricas de contratos. Visao temporal: eventos com data_decisao < ref_date."""
import numpy as np
import pandas as pd


REFUSE_REASONS = ["HC", "LIMIT", "SCO", "SCOFR", "VERIF", "SYSTEM"]


def build_contract_features(
    apps: pd.DataFrame, emp: pd.DataFrame
) -> pd.DataFrame:
    """Para cada (id_cliente, ref_date) em apps, agrega historico de contratos com data_decisao < ref_date.

    apps: DataFrame com colunas id_cliente, ref_date, id_contrato (NaN para submissao). Pode trazer outras.
    emp: historico de emprestimos (deduplicado, com colunas tipicas).
    """
    # cross-join cliente x contratos do mesmo cliente
    join = apps[["id_cliente", "ref_date"]].drop_duplicates().merge(
        emp[
            [
                "id_cliente", "id_contrato", "data_decisao", "status_contrato", "tipo_contrato",
                "valor_credito", "valor_parcela", "qtd_parcelas_planejadas", "motivo_recusa",
                "flag_seguro_contratado", "canal_venda", "tipo_produto",
            ]
        ],
        on="id_cliente",
        how="left",
    )
    # filtro temporal
    valid = join["data_decisao"].notna() & (join["data_decisao"] < join["ref_date"])
    h = join[valid].copy()

    # contadores por status
    h["is_approved"] = (h["status_contrato"] == "Approved").astype(int)
    h["is_refused"] = (h["status_contrato"] == "Refused").astype(int)
    h["is_canceled"] = (h["status_contrato"] == "Canceled").astype(int)
    h["is_unused"] = (h["status_contrato"] == "Unused offer").astype(int)

    # janela temporal (3, 6, 12 meses antes do ref_date)
    h["dias_antes"] = (h["ref_date"] - h["data_decisao"]).dt.days
    h["last_3m"] = (h["dias_antes"] <= 90).astype(int)
    h["last_6m"] = (h["dias_antes"] <= 180).astype(int)
    h["last_12m"] = (h["dias_antes"] <= 365).astype(int)

    # motivos de recusa one-hot
    for r in REFUSE_REASONS:
        h[f"recusa_{r}"] = ((h["status_contrato"] == "Refused") & (h["motivo_recusa"] == r)).astype(int)

    # agregacao por (id_cliente, ref_date)
    grp = h.groupby(["id_cliente", "ref_date"], dropna=False)
    agg = grp.agg(
        qtd_contratos_previos=("id_contrato", "count"),
        qtd_aprovados_prev=("is_approved", "sum"),
        qtd_recusados_prev=("is_refused", "sum"),
        qtd_cancelados_prev=("is_canceled", "sum"),
        qtd_unused_offer_prev=("is_unused", "sum"),
        qtd_contratos_3m=("last_3m", "sum"),
        qtd_contratos_6m=("last_6m", "sum"),
        qtd_contratos_12m=("last_12m", "sum"),
        dias_desde_ultimo_contrato=("dias_antes", "min"),
        dias_desde_primeiro_contrato=("dias_antes", "max"),
        valor_credito_medio_hist=("valor_credito", "mean"),
        valor_credito_max_hist=("valor_credito", "max"),
        valor_parcela_medio_hist=("valor_parcela", "mean"),
        qtd_parcelas_media_hist=("qtd_parcelas_planejadas", "mean"),
        pct_seguro_hist=("flag_seguro_contratado", "mean"),
        **{f"qtd_recusa_{r}": (f"recusa_{r}", "sum") for r in REFUSE_REASONS},
    ).reset_index()

    # taxas
    n = agg["qtd_contratos_previos"].replace(0, np.nan)
    agg["taxa_aprovacao_hist"] = agg["qtd_aprovados_prev"] / n
    agg["taxa_recusa_hist"] = agg["qtd_recusados_prev"] / n
    agg["taxa_cancelamento_hist"] = agg["qtd_cancelados_prev"] / n

    # produto / canal mais frequentes
    def top_mode(s):
        s = s.dropna()
        return s.mode().iloc[0] if len(s) else np.nan

    aux = grp.agg(
        produto_mais_freq=("tipo_produto", top_mode),
        canal_mais_freq=("canal_venda", top_mode),
        tipo_contrato_mais_freq=("tipo_contrato", top_mode),
        motivo_recusa_mais_freq=("motivo_recusa", top_mode),
    ).reset_index()
    agg = agg.merge(aux, on=["id_cliente", "ref_date"], how="left")

    # join de volta nas apps
    out = apps.merge(agg, on=["id_cliente", "ref_date"], how="left")
    # cold-start
    fill_zero = [
        "qtd_contratos_previos", "qtd_aprovados_prev", "qtd_recusados_prev",
        "qtd_cancelados_prev", "qtd_unused_offer_prev", "qtd_contratos_3m",
        "qtd_contratos_6m", "qtd_contratos_12m",
    ] + [f"qtd_recusa_{r}" for r in REFUSE_REASONS]
    for c in fill_zero:
        out[c] = out[c].fillna(0).astype(int)
    out["flag_sem_historico_credito"] = (out["qtd_contratos_previos"] == 0).astype(int)

    return out
