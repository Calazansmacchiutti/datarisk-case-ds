"""Política de crédito a partir de probabilidades calibradas.

Thresholds derivados de percentis da PD na base de validação, não absolutos.
Justificativa: bad rate observada é baixa (~1%), thresholds absolutos como 5% / 10%
classificariam quase todos como baixo risco. Percentis da validação refletem o
ranqueamento relativo do modelo e geram política operável.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def derive_thresholds_from_validation(pd_val: np.ndarray, q_low: float = 0.70, q_med: float = 0.90, q_high: float = 0.97) -> dict:
    """Calcula t1, t2, t3 a partir de percentis da PD na validacao."""
    return {
        "t_low": float(np.quantile(pd_val, q_low)),
        "t_med": float(np.quantile(pd_val, q_med)),
        "t_high": float(np.quantile(pd_val, q_high)),
        "q_low": q_low, "q_med": q_med, "q_high": q_high,
    }


def assign_band(p: float, t_low: float, t_med: float, t_high: float) -> str:
    if p <= t_low:
        return "verde"
    if p <= t_med:
        return "amarela"
    if p <= t_high:
        return "laranja"
    return "vermelha"


def base_decision(band: str) -> str:
    return {
        "verde": "aprovar_automatico",
        "amarela": "aprovar_com_ajuste",
        "laranja": "analise_manual",
        "vermelha": "rejeitar",
    }[band]


def apply_parallel_rules(row: pd.Series, comp_max: float) -> tuple[str, list[str]]:
    """Regras paralelas aplicadas sobre a faixa derivada da PD.

    NOTA: a regra de comprometimento de renda (>comp_max) foi removida apos
    diagnostico que mostrou que 99,4% dos clientes da submissao tem essa razao
    superior a 40%. A feature continua no modelo, mas nao gera regra paralela.
    Ver PIPELINE_LOG.md, iteracao 4 (D-011) e arquivo de decisoes.

    Cold-start (verde) -> rebaixa para amarela (D-012).
    """
    band = row["faixa_risco"]
    motivos = []

    if row.get("flag_sem_historico_credito", 0) == 1 and band == "verde":
        band = "amarela"
        motivos.append("cold-start: aprovar com limite reduzido")

    if not motivos:
        if band == "vermelha":
            motivos.append("PD elevada")
        elif band == "laranja":
            motivos.append("PD alta, comportamento limitrofe")
        elif band == "amarela":
            motivos.append("PD intermediaria")
        else:
            motivos.append("PD baixa")

    return band, motivos


def build_policy_output(
    submission: pd.DataFrame,
    base_submissao: pd.DataFrame,
    base_features: pd.DataFrame,
    pd_val: np.ndarray,
    config: dict,
    output_path: str,
) -> pd.DataFrame:
    pol = config["policy"]
    thresholds = derive_thresholds_from_validation(pd_val)
    logger.info(
        f"thresholds derivados (val): t_low={thresholds['t_low']:.4f} (q{thresholds['q_low']:.0%}), "
        f"t_med={thresholds['t_med']:.4f} (q{thresholds['q_med']:.0%}), "
        f"t_high={thresholds['t_high']:.4f} (q{thresholds['q_high']:.0%})"
    )

    df = submission.merge(
        base_submissao[["id_cliente", "valor_credito", "valor_parcela"]], on="id_cliente"
    ).merge(
        base_features[["id_cliente", "valor_parcela_sobre_renda_mensal", "flag_sem_historico_credito"]],
        on="id_cliente",
        how="left",
    )

    df["faixa_risco"] = df["probabilidade_inadimplencia"].apply(
        lambda p: assign_band(p, thresholds["t_low"], thresholds["t_med"], thresholds["t_high"])
    )
    df["faixa_inicial"] = df["faixa_risco"]

    novas_faixas, motivos_list = [], []
    for _, row in df.iterrows():
        new_band, motivos = apply_parallel_rules(row, pol["comprometimento_renda_max"])
        novas_faixas.append(new_band)
        motivos_list.append("; ".join(motivos))
    df["faixa_risco"] = novas_faixas
    df["motivo_principal"] = motivos_list
    df["decisao_sugerida"] = df["faixa_risco"].apply(base_decision)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cols_out = [
        "id_cliente", "probabilidade_inadimplencia", "faixa_risco",
        "decisao_sugerida", "motivo_principal",
    ]
    df[cols_out].to_csv(output_path, index=False)
    logger.info(f"saida de politica salva em {output_path}")

    summary = df.groupby("decisao_sugerida").agg(
        n=("id_cliente", "size"),
        pd_media=("probabilidade_inadimplencia", "mean"),
        valor_credito_total=("valor_credito", "sum"),
    ).reset_index()
    summary["pct"] = (summary["n"] / len(df) * 100).round(2)
    logger.info(f"resumo da politica:\n{summary.to_string(index=False)}")

    band_summary = df.groupby("faixa_risco").size().reset_index(name="n").sort_values("n", ascending=False)
    band_summary["pct"] = (band_summary["n"] / len(df) * 100).round(2)
    logger.info(f"distribuicao por faixa:\n{band_summary.to_string(index=False)}")

    return df, thresholds
