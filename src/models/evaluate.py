"""Avaliação: métricas globais, por segmento, decis."""
from pathlib import Path
import pandas as pd

from src.utils.io import save_json
from src.utils.logger import get_logger
from src.utils.metrics import basic_metrics, decile_table

logger = get_logger(__name__)


def evaluate_split(model, X, y, name: str) -> dict:
    proba = model.predict_proba(X)[:, 1]
    m = basic_metrics(y, proba)
    logger.info(
        f"[{name}] n={m['n']:,}  bad_rate={m['bad_rate']:.2%}  AUC={m['auc']:.4f}  "
        f"KS={m['ks']:.4f}  Brier={m['brier']:.4f}  LogLoss={m['log_loss']:.4f}"
    )
    return {"split": name, **m}


def evaluate_by_segment(model, X, y, seg_series: pd.Series, name: str) -> dict:
    out = {}
    for seg_value in seg_series.unique():
        mask = (seg_series == seg_value).values
        if mask.sum() == 0:
            continue
        m = basic_metrics(y[mask], model.predict_proba(X[mask])[:, 1])
        out[str(seg_value)] = m
        logger.info(
            f"  [{name}|{seg_value}] n={m['n']:,}  bad_rate={m['bad_rate']:.2%}  "
            f"AUC={m['auc']:.4f}  KS={m['ks']:.4f}"
        )
    return out


def evaluate_secondary_target(model, X, y_secondary, name: str) -> dict:
    """Avalia o modelo treinado em FPD5 contra o EVER15MOB03 para validacao de robustez."""
    valid = y_secondary >= 0  # -1 indica fora do cohort secundario
    if valid.sum() == 0:
        return {}
    proba = model.predict_proba(X[valid])[:, 1]
    m = basic_metrics(y_secondary[valid], proba)
    logger.info(
        f"[{name}|robustez EVER15MOB03] n={m['n']:,}  bad_rate={m['bad_rate']:.2%}  "
        f"AUC={m['auc']:.4f}  KS={m['ks']:.4f}"
    )
    return m


def save_decile_report(model, X, y, path: str) -> pd.DataFrame:
    proba = model.predict_proba(X)[:, 1]
    table = decile_table(y, proba)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    logger.info(f"tabela de decis salva em {path}")
    return table


def save_metrics(metrics: dict, path: str) -> None:
    save_json(metrics, path)
    logger.info(f"metricas salvas em {path}")
