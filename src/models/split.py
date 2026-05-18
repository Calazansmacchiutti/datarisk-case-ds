"""Split temporal da base de treino."""
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def temporal_split(df: pd.DataFrame, config: dict, ref_col: str = "data_decisao") -> dict:
    """Retorna dicionario {train, val, test} com os indices/dataframes ja separados.

    Suporta train_start opcional para janela de treino reduzida (D-013).
    """
    cfg = config["split"]
    ref = pd.to_datetime(df[ref_col])
    train_end = pd.Timestamp(cfg["train_end"])
    val_start = pd.Timestamp(cfg["validation_start"])
    val_end = pd.Timestamp(cfg["validation_end"])
    test_start = pd.Timestamp(cfg["test_start"])

    train_mask = ref <= train_end
    if cfg.get("train_start"):
        train_start_ts = pd.Timestamp(cfg["train_start"])
        train_mask &= (ref >= train_start_ts)
        logger.info(f"janela de treino restringida a partir de {cfg['train_start']}")

    val_mask = (ref >= val_start) & (ref <= val_end)
    test_mask = ref >= test_start

    out = {"train": df[train_mask].copy(), "val": df[val_mask].copy(), "test": df[test_mask].copy()}
    for name, sub in out.items():
        if len(sub) > 0:
            logger.info(f"split {name}: {len(sub):,} contratos, periodo {sub[ref_col].min()} a {sub[ref_col].max()}")
        else:
            logger.warning(f"split {name}: VAZIO")
    return out
