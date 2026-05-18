"""Leitura das bases brutas."""
from pathlib import Path
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

DATE_COLS = {
    "base_cadastral": ["data_nascimento"],
    "base_submissao": ["data_solicitacao"],
    "historico_emprestimos": [
        "data_decisao",
        "data_liberacao",
        "data_primeiro_vencimento",
        "data_ultimo_vencimento_original",
        "data_ultimo_vencimento",
        "data_encerramento",
    ],
    "historico_parcelas": ["data_prevista_pagamento", "data_real_pagamento"],
}


def _read_parquet(path: Path, date_cols: list[str]) -> pd.DataFrame:
    df = pd.read_parquet(path)
    for c in date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def load_raw_data(config: dict) -> dict[str, pd.DataFrame]:
    raw = Path(config["paths"]["raw_data_dir"])
    files = config["files"]
    out = {}
    for key in ["base_cadastral", "base_submissao", "historico_emprestimos", "historico_parcelas"]:
        path = raw / files[key]
        logger.info(f"lendo {key} de {path}")
        out[key] = _read_parquet(path, DATE_COLS[key])
        logger.info(f"  shape: {out[key].shape}")
    return out
