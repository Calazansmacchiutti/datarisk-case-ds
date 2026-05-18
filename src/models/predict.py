"""Geração da submissão final."""
from pathlib import Path
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def generate_submission(model, X_score, ids: pd.Series, output_path: str) -> pd.DataFrame:
    proba = model.predict_proba(X_score)[:, 1]
    out = pd.DataFrame({"id_cliente": ids.values, "probabilidade_inadimplencia": proba})

    # validacoes
    assert out["id_cliente"].is_unique, "id_cliente duplicado na submissao"
    assert out["probabilidade_inadimplencia"].notna().all(), "probabilidade nula encontrada"
    assert ((out["probabilidade_inadimplencia"] >= 0) & (out["probabilidade_inadimplencia"] <= 1)).all(), \
        "probabilidade fora de [0,1]"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    logger.info(
        f"submissao salva em {output_path} | linhas: {len(out):,} | "
        f"PD media: {out['probabilidade_inadimplencia'].mean():.4f} | "
        f"PD mediana: {out['probabilidade_inadimplencia'].median():.4f}"
    )
    return out
