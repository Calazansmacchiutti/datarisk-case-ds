"""Calibração isotônica/sigmoide manual das probabilidades.

A API CalibratedClassifierCV(cv='prefit') foi removida no sklearn 1.5+.
Implementação manual via IsotonicRegression / Platt scaling preserva o estimador original.

Nota sobre Platt scaling (sigmoid):
- Platt original ajusta uma sigmoid sobre o LOGIT da probabilidade bruta, não sobre a probabilidade.
- Em datasets com bad rate baixa (<1%), usar C=1.0 padrão na LogisticRegression deixa
  a regularização dominar e o coeficiente pode ficar próximo de zero ou negativo.
- Solução: aplicar logit(p) e usar C alto (sem regularização efetiva).
- Ver PIPELINE_LOG.md, iteração 5 (S-07) para o diagnóstico completo.
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _safe_logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


class CalibratedPipeline:
    """Aplica o pipeline original e calibra a saida via mapping monotonico."""

    def __init__(self, base_pipeline, X_val_raw, y_val, method: str = "isotonic"):
        self.pipeline = base_pipeline
        self.method = method
        proba_raw = base_pipeline.predict_proba(X_val_raw)[:, 1]
        if method == "isotonic":
            self.cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self.cal.fit(proba_raw, y_val)
        elif method == "sigmoid":
            # Platt scaling: ajusta logit + LR sem regularizacao efetiva
            logit_raw = _safe_logit(proba_raw)
            self.cal = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
            self.cal.fit(logit_raw.reshape(-1, 1), y_val)
            # checagem de sanidade: coef deve ser positivo (mais score -> mais bad)
            coef = float(self.cal.coef_[0, 0])
            if coef <= 0:
                logger.warning(
                    f"Platt scaling com coeficiente <= 0 ({coef:.4f}). "
                    "Possivel inversao de sinal. Verificar volume de bads na validacao."
                )
        else:
            raise ValueError(f"metodo de calibracao desconhecido: {method}")
        logger.info(f"calibracao {method} aplicada com {len(y_val):,} amostras de validacao")

    def predict_proba(self, X_raw):
        proba_raw = self.pipeline.predict_proba(X_raw)[:, 1]
        if self.method == "isotonic":
            cal = np.clip(self.cal.predict(proba_raw), 0.0, 1.0)
        else:
            logit_raw = _safe_logit(proba_raw)
            cal = self.cal.predict_proba(logit_raw.reshape(-1, 1))[:, 1]
        return np.column_stack([1 - cal, cal])
