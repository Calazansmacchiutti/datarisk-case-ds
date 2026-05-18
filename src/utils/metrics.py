"""Métricas de risco."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """KS = max |TPR - FPR| varrendo thresholds."""
    df = pd.DataFrame({"y": y_true, "score": y_score}).sort_values("score")
    total_pos = df["y"].sum()
    total_neg = len(df) - total_pos
    if total_pos == 0 or total_neg == 0:
        return float("nan")
    df["cum_pos"] = df["y"].cumsum() / total_pos
    df["cum_neg"] = (1 - df["y"]).cumsum() / total_neg
    return float((df["cum_pos"] - df["cum_neg"]).abs().max())


def basic_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    out = {
        "n": int(len(y_true)),
        "bad_rate": float(y_true.mean()) if len(y_true) else float("nan"),
        "auc": float("nan"),
        "ks": float("nan"),
        "ap": float("nan"),
        "brier": float("nan"),
        "log_loss": float("nan"),
    }
    if len(np.unique(y_true)) < 2:
        return out
    out["auc"] = float(roc_auc_score(y_true, y_score))
    out["ks"] = ks_statistic(y_true, y_score)
    out["ap"] = float(average_precision_score(y_true, y_score))
    out["brier"] = float(brier_score_loss(y_true, y_score))
    out["log_loss"] = float(log_loss(y_true, np.clip(y_score, 1e-9, 1 - 1e-9)))
    return out


def decile_table(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Tabela de decis ordenada do menor para o maior risco."""
    df = pd.DataFrame({"y": np.asarray(y_true), "score": np.asarray(y_score)})
    df["decil"] = pd.qcut(df["score"].rank(method="first"), q=n_bins, labels=False) + 1
    g = df.groupby("decil").agg(
        n=("y", "size"),
        bads=("y", "sum"),
        score_mean=("score", "mean"),
        score_min=("score", "min"),
        score_max=("score", "max"),
    )
    g["bad_rate"] = g["bads"] / g["n"]
    total_bads = df["y"].sum()
    g["pct_bads_acumulado"] = g.sort_index(ascending=False)["bads"].cumsum().sort_index() / max(total_bads, 1)
    return g.reset_index()
