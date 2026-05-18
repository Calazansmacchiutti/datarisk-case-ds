"""Runner de experimentos controlados.

Permite rodar a pipeline com config override e capturar metricas em JSON,
sem regerar a submissao final. Cada experimento muda UMA variavel.

Uso:
    python -m src.experiment --name p01_sigmoid --override calibration_method=sigmoid
    python -m src.experiment --name d013_treino_recente --override split.train_start=2022-01-01
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load_data import load_raw_data
from src.data.preprocess import preprocess
from src.data.validate_data import validate_raw_data
from src.features.build_features import (
    build_training_applications,
    make_features,
    split_X_y,
)
from src.models.calibrate import CalibratedPipeline
from src.models.evaluate import basic_metrics
from src.models.split import temporal_split
from src.models.train import train_lightgbm, train_logistic
from src.target.build_target import build_targets
from src.utils.io import load_config, save_json
from src.utils.logger import get_logger
from src.utils.metrics import decile_table, ks_statistic

logger = get_logger("experiment")


def apply_override(cfg: dict, key_path: str, value):
    """Aplica override do tipo 'a.b.c=valor' no config."""
    keys = key_path.split(".")
    cur = cfg
    for k in keys[:-1]:
        cur = cur[k]
    # parse numerico se possivel
    try:
        if "." in str(value) and value.replace(".", "").replace("-", "").isdigit():
            value = float(value)
        elif str(value).lstrip("-").isdigit():
            value = int(value)
    except Exception:
        pass
    cur[keys[-1]] = value
    logger.info(f"override aplicado: {key_path} = {value}")


def metrics_for_segment(y, proba, name=""):
    m = basic_metrics(y, proba)
    return m


def run_experiment(name: str, overrides: list[str], config_path: str = "config/config.yaml") -> dict:
    """Roda pipeline ate avaliacao (sem submissao) e retorna metricas."""
    t0 = time.time()
    cfg = load_config(config_path)
    for ov in overrides:
        key, val = ov.split("=", 1)
        apply_override(cfg, key, val)

    # filtragem opcional: train_start
    train_start = cfg.get("split", {}).get("train_start", None)

    data = load_raw_data(cfg)
    validate_raw_data(data)
    data = preprocess(data, cfg)
    targets = build_targets(data["historico_emprestimos"], data["historico_parcelas"], cfg)
    apps_train = build_training_applications(data["historico_emprestimos"], targets)

    feats_train = make_features(
        apps_train,
        data["base_cadastral"],
        data["historico_emprestimos"],
        data["historico_parcelas"],
        ref_col="data_decisao",
        mode="train",
    )

    if train_start is not None:
        before = len(feats_train)
        cutoff = pd.Timestamp(train_start)
        feats_train = feats_train[feats_train["data_decisao"] >= cutoff].copy()
        logger.info(f"train_start={train_start}: {before:,} -> {len(feats_train):,}")

    splits = temporal_split(feats_train, cfg, ref_col="data_decisao")

    target_col = "target_fpd5"
    secondary_col = "target_ever15mob03"

    X_tr, num_cols, cat_cols, y_tr = split_X_y(splits["train"], target_col=target_col)
    X_val, _, _, y_val = split_X_y(splits["val"], target_col=target_col)
    X_te, _, _, y_te = split_X_y(splits["test"], target_col=target_col)
    y_val_sec = splits["val"][secondary_col].values
    y_te_sec = splits["test"][secondary_col].values

    lgbm = train_lightgbm(
        X_tr, y_tr, X_val, y_val, num_cols, cat_cols,
        lgb_params=cfg["model"]["lightgbm_params"],
        random_state=cfg["project"]["random_state"],
    )

    cal_method = cfg["model"].get("calibration_method", "isotonic")
    calibrated = CalibratedPipeline(lgbm, X_val, y_val, method=cal_method)

    proba_val = calibrated.predict_proba(X_val)[:, 1]
    proba_te = calibrated.predict_proba(X_te)[:, 1]
    proba_tr = calibrated.predict_proba(X_tr)[:, 1]

    out = {
        "name": name,
        "overrides": overrides,
        "elapsed_s": round(time.time() - t0, 1),
        "n_train": int(len(y_tr)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_te)),
        "bad_rate_train": float(y_tr.mean()),
        "bad_rate_val": float(y_val.mean()),
        "bad_rate_test": float(y_te.mean()),
        "metrics_train": basic_metrics(y_tr, proba_tr),
        "metrics_val": basic_metrics(y_val, proba_val),
        "metrics_test": basic_metrics(y_te, proba_te),
        "calibration_method": cal_method,
        "lgbm_params": cfg["model"]["lightgbm_params"],
        "split": cfg.get("split", {}),
    }

    # robustez secundario
    valid_v = y_val_sec >= 0
    valid_t = y_te_sec >= 0
    out["metrics_val_secondary"] = basic_metrics(y_val_sec[valid_v], proba_val[valid_v]) if valid_v.sum() > 0 else {}
    out["metrics_test_secondary"] = basic_metrics(y_te_sec[valid_t], proba_te[valid_t]) if valid_t.sum() > 0 else {}

    # por segmento (val e test)
    seg_val = splits["val"]["tipo_contrato"].values
    seg_te = splits["test"]["tipo_contrato"].values
    out["by_segment_val"] = {}
    out["by_segment_test"] = {}
    for seg in np.unique(seg_val):
        mask = seg_val == seg
        out["by_segment_val"][str(seg)] = basic_metrics(y_val[mask], proba_val[mask])
    for seg in np.unique(seg_te):
        mask = seg_te == seg
        out["by_segment_test"][str(seg)] = basic_metrics(y_te[mask], proba_te[mask])

    # decis no teste (compactado)
    if len(np.unique(y_te)) >= 2:
        deciles = decile_table(y_te, proba_te, n_bins=10)
        # checa colapso: quantos decis tem score_mean unico?
        out["decile_test_unique_scores"] = int(deciles["score_mean"].nunique())
        out["decile_test_score_std"] = float(deciles["score_mean"].std())
        out["decile_test_lift_top10"] = float(
            deciles.iloc[-1]["bad_rate"] / max(y_te.mean(), 1e-9)
        )
        out["decile_test_table"] = deciles.to_dict(orient="records")

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="nome do experimento")
    parser.add_argument(
        "--override", action="append", default=[],
        help="override no formato chave.subchave=valor (pode repetir)"
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--out", default="experiments")
    args = parser.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    result = run_experiment(args.name, args.override, args.config)
    out_path = Path(args.out) / f"{args.name}.json"
    save_json(result, str(out_path))
    logger.info(f"resultado salvo em {out_path}")

    # resumo no stdout
    print("\n" + "=" * 70)
    print(f"EXPERIMENT: {args.name}")
    print("=" * 70)
    for split_name in ["train", "val", "test"]:
        m = result[f"metrics_{split_name}"]
        print(f"  [{split_name}] AUC={m['auc']:.4f} KS={m['ks']:.4f} "
              f"Brier={m['brier']:.4f} bad_rate={m['bad_rate']:.2%}")
    if "decile_test_unique_scores" in result:
        print(f"\n  Decis: {result['decile_test_unique_scores']}/10 scores únicos, "
              f"lift_top10={result['decile_test_lift_top10']:.2f}x")
    print(f"\n  Tempo: {result['elapsed_s']}s")
