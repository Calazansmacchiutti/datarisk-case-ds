"""Orquestrador end-to-end do pipeline de risco de crédito."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load_data import load_raw_data
from src.data.preprocess import preprocess
from src.data.validate_data import validate_raw_data
from src.features.build_features import (
    build_scoring_applications,
    build_training_applications,
    make_features,
    split_X_y,
)
from src.models.calibrate import CalibratedPipeline
from src.models.evaluate import (
    evaluate_by_segment,
    evaluate_secondary_target,
    evaluate_split,
    save_decile_report,
    save_metrics,
)
from src.models.predict import generate_submission
from src.models.split import temporal_split
from src.models.train import save_model, train_lightgbm, train_logistic
from src.policy.credit_policy import build_policy_output
from src.target.build_target import build_targets
from src.utils.io import ensure_dir, load_config
from src.utils.logger import get_logger

logger = get_logger("pipeline")


def main(config_path: str) -> None:
    config = load_config(config_path)
    paths = config["paths"]
    files = config["files"]
    ensure_dir(paths["output_dir"])
    ensure_dir(paths["report_dir"])
    ensure_dir(paths["model_dir"])

    # 1. carregar e validar
    logger.info("=" * 70 + "\n[1] Carregando bases\n" + "=" * 70)
    data = load_raw_data(config)
    validate_raw_data(data)

    # 2. preprocess
    logger.info("=" * 70 + "\n[2] Pre-processamento\n" + "=" * 70)
    data = preprocess(data, config)

    # 3. targets
    logger.info("=" * 70 + "\n[3] Construindo targets\n" + "=" * 70)
    targets = build_targets(data["historico_emprestimos"], data["historico_parcelas"], config)

    # 4. base de treino com features
    logger.info("=" * 70 + "\n[4] Features de treino\n" + "=" * 70)
    apps_train = build_training_applications(data["historico_emprestimos"], targets)
    feats_train = make_features(
        apps_train,
        data["base_cadastral"],
        data["historico_emprestimos"],
        data["historico_parcelas"],
        ref_col="data_decisao",
        mode="train",
    )

    # 5. split temporal
    logger.info("=" * 70 + "\n[5] Split temporal\n" + "=" * 70)
    splits = temporal_split(feats_train, config, ref_col="data_decisao")

    # 6. preparar X/y
    target_col = "target_fpd5"
    secondary_col = "target_ever15mob03"

    X_tr, num_cols, cat_cols, y_tr = split_X_y(splits["train"], target_col=target_col)
    X_val, _, _, y_val = split_X_y(splits["val"], target_col=target_col)
    X_te, _, _, y_te = split_X_y(splits["test"], target_col=target_col)
    y_val_sec = splits["val"][secondary_col].values
    y_te_sec = splits["test"][secondary_col].values

    logger.info(f"colunas numericas: {len(num_cols)}, categoricas: {len(cat_cols)}")
    logger.info(f"y_tr bad rate: {y_tr.mean():.2%}")

    # 7. baseline + lightgbm
    logger.info("=" * 70 + "\n[6] Treinando modelos\n" + "=" * 70)
    baseline = train_logistic(X_tr, y_tr, num_cols, cat_cols, random_state=config["project"]["random_state"])
    lgbm = train_lightgbm(
        X_tr, y_tr, X_val, y_val, num_cols, cat_cols,
        lgb_params=config["model"]["lightgbm_params"],
        random_state=config["project"]["random_state"],
    )

    # 8. avaliacao
    logger.info("=" * 70 + "\n[7] Avaliando modelos\n" + "=" * 70)
    metrics = {"baseline": {}, "lightgbm": {}, "lightgbm_calibrated": {}}

    for name, m in [("baseline", baseline), ("lightgbm", lgbm)]:
        metrics[name]["train"] = evaluate_split(m, X_tr, y_tr, f"{name}|train")
        metrics[name]["val"] = evaluate_split(m, X_val, y_val, f"{name}|val")
        metrics[name]["test"] = evaluate_split(m, X_te, y_te, f"{name}|test")

    logger.info("avaliacao por tipo de contrato (LightGBM, teste)")
    metrics["lightgbm"]["test_by_segment"] = evaluate_by_segment(
        lgbm, X_te, y_te, splits["test"]["tipo_contrato"], name="lightgbm|test"
    )

    metrics["lightgbm"]["robustez_secundario_val"] = evaluate_secondary_target(
        lgbm, X_val, y_val_sec, name="lightgbm|val"
    )
    metrics["lightgbm"]["robustez_secundario_test"] = evaluate_secondary_target(
        lgbm, X_te, y_te_sec, name="lightgbm|test"
    )

    # 9. calibracao
    logger.info("=" * 70 + "\n[8] Calibrando modelo final\n" + "=" * 70)
    calibrated = CalibratedPipeline(lgbm, X_val, y_val, method=config["model"]["calibration_method"])
    metrics["lightgbm_calibrated"]["val"] = evaluate_split(calibrated, X_val, y_val, "calibrated|val")
    metrics["lightgbm_calibrated"]["test"] = evaluate_split(calibrated, X_te, y_te, "calibrated|test")
    metrics["lightgbm_calibrated"]["test_by_segment"] = evaluate_by_segment(
        calibrated, X_te, y_te, splits["test"]["tipo_contrato"], name="calibrated|test"
    )

    save_decile_report(calibrated, X_te, y_te, str(Path(paths["report_dir"]) / "decis_test_fpd5.csv"))
    save_metrics(metrics, str(Path(paths["report_dir"]) / "metricas_modelo.json"))

    # 10. submissao
    logger.info("=" * 70 + "\n[9] Gerando submissao\n" + "=" * 70)
    apps_score = build_scoring_applications(data["base_submissao"])
    feats_score = make_features(
        apps_score,
        data["base_cadastral"],
        data["historico_emprestimos"],
        data["historico_parcelas"],
        ref_col="data_solicitacao",
        mode="score",
    )
    X_score, _, _, _ = split_X_y(feats_score)
    submission_path = str(Path(paths["output_dir"]) / files["submission"])
    submission = generate_submission(
        calibrated, X_score, feats_score["id_cliente"], submission_path
    )

    # 11. politica (thresholds derivados da distribuicao de PD na validacao)
    logger.info("=" * 70 + "\n[10] Politica de credito\n" + "=" * 70)
    pd_val = calibrated.predict_proba(X_val)[:, 1]
    policy_path = str(Path(paths["output_dir"]) / "decisoes_credito_simuladas.csv")
    _, thresholds = build_policy_output(
        submission, data["base_submissao"], feats_score, pd_val, config, policy_path
    )
    save_metrics({**metrics, "policy_thresholds": thresholds},
                 str(Path(paths["report_dir"]) / "metricas_modelo.json"))

    # 12. salvar modelo
    save_model(lgbm, str(Path(paths["model_dir"]) / "lightgbm.joblib"))

    logger.info("=" * 70 + "\nPipeline concluido com sucesso\n" + "=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    main(args.config)
