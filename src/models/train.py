"""Treino do baseline (logistico) e do modelo principal (LightGBM)."""
from pathlib import Path
import joblib
import lightgbm as lgb
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _build_preprocessor(num_cols: list[str], cat_cols: list[str], scale: bool) -> ColumnTransformer:
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(num_steps)

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="OUTROS")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01, sparse_output=False)),
    ])

    return ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])


def train_logistic(X_train, y_train, num_cols, cat_cols, random_state: int = 42) -> Pipeline:
    pre = _build_preprocessor(num_cols, cat_cols, scale=True)
    pipe = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=random_state)),
    ])
    pipe.fit(X_train, y_train)
    logger.info(f"baseline logistico treinado em {len(X_train):,} amostras")
    return pipe


def train_lightgbm(
    X_train, y_train, X_val, y_val, num_cols, cat_cols, lgb_params: dict, random_state: int = 42,
    disable_early_stopping: bool = False,
) -> Pipeline:
    pre = _build_preprocessor(num_cols, cat_cols, scale=False)
    params = dict(lgb_params)
    n_est = params.pop("n_estimators", 500)
    clf = lgb.LGBMClassifier(n_estimators=n_est, random_state=random_state, **params)

    pipe = Pipeline([("pre", pre), ("clf", clf)])

    Xt = pre.fit_transform(X_train)
    Xv = pre.transform(X_val)
    if disable_early_stopping:
        clf.fit(Xt, y_train)
        logger.info(f"LightGBM treinado em {len(X_train):,} amostras (n_estimators={n_est}, sem early stopping)")
    else:
        clf.fit(
            Xt, y_train,
            eval_set=[(Xv, y_val)],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
        )
        logger.info(f"LightGBM treinado em {len(X_train):,} amostras (best iteration: {clf.best_iteration_})")
    pipe.named_steps["pre"] = pre
    pipe.named_steps["clf"] = clf
    return pipe


def save_model(model, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"modelo salvo em {path}")
