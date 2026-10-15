"""Evaluación comparable de los candidatos V2 sobre un fold preparado."""

import numpy as np
import pandas as pd

from src.features.model_features import CATEGORICAL_FEATURES, MODEL_FEATURES
from src.models.baseline import seasonal_weekday_baseline
from src.models.catboost_model import fit_catboost
from src.models.categorical import fit_category_schema
from src.models.experiment import prediction_metrics
from src.models.lightgbm_model import fit_fixed_iterations
from src.models.recursive_forecast import recursive_forecast
from src.models.tsb import tsb_forecast


KEYS = ["date", "store_nbr", "family"]


def validate_prediction_coverage(truth, prediction):
    """Exige exactamente la misma población y grano antes de puntuar."""

    truth_keys = pd.MultiIndex.from_frame(truth[KEYS])
    prediction_keys = pd.MultiIndex.from_frame(prediction[KEYS])
    if not truth_keys.is_unique:
        raise ValueError("La verdad contiene claves fecha-tienda-familia duplicadas")
    if not prediction_keys.is_unique:
        raise ValueError("La predicción contiene claves fecha-tienda-familia duplicadas")
    missing = truth_keys.difference(prediction_keys)
    extra = prediction_keys.difference(truth_keys)
    if len(missing) or len(extra):
        raise ValueError(
            "La predicción no cubre exactamente la población evaluada: "
            f"faltan={len(missing)}, sobran={len(extra)}"
        )


def _score_frame(truth, prediction, prediction_col, model_name, fold):
    validate_prediction_coverage(truth, prediction)
    scored = truth.merge(
        prediction, on=KEYS, validate="one_to_one"
    )
    if len(scored) != len(truth):
        raise AssertionError("El merge de evaluación alteró la población")
    return {
        "fold": int(fold),
        "model": model_name,
        **prediction_metrics(scored["sales"], scored[prediction_col]),
    }, scored


def evaluate_v2_fold(
    prepared,
    fold,
    lgbm_original_estimators=1029,
    lgbm_optimized_params=None,
    lgbm_optimized_estimators=600,
    catboost_estimators=600,
    tsb_params=(0.2, 0.2),
    return_artifacts=False,
):
    train_features = prepared["train_features"]
    X_train, schema = fit_category_schema(
        train_features[MODEL_FEATURES].copy(), CATEGORICAL_FEATURES
    )
    y_train = train_features["sales"]
    truth = prepared["validation"][["date", "store_nbr", "family", "sales"]].copy()
    start, end = truth["date"].min(), truth["date"].max()
    predictions = {}
    fitted_artifacts = {"category_schema": schema}

    baseline = seasonal_weekday_baseline(prepared["train"], prepared["validation"])[
        ["date", "store_nbr", "family", "pred_baseline"]
    ]
    predictions["Seasonal baseline"] = (baseline, "pred_baseline")

    model_specs = [
        ("LightGBM original", lgbm_original_estimators, None),
        ("LightGBM optimized", lgbm_optimized_estimators, lgbm_optimized_params or {}),
    ]
    for name, estimators, params in model_specs:
        model = fit_fixed_iterations(
            X_train, y_train, CATEGORICAL_FEATURES, estimators, params=params
        )
        pred = recursive_forecast(
            model, prepared["features_base"], prepared["train"], start, end,
            MODEL_FEATURES, schema,
        ).rename(columns={"prediction": "prediction_model"})
        predictions[name] = (pred, "prediction_model")
        if name == "LightGBM optimized":
            fitted_artifacts["lightgbm_optimized"] = model

    catboost = fit_catboost(
        X_train, y_train, CATEGORICAL_FEATURES, n_estimators=catboost_estimators
    )
    cat_pred = recursive_forecast(
        catboost, prepared["features_base"], prepared["train"], start, end,
        MODEL_FEATURES, schema,
    ).rename(columns={"prediction": "prediction_model"})
    predictions["CatBoost"] = (cat_pred, "prediction_model")

    tsb_pred = tsb_forecast(
        prepared["train"], prepared["validation"], alpha=tsb_params[0], beta=tsb_params[1]
    )
    predictions["TSB"] = (tsb_pred, "prediction_tsb")

    rows, scored_frames = [], {}
    for name, (prediction, prediction_col) in predictions.items():
        row, scored = _score_frame(truth, prediction, prediction_col, name, fold)
        rows.append(row)
        scored_frames[name] = scored
    result = (pd.DataFrame(rows), scored_frames)
    if return_artifacts:
        return (*result, fitted_artifacts)
    return result


def diagnostics_by(frame, prediction_col, group_columns):
    rows = []
    for keys, group in frame.groupby(group_columns, observed=True, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_columns, keys))
        row.update(prediction_metrics(group["sales"], group[prediction_col]))
        row["n_obs"] = len(group)
        row["n_series"] = group[["store_nbr", "family"]].drop_duplicates().shape[0]
        rows.append(row)
    return pd.DataFrame(rows)
