import numpy as np
import pandas as pd

from src.features.model_features import CATEGORICAL_FEATURES, MODEL_FEATURES
from src.models.baseline import (
    aggregate_percentage_bias,
    mae,
    mean_signed_error,
    over_under_magnitudes,
    rmsle,
    seasonal_weekday_baseline,
    wape,
)
from src.models.categorical import apply_category_schema, fit_category_schema
from src.models.lightgbm_model import fit_fixed_iterations, fit_with_early_stopping
from src.models.modeling_pipeline import prepare_cutoff_dataset
from src.models.recursive_forecast import recursive_forecast


def prediction_metrics(y_true, y_pred):
    metrics = {
        "RMSLE": rmsle(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "MSE_signed": mean_signed_error(y_true, y_pred),
        "WAPE_pct": wape(y_true, y_pred),
        "Bias_pct": aggregate_percentage_bias(y_true, y_pred),
    }
    metrics.update(over_under_magnitudes(y_true, y_pred))
    return metrics


def segment_metrics(frame, prediction_col, group_cols=()):
    """Métricas comparables para el total o segmentos de un forecast."""

    group_cols = list(group_cols)
    grouped = (
        frame.groupby(group_cols, observed=True, dropna=False)
        if group_cols
        else [((), frame)]
    )
    rows = []
    for keys, group in grouped:
        if group_cols and not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys if group_cols else ()))
        errors = group[prediction_col].to_numpy() - group["sales"].to_numpy()
        row.update(prediction_metrics(group["sales"], group[prediction_col]))
        row.update(
            {
                "pct_zero_sales": float(group["sales"].eq(0).mean() * 100),
                "pct_overpredicted": float((errors > 0).mean() * 100),
                "pct_underpredicted": float((errors < 0).mean() * 100),
                "n_obs": int(len(group)),
                "n_series": int(
                    group[["store_nbr", "family"]].drop_duplicates().shape[0]
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _training_matrices(prepared):
    train_features = prepared["train_features"]
    X_train = train_features[MODEL_FEATURES].copy()
    y_train = train_features["sales"].copy()
    X_train, category_schema = fit_category_schema(
        X_train,
        CATEGORICAL_FEATURES,
    )

    selection = prepared["selection_features"]
    X_selection = selection[MODEL_FEATURES].copy()
    X_selection, unknown = apply_category_schema(
        X_selection,
        category_schema,
    )
    if unknown:
        raise AssertionError(f"Categorías desconocidas en selección: {unknown}")

    return (
        X_train,
        y_train,
        X_selection,
        selection["sales"].copy(),
        category_schema,
    )


def _evaluate_models(prepared, model, category_schema, start, end):
    recursive_predictions = recursive_forecast(
        model=model,
        features_base=prepared["features_base"],
        train=prepared["train"],
        validation_start=start,
        validation_end=end,
        model_features=MODEL_FEATURES,
        category_schema=category_schema,
    )
    truth = prepared["validation"][
        ["date", "store_nbr", "family", "sales"]
    ].copy()
    lgbm_eval = truth.merge(
        recursive_predictions,
        on=["date", "store_nbr", "family"],
        how="left",
        validate="one_to_one",
    )
    if lgbm_eval["prediction"].isna().any():
        raise AssertionError("Faltan predicciones LightGBM en el horizonte")

    baseline_eval = seasonal_weekday_baseline(
        train=prepared["train"],
        validation=prepared["validation"],
        lookback_days=28,
    )

    return lgbm_eval, baseline_eval


def run_backtest_fold(
    datasets,
    train_end,
    validation_start,
    validation_end,
    fold,
    max_estimators=1000,
    stopping_rounds=50,
):
    """Entrena, selecciona iteraciones y evalúa un fold anterior al test."""

    prepared = prepare_cutoff_dataset(
        sales=datasets["sales"],
        stores=datasets["stores"],
        holidays=datasets["holidays"],
        train_end=train_end,
        forecast_start=validation_start,
        forecast_end=validation_end,
    )
    X_train, y_train, X_selection, y_selection, schema = _training_matrices(
        prepared
    )
    model, best_iteration = fit_with_early_stopping(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_selection,
        y_valid=y_selection,
        categorical_features=CATEGORICAL_FEATURES,
        max_estimators=max_estimators,
        stopping_rounds=stopping_rounds,
    )
    lgbm_eval, baseline_eval = _evaluate_models(
        prepared,
        model=model,
        category_schema=schema,
        start=validation_start,
        end=validation_end,
    )

    rows = []
    for model_name, frame, prediction_col in [
        ("Seasonal baseline", baseline_eval, "pred_baseline"),
        ("LightGBM recursive", lgbm_eval, "prediction"),
    ]:
        metrics = prediction_metrics(frame["sales"], frame[prediction_col])
        rows.append(
            {
                "fold": int(fold),
                "train_end": pd.Timestamp(train_end),
                "validation_start": pd.Timestamp(validation_start),
                "validation_end": pd.Timestamp(validation_end),
                "model": model_name,
                "best_iteration": (
                    best_iteration if model_name == "LightGBM recursive" else np.nan
                ),
                "known_series": prepared["validation"][
                    ["store_nbr", "family"]
                ].drop_duplicates().shape[0],
                "cold_start_series": len(prepared["cold_start_keys"]),
                **metrics,
            }
        )

    return {
        "summary": pd.DataFrame(rows),
        "model": model,
        "category_schema": schema,
        "prepared": prepared,
        "lgbm_eval": lgbm_eval,
        "baseline_eval": baseline_eval,
    }


def run_final_test(
    datasets,
    train_end,
    test_start,
    test_end,
    n_estimators,
):
    """Reentrena hasta train_end y toca el test final una única vez."""

    prepared = prepare_cutoff_dataset(
        sales=datasets["sales"],
        stores=datasets["stores"],
        holidays=datasets["holidays"],
        train_end=train_end,
        forecast_start=test_start,
        forecast_end=test_end,
    )
    X_train = prepared["train_features"][MODEL_FEATURES].copy()
    y_train = prepared["train_features"]["sales"].copy()
    X_train, schema = fit_category_schema(X_train, CATEGORICAL_FEATURES)
    model = fit_fixed_iterations(
        X_train=X_train,
        y_train=y_train,
        categorical_features=CATEGORICAL_FEATURES,
        n_estimators=n_estimators,
    )
    lgbm_eval, baseline_eval = _evaluate_models(
        prepared,
        model=model,
        category_schema=schema,
        start=test_start,
        end=test_end,
    )

    rows = []
    for model_name, frame, prediction_col in [
        ("Seasonal baseline", baseline_eval, "pred_baseline"),
        ("LightGBM recursive", lgbm_eval, "prediction"),
    ]:
        rows.append(
            {
                "model": model_name,
                **prediction_metrics(frame["sales"], frame[prediction_col]),
            }
        )

    return {
        "summary": pd.DataFrame(rows),
        "model": model,
        "category_schema": schema,
        "prepared": prepared,
        "lgbm_eval": lgbm_eval,
        "baseline_eval": baseline_eval,
    }
