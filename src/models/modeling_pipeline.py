import pandas as pd

from src.data.panel_diagnostics import (
    build_modeling_panel,
    split_future_known_and_cold_start,
)
from src.features.model_features import (
    add_known_features,
    add_lag_features,
    add_rolling_features,
)


def prepare_cutoff_dataset(
    sales,
    stores,
    holidays,
    train_end,
    forecast_start,
    forecast_end,
    lags=(1, 7, 14, 28),
    rolling_windows=(7, 14, 28),
):
    """Prepara train y horizonte futuro con un cutoff de elegibilidad explícito."""

    train_end = pd.Timestamp(train_end)
    forecast_start = pd.Timestamp(forecast_start)
    forecast_end = pd.Timestamp(forecast_end)
    if forecast_start != train_end + pd.Timedelta(days=1):
        raise ValueError("El horizonte debe comenzar el día posterior a train_end")

    panel, annotated, eligibility = build_modeling_panel(
        sales,
        cutoff=train_end,
    )
    train = panel.loc[panel["date"].le(train_end)].copy()
    validation, cold_start, cold_keys = split_future_known_and_cold_start(
        annotated,
        start=forecast_start,
        end=forecast_end,
    )

    modeling_rows = pd.concat([train, validation], ignore_index=True)
    features_base = add_known_features(
        sales_modelable=modeling_rows,
        sales=sales,
        stores=stores,
        holidays=holidays,
    )

    train_base = features_base.loc[features_base["date"].le(train_end)].copy()
    train_features = add_rolling_features(
        add_lag_features(train_base, lags=lags),
        windows=rolling_windows,
    )

    # El primer día del fold puede usarse para early stopping: todos sus lags
    # y rolling terminan en train_end y, por construcción, no usan su target.
    selection_base = features_base.loc[
        features_base["date"].le(forecast_start)
    ].copy()
    selection_features = add_rolling_features(
        add_lag_features(selection_base, lags=lags),
        windows=rolling_windows,
    )
    selection_features = selection_features.loc[
        selection_features["date"].eq(forecast_start)
    ].copy()

    return {
        "eligibility": eligibility,
        "train": train,
        "validation": validation,
        "cold_start": cold_start,
        "cold_start_keys": cold_keys,
        "features_base": features_base,
        "train_features": train_features,
        "selection_features": selection_features,
    }
