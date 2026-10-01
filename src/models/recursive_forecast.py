import numpy as np
import pandas as pd

from src.models.categorical import apply_category_schema


def build_recursive_features_for_date(
    features_base,
    history,
    current_date,
    model_features,
    category_schema,
    lags=(1, 7, 14, 28),
    rolling_windows=(7, 14, 28),
):
    """Construye una matriz diaria con la misma semántica que entrenamiento."""

    current_date = pd.Timestamp(current_date)
    current = features_base.loc[features_base["date"].eq(current_date)].copy()
    if current.empty:
        raise ValueError(f"No hay filas para pronosticar {current_date.date()}")

    for lag in lags:
        lag_date = current_date - pd.Timedelta(days=lag)
        lag_values = history.loc[
            history["date"].eq(lag_date),
            ["store_nbr", "family", "history_sales"],
        ].rename(columns={"history_sales": f"lag_{lag}"})
        current = current.merge(
            lag_values,
            on=["store_nbr", "family"],
            how="left",
            validate="one_to_one",
        )

    for window in rolling_windows:
        start_date = current_date - pd.Timedelta(days=window)
        hist_window = history.loc[
            history["date"].ge(start_date)
            & history["date"].lt(current_date)
        ]
        rolling_values = hist_window.groupby(
            ["store_nbr", "family"],
            as_index=False,
        ).agg(**{f"rolling_mean_{window}": ("history_sales", "mean")})
        current = current.merge(
            rolling_values,
            on=["store_nbr", "family"],
            how="left",
            validate="one_to_one",
        )

    X_day = current[model_features].copy()
    X_day, unknown_categories = apply_category_schema(
        X_day,
        category_schema=category_schema,
        fail_on_unknown=True,
    )

    return current, X_day, unknown_categories


def recursive_forecast(
    model,
    features_base,
    train,
    validation_start,
    validation_end,
    model_features,
    category_schema,
    lags=(1, 7, 14, 28),
    rolling_windows=(7, 14, 28),
):
    """
    Genera un forecast recursivo día por día.

    La historia comienza únicamente con ventas conocidas de train.
    Las predicciones generadas se incorporan como historia para
    los días siguientes.
    """

    history = train[["date", "store_nbr", "family", "sales"]].copy()

    history = history.rename(columns={"sales": "history_sales"})

    predictions = []

    forecast_dates = pd.date_range(
        validation_start,
        validation_end,
        freq="D",
    )

    for current_date in forecast_dates:

        current, X_day, _ = build_recursive_features_for_date(
            features_base=features_base,
            history=history,
            current_date=current_date,
            model_features=model_features,
            category_schema=category_schema,
            lags=lags,
            rolling_windows=rolling_windows,
        )

        predictions_log = model.predict(X_day)

        predictions_day = np.expm1(predictions_log)

        predictions_day = np.clip(
            predictions_day,
            0,
            None,
        )

        result_day = current[["date", "store_nbr", "family"]].copy()

        result_day["prediction"] = predictions_day

        predictions.append(result_day)

        # -----------------------------------------------
        # LA PREDICCIÓN PASA A FORMAR PARTE DE LA HISTORIA
        # -----------------------------------------------
        new_history = result_day.rename(columns={"prediction": "history_sales"})

        history = pd.concat(
            [history, new_history],
            ignore_index=True,
        )

    result = pd.concat(
        predictions,
        ignore_index=True,
    )
    expected_dates = pd.DatetimeIndex(forecast_dates)
    observed_dates = pd.DatetimeIndex(result["date"].drop_duplicates())
    if not observed_dates.equals(expected_dates):
        raise AssertionError(
            "El forecast no cubre exactamente todas las fechas del horizonte."
        )

    return result
