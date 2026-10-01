import numpy as np
import pandas as pd


def recursive_forecast(
    model,
    features_base,
    train,
    validation_start,
    validation_end,
    model_features,
    categorical_features,
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

        # Features conocidas del día.
        # sales NO forma parte de model_features.
        current = features_base[features_base["date"].eq(current_date)].copy()

        # -----------------------------------------------
        # LAGS
        # -----------------------------------------------
        for lag in lags:

            lag_date = current_date - pd.Timedelta(days=lag)

            lag_values = history[history["date"].eq(lag_date)][
                ["store_nbr", "family", "history_sales"]
            ].rename(columns={"history_sales": f"lag_{lag}"})

            current = current.merge(
                lag_values,
                on=["store_nbr", "family"],
                how="left",
                validate="one_to_one",
            )

        # -----------------------------------------------
        # ROLLING
        # -----------------------------------------------
        for window in rolling_windows:

            start_date = current_date - pd.Timedelta(days=window)

            hist_window = history[
                history["date"].ge(start_date) & history["date"].lt(current_date)
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

        # -----------------------------------------------
        # MATRIZ DEL MODELO
        # -----------------------------------------------
        X_day = current[model_features].copy()

        # Usar exactamente las categorías aprendidas
        # durante entrenamiento.
        for col in categorical_features:
            X_day[col] = X_day[col].astype("category")

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

    return pd.concat(
        predictions,
        ignore_index=True,
    )
