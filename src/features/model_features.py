import pandas as pd

from src.features.build_calendar import build_calendar


MODEL_FEATURES = [
    "store_nbr",
    "family",
    "year",
    "month",
    "day",
    "day_of_week",
    "week_of_year",
    "is_weekend",
    "onpromotion",
    "city",
    "state",
    "type",
    "cluster",
    "is_holiday",
    "is_transfer",
    "is_additional",
    "is_bridge",
    "is_work_day",
    "is_new_year_closure",
    "scheduled_event_count",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
]

CATEGORICAL_FEATURES = [
    "store_nbr",
    "family",
    "city",
    "state",
    "type",
    "cluster",
]

FEATURE_AVAILABILITY = {
    "store_nbr": "Identidad conocida antes del horizonte.",
    "family": "Identidad conocida antes del horizonte.",
    "year": "Calendario determinístico.",
    "month": "Calendario determinístico.",
    "day": "Calendario determinístico.",
    "day_of_week": "Calendario determinístico.",
    "week_of_year": "Calendario determinístico.",
    "is_weekend": "Calendario determinístico.",
    "onpromotion": (
        "Disponible sólo bajo el supuesto Favorita: el plan promocional "
        "futuro se conoce antes de emitir el forecast."
    ),
    "city": "Metadata estática conocida.",
    "state": "Metadata estática conocida.",
    "type": "Metadata estática conocida.",
    "cluster": "Metadata estática conocida.",
    "is_holiday": "Feriado programado y no trasladado.",
    "is_transfer": "Traslado programado del calendario oficial.",
    "is_additional": "Día adicional programado.",
    "is_bridge": "Puente programado.",
    "is_work_day": "Día laboral compensatorio programado.",
    "is_new_year_closure": "Regla de calendario determinística.",
    "scheduled_event_count": (
        "Cuenta sólo Holiday, Transfer, Additional, Bridge y Work Day; "
        "excluye Event por posible conocimiento ex post."
    ),
    "lag_1": "Sólo target histórico disponible antes del día pronosticado.",
    "lag_7": "Sólo target histórico disponible antes del día pronosticado.",
    "lag_14": "Sólo target histórico disponible antes del día pronosticado.",
    "lag_28": "Sólo target histórico disponible antes del día pronosticado.",
    "rolling_mean_7": "Ventana histórica [t-7, t), sin el día actual.",
    "rolling_mean_14": "Ventana histórica [t-14, t), sin el día actual.",
    "rolling_mean_28": "Ventana histórica [t-28, t), sin el día actual.",
}


def feature_availability_table():
    """Devuelve el contrato auditable de disponibilidad del modelo."""

    return pd.DataFrame(
        [
            {"feature": feature, "availability": FEATURE_AVAILABILITY[feature]}
            for feature in MODEL_FEATURES
        ]
    )


def add_known_features(
    sales_modelable,
    sales,
    stores,
    holidays,
):
    """
    Agrega variables conocidas o determinables antes de realizar
    el pronóstico.

    No utiliza valores futuros de sales. ``onpromotion`` se considera conocida
    sólo bajo el supuesto de competencia Favorita. Los eventos ``type=Event``
    se conservan para diagnóstico pero no forman parte de ``MODEL_FEATURES``.
    """

    data = sales_modelable.copy()

    # ---------------------------------------------------------
    # 1. Features de calendario
    # ---------------------------------------------------------
    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    data["day"] = data["date"].dt.day
    data["day_of_week"] = data["date"].dt.dayofweek
    data["week_of_year"] = data["date"].dt.isocalendar().week.astype(int)

    data["is_weekend"] = (data["day_of_week"].isin([5, 6])).astype("int8")

    # ---------------------------------------------------------
    # 2. Metadata estática de tienda
    # ---------------------------------------------------------
    store_features = stores[
        [
            "store_nbr",
            "city",
            "state",
            "type",
            "cluster",
        ]
    ].copy()

    data = data.merge(
        store_features,
        on="store_nbr",
        how="left",
        validate="many_to_one",
    )

    # ---------------------------------------------------------
    # 3. Feriados / eventos
    # ---------------------------------------------------------
    calendar = build_calendar(
        sales=sales,
        stores=stores,
        holidays=holidays,
    )

    calendar_features = calendar[
        [
            "date",
            "store_nbr",
            "is_holiday",
            "is_event",
            "is_transfer",
            "is_additional",
            "is_bridge",
            "is_work_day",
            "is_new_year_closure",
            "event_count",
            "scheduled_event_count",
            "unplanned_event_count",
        ]
    ].copy()

    data = data.merge(
        calendar_features,
        on=["date", "store_nbr"],
        how="left",
        validate="many_to_one",
    )

    return data


def add_lag_features(
    data,
    lags=(1, 7, 14, 28),
):
    """
    Agrega lags de ventas por tienda-familia usando días calendario.

    Cada lag representa exactamente N días anteriores, no N filas anteriores.
    """

    result = data.copy()

    history = result[["date", "store_nbr", "family", "sales"]].copy()

    for lag in lags:

        lag_data = history.copy()

        lag_data["date"] = lag_data["date"] + pd.Timedelta(days=lag)

        lag_data = lag_data.rename(columns={"sales": f"lag_{lag}"})

        result = result.merge(
            lag_data,
            on=["date", "store_nbr", "family"],
            how="left",
            validate="one_to_one",
        )

    return result


def add_rolling_features(
    data,
    windows=(7, 14, 28),
):
    """
    Agrega medias móviles históricas por tienda-familia.

    Cada ventana utiliza únicamente fechas anteriores al día objetivo.
    La fecha actual queda excluida con closed="left".
    """

    result = data.copy()

    history = (
        result[
            ["date", "store_nbr", "family", "sales"]
        ]
        .sort_values(
            ["store_nbr", "family", "date"]
        )
        .set_index("date")
    )

    for window in windows:

        rolling_mean = (
            history
            .groupby(
                ["store_nbr", "family"]
            )["sales"]
            .rolling(
                f"{window}D",
                closed="left",
                min_periods=1,
            )
            .mean()
            .rename(
                f"rolling_mean_{window}"
            )
            .reset_index()
        )

        result = result.merge(
            rolling_mean,
            on=[
                "store_nbr",
                "family",
                "date",
            ],
            how="left",
            validate="one_to_one",
        )

    return result
