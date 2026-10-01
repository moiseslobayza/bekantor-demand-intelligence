import pandas as pd

from src.features.build_calendar import build_calendar


def add_known_features(
    sales_modelable,
    sales,
    stores,
    holidays,
):
    """
    Agrega variables conocidas o determinables antes de realizar
    el pronóstico.

    No utiliza valores futuros de sales.
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
