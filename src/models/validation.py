import pandas as pd


def fixed_temporal_split(data, validation_start, validation_end, date_col="date"):
    """Corta por fechas explícitas y excluye todo lo posterior a validación."""

    validation_start = pd.Timestamp(validation_start)
    validation_end = pd.Timestamp(validation_end)
    if validation_end < validation_start:
        raise ValueError("validation_end debe ser mayor o igual a validation_start")

    train = data.loc[data[date_col].lt(validation_start)].copy()
    validation = data.loc[
        data[date_col].between(validation_start, validation_end)
    ].copy()
    return train, validation


def rolling_origin_windows(final_test_start, n_windows=3, horizon_days=15):
    """Genera ventanas anteriores, no solapadas y con entrenamiento expansivo."""

    final_test_start = pd.Timestamp(final_test_start)
    rows = []
    for fold in range(n_windows, 0, -1):
        validation_end = (
            final_test_start
            - pd.Timedelta(days=1)
            - pd.Timedelta(days=(fold - 1) * horizon_days)
        )
        validation_start = validation_end - pd.Timedelta(days=horizon_days - 1)
        rows.append(
            {
                "fold": len(rows) + 1,
                "train_end": validation_start - pd.Timedelta(days=1),
                "validation_start": validation_start,
                "validation_end": validation_end,
            }
        )

    return pd.DataFrame(rows)


def temporal_train_valid_split(
    data,
    validation_days=15,
    date_col="date",
):
    """
    Divide un dataset temporal en entrenamiento y validación.

    Los últimos `validation_days` días se reservan para validación.
    No se mezclan ni aleatorizan observaciones.
    """

    data = data.copy()

    validation_end = data[date_col].max()

    validation_start = (
        validation_end
        - pd.Timedelta(days=validation_days - 1)
    )

    train, validation = fixed_temporal_split(
        data,
        validation_start=validation_start,
        validation_end=validation_end,
        date_col=date_col,
    )

    return train, validation, validation_start, validation_end
