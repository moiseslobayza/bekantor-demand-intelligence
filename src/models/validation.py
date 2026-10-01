import pandas as pd


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

    train = data[
        data[date_col] < validation_start
    ].copy()

    validation = data[
        data[date_col] >= validation_start
    ].copy()

    return train, validation, validation_start, validation_end