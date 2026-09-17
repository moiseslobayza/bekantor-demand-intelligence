import numpy as np
import pandas as pd


def temporal_train_valid_split(data, validation_days=15, date_col="date"):
    """Reserva los últimos `validation_days` como validación temporal."""
    validation_end = data[date_col].max()
    validation_start = validation_end - pd.Timedelta(days=validation_days - 1)

    train = data[data[date_col] < validation_start].copy()
    validation = data[data[date_col] >= validation_start].copy()

    return train, validation, validation_start, validation_end


def seasonal_weekday_baseline(
    train,
    validation,
    lookback_days=28,
    date_col="date",
    target_col="sales",
):
    """Baseline estacional tienda-familia-día de semana.

    Se conserva como módulo para la futura etapa de modelado, pero no debe
    ejecutarse ni evaluarse dentro del EDA final.
    """
    validation_start = validation[date_col].min()
    baseline_start = validation_start - pd.Timedelta(days=lookback_days)

    source = train[train[date_col] >= baseline_start].copy()
    source["day_of_week"] = source[date_col].dt.dayofweek

    seasonal = (
        source.groupby(["store_nbr", "family", "day_of_week"], as_index=False)
        .agg(pred_baseline=(target_col, "mean"))
    )

    result = validation.copy()
    result["day_of_week"] = result[date_col].dt.dayofweek
    result = result.merge(
        seasonal,
        on=["store_nbr", "family", "day_of_week"],
        how="left",
    )

    fallback = (
        train.groupby(["store_nbr", "family"], as_index=False)
        .agg(fallback_mean=(target_col, "mean"))
    )
    result = result.merge(fallback, on=["store_nbr", "family"], how="left")
    result["pred_baseline"] = (
        result["pred_baseline"]
        .fillna(result["fallback_mean"])
        .fillna(0)
        .clip(lower=0)
    )

    return result


def rmsle(y_true, y_pred):
    """Root Mean Squared Logarithmic Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if (y_true < 0).any() or (y_pred < 0).any():
        raise ValueError("RMSLE requiere valores no negativos.")

    return float(
        np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2))
    )
