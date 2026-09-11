import numpy as np
import pandas as pd


def add_store_operation_flags(sales):
    """
    Identifica la fecha de apertura observada de cada tienda
    y clasifica los registros según si la tienda ya estaba operativa.
    """

    sales = sales.copy()

    # Primera fecha con ventas positivas por tienda
    apertura = (
        sales.loc[sales["sales"] > 0]
        .groupby("store_nbr")["date"]
        .min()
    )

    sales["tienda_operativa"] = (
        sales["date"] >= sales["store_nbr"].map(apertura)
    )

    return sales, apertura


def add_zero_taxonomy(sales):
    """
    Clasifica los ceros en:
    - Pre-apertura
    - Familia no surtida
    - Cero real de demanda
    - Venta positiva

    Nota:
    'familia_surtida' se utiliza como diagnóstico EDA y no
    debe utilizarse directamente como predictor.
    """

    sales = sales.copy()

    # ¿Esta tienda-familia registra al menos una venta positiva
    # durante el período observado?
    sales["familia_surtida"] = (
        sales.groupby(["store_nbr", "family"])["sales"]
        .transform(lambda x: (x > 0).any())
    )

    conditions = [
        (sales["sales"] == 0) &
        (~sales["tienda_operativa"]),

        (sales["sales"] == 0) &
        (sales["tienda_operativa"]) &
        (~sales["familia_surtida"]),

        (sales["sales"] == 0) &
        (sales["tienda_operativa"]) &
        (sales["familia_surtida"])
    ]

    choices = [
        "Pre-apertura",
        "Familia no surtida",
        "Cero real de demanda"
    ]

    sales["tipo_cero"] = np.select(
        conditions,
        choices,
        default="Venta positiva"
    )

    return sales


def find_missing_dates(sales):
    """
    Detecta fechas completamente ausentes del panel.
    """

    full_calendar = pd.date_range(
        start=sales["date"].min(),
        end=sales["date"].max(),
        freq="D"
    )

    missing_dates = full_calendar.difference(
        pd.DatetimeIndex(sales["date"].unique())
    )

    return missing_dates