import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelingEligibility:
    """Universo observable construido sin mirar después de ``cutoff``."""

    cutoff: pd.Timestamp
    store_openings: pd.DataFrame
    known_series: pd.DataFrame


def derive_modeling_eligibility(
    sales,
    cutoff,
    date_col="date",
    target_col="sales",
):
    """Deriva aperturas y series conocidas usando sólo historia hasta cutoff."""

    cutoff = pd.Timestamp(cutoff)
    history = sales.loc[sales[date_col].le(cutoff)].copy()
    positive = history.loc[history[target_col].gt(0)]

    store_openings = (
        positive.groupby("store_nbr", as_index=False)[date_col]
        .min()
        .rename(columns={date_col: "opening_date"})
    )
    known_series = (
        positive[["store_nbr", "family"]]
        .drop_duplicates()
        .assign(series_known_at_cutoff=True)
    )

    return ModelingEligibility(
        cutoff=cutoff,
        store_openings=store_openings,
        known_series=known_series,
    )


def apply_modeling_eligibility(sales, eligibility, date_col="date"):
    """Anota un panel sin usar targets posteriores al cutoff de elegibilidad."""

    result = sales.copy()
    result = result.merge(
        eligibility.store_openings,
        on="store_nbr",
        how="left",
        validate="many_to_one",
    )
    result = result.merge(
        eligibility.known_series,
        on=["store_nbr", "family"],
        how="left",
        validate="many_to_one",
    )

    result["store_known_at_cutoff"] = result["opening_date"].notna()
    result["series_known_at_cutoff"] = (
        result["series_known_at_cutoff"].fillna(False).astype(bool)
    )
    result["tienda_operativa"] = (
        result["store_known_at_cutoff"]
        & result[date_col].ge(result["opening_date"])
    )
    result["modelable_from_cutoff"] = (
        result["tienda_operativa"]
        & result["series_known_at_cutoff"]
    )

    return result


def build_modeling_panel(sales, cutoff, date_col="date", target_col="sales"):
    """Construye el panel modelable con elegibilidad disponible al cutoff."""

    eligibility = derive_modeling_eligibility(
        sales,
        cutoff=cutoff,
        date_col=date_col,
        target_col=target_col,
    )
    annotated = apply_modeling_eligibility(
        sales,
        eligibility=eligibility,
        date_col=date_col,
    )
    panel = annotated.loc[annotated["modelable_from_cutoff"]].copy()
    return panel, annotated, eligibility


def split_future_known_and_cold_start(
    annotated_sales,
    start,
    end,
    date_col="date",
    target_col="sales",
):
    """Separa series conocidas y cold-start sin ocultar demanda futura nueva.

    La demanda futura se consulta sólo para reportar qué series desconocidas
    efectivamente aparecen; nunca modifica el universo conocido ni el modelo.
    """

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    future = annotated_sales.loc[
        annotated_sales[date_col].between(start, end)
    ].copy()

    known = future.loc[future["series_known_at_cutoff"]].copy()
    cold_keys = (
        future.loc[
            ~future["series_known_at_cutoff"] & future[target_col].gt(0),
            ["store_nbr", "family"],
        ]
        .drop_duplicates()
        .assign(is_cold_start=True)
    )
    cold = future.merge(
        cold_keys,
        on=["store_nbr", "family"],
        how="inner",
        validate="many_to_one",
    )

    return known, cold, cold_keys.drop(columns="is_cold_start")


def add_store_operation_flags(sales):
    """
    Identifica la fecha de apertura observada de cada tienda.

    Esta función conserva el diagnóstico histórico del EDA y utiliza todo el
    DataFrame recibido. Para modelado debe usarse ``build_modeling_panel`` con
    un cutoff explícito.
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
