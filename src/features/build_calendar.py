import pandas as pd


def build_calendar(sales, stores, holidays):
    """
    Construye un calendario diario por tienda.

    Considera:
    - feriados nacionales;
    - feriados regionales según state;
    - feriados locales según city;
    - Holiday, Event, Transfer, Additional, Bridge y Work Day;
    - múltiples eventos en una misma fecha sin perder información.

    Devuelve una fila por fecha y tienda.
    """

    # Calendario diario completo
    dates = pd.DataFrame({
        "date": pd.date_range(
            sales["date"].min(),
            sales["date"].max(),
            freq="D"
        )
    })

    # Producto cartesiano fecha × tienda
    calendar = dates.merge(
        stores,
        how="cross"
    )

    # Un Holiday trasladado NO se celebra en su fecha original.
    # El día efectivo aparece como type="Transfer".
    active_holidays = holidays[
        ~(
            (holidays["type"] == "Holiday") &
            (holidays["transferred"] == True)
        )
    ].copy()

    # Solo período cubierto por ventas
    active_holidays = active_holidays[
        active_holidays["date"].between(
            sales["date"].min(),
            sales["date"].max()
        )
    ]

    records = []

    # Aplicar cada evento solamente a las tiendas correspondientes
    for _, event in active_holidays.iterrows():

        if event["locale"] == "National":
            affected_stores = stores

        elif event["locale"] == "Regional":
            affected_stores = stores[
                stores["state"] == event["locale_name"]
            ]

        elif event["locale"] == "Local":
            affected_stores = stores[
                stores["city"] == event["locale_name"]
            ]

        else:
            continue

        for store_nbr in affected_stores["store_nbr"]:

            records.append({
                "date": event["date"],
                "store_nbr": store_nbr,
                "event_type": event["type"],
                "description": event["description"]
            })

    events = pd.DataFrame(records)

    # Si no hubiera eventos evitamos error
    if events.empty:
        return calendar

    # Número de eventos que afectan a cada tienda-fecha
    event_count = (
        events
        .groupby(
            ["date", "store_nbr"],
            as_index=False
        )
        .size()
        .rename(columns={"size": "event_count"})
    )

    # Conservar todas las descripciones de un mismo día
    descriptions = (
        events
        .groupby(
            ["date", "store_nbr"]
        )["description"]
        .agg(
            lambda x: " | ".join(
                sorted(set(x))
            )
        )
        .reset_index()
        .rename(
            columns={"description": "event_descriptions"}
        )
    )

    # Una columna binaria por tipo
    event_flags = pd.crosstab(
        [events["date"], events["store_nbr"]],
        events["event_type"]
    )

    event_flags = (
        event_flags
        .clip(upper=1)
        .reset_index()
    )

    rename_types = {
        "Holiday": "is_holiday",
        "Event": "is_event",
        "Transfer": "is_transfer",
        "Additional": "is_additional",
        "Bridge": "is_bridge",
        "Work Day": "is_work_day"
    }

    event_flags = event_flags.rename(
        columns=rename_types
    )

    calendar = calendar.merge(
        event_flags,
        on=["date", "store_nbr"],
        how="left"
    )

    calendar = calendar.merge(
        event_count,
        on=["date", "store_nbr"],
        how="left"
    )

    calendar = calendar.merge(
        descriptions,
        on=["date", "store_nbr"],
        how="left"
    )

    flag_columns = [
        "is_holiday",
        "is_event",
        "is_transfer",
        "is_additional",
        "is_bridge",
        "is_work_day"
    ]

    # Garantizar que existan todas las columnas
    for col in flag_columns:
        if col not in calendar.columns:
            calendar[col] = 0

    calendar[flag_columns] = (
        calendar[flag_columns]
        .fillna(0)
        .astype("int8")
    )

    calendar["event_count"] = (
        calendar["event_count"]
        .fillna(0)
        .astype(int)
    )

    # Cierre estructural detectado en el EDA
    calendar["is_new_year_closure"] = (
        (calendar["date"].dt.month == 1) &
        (calendar["date"].dt.day == 1)
    ).astype("int8")

    return calendar