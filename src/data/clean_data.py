import pandas as pd

def minimal_cleaning(datasets):

    cleaned = {
        name: df.copy()
        for name, df in datasets.items()
    }

    for name in ["sales", "transactions", "holidays", "oil"]:
        cleaned[name] = (
            cleaned[name]
            .sort_values("date")
            .reset_index(drop=True)
        )

    # Calendario completo del petróleo
    sales_min_date = cleaned["sales"]["date"].min()
    sales_max_date = cleaned["sales"]["date"].max()

    full_calendar = pd.date_range(
        sales_min_date,
        sales_max_date,
        freq="D"
    )

    cleaned["oil"] = (
        cleaned["oil"]
        .set_index("date")
        .reindex(full_calendar)
        .rename_axis("date")
        .reset_index()
    )

    # Último precio conocido: sin mirar hacia el futuro
    cleaned["oil"]["dcoilwtico"] = (
        cleaned["oil"]["dcoilwtico"]
        .ffill()
    )

    return cleaned