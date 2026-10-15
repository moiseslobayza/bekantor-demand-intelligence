"""Escenarios reproducibles de disponibilidad promocional futura."""

import pandas as pd


SCENARIOS = {"full", "partial", "none"}


def fit_historical_promotion_rates(history, cutoff):
    """Ajusta medias sólo con registros observados hasta el cutoff."""

    cutoff = pd.Timestamp(cutoff)
    past = history.loc[history["date"].le(cutoff)].copy()
    if past.empty:
        raise ValueError("No hay historia promocional anterior al cutoff")
    past["day_of_week"] = past["date"].dt.dayofweek
    detailed = past.groupby(
        ["store_nbr", "family", "day_of_week"], observed=True
    )["onpromotion"].mean()
    family = past.groupby(["family", "day_of_week"], observed=True)["onpromotion"].mean()
    weekday = past.groupby("day_of_week")["onpromotion"].mean()
    return {
        "cutoff": cutoff,
        "detailed": detailed,
        "family": family,
        "weekday": weekday,
        "global": float(past["onpromotion"].mean()),
        "max_history_date": past["date"].max(),
    }


def estimated_promotions(keys, rates):
    values = []
    for row in keys.itertuples(index=False):
        dow = pd.Timestamp(row.date).dayofweek
        value = rates["detailed"].get((row.store_nbr, row.family, dow))
        if pd.isna(value):
            value = rates["family"].get((row.family, dow))
        if pd.isna(value):
            value = rates["weekday"].get(dow, rates["global"])
        values.append(max(0.0, float(value)))
    return values


def make_promotion_transform(scenario, historical_sales, cutoff, confirmed_days=7):
    """Crea un transform para forecast recursivo sin mirar ventas futuras.

    En ``partial`` se asume explícitamente que sólo los primeros
    ``confirmed_days`` del horizonte tienen calendario confirmado. El resto se
    estima con tasas históricas. En ``none`` se estima todo el horizonte.
    """

    if scenario not in SCENARIOS:
        raise ValueError(f"Escenario inválido: {scenario}")
    rates = fit_historical_promotion_rates(historical_sales, cutoff)
    cutoff = pd.Timestamp(cutoff)

    def transform(X_day, keys, current_date):
        lead = (pd.Timestamp(current_date) - cutoff).days
        confirmed = scenario == "full" or (scenario == "partial" and lead <= confirmed_days)
        if not confirmed:
            X_day["onpromotion"] = estimated_promotions(keys, rates)
        return X_day

    transform.audit = {
        "scenario": scenario,
        "cutoff": cutoff,
        "confirmed_days": confirmed_days if scenario == "partial" else None,
        "history_max_date": rates["max_history_date"],
    }
    return transform
