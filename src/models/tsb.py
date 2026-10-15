"""Teunter-Syntetos-Babai para demanda intermitente, sin datos futuros."""

import numpy as np
import pandas as pd


def fit_tsb_series(values, alpha=0.2, beta=0.2):
    """Estima probabilidad de ocurrencia y tamaño positivo de una serie."""

    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("TSB requiere una serie unidimensional no vacía")
    if (values < 0).any():
        raise ValueError("TSB requiere demanda no negativa")
    if not 0 < alpha <= 1 or not 0 < beta <= 1:
        raise ValueError("alpha y beta deben pertenecer a (0, 1]")

    positive = values[values > 0]
    if len(positive) == 0:
        return {"probability": 0.0, "size": 0.0, "forecast": 0.0}

    probability = float(np.mean(values > 0))
    size = float(positive[0])
    for value in values:
        occurrence = float(value > 0)
        probability += beta * (occurrence - probability)
        if occurrence:
            size += alpha * (value - size)
    return {
        "probability": float(np.clip(probability, 0, 1)),
        "size": max(0.0, size),
        "forecast": max(0.0, probability * size),
    }


def tsb_forecast(train, validation, alpha=0.2, beta=0.2):
    """Pronóstico constante por tienda-familia para el horizonte elegible."""

    rows = []
    grouped = train.sort_values("date").groupby(["store_nbr", "family"], observed=True)
    states = {
        key: fit_tsb_series(group["sales"].to_numpy(), alpha=alpha, beta=beta)
        for key, group in grouped
    }
    for row in validation[["date", "store_nbr", "family"]].itertuples(index=False):
        state = states.get((row.store_nbr, row.family), {"forecast": 0.0})
        rows.append(
            {
                "date": row.date,
                "store_nbr": row.store_nbr,
                "family": row.family,
                "prediction_tsb": state["forecast"],
            }
        )
    return pd.DataFrame(rows)


def select_tsb_parameters(fold_frames, candidates=None):
    """Selecciona alpha/beta sólo con folds temporales suministrados."""

    from src.models.baseline import rmsle

    candidates = candidates or [(a, b) for a in (0.1, 0.2, 0.3) for b in (0.1, 0.2, 0.3)]
    rows = []
    for alpha, beta in candidates:
        scores = []
        for prepared in fold_frames:
            pred = tsb_forecast(prepared["train"], prepared["validation"], alpha, beta)
            scored = prepared["validation"].merge(
                pred, on=["date", "store_nbr", "family"], validate="one_to_one"
            )
            if len(scored) != len(prepared["validation"]):
                raise ValueError("TSB no cubre exactamente la población del fold")
            scores.append(rmsle(scored["sales"], scored["prediction_tsb"]))
        rows.append({"alpha": alpha, "beta": beta, "RMSLE_mean": float(np.mean(scores))})
    results = pd.DataFrame(rows).sort_values(["RMSLE_mean", "alpha", "beta"])
    best = results.iloc[0]
    return (float(best["alpha"]), float(best["beta"])), results.reset_index(drop=True)
