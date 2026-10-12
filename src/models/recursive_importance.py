import numpy as np
import pandas as pd

from src.models.baseline import rmsle
from src.models.recursive_forecast import recursive_forecast


KEY_COLUMNS = ["date", "store_nbr", "family"]


def permute_feature_within_date(X_day, keys, feature, rng):
    """Permuta una feature sólo entre series de la misma fecha.

    Devuelve también una tabla de auditoría que conserva la fecha y serie de
    origen de cada valor. La función acepta más de una fecha para que la
    restricción temporal pueda probarse de forma aislada, aunque el forecast la
    invoca una vez por día.
    """

    if len(X_day) != len(keys):
        raise ValueError("X_day y keys deben tener la misma cantidad de filas")
    if feature not in X_day.columns:
        raise ValueError(f"Feature inexistente: {feature}")

    transformed = X_day.copy()
    values = X_day[feature].to_numpy(copy=True)
    dates = keys["date"].to_numpy(copy=True)
    stores = keys["store_nbr"].to_numpy(copy=True)
    families = keys["family"].to_numpy(copy=True)
    permuted_values = np.empty_like(values)
    source_dates = np.empty_like(dates)
    source_stores = np.empty_like(stores)
    source_families = np.empty_like(families)

    for positions in keys.groupby("date", sort=False).indices.values():
        positions = np.asarray(positions, dtype=int)
        source_positions = rng.permutation(positions)
        permuted_values[positions] = values[source_positions]
        source_dates[positions] = dates[source_positions]
        source_stores[positions] = stores[source_positions]
        source_families[positions] = families[source_positions]

    transformed[feature] = permuted_values
    audit = keys[KEY_COLUMNS].reset_index(drop=True).copy()
    audit["source_date"] = source_dates
    audit["source_store_nbr"] = source_stores
    audit["source_family"] = source_families
    if not audit["date"].eq(audit["source_date"]).all():
        raise AssertionError("La permutación cruzó fechas del horizonte")
    return transformed, audit


def recursive_permutation_importance(
    model,
    features_base,
    train,
    validation,
    validation_start,
    validation_end,
    model_features,
    category_schema,
    n_repeats=3,
    random_state=42,
    lags=(1, 7, 14, 28),
    rolling_windows=(7, 14, 28),
):
    """Permutation importance compatible con un forecast recursivo.

    Primero ejecuta el recorrido base para obtener el score de referencia. Para
    cada feature y repetición vuelve a ejecutar todo el horizonte y permuta la
    feature, en cada día, sólo entre las series presentes en esa fecha. Por lo
    tanto los lags y rolling de ese día se reconstruyen desde la historia del
    recorrido perturbado antes de permutarse, y las predicciones resultantes se
    realimentan en los días posteriores. El target real se usa sólo para medir
    RMSLE. La medida expresa sensibilidad predictiva, no causalidad.
    """

    if n_repeats < 1:
        raise ValueError("n_repeats debe ser al menos 1")

    baseline_predictions, trace = recursive_forecast(
        model=model,
        features_base=features_base,
        train=train,
        validation_start=validation_start,
        validation_end=validation_end,
        model_features=model_features,
        category_schema=category_schema,
        lags=lags,
        rolling_windows=rolling_windows,
        return_feature_trace=True,
    )
    truth = validation[KEY_COLUMNS + ["sales"]].copy()
    baseline_eval = truth.merge(
        baseline_predictions,
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    if baseline_eval["prediction"].isna().any():
        raise AssertionError("Faltan predicciones en el recorrido recursivo base")
    baseline_score = rmsle(baseline_eval["sales"], baseline_eval["prediction"])

    rng = np.random.default_rng(random_state)
    rows = []
    for feature in model_features:
        increases = []
        for repeat in range(n_repeats):
            def replace_feature(X_day, keys, current_date):
                transformed, audit = permute_feature_within_date(
                    X_day=X_day,
                    keys=keys,
                    feature=feature,
                    rng=rng,
                )
                if not audit["source_date"].eq(current_date).all():
                    raise AssertionError(
                        f"{feature} recibió valores de otra fecha en {current_date}"
                    )
                return transformed

            perturbed = recursive_forecast(
                model=model,
                features_base=features_base,
                train=train,
                validation_start=validation_start,
                validation_end=validation_end,
                model_features=model_features,
                category_schema=category_schema,
                lags=lags,
                rolling_windows=rolling_windows,
                feature_transform=replace_feature,
            )
            perturbed_eval = truth.merge(
                perturbed,
                on=KEY_COLUMNS,
                how="left",
                validate="one_to_one",
            )
            score = rmsle(
                perturbed_eval["sales"], perturbed_eval["prediction"]
            )
            increases.append(score - baseline_score)

        rows.append(
            {
                "feature": feature,
                "importance_mean": float(np.mean(increases)),
                "importance_std": float(np.std(increases, ddof=0)),
                "baseline_RMSLE": baseline_score,
            }
        )

    result = pd.DataFrame(rows).sort_values(
        "importance_mean", ascending=False
    ).reset_index(drop=True)
    return result, baseline_predictions, trace
