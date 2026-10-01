import pandas as pd


def fit_category_schema(data, categorical_features):
    """Convierte categóricas y conserva exactamente los niveles de train."""

    result = data.copy()
    schema = {}

    for col in categorical_features:
        if col not in result.columns:
            raise ValueError(f"Falta la columna categórica requerida: {col}")
        result[col] = result[col].astype("category")
        schema[col] = list(result[col].cat.categories)

    return result, schema


def apply_category_schema(data, category_schema, fail_on_unknown=True):
    """Aplica niveles de train y detecta categorías desconocidas.

    Si ``fail_on_unknown`` es verdadero, falla antes de que el modelo reciba
    códigos categóricos ambiguos. También devuelve el detalle para auditoría.
    """

    result = data.copy()
    unknown = {}

    for col, categories in category_schema.items():
        if col not in result.columns:
            raise ValueError(f"Falta la columna categórica requerida: {col}")

        known = pd.Index(categories)
        values = pd.Index(result[col].dropna().unique())
        unseen = values.difference(known).tolist()
        if unseen:
            unknown[col] = unseen

    if unknown and fail_on_unknown:
        details = "; ".join(
            f"{col}={values}" for col, values in sorted(unknown.items())
        )
        raise ValueError(
            "Categorías desconocidas detectadas antes de predict: " + details
        )

    for col, categories in category_schema.items():
        result[col] = pd.Categorical(result[col], categories=categories)

    return result, unknown
