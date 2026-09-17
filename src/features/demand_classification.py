import numpy as np
import pandas as pd

ADI_THRESHOLD = 1.32
CV2_THRESHOLD = 0.49


def classify_demand_pattern(adi, cv2):
    """Clasifica una serie según el esquema ADI + CV² de Syntetos-Boylan.

    Categorías:
    - Regular: ADI < 1.32 y CV² < 0.49
    - Errática: ADI < 1.32 y CV² >= 0.49
    - Intermitente: ADI >= 1.32 y CV² < 0.49
    - Lumpy: ADI >= 1.32 y CV² >= 0.49
    """
    if pd.isna(adi) or pd.isna(cv2):
        return "Sin demanda positiva"
    if adi < ADI_THRESHOLD and cv2 < CV2_THRESHOLD:
        return "Regular"
    if adi < ADI_THRESHOLD and cv2 >= CV2_THRESHOLD:
        return "Errática"
    if adi >= ADI_THRESHOLD and cv2 < CV2_THRESHOLD:
        return "Intermitente"
    return "Lumpy"


def build_demand_profile(
    sales_modelable,
    group_cols=("store_nbr", "family"),
    date_col="date",
    sales_col="sales",
):
    """Calcula ADI, CV² y clase de demanda para cada serie.

    Esta función debe recibir el panel modelable, es decir, sin períodos de
    preapertura ni combinaciones tienda-familia no comercializadas.

    ADI se calcula como períodos observados / períodos con demanda positiva.
    CV² se calcula sobre la magnitud de la demanda positiva usando desviación
    estándar poblacional (ddof=0), lo que permite clasificar también series
    con una sola observación positiva.
    """
    required = set(group_cols) | {date_col, sales_col}
    missing = required.difference(sales_modelable.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")

    rows = []

    for keys, group in sales_modelable.groupby(list(group_cols), sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        values = group[sales_col].astype(float)
        positive = values[values > 0]
        total_periods = group[date_col].nunique()
        positive_periods = int((values > 0).sum())

        if positive_periods == 0:
            adi = np.nan
            cv2 = np.nan
        else:
            adi = total_periods / positive_periods
            mean_positive = positive.mean()
            cv2 = (
                (positive.std(ddof=0) / mean_positive) ** 2
                if mean_positive > 0
                else np.nan
            )

        row = dict(zip(group_cols, keys))
        row.update(
            {
                "periodos": int(total_periods),
                "periodos_positivos": positive_periods,
                "pct_ceros": float((values == 0).mean() * 100),
                "ventas_totales": float(values.sum()),
                "adi": float(adi) if pd.notna(adi) else np.nan,
                "cv2": float(cv2) if pd.notna(cv2) else np.nan,
                "tipo_demanda": classify_demand_pattern(adi, cv2),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)
