import pandas as pd


def audit_dataframe(name, df):
    print("\n" + "=" * 70)
    print(f"DATASET: {name.upper()}")
    print("=" * 70)

    print(f"\nFilas: {df.shape[0]:,}")
    print(f"Columnas: {df.shape[1]}")

    print("\nCOLUMNAS Y TIPOS:")
    print(df.dtypes)

    print("\nNULOS:")
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0]

    if nulls.empty:
        print("Sin valores nulos.")
    else:
        print(nulls)

    print("\nDUPLICADOS:")
    print(df.duplicated().sum())

    print("\nPRIMERAS FILAS:")
    print(df.head())

    print("\nRESUMEN ESTADÍSTICO:")
    print(df.describe(include="all"))


def build_audit_summary(datasets):
    """Genera un resumen compacto de calidad para todos los datasets."""

    summary = []

    for name, df in datasets.items():

        null_count = df.isna().sum().sum()
        null_columns = (df.isna().sum() > 0).sum()

        row = {
            "dataset": name,
            "rows": len(df),
            "columns": df.shape[1],
            "null_values": null_count,
            "columns_with_nulls": null_columns,
            "duplicates": df.duplicated().sum(),
        }

        # Si existe una columna date, agregamos rango temporal
        if "date" in df.columns:
            row["date_min"] = df["date"].min()
            row["date_max"] = df["date"].max()
        else:
            row["date_min"] = None
            row["date_max"] = None

        summary.append(row)

    return pd.DataFrame(summary)