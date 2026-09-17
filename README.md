# BEKANTOR Demand Intelligence

Proyecto de análisis de demanda retail sobre el dataset **Corporación Favorita Store Sales**.

El EDA está orientado a forecasting: separa ceros estructurales de demanda real, respeta el orden temporal, evita data leakage y estudia estacionalidad, promociones, calendario comercial, eventos extraordinarios y heterogeneidad entre series tienda–familia.

## Estructura actual

- `notebooks/04_eda.ipynb`: EDA final.
- `src/data/`: carga, auditoría, limpieza y diagnóstico estructural.
- `src/features/build_calendar.py`: calendario fecha–tienda.
- `src/features/demand_classification.py`: clasificación ADI + CV².
- `src/features/statistical_analysis.py`: intervalos bootstrap.
- `src/models/baseline.py`: baseline estacional reservado para la próxima etapa de modelado; no se ejecuta dentro del EDA.

## Reproducibilidad local

El proyecto fue **verificado ejecutando 94/94 celdas de `notebooks/04_eda.ipynb` con CPython 3.14.7 y Pandas 3.0.0**. Python 3.14.7 es el entorno verificado, no un requisito exclusivo; la ejecución con otras versiones de Python no fue verificada. `requirements.txt` fija las versiones de las dependencias principales utilizadas en esa ejecución.

1. Crear y activar un entorno virtual.
2. Instalar dependencias:

```bash
pip install -r requirements.txt
```

3. Colocar los archivos originales en `data/raw/`:

```text
train.csv
stores.csv
transactions.csv
holidays_events.csv
oil.csv
```

4. Abrir `notebooks/04_eda.ipynb` y ejecutar **Restart + Run All** desde la raíz del proyecto o desde la carpeta `notebooks/`.

`load_raw_data()` usa `data/raw/` por defecto, pero también acepta una ruta configurable mediante `load_raw_data(data_dir=...)`.

## Nota sobre petróleo

Después de reindexar al calendario diario y aplicar únicamente `ffill`, queda un valor nulo el **2013-01-01**. Se conserva porque no existe una cotización previa conocida. No se utiliza `bfill` ni interpolación con valores futuros para evitar leakage.

## Próxima etapa

La validación temporal, el baseline estacional, RMSLE y los modelos predictivos se evaluarán en la etapa de modelado. El módulo de baseline permanece en el repositorio para reutilizarlo posteriormente, pero no forma parte del EDA entregable.
