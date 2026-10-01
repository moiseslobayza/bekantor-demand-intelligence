# BEKANTOR Demand Intelligence

Proyecto de análisis y forecasting de demanda retail sobre el dataset **Corporación Favorita Store Sales**.

## Entregables

- `notebooks/04_eda.ipynb`: EDA orientado a forecasting.
- `notebooks/05_modeling.ipynb`: baseline estacional, LightGBM recursivo, rolling-origin backtesting, test final y diagnósticos.
- `src/data/`: carga, limpieza y elegibilidad temporal del panel.
- `src/features/`: calendario, contrato de features, lags, rolling y clasificación ADI/CV².
- `src/models/`: baseline, esquema categórico, validación, entrenamiento y forecast recursivo.
- `tests/test_forecasting_pipeline.py`: tests automáticos de las garantías temporales del pipeline.

## Metodología de modelado

El intervalo **01/08/2017–15/08/2017** se reserva como test final y no participa en la selección del modelo. Aperturas, familias surtidas y universo modelable se derivan sólo con datos disponibles hasta el cutoff de cada ejecución. Las series que aparecen después del cutoff se conservan y se reportan como cold-start por separado.

La selección del número de árboles usa tres ventanas rolling-origin anteriores al test, todas de 15 días y con entrenamiento expansivo. Los `best_iteration` fueron `1029`, `369` y `1336`; se fijó su mediana (`1029`) antes de reentrenar con toda la historia hasta el 31/07/2017 y evaluar el test una única vez. El techo técnico fue 2000 árboles y ningún fold lo alcanzó.

Los lags se construyen por fecha calendario. Los rolling usan `[t-w, t)` y excluyen el día actual. El forecast es recursivo: después del cutoff sólo incorpora sus propias predicciones a la historia, nunca las ventas reales futuras.

`onpromotion` se utiliza bajo el supuesto de la competencia Favorita de que las promociones futuras están disponibles. Los eventos `type=Event`, potencialmente conocidos ex post, no son predictivos; sólo entran feriados y eventos programados. Las categorías se aprenden de `X_train`, se reaplican sin cambios en cada día recursivo y una categoría desconocida produce un error explícito antes de `predict`.

## Resultados validados

Backtesting anterior al test final (media ± desvío estándar, tres folds):

| Modelo | RMSLE | MAE | Error firmado medio |
|---|---:|---:|---:|
| Baseline estacional | 0.442592 ± 0.013444 | 69.639352 ± 6.235829 | 10.988712 ± 16.210790 |
| LightGBM recursivo | 0.390971 ± 0.009886 | 57.220365 ± 0.629283 | -4.462179 ± 15.685443 |

Test final 01/08/2017–15/08/2017:

| Modelo | RMSLE | MAE | Error firmado medio |
|---|---:|---:|---:|
| Baseline estacional | 0.536571 | 85.477984 | 4.223710 |
| LightGBM recursivo | 0.398867 | 65.176282 | 16.999931 |

La corrección del universo produjo **0 diferencias de series y 0 diferencias de aperturas** en el test final actual, y no hubo series cold-start en ese período. El baseline es idéntico al resultado anterior. Frente al LightGBM previo (`0.405895 / 70.298360`), el nuevo pipeline mejora RMSLE en `0.007028` y MAE en `5.122078`. Es el efecto conjunto del contrato de features corregido, el esquema categórico estable y la cantidad de árboles seleccionada exclusivamente antes del test; no se interpreta como atribución causal individual.

El notebook también reporta error por horizonte, tienda, familia y tipo ADI/CV²; permutation importance fuera de entrenamiento; e importancias gain/split, que describen uso interno del modelo y no causalidad.

## Reproducibilidad

La ejecución validada utilizó CPython 3.14.7, pandas 3.0.0, scikit-learn 1.9.1 y LightGBM 4.7.0. Las dependencias están fijadas en `requirements.txt`; el notebook no instala paquetes.

1. Crear y activar un entorno virtual.
2. Instalar dependencias:

```bash
pip install -r requirements.txt
```

3. Colocar en `data/raw/`:

```text
train.csv
stores.csv
transactions.csv
holidays_events.csv
oil.csv
```

4. Ejecutar tests:

```bash
python -m pytest -q
```

5. Ejecutar el notebook desde un kernel limpio:

```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebooks/05_modeling.ipynb
```

La ejecución validada completó las 20 celdas de código sin errores y los 8 tests automáticos. `load_raw_data()` usa `data/raw/` por defecto y también acepta `load_raw_data(data_dir=...)`.

## Nota sobre petróleo

Después de reindexar al calendario diario y aplicar únicamente `ffill`, queda un valor nulo el **2013-01-01**. Se conserva porque no existe una cotización previa conocida. No se utiliza `bfill` ni interpolación con valores futuros para evitar leakage.
