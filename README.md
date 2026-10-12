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

La permutation importance también respeta ese recorrido: se calcula sobre el último fold previo al test, permuta cada feature exclusivamente entre series de la misma fecha y vuelve a pronosticar los 15 días completos en cada repetición. Antes de permutar, los lags y rolling diarios se reconstruyen con la historia del recorrido perturbado; luego sus predicciones alimentan los días siguientes. Las ventas reales de validación sólo se usan para puntuar RMSLE. La medida expresa sensibilidad predictiva, no causalidad; las features constantes entre series de un mismo día tienen importancia cero por construcción.

`onpromotion` se utiliza bajo el supuesto de la competencia Favorita de que el plan promocional futuro está disponible antes del horizonte. En un uso empresarial sólo sería válido si proviniera de un calendario comercial aprobado; un registro retrospectivo no sería información admisible. Los eventos `type=Event`, potencialmente conocidos ex post, no son predictivos; sólo entran feriados y eventos programados.

### Limitación de aislamiento temporal

El EDA aprobado se había realizado previamente sobre datos que llegan al 15/08/2017. Por eso no se afirma independencia humana absoluta del test. Las decisiones predictivas se recalculan sin ese período: elegibilidad, features históricas, esquema categórico y clasificación ADI/CV² usan datos anteriores al cutoff; los árboles se seleccionan únicamente con folds anteriores al 01/08; y no se ajustan hiperparámetros tras observar el test. El EDA previo queda como análisis descriptivo retrospectivo y esta exposición previa se documenta como limitación.

## Resultados validados

Backtesting anterior al test final (media ± desvío estándar, tres folds):

| Modelo | RMSLE | MAE | WAPE | Error firmado medio | Sesgo agregado |
|---|---:|---:|---:|---:|---:|
| Baseline estacional | 0.442592 ± 0.013444 | 69.639352 ± 6.235829 | 14.0994% ± 1.3970 | 10.988712 ± 16.210790 | 2.2658% |
| LightGBM recursivo | 0.390971 ± 0.009886 | 57.220365 ± 0.629283 | 11.5795% ± 0.2155 | -4.462179 ± 15.685443 | -0.9144% |

Test final 01/08/2017–15/08/2017:

| Modelo | RMSLE | MAE | WAPE | Error firmado medio | Sesgo agregado | Sobrepredicción | Subpredicción |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline estacional | 0.536571 | 85.477984 | 17.8301% | 4.223710 | 0.8810% | 1,163,207 | 1,053,665 |
| LightGBM recursivo | 0.398867 | 65.176282 | 13.5953% | 16.999931 | 3.5461% | 1,065,620 | 624,727 |

La corrección del universo produjo **0 diferencias de series y 0 diferencias de aperturas** en el test final actual, y no hubo series cold-start en ese período. El baseline es idéntico al resultado anterior. Frente al LightGBM previo (`0.405895 / 70.298360`), el nuevo pipeline mejora RMSLE en `0.007028` y MAE en `5.122078`. Es el efecto conjunto del contrato de features corregido, el esquema categórico estable y la cantidad de árboles seleccionada exclusivamente antes del test; no se interpreta como atribución causal individual.

WAPE retorna `NaN` para segmentos sin ventas reales, en vez de ocultar la división por cero. El signo se define como `predicción - real`: la subpredicción señala riesgo potencial de faltantes y la sobrepredicción riesgo potencial de sobrestock, pero sin datos de inventario, quiebres o costos no se afirma que esos efectos hayan ocurrido. RMSLE y WAPE no son equivalentes: el primero enfatiza errores relativos y el segundo unidades absolutas ponderadas por volumen.

### Diagnóstico de segmentos difíciles

| Segmento | Modelo | RMSLE | MAE | WAPE | Sesgo agregado | % ceros test | Series |
|---|---|---:|---:|---:|---:|---:|---:|
| LINGERIE | Baseline | 0.7169 | 4.4750 | 56.7520% | -21.5320% | 8.3951% | 54 |
| LINGERIE | LightGBM | 0.6215 | 3.9432 | 50.0080% | -23.0140% | 8.3951% | 54 |
| GROCERY II | Baseline | 0.6952 | 13.3926 | 46.7083% | 18.5640% | 1.8519% | 54 |
| GROCERY II | LightGBM | 0.6120 | 11.0205 | 38.4352% | 8.6703% | 1.8519% | 54 |
| Intermitente | Baseline | 0.4487 | 18.2013 | 18.9617% | 6.0032% | 31.5892% | 409 |
| Intermitente | LightGBM | 0.4009 | 15.7370 | 16.3945% | 4.9536% | 31.5892% | 409 |
| Lumpy | Baseline | 0.9446 | 16.5281 | 65.3811% | -45.4417% | 25.4367% | 271 |
| Lumpy | LightGBM | 0.5451 | 11.0332 | 43.6446% | -22.0707% | 25.4367% | 271 |

LightGBM mejora las tres métricas en los cuatro focos, pero LINGERIE y especialmente lumpy conservan WAPE alto y sesgo de subpredicción. Es evidencia para evaluar más adelante un hurdle de dos etapas, no una justificación suficiente para implementarlo todavía.

La sensibilidad sin promociones futuras, manteniendo el mismo modelo, empeora RMSLE de `0.3989` a `0.4925` y WAPE de `13.5953%` a `16.6070%`; el forecast total cae de 12.87 M a 11.19 M. Esto muestra dependencia predictiva del plan promocional, no causalidad.

La importance recursiva del último fold ubica primero `rolling_mean_7` (+1.3446 RMSLE), `lag_1` (+0.8280), `onpromotion` (+0.4338) y `family` (+0.3207). Las importancias gain/split se conservan como descripción interna del modelo, no como efecto causal.

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

5. Reconstruir y ejecutar el notebook desde un kernel limpio:

```bash
python scripts/rebuild_modeling_notebook.py
python scripts/execute_modeling_notebook.py
```

La ejecución validada completó las 21 celdas de código sin errores y los 11 tests automáticos. El runtime de validación no podía cargar la integración LightGBM/scikit-learn por una política local de Windows; el pipeline usó el fallback a la API nativa del mismo LightGBM, con idénticos parámetros y resultados a la versión ya auditada. En entornos normales se conserva la integración original. `load_raw_data()` usa `data/raw/` por defecto y también acepta `load_raw_data(data_dir=...)`.

## Nota sobre petróleo

Después de reindexar al calendario diario y aplicar únicamente `ffill`, queda un valor nulo el **2013-01-01**. Se conserva porque no existe una cotización previa conocida. No se utiliza `bfill` ni interpolación con valores futuros para evitar leakage.
