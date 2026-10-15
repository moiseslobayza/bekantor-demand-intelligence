# Forecasting V2 — protocolo metodológico

## Ventanas y selección

La V2 fija seis folds rolling-origin de 15 días anteriores al 01/08/2017. Los
primeros cuatro son de `tuning`; Optuna minimiza allí el RMSLE medio de forecasts
recursivos completos. Los dos últimos son `development_evaluation` y no se usan
en la búsqueda. Agosto se evalúa, si se ejecuta, sólo después de congelar modelo,
parámetros y número de árboles; se etiqueta como evaluación final previamente
expuesta en el EDA, no como holdout perfectamente ciego.

El número de árboles LightGBM se estima con early stopping sobre un día interno
anterior a cada cutoff externo, cuyos lags son enteramente observados. Después
se reentrena con toda la historia del fold y el objetivo de Optuna se calcula
sobre los 15 días recursivos. Si al menos la mitad de los folds queda dentro de
las últimas 30 iteraciones del máximo, la corrida completa se detiene antes de
mirar los folds externos: el límite todavía sería vinculante.

Cada fold vuelve a derivar aperturas, series elegibles, esquema categórico y
features desde su propio cutoff. El horizonte nunca incorpora ventas reales en
lags o rolling. El pruning sólo puede decidir después de observar el resultado
completo de uno o más folds recursivos; no reemplaza el horizonte por una
validación tabular.

## Modelos

- `Seasonal baseline`: media tienda–familia–día de semana de los 28 días previos.
- `LightGBM original`: configuración V1 congelada.
- `LightGBM optimized`: busca `num_leaves`, `max_depth`, `learning_rate`,
  `min_child_samples` y `reg_lambda`; semilla 42.
- `CatBoost`: mismo target `log1p`, universo, features y recorrido recursivo;
  las categóricas se declaran al modelo y no se calculan target encodings fuera
  de CatBoost. Su número de iteraciones se calibra por early stopping en las
  mismas ventanas internas de los cuatro folds de tuning, con máximo 1.000, y
  se congela mediante la mediana antes de evaluar los dos folds externos.
- `TSB`: actualiza probabilidad de ocurrencia y tamaño positivo por
  tienda–familia. Sus parámetros alpha/beta se eligen sólo en folds de tuning.

La selección principal se define por RMSLE medio de tuning, con WAPE, sesgo y
estabilidad entre folds como guardas. Las comparaciones segmentadas conservan
la misma población y período. TSB no se declara superior sin evidencia y un
híbrido sólo se discute si gana en segmentos previamente definidos. Los
diagnósticos de segmentos se guardan por separado para ambos folds externos;
una mejora aislada en GROCERY II no alcanza para seleccionar TSB.

## Inventario

El simulador convierte pronósticos en una recomendación hipotética. Stock
inicial, lead time, nivel de servicio y costos son parámetros, no campos
observados. El stock de seguridad usa el desvío de errores diarios de desarrollo
y un cuantil Normal explícito. Las ventas observadas no garantizan demanda no
restringida; por ello “faltante”, “excedente” y “costo” siempre significan
resultados del escenario, no hechos históricos.

Con lead time de tres días, la orden llega al inicio del día 4. El stock inicial
es el único disponible durante los días 1–3 y los faltantes potenciales de ese
intervalo se registran como ventas perdidas, no como backorders. La reposición
apunta a la demanda desde la recepción hasta el final del horizonte más stock
de seguridad, descontando sólo el stock proyectado que llega vivo al día 4.

El caso demostrativo se elige sin mirar agosto: dentro del último fold externo,
se toma la serie de mayor volumen observado entre los focos predefinidos
(intermitente, lumpy, LINGERIE o GROCERY II). El stock de seguridad se estima
con errores de la misma serie en el fold externo anterior.

## Ejecución

El modo rápido reduce trials/árboles para verificar el pipeline. El protocolo
completo recomendado usa 20 trials. Los folds preparados se reutilizan en
memoria dentro del proceso y Optuna persiste cada trial en un journal, con una firma
de datos y protocolo que evita reanudar un estudio incompatible:

```bash
python scripts/run_forecasting_v2.py --output-dir outputs/forecasting_v2_full --trials 20 --max-estimators 1500 --catboost-max-iterations 1000
python scripts/rebuild_forecasting_v2_notebook.py
python scripts/execute_forecasting_v2_notebook.py
```

Ninguna métrica se completa manualmente: el notebook lee artefactos generados o
marca con claridad qué experimento no fue ejecutado.
