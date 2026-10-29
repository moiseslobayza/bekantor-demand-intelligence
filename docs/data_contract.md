# Contrato de datos — Forecasting V2

## Fuente y archivos

La fuente es la competencia **Corporación Favorita Grocery Sales Forecasting**
de Kaggle. La V2 requiere los archivos originales `train.csv`, `stores.csv`,
`transactions.csv`, `holidays_events.csv` y `oil.csv`. No se versionan datos,
credenciales ni tokens.

Competencia y condiciones de acceso:
<https://www.kaggle.com/c/favorita-grocery-sales-forecasting>. La descarga
requiere una cuenta de Kaggle y la aceptación previa de las reglas. Las
credenciales deben configurarse fuera del repositorio siguiendo la
documentación oficial de la API de Kaggle.

Descarga con Kaggle API (después de aceptar las reglas de la competencia y
configurar las credenciales según la documentación oficial de Kaggle):

```bash
kaggle competitions download -c favorita-grocery-sales-forecasting
unzip favorita-grocery-sales-forecasting.zip -d data/raw
```

También se pueden descargar manualmente desde Kaggle y extraer esos cinco CSV.
El código busca `data/raw/` de forma predeterminada. Para una ruta portable:

```powershell
$env:FAVORITA_DATA_DIR = "C:\ruta\a\favorita"
python -c "from src.data.load_data import load_raw_data; load_raw_data()"
```

En Google Colab se puede definir, por ejemplo,
`os.environ["FAVORITA_DATA_DIR"] = "/content/favorita"`; montar Drive es
opcional, no un requisito del proyecto. `validate_raw_files()` enumera los
archivos faltantes antes de iniciar un experimento costoso.

Para reproducir solamente las tablas y gráficos del cierre no se necesitan los
CSV: basta con clonar el commit de cierre y usar los artefactos versionados. La
descarga de Kaggle corresponde únicamente a la reproducción experimental.

## Disponibilidad al emitir el pronóstico

| Fuente / feature | Disponibilidad admisible |
|---|---|
| Ventas (`sales`) | Sólo fechas hasta el cutoff. Después del cutoff se usan únicamente para evaluación. |
| Lags y rolling | Se construyen con ventas previas al cutoff y luego con predicciones propias. |
| Tienda, familia y metadata de tienda | Estática y conocida. |
| Calendario | Fecha y día de semana son determinísticos. Feriados programados se admiten; eventos potencialmente conocidos ex post se excluyen del modelo. |
| `onpromotion` | Sólo es conocida anticipadamente si proviene de un calendario comercial confirmado. |
| Transacciones y petróleo | No forman parte de las features V2 actuales. |

El dataset no contiene fecha de aprobación ni de publicación de las
promociones. Por eso el escenario “información completa” es una referencia de
competencia, no disponibilidad empresarial comprobada. Los escenarios parcial
y sin calendario sustituyen promociones no confirmadas con promedios históricos
calculados exclusivamente hasta el cutoff. Esta comparación es predictiva y no
causal.

Las filas diarias del panel distinguen un cero observado de una fila ausente.
TSB sólo trata como cero los registros presentes con `sales == 0`; no crea ceros
a partir de datos faltantes.

## Instalación reproducible

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pytest -q
```

En Linux/macOS, la activación equivalente es `source .venv/bin/activate`.
Optuna, CatBoost y LightGBM están fijados en `requirements.txt`. Los resultados
de experimentos se escriben en `outputs/forecasting_v2/`; no sustituyen a los
datos originales.
