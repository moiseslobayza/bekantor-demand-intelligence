# BEKANTOR Demand Intelligence — auditoría académica previa a la defensa

Fecha: 29 de octubre de 2026
Alcance: rama `forecasting-final`, commit `2d283ae2a0f7d2fed1d3b8b87e9110b81126befb`

## Veredicto

**APTO PARA DEFENSA, con ejecución controlada por secciones.** Las métricas,
los escenarios promocionales y el inventario reconcilian con los artefactos
versionados. La defensa debe usar la modalidad de reproducción de resultados y
no “Run all”: el TP7 conserva capítulos históricos con entrenamiento costoso.

El TP7 consolidado original `PF_Demanda_tp7.ipynb` estaba sin seguimiento y
con sus 198 celdas de código sin ejecutar. Se preservó y se generó una copia
documental. El bloque final independiente sí se validó sin entrenamiento.

## 1. Git y preparación

- Rama encontrada: `forecasting-final`.
- `forecasting-final` apunta exactamente al commit solicitado.
- No se creó `defensa-final`: el árbol ya contenía nueve archivos sin
  seguimiento (`PF_Demanda_tp7.ipynb` y ocho salidas finales). Se conservaron.
- No se hizo staging, commit, push, merge ni modificación de ramas.
- TP7 consolidado localizado: `PF_Demanda_tp7.ipynb`; no se confundió con
  `notebooks/07_project_closure.ipynb`.

## 2. Archivos revisados

- TP7 consolidado y notebook de cierre: `PF_Demanda_tp7.ipynb`,
  `notebooks/07_project_closure.ipynb`.
- Metodología y datos: `README.md`, `requirements.txt`,
  `docs/data_contract.md`, `docs/forecasting_v2_methodology.md`,
  `docs/forecasting_final_conclusions.md`.
- Los 14 archivos de `outputs/forecasting_v2_full/` y los cuatro artefactos
  de datos de `outputs/forecasting_final/`.
- Auditores, constructores y pruebas de cierre en `scripts/`, `src/models/` y
  `tests/`.
- Notebooks protegidos 04–06 mediante sus huellas en las pruebas existentes.

## 3. Resultados numéricos y metodológicos

### Modelos V2

| Modelo | RMSLE medio | WAPE medio |
|---|---:|---:|
| LightGBM optimizado | 0,389282 | 10,955680 % |
| LightGBM original | 0,390548 | 11,104623 % |
| CatBoost | 0,401935 | 12,755792 % |
| Seasonal baseline | 0,439960 | 13,293685 % |
| TSB | 0,479565 | 23,143631 % |

Recomputado desde `model_comparison.csv`:

- mejora absoluta RMSLE: `0,001266503` (redondeada: `0,001267`);
- mejora relativa: `0,324288 %` (redondeada: `0,324 %`);
- reducción WAPE: `0,148943` puntos porcentuales;
- cuatro folds de tuning y dos folds externos de desarrollo, todos de 15 días;
- V2 termina el 31/07/2017 y no evalúa agosto;
- 20 trials Optuna: 17 completos y 3 podados;
- Optuna: 19.594,10 s (5 h 26 min 34 s); corrida completa: 33.567,52 s
  (9 h 19 min 28 s).

LightGBM optimizado es el mejor resultado observado porque obtuvo el menor
RMSLE y WAPE medios bajo la misma población y los mismos dos folds externos.
La mejora es pequeña y dos orígenes no prueban superioridad general. Tampoco
puede traducirse `0,324 %` de RMSLE a igual porcentaje de ahorro o faltantes.

**Hallazgo adicional:** CatBoost alcanzó el máximo de 1.000 iteraciones en 2 de
4 folds de calibración (`654, 1000, 1000, 831`). Se conserva como benchmark con
presupuesto parcialmente vinculante.

### Promociones

| Disponibilidad | RMSLE | WAPE |
|---|---:|---:|
| 15 días confirmados | 0,400791 | 11,752136 % |
| 7 días confirmados | 0,427978 | 11,932466 % |
| 0 días confirmados | 0,461793 | 12,225953 % |

Las promociones no confirmadas se imputan con tasas históricas calculadas
hasta el cutoff. Son escenarios de información, no una evaluación causal: el
dataset no registra cuándo se aprobó el calendario, su profundidad, ejecución,
precio, canibalización ni disponibilidad real de stock.

### Inventario

El escenario de estrés usa tienda 18–`PRODUCE`, stock inicial común de
`2.951,916`, stock de seguridad `445,804`, lead time de tres días y recepción
al inicio del día 4. Los cinco modelos comparten demanda de comparación,
política, costos y demás supuestos. El primer faltante aparece en el día 2 y
antes de la recepción se acumulan `3.786,294` unidades potencialmente no
atendidas.

La compra no usa ventas futuras: éstas sólo evalúan el escenario como proxy de
demanda. La serie fue elegida retrospectivamente; costos, faltantes y ahorro
son simulados. Una orden que llega el día 4 no puede corregir una venta perdida
en los días 2–3, y la política no acumula backorders.

## 4. Problemas encontrados y correcciones

1. **TP7 sin ejecutar.** Las 198 celdas de código del archivo original tenían
   `execution_count = null` y no había salidas. Se preservó el original; la
   copia identifica la modalidad segura y las celdas costosas.
2. **Ruta personal de Drive.** La celda de datos estaba fijada a una carpeta
   personal. La copia busca `FAVORITA_DATA_DIR`, `data/raw/`,
   `/content/favorita` y deja Drive como fallback opcional.
3. **Modalidades mezcladas.** El TP7 podía reconstruir resultados, pero antes
   contenía entrenamiento V1. Se agregó una separación explícita entre
   reproducción de resultados y reproducción experimental.
4. **Bibliografía ausente.** Se incorporaron referencias APA 7 verificadas para
   Kaggle, LightGBM, CatBoost, Optuna, TSB, rolling-origin, métricas e
   inventarios, con citas dentro de la explicación metodológica.
5. **Límite CatBoost no visible en la conclusión.** Se documentó el tope en 2/4
   folds.

No se cambiaron métricas, modelos, features, hiperparámetros, datasets ni
artefactos históricos.

## 5. Reproducibilidad

### A. Reproducción de resultados

No requiere Kaggle ni Drive. Clonar el commit de cierre, instalar dependencias
para las herramientas de notebook y ejecutar desde **“Preparar cierre desde
GitHub (commit fijo, sin entrenamiento)”**. Ese bloque verifica el commit, las
14 huellas SHA-256 y variantes LF/CRLF, y reconstruye tablas y gráficos desde
CSV/JSON.

### B. Reproducción experimental

Requiere aceptar las reglas de Kaggle, descargar `train.csv`, `stores.csv`,
`transactions.csv`, `holidays_events.csv` y `oil.csv`, instalar
`requirements.txt` y ejecutar el pipeline V2 completo. El tiempo observado fue
9 h 19 min 28 s. No se repitió esta modalidad durante la auditoría.

## 6. Controles ejecutados

- `python scripts/audit_forecasting_v2_outputs.py`: aprobado; 14 artefactos,
  agregados, promociones, segmentos e inventario reconciliados.
- `python scripts/audit_forecasting_final_outputs.py`: aprobado; huellas,
  política prospectiva, costos y notebook 07 verificados.
- `.venv-v2\\Scripts\\python.exe -m pytest -q -p no:cacheprovider --basetemp .codex_pytest_tmp`:
  **36 pruebas aprobadas**.
- Bloque final del TP7: **25 celdas ejecutadas sin error** con artefactos locales
  y sin entrenamiento; se suprimió sólo la apertura interactiva de Plotly.
- Validación de estructura, sintaxis, celdas sin ejecutar y salidas con error.
- `git diff --check` y estado final de Git.

## 7. Cobertura de la revisión

Los denominadores son grupos de afirmaciones o secciones del alcance, no un
porcentaje de exactitud ni una certificación de cada celda.

### Calidad y completitud del TP7

| Categoría | Defectos observados | Evaluación |
|---|---:|---|
| Utilidad y completitud | 1 / 9 | El TP7 original sigue sin ejecución completa; la copia separa y documenta las dos modalidades. |
| Claridad analítica | 0 / 7 | Métricas, supuestos, causalidad y límites quedan separados. |
| Consistencia visual e interacción | Desconocido | No se renderizó el TP7 completo; los PNG/HTML finales existentes sí son no vacíos. |

### Corrección y robustez analítica

| Categoría | Defectos observados | Evaluación |
|---|---:|---|
| Autoridad de fuentes | 0 / 8 | Referencias verificadas y enlazadas. |
| Exactitud numérica | 0 / 5 | Cinco modelos y diferencias principales reconciliados. |
| Consistencia entre artefactos | 0 / 4 | CSV, JSON, documentos y notebook 07 coinciden. |
| Controles de calidad | 0 / 4 | Huellas, LF/CRLF, población, fechas y supuestos verificados. |
| Soporte de conclusiones | 0 / 6 | Se evitan causalidad, ahorro real y superioridad universal. |

## 8. Limitaciones pendientes

- No se ejecutó el TP7 completo porque eso iniciaría entrenamientos prohibidos
  y de varias horas.
- No se validó visualmente el notebook consolidado renderizado de punta a punta.
- No hay inventario, órdenes, lead times, márgenes, costos ni promociones
  versionadas de una empresa real.
- Dos folds externos no permiten significancia ni generalización universal.
- La rama `defensa-final` fue creada desde el commit de cierre y conserva los
  archivos originales y corregidos sin sobrescrituras.
