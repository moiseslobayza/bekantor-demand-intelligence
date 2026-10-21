# BEKANTOR Demand Intelligence — conclusiones finales

## Alcance y veredicto metodológico

BEKANTOR evolucionó desde un análisis exploratorio del dataset Corporación
Favorita hacia un prototipo reproducible que combina pronóstico recursivo,
validación temporal y simulación de inventario. No es un ERP ni un sistema de
reposición listo para producción: es un experimento académico de apoyo a
decisiones, con resultados cuantitativos auditables y supuestos empresariales
explícitos.

La recomendación técnica es conservar **LightGBM optimizado como el mejor
candidato observado**, sin afirmar que sea universalmente superior al
LightGBM original. La diferencia entre ambos es pequeña y fue observada en
solamente dos ventanas externas de desarrollo.

Fuentes de este cierre:

- los 14 artefactos originales de `outputs/forecasting_v2_full/`;
- el simulador `src/models/inventory.py`;
- los nuevos artefactos determinísticos de `outputs/forecasting_final/`.

No se ejecutó un nuevo entrenamiento ni se utilizó agosto para seleccionar,
ajustar o mejorar modelos.

## 1. Qué significa la mejora de la optimización

### Resultado observado

| Modelo | RMSLE medio | WAPE medio |
|---|---:|---:|
| LightGBM optimizado | 0,389282 | 10,955680 % |
| LightGBM original | 0,390548 | 11,104623 % |
| Diferencia, optimizado − original | −0,001267 | −0,148943 pp |

La reducción relativa de RMSLE es:

\[
\frac{0,390548 - 0,389282}{0,390548}\times 100 = 0,324\%.
\]

RMSLE calcula la raíz del error cuadrático medio en la escala `log1p`. La
diferencia absoluta de `0,001267` indica una reducción muy pequeña de la
magnitud agregada de los errores logarítmicos. No equivale a 0,324 % menos
unidades, 0,324 % menos faltantes ni 0,324 % de ahorro. Tampoco permite inferir
por sí sola una mejora comercial.

### Por qué Optuna sigue siendo útil

Optuna aporta valor metodológico aunque la ganancia final sea modesta:

1. explora sistemáticamente combinaciones definidas de hiperparámetros;
2. mantiene la búsqueda dentro de cuatro folds temporales de tuning;
3. permite podar configuraciones poco prometedoras;
4. registra los 20 trials, sus parámetros, duración y resultado;
5. separa la elección de parámetros de los folds 5 y 6 usados para evaluación.

De los 20 trials, 17 terminaron y 3 fueron podados. La etapa Optuna consumió
`19.594,10` segundos, aproximadamente **5 h 26 min 34 s**. La corrida completa
consumió `33.567,52` segundos, aproximadamente **9 h 19 min 28 s**. Ese costo
computacional produjo una mejora incremental, no un salto de desempeño.

### Límite de la evidencia

Los dos modelos se compararon externamente en los mismos folds 5 y 6 y el
optimizado obtuvo menor RMSLE en ambos. Sin embargo, dos orígenes temporales
representan sólo dos condiciones de calendario, aun cuando cada ventana
contenga muchas observaciones tienda–familia–día. Las observaciones dentro de
una ventana no deben confundirse con muchas réplicas temporales independientes.

Por ello no se calculan pruebas de significancia ni intervalos de confianza.
La conclusión defendible es: **el LightGBM optimizado fue el mejor candidato
observado, pero la evidencia de superioridad sobre el original es limitada**.

## 2. Validación temporal: V1 y V2 no son el mismo experimento

| Versión | Ventanas de selección | Evaluación informada | Relación con agosto | Interpretación |
|---|---|---|---|---|
| V1 | rolling-origin anterior al corte | 01/08/2017–15/08/2017 | agosto ya había sido explorado en el EDA | resultado retrospectivo útil, no test humano perfectamente ciego |
| V2 | folds 1–4, todos anteriores a agosto | folds 5–6, 02/07/2017–31/07/2017 | agosto queda fuera de selección y evaluación V2 | evaluación externa dentro del desarrollo |

### Cronología V2

| Fold | Fin de entrenamiento | Validación | Rol |
|---:|---|---|---|
| 1 | 02/05/2017 | 03/05–17/05 | tuning |
| 2 | 17/05/2017 | 18/05–01/06 | tuning |
| 3 | 01/06/2017 | 02/06–16/06 | tuning |
| 4 | 16/06/2017 | 17/06–01/07 | tuning |
| 5 | 01/07/2017 | 02/07–16/07 | evaluación externa de desarrollo |
| 6 | 16/07/2017 | 17/07–31/07 | evaluación externa de desarrollo |

No corresponde comparar directamente la métrica de agosto de V1 con las
métricas de julio de V2 como si fueran tratamientos evaluados sobre la misma
muestra. Cambian la ventana, el contexto temporal, la población efectiva y el
protocolo de selección. Una diferencia puede provenir del período y no del
modelo.

Los folds 5 y 6 son “externos” respecto de Optuna: no participaron de la
búsqueda. Sin embargo, siguen perteneciendo al ciclo de desarrollo conocido y
documentado. Un test final perfectamente ciego requeriría un período o conjunto
operativo no explorado ni utilizado en decisiones humanas previas.

## 3. Disponibilidad empresarial de promociones

### Escenarios informativos

| Disponibilidad al emitir el pronóstico | Días confirmados | RMSLE | Variación frente a completa | WAPE |
|---|---:|---:|---:|---:|
| Promociones completas | 15 | 0,400791 | referencia | 11,752136 % |
| Siete días confirmados | 7 | 0,427978 | +0,027187; +6,78 % relativo | 11,932466 % |
| Sin calendario confirmado | 0 | 0,461793 | +0,061002; +15,22 % relativo | 12,225953 % |

Un **calendario promocional aprobado** es el plan que la empresa ha autorizado
y puede comunicar a forecasting antes de emitir el pronóstico: productos o
familias, tiendas, fechas, mecánica y profundidad promocional. En una operación
real podrían estar confirmados los quince días, sólo una parte del horizonte o
ninguno, según el proceso comercial.

Cuando una promoción futura no está confirmada, BEKANTOR no la reemplaza con
información posterior. El escenario usa tasas históricas calculadas hasta el
corte disponible para imputar la variable promocional. De este modo conserva la
regla temporal, aunque agrega incertidumbre.

El deterioro de RMSLE y WAPE indica que la planificación pierde precisión al
reducirse la visibilidad promocional. El total pronosticado también cae de
12,05 millones con calendario completo a 11,79 millones sin confirmación. Esto
puede ampliar el riesgo de planificar compras con una señal incompleta, pero no
demuestra que las promociones causen ventas ni cuantifica un costo real.

El dataset no contiene la fecha en que cada promoción fue aprobada. Por lo
tanto, los tres casos son **supuestos de disponibilidad informativa**, no una
reconstrucción histórica de qué conocía la empresa en cada fecha.

Recomendaciones operativas:

- registrar fecha y responsable de aprobación, modificación y cancelación;
- versionar el calendario para reconstruir exactamente qué se conocía en cada
  corte de forecast;
- definir un SLA de comunicación entre Comercial, Abastecimiento y Datos;
- registrar alcance por tienda–familia–producto y profundidad promocional;
- medir por separado desempeño con calendario completo, parcial y no
  confirmado.

## 4. Segundo escenario de inventario: stock inicial insuficiente

### Supuestos comunes

La tienda 18 y la familia `PRODUCE` fueron **seleccionadas retrospectivamente
en V2 según las ventas reales observadas durante la validación**. Se conserva
esa combinación exclusivamente como caso demostrativo para comparar los dos
escenarios; no representa una selección operativa prospectiva ni demuestra que
sea una serie representativa del negocio.

Una vez seleccionado el caso, el stock inicial y las reposiciones se
determinan con pronósticos y supuestos disponibles antes del horizonte. Las
ventas reales futuras se utilizan solamente para evaluar el resultado simulado
y no intervienen en esas decisiones.

Se conserva el horizonte de 15 días, lead time de 3 días, recepción al inicio
del día 4, nivel de servicio objetivo de 95 %, stock de seguridad de `445,804`
unidades y los mismos costos hipotéticos del caso V2.

El nuevo stock inicial es **2.951,916 unidades**, exactamente 10 % del
pronóstico total del baseline estacional (`29.519,159`). La regla sólo usa un
pronóstico disponible al decidir; no mira las ventas futuras reales. La misma
cantidad se aplica a los cinco modelos.

Las ventas observadas comunes se usan después como proxy de demanda para
evaluar el escenario. No permiten reconstruir demanda irrestricta ni ventas
realmente perdidas. Por eso el resultado se denomina “faltante potencial
simulado”.

### Resultado cuantitativo

| Modelo | Demanda pronosticada | Compra recomendada | Primer faltante | Unidades potencialmente no atendidas | Stock final | Costo simulado total |
|---|---:|---:|---:|---:|---:|---:|
| Seasonal baseline | 29.519,159 | 23.348,762 | día 2 | 3.786,294 | 2.292,529 | 33.754,935 |
| LightGBM original | 30.390,579 | 24.022,592 | día 2 | 3.786,294 | 2.966,359 | 34.590,484 |
| LightGBM optimizado | 29.424,287 | 23.035,018 | día 2 | 3.786,294 | 1.978,785 | 33.365,892 |
| CatBoost | 29.727,530 | 23.552,777 | día 2 | 3.786,294 | 2.496,544 | 34.007,914 |
| TSB | 29.919,121 | 24.381,101 | día 2 | 3.786,294 | 3.324,868 | 35.035,036 |

La demanda comparativa y el stock inicial son comunes, por lo que los cinco
modelos sufren el mismo faltante antes de la recepción: `3.786,294` unidades
potenciales, en los días 2 y 3. La mercadería llega recién al inicio del día 4.
La política trata el faltante como venta potencial perdida: no genera backlog y
no se suma a la compra recomendada. La reposición cubre la demanda pronosticada
desde la recepción más el stock de seguridad, por eso cada modelo recomienda
una cantidad distinta.

Los costos combinan compra, tenencia diaria y penalización hipotética por
faltante. Son resultados de simulación, no ahorros ni pérdidas reales. Que
LightGBM optimizado tenga el menor costo en este escenario no prueba una ventaja
económica general: depende de parámetros supuestos y de una sola serie.

## 5. TSB y demanda irregular

TSB (Teunter–Syntetos–Babai) separa dos componentes: la probabilidad de que
ocurra demanda y el tamaño esperado cuando ocurre. Por eso puede ser adecuado
para series con muchos ceros y obsolescencia o intervalos irregulares entre
ventas.

Sin embargo, TSB se ajusta por serie y utiliza una estructura mucho más
restringida. El modelo global LightGBM comparte señal entre tiendas y familias y
dispone de calendario, promociones, atributos categóricos, lags y estadísticas
móviles. La presencia de ceros no garantiza que la descomposición de TSB
capture mejor toda la estructura predictiva.

Promedios de los folds externos:

| Segmento | LightGBM optimizado RMSLE | TSB RMSLE | Hallazgo |
|---|---:|---:|---|
| Intermitente | 0,396275 | 0,456521 | LightGBM fue mejor en ambos folds |
| Lumpy | 0,521920 | 0,602332 | TSB no resolvió la alta variabilidad |
| LINGERIE | 0,612756 | 0,688252 | TSB quedó por detrás en el promedio |
| GROCERY II | 0,594143 | 0,616418 | TSB sólo tuvo una mejora puntual en fold 6 |

En `GROCERY II`, TSB obtuvo RMSLE `0,603039` en el fold 6 frente a `0,604174`
de LightGBM optimizado: una ventaja puntual de `0,001135`. En el fold 5 quedó
peor (`0,629796` frente a `0,584111`). El promedio externo y el resultado global
favorecen LightGBM. Esa inversión entre ventanas es evidencia contra adoptar
TSB a partir de un único segmento–fold.

Es razonable plantear como hipótesis que TSB puede ayudar en subconjuntos con
ceros persistentes y poca señal exógena, pero los resultados actuales no
demuestran una superioridad estable. Recomendamos LightGBM como modelo general
y estudiar modelos especializados sólo cuando haya volumen suficiente de
series, validación temporal adicional, estabilidad entre ventanas y una mejora
operativamente relevante.

## 6. Tres decisiones comerciales concretas

### A. Planificación de compras y reposición

| Elemento | Aplicación |
|---|---|
| Información necesaria | forecast diario, stock disponible, pedidos en tránsito, lead time, stock de seguridad, múltiplos y costos |
| Resultado BEKANTOR | demanda esperada y cantidad sugerida al momento de recepción |
| Acción | emitir o revisar una orden de compra y priorizar series con riesgo de faltante |
| Riesgo | errores de forecast, lead time variable y ventas censuradas por quiebres |
| Datos reales faltantes | inventario, recepciones, órdenes, backorders, merma y nivel de servicio observado |

### B. Planificación de promociones

| Elemento | Aplicación |
|---|---|
| Información necesaria | calendario versionado, fecha de aprobación, tienda–producto, profundidad y vigencia |
| Resultado BEKANTOR | forecast bajo disponibilidad completa, parcial o no confirmada |
| Acción | adelantar abastecimiento o elevar una alerta cuando el plan cambia tarde |
| Riesgo | el análisis es predictivo, no causal; una promoción puede estar correlacionada con otras decisiones |
| Datos reales faltantes | aprobación histórica, precio, descuento, exhibición, canibalización y ejecución efectiva |

### C. Priorización de productos y familias

| Elemento | Aplicación |
|---|---|
| Información necesaria | métricas por serie/familia, ADI/CV², sesgo, volumen y faltantes simulados |
| Resultado BEKANTOR | ranking de segmentos difíciles y señales de sub/sobrepredicción |
| Acción | revisión humana más frecuente, excepción operativa o piloto especializado |
| Riesgo | un promedio puede ocultar heterogeneidad; error alto no prueba causa ni valor económico |
| Datos reales faltantes | margen, criticidad, sustitución, inventario, shelf life y costo de quiebre |

## 7. Conclusión final

1. **Forecasting.** LightGBM optimizado obtuvo el mejor resultado medio
   observado (`RMSLE 0,389282`, `WAPE 10,955680 %`) frente a cuatro alternativas.
2. **Validación y optimización.** La separación de cuatro folds de tuning y dos
   folds externos anteriores a agosto mejora el rigor, pero no constituye un
   test final perfectamente ciego ni demuestra superioridad universal.
3. **Aplicación empresarial.** Los pronósticos pueden alimentar decisiones de
   reposición, coordinación promocional y priorización de excepciones cuando se
   combinan con supuestos operativos explícitos.
4. **Limitaciones.** No hay inventario real, demanda perdida observada, fechas
   de aprobación promocional, lead times efectivos, costos ni márgenes reales.
   Los escenarios no identifican efectos causales ni ahorros.
5. **Trabajo futuro.** Validar con inventarios y órdenes reales, versionar
   promociones, ampliar orígenes temporales externos, medir calibración por
   nivel de servicio y evaluar modelos especializados sólo en segmentos donde
   exista evidencia estable.

El principal avance del proyecto no es una décima adicional de precisión, sino
haber convertido el EDA en un sistema experimental reproducible que explicita
qué se conoce al momento de decidir, separa selección y evaluación, y traduce
pronósticos a escenarios operativos sin ocultar sus límites.
