# BEKANTOR Demand Intelligence — resumen ejecutivo para la defensa

## Mensaje central

BEKANTOR es un prototipo académico reproducible de apoyo a decisiones de
demanda, no un ERP ni una prueba de ahorro real. Su aporte principal es separar
qué información se conoce al emitir el pronóstico, validar temporalmente los
modelos y traducir el forecast a escenarios operativos con supuestos visibles.

## Resultados que sí están comprobados

- LightGBM optimizado obtuvo el mejor resultado observado: RMSLE `0,389282` y
  WAPE `10,955680 %` en dos folds externos de desarrollo.
- Frente al LightGBM original, la mejora fue `0,001267` RMSLE, `0,324 %`
  relativa y `0,148943` puntos porcentuales de WAPE.
- La mejora es consistente en las dos ventanas, pero pequeña. Dos folds no
  demuestran superioridad universal ni justifican una equivalencia económica.
- Optuna aportó trazabilidad —20 trials, 17 completos y 3 podados— a un costo
  de 5 h 26 min; la corrida completa tomó 9 h 19 min.
- V1 y V2 no son la misma evaluación: V1 informa agosto; V2 selecciona con
  cuatro folds y evalúa en dos ventanas externas de julio.

## Promociones

El RMSLE empeora de `0,400791` con 15 días confirmados a `0,427978` con 7 días
y `0,461793` sin confirmaciones. Esto demuestra dependencia predictiva de la
información promocional, no causalidad. Para una empresa real se necesitan
fechas de aprobación, versiones del calendario, profundidad, precio, alcance y
ejecución efectiva.

## Inventario

El caso tienda 18–`PRODUCE` es retrospectivo y demostrativo. Con stock inicial
`2.951,916`, lead time de tres días y recepción al inicio del día 4, el primer
faltante potencial aparece en el día 2 y se acumulan `3.786,294` unidades antes
de la recepción. Los cinco modelos comparten stock, demanda, seguridad, plazo,
costos y política. Las ventas futuras evalúan el escenario; no fijan la compra.
Los costos son hipotéticos y no representan pérdidas ni ahorros observados.

## Tres decisiones empresariales defendibles

1. **Cuánto reponer.** Combinar el forecast con stock real, pedidos en tránsito,
   lead time y stock de seguridad; alertar faltantes anteriores a la recepción.
2. **Cómo coordinar promociones.** Exigir un calendario comercial versionado y
   confirmado antes del corte del forecast; elevar alertas por cambios tardíos.
3. **Dónde concentrar revisión humana.** Priorizar familias o tiendas con error,
   sesgo o demanda irregular altos, incorporando margen, criticidad y vida útil.

## Frases para la exposición oral

- “LightGBM optimizado fue el mejor candidato observado; no afirmamos que sea
  universalmente superior.”
- “Una mejora de 0,324 % en RMSLE no equivale a 0,324 % de ahorro.”
- “Los escenarios promocionales son predictivos, no causales.”
- “Las ventas observadas son un proxy para evaluar el inventario simulado, no
  demanda irrestricta comprobada.”
- “El valor del proyecto está en la trazabilidad y en hacer explícitos los
  supuestos, no en agregar más modelos.”

## Cierre

El proyecto es **apto para defensa** si se presenta la reproducción de
resultados desde artefactos congelados, se explican las limitaciones y no se
confunden resultados predictivos con impacto económico real. No usar “Run all”
durante la defensa: el notebook conserva entrenamientos históricos por
trazabilidad.
