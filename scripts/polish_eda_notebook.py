import json
from pathlib import Path

NOTEBOOK_PATH = Path("notebooks/04_eda.ipynb")


def as_lines(text: str):
    return text.splitlines(keepends=True)


def source_text(cell):
    src = cell.get("source", [])
    return "".join(src) if isinstance(src, list) else src


with NOTEBOOK_PATH.open("r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]
by_id = {cell.get("id"): cell for cell in cells}


def replace_cell(cell_id: str, text: str):
    if cell_id not in by_id:
        raise RuntimeError(f"No se encontró la celda {cell_id}")
    by_id[cell_id]["source"] = as_lines(text)


def replace_first_line(cell_id: str, new_first_line: str):
    if cell_id not in by_id:
        raise RuntimeError(f"No se encontró la celda {cell_id}")
    text = source_text(by_id[cell_id])
    lines = text.splitlines()
    if not lines:
        raise RuntimeError(f"La celda {cell_id} está vacía")
    lines[0] = new_first_line
    by_id[cell_id]["source"] = as_lines("\n".join(lines))


# -----------------------------------------------------------------------------
# 1) Acabado visual y jerarquía de títulos
# -----------------------------------------------------------------------------
replace_cell("7782153f", "## Librerías y configuración")
replace_cell("3796c461", "### Conexión del notebook con el proyecto")
replace_cell("e6097d3f", "## 6.1 Distribución de las ventas")
replace_cell("ea6e42cf", "### Distribución de ventas (muestra)")
replace_cell("4331b5b6", "## 6.2 Evolución temporal de las ventas")
replace_cell("dd566f44", "### Comparación mensual sin agosto de 2017 parcial")
replace_cell("716f7847", "## 6.3 Feriados y eventos — análisis preliminar")
replace_cell("ec3ff557", "## 6.4 Ventas por familia de producto")
replace_cell("2cfe43ec", "## 6.5 Ventas por tienda")

# Títulos redundantes que ya tenían una sección explicativa inmediatamente debajo.
redundant_ids = {
    "54e287dc",  # # promociones vs ventas.
    "ec82ad48",  # # transacciones vs ventas.
    "5679f3da",  # # petróleo vs ventas.
    "87c74800",  # # outliers y anomalías
}
cells = [cell for cell in cells if cell.get("id") not in redundant_ids]
nb["cells"] = cells
by_id = {cell.get("id"): cell for cell in cells}

# Promover Transacciones y Petróleo a secciones principales y renumerar lo posterior.
replace_first_line("413fa944", "## 6.8 Transacciones y ventas")
replace_first_line("3b260ace", "## 6.9 Petróleo y ventas")
replace_first_line("082994a3", "## 6.10 Outliers y anomalías temporales")
replace_first_line("b55c0be1", "## 6.11 Calendario completo de feriados y eventos")
replace_first_line("3eee94c4", "## 6.12 Estructura de las series tienda–familia")
replace_first_line("c4cd1322", "## 6.13 Estrategia de validación temporal y baseline")

# -----------------------------------------------------------------------------
# 2) Síntesis final actualizada con los hallazgos definitivos del EDA
# -----------------------------------------------------------------------------
summary = """# Síntesis del EDA preliminar

El análisis exploratorio muestra que la demanda de Corporación Favorita presenta una estructura fuertemente temporal, heterogénea y concentrada entre tiendas y familias de productos.

Los registros con `sales = 0` no representan un único fenómeno. Del 31,30 % de ceros presentes en el panel original, una parte corresponde a períodos previos a la apertura de tiendas o a familias que determinados establecimientos no comercializan. Una vez considerados estos casos estructurales, los ceros restantes representan períodos reales sin ventas y constituyen información relevante sobre la dinámica de la demanda.

La demanda presenta una marcada estacionalidad. Los fines de semana, especialmente los domingos, registran niveles considerablemente superiores al promedio, mientras que los jueves muestran sistemáticamente menores ventas. También se observan patrones dentro del mes, con mayor actividad al comienzo, alrededor de la quincena y hacia el cierre mensual. Estos resultados justifican incorporar variables de calendario y rezagos temporales.

Las promociones presentan una asociación positiva con las ventas incluso al comparar observaciones dentro de una misma combinación tienda–familia desde 2014. El efecto, sin embargo, varía considerablemente entre familias y debe interpretarse como asociación y no como evidencia causal.

Los feriados y eventos tampoco presentan un efecto uniforme. Su impacto depende del tipo de acontecimiento y de su alcance nacional, regional o local. El calendario construido a nivel fecha–tienda permite conservar esta información y representar correctamente fechas con múltiples eventos.

El análisis robusto de anomalías mediante mediana móvil y MAD identificó 21 días con comportamiento excepcional. Estos valores no serán eliminados automáticamente, ya que varios corresponden a acontecimientos reales. El terremoto de Manabí de abril de 2016 constituye un ejemplo claro: las ventas aumentaron fuertemente durante los días posteriores al evento. Asimismo, los días 1 de enero muestran un comportamiento excepcional asociado al funcionamiento comercial y no deben tratarse como demanda ordinaria.

El problema está compuesto por 1.782 series tienda–familia. En 2017, 1.650 presentan actividad regular, 65 muestran demanda altamente intermitente y 67 permanecen sin ventas. Además, las 200 series de mayor volumen concentran aproximadamente el 76,5 % de las ventas, evidenciando una fuerte concentración comercial.

Las transacciones presentan una elevada relación contemporánea con las ventas (r = 0,837), que también se mantiene al analizar cambios diarios (r = 0,783). Sin embargo, las transacciones del mismo día no pueden utilizarse directamente en un pronóstico ex ante. El rezago de siete días conserva una asociación importante (r = 0,605), consistente con la estacionalidad semanal observada.

El precio del petróleo presenta una correlación negativa moderada con las ventas cuando se comparan sus niveles (r = -0,627). No obstante, esta asociación prácticamente desaparece al analizar primeras diferencias y rezagos, por lo que su capacidad explicativa de corto plazo parece limitada bajo las transformaciones estudiadas.

Finalmente, se definió una estrategia de validación estrictamente temporal, reservando los últimos 15 días disponibles como conjunto de validación. Como referencia inicial se construyó un baseline estacional semanal utilizando exclusivamente información previa, obteniendo un RMSLE de 0,5285. Este resultado servirá como referencia mínima para evaluar posteriormente los modelos predictivos.
"""

summary_id = "a69122f6"
if summary_id not in by_id:
    raise RuntimeError("No se encontró la síntesis actual")

summary_cell = by_id[summary_id]
summary_cell["source"] = as_lines(summary)

# Mover la síntesis al final absoluto del notebook.
nb["cells"] = [cell for cell in nb["cells"] if cell.get("id") != summary_id]
nb["cells"].append(summary_cell)

# -----------------------------------------------------------------------------
# 3) Validaciones mínimas para evitar modificar accidentalmente el notebook
# -----------------------------------------------------------------------------
texts = [source_text(cell) for cell in nb["cells"] if cell.get("cell_type") == "markdown"]

required = [
    "## 6.1 Distribución de las ventas",
    "## 6.6 Promociones y ventas — efecto controlado",
    "## 6.7 Estacionalidad de la demanda",
    "## 6.8 Transacciones y ventas",
    "## 6.9 Petróleo y ventas",
    "## 6.10 Outliers y anomalías temporales",
    "## 6.13 Estrategia de validación temporal y baseline",
    "# Síntesis del EDA preliminar",
]

for heading in required:
    if not any(heading in text for text in texts):
        raise RuntimeError(f"Falta una sección esperada: {heading}")

last_markdown = next(
    cell for cell in reversed(nb["cells"]) if cell.get("cell_type") == "markdown"
)
if last_markdown.get("id") != summary_id:
    raise RuntimeError("La síntesis no quedó al final")

with NOTEBOOK_PATH.open("w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
    f.write("\n")

print("Notebook actualizado correctamente.")
