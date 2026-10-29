"""Crea una copia documental y portable del TP7 consolidado para la defensa.

No ejecuta notebooks, no entrena modelos y no modifica el archivo entregado.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]


REPRODUCIBILITY_CELL = r'''# @title Modalidades de reproducción para la defensa
from IPython.display import display, HTML

display(HTML(r"""
<div style="margin:20px 0;padding:20px 22px;background:#07111F;border:1px solid #155E75;border-left:5px solid #22D3EE;border-radius:14px;color:#CBD5E1;line-height:1.75;font-size:14px">
  <div style="font-size:11px;letter-spacing:1.8px;color:#67E8F9;font-weight:850">REPRODUCIBILIDAD · DOS MODALIDADES</div>
  <div style="font-size:21px;color:#F8FAFC;font-weight:850;margin:6px 0 12px">Elegir el recorrido antes de ejecutar</div>
  <p><b style="color:#A7F3D0">A · Reproducción de resultados (defensa).</b> Ir al bloque <b>“Preparar cierre desde GitHub (commit fijo, sin entrenamiento)”</b> y ejecutar desde allí hasta el final. Reconstruye tablas y gráficos desde CSV/JSON versionados; no requiere Kaggle, Drive, LightGBM, CatBoost ni Optuna.</p>
  <p><b style="color:#FDE68A">B · Reproducción experimental.</b> Repite preparación, entrenamiento, rolling-origin, tuning y simulaciones. Requiere aceptar las reglas de Kaggle, descargar los cinco CSV, instalar <code>requirements.txt</code> y reservar varias horas. Usar <code>data/raw/</code> o <code>FAVORITA_DATA_DIR</code>; Drive es opcional.</p>
  <p style="margin-bottom:0"><b style="color:#FDA4AF">No usar “Run all” durante la defensa.</b> Los capítulos históricos conservan las celdas de entrenamiento para trazabilidad. No almacenar tokens ni credenciales en el notebook.</p>
</div>
"""))
'''


PORTABLE_DATA_CELL = r'''# @title Localizar los CSV de Kaggle (Drive opcional)
from pathlib import Path
import os

required_files = [
    "train.csv",
    "stores.csv",
    "transactions.csv",
    "holidays_events.csv",
    "oil.csv",
]

candidates = []
if os.environ.get("FAVORITA_DATA_DIR"):
    candidates.append(Path(os.environ["FAVORITA_DATA_DIR"]).expanduser())
candidates.extend([
    Path.cwd() / "data" / "raw",
    PROJECT_ROOT / "data" / "raw",
    Path("/content/favorita"),
])

DATA_DIR = next(
    (path for path in candidates if all((path / name).is_file() for name in required_files)),
    None,
)

if DATA_DIR is None and Path("/content").is_dir():
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        legacy_drive = Path("/content/drive/MyDrive/csv/csv demanda favorita")
        if all((legacy_drive / name).is_file() for name in required_files):
            DATA_DIR = legacy_drive
    except ImportError:
        pass

if DATA_DIR is None:
    raise FileNotFoundError(
        "No se encontraron los CSV. Aceptá las reglas de Kaggle y ubicá los cinco archivos "
        "en data/raw, /content/favorita o una ruta indicada por FAVORITA_DATA_DIR. "
        "Google Drive es opcional. Ver docs/data_contract.md."
    )

print("Proyecto:", PROJECT_ROOT)
print("Datos:", DATA_DIR)
print("Datos encontrados correctamente.")
'''


REFERENCES_CELL = r'''# @title Referencias bibliográficas (APA 7)
from IPython.display import display, HTML

display(HTML(r"""
<div style="margin:24px 0;padding:22px 24px;background:#05070B;border:1px solid #1E293B;border-radius:14px;color:#CBD5E1;line-height:1.72;font-size:13px">
  <div style="font-size:11px;letter-spacing:1.8px;color:#C4B5FD;font-weight:850">REFERENCIAS BIBLIOGRÁFICAS · APA 7</div>
  <div style="font-size:22px;color:#F8FAFC;font-weight:850;margin:7px 0 14px">Fundamentos del experimento</div>
  <p>La validación rolling-origin sigue a Tashman (2000); LightGBM, CatBoost, Optuna y TSB se documentan con sus trabajos originales. RMSLE y WAPE son medidas predictivas y no equivalen a ahorro, nivel de servicio ni causalidad comercial (Hyndman &amp; Koehler, 2006).</p>
  <ol style="padding-left:22px">
    <li>Akiba, T., Sano, S., Yanase, T., Ohta, T., &amp; Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. <i>Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery &amp; Data Mining</i>, 2623–2631. <a style="color:#67E8F9" href="https://doi.org/10.1145/3292500.3330701">https://doi.org/10.1145/3292500.3330701</a></li>
    <li>Hyndman, R. J., &amp; Koehler, A. B. (2006). Another look at measures of forecast accuracy. <i>International Journal of Forecasting, 22</i>(4), 679–688. <a style="color:#67E8F9" href="https://doi.org/10.1016/j.ijforecast.2006.03.001">https://doi.org/10.1016/j.ijforecast.2006.03.001</a></li>
    <li>Kaggle. (2017). <i>Corporación Favorita Grocery Sales Forecasting</i> [Data set and competition]. <a style="color:#67E8F9" href="https://www.kaggle.com/c/favorita-grocery-sales-forecasting">Competition page</a>.</li>
    <li>Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., &amp; Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. <i>Advances in Neural Information Processing Systems, 30</i>. <a style="color:#67E8F9" href="https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html">Paper</a>.</li>
    <li>Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., &amp; Gulin, A. (2018). CatBoost: Unbiased boosting with categorical features. <i>Advances in Neural Information Processing Systems, 31</i>. <a style="color:#67E8F9" href="https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html">Paper</a>.</li>
    <li>Silver, E. A., Pyke, D. F., &amp; Thomas, D. J. (2017). <i>Inventory and production management in supply chains</i> (4th ed.). CRC Press. <a style="color:#67E8F9" href="https://www.routledge.com/Inventory-and-Production-Management-in-Supply-Chains/Silver-Pyke-Thomas/p/book/9781466558618">Publisher</a>.</li>
    <li>Tashman, L. J. (2000). Out-of-sample tests of forecasting accuracy: An analysis and review. <i>International Journal of Forecasting, 16</i>(4), 437–450. <a style="color:#67E8F9" href="https://doi.org/10.1016/S0169-2070(00)00065-0">https://doi.org/10.1016/S0169-2070(00)00065-0</a></li>
    <li>Teunter, R. H., Syntetos, A. A., &amp; Babai, M. Z. (2011). Intermittent demand: Linking forecasting to inventory obsolescence. <i>European Journal of Operational Research, 214</i>(3), 606–615. <a style="color:#67E8F9" href="https://doi.org/10.1016/j.ejor.2011.05.018">https://doi.org/10.1016/j.ejor.2011.05.018</a></li>
  </ol>
</div>
"""))
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="PF_Demanda_tp7.ipynb")
    parser.add_argument("--output", default="PF_Demanda_tp7_defensa_final.ipynb")
    args = parser.parse_args()

    source = ROOT / args.source
    output = ROOT / args.output
    if source.resolve() == output.resolve():
        raise ValueError("El archivo de salida debe ser distinto del TP7 original")

    notebook = nbformat.read(source, as_version=4)

    reproducibility = nbformat.v4.new_code_cell(
        source=REPRODUCIBILITY_CELL,
        metadata={"tags": ["documentation", "defense-entrypoint"]},
    )
    notebook.cells.insert(1, reproducibility)

    data_index = next(
        i
        for i, cell in enumerate(notebook.cells)
        if "Conectar Google Drive y localizar los CSV" in cell.source
    )
    notebook.cells[data_index].source = PORTABLE_DATA_CELL
    notebook.cells[data_index].metadata.setdefault("tags", []).extend(
        ["requires-favorita-data", "portable-data-path"]
    )

    expensive_markers = (
        "Ejecutar backtesting",
        "Permutation importance recursiva temporalmente válida",
        "Ejecutar test final",
        "Sensibilidad del pronóstico a promociones futuras",
    )
    for cell in notebook.cells:
        if any(marker in cell.source for marker in expensive_markers):
            tags = cell.metadata.setdefault("tags", [])
            if "expensive-experimental-reproduction" not in tags:
                tags.append("expensive-experimental-reproduction")

    references = nbformat.v4.new_code_cell(
        source=REFERENCES_CELL,
        metadata={"tags": ["documentation", "references"]},
    )
    final_links_index = next(
        i
        for i, cell in enumerate(notebook.cells)
        if "Código, resultados y reproducción del proyecto" in cell.source
    )
    notebook.cells.insert(final_links_index, references)

    notebook.metadata["bekantor_defense_audit"] = {
        "source": source.name,
        "source_preserved": True,
        "results_commit": "2d283ae2a0f7d2fed1d3b8b87e9110b81126befb",
        "recommended_mode": "results_reproduction_without_training",
    }
    nbformat.write(notebook, output)
    print(f"Copia documental creada: {output}")
    print(f"TP7 original preservado: {source}")


if __name__ == "__main__":
    main()
