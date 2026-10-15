"""Render HTML local del notebook V2 para inspección visual."""

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "outputs" / "jupyter_runtime_full"
RUNTIME.mkdir(parents=True, exist_ok=True)
os.environ["JUPYTER_CONFIG_DIR"] = str(RUNTIME / "config")
os.environ["JUPYTER_DATA_DIR"] = str(RUNTIME / "data")
os.environ["MPLCONFIGDIR"] = str(RUNTIME / "matplotlib")
os.environ["MPLBACKEND"] = "Agg"

import nbformat
from nbconvert import HTMLExporter


source = ROOT / "notebooks" / "06_forecasting_v2.ipynb"
target = ROOT / "outputs" / "forecasting_v2_full" / "06_forecasting_v2.html"
notebook = nbformat.read(source, as_version=4)
exporter = HTMLExporter(template_name="lab")
html, _ = exporter.from_notebook_node(notebook)
target.write_text(html, encoding="utf-8")
print(f"HTML renderizado: {target}")
