"""Ejecuta 05_modeling.ipynb desde un kernel limpio y guarda sus salidas."""

from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "05_modeling.ipynb"


notebook = nbformat.read(NOTEBOOK, as_version=4)
client = NotebookClient(
    notebook,
    timeout=1800,
    kernel_name="python3",
    resources={"metadata": {"path": str(ROOT)}},
)
client.execute()
nbformat.write(notebook, NOTEBOOK)
print(f"Notebook ejecutado y guardado: {NOTEBOOK}")
