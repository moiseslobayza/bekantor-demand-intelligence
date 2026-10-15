"""Ejecuta 06_forecasting_v2.ipynb desde un kernel limpio y guarda salidas."""

import os
import asyncio
import sys
from pathlib import Path

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "06_forecasting_v2.ipynb"
JUPYTER_RUNTIME = ROOT / "outputs" / "jupyter_runtime_full"
CONFIG_DIR = JUPYTER_RUNTIME / "config"
DATA_DIR = JUPYTER_RUNTIME / "data"
TEMP_DIR = JUPYTER_RUNTIME / "temp"
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ["JUPYTER_CONFIG_DIR"] = str(CONFIG_DIR)
os.environ["JUPYTER_DATA_DIR"] = str(DATA_DIR)
os.environ["JUPYTER_PATH"] = str(JUPYTER_RUNTIME / "share" / "jupyter")
os.environ["TMP"] = str(TEMP_DIR)
os.environ["TEMP"] = str(TEMP_DIR)
os.environ["MPLCONFIGDIR"] = str(JUPYTER_RUNTIME / "matplotlib")

import nbformat
from ipykernel.kernelspec import install
from nbclient import NotebookClient

KERNEL_NAME = "bekantor-v2-runtime"
kernel_spec = JUPYTER_RUNTIME / "share" / "jupyter" / "kernels" / KERNEL_NAME / "kernel.json"
if not kernel_spec.is_file():
    print(f"Instalando kernel local en {JUPYTER_RUNTIME}", flush=True)
    install(
        kernel_name=KERNEL_NAME,
        display_name="BEKANTOR V2",
        prefix=str(JUPYTER_RUNTIME),
    )

notebook = nbformat.read(NOTEBOOK, as_version=4)
print(f"Ejecutando {len(notebook.cells)} celdas con {KERNEL_NAME}", flush=True)


def report_cell(*, cell, cell_index, **_kwargs):
    if cell.cell_type == "code":
        print(f"  celda de código {cell_index}", flush=True)


client = NotebookClient(
    notebook,
    timeout=1800,
    kernel_name=KERNEL_NAME,
    resources={"metadata": {"path": str(ROOT)}},
    on_cell_start=report_cell,
)
client.execute()
nbformat.write(notebook, NOTEBOOK)
print(f"Notebook ejecutado y guardado: {NOTEBOOK}")
