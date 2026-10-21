"""Auditoría determinística del cierre académico, sin entrenamientos."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import nbformat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.inventory_closure import (
    EXPECTED_MODELS,
    V2_REQUIRED_FILES,
    v2_source_fingerprints,
)


V2 = ROOT / "outputs" / "forecasting_v2_full"
FINAL = ROOT / "outputs" / "forecasting_final"
NOTEBOOK = ROOT / "notebooks" / "07_project_closure.ipynb"
DOCUMENT = ROOT / "docs" / "forecasting_final_conclusions.md"


def main() -> None:
    required_final = {
        "inventory_stress_summary.csv",
        "inventory_stress_trajectory.csv",
        "inventory_stress_assumptions.json",
        "source_v2_sha256.json",
        "07_project_closure.html",
        "validation_timeline.png",
        "model_comparison.png",
        "promotion_availability.png",
        "inventory_stress_trajectory.png",
        "inventory_stress_costs.png",
        "tsb_segment_diagnostics.png",
    }
    missing = sorted(name for name in required_final if not (FINAL / name).is_file())
    empty = sorted(
        name for name in required_final
        if (FINAL / name).is_file() and (FINAL / name).stat().st_size == 0
    )
    if missing or empty:
        raise AssertionError(f"Artefactos faltantes={missing}; vacíos={empty}")

    recorded_hashes = json.loads(
        (FINAL / "source_v2_sha256.json").read_text(encoding="utf-8")
    )
    if len(recorded_hashes) != len(V2_REQUIRED_FILES) == 14:
        raise AssertionError("La manifestación V2 debe contener 14 artefactos")
    if recorded_hashes != v2_source_fingerprints(V2):
        raise AssertionError("Cambió al menos un artefacto fuente V2")

    summary = pd.read_csv(FINAL / "inventory_stress_summary.csv")
    trajectory = pd.read_csv(FINAL / "inventory_stress_trajectory.csv")
    assumptions = json.loads(
        (FINAL / "inventory_stress_assumptions.json").read_text(encoding="utf-8")
    )
    if tuple(summary["model"]) != EXPECTED_MODELS:
        raise AssertionError("Conjunto de modelos inesperado")
    common = [
        "initial_stock_assumption",
        "safety_stock",
        "service_level",
        "lead_time_days",
        "unit_cost",
        "holding_cost_per_unit",
        "shortage_penalty_per_unit",
        "simulated_demand",
    ]
    for column in common:
        if summary[column].nunique() != 1:
            raise AssertionError(f"Supuesto no común: {column}")
    if not summary["future_actual_used_for_decision"].eq(False).all():
        raise AssertionError("La decisión declara uso de información futura")
    if not summary["first_shortage_day"].eq(2).all():
        raise AssertionError("No se detectó el faltante esperado en día 2")
    if not trajectory.loc[trajectory["horizon_day"].ne(4), "receipt"].eq(0).all():
        raise AssertionError("Existe una recepción fuera del día 4")
    if not np.allclose(
        summary["simulated_total_cost"],
        summary["purchase_cost"] + summary["holding_cost"] + summary["shortage_cost"],
    ):
        raise AssertionError("Los costos no reconcilian")
    if assumptions["lost_sales_policy"].find("no se acumulan") < 0:
        raise AssertionError("La política de ventas perdidas no está documentada")
    if assumptions.get("case_selection_retrospective") is not True:
        raise AssertionError("No se declara la selección retrospectiva del caso")
    if assumptions.get("case_selection_basis") != "observed_validation_sales":
        raise AssertionError("La base de selección retrospectiva no coincide")
    if assumptions.get("future_actual_used_for_decision") is not False:
        raise AssertionError("La decisión declara uso de ventas futuras")

    notebook = nbformat.read(NOTEBOOK, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    if not code_cells or any(cell.execution_count is None for cell in code_cells):
        raise AssertionError("El notebook no está ejecutado de punta a punta")
    errors = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    if errors:
        raise AssertionError(f"El notebook contiene errores: {errors}")
    notebook_code = "\n".join("".join(cell.source) for cell in code_cells).lower()
    forbidden = [
        "optimize_lightgbm_recursive(",
        "fit_lightgbm(",
        "fit_catboost(",
        "select_tsb_parameters(",
        "run_forecasting_v2",
    ]
    used_forbidden = [token for token in forbidden if token in notebook_code]
    if used_forbidden:
        raise AssertionError(f"El notebook contiene entrenamiento: {used_forbidden}")

    document_text = DOCUMENT.read_text(encoding="utf-8").lower()
    notebook_markdown = "\n".join(
        "".join(cell.source)
        for cell in notebook.cells
        if cell.cell_type == "markdown"
    ).lower()
    required_disclosures = [
        "seleccionadas retrospectivamente",
        "ventas reales observadas durante la validación",
        "caso demostrativo",
        "selección operativa prospectiva",
        "ventas reales futuras se utilizan solamente para evaluar",
    ]
    for disclosure in required_disclosures:
        if disclosure not in document_text or disclosure not in notebook_markdown:
            raise AssertionError(f"Falta divulgación metodológica: {disclosure}")
    if "01/08 es límite exclusivo" not in notebook_code:
        raise AssertionError("La cronología no aclara el límite exclusivo de agosto")

    print("OK: 14 artefactos V2 conservados por SHA-256")
    print("OK: cinco modelos con supuestos comunes y decisión sin ventas futuras")
    print("OK: recepción sólo en día 4; faltante detectado en día 2")
    print("OK: ventas perdidas, stock final y costos reconciliados")
    print("OK: selección retrospectiva y decisiones prospectivas diferenciadas")
    print("OK: notebook ejecutado, sin errores ni llamadas de entrenamiento")
    print("OK: artefactos y gráficos finales presentes y no vacíos")


if __name__ == "__main__":
    main()
