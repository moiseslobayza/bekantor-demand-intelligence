"""Escenario de inventario de cierre construido sin reentrenar modelos.

El módulo reutiliza los pronósticos y la demanda comparativa almacenados en los
artefactos V2. La tienda 18 y la familia PRODUCE fueron seleccionadas
retrospectivamente en V2 según ventas observadas durante la validación. Una vez
fijado ese caso demostrativo, las ventas observadas se usan exclusivamente para
evaluar la simulación; no intervienen en el stock inicial, el stock de seguridad
ni la cantidad recomendada a comprar.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.inventory import simulate_inventory


EXPECTED_MODELS = (
    "Seasonal baseline",
    "LightGBM original",
    "LightGBM optimized",
    "CatBoost",
    "TSB",
)
V2_REQUIRED_FILES = (
    "best_lightgbm.json",
    "catboost_iterations.json",
    "fold_metrics.csv",
    "folds.csv",
    "inventory_example_summary.csv",
    "inventory_example_trajectory.csv",
    "inventory_sensitivity.csv",
    "model_comparison.csv",
    "optuna_study.log",
    "optuna_trials.csv",
    "promotion_scenarios.csv",
    "run_manifest.json",
    "segment_diagnostics.csv",
    "tsb_parameter_search.csv",
)


def sha256_file(path: Path) -> str:
    """Devuelve la huella SHA-256 de un archivo sin modificarlo."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def v2_source_fingerprints(source_dir: Path) -> dict[str, str]:
    """Calcula huellas de los 14 artefactos requeridos por la auditoría V2."""

    source_dir = Path(source_dir)
    missing = [name for name in V2_REQUIRED_FILES if not (source_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Faltan artefactos V2: {missing}")
    return {name: sha256_file(source_dir / name) for name in V2_REQUIRED_FILES}


def build_inventory_stress_scenario(
    source_dir: Path,
    *,
    initial_stock_fraction: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Construye el escenario de faltantes a partir de artefactos V2.

    El stock inicial es una fracción del pronóstico total del baseline
    estacional, disponible al momento de decisión. La demanda observada común
    sólo sirve como escenario de evaluación y no altera la reposición.
    """

    if not 0 < initial_stock_fraction < 1:
        raise ValueError("initial_stock_fraction debe estar entre 0 y 1")

    source_dir = Path(source_dir)
    original_summary = pd.read_csv(source_dir / "inventory_example_summary.csv")
    original_trajectory = pd.read_csv(source_dir / "inventory_example_trajectory.csv")
    folds = pd.read_csv(
        source_dir / "folds.csv",
        parse_dates=["train_end", "validation_start", "validation_end"],
    )

    if tuple(original_summary["model"]) != EXPECTED_MODELS:
        raise AssertionError("El orden o conjunto de modelos V2 cambió")

    reference_model = "Seasonal baseline"
    reference_summary = original_summary.loc[
        original_summary["model"].eq(reference_model)
    ].iloc[0]
    reference_trajectory = original_trajectory.loc[
        original_trajectory["model"].eq(reference_model)
    ].sort_values("horizon_day")

    common_actual = reference_trajectory["observed_sales"].to_numpy(dtype=float)
    common_safety_stock = float(reference_summary["safety_stock"])
    initial_stock = (
        float(reference_summary["forecast_demand"]) * float(initial_stock_fraction)
    )
    lead_time_days = int(reference_summary["lead_time_days"])
    receipt_day = lead_time_days + 1
    scenario_start = folds.loc[
        folds["fold"].eq(6), "validation_start"
    ].iloc[0]
    safety_source_end = pd.Timestamp(
        reference_summary["safety_stock_source_validation_end"]
    )
    if safety_source_end >= scenario_start:
        raise AssertionError("El stock de seguridad contiene información futura")

    summary_rows: list[dict] = []
    trajectory_rows: list[pd.DataFrame] = []
    for item in original_summary.itertuples(index=False):
        prior = original_trajectory.loc[
            original_trajectory["model"].eq(item.model)
        ].sort_values("horizon_day")
        model_actual = prior["observed_sales"].to_numpy(dtype=float)
        if not np.allclose(model_actual, common_actual):
            raise AssertionError("La demanda comparativa no coincide entre modelos")
        forecast = prior["forecast"].to_numpy(dtype=float)

        simulated_summary, simulated_trajectory = simulate_inventory(
            forecast,
            actual=common_actual,
            initial_stock=initial_stock,
            lead_time_days=lead_time_days,
            safety_stock=common_safety_stock,
            unit_cost=float(reference_summary["unit_cost"]),
            holding_cost_per_unit=float(
                reference_summary["holding_cost_per_unit"]
            ),
            shortage_penalty_per_unit=float(
                reference_summary["shortage_penalty_per_unit"]
            ),
        )
        shortage_days = simulated_trajectory.loc[
            simulated_trajectory["potential_shortage"].gt(0), "horizon_day"
        ]
        first_shortage_day = int(shortage_days.iloc[0]) if len(shortage_days) else None
        pre_receipt_shortage = float(
            simulated_trajectory.loc[
                simulated_trajectory["horizon_day"].lt(receipt_day),
                "potential_shortage",
            ].sum()
        )
        post_receipt_shortage = float(
            simulated_trajectory.loc[
                simulated_trajectory["horizon_day"].ge(receipt_day),
                "potential_shortage",
            ].sum()
        )
        summary_rows.append(
            {
                "store_nbr": int(item.store_nbr),
                "family": item.family,
                "model": item.model,
                "policy_reference_model": reference_model,
                "initial_stock_policy": "10_pct_reference_forecast",
                "initial_stock_fraction": float(initial_stock_fraction),
                "safety_stock_policy": "prior_external_fold_reference_errors",
                "demand_comparison_policy": "common_observed_sales_as_simulation_proxy",
                "future_actual_used_for_decision": False,
                "lost_sales_backordered": False,
                "service_level": float(reference_summary["service_level"]),
                "lead_time_days": lead_time_days,
                "receipt_day": receipt_day,
                "safety_stock_source_validation_end": safety_source_end.date().isoformat(),
                "first_shortage_day": first_shortage_day,
                "pre_receipt_potential_shortage": pre_receipt_shortage,
                "post_receipt_potential_shortage": post_receipt_shortage,
                "unrecovered_potential_lost_sales": float(
                    simulated_summary["potential_shortage"]
                ),
                **simulated_summary,
            }
        )

        simulated_trajectory.insert(0, "model", item.model)
        simulated_trajectory["stockout_flag"] = simulated_trajectory[
            "potential_shortage"
        ].gt(0)
        simulated_trajectory["cumulative_potential_shortage"] = (
            simulated_trajectory["potential_shortage"].cumsum()
        )
        trajectory_rows.append(simulated_trajectory)

    stress_summary = pd.DataFrame(summary_rows)
    stress_trajectory = pd.concat(trajectory_rows, ignore_index=True)
    assumptions = {
        "scenario": "stock_inicial_insuficiente",
        "store_nbr": int(reference_summary["store_nbr"]),
        "family": reference_summary["family"],
        "case_selection_retrospective": True,
        "case_selection_basis": "observed_validation_sales",
        "case_selection_role": (
            "Caso demostrativo conservado de V2; no es una selección operativa "
            "prospectiva."
        ),
        "future_actual_used_for_decision": False,
        "horizon_days": int(reference_trajectory["horizon_day"].nunique()),
        "lead_time_days": lead_time_days,
        "receipt_day": receipt_day,
        "service_level": float(reference_summary["service_level"]),
        "initial_stock_policy": "10_pct_reference_forecast",
        "initial_stock_fraction": float(initial_stock_fraction),
        "initial_stock_assumption": initial_stock,
        "reference_forecast_total": float(reference_summary["forecast_demand"]),
        "safety_stock_policy": "prior_external_fold_reference_errors",
        "safety_stock": common_safety_stock,
        "safety_stock_source_validation_end": safety_source_end.date().isoformat(),
        "scenario_validation_start": scenario_start.date().isoformat(),
        "unit_cost": float(reference_summary["unit_cost"]),
        "holding_cost_per_unit": float(
            reference_summary["holding_cost_per_unit"]
        ),
        "shortage_penalty_per_unit": float(
            reference_summary["shortage_penalty_per_unit"]
        ),
        "demand_role": (
            "Las ventas observadas comunes son un proxy de demanda para evaluar la "
            "simulación; no fijan stock inicial, seguridad ni reposición."
        ),
        "lost_sales_policy": (
            "Los faltantes son ventas potencialmente no atendidas y no se acumulan "
            "como pedidos pendientes ni se recuperan tras la recepción."
        ),
        "business_status": (
            "Todos los parámetros de inventario y costos son supuestos; los costos "
            "son simulados y no representan ahorro real."
        ),
    }
    return stress_summary, stress_trajectory, assumptions


def write_inventory_stress_artifacts(
    source_dir: Path,
    output_dir: Path,
    *,
    initial_stock_fraction: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Escribe los artefactos determinísticos del escenario de cierre."""

    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary, trajectory, assumptions = build_inventory_stress_scenario(
        source_dir,
        initial_stock_fraction=initial_stock_fraction,
    )
    summary.to_csv(output_dir / "inventory_stress_summary.csv", index=False)
    trajectory.to_csv(output_dir / "inventory_stress_trajectory.csv", index=False)
    (output_dir / "inventory_stress_assumptions.json").write_text(
        json.dumps(assumptions, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "source_v2_sha256.json").write_text(
        json.dumps(v2_source_fingerprints(source_dir), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary, trajectory, assumptions
