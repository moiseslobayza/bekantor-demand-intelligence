"""Regenera inventario del caso ya ejecutado sin reentrenar modelos."""

from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.inventory import inventory_sensitivity, simulate_inventory


parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", default="outputs/forecasting_v2_full")
args = parser.parse_args()

OUT = ROOT / args.output_dir
summary = pd.read_csv(OUT / "inventory_example_summary.csv")
trajectory = pd.read_csv(OUT / "inventory_example_trajectory.csv")
reference_model = "Seasonal baseline"
reference_summary = summary.loc[summary["model"].eq(reference_model)].iloc[0]
reference_trajectory = trajectory.loc[
    trajectory["model"].eq(reference_model)
].sort_values("horizon_day")
common_initial_stock = float(reference_summary["initial_stock_assumption"])
common_safety_stock = float(reference_summary["safety_stock"])
common_actual = reference_trajectory["observed_sales"].to_numpy()
summary_rows = []
trajectory_rows = []
sensitivity_rows = []
for item in summary.itertuples(index=False):
    prior = trajectory.loc[trajectory["model"].eq(item.model)].sort_values("horizon_day")
    forecast = prior["forecast"].to_numpy()
    if not np.allclose(prior["observed_sales"].to_numpy(), common_actual):
        raise AssertionError("La demanda observada no coincide entre modelos")
    revised_summary, revised_trajectory = simulate_inventory(
        forecast, actual=common_actual,
        initial_stock=common_initial_stock,
        lead_time_days=3, safety_stock=common_safety_stock,
        unit_cost=1.0, holding_cost_per_unit=0.02,
        shortage_penalty_per_unit=2.0,
    )
    summary_rows.append({
        "store_nbr": item.store_nbr,
        "family": item.family,
        "model": item.model,
        "policy_reference_model": reference_model,
        "initial_stock_policy": "25_pct_reference_forecast",
        "safety_stock_policy": "prior_external_fold_reference_errors",
        "demand_comparison_policy": "common_observed_sales",
        "service_level": 0.95,
        "lead_time_days": 3,
        "safety_stock_source_validation_end": "2017-07-16",
        "safety_stock": common_safety_stock,
        "initial_stock_assumption": common_initial_stock,
        **revised_summary,
    })
    revised_trajectory.insert(0, "model", item.model)
    trajectory_rows.append(revised_trajectory)
    sensitivity = inventory_sensitivity(
        forecast, error_pct=0.15,
        base_demand=common_actual,
        initial_stock=common_initial_stock,
        lead_time_days=3, safety_stock=common_safety_stock,
        unit_cost=1.0, holding_cost_per_unit=0.02,
        shortage_penalty_per_unit=2.0,
    )
    sensitivity.insert(0, "model", item.model)
    sensitivity.insert(1, "service_level", 0.95)
    sensitivity_rows.append(sensitivity)

pd.DataFrame(summary_rows).to_csv(OUT / "inventory_example_summary.csv", index=False)
pd.concat(trajectory_rows, ignore_index=True).to_csv(
    OUT / "inventory_example_trajectory.csv", index=False
)
pd.concat(sensitivity_rows, ignore_index=True).to_csv(
    OUT / "inventory_sensitivity.csv", index=False
)
print(f"Inventario regenerado en: {OUT}")
