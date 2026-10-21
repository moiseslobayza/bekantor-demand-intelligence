"""Auditoría del cierre académico sin ejecutar entrenamientos."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.inventory import simulate_inventory
from src.models.inventory_closure import (
    EXPECTED_MODELS,
    V2_REQUIRED_FILES,
    build_inventory_stress_scenario,
    v2_source_fingerprints,
    write_inventory_stress_artifacts,
)


ROOT = Path(__file__).resolve().parents[1]
V2_DIR = ROOT / "outputs" / "forecasting_v2_full"

EXPECTED_V2_HASHES = {
    "best_lightgbm.json": "7ce4278e91e9b71357f9132847dcfd4ca4c9b9a58ee93fd1d317af09c75f75f2",
    "catboost_iterations.json": "f2d065a2abea6fc439ffdc0d9fdc6dc89ef2669884f895a01a97b9ff44912d2d",
    "fold_metrics.csv": "c7ef5e390291b727bab797370ba3e0181a7688358a8fcac49bc1361a4464056e",
    "folds.csv": "205f4a951a9e584f7e130ec51825598b79e58e2525818fa450370c4b0a99028c",
    "inventory_example_summary.csv": "5ef727498e031e3d3980ab46ee2c2c51b4b37ebd9f484243dd22a03a7c345ceb",
    "inventory_example_trajectory.csv": "9c73352382b4100f3baf9e8b22c32244ef8823ee73488f554d291501659b21d7",
    "inventory_sensitivity.csv": "197c48d181caba5036b5bb83f1e9e34216f4e86c7e59c5a095b0d8538aea93b7",
    "model_comparison.csv": "4480729a91588a859a29275c56863c0336714e46255471bebc3bdf42d6368adb",
    "optuna_study.log": "f982818153994c2788062456c7c504afcf4587f3d82f9c9e02b215699675231c",
    "optuna_trials.csv": "554a0f1bfb16881155f250995ee3c5f0882fe21e5ddde3293670f68fbb059f45",
    "promotion_scenarios.csv": "f455548303b07974cf6065b7098daea658f1e99aea172f2116048e09f421803e",
    "run_manifest.json": "659c8ca39eba2b99547ad23b6621d0750365866b0ba1a884e2303164d5cce919",
    "segment_diagnostics.csv": "b0f5185924b945e577ee3712c563bb55f954438724dac172dd7362839e905281",
    "tsb_parameter_search.csv": "8dcc06cec0d4a3b8d4060ee7b02838604b61d94603667ecf6ba00e9237cf3dd1",
}

EXPECTED_PROTECTED_NOTEBOOK_HASHES = {
    "04_eda.ipynb": "0ad140e3b114c6df9edb1a93b1e27a99aad2e695fee1c327919d89aea24a5758",
    "05_modeling.ipynb": "81db4f53e5cde35424f33329d3bc730308c145f11811bb3b297ef326c416cc35",
    "06_forecasting_v2.ipynb": "94d619008d398f6d41baf33b3f9c4a48a00d9f6624eee8a91f2b6b1a4aef8673",
}


@pytest.fixture(scope="module")
def stress_result():
    return build_inventory_stress_scenario(V2_DIR)


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v2_artifacts_and_approved_notebooks_remain_byte_identical():
    assert len(V2_REQUIRED_FILES) == 14
    assert v2_source_fingerprints(V2_DIR) == EXPECTED_V2_HASHES
    actual_notebooks = {
        name: _file_hash(ROOT / "notebooks" / name)
        for name in EXPECTED_PROTECTED_NOTEBOOK_HASHES
    }
    assert actual_notebooks == EXPECTED_PROTECTED_NOTEBOOK_HASHES


def test_stress_assumptions_are_common_and_cutoff_safe(stress_result):
    summary, trajectory, assumptions = stress_result
    assert tuple(summary["model"]) == EXPECTED_MODELS
    common_columns = [
        "initial_stock_assumption",
        "safety_stock",
        "service_level",
        "lead_time_days",
        "unit_cost",
        "holding_cost_per_unit",
        "shortage_penalty_per_unit",
        "simulated_demand",
    ]
    assert all(summary[column].nunique() == 1 for column in common_columns)
    assert summary["future_actual_used_for_decision"].eq(False).all()
    assert summary["lost_sales_backordered"].eq(False).all()
    assert assumptions["case_selection_retrospective"] is True
    assert assumptions["case_selection_basis"] == "observed_validation_sales"
    assert assumptions["future_actual_used_for_decision"] is False
    assert assumptions["initial_stock_policy"] == "10_pct_reference_forecast"
    assert assumptions["initial_stock_assumption"] == pytest.approx(
        assumptions["reference_forecast_total"] * 0.10
    )
    assert pd.Timestamp(assumptions["safety_stock_source_validation_end"]) < pd.Timestamp(
        assumptions["scenario_validation_start"]
    )
    observed_by_day = trajectory.pivot(
        index="horizon_day", columns="model", values="observed_sales"
    )
    assert observed_by_day.nunique(axis=1).eq(1).all()


def test_retrospective_case_selection_is_documented_consistently(stress_result):
    _, _, assumptions = stress_result
    original = pd.read_csv(V2_DIR / "inventory_example_summary.csv")
    assert original["store_nbr"].eq(assumptions["store_nbr"]).all()
    assert original["family"].eq(assumptions["family"]).all()

    document = (ROOT / "docs" / "forecasting_final_conclusions.md").read_text(
        encoding="utf-8"
    ).lower()
    notebook = json.loads(
        (ROOT / "notebooks" / "07_project_closure.ipynb").read_text(
            encoding="utf-8"
        )
    )
    notebook_markdown = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    ).lower()
    generator = (ROOT / "scripts" / "build_project_closure_notebook.py").read_text(
        encoding="utf-8"
    ).lower()
    required_disclosures = [
        "seleccionadas retrospectivamente",
        "ventas reales observadas durante la validación",
        "caso demostrativo",
        "selección operativa prospectiva",
        "ventas reales futuras se utilizan solamente para evaluar",
    ]
    for disclosure in required_disclosures:
        assert disclosure in document
        assert disclosure in notebook_markdown
        assert disclosure in generator


def test_validation_timeline_marks_august_as_exclusive_boundary():
    folds = pd.read_csv(V2_DIR / "folds.csv", parse_dates=["validation_end"])
    assert folds["validation_end"].max() == pd.Timestamp("2017-07-31")
    notebook = json.loads(
        (ROOT / "notebooks" / "07_project_closure.ipynb").read_text(
            encoding="utf-8"
        )
    )
    notebook_source = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"]
    )
    assert "01/08 es límite exclusivo" in notebook_source


def test_receipt_occurs_only_on_day_four_and_stockout_is_detected(stress_result):
    summary, trajectory, _ = stress_result
    assert trajectory.loc[trajectory["horizon_day"].ne(4), "receipt"].eq(0).all()
    receipts = trajectory.loc[
        trajectory["horizon_day"].eq(4), ["model", "receipt"]
    ]
    receipt_check = summary[["model", "recommended_replenishment"]].merge(
        receipts, on="model", validate="one_to_one"
    )
    np.testing.assert_allclose(
        receipt_check["recommended_replenishment"], receipt_check["receipt"]
    )
    assert summary["first_shortage_day"].eq(2).all()
    assert summary["pre_receipt_potential_shortage"].gt(0).all()
    assert summary["post_receipt_potential_shortage"].eq(0).all()


def test_lost_sales_are_not_recovered_and_final_stock_reconciles(stress_result):
    summary, trajectory, _ = stress_result
    assert summary["potential_shortage"].nunique() == 1
    for item in summary.itertuples(index=False):
        model_rows = trajectory.loc[trajectory["model"].eq(item.model)]
        assert item.unrecovered_potential_lost_sales == pytest.approx(
            model_rows["potential_shortage"].sum()
        )
        assert item.ending_inventory == pytest.approx(
            item.initial_stock_assumption
            + model_rows["receipt"].sum()
            - model_rows["served"].sum()
        )


def test_purchase_decision_is_invariant_to_future_actuals():
    forecast = [10, 12, 14, 16, 18]
    assumptions = dict(
        initial_stock=5,
        lead_time_days=3,
        safety_stock=2,
        unit_cost=1,
        holding_cost_per_unit=0.02,
        shortage_penalty_per_unit=2,
    )
    low, _ = simulate_inventory(forecast, actual=[1, 1, 1, 1, 1], **assumptions)
    high, _ = simulate_inventory(forecast, actual=[50, 50, 50, 50, 50], **assumptions)
    assert low["recommended_replenishment"] == pytest.approx(
        high["recommended_replenishment"]
    )


def test_costs_and_model_differences_reconcile(stress_result):
    summary, _, _ = stress_result
    np.testing.assert_allclose(
        summary["simulated_total_cost"],
        summary["purchase_cost"]
        + summary["holding_cost"]
        + summary["shortage_cost"],
    )
    assert summary["recommended_replenishment"].nunique() == len(EXPECTED_MODELS)
    assert summary["ending_inventory"].nunique() == len(EXPECTED_MODELS)
    assert summary["simulated_total_cost"].nunique() == len(EXPECTED_MODELS)


def test_new_artifacts_are_reproducible(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_inventory_stress_artifacts(V2_DIR, first)
    write_inventory_stress_artifacts(V2_DIR, second)
    expected_files = {
        "inventory_stress_summary.csv",
        "inventory_stress_trajectory.csv",
        "inventory_stress_assumptions.json",
        "source_v2_sha256.json",
    }
    assert {path.name for path in first.iterdir()} == expected_files
    assert {path.name for path in second.iterdir()} == expected_files
    for name in expected_files:
        assert (first / name).read_bytes() == (second / name).read_bytes()
    manifest = json.loads((first / "source_v2_sha256.json").read_text())
    assert manifest == EXPECTED_V2_HASHES
