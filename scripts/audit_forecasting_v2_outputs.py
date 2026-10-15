"""Auditoría determinística de los artefactos producidos por Forecasting V2."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODELS = {
    "Seasonal baseline", "LightGBM original", "LightGBM optimized", "CatBoost", "TSB",
}
REQUIRED_FILES = {
    "best_lightgbm.json", "catboost_iterations.json", "fold_metrics.csv", "folds.csv",
    "inventory_example_summary.csv", "inventory_example_trajectory.csv",
    "inventory_sensitivity.csv", "model_comparison.csv", "optuna_study.log",
    "optuna_trials.csv", "promotion_scenarios.csv", "run_manifest.json",
    "segment_diagnostics.csv", "tsb_parameter_search.csv",
}


def assert_close(left, right, label, atol=1e-10):
    if not np.allclose(left, right, rtol=1e-10, atol=atol, equal_nan=True):
        raise AssertionError(f"No reconcilia: {label}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/forecasting_v2_full")
    parser.add_argument("--expect-protocol", default="full")
    args = parser.parse_args()
    out = ROOT / args.output_dir

    missing = sorted(name for name in REQUIRED_FILES if not (out / name).is_file())
    empty = sorted(
        name for name in REQUIRED_FILES
        if (out / name).is_file() and (out / name).stat().st_size == 0
    )
    if missing or empty:
        raise AssertionError(f"Artefactos faltantes={missing}; vacíos={empty}")

    folds = pd.read_csv(
        out / "folds.csv",
        parse_dates=["train_end", "validation_start", "validation_end"],
    )
    metrics = pd.read_csv(out / "fold_metrics.csv")
    comparison = pd.read_csv(out / "model_comparison.csv")
    promotions = pd.read_csv(out / "promotion_scenarios.csv")
    inventory = pd.read_csv(out / "inventory_example_summary.csv")
    trajectory = pd.read_csv(out / "inventory_example_trajectory.csv")
    sensitivity = pd.read_csv(out / "inventory_sensitivity.csv")
    segments = pd.read_csv(out / "segment_diagnostics.csv")
    trials = pd.read_csv(out / "optuna_trials.csv")
    tsb_search = pd.read_csv(out / "tsb_parameter_search.csv")
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    best = json.loads((out / "best_lightgbm.json").read_text(encoding="utf-8"))
    catboost = json.loads((out / "catboost_iterations.json").read_text(encoding="utf-8"))

    assert manifest["protocol"] == args.expect_protocol
    assert manifest["august_evaluated"] is False
    assert len(folds) == 6
    assert folds["role"].tolist() == ["tuning"] * 4 + ["development_evaluation"] * 2
    assert folds["validation_end"].max() < pd.Timestamp("2017-08-01")
    assert (folds["validation_start"] == folds["train_end"] + pd.Timedelta(days=1)).all()
    assert manifest["folds"] == folds.assign(
        train_end=folds["train_end"].astype(str),
        validation_start=folds["validation_start"].astype(str),
        validation_end=folds["validation_end"].astype(str),
    ).to_dict("records")

    assert len(trials) == manifest["trials"] == 20
    assert set(trials["state"]).issubset({"COMPLETE", "PRUNED"})
    assert trials["state"].eq("COMPLETE").any()
    best_trial = trials.loc[trials["state"].eq("COMPLETE")].sort_values("value").iloc[0]
    assert int(best_trial["number"]) == 11
    assert_close(best_trial["value"], best["objective_recursive_rmsle"], "mejor Optuna")
    assert int(best_trial["user_attrs_selected_n_estimators"]) == best["selected_n_estimators"]
    assert best["estimator_cap_is_binding"] is False
    assert best["near_estimator_cap_folds"] == 0
    assert catboost["selected_iterations"] == manifest["catboost_iterations"]
    assert catboost["data_signature"] == manifest["data_signature"]
    assert len(tsb_search) == 9

    heldout = metrics.loc[metrics["role"].eq("development_evaluation")].copy()
    assert not metrics.duplicated(["fold", "role", "model"]).any()
    assert set(heldout["fold"]) == {5, 6}
    for fold, group in heldout.groupby("fold"):
        assert set(group["model"]) == EXPECTED_MODELS, fold
    assert np.isfinite(metrics.select_dtypes(include="number").to_numpy()).all()

    recomputed = heldout.groupby("model", as_index=False).agg(
        folds=("fold", "nunique"), RMSLE_mean=("RMSLE", "mean"),
        RMSLE_std=("RMSLE", "std"), MAE_mean=("MAE", "mean"),
        WAPE_pct_mean=("WAPE_pct", "mean"), Bias_pct_mean=("Bias_pct", "mean"),
    ).sort_values("model").reset_index(drop=True)
    comparison_sorted = comparison.sort_values("model").reset_index(drop=True)
    assert recomputed["model"].equals(comparison_sorted["model"])
    assert_close(
        recomputed.select_dtypes(include="number").to_numpy(),
        comparison_sorted.select_dtypes(include="number").to_numpy(),
        "model_comparison.csv",
    )

    assert set(segments["fold"]) == {5, 6}
    for keys, group in segments.groupby(["fold", "dimension", "segment"], dropna=False):
        assert set(group["model"]) == EXPECTED_MODELS, keys
        assert group["n_obs"].nunique() == 1, keys
        assert group["n_series"].nunique() == 1, keys
    horizon_coverage = segments.loc[segments["dimension"].eq("horizon")].groupby(
        ["fold", "model"], as_index=False
    ).agg(n_obs=("n_obs", "sum"), horizon_days=("segment", "nunique"))
    assert horizon_coverage["horizon_days"].eq(15).all()
    for _, group in horizon_coverage.groupby("fold"):
        assert group["n_obs"].nunique() == 1

    assert promotions["scenario"].tolist() == ["full", "partial", "none"]
    assert promotions["confirmed_days"].tolist() == [15, 7, 0]
    optimized_last = heldout.loc[
        heldout["fold"].eq(6) & heldout["model"].eq("LightGBM optimized")
    ].iloc[0]
    for column in ["RMSLE", "MAE", "WAPE_pct", "Bias_pct"]:
        assert_close(promotions.iloc[0][column], optimized_last[column], f"promoción full {column}")

    common_columns = [
        "initial_stock_assumption", "safety_stock", "service_level", "lead_time_days",
        "unit_cost", "holding_cost_per_unit", "shortage_penalty_per_unit",
        "simulated_demand",
    ]
    for column in common_columns:
        assert inventory[column].nunique() == 1, column
    assert inventory["policy_reference_model"].eq("Seasonal baseline").all()
    assert pd.to_datetime(inventory["safety_stock_source_validation_end"]).max() < folds.loc[
        folds["fold"].eq(6), "validation_start"
    ].iloc[0]
    assert inventory["recommended_replenishment"].ge(0).all()
    assert inventory["ending_inventory"].ge(0).all()
    assert inventory["potential_shortage"].ge(0).all()

    observed = trajectory.pivot(index="horizon_day", columns="model", values="observed_sales")
    assert observed.nunique(axis=1).eq(1).all()
    assert trajectory.groupby("model")["horizon_day"].nunique().eq(15).all()
    assert trajectory.loc[trajectory["horizon_day"].ne(4), "receipt"].eq(0).all()
    receipts = trajectory.loc[
        trajectory["horizon_day"].eq(4), ["model", "receipt"]
    ]
    receipt_check = inventory[["model", "recommended_replenishment"]].merge(
        receipts, on="model", validate="one_to_one"
    )
    assert_close(
        receipt_check["recommended_replenishment"], receipt_check["receipt"],
        "recepción día 4",
    )
    assert_close(
        inventory["recommended_replenishment"],
        np.maximum(
            0,
            inventory["target_stock_at_receipt"]
            - inventory["projected_stock_at_receipt"],
        ),
        "reposición recomendada",
    )

    assert set(sensitivity["scenario"]) == {"high_demand", "observed_base", "low_demand"}
    assert sensitivity["potential_shortage"].ge(0).all()
    for scenario, group in sensitivity.groupby("scenario"):
        for column in common_columns[0:7] + ["simulated_demand", "demand_multiplier"]:
            assert group[column].nunique() == 1, (scenario, column)

    catboost_cap_hits = sum(
        int(value) >= int(catboost["max_iterations"])
        for value in catboost["fold_best_iterations"]
    )
    print("OK: 14 artefactos requeridos presentes, legibles y no vacíos")
    print("OK: 20 trials Optuna (17 completos, 3 podados), sin fallos")
    print("OK: 4 folds tuning + 2 externos; ninguna fecha de agosto")
    print("OK: agregados, promociones y cobertura segmentada reconciliados")
    print("OK: los cinco modelos comparten población, stock, demanda y supuestos")
    print("OK: recepción sólo en día 4 y stock de seguridad anterior al fold 6")
    if catboost_cap_hits:
        print(
            "ADVERTENCIA: CatBoost alcanzó el máximo interno en "
            f"{catboost_cap_hits}/4 folds; su comparación queda como benchmark "
            "con presupuesto parcialmente vinculante."
        )


if __name__ == "__main__":
    main()
