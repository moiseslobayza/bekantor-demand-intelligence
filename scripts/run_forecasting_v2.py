"""Ejecuta Forecasting V2 y guarda artefactos reproducibles.

El modo ``--quick`` conserva universo, horizonte y temporalidad; reduce el
presupuesto de búsqueda y árboles. No ejecuta agosto.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.clean_data import minimal_cleaning
from src.data.load_data import load_raw_data
from src.features.demand_classification import build_demand_profile
from src.features.model_features import CATEGORICAL_FEATURES, MODEL_FEATURES
from src.features.promotion_scenarios import make_promotion_transform
from src.models.baseline import seasonal_weekday_baseline
from src.models.catboost_model import select_catboost_iterations
from src.models.categorical import fit_category_schema
from src.models.experiment import prediction_metrics
from src.models.inventory import inventory_sensitivity, safety_stock_from_errors, simulate_inventory
from src.models.lightgbm_model import fit_fixed_iterations
from src.models.modeling_pipeline import prepare_cutoff_dataset
from src.models.optuna_search import optimize_lightgbm_recursive, save_study_results
from src.models.recursive_forecast import recursive_forecast
from src.models.tsb import select_tsb_parameters, tsb_forecast
from src.models.v2_evaluation import evaluate_v2_fold, validate_prediction_coverage
from src.models.validation import v2_development_windows


KEYS = ["date", "store_nbr", "family"]
PROTOCOL_VERSION = "forecasting-v2.1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir", default="outputs/forecasting_v2")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--max-estimators", type=int, default=1500)
    parser.add_argument(
        "--catboost-iterations", type=int,
        help="Fija iteraciones CatBoost; si se omite se calibran sólo en tuning.",
    )
    parser.add_argument("--catboost-max-iterations", type=int, default=1000)
    parser.add_argument(
        "--study-storage",
        help="Archivo journal de Optuna; por defecto vive en output-dir.",
    )
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def prepare(datasets, row):
    return prepare_cutoff_dataset(
        datasets["sales"], datasets["stores"], datasets["holidays"],
        row.train_end, row.validation_start, row.validation_end,
    )


def dataset_signature(datasets):
    """Firma el contenido que afecta features para no mezclar estudios."""

    digest = hashlib.sha256(PROTOCOL_VERSION.encode("utf-8"))
    selected_columns = {
        "sales": ["date", "store_nbr", "family", "sales", "onpromotion"],
        "stores": list(datasets["stores"].columns),
        "holidays": list(datasets["holidays"].columns),
    }
    for name in ("sales", "stores", "holidays"):
        frame = datasets[name][selected_columns[name]]
        if "date" in frame.columns:
            frame = frame.loc[frame["date"].lt("2017-08-01")]
        digest.update(name.encode("utf-8"))
        digest.update(str(frame.shape).encode("utf-8"))
        digest.update(
            pd.util.hash_pandas_object(frame, index=False, categorize=True)
            .to_numpy(dtype="uint64", copy=False)
            .tobytes()
        )
    return digest.hexdigest()[:16]


def baseline_tsb_rows(prepared, fold, role, tsb_params):
    truth = prepared["validation"][[*KEYS, "sales"]]
    baseline = seasonal_weekday_baseline(prepared["train"], prepared["validation"])
    tsb = tsb_forecast(prepared["train"], prepared["validation"], *tsb_params)
    rows = []
    for model, frame, column in [
        ("Seasonal baseline", baseline, "pred_baseline"),
        ("TSB", tsb, "prediction_tsb"),
    ]:
        validate_prediction_coverage(truth, frame)
        scored = truth.merge(frame[[*KEYS, column]], on=KEYS, validate="one_to_one")
        rows.append({"fold": fold, "role": role, "model": model, **prediction_metrics(scored["sales"], scored[column])})
    return rows


def promotion_experiment(
    prepared,
    params,
    n_estimators,
    confirmed_days=7,
    fitted_model=None,
    category_schema=None,
    full_prediction=None,
):
    train_features = prepared["train_features"]
    model, schema = fitted_model, category_schema
    if model is None or schema is None:
        X_train, schema = fit_category_schema(train_features[MODEL_FEATURES], CATEGORICAL_FEATURES)
        model = fit_fixed_iterations(
            X_train, train_features["sales"], CATEGORICAL_FEATURES,
            n_estimators=n_estimators, params=params,
        )
    truth = prepared["validation"][[*KEYS, "sales"]]
    start, end = truth["date"].min(), truth["date"].max()
    rows, predictions = [], {}
    for scenario in ["full", "partial", "none"]:
        transform = make_promotion_transform(
            scenario, prepared["train"], prepared["train"]["date"].max(),
            confirmed_days=confirmed_days,
        )
        if scenario == "full" and full_prediction is not None:
            pred = full_prediction[[*KEYS, "prediction_model"]].rename(
                columns={"prediction_model": "prediction_full"}
            )
        else:
            pred = recursive_forecast(
                model, prepared["features_base"], prepared["train"], start, end,
                MODEL_FEATURES, schema, feature_transform=transform,
            ).rename(columns={"prediction": f"prediction_{scenario}"})
        validate_prediction_coverage(truth, pred)
        scored = truth.merge(pred, on=KEYS, validate="one_to_one")
        column = f"prediction_{scenario}"
        rows.append({
            "scenario": scenario,
            "confirmed_days": confirmed_days if scenario == "partial" else (15 if scenario == "full" else 0),
            "forecast_total": float(scored[column].sum()),
            **prediction_metrics(scored["sales"], scored[column]),
        })
        predictions[scenario] = scored
    return pd.DataFrame(rows), predictions


def segment_outputs(prepared_by_fold, scored_by_fold, output_dir):
    tables = []
    profiles = {}
    for fold in sorted(scored_by_fold):
        prepared = prepared_by_fold[fold]
        profile = build_demand_profile(prepared["train"])
        profiles[fold] = profile
        known = prepared["features_base"][[*KEYS, "is_holiday", "onpromotion"]]
        for model, scored in scored_by_fold[fold].items():
            pred_col = next(c for c in scored.columns if c.startswith("pred") and c != "prediction") if "prediction" not in scored else "prediction"
            if "prediction_model" in scored:
                pred_col = "prediction_model"
            if "prediction_tsb" in scored:
                pred_col = "prediction_tsb"
            enriched = scored.merge(profile[["store_nbr", "family", "tipo_demanda"]], on=["store_nbr", "family"], validate="many_to_one")
            enriched = enriched.merge(known, on=KEYS, validate="one_to_one")
            enriched["horizon_day"] = (enriched["date"] - enriched["date"].min()).dt.days + 1
            enriched["day_type"] = np.where(enriched["is_holiday"].gt(0), "holiday", "ordinary")
            enriched["promotion_level"] = pd.cut(
                enriched["onpromotion"], [-np.inf, 0, 5, np.inf],
                labels=["none", "low_1_5", "high_6_plus"],
            )
            for columns, label in [
                (["family"], "family"), (["store_nbr"], "store"),
                (["tipo_demanda"], "demand_type"), (["horizon_day"], "horizon"),
                (["day_type"], "day_type"), (["promotion_level"], "promotion_level"),
            ]:
                for keys, group in enriched.groupby(columns, observed=True):
                    keys = keys if isinstance(keys, tuple) else (keys,)
                    row = {
                        "fold": int(fold), "model": model, "dimension": label,
                        "segment": " | ".join(map(str, keys)),
                    }
                    row.update(prediction_metrics(group["sales"], group[pred_col]))
                    row.update({"n_obs": len(group), "n_series": group[["store_nbr", "family"]].drop_duplicates().shape[0]})
                    tables.append(row)
    result = pd.DataFrame(tables)
    result.to_csv(output_dir / "segment_diagnostics.csv", index=False)
    return result, profiles


def inventory_example(last_prepared, last_scored, previous_scored, profile, output_dir):
    truth = last_prepared["validation"][[*KEYS, "sales"]].merge(
        profile[["store_nbr", "family", "tipo_demanda"]],
        on=["store_nbr", "family"], validate="many_to_one",
    )
    focus = truth.loc[
        truth["tipo_demanda"].isin(["Intermitente", "Lumpy"])
        | truth["family"].isin(["LINGERIE", "GROCERY II"])
    ]
    volumes = focus.groupby(["store_nbr", "family"], as_index=False)["sales"].sum()
    if volumes.empty:
        volumes = truth.groupby(["store_nbr", "family"], as_index=False)["sales"].sum()
    selected = volumes.sort_values(["sales", "store_nbr", "family"], ascending=[False, True, True]).iloc[0]
    store, family = int(selected["store_nbr"]), selected["family"]
    reference_model = "Seasonal baseline"
    reference = last_scored[reference_model].loc[
        (last_scored[reference_model]["store_nbr"] == store)
        & (last_scored[reference_model]["family"] == family)
    ].sort_values("date")
    reference_previous = previous_scored[reference_model].loc[
        (previous_scored[reference_model]["store_nbr"] == store)
        & (previous_scored[reference_model]["family"] == family)
    ].sort_values("date")
    common_actual = reference["sales"].to_numpy()
    common_initial_stock = float(reference["pred_baseline"].sum() * 0.25)
    common_safety_stock = safety_stock_from_errors(
        reference_previous["pred_baseline"].to_numpy()
        - reference_previous["sales"].to_numpy(),
        lead_time_days=3,
        service_level=0.95,
    )
    rows, trajectories, sensitivities = [], [], []
    for model, scored in last_scored.items():
        subset = scored.loc[(scored["store_nbr"] == store) & (scored["family"] == family)].sort_values("date")
        if not np.array_equal(subset["date"].to_numpy(), reference["date"].to_numpy()):
            raise AssertionError("Los modelos no comparten las fechas del caso de inventario")
        if not np.allclose(subset["sales"].to_numpy(), common_actual):
            raise AssertionError("Los modelos no comparten la demanda de comparación")
        pred_col = "prediction_model" if "prediction_model" in subset else (
            "prediction_tsb" if "prediction_tsb" in subset else "pred_baseline"
        )
        forecast = subset[pred_col].to_numpy()
        summary, trajectory = simulate_inventory(
            forecast, actual=common_actual, initial_stock=common_initial_stock,
            lead_time_days=3, safety_stock=common_safety_stock, unit_cost=1.0,
            holding_cost_per_unit=0.02, shortage_penalty_per_unit=2.0,
        )
        rows.append({
            "store_nbr": store,
            "family": family,
            "model": model,
            "policy_reference_model": reference_model,
            "initial_stock_policy": "25_pct_reference_forecast",
            "safety_stock_policy": "prior_external_fold_reference_errors",
            "demand_comparison_policy": "common_observed_sales",
            "service_level": 0.95,
            "lead_time_days": 3,
            "safety_stock_source_validation_end": str(
                reference_previous["date"].max().date()
            ),
            "safety_stock": common_safety_stock,
            "initial_stock_assumption": common_initial_stock,
            **summary,
        })
        trajectory.insert(0, "model", model)
        trajectories.append(trajectory)
        sensitivity = inventory_sensitivity(
            forecast,
            error_pct=0.15,
            base_demand=common_actual,
            initial_stock=common_initial_stock,
            lead_time_days=3,
            safety_stock=common_safety_stock,
            unit_cost=1.0,
            holding_cost_per_unit=0.02,
            shortage_penalty_per_unit=2.0,
        )
        sensitivity.insert(0, "model", model)
        sensitivity.insert(1, "service_level", 0.95)
        sensitivities.append(sensitivity)
    pd.DataFrame(rows).to_csv(output_dir / "inventory_example_summary.csv", index=False)
    pd.concat(trajectories, ignore_index=True).to_csv(output_dir / "inventory_example_trajectory.csv", index=False)
    pd.concat(sensitivities, ignore_index=True).to_csv(output_dir / "inventory_sensitivity.csv", index=False)
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    started = time.time()
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_seconds = {}
    if args.quick:
        args.trials = min(args.trials, 2)
        args.max_estimators = min(args.max_estimators, 250)
        args.catboost_max_iterations = min(args.catboost_max_iterations, 200)
        args.catboost_iterations = min(args.catboost_iterations or 200, 200)

    stage_started = time.perf_counter()
    datasets = minimal_cleaning(load_raw_data(args.data_dir))
    data_signature = dataset_signature(datasets)
    stage_seconds["load_clean_and_signature"] = time.perf_counter() - stage_started
    windows = v2_development_windows()
    windows.to_csv(output_dir / "folds.csv", index=False)
    print(windows.to_string(index=False), flush=True)

    tuning_prepared = []
    stage_started = time.perf_counter()
    for row in windows.loc[windows["role"].eq("tuning")].itertuples(index=False):
        print(f"Preparando fold tuning {row.fold}", flush=True)
        tuning_prepared.append(prepare(datasets, row))
    stage_seconds["prepare_tuning_folds"] = time.perf_counter() - stage_started

    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock

    study_storage_path = (
        Path(args.study_storage).resolve()
        if args.study_storage
        else (output_dir / "optuna_study.log").resolve()
    )
    try:
        study_storage_reference = study_storage_path.relative_to(ROOT).as_posix()
    except ValueError:
        study_storage_reference = study_storage_path.name
    study_storage_path.parent.mkdir(parents=True, exist_ok=True)
    study_storage = JournalStorage(
        JournalFileBackend(
            str(study_storage_path),
            lock_obj=JournalFileOpenLock(str(study_storage_path)),
        )
    )
    study_name = (
        f"{PROTOCOL_VERSION}-data-{data_signature}-"
        f"tuning4-h15-max{args.max_estimators}"
    )
    stage_started = time.perf_counter()
    study = optimize_lightgbm_recursive(
        tuning_prepared, n_trials=args.trials,
        max_estimators=args.max_estimators, stopping_rounds=30,
        storage=study_storage, study_name=study_name,
    )
    stage_seconds["optuna_lightgbm"] = time.perf_counter() - stage_started
    best, _ = save_study_results(study, output_dir)
    best_params = {k: v for k, v in study.best_params.items()}
    selected_estimators = int(best["selected_n_estimators"])
    fold_iterations = best.get("tuning_fold_best_iterations") or []
    near_cap = sum(
        int(iterations) > args.max_estimators - 30
        for iterations in fold_iterations
    )
    best["near_estimator_cap_folds"] = near_cap
    best["estimator_cap_is_binding"] = near_cap >= len(fold_iterations) / 2
    (output_dir / "best_lightgbm.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    print("Mejor LightGBM:", best, flush=True)
    if best["estimator_cap_is_binding"] and not args.quick:
        raise RuntimeError(
            "El máximo de árboles sigue limitando el trial ganador; aumente "
            "--max-estimators antes de evaluar los folds externos."
        )

    stage_started = time.perf_counter()
    catboost_budget_path = output_dir / "catboost_iterations.json"
    if args.catboost_iterations is None:
        cached_catboost = None
        if catboost_budget_path.exists():
            candidate = json.loads(catboost_budget_path.read_text(encoding="utf-8"))
            if (
                candidate.get("protocol_version") == PROTOCOL_VERSION
                and candidate.get("data_signature") == data_signature
                and candidate.get("max_iterations") == args.catboost_max_iterations
                and candidate.get("folds") == [1, 2, 3, 4]
                and candidate.get("source") in {
                    "tuning_inner_early_stopping_median",
                    "cached_tuning_inner_early_stopping_median",
                }
            ):
                cached_catboost = candidate
        if cached_catboost:
            catboost_iterations = int(cached_catboost["selected_iterations"])
            catboost_fold_iterations = cached_catboost["fold_best_iterations"]
            catboost_budget_source = "cached_tuning_inner_early_stopping_median"
        else:
            catboost_iterations, catboost_fold_iterations = select_catboost_iterations(
                tuning_prepared,
                max_iterations=args.catboost_max_iterations,
                stopping_rounds=50,
            )
            catboost_budget_source = "tuning_inner_early_stopping_median"
    else:
        catboost_iterations = int(args.catboost_iterations)
        catboost_fold_iterations = None
        catboost_budget_source = "fixed_cli"
    catboost_budget_path.write_text(
        json.dumps({
            "selected_iterations": catboost_iterations,
            "fold_best_iterations": catboost_fold_iterations,
            "max_iterations": args.catboost_max_iterations,
            "source": catboost_budget_source,
            "protocol_version": PROTOCOL_VERSION,
            "data_signature": data_signature,
            "folds": [1, 2, 3, 4],
        }, indent=2),
        encoding="utf-8",
    )
    stage_seconds["catboost_iteration_calibration"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    tsb_params, tsb_search = select_tsb_parameters(tuning_prepared)
    tsb_search.to_csv(output_dir / "tsb_parameter_search.csv", index=False)

    cheap_rows = []
    for row, prepared in zip(windows.loc[windows["role"].eq("tuning")].itertuples(index=False), tuning_prepared):
        cheap_rows.extend(baseline_tsb_rows(prepared, row.fold, row.role, tsb_params))
    stage_seconds["tsb_selection_and_tuning_diagnostics"] = time.perf_counter() - stage_started
    del tuning_prepared
    gc.collect()

    all_rows = cheap_rows
    scored_by_fold = {}
    prepared_by_fold = {}
    evaluation_fold_seconds = {}
    last_eval_fold = int(windows.loc[windows["role"].eq("development_evaluation"), "fold"].max())
    last_fold_artifacts = None
    for row in windows.loc[windows["role"].eq("development_evaluation")].itertuples(index=False):
        print(f"Evaluando fold externo {row.fold}", flush=True)
        fold_started = time.perf_counter()
        prepared = prepare(datasets, row)
        evaluation_result = evaluate_v2_fold(
            prepared, row.fold,
            lgbm_optimized_params=best_params,
            lgbm_optimized_estimators=selected_estimators,
            catboost_estimators=catboost_iterations,
            tsb_params=tsb_params,
            return_artifacts=int(row.fold) == last_eval_fold,
        )
        if int(row.fold) == last_eval_fold:
            summary, scored, last_fold_artifacts = evaluation_result
        else:
            summary, scored = evaluation_result
        summary["role"] = row.role
        all_rows.extend(summary.to_dict("records"))
        scored_by_fold[row.fold] = scored
        prepared_by_fold[row.fold] = prepared
        evaluation_fold_seconds[str(row.fold)] = time.perf_counter() - fold_started
    stage_seconds["development_evaluation"] = sum(evaluation_fold_seconds.values())

    postprocess_started = time.perf_counter()
    metrics = pd.DataFrame(all_rows)
    metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    heldout = metrics.loc[metrics["role"].eq("development_evaluation")]
    aggregate = heldout.groupby("model", as_index=False).agg(
        folds=("fold", "nunique"), RMSLE_mean=("RMSLE", "mean"), RMSLE_std=("RMSLE", "std"),
        MAE_mean=("MAE", "mean"), WAPE_pct_mean=("WAPE_pct", "mean"), Bias_pct_mean=("Bias_pct", "mean"),
    )
    aggregate.to_csv(output_dir / "model_comparison.csv", index=False)

    eval_folds = sorted(scored_by_fold)
    last_fold = eval_folds[-1]
    last_prepared = prepared_by_fold[last_fold]
    promotion, _ = promotion_experiment(
        last_prepared,
        best_params,
        selected_estimators,
        fitted_model=last_fold_artifacts["lightgbm_optimized"],
        category_schema=last_fold_artifacts["category_schema"],
        full_prediction=scored_by_fold[last_fold]["LightGBM optimized"],
    )
    promotion.to_csv(output_dir / "promotion_scenarios.csv", index=False)
    _, profiles = segment_outputs(prepared_by_fold, scored_by_fold, output_dir)
    inventory = inventory_example(
        last_prepared, scored_by_fold[last_fold], scored_by_fold[eval_folds[-2]], profiles[last_fold], output_dir
    )
    stage_seconds["diagnostics_promotions_inventory"] = time.perf_counter() - postprocess_started

    manifest = {
        "protocol": "quick_verification" if args.quick else "full",
        "august_evaluated": False,
        "trials": args.trials,
        "max_estimators": args.max_estimators,
        "catboost_iterations": catboost_iterations,
        "catboost_iteration_source": catboost_budget_source,
        "catboost_max_iterations": args.catboost_max_iterations,
        "tsb_alpha": tsb_params[0],
        "tsb_beta": tsb_params[1],
        "python": platform.python_version(),
        "elapsed_seconds": time.time() - started,
        "stage_seconds": stage_seconds,
        "evaluation_fold_seconds": evaluation_fold_seconds,
        "prepared_data_cache": "in_memory_once_per_cutoff",
        "study_name": study_name,
        "study_storage": study_storage_reference,
        "data_signature": data_signature,
        "folds": windows.assign(
            train_end=windows["train_end"].astype(str),
            validation_start=windows["validation_start"].astype(str),
            validation_end=windows["validation_end"].astype(str),
        ).to_dict("records"),
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nCOMPARACIÓN HELD-OUT\n", aggregate.to_string(index=False), flush=True)
    print("\nPROMOCIONES\n", promotion.to_string(index=False), flush=True)
    print("\nINVENTARIO\n", inventory.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
