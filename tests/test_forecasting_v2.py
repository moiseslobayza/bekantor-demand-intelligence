import numpy as np
import pandas as pd
import pytest

from src.data.load_data import resolve_data_dir, validate_raw_files
from src.features.promotion_scenarios import (
    fit_historical_promotion_rates,
    make_promotion_transform,
)
from src.models.inventory import (
    inventory_sensitivity,
    safety_stock_from_errors,
    simulate_inventory,
)
from src.models.lightgbm_model import LIGHTGBM_BASE_PARAMS
from src.models.tsb import fit_tsb_series, select_tsb_parameters, tsb_forecast
from src.models.v2_evaluation import validate_prediction_coverage
from src.models.validation import rolling_origin_windows, v2_development_windows
from scripts.run_forecasting_v2 import dataset_signature


def test_v2_folds_are_non_overlapping_and_roles_are_predefined():
    windows = v2_development_windows()
    assert len(windows) == 6
    assert windows["role"].tolist() == ["tuning"] * 4 + ["development_evaluation"] * 2
    assert windows["validation_end"].max() < pd.Timestamp("2017-08-01")
    assert (windows["validation_start"] == windows["train_end"] + pd.Timedelta(days=1)).all()
    assert (windows["validation_start"].iloc[1:].to_numpy() > windows["validation_end"].iloc[:-1].to_numpy()).all()


def test_v1_window_and_parameters_remain_unchanged():
    assert len(rolling_origin_windows("2017-08-01", 3, 15)) == 3
    assert LIGHTGBM_BASE_PARAMS["num_leaves"] == 63
    assert LIGHTGBM_BASE_PARAMS["learning_rate"] == 0.05
    assert LIGHTGBM_BASE_PARAMS["min_child_samples"] == 100


def test_study_signature_ignores_august_but_detects_pre_august_changes():
    datasets = {
        "sales": pd.DataFrame({
            "date": pd.to_datetime(["2017-07-31", "2017-08-01"]),
            "store_nbr": [1, 1], "family": ["A", "A"],
            "sales": [10.0, 20.0], "onpromotion": [0, 1],
        }),
        "stores": pd.DataFrame({"store_nbr": [1], "city": ["Quito"]}),
        "holidays": pd.DataFrame({
            "date": pd.to_datetime(["2017-07-31", "2017-08-01"]),
            "type": ["Holiday", "Holiday"],
        }),
    }
    original = dataset_signature(datasets)
    august_changed = {name: frame.copy() for name, frame in datasets.items()}
    august_changed["sales"].loc[1, "sales"] = 999999
    august_changed["holidays"].loc[1, "type"] = "Transfer"
    assert dataset_signature(august_changed) == original
    july_changed = {name: frame.copy() for name, frame in datasets.items()}
    july_changed["sales"].loc[0, "sales"] = 11
    assert dataset_signature(july_changed) != original


def test_portable_data_dir_reports_all_missing_files(monkeypatch):
    missing_path = resolve_data_dir("data/definitely-missing-v2")
    monkeypatch.setenv("FAVORITA_DATA_DIR", str(missing_path))
    assert resolve_data_dir() == missing_path
    with pytest.raises(FileNotFoundError, match="train.csv") as error:
        validate_raw_files()
    assert "holidays_events.csv" in str(error.value)


def test_tsb_handles_zero_intermittent_and_nonnegative_forecasts():
    assert fit_tsb_series([0, 0, 0])["forecast"] == 0
    state = fit_tsb_series([0, 5, 0, 0, 10], alpha=0.2, beta=0.3)
    assert 0 <= state["probability"] <= 1
    assert state["size"] > 0
    assert state["forecast"] >= 0

    train = pd.DataFrame({
        "date": pd.date_range("2017-01-01", periods=5),
        "store_nbr": [1] * 5,
        "family": ["A"] * 5,
        "sales": [0, 5, 0, 0, 10],
    })
    validation = pd.DataFrame({
        "date": pd.date_range("2017-01-06", periods=2),
        "store_nbr": [1] * 2,
        "family": ["A"] * 2,
        "sales": [0, 2],
    })
    pred = tsb_forecast(train, validation)
    assert len(pred) == 2
    assert pred["prediction_tsb"].ge(0).all()


def test_tsb_recurrence_counts_zero_demand_as_nonoccurrence():
    state = fit_tsb_series([0, 5, 0], alpha=0.2, beta=0.3)
    expected_probability = ((((1 / 3) * 0.7) * 0.7) + 0.3) * 0.7
    assert state["probability"] == pytest.approx(expected_probability)
    assert state["size"] == pytest.approx(5.0)
    assert state["forecast"] == pytest.approx(expected_probability * 5.0)


def test_tsb_parameter_selection_uses_only_supplied_folds():
    train = pd.DataFrame({
        "date": pd.date_range("2017-01-01", periods=4),
        "store_nbr": [1] * 4,
        "family": ["A"] * 4,
        "sales": [0, 3, 0, 4],
    })
    validation = pd.DataFrame({
        "date": pd.date_range("2017-01-05", periods=2),
        "store_nbr": [1] * 2,
        "family": ["A"] * 2,
        "sales": [0, 5],
    })
    best, results = select_tsb_parameters(
        [{"train": train, "validation": validation}], candidates=[(0.1, 0.1), (0.3, 0.3)]
    )
    assert best in {(0.1, 0.1), (0.3, 0.3)}
    assert len(results) == 2


def test_promotion_imputation_is_cutoff_safe_and_future_invariant():
    history = pd.DataFrame({
        "date": pd.to_datetime(["2017-01-01", "2017-01-08", "2017-01-15", "2017-02-01"]),
        "store_nbr": [1, 1, 1, 1],
        "family": ["A"] * 4,
        "onpromotion": [0, 2, 4, 999],
    })
    rates = fit_historical_promotion_rates(history, "2017-01-15")
    assert rates["max_history_date"] == pd.Timestamp("2017-01-15")
    assert rates["global"] == 2.0

    X = pd.DataFrame({"onpromotion": [999.0]})
    keys = pd.DataFrame({"date": [pd.Timestamp("2017-01-22")], "store_nbr": [1], "family": ["A"]})
    transform = make_promotion_transform("none", history, "2017-01-15")
    first = transform(X.copy(), keys, "2017-01-22")
    altered = history.copy()
    altered.loc[altered["date"].gt("2017-01-15"), "onpromotion"] = 123456
    second = make_promotion_transform("none", altered, "2017-01-15")(X.copy(), keys, "2017-01-22")
    pd.testing.assert_frame_equal(first, second)
    assert first["onpromotion"].iloc[0] != 0


def test_partial_promotion_keeps_only_confirmed_leads():
    history = pd.DataFrame({
        "date": pd.to_datetime(["2017-01-01", "2017-01-08"]),
        "store_nbr": [1, 1], "family": ["A", "A"], "onpromotion": [1, 3],
    })
    transform = make_promotion_transform("partial", history, "2017-01-08", confirmed_days=2)
    X = pd.DataFrame({"onpromotion": [99.0]})
    key1 = pd.DataFrame({"date": [pd.Timestamp("2017-01-10")], "store_nbr": [1], "family": ["A"]})
    key2 = key1.assign(date=pd.Timestamp("2017-01-11"))
    assert transform(X.copy(), key1, "2017-01-10")["onpromotion"].iloc[0] == 99
    assert transform(X.copy(), key2, "2017-01-11")["onpromotion"].iloc[0] != 99


def test_inventory_replenishment_nonnegative_and_costs_reconcile():
    safety = safety_stock_from_errors([-2, 0, 2, 1], lead_time_days=2, service_level=0.95)
    assert safety >= 0
    summary, trajectory = simulate_inventory(
        [10, 10, 10], actual=[12, 8, 15], initial_stock=5,
        lead_time_days=1, safety_stock=safety, unit_cost=2,
        holding_cost_per_unit=0.1, shortage_penalty_per_unit=4,
    )
    assert summary["recommended_replenishment"] >= 0
    assert trajectory["ending_inventory"].ge(0).all()
    assert np.isclose(
        summary["simulated_total_cost"],
        summary["purchase_cost"] + summary["holding_cost"] + summary["shortage_cost"],
    )
    sensitivity = inventory_sensitivity(
        [10, 10], error_pct=0.2, base_demand=[8, 12], initial_stock=0
    )
    assert sensitivity["scenario"].tolist() == [
        "high_demand", "observed_base", "low_demand"
    ]
    assert sensitivity["simulated_demand"].tolist() == pytest.approx([24, 20, 16])


def test_inventory_sensitivity_uses_common_demand_across_forecasts():
    common_demand = [8, 12]
    first = inventory_sensitivity(
        [9, 9], error_pct=0.15, base_demand=common_demand, initial_stock=5
    )
    second = inventory_sensitivity(
        [20, 1], error_pct=0.15, base_demand=common_demand, initial_stock=5
    )
    assert first["simulated_demand"].tolist() == pytest.approx(
        second["simulated_demand"].tolist()
    )


def test_inventory_lead_time_uses_pre_receipt_stock_and_does_not_reorder_lost_sales():
    summary, trajectory = simulate_inventory(
        [10, 10, 10, 10, 10], actual=[10, 10, 10, 10, 10],
        initial_stock=5, lead_time_days=3, safety_stock=2,
    )
    assert trajectory.loc[:2, "receipt"].eq(0).all()
    assert trajectory.loc[3, "receipt"] == pytest.approx(22.0)
    assert trajectory.loc[:2, "potential_shortage"].sum() == pytest.approx(25.0)
    assert summary["lead_time_demand"] == pytest.approx(30.0)
    assert summary["projected_stock_at_receipt"] == pytest.approx(0.0)
    assert summary["target_stock_at_receipt"] == pytest.approx(22.0)
    assert summary["recommended_replenishment"] == pytest.approx(22.0)
    assert summary["ending_inventory"] == pytest.approx(2.0)


def test_prediction_coverage_rejects_missing_or_extra_observations():
    truth = pd.DataFrame({
        "date": pd.to_datetime(["2017-07-01", "2017-07-02"]),
        "store_nbr": [1, 1], "family": ["A", "A"], "sales": [1, 2],
    })
    valid = truth.drop(columns="sales").assign(prediction=[1.0, 2.0])
    validate_prediction_coverage(truth, valid)
    with pytest.raises(ValueError, match="faltan=1"):
        validate_prediction_coverage(truth, valid.iloc[:1])
    extra = pd.concat([
        valid,
        pd.DataFrame({
            "date": [pd.Timestamp("2017-07-03")], "store_nbr": [1],
            "family": ["A"], "prediction": [3.0],
        }),
    ], ignore_index=True)
    with pytest.raises(ValueError, match="sobran=1"):
        validate_prediction_coverage(truth, extra)


def test_optuna_storage_resumes_to_target_total_trials(tmp_path, monkeypatch):
    pytest.importorskip("optuna")
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
    import src.models.optuna_search as search

    prepared = {
        "validation": pd.DataFrame({"date": [pd.Timestamp("2017-07-01")]}),
    }
    monkeypatch.setattr(
        search,
        "_fold_score",
        lambda prepared, params, max_estimators, stopping_rounds: (
            float(params["learning_rate"]), 10
        ),
    )
    journal_path = str(tmp_path / "study.log")
    storage = JournalStorage(
        JournalFileBackend(
            journal_path,
            lock_obj=JournalFileOpenLock(journal_path),
        )
    )
    first = search.optimize_lightgbm_recursive(
        [prepared], n_trials=2, storage=storage, study_name="resume-test"
    )
    first_trial_count = len(first.trials)
    second = search.optimize_lightgbm_recursive(
        [prepared], n_trials=3, storage=storage, study_name="resume-test"
    )
    assert first_trial_count == 2
    assert len(second.trials) == 3
    assert second.trials[2].params != second.trials[0].params


def test_catboost_categorical_adapter_when_dependency_is_available():
    pytest.importorskip("catboost")
    from src.models.catboost_model import fit_catboost

    X = pd.DataFrame({"family": pd.Categorical(["A", "B", "A", "B"]), "lag_1": [0, 1, 2, 3]})
    model = fit_catboost(X, pd.Series([0, 1, 2, 3]), ["family"], n_estimators=5)
    pred = model.predict(X)
    assert len(pred) == 4
    assert np.isfinite(pred).all()
