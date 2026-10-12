import numpy as np
import pandas as pd
import pytest

from src.data.panel_diagnostics import (
    apply_modeling_eligibility,
    derive_modeling_eligibility,
    split_future_known_and_cold_start,
)
from src.features.model_features import add_lag_features, add_rolling_features
from src.models.baseline import (
    aggregate_percentage_bias,
    over_under_magnitudes,
    wape,
)
from src.models.categorical import apply_category_schema, fit_category_schema
from src.models.recursive_importance import (
    permute_feature_within_date,
    recursive_permutation_importance,
)
from src.models.recursive_forecast import (
    build_recursive_features_for_date,
    recursive_forecast,
)
from src.models.validation import fixed_temporal_split, rolling_origin_windows


def test_fixed_temporal_split_has_no_overlap_or_future_rows():
    data = pd.DataFrame(
        {"date": pd.date_range("2017-01-01", periods=10), "sales": range(10)}
    )
    train, validation = fixed_temporal_split(
        data,
        validation_start="2017-01-06",
        validation_end="2017-01-08",
    )

    assert train["date"].max() == pd.Timestamp("2017-01-05")
    assert validation["date"].tolist() == list(
        pd.date_range("2017-01-06", "2017-01-08")
    )
    assert not validation["date"].gt("2017-01-08").any()
    assert len(rolling_origin_windows("2017-08-01", 3, 15)) == 3


def test_eligibility_uses_train_only_and_retains_cold_start():
    sales = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2017-01-01", "2017-01-02", "2017-01-01", "2017-01-02"]
            ),
            "store_nbr": [1, 1, 1, 1],
            "family": ["KNOWN", "KNOWN", "COLD", "COLD"],
            "sales": [2.0, 0.0, 0.0, 3.0],
        }
    )
    eligibility = derive_modeling_eligibility(sales, cutoff="2017-01-01")
    annotated = apply_modeling_eligibility(sales, eligibility)
    _, cold, cold_keys = split_future_known_and_cold_start(
        annotated,
        start="2017-01-02",
        end="2017-01-02",
    )

    assert eligibility.known_series["family"].tolist() == ["KNOWN"]
    assert cold_keys["family"].tolist() == ["COLD"]
    assert cold["sales"].tolist() == [3.0]


def test_lags_use_calendar_dates_not_row_positions():
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2017-01-01", "2017-01-03"]),
            "store_nbr": [1, 1],
            "family": ["A", "A"],
            "sales": [10.0, 30.0],
        }
    )
    result = add_lag_features(data, lags=(1, 2))

    assert pd.isna(result.loc[1, "lag_1"])
    assert result.loc[1, "lag_2"] == 10.0


def test_rolling_excludes_current_day():
    data = pd.DataFrame(
        {
            "date": pd.date_range("2017-01-01", periods=3),
            "store_nbr": [1, 1, 1],
            "family": ["A", "A", "A"],
            "sales": [10.0, 20.0, 1000.0],
        }
    )
    result = add_rolling_features(data, windows=(2,))

    assert result.loc[2, "rolling_mean_2"] == 15.0


def _recursive_fixture():
    dates = pd.date_range("2017-01-01", periods=7)
    base = pd.DataFrame(
        {
            "date": dates,
            "store_nbr": [1] * len(dates),
            "family": ["A"] * len(dates),
            "sales": np.arange(1.0, len(dates) + 1),
        }
    )
    train = base.loc[base["date"].le("2017-01-04")].copy()
    train_x = train[["store_nbr", "family"]].copy()
    _, schema = fit_category_schema(train_x, ["store_nbr", "family"])
    return base, train, schema


def test_historical_and_recursive_features_have_parity():
    base, train, schema = _recursive_fixture()
    historical = add_rolling_features(
        add_lag_features(base, lags=(1, 2)),
        windows=(2,),
    )
    history = train[["date", "store_nbr", "family", "sales"]].rename(
        columns={"sales": "history_sales"}
    )
    _, recursive_x, _ = build_recursive_features_for_date(
        features_base=base,
        history=history,
        current_date="2017-01-05",
        model_features=[
            "store_nbr",
            "family",
            "lag_1",
            "lag_2",
            "rolling_mean_2",
        ],
        category_schema=schema,
        lags=(1, 2),
        rolling_windows=(2,),
    )
    expected = historical.loc[historical["date"].eq("2017-01-05")].iloc[0]

    assert recursive_x.iloc[0]["lag_1"] == expected["lag_1"]
    assert recursive_x.iloc[0]["lag_2"] == expected["lag_2"]
    assert recursive_x.iloc[0]["rolling_mean_2"] == expected["rolling_mean_2"]


class _RecordingModel:
    def __init__(self, predictions):
        self.predictions = iter(predictions)
        self.frames = []

    def predict(self, X):
        self.frames.append(X.copy())
        value = next(self.predictions)
        return np.full(len(X), np.log1p(value))


def test_recursive_forecast_never_uses_future_truth():
    base, train, schema = _recursive_fixture()
    model = _RecordingModel([100.0, 200.0, 300.0])
    recursive_forecast(
        model=model,
        features_base=base,
        train=train,
        validation_start="2017-01-05",
        validation_end="2017-01-07",
        model_features=["store_nbr", "family", "lag_1"],
        category_schema=schema,
        lags=(1,),
        rolling_windows=(),
    )

    assert np.isclose(model.frames[1].iloc[0]["lag_1"], 100.0)
    assert np.isclose(model.frames[2].iloc[0]["lag_1"], 200.0)
    assert not np.isclose(model.frames[1].iloc[0]["lag_1"], 5.0)


def test_category_schema_is_exact_and_rejects_unknowns():
    train = pd.DataFrame({"family": ["A", "B", "A"]})
    _, schema = fit_category_schema(train, ["family"])
    known, unknown = apply_category_schema(
        pd.DataFrame({"family": ["B"]}), schema
    )

    assert list(known["family"].cat.categories) == ["A", "B"]
    assert unknown == {}
    with pytest.raises(ValueError, match="Categorías desconocidas"):
        apply_category_schema(pd.DataFrame({"family": ["C"]}), schema)


def test_recursive_forecast_covers_complete_horizon():
    base, train, schema = _recursive_fixture()
    model = _RecordingModel([10.0, 10.0, 10.0])
    result = recursive_forecast(
        model=model,
        features_base=base,
        train=train,
        validation_start="2017-01-05",
        validation_end="2017-01-07",
        model_features=["store_nbr", "family", "lag_1"],
        category_schema=schema,
        lags=(1,),
        rolling_windows=(),
    )

    assert result["date"].tolist() == list(
        pd.date_range("2017-01-05", "2017-01-07")
    )
    assert len(result) == 3


def test_business_metrics_handle_zero_denominator_and_signed_magnitudes():
    assert np.isnan(wape([0, 0], [1, 2]))
    assert np.isnan(aggregate_percentage_bias([0, 0], [1, 2]))
    assert np.isclose(wape([10, 20], [12, 15]), 7 / 30 * 100)
    assert np.isclose(aggregate_percentage_bias([10, 20], [12, 15]), -10.0)
    assert over_under_magnitudes([10, 20], [12, 15]) == {
        "Overprediction": 2.0,
        "Underprediction": 5.0,
    }


def test_feature_transform_is_reforecast_recursively():
    base, train, schema = _recursive_fixture()
    model = _RecordingModel([100.0, 200.0, 300.0])

    def perturb_first_day(X_day, keys, current_date):
        if current_date == pd.Timestamp("2017-01-05"):
            X_day["lag_1"] = 999.0
        return X_day

    recursive_forecast(
        model=model,
        features_base=base,
        train=train,
        validation_start="2017-01-05",
        validation_end="2017-01-07",
        model_features=["store_nbr", "family", "lag_1"],
        category_schema=schema,
        lags=(1,),
        rolling_windows=(),
        feature_transform=perturb_first_day,
    )

    assert model.frames[0].iloc[0]["lag_1"] == 999.0
    assert np.isclose(model.frames[1].iloc[0]["lag_1"], 100.0)
    assert np.isclose(model.frames[2].iloc[0]["lag_1"], 200.0)
    assert list(model.frames[0]["family"].cat.categories) == ["A"]


class _LagModel:
    def __init__(self):
        self.frames = []

    def predict(self, X):
        self.frames.append(X.copy())
        return np.log1p(X["lag_1"].fillna(0).to_numpy(dtype=float))


def test_permutation_never_crosses_forecast_dates():
    keys = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2017-01-05", "2017-01-05", "2017-01-06", "2017-01-06"]
            ),
            "store_nbr": [1, 2, 1, 2],
            "family": ["A", "A", "A", "A"],
        }
    )
    X = pd.DataFrame({"lag_1": [10.0, 20.0, 100.0, 200.0]})
    permuted, audit = permute_feature_within_date(
        X_day=X,
        keys=keys,
        feature="lag_1",
        rng=np.random.default_rng(4),
    )

    assert audit["date"].eq(audit["source_date"]).all()
    for date, positions in keys.groupby("date").indices.items():
        assert sorted(permuted.iloc[positions]["lag_1"]) == sorted(
            X.iloc[positions]["lag_1"]
        )
        assert set(audit.iloc[positions]["source_date"]) == {date}


def test_recursive_importance_features_do_not_depend_on_validation_truth():
    base, train, schema = _recursive_fixture()
    validation = base.loc[base["date"].gt("2017-01-04")].copy()
    altered = validation.copy()
    altered["sales"] = [9000.0, 8000.0, 7000.0]
    altered_base = base.copy()
    altered_base.loc[altered_base["date"].gt("2017-01-04"), "sales"] = altered[
        "sales"
    ].to_numpy()

    base_model = _LagModel()
    common = dict(
        model=base_model,
        features_base=base,
        train=train,
        validation_start="2017-01-05",
        validation_end="2017-01-07",
        model_features=["store_nbr", "family", "lag_1"],
        category_schema=schema,
        n_repeats=1,
        random_state=7,
        lags=(1,),
        rolling_windows=(),
    )
    _, predictions, trace = recursive_permutation_importance(
        validation=validation, **common
    )
    altered_model = _LagModel()
    altered_common = {
        **common,
        "model": altered_model,
        "features_base": altered_base,
    }
    _, altered_predictions, altered_trace = recursive_permutation_importance(
        validation=altered, **altered_common
    )

    pd.testing.assert_frame_equal(predictions, altered_predictions)
    pd.testing.assert_frame_equal(trace, altered_trace)
    assert len(base_model.frames) == len(altered_model.frames)
    for base_frame, altered_frame in zip(base_model.frames, altered_model.frames):
        pd.testing.assert_frame_equal(base_frame, altered_frame)
    assert trace.loc[1, "lag_1"] == predictions.loc[0, "prediction"]
