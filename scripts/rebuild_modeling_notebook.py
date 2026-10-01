"""Reconstruye el notebook de modelado con el pipeline auditable actual."""

from pathlib import Path
import textwrap

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "05_modeling.ipynb"


def md(text):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


cells = [
    md(
        """
        # Forecasting de demanda — LightGBM recursivo sin leakage

        ## tl;dr

        Este notebook conserva **01/08/2017–15/08/2017 como test final**. La
        elegibilidad se deriva en cada cutoff, tres ventanas anteriores se usan
        para backtesting y selección del número de árboles, y el test se evalúa
        una única vez después de fijar `best_iteration`.

        Los resultados ejecutados, su comparación con la versión anterior y las
        conclusiones aparecen al final del notebook.
        """
    ),
    md(
        """
        ## Contexto y método

        ### Supuestos clave

        - Las promociones futuras (`onpromotion`) están disponibles antes del
          horizonte, como en el esquema de competencia Favorita.
        - Sólo se utilizan feriados y días especiales programados. Los registros
          `type=Event`, potencialmente conocidos ex post, quedan fuera del modelo.
        - Un error firmado positivo significa sobrepredicción y uno negativo,
          subpredicción.
        - Las series nuevas se reportan como *cold-start* y no modifican el
          universo conocido en el cutoff.
        """
    ),
    code(
        """
        from pathlib import Path
        import gc
        import sys

        import lightgbm as lgb
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import sklearn
        from IPython.display import display
        from sklearn.inspection import permutation_importance

        PROJECT_ROOT = Path.cwd()
        if PROJECT_ROOT.name == "notebooks":
            PROJECT_ROOT = PROJECT_ROOT.parent
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.append(str(PROJECT_ROOT))

        pd.set_option("display.max_columns", 50)
        plt.style.use("seaborn-v0_8-whitegrid")

        FINAL_TRAIN_END = pd.Timestamp("2017-07-31")
        FINAL_TEST_START = pd.Timestamp("2017-08-01")
        FINAL_TEST_END = pd.Timestamp("2017-08-15")
        HORIZON_DAYS = 15
        N_BACKTEST_WINDOWS = 3

        print("Proyecto:", PROJECT_ROOT)
        print("Test final:", FINAL_TEST_START.date(), "→", FINAL_TEST_END.date())
        print(
            "Versiones validadas:",
            f"pandas={pd.__version__}",
            f"scikit-learn={sklearn.__version__}",
            f"lightgbm={lgb.__version__}",
        )
        """
    ),
    md("## Datos y contrato de features"),
    code(
        """
        from src.data.load_data import load_raw_data
        from src.data.clean_data import minimal_cleaning
        from src.features.model_features import (
            CATEGORICAL_FEATURES,
            MODEL_FEATURES,
            feature_availability_table,
        )

        datasets = minimal_cleaning(load_raw_data())
        sales = datasets["sales"]

        print(f"Ventas: {len(sales):,} filas")
        print(sales["date"].min().date(), "→", sales["date"].max().date())
        display(feature_availability_table())
        """
    ),
    code(
        """
        assert "is_event" not in MODEL_FEATURES
        assert "event_count" not in MODEL_FEATURES
        assert "scheduled_event_count" in MODEL_FEATURES
        assert FINAL_TEST_START == FINAL_TRAIN_END + pd.Timedelta(days=1)

        print("✓ Eventos potencialmente ex post excluidos")
        print("✓ Sólo scheduled_event_count entra al modelo")
        print("✓ Test final separado del desarrollo")
        """
    ),
    md("### Verificación del universo corregido"),
    code(
        """
        from src.data.panel_diagnostics import (
            add_store_operation_flags,
            add_zero_taxonomy,
            build_modeling_panel,
        )

        legacy_sales, legacy_openings = add_store_operation_flags(sales)
        legacy_sales = add_zero_taxonomy(legacy_sales)
        legacy_series = (
            legacy_sales.loc[
                legacy_sales["tienda_operativa"]
                & legacy_sales["familia_surtida"],
                ["store_nbr", "family"],
            ]
            .drop_duplicates()
            .sort_values(["store_nbr", "family"])
            .reset_index(drop=True)
        )

        _, _, final_eligibility = build_modeling_panel(
            sales,
            cutoff=FINAL_TRAIN_END,
        )
        corrected_series = (
            final_eligibility.known_series[["store_nbr", "family"]]
            .sort_values(["store_nbr", "family"])
            .reset_index(drop=True)
        )
        corrected_openings = (
            final_eligibility.store_openings.set_index("store_nbr")["opening_date"]
        )

        universe_differences = len(
            legacy_series.merge(
                corrected_series,
                on=["store_nbr", "family"],
                how="outer",
                indicator=True,
            ).query("_merge != 'both'")
        )
        opening_differences = int(
            (
                legacy_openings.sort_index()
                != corrected_openings.reindex(legacy_openings.index)
            ).sum()
        )

        print("Diferencias de universo:", universe_differences)
        print("Diferencias de aperturas:", opening_differences)
        print("Series conocidas al 31/07/2017:", len(corrected_series))

        assert universe_differences == 0
        assert opening_differences == 0

        del legacy_sales, legacy_series, corrected_series
        gc.collect()
        """
    ),
    md("## Rolling-origin backtesting y selección de árboles"),
    code(
        """
        from src.models.validation import rolling_origin_windows

        backtest_windows = rolling_origin_windows(
            final_test_start=FINAL_TEST_START,
            n_windows=N_BACKTEST_WINDOWS,
            horizon_days=HORIZON_DAYS,
        )
        display(backtest_windows)

        assert backtest_windows["validation_end"].max() < FINAL_TEST_START
        assert (
            backtest_windows["validation_start"]
            == backtest_windows["train_end"] + pd.Timedelta(days=1)
        ).all()
        """
    ),
    code(
        """
        from src.models.experiment import run_backtest_fold

        backtest_summaries = []
        last_fold_result = None

        for row in backtest_windows.itertuples(index=False):
            print(
                f"Fold {row.fold}: train hasta {row.train_end.date()} | "
                f"validación {row.validation_start.date()}–{row.validation_end.date()}"
            )
            fold_result = run_backtest_fold(
                datasets=datasets,
                train_end=row.train_end,
                validation_start=row.validation_start,
                validation_end=row.validation_end,
                fold=row.fold,
                # Techo técnico holgado: early stopping, no el cap, debe fijar
                # la cantidad de árboles. No es una búsqueda de hiperparámetros.
                max_estimators=2000,
                stopping_rounds=50,
            )
            backtest_summaries.append(fold_result["summary"])

            if row.fold == N_BACKTEST_WINDOWS:
                last_fold_result = fold_result
            else:
                del fold_result
                gc.collect()

        backtest_results = pd.concat(backtest_summaries, ignore_index=True)
        display(backtest_results.round(6))
        """
    ),
    code(
        """
        backtest_aggregate = (
            backtest_results.groupby("model", as_index=False)
            .agg(
                RMSLE_mean=("RMSLE", "mean"),
                RMSLE_std=("RMSLE", "std"),
                MAE_mean=("MAE", "mean"),
                MAE_std=("MAE", "std"),
                signed_error_mean=("MSE_signed", "mean"),
                signed_error_std=("MSE_signed", "std"),
            )
        )

        selected_iterations = backtest_results.loc[
            backtest_results["model"].eq("LightGBM recursive"),
            "best_iteration",
        ].astype(int)
        FINAL_N_ESTIMATORS = int(np.median(selected_iterations))

        display(backtest_aggregate.round(6))
        print("best_iteration por fold:", selected_iterations.tolist())
        print("Árboles fijados para el modelo final:", FINAL_N_ESTIMATORS)
        """
    ),
    code(
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for model_name, group in backtest_results.groupby("model"):
            axes[0].plot(group["fold"], group["RMSLE"], marker="o", label=model_name)
            axes[1].plot(group["fold"], group["MAE"], marker="o", label=model_name)

        axes[0].set(title="RMSLE por fold previo al test", xlabel="Fold", ylabel="RMSLE")
        axes[1].set(title="MAE por fold previo al test", xlabel="Fold", ylabel="MAE")
        axes[0].legend()
        axes[1].legend()
        plt.tight_layout()
        plt.show()
        """
    ),
    md("### Permutation importance fuera de entrenamiento"),
    code(
        """
        from src.features.model_features import add_lag_features, add_rolling_features
        from src.models.categorical import apply_category_schema

        last_prepared = last_fold_result["prepared"]
        last_historical_features = add_rolling_features(
            add_lag_features(last_prepared["features_base"]),
            windows=(7, 14, 28),
        )
        permutation_frame = last_historical_features.loc[
            last_historical_features["date"].gt(
                backtest_windows.iloc[-1]["validation_start"]
            )
        ].copy()
        if len(permutation_frame) > 50_000:
            permutation_frame = permutation_frame.sample(50_000, random_state=42)

        X_permutation = permutation_frame[MODEL_FEATURES].copy()
        X_permutation, permutation_unknown = apply_category_schema(
            X_permutation,
            last_fold_result["category_schema"],
        )
        assert permutation_unknown == {}

        permutation = permutation_importance(
            last_fold_result["model"],
            X_permutation,
            np.log1p(permutation_frame["sales"]),
            scoring="neg_mean_squared_error",
            n_repeats=3,
            random_state=42,
            n_jobs=-1,
        )
        permutation_table = (
            pd.DataFrame(
                {
                    "feature": MODEL_FEATURES,
                    "importance_mean": permutation.importances_mean,
                    "importance_std": permutation.importances_std,
                }
            )
            .sort_values("importance_mean", ascending=False)
            .reset_index(drop=True)
        )
        display(permutation_table.head(15).round(6))
        """
    ),
    code(
        """
        permutation_plot = permutation_table.head(15).sort_values("importance_mean")
        plt.figure(figsize=(8, 6))
        plt.barh(
            permutation_plot["feature"],
            permutation_plot["importance_mean"],
            xerr=permutation_plot["importance_std"],
        )
        plt.title("Permutation importance — último fold previo al test")
        plt.xlabel("Aumento medio del MSE log al permutar")
        plt.tight_layout()
        plt.show()

        del last_historical_features, permutation_frame, X_permutation
        del last_fold_result, last_prepared
        gc.collect()
        """
    ),
    md(
        """
        ## Test final intocable

        A partir de este punto `best_iteration` ya está fijado. Se reentrena con
        toda la información disponible hasta el 31/07/2017 y se evalúa el test
        01/08/2017–15/08/2017 una sola vez.
        """
    ),
    code(
        """
        from src.models.experiment import run_final_test

        final_result = run_final_test(
            datasets=datasets,
            train_end=FINAL_TRAIN_END,
            test_start=FINAL_TEST_START,
            test_end=FINAL_TEST_END,
            n_estimators=FINAL_N_ESTIMATORS,
        )

        final_summary = final_result["summary"]
        display(final_summary.round(6))
        print("Cold-start en test final:", len(final_result["prepared"]["cold_start_keys"]))
        assert len(final_result["prepared"]["cold_start_keys"]) == 0
        """
    ),
    md("### Comparación con la versión anterior"),
    code(
        """
        previous_results = pd.DataFrame(
            [
                {"model": "Seasonal baseline", "previous_RMSLE": 0.536571, "previous_MAE": 85.477984},
                {"model": "LightGBM recursive", "previous_RMSLE": 0.405895, "previous_MAE": 70.298360},
            ]
        )
        final_comparison = final_summary.merge(previous_results, on="model", how="left")
        final_comparison["delta_RMSLE"] = (
            final_comparison["RMSLE"] - final_comparison["previous_RMSLE"]
        )
        final_comparison["delta_MAE"] = (
            final_comparison["MAE"] - final_comparison["previous_MAE"]
        )
        display(final_comparison.round(6))
        """
    ),
    md("## Diagnóstico del test final"),
    code(
        """
        from src.models.baseline import mae, mean_signed_error, rmsle

        lgbm_eval = final_result["lgbm_eval"].copy()
        baseline_eval = final_result["baseline_eval"].copy()
        comparison_eval = lgbm_eval.merge(
            baseline_eval[["date", "store_nbr", "family", "pred_baseline"]],
            on=["date", "store_nbr", "family"],
            how="left",
            validate="one_to_one",
        )
        comparison_eval["horizon_day"] = (
            comparison_eval["date"] - FINAL_TEST_START
        ).dt.days + 1

        daily_rows = []
        for horizon_day, group in comparison_eval.groupby("horizon_day"):
            for model_name, prediction_col in [
                ("Seasonal baseline", "pred_baseline"),
                ("LightGBM recursive", "prediction"),
            ]:
                daily_rows.append(
                    {
                        "horizon_day": horizon_day,
                        "model": model_name,
                        "RMSLE": rmsle(group["sales"], group[prediction_col]),
                        "MAE": mae(group["sales"], group[prediction_col]),
                        "MSE_signed": mean_signed_error(
                            group["sales"], group[prediction_col]
                        ),
                    }
                )
        daily_diagnostics = pd.DataFrame(daily_rows)
        display(daily_diagnostics.round(6))
        """
    ),
    code(
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for model_name, group in daily_diagnostics.groupby("model"):
            axes[0].plot(group["horizon_day"], group["RMSLE"], marker="o", label=model_name)
            axes[1].plot(group["horizon_day"], group["MAE"], marker="o", label=model_name)
            axes[2].plot(group["horizon_day"], group["MSE_signed"], marker="o", label=model_name)

        axes[0].set(title="RMSLE por horizonte", xlabel="Día", ylabel="RMSLE")
        axes[1].set(title="MAE por horizonte", xlabel="Día", ylabel="MAE")
        axes[2].axhline(0, color="black", linewidth=1)
        axes[2].set(title="Error firmado por horizonte", xlabel="Día", ylabel="Predicción - real")
        for ax in axes:
            ax.legend()
        plt.tight_layout()
        plt.show()
        """
    ),
    code(
        """
        def grouped_metrics(data, group_col, prediction_col):
            rows = []
            for value, group in data.groupby(group_col, observed=True):
                rows.append(
                    {
                        group_col: value,
                        "RMSLE": rmsle(group["sales"], group[prediction_col]),
                        "MAE": mae(group["sales"], group[prediction_col]),
                        "MSE_signed": mean_signed_error(
                            group["sales"], group[prediction_col]
                        ),
                        "n_obs": len(group),
                    }
                )
            return pd.DataFrame(rows)

        store_diagnostics = grouped_metrics(lgbm_eval, "store_nbr", "prediction")
        family_diagnostics = grouped_metrics(lgbm_eval, "family", "prediction")

        display(store_diagnostics.sort_values("RMSLE", ascending=False).head(15).round(6))
        display(family_diagnostics.sort_values("RMSLE", ascending=False).head(15).round(6))
        """
    ),
    md("### Comparación por tipo de demanda ADI/CV²"),
    code(
        """
        from src.features.demand_classification import build_demand_profile

        train_demand_profile = build_demand_profile(final_result["prepared"]["train"])
        demand_eval = comparison_eval.merge(
            train_demand_profile[["store_nbr", "family", "tipo_demanda"]],
            on=["store_nbr", "family"],
            how="left",
            validate="many_to_one",
        )
        assert demand_eval["tipo_demanda"].notna().all()

        demand_rows = []
        for demand_type, group in demand_eval.groupby("tipo_demanda"):
            for model_name, prediction_col in [
                ("Seasonal baseline", "pred_baseline"),
                ("LightGBM recursive", "prediction"),
            ]:
                demand_rows.append(
                    {
                        "tipo_demanda": demand_type,
                        "model": model_name,
                        "RMSLE": rmsle(group["sales"], group[prediction_col]),
                        "MAE": mae(group["sales"], group[prediction_col]),
                        "MSE_signed": mean_signed_error(
                            group["sales"], group[prediction_col]
                        ),
                        "series": group[["store_nbr", "family"]].drop_duplicates().shape[0],
                    }
                )
        demand_diagnostics = pd.DataFrame(demand_rows)
        display(demand_diagnostics.round(6))
        """
    ),
    code(
        """
        pivot = demand_diagnostics.pivot(
            index="tipo_demanda", columns="model", values="RMSLE"
        )
        pivot.plot(kind="bar", figsize=(9, 5))
        plt.title("RMSLE por tipo de demanda — test final")
        plt.xlabel("Tipo ADI/CV²")
        plt.ylabel("RMSLE")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()
        """
    ),
    md("### Feature importance del modelo final"),
    code(
        """
        final_model = final_result["model"]
        feature_importance = pd.DataFrame(
            {
                "feature": MODEL_FEATURES,
                "gain": final_model.booster_.feature_importance(importance_type="gain"),
                "splits": final_model.booster_.feature_importance(importance_type="split"),
            }
        )
        feature_importance["gain_pct"] = (
            feature_importance["gain"] / feature_importance["gain"].sum() * 100
        )
        feature_importance = feature_importance.sort_values(
            "gain", ascending=False
        ).reset_index(drop=True)
        display(feature_importance.head(15).round(6))

        print(
            "Gain y split describen el uso interno de features por LightGBM; "
            "no miden causalidad."
        )
        """
    ),
    code(
        """
        importance_plot = feature_importance.head(15).sort_values("gain_pct")
        plt.figure(figsize=(8, 6))
        plt.barh(importance_plot["feature"], importance_plot["gain_pct"])
        plt.title("Feature importance por gain — modelo final")
        plt.xlabel("% de gain total")
        plt.tight_layout()
        plt.show()
        """
    ),
    md("## Takeaways y controles finales"),
    code(
        """
        assert final_result["prepared"]["train"]["date"].max() == FINAL_TRAIN_END
        assert comparison_eval["date"].min() == FINAL_TEST_START
        assert comparison_eval["date"].max() == FINAL_TEST_END
        assert comparison_eval["date"].nunique() == HORIZON_DAYS
        assert comparison_eval["prediction"].notna().all()
        assert np.isfinite(comparison_eval["prediction"]).all()
        assert (comparison_eval["prediction"] >= 0).all()
        assert set(final_result["category_schema"]) == set(CATEGORICAL_FEATURES)

        print("RESULTADO FINAL")
        display(final_comparison.round(6))
        print("BACKTESTING")
        display(backtest_aggregate.round(6))
        print("Árboles seleccionados:", FINAL_N_ESTIMATORS)
        print("Universo vs versión anterior: 0 diferencias")
        print("Cold-start en test final: 0 series")
        print("✓ Test final cubierto completamente y evaluado en forma recursiva")
        """
    ),
]


notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.14.7",
        },
    },
)
nbf.validate(notebook)
nbf.write(notebook, OUTPUT)
print(f"Notebook reconstruido: {OUTPUT}")
