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

        ### Alcance del aislamiento temporal

        El EDA histórico aprobado fue realizado previamente sobre datos que
        llegan hasta el 15/08/2017, por lo que **no se afirma independencia
        absoluta del test**. Este notebook separa el uso retrospectivo del EDA
        de las decisiones predictivas: contrato de disponibilidad, universo,
        clasificación ADI/CV², esquema categórico y entrenamiento se calculan
        de nuevo con información anterior al cutoff; la cantidad de árboles se
        fija sólo con los tres folds previos al 01/08. No se ajustan
        hiperparámetros después de observar el test.

        Esta limitación humana/procedimental permanece: el equipo ya había
        visto descriptivos del período final. Por eso el test se interpreta
        como evaluación final congelada del pipeline, no como evidencia de un
        cegamiento experimental perfecto.
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
        _ = gc.collect()
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
                _ = gc.collect()

        backtest_results = pd.concat(backtest_summaries, ignore_index=True)
        backtest_display = backtest_results.copy()
        numeric_columns = backtest_display.select_dtypes(include="number").columns
        backtest_display[numeric_columns] = backtest_display[numeric_columns].round(6)
        display(backtest_display)
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
                WAPE_mean=("WAPE_pct", "mean"),
                WAPE_std=("WAPE_pct", "std"),
                signed_error_mean=("MSE_signed", "mean"),
                signed_error_std=("MSE_signed", "std"),
                bias_pct_mean=("Bias_pct", "mean"),
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
    md(
        """
        ### Permutation importance recursiva fuera de entrenamiento

        La permutation importance tabular convencional no es comparable con
        este forecast: reconstruir lags sobre la validación incorporaría ventas
        reales que no existen al emitir el pronóstico. Aquí cada perturbación
        vuelve a recorrer los 15 días. Sólo permuta features generadas por el
        recorrido base, realimenta predicciones perturbadas y nunca usa ventas
        reales para lags o rolling. Cada feature se permuta sólo entre series
        de la misma fecha; los lags y rolling se reconstruyen primero desde la
        historia del recorrido perturbado. La importancia es el aumento de
        RMSLE y mide sensibilidad predictiva, no causalidad. Las features
        constantes entre series de un mismo día tienen importancia cero por
        construcción bajo este procedimiento.
        """
    ),
    code(
        """
        from src.models.recursive_importance import recursive_permutation_importance

        last_prepared = last_fold_result["prepared"]
        permutation_table, permutation_predictions, permutation_trace = (
            recursive_permutation_importance(
                model=last_fold_result["model"],
                features_base=last_prepared["features_base"],
                train=last_prepared["train"],
                validation=last_prepared["validation"],
                validation_start=backtest_windows.iloc[-1]["validation_start"],
                validation_end=backtest_windows.iloc[-1]["validation_end"],
                model_features=MODEL_FEATURES,
                category_schema=last_fold_result["category_schema"],
                n_repeats=3,
                random_state=42,
            )
        )
        assert permutation_trace["date"].min() > last_prepared["train"]["date"].max()
        assert len(permutation_predictions) == len(last_prepared["validation"])
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
        plt.xlabel("Aumento medio de RMSLE al permutar y re-pronosticar")
        plt.tight_layout()
        plt.show()

        del permutation_predictions, permutation_trace
        del last_fold_result, last_prepared
        _ = gc.collect()
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
    md(
        """
        ## Diagnóstico comercial del test final

        WAPE pondera el error absoluto por el volumen real; si un segmento no
        tiene ventas, queda como `NaN` en vez de dividir por cero. Un sesgo
        negativo indica subpredicción (riesgo potencial de faltantes) y uno
        positivo sobrepredicción (riesgo potencial de sobrestock). Son riesgos
        inferidos del forecast: sin inventario, quiebres ni costos no se afirma
        que hayan ocurrido faltantes, pérdidas o excedentes reales.

        RMSLE y WAPE responden preguntas diferentes: RMSLE enfatiza errores
        relativos y reduce el peso de grandes volúmenes; WAPE refleja unidades
        absolutas sobre el volumen total. Mejorar una no garantiza mejorar la
        otra.
        """
    ),
    code(
        """
        from src.models.experiment import prediction_metrics, segment_metrics

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
                daily_rows.append({
                    "horizon_day": horizon_day,
                    "model": model_name,
                    **prediction_metrics(group["sales"], group[prediction_col]),
                })
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
        def compare_segments(data, group_cols):
            frames = []
            for model_name, prediction_col in [
                ("Seasonal baseline", "pred_baseline"),
                ("LightGBM recursive", "prediction"),
            ]:
                metrics = segment_metrics(data, prediction_col, group_cols)
                metrics["model"] = model_name
                frames.append(metrics)
            return pd.concat(frames, ignore_index=True)

        store_diagnostics = compare_segments(comparison_eval, ["store_nbr"])
        family_diagnostics = compare_segments(comparison_eval, ["family"])

        display(store_diagnostics.sort_values("RMSLE", ascending=False).head(20).round(4))
        display(family_diagnostics.sort_values("RMSLE", ascending=False).head(20).round(4))
        """
    ),
    md(
        """
        ### Familias difíciles y patrones ADI/CV²

        La clasificación se calcula exclusivamente con train hasta 31/07. Se
        comparan ambos modelos para LINGERIE, GROCERY II, demanda intermitente y
        lumpy, incluyendo ceros, frecuencia del signo del error, magnitudes y
        tamaños de muestra. Esto genera evidencia para decidir más adelante si
        vale la pena probar un hurdle; aquí no se entrena ningún segundo modelo.
        """
    ),
    code(
        """
        from src.features.demand_classification import build_demand_profile

        train_demand_profile = build_demand_profile(final_result["prepared"]["train"])
        demand_eval = comparison_eval.merge(
            train_demand_profile[
                ["store_nbr", "family", "tipo_demanda", "pct_ceros"]
            ],
            on=["store_nbr", "family"],
            how="left",
            validate="many_to_one",
        )
        assert demand_eval["tipo_demanda"].notna().all()

        demand_diagnostics = compare_segments(demand_eval, ["tipo_demanda"])
        difficult_demand = demand_diagnostics.loc[
            demand_diagnostics["tipo_demanda"].isin(["Intermitente", "Lumpy"])
        ].copy()
        difficult_families = family_diagnostics.loc[
            family_diagnostics["family"].isin(["LINGERIE", "GROCERY II"])
        ].copy()

        display(demand_diagnostics.round(4))
        print("Foco: demanda intermitente y lumpy")
        display(difficult_demand.round(4))
        print("Foco: LINGERIE y GROCERY II")
        display(difficult_families.round(4))

        profile_sizes = (
            train_demand_profile.groupby("tipo_demanda", as_index=False)
            .agg(
                series=("family", "size"),
                historical_zero_pct=("pct_ceros", "mean"),
            )
        )
        display(profile_sizes.round(4))
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
    md(
        """
        ### Sensibilidad a promociones futuras

        En el escenario de Corporación Favorita se asume que `onpromotion`
        representa un plan promocional conocido antes de los 15 días. En una
        empresa real sólo sería admisible si proviene del calendario comercial
        aprobado; un registro completado retrospectivamente no estaría
        disponible y no debería usarse.

        Como sensibilidad, se mantiene el mismo modelo ya fijado y se vuelve a
        pronosticar el test forzando `onpromotion=0`. No se reentrena ni se usa
        la venta real para construir features. La comparación mide cuánto
        depende el forecast de disponer del plan, no estima causalidad de las
        promociones.
        """
    ),
    code(
        """
        from src.models.recursive_forecast import recursive_forecast

        def without_future_promotions(X_day, keys, current_date):
            X_day["onpromotion"] = 0
            return X_day

        no_promo_predictions = recursive_forecast(
            model=final_result["model"],
            features_base=final_result["prepared"]["features_base"],
            train=final_result["prepared"]["train"],
            validation_start=FINAL_TEST_START,
            validation_end=FINAL_TEST_END,
            model_features=MODEL_FEATURES,
            category_schema=final_result["category_schema"],
            feature_transform=without_future_promotions,
        ).rename(columns={"prediction": "prediction_no_future_promo"})

        promotion_sensitivity = comparison_eval.merge(
            no_promo_predictions,
            on=["date", "store_nbr", "family"],
            how="left",
            validate="one_to_one",
        )
        promotion_rows = []
        for scenario, prediction_col in [
            ("Plan futuro disponible", "prediction"),
            ("Sin información promocional futura", "prediction_no_future_promo"),
        ]:
            promotion_rows.append({
                "scenario": scenario,
                "forecast_total": promotion_sensitivity[prediction_col].sum(),
                **prediction_metrics(
                    promotion_sensitivity["sales"],
                    promotion_sensitivity[prediction_col],
                ),
            })
        promotion_sensitivity_summary = pd.DataFrame(promotion_rows)
        display(promotion_sensitivity_summary.round(4))
        print(
            "Filas con promoción planificada en el horizonte:",
            int(final_result["prepared"]["features_base"].loc[
                lambda x: x["date"].between(FINAL_TEST_START, FINAL_TEST_END),
                "onpromotion",
            ].gt(0).sum()),
        )
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
