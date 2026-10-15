"""Reconstruye el notebook académico V2 desde la corrida completa."""

from pathlib import Path
import json
import textwrap

import nbformat as nbf
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs" / "forecasting_v2_full"
NOTEBOOK = ROOT / "notebooks" / "06_forecasting_v2.ipynb"


def md(value):
    return nbf.v4.new_markdown_cell(textwrap.dedent(value).strip())


def code(value):
    return nbf.v4.new_code_cell(textwrap.dedent(value).strip())


required = [
    "run_manifest.json", "best_lightgbm.json", "catboost_iterations.json",
    "folds.csv", "fold_metrics.csv", "model_comparison.csv", "optuna_trials.csv",
    "tsb_parameter_search.csv", "segment_diagnostics.csv",
    "promotion_scenarios.csv", "inventory_example_summary.csv",
    "inventory_example_trajectory.csv", "inventory_sensitivity.csv",
]
missing = [name for name in required if not (OUTPUTS / name).is_file()]
if missing:
    raise FileNotFoundError(f"Faltan artefactos de la corrida completa: {missing}")

manifest = json.loads((OUTPUTS / "run_manifest.json").read_text(encoding="utf-8"))
best = json.loads((OUTPUTS / "best_lightgbm.json").read_text(encoding="utf-8"))
cat_budget = json.loads((OUTPUTS / "catboost_iterations.json").read_text(encoding="utf-8"))
comparison = pd.read_csv(OUTPUTS / "model_comparison.csv")
winner = comparison.sort_values("RMSLE_mean").iloc[0]
optimized = comparison.loc[comparison["model"].eq("LightGBM optimized")].iloc[0]
original = comparison.loc[comparison["model"].eq("LightGBM original")].iloc[0]
catboost = comparison.loc[comparison["model"].eq("CatBoost")].iloc[0]
tsb = comparison.loc[comparison["model"].eq("TSB")].iloc[0]
absolute_gain = original["RMSLE_mean"] - optimized["RMSLE_mean"]
relative_gain_pct = absolute_gain / original["RMSLE_mean"] * 100
cat_cap_hits = sum(
    value >= cat_budget["max_iterations"]
    for value in cat_budget["fold_best_iterations"]
)

cells = [
    md(
        f"""
        # BEKANTOR Demand Intelligence — Forecasting V2

        ## tl;dr

        Esta es la **corrida completa**: {manifest['trials']} trials de Optuna,
        máximo {manifest['max_estimators']} árboles LightGBM y CatBoost congelado
        en {manifest['catboost_iterations']} iteraciones después de calibrarlo sólo
        con los cuatro folds de tuning. Los dos folds externos terminan el
        31/07/2017; **agosto no fue usado ni evaluado**.

        **LightGBM optimizado** obtuvo el menor RMSLE externo medio:
        **{optimized['RMSLE_mean']:.6f}**, frente a **{original['RMSLE_mean']:.6f}**
        del LightGBM original. La diferencia es **{absolute_gain:.6f}**
        ({relative_gain_pct:.2f}% relativa) y favorece al optimizado en ambos folds,
        pero sigue siendo pequeña y se basa en sólo dos ventanas externas.

        CatBoost obtuvo **{catboost['RMSLE_mean']:.6f}** y TSB
        **{tsb['RMSLE_mean']:.6f}**. El simulador de inventario fue recalculado con
        stock inicial, stock de seguridad, demanda y costos comunes para todos los
        modelos, de modo que las diferencias reflejen sus decisiones de forecast.
        """
    ),
    md(
        """
        ## Contexto y métodos

        Cada fold deriva su universo desde su cutoff y pronostica 15 días de forma
        recursiva. Optuna observa exclusivamente los folds 1–4; los folds 5–6 se
        reservan para comparación externa. El número de árboles LightGBM se
        calibra en una ventana interna y la función objetivo usa el horizonte
        recursivo completo.

        ### Supuestos clave

        - La comparación externa exige las mismas claves fecha–tienda–familia para todos los modelos.
        - El escenario promocional parcial confirma 7 días y estima el resto con historia anterior al cutoff.
        - El caso de inventario usa una política comercial común: stock inicial igual al 25% del forecast del baseline estacional; stock de seguridad común al 95%, estimado con sus errores del fold externo anterior; lead time de 3 días; costos idénticos.
        - Las ventas observadas son una demanda de comparación común, no evidencia de demanda irrestricta ni de faltantes históricos.
        """
    ),
    code(
        """
        from pathlib import Path
        import json
        import platform

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        ROOT = Path.cwd()
        OUT = ROOT / "outputs" / "forecasting_v2_full"
        plt.style.use("seaborn-v0_8-whitegrid")
        MODEL_COLORS = {
            "LightGBM optimized": "#2563eb",
            "LightGBM original": "#d97706",
            "CatBoost": "#7c3aed",
            "Seasonal baseline": "#64748b",
            "TSB": "#0f766e",
        }
        pd.set_option("display.max_columns", 40)

        manifest = json.loads((OUT / "run_manifest.json").read_text(encoding="utf-8"))
        best = json.loads((OUT / "best_lightgbm.json").read_text(encoding="utf-8"))
        cat_budget = json.loads((OUT / "catboost_iterations.json").read_text(encoding="utf-8"))
        assert manifest["protocol"] == "full"
        assert manifest["august_evaluated"] is False
        print("Python:", platform.python_version())
        print("Duración total (h):", round(manifest["elapsed_seconds"] / 3600, 2))
        print("LightGBM seleccionado (árboles):", best["selected_n_estimators"])
        print("CatBoost seleccionado (iteraciones):", cat_budget["selected_iterations"])
        """
    ),
    md("## Datos y contrato temporal"),
    code(
        """
        folds = pd.read_csv(OUT / "folds.csv", parse_dates=["train_end", "validation_start", "validation_end"])
        assert len(folds) == 6
        assert folds["role"].tolist() == ["tuning"] * 4 + ["development_evaluation"] * 2
        assert folds["validation_end"].max() < pd.Timestamp("2017-08-01")
        display(folds)
        """
    ),
    md("## Resultados"),
    md("### Optimización LightGBM, CatBoost y TSB"),
    code(
        """
        trials = pd.read_csv(OUT / "optuna_trials.csv")
        tsb_search = pd.read_csv(OUT / "tsb_parameter_search.csv")
        print("Estados Optuna:", trials["state"].value_counts().to_dict())
        display(pd.DataFrame([best]).round(6))
        display(pd.DataFrame([cat_budget]))
        display(tsb_search.round(6))

        fig, ax = plt.subplots(figsize=(9, 4.5))
        state_colors = {"COMPLETE": "#2563eb", "PRUNED": "#94a3b8"}
        for state, group in trials.groupby("state"):
            ax.scatter(group["number"], group["value"], s=65,
                       color=state_colors[state], label=state.title())
        ax.set(title="Optuna — RMSLE recursivo medio en 4 folds de tuning",
               xlabel="Trial", ylabel="RMSLE medio")
        ax.set_xticks(range(0, len(trials), 2))
        ax.legend(frameon=False)
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        f"""
        CatBoost llegó al máximo de {cat_budget['max_iterations']} iteraciones en
        {cat_cap_hits} de 4 ventanas internas. Su resultado externo sigue siendo
        una medición válida del modelo congelado en {cat_budget['selected_iterations']}
        iteraciones, pero no demuestra que ése sea el presupuesto óptimo de CatBoost.
        """
    ),
    md("### Comparación consolidada en desarrollo externo"),
    code(
        """
        comparison = pd.read_csv(OUT / "model_comparison.csv").sort_values("RMSLE_mean")
        display(comparison.round(6))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        colors = [MODEL_COLORS[name] for name in comparison["model"]]
        axes[0].barh(comparison["model"], comparison["RMSLE_mean"],
                     xerr=comparison["RMSLE_std"], color=colors, capsize=3)
        axes[0].invert_yaxis()
        axes[0].set(title="RMSLE medio ± desvío entre folds externos",
                    xlabel="RMSLE", ylabel="")
        axes[0].set_xlim(left=0)
        axes[1].barh(comparison["model"], comparison["WAPE_pct_mean"], color=colors)
        axes[1].invert_yaxis()
        axes[1].set(title="WAPE medio — folds externos", xlabel="WAPE (%)", ylabel="")
        axes[1].set_xlim(left=0)
        plt.tight_layout()
        plt.show()
        """
    ),
    code(
        """
        fold_metrics = pd.read_csv(OUT / "fold_metrics.csv")
        heldout = fold_metrics.loc[fold_metrics["role"].eq("development_evaluation")]
        pivot = heldout.pivot(index="fold", columns="model", values="RMSLE")
        ax = pivot.plot(marker="o", figsize=(10, 5),
                        color=[MODEL_COLORS[name] for name in pivot.columns])
        ax.set(title="RMSLE por fold externo", xlabel="Fold", ylabel="RMSLE")
        ax.set_xticks([5, 6])
        plt.tight_layout()
        plt.show()
        display(heldout.round(6))
        """
    ),
    md("### Población común y comportamiento por horizonte"),
    code(
        """
        segments = pd.read_csv(OUT / "segment_diagnostics.csv")
        horizon = segments.loc[segments["dimension"].eq("horizon")].copy()
        horizon["segment"] = horizon["segment"].astype(int)
        coverage = horizon.groupby(["fold", "model"], as_index=False).agg(
            observations=("n_obs", "sum"), horizon_days=("segment", "nunique"),
            series=("n_series", "max"),
        )
        assert coverage.groupby("fold")["observations"].nunique().eq(1).all()
        assert coverage["horizon_days"].eq(15).all()
        display(coverage)

        horizon_mean = horizon.groupby(["segment", "model"], as_index=False)["RMSLE"].mean()
        horizon_pivot = horizon_mean.pivot(index="segment", columns="model", values="RMSLE")
        ax = horizon_pivot.plot(marker="o", figsize=(10, 5),
                                color=[MODEL_COLORS[name] for name in horizon_pivot.columns])
        ax.set(title="RMSLE por día — media de los 2 folds externos",
               xlabel="Día del horizonte", ylabel="RMSLE")
        plt.tight_layout()
        plt.show()
        """
    ),
    md("### Familias y tipos ADI/CV²"),
    code(
        """
        focus = segments.loc[
            ((segments["dimension"] == "family") & segments["segment"].isin(["LINGERIE", "GROCERY II"]))
            | ((segments["dimension"] == "demand_type") & segments["segment"].isin(["Intermitente", "Lumpy"]))
        ]
        display(focus[["fold", "model", "dimension", "segment", "RMSLE", "MAE",
                       "WAPE_pct", "Bias_pct", "n_obs", "n_series"]].round(4))

        demand_type = focus.loc[focus["dimension"].eq("demand_type")].groupby(
            ["segment", "model"], as_index=False
        )["WAPE_pct"].mean()
        chart = demand_type.pivot(index="segment", columns="model", values="WAPE_pct")
        chart.plot(kind="bar", figsize=(11, 5),
                   color=[MODEL_COLORS[name] for name in chart.columns])
        plt.title("WAPE por tipo de demanda — media de 2 folds externos")
        plt.xlabel("Tipo ADI/CV²")
        plt.ylabel("WAPE (%)")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()
        """
    ),
    md("### Escenarios de promociones"),
    code(
        """
        promotions = pd.read_csv(OUT / "promotion_scenarios.csv")
        display(promotions.round(4))
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].bar(promotions["scenario"], promotions["RMSLE"], color="#2563eb")
        axes[0].set(title="RMSLE por información promocional", ylabel="RMSLE")
        axes[0].set_ylim(bottom=0)
        axes[1].bar(promotions["scenario"], promotions["forecast_total"] / 1e6,
                    color="#d97706")
        axes[1].set(title="Forecast acumulado", ylabel="Millones de unidades")
        axes[1].set_ylim(bottom=0)
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        La sensibilidad promocional no estima causalidad. Mide el deterioro del
        mismo sistema cuando promociones no confirmadas se sustituyen por tasas
        históricas calculadas únicamente hasta el cutoff.
        """
    ),
    md("### Simulación de inventario con política común"),
    code(
        """
        inventory = pd.read_csv(OUT / "inventory_example_summary.csv")
        trajectory = pd.read_csv(OUT / "inventory_example_trajectory.csv")
        sensitivity = pd.read_csv(OUT / "inventory_sensitivity.csv")
        common = ["initial_stock_assumption", "safety_stock", "service_level",
                  "lead_time_days", "unit_cost", "holding_cost_per_unit",
                  "shortage_penalty_per_unit", "simulated_demand"]
        assert all(inventory[column].nunique() == 1 for column in common)
        display(inventory[["model", *common, "forecast_demand",
                           "recommended_replenishment", "ending_inventory",
                           "potential_shortage", "simulated_total_cost"]].round(2))
        display(sensitivity[["model", "scenario", "demand_multiplier",
                             "simulated_demand", "potential_shortage",
                             "ending_inventory", "simulated_total_cost"]].round(2))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        for model, group in trajectory.groupby("model"):
            axes[0].plot(group["horizon_day"], group["ending_inventory"],
                         marker="o", label=model, color=MODEL_COLORS[model])
        axes[0].set(title="Inventario final diario — demanda observada común",
                    xlabel="Día del horizonte", ylabel="Unidades")
        axes[0].legend(ncol=2, frameon=False)
        order = inventory.sort_values("simulated_total_cost")
        axes[1].barh(order["model"], order["simulated_total_cost"],
                     color=[MODEL_COLORS[name] for name in order["model"]])
        axes[1].invert_yaxis()
        axes[1].set(title="Costo simulado — mismos supuestos",
                    xlabel="Unidades monetarias", ylabel="")
        axes[1].set_xlim(left=0)
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        f"""
        ## Takeaways

        1. LightGBM optimizado mejora al original en los dos folds externos y
           reduce el RMSLE medio de {original['RMSLE_mean']:.6f} a
           {optimized['RMSLE_mean']:.6f}; la mejora relativa es {relative_gain_pct:.2f}%.
        2. La ventaja es consistente pero pequeña. Dos folds externos no permiten
           afirmar significancia estadística ni garantizan superioridad futura.
        3. CatBoost y TSB no superan a ambos LightGBM globalmente. CatBoost debe
           interpretarse con la salvedad de que 2/4 calibraciones internas tocaron
           el máximo de 1.000 iteraciones.
        4. Menor confirmación promocional empeora RMSLE y reduce el forecast total;
           este resultado es sensibilidad predictiva, no efecto causal.
        5. Con inventario, demanda, lead time, servicio y costos comunes, LightGBM
           optimizado produce el menor costo del caso demostrativo. Sigue siendo
           un escenario hipotético, no una medición de inventario real.

        ## Limitaciones

        - Sólo dos folds externos comparan los cinco modelos costosos.
        - Agosto no fue evaluado ni utilizado para seleccionar modelos.
        - Las ventas observadas pueden estar censuradas por faltantes desconocidos.
        - La política de inventario usa un baseline común porque Favorita no aporta stock histórico.
        - Los escenarios ±15% son estrés operativo, no intervalos de confianza.
        """
    ),
]

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": manifest["python"]},
    },
)
nbf.validate(notebook)
nbf.write(notebook, NOTEBOOK)
print(f"Notebook reconstruido: {NOTEBOOK}")
