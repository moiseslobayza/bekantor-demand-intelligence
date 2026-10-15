"""Optimización LightGBM evaluada con pronósticos recursivos completos."""

from pathlib import Path

import numpy as np

from src.features.model_features import CATEGORICAL_FEATURES, MODEL_FEATURES
from src.models.baseline import rmsle
from src.models.categorical import apply_category_schema, fit_category_schema
from src.models.lightgbm_model import fit_fixed_iterations, fit_with_early_stopping
from src.models.recursive_forecast import recursive_forecast
from src.models.v2_evaluation import validate_prediction_coverage


def suggest_lightgbm_params(trial):
    max_depth = trial.suggest_int("max_depth", 4, 10)
    return {
        "num_leaves": trial.suggest_int("num_leaves", 16, min(255, 2**max_depth)),
        "max_depth": max_depth,
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 30, 250),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "random_state": 42,
        "deterministic": True,
        "force_col_wise": True,
    }


def _fold_score(prepared, params, max_estimators, stopping_rounds):
    train_features = prepared["train_features"]
    outer_train_end = prepared["train"]["date"].max()
    inner_selection_date = outer_train_end - np.timedelta64(14, "D")
    inner_train = train_features.loc[
        train_features["date"].lt(inner_selection_date)
    ].copy()
    inner_selection = train_features.loc[
        train_features["date"].eq(inner_selection_date)
    ].copy()
    if inner_selection.empty:
        raise ValueError("No hay ventana interna para early stopping")

    X_inner_train, inner_schema = fit_category_schema(
        inner_train[MODEL_FEATURES].copy(), CATEGORICAL_FEATURES
    )
    X_selection, unknown = apply_category_schema(
        inner_selection[MODEL_FEATURES].copy(), inner_schema
    )
    if unknown:
        raise AssertionError(f"Categorías desconocidas: {unknown}")
    _, best_iteration = fit_with_early_stopping(
        X_inner_train,
        inner_train["sales"],
        X_selection,
        inner_selection["sales"],
        CATEGORICAL_FEATURES,
        max_estimators=max_estimators,
        stopping_rounds=stopping_rounds,
        params=params,
    )
    X_train, schema = fit_category_schema(
        train_features[MODEL_FEATURES].copy(), CATEGORICAL_FEATURES
    )
    model = fit_fixed_iterations(
        X_train,
        train_features["sales"],
        CATEGORICAL_FEATURES,
        best_iteration,
        params=params,
    )
    validation = prepared["validation"]
    prediction = recursive_forecast(
        model,
        prepared["features_base"],
        prepared["train"],
        validation["date"].min(),
        validation["date"].max(),
        MODEL_FEATURES,
        schema,
    )
    validate_prediction_coverage(validation, prediction)
    scored = validation.merge(
        prediction, on=["date", "store_nbr", "family"], validate="one_to_one"
    )
    return rmsle(scored["sales"], scored["prediction"]), int(best_iteration)


def optimize_lightgbm_recursive(
    prepared_folds,
    n_trials=20,
    timeout=None,
    max_estimators=1500,
    stopping_rounds=50,
    storage=None,
    study_name="bekantor-lightgbm-v2",
    random_seed=42,
):
    """Optimiza RMSLE medio; ``prepared_folds`` debe excluir agosto y dev-eval."""

    import optuna

    if not prepared_folds:
        raise ValueError("Se requiere al menos un fold de tuning")
    for prepared in prepared_folds:
        if prepared["validation"]["date"].max() >= np.datetime64("2017-08-01"):
            raise ValueError("Optuna no puede recibir agosto ni fechas posteriores")

    sampler = optuna.samplers.TPESampler(seed=random_seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=min(5, n_trials), n_warmup_steps=1)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        study_name=study_name,
        load_if_exists=bool(storage),
    )

    def objective(trial):
        params = suggest_lightgbm_params(trial)
        scores, iterations = [], []
        for fold_index, prepared in enumerate(prepared_folds):
            score, best_iteration = _fold_score(
                prepared, params, max_estimators, stopping_rounds
            )
            scores.append(score)
            iterations.append(best_iteration)
            trial.report(float(np.mean(scores)), step=fold_index)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("fold_rmsle", scores)
        trial.set_user_attr("best_iterations", iterations)
        trial.set_user_attr("selected_n_estimators", int(np.median(iterations)))
        return float(np.mean(scores))

    terminal_states = {
        optuna.trial.TrialState.COMPLETE,
        optuna.trial.TrialState.PRUNED,
    }
    existing_trials = sum(trial.state in terminal_states for trial in study.trials)
    remaining_trials = max(0, int(n_trials) - existing_trials)
    if remaining_trials:
        # Optuna no persiste el estado RNG del sampler. Al reanudar, desplazar
        # la semilla evita repetir desde el inicio la misma secuencia aleatoria.
        if existing_trials:
            study.sampler = optuna.samplers.TPESampler(
                seed=random_seed + existing_trials
            )
        study.optimize(
            objective,
            n_trials=remaining_trials,
            timeout=timeout,
            gc_after_trial=True,
        )
    return study


def save_study_results(study, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trials_path = output_dir / "optuna_trials.csv"
    study.trials_dataframe().to_csv(trials_path, index=False)
    best = {
        **study.best_params,
        "objective_recursive_rmsle": study.best_value,
        "selected_n_estimators": study.best_trial.user_attrs.get("selected_n_estimators"),
        "tuning_fold_best_iterations": study.best_trial.user_attrs.get("best_iterations"),
        "tuning_fold_rmsle": study.best_trial.user_attrs.get("fold_rmsle"),
    }
    return best, trials_path
