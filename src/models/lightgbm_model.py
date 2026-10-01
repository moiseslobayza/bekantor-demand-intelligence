import lightgbm as lgb
import numpy as np


LIGHTGBM_BASE_PARAMS = {
    "objective": "regression",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 100,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
    "deterministic": True,
    "force_col_wise": True,
}


def make_lgbm_regressor(n_estimators):
    """Crea el primer modelo LightGBM con configuración reproducible."""

    return lgb.LGBMRegressor(
        n_estimators=int(n_estimators),
        **LIGHTGBM_BASE_PARAMS,
    )


def fit_with_early_stopping(
    X_train,
    y_train,
    X_valid,
    y_valid,
    categorical_features,
    max_estimators=1000,
    stopping_rounds=50,
):
    """Selecciona iteraciones sin usar el test final."""

    model = make_lgbm_regressor(max_estimators)
    model.fit(
        X_train,
        np.log1p(y_train),
        eval_X=X_valid,
        eval_y=np.log1p(y_valid),
        eval_metric="l2",
        categorical_feature=categorical_features,
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=stopping_rounds,
                verbose=False,
            )
        ],
    )
    return model, int(model.best_iteration_)


def fit_fixed_iterations(
    X_train,
    y_train,
    categorical_features,
    n_estimators,
):
    """Reentrena con la cantidad de árboles ya fijada."""

    model = make_lgbm_regressor(n_estimators)
    model.fit(
        X_train,
        np.log1p(y_train),
        categorical_feature=categorical_features,
    )
    return model
