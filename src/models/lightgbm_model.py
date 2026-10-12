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


class _NativeLightGBMRegressor:
    """Adaptador mínimo para usar la API nativa si sklearn no está disponible."""

    def __init__(self, booster, best_iteration):
        self.booster_ = booster
        self.best_iteration_ = int(best_iteration)

    def predict(self, X):
        return self.booster_.predict(X, num_iteration=self.best_iteration_)


def _native_train(
    X_train,
    y_train,
    categorical_features,
    n_estimators,
    X_valid=None,
    y_valid=None,
    stopping_rounds=None,
):
    """Entrena el mismo LightGBM mediante su API nativa, sin sklearn."""

    train_set = lgb.Dataset(
        X_train,
        label=np.log1p(y_train),
        categorical_feature=categorical_features,
        free_raw_data=False,
    )
    valid_sets = None
    callbacks = None
    if X_valid is not None:
        valid_set = lgb.Dataset(
            X_valid,
            label=np.log1p(y_valid),
            categorical_feature=categorical_features,
            reference=train_set,
            free_raw_data=False,
        )
        valid_sets = [valid_set]
        callbacks = [
            lgb.early_stopping(stopping_rounds=stopping_rounds, verbose=False)
        ]

    booster = lgb.train(
        params=LIGHTGBM_BASE_PARAMS,
        train_set=train_set,
        num_boost_round=int(n_estimators),
        valid_sets=valid_sets,
        callbacks=callbacks,
    )
    best_iteration = booster.best_iteration or int(n_estimators)
    return _NativeLightGBMRegressor(booster, best_iteration)


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

    if not lgb.compat.SKLEARN_INSTALLED:
        model = _native_train(
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            categorical_features=categorical_features,
            n_estimators=max_estimators,
            stopping_rounds=stopping_rounds,
        )
        return model, model.best_iteration_

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

    if not lgb.compat.SKLEARN_INSTALLED:
        return _native_train(
            X_train=X_train,
            y_train=y_train,
            categorical_features=categorical_features,
            n_estimators=n_estimators,
        )

    model = make_lgbm_regressor(n_estimators)
    model.fit(
        X_train,
        np.log1p(y_train),
        categorical_feature=categorical_features,
    )
    return model
