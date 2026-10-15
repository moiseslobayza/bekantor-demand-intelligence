"""Adaptador CatBoost con categóricas explícitas y salida en escala log1p."""

import numpy as np
import pandas as pd

from src.features.model_features import CATEGORICAL_FEATURES, MODEL_FEATURES


CATBOOST_BASE_PARAMS = {
    "loss_function": "RMSE",
    "learning_rate": 0.05,
    "depth": 8,
    "l2_leaf_reg": 5.0,
    "random_seed": 42,
    "thread_count": -1,
    "verbose": False,
    "allow_writing_files": False,
}


def _catboost_frame(data, categorical_features):
    frame = data.copy()
    for column in categorical_features:
        frame[column] = frame[column].astype("string").fillna("__MISSING__").astype(str)
    return frame


class CatBoostLogRegressor:
    """Normaliza dtypes para que entrenamiento y forecast sean idénticos."""

    def __init__(self, model, categorical_features):
        self.model = model
        self.categorical_features = tuple(categorical_features)
        best_iteration = model.get_best_iteration()
        self.best_iteration_ = (
            int(best_iteration) + 1
            if best_iteration is not None and best_iteration >= 0
            else int(model.tree_count_)
        )

    def predict(self, X):
        return self.model.predict(_catboost_frame(X, self.categorical_features))


def fit_catboost(
    X_train,
    y_train,
    categorical_features,
    n_estimators=800,
    X_valid=None,
    y_valid=None,
    stopping_rounds=50,
    params=None,
):
    """Entrena CatBoost sin target statistics calculadas externamente."""

    from catboost import CatBoostRegressor

    model_params = {**CATBOOST_BASE_PARAMS, **(params or {})}
    model = CatBoostRegressor(iterations=int(n_estimators), **model_params)
    X_train_cb = _catboost_frame(X_train, categorical_features)
    fit_kwargs = {"cat_features": list(categorical_features)}
    if X_valid is not None:
        fit_kwargs.update(
            {
                "eval_set": (
                    _catboost_frame(X_valid, categorical_features),
                    np.log1p(y_valid),
                ),
                "early_stopping_rounds": int(stopping_rounds),
                "use_best_model": True,
            }
        )
    model.fit(X_train_cb, np.log1p(y_train), **fit_kwargs)
    return CatBoostLogRegressor(model, categorical_features)


def select_catboost_iterations(
    prepared_folds,
    max_iterations=1000,
    stopping_rounds=50,
):
    """Calibra árboles sólo con ventanas internas de los folds de tuning.

    Se usa un único día interno cuyos lags son completamente observados. Así el
    detector de sobreajuste no ve el horizonte externo ni recibe lags con
    ventas reales posteriores a su cutoff. Los parámetros quedan fijos; sólo se
    selecciona un presupuesto robusto mediante la mediana entre folds.
    """

    best_iterations = []
    for prepared in prepared_folds:
        train_features = prepared["train_features"]
        outer_train_end = pd.Timestamp(prepared["train"]["date"].max())
        inner_selection_date = outer_train_end - pd.Timedelta(days=14)
        inner_train = train_features.loc[
            train_features["date"].lt(inner_selection_date)
        ]
        inner_selection = train_features.loc[
            train_features["date"].eq(inner_selection_date)
        ]
        if inner_selection.empty:
            raise ValueError("No hay ventana interna para calibrar CatBoost")
        model = fit_catboost(
            inner_train[MODEL_FEATURES],
            inner_train["sales"],
            CATEGORICAL_FEATURES,
            n_estimators=max_iterations,
            X_valid=inner_selection[MODEL_FEATURES],
            y_valid=inner_selection["sales"],
            stopping_rounds=stopping_rounds,
        )
        best_iterations.append(int(model.best_iteration_))
    selected = max(1, int(np.median(best_iterations)))
    return selected, best_iterations
