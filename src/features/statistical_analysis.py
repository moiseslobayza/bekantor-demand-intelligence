import numpy as np
import pandas as pd


def bootstrap_ci(
    values,
    statistic=np.median,
    n_boot=2000,
    confidence=0.95,
    random_state=42,
):
    """Calcula un intervalo bootstrap para una estadística univariada.

    Se eliminan NaN antes del remuestreo. Devuelve (estimación, límite inferior,
    límite superior, n_observaciones).
    """
    values = pd.Series(values, dtype="float64").dropna().to_numpy()
    n = len(values)

    if n == 0:
        return np.nan, np.nan, np.nan, 0

    estimate = float(statistic(values))

    if n == 1:
        return estimate, estimate, estimate, 1

    rng = np.random.default_rng(random_state)
    indices = rng.integers(0, n, size=(n_boot, n))
    samples = values[indices]
    boot_stats = np.apply_along_axis(statistic, 1, samples)

    alpha = 1 - confidence
    low = float(np.quantile(boot_stats, alpha / 2))
    high = float(np.quantile(boot_stats, 1 - alpha / 2))

    return estimate, low, high, n


def bootstrap_ratio_ci(
    numerator,
    denominator,
    statistic=np.mean,
    n_boot=2000,
    confidence=0.95,
    random_state=42,
):
    """Intervalo bootstrap del cociente statistic(numerador)/statistic(denominador).

    El remuestreo se realiza de manera independiente en ambos grupos. Es útil
    para cuantificar incertidumbre de lifts descriptivos sin asumir normalidad.
    """
    numerator = pd.Series(numerator, dtype="float64").dropna().to_numpy()
    denominator = pd.Series(denominator, dtype="float64").dropna().to_numpy()

    if len(numerator) == 0 or len(denominator) == 0:
        return np.nan, np.nan, np.nan, len(numerator), len(denominator)

    denominator_stat = statistic(denominator)
    if denominator_stat == 0:
        return np.nan, np.nan, np.nan, len(numerator), len(denominator)

    estimate = float(statistic(numerator) / denominator_stat)
    rng = np.random.default_rng(random_state)

    num_idx = rng.integers(0, len(numerator), size=(n_boot, len(numerator)))
    den_idx = rng.integers(0, len(denominator), size=(n_boot, len(denominator)))

    num_stats = np.apply_along_axis(statistic, 1, numerator[num_idx])
    den_stats = np.apply_along_axis(statistic, 1, denominator[den_idx])

    valid = den_stats != 0
    boot_ratio = num_stats[valid] / den_stats[valid]

    alpha = 1 - confidence
    low = float(np.quantile(boot_ratio, alpha / 2))
    high = float(np.quantile(boot_ratio, 1 - alpha / 2))

    return estimate, low, high, len(numerator), len(denominator)
