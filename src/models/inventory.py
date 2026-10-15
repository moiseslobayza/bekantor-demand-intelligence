"""Simulador decisional: no presupone inventario histórico de Favorita."""

from statistics import NormalDist

import numpy as np
import pandas as pd


def safety_stock_from_errors(errors, lead_time_days, service_level=0.95):
    """Stock de seguridad normal usando errores diarios de desarrollo."""

    errors = np.asarray(errors, dtype=float)
    if lead_time_days < 0:
        raise ValueError("El plazo de reposición no puede ser negativo")
    if not 0.5 < service_level < 1:
        raise ValueError("service_level debe estar entre 0.5 y 1")
    if errors.size < 2 or lead_time_days == 0:
        return 0.0
    z_value = NormalDist().inv_cdf(service_level)
    return max(0.0, float(z_value * np.std(errors, ddof=1) * np.sqrt(lead_time_days)))


def simulate_inventory(
    forecast,
    actual=None,
    initial_stock=0.0,
    lead_time_days=0,
    safety_stock=0.0,
    unit_cost=1.0,
    holding_cost_per_unit=0.0,
    shortage_penalty_per_unit=0.0,
):
    """Convierte un forecast diario en una recomendación y trayectoria simple.

    ``actual`` representa ventas observadas, no demanda irrestricta; cualquier
    faltante calculado es un escenario hipotético.
    """

    forecast = np.clip(np.asarray(forecast, dtype=float), 0, None)
    if actual is None:
        actual = forecast.copy()
    actual = np.clip(np.asarray(actual, dtype=float), 0, None)
    if len(actual) != len(forecast):
        raise ValueError("forecast y actual deben tener igual longitud")
    if int(lead_time_days) != lead_time_days:
        raise ValueError("lead_time_days debe ser un entero")
    if min(initial_stock, lead_time_days, safety_stock, unit_cost, holding_cost_per_unit, shortage_penalty_per_unit) < 0:
        raise ValueError("Los supuestos de inventario y costos deben ser no negativos")

    # La orden se decide antes del día 1 y llega al inicio del día
    # ``lead_time_days + 1``. Por lo tanto, el stock inicial sólo puede cubrir
    # la demanda previa a la recepción. La reposición debe apuntar a la demanda
    # restante, no a ventas potencialmente perdidas durante el lead time.
    lead_time_days = int(lead_time_days)
    lead_time_demand = float(forecast[:lead_time_days].sum())
    projected_stock_at_receipt = max(0.0, float(initial_stock) - lead_time_demand)
    post_receipt_demand = float(forecast[lead_time_days:].sum())
    target_stock_at_receipt = post_receipt_demand + float(safety_stock)
    replenishment = max(0.0, target_stock_at_receipt - projected_stock_at_receipt)
    on_hand = float(initial_stock)
    rows = []
    for day, (expected, observed) in enumerate(zip(forecast, actual), start=1):
        receipt = replenishment if day == lead_time_days + 1 else 0.0
        on_hand += receipt
        served = min(on_hand, observed)
        shortage = max(0.0, observed - on_hand)
        on_hand = max(0.0, on_hand - observed)
        rows.append(
            {
                "horizon_day": day,
                "forecast": expected,
                "observed_sales": observed,
                "receipt": receipt,
                "served": served,
                "ending_inventory": on_hand,
                "potential_shortage": shortage,
            }
        )
    trajectory = pd.DataFrame(rows)
    summary = {
        "forecast_demand": float(forecast.sum()),
        "simulated_demand": float(actual.sum()),
        "initial_stock_assumption": float(initial_stock),
        "lead_time_days": lead_time_days,
        "safety_stock": float(safety_stock),
        "unit_cost": float(unit_cost),
        "holding_cost_per_unit": float(holding_cost_per_unit),
        "shortage_penalty_per_unit": float(shortage_penalty_per_unit),
        "lead_time_demand": lead_time_demand,
        "projected_stock_at_receipt": projected_stock_at_receipt,
        "target_stock_at_receipt": target_stock_at_receipt,
        # Alias conservado para no romper consumidores previos. Desde V2.1
        # significa el objetivo al momento de recepción, no al día de decisión.
        "target_stock": target_stock_at_receipt,
        "recommended_replenishment": replenishment,
        "ending_inventory": float(trajectory["ending_inventory"].iloc[-1]),
        "potential_shortage": float(trajectory["potential_shortage"].sum()),
        "purchase_cost": replenishment * unit_cost,
        "holding_cost": float(trajectory["ending_inventory"].sum()) * holding_cost_per_unit,
        "shortage_cost": float(trajectory["potential_shortage"].sum()) * shortage_penalty_per_unit,
    }
    summary["simulated_total_cost"] = summary["purchase_cost"] + summary["holding_cost"] + summary["shortage_cost"]
    return summary, trajectory


def inventory_sensitivity(forecast, error_pct=0.15, base_demand=None, **kwargs):
    """Compara una demanda común base y ±error; no son intervalos de confianza."""

    forecast = np.asarray(forecast, dtype=float)
    if base_demand is None:
        base_demand = forecast.copy()
    base_demand = np.asarray(base_demand, dtype=float)
    if len(base_demand) != len(forecast):
        raise ValueError("base_demand y forecast deben tener igual longitud")
    rows = []
    for label, factor in [
        ("high_demand", 1 + error_pct),
        ("observed_base", 1.0),
        ("low_demand", 1 - error_pct),
    ]:
        summary, _ = simulate_inventory(
            forecast, actual=base_demand * factor, **kwargs
        )
        rows.append({"scenario": label, "demand_multiplier": factor, **summary})
    return pd.DataFrame(rows)
