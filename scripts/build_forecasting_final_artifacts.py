"""Genera el escenario final de inventario desde artefactos V2 existentes."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.inventory_closure import write_inventory_stress_artifacts


def main() -> None:
    source_dir = ROOT / "outputs" / "forecasting_v2_full"
    output_dir = ROOT / "outputs" / "forecasting_final"
    summary, _, assumptions = write_inventory_stress_artifacts(
        source_dir,
        output_dir,
        initial_stock_fraction=0.10,
    )
    first_shortage = int(summary["first_shortage_day"].dropna().min())
    shortage = float(summary["potential_shortage"].iloc[0])
    print(f"Artefactos de cierre: {output_dir}")
    print(f"Stock inicial supuesto: {assumptions['initial_stock_assumption']:.3f}")
    print(f"Primer faltante simulado: día {first_shortage}")
    print(f"Unidades potencialmente no atendidas: {shortage:.3f}")


if __name__ == "__main__":
    main()
