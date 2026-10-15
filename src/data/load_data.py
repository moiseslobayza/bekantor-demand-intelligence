import os
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"
REQUIRED_FAVORITA_FILES = {
    "sales": "train.csv",
    "stores": "stores.csv",
    "transactions": "transactions.csv",
    "holidays": "holidays_events.csv",
    "oil": "oil.csv",
}


def resolve_data_dir(data_dir=None):
    """Resuelve una ruta portable, con ``FAVORITA_DATA_DIR`` como override."""

    candidate = data_dir or os.environ.get("FAVORITA_DATA_DIR")
    return Path(candidate).expanduser().resolve() if candidate else DEFAULT_RAW_DATA_PATH


def validate_raw_files(data_dir=None):
    """Falla con un mensaje accionable sin intentar descargar ni pedir secretos."""

    root = resolve_data_dir(data_dir)
    missing = [name for name in REQUIRED_FAVORITA_FILES.values() if not (root / name).is_file()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(
            f"Faltan archivos de Corporacion Favorita en {root}: {joined}. "
            "Defina FAVORITA_DATA_DIR o pase data_dir explícitamente."
        )
    return root

def load_raw_data(data_dir=None):
    data_dir = validate_raw_files(data_dir)

    datasets = {
        key: pd.read_csv(
            data_dir / filename,
            parse_dates=["date"] if key not in {"stores"} else None,
        )
        for key, filename in REQUIRED_FAVORITA_FILES.items()
    }

    return datasets
