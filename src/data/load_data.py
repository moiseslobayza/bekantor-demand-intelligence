from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"

def load_raw_data(data_dir=None):
    if data_dir is None:
        data_dir = DEFAULT_RAW_DATA_PATH
    else:
        data_dir = Path(data_dir)

    datasets = {
        "sales": pd.read_csv(data_dir / "train.csv", parse_dates=["date"]),
        "stores": pd.read_csv(data_dir / "stores.csv"),
        "transactions": pd.read_csv(data_dir / "transactions.csv", parse_dates=["date"]),
        "holidays": pd.read_csv(data_dir / "holidays_events.csv", parse_dates=["date"]),
        "oil": pd.read_csv(data_dir / "oil.csv", parse_dates=["date"]),
    }

    return datasets