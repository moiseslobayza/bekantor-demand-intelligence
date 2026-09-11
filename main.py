from src.data.load_data import load_raw_data
from src.data.audit_data import build_audit_summary
from src.data.clean_data import minimal_cleaning


datasets = load_raw_data()

print("\nAUDITORÍA - DATOS CRUDOS\n")
print(build_audit_summary(datasets).to_string(index=False))

cleaned_datasets = minimal_cleaning(datasets)

print("\nAUDITORÍA - DESPUÉS DE LIMPIEZA MÍNIMA\n")
print(build_audit_summary(cleaned_datasets).to_string(index=False))