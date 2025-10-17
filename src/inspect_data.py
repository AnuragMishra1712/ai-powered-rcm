import os
import json
import pandas as pd

DATA_DIR = "data"
SCHEMA_DIR = "models"
os.makedirs(SCHEMA_DIR, exist_ok=True)

def inspect_csv(filename):
    """Inspect CSV and return feature metadata."""
    path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(path)
    print(f"\n--- {filename} ---")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}\n")
    print(df.dtypes)
    print("\nMissing values per column:")
    print(df.isna().sum())
    print("\nSample rows:")
    print(df.head(3))
    
    # Infer schema
    schema = {}
    for col in df.columns:
        dtype = str(df[col].dtype)
        nunique = df[col].nunique()
        example = df[col].dropna().iloc[0] if df[col].notna().any() else None
        schema[col] = {
            "dtype": dtype,
            "unique_values": int(nunique),
            "example_value": str(example)
        }
    return schema

def main():
    all_schemas = {}
    csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    for f in csv_files:
        schema = inspect_csv(f)
        base = f.replace(".csv", "")
        json_path = os.path.join(SCHEMA_DIR, f"{base}_schema.json")
        with open(json_path, "w") as jf:
            json.dump(schema, jf, indent=4)
        all_schemas[base] = schema
    print("\n✅ All schemas exported to models/*.json")

if __name__ == "__main__":
    main()

