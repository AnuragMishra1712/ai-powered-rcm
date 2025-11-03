import pandas as pd
from sklearn.preprocessing import LabelEncoder


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Converts all object/category columns to numeric labels."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object" or str(df[col].dtype).startswith("category"):
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    return df


def align_features(df: pd.DataFrame, expected_features: list[str]) -> pd.DataFrame:
    """Ensures feature order and missing columns match training schema."""
    df = df.copy()
    for f in expected_features:
        if f not in df.columns:
            df[f] = 0
    df = df[expected_features]
    return df
