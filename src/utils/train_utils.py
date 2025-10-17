import os
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# Helper to evaluate models
# ============================================================
def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    proba = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else preds
    )

    return {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds), 4),
        "recall": round(recall_score(y_test, preds), 4),
        "f1": round(f1_score(y_test, preds), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
    }


# ============================================================
# Tabular Model Training
# ============================================================
def train_tabular_models(
    df,
    target_col,
    num_cols,
    cat_cols,
    bool_cols,
    model_dir,
    model_name,
    test_size=0.2,
):
    os.makedirs(model_dir, exist_ok=True)

    # --- Preprocess ---
    df = df.copy()
    df[bool_cols] = df[bool_cols].astype(int)
    X = df[num_cols + cat_cols + bool_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # --- Prepare numeric + categorical pipelines ---
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="passthrough",
    )

    # --- Candidate models ---
    models = {
        "LogisticRegression": Pipeline(
            [("pre", preprocessor), ("clf", LogisticRegression(max_iter=500))]
        ),
        "RandomForest": Pipeline(
            [("pre", preprocessor), ("clf", RandomForestClassifier(n_estimators=200))]
        ),
        "LightGBM": Pipeline(
            [("pre", preprocessor), ("clf", LGBMClassifier(n_estimators=300))]
        ),
        "XGBoost": Pipeline(
            [("pre", preprocessor), ("clf", XGBClassifier(use_label_encoder=False, eval_metric="logloss"))]
        ),
    }

    # CatBoost handles categoricals directly
    catboost_model = CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.1, verbose=False
    )
    models["CatBoost"] = catboost_model

    # --- Train and Evaluate ---
    results = {}
    for name, model in models.items():
        print(f"🧠 Training {name} for {model_name} ...")
        if name == "CatBoost":
            model.fit(
                X_train, y_train, cat_features=cat_cols, eval_set=(X_test, y_test)
            )
        else:
            model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)
        results[name] = metrics
        print(f"{name} metrics: {metrics}")

    # --- Choose best model ---
    best_model_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = models[best_model_name]
    print(f"✅ Best model for {model_name}: {best_model_name}")

    # --- Save model + schema + metrics ---
    joblib.dump(best_model, os.path.join(model_dir, f"{model_name}_model.pkl"))

    schema = {"num_cols": num_cols, "cat_cols": cat_cols, "bool_cols": bool_cols}
    with open(os.path.join(model_dir, f"{model_name}_schema.json"), "w") as f:
        json.dump(schema, f, indent=4)

    with open(os.path.join(model_dir, f"{model_name}_metrics.json"), "w") as f:
        json.dump(results, f, indent=4)

    print("📦 Model, schema, and metrics saved successfully!\n")
    return best_model_name, results
