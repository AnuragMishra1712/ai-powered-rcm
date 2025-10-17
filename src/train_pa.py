import pandas as pd
import numpy as np
import os
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score

# ---------------------------------------------------------
# ✅ 1. Load Data
# ---------------------------------------------------------
data_path = "data/pa_data.csv"
print(f"📂 Loading PA Data from: {data_path}")
df = pd.read_csv(data_path)
print(f"✅ Loaded {len(df)} rows and {df.shape[1]} columns\n")

# ---------------------------------------------------------
# ✅ 2. Identify categorical features
# ---------------------------------------------------------
cat_cols = [
    "claim_id", "gender", "medical_specialty", "insurance_type",
    "plan_type", "hospital_region", "payer_id", "diagnosis_code", "claim_category"
]

print(f"🧩 Found {len(cat_cols)} categorical columns: {cat_cols}")
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

print("✅ All categorical columns encoded.\n")

# ---------------------------------------------------------
# ✅ 3. Prepare features and target
# ---------------------------------------------------------
target = "pa_required"   # <-- corrected target column
if target not in df.columns:
    raise ValueError(f"❌ Target column '{target}' not found in dataset!")

X = df.drop(columns=[target])
y = df[target].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"🔹 Train shape: {X_train.shape} | Test shape: {X_test.shape}\n")

# ---------------------------------------------------------
# ✅ 4. Train LightGBM
# ---------------------------------------------------------
train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
valid_data = lgb.Dataset(X_test, label=y_test, categorical_feature=cat_cols)

params = {
    "objective": "binary",
    "metric": ["auc", "binary_logloss"],
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 42
}

print("🚀 Training LightGBM PA Model...")
model = lgb.train(
    params,
    train_set=train_data,
    valid_sets=[train_data, valid_data],
    valid_names=["train", "valid"],
    num_boost_round=300,
    callbacks=[
        lgb.early_stopping(stopping_rounds=30),
        lgb.log_evaluation(period=20)
    ]
)

# ---------------------------------------------------------
# ✅ 5. Evaluate Model
# ---------------------------------------------------------
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

roc_auc = roc_auc_score(y_test, y_pred_prob)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ PA Model | acc: {acc:.4f} | roc_auc: {roc_auc:.4f}")

# ---------------------------------------------------------
# ✅ 6. Save Model
# ---------------------------------------------------------
os.makedirs("models", exist_ok=True)
model.save_model("models/pa_model.txt")
joblib.dump(cat_cols, "models/pa_categorical_cols.pkl")

print("\n💾 Model saved:")
print(" - models/pa_model.txt")
print(" - models/pa_categorical_cols.pkl")
print("\n🎯 Prior Authorization model trained and ready!")
