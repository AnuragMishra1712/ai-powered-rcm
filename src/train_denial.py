import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import os

# ----------------------------
# CONFIG
# ----------------------------
DATA_PATH = "data/denial_data.csv"
MODEL_PATH = "models/denial_model.cbm"
SCHEMA_PATH = "models/denial_schema.pkl"
os.makedirs("models", exist_ok=True)

# ----------------------------
# LOAD DATA
# ----------------------------
print("📂 Loading Denial Data from:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
print(f"✅ Loaded {df.shape[0]} rows and {df.shape[1]} columns\n")

# ----------------------------
# DEFINE TARGET & FEATURES
# ----------------------------
target = "denied"
if target not in df.columns:
    raise ValueError(f"❌ Target column '{target}' not found in dataset.")

X = df.drop(columns=[target])
y = df[target]

# Identify column types
cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

print("🧩 Feature Summary:")
print(f" - Categorical columns: {cat_cols}")
print(f" - Numeric columns: {num_cols}\n")

# ----------------------------
# SPLIT DATA
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"🔹 Train shape: {X_train.shape} | Test shape: {X_test.shape}\n")

# ----------------------------
# CATBOOST MODEL TRAINING
# ----------------------------
print("🚀 Training CatBoost Denial Prediction Model...")
model = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.05,
    eval_metric="AUC",
    cat_features=cat_cols,
    verbose=100
)

train_pool = Pool(X_train, y_train, cat_features=cat_cols)
test_pool = Pool(X_test, y_test, cat_features=cat_cols)

model.fit(train_pool, eval_set=test_pool)

# ----------------------------
# EVALUATE
# ----------------------------
pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, pred)
roc_auc = roc_auc_score(y_test, proba)

print(f"\n✅ Denial Model Trained Successfully!")
print(f"   Accuracy: {acc:.4f}")
print(f"   ROC-AUC:  {roc_auc:.4f}")

# ----------------------------
# SAVE MODEL & SCHEMA
# ----------------------------
model.save_model(MODEL_PATH)
schema = {"cat_cols": cat_cols, "num_cols": num_cols, "target": target}
joblib.dump(schema, SCHEMA_PATH)

print(f"\n💾 Model saved to: {MODEL_PATH}")
print(f"💾 Schema saved to: {SCHEMA_PATH}")
print("🎯 Done!\n")
