import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from utils.ai_bot import billing_followup_bot

# ----------------------------
# CONFIG
# ----------------------------
DATA_PATH = "data/billing_data.csv"
MODEL_PATH = "models/billing_model.json"
SCHEMA_PATH = "models/billing_schema.pkl"
os.makedirs("models", exist_ok=True)

# ----------------------------
# LOAD DATA
# ----------------------------
print(f"📂 Loading Billing Data from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"✅ Loaded {df.shape[0]} rows and {df.shape[1]} columns\n")

# ----------------------------
# DEFINE TARGET & FEATURES
# ----------------------------
target = "paid_on_time"
if target not in df.columns:
    raise ValueError(f"❌ Target column '{target}' not found in dataset.")

X = df.drop(columns=[target])
y = df[target]

# ----------------------------
# HANDLE CATEGORICALS
# ----------------------------
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
print(f"🧩 Found {len(cat_cols)} categorical columns: {cat_cols}")

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

print("✅ All categorical columns encoded.\n")

# ----------------------------
# SPLIT DATA
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"🔹 Train shape: {X_train.shape} | Test shape: {X_test.shape}\n")

# ----------------------------
# TRAIN MODEL (XGBoost)
# ----------------------------
print("🚀 Training XGBoost Billing Model...")

model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)

model.fit(X_train, y_train)
print("✅ Model training complete.\n")

# ----------------------------
# EVALUATE
# ----------------------------
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)

print(f"📊 Evaluation Metrics:")
print(f"   Accuracy: {acc:.4f}")
print(f"   ROC-AUC:  {roc_auc:.4f}\n")

# ----------------------------
# SAVE MODEL & SCHEMA
# ----------------------------
model.save_model(MODEL_PATH)
schema = {"cat_cols": cat_cols, "encoders": encoders, "target": target}
joblib.dump(schema, SCHEMA_PATH)

print(f"💾 Model saved to: {MODEL_PATH}")
print(f"💾 Schema saved to: {SCHEMA_PATH}\n")

# ----------------------------
# AI BOT SIMULATION
# ----------------------------
print("🤖 Running AI Bot for Patient Billing follow-up simulation...")

sample = X_test.sample(1, random_state=42)
patient_id = sample.get("patient_id", pd.Series(["Unknown"])).iloc[0]
balance_due = sample.get("balance_due", pd.Series([0.0])).iloc[0]
risk_score = float(model.predict_proba(sample)[0][1])

ai_response = billing_followup_bot(patient_id, round(risk_score, 3), round(balance_due, 2))

print(f"🧾 Patient ID: {patient_id}")
print(f"💰 Balance Due: ${balance_due}")
print(f"📉 Predicted Payment Risk Score: {risk_score:.3f}")
print(f"🗣️ AI BOT says: {ai_response}\n")

print("🎯 Billing model training complete.\n")
