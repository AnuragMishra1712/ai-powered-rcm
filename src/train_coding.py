import pandas as pd
import numpy as np
import os
import ast
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, f1_score
from lightgbm import LGBMClassifier

# ---------------------------------------------------------
# ✅ 1. Load Data
# ---------------------------------------------------------
data_path = "data/coding_data.csv"
print(f"📂 Loading Coding Data from: {data_path}")

df = pd.read_csv(data_path)
print(f"✅ Loaded {len(df)} rows and {df.shape[1]} columns\n")

# Parse lists safely (CPT and ICD columns are stringified lists)
for col in ["final_billed_cpt", "final_icd10"]:
    df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

# ---------------------------------------------------------
# ✅ 2. Prepare Inputs
# ---------------------------------------------------------
text_col = "note_text"
y_cpt = df["final_billed_cpt"]
y_icd = df["final_icd10"]

X_train, X_test, y_train_cpt, y_test_cpt = train_test_split(df[text_col], y_cpt, test_size=0.2, random_state=42)
X_train2, X_test2, y_train_icd, y_test_icd = train_test_split(df[text_col], y_icd, test_size=0.2, random_state=42)

# TF-IDF Vectorization
print("🔠 Building TF-IDF representation (unigrams + bigrams)...")
vectorizer = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 2),
    stop_words='english'
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ---------------------------------------------------------
# ✅ 3. Multi-Label Binarization
# ---------------------------------------------------------
mlb_cpt = MultiLabelBinarizer()
mlb_icd = MultiLabelBinarizer()

y_train_cpt_bin = mlb_cpt.fit_transform(y_train_cpt)
y_test_cpt_bin = mlb_cpt.transform(y_test_cpt)

y_train_icd_bin = mlb_icd.fit_transform(y_train_icd)
y_test_icd_bin = mlb_icd.transform(y_test_icd)

# ---------------------------------------------------------
# ✅ 4. Train LightGBM for CPT & ICD10
# ---------------------------------------------------------
def train_model(X_train, y_train, X_test, y_test, code_type):
    print(f"\n🚀 Training LightGBM model for {code_type} codes...")
    model = MultiOutputClassifier(
        LGBMClassifier(
            n_estimators=150,
            learning_rate=0.08,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.8,
            objective="binary",
            random_state=42,
            n_jobs=-1
        )
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    f1 = f1_score(y_test, preds, average="micro")
    print(f"✅ {code_type} Model | F1: {f1:.4f}")
    print(classification_report(y_test, preds, target_names=mlb_cpt.classes_ if code_type=='CPT' else mlb_icd.classes_))
    return model


model_cpt = train_model(X_train_tfidf, y_train_cpt_bin, X_test_tfidf, y_test_cpt_bin, "CPT")
model_icd = train_model(X_train_tfidf, y_train_icd_bin, X_test_tfidf, y_test_icd_bin, "ICD10")

# ---------------------------------------------------------
# ✅ 5. Save Models
# ---------------------------------------------------------
os.makedirs("models", exist_ok=True)

joblib.dump(model_cpt, "models/coding_model_cpt.pkl")
joblib.dump(model_icd, "models/coding_model_icd.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
joblib.dump(mlb_cpt, "models/mlb_cpt.pkl")
joblib.dump(mlb_icd, "models/mlb_icd.pkl")

print("\n💾 Models saved:")
print(" - models/coding_model_cpt.pkl")
print(" - models/coding_model_icd.pkl")
print(" - models/tfidf_vectorizer.pkl")
print(" - models/mlb_cpt.pkl")
print(" - models/mlb_icd.pkl")

print("\n🎯 All coding models trained and ready!")
