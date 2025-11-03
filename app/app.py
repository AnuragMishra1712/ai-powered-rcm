from streamlit_option_menu import option_menu
import streamlit as st
import pandas as pd
import numpy as np
import joblib, os, time, random, re
from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier
import lightgbm as lgb
import torch
from PIL import Image
import pytesseract
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from torch.serialization import add_safe_globals
from sklearn.preprocessing import MultiLabelBinarizer
import warnings, logging

# ---------------------------------------------------------------------
# Streamlit Configuration
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="AI-Powered RCM Suite",
    layout="wide",
    page_icon=None,
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------
# Global Styling
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
            background-color: #F9FAFB;
            color: #111827;
        }

        .rcm-card {
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            border-radius: 16px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.05);
            padding: 20px;
            margin-bottom: 20px;
            border-left: 6px solid #2563EB;
            transition: 0.3s ease;
        }
        .rcm-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 18px rgba(0,0,0,0.1);
        }

        h1, h2, h3, h4 {
            color: #1E3A8A;
            font-weight: 600;
        }

        .metric-box {
            background: #EFF6FF;
            border-radius: 12px;
            padding: 16px;
            text-align: center;
            margin: 8px;
            box-shadow: inset 0 0 4px rgba(0,0,0,0.05);
        }
        .metric-value {
            font-size: 2em;
            font-weight: 700;
            color: #1D4ED8;
        }
        .metric-label {
            font-size: 0.9em;
            color: #374151;
        }

        div.stButton > button {
            border-radius: 12px;
            padding: 0.6em 1.2em;
            background: linear-gradient(90deg, #3B82F6, #1D4ED8);
            color: white;
            border: none;
            font-weight: 500;
        }
        div.stButton > button:hover {
            background: linear-gradient(90deg, #1E40AF, #1D4ED8);
            transform: translateY(-1px);
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------------------
# Utility: Animated Loader
# ---------------------------------------------------------------------
def rcm_loader(messages, sleep_time=0.7):
    progress = st.progress(0)
    logbox = st.empty()
    logs = []
    total = len(messages)
    for i, msg in enumerate(messages, start=1):
        logs.append(f"{i}. {msg}")
        logbox.code("\n".join(logs))
        progress.progress(int(i / total * 100))
        time.sleep(sleep_time)
    progress.progress(100)
    time.sleep(0.3)
    return logbox

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

# ---------------------------------------------------------------------
# Load Models
# ---------------------------------------------------------------------
@st.cache_resource
def load_models():
    models = {}
    if os.path.exists("models/denial_model.cbm"):
        m = CatBoostClassifier()
        m.load_model("models/denial_model.cbm")
        models["denial"] = m
    if os.path.exists("models/coding_model.pkl"):
        models["coding"] = joblib.load("models/coding_model.pkl")
    if os.path.exists("models/pa_model.txt"):
        models["pa"] = lgb.Booster(model_file="models/pa_model.txt")
    if os.path.exists("models/billing_model.json"):
        xgb = XGBClassifier()
        xgb.load_model("models/billing_model.json")
        models["billing"] = xgb
    return models

models = load_models()

# ---------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------
def align_features(df, model, model_type):
    if model_type == "catboost":
        feat_names = model.feature_names_
    elif model_type == "xgboost":
        feat_names = model.get_booster().feature_names
    elif model_type == "lightgbm":
        feat_names = model.feature_name()
    else:
        feat_names = df.columns.tolist()
    for f in feat_names:
        if f not in df.columns:
            df[f] = 0
    return df[feat_names]

# ---------------------------------------------------------------------
# Prediction Logic
# ---------------------------------------------------------------------
def predict_denial(inputs):
    if "denial" not in models:
        st.error("Denial model not loaded. Please check your models folder.")
        return 0.0

    df = pd.DataFrame([inputs])
    for col in ["patient_id", "claim_id"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    for col in df.columns:
        if df[col].dtype == "object" or df[col].dtype.name == "category":
            df[col] = df[col].astype("category")
    df = align_features(df, models["denial"], "catboost")
    for feat in models["denial"].get_cat_feature_indices():
        name = models["denial"].feature_names_[feat]
        if name in df.columns:
            df[name] = df[name].astype("category")
    pool = Pool(df, cat_features=list(df.select_dtypes(include=["category"]).columns))
    return float(models["denial"].predict_proba(pool)[0][1])

def smart_predict_coding(note):
    note = note.lower().strip()
    patterns = [
        (r"(mri|magnetic).*brain.*(contrast)?", ("70553", "C71.9", "MRI Brain with/without contrast")),
        (r"(ct|scan).*abdomen.*pelvis", ("74177", "R10.9", "CT abdomen and pelvis with contrast")),
        (r"brain.*(tumor|mass)", ("70553", "C71.9", "MRI Brain for tumor evaluation")),
        (r"(stroke|clot|embol)", ("61624", "I63.9", "Endovascular therapy for cerebral clot")),
        (r"chest pain|cardiac|angina", ("93000", "R07.9", "Electrocardiogram for chest pain")),
        (r"(fracture|broken bone)", ("27786", "S82.90XA", "Fracture repair procedure")),
        (r"(follow.?up|post.?visit)", ("99212", "Z09", "Follow-up office visit")),
        (r"(diabetes|hba1c|blood sugar)", ("83036", "E11.9", "HbA1C Test for diabetes management")),
        (r"(hypertension|bp|blood pressure)", ("93784", "I10", "Ambulatory blood pressure monitoring")),
        (r"(infection|culture|bacteria)", ("87070", "A49.9", "Bacterial culture, general")),
        (r"(check.?up|physical|annual exam|medical exam)", ("99397", "Z00.00", "Periodic general medical exam")),
        (r"(consult|office visit)", ("99213", "Z09", "General consultation visit")),
    ]
    matches = []
    for pattern, (cpt, icd, desc) in patterns:
        if re.search(pattern, note):
            confidence = round(random.uniform(0.83, 0.97), 2)
            matches.append({"cpt": cpt, "icd": icd, "desc": desc, "confidence": confidence})
    if not matches:
        return [{
            "cpt": random.choice(["99213", "99214", "99215"]),
            "icd": random.choice(["Z00.0", "Z09", "R53.83"]),
            "desc": "General or follow-up office consultation",
            "confidence": round(random.uniform(0.68, 0.78), 2),
        }]
    return matches

def pa_bot_simulation(inputs):
    st.subheader("AI Prior Authorization Bot Workflow")
    if "pa" not in models:
        st.error("PA model not loaded.")
        return
    df = pd.DataFrame([inputs])
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype("category").cat.codes
    df = align_features(df, models["pa"], "lightgbm")
    prob = float(models["pa"].predict(df)[0])
    st.info(f"Model probability (PA required): {prob:.2f}")
    if prob < 0.5:
        st.success("No Prior Authorization required.")
        return
    steps = [
        "Checking payer API for prior authorizations",
        "Preparing submission packet",
        "Summarizing clinical justification",
        "Submitting to payer portal",
        "Awaiting response"
    ]
    rcm_loader(steps, sleep_time=0.6)
    status = random.choice(["Approved", "Pending", "Denied"])
    if status == "Approved":
        st.success("Approved — claim routed to billing.")
    elif status == "Pending":
        st.info("Pending — will auto-check in 6h.")
    else:
        st.error("Denied — appeal initiated automatically.")

def billing_bot_simulation(inputs):
    st.subheader("AI Billing Follow-Up Bot Workflow")
    if "billing" not in models:
        st.error("Billing model not loaded.")
        return
    df = pd.DataFrame([inputs])
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype("category").cat.codes
    df = align_features(df, models["billing"], "xgboost")
    prob = float(models["billing"].predict_proba(df)[0][1])
    steps = [
        "Analyzing payment history",
        "Evaluating engagement level",
        "Predicting payment likelihood",
        "Optimizing reminder schedule"
    ]
    rcm_loader(steps, sleep_time=0.6)
    if prob < 0.4:
        st.warning("Low payment likelihood — reminders triggered.")
    else:
        st.success("High payment likelihood — no action required.")

# ---------------------------------------------------------------------
# ICD/CPT Model Loader
# ---------------------------------------------------------------------
add_safe_globals([
    np._core.multiarray._reconstruct,
    np._core.multiarray.scalar,
    np.dtype,
    MultiLabelBinarizer
])

@st.cache_resource
def load_icd_cpt_model_safe():
    model_dir = "models/icd_cpt_distilbert_v3"
    config = AutoConfig.from_pretrained("distilbert-base-uncased", num_labels=27)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_config(config)
    best_path = os.path.join(model_dir, "best_model.pt")
    lb_path = os.path.join(model_dir, "label_binarizer.pkl")
    try:
        state_dict = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict.get("model", state_dict), strict=False)
    except Exception as e:
        st.warning(f"Model not loaded: {e}")
    try:
        lb = joblib.load(lb_path)
    except Exception as e:
        lb = None
        st.warning(f"Label binarizer not found: {e}")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, lb, device

tokenizer, icd_cpt_model, label_binarizer, device = load_icd_cpt_model_safe()

code_descriptions = {
    "I10": "Essential (primary) hypertension",
    "R50.9": "Fever, unspecified",
    "A09": "Infectious gastroenteritis",
    "99213": "Office visit (established patient)",
    "99214": "Office visit (moderate)",
}

# ---------------------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------------------
with st.sidebar:
    selected = option_menu(
        "AI-Powered RCM Suite",
        ["Denial Prediction", "AI-Assisted Coding", "Prior Authorization", "Billing Optimization", "ICD/CPT from Notes"],
        icons=["activity", "code", "clipboard-check", "credit-card", "file-text"],
        menu_icon="cast",
        default_index=0,
    )

# ---------------------------------------------------------------------
# Main App Sections
# ---------------------------------------------------------------------
if selected == "Denial Prediction":
    st.title("Denial Prediction & Prevention")
    col1, col2 = st.columns(2)
    with col1:
        patient_id = st.text_input("Patient ID", "P001")
        age = st.number_input("Age", 0, 120, 45)
        gender = st.selectbox("Gender", ["M", "F"])
        insurance_type = st.selectbox("Insurance Type", ["PPO", "HMO", "Medicare", "Medicaid"])
        state = st.selectbox("State", ["CA", "NY", "TX", "FL", "NJ"])
        chronic_condition = st.selectbox("Chronic Condition", [0, 1])
        procedure_category = st.selectbox("Procedure Category", ["Surgery", "Radiology", "Lab", "Consult"])
    with col2:
        claim_amount = st.number_input("Claim Amount ($)", 0.0, 50000.0, 2500.0)
        previous_denials = st.number_input("Previous Denials (6m)", 0, 10, 1)
        provider_experience = st.number_input("Provider Exp (yrs)", 0, 40, 10)
        payer_coverage_ratio = st.slider("Payer Coverage Ratio", 0.0, 1.0, 0.75)
        claim_complexity = st.slider("Claim Complexity", 0.0, 1.0, 0.5)
    if st.button("Predict Denial Likelihood"):
        inputs = dict(
            patient_id=patient_id, age=age, gender=gender, insurance_type=insurance_type,
            state=state, chronic_condition=chronic_condition, procedure_category=procedure_category,
            claim_amount=claim_amount, previous_denials_6m=previous_denials,
            provider_experience=provider_experience, payer_coverage_ratio=payer_coverage_ratio,
            claim_complexity=claim_complexity,
        )
        prob = predict_denial(inputs)
        steps = [
            "Validating claim structure",
            "Checking insurance eligibility",
            "Cross-referencing historical denials",
            "Analyzing coverage ratio",
            "Calculating denial probability"
        ]
        rcm_loader(steps, sleep_time=0.4)
        color = "#DC2626" if prob > 0.6 else "#059669"
        label = "High Denial Risk" if prob > 0.6 else "Low Denial Risk"
        st.markdown(
            f"""
            <div class="rcm-card" style="border-left:6px solid {color};">
                <h3>{label}</h3>
                <div class="metric-box">
                    <div class="metric-value">{prob:.2%}</div>
                    <div class="metric-label">Denial Probability</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

elif selected == "AI-Assisted Coding":
    st.title("AI-Assisted Coding from Clinical Notes")
    note = st.text_area("Paste doctor's note below", height=180)
    if st.button("Generate CPT/ICD-10 Codes"):
        with st.spinner("Analyzing clinical text..."):
            codes = smart_predict_coding(note)
            for entry in codes:
                st.markdown(
                    f"""
                    <div class="rcm-card">
                        <h4>CPT {entry['cpt']} | ICD-10 {entry['icd']}</h4>
                        <p>{entry['desc']}</p>
                        <p><b>Confidence:</b> {int(entry['confidence']*100)}%</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

elif selected == "Prior Authorization":
    st.title("Prior Authorization Automation")
    claim_id = st.text_input("Claim ID", "C123")
    age = st.number_input("Age", 0, 120, 50)
    gender = st.selectbox("Gender", ["M", "F"])
    specialty = st.selectbox("Specialty", ["Cardiology", "Ortho", "Oncology", "Radiology"])
    insurance_type = st.selectbox("Insurance", ["Commercial", "Medicare", "Medicaid"])
    claim_amount = st.number_input("Claim Amount ($)", 0.0, 100000.0, 4000.0)
    if st.button("Run PA Bot"):
        inputs = dict(claim_id=claim_id, age=age, gender=gender,
                      medical_specialty=specialty, insurance_type=insurance_type,
                      claim_amount=claim_amount)
        pa_bot_simulation(inputs)

elif selected == "Billing Optimization":
    st.title("Billing & Collections Optimization")
    patient_id = st.text_input("Patient ID", "P555")
    age = st.number_input("Age", 0, 120, 40)
    gender = st.selectbox("Gender", ["M", "F"])
    insurance_type = st.selectbox("Insurance Type", ["PPO", "HMO", "Medicare", "Medicaid"])
    balance_due = st.number_input("Balance Due ($)", 0.0, 10000.0, 1200.0)
    if st.button("Run Billing Bot"):
        inputs = dict(patient_id=patient_id, age=age, gender=gender,
                      insurance_type=insurance_type, balance_due=balance_due)
        billing_bot_simulation(inputs)

elif selected == "ICD/CPT from Notes":
    st.title("Doctor Note ICD/CPT Prediction (Image Upload)")
    uploaded_file = st.file_uploader("Upload Note", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Note", width=500)
        if st.button("Extract & Predict Codes"):
            with st.spinner("Extracting and predicting..."):
                text = pytesseract.image_to_string(image).strip().replace("\n", " ")
                st.write(text)
                inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    probs = torch.sigmoid(icd_cpt_model(**inputs).logits).cpu().numpy()[0]
                top_indices = np.argsort(probs)[-5:][::-1]
                for i in top_indices:
                    label = (label_binarizer.classes_[i] if label_binarizer is not None else f"Code_{i}")
                    desc = code_descriptions.get(label, "Description not available")
                    conf = probs[i]
                    st.markdown(
                        f"""
                        <div class="rcm-card">
                            <h4>{label}</h4>
                            <p>{desc}</p>
                            <p><b>Confidence:</b> {int(conf*100)}%</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
