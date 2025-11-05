import os
import torch
import requests

MODEL_PATH = "models/icd_cpt_distilbert_v3/best_model.pt"
GDRIVE_URL = "https://drive.google.com/uc?id=YOUR_FILE_ID"  # <-- Replace with your actual ID

# Check and download model
if not os.path.exists(MODEL_PATH):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print("🔽 Downloading model from Google Drive...")
    r = requests.get(GDRIVE_URL, allow_redirects=True)
    open(MODEL_PATH, "wb").write(r.content)
    print("✅ Model downloaded successfully.")

# Load the model
model = torch.load(MODEL_PATH, map_location=torch.device("cpu"))

import sys, os, re, time, random, datetime, warnings, logging
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from sklearn.preprocessing import MultiLabelBinarizer
from streamlit_option_menu import option_menu

# ---------------------------------------------------------------------
# Fix Python Imports for Streamlit
# ---------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------
# Local Module Imports
# ---------------------------------------------------------------------
from modules.denial import render_denial_prediction
from modules.coding import render_coding_assistant
from modules.prior_auth import render_prior_auth
from modules.billing import render_billing
from modules.icd_notes import render_icd_notes
from dashboard.dashboard_view import render_dashboard

# ---------------------------------------------------------------------
# Streamlit Config
# ---------------------------------------------------------------------
st.set_page_config(page_title="AI-Powered RCM Console", layout="wide")
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# ---------------------------------------------------------------------
# Global CSS (AWS-Style)
# ---------------------------------------------------------------------
st.markdown("""
<style>
body {
    font-family: 'Inter', sans-serif;
    background-color: #F9FAFB;
    color: #111827;
}
.main-header {
    background-color:#1E3A8A;
    color:white;
    padding:1rem 2rem;
    border-radius:0 0 6px 6px;
    position: sticky;
    top: 0;
    z-index: 999;
}
.rcm-card {
    background:white;
    border-radius:10px;
    box-shadow:0 1px 4px rgba(0,0,0,0.08);
    padding:1rem 1.25rem;
    margin-bottom:1rem;
}
footer {
    margin-top:2rem;
    text-align:center;
    color:#6B7280;
    font-size:0.9rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.markdown(
    '<div class="main-header"><h2>AI-Powered RCM Console</h2>'
    '<p>Unified Interface for Revenue Cycle Intelligence</p></div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------------------------
# MODEL LOADER
# ---------------------------------------------------------------------
@st.cache_resource
def load_models():
    models = {}

    # ---- Denial Model ----
    if os.path.exists("models/denial_model.cbm"):
        m = CatBoostClassifier()
        m.load_model("models/denial_model.cbm")
        models["denial"] = m

    # ---- Billing Model ----
    if os.path.exists("models/billing_model.json"):
        xgb = XGBClassifier()
        xgb.load_model("models/billing_model.json")
        models["billing"] = xgb

    # ---- Prior Authorization ----
    if os.path.exists("models/pa_model.txt"):
        models["pa"] = lgb.Booster(model_file="models/pa_model.txt")

    # ---- AI Coding Models ----
    try:
        if os.path.exists("models/coding_model_icd.pkl"):
            models["coding_icd"] = joblib.load("models/coding_model_icd.pkl")
        if os.path.exists("models/coding_model_cpt.pkl"):
            models["coding_cpt"] = joblib.load("models/coding_model_cpt.pkl")
        if os.path.exists("models/tfidf_vectorizer.pkl"):
            models["tfidf_vectorizer"] = joblib.load("models/tfidf_vectorizer.pkl")
        if os.path.exists("models/mlb_icd.pkl"):
            models["mlb_icd"] = joblib.load("models/mlb_icd.pkl")
        if os.path.exists("models/mlb_cpt.pkl"):
            models["mlb_cpt"] = joblib.load("models/mlb_cpt.pkl")
        st.success("AI-Assisted Coding models loaded successfully.")
    except Exception as e:
        st.warning(f"Could not load some coding models: {e}")

    # ---- ICD/CPT Text Classification ----
    models["icd_cpt"] = True

    return models


# --- Initialize Models Early ---
models = load_models()

# ---------------------------------------------------------------------
# ICD/CPT Model Loader (DistilBERT)
# ---------------------------------------------------------------------
@st.cache_resource
def load_icd_cpt_model_safe():
    model_dir = "models/icd_cpt_distilbert_v3"
    config = AutoConfig.from_pretrained("distilbert-base-uncased", num_labels=27)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_config(config)
    best = os.path.join(model_dir, "best_model.pt")
    lb = os.path.join(model_dir, "label_binarizer.pkl")

    try:
        state = torch.load(best, map_location="cpu")
        model.load_state_dict(state.get("model", state), strict=False)
    except Exception:
        pass
    try:
        labels = joblib.load(lb)
    except Exception:
        labels = None

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, labels, device


tokenizer, icd_model, label_binarizer, device = load_icd_cpt_model_safe()

# ---------------------------------------------------------------------
# NAVIGATION BAR (TOP MENU)
# ---------------------------------------------------------------------
selected = option_menu(
    None,
    ["Dashboard", "Denial Prediction", "AI-Assisted Coding",
     "Prior Authorization", "Billing Optimization", "ICD/CPT from Notes"],
    icons=["bar-chart", "activity", "code", "clipboard-check", "credit-card", "file-text"],
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#1E3A8A", "justify-content": "center"},
        "nav-link": {"font-size": "15px", "font-weight": "500", "color": "#E0E7FF", "padding": "10px 16px"},
        "nav-link-selected": {"background-color": "#F9FAFB", "color": "#1E3A8A", "font-weight": "600", "border-bottom": "4px solid #2563EB"},
        "icon": {"color": "#E0E7FF", "font-size": "18px"},
    },
)

# ---------------------------------------------------------------------
# TAB ROUTING
# ---------------------------------------------------------------------
if selected == "Dashboard":
    render_dashboard()

elif selected == "Denial Prediction":
    render_denial_prediction(models)

elif selected == "AI-Assisted Coding":
    render_coding_assistant(models)

elif selected == "Prior Authorization":
    render_prior_auth(models)

elif selected == "Billing Optimization":
    render_billing(models)

elif selected == "ICD/CPT from Notes":
    render_icd_notes(tokenizer, icd_model, label_binarizer, device)

# ---------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------
st.markdown(
    "<footer>© 2025 AI-Powered RCM Suite • Designed by Anurag Mishra</footer>",
    unsafe_allow_html=True
)
