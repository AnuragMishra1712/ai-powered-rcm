import streamlit as st
import pandas as pd
import time
import joblib
import lightgbm as lgb
from utils.db_utils import log_metric


def render_prior_auth(models):
    st.title("Prior Authorization Automation")
    st.markdown(
        "Predict whether a claim requires prior authorization and simulate automated submission workflow."
    )

    # ---------------------- Load model ---------------------- #
    model_file = "models/pa_model.txt"
    try:
        booster = lgb.Booster(model_file=model_file)
    except Exception as e:
        st.error(f"Failed to load LightGBM model: {e}")
        return

    # ---------------------- Load categorical metadata ---------------------- #
    cat_cols_path = "models/pa_categorical_cols.pkl"
    try:
        pa_cat_cols = joblib.load(cat_cols_path)
        if not isinstance(pa_cat_cols, list):
            pa_cat_cols = list(pa_cat_cols)
    except Exception:
        pa_cat_cols = []
        st.warning("⚠️ No categorical column metadata found. Using fallback.")

    model_features = booster.feature_name()

    # ---------------------- INPUT FORM ---------------------- #
    with st.form("pa_form"):
        st.markdown("#### Enter Claim & Patient Details")

        c1, c2, c3 = st.columns(3)
        with c1:
            claim_id = st.text_input("Claim ID", "C123")
            age = st.number_input("Age", 0, 120, 45)
            gender = st.selectbox("Gender", ["M", "F"])
            insurance_type = st.selectbox(
                "Insurance Type", ["Commercial", "Medicare", "Medicaid", "PPO", "HMO"]
            )

        with c2:
            provider_specialty = st.selectbox(
                "Provider Specialty",
                ["Cardiology", "Orthopedics", "Oncology", "Radiology", "Primary Care"],
            )
            state = st.selectbox("State", ["CA", "NY", "TX", "FL", "NJ"])
            chronic_condition = st.selectbox("Chronic Condition", [0, 1])
            claim_amount = st.number_input("Claim Amount ($)", 0.0, 100000.0, 4000.0)

        with c3:
            procedure_category = st.selectbox(
                "Procedure Category", ["Imaging", "Surgery", "Lab", "Consult", "Therapy"]
            )
            provider_experience = st.number_input("Provider Experience (yrs)", 0, 40, 10)
            patient_history_flag = st.selectbox("Patient History Available", [0, 1])
            coverage_ratio = st.slider("Payer Coverage Ratio", 0.0, 1.0, 0.8)

        submitted = st.form_submit_button("Run PA Prediction")

    if not submitted:
        return

    # ---------------------- DATA PREPARATION ---------------------- #
    start = time.time()

    df = pd.DataFrame(
        [
            {
                "claim_id": claim_id,
                "age": age,
                "gender": gender,
                "insurance_type": insurance_type,
                "provider_specialty": provider_specialty,
                "state": state,
                "chronic_condition": chronic_condition,
                "claim_amount": claim_amount,
                "procedure_category": procedure_category,
                "provider_experience": provider_experience,
                "patient_history_flag": patient_history_flag,
                "payer_coverage_ratio": coverage_ratio,
            }
        ]
    )

    # Add missing columns
    for f in model_features:
        if f not in df.columns:
            df[f] = 0

    df = df[model_features]

    # Convert all to numeric safely (LightGBM Booster doesn't need category dtype)
    for col in df.columns:
        if df[col].dtype == "object" or df[col].dtype.name == "category":
            df[col] = pd.factorize(df[col])[0]  # convert category → integer code
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    st.write("✅ Model expects:", len(model_features), "features")
    st.write("✅ Input provided:", len(df.columns))

    # ---------------------- PREDICTION ---------------------- #
    try:
        prob = float(booster.predict(df)[0])
        runtime = time.time() - start
        log_metric("Prior Authorization", prob, runtime)
    except Exception as e:
        st.error(f"Model inference failed: {e}")
        return

    # ---------------------- DISPLAY ---------------------- #
    st.markdown("---")
    st.subheader("Prediction Summary")

    c1, c2 = st.columns(2)
    c1.metric("Prior Authorization Probability", f"{prob:.2%}")
    c2.metric("Model Runtime", f"{runtime:.2f} sec")

    if prob < 0.4:
        st.success("No Prior Authorization required.")
    elif prob < 0.7:
        st.warning("Prior Authorization may be required — verify policy.")
    else:
        st.error("High probability — Prior Authorization is required.")

    # ---------------------- SIMULATION ---------------------- #
    st.markdown("#### Automated Submission Workflow")
    if prob >= 0.7:
        steps = [
            "Checking payer rules...",
            "Preparing prior authorization request...",
            "Submitting to payer portal...",
            "Awaiting payer response...",
        ]
        progress = st.progress(0)
        for i, step in enumerate(steps):
            progress.progress(int((i + 1) / len(steps) * 100))
            st.text(step)
            time.sleep(0.5)
        st.success("Submission completed — awaiting payer decision.")
    elif prob < 0.4:
        st.info("No prior authorization required — claim can be submitted directly.")
    else:
        st.info("Review required — insufficient data for full automation.")

    st.caption("All PA predictions are logged to the RCM Dashboard for monitoring.")
