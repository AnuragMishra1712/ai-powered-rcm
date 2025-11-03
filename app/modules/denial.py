import streamlit as st
import pandas as pd
import time
from catboost import Pool
from utils.db_utils import log_metric


def render_denial_prediction(models):
    st.title("Denial Prediction & Prevention")
    st.markdown(
        "Predict the likelihood of a claim being denied using patient, provider, and claim details."
    )

    model = models.get("denial")
    if not model:
        st.error("Denial model not loaded.")
        return

    # ---------------------- INPUT FORM ---------------------- #
    with st.form("denial_form"):
        st.markdown("#### Enter Claim Details")

        c1, c2, c3 = st.columns(3)

        with c1:
            patient_id = st.number_input("Patient ID", 1001, 999999, 1001)
            age = st.number_input("Age", 0, 120, 45)
            gender = st.selectbox("Gender", ["M", "F"])
            insurance_type = st.selectbox(
                "Insurance Type", ["PPO", "HMO", "Medicare", "Medicaid"]
            )

        with c2:
            state = st.selectbox("State", ["CA", "NY", "TX", "FL", "NJ"])
            provider_specialty = st.selectbox(
                "Provider Specialty",
                ["Cardiology", "Orthopedics", "Oncology", "Radiology", "Family Medicine"],
            )
            has_chronic_condition = st.selectbox("Chronic Condition", [0, 1])
            procedure_category = st.selectbox(
                "Procedure Category", ["Surgery", "Radiology", "Lab", "Consult"]
            )

        with c3:
            claim_amount = st.number_input("Claim Amount ($)", 0.0, 50000.0, 2500.0)
            prev_denials = st.number_input("Prev Denials (6m)", 0, 10, 1)
            provider_exp = st.number_input("Provider Experience (yrs)", 0, 40, 10)
            coverage_ratio = st.slider("Coverage Ratio", 0.0, 1.0, 0.75)

        claim_complexity = st.slider("Claim Complexity", 0.0, 1.0, 0.5)

        submitted = st.form_submit_button("Predict Denial")

    if not submitted:
        return

    # ---------------------- DATA PREPARATION ---------------------- #
    start = time.time()
    df = pd.DataFrame(
        [
            {
                "patient_id": patient_id,
                "age": age,
                "gender": gender,
                "insurance_type": insurance_type,
                "state": state,
                "provider_specialty": provider_specialty,
                "has_chronic_condition": has_chronic_condition,
                "procedure_category": procedure_category,
                "claim_amount": claim_amount,
                "previous_denials_6m": prev_denials,
                "provider_experience": provider_exp,
                "payer_coverage_ratio": coverage_ratio,
                "claim_complexity": claim_complexity,
            }
        ]
    )

    # Align with model features
    model_features = model.feature_names_
    for f in model_features:
        if f not in df.columns:
            df[f] = 0
    df = df[model_features]

    # Convert categorical + numeric dtypes
    cat_cols = [
        f
        for f in df.columns
        if df[f].dtype == "object" or str(df[f].dtype).startswith("category")
    ]
    for c in cat_cols:
        df[c] = df[c].astype("category")

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype("category")
        elif df[col].dtype.name == "category":
            continue
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Enforce model-declared categorical columns
    try:
        model_cat_indices = model.get_cat_feature_indices()
        model_cat_names = [model.feature_names_[i] for i in model_cat_indices]
        for c in model_cat_names:
            if c in df.columns:
                df[c] = df[c].astype("category")
                if c not in cat_cols:
                    cat_cols.append(c)
    except Exception as e:
        st.warning(f"Could not align categorical columns: {e}")

    # ---------------------- PREDICTION ---------------------- #
    try:
        pool = Pool(df, cat_features=cat_cols)
        prob = float(model.predict_proba(pool)[0][1])
        runtime = time.time() - start
        log_metric("Denial Prediction", prob, runtime)
    except Exception as e:
        st.error(f"Model inference failed: {e}")
        return

    # ---------------------- DISPLAY RESULTS ---------------------- #
    st.markdown("---")
    st.subheader("Prediction Summary")

    col1, col2 = st.columns(2)
    col1.metric("Denial Probability", f"{prob:.2%}")
    col2.metric("Model Runtime", f"{runtime:.2f} sec")

    # Risk tiers
    if prob > 0.7:
        st.error("High denial risk — immediate review recommended.")
    elif prob > 0.4:
        st.warning("Moderate risk — verify documentation and payer rules.")
    else:
        st.success("Low denial risk — claim likely to be approved.")

    # Recommendations
    st.markdown("#### Recommended Actions")
    if prob > 0.7:
        st.write(
            "- Verify coverage limits\n"
            "- Ensure all supporting documents attached\n"
            "- Review provider coding accuracy"
        )
    elif prob > 0.4:
        st.write(
            "- Check if prior authorization is needed\n"
            "- Validate previous denial reasons for similar cases"
        )
    else:
        st.write("- Proceed with submission confidently — no major risk detected.")

    st.caption("All predictions are logged to the RCM Dashboard for live monitoring.")
