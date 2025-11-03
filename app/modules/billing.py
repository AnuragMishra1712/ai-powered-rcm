import streamlit as st
import pandas as pd
import time
from utils.db_utils import log_metric


def render_billing(models):
    st.title("Billing Optimization")
    st.markdown(
        "Predict payment likelihood and recommend the best billing strategy based on patient and claim details."
    )

    model = models.get("billing")
    if not model:
        st.error("Billing model not loaded.")
        return

    model_features = model.get_booster().feature_names if hasattr(model, "get_booster") else []

    # ---------------------- INPUT FORM ---------------------- #
    with st.form("billing_form"):
        st.markdown("#### Enter Billing & Payment Details")

        c1, c2, c3 = st.columns(3)
        with c1:
            patient_id = st.text_input("Patient ID", "P001")
            age = st.number_input("Age", 0, 120, 45)
            gender = st.selectbox("Gender", ["M", "F"])
            insurance_type = st.selectbox("Insurance Type", ["PPO", "HMO", "Medicare", "Medicaid", "Commercial"])

        with c2:
            balance_due = st.number_input("Balance Due ($)", 0.0, 50000.0, 1500.0)
            num_reminders_sent = st.number_input("Reminders Sent", 0, 10, 1)
            last_payment_days = st.number_input("Days Since Last Payment", 0, 365, 45)
            income_bracket = st.selectbox("Income Bracket", ["Low", "Medium", "High"])

        with c3:
            visit_type = st.selectbox("Visit Type", ["Routine", "Emergency", "Specialist", "Follow-Up"])
            has_payment_plan = st.selectbox("Has Payment Plan", [0, 1])
            credit_score = st.number_input("Credit Score", 300, 850, 680)
            payment_method = st.selectbox("Payment Method", ["Credit Card", "Insurance", "Cash", "Online"])

        submitted = st.form_submit_button("Run Billing Optimization")

    if not submitted:
        return

    # ---------------------- DATA PREP ---------------------- #
    start = time.time()

    df = pd.DataFrame(
        [
            {
                "patient_id": patient_id,
                "age": age,
                "gender": gender,
                "insurance_type": insurance_type,
                "balance_due": balance_due,
                "num_reminders_sent": num_reminders_sent,
                "last_payment_days": last_payment_days,
                "income_bracket": income_bracket,
                "visit_type": visit_type,
                "has_payment_plan": has_payment_plan,
                "credit_score": credit_score,
                "payment_method": payment_method,
            }
        ]
    )

    # Add any missing columns based on model training features
    for f in model_features:
        if f not in df.columns:
            df[f] = 0
    df = df[model_features]

    # Convert all non-numeric to numeric safely
    for col in df.columns:
        if df[col].dtype == "object" or df[col].dtype.name == "category":
            df[col] = pd.factorize(df[col])[0]
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # ---------------------- PREDICTION ---------------------- #
    try:
        prob = float(model.predict(df)[0])
        runtime = time.time() - start
        log_metric("Billing Optimization", prob, runtime)
    except Exception as e:
        st.error(f"Model inference failed: {e}")
        return

    # ---------------------- RESULTS ---------------------- #
    st.markdown("---")
    st.subheader("Billing Optimization Results")

    c1, c2 = st.columns(2)
    c1.metric("Payment Probability", f"{prob:.2%}")
    c2.metric("Model Runtime", f"{runtime:.2f} sec")

    if prob < 0.4:
        st.error("High risk of late or missed payment. Recommend early reminders or payment plan.")
    elif prob < 0.7:
        st.warning("Moderate payment likelihood. Consider small discounts or additional reminders.")
    else:
        st.success("High payment likelihood. Proceed with standard billing.")

    # ---------------------- RECOMMENDED ACTIONS ---------------------- #
    st.markdown("### Recommended Actions")
    if prob < 0.4:
        st.write(
            "- Offer installment plan or extended due date\n"
            "- Send follow-up reminders via SMS and email\n"
            "- Check payer coverage ratio for secondary claim"
        )
    elif prob < 0.7:
        st.write(
            "- Offer small discount (2-5%) for early payment\n"
            "- Send automated reminder with payment link"
        )
    else:
        st.write(
            "- Proceed with automated invoice\n"
            "- No additional interventions required"
        )

    st.caption("All billing predictions are logged in the RCM Dashboard for ongoing monitoring.")
