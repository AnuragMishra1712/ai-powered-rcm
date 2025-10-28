# # # # # import streamlit as st
# # # # # import pandas as pd
# # # # # import numpy as np
# # # # # import joblib, os, time, random
# # # # # from catboost import CatBoostClassifier, Pool
# # # # # from xgboost import XGBClassifier
# # # # # import lightgbm as lgb

# # # # # st.set_page_config(page_title="AI-Powered RCM Suite", layout="wide")

# # # # # # ---------------------------------------------------------------------
# # # # # # 🧠 Load All Models
# # # # # # ---------------------------------------------------------------------
# # # # # @st.cache_resource
# # # # # def load_models():
# # # # #     models = {}
# # # # #     if os.path.exists("models/denial_model.cbm"):
# # # # #         m = CatBoostClassifier()
# # # # #         m.load_model("models/denial_model.cbm")
# # # # #         models["denial"] = m
# # # # #     if os.path.exists("models/coding_model.pkl"):
# # # # #         models["coding"] = joblib.load("models/coding_model.pkl")
# # # # #     if os.path.exists("models/pa_model.txt"):
# # # # #         models["pa"] = lgb.Booster(model_file="models/pa_model.txt")
# # # # #     if os.path.exists("models/billing_model.json"):
# # # # #         xgb = XGBClassifier()
# # # # #         xgb.load_model("models/billing_model.json")
# # # # #         models["billing"] = xgb
# # # # #     return models

# # # # # models = load_models()

# # # # # # ---------------------------------------------------------------------
# # # # # # 🧩 Feature Alignment Helper
# # # # # # ---------------------------------------------------------------------
# # # # # def align_features(df, model, model_type):
# # # # #     if model_type == "catboost":
# # # # #         feat_names = model.feature_names_
# # # # #     elif model_type == "xgboost":
# # # # #         feat_names = model.get_booster().feature_names
# # # # #     elif model_type == "lightgbm":
# # # # #         feat_names = model.feature_name()
# # # # #     else:
# # # # #         return df

# # # # #     for f in feat_names:
# # # # #         if f not in df.columns:
# # # # #             df[f] = 0
# # # # #     return df[feat_names]

# # # # # # ---------------------------------------------------------------------
# # # # # # 🧾 Denial Prediction
# # # # # # ---------------------------------------------------------------------
# # # # # def predict_denial(inputs):
# # # # #     df = pd.DataFrame([inputs])

# # # # #     # Drop identifiers that cause non-numeric errors
# # # # #     for col in ["patient_id", "claim_id"]:
# # # # #         if col in df.columns:
# # # # #             df = df.drop(columns=[col])

# # # # #     # Convert object columns to categorical
# # # # #     for c in df.select_dtypes(include=["object"]).columns:
# # # # #         df[c] = df[c].astype("category")

# # # # #     df = align_features(df, models["denial"], "catboost")
# # # # #     pool = Pool(df, cat_features=list(df.select_dtypes(include=["category"]).columns))
# # # # #     return float(models["denial"].predict_proba(pool)[0][1])

# # # # # # ---------------------------------------------------------------------
# # # # # # 🧠 Coding Prediction
# # # # # # ---------------------------------------------------------------------
# # # # # def predict_coding(note):
# # # # #     note = note.lower()
# # # # #     if "colonoscopy" in note:
# # # # #         return ["45378"], ["K62.5"]
# # # # #     elif "chest pain" in note or "angina" in note:
# # # # #         return ["93000", "99284"], ["I20.9"]
# # # # #     elif "diabetes" in note:
# # # # #         return ["83036"], ["E11.9"]
# # # # #     elif "hypertension" in note:
# # # # #         return ["93015"], ["I10"]
# # # # #     else:
# # # # #         return ["99213"], ["Z00.0"]

# # # # # # ---------------------------------------------------------------------
# # # # # # 🤖 Prior Authorization BOT
# # # # # # ---------------------------------------------------------------------
# # # # # def pa_bot_simulation(inputs):
# # # # #     st.subheader("AI Prior Authorization Bot Workflow")

# # # # #     df = pd.DataFrame([inputs])
# # # # #     for c in df.select_dtypes(include=["object"]).columns:
# # # # #         df[c] = df[c].astype("category").cat.codes

# # # # #     df = align_features(df, models["pa"], "lightgbm")

# # # # #     prob = float(models["pa"].predict(df)[0])
# # # # #     st.info(f"Model probability (PA required): {prob:.2f}")

# # # # #     if prob < 0.5:
# # # # #         st.success("✅ No Prior Authorization required.")
# # # # #         return

# # # # #     st.warning("⚠️ Prior Authorization required. Initiating automation...")

# # # # #     progress = st.progress(0)
# # # # #     logbox = st.empty()
# # # # #     logs = []

# # # # #     def log(msg, step, wait=1.1):
# # # # #         logs.append(msg)
# # # # #         logbox.code("\n".join(logs))
# # # # #         progress.progress(step)
# # # # #         time.sleep(wait)

# # # # #     log("🔍 Checking payer API for prior authorizations...", 10)
# # # # #     log("📁 No record found — preparing submission packet...", 25)
# # # # #     log("🧠 Using NLP to summarize clinical justification...", 45)
# # # # #     log("📤 Submitting request via payer integration...", 70)
# # # # #     log("⏳ Awaiting payer decision...", 90)

# # # # #     status = random.choice(["Approved", "Pending", "Denied"])
# # # # #     log(f"📨 Response received: {status}", 100)

# # # # #     if status == "Approved":
# # # # #         st.success("✅ Approved. Claim routed for billing.")
# # # # #     elif status == "Pending":
# # # # #         st.info("⌛ Pending. Bot will auto-poll every 6 hours.")
# # # # #     else:
# # # # #         st.error("❌ Denied. Provider notified & appeal initiated.")

# # # # # # ---------------------------------------------------------------------
# # # # # # 💳 Billing Optimization BOT
# # # # # # ---------------------------------------------------------------------
# # # # # def billing_bot_simulation(inputs):
# # # # #     st.subheader("AI Billing Follow-Up Bot Workflow")

# # # # #     df = pd.DataFrame([inputs])

# # # # #     # Drop identifiers
# # # # #     for col in ["patient_id", "claim_id"]:
# # # # #         if col in df.columns:
# # # # #             df = df.drop(columns=[col])

# # # # #     # Encode categorical columns
# # # # #     for c in df.select_dtypes(include=["object"]).columns:
# # # # #         df[c] = df[c].astype("category").cat.codes

# # # # #     df = align_features(df, models["billing"], "xgboost")

# # # # #     prob = float(models["billing"].predict_proba(df)[0][1])
# # # # #     st.metric("Payment Probability", f"{prob:.2f}")

# # # # #     progress = st.progress(0)
# # # # #     logbox = st.empty()
# # # # #     logs = []

# # # # #     def log(msg, step, wait=1.1):
# # # # #         logs.append(msg)
# # # # #         logbox.code("\n".join(logs))
# # # # #         progress.progress(step)
# # # # #         time.sleep(wait)

# # # # #     if prob < 0.4:
# # # # #         log("📞 Low payment likelihood detected — initiating contact...", 20)
# # # # #         log("📧 Sending personalized reminder email...", 40)
# # # # #         log("💬 Scheduling SMS payment reminder...", 60)
# # # # #         log("🤖 AI agent recommending payment plan options...", 80)
# # # # #         log("🧾 Following up with billing team for escalation...", 100)
# # # # #         st.warning("Low payment probability. Follow-up plan generated.")
# # # # #     else:
# # # # #         log("✅ High payment likelihood detected.", 100)
# # # # #         st.success("No further action needed — likely on-time payment.")

# # # # # # ---------------------------------------------------------------------
# # # # # # 🎯 Streamlit Layout
# # # # # # ---------------------------------------------------------------------
# # # # # st.title("AI-Powered Revenue Cycle Management (RCM) Suite")
# # # # # st.caption("Predict • Automate • Optimize")
# # # # # st.markdown("---")

# # # # # tabs = st.tabs([
# # # # #     "Denial Prediction & Prevention",
# # # # #     "AI-Assisted Coding",
# # # # #     "Prior Authorization Automation",
# # # # #     "Billing & Collections Optimization"
# # # # # ])

# # # # # # ---------------------------------------------------------------------
# # # # # # 🧾 Tab 1: Denial Prediction
# # # # # # ---------------------------------------------------------------------
# # # # # with tabs[0]:
# # # # #     st.header("Denial Prediction & Prevention")
# # # # #     col1, col2 = st.columns(2)
# # # # #     with col1:
# # # # #         patient_id = st.text_input("Patient ID", "P001", key="denial_pid")
# # # # #         age = st.number_input("Age", 0, 120, 45, key="denial_age")
# # # # #         gender = st.selectbox("Gender", ["M", "F"], key="denial_gender")
# # # # #         insurance_type = st.selectbox("Insurance Type", ["PPO", "HMO", "Medicare", "Medicaid"], key="denial_ins_type")
# # # # #         state = st.selectbox("State", ["CA", "NY", "TX", "FL", "NJ"], key="denial_state")
# # # # #         chronic_condition = st.selectbox("Chronic Condition", [0, 1], key="denial_chronic")
# # # # #     with col2:
# # # # #         claim_amount = st.number_input("Claim Amount ($)", 0.0, 50000.0, 2500.0, key="denial_claim_amt")
# # # # #         previous_denials = st.number_input("Previous Denials (6 months)", 0, 10, 1, key="denial_prev")
# # # # #         provider_experience = st.number_input("Provider Experience (yrs)", 0, 40, 10, key="denial_exp")
# # # # #         payer_coverage_ratio = st.slider("Payer Coverage Ratio", 0.0, 1.0, 0.75, key="denial_pcr")
# # # # #         claim_complexity = st.slider("Claim Complexity (0-1)", 0.0, 1.0, 0.5, key="denial_complex")

# # # # #     if st.button("Predict Denial Likelihood", key="denial_button"):
# # # # #         try:
# # # # #             inputs = dict(
# # # # #                 patient_id=patient_id,
# # # # #                 age=age,
# # # # #                 gender=gender,
# # # # #                 insurance_type=insurance_type,
# # # # #                 state=state,
# # # # #                 chronic_condition=chronic_condition,
# # # # #                 claim_amount=claim_amount,
# # # # #                 previous_denials_6m=previous_denials,
# # # # #                 provider_experience=provider_experience,
# # # # #                 payer_coverage_ratio=payer_coverage_ratio,
# # # # #                 claim_complexity=claim_complexity,
# # # # #             )
# # # # #             prob = predict_denial(inputs)
# # # # #             st.metric("Denial Probability", f"{prob:.2f}")
# # # # #             if prob > 0.6:
# # # # #                 st.error("⚠️ High denial risk — recommend pre-submission QA review.")
# # # # #             else:
# # # # #                 st.success("✅ Low denial likelihood — claim can proceed.")
# # # # #         except Exception as e:
# # # # #             st.error(f"Prediction error: {e}")

# # # # # # ---------------------------------------------------------------------
# # # # # # 🧠 Tab 2: Coding
# # # # # # ---------------------------------------------------------------------
# # # # # with tabs[1]:
# # # # #     st.header("AI-Assisted Coding from Clinical Notes")
# # # # #     note = st.text_area("Enter Doctor's Note", key="coding_note")
# # # # #     if st.button("Generate CPT/ICD-10 Codes", key="coding_button"):
# # # # #         cpt, icd = predict_coding(note)
# # # # #         st.write("**Predicted CPT Codes:**", cpt)
# # # # #         st.write("**Predicted ICD-10 Codes:**", icd)

# # # # # # ---------------------------------------------------------------------
# # # # # # 🤖 Tab 3: Prior Authorization
# # # # # # ---------------------------------------------------------------------
# # # # # with tabs[2]:
# # # # #     st.header("Prior Authorization Automation")
# # # # #     col1, col2 = st.columns(2)
# # # # #     with col1:
# # # # #         claim_id = st.text_input("Claim ID", "C123", key="pa_claimid")
# # # # #         age = st.number_input("Age", 0, 120, 50, key="pa_age")
# # # # #         gender = st.selectbox("Gender", ["M", "F"], key="pa_gender")
# # # # #         medical_specialty = st.selectbox("Specialty", ["Cardiology", "Orthopedics", "Oncology", "Radiology"], key="pa_spec")
# # # # #         insurance_type = st.selectbox("Insurance", ["Commercial", "Medicare", "Medicaid"], key="pa_insurance")
# # # # #         claim_amount = st.number_input("Claim Amount ($)", 0.0, 100000.0, 4000.0, key="pa_amount")
# # # # #     with col2:
# # # # #         claim_category = st.selectbox("Claim Category", ["Regular", "High Value", "Surgery", "Imaging"], key="pa_category")
# # # # #         plan_type = st.selectbox("Plan Type", ["HMO", "PPO"], key="pa_plan")
# # # # #         hospital_region = st.selectbox("Region", ["East", "West", "North", "South"], key="pa_region")
# # # # #         risk_score = st.slider("Patient Risk Score", 0.0, 1.0, 0.4, key="pa_risk")
# # # # #         urgent_case = st.selectbox("Urgent Case?", [0, 1], key="pa_urgent")

# # # # #     if st.button("Run Prior Authorization Bot", key="pa_button"):
# # # # #         try:
# # # # #             inputs = dict(
# # # # #                 claim_id=claim_id,
# # # # #                 age=age,
# # # # #                 gender=gender,
# # # # #                 medical_specialty=medical_specialty,
# # # # #                 insurance_type=insurance_type,
# # # # #                 plan_type=plan_type,
# # # # #                 hospital_region=hospital_region,
# # # # #                 claim_amount=claim_amount,
# # # # #                 claim_category=claim_category,
# # # # #                 risk_score=risk_score,
# # # # #                 urgent_case=urgent_case,
# # # # #             )
# # # # #             pa_bot_simulation(inputs)
# # # # #         except Exception as e:
# # # # #             st.error(f"Prediction error: {e}")

# # # # # # ---------------------------------------------------------------------
# # # # # # 💳 Tab 4: Billing
# # # # # # ---------------------------------------------------------------------
# # # # # with tabs[3]:
# # # # #     st.header("Patient Billing & Collections Optimization")
# # # # #     col1, col2 = st.columns(2)
# # # # #     with col1:
# # # # #         patient_id = st.text_input("Patient ID", "P555", key="bill_pid")
# # # # #         age = st.number_input("Age", 0, 120, 40, key="bill_age")
# # # # #         gender = st.selectbox("Gender", ["M", "F"], key="bill_gender")
# # # # #         insurance_type = st.selectbox("Insurance Type", ["PPO", "HMO", "Medicare", "Medicaid"], key="bill_ins_type")
# # # # #         balance_due = st.number_input("Balance Due ($)", 0.0, 10000.0, 1200.0, key="bill_balance")
# # # # #         num_reminders = st.number_input("Reminders Sent", 0, 10, 1, key="bill_reminders")
# # # # #     with col2:
# # # # #         credit_score = st.slider("Credit Score", 300, 850, 680, key="bill_credit")
# # # # #         patient_engagement_score = st.slider("Engagement Score", 0.0, 1.0, 0.6, key="bill_engage")
# # # # #         days_in_ar = st.number_input("Days in AR", 0, 120, 30, key="bill_daysar")
# # # # #         visit_type = st.selectbox("Visit Type", ["Inpatient", "Outpatient", "ER"], key="bill_visit")
# # # # #         has_payment_plan = st.selectbox("Payment Plan Exists", [0, 1], key="bill_plan")

# # # # #     if st.button("Run Billing Follow-Up Bot", key="bill_button"):
# # # # #         try:
# # # # #             inputs = dict(
# # # # #                 patient_id=patient_id,
# # # # #                 age=age,
# # # # #                 gender=gender,
# # # # #                 insurance_type=insurance_type,
# # # # #                 balance_due=balance_due,
# # # # #                 num_reminders_sent=num_reminders,
# # # # #                 credit_score=credit_score,
# # # # #                 patient_engagement_score=patient_engagement_score,
# # # # #                 days_in_ar=days_in_ar,
# # # # #                 visit_type=visit_type,
# # # # #                 has_payment_plan=has_payment_plan,
# # # # #             )
# # # # #             billing_bot_simulation(inputs)
# # # # #         except Exception as e:
# # # # #             st.error(f"Prediction error: {e}")
# # # # import streamlit as st
# # # # import pandas as pd
# # # # import numpy as np
# # # # import joblib, os, time, random
# # # # from catboost import CatBoostClassifier, Pool
# # # # from xgboost import XGBClassifier
# # # # import lightgbm as lgb

# # # # st.set_page_config(page_title="AI-Powered RCM Suite", layout="wide")

# # # # # ---------------------------------------------------------------------
# # # # # 🧠 Load Models
# # # # # ---------------------------------------------------------------------
# # # # @st.cache_resource
# # # # def load_models():
# # # #     models = {}
# # # #     if os.path.exists("models/denial_model.cbm"):
# # # #         m = CatBoostClassifier()
# # # #         m.load_model("models/denial_model.cbm")
# # # #         models["denial"] = m
# # # #     if os.path.exists("models/coding_model.pkl"):
# # # #         models["coding"] = joblib.load("models/coding_model.pkl")
# # # #     if os.path.exists("models/pa_model.txt"):
# # # #         models["pa"] = lgb.Booster(model_file="models/pa_model.txt")
# # # #     if os.path.exists("models/billing_model.json"):
# # # #         xgb = XGBClassifier()
# # # #         xgb.load_model("models/billing_model.json")
# # # #         models["billing"] = xgb
# # # #     return models

# # # # models = load_models()

# # # # # ---------------------------------------------------------------------
# # # # # 🧩 Align Features
# # # # # ---------------------------------------------------------------------
# # # # def align_features(df, model, model_type):
# # # #     if model_type == "catboost":
# # # #         feat_names = model.feature_names_
# # # #     elif model_type == "xgboost":
# # # #         feat_names = model.get_booster().feature_names
# # # #     elif model_type == "lightgbm":
# # # #         feat_names = model.feature_name()
# # # #     else:
# # # #         feat_names = df.columns.tolist()

# # # #     # Add missing features
# # # #     for f in feat_names:
# # # #         if f not in df.columns:
# # # #             df[f] = 0

# # # #     # Drop extras not in model
# # # #     df = df[feat_names]
# # # #     return df

# # # # # ---------------------------------------------------------------------
# # # # # 🧾 Denial Prediction (CatBoost)
# # # # # ---------------------------------------------------------------------
# # # # def predict_denial(inputs):
# # # #     df = pd.DataFrame([inputs])

# # # #     # Drop IDs — not predictive
# # # #     for col in ["patient_id", "claim_id"]:
# # # #         if col in df.columns:
# # # #             df = df.drop(columns=[col])

# # # #     # Convert ALL non-numeric features to category for CatBoost
# # # #     cat_features = []
# # # #     for col in df.columns:
# # # #         if df[col].dtype == "object" or df[col].dtype.name == "category":
# # # #             df[col] = df[col].astype("category")
# # # #             cat_features.append(col)

# # # #     df = align_features(df, models["denial"], "catboost")

# # # #     # Make sure expected categorical features exist
# # # #     for feat in models["denial"].get_cat_feature_indices():
# # # #         name = models["denial"].feature_names_[feat]
# # # #         if name in df.columns:
# # # #             df[name] = df[name].astype("category")

# # # #     pool = Pool(df, cat_features=list(df.select_dtypes(include=["category"]).columns))
# # # #     return float(models["denial"].predict_proba(pool)[0][1])

# # # # # ---------------------------------------------------------------------
# # # # # 🧠 Coding Prediction
# # # # # ---------------------------------------------------------------------
# # # # def predict_coding(note):
# # # #     note = note.lower()
# # # #     if "colonoscopy" in note:
# # # #         return ["45378"], ["K62.5"]
# # # #     elif "chest pain" in note or "angina" in note:
# # # #         return ["93000", "99284"], ["I20.9"]
# # # #     elif "diabetes" in note:
# # # #         return ["83036"], ["E11.9"]
# # # #     elif "hypertension" in note:
# # # #         return ["93015"], ["I10"]
# # # #     else:
# # # #         return ["99213"], ["Z00.0"]

# # # # # ---------------------------------------------------------------------
# # # # # 🤖 Prior Authorization BOT (LightGBM)
# # # # # ---------------------------------------------------------------------
# # # # def pa_bot_simulation(inputs):
# # # #     st.subheader("AI Prior Authorization Bot Workflow")

# # # #     df = pd.DataFrame([inputs])
# # # #     for c in df.select_dtypes(include=["object"]).columns:
# # # #         df[c] = df[c].astype("category").cat.codes
# # # #     df = align_features(df, models["pa"], "lightgbm")

# # # #     prob = float(models["pa"].predict(df)[0])
# # # #     st.info(f"Model probability (PA required): {prob:.2f}")

# # # #     if prob < 0.5:
# # # #         st.success("✅ No Prior Authorization required.")
# # # #         return

# # # #     st.warning("⚠️ Prior Authorization required. Initiating automation...")
# # # #     progress = st.progress(0)
# # # #     logbox = st.empty()
# # # #     logs = []

# # # #     def log(msg, step, wait=1.0):
# # # #         logs.append(msg)
# # # #         logbox.code("\n".join(logs))
# # # #         progress.progress(step)
# # # #         time.sleep(wait)

# # # #     log("🔍 Checking payer API for prior authorizations...", 10)
# # # #     log("📁 No record found — preparing submission packet...", 30)
# # # #     log("🧠 Summarizing clinical justification...", 50)
# # # #     log("📤 Submitting via payer portal...", 75)
# # # #     log("⏳ Awaiting response...", 90)

# # # #     status = random.choice(["Approved", "Pending", "Denied"])
# # # #     log(f"📨 Response: {status}", 100)

# # # #     if status == "Approved":
# # # #         st.success("✅ Approved — claim routed to billing.")
# # # #     elif status == "Pending":
# # # #         st.info("⌛ Pending — bot will auto-check every 6h.")
# # # #     else:
# # # #         st.error("❌ Denied — appeal initiated automatically.")

# # # # # ---------------------------------------------------------------------
# # # # # 💳 Billing Optimization BOT (XGBoost)
# # # # # ---------------------------------------------------------------------
# # # # def billing_bot_simulation(inputs):
# # # #     st.subheader("AI Billing Follow-Up Bot Workflow")

# # # #     df = pd.DataFrame([inputs])

# # # #     # Drop IDs
# # # #     for col in ["patient_id", "claim_id"]:
# # # #         if col in df.columns:
# # # #             df = df.drop(columns=[col])

# # # #     # Encode objects
# # # #     for c in df.select_dtypes(include=["object"]).columns:
# # # #         df[c] = df[c].astype("category").cat.codes

# # # #     df = align_features(df, models["billing"], "xgboost")
# # # #     prob = float(models["billing"].predict_proba(df)[0][1])
# # # #     st.metric("Payment Probability", f"{prob:.2f}")

# # # #     progress = st.progress(0)
# # # #     logbox = st.empty()
# # # #     logs = []

# # # #     def log(msg, step, wait=1.0):
# # # #         logs.append(msg)
# # # #         logbox.code("\n".join(logs))
# # # #         progress.progress(step)
# # # #         time.sleep(wait)

# # # #     if prob < 0.4:
# # # #         log("📞 Low payment likelihood detected...", 20)
# # # #         log("📧 Sending personalized reminder email...", 40)
# # # #         log("💬 Scheduling SMS follow-up...", 60)
# # # #         log("🤖 Suggesting payment plan...", 80)
# # # #         log("🧾 Notifying billing team...", 100)
# # # #         st.warning("Follow-up plan generated.")
# # # #     else:
# # # #         log("✅ High payment likelihood detected.", 100)
# # # #         st.success("No further action required.")

# # # # # ---------------------------------------------------------------------
# # # # # 🎯 Layout
# # # # # ---------------------------------------------------------------------
# # # # st.title("AI-Powered Revenue Cycle Management (RCM) Suite")
# # # # st.caption("Predict • Automate • Optimize")
# # # # st.markdown("---")

# # # # tabs = st.tabs([
# # # #     "Denial Prediction & Prevention",
# # # #     "AI-Assisted Coding",
# # # #     "Prior Authorization Automation",
# # # #     "Billing & Collections Optimization"
# # # # ])

# # # # # ---------------------------------------------------------------------
# # # # # 🧾 Tab 1: Denial Prediction
# # # # # ---------------------------------------------------------------------
# # # # with tabs[0]:
# # # #     st.header("Denial Prediction & Prevention")
# # # #     col1, col2 = st.columns(2)
# # # #     with col1:
# # # #         patient_id = st.text_input("Patient ID", "P001", key="den_pid")
# # # #         age = st.number_input("Age", 0, 120, 45, key="den_age")
# # # #         gender = st.selectbox("Gender", ["M", "F"], key="den_gender")
# # # #         insurance_type = st.selectbox("Insurance Type", ["PPO", "HMO", "Medicare", "Medicaid"], key="den_ins")
# # # #         state = st.selectbox("State", ["CA", "NY", "TX", "FL", "NJ"], key="den_state")
# # # #         chronic_condition = st.selectbox("Chronic Condition", [0, 1], key="den_chronic")
# # # #         procedure_category = st.selectbox("Procedure Category", ["Surgery", "Radiology", "Lab", "Consult"], key="den_proc")
# # # #     with col2:
# # # #         claim_amount = st.number_input("Claim Amount ($)", 0.0, 50000.0, 2500.0, key="den_claim_amt")
# # # #         previous_denials = st.number_input("Previous Denials (6m)", 0, 10, 1, key="den_prev")
# # # #         provider_experience = st.number_input("Provider Exp (yrs)", 0, 40, 10, key="den_exp")
# # # #         payer_coverage_ratio = st.slider("Payer Coverage Ratio", 0.0, 1.0, 0.75, key="den_pcr")
# # # #         claim_complexity = st.slider("Claim Complexity", 0.0, 1.0, 0.5, key="den_complex")

# # # #     if st.button("Predict Denial Likelihood", key="den_btn"):
# # # #         try:
# # # #             inputs = dict(
# # # #                 patient_id=patient_id,
# # # #                 age=age,
# # # #                 gender=gender,
# # # #                 insurance_type=insurance_type,
# # # #                 state=state,
# # # #                 chronic_condition=chronic_condition,
# # # #                 procedure_category=procedure_category,
# # # #                 claim_amount=claim_amount,
# # # #                 previous_denials_6m=previous_denials,
# # # #                 provider_experience=provider_experience,
# # # #                 payer_coverage_ratio=payer_coverage_ratio,
# # # #                 claim_complexity=claim_complexity,
# # # #             )
# # # #             prob = predict_denial(inputs)
# # # #             st.metric("Denial Probability", f"{prob:.2f}")
# # # #             if prob > 0.6:
# # # #                 st.error("⚠️ High denial risk — QA review advised.")
# # # #             else:
# # # #                 st.success("✅ Low denial likelihood — claim can proceed.")
# # # #         except Exception as e:
# # # #             st.error(f"Prediction error: {e}")

# # # # # ---------------------------------------------------------------------
# # # # # 🧠 Tab 2: Coding
# # # # # ---------------------------------------------------------------------
# # # # with tabs[1]:
# # # #     st.header("AI-Assisted Coding from Clinical Notes")
# # # #     note = st.text_area("Enter Doctor’s Note", key="code_note")
# # # #     if st.button("Generate CPT/ICD-10 Codes", key="code_btn"):
# # # #         cpt, icd = predict_coding(note)
# # # #         st.write("**Predicted CPT Codes:**", cpt)
# # # #         st.write("**Predicted ICD-10 Codes:**", icd)

# # # # # ---------------------------------------------------------------------
# # # # # 🤖 Tab 3: PA Automation
# # # # # ---------------------------------------------------------------------
# # # # with tabs[2]:
# # # #     st.header("Prior Authorization Automation")
# # # #     col1, col2 = st.columns(2)
# # # #     with col1:
# # # #         claim_id = st.text_input("Claim ID", "C123", key="pa_claimid")
# # # #         age = st.number_input("Age", 0, 120, 50, key="pa_age")
# # # #         gender = st.selectbox("Gender", ["M", "F"], key="pa_gender")
# # # #         specialty = st.selectbox("Specialty", ["Cardiology", "Ortho", "Oncology", "Radiology"], key="pa_spec")
# # # #         insurance_type = st.selectbox("Insurance", ["Commercial", "Medicare", "Medicaid"], key="pa_ins")
# # # #         claim_amount = st.number_input("Claim Amount ($)", 0.0, 100000.0, 4000.0, key="pa_amount")
# # # #     with col2:
# # # #         category = st.selectbox("Claim Category", ["Regular", "High Value", "Surgery", "Imaging"], key="pa_category")
# # # #         plan_type = st.selectbox("Plan Type", ["HMO", "PPO"], key="pa_plan")
# # # #         region = st.selectbox("Region", ["East", "West", "North", "South"], key="pa_region")
# # # #         risk_score = st.slider("Risk Score", 0.0, 1.0, 0.4, key="pa_risk")
# # # #         urgent = st.selectbox("Urgent?", [0, 1], key="pa_urgent")

# # # #     if st.button("Run PA Bot", key="pa_btn"):
# # # #         try:
# # # #             inputs = dict(
# # # #                 claim_id=claim_id,
# # # #                 age=age,
# # # #                 gender=gender,
# # # #                 medical_specialty=specialty,
# # # #                 insurance_type=insurance_type,
# # # #                 plan_type=plan_type,
# # # #                 hospital_region=region,
# # # #                 claim_amount=claim_amount,
# # # #                 claim_category=category,
# # # #                 risk_score=risk_score,
# # # #                 urgent_case=urgent,
# # # #             )
# # # #             pa_bot_simulation(inputs)
# # # #         except Exception as e:
# # # #             st.error(f"Prediction error: {e}")

# # # # # ---------------------------------------------------------------------
# # # # # 💳 Tab 4: Billing
# # # # # ---------------------------------------------------------------------
# # # # with tabs[3]:
# # # #     st.header("Billing & Collections Optimization")
# # # #     col1, col2 = st.columns(2)
# # # #     with col1:
# # # #         patient_id = st.text_input("Patient ID", "P555", key="bill_pid")
# # # #         age = st.number_input("Age", 0, 120, 40, key="bill_age")
# # # #         gender = st.selectbox("Gender", ["M", "F"], key="bill_gender")
# # # #         insurance_type = st.selectbox("Insurance Type", ["PPO", "HMO", "Medicare", "Medicaid"], key="bill_ins")
# # # #         balance_due = st.number_input("Balance Due ($)", 0.0, 10000.0, 1200.0, key="bill_balance")
# # # #         num_reminders = st.number_input("Reminders Sent", 0, 10, 1, key="bill_rem")
# # # #     with col2:
# # # #         credit_score = st.slider("Credit Score", 300, 850, 680, key="bill_credit")
# # # #         patient_eng = st.slider("Engagement Score", 0.0, 1.0, 0.6, key="bill_eng")
# # # #         days_in_ar = st.number_input("Days in AR", 0, 120, 30, key="bill_days")
# # # #         visit_type = st.selectbox("Visit Type", ["Inpatient", "Outpatient", "ER"], key="bill_visit")
# # # #         has_plan = st.selectbox("Payment Plan Exists", [0, 1], key="bill_plan")

# # # #     if st.button("Run Billing Bot", key="bill_btn"):
# # # #         try:
# # # #             inputs = dict(
# # # #                 patient_id=patient_id,
# # # #                 age=age,
# # # #                 gender=gender,
# # # #                 insurance_type=insurance_type,
# # # #                 balance_due=balance_due,
# # # #                 num_reminders_sent=num_reminders,
# # # #                 credit_score=credit_score,
# # # #                 patient_engagement_score=patient_eng,
# # # #                 days_in_ar=days_in_ar,
# # # #                 visit_type=visit_type,
# # # #                 has_payment_plan=has_plan,
# # # #             )
# # # #             billing_bot_simulation(inputs)
# # # #         except Exception as e:
# # # #             st.error(f"Prediction error: {e}")
# # # # import streamlit as st
# # # # import pandas as pd
# # # # import numpy as np
# # # # import joblib, os, time, random
# # # # from catboost import CatBoostClassifier, Pool
# # # # from xgboost import XGBClassifier
# # # # import lightgbm as lgb

# # # # st.set_page_config(page_title="AI-Powered RCM Suite", layout="wide")

# # # # # ---------------------------------------------------------------------
# # # # # 🧠 Load Models
# # # # # ---------------------------------------------------------------------
# # # # @st.cache_resource
# # # # def load_models():
# # # #     models = {}
# # # #     if os.path.exists("models/denial_model.cbm"):
# # # #         m = CatBoostClassifier()
# # # #         m.load_model("models/denial_model.cbm")
# # # #         models["denial"] = m
# # # #     if os.path.exists("models/coding_model.pkl"):
# # # #         models["coding"] = joblib.load("models/coding_model.pkl")
# # # #     if os.path.exists("models/pa_model.txt"):
# # # #         models["pa"] = lgb.Booster(model_file="models/pa_model.txt")
# # # #     if os.path.exists("models/billing_model.json"):
# # # #         xgb = XGBClassifier()
# # # #         xgb.load_model("models/billing_model.json")
# # # #         models["billing"] = xgb
# # # #     return models

# # # # models = load_models()

# # # # # ---------------------------------------------------------------------
# # # # # 🧩 Align Features
# # # # # ---------------------------------------------------------------------
# # # # def align_features(df, model, model_type):
# # # #     if model_type == "catboost":
# # # #         feat_names = model.feature_names_
# # # #     elif model_type == "xgboost":
# # # #         feat_names = model.get_booster().feature_names
# # # #     elif model_type == "lightgbm":
# # # #         feat_names = model.feature_name()
# # # #     else:
# # # #         feat_names = df.columns.tolist()

# # # #     for f in feat_names:
# # # #         if f not in df.columns:
# # # #             df[f] = 0
# # # #     df = df[feat_names]
# # # #     return df

# # # # # ---------------------------------------------------------------------
# # # # # 🧾 Denial Prediction (CatBoost)
# # # # # ---------------------------------------------------------------------
# # # # def predict_denial(inputs):
# # # #     df = pd.DataFrame([inputs])

# # # #     for col in ["patient_id", "claim_id"]:
# # # #         if col in df.columns:
# # # #             df = df.drop(columns=[col])

# # # #     cat_features = []
# # # #     for col in df.columns:
# # # #         if df[col].dtype == "object" or df[col].dtype.name == "category":
# # # #             df[col] = df[col].astype("category")
# # # #             cat_features.append(col)

# # # #     df = align_features(df, models["denial"], "catboost")

# # # #     for feat in models["denial"].get_cat_feature_indices():
# # # #         name = models["denial"].feature_names_[feat]
# # # #         if name in df.columns:
# # # #             df[name] = df[name].astype("category")

# # # #     pool = Pool(df, cat_features=list(df.select_dtypes(include=["category"]).columns))
# # # #     return float(models["denial"].predict_proba(pool)[0][1])

# # # # # ---------------------------------------------------------------------
# # # # # 🧠 Smarter AI-Assisted Coding
# # # # # ---------------------------------------------------------------------
# # # # def smart_predict_coding(note):
# # # #     """
# # # #     Smarter simulated AI coding engine
# # # #     Matches patterns + adds confidence scoring
# # # #     """
# # # #     note = note.lower()
# # # #     cpt_icd_map = {
# # # #         "colonoscopy": [("45378", "K62.5", "Diagnostic colonoscopy")],
# # # #         "chest pain": [("93000", "I20.9", "Electrocardiogram for chest pain")],
# # # #         "diabetes": [("83036", "E11.9", "HbA1C Test for diabetes management")],
# # # #         "hypertension": [("93015", "I10", "Cardiac stress test for hypertension")],
# # # #         "mri brain": [("70551", "G93.9", "MRI Brain without contrast")],
# # # #         "abdominal pain": [("99213", "R10.9", "Evaluation of abdominal pain")],
# # # #         "fracture": [("27786", "S82.90XA", "Fracture repair procedure")],
# # # #         "follow-up": [("99212", "Z09", "Follow-up visit after treatment")],
# # # #         "physical": [("99396", "Z00.00", "Annual physical exam")],
# # # #     }

# # # #     matches = []
# # # #     for key, values in cpt_icd_map.items():
# # # #         if key in note:
# # # #             for cpt, icd, desc in values:
# # # #                 confidence = round(random.uniform(0.78, 0.97), 2)
# # # #                 matches.append({"cpt": cpt, "icd": icd, "desc": desc, "confidence": confidence})

# # # #     if not matches:
# # # #         matches = [
# # # #             {
# # # #                 "cpt": random.choice(["99213", "99214"]),
# # # #                 "icd": random.choice(["Z00.0", "Z09"]),
# # # #                 "desc": "General office consultation",
# # # #                 "confidence": 0.72,
# # # #             }
# # # #         ]
# # # #     return matches

# # # # # ---------------------------------------------------------------------
# # # # # 🤖 Prior Authorization BOT (LightGBM)
# # # # # ---------------------------------------------------------------------
# # # # def pa_bot_simulation(inputs):
# # # #     st.subheader("AI Prior Authorization Bot Workflow")

# # # #     df = pd.DataFrame([inputs])
# # # #     for c in df.select_dtypes(include=["object"]).columns:
# # # #         df[c] = df[c].astype("category").cat.codes
# # # #     df = align_features(df, models["pa"], "lightgbm")

# # # #     prob = float(models["pa"].predict(df)[0])
# # # #     st.info(f"Model probability (PA required): {prob:.2f}")

# # # #     if prob < 0.5:
# # # #         st.success("✅ No Prior Authorization required.")
# # # #         return

# # # #     st.warning("⚠️ Prior Authorization required. Initiating automation...")
# # # #     progress = st.progress(0)
# # # #     logbox = st.empty()
# # # #     logs = []

# # # #     def log(msg, step, wait=1.0):
# # # #         logs.append(msg)
# # # #         logbox.code("\n".join(logs))
# # # #         progress.progress(step)
# # # #         time.sleep(wait)

# # # #     log("🔍 Checking payer API for prior authorizations...", 10)
# # # #     log("📁 No record found — preparing submission packet...", 30)
# # # #     log("🧠 Summarizing clinical justification...", 50)
# # # #     log("📤 Submitting via payer portal...", 75)
# # # #     log("⏳ Awaiting response...", 90)

# # # #     status = random.choice(["Approved", "Pending", "Denied"])
# # # #     log(f"📨 Response: {status}", 100)

# # # #     if status == "Approved":
# # # #         st.success("✅ Approved — claim routed to billing.")
# # # #     elif status == "Pending":
# # # #         st.info("⌛ Pending — bot will auto-check every 6h.")
# # # #     else:
# # # #         st.error("❌ Denied — appeal initiated automatically.")

# # # # # ---------------------------------------------------------------------
# # # # # 💳 Billing Optimization BOT (XGBoost)
# # # # # ---------------------------------------------------------------------
# # # # def billing_bot_simulation(inputs):
# # # #     st.subheader("AI Billing Follow-Up Bot Workflow")

# # # #     df = pd.DataFrame([inputs])

# # # #     for col in ["patient_id", "claim_id"]:
# # # #         if col in df.columns:
# # # #             df = df.drop(columns=[col])

# # # #     for c in df.select_dtypes(include=["object"]).columns:
# # # #         df[c] = df[c].astype("category").cat.codes

# # # #     df = align_features(df, models["billing"], "xgboost")
# # # #     prob = float(models["billing"].predict_proba(df)[0][1])
# # # #     st.metric("Payment Probability", f"{prob:.2f}")

# # # #     progress = st.progress(0)
# # # #     logbox = st.empty()
# # # #     logs = []

# # # #     def log(msg, step, wait=1.0):
# # # #         logs.append(msg)
# # # #         logbox.code("\n".join(logs))
# # # #         progress.progress(step)
# # # #         time.sleep(wait)

# # # #     if prob < 0.4:
# # # #         log("📞 Low payment likelihood detected...", 20)
# # # #         log("📧 Sending personalized reminder email...", 40)
# # # #         log("💬 Scheduling SMS follow-up...", 60)
# # # #         log("🤖 Suggesting payment plan...", 80)
# # # #         log("🧾 Notifying billing team...", 100)
# # # #         st.warning("Follow-up plan generated.")
# # # #     else:
# # # #         log("✅ High payment likelihood detected.", 100)
# # # #         st.success("No further action required.")

# # # # # ---------------------------------------------------------------------
# # # # # 🎯 Layout
# # # # # ---------------------------------------------------------------------
# # # # st.title("AI-Powered Revenue Cycle Management (RCM) Suite")
# # # # st.caption("Predict • Automate • Optimize")
# # # # st.markdown("---")

# # # # tabs = st.tabs([
# # # #     "Denial Prediction & Prevention",
# # # #     "AI-Assisted Coding",
# # # #     "Prior Authorization Automation",
# # # #     "Billing & Collections Optimization"
# # # # ])

# # # # # ---------------------------------------------------------------------
# # # # # 🧾 Tab 1: Denial Prediction
# # # # # ---------------------------------------------------------------------
# # # # with tabs[0]:
# # # #     st.header("Denial Prediction & Prevention")
# # # #     col1, col2 = st.columns(2)
# # # #     with col1:
# # # #         patient_id = st.text_input("Patient ID", "P001", key="den_pid")
# # # #         age = st.number_input("Age", 0, 120, 45, key="den_age")
# # # #         gender = st.selectbox("Gender", ["M", "F"], key="den_gender")
# # # #         insurance_type = st.selectbox("Insurance Type", ["PPO", "HMO", "Medicare", "Medicaid"], key="den_ins")
# # # #         state = st.selectbox("State", ["CA", "NY", "TX", "FL", "NJ"], key="den_state")
# # # #         chronic_condition = st.selectbox("Chronic Condition", [0, 1], key="den_chronic")
# # # #         procedure_category = st.selectbox("Procedure Category", ["Surgery", "Radiology", "Lab", "Consult"], key="den_proc")
# # # #     with col2:
# # # #         claim_amount = st.number_input("Claim Amount ($)", 0.0, 50000.0, 2500.0, key="den_claim_amt")
# # # #         previous_denials = st.number_input("Previous Denials (6m)", 0, 10, 1, key="den_prev")
# # # #         provider_experience = st.number_input("Provider Exp (yrs)", 0, 40, 10, key="den_exp")
# # # #         payer_coverage_ratio = st.slider("Payer Coverage Ratio", 0.0, 1.0, 0.75, key="den_pcr")
# # # #         claim_complexity = st.slider("Claim Complexity", 0.0, 1.0, 0.5, key="den_complex")

# # # #     if st.button("Predict Denial Likelihood", key="den_btn"):
# # # #         try:
# # # #             inputs = dict(
# # # #                 patient_id=patient_id,
# # # #                 age=age,
# # # #                 gender=gender,
# # # #                 insurance_type=insurance_type,
# # # #                 state=state,
# # # #                 chronic_condition=chronic_condition,
# # # #                 procedure_category=procedure_category,
# # # #                 claim_amount=claim_amount,
# # # #                 previous_denials_6m=previous_denials,
# # # #                 provider_experience=provider_experience,
# # # #                 payer_coverage_ratio=payer_coverage_ratio,
# # # #                 claim_complexity=claim_complexity,
# # # #             )
# # # #             prob = predict_denial(inputs)
# # # #             st.metric("Denial Probability", f"{prob:.2f}")
# # # #             if prob > 0.6:
# # # #                 st.error("⚠️ High denial risk — QA review advised.")
# # # #             else:
# # # #                 st.success("✅ Low denial likelihood — claim can proceed.")
# # # #         except Exception as e:
# # # #             st.error(f"Prediction error: {e}")

# # # # # ---------------------------------------------------------------------
# # # # # 🧠 Tab 2: AI-Assisted Coding
# # # # # ---------------------------------------------------------------------
# # # # with tabs[1]:
# # # #     st.header("🩺 AI-Assisted Coding from Clinical Notes")
# # # #     st.caption("Let AI extract CPT and ICD-10 codes intelligently from unstructured clinical notes.")

# # # #     note = st.text_area(
# # # #         "📝 Paste or type doctor's note below",
# # # #         height=180,
# # # #         placeholder="e.g., Patient presents with chest pain and hypertension for 2 weeks...",
# # # #         key="coding_note",
# # # #     )

# # # #     col1, col2 = st.columns([1, 1])
# # # #     with col1:
# # # #         if st.button("✨ Generate CPT/ICD-10 Codes", key="coding_generate"):
# # # #             with st.spinner("Analyzing clinical text..."):
# # # #                 progress = st.progress(0)
# # # #                 for i in range(0, 100, 20):
# # # #                     time.sleep(0.15)
# # # #                     progress.progress(i + 10)

# # # #                 codes = smart_predict_coding(note)
# # # #                 progress.progress(100)

# # # #                 st.success(f"✅ {len(codes)} codes generated.")
# # # #                 st.markdown("---")

# # # #                 for idx, entry in enumerate(codes, 1):
# # # #                     st.markdown(
# # # #                         f"""
# # # #                         <div style="background-color:#f6f8fa; padding:14px; border-radius:10px; margin-bottom:10px; border-left:5px solid #4CAF50;">
# # # #                         <h4 style="margin-bottom:4px;">💉 CPT {entry['cpt']}  |  ICD-10 {entry['icd']}</h4>
# # # #                         <p style="margin-bottom:4px; font-size:14px; color:#333;">{entry['desc']}</p>
# # # #                         <div style="font-size:13px; color:#555;">Confidence: <b>{int(entry['confidence']*100)}%</b></div>
# # # #                         </div>
# # # #                         """,
# # # #                         unsafe_allow_html=True,
# # # #                     )
# # # #     with col2:
# # # #         if st.button("🔁 Regenerate Suggestions", key="coding_refresh"):
# # # #             st.experimental_rerun()

# # # # # ---------------------------------------------------------------------
# # # # # 🤖 Tab 3: PA Automation
# # # # # ---------------------------------------------------------------------
# # # # with tabs[2]:
# # # #     st.header("Prior Authorization Automation")
# # # #     col1, col2 = st.columns(2)
# # # #     with col1:
# # # #         claim_id = st.text_input("Claim ID", "C123", key="pa_claimid")
# # # #         age = st.number_input("Age", 0, 120, 50, key="pa_age")
# # # #         gender = st.selectbox("Gender", ["M", "F"], key="pa_gender")
# # # #         specialty = st.selectbox("Specialty", ["Cardiology", "Ortho", "Oncology", "Radiology"], key="pa_spec")
# # # #         insurance_type = st.selectbox("Insurance", ["Commercial", "Medicare", "Medicaid"], key="pa_ins")
# # # #         claim_amount = st.number_input("Claim Amount ($)", 0.0, 100000.0, 4000.0, key="pa_amount")
# # # #     with col2:
# # # #         category = st.selectbox("Claim Category", ["Regular", "High Value", "Surgery", "Imaging"], key="pa_category")
# # # #         plan_type = st.selectbox("Plan Type", ["HMO", "PPO"], key="pa_plan")
# # # #         region = st.selectbox("Region", ["East", "West", "North", "South"], key="pa_region")
# # # #         risk_score = st.slider("Risk Score", 0.0, 1.0, 0.4, key="pa_risk")
# # # #         urgent = st.selectbox("Urgent?", [0, 1], key="pa_urgent")

# # # #     if st.button("Run PA Bot", key="pa_btn"):
# # # #         try:
# # # #             inputs = dict(
# # # #                 claim_id=claim_id,
# # # #                 age=age,
# # # #                 gender=gender,
# # # #                 medical_specialty=specialty,
# # # #                 insurance_type=insurance_type,
# # # #                 plan_type=plan_type,
# # # #                 hospital_region=region,
# # # #                 claim_amount=claim_amount,
# # # #                 claim_category=category,
# # # #                 risk_score=risk_score,
# # # #                 urgent_case=urgent,
# # # #             )
# # # #             pa_bot_simulation(inputs)
# # # #         except Exception as e:
# # # #             st.error(f"Prediction error: {e}")

# # # # # ---------------------------------------------------------------------
# # # # # 💳 Tab 4: Billing
# # # # # ---------------------------------------------------------------------
# # # # with tabs[3]:
# # # #     st.header("Billing & Collections Optimization")
# # # #     col1, col2 = st.columns(2)
# # # #     with col1:
# # # #         patient_id = st.text_input("Patient ID", "P555", key="bill_pid")
# # # #         age = st.number_input("Age", 0, 120, 40, key="bill_age")
# # # #         gender = st.selectbox("Gender", ["M", "F"], key="bill_gender")
# # # #         insurance_type = st.selectbox("Insurance Type", ["PPO", "HMO", "Medicare", "Medicaid"], key="bill_ins")
# # # #         balance_due = st.number_input("Balance Due ($)", 0.0, 10000.0, 1200.0, key="bill_balance")
# # # #         num_reminders = st.number_input("Reminders Sent", 0, 10, 1, key="bill_rem")
# # # #     with col2:
# # # #         credit_score = st.slider("Credit Score", 300, 850, 680, key="bill_credit")
# # # #         patient_eng = st.slider("Engagement Score", 0.0, 1.0, 0.6, key="bill_eng")
# # # #         days_in_ar = st.number_input("Days in AR", 0, 120, 30, key="bill_days")
# # # #         visit_type = st.selectbox("Visit Type", ["Inpatient", "Outpatient", "ER"], key="bill_visit")
# # # #         has_plan = st.selectbox("Payment Plan Exists", [0, 1], key="bill_plan")

# # # #     if st.button("Run Billing Bot", key="bill_btn"):
# # # #         try:
# # # #             inputs = dict(
# # # #                 patient_id=patient_id,
# # # #                 age=age,
# # # #                 gender=gender,
# # # #                 insurance_type=insurance_type,
# # # #                 balance_due=balance_due,
# # # #                 num_reminders_sent=num_reminders,
# # # #                 credit_score=credit_score,
# # # #                 patient_engagement_score=patient_eng,
# # # #                 days_in_ar=days_in_ar,
# # # #                 visit_type=visit_type,
# # # #                 has_payment_plan=has_plan,
# # # #             )
# # # #             billing_bot_simulation(inputs)
# # # #         except Exception as e:
# # # #             st.error(f"Prediction error: {e}")
# # # import streamlit as st
# # # import pandas as pd
# # # import numpy as np
# # # import joblib, os, time, random
# # # from catboost import CatBoostClassifier, Pool
# # # from xgboost import XGBClassifier
# # # import lightgbm as lgb

# # # st.set_page_config(page_title="AI-Powered RCM Suite", layout="wide")

# # # # ---------------------------------------------------------------------
# # # # 🧠 Load Models
# # # # ---------------------------------------------------------------------
# # # @st.cache_resource
# # # def load_models():
# # #     models = {}
# # #     if os.path.exists("models/denial_model.cbm"):
# # #         m = CatBoostClassifier()
# # #         m.load_model("models/denial_model.cbm")
# # #         models["denial"] = m
# # #     if os.path.exists("models/coding_model.pkl"):
# # #         models["coding"] = joblib.load("models/coding_model.pkl")
# # #     if os.path.exists("models/pa_model.txt"):
# # #         models["pa"] = lgb.Booster(model_file="models/pa_model.txt")
# # #     if os.path.exists("models/billing_model.json"):
# # #         xgb = XGBClassifier()
# # #         xgb.load_model("models/billing_model.json")
# # #         models["billing"] = xgb
# # #     return models

# # # models = load_models()

# # # # ---------------------------------------------------------------------
# # # # 🧩 Align Features
# # # # ---------------------------------------------------------------------
# # # def align_features(df, model, model_type):
# # #     if model_type == "catboost":
# # #         feat_names = model.feature_names_
# # #     elif model_type == "xgboost":
# # #         feat_names = model.get_booster().feature_names
# # #     elif model_type == "lightgbm":
# # #         feat_names = model.feature_name()
# # #     else:
# # #         feat_names = df.columns.tolist()

# # #     for f in feat_names:
# # #         if f not in df.columns:
# # #             df[f] = 0
# # #     df = df[feat_names]
# # #     return df

# # # # ---------------------------------------------------------------------
# # # # 🧾 Denial Prediction (CatBoost)
# # # # ---------------------------------------------------------------------
# # # def predict_denial(inputs):
# # #     df = pd.DataFrame([inputs])

# # #     for col in ["patient_id", "claim_id"]:
# # #         if col in df.columns:
# # #             df = df.drop(columns=[col])

# # #     cat_features = []
# # #     for col in df.columns:
# # #         if df[col].dtype == "object" or df[col].dtype.name == "category":
# # #             df[col] = df[col].astype("category")
# # #             cat_features.append(col)

# # #     df = align_features(df, models["denial"], "catboost")

# # #     for feat in models["denial"].get_cat_feature_indices():
# # #         name = models["denial"].feature_names_[feat]
# # #         if name in df.columns:
# # #             df[name] = df[name].astype("category")

# # #     pool = Pool(df, cat_features=list(df.select_dtypes(include=["category"]).columns))
# # #     return float(models["denial"].predict_proba(pool)[0][1])

# # # # ---------------------------------------------------------------------
# # # # 🧠 Smarter AI-Assisted Coding (Enhanced)
# # # # ---------------------------------------------------------------------
# # # def smart_predict_coding(note):
# # #     """
# # #     Enhanced AI-assisted coding logic:
# # #     • Smarter keyword + synonym mapping
# # #     • Distinct CPT/ICD outputs for variety
# # #     """
# # #     note = note.lower()
# # #     keyword_map = {
# # #         "brain tumor": [("70553", "C71.9", "MRI Brain with and without contrast for tumor evaluation")],
# # #         "brain clot": [("61624", "I63.9", "Endovascular therapy for cerebral clot")],
# # #         "stroke": [("70450", "I63.9", "CT Head without contrast for stroke assessment")],
# # #         "heart": [("93306", "I20.9", "Echocardiography, transthoracic, complete")],
# # #         "chest pain": [("93000", "R07.9", "Electrocardiogram for chest pain")],
# # #         "abdominal pain": [("74177", "R10.9", "CT abdomen and pelvis with contrast")],
# # #         "fracture": [("27786", "S82.90XA", "Fracture repair procedure")],
# # #         "follow-up": [("99212", "Z09", "Follow-up office visit")],
# # #         "diabetes": [("83036", "E11.9", "HbA1C test for diabetes management")],
# # #         "hypertension": [("93784", "I10", "Ambulatory blood pressure monitoring")],
# # #         "infection": [("87070", "A49.9", "Bacterial culture, general")],
# # #         "checkup": [("99397", "Z00.00", "Periodic general medical exam")],
# # #     }

# # #     matches = []
# # #     for key, values in keyword_map.items():
# # #         if key in note:
# # #             for cpt, icd, desc in values:
# # #                 confidence = round(random.uniform(0.83, 0.97), 2)
# # #                 matches.append({"cpt": cpt, "icd": icd, "desc": desc, "confidence": confidence})

# # #     if not matches:
# # #         matches = [
# # #             {
# # #                 "cpt": random.choice(["99213", "99214", "99215"]),
# # #                 "icd": random.choice(["Z00.0", "Z09", "R53.83"]),
# # #                 "desc": "General or follow-up office consultation",
# # #                 "confidence": round(random.uniform(0.68, 0.78), 2),
# # #             }
# # #         ]
# # #     return matches

# # # # ---------------------------------------------------------------------
# # # # 🤖 Prior Authorization BOT (LightGBM)
# # # # ---------------------------------------------------------------------
# # # def pa_bot_simulation(inputs):
# # #     st.subheader("AI Prior Authorization Bot Workflow")

# # #     df = pd.DataFrame([inputs])
# # #     for c in df.select_dtypes(include=["object"]).columns:
# # #         df[c] = df[c].astype("category").cat.codes
# # #     df = align_features(df, models["pa"], "lightgbm")

# # #     prob = float(models["pa"].predict(df)[0])
# # #     st.info(f"Model probability (PA required): {prob:.2f}")

# # #     if prob < 0.5:
# # #         st.success("✅ No Prior Authorization required.")
# # #         return

# # #     st.warning("⚠️ Prior Authorization required. Initiating automation...")
# # #     progress = st.progress(0)
# # #     logbox = st.empty()
# # #     logs = []

# # #     def log(msg, step, wait=1.0):
# # #         logs.append(msg)
# # #         logbox.code("\n".join(logs))
# # #         progress.progress(step)
# # #         time.sleep(wait)

# # #     log("🔍 Checking payer API for prior authorizations...", 10)
# # #     log("📁 No record found — preparing submission packet...", 30)
# # #     log("🧠 Summarizing clinical justification...", 50)
# # #     log("📤 Submitting via payer portal...", 75)
# # #     log("⏳ Awaiting response...", 90)

# # #     status = random.choice(["Approved", "Pending", "Denied"])
# # #     log(f"📨 Response: {status}", 100)

# # #     if status == "Approved":
# # #         st.success("✅ Approved — claim routed to billing.")
# # #     elif status == "Pending":
# # #         st.info("⌛ Pending — bot will auto-check every 6h.")
# # #     else:
# # #         st.error("❌ Denied — appeal initiated automatically.")

# # # # ---------------------------------------------------------------------
# # # # 💳 Billing Optimization BOT (XGBoost)
# # # # ---------------------------------------------------------------------
# # # def billing_bot_simulation(inputs):
# # #     st.subheader("AI Billing Follow-Up Bot Workflow")

# # #     df = pd.DataFrame([inputs])

# # #     for col in ["patient_id", "claim_id"]:
# # #         if col in df.columns:
# # #             df = df.drop(columns=[col])

# # #     for c in df.select_dtypes(include=["object"]).columns:
# # #         df[c] = df[c].astype("category").cat.codes

# # #     df = align_features(df, models["billing"], "xgboost")
# # #     prob = float(models["billing"].predict_proba(df)[0][1])
# # #     st.metric("Payment Probability", f"{prob:.2f}")

# # #     progress = st.progress(0)
# # #     logbox = st.empty()
# # #     logs = []

# # #     def log(msg, step, wait=1.0):
# # #         logs.append(msg)
# # #         logbox.code("\n".join(logs))
# # #         progress.progress(step)
# # #         time.sleep(wait)

# # #     if prob < 0.4:
# # #         log("📞 Low payment likelihood detected...", 20)
# # #         log("📧 Sending personalized reminder email...", 40)
# # #         log("💬 Scheduling SMS follow-up...", 60)
# # #         log("🤖 Suggesting payment plan...", 80)
# # #         log("🧾 Notifying billing team...", 100)
# # #         st.warning("Follow-up plan generated.")
# # #     else:
# # #         log("✅ High payment likelihood detected.", 100)
# # #         st.success("No further action required.")

# # # # ---------------------------------------------------------------------
# # # # 🎯 Layout
# # # # ---------------------------------------------------------------------
# # # st.title("AI-Powered Revenue Cycle Management (RCM) Suite")
# # # st.caption("Predict • Automate • Optimize")
# # # st.markdown("---")

# # # tabs = st.tabs([
# # #     "Denial Prediction & Prevention",
# # #     "AI-Assisted Coding",
# # #     "Prior Authorization Automation",
# # #     "Billing & Collections Optimization"
# # # ])

# # # # ---------------------------------------------------------------------
# # # # 🧾 Tab 1: Denial Prediction
# # # # ---------------------------------------------------------------------
# # # with tabs[0]:
# # #     st.header("Denial Prediction & Prevention")
# # #     col1, col2 = st.columns(2)
# # #     with col1:
# # #         patient_id = st.text_input("Patient ID", "P001", key="den_pid")
# # #         age = st.number_input("Age", 0, 120, 45, key="den_age")
# # #         gender = st.selectbox("Gender", ["M", "F"], key="den_gender")
# # #         insurance_type = st.selectbox("Insurance Type", ["PPO", "HMO", "Medicare", "Medicaid"], key="den_ins")
# # #         state = st.selectbox("State", ["CA", "NY", "TX", "FL", "NJ"], key="den_state")
# # #         chronic_condition = st.selectbox("Chronic Condition", [0, 1], key="den_chronic")
# # #         procedure_category = st.selectbox("Procedure Category", ["Surgery", "Radiology", "Lab", "Consult"], key="den_proc")
# # #     with col2:
# # #         claim_amount = st.number_input("Claim Amount ($)", 0.0, 50000.0, 2500.0, key="den_claim_amt")
# # #         previous_denials = st.number_input("Previous Denials (6m)", 0, 10, 1, key="den_prev")
# # #         provider_experience = st.number_input("Provider Exp (yrs)", 0, 40, 10, key="den_exp")
# # #         payer_coverage_ratio = st.slider("Payer Coverage Ratio", 0.0, 1.0, 0.75, key="den_pcr")
# # #         claim_complexity = st.slider("Claim Complexity", 0.0, 1.0, 0.5, key="den_complex")

# # #     if st.button("Predict Denial Likelihood", key="den_btn"):
# # #         try:
# # #             inputs = dict(
# # #                 patient_id=patient_id,
# # #                 age=age,
# # #                 gender=gender,
# # #                 insurance_type=insurance_type,
# # #                 state=state,
# # #                 chronic_condition=chronic_condition,
# # #                 procedure_category=procedure_category,
# # #                 claim_amount=claim_amount,
# # #                 previous_denials_6m=previous_denials,
# # #                 provider_experience=provider_experience,
# # #                 payer_coverage_ratio=payer_coverage_ratio,
# # #                 claim_complexity=claim_complexity,
# # #             )
# # #             prob = predict_denial(inputs)
# # #             st.metric("Denial Probability", f"{prob:.2f}")
# # #             if prob > 0.6:
# # #                 st.error("⚠️ High denial risk — QA review advised.")
# # #             else:
# # #                 st.success("✅ Low denial likelihood — claim can proceed.")
# # #         except Exception as e:
# # #             st.error(f"Prediction error: {e}")

# # # # ---------------------------------------------------------------------
# # # # 🧠 Tab 2: AI-Assisted Coding (with bright contrast)
# # # # ---------------------------------------------------------------------
# # # with tabs[1]:
# # #     st.header("🩺 AI-Assisted Coding from Clinical Notes")
# # #     st.caption("Let AI extract CPT and ICD-10 codes intelligently from unstructured clinical notes.")

# # #     note = st.text_area(
# # #         "📝 Paste or type doctor's note below",
# # #         height=180,
# # #         placeholder="e.g., Patient presents with chest pain and hypertension for 2 weeks...",
# # #         key="coding_note",
# # #     )

# # #     col1, col2 = st.columns([1, 1])
# # #     with col1:
# # #         if st.button("✨ Generate CPT/ICD-10 Codes", key="coding_generate"):
# # #             with st.spinner("Analyzing clinical text..."):
# # #                 progress = st.progress(0)
# # #                 for i in range(0, 100, 20):
# # #                     time.sleep(0.15)
# # #                     progress.progress(i + 10)

# # #                 codes = smart_predict_coding(note)
# # #                 progress.progress(100)

# # #                 st.success(f"✅ {len(codes)} codes generated.")
# # #                 st.markdown("---")

# # #                 for entry in codes:
# # #                     st.markdown(
# # #                         f"""
# # #                         <div style="
# # #                             background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
# # #                             padding:16px;
# # #                             border-radius:14px;
# # #                             margin-bottom:14px;
# # #                             box-shadow: 0 3px 6px rgba(0,0,0,0.15);
# # #                             border-left:6px solid #1565c0;
# # #                         ">
# # #                         <h4 style="color:#0d47a1;margin-bottom:4px;">
# # #                             💉 <b>CPT {entry['cpt']}</b>  &nbsp;|&nbsp;  🧠 <b>ICD-10 {entry['icd']}</b>
# # #                         </h4>
# # #                         <p style="color:#1a237e;font-size:14px;margin-bottom:6px;">{entry['desc']}</p>
# # #                         <p style="color:#0d47a1;font-size:13px;">Confidence: <b>{int(entry['confidence']*100)}%</b></p>
# # #                         </div>
# # #                         """,
# # #                         unsafe_allow_html=True,
# # #                     )
# # #     with col2:
# # #         if st.button("🔁 Regenerate Suggestions", key="coding_refresh"):
# # #             st.experimental_rerun()

# # # # ---------------------------------------------------------------------
# # # # 🤖 Tab 3: PA Automation
# # # # ---------------------------------------------------------------------
# # # with tabs[2]:
# # #     st.header("Prior Authorization Automation")
# # #     col1, col2 = st.columns(2)
# # #     with col1:
# # #         claim_id = st.text_input("Claim ID", "C123", key="pa_claimid")
# # #         age = st.number_input("Age", 0, 120, 50, key="pa_age")
# # #         gender = st.selectbox("Gender", ["M", "F"], key="pa_gender")
# # #         specialty = st.selectbox("Specialty", ["Cardiology", "Ortho", "Oncology", "Radiology"], key="pa_spec")
# # #         insurance_type = st.selectbox("Insurance", ["Commercial", "Medicare", "Medicaid"], key="pa_ins")
# # #         claim_amount = st.number_input("Claim Amount ($)", 0.0, 100000.0, 4000.0, key="pa_amount")
# # #     with col2:
# # #         category = st.selectbox("Claim Category", ["Regular", "High Value", "Surgery", "Imaging"], key="pa_category")
# # #         plan_type = st.selectbox("Plan Type", ["HMO", "PPO"], key="pa_plan")
# # #         region = st.selectbox("Region", ["East", "West", "North", "South"], key="pa_region")
# # #         risk_score = st.slider("Risk Score", 0.0, 1.0, 0.4, key="pa_risk")
# # #         urgent = st.selectbox("Urgent?", [0, 1], key="pa_urgent")

# # #     if st.button("Run PA Bot", key="pa_btn"):
# # #         try:
# # #             inputs = dict(
# # #                 claim_id=claim_id,
# # #                 age=age,
# # #                 gender=gender,
# # #                 medical_specialty=specialty,
# # #                 insurance_type=insurance_type,
# # #                 plan_type=plan_type,
# # #                 hospital_region=region,
# # #                 claim_amount=claim_amount,
# # #                 claim_category=category,
# # #                 risk_score=risk_score,
# # #                 urgent_case=urgent,
# # #             )
# # #             pa_bot_simulation(inputs)
# # #         except Exception as e:
# # #             st.error(f"Prediction error: {e}")

# # # # ---------------------------------------------------------------------
# # # # 💳 Tab 4: Billing
# # # # ---------------------------------------------------------------------
# # # with tabs[3]:
# # #     st.header("Billing & Collections Optimization")
# # #     col1, col2 = st.columns(2)
# # #     with col1:
# # #         patient_id = st.text_input("Patient ID", "P555", key="bill_pid")
# # #         age = st.number_input("Age", 0, 120, 40, key="bill_age")
# # #         gender = st.selectbox("Gender", ["M", "F"], key="bill_gender")
# # #         insurance_type = st.selectbox("Insurance Type", ["PPO", "HMO", "Medicare", "Medicaid"], key="bill_ins")
# # #         balance_due = st.number_input("Balance Due ($)", 0.0, 10000.0, 1200.0, key="bill_balance")
# # #         num_reminders = st.number_input("Reminders Sent", 0, 10, 1, key="bill_rem")
# # #     with col2:
# # #         credit_score = st.slider("Credit Score", 300, 850, 680, key="bill_credit")
# # #         patient_eng = st.slider("Engagement Score", 0.0, 1.0, 0.6, key="bill_eng")
# # #         days_in_ar = st.number_input("Days in AR", 0, 120, 30, key="bill_days")
# # #         visit_type = st.selectbox("Visit Type", ["Inpatient", "Outpatient", "ER"], key="bill_visit")
# # #         has_plan = st.selectbox("Payment Plan Exists", [0, 1], key="bill_plan")

# # #     if st.button("Run Billing Bot", key="bill_btn"):
# # #         try:
# # #             inputs = dict(
# # #                 patient_id=patient_id,
# # #                 age=age,
# # #                 gender=gender,
# # #                 insurance_type=insurance_type,
# # #                 balance_due=balance_due,
# # #                 num_reminders_sent=num_reminders,
# # #                 credit_score=credit_score,
# # #                 patient_engagement_score=patient_eng,
# # #                 days_in_ar=days_in_ar,
# # #                 visit_type=visit_type,
# # #                 has_payment_plan=has_plan,
# # #             )
# # #             billing_bot_simulation(inputs)
# # #         except Exception as e:
# # #             st.error(f"Prediction error: {e}")
# # import streamlit as st
# # import pandas as pd
# # import numpy as np
# # import joblib, os, time, random, re
# # from catboost import CatBoostClassifier, Pool
# # from xgboost import XGBClassifier
# # import lightgbm as lgb

# # st.set_page_config(page_title="AI-Powered RCM Suite", layout="wide")

# # # ---------------------------------------------------------------------
# # # 🧠 Load Models
# # # ---------------------------------------------------------------------
# # @st.cache_resource
# # def load_models():
# #     models = {}
# #     if os.path.exists("models/denial_model.cbm"):
# #         m = CatBoostClassifier()
# #         m.load_model("models/denial_model.cbm")
# #         models["denial"] = m
# #     if os.path.exists("models/coding_model.pkl"):
# #         models["coding"] = joblib.load("models/coding_model.pkl")
# #     if os.path.exists("models/pa_model.txt"):
# #         models["pa"] = lgb.Booster(model_file="models/pa_model.txt")
# #     if os.path.exists("models/billing_model.json"):
# #         xgb = XGBClassifier()
# #         xgb.load_model("models/billing_model.json")
# #         models["billing"] = xgb
# #     return models

# # models = load_models()

# # # ---------------------------------------------------------------------
# # # 🧩 Align Features
# # # ---------------------------------------------------------------------
# # def align_features(df, model, model_type):
# #     if model_type == "catboost":
# #         feat_names = model.feature_names_
# #     elif model_type == "xgboost":
# #         feat_names = model.get_booster().feature_names
# #     elif model_type == "lightgbm":
# #         feat_names = model.feature_name()
# #     else:
# #         feat_names = df.columns.tolist()
# #     for f in feat_names:
# #         if f not in df.columns:
# #             df[f] = 0
# #     df = df[feat_names]
# #     return df

# # # ---------------------------------------------------------------------
# # # 🧾 Denial Prediction (CatBoost)
# # # ---------------------------------------------------------------------
# # def predict_denial(inputs):
# #     df = pd.DataFrame([inputs])
# #     for col in ["patient_id", "claim_id"]:
# #         if col in df.columns:
# #             df = df.drop(columns=[col])
# #     for col in df.columns:
# #         if df[col].dtype == "object":
# #             df[col] = df[col].astype("category")
# #     df = align_features(df, models["denial"], "catboost")
# #     pool = Pool(df, cat_features=list(df.select_dtypes(include=["category"]).columns))
# #     return float(models["denial"].predict_proba(pool)[0][1])

# # # ---------------------------------------------------------------------
# # # 🧠 Smarter AI-Assisted Coding (Semantic)
# # # ---------------------------------------------------------------------
# # def smart_predict_coding(note):
# #     note = note.lower().strip()

# #     patterns = [
# #         (r"(mri|magnetic).*brain.*(contrast)?", ("70553", "C71.9", "MRI Brain with/without contrast")),
# #         (r"(ct|scan).*abdomen.*pelvis", ("74177", "R10.9", "CT abdomen and pelvis with contrast")),
# #         (r"brain.*(tumor|mass)", ("70553", "C71.9", "MRI Brain for tumor evaluation")),
# #         (r"(stroke|clot|embol)", ("61624", "I63.9", "Endovascular therapy for cerebral clot")),
# #         (r"chest pain|cardiac|angina", ("93000", "R07.9", "Electrocardiogram for chest pain")),
# #         (r"(fracture|broken bone)", ("27786", "S82.90XA", "Fracture repair procedure")),
# #         (r"(follow.?up|post.?visit)", ("99212", "Z09", "Follow-up office visit")),
# #         (r"(diabetes|hba1c|blood sugar)", ("83036", "E11.9", "HbA1C Test for diabetes management")),
# #         (r"(hypertension|bp|blood pressure)", ("93784", "I10", "Ambulatory blood pressure monitoring")),
# #         (r"(infection|culture|bacteria)", ("87070", "A49.9", "Bacterial culture, general")),
# #         (r"(check.?up|physical|annual exam|medical exam)", ("99397", "Z00.00", "Periodic general medical exam")),
# #         (r"(consult|office visit)", ("99213", "Z09", "General consultation visit")),
# #     ]

# #     for pattern, (cpt, icd, desc) in patterns:
# #         if re.search(pattern, note):
# #             confidence = round(random.uniform(0.83, 0.97), 2)
# #             return [{"cpt": cpt, "icd": icd, "desc": desc, "confidence": confidence}]

# #     # fallback default
# #     return [{
# #         "cpt": random.choice(["99213", "99214", "99215"]),
# #         "icd": random.choice(["Z00.0", "Z09", "R53.83"]),
# #         "desc": "General or follow-up office consultation",
# #         "confidence": round(random.uniform(0.68, 0.78), 2),
# #     }]

# # # ---------------------------------------------------------------------
# # # 🤖 Prior Authorization BOT (LightGBM)
# # # ---------------------------------------------------------------------
# # def pa_bot_simulation(inputs):
# #     st.subheader("AI Prior Authorization Bot Workflow")
# #     df = pd.DataFrame([inputs])
# #     for c in df.select_dtypes(include=["object"]).columns:
# #         df[c] = df[c].astype("category").cat.codes
# #     df = align_features(df, models["pa"], "lightgbm")
# #     prob = float(models["pa"].predict(df)[0])
# #     st.info(f"Model probability (PA required): {prob:.2f}")
# #     if prob < 0.5:
# #         st.success("✅ No Prior Authorization required.")
# #         return
# #     st.warning("⚠️ Prior Authorization required. Initiating automation...")
# #     progress = st.progress(0)
# #     logbox = st.empty()
# #     logs = []
# #     def log(msg, step, wait=1.0):
# #         logs.append(msg)
# #         logbox.code("\n".join(logs))
# #         progress.progress(step)
# #         time.sleep(wait)
# #     log("🔍 Checking payer API for prior authorizations...", 10)
# #     log("📁 Preparing submission packet...", 30)
# #     log("🧠 Summarizing clinical justification...", 50)
# #     log("📤 Submitting via payer portal...", 75)
# #     log("⏳ Awaiting response...", 90)
# #     status = random.choice(["Approved", "Pending", "Denied"])
# #     log(f"📨 Response: {status}", 100)
# #     if status == "Approved":
# #         st.success("✅ Approved — claim routed to billing.")
# #     elif status == "Pending":
# #         st.info("⌛ Pending — bot will auto-check every 6h.")
# #     else:
# #         st.error("❌ Denied — appeal initiated automatically.")

# # # ---------------------------------------------------------------------
# # # 💳 Billing Optimization BOT (XGBoost)
# # # ---------------------------------------------------------------------
# # def billing_bot_simulation(inputs):
# #     st.subheader("AI Billing Follow-Up Bot Workflow")
# #     df = pd.DataFrame([inputs])
# #     for col in ["patient_id", "claim_id"]:
# #         if col in df.columns:
# #             df = df.drop(columns=[col])
# #     for c in df.select_dtypes(include=["object"]).columns:
# #         df[c] = df[c].astype("category").cat.codes
# #     df = align_features(df, models["billing"], "xgboost")
# #     prob = float(models["billing"].predict_proba(df)[0][1])
# #     st.metric("Payment Probability", f"{prob:.2f}")
# #     progress = st.progress(0)
# #     logbox = st.empty()
# #     logs = []
# #     def log(msg, step, wait=1.0):
# #         logs.append(msg)
# #         logbox.code("\n".join(logs))
# #         progress.progress(step)
# #         time.sleep(wait)
# #     if prob < 0.4:
# #         log("📞 Low payment likelihood detected...", 20)
# #         log("📧 Sending personalized reminder email...", 40)
# #         log("💬 Scheduling SMS follow-up...", 60)
# #         log("🤖 Suggesting payment plan...", 80)
# #         log("🧾 Notifying billing team...", 100)
# #         st.warning("Follow-up plan generated.")
# #     else:
# #         log("✅ High payment likelihood detected.", 100)
# #         st.success("No further action required.")

# # # ---------------------------------------------------------------------
# # # 🎯 Layout
# # # ---------------------------------------------------------------------
# # st.title("AI-Powered Revenue Cycle Management (RCM) Suite")
# # st.caption("Predict • Automate • Optimize")
# # st.markdown("---")

# # tabs = st.tabs([
# #     "Denial Prediction & Prevention",
# #     "AI-Assisted Coding",
# #     "Prior Authorization Automation",
# #     "Billing & Collections Optimization"
# # ])

# # # ---------------------------------------------------------------------
# # # 🧾 Tab 1: Denial Prediction
# # # ---------------------------------------------------------------------
# # with tabs[0]:
# #     st.header("Denial Prediction & Prevention")
# #     col1, col2 = st.columns(2)
# #     with col1:
# #         patient_id = st.text_input("Patient ID", "P001", key="den_pid")
# #         age = st.number_input("Age", 0, 120, 45, key="den_age")
# #         gender = st.selectbox("Gender", ["M", "F"], key="den_gender")
# #         insurance_type = st.selectbox("Insurance Type", ["PPO", "HMO", "Medicare", "Medicaid"], key="den_ins")
# #         state = st.selectbox("State", ["CA", "NY", "TX", "FL", "NJ"], key="den_state")
# #         chronic_condition = st.selectbox("Chronic Condition", [0, 1], key="den_chronic")
# #         procedure_category = st.selectbox("Procedure Category", ["Surgery", "Radiology", "Lab", "Consult"], key="den_proc")
# #     with col2:
# #         claim_amount = st.number_input("Claim Amount ($)", 0.0, 50000.0, 2500.0, key="den_claim_amt")
# #         previous_denials = st.number_input("Previous Denials (6m)", 0, 10, 1, key="den_prev")
# #         provider_experience = st.number_input("Provider Exp (yrs)", 0, 40, 10, key="den_exp")
# #         payer_coverage_ratio = st.slider("Payer Coverage Ratio", 0.0, 1.0, 0.75, key="den_pcr")
# #         claim_complexity = st.slider("Claim Complexity", 0.0, 1.0, 0.5, key="den_complex")
# #     if st.button("Predict Denial Likelihood", key="den_btn"):
# #         try:
# #             inputs = dict(
# #                 patient_id=patient_id,
# #                 age=age,
# #                 gender=gender,
# #                 insurance_type=insurance_type,
# #                 state=state,
# #                 chronic_condition=chronic_condition,
# #                 procedure_category=procedure_category,
# #                 claim_amount=claim_amount,
# #                 previous_denials_6m=previous_denials,
# #                 provider_experience=provider_experience,
# #                 payer_coverage_ratio=payer_coverage_ratio,
# #                 claim_complexity=claim_complexity,
# #             )
# #             prob = predict_denial(inputs)
# #             st.metric("Denial Probability", f"{prob:.2f}")
# #             if prob > 0.6:
# #                 st.error("⚠️ High denial risk — QA review advised.")
# #             else:
# #                 st.success("✅ Low denial likelihood — claim can proceed.")
# #         except Exception as e:
# #             st.error(f"Prediction error: {e}")

# # # ---------------------------------------------------------------------
# # # 🧠 Tab 2: AI-Assisted Coding (with bright contrast)
# # # ---------------------------------------------------------------------
# # with tabs[1]:
# #     st.header("🩺 AI-Assisted Coding from Clinical Notes")
# #     st.caption("Let AI extract CPT and ICD-10 codes intelligently from unstructured clinical notes.")
# #     note = st.text_area("📝 Paste or type doctor's note below", height=180, placeholder="e.g., Patient presents with chest pain and hypertension for 2 weeks...", key="coding_note")
# #     col1, col2 = st.columns([1, 1])
# #     with col1:
# #         if st.button("✨ Generate CPT/ICD-10 Codes", key="coding_generate"):
# #             with st.spinner("Analyzing clinical text..."):
# #                 progress = st.progress(0)
# #                 for i in range(0, 100, 20):
# #                     time.sleep(0.15)
# #                     progress.progress(i + 10)
# #                 codes = smart_predict_coding(note)
# #                 progress.progress(100)
# #                 st.success(f"✅ {len(codes)} codes generated.")
# #                 st.markdown("---")
# #                 for entry in codes:
# #                     st.markdown(
# #                         f"""
# #                         <div style="background: linear-gradient(135deg, #bbdefb 0%, #e3f2fd 100%); padding:16px; border-radius:14px; margin-bottom:14px; box-shadow: 0 3px 6px rgba(0,0,0,0.15); border-left:6px solid #1565c0;">
# #                         <h4 style="color:#0d47a1;margin-bottom:4px;">💉 <b>CPT {entry['cpt']}</b>  |  🧠 <b>ICD-10 {entry['icd']}</b></h4>
# #                         <p style="color:#1a237e;font-size:14px;margin-bottom:6px;">{entry['desc']}</p>
# #                         <p style="color:#0d47a1;font-size:13px;">Confidence: <b>{int(entry['confidence']*100)}%</b></p>
# #                         </div>
# #                         """,
# #                         unsafe_allow_html=True,
# #                     )
# #     with col2:
# #         if st.button("🔁 Regenerate Suggestions", key="coding_refresh"):
# #             st.experimental_rerun()

# # # ---------------------------------------------------------------------
# # # 🤖 Tab 3: PA Automation
# # # ---------------------------------------------------------------------
# # with tabs[2]:
# #     st.header("Prior Authorization Automation")
# #     col1, col2 = st.columns(2)
# #     with col1:
# #         claim_id = st.text_input("Claim ID", "C123", key="pa_claimid")
# #         age = st.number_input("Age", 0, 120, 50, key="pa_age")
# #         gender = st.selectbox("Gender", ["M", "F"], key="pa_gender")
# #         specialty = st.selectbox("Specialty", ["Cardiology", "Ortho", "Oncology", "Radiology"], key="pa_spec")
# #         insurance_type = st.selectbox("Insurance", ["Commercial", "Medicare", "Medicaid"], key="pa_ins")
# #         claim_amount = st.number_input("Claim Amount ($)", 0.0, 100000.0, 4000.0, key="pa_amount")
# #     with col2:
# #         category = st.selectbox("Claim Category", ["Regular", "High Value", "Surgery", "Imaging"], key="pa_category")
# #         plan_type = st.selectbox("Plan Type", ["HMO", "PPO"], key="pa_plan")
# #         region = st.selectbox("Region", ["East", "West", "North", "South"], key="pa_region")
# #         risk_score = st.slider("Risk Score", 0.0, 1.0, 0.4, key="pa_risk")
# #         urgent = st.selectbox("Urgent?", [0, 1], key="pa_urgent")
# #     if st.button("Run PA Bot", key="pa_btn"):
# #         try:
# #             inputs = dict(
# #                 claim_id=claim_id,
# #                 age=age,
# #                 gender=gender,
# #                 medical_specialty=specialty,
# #                 insurance_type=insurance_type,
# #                 plan_type=plan_type,
# #                 hospital_region=region,
# #                 claim_amount=claim_amount,
# #                 claim_category=category,
# #                 risk_score=risk_score,
# #                 urgent_case=urgent,
# #             )
# #             pa_bot_simulation(inputs)
# #         except Exception as e:
# #             st.error(f"Prediction error: {e}")

# # # ---------------------------------------------------------------------
# # # 💳 Tab 4: Billing
# # # ---------------------------------------------------------------------
# # with tabs[3]:
# #     st.header("Billing & Collections Optimization")
# #     col1, col2 = st.columns(2)
# #     with col1:
# #         patient_id = st.text_input("Patient ID", "P555", key="bill_pid")
# #         age = st.number_input("Age", 0, 120, 40, key="bill_age")
# #         gender = st.selectbox("Gender", ["M", "F"], key="bill_gender")
# #         insurance_type = st.selectbox("Insurance Type", ["PPO", "HMO", "Medicare", "Medicaid"], key="bill_ins")
# #         balance_due = st.number_input("Balance Due ($)", 0.0, 10000.0, 1200.0, key="bill_balance")
# #         num_reminders = st.number_input("Reminders Sent", 0, 10, 1, key="bill_rem")
# #     with col2:
# #         credit_score = st.slider("Credit Score", 300, 850, 680, key="bill_credit")
# #         patient_eng = st.slider("Engagement Score", 0.0, 1.0, 0.6, key="bill_eng")
# #         days_in_ar = st.number_input("Days in AR", 0, 120, 30, key="bill_days")
# #         visit_type = st.selectbox("Visit Type", ["Inpatient", "Outpatient", "ER"], key="bill_visit")
# #         has_plan = st.selectbox("Payment Plan Exists", [0, 1], key="bill_plan")
# #     if st.button("Run Billing Bot", key="bill_btn"):
# #         try:
# #             inputs = dict(
# #                 patient_id=patient_id,
# #                 age=age,
# #                 gender=gender,
# #                 insurance_type=insurance_type,
# #                 balance_due=balance_due,
# #                 num_reminders_sent=num_reminders,
# #                 credit_score=credit_score,
# #                 patient_engagement_score=patient_eng,
# #                 days_in_ar=days_in_ar,
# #                 visit_type=visit_type,
# #                 has_payment_plan=has_plan,
# #             )
# #             billing_bot_simulation(inputs)
# #         except Exception as e:
# #             st.error(f"Prediction error: {e}")
# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib, os, time, random, re
# from catboost import CatBoostClassifier, Pool
# from xgboost import XGBClassifier
# import lightgbm as lgb

# st.set_page_config(page_title="AI-Powered RCM Suite", layout="wide")

# # ---------------------------------------------------------------------
# # 🧠 Load Models
# # ---------------------------------------------------------------------
# @st.cache_resource
# def load_models():
#     models = {}
#     if os.path.exists("models/denial_model.cbm"):
#         m = CatBoostClassifier()
#         m.load_model("models/denial_model.cbm")
#         models["denial"] = m
#     if os.path.exists("models/coding_model.pkl"):
#         models["coding"] = joblib.load("models/coding_model.pkl")
#     if os.path.exists("models/pa_model.txt"):
#         models["pa"] = lgb.Booster(model_file="models/pa_model.txt")
#     if os.path.exists("models/billing_model.json"):
#         xgb = XGBClassifier()
#         xgb.load_model("models/billing_model.json")
#         models["billing"] = xgb
#     return models

# models = load_models()

# # ---------------------------------------------------------------------
# # 🧩 Align Features
# # ---------------------------------------------------------------------
# def align_features(df, model, model_type):
#     if model_type == "catboost":
#         feat_names = model.feature_names_
#     elif model_type == "xgboost":
#         feat_names = model.get_booster().feature_names
#     elif model_type == "lightgbm":
#         feat_names = model.feature_name()
#     else:
#         feat_names = df.columns.tolist()

#     for f in feat_names:
#         if f not in df.columns:
#             df[f] = 0
#     df = df[feat_names]
#     return df

# # ---------------------------------------------------------------------
# # 🧾 Denial Prediction (CatBoost)
# # ---------------------------------------------------------------------
# def predict_denial(inputs):
#     df = pd.DataFrame([inputs])

#     for col in ["patient_id", "claim_id"]:
#         if col in df.columns:
#             df = df.drop(columns=[col])

#     cat_features = []
#     for col in df.columns:
#         if df[col].dtype == "object" or df[col].dtype.name == "category":
#             df[col] = df[col].astype("category")
#             cat_features.append(col)

#     df = align_features(df, models["denial"], "catboost")

#     for feat in models["denial"].get_cat_feature_indices():
#         name = models["denial"].feature_names_[feat]
#         if name in df.columns:
#             df[name] = df[name].astype("category")

#     pool = Pool(df, cat_features=list(df.select_dtypes(include=["category"]).columns))
#     return float(models["denial"].predict_proba(pool)[0][1])

# # ---------------------------------------------------------------------
# # 🧠 AI-Assisted Coding (semantic enhancement only)
# # ---------------------------------------------------------------------
# def smart_predict_coding(note):
#     note = note.lower().strip()

#     # Use regex & semantic matching for better mapping
#     patterns = [
#         (r"(mri|magnetic).*brain.*(contrast)?", ("70553", "C71.9", "MRI Brain with/without contrast")),
#         (r"(ct|scan).*abdomen.*pelvis", ("74177", "R10.9", "CT abdomen and pelvis with contrast")),
#         (r"brain.*(tumor|mass)", ("70553", "C71.9", "MRI Brain for tumor evaluation")),
#         (r"(stroke|clot|embol)", ("61624", "I63.9", "Endovascular therapy for cerebral clot")),
#         (r"chest pain|cardiac|angina", ("93000", "R07.9", "Electrocardiogram for chest pain")),
#         (r"(fracture|broken bone)", ("27786", "S82.90XA", "Fracture repair procedure")),
#         (r"(follow.?up|post.?visit)", ("99212", "Z09", "Follow-up office visit")),
#         (r"(diabetes|hba1c|blood sugar)", ("83036", "E11.9", "HbA1C Test for diabetes management")),
#         (r"(hypertension|bp|blood pressure)", ("93784", "I10", "Ambulatory blood pressure monitoring")),
#         (r"(infection|culture|bacteria)", ("87070", "A49.9", "Bacterial culture, general")),
#         (r"(check.?up|physical|annual exam|medical exam)", ("99397", "Z00.00", "Periodic general medical exam")),
#         (r"(consult|office visit)", ("99213", "Z09", "General consultation visit")),
#     ]

#     for pattern, (cpt, icd, desc) in patterns:
#         if re.search(pattern, note):
#             confidence = round(random.uniform(0.83, 0.97), 2)
#             return [{"cpt": cpt, "icd": icd, "desc": desc, "confidence": confidence}]

#     # Fallback (default)
#     return [{
#         "cpt": random.choice(["99213", "99214", "99215"]),
#         "icd": random.choice(["Z00.0", "Z09", "R53.83"]),
#         "desc": "General or follow-up office consultation",
#         "confidence": round(random.uniform(0.68, 0.78), 2),
#     }]

# # ---------------------------------------------------------------------
# # 🤖 Prior Authorization BOT (LightGBM)
# # ---------------------------------------------------------------------
# def pa_bot_simulation(inputs):
#     st.subheader("AI Prior Authorization Bot Workflow")

#     df = pd.DataFrame([inputs])
#     for c in df.select_dtypes(include=["object"]).columns:
#         df[c] = df[c].astype("category").cat.codes
#     df = align_features(df, models["pa"], "lightgbm")

#     prob = float(models["pa"].predict(df)[0])
#     st.info(f"Model probability (PA required): {prob:.2f}")

#     if prob < 0.5:
#         st.success("✅ No Prior Authorization required.")
#         return

#     st.warning("⚠️ Prior Authorization required. Initiating automation...")
#     progress = st.progress(0)
#     logbox = st.empty()
#     logs = []

#     def log(msg, step, wait=1.0):
#         logs.append(msg)
#         logbox.code("\n".join(logs))
#         progress.progress(step)
#         time.sleep(wait)

#     log("🔍 Checking payer API for prior authorizations...", 10)
#     log("📁 No record found — preparing submission packet...", 30)
#     log("🧠 Summarizing clinical justification...", 50)
#     log("📤 Submitting via payer portal...", 75)
#     log("⏳ Awaiting response...", 90)

#     status = random.choice(["Approved", "Pending", "Denied"])
#     log(f"📨 Response: {status}", 100)

#     if status == "Approved":
#         st.success("✅ Approved — claim routed to billing.")
#     elif status == "Pending":
#         st.info("⌛ Pending — bot will auto-check every 6h.")
#     else:
#         st.error("❌ Denied — appeal initiated automatically.")

# # ---------------------------------------------------------------------
# # 💳 Billing Optimization BOT (XGBoost)
# # ---------------------------------------------------------------------
# def billing_bot_simulation(inputs):
#     st.subheader("AI Billing Follow-Up Bot Workflow")

#     df = pd.DataFrame([inputs])

#     for col in ["patient_id", "claim_id"]:
#         if col in df.columns:
#             df = df.drop(columns=[col])

#     for c in df.select_dtypes(include=["object"]).columns:
#         df[c] = df[c].astype("category").cat.codes

#     df = align_features(df, models["billing"], "xgboost")
#     prob = float(models["billing"].predict_proba(df)[0][1])
#     st.metric("Payment Probability", f"{prob:.2f}")

#     progress = st.progress(0)
#     logbox = st.empty()
#     logs = []

#     def log(msg, step, wait=1.0):
#         logs.append(msg)
#         logbox.code("\n".join(logs))
#         progress.progress(step)
#         time.sleep(wait)

#     if prob < 0.4:
#         log("📞 Low payment likelihood detected...", 20)
#         log("📧 Sending personalized reminder email...", 40)
#         log("💬 Scheduling SMS follow-up...", 60)
#         log("🤖 Suggesting payment plan...", 80)
#         log("🧾 Notifying billing team...", 100)
#         st.warning("Follow-up plan generated.")
#     else:
#         log("✅ High payment likelihood detected.", 100)
#         st.success("No further action required.")

# # ---------------------------------------------------------------------
# # 🎯 Layout
# # ---------------------------------------------------------------------
# st.title("AI-Powered Revenue Cycle Management (RCM) Suite")
# st.caption("Predict • Automate • Optimize")
# st.markdown("---")

# tabs = st.tabs([
#     "Denial Prediction & Prevention",
#     "AI-Assisted Coding",
#     "Prior Authorization Automation",
#     "Billing & Collections Optimization"
# ])

# # ---------------------------------------------------------------------
# # 🧾 Tab 1: Denial Prediction (UNCHANGED)
# # ---------------------------------------------------------------------
# with tabs[0]:
#     st.header("Denial Prediction & Prevention")
#     col1, col2 = st.columns(2)
#     with col1:
#         patient_id = st.text_input("Patient ID", "P001", key="den_pid")
#         age = st.number_input("Age", 0, 120, 45, key="den_age")
#         gender = st.selectbox("Gender", ["M", "F"], key="den_gender")
#         insurance_type = st.selectbox("Insurance Type", ["PPO", "HMO", "Medicare", "Medicaid"], key="den_ins")
#         state = st.selectbox("State", ["CA", "NY", "TX", "FL", "NJ"], key="den_state")
#         chronic_condition = st.selectbox("Chronic Condition", [0, 1], key="den_chronic")
#         procedure_category = st.selectbox("Procedure Category", ["Surgery", "Radiology", "Lab", "Consult"], key="den_proc")
#     with col2:
#         claim_amount = st.number_input("Claim Amount ($)", 0.0, 50000.0, 2500.0, key="den_claim_amt")
#         previous_denials = st.number_input("Previous Denials (6m)", 0, 10, 1, key="den_prev")
#         provider_experience = st.number_input("Provider Exp (yrs)", 0, 40, 10, key="den_exp")
#         payer_coverage_ratio = st.slider("Payer Coverage Ratio", 0.0, 1.0, 0.75, key="den_pcr")
#         claim_complexity = st.slider("Claim Complexity", 0.0, 1.0, 0.5, key="den_complex")

#     if st.button("Predict Denial Likelihood", key="den_btn"):
#         try:
#             inputs = dict(
#                 patient_id=patient_id,
#                 age=age,
#                 gender=gender,
#                 insurance_type=insurance_type,
#                 state=state,
#                 chronic_condition=chronic_condition,
#                 procedure_category=procedure_category,
#                 claim_amount=claim_amount,
#                 previous_denials_6m=previous_denials,
#                 provider_experience=provider_experience,
#                 payer_coverage_ratio=payer_coverage_ratio,
#                 claim_complexity=claim_complexity,
#             )
#             prob = predict_denial(inputs)
#             st.metric("Denial Probability", f"{prob:.2f}")
#             if prob > 0.6:
#                 st.error("⚠️ High denial risk — QA review advised.")
#             else:
#                 st.success("✅ Low denial likelihood — claim can proceed.")
#         except Exception as e:
#             st.error(f"Prediction error: {e}")

# # ---------------------------------------------------------------------
# # 🧠 Tab 2: AI-Assisted Coding (ONLY color + mapping updated)
# # ---------------------------------------------------------------------
# with tabs[1]:
#     st.header("🩺 AI-Assisted Coding from Clinical Notes")
#     st.caption("Let AI extract CPT and ICD-10 codes intelligently from unstructured clinical notes.")

#     note = st.text_area(
#         "📝 Paste or type doctor's note below",
#         height=180,
#         placeholder="e.g., Patient presents with chest pain and hypertension for 2 weeks...",
#         key="coding_note",
#     )

#     col1, col2 = st.columns([1, 1])
#     with col1:
#         if st.button("✨ Generate CPT/ICD-10 Codes", key="coding_generate"):
#             with st.spinner("Analyzing clinical text..."):
#                 progress = st.progress(0)
#                 for i in range(0, 100, 20):
#                     time.sleep(0.15)
#                     progress.progress(i + 10)
#                 codes = smart_predict_coding(note)
#                 progress.progress(100)
#                 st.success(f"✅ {len(codes)} codes generated.")
#                 st.markdown("---")
#                 for entry in codes:
#                     st.markdown(
#                         f"""
#                         <div style="background: linear-gradient(135deg, #bbdefb 0%, #e3f2fd 100%); padding:16px; border-radius:14px; margin-bottom:14px; box-shadow: 0 3px 6px rgba(0,0,0,0.15); border-left:6px solid #1565c0;">
#                         <h4 style="color:#0d47a1;margin-bottom:4px;">💉 <b>CPT {entry['cpt']}</b>  |  🧠 <b>ICD-10 {entry['icd']}</b></h4>
#                         <p style="color:#1a237e;font-size:14px;margin-bottom:6px;">{entry['desc']}</p>
#                         <p style="color:#0d47a1;font-size:13px;">Confidence: <b>{int(entry['confidence']*100)}%</b></p>
#                         </div>
#                         """,
#                         unsafe_allow_html=True,
#                     )
#     with col2:
#         if st.button("🔁 Regenerate Suggestions", key="coding_refresh"):
#             st.experimental_rerun()

# # ---------------------------------------------------------------------
# # 🤖 Tab 3: PA Automation (UNCHANGED)
# # ---------------------------------------------------------------------
# with tabs[2]:
#     st.header("Prior Authorization Automation")
#     col1, col2 = st.columns(2)
#     with col1:
#         claim_id = st.text_input("Claim ID", "C123", key="pa_claimid")
#         age = st.number_input("Age", 0, 120, 50, key="pa_age")
#         gender = st.selectbox("Gender", ["M", "F"], key="pa_gender")
#         specialty = st.selectbox("Specialty", ["Cardiology", "Ortho", "Oncology", "Radiology"], key="pa_spec")
#         insurance_type = st.selectbox("Insurance", ["Commercial", "Medicare", "Medicaid"], key="pa_ins")
#         claim_amount = st.number_input("Claim Amount ($)", 0.0, 100000.0, 4000.0, key="pa_amount")
#     with col2:
#         category = st.selectbox("Claim Category", ["Regular", "High Value", "Surgery", "Imaging"], key="pa_category")
#         plan_type = st.selectbox("Plan Type", ["HMO", "PPO"], key="pa_plan")
#         region = st.selectbox("Region", ["East", "West", "North", "South"], key="pa_region")
#         risk_score = st.slider("Risk Score", 0.0, 1.0, 0.4, key="pa_risk")
#         urgent = st.selectbox("Urgent?", [0, 1], key="pa_urgent")

#     if st.button("Run PA Bot", key="pa_btn"):
#         try:
#             inputs = dict(
#                 claim_id=claim_id,
#                 age=age,
#                 gender=gender,
#                 medical_specialty=specialty,
#                 insurance_type=insurance_type,
#                 plan_type=plan_type,
#                 hospital_region=region,
#                 claim_amount=claim_amount,
#                 claim_category=category,
#                 risk_score=risk_score,
#                 urgent_case=urgent,
#             )
#             pa_bot_simulation(inputs)
#         except Exception as e:
#             st.error(f"Prediction error: {e}")

# # ---------------------------------------------------------------------
# # 💳 Tab 4: Billing (UNCHANGED)
# # ---------------------------------------------------------------------
# with tabs[3]:
#     st.header("Billing & Collections Optimization")
#     col1, col2 = st.columns(2)
#     with col1:
#         patient_id = st.text_input("Patient ID", "P555", key="bill_pid")
#         age = st.number_input("Age", 0, 120, 40, key="bill_age")
#         gender = st.selectbox("Gender", ["M", "F"], key="bill_gender")
#         insurance_type = st.selectbox("Insurance Type", ["PPO", "HMO", "Medicare", "Medicaid"], key="bill_ins")
#         balance_due = st.number_input("Balance Due ($)", 0.0, 10000.0, 1200.0, key="bill_balance")
#         num_reminders = st.number_input("Reminders Sent", 0, 10, 1, key="bill_rem")
#     with col2:
#         credit_score = st.slider("Credit Score", 300, 850, 680, key="bill_credit")
#         patient_eng = st.slider("Engagement Score", 0.0, 1.0, 0.6, key="bill_eng")
#         days_in_ar = st.number_input("Days in AR", 0, 120, 30, key="bill_days")
#         visit_type = st.selectbox("Visit Type", ["Inpatient", "Outpatient", "ER"], key="bill_visit")
#         has_plan = st.selectbox("Payment Plan Exists", [0, 1], key="bill_plan")

#     if st.button("Run Billing Bot", key="bill_btn"):
#         try:
#             inputs = dict(
#                 patient_id=patient_id,
#                 age=age,
#                 gender=gender,
#                 insurance_type=insurance_type,
#                 balance_due=balance_due,
#                 num_reminders_sent=num_reminders,
#                 credit_score=credit_score,
#                 patient_engagement_score=patient_eng,
#                 days_in_ar=days_in_ar,
#                 visit_type=visit_type,
#                 has_payment_plan=has_plan,
#             )
#             billing_bot_simulation(inputs)
#         except Exception as e:
#             st.error(f"Prediction error: {e}")

# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib, os, time, random, re
# from catboost import CatBoostClassifier, Pool
# from xgboost import XGBClassifier
# import lightgbm as lgb

# st.set_page_config(page_title="AI-Powered RCM Suite", layout="wide")

# # ---------------------------------------------------------------------
# # 🧠 Load Models
# # ---------------------------------------------------------------------
# @st.cache_resource
# def load_models():
#     models = {}
#     if os.path.exists("models/denial_model.cbm"):
#         m = CatBoostClassifier()
#         m.load_model("models/denial_model.cbm")
#         models["denial"] = m
#     if os.path.exists("models/coding_model.pkl"):
#         models["coding"] = joblib.load("models/coding_model.pkl")
#     if os.path.exists("models/pa_model.txt"):
#         models["pa"] = lgb.Booster(model_file="models/pa_model.txt")
#     if os.path.exists("models/billing_model.json"):
#         xgb = XGBClassifier()
#         xgb.load_model("models/billing_model.json")
#         models["billing"] = xgb
#     return models

# models = load_models()

# # ---------------------------------------------------------------------
# # 🧩 Align Features
# # ---------------------------------------------------------------------
# def align_features(df, model, model_type):
#     if model_type == "catboost":
#         feat_names = model.feature_names_
#     elif model_type == "xgboost":
#         feat_names = model.get_booster().feature_names
#     elif model_type == "lightgbm":
#         feat_names = model.feature_name()
#     else:
#         feat_names = df.columns.tolist()

#     for f in feat_names:
#         if f not in df.columns:
#             df[f] = 0
#     df = df[feat_names]
#     return df

# # ---------------------------------------------------------------------
# # 🧾 Denial Prediction (CatBoost)
# # ---------------------------------------------------------------------
# def predict_denial(inputs):
#     df = pd.DataFrame([inputs])

#     for col in ["patient_id", "claim_id"]:
#         if col in df.columns:
#             df = df.drop(columns=[col])

#     cat_features = []
#     for col in df.columns:
#         if df[col].dtype == "object" or df[col].dtype.name == "category":
#             df[col] = df[col].astype("category")
#             cat_features.append(col)

#     df = align_features(df, models["denial"], "catboost")

#     for feat in models["denial"].get_cat_feature_indices():
#         name = models["denial"].feature_names_[feat]
#         if name in df.columns:
#             df[name] = df[name].astype("category")

#     pool = Pool(df, cat_features=list(df.select_dtypes(include=["category"]).columns))
#     return float(models["denial"].predict_proba(pool)[0][1])

# # ---------------------------------------------------------------------
# # 🧠 AI-Assisted Coding (semantic enhancement only)
# # ---------------------------------------------------------------------
# def smart_predict_coding(note):
#     note = note.lower().strip()

#     # Use regex & semantic matching for better mapping
#     patterns = [
#         (r"(mri|magnetic).*brain.*(contrast)?", ("70553", "C71.9", "MRI Brain with/without contrast")),
#         (r"(ct|scan).*abdomen.*pelvis", ("74177", "R10.9", "CT abdomen and pelvis with contrast")),
#         (r"brain.*(tumor|mass)", ("70553", "C71.9", "MRI Brain for tumor evaluation")),
#         (r"(stroke|clot|embol)", ("61624", "I63.9", "Endovascular therapy for cerebral clot")),
#         (r"chest pain|cardiac|angina", ("93000", "R07.9", "Electrocardiogram for chest pain")),
#         (r"(fracture|broken bone)", ("27786", "S82.90XA", "Fracture repair procedure")),
#         (r"(follow.?up|post.?visit)", ("99212", "Z09", "Follow-up office visit")),
#         (r"(diabetes|hba1c|blood sugar)", ("83036", "E11.9", "HbA1C Test for diabetes management")),
#         (r"(hypertension|bp|blood pressure)", ("93784", "I10", "Ambulatory blood pressure monitoring")),
#         (r"(infection|culture|bacteria)", ("87070", "A49.9", "Bacterial culture, general")),
#         (r"(check.?up|physical|annual exam|medical exam)", ("99397", "Z00.00", "Periodic general medical exam")),
#         (r"(consult|office visit)", ("99213", "Z09", "General consultation visit")),
#     ]

#     for pattern, (cpt, icd, desc) in patterns:
#         if re.search(pattern, note):
#             confidence = round(random.uniform(0.83, 0.97), 2)
#             return [{"cpt": cpt, "icd": icd, "desc": desc, "confidence": confidence}]

#     # Fallback (default)
#     return [{
#         "cpt": random.choice(["99213", "99214", "99215"]),
#         "icd": random.choice(["Z00.0", "Z09", "R53.83"]),
#         "desc": "General or follow-up office consultation",
#         "confidence": round(random.uniform(0.68, 0.78), 2),
#     }]

# # ---------------------------------------------------------------------
# # 🤖 Prior Authorization BOT (LightGBM)
# # ---------------------------------------------------------------------
# def pa_bot_simulation(inputs):
#     st.subheader("AI Prior Authorization Bot Workflow")

#     df = pd.DataFrame([inputs])
#     for c in df.select_dtypes(include=["object"]).columns:
#         df[c] = df[c].astype("category").cat.codes
#     df = align_features(df, models["pa"], "lightgbm")

#     prob = float(models["pa"].predict(df)[0])
#     st.info(f"Model probability (PA required): {prob:.2f}")

#     if prob < 0.5:
#         st.success("✅ No Prior Authorization required.")
#         return

#     st.warning("⚠️ Prior Authorization required. Initiating automation...")
#     progress = st.progress(0)
#     logbox = st.empty()
#     logs = []

#     def log(msg, step, wait=1.0):
#         logs.append(msg)
#         logbox.code("\n".join(logs))
#         progress.progress(step)
#         time.sleep(wait)

#     log("🔍 Checking payer API for prior authorizations...", 10)
#     log("📁 No record found — preparing submission packet...", 30)
#     log("🧠 Summarizing clinical justification...", 50)
#     log("📤 Submitting via payer portal...", 75)
#     log("⏳ Awaiting response...", 90)

#     status = random.choice(["Approved", "Pending", "Denied"])
#     log(f"📨 Response: {status}", 100)

#     if status == "Approved":
#         st.success("✅ Approved — claim routed to billing.")
#     elif status == "Pending":
#         st.info("⌛ Pending — bot will auto-check every 6h.")
#     else:
#         st.error("❌ Denied — appeal initiated automatically.")

# # ---------------------------------------------------------------------
# # 💳 Billing Optimization BOT (XGBoost)
# # ---------------------------------------------------------------------
# def billing_bot_simulation(inputs):
#     st.subheader("AI Billing Follow-Up Bot Workflow")

#     df = pd.DataFrame([inputs])

#     for col in ["patient_id", "claim_id"]:
#         if col in df.columns:
#             df = df.drop(columns=[col])

#     for c in df.select_dtypes(include=["object"]).columns:
#         df[c] = df[c].astype("category").cat.codes

#     df = align_features(df, models["billing"], "xgboost")
#     prob = float(models["billing"].predict_proba(df)[0][1])
#     st.metric("Payment Probability", f"{prob:.2f}")

#     progress = st.progress(0)
#     logbox = st.empty()
#     logs = []

#     def log(msg, step, wait=1.0):
#         logs.append(msg)
#         logbox.code("\n".join(logs))
#         progress.progress(step)
#         time.sleep(wait)

#     if prob < 0.4:
#         log("📞 Low payment likelihood detected...", 20)
#         log("📧 Sending personalized reminder email...", 40)
#         log("💬 Scheduling SMS follow-up...", 60)
#         log("🤖 Suggesting payment plan...", 80)
#         log("🧾 Notifying billing team...", 100)
#         st.warning("Follow-up plan generated.")
#     else:
#         log("✅ High payment likelihood detected.", 100)
#         st.success("No further action required.")

# # ---------------------------------------------------------------------
# # 🎯 Layout
# # ---------------------------------------------------------------------
# st.title("AI-Powered Revenue Cycle Management (RCM) Suite")
# st.caption("Predict • Automate • Optimize")
# st.markdown("---")

# tabs = st.tabs([
#     "Denial Prediction & Prevention",
#     "AI-Assisted Coding",
#     "Prior Authorization Automation",
#     "Billing & Collections Optimization"
# ])

# # ---------------------------------------------------------------------
# # 🧾 Tab 1: Denial Prediction (UNCHANGED)
# # ---------------------------------------------------------------------
# with tabs[0]:
#     st.header("Denial Prediction & Prevention")
#     col1, col2 = st.columns(2)
#     with col1:
#         patient_id = st.text_input("Patient ID", "P001", key="den_pid")
#         age = st.number_input("Age", 0, 120, 45, key="den_age")
#         gender = st.selectbox("Gender", ["M", "F"], key="den_gender")
#         insurance_type = st.selectbox("Insurance Type", ["PPO", "HMO", "Medicare", "Medicaid"], key="den_ins")
#         state = st.selectbox("State", ["CA", "NY", "TX", "FL", "NJ"], key="den_state")
#         chronic_condition = st.selectbox("Chronic Condition", [0, 1], key="den_chronic")
#         procedure_category = st.selectbox("Procedure Category", ["Surgery", "Radiology", "Lab", "Consult"], key="den_proc")
#     with col2:
#         claim_amount = st.number_input("Claim Amount ($)", 0.0, 50000.0, 2500.0, key="den_claim_amt")
#         previous_denials = st.number_input("Previous Denials (6m)", 0, 10, 1, key="den_prev")
#         provider_experience = st.number_input("Provider Exp (yrs)", 0, 40, 10, key="den_exp")
#         payer_coverage_ratio = st.slider("Payer Coverage Ratio", 0.0, 1.0, 0.75, key="den_pcr")
#         claim_complexity = st.slider("Claim Complexity", 0.0, 1.0, 0.5, key="den_complex")

#     if st.button("Predict Denial Likelihood", key="den_btn"):
#         try:
#             inputs = dict(
#                 patient_id=patient_id,
#                 age=age,
#                 gender=gender,
#                 insurance_type=insurance_type,
#                 state=state,
#                 chronic_condition=chronic_condition,
#                 procedure_category=procedure_category,
#                 claim_amount=claim_amount,
#                 previous_denials_6m=previous_denials,
#                 provider_experience=provider_experience,
#                 payer_coverage_ratio=payer_coverage_ratio,
#                 claim_complexity=claim_complexity,
#             )
#             prob = predict_denial(inputs)
#             st.metric("Denial Probability", f"{prob:.2f}")
#             if prob > 0.6:
#                 st.error("⚠️ High denial risk — QA review advised.")
#             else:
#                 st.success("✅ Low denial likelihood — claim can proceed.")
#         except Exception as e:
#             st.error(f"Prediction error: {e}")

# # ---------------------------------------------------------------------
# # 🧠 Tab 2: AI-Assisted Coding (ONLY color + mapping updated)
# # ---------------------------------------------------------------------
# with tabs[1]:
#     st.header("🩺 AI-Assisted Coding from Clinical Notes")
#     st.caption("Let AI extract CPT and ICD-10 codes intelligently from unstructured clinical notes.")

#     note = st.text_area(
#         "📝 Paste or type doctor's note below",
#         height=180,
#         placeholder="e.g., Patient presents with chest pain and hypertension for 2 weeks...",
#         key="coding_note",
#     )

#     col1, col2 = st.columns([1, 1])
#     with col1:
#         if st.button("✨ Generate CPT/ICD-10 Codes", key="coding_generate"):
#             with st.spinner("Analyzing clinical text..."):
#                 progress = st.progress(0)
#                 for i in range(0, 100, 20):
#                     time.sleep(0.15)
#                     progress.progress(i + 10)
#                 codes = smart_predict_coding(note)
#                 progress.progress(100)
#                 st.success(f"✅ {len(codes)} codes generated.")
#                 st.markdown("---")
#                 for entry in codes:
#                     st.markdown(
#                         f"""
#                         <div style="background: linear-gradient(135deg, #bbdefb 0%, #e3f2fd 100%); padding:16px; border-radius:14px; margin-bottom:14px; box-shadow: 0 3px 6px rgba(0,0,0,0.15); border-left:6px solid #1565c0;">
#                         <h4 style="color:#0d47a1;margin-bottom:4px;">💉 <b>CPT {entry['cpt']}</b>  |  🧠 <b>ICD-10 {entry['icd']}</b></h4>
#                         <p style="color:#1a237e;font-size:14px;margin-bottom:6px;">{entry['desc']}</p>
#                         <p style="color:#0d47a1;font-size:13px;">Confidence: <b>{int(entry['confidence']*100)}%</b></p>
#                         </div>
#                         """,
#                         unsafe_allow_html=True,
#                     )
#     with col2:
#         if st.button("🔁 Regenerate Suggestions", key="coding_refresh"):
#             st.experimental_rerun()

# # ---------------------------------------------------------------------
# # 🤖 Tab 3: PA Automation (UNCHANGED)
# # ---------------------------------------------------------------------
# with tabs[2]:
#     st.header("Prior Authorization Automation")
#     col1, col2 = st.columns(2)
#     with col1:
#         claim_id = st.text_input("Claim ID", "C123", key="pa_claimid")
#         age = st.number_input("Age", 0, 120, 50, key="pa_age")
#         gender = st.selectbox("Gender", ["M", "F"], key="pa_gender")
#         specialty = st.selectbox("Specialty", ["Cardiology", "Ortho", "Oncology", "Radiology"], key="pa_spec")
#         insurance_type = st.selectbox("Insurance", ["Commercial", "Medicare", "Medicaid"], key="pa_ins")
#         claim_amount = st.number_input("Claim Amount ($)", 0.0, 100000.0, 4000.0, key="pa_amount")
#     with col2:
#         category = st.selectbox("Claim Category", ["Regular", "High Value", "Surgery", "Imaging"], key="pa_category")
#         plan_type = st.selectbox("Plan Type", ["HMO", "PPO"], key="pa_plan")
#         region = st.selectbox("Region", ["East", "West", "North", "South"], key="pa_region")
#         risk_score = st.slider("Risk Score", 0.0, 1.0, 0.4, key="pa_risk")
#         urgent = st.selectbox("Urgent?", [0, 1], key="pa_urgent")

#     if st.button("Run PA Bot", key="pa_btn"):
#         try:
#             inputs = dict(
#                 claim_id=claim_id,
#                 age=age,
#                 gender=gender,
#                 medical_specialty=specialty,
#                 insurance_type=insurance_type,
#                 plan_type=plan_type,
#                 hospital_region=region,
#                 claim_amount=claim_amount,
#                 claim_category=category,
#                 risk_score=risk_score,
#                 urgent_case=urgent,
#             )
#             pa_bot_simulation(inputs)
#         except Exception as e:
#             st.error(f"Prediction error: {e}")

# # ---------------------------------------------------------------------
# # 💳 Tab 4: Billing (UNCHANGED)
# # ---------------------------------------------------------------------
# with tabs[3]:
#     st.header("Billing & Collections Optimization")
#     col1, col2 = st.columns(2)
#     with col1:
#         patient_id = st.text_input("Patient ID", "P555", key="bill_pid")
#         age = st.number_input("Age", 0, 120, 40, key="bill_age")
#         gender = st.selectbox("Gender", ["M", "F"], key="bill_gender")
#         insurance_type = st.selectbox("Insurance Type", ["PPO", "HMO", "Medicare", "Medicaid"], key="bill_ins")
#         balance_due = st.number_input("Balance Due ($)", 0.0, 10000.0, 1200.0, key="bill_balance")
#         num_reminders = st.number_input("Reminders Sent", 0, 10, 1, key="bill_rem")
#     with col2:
#         credit_score = st.slider("Credit Score", 300, 850, 680, key="bill_credit")
#         patient_eng = st.slider("Engagement Score", 0.0, 1.0, 0.6, key="bill_eng")
#         days_in_ar = st.number_input("Days in AR", 0, 120, 30, key="bill_days")
#         visit_type = st.selectbox("Visit Type", ["Inpatient", "Outpatient", "ER"], key="bill_visit")
#         has_plan = st.selectbox("Payment Plan Exists", [0, 1], key="bill_plan")

#     if st.button("Run Billing Bot", key="bill_btn"):
#         try:
#             inputs = dict(
#                 patient_id=patient_id,
#                 age=age,
#                 gender=gender,
#                 insurance_type=insurance_type,
#                 balance_due=balance_due,
#                 num_reminders_sent=num_reminders,
#                 credit_score=credit_score,
#                 patient_engagement_score=patient_eng,
#                 days_in_ar=days_in_ar,
#                 visit_type=visit_type,
#                 has_payment_plan=has_plan,
#             )
#             billing_bot_simulation(inputs)
#         except Exception as e:
#             st.error(f"Prediction error: {e}")


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

# from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import joblib

st.set_page_config(page_title="AI-Powered RCM Suite", layout="wide")

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

    st.write("Loaded models:", list(models.keys()))
    return models

models = load_models()

# ---------------------------------------------------------------------
# Align Features
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
    df = df[feat_names]
    return df

# ---------------------------------------------------------------------
# Denial Prediction (CatBoost)
# ---------------------------------------------------------------------
def predict_denial(inputs):
    if "denial" not in models:
        st.error("Denial model not loaded. Please check your models folder.")
        return 0.0

    df = pd.DataFrame([inputs])
    for col in ["patient_id", "claim_id"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    cat_features = []
    for col in df.columns:
        if df[col].dtype == "object" or df[col].dtype.name == "category":
            df[col] = df[col].astype("category")
            cat_features.append(col)

    df = align_features(df, models["denial"], "catboost")

    for feat in models["denial"].get_cat_feature_indices():
        name = models["denial"].feature_names_[feat]
        if name in df.columns:
            df[name] = df[name].astype("category")

    pool = Pool(df, cat_features=list(df.select_dtypes(include=["category"]).columns))
    return float(models["denial"].predict_proba(pool)[0][1])

# ---------------------------------------------------------------------
# AI-Assisted Coding (semantic enhancement)
# ---------------------------------------------------------------------
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

    if matches:
        return matches

    return [{
        "cpt": random.choice(["99213", "99214", "99215"]),
        "icd": random.choice(["Z00.0", "Z09", "R53.83"]),
        "desc": "General or follow-up office consultation",
        "confidence": round(random.uniform(0.68, 0.78), 2),
    }]

# ---------------------------------------------------------------------
# Prior Authorization BOT (LightGBM)
# ---------------------------------------------------------------------
def pa_bot_simulation(inputs):
    st.subheader("AI Prior Authorization Bot Workflow")

    if "pa" not in models:
        st.error("PA model not loaded. Please check your models folder.")
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

    st.warning("Prior Authorization required. Initiating automation...")
    progress = st.progress(0)
    logbox = st.empty()
    logs = []

    def log(msg, step, wait=1.0):
        logs.append(msg)
        logbox.code("\n".join(logs))
        progress.progress(step)
        time.sleep(wait)

    log("Checking payer API for prior authorizations...", 10)
    log("No record found — preparing submission packet...", 30)
    log("Summarizing clinical justification...", 50)
    log("Submitting via payer portal...", 75)
    log("Awaiting response...", 90)

    status = random.choice(["Approved", "Pending", "Denied"])
    log(f"Response: {status}", 100)

    if status == "Approved":
        st.success("Approved — claim routed to billing.")
    elif status == "Pending":
        st.info("Pending — bot will auto-check every 6h.")
    else:
        st.error("Denied — appeal initiated automatically.")

# ---------------------------------------------------------------------
# Billing Optimization BOT (XGBoost)
# ---------------------------------------------------------------------
def billing_bot_simulation(inputs):
    st.subheader("AI Billing Follow-Up Bot Workflow")

    if "billing" not in models:
        st.error("Billing model not loaded. Please check your models folder.")
        return

    df = pd.DataFrame([inputs])
    for col in ["patient_id", "claim_id"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype("category").cat.codes

    df = align_features(df, models["billing"], "xgboost")
    prob = float(models["billing"].predict_proba(df)[0][1])
    st.metric("Payment Probability", f"{prob:.2f}")

    progress = st.progress(0)
    logbox = st.empty()
    logs = []

    def log(msg, step, wait=1.0):
        logs.append(msg)
        logbox.code("\n".join(logs))
        progress.progress(step)
        time.sleep(wait)

    if prob < 0.4:
        log("Low payment likelihood detected...", 20)
        log("Sending personalized reminder email...", 40)
        log("Scheduling SMS follow-up...", 60)
        log("Suggesting payment plan...", 80)
        log("Notifying billing team...", 100)
        st.warning("Follow-up plan generated.")
    else:
        log("High payment likelihood detected.", 100)
        st.success("No further action required.")

# ---------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------
st.title("AI-Powered Revenue Cycle Management (RCM) Suite")
st.caption("Predict • Automate • Optimize")
st.markdown("---")

tabs = st.tabs([
    "Denial Prediction & Prevention",
    "AI-Assisted Coding",
    "Prior Authorization Automation",
    "Billing & Collections Optimization",
    "Doctor Note ICD/CPT Prediction"
])

# ---------------------------------------------------------------------
# Tab 1: Denial Prediction
# ---------------------------------------------------------------------
with tabs[0]:
    st.header("Denial Prediction & Prevention")
    col1, col2 = st.columns(2)
    with col1:
        patient_id = st.text_input("Patient ID", "P001", key="den_pid")
        age = st.number_input("Age", 0, 120, 45, key="den_age")
        gender = st.selectbox("Gender", ["M", "F"], key="den_gender")
        insurance_type = st.selectbox("Insurance Type", ["PPO", "HMO", "Medicare", "Medicaid"], key="den_ins")
        state = st.selectbox("State", ["CA", "NY", "TX", "FL", "NJ"], key="den_state")
        chronic_condition = st.selectbox("Chronic Condition", [0, 1], key="den_chronic")
        procedure_category = st.selectbox("Procedure Category", ["Surgery", "Radiology", "Lab", "Consult"], key="den_proc")
    with col2:
        claim_amount = st.number_input("Claim Amount ($)", 0.0, 50000.0, 2500.0, key="den_claim_amt")
        previous_denials = st.number_input("Previous Denials (6m)", 0, 10, 1, key="den_prev")
        provider_experience = st.number_input("Provider Exp (yrs)", 0, 40, 10, key="den_exp")
        payer_coverage_ratio = st.slider("Payer Coverage Ratio", 0.0, 1.0, 0.75, key="den_pcr")
        claim_complexity = st.slider("Claim Complexity", 0.0, 1.0, 0.5, key="den_complex")

    if st.button("Predict Denial Likelihood", key="den_btn"):
        try:
            inputs = dict(
                patient_id=patient_id,
                age=age,
                gender=gender,
                insurance_type=insurance_type,
                state=state,
                chronic_condition=chronic_condition,
                procedure_category=procedure_category,
                claim_amount=claim_amount,
                previous_denials_6m=previous_denials,
                provider_experience=provider_experience,
                payer_coverage_ratio=payer_coverage_ratio,
                claim_complexity=claim_complexity,
            )
            prob = predict_denial(inputs)
            st.metric("Denial Probability", f"{prob:.2f}")
            if prob > 0.6:
                st.error("High denial risk — QA review advised.")
            else:
                st.success("Low denial likelihood — claim can proceed.")
        except Exception as e:
            st.error(f"Prediction error: {e}")

# ---------------------------------------------------------------------
# Tab 2: AI-Assisted Coding
# ---------------------------------------------------------------------
with tabs[1]:
    st.header("AI-Assisted Coding from Clinical Notes")
    st.caption("Extract CPT and ICD-10 codes intelligently from clinical notes.")
    note = st.text_area(
        "Paste or type doctor's note below",
        height=180,
        placeholder="e.g., Patient presents with chest pain and hypertension for 2 weeks...",
        key="coding_note",
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Generate CPT/ICD-10 Codes", key="coding_generate"):
            with st.spinner("Analyzing clinical text..."):
                progress = st.progress(0)
                for i in range(0, 100, 20):
                    time.sleep(0.15)
                    progress.progress(i + 10)
                codes = smart_predict_coding(note)
                progress.progress(100)
                st.success(f"{len(codes)} codes generated.")
                st.markdown("---")
                for entry in codes:
                    st.markdown(
                        f"""
                        <div style="background: linear-gradient(135deg, #dfe9f3 0%, #ffffff 100%); padding:16px; border-radius:14px; margin-bottom:14px; box-shadow: 0 3px 6px rgba(0,0,0,0.15); border-left:6px solid #00509e;">
                        <h4 style="color:#003f7d;margin-bottom:4px;">CPT {entry['cpt']}  |  ICD-10 {entry['icd']}</h4>
                        <p style="color:#1a1a1a;font-size:14px;margin-bottom:6px;">{entry['desc']}</p>
                        <p style="color:#0d47a1;font-size:13px;">Confidence: <b>{int(entry['confidence']*100)}%</b></p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
    with col2:
        if st.button("Regenerate Suggestions", key="coding_refresh"):
            st.experimental_rerun()

# ---------------------------------------------------------------------
# Tab 3: PA Automation
# ---------------------------------------------------------------------
with tabs[2]:
    st.header("Prior Authorization Automation")
    col1, col2 = st.columns(2)
    with col1:
        claim_id = st.text_input("Claim ID", "C123", key="pa_claimid")
        age = st.number_input("Age", 0, 120, 50, key="pa_age")
        gender = st.selectbox("Gender", ["M", "F"], key="pa_gender")
        specialty = st.selectbox("Specialty", ["Cardiology", "Ortho", "Oncology", "Radiology"], key="pa_spec")
        insurance_type = st.selectbox("Insurance", ["Commercial", "Medicare", "Medicaid"], key="pa_ins")
        claim_amount = st.number_input("Claim Amount ($)", 0.0, 100000.0, 4000.0, key="pa_amount")
    with col2:
        category = st.selectbox("Claim Category", ["Regular", "High Value", "Surgery", "Imaging"], key="pa_category")
        plan_type = st.selectbox("Plan Type", ["HMO", "PPO"], key="pa_plan")
        region = st.selectbox("Region", ["East", "West", "North", "South"], key="pa_region")
        risk_score = st.slider("Risk Score", 0.0, 1.0, 0.4, key="pa_risk")
        urgent = st.selectbox("Urgent?", [0, 1], key="pa_urgent")

    if st.button("Run PA Bot", key="pa_btn"):
        try:
            inputs = dict(
                claim_id=claim_id,
                age=age,
                gender=gender,
                medical_specialty=specialty,
                insurance_type=insurance_type,
                plan_type=plan_type,
                hospital_region=region,
                claim_amount=claim_amount,
                claim_category=category,
                risk_score=risk_score,
                urgent_case=urgent,
            )
            pa_bot_simulation(inputs)
        except Exception as e:
            st.error(f"Prediction error: {e}")

# ---------------------------------------------------------------------
# Tab 4: Billing
# ---------------------------------------------------------------------
with tabs[3]:
    st.header("Billing & Collections Optimization")
    col1, col2 = st.columns(2)
    with col1:
        patient_id = st.text_input("Patient ID", "P555", key="bill_pid")
        age = st.number_input("Age", 0, 120, 40, key="bill_age")
        gender = st.selectbox("Gender", ["M", "F"], key="bill_gender")
        insurance_type = st.selectbox("Insurance Type", ["PPO", "HMO", "Medicare", "Medicaid"], key="bill_ins")
        balance_due = st.number_input("Balance Due ($)", 0.0, 10000.0, 1200.0, key="bill_balance")
        num_reminders = st.number_input("Reminders Sent", 0, 10, 1, key="bill_rem")
    with col2:
        credit_score = st.slider("Credit Score", 300, 850, 680, key="bill_credit")
        patient_eng = st.slider("Engagement Score", 0.0, 1.0, 0.6, key="bill_eng")
        days_in_ar = st.number_input("Days in AR", 0, 120, 30, key="bill_days")
        visit_type = st.selectbox("Visit Type", ["Inpatient", "Outpatient", "ER"], key="bill_visit")
        has_plan = st.selectbox("Payment Plan Exists", [0, 1], key="bill_plan")

    if st.button("Run Billing Bot", key="bill_btn"):
        try:
            inputs = dict(
                patient_id=patient_id,
                age=age,
                gender=gender,
                insurance_type=insurance_type,
                balance_due=balance_due,
                num_reminders_sent=num_reminders,
                credit_score=credit_score,
                patient_engagement_score=patient_eng,
                days_in_ar=days_in_ar,
                visit_type=visit_type,
                has_payment_plan=has_plan,
            )
            billing_bot_simulation(inputs)
        except Exception as e:
            st.error(f"Prediction error: {e}")

# # ---------------------------------------------------------------------
# # Tab 5: Doctor Note ICD/CPT Prediction (Image Upload)
# # ---------------------------------------------------------------------
# import os, sys, numpy as np, torch, joblib
# from PIL import Image
# import pytesseract
# from torch.serialization import add_safe_globals
# from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification


# # ✅ Prevent deadlocks & reduce startup lag
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# sys.setrecursionlimit(2000)

# add_safe_globals([np._core.multiarray.scalar])

# @st.cache_resource(show_spinner=False)
# def load_icd_cpt_model_safe():
#     """Safely load the ICD/CPT DistilBERT model & label encoder."""
#     from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

#     model_dir = "models/icd_cpt_distilbert_v3"
#     model_name = "bert-base-uncased"
#     num_labels = 27

#     config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForSequenceClassification.from_config(config)

#     ckpt_candidates = [
#         os.path.join(model_dir, "best_model.pt"),
#         os.path.join(model_dir, "pytorch_model.bin"),
#         os.path.join(model_dir, "model.safetensors"),
#     ]

#     model_loaded = False
#     for ckpt in ckpt_candidates:
#         if os.path.exists(ckpt):
#             try:
#                 state = torch.load(ckpt, map_location="cpu", weights_only=False)
#                 if isinstance(state, dict):
#                     if "model" in state:
#                         state = state["model"]
#                     elif "state_dict" in state:
#                         state = state["state_dict"]
#                 missing, unexpected = model.load_state_dict(state, strict=False)
#                 if missing:
#                     st.warning(f"⚠️ Missing keys: {len(missing)}")
#                 if unexpected:
#                     st.info(f"ℹ️ Ignored extra keys: {unexpected[:3]} ... ({len(unexpected)} total)")
#                 st.success(f"✅ Loaded model from {os.path.basename(ckpt)}")
#                 model_loaded = True
#                 break
#             except Exception as e:
#                 st.error(f"❌ Could not load {ckpt}: {e}")
#     if not model_loaded:
#         st.warning("⚠️ No model weights found — using random predictions.")

#     # ✅ Load label binarizer robustly (.pkl or .pt)
#     lb = None
#     for ext in ["pkl", "pt"]:
#         path = os.path.join(model_dir, f"label_binarizer.{ext}")
#         if os.path.exists(path):
#             try:
#                 lb = joblib.load(path) if ext == "pkl" else torch.load(path, map_location="cpu")
#                 st.success(f"✅ Loaded label binarizer ({ext})")
#                 break
#             except Exception as e:
#                 st.warning(f"⚠️ Could not load label_binarizer.{ext}: {e}")
#     if lb is None:
#         st.warning("⚠️ No label binarizer found — fallback to generic Code_X labels.")

#     # ✅ Device setup
#     device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#     model.to(device)
#     model.eval()
#     return tokenizer, model, lb, device


# # 🔹 Load model once (cached)
# tokenizer, icd_cpt_model, label_binarizer, device = load_icd_cpt_model_safe()

# # ---------------------------------------------------------------------
# # Streamlit UI for the ICD/CPT Predictor
# # ---------------------------------------------------------------------
# with tabs[4]:
#     st.header("🧠 Doctor Note ICD/CPT Prediction (Image Upload)")
#     st.caption(
#         "Upload a handwritten note image (.png/.jpg). The app will extract text and predict probable ICD/CPT codes."
#     )

#     uploaded_file = st.file_uploader(
#         "Upload Doctor's Handwritten Note",
#         type=["png", "jpg", "jpeg"],
#         key="icd_upload5"
#     )

#     if uploaded_file:
#         image = Image.open(uploaded_file)
#         st.image(image, caption="Uploaded Note", width="stretch")

#         if st.button("Extract & Predict Codes", key="predict_icd_cpt5"):
#             with st.spinner("Extracting text and predicting..."):
#                 # 🔹 OCR extraction
#                 extracted_text = pytesseract.image_to_string(image)
#                 cleaned_text = extracted_text.strip().replace("\n", " ")

#                 st.subheader("📋 Extracted Text")
#                 st.write(cleaned_text if cleaned_text else "_(No readable text found)_")

#                 if not cleaned_text:
#                     st.stop()

#                 # 🔹 Tokenize & Predict
#                 inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
#                 inputs = {k: v.to(device) for k, v in inputs.items()}

#                 with torch.no_grad():
#                     probs = torch.sigmoid(icd_cpt_model(**inputs).logits).cpu().numpy()[0]

#                 # 🔹 Top 5 predictions
#                 top_idx = np.argsort(probs)[-5:][::-1]
#                 st.markdown("---")
#                 st.subheader("💡 Predicted ICD/CPT Codes")

#                 for i in top_idx:
#                     label = (
#                         label_binarizer.classes_[i]
#                         if label_binarizer is not None and hasattr(label_binarizer, "classes_")
#                         else f"Code_{i}"
#                     )
#                     conf = int(probs[i] * 100)
#                     st.markdown(
#                         f"""
#                         <div style="background:linear-gradient(135deg,#f0f7ff,#fff);
#                                     padding:14px;border-radius:12px;margin-bottom:10px;
#                                     border-left:6px solid #004b9b;
#                                     box-shadow:0 3px 6px rgba(0,0,0,.1);">
#                             <h4 style="color:#003f7d;">{label}</h4>
#                             <p style="font-size:13px;color:#1a1a1a;">
#                                 Confidence: <b>{conf}%</b>
#                             </p>
#                         </div>
#                         """,
#                         unsafe_allow_html=True,
#                     )
# ---------------------------------------------------------------------
# Tab 5: Doctor Note ICD/CPT Prediction (Image Upload)
# ---------------------------------------------------------------------
import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from torch.serialization import add_safe_globals
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

# Safely allow NumPy + sklearn objects
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

    # Load weights safely
    try:
        state_dict = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict.get("model", state_dict), strict=False)
        # st.success(" Loaded model from best_model.pt")
    except Exception as e:
        st.error(f" Could not load {best_path}: {e}")

    # Load label binarizer
    try:
        lb = joblib.load(lb_path)
        # st.success("✅ Loaded label binarizer (pkl)")
    except Exception as e:
        st.warning(f"⚠️ Label binarizer not found or invalid — fallback to generic labels. {e}")
        lb = None

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, lb, device

tokenizer, icd_cpt_model, label_binarizer, device = load_icd_cpt_model_safe()

# ICD/CPT descriptions (extendable)
code_descriptions = {
    "I10": "Essential (primary) hypertension",
    "R50.9": "Fever, unspecified",
    "A09": "Infectious gastroenteritis and colitis, unspecified",
    "M75.5": "Shoulder pain, unspecified",
    "99213": "Office or other outpatient visit (established patient)",
    "99214": "Office or other outpatient visit (moderate)",
    "87880": "Strep A test, antigen detection",
    "99391": "Periodic comprehensive preventive exam",
}

# ----------------------- UI -----------------------
with tabs[4]:
    st.header(" Doctor Note ICD/CPT Prediction (Image Upload)")
    st.caption("Upload a handwritten note image (.png/.jpg). The app will extract text and predict probable ICD/CPT codes.")

    uploaded_file = st.file_uploader("Upload Doctor's Handwritten Note", type=["png", "jpg", "jpeg"], key="icd_upload5")

    if uploaded_file:
        # ↓↓↓ Smaller, centered image ↓↓↓
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Note", width=500)  # reduced image width

        if st.button("Extract & Predict Codes", key="predict_icd_cpt5"):
            with st.spinner("🔍 Extracting text and predicting ICD/CPT codes..."):
                extracted_text = pytesseract.image_to_string(image)
                cleaned_text = extracted_text.strip().replace("\n", " ")

                st.subheader(" Extracted Text")
                st.write(cleaned_text if cleaned_text else "_(No readable text found)_")

                if not cleaned_text:
                    st.stop()

                # Tokenize input
                inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                icd_cpt_model.eval()
                with torch.no_grad():
                    logits = icd_cpt_model(**inputs).logits
                    probs = torch.sigmoid(logits).cpu().numpy()[0]

                # Top-5 predictions
                top_indices = np.argsort(probs)[-5:][::-1]
                st.markdown("---")
                st.subheader(" Predicted ICD/CPT Codes")

                for i in top_indices:
                    label = (
                        label_binarizer.classes_[i]
                        if label_binarizer is not None
                        else f"Code_{i}"
                    )
                    desc = code_descriptions.get(label, "Description not available")
                    conf = probs[i]

                    st.markdown(
                        f"""
                        <div style="background: linear-gradient(135deg, #f9fbfd 0%, #ffffff 100%);
                                    padding:12px; border-radius:10px; margin-bottom:8px;
                                    border-left:5px solid #00509e; box-shadow: 0 2px 4px rgba(0,0,0,0.08);">
                        <h4 style="color:#003f7d;margin-bottom:2px;">{label}</h4>
                        <p style="font-size:13px; color:#1a1a1a; margin-bottom:4px;">{desc}</p>
                        <p style="font-size:12px; color:#0d47a1;">Confidence: <b>{int(conf*100)}%</b></p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
