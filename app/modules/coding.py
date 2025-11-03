import streamlit as st
import numpy as np
import re
import time
from utils.db_utils import log_metric


# ---------------------------------------------------------------------
# AI-Assisted Coding
# ---------------------------------------------------------------------
def render_coding_assistant(models):
    st.title("AI-Assisted Coding")
    st.markdown(
        "Automatically predict **ICD-10** and **CPT** codes from free-text clinical notes."
    )

    tfidf_vectorizer = models.get("tfidf_vectorizer")
    model_icd = models.get("coding_icd")
    model_cpt = models.get("coding_cpt")
    mlb_icd = models.get("mlb_icd")
    mlb_cpt = models.get("mlb_cpt")

    # --- Check model availability ---
    if not all([tfidf_vectorizer, model_icd, model_cpt, mlb_icd, mlb_cpt]):
        st.warning("⚠️ Some AI models missing — switching to intelligent rule-based fallback.")
        render_rule_based_coding()
        return

    # --- Input ---
    note = st.text_area("Enter or paste doctor's note:", height=180, key="note_text")
    generate = st.button("Generate Codes", use_container_width=True)

    if not (generate and note.strip()):
        return

    start = time.time()
    X = tfidf_vectorizer.transform([note])

    # --- ICD Prediction ---
    icd_probs_raw = model_icd.predict_proba(X)
    icd_probs = np.array(icd_probs_raw).squeeze()
    if icd_probs.ndim > 1:
        icd_probs = icd_probs[0]
    icd_labels = mlb_icd.classes_
    icd_top = np.argsort(icd_probs)[-5:][::-1]

    # --- CPT Prediction ---
    cpt_probs_raw = model_cpt.predict_proba(X)
    cpt_probs = np.array(cpt_probs_raw).squeeze()
    if cpt_probs.ndim > 1:
        cpt_probs = cpt_probs[0]
    cpt_labels = mlb_cpt.classes_
    cpt_top = np.argsort(cpt_probs)[-5:][::-1]

    runtime = time.time() - start
    confidence = float(np.max([np.max(icd_probs), np.max(cpt_probs)]))
    log_metric("AI-Assisted Coding", confidence, runtime)

    # --- Display ICD Results ---
    st.markdown("### Predicted ICD-10 Codes")
    for idx in icd_top:
        st.markdown(
            f"""
            <div class='rcm-card'>
                <b>{icd_labels[idx]}</b> — Confidence: {float(icd_probs[idx])*100:.1f}%
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- Display CPT Results ---
    st.markdown("### Predicted CPT Codes")
    for idx in cpt_top:
        st.markdown(
            f"""
            <div class='rcm-card'>
                <b>{cpt_labels[idx]}</b> — Confidence: {float(cpt_probs[idx])*100:.1f}%
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.success("✅ AI models successfully generated medical codes.")


# ---------------------------------------------------------------------
# Fallback Rule-Based Coding
# ---------------------------------------------------------------------
def render_rule_based_coding():
    note = st.text_area("Enter or paste doctor's note:", height=180, key="fallback_note")
    if st.button("Generate Codes (Rule-Based)", use_container_width=True):
        note = note.lower()
        rules = [
            (r"mri.*brain", ("70553", "C71.9", "MRI Brain with/without Contrast")),
            (r"ct.*abdomen", ("74177", "R10.9", "CT Abdomen & Pelvis")),
            (r"fracture", ("27786", "S82.90XA", "Fracture Repair")),
            (r"diabetes", ("83036", "E11.9", "HbA1C Test")),
            (r"hypertension", ("93784", "I10", "BP Monitoring")),
        ]
        matches = []
        for pat, (cpt, icd, desc) in rules:
            if re.search(pat, note):
                matches.append({"CPT": cpt, "ICD": icd, "Description": desc, "Confidence": 0.9})
        if not matches:
            matches.append({"CPT": "99213", "ICD": "Z09", "Description": "Follow-Up Visit", "Confidence": 0.75})

        for m in matches:
            st.markdown(
                f"""
                <div class='rcm-card'>
                    <b>CPT:</b> {m['CPT']} | <b>ICD:</b> {m['ICD']}<br>
                    {m['Description']}<br>
                    <i>Confidence:</i> {m['Confidence']*100:.0f}%
                </div>
                """,
                unsafe_allow_html=True,
            )
