import streamlit as st
import torch
import numpy as np
from PIL import Image
import pytesseract
import time
from utils.db_utils import log_metric

def render_icd_notes(tokenizer, icd_model, label_binarizer, device):
    st.title("ICD/CPT Prediction from Notes")
    st.markdown("Upload scanned or handwritten notes to extract codes.")

    uploaded = st.file_uploader("Upload Note", type=["png", "jpg", "jpeg"])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded Note", width=500)

        if st.button("Extract & Predict"):
            start = time.time()
            text = pytesseract.image_to_string(img).strip().replace("\n", " ")
            st.text_area("Extracted Text", text, height=150)

            with torch.no_grad():
                inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                probs = torch.sigmoid(icd_model(**inputs).logits).cpu().numpy()[0]
                top = np.argsort(probs)[-5:][::-1]

                st.markdown("### Predicted Codes")
                for i in top:
                    label = label_binarizer.classes_[i] if label_binarizer is not None else f"Code_{i}"
                    st.write(f"{label}: {probs[i]*100:.1f}% confidence")

            runtime = time.time() - start
            avg_conf = float(np.mean(probs[top]))
            log_metric("ICD/CPT from Notes", avg_conf, runtime)
