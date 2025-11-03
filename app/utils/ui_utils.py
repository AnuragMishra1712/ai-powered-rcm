import streamlit as st
import time

# ---------------------------------------------------------------------
# Unified AWS-style UI utilities for RCM modules
# ---------------------------------------------------------------------

def rcm_card(title, content, color="#2563EB"):
    """Reusable RCM-style card with a colored left border."""
    st.markdown(
        f"""
        <div style='background:white;border-radius:10px;
                    box-shadow:0 1px 4px rgba(0,0,0,0.08);
                    padding:1rem 1.25rem;margin-bottom:1rem;
                    border-left:6px solid {color};'>
            <h4 style='margin-bottom:0.25rem;color:#111827;'>{title}</h4>
            <p style='color:#374151;font-size:0.95rem;margin:0;'>{content}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def rcm_loader(steps, delay=0.5):
    """Displays a progress bar that advances through named steps."""
    progress = st.progress(0)
    log = st.empty()
    for i, s in enumerate(steps, start=1):
        log.info(f"🔄 {s}")
        progress.progress(int(i / len(steps) * 100))
        time.sleep(delay)
    progress.progress(100)
    time.sleep(0.2)


def risk_badge(prob):
    """Returns a risk label, description, and color based on denial probability."""
    if prob > 0.8:
        return ("🚨 Critical Risk", "Immediate review required; high chance of denial.", "#B91C1C")
    elif prob > 0.6:
        return ("⚠️ High Risk", "Significant risk; verify documentation and payer coverage.", "#DC2626")
    elif prob > 0.4:
        return ("🟠 Moderate Risk", "Some risk detected; review before claim submission.", "#F59E0B")
    else:
        return ("✅ Low Risk", "Claim likely to be approved; proceed with normal workflow.", "#059669")


def metric_row(metrics):
    """Displays a row of Streamlit metrics nicely."""
    cols = st.columns(len(metrics))
    for i, (label, value) in enumerate(metrics.items()):
        cols[i].metric(label, value)
