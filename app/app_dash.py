import streamlit as st
import pandas as pd
import datetime
import random

# ---------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="AI-Powered RCM Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------------------
# Global Styles
# ---------------------------------------------------------------------
st.markdown("""
    <style>
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            background-color: #F9FAFB;
            color: #111827;
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #E5E7EB;
        }
        .env-pill {
            background: #DCFCE7;
            color: #166534;
            padding: 6px 14px;
            border-radius: 9999px;
            font-size: 13px;
            font-weight: 500;
        }
        .service-card {
            background: #FFFFFF;
            border-radius: 16px;
            padding: 22px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.05);
            transition: all 0.2s ease;
            border: 1px solid #E5E7EB;
        }
        .service-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 16px rgba(0,0,0,0.08);
            border-color: #2563EB;
        }
        .metric {
            font-size: 24px;
            font-weight: 600;
            color: #1E3A8A;
        }
        .label {
            font-size: 13px;
            color: #6B7280;
        }
        .footer {
            text-align: center;
            font-size: 13px;
            color: #9CA3AF;
            margin-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
col1, col2 = st.columns([0.8, 0.2])
with col1:
    st.markdown("<h1 style='color:#1E3A8A;margin-bottom:-8px;'>AI-Powered RCM Suite</h1>", unsafe_allow_html=True)
    st.caption("Executive Overview • Predict • Automate • Optimize")
with col2:
    st.markdown("<div style='text-align:right;' class='env-pill'>Production</div>", unsafe_allow_html=True)

st.markdown("---")

# ---------------------------------------------------------------------
# Simulated Metrics
# ---------------------------------------------------------------------
today = datetime.datetime.now().strftime("%b %d, %Y")
mock_metrics = {
    "Denial Prediction": {
        "accuracy": "94.1%",
        "claims_today": 187,
        "last_run": today,
        "status": "Operational"
    },
    "AI-Assisted Coding": {
        "accuracy": "91.6%",
        "notes_processed": 243,
        "last_run": today,
        "status": "Stable"
    },
    "Prior Authorization": {
        "efficiency_gain": "32%",
        "pending_cases": 48,
        "last_run": today,
        "status": "Active"
    },
    "Billing Optimization": {
        "collection_rate": "89%",
        "followups_today": 61,
        "last_run": today,
        "status": "Stable"
    },
    "ICD/CPT from Notes": {
        "inference_latency": "1.3s",
        "images_processed": 72,
        "last_run": today,
        "status": "Operational"
    }
}

# ---------------------------------------------------------------------
# Service Cards
# ---------------------------------------------------------------------
st.subheader("System Modules")

cols = st.columns(3)
keys = list(mock_metrics.keys())

for i, service in enumerate(keys):
    metrics = mock_metrics[service]
    with cols[i % 3]:
        st.markdown(f"""
            <div class='service-card'>
                <h3 style='margin-bottom:4px;'>{service}</h3>
                <p class='label'>{metrics["status"]} • Last Run: {metrics["last_run"]}</p>
                <div style='margin-top:10px;'>
                    <div class='metric'>{list(metrics.values())[0]}</div>
                    <div class='label'>{list(metrics.keys())[0].replace("_", " ").capitalize()}</div>
                </div>
                <div style='margin-top:10px;'>
                    <div class='label'>Throughput: {list(metrics.values())[1]}</div>
                </div>
                <a href='#' target='_self'>
                    <div style='margin-top:12px;background:#2563EB;color:white;padding:8px 14px;
                                text-align:center;border-radius:8px;font-weight:500;'>Open Module</div>
                </a>
            </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Analytics Snapshot
# ---------------------------------------------------------------------
st.markdown("### Performance Summary")

data = {
    "Module": list(mock_metrics.keys()),
    "Availability (%)": [99.8, 99.2, 98.7, 99.1, 99.5],
    "Avg Response Time (s)": [0.7, 1.2, 1.8, 1.1, 1.4],
    "Last Updated": [today] * 5
}
df = pd.DataFrame(data)
st.dataframe(df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------
st.markdown(
    "<div class='footer'>© 2025 AI-Powered RCM Suite • Built for Scalability & Insight</div>",
    unsafe_allow_html=True
)
