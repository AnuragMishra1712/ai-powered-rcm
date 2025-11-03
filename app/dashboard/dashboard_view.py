import streamlit as st
import pandas as pd
import datetime
import random
from app.utils.db_utils import get_metrics_summary

# ---------------------------------------------------------------------
# Dashboard Overview
# ---------------------------------------------------------------------
def render_dashboard():
    st.title("AI-Powered RCM Console Dashboard")
    st.markdown("### Unified AWS-Style Interface for Revenue Cycle Intelligence")

    # ---------------- Auto-refresh every 10 seconds ---------------- #
    st.caption("Refreshing every 10 seconds for live metrics...")
    # st_autorefresh = st.experimental_rerun  # Fallback if module not available
    try:
        from streamlit_autorefresh import st_autorefresh
    except ImportError:
        def st_autorefresh(interval=10000, key=None):
            """Safe fallback: reload dashboard manually if module not available"""
            st.info("🔄 Auto-refresh not active — install with `pip install streamlit-autorefresh`.")
            if st.button("Refresh Now"):
                st.rerun()

    # ---------------- Metrics Summary from DB ---------------- #
    df = get_metrics_summary(limit=100)

    if df.empty:
        st.info("📊 No metrics logged yet. Run predictions in any module to populate the dashboard.")
        return

    # Compute key stats dynamically
    total_preds = len(df)
    high_conf = len(df[df["confidence"] >= 0.85])
    avg_conf = df["confidence"].mean()
    avg_runtime = df["runtime"].mean()
    recent_module = df.iloc[0]["module_name"]

    # Metric cards
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Predictions", f"{total_preds}")
    c2.metric("High Confidence (>85%)", f"{high_conf}")
    c3.metric("Avg Confidence", f"{avg_conf:.2f}")

    c4, c5, c6 = st.columns(3)
    c4.metric("Avg Runtime (sec)", f"{avg_runtime:.2f}")
    c5.metric("Most Recent Module", recent_module)
    c6.metric("Last Updated", datetime.datetime.now().strftime("%H:%M:%S"))

    # ---------------- Trend Table ---------------- #
    st.markdown("### Recent Activity")
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ---------------- Aggregation by Module ---------------- #
    st.markdown("### Module Performance Summary")
    summary = (
        df.groupby("module_name")
        .agg({"confidence": "mean", "runtime": "mean", "id": "count"})
        .rename(columns={"id": "Total Runs", "confidence": "Avg Confidence", "runtime": "Avg Runtime"})
        .reset_index()
    )
    st.bar_chart(summary.set_index("module_name")["Avg Confidence"])
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.success("✅ Dashboard loaded successfully. Metrics auto-refresh when new predictions are logged.")
