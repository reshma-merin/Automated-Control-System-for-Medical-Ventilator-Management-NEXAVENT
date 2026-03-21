"""
CNC Predictive Maintenance — Streamlit App
==========================================
Run: streamlit run app/streamlit_app.py
"""
import streamlit as st

st.set_page_config(
    page_title="CNC PredictIQ",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("⚙️ CNC PredictIQ")
st.sidebar.caption("Predictive Maintenance System")
st.sidebar.divider()
st.sidebar.markdown("""
**Pages**
- 📊 Dashboard
- 📂 CSV Input
- 🚨 Alerts
- 🔧 Work Orders
""")

st.title("⚙️ CNC PredictIQ — Predictive Maintenance")
st.markdown("""
An end-to-end AI system for CNC machine predictive maintenance.
Upload operator parameters, get instant failure predictions, anomaly alerts and automated work orders.
""")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Dataset",      "AI4I 2020",  "10,000 real rows")
c2.metric("ML models",    "4",          "LSTM · AE · RF · GBR")
c3.metric("Failure modes","5",          "TWF HDF PWF OSF RNF")
c4.metric("Training time","< 10 min",   "CPU only")
