"""Page 1 — Live Dashboard"""
import streamlit as st
import httpx, random, numpy as np
import plotly.graph_objects as go

API = "http://localhost:8000"
st.title("📊 Live Dashboard")

if st.button("🔄 Refresh"):
    st.rerun()

try:
    health = httpx.get(f"{API}/api/health", timeout=3).json()
    st.success(f"API online · models ready: {health['models_ready']}")
except:
    st.warning("API offline — showing demo data")

st.divider()

# Fleet overview
MACHINES = [
    {"id": "CNC-01", "loc": "Cell A", "risk": random.uniform(0.05, 0.25)},
    {"id": "CNC-02", "loc": "Cell A", "risk": random.uniform(0.65, 0.85)},
    {"id": "CNC-03", "loc": "Cell B", "risk": random.uniform(0.35, 0.55)},
    {"id": "CNC-04", "loc": "Cell B", "risk": random.uniform(0.10, 0.30)},
    {"id": "CNC-05", "loc": "Cell C", "risk": random.uniform(0.70, 0.90)},
    {"id": "CNC-06", "loc": "Cell C", "risk": random.uniform(0.15, 0.35)},
]

cols = st.columns(len(MACHINES))
for i, m in enumerate(MACHINES):
    sev = "🔴" if m["risk"] > 0.6 else "🟡" if m["risk"] > 0.35 else "🟢"
    cols[i].metric(f"{sev} {m['id']}", f"{m['risk']:.0%} risk", m["loc"])

st.divider()
st.subheader("Sensor trend — CNC-02 (last 50 readings)")

np.random.seed(42)
x = list(range(50))
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=1500 + np.cumsum(np.random.randn(50)*10),
                          name="RPM", line=dict(color="#378ADD")))
fig.add_trace(go.Scatter(x=x, y=40 + np.cumsum(np.random.randn(50)*0.5),
                          name="Torque (Nm)", line=dict(color="#EF9F27")))
fig.add_trace(go.Scatter(x=x, y=np.linspace(80, 210, 50) + np.random.randn(50)*2,
                          name="Tool wear (min)", line=dict(color="#E24B4A")))
fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                  legend=dict(orientation="h", y=1.1))
st.plotly_chart(fig, use_container_width=True)
