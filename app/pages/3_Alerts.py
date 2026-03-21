"""Page 3 — Alerts"""
import streamlit as st, httpx
from datetime import datetime

API = "http://localhost:8000"
st.title("🚨 Active Alerts")

sev = st.selectbox("Filter", ["all", "critical", "warning"])
try:
    url = f"{API}/api/alerts" + (f"?severity={sev}" if sev != "all" else "")
    alerts = httpx.get(url, timeout=4).json().get("alerts", [])
except:
    alerts = [
        {"id":"A1","machine_id":"CNC-02","type":"failure_prediction","severity":"critical",
         "message":"Failure risk 78% on CNC-02 — tool wear critical","timestamp":datetime.now().isoformat(),"value":0.78},
        {"id":"A2","machine_id":"CNC-05","type":"low_rul","severity":"critical",
         "message":"Tool RUL: 14 min remaining on CNC-05","timestamp":datetime.now().isoformat(),"value":14},
        {"id":"A3","machine_id":"CNC-03","type":"anomaly_detected","severity":"warning",
         "message":"Sensor anomaly score 0.61 on CNC-03","timestamp":datetime.now().isoformat(),"value":0.61},
        {"id":"A4","machine_id":"CNC-01","type":"wrong_parameter","severity":"critical",
         "message":"Possible wrong RPM entry on CNC-01 — verify job sheet","timestamp":datetime.now().isoformat(),"value":0.82},
    ]

n_crit = sum(1 for a in alerts if a["severity"]=="critical")
n_warn = sum(1 for a in alerts if a["severity"]=="warning")
if n_crit: st.error(f"{n_crit} critical alerts require immediate action")
if n_warn: st.warning(f"{n_warn} warnings")

for a in alerts:
    icon = {"critical":"🔴","warning":"🟡","ok":"🟢"}.get(a["severity"],"⚪")
    with st.container(border=True):
        c1, c2 = st.columns([3,1])
        c1.markdown(f"**{icon} {a['type'].replace('_',' ').title()}** — `{a['machine_id']}`")
        c1.caption(a["message"])
        c2.caption(a.get("timestamp","")[:19])
        c2.caption(f"Score: {a.get('value','-')}")
