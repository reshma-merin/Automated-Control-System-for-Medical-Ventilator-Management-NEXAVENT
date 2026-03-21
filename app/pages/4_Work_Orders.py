"""Page 4 — Work Orders"""
import streamlit as st, httpx
from datetime import datetime

API = "http://localhost:8000"
st.title("🔧 Auto-Generated Work Orders")

try:
    wos = httpx.get(f"{API}/api/work-orders", timeout=4).json().get("work_orders", [])
except:
    wos = [
        {"id":"WO-001","machine_id":"CNC-02","priority":"URGENT","eta":"ASAP — within 1 hour",
         "failure_risk":0.78,"rul_minutes":18,"status":"open","created_at":datetime.now().isoformat(),
         "actions":["Replace cutting tool — wear limit reached",
                    "Inspect spindle bearings for unusual vibration",
                    "Reduce feed rate 15% until maintenance complete",
                    "Alert shift supervisor and log in CMMS"]},
        {"id":"WO-002","machine_id":"CNC-05","priority":"URGENT","eta":"ASAP — within 1 hour",
         "failure_risk":0.72,"rul_minutes":14,"status":"open","created_at":datetime.now().isoformat(),
         "actions":["Schedule tool change within 14 minutes",
                    "Order replacement insert from tool crib now",
                    "Verify coolant flow to cutting zone"]},
        {"id":"WO-003","machine_id":"CNC-03","priority":"HIGH","eta":"Within 4 hours",
         "failure_risk":0.45,"rul_minutes":55,"status":"open","created_at":datetime.now().isoformat(),
         "actions":["Run sensor self-diagnostic cycle",
                    "Check spindle bearings for vibration",
                    "Review last 10 cycles for trend deviations"]},
    ]

if not wos:
    st.info("No open work orders.")

for wo in wos:
    icon = {"URGENT":"🔴","HIGH":"🟡","ROUTINE":"🟢"}.get(wo["priority"],"⚪")
    with st.container(border=True):
        c1, c2, c3 = st.columns([2,1,1])
        c1.markdown(f"**{icon} {wo['id']}** — `{wo['machine_id']}`")
        c1.caption(f"ETA: {wo['eta']}")
        c2.metric("Failure risk", f"{wo.get('failure_risk',0):.0%}")
        c3.metric("RUL", f"{wo.get('rul_minutes',0):.0f} min")
        st.markdown("**Recommended actions:**")
        for i, action in enumerate(wo.get("actions",[]), 1):
            st.markdown(f"{i}. {action}")
        col_a, col_b = st.columns([1,5])
        if col_a.button("✅ Close", key=wo["id"]):
            try:
                httpx.patch(f"{API}/api/work-orders/{wo['id']}/close", timeout=5)
                st.success("Closed")
                st.rerun()
            except:
                st.error("API offline")
        col_b.caption(f"Status: {wo.get('status')} | {wo.get('created_at','')[:19]}")
