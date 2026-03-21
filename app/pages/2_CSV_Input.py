"""Page 2 — CSV Input"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))

import streamlit as st
import pandas as pd
import httpx
from datetime import datetime
from csv_input import validate_row, SAFE_RANGES, COLUMNS

API = "http://localhost:8000"
st.title("📂 CSV Parameter Input")
st.caption("Upload operator parameters. Each row is validated before reaching the machine.")

with st.expander("📋 Download template + parameter ranges"):
    ranges_df = pd.DataFrame([
        {"Parameter": k, "Min": v["min"], "Max": v["max"], "Unit": v["unit"]}
        for k, v in SAFE_RANGES.items()
    ])
    st.dataframe(ranges_df, use_container_width=True, hide_index=True)
    sample = {"timestamp": [datetime.now().isoformat()], "machine_id": ["CNC-01"],
              "operator_id": ["OP-42"], "product_type": ["M"],
              "air_temp_K": [300.1], "process_temp_K": [310.5],
              "rpm": [1500], "torque_Nm": [42.3], "tool_wear_min": [88], "notes": ["setup"]}
    st.download_button("⬇ Download template", pd.DataFrame(sample).to_csv(index=False).encode(),
                       "cnc_template.csv", "text/csv")

st.divider()
uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.success(f"Loaded {len(df)} rows from {uploaded.name}")

    results = []
    for idx, row in df.iterrows():
        v = validate_row(row)
        results.append({"row": idx+1, "machine_id": row.get("machine_id","?"),
                         "status": v["status"], "_v": v, "_row": row})

    n_ok   = sum(1 for r in results if r["status"] == "ok")
    n_warn = sum(1 for r in results if r["status"] == "warning")
    n_crit = sum(1 for r in results if r["status"] == "critical")

    c1, c2, c3 = st.columns(3)
    c1.metric("✅ Valid", n_ok)
    c2.metric("⚠️ Warning", n_warn)
    c3.metric("❌ Blocked", n_crit)
    st.divider()

    for r in results:
        icon = {"ok":"✅","warning":"⚠️","critical":"❌"}[r["status"]]
        with st.expander(f"{icon} Row {r['row']} — {r['machine_id']} [{r['status'].upper()}]"):
            if r["_v"]["errors"]:
                st.error("Blocked — critical parameter error:")
                for e in r["_v"]["errors"]:
                    if isinstance(e, dict):
                        st.markdown(f"- {e['message']}")
            if r["_v"]["warnings"]:
                st.warning("Warnings:")
                for w in r["_v"]["warnings"]:
                    if isinstance(w, dict):
                        st.markdown(f"- {w['message']}")
            if r["status"] == "ok":
                st.success("All parameters within safe range.")

    submittable = [r for r in results if r["status"] != "critical"]
    if submittable and st.button(f"🚀 Submit {len(submittable)} valid rows", type="primary"):
        prog = st.progress(0)
        out = []
        for i, r in enumerate(submittable):
            row = r["_row"]
            payload = {"machine_id": str(row.get("machine_id","CNC-01")),
                       "operator_id": str(row.get("operator_id","")),
                       "timestamp": str(row.get("timestamp", datetime.now().isoformat())),
                       "product_type": str(row.get("product_type","M")),
                       "air_temp_K": float(row.get("air_temp_K", 300)),
                       "process_temp_K": float(row.get("process_temp_K", 310)),
                       "rpm": float(row.get("rpm", 1500)),
                       "torque_Nm": float(row.get("torque_Nm", 40)),
                       "tool_wear_min": float(row.get("tool_wear_min", 0))}
            try:
                resp = httpx.post(f"{API}/api/ingest", json=payload, timeout=8)
                d = resp.json() if resp.status_code == 200 else {}
                out.append({"machine": payload["machine_id"],
                             "severity": d.get("severity","—"),
                             "failure_risk": f"{d.get('scores',{}).get('failure_risk',0):.1%}",
                             "RUL (min)": f"{d.get('scores',{}).get('rul_minutes',0):.0f}"})
            except:
                out.append({"machine": payload["machine_id"], "error": "API offline"})
            prog.progress((i+1)/len(submittable))
        st.success("Done!")
        st.dataframe(pd.DataFrame(out), use_container_width=True)
