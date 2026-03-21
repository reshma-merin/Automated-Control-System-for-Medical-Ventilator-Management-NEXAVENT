"""
CNC Predictive Maintenance — FastAPI Backend
=============================================
Endpoints:
  POST /api/ingest          — accept sensor row, run all 4 models
  POST /api/validate_csv    — validate operator CSV before submission
  GET  /api/alerts          — list active alerts
  GET  /api/work-orders     — list auto-generated work orders
  GET  /api/health          — health check

Run:
    uvicorn api.main:app --reload --port 8000
"""

import os
import json
import time
import joblib
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from typing import Optional
from loguru import logger
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.train_all import LSTMFailureModel, Conv1DAutoencoder

MODEL_DIR = "models/saved"
DATA_DIR  = "data/processed"

# ── Thresholds for alert severity ─────────────────────────────────────────────
ALERT_THRESHOLDS = {
    "failure_risk":   {"warning": 0.30, "critical": 0.60},
    "anomaly_score":  {"warning": 0.40, "critical": 0.70},
    "wrong_param":    {"warning": 0.40, "critical": 0.65},
    "rul_minutes":    {"warning": 60,   "critical": 20},
}

app = FastAPI(
    title="CNC Predictive Maintenance",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# In-memory stores (swap for a DB in production)
alert_store: list = []
work_order_store: list = []
machine_store: dict = {}


# ── Model registry ─────────────────────────────────────────────────────────────
class Models:
    def __init__(self):
        self.lstm = None
        self.ae = None
        self.rf = None
        self.gbr = None
        self.scaler = None
        self.meta = None
        self.ae_threshold = 0.5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ready = False

    def load(self):
        try:
            with open(f"{MODEL_DIR}/model_meta.json") as f:
                self.meta = json.load(f)

            n = self.meta["n_features"]

            # LSTM
            self.lstm = LSTMFailureModel(n).to(self.device)
            self.lstm.load_state_dict(
                torch.load(f"{MODEL_DIR}/lstm_best.pt", map_location=self.device))
            self.lstm.eval()

            # Autoencoder
            self.ae = Conv1DAutoencoder(n).to(self.device)
            self.ae.load_state_dict(
                torch.load(f"{MODEL_DIR}/autoencoder_best.pt", map_location=self.device))
            self.ae.eval()
            self.ae_threshold = self.meta["models"]["autoencoder"]["threshold"]

            # Sklearn
            self.rf  = joblib.load(f"{MODEL_DIR}/rf_classifier.pkl")
            self.gbr = joblib.load(f"{MODEL_DIR}/gbr_rul.pkl")
            self.scaler = joblib.load(f"{DATA_DIR}/scaler.pkl")

            self.ready = True
            logger.success("All models loaded")
        except Exception as e:
            logger.warning(f"Models not found ({e}) — running in demo mode")
            self.ready = False


models = Models()


@app.on_event("startup")
async def startup():
    models.load()


# ── Schemas ────────────────────────────────────────────────────────────────────
class SensorReading(BaseModel):
    machine_id: str = "CNC-01"
    operator_id: str = ""
    timestamp: str = ""
    air_temp_K: float
    process_temp_K: float
    rpm: float
    torque_Nm: float
    tool_wear_min: float
    product_type: str = "M"    # L / M / H


class CSVRow(BaseModel):
    machine_id: str
    operator_id: str
    air_temp_K: float
    process_temp_K: float
    rpm: float
    torque_Nm: float
    tool_wear_min: float
    product_type: str = "M"


# ── Inference helpers ──────────────────────────────────────────────────────────
def build_feature_vector(r: SensorReading) -> np.ndarray:
    """Replicate the same feature engineering from preprocess.py."""
    import math
    power_W       = r.torque_Nm * (r.rpm * 2 * math.pi / 60)
    temp_delta    = r.process_temp_K - r.air_temp_K
    wear_rate     = r.tool_wear_min / 240.0
    torque_x_rpm  = r.torque_Nm * r.rpm

    return np.array([
        r.air_temp_K, r.process_temp_K, r.rpm,
        r.torque_Nm, r.tool_wear_min,
        power_W, temp_delta, wear_rate, torque_x_rpm,
    ], dtype=np.float32)


def run_models(feat: np.ndarray) -> dict:
    """Run all 4 models. Returns raw scores."""
    if not models.ready:
        # Demo mode — realistic-looking mock scores
        rng = np.random.default_rng(int(feat.sum() * 1000) % 9999)
        return {
            "failure_risk":    float(rng.beta(1.5, 6)),
            "anomaly_score":   float(rng.beta(1.5, 5)),
            "wrong_param_prob":float(rng.beta(1, 9)),
            "rul_minutes":     float(rng.uniform(20, 220)),
        }

    scaled = models.scaler.transform(feat.reshape(1, -1)).astype(np.float32)

    # 1. LSTM failure risk
    x_lstm = torch.FloatTensor(scaled).unsqueeze(1).to(models.device)
    with torch.no_grad():
        failure_risk = float(models.lstm(x_lstm).cpu().item())

    # 2. Autoencoder anomaly score
    x_ae = torch.FloatTensor(scaled).unsqueeze(-1).to(models.device)
    with torch.no_grad():
        ae_err = float(models.ae.reconstruction_error(x_ae).cpu().item())
    anomaly_score = min(ae_err / (models.ae_threshold * 2 + 1e-8), 1.0)

    # 3. RF wrong-param probability
    wrong_param_prob = float(models.rf.predict_proba(scaled)[0, 1])

    # 4. GBR RUL
    rul = float(np.clip(models.gbr.predict(scaled)[0], 0, 240))

    return {
        "failure_risk":     failure_risk,
        "anomaly_score":    anomaly_score,
        "wrong_param_prob": wrong_param_prob,
        "rul_minutes":      rul,
    }


def get_severity(scores: dict, validation_status: str) -> str:
    t = ALERT_THRESHOLDS
    if (scores["failure_risk"]    > t["failure_risk"]["critical"]  or
        scores["rul_minutes"]     < t["rul_minutes"]["critical"]   or
        scores["wrong_param_prob"]> t["wrong_param"]["critical"]   or
        validation_status == "critical"):
        return "critical"
    if (scores["failure_risk"]    > t["failure_risk"]["warning"]   or
        scores["anomaly_score"]   > t["anomaly_score"]["warning"]  or
        scores["rul_minutes"]     < t["rul_minutes"]["warning"]    or
        validation_status == "warning"):
        return "warning"
    return "ok"


def make_alerts(machine_id: str, scores: dict) -> list:
    ts = datetime.now().isoformat()
    alerts = []
    t = ALERT_THRESHOLDS

    if scores["failure_risk"] > t["failure_risk"]["warning"]:
        alerts.append({
            "id": f"ALT-{int(time.time())}-FAIL",
            "machine_id": machine_id,
            "type": "failure_prediction",
            "severity": "critical" if scores["failure_risk"] > t["failure_risk"]["critical"] else "warning",
            "message": f"Failure risk {scores['failure_risk']:.0%} detected on {machine_id}",
            "value": scores["failure_risk"],
            "timestamp": ts,
        })

    if scores["anomaly_score"] > t["anomaly_score"]["warning"]:
        alerts.append({
            "id": f"ALT-{int(time.time())}-ANOM",
            "machine_id": machine_id,
            "type": "anomaly_detected",
            "severity": "critical" if scores["anomaly_score"] > t["anomaly_score"]["critical"] else "warning",
            "message": f"Sensor anomaly detected (score={scores['anomaly_score']:.2f}) on {machine_id}",
            "value": scores["anomaly_score"],
            "timestamp": ts,
        })

    if scores["wrong_param_prob"] > t["wrong_param"]["warning"]:
        alerts.append({
            "id": f"ALT-{int(time.time())}-PARAM",
            "machine_id": machine_id,
            "type": "wrong_parameter",
            "severity": "critical",
            "message": f"Possible wrong parameter entry on {machine_id} — verify against job sheet",
            "value": scores["wrong_param_prob"],
            "timestamp": ts,
        })

    if scores["rul_minutes"] < t["rul_minutes"]["warning"]:
        alerts.append({
            "id": f"ALT-{int(time.time())}-RUL",
            "machine_id": machine_id,
            "type": "low_rul",
            "severity": "critical" if scores["rul_minutes"] < t["rul_minutes"]["critical"] else "warning",
            "message": f"Tool RUL critical: {scores['rul_minutes']:.0f} min remaining on {machine_id}",
            "value": scores["rul_minutes"],
            "timestamp": ts,
        })

    return alerts


def get_solutions(alerts: list, scores: dict) -> list:
    """Map alert types to automated solutions from the dataset's failure modes."""
    failure_solutions = {
        "failure_prediction": [
            "Inspect cutting tool for wear — replace if wear > 200 min",
            "Check coolant flow to cutting zone",
            "Reduce feed rate by 15% to lower mechanical stress",
            "Prepare standby tooling for immediate swap",
            "Alert shift supervisor and log in maintenance system",
        ],
        "anomaly_detected": [
            "Run full sensor self-diagnostic cycle",
            "Check spindle bearings for unusual vibration or noise",
            "Verify workpiece clamping — chatter may be distorting readings",
            "Inspect coolant nozzle for blockage",
            "Review last 10 cycles for trend deviations",
        ],
        "wrong_parameter": [
            "STOP — verify all parameters against the job traveller",
            "Cross-check spindle RPM against material and tool spec",
            "Confirm torque limit matches workpiece material hardness",
            "Verify feed rate is within tool manufacturer's recommended range",
            "Get second operator sign-off before resuming",
        ],
        "low_rul": [
            f"Schedule tool change within {scores['rul_minutes']:.0f} minutes",
            "Order replacement insert from tool crib now",
            "Log tool life data in the CMMS for trend analysis",
            "Check tool path program for unnecessary extra passes",
        ],
    }

    solutions = []
    seen = set()
    for alert in alerts:
        atype = alert["type"]
        if atype not in seen and atype in failure_solutions:
            solutions.extend(failure_solutions[atype][:3])
            seen.add(atype)
    return solutions


def make_work_order(machine_id: str, scores: dict, severity: str, solutions: list) -> dict:
    eta = {"critical": "ASAP — within 1 hour",
           "warning": "Within 4 hours",
           "ok": "Next scheduled maintenance"}
    return {
        "id": f"WO-{int(time.time())}",
        "machine_id": machine_id,
        "priority": {"critical": "URGENT", "warning": "HIGH", "ok": "ROUTINE"}[severity],
        "eta": eta[severity],
        "failure_risk": scores["failure_risk"],
        "rul_minutes": scores["rul_minutes"],
        "actions": solutions[:5],
        "status": "open",
        "created_at": datetime.now().isoformat(),
    }


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {"status": "ok", "models_ready": models.ready,
            "timestamp": datetime.now().isoformat()}


@app.post("/api/ingest")
def ingest(reading: SensorReading):
    """Accept a sensor reading, run all 4 models, generate alerts + work orders."""
    ts = reading.timestamp or datetime.now().isoformat()

    # Validate operator parameters
    validation_status = "ok"
    if models.meta:
        ranges = models.meta.get("operator_ranges", {})
        field_map = {
            "Air temperature [K]":     reading.air_temp_K,
            "Process temperature [K]": reading.process_temp_K,
            "Rotational speed [rpm]":  reading.rpm,
            "Torque [Nm]":             reading.torque_Nm,
            "Tool wear [min]":         reading.tool_wear_min,
        }
        for param, val in field_map.items():
            if param in ranges:
                r = ranges[param]
                if val < r["min"] or val > r["max"]:
                    validation_status = "warning"

    feat   = build_feature_vector(reading)
    scores = run_models(feat)
    severity = get_severity(scores, validation_status)
    alerts = make_alerts(reading.machine_id, scores)
    solutions = get_solutions(alerts, scores)

    alert_store.extend(alerts)

    if severity in ("warning", "critical") and solutions:
        wo = make_work_order(reading.machine_id, scores, severity, solutions)
        work_order_store.append(wo)

    machine_store[reading.machine_id] = {
        "machine_id": reading.machine_id,
        "last_seen": ts,
        "scores": scores,
        "severity": severity,
        "validation_status": validation_status,
    }

    return {
        "machine_id":    reading.machine_id,
        "timestamp":     ts,
        "severity":      severity,
        "validation_status": validation_status,
        "scores":        scores,
        "alerts":        alerts,
        "solutions":     solutions,
    }


@app.post("/api/validate_csv")
def validate_csv(row: CSVRow):
    """Validate a single CSV row before it's submitted."""
    errors, warnings = [], []

    if models.meta:
        ranges = models.meta.get("operator_ranges", {})
        checks = {
            "Air temperature [K]":     row.air_temp_K,
            "Process temperature [K]": row.process_temp_K,
            "Rotational speed [rpm]":  row.rpm,
            "Torque [Nm]":             row.torque_Nm,
            "Tool wear [min]":         row.tool_wear_min,
        }
        for param, val in checks.items():
            if param in ranges:
                r = ranges[param]
                if val < r["min"] * 0.8 or val > r["max"] * 1.2:
                    errors.append(f"CRITICAL: {param}={val} far outside limits [{r['min']}, {r['max']}]")
                elif val < r["min"] or val > r["max"]:
                    warnings.append(f"WARNING: {param}={val} outside safe range [{r['min']}, {r['max']}]")

    status = "critical" if errors else ("warning" if warnings else "ok")
    return {"status": status, "errors": errors, "warnings": warnings}


@app.get("/api/alerts")
def list_alerts(severity: Optional[str] = None, limit: int = 50):
    result = list(reversed(alert_store[-limit:]))
    if severity:
        result = [a for a in result if a["severity"] == severity]
    return {"alerts": result, "count": len(result)}


@app.get("/api/work-orders")
def list_work_orders():
    return {"work_orders": list(reversed(work_order_store)), "count": len(work_order_store)}


@app.get("/api/machines")
def list_machines():
    return {"machines": list(machine_store.values()), "count": len(machine_store)}


@app.patch("/api/work-orders/{wo_id}/close")
def close_work_order(wo_id: str):
    for wo in work_order_store:
        if wo["id"] == wo_id:
            wo["status"] = "closed"
            wo["closed_at"] = datetime.now().isoformat()
            return wo
    raise HTTPException(404, "Work order not found")
