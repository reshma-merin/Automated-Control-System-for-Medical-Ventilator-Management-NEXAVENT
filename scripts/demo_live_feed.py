"""
MachineMind — Live Demo Feed
==============================
Simulates 6 CNC machines sending sensor readings to the API every few seconds.
Run this during your Tableau demo so charts update live.

CNC-02 and CNC-05 are scripted to deteriorate over time — their failure risk
climbs, RUL drops, and eventually a critical alert fires. This is your
demo wow moment.

Usage:
    python scripts/demo_live_feed.py
    python scripts/demo_live_feed.py --speed 2   # 2 seconds between readings
    python scripts/demo_live_feed.py --crisis     # CNC-02 goes critical immediately
"""

import time
import math
import random
import argparse
import httpx
from datetime import datetime
from loguru import logger

API_BASE = "http://localhost:8000"

# ── Machine profiles ───────────────────────────────────────────────────────────
# Each machine has a base state. CNC-02 and CNC-05 degrade over time.
MACHINES = {
    "CNC-01": {"base_wear": 40,  "base_torque": 38, "rpm": 1550, "degrade": False},
    "CNC-02": {"base_wear": 160, "base_torque": 55, "rpm": 2200, "degrade": True},   # will fail
    "CNC-03": {"base_wear": 80,  "base_torque": 42, "rpm": 1800, "degrade": False},
    "CNC-04": {"base_wear": 20,  "base_torque": 35, "rpm": 1400, "degrade": False},
    "CNC-05": {"base_wear": 185, "base_torque": 60, "rpm": 2500, "degrade": True},   # will fail
    "CNC-06": {"base_wear": 55,  "base_torque": 40, "rpm": 1650, "degrade": False},
}


def make_reading(machine_id: str, profile: dict, tick: int, crisis: bool = False) -> dict:
    """Generate a realistic sensor reading for a machine at a given time tick."""
    rng = random.Random(hash(machine_id) + tick)

    wear = profile["base_wear"]
    torque = profile["base_torque"]
    rpm = profile["rpm"]

    # Degrading machines get worse over time
    if profile["degrade"]:
        wear = min(wear + tick * 1.5, 238)
        torque = min(torque + tick * 0.3, 68)

    if crisis:
        wear = 235
        torque = 68
        rpm = 2600

    # Add realistic noise
    wear   = wear   + rng.uniform(-2, 2)
    torque = torque + rng.uniform(-1.5, 1.5)
    rpm    = rpm    + rng.randint(-50, 50)

    air_temp     = 299.5 + rng.uniform(-1.5, 1.5)
    process_temp = air_temp + 9.8 + rng.uniform(-0.5, 0.5)

    return {
        "machine_id":     machine_id,
        "operator_id":    f"OP-{hash(machine_id) % 99:02d}",
        "product_type":   rng.choice(["L", "M", "H"]),
        "air_temp_K":     round(air_temp, 2),
        "process_temp_K": round(process_temp, 2),
        "rpm":            max(1200, min(2800, rpm)),
        "torque_Nm":      round(max(10, min(70, torque)), 2),
        "tool_wear_min":  round(max(0, min(240, wear)), 1),
    }


def send_reading(payload: dict) -> dict | None:
    try:
        resp = httpx.post(f"{API_BASE}/api/ingest", json=payload, timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except httpx.ConnectError:
        logger.error("API not reachable — is FastAPI running on port 8000?")
    return None


def main(speed: int = 3, crisis: bool = False):
    logger.info("=" * 55)
    logger.info(" MachineMind — Live Demo Feed")
    logger.info(f" Sending readings every {speed}s | Crisis mode: {crisis}")
    logger.info(" Press Ctrl+C to stop")
    logger.info("=" * 55)

    # Check API is up
    try:
        health = httpx.get(f"{API_BASE}/api/health", timeout=3).json()
        logger.success(f"API online — models ready: {health.get('models_ready')}")
    except:
        logger.error("Cannot reach API. Start it first: uvicorn api.main:app --port 8000")
        return

    tick = 0
    while True:
        tick += 1
        logger.info(f"\n--- Tick {tick} | {datetime.now().strftime('%H:%M:%S')} ---")

        for machine_id, profile in MACHINES.items():
            is_crisis = crisis and machine_id in ("CNC-02", "CNC-05")
            payload = make_reading(machine_id, profile, tick, is_crisis)
            result  = send_reading(payload)

            if result:
                scores   = result.get("scores", {})
                severity = result.get("severity", "ok")
                icon     = {"critical": "🔴", "warning": "🟡", "ok": "🟢"}.get(severity, "⚪")
                logger.info(
                    f"  {icon} {machine_id} | "
                    f"risk={scores.get('failure_risk', 0):.0%} | "
                    f"RUL={scores.get('rul_minutes', 0):.0f}min | "
                    f"wear={payload['tool_wear_min']:.0f}min"
                )

                if result.get("alerts"):
                    for a in result["alerts"]:
                        logger.warning(f"    ALERT: {a['type']} — {a['message'][:60]}")

        logger.info(f"  Refresh Tableau now to see updates →")
        time.sleep(speed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed",  type=int, default=3, help="Seconds between readings")
    parser.add_argument("--crisis", action="store_true",  help="CNC-02 and CNC-05 go critical immediately")
    args = parser.parse_args()
    main(args.speed, args.crisis)