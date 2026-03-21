"""
CNC Operator CSV Input Processor
==================================
The operator fills in a CSV with machine parameters.
This script validates every row and submits safe ones to the API.

Usage:
    python scripts/csv_input.py --generate_template
    python scripts/csv_input.py --csv operator_input.csv
    python scripts/csv_input.py --csv operator_input.csv --dry_run
"""

import os
import sys
import json
import argparse
import pandas as pd
import httpx
from datetime import datetime
from loguru import logger

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

SAFE_RANGES = {
    "air_temp_K":        {"min": 295.0, "max": 305.0,  "unit": "K"},
    "process_temp_K":    {"min": 305.0, "max": 315.0,  "unit": "K"},
    "rpm":               {"min": 1200,  "max": 2800,   "unit": "rpm"},
    "torque_Nm":         {"min": 10.0,  "max": 70.0,   "unit": "Nm"},
    "tool_wear_min":     {"min": 0,     "max": 240,    "unit": "min"},
}

COLUMNS = ["timestamp", "machine_id", "operator_id", "product_type",
           "air_temp_K", "process_temp_K", "rpm", "torque_Nm",
           "tool_wear_min", "notes"]


def generate_template(path="cnc_template.csv"):
    sample = {
        "timestamp":      [datetime.now().isoformat()],
        "machine_id":     ["CNC-01"],
        "operator_id":    ["OP-42"],
        "product_type":   ["M"],
        "air_temp_K":     [300.1],
        "process_temp_K": [310.5],
        "rpm":            [1500],
        "torque_Nm":      [42.3],
        "tool_wear_min":  [88],
        "notes":          ["Routine job setup"],
    }
    pd.DataFrame(sample).to_csv(path, index=False)
    logger.success(f"Template saved → {path}")
    print("\nSafe parameter ranges:")
    for param, r in SAFE_RANGES.items():
        print(f"  {param:<22} {r['min']} – {r['max']} {r['unit']}")


def validate_row(row: pd.Series) -> dict:
    errors, warnings = [], []

    for param, limits in SAFE_RANGES.items():
        if param not in row or pd.isna(row[param]):
            warnings.append(f"{param}: missing — using default")
            continue
        val = float(row[param])
        # Hard block: 20% outside safe range
        if val < limits["min"] * 0.80 or val > limits["max"] * 1.20:
            errors.append({
                "param": param, "value": val, "unit": limits["unit"],
                "message": f"BLOCKED: {val} {limits['unit']} is dangerously outside "
                           f"[{limits['min']}, {limits['max']}]"
            })
        elif val < limits["min"] or val > limits["max"]:
            warnings.append({
                "param": param, "value": val, "unit": limits["unit"],
                "message": f"WARNING: {val} {limits['unit']} outside safe range "
                           f"[{limits['min']}, {limits['max']}]"
            })

    # Cross-parameter: power = torque × rpm × 2π/60
    if "torque_Nm" in row and "rpm" in row:
        import math
        power = float(row["torque_Nm"]) * float(row["rpm"]) * 2 * math.pi / 60
        if power > 9000:
            errors.append({"param": "power", "value": round(power, 1),
                           "unit": "W", "message": f"BLOCKED: Power {power:.0f}W exceeds 9000W limit"})
        elif power < 3500:
            warnings.append({"param": "power", "value": round(power, 1),
                             "unit": "W", "message": f"WARNING: Power {power:.0f}W below 3500W minimum"})

    status = "critical" if errors else ("warning" if warnings else "ok")
    return {"status": status, "errors": errors, "warnings": warnings}


def submit_row(row: pd.Series, api_base: str):
    payload = {
        "machine_id":    str(row.get("machine_id", "CNC-01")),
        "operator_id":   str(row.get("operator_id", "")),
        "timestamp":     str(row.get("timestamp", datetime.now().isoformat())),
        "product_type":  str(row.get("product_type", "M")),
        "air_temp_K":    float(row.get("air_temp_K", 300)),
        "process_temp_K":float(row.get("process_temp_K", 310)),
        "rpm":           float(row.get("rpm", 1500)),
        "torque_Nm":     float(row.get("torque_Nm", 40)),
        "tool_wear_min": float(row.get("tool_wear_min", 0)),
    }
    try:
        resp = httpx.post(f"{api_base}/api/ingest", json=payload, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            logger.debug(f"  → severity={data['severity']} | "
                        f"failure={data['scores']['failure_risk']:.1%} | "
                        f"RUL={data['scores']['rul_minutes']:.0f}min")
            return data
        else:
            logger.warning(f"  API error {resp.status_code}: {resp.text[:100]}")
    except httpx.ConnectError:
        logger.warning("  API offline — row logged locally only")
    return None


def process_csv(csv_path: str, dry_run: bool = False):
    logger.info(f"Loading {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows")

    submitted, blocked, warned = 0, 0, 0
    print("\n" + "=" * 60)

    for idx, row in df.iterrows():
        v = validate_row(row)
        row_label = f"Row {idx+1} | {row.get('machine_id','?')} | {row.get('operator_id','?')}"

        if v["status"] == "critical":
            blocked += 1
            print(f"\n[BLOCKED] {row_label}")
            for err in v["errors"]:
                print(f"  ✖ {err['message']}")
        elif v["status"] == "warning":
            warned += 1
            print(f"\n[WARNING] {row_label}")
            for w in v["warnings"]:
                if isinstance(w, dict):
                    print(f"  ⚠ {w['message']}")
            if not dry_run:
                submit_row(row, API_BASE)
            submitted += 1
        else:
            print(f"\n[OK]      {row_label}")
            if not dry_run:
                submit_row(row, API_BASE)
            submitted += 1

    print("\n" + "=" * 60)
    print(f"Summary: {submitted} submitted | {warned} warnings | {blocked} blocked")
    print("=" * 60)

    if blocked > 0:
        logger.error(f"{blocked} rows blocked due to dangerous parameters.")
        sys.exit(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",               help="Path to operator CSV")
    parser.add_argument("--generate_template", action="store_true")
    parser.add_argument("--dry_run",           action="store_true",
                        help="Validate only, do not submit to API")
    args = parser.parse_args()

    if args.generate_template:
        generate_template()
    elif args.csv:
        process_csv(args.csv, args.dry_run)
    else:
        parser.print_help()
