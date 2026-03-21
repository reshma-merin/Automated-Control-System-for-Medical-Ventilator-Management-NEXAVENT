"""
CNC Predictive Maintenance — Data Preprocessing
=================================================
Downloads the AI4I 2020 dataset from UCI, runs EDA,
engineers features, labels wrong operator inputs,
scales and splits into train/val/test sets.

Run:
    python data/preprocess.py

Outputs (saved to data/processed/):
    X_train.npy, X_val.npy, X_test.npy
    y_failure_train.npy  ... (binary — machine failure)
    y_anomaly_train.npy  ... (binary — anomaly flag)
    y_wrongparam_train.npy .. (binary — wrong operator input)
    y_rul_train.npy      ... (float — remaining tool life in mins)
    scaler.pkl
    feature_names.json
    eda_report.json
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from loguru import logger

OUTPUT_DIR = "data/processed"
RANDOM_STATE = 42

# ── Safe operating ranges for operator-entered parameters ─────────────────────
# Values outside these → wrong_param = 1
OPERATOR_RANGES = {
    "Air temperature [K]":        {"min": 295.0, "max": 305.0},
    "Process temperature [K]":    {"min": 305.0, "max": 315.0},
    "Rotational speed [rpm]":     {"min": 1200,  "max": 2800},
    "Torque [Nm]":                {"min": 10.0,  "max": 70.0},
    "Tool wear [min]":            {"min": 0,     "max": 240},
}

# ── Failure mode mapping → human readable solutions ───────────────────────────
FAILURE_SOLUTIONS = {
    "TWF": [  # Tool Wear Failure
        "Replace cutting tool immediately — wear limit exceeded",
        "Inspect tool holder for damage or misalignment",
        "Review tool life settings and reduce cut depth by 10%",
        "Check coolant flow rate to cutting zone",
    ],
    "HDF": [  # Heat Dissipation Failure
        "Check coolant system — flow rate may be insufficient",
        "Reduce spindle speed by 15% to lower heat generation",
        "Inspect and clean heat exchanger fins",
        "Verify ambient temperature in machining area",
    ],
    "PWF": [  # Power Failure
        "Check spindle motor and drive unit for faults",
        "Verify power supply voltage is within spec (±5%)",
        "Inspect electrical connections in the control cabinet",
        "Reduce torque load — current cut parameters too aggressive",
    ],
    "OSF": [  # Overstrain Failure
        "Reduce torque setpoint — overstrain threshold exceeded",
        "Decrease feed rate by 20% to reduce mechanical load",
        "Check workpiece clamping — vibration may be amplifying torque",
        "Inspect gearbox and spindle bearings for wear",
    ],
    "RNF": [  # Random Failure
        "Run full diagnostic cycle on all subsystems",
        "Check machine logs for intermittent error codes",
        "Inspect all sensor connections for loose contacts",
        "Schedule preventive maintenance inspection",
    ],
}


def download_dataset() -> pd.DataFrame:
    """Fetch AI4I 2020 from UCI repo — one line, no sign-up needed."""
    logger.info("Downloading AI4I 2020 dataset from UCI...")
    repo = fetch_ucirepo(id=601)
    X = repo.data.features
    y = repo.data.targets
    df = pd.concat([X, y], axis=1)
    logger.success(f"Downloaded: {df.shape[0]} rows × {df.shape[1]} columns")
    logger.info(f"  Raw columns: {list(df.columns)}")

    # The ucimlrepo library sometimes strips the unit suffixes from column names.
    # We normalise here so the rest of the pipeline always sees the full names.
    col_map = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        if "air temp" in col_lower:
            col_map[col] = "Air temperature [K]"
        elif "process temp" in col_lower:
            col_map[col] = "Process temperature [K]"
        elif "rotational" in col_lower or "rpm" in col_lower:
            col_map[col] = "Rotational speed [rpm]"
        elif "torque" in col_lower:
            col_map[col] = "Torque [Nm]"
        elif "tool wear" in col_lower:
            col_map[col] = "Tool wear [min]"
        elif "machine failure" in col_lower:
            col_map[col] = "Machine failure"
        elif col_lower in ("twf", "hdf", "pwf", "osf", "rnf"):
            col_map[col] = col.upper()
    if col_map:
        df = df.rename(columns=col_map)
        logger.info(f"  Normalised columns: {list(df.columns)}")
    return df


def run_eda(df: pd.DataFrame) -> dict:
    """Quick EDA — distributions, class balance, correlations."""
    logger.info("Running EDA...")

    fail_col  = next((c for c in df.columns if "machine failure" in c.lower()), "Machine failure")
    mode_cols = [c for c in df.columns if c.upper() in ("TWF", "HDF", "PWF", "OSF", "RNF")]

    report = {
        "shape": list(df.shape),
        "null_counts": df.isnull().sum().to_dict(),
        "failure_rate": float(df[fail_col].mean()),
        "failure_modes": {
            col: {"count": int(df[col].sum()), "rate": float(df[col].mean())}
            for col in mode_cols
        },
        "feature_stats": df.describe().to_dict(),
    }

    logger.info(f"  Overall failure rate  : {report['failure_rate']:.2%}")
    for mode, stats in report["failure_modes"].items():
        logger.info(f"  {mode}: {stats['count']} failures ({stats['rate']:.2%})")

    return report


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features that the models will benefit from:
      - power        = torque × (rpm × 2π/60)   [Watts]
      - temp_delta   = process_temp - air_temp    [K]
      - wear_rate    = tool_wear / max_tool_wear  [0–1]
      - torque_rpm   = torque × rpm               [interaction]
    """
    logger.info("Engineering features...")
    df = df.copy()

    df["power_W"] = df["Torque [Nm]"] * (df["Rotational speed [rpm]"] * 2 * np.pi / 60)
    df["temp_delta_K"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["wear_rate"] = df["Tool wear [min]"] / 240.0
    df["torque_x_rpm"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"]

    logger.info(f"  Added 4 engineered features → total {df.shape[1]} columns")
    return df


def label_wrong_params(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag rows where any operator-entered parameter is outside safe range.
    In a real system this fires before the machine runs.
    Here we create supervised labels for the Random Forest classifier.
    """
    df = df.copy()
    df["wrong_param"] = 0
    df["wrong_param_detail"] = ""

    for col, limits in OPERATOR_RANGES.items():
        if col not in df.columns:
            continue
        mask = (df[col] < limits["min"]) | (df[col] > limits["max"])
        df.loc[mask, "wrong_param"] = 1
        df.loc[mask, "wrong_param_detail"] += col + "; "

    n_flagged = df["wrong_param"].sum()
    logger.info(f"Wrong param labels: {n_flagged} rows ({n_flagged/len(df):.2%})")
    return df


def compute_rul(df: pd.DataFrame) -> pd.DataFrame:
    """
    RUL = remaining tool life in minutes.
    Tool fails between 200–240 mins wear → RUL = 240 - tool_wear, clipped at 0.
    """
    df = df.copy()
    df["rul_minutes"] = (240 - df["Tool wear [min]"]).clip(lower=0).astype(float)
    logger.info(f"RUL computed: mean={df['rul_minutes'].mean():.1f} min, "
                f"min={df['rul_minutes'].min():.0f}, max={df['rul_minutes'].max():.0f}")
    return df


def build_arrays(df: pd.DataFrame) -> tuple:
    """Select feature columns and build numpy arrays for all 4 targets."""

    feature_cols = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "power_W",
        "temp_delta_K",
        "wear_rate",
        "torque_x_rpm",
    ]

    X = df[feature_cols].values.astype(np.float32)

    fail_col = next((c for c in df.columns if "machine failure" in c.lower()), "Machine failure")
    y_failure   = df[fail_col].values.astype(np.int32)
    y_anomaly   = df["wrong_param"].values.astype(np.int32)
    y_wrongparam = df["wrong_param"].values.astype(np.int32)
    y_rul       = df["rul_minutes"].values.astype(np.float32)

    logger.info(f"Feature matrix: {X.shape}")
    logger.info(f"Failure labels: {y_failure.mean():.2%} positive")
    logger.info(f"Wrong param labels: {y_wrongparam.mean():.2%} positive")

    return X, y_failure, y_anomaly, y_wrongparam, y_rul, feature_cols


def split_and_scale(X, y_failure, y_anomaly, y_wrongparam, y_rul):
    """70% train / 15% val / 15% test. Scaler fit on train only."""

    # First split: 70% train, 30% temp
    X_train, X_temp, yf_train, yf_temp, \
    ya_train, ya_temp, yw_train, yw_temp, \
    yr_train, yr_temp = train_test_split(
        X, y_failure, y_anomaly, y_wrongparam, y_rul,
        test_size=0.30, random_state=RANDOM_STATE, stratify=y_failure
    )

    # Second split: 50/50 of temp → 15% val, 15% test
    X_val, X_test, yf_val, yf_test, \
    ya_val, ya_test, yw_val, yw_test, \
    yr_val, yr_test = train_test_split(
        X_temp, yf_temp, ya_temp, yw_temp, yr_temp,
        test_size=0.50, random_state=RANDOM_STATE, stratify=yf_temp
    )

    # Scale — fit on train only to prevent leakage
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val   = scaler.transform(X_val).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    logger.info(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    return (X_train, X_val, X_test,
            yf_train, yf_val, yf_test,
            ya_train, ya_val, ya_test,
            yw_train, yw_val, yw_test,
            yr_train, yr_val, yr_test,
            scaler)


def save_outputs(output_dir, splits, scaler, feature_cols, eda_report):
    """Save all arrays, scaler, and config to disk."""
    os.makedirs(output_dir, exist_ok=True)

    (X_train, X_val, X_test,
     yf_train, yf_val, yf_test,
     ya_train, ya_val, ya_test,
     yw_train, yw_val, yw_test,
     yr_train, yr_val, yr_test) = splits

    for name, arr in [
        ("X_train", X_train),         ("X_val", X_val),         ("X_test", X_test),
        ("y_failure_train", yf_train), ("y_failure_val", yf_val), ("y_failure_test", yf_test),
        ("y_anomaly_train", ya_train), ("y_anomaly_val", ya_val), ("y_anomaly_test", ya_test),
        ("y_wrongparam_train", yw_train),("y_wrongparam_val", yw_val),("y_wrongparam_test", yw_test),
        ("y_rul_train", yr_train),     ("y_rul_val", yr_val),     ("y_rul_test", yr_test),
    ]:
        np.save(f"{output_dir}/{name}.npy", arr)

    joblib.dump(scaler, f"{output_dir}/scaler.pkl")

    config = {
        "feature_names": feature_cols,
        "n_features": len(feature_cols),
        "operator_ranges": OPERATOR_RANGES,
        "failure_solutions": FAILURE_SOLUTIONS,
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
        "failure_rate_train": float(yf_train.mean()),
        "wrongparam_rate_train": float(yw_train.mean()),
        "mean_rul_train": float(yr_train.mean()),
    }

    with open(f"{output_dir}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    with open(f"{output_dir}/eda_report.json", "w") as f:
        json.dump(eda_report, f, indent=2, default=str)

    logger.success(f"All outputs saved to {output_dir}/")


def main():
    logger.info("=" * 55)
    logger.info(" CNC Predictive Maintenance — Preprocessing Pipeline")
    logger.info("=" * 55)

    df = download_dataset()
    eda_report = run_eda(df)
    df = engineer_features(df)
    df = label_wrong_params(df)
    df = compute_rul(df)

    X, y_failure, y_anomaly, y_wrongparam, y_rul, feature_cols = build_arrays(df)

    splits_and_scaler = split_and_scale(
        X, y_failure, y_anomaly, y_wrongparam, y_rul
    )
    scaler = splits_and_scaler[-1]
    splits = splits_and_scaler[:-1]

    save_outputs(OUTPUT_DIR, splits, scaler, feature_cols, eda_report)

    logger.success("=" * 55)
    logger.success(" Preprocessing complete!")
    logger.success(f" Features  : {len(feature_cols)}")
    logger.success(f" Train rows: {splits[0].shape[0]:,}")
    logger.success(" Next: python models/train_all.py")
    logger.success("=" * 55)


if __name__ == "__main__":
    main()