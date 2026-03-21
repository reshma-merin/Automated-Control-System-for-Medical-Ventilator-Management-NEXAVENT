"""
CNC Predictive Maintenance — Train All 4 Models
=================================================
Trains in sequence, logs everything to MLflow.
Total training time: ~5–10 minutes on CPU.

Models:
  1. LSTM           — failure prediction (binary classification)
  2. Autoencoder    — anomaly detection (unsupervised)
  3. Random Forest  — wrong parameter detection (binary classification)
  4. GBR            — remaining useful life regression

Run:
    python models/train_all.py

Outputs saved to models/saved/:
    lstm_best.pt
    autoencoder_best.pt
    rf_classifier.pkl
    gbr_rul.pkl
    model_meta.json
"""

import os
import json
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import (roc_auc_score, classification_report,
                             mean_absolute_error, r2_score)
import mlflow
import mlflow.pytorch
import mlflow.sklearn
from loguru import logger

DATA_DIR  = "data/processed"
MODEL_DIR = "models/saved"
os.makedirs(MODEL_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════
# MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════

class LSTMFailureModel(nn.Module):
    """
    BiLSTM with attention for failure prediction.
    Input shape: (batch, seq_len=1, n_features)
    For tabular data we treat each row as a sequence of length 1.
    The power comes from the dense head — swap seq_len for real
    time-windows if you have streaming sensor data.
    """
    def __init__(self, n_features: int, hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=2,
                            batch_first=True, bidirectional=True,
                            dropout=dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class Conv1DAutoencoder(nn.Module):
    """
    1D convolutional autoencoder for anomaly detection.
    Trained on NORMAL samples only.
    High reconstruction error at inference = anomaly.
    Input shape: (batch, n_features, 1)
    """
    def __init__(self, n_features: int, latent: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(32, latent, kernel_size=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent, 32, kernel_size=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, n_features, kernel_size=1),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def reconstruction_error(self, x):
        return ((x - self.forward(x)) ** 2).mean(dim=(1, 2))


# ══════════════════════════════════════════════════════════════
# 1. LSTM — FAILURE PREDICTION
# ══════════════════════════════════════════════════════════════

def train_lstm():
    logger.info("── Training LSTM failure predictor ──────────────────")

    X_train = np.load(f"{DATA_DIR}/X_train.npy")
    y_train = np.load(f"{DATA_DIR}/y_failure_train.npy")
    X_val   = np.load(f"{DATA_DIR}/X_val.npy")
    y_val   = np.load(f"{DATA_DIR}/y_failure_val.npy")
    X_test  = np.load(f"{DATA_DIR}/X_test.npy")
    y_test  = np.load(f"{DATA_DIR}/y_failure_test.npy")

    n_features = X_train.shape[1]

    # Add seq_len=1 dimension for LSTM
    to_seq = lambda a: torch.FloatTensor(a).unsqueeze(1)

    train_loader = DataLoader(
        TensorDataset(to_seq(X_train), torch.FloatTensor(y_train)),
        batch_size=256, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(to_seq(X_val), torch.FloatTensor(y_val)),
        batch_size=512
    )

    pos_weight = torch.tensor([(1 - y_train.mean()) / (y_train.mean() + 1e-6)])
    model = LSTMFailureModel(n_features).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    mlflow.set_experiment("cnc-failure-prediction")
    with mlflow.start_run(run_name="lstm-bilstm"):
        mlflow.log_params({"model": "BiLSTM", "hidden": 64, "lr": 1e-3,
                           "n_features": n_features})
        best_auc, best_epoch = 0, 0
        patience = 8

        for epoch in range(40):
            model.train()
            total_loss = 0
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                optimizer.zero_grad()
                pred = model(Xb)
                loss = criterion(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            model.eval()
            preds, labels = [], []
            with torch.no_grad():
                for Xb, yb in val_loader:
                    preds.extend(model(Xb.to(device)).cpu().numpy())
                    labels.extend(yb.numpy())

            val_auc = roc_auc_score(labels, preds)
            avg_loss = total_loss / len(train_loader)
            mlflow.log_metrics({"train_loss": avg_loss, "val_auc": val_auc}, step=epoch)
            scheduler.step()

            if val_auc > best_auc:
                best_auc = val_auc
                best_epoch = epoch
                torch.save(model.state_dict(), f"{MODEL_DIR}/lstm_best.pt")

            elif epoch - best_epoch >= patience:
                logger.info(f"  Early stop at epoch {epoch+1}")
                break

            if (epoch + 1) % 5 == 0:
                logger.info(f"  Epoch {epoch+1:02d} | loss={avg_loss:.4f} | val_AUC={val_auc:.4f}")

        # Test evaluation
        model.load_state_dict(torch.load(f"{MODEL_DIR}/lstm_best.pt",
                                          map_location=device))
        model.eval()
        test_preds = []
        with torch.no_grad():
            for i in range(0, len(X_test), 512):
                batch = to_seq(X_test[i:i+512]).to(device)
                test_preds.extend(model(batch).cpu().numpy())

        test_auc = roc_auc_score(y_test, test_preds)
        binary = (np.array(test_preds) > 0.5).astype(int)
        report = classification_report(y_test, binary, output_dict=True)

        mlflow.log_metrics({"test_auc": test_auc,
                             "test_f1": report.get("1", {}).get("f1-score", 0)})
        mlflow.pytorch.log_model(model, "lstm_model",
                                  registered_model_name="CNC_FailureLSTM")

        logger.success(f"  LSTM: test AUC={test_auc:.4f} | "
                       f"F1={report.get('1',{}).get('f1-score',0):.4f}")
        return test_auc


# ══════════════════════════════════════════════════════════════
# 2. AUTOENCODER — ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════

def train_autoencoder():
    logger.info("── Training Autoencoder anomaly detector ────────────")

    X_train = np.load(f"{DATA_DIR}/X_train.npy")
    y_train = np.load(f"{DATA_DIR}/y_anomaly_train.npy")
    X_val   = np.load(f"{DATA_DIR}/X_val.npy")
    y_val   = np.load(f"{DATA_DIR}/y_anomaly_val.npy")
    X_test  = np.load(f"{DATA_DIR}/X_test.npy")
    y_test  = np.load(f"{DATA_DIR}/y_anomaly_test.npy")

    n_features = X_train.shape[1]

    # Train on NORMAL samples only
    X_normal = X_train[y_train == 0]
    logger.info(f"  Training on {len(X_normal):,} normal samples")

    # Shape: (batch, n_features, 1) for Conv1d
    to_conv = lambda a: torch.FloatTensor(a).unsqueeze(-1)

    train_loader = DataLoader(
        TensorDataset(to_conv(X_normal)),
        batch_size=256, shuffle=True
    )

    model = Conv1DAutoencoder(n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    mlflow.set_experiment("cnc-anomaly-detection")
    with mlflow.start_run(run_name="conv1d-autoencoder"):
        mlflow.log_params({"model": "Conv1DAutoencoder", "latent": 8,
                           "n_features": n_features})
        best_auc, best_epoch = 0, 0

        for epoch in range(30):
            model.train()
            total_loss = 0
            for (Xb,) in train_loader:
                Xb = Xb.to(device)
                optimizer.zero_grad()
                recon = model(Xb)
                loss = nn.MSELoss()(recon, Xb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            model.eval()
            with torch.no_grad():
                val_errors = model.reconstruction_error(
                    to_conv(X_val).to(device)
                ).cpu().numpy()

            val_auc = roc_auc_score(y_val, val_errors)
            avg_loss = total_loss / len(train_loader)
            mlflow.log_metrics({"train_loss": avg_loss, "val_auc": val_auc}, step=epoch)

            if val_auc > best_auc:
                best_auc = val_auc
                best_epoch = epoch
                torch.save(model.state_dict(), f"{MODEL_DIR}/autoencoder_best.pt")

            if (epoch + 1) % 5 == 0:
                logger.info(f"  Epoch {epoch+1:02d} | loss={avg_loss:.6f} | val_AUC={val_auc:.4f}")

        # Threshold = 95th percentile of normal reconstruction errors
        model.eval()
        with torch.no_grad():
            normal_errors = model.reconstruction_error(
                to_conv(X_normal).to(device)
            ).cpu().numpy()
        threshold = float(np.percentile(normal_errors, 95))

        # Test
        model.load_state_dict(torch.load(f"{MODEL_DIR}/autoencoder_best.pt",
                                          map_location=device))
        model.eval()
        with torch.no_grad():
            test_errors = model.reconstruction_error(
                to_conv(X_test).to(device)
            ).cpu().numpy()
        test_auc = roc_auc_score(y_test, test_errors)

        mlflow.log_metrics({"test_auc": test_auc, "threshold": threshold})
        mlflow.pytorch.log_model(model, "autoencoder_model",
                                  registered_model_name="CNC_AnomalyAE")

        # Save threshold
        with open(f"{MODEL_DIR}/ae_threshold.json", "w") as f:
            json.dump({"threshold": threshold}, f)

        logger.success(f"  Autoencoder: test AUC={test_auc:.4f} | "
                       f"threshold={threshold:.6f}")
        return test_auc, threshold


# ══════════════════════════════════════════════════════════════
# 3. RANDOM FOREST — WRONG PARAMETER DETECTION
# ══════════════════════════════════════════════════════════════

def train_random_forest():
    logger.info("── Training Random Forest wrong-param classifier ────")

    X_train = np.load(f"{DATA_DIR}/X_train.npy")
    y_train = np.load(f"{DATA_DIR}/y_wrongparam_train.npy")
    X_val   = np.load(f"{DATA_DIR}/X_val.npy")
    y_val   = np.load(f"{DATA_DIR}/y_wrongparam_val.npy")
    X_test  = np.load(f"{DATA_DIR}/X_test.npy")
    y_test  = np.load(f"{DATA_DIR}/y_wrongparam_test.npy")

    mlflow.set_experiment("cnc-wrong-param-detection")
    with mlflow.start_run(run_name="random-forest"):
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=4,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
        model.fit(X_train, y_train)

        val_proba  = model.predict_proba(X_val)[:, 1]
        val_auc    = roc_auc_score(y_val, val_proba)
        test_proba = model.predict_proba(X_test)[:, 1]
        test_preds = model.predict(X_test)
        test_auc   = roc_auc_score(y_test, test_proba)
        report     = classification_report(y_test, test_preds, output_dict=True)

        mlflow.log_params({"model": "RandomForest", "n_estimators": 200,
                            "max_depth": 10})
        mlflow.log_metrics({"val_auc": val_auc, "test_auc": test_auc,
                             "test_f1": report.get("1", {}).get("f1-score", 0)})
        mlflow.sklearn.log_model(model, "rf_model",
                                  registered_model_name="CNC_WrongParamRF")

        joblib.dump(model, f"{MODEL_DIR}/rf_classifier.pkl")
        logger.success(f"  Random Forest: test AUC={test_auc:.4f} | "
                       f"F1={report.get('1',{}).get('f1-score',0):.4f}")
        return test_auc


# ══════════════════════════════════════════════════════════════
# 4. GRADIENT BOOSTING — RUL REGRESSION
# ══════════════════════════════════════════════════════════════

def train_gbr():
    logger.info("── Training GBR remaining useful life regressor ─────")

    X_train = np.load(f"{DATA_DIR}/X_train.npy")
    y_train = np.load(f"{DATA_DIR}/y_rul_train.npy")
    X_val   = np.load(f"{DATA_DIR}/X_val.npy")
    y_val   = np.load(f"{DATA_DIR}/y_rul_val.npy")
    X_test  = np.load(f"{DATA_DIR}/X_test.npy")
    y_test  = np.load(f"{DATA_DIR}/y_rul_test.npy")

    mlflow.set_experiment("cnc-rul-regression")
    with mlflow.start_run(run_name="gradient-boosting-rul"):
        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
        model.fit(X_train, y_train)

        val_preds  = model.predict(X_val).clip(0, 240)
        val_mae    = mean_absolute_error(y_val, val_preds)
        test_preds = model.predict(X_test).clip(0, 240)
        test_mae   = mean_absolute_error(y_test, test_preds)
        test_r2    = r2_score(y_test, test_preds)

        mlflow.log_params({"model": "GradientBoosting", "n_estimators": 200,
                            "learning_rate": 0.05})
        mlflow.log_metrics({"val_mae": val_mae, "test_mae": test_mae,
                             "test_r2": test_r2})
        mlflow.sklearn.log_model(model, "gbr_model",
                                  registered_model_name="CNC_RUL_GBR")

        joblib.dump(model, f"{MODEL_DIR}/gbr_rul.pkl")
        logger.success(f"  GBR: test MAE={test_mae:.2f} min | R²={test_r2:.4f}")
        return test_mae, test_r2


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 55)
    logger.info(" CNC Predictive Maintenance — Training All Models")
    logger.info(f" Device: {device}")
    logger.info("=" * 55)

    with open(f"{DATA_DIR}/config.json") as f:
        config = json.load(f)

    lstm_auc            = train_lstm()
    ae_auc, ae_thresh   = train_autoencoder()
    rf_auc              = train_random_forest()
    gbr_mae, gbr_r2     = train_gbr()

    # Save combined metadata for the API to load
    meta = {
        "n_features": config["n_features"],
        "feature_names": config["feature_names"],
        "operator_ranges": config["operator_ranges"],
        "failure_solutions": config["failure_solutions"],
        "models": {
            "lstm":        {"test_auc": lstm_auc,  "threshold": 0.5},
            "autoencoder": {"test_auc": ae_auc,    "threshold": ae_thresh},
            "rf":          {"test_auc": rf_auc,    "threshold": 0.5},
            "gbr":         {"test_mae_min": gbr_mae, "test_r2": gbr_r2},
        }
    }
    with open(f"{MODEL_DIR}/model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.success("=" * 55)
    logger.success(" All models trained!")
    logger.success(f"  LSTM AUC         : {lstm_auc:.4f}")
    logger.success(f"  Autoencoder AUC  : {ae_auc:.4f}")
    logger.success(f"  Random Forest AUC: {rf_auc:.4f}")
    logger.success(f"  GBR MAE          : {gbr_mae:.2f} min")
    logger.success(" Next: python api/main.py")
    logger.success("=" * 55)


if __name__ == "__main__":
    main()
