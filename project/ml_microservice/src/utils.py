from __future__ import annotations
import json, os
from dataclasses import dataclass
from typing import Dict, Any, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, f1_score,
    precision_score, recall_score, RocCurveDisplay, PrecisionRecallDisplay,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

@dataclass
class Config:
    data: Dict[str, Any]
    features: Dict[str, Any]
    model: Dict[str, Any]
    cv: Dict[str, Any]
    outputs: Dict[str, Any]

def load_config(path: str) -> Config:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("outputs", {"artifacts_dir": "artifacts"})
    return Config(**cfg)

def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df

def make_split(df: pd.DataFrame, target: str, test_size: float, random_state: int):
    X = df.drop(columns=[target]); y = df[target]
    stratify = y if y.nunique() == 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    return X_train, X_test, y_train, y_test, X_train.index.values, X_test.index.values

def cv_strategy_from_config(cv_cfg: Dict[str, Any]):
    name = cv_cfg.get("strategy", "StratifiedKFold")
    n_splits = int(cv_cfg.get("n_splits", 5))
    shuffle = bool(cv_cfg.get("shuffle", True))
    random_state = cv_cfg.get("random_state", 42)
    if name == "StratifiedKFold":
        return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    if name == "KFold":
        return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    raise ValueError(f"Unsupported CV strategy: {name}")

def ensure_dir(path: str): os.makedirs(path, exist_ok=True)

def save_split_indices(artifacts_dir: str, train_idx, test_idx) -> str:
    ensure_dir(artifacts_dir)
    split_path = os.path.join(artifacts_dir, "split.json")
    with open(split_path, "w") as f:
        json.dump({"train_idx": list(map(int, train_idx)), "test_idx": list(map(int, test_idx))}, f)
    return split_path

def load_split_indices(path: str):
    with open(path, "r") as f:
        data = json.load(f)
    return np.array(data["train_idx"]), np.array(data["test_idx"])

def compute_metrics_binary(y_true, y_proba, y_pred) -> Dict[str, float]:
    metrics = {}
    if y_proba is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        metrics["avg_precision"] = float(average_precision_score(y_true, y_proba))
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    return metrics

def plot_and_save_roc(y_true, y_proba, path: str) -> Optional[str]:
    if y_proba is None: return None
    fig = plt.figure(); RocCurveDisplay.from_predictions(y_true, y_proba)
    plt.title("ROC Curve"); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path

def plot_and_save_pr(y_true, y_proba, path: str) -> Optional[str]:
    if y_proba is None: return None
    fig = plt.figure(); PrecisionRecallDisplay.from_predictions(y_true, y_proba)
    plt.title("Precision-Recall Curve"); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path

def plot_and_save_confusion(y_true, y_pred, path: str) -> str:
    fig = plt.figure(); ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.title("Confusion Matrix"); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path
