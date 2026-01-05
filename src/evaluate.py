"""Evaluation utilities for trained models."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics

from .model import TrainedModel



METRIC_COLUMNS = ["precision", "recall", "f1", "accuracy", "roc_auc"]


def evaluate_model(model: TrainedModel, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    probs = model.pipeline.predict_proba(X)[:, 1]
    preds = (probs >= model.threshold).astype(int)
    return {
        "precision": metrics.precision_score(y, preds),
        "recall": metrics.recall_score(y, preds),
        "f1": metrics.f1_score(y, preds),
        "accuracy": metrics.accuracy_score(y, preds),
        "roc_auc": metrics.roc_auc_score(y, probs),
    }


def save_confusion_matrix(model: TrainedModel, X: pd.DataFrame, y: pd.Series, path: Path) -> Path:
    probs = model.pipeline.predict_proba(X)[:, 1]
    preds = (probs >= model.threshold).astype(int)
    cm = metrics.confusion_matrix(y, preds)

    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix - {model.name}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def save_roc_curve(model: TrainedModel, X: pd.DataFrame, y: pd.Series, path: Path) -> Path:
    probs = model.pipeline.predict_proba(X)[:, 1]
    fpr, tpr, _ = metrics.roc_curve(y, probs)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"ROC AUC = {metrics.roc_auc_score(y, probs):.2f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.set_title(f"ROC Curve - {model.name}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def save_precision_recall_curve(model: TrainedModel, X: pd.DataFrame, y: pd.Series, path: Path) -> Path:
    probs = model.pipeline.predict_proba(X)[:, 1]
    precision, recall, _ = metrics.precision_recall_curve(y, probs)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve - {model.name}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def summarize_metrics(metrics_dict: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    rows = []
    for name, metric in metrics_dict.items():
        row = {"model": name}
        row.update({k: metric.get(k) for k in METRIC_COLUMNS})
        rows.append(row)
    return pd.DataFrame(rows)
