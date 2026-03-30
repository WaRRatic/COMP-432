from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)

def classification_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Return the core metrics used to compare classification baselines."""
    # Evaluation for first classification baseline
    metrics = {
        # Accuracy for correct predictions
        "accuracy": float(accuracy_score(y_true, y_pred)),
        # Macro F1 to give equal % to each class 
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        # Macro precision 
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        # Macro recall 
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    return metrics

def regression_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Return the core metrics used to compare regression baselines."""
    metrics = {
        # Average absolute prediction error in days
        "mae": float(mean_absolute_error(y_true, y_pred)),
        # Root mean squared error penalizes larger mistakes more strongly
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }
    return metrics

def save_json(payload: dict[str, object], output_path: Path) -> None:
    """Write a small JSON summary file for the training run."""
    # Output folder
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # summary save 
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
