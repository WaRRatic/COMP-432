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
    precision_recall_fscore_support,
    recall_score,
)

def multiclass_brier_score(
    y_true: pd.Series,
    probabilities: np.ndarray,
    labels: list[str],
) -> float:
    """Compute a multiclass Brier score from predicted probabilities.
    Args:
        y_true: True class labels as a pandas Series.
        probabilities: Predicted class probabilities with shape `(n_samples, n_classes)`.
        labels: Class labels in the same order as the probability columns.
    Returns:
        A float with the average multiclass Brier score. Lower is better.
    Example:
        >>> multiclass_brier_score(pd.Series(["A", "B"]), np.array([[0.8, 0.2], [0.3, 0.7]]), ["A", "B"])
    """
    label_to_index = {label: index for index, label in enumerate(labels)}
    truth = np.zeros((len(y_true), len(labels)), dtype=float)
    for row_index, label in enumerate(y_true):
        truth[row_index, label_to_index[label]] = 1.0
    return float(np.mean(np.sum((probabilities - truth) ** 2, axis=1)))

def classification_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    probabilities: np.ndarray | None = None,
    labels: list[str] | None = None,
) -> dict[str, float]:
    """Compute the metrics used to compare classification baselines.
    Args:
        y_true: True class labels.
        y_pred: Predicted class labels.
        probabilities: Optional predicted probabilities for each class.
        labels: Optional class labels matching the probability columns.
    Returns:
        A dictionary with accuracy, macro F1, macro precision, macro recall,
        and optionally multiclass Brier score.
    Example:
        >>> classification_metrics(pd.Series(["A", "B"]), np.array(["A", "B"]))
    """
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
    if probabilities is not None and labels is not None:
        # Add one probability-quality summary without changing model selection flow.
        metrics["multiclass_brier"] = multiclass_brier_score(y_true, probabilities, labels)
    return metrics

def regression_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute the metrics used to compare regression baselines
    Args:
        y_true: True regression targets.
        y_pred: Predicted regression targets.
    Returns:
        A dictionary with MAE and RMSE.
    Example:
        >>> regression_metrics(pd.Series([1.0, 2.0]), np.array([1.5, 2.5]))
    """
    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        # RMSE penalizes larger misses more than MAE.
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }
    return metrics
# confidence versus observed accuracy summary table for pred prob
def confidence_reliability_table(
    y_true: pd.Series,
    probabilities: np.ndarray,
    labels: list[str],
    bins: int = 10,
) -> pd.DataFrame:
    """Build a small confidence-versus-accuracy summary table.
    Args:
        y_true: True class labels.
        probabilities: Predicted class probabilities.
        labels: Class labels matching the probability columns.
        bins: Number of confidence bins to summarize.
    Returns:
        A pandas DataFrame with one row per non-empty confidence bin.
    Example:
        >>> confidence_reliability_table(pd.Series(["A"]), np.array([[0.8, 0.2]]), ["A", "B"])
    """
    confidence = probabilities.max(axis=1)
    predictions = np.asarray(labels)[probabilities.argmax(axis=1)]
    correct = (predictions == y_true.to_numpy()).astype(float)
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.clip(np.digitize(confidence, bin_edges, right=True) - 1, 0, bins - 1)
    rows: list[dict[str, float | int]] = []
    for bin_index in range(bins):
        mask = bin_ids == bin_index
        if not mask.any():
            continue
        rows.append(
            {
                "bin": int(bin_index),
                "avg_confidence": float(confidence[mask].mean()),
                "accuracy": float(correct[mask].mean()),
                "count": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)

def classification_report_table(
    y_true: pd.Series,
    y_pred: np.ndarray,
    labels: list[str],
) -> pd.DataFrame:
    """Build a small per-class precision, recall, and F1 summary table."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    rows: list[dict[str, float | int | str]] = []
    for index, label in enumerate(labels):
        rows.append(
            {
                "label": label,
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
        )
    return pd.DataFrame(rows)

def save_json(payload: dict[str, object], output_path: Path) -> None:
    """Write a JSON payload to disk for a training run.
    Args:
        payload: Dictionary to serialize as JSON.
        output_path: Destination path for the JSON file.
    Returns:
        None. The function writes the file to disk.
    Example:
        >>> save_json({"ok": True}, Path("summary.json"))
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # summary save 
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
