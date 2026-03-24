from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def classification_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Return the core metrics used to compare classification baselines."""
    # Evaluation for first classification baseline
    metrics = {
        # Accuracy measures the overall fraction of correct predictions.
        "accuracy": float(accuracy_score(y_true, y_pred)),
        # Macro F1 gives equal importance to each class, even rare ones.
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        # Macro precision shows how correct the predicted labels are on average across classes.
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        # Macro recall shows how well the model finds each class on average.
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    return metrics

def save_json(payload: dict[str, object], output_path: Path) -> None:
    """Write a small JSON summary file for the training run."""
    # Create the output folder if it does not exist yet.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Save the summary in JSON so it can be read later without rerunning training.
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
