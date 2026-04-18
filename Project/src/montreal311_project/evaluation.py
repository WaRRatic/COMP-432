"""Compute and save the evaluation outputs used in the project.

This module contains the metric helpers, summary tables, and small reporting
utilities used by the training scripts.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
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

def split_conformal_interval_radius(
    y_true: pd.Series,
    y_pred: np.ndarray,
    alpha: float = 0.1,
) -> float:
    """Estimate one split-conformal interval radius from calibration errors."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1.")
    errors = np.abs(y_true.to_numpy(dtype=float) - np.asarray(y_pred, dtype=float))
    if errors.size == 0:
        raise ValueError("Calibration errors are required to build conformal intervals.")
    sorted_errors = np.sort(errors)
    rank = int(np.ceil((sorted_errors.size + 1) * (1.0 - alpha))) - 1
    rank = min(max(rank, 0), sorted_errors.size - 1)
    return float(sorted_errors[rank])

def regression_interval_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    interval_radius: float,
) -> dict[str, float]:
    """Summarize empirical coverage and interval width for regression predictions."""
    predictions = np.asarray(y_pred, dtype=float)
    lower_bound = predictions - interval_radius
    upper_bound = predictions + interval_radius
    targets = y_true.to_numpy(dtype=float)
    covered = (targets >= lower_bound) & (targets <= upper_bound)
    interval_width = upper_bound - lower_bound
    return {
        "coverage": float(np.mean(covered)),
        "mean_interval_width": float(np.mean(interval_width)),
        "median_interval_width": float(np.median(interval_width)),
    }

def regression_interval_table(
    y_true: pd.Series,
    y_pred: np.ndarray,
    interval_radius: float,
) -> pd.DataFrame:
    """Build a small table with prediction intervals for regression outputs."""
    predictions = np.asarray(y_pred, dtype=float)
    lower_bound = predictions - interval_radius
    upper_bound = predictions + interval_radius
    targets = y_true.to_numpy(dtype=float)
    covered = (targets >= lower_bound) & (targets <= upper_bound)
    return pd.DataFrame(
        {
            "actual": targets,
            "prediction": predictions,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "covered": covered,
        }
    )

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

def confusion_matrix_table(
    y_true: pd.Series,
    y_pred: np.ndarray,
    labels: list[str],
) -> pd.DataFrame:
    """Build a labeled confusion-matrix table for classification results"""
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(matrix, index=labels, columns=labels)

def grouped_classification_metrics_table(
    groups: pd.Series,
    y_true: pd.Series,
    y_pred: np.ndarray,
    labels: list[str],
    min_count: int = 50,
) -> pd.DataFrame:
    """Build a small grouped accuracy and macro-F1 summary table."""
    grouped_frame = pd.DataFrame(
        {
            "group": groups,
            "y_true": y_true,
            "y_pred": y_pred,
        }
    ).dropna(subset=["group"])
    rows: list[dict[str, float | int | str]] = []
    for group_value, group_frame in grouped_frame.groupby("group"):
        if len(group_frame) < min_count:
            continue
        rows.append(
            {
                "group": str(group_value),
                "count": int(len(group_frame)),
                "accuracy": float(
                    accuracy_score(group_frame["y_true"], group_frame["y_pred"])
                ),
                "macro_f1": float(
                    f1_score(
                        group_frame["y_true"],
                        group_frame["y_pred"],
                        labels=labels,
                        average="macro",
                        zero_division=0,
                    )
                ),
            }
        )
    return pd.DataFrame(rows)

def linear_feature_contribution_table(
    estimator: object,
    top_n: int = 20,
) -> pd.DataFrame:
    """Build a small top-coefficient table for fitted linear models."""
    preprocessor, coefficients, groups = _extract_linear_model_parts(estimator)
    if preprocessor is None or coefficients is None:
        return pd.DataFrame()

    feature_names = _get_preprocessor_feature_names(preprocessor)
    if not feature_names:
        return pd.DataFrame()

    coefficient_matrix = np.asarray(coefficients, dtype=float)
    if coefficient_matrix.ndim == 1:
        coefficient_matrix = coefficient_matrix.reshape(1, -1)

    if coefficient_matrix.shape[1] != len(feature_names):
        return pd.DataFrame()

    if groups is None:
        groups = ["target"]

    rows: list[dict[str, float | int | str]] = []
    for group_name, coefficient_row in zip(groups, coefficient_matrix):
        ranked_indices = np.argsort(np.abs(coefficient_row))[::-1][:top_n]
        for rank, feature_index in enumerate(ranked_indices, start=1):
            coefficient = float(coefficient_row[feature_index])
            rows.append(
                {
                    "group": str(group_name),
                    "rank": int(rank),
                    "feature": feature_names[feature_index],
                    "coefficient": coefficient,
                    "abs_coefficient": float(abs(coefficient)),
                    "direction": "positive" if coefficient >= 0.0 else "negative",
                }
            )
    return pd.DataFrame(rows)

def _extract_linear_model_parts(
    estimator: object,
) -> tuple[ColumnTransformer | None, np.ndarray | None, list[str] | None]:
    """Return preprocessor, coefficients, and group labels for a fitted linear model."""
    if isinstance(estimator, Pipeline):
        preprocessor = estimator.named_steps.get("preprocessor")
        model = estimator.named_steps.get("model")
        coefficients = _extract_classifier_coefficients(model)
        groups = _extract_model_groups(model)
        if isinstance(preprocessor, ColumnTransformer) and coefficients is not None:
            return preprocessor, coefficients, groups

    if isinstance(estimator, TransformedTargetRegressor):
        regressor = getattr(estimator, "regressor_", None)
        if isinstance(regressor, Pipeline):
            preprocessor = regressor.named_steps.get("preprocessor")
            model = regressor.named_steps.get("model")
            coefficients = getattr(model, "coef_", None)
            if isinstance(preprocessor, ColumnTransformer) and coefficients is not None:
                return preprocessor, np.asarray(coefficients, dtype=float), ["target"]

    return None, None, None

def _extract_classifier_coefficients(model: object) -> np.ndarray | None:
    """Return linear classification coefficients when the fitted model exposes them."""
    if model is None:
        return None
    if hasattr(model, "coef_"):
        return np.asarray(model.coef_, dtype=float)
    if isinstance(model, CalibratedClassifierCV):
        coefficient_rows: list[np.ndarray] = []
        for calibrated in getattr(model, "calibrated_classifiers_", []):
            base_estimator = getattr(calibrated, "estimator", None)
            if base_estimator is not None and hasattr(base_estimator, "coef_"):
                coefficient_rows.append(np.asarray(base_estimator.coef_, dtype=float))
        if coefficient_rows:
            return np.mean(np.stack(coefficient_rows, axis=0), axis=0)
    return None

def _extract_model_groups(model: object) -> list[str] | None:
    """Return the group labels that match the coefficient rows."""
    if model is None:
        return None
    classes = getattr(model, "classes_", None)
    if classes is None:
        return None
    return [str(label) for label in classes]

def _get_preprocessor_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Return feature names from the current column-transformer pipelines."""
    feature_names: list[str] = []
    for transformer_name, transformer, columns in preprocessor.transformers_:
        if transformer == "drop":
            continue
        if transformer_name == "text":
            tfidf = transformer.named_steps["tfidf"]
            feature_names.extend(
                [f"{transformer_name}__{name}" for name in tfidf.get_feature_names_out()]
            )
            continue
        if transformer_name == "categorical":
            one_hot = transformer.named_steps["one_hot"]
            feature_names.extend(one_hot.get_feature_names_out(columns).tolist())
            continue
        if transformer_name == "numeric":
            feature_names.extend([str(column) for column in columns])
    return feature_names

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
