from __future__ import annotations
import argparse
import sys
from pathlib import Path
import joblib
import pandas as pd

# path to script
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
# Add Project/src to Python's import path to import
# montreal311_project.* when run directly from the command line
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Data-loading and target-preparation helpers
from montreal311_project.data import load_requests, prepare_classification_frame

# Evaluation helpers to score models and save outputs
from montreal311_project.evaluation import (
    classification_report_table,
    classification_metrics,
    confusion_matrix_table,
    confidence_reliability_table,
    grouped_classification_metrics_table,
    linear_feature_contribution_table,
    save_json,
)

# The list of baseline classifiers to compare in training
from montreal311_project.modeling import build_classification_models

# Central project paths for default input/output locations
from montreal311_project.paths import OUTPUTS_DIR, PROCESSED_DATA_DIR

# Time-based split helper for chronol. order
from montreal311_project.splits import split_by_time

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for classification training.
    Args:
        None.
    Returns:
        argparse.Namespace: Parsed options for input data, output directory,
        split dates, and optional row limit.
    Example:
        >>> args = parse_args()
    """
    parser = argparse.ArgumentParser(description="Train and compare classification baselines.")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROCESSED_DATA_DIR / "requetes311_2019_2021_sample_300k.csv.gz",
        help="Path to the representative subset CSV.gz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS_DIR / "classification",
        help="Directory for metrics and saved model artifacts.",
    )
    parser.add_argument("--train-end", default="2020-12-31 23:59:59")
    parser.add_argument("--validation-end", default="2021-06-30 23:59:59")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row cap for a quick smoke test.",
    )
    return parser.parse_args()

def main() -> None:
    """Train, compare, and save the current classification baselines.
    Args:
        None.
    Returns:
        None. The function writes model comparison outputs, a JSON summary,
        an optional reliability table, and the best fitted model to disk.
    Example:
        >>> main()
    """
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input subset not found: {args.input}")

    frame = load_requests(args.input, nrows=args.max_rows)
    prepared = prepare_classification_frame(frame)

    train, validation, test = split_by_time(
        prepared,
        train_end=args.train_end,
        validation_end=args.validation_end,
    )

    if train.empty or validation.empty or test.empty:
        raise ValueError("One of the chronological splits is empty. Adjust the dates or the input sample.")
    if train["NATURE_TARGET"].nunique() < 2:
        raise ValueError("Training data must contain at least two classes for classification.")
    labels = sorted(train["NATURE_TARGET"].unique().tolist())
    feature_columns = [column for column in prepared.columns if column != "NATURE_TARGET"]
    results: list[dict[str, object]] = []
    best_name = None
    best_macro_f1 = float("-inf")

    for spec in build_classification_models():
        estimator = spec.build_estimator()
        try:
            estimator.fit(train[feature_columns], train["NATURE_TARGET"])
            validation_pred = estimator.predict(validation[feature_columns])
            if hasattr(estimator, "predict_proba"):
                # Reliability metrics (models that expose probabilities)
                validation_proba = estimator.predict_proba(validation[feature_columns])
            else:
                validation_proba = None

            validation_metrics = classification_metrics(
                validation["NATURE_TARGET"],
                validation_pred,
                probabilities=validation_proba,
                labels=labels,
            )

            # summary for current model saved
            result_row = {
                "model": spec.name,
                "feature_view": spec.feature_view,
                "error": "",
            }
            for key, value in validation_metrics.items():
                result_row[f"validation_{key}"] = value
            results.append(result_row)

            # choosing best validation macro-F1 score
            if validation_metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = validation_metrics["macro_f1"]
                best_name = spec.name
        except Exception as exc:
            error_row = {
                "model": spec.name,
                "feature_view": spec.feature_view,
                "error": str(exc),
            }
            results.append(error_row)

    comparison = pd.DataFrame(results)

    if "validation_macro_f1" in comparison.columns:
        comparison = comparison.sort_values("validation_macro_f1", ascending=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output_dir / "model_comparison.csv", index=False)
    if best_name is None:
        raise RuntimeError("No classification model trained successfully. Check model_comparison.csv for details.")

    best_spec = None
    for spec in build_classification_models():
        if spec.name == best_name:
            best_spec = spec
            break
    if best_spec is None:
        raise RuntimeError("Best model was selected by name but could not be rebuilt.")

    best_estimator = best_spec.build_estimator()
    train_validation = pd.concat([train, validation], ignore_index=True)
    best_estimator.fit(train_validation[feature_columns], train_validation["NATURE_TARGET"])
    final_labels = sorted(train_validation["NATURE_TARGET"].unique().tolist())

    final_test_pred = best_estimator.predict(test[feature_columns])

    if hasattr(best_estimator, "predict_proba"):
        final_test_proba = best_estimator.predict_proba(test[feature_columns])
    else:
        final_test_proba = None

    # Computing final test metrics for selected model
    final_test_metrics = classification_metrics(
        test["NATURE_TARGET"],
        final_test_pred,
        probabilities=final_test_proba,
        labels=final_labels,
    )
    # summary of the run
    summary: dict[str, object] = {
        "best_model": best_name,
        "labels": final_labels,
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "final_test_metrics": final_test_metrics,
    }

    save_json(summary, args.output_dir / "summary.json")

    class_report = classification_report_table(
        test["NATURE_TARGET"],
        final_test_pred,
        final_labels,
    )
    class_report.to_csv(args.output_dir / "best_model_class_report.csv", index=False)

    confusion_table = confusion_matrix_table(
        test["NATURE_TARGET"],
        final_test_pred,
        final_labels,
    )
    confusion_table.to_csv(args.output_dir / "best_model_confusion_matrix.csv", index=True)

    by_month = grouped_classification_metrics_table(
        test["creation_month"],
        test["NATURE_TARGET"],
        final_test_pred,
        final_labels,
    )
    by_month.to_csv(args.output_dir / "best_model_by_month.csv", index=False)

    by_borough = grouped_classification_metrics_table(
        test["ARRONDISSEMENT"],
        test["NATURE_TARGET"],
        final_test_pred,
        final_labels,
    )
    by_borough.to_csv(args.output_dir / "best_model_by_borough.csv", index=False)

    if final_test_proba is not None:
        # Saving a reliability artifact for the best classifier
        reliability = confidence_reliability_table(
            test["NATURE_TARGET"],
            final_test_proba,
            final_labels,
        )
        reliability.to_csv(args.output_dir / "best_model_reliability.csv", index=False)

    feature_contributions = linear_feature_contribution_table(best_estimator)
    if not feature_contributions.empty:
        feature_contributions.to_csv(
            args.output_dir / "best_model_feature_contributions.csv",
            index=False,
        )

    joblib.dump(best_estimator, args.output_dir / "best_model.joblib")

    print(f"Saved classification results to {args.output_dir}")

if __name__ == "__main__":
    main()