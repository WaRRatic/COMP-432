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
    classification_metrics,
    save_json,
)

# The list of baseline classifiers to compare in training
from montreal311_project.modeling import build_classification_models

# Central project paths for default input/output locations
from montreal311_project.paths import OUTPUTS_DIR, PROCESSED_DATA_DIR

# Time-based split helper for chronol. order
from montreal311_project.splits import split_by_time

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and compare classification baselines.")

    # Path to the representative subset
    parser.add_argument(
        "--input",
        type=Path,
        default=PROCESSED_DATA_DIR / "requetes311_2019_2021_sample_300k.csv.gz",
        help="Path to the representative subset CSV.gz.",
    )
    # Directory for comparison table/summary/best model
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS_DIR / "classification",
        help="Directory for metrics and saved model artifacts.",
    )
    # stop training time window
    parser.add_argument("--train-end", default="2020-12-31 23:59:59")
    # stop validation time window
    parser.add_argument("--validation-end", default="2021-06-30 23:59:59")
    # row limit for testing
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row cap for a quick smoke test.",
    )
    return parser.parse_args()

def main() -> None:
    # Reading cml options
    args = parse_args()
    # Fail if expected subset file dne
    if not args.input.exists():
        raise FileNotFoundError(f"Input subset not found: {args.input}")

    # Loading sampled requests file into a dataframe
    frame = load_requests(args.input, nrows=args.max_rows)
    # only rows valid for classification
    prepared = prepare_classification_frame(frame)

    # Cronolog split into trai/validation/test windows
    train, validation, test = split_by_time(
        prepared,
        train_end=args.train_end,
        validation_end=args.validation_end,
    )

    # Stop if time window has no rows
    if train.empty or validation.empty or test.empty:
        raise ValueError("One of the chronological splits is empty. Adjust the dates or the input sample.")
    # at least two targets for classification
    if train["NATURE_TARGET"].nunique() < 2:
        raise ValueError("Training data must contain at least two classes for classification.")
    # ordered label list for class order metrics
    labels = sorted(train["NATURE_TARGET"].unique().tolist())
    # column as model feature (except nature)
    feature_columns = [column for column in prepared.columns if column != "NATURE_TARGET"]
    # one result row per model
    results: list[dict[str, object]] = []
    best_name = None
    # Start below F1 - first successful model wins
    best_macro_f1 = float("-inf")

    # Loops over each baseline classifier
    for spec in build_classification_models():
        # Build a fresh estimator for the run
        estimator = spec.build_estimator()
        try:
            # model on training split only
            estimator.fit(train[feature_columns], train["NATURE_TARGET"])
            # Predicting labels on the validation split
            validation_pred = estimator.predict(validation[feature_columns])

            # validation metrics for model comparison
            validation_metrics = classification_metrics(
                validation["NATURE_TARGET"],
                validation_pred,
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

    # Convert all per-model result rows into a dataframe for easier sorting and saving
    comparison = pd.DataFrame(results)

    # Sort by validation macro-F1 when tpossible
    if "validation_macro_f1" in comparison.columns:
        comparison = comparison.sort_values("validation_macro_f1", ascending=False)
    # output directory check
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Saving full model comparison table
    comparison.to_csv(args.output_dir / "model_comparison.csv", index=False)
    # if all fail:
    if best_name is None:
        raise RuntimeError("No classification model trained successfully. Check model_comparison.csv for details.")

    # Rebuilding winning model
    best_spec = None
    for spec in build_classification_models():
        if spec.name == best_name:
            best_spec = spec
            break
    if best_spec is None:
        raise RuntimeError("Best model was selected by name but could not be rebuilt.")

    best_estimator = best_spec.build_estimator()
    # Combining train and validation after best model selected
    train_validation = pd.concat([train, validation], ignore_index=True)
    # fitting the winning model on pre-test data
    best_estimator.fit(train_validation[feature_columns], train_validation["NATURE_TARGET"])
    final_labels = sorted(train_validation["NATURE_TARGET"].unique().tolist())

    # Final predictions on test split
    final_test_pred = best_estimator.predict(test[feature_columns])

    # Final probabilities if supportes
    if hasattr(best_estimator, "predict_proba"):
        final_test_proba = best_estimator.predict_proba(test[feature_columns])
    else:
        final_test_proba = None

    # Computing final test metrics for selected model
    final_test_metrics = classification_metrics(
        test["NATURE_TARGET"],
        final_test_pred,
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

    # Saving summary as JSON
    save_json(summary, args.output_dir / "summary.json")

    # Saving fitted best estimator
    joblib.dump(best_estimator, args.output_dir / "best_model.joblib")

    print(f"Saved classification results to {args.output_dir}")

if __name__ == "__main__":
    main()