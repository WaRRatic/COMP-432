"""Train and save the Montreal 311 regression baselines.

This script compares a small set of regression baselines on the prepared
Montreal 311 data, keeps the best model based on validation MAE, and saves
the main files used by the final report.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import joblib
import pandas as pd

# Script location helpers so the package (imported and run directly)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# data-preparation and training helpers
from montreal311_project.data import load_requests, prepare_regression_frame
from montreal311_project.evaluation import (
    linear_feature_contribution_table,
    regression_interval_metrics,
    regression_interval_table,
    regression_metrics,
    save_json,
    split_conformal_interval_radius,
)
from montreal311_project.modeling import build_regression_models
from montreal311_project.paths import OUTPUTS_DIR, PROCESSED_DATA_DIR
from montreal311_project.splits import split_by_time

def parse_args() -> argparse.Namespace:
    """Collect the settings used for a regression training run.

    The script reads the subset path, output folder, split boundaries, and the
    conformal interval setting from these arguments so the same run can be
    repeated later without guessing any configuration details.
    Returns:
        argparse.Namespace: Parsed input path, output path, split dates, and
        conformal interval settings.
    Example:
        >>> args = parse_args()
    """
    # Input settings for data, split dates, and output location
    parser = argparse.ArgumentParser(description="Train and compare regression baselines.")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROCESSED_DATA_DIR / "requetes311_2019_2021_sample_300k.csv.gz",
        help="Path to the subset CSV.gz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS_DIR / "regression",
        help="Directory for saved outputs.",
    )
    parser.add_argument("--train-end", default="2020-12-31 23:59:59")
    parser.add_argument("--validation-end", default="2021-06-30 23:59:59")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row limit for a quick test.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Miscoverage level for split-conformal prediction intervals.",
    )
    return parser.parse_args()

def main() -> None:
    """Run the full regression baseline workflow and save the outputs.

    The workflow loads the prepared data, applies the time-based split,
    compares models on the validation set, and saves the final test results.
    """
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input subset not found: {args.input}")

    # Load the subset and keep only rows usable for regression
    frame = load_requests(args.input, nrows=args.max_rows)
    prepared = prepare_regression_frame(frame)

    # Chronological split (learning from older requests and testing on newer)
    train, validation, test = split_by_time(
        prepared,
        train_end=args.train_end,
        validation_end=args.validation_end,
    )
    if train.empty or validation.empty or test.empty:
        raise ValueError("One of the chronological splits is empty. Adjust the dates or the input sample.")

    feature_columns = [column for column in prepared.columns if column != "resolution_time_days"]
    results: list[dict[str, object]] = []
    best_name = None
    best_validation_mae = float("inf")

    # Compare each baseline using validation MAE.
    for spec in build_regression_models():
        estimator = spec.build_estimator()
        estimator.fit(train[feature_columns], train["resolution_time_days"])

        validation_pred = estimator.predict(validation[feature_columns])
        validation_metrics = regression_metrics(
            validation["resolution_time_days"],
            validation_pred,
        )

        result_row = {
            "model": spec.name,
            "feature_view": spec.feature_view,
            "error": "",
        }
        for key, value in validation_metrics.items():
            result_row[f"validation_{key}"] = value
        results.append(result_row)

        if validation_metrics["mae"] < best_validation_mae:
            best_validation_mae = validation_metrics["mae"]
            best_name = spec.name

    comparison = pd.DataFrame(results)
    if "validation_mae" in comparison.columns:
        comparison = comparison.sort_values("validation_mae", ascending=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output_dir / "model_comparison.csv", index=False)

    if best_name is None:
        raise RuntimeError("No regression model trained successfully. Check model_comparison.csv for details.")

    best_spec = None
    for spec in build_regression_models():
        if spec.name == best_name:
            best_spec = spec
            break
    if best_spec is None:
        raise RuntimeError("Best model was selected by name but could not be rebuilt.")

    # Fit once on train for interval calibration, then refit on train and validation
    best_estimator = best_spec.build_estimator()
    calibration_estimator = best_spec.build_estimator()
    calibration_estimator.fit(train[feature_columns], train["resolution_time_days"])

    validation_pred = calibration_estimator.predict(validation[feature_columns])
    interval_radius = split_conformal_interval_radius(
        validation["resolution_time_days"],
        validation_pred,
        alpha=args.alpha,
    )
    conformal_test_pred = calibration_estimator.predict(test[feature_columns])
    conformal_metrics = regression_interval_metrics(
        test["resolution_time_days"],
        conformal_test_pred,
        interval_radius,
    )

    train_validation = pd.concat([train, validation], ignore_index=True)
    best_estimator.fit(train_validation[feature_columns], train_validation["resolution_time_days"])
    final_test_pred = best_estimator.predict(test[feature_columns])
    final_test_metrics = regression_metrics(test["resolution_time_days"], final_test_pred)

    summary: dict[str, object] = {
        "best_model": best_name,
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "final_test_metrics": final_test_metrics,
        "conformal_interval": {
            "alpha": args.alpha,
            "nominal_coverage": float(1.0 - args.alpha),
            "calibration_rows": int(len(validation)),
            "interval_radius_days": interval_radius,
            "test_metrics": conformal_metrics,
        },
    }

    save_json(summary, args.output_dir / "summary.json")
    regression_interval_table(
        test["resolution_time_days"],
        conformal_test_pred,
        interval_radius,
    ).to_csv(args.output_dir / "conformal_test_predictions.csv", index=False)

    # Save linear feature weights only when the chosen model exposes them.
    feature_contributions = linear_feature_contribution_table(best_estimator)
    if not feature_contributions.empty:
        feature_contributions.to_csv(
            args.output_dir / "best_model_feature_contributions.csv",
            index=False,
        )

    joblib.dump(best_estimator, args.output_dir / "best_model.joblib")

    print(f"Saved regression results to {args.output_dir}")

if __name__ == "__main__":
    main()