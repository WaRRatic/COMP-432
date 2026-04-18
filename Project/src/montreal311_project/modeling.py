"""Define the baseline models used in the Montreal 311 project.

This module contains the model specifications and helper builders for the
classification and regression experiments.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
# Simple regression reference model
from sklearn.dummy import DummyRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.calibration import CalibratedClassifierCV
# First linear regression baselines with regularization
from sklearn.linear_model import ElasticNet
# First linear regression baseline with regularization
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from .preprocessing import (
    build_regression_preprocessor,
    build_sparse_preprocessor,
    build_tabular_preprocessor,
    build_text_only_preprocessor,
)

@dataclass(frozen=True)
class ModelSpec:
    """Store the metadata needed to rebuild one baseline model for training and evaluation."""
    name: str
    feature_view: str
    build_estimator: Callable[[], object]

class ConditionalMedianRegressor(BaseEstimator, RegressorMixin):
    """Predict resolution time from simple training-set median lookups."""

    def __init__(self) -> None:
        """Initialize the grouped-median regression baseline."""
        self.global_median_: float | None = None
        self.by_activity_: dict[str, float] = {}
        self.by_borough_: dict[str, float] = {}
        self.by_activity_borough_: dict[tuple[str, str], float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ConditionalMedianRegressor":
        """Fit grouped medians from training rows only."""
        frame = self._build_lookup_frame(X, y)
        self.global_median_ = float(frame["target"].median())
        self.by_activity_ = frame.groupby("ACTI_NOM")["target"].median().to_dict()
        self.by_borough_ = frame.groupby("ARRONDISSEMENT")["target"].median().to_dict()
        self.by_activity_borough_ = (
            frame.groupby(["ACTI_NOM", "ARRONDISSEMENT"])["target"].median().to_dict()
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict with hierarchical fallback from specific groups to the global median."""
        if self.global_median_ is None:
            raise ValueError("The conditional median baseline must be fitted before prediction.")
        frame = self._build_feature_frame(X)
        predictions: list[float] = []
        for row in frame.itertuples(index=False):
            activity_key = row.ACTI_NOM
            borough_key = row.ARRONDISSEMENT
            combined_key = (activity_key, borough_key)
            if combined_key in self.by_activity_borough_:
                predictions.append(float(self.by_activity_borough_[combined_key]))
            elif activity_key in self.by_activity_:
                predictions.append(float(self.by_activity_[activity_key]))
            elif borough_key in self.by_borough_:
                predictions.append(float(self.by_borough_[borough_key]))
            else:
                predictions.append(float(self.global_median_))
        return np.asarray(predictions, dtype=float)

    def _build_feature_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract the grouping columns used by the baseline from a feature frame."""
        frame = pd.DataFrame(index=X.index)
        if "ACTI_NOM" in X.columns:
            frame["ACTI_NOM"] = X["ACTI_NOM"].fillna("missing").astype(str)
        else:
            frame["ACTI_NOM"] = "missing"
        if "ARRONDISSEMENT" in X.columns:
            frame["ARRONDISSEMENT"] = X["ARRONDISSEMENT"].fillna("missing").astype(str)
        else:
            frame["ARRONDISSEMENT"] = "missing"
        return frame

    def _build_lookup_frame(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Build the training lookup table for grouped median prediction."""
        frame = self._build_feature_frame(X)
        frame["target"] = pd.Series(y, index=X.index).astype(float)
        return frame

def build_stabilized_regression_estimator(regressor: object) -> TransformedTargetRegressor:
    """Wrap a regression pipeline with a simple training-only log target transform."""
    return TransformedTargetRegressor(
        # same feature pipeline, but fitting the linear model on log1p(days)
        regressor=Pipeline(
            steps=[
                ("preprocessor", build_regression_preprocessor()),
                ("model", regressor),
            ]
        ),
        # Compressing very long resolution times so the linear baselines are more stable
        func=np.log1p,
        # Convert predictions back to the original unit (days)
        inverse_func=np.expm1,
    )

def build_calibrated_linear_svc_estimator() -> Pipeline:
    """Build a calibrated linear SVM classifier on the existing sparse feature view."""
    return Pipeline(
        steps=[
            ("preprocessor", build_sparse_preprocessor()),
            (
                "model",
                CalibratedClassifierCV(
                    estimator=LinearSVC(class_weight="balanced", random_state=42),
                    cv=3,
                ),
            ),
        ]
    )

def build_classification_models() -> list[ModelSpec]:
    """Return the classification baselines for the current comparison step."""
    return [
        ModelSpec(
            # reference point  for model performance (predicting the most common class)
            name="dummy_prior",
            feature_view="all_features_ignored",
            build_estimator=lambda: DummyClassifier(strategy="prior"),
        ),
        ModelSpec(
            name="logistic_text_only",
            feature_view="text_only",
            build_estimator=lambda: Pipeline(
                steps=[
                    ("preprocessor", build_text_only_preprocessor()),
                    (
                        "model",
                        LogisticRegression(
                            solver="saga",
                            max_iter=2_000,
                            class_weight="balanced",
                            random_state=42,
                        ),
                    ),
                ]
            ),
        ),
        ModelSpec(
            name="logistic_tabular_only",
            feature_view="categorical_numeric",
            build_estimator=lambda: Pipeline(
                steps=[
                    ("preprocessor", build_tabular_preprocessor()),
                    (
                        "model",
                        LogisticRegression(
                            solver="saga",
                            max_iter=2_000,
                            class_weight="balanced",
                            random_state=42,
                        ),
                    ),
                ]
            ),
        ),
        ModelSpec(
            name="logistic_sparse_combined",
            feature_view="text_categorical_numeric",
            build_estimator=lambda: Pipeline(
                steps=[
                    # raw dataframe columns into model-ready features
                    ("preprocessor", build_sparse_preprocessor()),
                    (
                        "model",
                        # Logistic regression baseline
                        LogisticRegression(
                            solver="saga",
                            max_iter=2_000,
                            class_weight="balanced",
                            random_state=42,
                        ),
                    ),
                ]
            ),
        ),
        ModelSpec(
            name="linear_svc_calibrated_sparse",
            feature_view="text_categorical_numeric",
            build_estimator=build_calibrated_linear_svc_estimator,
        ),
    ]

def build_regression_models() -> list[ModelSpec]:
    """Return the regression baselines for the current comparison step."""
    return [
        ModelSpec(
            # Reference point: always predicts the training-set average
            name="dummy_mean",
            feature_view="all_features_ignored",
            build_estimator=lambda: DummyRegressor(strategy="mean"),
        ),
        ModelSpec(
            name="conditional_median_lookup",
            feature_view="activity_borough_lookup",
            build_estimator=ConditionalMedianRegressor,
        ),
        ModelSpec(
            # First regression baseline using a stabilized target and linear features
            name="ridge_sparse_combined",
            feature_view="text_categorical_numeric",
            build_estimator=lambda: build_stabilized_regression_estimator(
                # ridge baseline is original, but training it on the stabilized target
                Ridge(alpha=1.0)
            ),
        ),
        ModelSpec(
            # Slightly stronger sparse linear baseline with l1/l2 regularization
            name="elasticnet_sparse_combined",
            feature_view="text_categorical_numeric",
            build_estimator=lambda: build_stabilized_regression_estimator(
                # Small regularized upgrade after ridge without changing the feature set
                ElasticNet(alpha=0.0005, l1_ratio=0.5, max_iter=5_000, random_state=42)
            ),
        ),
    ]
