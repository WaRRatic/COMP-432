from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
from sklearn.dummy import DummyClassifier
# Simple regression reference model
from sklearn.dummy import DummyRegressor
# First linear regression baseline with regularization
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from .preprocessing import build_regression_preprocessor, build_sparse_preprocessor

@dataclass(frozen=True)
class ModelSpec:
    # Baseline model definition
    name: str
    feature_view: str
    build_estimator: Callable[[], object]

def build_classification_models() -> list[ModelSpec]:
    """Return the classification baselines for the second project commit."""
    return [
        ModelSpec(
            # reference point  for model performance (predicting the most common class)
            name="dummy_prior",
            feature_view="all_features_ignored",
            build_estimator=lambda: DummyClassifier(strategy="prior"),
        ),
        ModelSpec(
            # first baseline: preprocess text/categories, train logistic regression.
            name="logistic_sparse_combined",
            feature_view="text_categorical",
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
    ]

def build_regression_models() -> list[ModelSpec]:
    """Return the regression baselines for the next project commit."""
    return [
        ModelSpec(
            # Reference point: always predicts the training-set average
            name="dummy_mean",
            feature_view="all_features_ignored",
            build_estimator=lambda: DummyRegressor(strategy="mean"),
        ),
        ModelSpec(
            # First regression baseline using text/categories/numeric features
            name="ridge_sparse_combined",
            feature_view="text_categorical_numeric",
            build_estimator=lambda: Pipeline(
                steps=[
                    # Turninng raw dataframe columns into model-ready features
                    ("preprocessor", build_regression_preprocessor()),
                    (
                        "model",
                        # Ridge regression baseline
                        Ridge(alpha=1.0),
                    ),
                ]
            ),
        ),
    ]
