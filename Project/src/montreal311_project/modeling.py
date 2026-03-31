from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
from sklearn.dummy import DummyClassifier
# Simple regression reference model
from sklearn.dummy import DummyRegressor
from sklearn.compose import TransformedTargetRegressor
# First linear regression baselines with regularization
from sklearn.linear_model import ElasticNet
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
    """Return the regression baselines for the fourth project commit."""
    return [
        ModelSpec(
            # Reference point: always predicts the training-set average
            name="dummy_mean",
            feature_view="all_features_ignored",
            build_estimator=lambda: DummyRegressor(strategy="mean"),
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
