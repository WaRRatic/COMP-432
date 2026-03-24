from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .preprocessing import build_sparse_preprocessor


@dataclass(frozen=True)
class ModelSpec:
    # Small container used to keep each baseline model definition together.
    name: str
    feature_view: str
    build_estimator: Callable[[], object]


def build_classification_models() -> list[ModelSpec]:
    """Return the classification baselines for the second project commit."""

    return [
        ModelSpec(
            # DummyClassifier gives a simple reference point by always following class frequencies.
            name="dummy_prior",
            feature_view="all_features_ignored",
            build_estimator=lambda: DummyClassifier(strategy="prior"),
        ),
        ModelSpec(
            # This is the first real baseline: preprocess the text/categories, then train logistic regression.
            name="logistic_sparse_combined",
            feature_view="text_categorical",
            build_estimator=lambda: Pipeline(
                steps=[
                    # Turn the raw dataframe columns into model-ready features.
                    ("preprocessor", build_sparse_preprocessor()),
                    (
                        "model",
                        # Logistic regression is a common baseline for text classification.
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
