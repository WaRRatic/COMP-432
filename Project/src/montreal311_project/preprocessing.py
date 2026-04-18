"""Build the feature preprocessing steps used by the project models.

This module defines the text, categorical, and numeric preprocessing pipelines
used for the classification and regression baselines.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer

# main categorical fields for the first classification baseline
# (available at request creation not to leak future info)
SPARSE_CATEGORICAL_COLUMNS = [
    "TYPE_LIEU_INTERV",
    "ARRONDISSEMENT",
    "ARRONDISSEMENT_GEO",
    "UNITE_RESP_PARENT",
    "PROVENANCE_ORIGINALE",
]

# simple numeric features available at request creation time for both tasks
NUMERIC_BASELINE_COLUMNS = [
    "creation_year",
    "creation_month",
    "creation_dayofweek",
    "creation_hour",
    "has_geo",
]

def extract_text_feature(values: pd.DataFrame | pd.Series | np.ndarray) -> np.ndarray:
    """Return a 1D string array for text vectorization."""
    # normalizing into a text column and filling missing values with empty strings
    if isinstance(values, pd.DataFrame):
        series = values.iloc[:, 0]
    elif isinstance(values, pd.Series):
        series = values
    else:
        # converts to series if input is a numpt array 
        array = np.asarray(values)
        if array.ndim == 2:
            array = array[:, 0]
        series = pd.Series(array)

    # Replace missing text with empty strings before vectorization
    return series.fillna("").astype(str).to_numpy()

def build_text_pipeline() -> Pipeline:
    """Build the shared TF-IDF text pipeline used by linear baselines."""
    return Pipeline(
        steps=[
            ("extract_text", FunctionTransformer(extract_text_feature, validate=False)),
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=4_000,
                    min_df=5,
                    ngram_range=(1, 2),
                ),
            ),
        ]
    )

def build_categorical_pipeline() -> Pipeline:
    """Build the shared categorical pipeline for creation-time columns."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=20,
                ),
            ),
        ]
    )

def build_numeric_pipeline() -> Pipeline:
    """Build the shared numeric pipeline for simple creation-time features."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

def build_text_only_preprocessor() -> ColumnTransformer:
    """Build a text-only classification feature view from `ACTI_NOM`."""
    return ColumnTransformer(
        transformers=[
            ("text", build_text_pipeline(), ["ACTI_NOM"]),
        ],
        sparse_threshold=0.3,
    )

def build_tabular_preprocessor() -> ColumnTransformer:
    """Build a tabular-only feature view from creation-time columns."""
    return ColumnTransformer(
        transformers=[
            ("categorical", build_categorical_pipeline(), SPARSE_CATEGORICAL_COLUMNS),
            ("numeric", build_numeric_pipeline(), NUMERIC_BASELINE_COLUMNS),
        ],
        sparse_threshold=0.3,
    )

def build_sparse_preprocessor() -> ColumnTransformer:
    """Build the current sparse classification feature view."""

    # Combine text, categorical, and simple creation-time numeric features
    return ColumnTransformer(
        transformers=[
            ("text", build_text_pipeline(), ["ACTI_NOM"]),
            ("categorical", build_categorical_pipeline(), SPARSE_CATEGORICAL_COLUMNS),
            ("numeric", build_numeric_pipeline(), NUMERIC_BASELINE_COLUMNS),
        ],
        sparse_threshold=0.3,
    )

def build_regression_preprocessor() -> ColumnTransformer:
    """Build the first regression feature view from creation-time columns."""
    # Combine text/categorical/numeric features into one model input matrix
    return ColumnTransformer(
        transformers=[
            ("text", build_text_pipeline(), ["ACTI_NOM"]),
            ("categorical", build_categorical_pipeline(), SPARSE_CATEGORICAL_COLUMNS),
            ("numeric", build_numeric_pipeline(), NUMERIC_BASELINE_COLUMNS),
        ],
        sparse_threshold=0.3,
    )
