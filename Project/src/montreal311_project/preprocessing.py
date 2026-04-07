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

def build_sparse_preprocessor() -> ColumnTransformer:
    """Build the current sparse classification feature view."""

    # Convert the activity name into TF-IDF text features for learning
    text_pipeline = Pipeline(
        steps=[
            ("extract_text", FunctionTransformer(extract_text_feature, validate=False)),
            (
                "tfidf",
                TfidfVectorizer(
                    # ocabulary size manageable for first baseline
                    max_features=4_000,
                    # ignoring rare words that don't generalize well
                    min_df=5,
                    # Using single words and short phrases
                    ngram_range=(1, 2),
                ),
            ),
        ]
    )

    # Filling missing category values, one-hot encoding for classifier
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            (
                "one_hot",
                OneHotEncoder(
                    # Ignore category values from the rest of the data
                    handle_unknown="ignore",
                    # Grouping very rare categories together to keep the feature space smaller
                    min_frequency=20,
                ),
            ),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    # Combine text, categorical, and simple creation-time numeric features
    return ColumnTransformer(
        transformers=[
            ("text", text_pipeline, ["ACTI_NOM"]),
            ("categorical", categorical_pipeline, SPARSE_CATEGORICAL_COLUMNS),
            ("numeric", numeric_pipeline, NUMERIC_BASELINE_COLUMNS),
        ],
        sparse_threshold=0.3,
    )

def build_regression_preprocessor() -> ColumnTransformer:
    """Build the first regression feature view from creation-time columns."""
    # Reuse the request text as TF-IDF features for the reg baseline
    text_pipeline = Pipeline(
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

    # One-hot encode categorical fields 
    categorical_pipeline = Pipeline(
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

    # Filling missing numeric creation-time values before regression
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    # Combine text/categorical/numeric features into one model input matrix
    return ColumnTransformer(
        transformers=[
            ("text", text_pipeline, ["ACTI_NOM"]),
            ("categorical", categorical_pipeline, SPARSE_CATEGORICAL_COLUMNS),
            ("numeric", numeric_pipeline, NUMERIC_BASELINE_COLUMNS),
        ],
        sparse_threshold=0.3,
    )