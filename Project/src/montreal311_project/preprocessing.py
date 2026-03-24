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
# They are all available when the request is created, so they do not leak future information.
SPARSE_CATEGORICAL_COLUMNS = [
    "TYPE_LIEU_INTERV",
    "ARRONDISSEMENT",
    "ARRONDISSEMENT_GEO",
    "UNITE_RESP_PARENT",
    "PROVENANCE_ORIGINALE",
]

def extract_text_feature(values: pd.DataFrame | pd.Series | np.ndarray) -> np.ndarray:
    # TfidfVectorizer expects a simple 1D sequence of strings.
    if isinstance(values, pd.DataFrame):
        series = values.iloc[:, 0]
    elif isinstance(values, pd.Series):
        series = values
    else:
        # ColumnTransformer can also pass numpy arrays depending on the pipeline step.
        array = np.asarray(values)
        if array.ndim == 2:
            array = array[:, 0]
        series = pd.Series(array)

    # Replace missing text with empty strings before vectorization.
    return series.fillna("").astype(str).to_numpy()

def build_sparse_preprocessor() -> ColumnTransformer:
    """Build the first text-and-category feature view for classification."""

    # Convert the activity name into TF-IDF text features.
    text_pipeline = Pipeline(
        steps=[
            ("extract_text", FunctionTransformer(extract_text_feature, validate=False)),
            (
                "tfidf",
                TfidfVectorizer(
                    # Keep the vocabulary size manageable for a first baseline.
                    max_features=4_000,
                    # Ignore very rare terms that are unlikely to help the first model.
                    min_df=5,
                    # Use single words and short phrases.
                    ngram_range=(1, 2),
                ),
            ),
        ]
    )

    # Fill missing category values, then one-hot encode them for the classifier.
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            (
                "one_hot",
                OneHotEncoder(
                    # Ignore category values that only appear later at validation/test time.
                    handle_unknown="ignore",
                    # Group very rare categories together to keep the feature space smaller.
                    min_frequency=20,
                ),
            ),
        ]
    )

    # Combine the text features and categorical features into one model input matrix.
    return ColumnTransformer(
        transformers=[
            ("text", text_pipeline, ["ACTI_NOM"]),
            ("categorical", categorical_pipeline, SPARSE_CATEGORICAL_COLUMNS),
        ],
        sparse_threshold=0.3,
    )
