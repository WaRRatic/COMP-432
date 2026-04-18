"""Loading and cleaning the Montreal 311 data used by the project.

This module handles shared data for both classification and regression tasks.
"""

from __future__ import annotations
import unicodedata
from pathlib import Path
import numpy as np
import pandas as pd

# Source columns grouped by role
PROVENANCE_COLUMNS = [
    "PROVENANCE_TELEPHONE",
    "PROVENANCE_COURRIEL",
    "PROVENANCE_PERSONNE",
    "PROVENANCE_COURRIER",
    "PROVENANCE_TELECOPIEUR",
    "PROVENANCE_INSTANCE",
    "PROVENANCE_MOBILE",
    "PROVENANCE_MEDIASOCIAUX",
    "PROVENANCE_SITEINTERNET",
]

LOCATION_COLUMNS = ["LOC_LONG", "LOC_LAT", "LOC_X", "LOC_Y", "LOC_ERREUR_GDT"]

STRING_COLUMNS = [
    "NATURE",
    "ACTI_NOM",
    "TYPE_LIEU_INTERV",
    "RUE",
    "RUE_INTERSECTION1",
    "RUE_INTERSECTION2",
    "ARRONDISSEMENT",
    "ARRONDISSEMENT_GEO",
    "LIN_CODE_POSTAL",
    "PROVENANCE_ORIGINALE",
    "UNITE_RESP_PARENT",
    "DERNIER_STATUT",
]

# Normalizing targets
CANONICAL_NATURES = {
    "information": "Information",
    "commentaire": "Commentaire",
    "requete": "Requete",
    "plainte": "Plainte",
}

CANONICAL_CLOSED_STATUSES = {
    "termine",
    "refuse",
    "annule",
    "supprime",
}

RAW_CSV_ENCODING = "utf-16"

def _empty_object_series(index: pd.Index | None = None) -> pd.Series:
    return pd.Series(index=index, dtype="object")

def maybe_fix_mojibake(value: object) -> object:
    """Cleans text values and tries to repair simple encoding problems.

    This helper only handles string values, removes extra whitespace,
    and fixes common mojibake patterns that appear in the raw dataset.
    """
    if not isinstance(value, str):
        return value

    # Collapsing extra whitespace before attempting any encoding repair
    text = " ".join(value.strip().split())
    if not text:
        return ""
    # repairing mojibake from UTF-8 text decoded as latin1
    if any(token in text for token in ("Ã", "Â", "â", "�")):
        try:
            repaired = text.encode("latin1").decode("utf-8")
        except UnicodeError:
            repaired = text
        if repaired:
            text = repaired

    return text

def fold_text(value: object) -> str:
    """Normalizes a value into lowercase ASCII text.

    This used to make text labels more consistent before matching or
    cleaning them.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    # Normalizing to lowercase ASCII for label cleanup
    text = str(maybe_fix_mojibake(value))
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return " ".join(ascii_text.lower().split())

def canonicalize_nature(value: object) -> str | None:
    """Maps a raw nature label into the small target set used in the project."""
    # Mapping cleaned labels into controlled set used by classifier
    folded = fold_text(value)
    return CANONICAL_NATURES.get(folded)

def read_raw_csv_kwargs() -> dict[str, object]:
    """Return the shared pandas CSV options for the raw Montreal 311 file."""
    return {
        "low_memory": False,
        "encoding": RAW_CSV_ENCODING,
        "encoding_errors": "replace",
    }

def read_csv_kwargs(csv_path: Path | str) -> dict[str, object]:
    """Return pandas CSV options for either the raw file or processed samples."""
    path = Path(csv_path)
    if path.suffix == ".gz":
        return {
            "low_memory": False,
        }
    return read_raw_csv_kwargs()

def load_requests(csv_path: Path | str, nrows: int | None = None) -> pd.DataFrame:
    """Load the raw Montreal 311 CSV using the shared file settings."""
    return pd.read_csv(
        csv_path,
        nrows=nrows,
        **read_csv_kwargs(csv_path),
    )

def parse_mixed_datetime(series: pd.Series) -> pd.Series:
    """Parses datetime values from the mixed formats used in the source data.

    The raw file contains more than one timestamp format, so this tries
    the main format first and then fills the remaining missing values with a
    second format.
    """
    # The source file mixes ISO-style strings with slash-delimited timestamps
    parsed = pd.to_datetime(series, format="ISO8601", errors="coerce")
    missing_mask = parsed.isna() & series.notna()
    if missing_mask.any():
        parsed.loc[missing_mask] = pd.to_datetime(
            series.loc[missing_mask],
            format="%Y/%m/%d %H:%M:%S",
            errors="coerce",
        )
    return parsed

def prepare_base_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Builds the shared cleaned dataframe used by both prediction tasks.

    This fn fixes text issues, parses timestamps, creates normalized
    helper columns, and adds simple creation-time features used later in the
    project.
    """
    # Shared cleaned dataframe used by both downstream tasks
    prepared = df.copy()

    # Cleaninig free-text and categorical columns befor normalization.
    for column in STRING_COLUMNS:
        if column in prepared.columns:
            prepared[column] = prepared[column].map(maybe_fix_mojibake)

    # Forced provenance and location fields to numeric (for NaN)
    for column in PROVENANCE_COLUMNS + LOCATION_COLUMNS:
        if column in prepared.columns:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    # Parsing timestamps for time-based splitting and resolution-time targets
    creation_source = prepared.get("DDS_DATE_CREATION", _empty_object_series(prepared.index))
    last_status_source = prepared.get("DATE_DERNIER_STATUT", _empty_object_series(prepared.index))
    prepared["creation_ts"] = parse_mixed_datetime(creation_source)
    prepared["last_status_ts"] = parse_mixed_datetime(last_status_source)

    # Creating normalized target/helper columns
    prepared["NATURE_TARGET"] = (
        prepared["NATURE"].map(canonicalize_nature)
        if "NATURE" in prepared.columns
        else _empty_object_series(prepared.index)
    )
    prepared["FINAL_STATUS_FOLDED"] = (
        prepared["DERNIER_STATUT"].map(fold_text)
        if "DERNIER_STATUT" in prepared.columns
        else _empty_object_series(prepared.index)
    )
    if "ACTI_NOM" in prepared.columns:
        prepared["ACTI_NOM"] = prepared["ACTI_NOM"].fillna("")
    else:
        prepared["ACTI_NOM"] = ""

    # Time-derived features that depend only on request creation time
    prepared["creation_year"] = prepared["creation_ts"].dt.year
    prepared["creation_month"] = prepared["creation_ts"].dt.month
    prepared["creation_dayofweek"] = prepared["creation_ts"].dt.dayofweek
    prepared["creation_hour"] = prepared["creation_ts"].dt.hour
    prepared["has_geo"] = (
        prepared[["LOC_LAT", "LOC_LONG"]].notna().all(axis=1).astype(float)
        if {"LOC_LAT", "LOC_LONG"}.issubset(prepared.columns)
        else 0.0
    )
    return prepared

def prepare_classification_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Returns the cleaned dataframe ready for classification training.

    This version keeps only rows with a valid creation timestamp and a usable
    normalized class label.
    """
    # Valid creation timestamp and a normalized request type for classification
    prepared = prepare_base_frame(df)
    # Drop rows missing fields bc they cannot contribute a supervised classification label
    prepared = prepared.dropna(subset=["creation_ts", "NATURE_TARGET"]).copy()
    return prepared

def prepare_regression_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Returns the cleaned dataframe ready for regression training.

    This version keeps only closed requests with valid timestamps and a
    non-negative resolution time in days.
    """
    # Restricting regression to requests with usable closing timestamp and closed status
    prepared = prepare_base_frame(df)
    prepared = prepared.dropna(subset=["creation_ts", "last_status_ts"]).copy()

    # Keep only closed requests for resolution time prediction
    prepared = prepared.loc[
        prepared["FINAL_STATUS_FOLDED"].isin(CANONICAL_CLOSED_STATUSES)
    ].copy()
    # Resolution time in days derived from timestamps (regression target)
    prepared["resolution_time_days"] = (
        prepared["last_status_ts"] - prepared["creation_ts"]
    ).dt.total_seconds() / 86400.0

    # excluding negative or missing duration
    prepared = prepared.loc[
        prepared["resolution_time_days"].notna()
        & (prepared["resolution_time_days"] >= 0)
    ].copy()
    return prepared
