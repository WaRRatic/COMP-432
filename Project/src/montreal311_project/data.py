from __future__ import annotations
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

# Group columns by role so the cleaning step is easier to follow.
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

# Normalize the labels we will later use as targets.
CANONICAL_NATURES = {
    "information": "Information",
    "commentaire": "Commentaire",
    "requete": "Requete",
    "plainte": "Plainte",
}

CANONICAL_CLOSED_STATUSES = {
    "terminee",
    "refusee",
    "annulee",
    "supprimee",
}


def _empty_object_series(index: pd.Index | None = None) -> pd.Series:
    return pd.Series(index=index, dtype="object")


def maybe_fix_mojibake(value: object) -> object:
    if not isinstance(value, str):
        return value

    # Clean extra whitespace first.
    text = " ".join(value.strip().split())
    if not text:
        return ""

    # Try to repair a few common encoding glitches.
    if any(token in text for token in ("Ã", "Â", "â", "�")):
        try:
            repaired = text.encode("latin1").decode("utf-8")
        except UnicodeError:
            repaired = text
        if repaired:
            text = repaired

    return text

def fold_text(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""

    # Fold text to lowercase ASCII so labels are easier to match consistently.
    text = str(maybe_fix_mojibake(value))
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return " ".join(ascii_text.lower().split())


def canonicalize_nature(value: object) -> str | None:
    # Map raw labels into the smaller cleaned set used later.
    folded = fold_text(value)
    return CANONICAL_NATURES.get(folded)


def load_requests(csv_path: Path | str, nrows: int | None = None) -> pd.DataFrame:
    # Replace bad characters instead of failing on a few broken rows.
    return pd.read_csv(
        csv_path,
        low_memory=False,
        nrows=nrows,
        encoding_errors="replace",
    )


def parse_mixed_datetime(series: pd.Series) -> pd.Series:
    # The file uses more than one datetime format.
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
    # Shared cleaned dataframe used by both tasks.
    prepared = df.copy()

    # Clean text-like columns first.
    for column in STRING_COLUMNS:
        if column in prepared.columns:
            prepared[column] = prepared[column].map(maybe_fix_mojibake)

    # Convert numeric fields and let invalid values become NaN.
    for column in PROVENANCE_COLUMNS + LOCATION_COLUMNS:
        if column in prepared.columns:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    # Parse the timestamps needed later for splitting and targets.
    creation_source = prepared.get("DDS_DATE_CREATION", _empty_object_series(prepared.index))
    last_status_source = prepared.get("DATE_DERNIER_STATUT", _empty_object_series(prepared.index))
    prepared["creation_ts"] = parse_mixed_datetime(creation_source)
    prepared["last_status_ts"] = parse_mixed_datetime(last_status_source)

    # Build cleaned helper columns once so both tasks can reuse them.
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

    # These features are safe because they only use the creation timestamp.
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
    # Classification needs a valid timestamp and a valid cleaned label.
    prepared = prepare_base_frame(df)
    # Rows missing either field cannot be used for supervised learning.
    prepared = prepared.dropna(subset=["creation_ts", "NATURE_TARGET"]).copy()
    return prepared


def prepare_regression_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Regression needs both timestamps and a closed-like final status.
    prepared = prepare_base_frame(df)
    prepared = prepared.dropna(subset=["creation_ts", "last_status_ts"]).copy()

    # Keep only requests that were actually closed.
    prepared = prepared.loc[
        prepared["FINAL_STATUS_FOLDED"].isin(CANONICAL_CLOSED_STATUSES)
    ].copy()

    # Resolution time in days is the regression target.
    prepared["resolution_time_days"] = (
        prepared["last_status_ts"] - prepared["creation_ts"]
    ).dt.total_seconds() / 86400.0

    # Drop invalid durations.
    prepared = prepared.loc[
        prepared["resolution_time_days"].notna()
        & (prepared["resolution_time_days"] >= 0)
    ].copy()
    return prepared
