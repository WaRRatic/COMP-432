from __future__ import annotations
import pandas as pd

def split_by_time(
    df: pd.DataFrame,
    train_end: str = "2020-12-31 23:59:59",
    validation_end: str = "2021-06-30 23:59:59",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return chronological train, validation, and test splits using `creation_ts`."""

    # Turn the boundary strings into pandas timestamps so we can compare them
    # directly against the dataframe's `creation_ts` column.
    train_end_ts = pd.Timestamp(train_end)
    validation_end_ts = pd.Timestamp(validation_end)

    # The windows are ordered in time so evaluation uses older requests to predict newer ones.
    # Training contains everything up to the first cutoff.
    train = df.loc[df["creation_ts"] <= train_end_ts].copy()

    # Validation contains the next block of time after training.
    validation = df.loc[
        (df["creation_ts"] > train_end_ts)
        & (df["creation_ts"] <= validation_end_ts)
    ].copy()

    # Test contains the newest requests, which simulates future unseen data.
    test = df.loc[df["creation_ts"] > validation_end_ts].copy()
    return train, validation, test
