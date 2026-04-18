"""Builds and saves the representative subset used by the project.

Handles the sampling logic for creating a smaller Montreal 311
dataset that still keeps the main month and request-type structure.
"""

from __future__ import annotations
import heapq
import json
from pathlib import Path
import numpy as np
import pandas as pd

from .data import prepare_base_frame, read_raw_csv_kwargs

def _trim_chunk_for_debug(
    chunk: pd.DataFrame,
    rows_processed: int,
    max_rows: int | None,
) -> pd.DataFrame | None:
    """Triming a chunk when a small debug row limit is being used.

    This helper keeps the normal chunk unchanged unless you set `max_rows` for a test run.
    """
    if max_rows is None:
        return chunk

    remaining = max_rows - rows_processed
    if remaining <= 0:
        return None
    if len(chunk) > remaining:
        return chunk.iloc[:remaining].copy()
    return chunk

def _prepare_sampling_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Adds grouping fields used for sampling.

    This is prepared with the shared data cleaning logic, filtered to rows
    with usable time and target values, and given a month key for grouping.
    """
    prepared = prepare_base_frame(chunk)
    prepared = prepared.dropna(subset=["creation_ts", "NATURE_TARGET"]).copy()
    prepared["sample_month"] = prepared["creation_ts"].dt.to_period("M").astype(str)
    return prepared

def _allocate_group_quotas(
    counts: pd.Series,
    target_rows: int,
    min_per_group: int,
) -> pd.Series:
    """Decides how many rows to sample from each month and nature group to create a subset.

    The quotas try to stay close to the original distribution while still
    giving small groups a minimum number of rows when possible.
    """
    # Removes empty groups and caps the requested sample size to available data
    counts = counts[counts > 0].sort_values(ascending=False)
    target_rows = min(int(target_rows), int(counts.sum()))

    # Proportional allocation to mirror the source distribution
    proportions = counts / counts.sum()
    raw = proportions * target_rows
    quota = np.floor(raw.to_numpy()).astype(int)

    # creating a small floor for each group to account for rare month/class
    minimum = np.minimum(counts.to_numpy(), min_per_group)
    quota = np.maximum(quota, minimum)
    quota = np.minimum(quota, counts.to_numpy())

    current_total = int(quota.sum())

    if current_total > target_rows:
        # Trimming the most reducible groups first if the min floor is above target
        excess = current_total - target_rows
        reducible = quota - minimum
        order = np.argsort(-(quota - minimum))
        while excess > 0 and reducible.sum() > 0:
            for index in order:
                if excess == 0:
                    break
                if reducible[index] > 0:
                    quota[index] -= 1
                    reducible[index] -= 1
                    excess -= 1
    if int(quota.sum()) < target_rows:
        # Extra rows to groups with space 
        deficit = target_rows - int(quota.sum())
        fractional = (raw - np.floor(raw)).to_numpy()
        capacity = counts.to_numpy() - quota
        order = np.lexsort((-capacity, -fractional))
        order = order[::-1]
        while deficit > 0 and capacity.sum() > 0:
            for index in order:
                if deficit == 0:
                    break
                if capacity[index] > 0:
                    quota[index] += 1
                    capacity[index] -= 1
                    deficit -= 1
    return pd.Series(quota, index=counts.index, dtype=int)

def save_subset(
    subset: pd.DataFrame,
    output_path: Path,
    metadata: dict[str, object],
) -> None:
    """Saving the sampled subset and its metadata to disk.

    This writes the sampled rows as a compressed CSV and stores the sampling
    metadata in a matching JSON file.
    """
    # Saving sampled rows and metadata to reproduce sampling decisions
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(output_path, index=False, compression="gzip")

    metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

def build_representative_subset_from_csv(
    csv_path: Path,
    target_rows: int = 300_000,
    min_per_group: int = 25,
    random_state: int = 42,
    chunk_size: int = 50_000,
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build a representative chronological subset from the raw Montreal 311 CSV."""
    # Pass 1: count valid rows by (month, normalized nature)
    usecols = ["NATURE", "DDS_DATE_CREATION"]
    counts_map: dict[tuple[str, str], int] = {}

    rows_seen = 0
    for chunk in pd.read_csv(
        csv_path,
        usecols=usecols,
        chunksize=chunk_size,
        **read_raw_csv_kwargs(),
    ):
        # `max_rows` is only for debugging small runs. real sampling should scan the whole file
        chunk = _trim_chunk_for_debug(chunk, rows_seen, max_rows)
        if chunk is None:
            break

        prepared = _prepare_sampling_chunk(chunk)
        grouped = prepared.groupby(["sample_month", "NATURE_TARGET"]).size()
        for key, value in grouped.items():
            counts_map[key] = counts_map.get(key, 0) + int(value)
        rows_seen += len(chunk)

    counts = pd.Series(counts_map, dtype=int).sort_values(ascending=False)
    quotas = _allocate_group_quotas(counts, target_rows=target_rows, min_per_group=min_per_group)

    # Pass 2: do bounded random selection inside each group using fixed-size heaps
    rng = np.random.default_rng(random_state)
    heaps: dict[tuple[str, str], list[tuple[float, int]]] = {
        key: [] for key in quotas.index
    }

    row_offset = 0
    for chunk in pd.read_csv(
        csv_path,
        usecols=usecols,
        chunksize=chunk_size,
        **read_raw_csv_kwargs(),
    ):
        # Apply the same optional debug cap during the row-selection pass
        chunk = _trim_chunk_for_debug(chunk, row_offset, max_rows)
        if chunk is None:
            break

        # Assign a stable global row id to recover chosen rows later
        chunk["__row_id__"] = np.arange(row_offset, row_offset + len(chunk))
        prepared = _prepare_sampling_chunk(chunk)

        for row_id, month, nature in prepared[["__row_id__", "sample_month", "NATURE_TARGET"]].itertuples(index=False):
            key = (month, nature)
            quota = int(quotas.get(key, 0))
            if quota <= 0:
                continue
            score = float(rng.random())
            heap = heaps[key]
            # Keep the rows with the best random scores for each group quota
            if len(heap) < quota:
                heapq.heappush(heap, (-score, int(row_id)))
            elif score < -heap[0][0]:
                heapq.heapreplace(heap, (-score, int(row_id)))

        row_offset += len(chunk)

    # Collect the selected global row ids sorted for final pass
    selected_row_ids = sorted(
        row_id
        for heap in heaps.values()
        for _, row_id in heap
    )

    # Pass 3: reading full rows for chosen ids and assemble final sampled dataset
    sampled_chunks: list[pd.DataFrame] = []
    pointer = 0
    row_offset = 0
    for chunk in pd.read_csv(
        csv_path,
        chunksize=chunk_size,
        **read_raw_csv_kwargs(),
    ):
        # Apply the same optional cap while recovering the final rows
        chunk = _trim_chunk_for_debug(chunk, row_offset, max_rows)
        if chunk is None:
            break

        chunk_end = row_offset + len(chunk)
        local_positions: list[int] = []
        # Convertijg global row ids into positions within the current chunk
        while pointer < len(selected_row_ids) and selected_row_ids[pointer] < chunk_end:
            if selected_row_ids[pointer] >= row_offset:
                local_positions.append(selected_row_ids[pointer] - row_offset)
            pointer += 1
        if local_positions:
            sampled_chunks.append(chunk.iloc[local_positions].copy())
        row_offset = chunk_end

    sampled = pd.concat(sampled_chunks, ignore_index=True)
    sampled = prepare_base_frame(sampled)
    # Sort chronologically so later split logic can operate directly on saved subset
    sampled = sampled.sort_values("creation_ts")

    metadata = {
        "source_rows_with_valid_time_and_nature": int(counts.sum()),
        "sampled_rows": int(len(sampled)),
        "target_rows": int(target_rows),
        "min_per_group": int(min_per_group),
        "random_state": int(random_state),
        "chunk_size": int(chunk_size),
        "max_rows": None if max_rows is None else int(max_rows),
        "monthly_nature_counts": {
            f"{month}|{nature}": int(count)
            for (month, nature), count in counts.items()
        },
        "monthly_nature_sampled_counts": {
            f"{month}|{nature}": int(count)
            for (month, nature), count in quotas.items()
        },
    }
    return sampled, metadata
