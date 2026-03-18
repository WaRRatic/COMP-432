from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from montreal311_project.paths import PROCESSED_DATA_DIR, RAW_DATA_PATH
from montreal311_project.sampling import build_representative_subset_from_csv, save_subset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the representative Montreal 311 subset.")
    parser.add_argument(
        "--input",
        type=Path,
        default=RAW_DATA_PATH,
        help="Path to the raw Montreal 311 CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DATA_DIR / "requetes311_2019_2021_sample_300k.csv.gz",
        help="Path for the sampled output CSV.gz.",
    )
    parser.add_argument("--target-rows", type=int, default=300_000)
    parser.add_argument("--min-per-group", type=int, default=25)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional debug cap for a partial scan.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Raw dataset not found: {args.input}")

    subset, metadata = build_representative_subset_from_csv(
        csv_path=args.input,
        target_rows=args.target_rows,
        min_per_group=args.min_per_group,
        random_state=args.random_state,
        chunk_size=args.chunk_size,
        max_rows=args.max_rows,
    )
    save_subset(subset, args.output, metadata)

    print(f"Saved {len(subset):,} rows to {args.output}")


if __name__ == "__main__":
    main()
