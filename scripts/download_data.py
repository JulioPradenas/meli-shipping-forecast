"""Download the Brazilian E-Commerce dataset from Kaggle.

This script downloads `olistbr/brazilian-ecommerce` into `data/raw/` and
unzips it. If the dataset is already present, the download is skipped.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --force  # re-download even if present
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import zipfile
from pathlib import Path

DATASET_REF = "olistbr/brazilian-ecommerce"
RAW_DATA_DIR = Path("data/raw")

# Files expected after unzipping (sanity check)
EXPECTED_FILES = [
    "olist_customers_dataset.csv",
    "olist_geolocation_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_orders_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "product_category_name_translation.csv",
]


def setup_logging() -> logging.Logger:
    """Configure module-level logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("download_data")


def already_downloaded(target_dir: Path) -> bool:
    """Check whether all expected CSV files are present in the target directory."""
    missing = [f for f in EXPECTED_FILES if not (target_dir / f).exists()]
    return len(missing) == 0


def download_dataset(target_dir: Path, log: logging.Logger) -> None:
    """Download and unzip the dataset using the Kaggle CLI."""
    target_dir.mkdir(parents=True, exist_ok=True)
    log.info("Downloading dataset %s to %s", DATASET_REF, target_dir)

    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET_REF, "-p", str(target_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        log.error("Kaggle CLI failed:\n%s", result.stderr)
        sys.exit(1)
    log.info("Download complete.")

    zip_files = list(target_dir.glob("*.zip"))
    if not zip_files:
        log.error("No zip file found after download. Aborting.")
        sys.exit(1)

    zip_path = zip_files[0]
    log.info("Extracting %s ...", zip_path.name)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(target_dir)
    zip_path.unlink()
    log.info("Extraction complete. Zip file removed.")


def verify_files(target_dir: Path, log: logging.Logger) -> None:
    """Verify all expected files are present and non-empty."""
    for fname in EXPECTED_FILES:
        fpath = target_dir / fname
        if not fpath.exists():
            log.error("Missing expected file: %s", fname)
            sys.exit(1)
        size_kb = fpath.stat().st_size / 1024
        log.info("  ✓ %-45s %8.1f KB", fname, size_kb)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files already exist",
    )
    args = parser.parse_args()

    log = setup_logging()

    if not args.force and already_downloaded(RAW_DATA_DIR):
        log.info("Dataset already present in %s. Skipping download.", RAW_DATA_DIR)
        log.info("Use --force to re-download.")
        verify_files(RAW_DATA_DIR, log)
        return

    download_dataset(RAW_DATA_DIR, log)
    verify_files(RAW_DATA_DIR, log)
    log.info("Done.")


if __name__ == "__main__":
    main()
