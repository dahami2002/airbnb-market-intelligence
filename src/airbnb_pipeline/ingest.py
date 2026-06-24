"""Download raw data files from Inside Airbnb."""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests

from .config import get_city_config, get_data_paths
from .utils import ensure_dir, file_size_mb

logger = logging.getLogger("airbnb_pipeline.ingest")

# HTTP download settings
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5
REQUEST_TIMEOUT = 120  # seconds


def ingest(city_key: str, force: bool = False) -> pd.DataFrame:
    """
    Download all data files for a city and generate ingestion metadata.

    Args:
        city_key: City identifier (e.g., 'bangkok').
        force: If True, re-download even if file already exists.

    Returns:
        DataFrame of ingestion metadata.
    """
    city_cfg = get_city_config(city_key)
    paths = get_data_paths(city_key, city_cfg)
    raw_dir = ensure_dir(paths["raw_dir"])
    processed_dir = ensure_dir(paths["processed_dir"])

    logger.info(
        "Starting ingestion for %s (snapshot: %s)",
        city_cfg["display_name"],
        city_cfg["snapshot_date"],
    )

    metadata_rows: List[Dict[str, Any]] = []

    for file_key, file_cfg in city_cfg["files"].items():
        url = file_cfg["url"]
        local_name = file_cfg["local_name"]
        local_path = raw_dir / local_name

        row = {
            "city": city_key,
            "file_key": file_key,
            "source_url": url,
            "local_path": str(local_path),
            "file_size_mb": 0.0,
            "downloaded_at": None,
            "status": "pending",
        }

        # Skip if file already exists and force=False
        if local_path.exists() and not force:
            row["file_size_mb"] = file_size_mb(local_path)
            row["status"] = "skipped_existing"
            row["downloaded_at"] = datetime.now(timezone.utc).isoformat()
            logger.info("Skipping %s — file already exists (%s MB)", local_name, row["file_size_mb"])
            metadata_rows.append(row)
            continue

        # Download with retry
        success = _download_file(url, local_path)
        if success:
            row["file_size_mb"] = file_size_mb(local_path)
            row["status"] = "success"
            row["downloaded_at"] = datetime.now(timezone.utc).isoformat()
            logger.info("Downloaded %s (%.2f MB)", local_name, row["file_size_mb"])
        else:
            row["status"] = "failed"
            row["downloaded_at"] = datetime.now(timezone.utc).isoformat()
            logger.error("Failed to download %s after %d retries", local_name, MAX_RETRIES)

        metadata_rows.append(row)

    # Save metadata
    metadata_df = pd.DataFrame(metadata_rows)
    metadata_path = processed_dir / "ingestion_metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)
    logger.info("Ingestion metadata saved to %s", metadata_path)

    # Summary
    success_count = (metadata_df["status"].isin(["success", "skipped_existing"])).sum()
    total_count = len(metadata_df)
    logger.info("Ingestion complete: %d/%d files available.", success_count, total_count)

    return metadata_df


def _download_file(url: str, local_path: Path) -> bool:
    """
    Download a single file with retry logic.

    Args:
        url: Source URL.
        local_path: Destination file path.

    Returns:
        True if download succeeded, False otherwise.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.debug("Download attempt %d/%d: %s", attempt, MAX_RETRIES, url)
            response = requests.get(url, timeout=REQUEST_TIMEOUT, stream=True)
            response.raise_for_status()

            # Write in chunks to handle large files
            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return True

        except requests.RequestException as e:
            logger.warning(
                "Attempt %d/%d failed for %s: %s",
                attempt, MAX_RETRIES, url, e,
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)

    return False
