"""Metadata downloader for bioRxiv and medRxiv preprints.

This module provides robust metadata downloading capabilities with:
- Comprehensive error handling with exponential backoff retry
- Resume capability for interrupted downloads
- Data validation and integrity checking
- Detailed logging with failed operation tracking
- Rate limiting for respectful API usage

Input:
    - Source name (biorxiv/medrxiv)
    - Date range (start_date to end_date)
    - Output file path for JSONL metadata
    - Resume flag for continuing interrupted downloads

Output:
    - JSONL files containing validated metadata entries
    - Detailed log files with operation status and errors
    - Failed chunk tracking files for recovery
    - Statistics on successful vs failed downloads
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

API_BASE = "https://api.biorxiv.org/details"
CHUNK_SIZE = 100  # max allowed
SOURCES = {"biorxiv": "2013-01-01", "medrxiv": "2019-01-01"}
END_DATE = "2025-07-01"
MAX_RETRIES = 5
RETRY_BACKOFF = 2
TIMEOUT = 30


def setup_logging(log_file: str) -> logging.Logger:
    """Setup logging configuration with file and console handlers.

    Args:
        log_file: Path to log file

    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def create_session() -> requests.Session:
    """Create a requests session with retry strategy.

    Returns:
        Configured session with exponential backoff retry strategy
    """
    session = requests.Session()
    retry_strategy = Retry(
        total=MAX_RETRIES,
        backoff_factor=RETRY_BACKOFF,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def get_total_count(
    session: requests.Session,
    source: str,
    start_date: str,
    end_date: str,
    logger: logging.Logger,
) -> int:
    """Get total count of available papers for a source and date range.

    Args:
        session: Configured requests session
        source: Source name (biorxiv/medrxiv)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        logger: Logger instance

    Returns:
        Total number of papers available

    Raises:
        Exception: If API request fails
    """
    url = f"{API_BASE}/{source}/{start_date}/{end_date}/0"
    try:
        response = session.get(url, timeout=TIMEOUT)
        response.raise_for_status()
        data = response.json()
        total = int(data["messages"][0]["total"])
        logger.info(f"Total papers available for {source}: {total}")
        return total
    except Exception as e:
        logger.error(f"Failed to get total count for {source}: {e}")
        raise


def fetch_chunk(
    session: requests.Session,
    source: str,
    start_date: str,
    end_date: str,
    offset: int,
    logger: logging.Logger,
) -> List[Dict[str, Any]]:
    """Fetch a single chunk of metadata with retry logic.

    Args:
        session: Configured requests session
        source: Source name (biorxiv/medrxiv)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        offset: Pagination offset
        logger: Logger instance

    Returns:
        List of metadata entries

    Raises:
        Exception: If all retry attempts fail
    """
    url = f"{API_BASE}/{source}/{start_date}/{end_date}/{offset}"

    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(url, timeout=TIMEOUT)
            response.raise_for_status()
            data = response.json()

            if "collection" not in data:
                logger.warning(
                    f"No collection in response for {source} offset {offset}"
                )
                return []

            collection = data["collection"]
            logger.debug(
                f"Fetched {len(collection)} entries for {source} offset {offset}"
            )
            return collection

        except requests.exceptions.RequestException as e:
            logger.warning(
                f"Attempt {attempt + 1} failed for {source} offset {offset}: {e}"
            )
            if attempt < MAX_RETRIES - 1:
                sleep_time = RETRY_BACKOFF**attempt
                logger.info(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                logger.error(f"All retry attempts failed for {source} offset {offset}")
                raise


def validate_entry(entry: Dict[str, Any], logger: logging.Logger) -> bool:
    """Validate a metadata entry for required fields.

    Args:
        entry: Metadata entry dictionary
        logger: Logger instance

    Returns:
        True if entry is valid, False otherwise
    """
    required_fields = ["doi", "title", "authors", "date", "version"]
    for field in required_fields:
        if field not in entry or not entry[field]:
            logger.warning(f"Missing or empty required field '{field}' in entry")
            return False
    return True


def fetch_all(
    source: str, start_date: str, end_date: str, out_file: str, resume: bool = False
) -> Tuple[int, List[int]]:
    """Fetch all metadata with comprehensive error handling and logging.

    Args:
        source: Source name (biorxiv/medrxiv)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        out_file: Output file path for JSONL metadata
        resume: Whether to resume from existing file

    Returns:
        Tuple of (successful_entries_count, failed_chunk_offsets)
    """
    log_file = f"logs/metadata_{source}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(log_file)

    logger.info(
        f"Starting metadata download for {source} from {start_date} to {end_date}"
    )

    session = create_session()
    total = get_total_count(session, source, start_date, end_date, logger)

    Path(out_file).parent.mkdir(parents=True, exist_ok=True)

    start_offset = 0
    if resume and Path(out_file).exists():
        with open(out_file, "r") as f:
            existing_count = sum(1 for _ in f)
        start_offset = (existing_count // CHUNK_SIZE) * CHUNK_SIZE
        logger.info(
            f"Resuming from offset {start_offset} (found {existing_count} existing entries)"
        )

    mode = "a" if resume and Path(out_file).exists() else "w"
    successful_entries = 0
    failed_chunks = []

    with open(out_file, mode) as f:
        with tqdm(
            total=total, initial=start_offset, desc=f"Downloading {source}"
        ) as pbar:
            for offset in range(start_offset, total, CHUNK_SIZE):
                try:
                    collection = fetch_chunk(
                        session, source, start_date, end_date, offset, logger
                    )

                    valid_entries = 0
                    for entry in collection:
                        if validate_entry(entry, logger):
                            f.write(json.dumps(entry) + "\n")
                            valid_entries += 1
                        else:
                            logger.warning(
                                f"Skipping invalid entry: {entry.get('doi', 'unknown')}"
                            )

                    successful_entries += valid_entries
                    pbar.update(len(collection))

                    time.sleep(0.1)  # Rate limiting

                except Exception as e:
                    logger.error(f"Failed to process chunk at offset {offset}: {e}")
                    failed_chunks.append(offset)
                    continue

    logger.info(f"Download completed for {source}")
    logger.info(f"Successfully downloaded: {successful_entries} entries")
    logger.info(f"Failed chunks: {len(failed_chunks)}")

    if failed_chunks:
        logger.warning(f"Failed offsets: {failed_chunks}")
        failed_file = f"logs/failed_chunks_{source}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(failed_file, "w") as f:
            json.dump({"source": source, "failed_offsets": failed_chunks}, f)
        logger.info(f"Failed chunks saved to {failed_file}")

    return successful_entries, failed_chunks


if __name__ == "__main__":
    for src, start in SOURCES.items():
        out_file = f"data/{src}_metadata.jsonl"
        try:
            successful, failed = fetch_all(src, start, END_DATE, out_file)
            print(
                f"Completed {src}: {successful} successful, {len(failed)} failed chunks"
            )
        except Exception as e:
            print(f"Failed to download {src}: {e}")
            continue
