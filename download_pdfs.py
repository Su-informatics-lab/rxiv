"""PDF downloader for bioRxiv and medRxiv preprints.

This module provides robust PDF downloading capabilities with:
- Multi-threaded downloads with configurable thread pools
- Comprehensive error handling and retry logic
- PDF validation and integrity checking with SHA256 hashing
- Resume capability for interrupted downloads
- Detailed logging and progress tracking
- Rate limiting for respectful server usage

Input:
    - Metadata entries from JSONL files containing DOIs and versions
    - Source name (biorxiv/medrxiv) for URL construction
    - Output directory for organized PDF storage
    - Resume flag for continuing interrupted downloads

Output:
    - Downloaded and validated PDF files organized by source
    - Detailed log files with download status and errors
    - Success/failure tracking files with hashes and statistics
    - Progress reports with real-time download statistics
"""

import hashlib
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

MAX_RETRIES = 5
NUM_THREADS = 8  # Reduced to be more respectful
TIMEOUT = 30
RETRY_BACKOFF = 2
CHUNK_SIZE = 1024 * 1024  # 1MB chunks for download


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


def safe_doi_to_filename(doi: str) -> str:
    """Convert DOI to safe filename by URL encoding.

    Args:
        doi: DOI string to convert

    Returns:
        URL-encoded filename safe for filesystem
    """
    return quote(doi.replace("/", "_"), safe="")


def construct_pdf_url(source: str, doi: str, version: str) -> str:
    """Construct PDF download URL for given source, DOI, and version.

    Args:
        source: Source name (biorxiv/medrxiv)
        doi: DOI of the paper
        version: Version number

    Returns:
        Complete URL to PDF file
    """
    base = (
        "https://www.biorxiv.org" if source == "biorxiv" else "https://www.medrxiv.org"
    )
    return f"{base}/content/{doi}v{version}.full.pdf"


def calculate_file_hash(file_path: Path) -> str:
    """Calculate SHA256 hash of a file for integrity verification.

    Args:
        file_path: Path to file to hash

    Returns:
        SHA256 hash as hexadecimal string
    """
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def validate_pdf(file_path: Path) -> bool:
    """Validate that downloaded file is a valid PDF by checking header.

    Args:
        file_path: Path to file to validate

    Returns:
        True if file is a valid PDF, False otherwise
    """
    try:
        with open(file_path, "rb") as f:
            header = f.read(4)
            return header == b"%PDF"
    except Exception:
        return False


def download_pdf(
    entry: Dict[str, Any],
    source: str,
    out_dir: str,
    session: requests.Session,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Download a single PDF with comprehensive error handling and validation.

    Args:
        entry: Metadata entry containing doi and version
        source: Source name (biorxiv/medrxiv)
        out_dir: Output directory for PDF files
        session: Configured requests session
        logger: Logger instance

    Returns:
        Dictionary with download status, DOI, filename, and optional hash/size
    """
    doi = entry["doi"]
    version = entry["version"]
    filename = f"{safe_doi_to_filename(doi)}v{version}.pdf"
    out_path = Path(out_dir) / filename

    if out_path.exists():
        if validate_pdf(out_path):
            return {"status": "exists", "doi": doi, "file": filename}
        else:
            logger.warning(f"Invalid PDF found, re-downloading: {doi}")
            out_path.unlink()

    url = construct_pdf_url(source, doi, version)

    for attempt in range(MAX_RETRIES):
        try:
            logger.debug(f"Downloading {doi} (attempt {attempt + 1})")
            response = session.get(url, timeout=TIMEOUT, stream=True)

            if response.status_code == 200:
                content_type = response.headers.get("Content-Type", "")
                if "application/pdf" in content_type:
                    # Download with progress and validation
                    temp_path = out_path.with_suffix(".tmp")

                    with open(temp_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                            if chunk:
                                f.write(chunk)

                    # Validate downloaded file
                    if validate_pdf(temp_path):
                        temp_path.rename(out_path)
                        file_hash = calculate_file_hash(out_path)
                        return {
                            "status": "downloaded",
                            "doi": doi,
                            "file": filename,
                            "hash": file_hash,
                            "size": out_path.stat().st_size,
                        }
                    else:
                        temp_path.unlink()
                        logger.warning(f"Downloaded file is not a valid PDF: {doi}")
                        return {"status": "invalid_pdf", "doi": doi, "file": filename}
                else:
                    logger.warning(
                        f"Unexpected content type '{content_type}' for {doi}"
                    )
                    return {
                        "status": "wrong_content_type",
                        "doi": doi,
                        "file": filename,
                    }
            elif response.status_code == 404:
                logger.warning(f"PDF not found (404) for {doi}")
                return {"status": "not_found", "doi": doi, "file": filename}
            elif response.status_code == 429:
                logger.warning(f"Rate limited (429) for {doi}")
                sleep_time = (RETRY_BACKOFF**attempt) * 2
                time.sleep(sleep_time)
            else:
                logger.warning(f"HTTP {response.status_code} for {doi}")

        except requests.exceptions.RequestException as e:
            logger.warning(f"Request failed for {doi} (attempt {attempt + 1}): {e}")
            if attempt < MAX_RETRIES - 1:
                sleep_time = RETRY_BACKOFF**attempt
                time.sleep(sleep_time)
        except Exception as e:
            logger.error(f"Unexpected error downloading {doi}: {e}")
            break

    return {"status": "failed", "doi": doi, "file": filename}


def load_entries(path: str) -> List[Dict[str, Any]]:
    """Load metadata entries from JSONL file with validation.

    Args:
        path: Path to JSONL metadata file

    Returns:
        List of validated metadata entries containing doi and version

    Raises:
        FileNotFoundError: If metadata file doesn't exist
    """
    entries = []
    try:
        with open(path) as f:
            for line_num, line in enumerate(f, 1):
                try:
                    entry = json.loads(line.strip())
                    if "doi" in entry and "version" in entry:
                        entries.append(entry)
                    else:
                        logging.warning(
                            f"Skipping invalid entry at line {line_num}: missing doi or version"
                        )
                except json.JSONDecodeError:
                    logging.warning(f"Skipping invalid JSON at line {line_num}")
    except FileNotFoundError:
        logging.error(f"Metadata file not found: {path}")
        raise
    return entries


def save_results(
    results: List[Dict[str, Any]], source: str, log_dir: str
) -> Tuple[int, int]:
    """Save download results to separate success and failure files.

    Args:
        results: List of download result dictionaries
        source: Source name for file naming
        log_dir: Directory to save result files

    Returns:
        Tuple of (successful_count, failed_count)
    """
    success_file = (
        Path(log_dir)
        / f"pdf_success_{source}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    failed_file = (
        Path(log_dir)
        / f"pdf_failed_{source}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    successful = [r for r in results if r["status"] in ["downloaded", "exists"]]
    failed = [r for r in results if r["status"] not in ["downloaded", "exists"]]

    with open(success_file, "w") as f:
        json.dump(successful, f, indent=2)

    with open(failed_file, "w") as f:
        json.dump(failed, f, indent=2)

    return len(successful), len(failed)


def run_parallel(
    entries: List[Dict[str, Any]], source: str, out_dir: str, resume: bool = False
) -> List[Dict[str, Any]]:
    """Run parallel PDF downloads with comprehensive logging and progress tracking.

    Args:
        entries: List of metadata entries to download
        source: Source name (biorxiv/medrxiv)
        out_dir: Output directory for PDF files
        resume: Whether to skip existing files

    Returns:
        List of download result dictionaries with status, DOI, and file info
    """
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"pdf_{source}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logging(log_file)

    logger.info(f"Starting PDF download for {source}")
    logger.info(f"Total entries to process: {len(entries)}")

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if resume:
        # Filter out already downloaded files
        original_count = len(entries)
        entries = [
            e
            for e in entries
            if not (
                Path(out_dir) / f"{safe_doi_to_filename(e['doi'])}v{e['version']}.pdf"
            ).exists()
        ]
        logger.info(
            f"Resume mode: skipping {original_count - len(entries)} existing files"
        )

    results = []
    session = create_session()

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = [
            executor.submit(download_pdf, entry, source, out_dir, session, logger)
            for entry in entries
        ]

        with tqdm(total=len(entries), desc=f"Downloading {source} PDFs") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)

                    if result["status"] == "downloaded":
                        pbar.set_postfix({"status": "downloaded"})
                    elif result["status"] == "exists":
                        pbar.set_postfix({"status": "exists"})
                    else:
                        pbar.set_postfix({"status": "failed"})

                    pbar.update(1)

                    # Add small delay to be respectful
                    time.sleep(0.1)

                except Exception as e:
                    logger.error(f"Future failed: {e}")
                    results.append(
                        {"status": "error", "doi": "unknown", "file": "unknown"}
                    )
                    pbar.update(1)

    # Save results
    successful, failed = save_results(results, source, log_dir)

    logger.info(f"Download completed for {source}")
    logger.info(f"Successfully downloaded: {successful}")
    logger.info(f"Failed downloads: {failed}")

    return results


if __name__ == "__main__":
    sources = ["biorxiv", "medrxiv"]
    for source in sources:
        metadata_file = f"data/{source}_metadata.jsonl"
        out_dir = f"pdfs/{source}"

        try:
            if not Path(metadata_file).exists():
                print(f"Metadata file not found: {metadata_file}")
                continue

            entries = load_entries(metadata_file)
            results = run_parallel(entries, source, out_dir, resume=True)

            downloaded = len([r for r in results if r["status"] == "downloaded"])
            exists = len([r for r in results if r["status"] == "exists"])
            failed = len(
                [r for r in results if r["status"] not in ["downloaded", "exists"]]
            )

            print(
                f"Completed {source}: {downloaded} downloaded, {exists} existing, {failed} failed"
            )

        except Exception as e:
            print(f"Failed to process {source}: {e}")
            continue
