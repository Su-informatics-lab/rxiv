# Rxiv Corpus Downloader

This repository contains a robust and resilient pipeline to fetch all metadata and full-text PDFs from bioRxiv and 
medRxiv preprints with comprehensive error handling, logging, and resume capabilities.

## Features

### Core Functionality
- **Comprehensive API Coverage**: Crawls metadata via official bioRxiv/medRxiv API
- **Resilient PDF Downloads**: Multi-threaded PDF downloads with validation
- **Smart Resume**: Automatically resume interrupted downloads
- **Data Validation**: PDF header validation and SHA256 hash calculation
- **Flexible Configuration**: JSON-based configuration with command-line overrides

### Robustness & Reliability
- **Error Handling**: Exponential backoff, retry strategies, and comprehensive exception handling
- **Logging**: Separate timestamped logs for all operations with detailed error tracking
- **Rate Limiting**: Respectful crawling with configurable delays
- **Data Integrity**: Validates downloaded files and tracks failed operations
- **Recovery Mechanisms**: Failed chunk tracking and automatic retry capabilities

### Scalability & Performance
- **Optimized Threading**: Configurable thread pools for efficient downloads
- **Memory Efficient**: Streaming downloads with chunked processing
- **Progress Tracking**: Real-time progress bars and detailed statistics
- **Batch Processing**: Efficient handling of large datasets

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download everything (recommended)
```bash
python main.py
```

### 3. Or use individual scripts
```bash
# Download metadata only
python main.py --metadata-only

# Download PDFs only
python main.py --pdfs-only
```

## Advanced Usage

### Command Line Options
```bash
# Process specific source
python main.py --source biorxiv

# Resume interrupted downloads
python main.py --resume

# Custom date range
python main.py --start-date 2020-01-01 --end-date 2024-12-31

# Adjust thread count
python main.py --threads 4

# Dry run (show what would be done)
python main.py --dry-run

# Custom configuration
python main.py --config custom_config.json
```

### Configuration

The pipeline uses `config.json` for settings. Key configurations include:

```json
{
  "api": {
    "timeout": 30,
    "max_retries": 5,
    "rate_limit_delay": 0.1
  },
  "download": {
    "num_threads": 8,
    "validate_pdfs": true,
    "calculate_hashes": true
  },
  "paths": {
    "data_dir": "data",
    "pdf_dir": "pdfs",
    "log_dir": "logs"
  }
}
```

## Directory Structure

```
rxiv/
├── main.py              # Main orchestration script
├── download_metadata.py # Metadata downloader
├── download_pdfs.py     # PDF downloader
├── config.py           # Configuration management
├── config.json         # Configuration file (auto-generated)
├── requirements.txt    # Python dependencies
├── data/              # Metadata files (JSONL format)
│   ├── biorxiv_metadata.jsonl
│   └── medrxiv_metadata.jsonl
├── pdfs/              # Downloaded PDFs
│   ├── biorxiv/
│   └── medrxiv/
└── logs/              # Comprehensive logs
    ├── metadata_*.log
    ├── pdf_*.log
    ├── failed_chunks_*.json
    └── results_*.json
```

## Monitoring & Troubleshooting

### Logs
- **Metadata logs**: `logs/metadata_[source]_[timestamp].log`
- **PDF logs**: `logs/pdf_[source]_[timestamp].log`
- **Failed operations**: `logs/failed_chunks_[source]_[timestamp].json`
- **Results summary**: `logs/results_[timestamp].json`

### Resume Capability
The pipeline automatically detects existing files and can resume from where it left off:
```bash
python main.py --resume
```

### Error Recovery
- Failed metadata chunks are tracked and can be retried
- Invalid PDFs are automatically re-downloaded
- Comprehensive error logging helps identify and resolve issues

## Data Output

### Metadata Format
Metadata is saved as JSONL files with validated entries containing:
- DOI, title, authors, publication date
- Abstract, categories, version information
- Publication status and journal details

### PDF Organization
PDFs are organized by source with filename format:
- `[URL-encoded-DOI]v[version].pdf`
- SHA256 hashes calculated for integrity verification
- Invalid files automatically detected and removed

## Performance

The pipeline is designed for large-scale operations:
- **Metadata**: ~100 entries per API call with automatic pagination
- **PDFs**: Configurable threading (default: 8 threads)
- **Rate limiting**: Built-in delays to respect server resources
- **Memory efficient**: Streaming downloads to handle large files

## Contributing

This pipeline follows best practices for:
- Respectful web scraping with rate limiting
- Comprehensive error handling and logging
- Data validation and integrity checks
- Resumable operations for large datasets

---
MIT License.
