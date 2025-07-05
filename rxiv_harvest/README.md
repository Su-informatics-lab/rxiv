# Academic Paper Harvesting

Downloads metadata and PDFs from bioRxiv and medRxiv repositories.

## Features

- **Dual-source**: bioRxiv and medRxiv support
- **Multi-threaded**: Parallel PDF downloads
- **Resume capability**: Continue interrupted downloads
- **Error handling**: Retry logic and failure tracking

## Installation

```bash
cd rxiv_harvest
pip install -r requirements.txt
```

## Usage

```bash
# Download all sources
python download.py

# Specific source
python download.py --source biorxiv

# Metadata only
python download.py --metadata-only

# Resume interrupted download
python download.py --resume
```

## Configuration

```json
{
  "sources": {
    "biorxiv": {"enabled": true, "start_date": "2020-01-01"},
    "medrxiv": {"enabled": true, "start_date": "2019-06-01"}
  },
  "download": {
    "num_threads": 8,
    "max_retries": 3
  }
}
```

## Output

```
data/
├── biorxiv_metadata.jsonl    # bioRxiv metadata
├── medrxiv_metadata.jsonl    # medRxiv metadata  
├── pdfs/
│   ├── biorxiv/             # bioRxiv PDFs
│   │   ├── 2024.01.15.576123v1.full.pdf
│   │   └── 2024.01.16.576234v1.full.pdf
│   └── medrxiv/             # medRxiv PDFs
│       ├── 2024.01.15.24300123v1.full.pdf
│       └── 2024.01.16.24300234v1.full.pdf
└── logs/
    ├── harvest_20240105_143022.log  # Main logs
    └── errors_20240105_143022.log   # Error logs
```

### Example Metadata Entry
```json
{
  "doi": "10.1101/2024.01.15.576123",
  "title": "CRISPR-Cas9 gene editing in cancer therapy",
  "authors": "Smith, J.; Doe, A.; Johnson, B.",
  "date": "2024-01-15",
  "category": "Cancer Biology",
  "abstract": "We demonstrate the therapeutic potential of CRISPR-Cas9...",
  "license": "cc_by",
  "published": "NA",
  "server": "biorxiv"
}
```

## Integration

Process with main pipeline:
```bash
cd ../processing
python process_papers.py ../rxiv_harvest/data/pdfs/biorxiv/ processed_papers/
```