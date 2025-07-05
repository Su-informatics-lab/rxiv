# Rxiv Corpus Downloader

A pipeline to download metadata and PDFs from bioRxiv and medRxiv with error handling, logging, and resume capabilities.

## Features

- **Comprehensive Downloads**: Fetches metadata and PDFs from bioRxiv/medRxiv
- **Resumable Design**: Automatically resume interrupted downloads
- **Error Handling**: Retry logic with exponential backoff
- **Data Validation**: PDF validation and integrity checks
- **Detailed Logging**: Comprehensive logs for monitoring and troubleshooting
## Usage

### Basic Usage
```bash
pip install -r requirements.txt
python main.py
```

### Common Options
```bash
# download only metadata
python main.py --metadata-only

# download only PDFs  
python main.py --pdfs-only

# process specific source
python main.py --source biorxiv

# resume interrupted downloads
python main.py --resume

# custom date range
python main.py --start-date 2020-01-01 --end-date 2024-12-31
```

## Output

- **Metadata**: Saved as JSONL files in `data/` directory
- **PDFs**: Organized by source in `pdfs/` directory  
- **Logs**: Detailed logs and error tracking in `logs/` directory

Files are automatically validated and resume capability handles interruptions.

## Contact

For questions or issues, please contact:
- **Haining Wang** - hw56@iu.edu

## License
MIT
