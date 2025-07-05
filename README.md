# Rxiv Corpus Downloader

This repository contains a simple and scalable pipeline to fetch all metadata and full-text PDFs from bioRxiv and medRxiv preprints.

## Features
- Crawls metadata via official API
- Downloads PDFs with retry logic
- Uses multithreading for scalability

## Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download metadata
```bash
python download_metadata.py
```

### 3. Download PDFs
```bash
python download_pdfs.py
```

Downloaded files will be saved in the `data/` and `pdfs/` directories.


---
MIT License.
