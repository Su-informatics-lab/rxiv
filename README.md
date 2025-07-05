# Knowledge Base for DelPHEA

End-to-end system for harvesting, processing, and serving academic papers through a RAG API.

## Architecture

```
HARVEST → PROCESSING → RAG API
```

## Components

### [Academic Papers Harvest](./rxiv_harvest/README.md)
Downloads papers from bioRxiv and medRxiv.

### [Clinical Guidelines Harvest](./guideline_harvest/README.md)
Harvests US clinical guidelines using crawl4ai.

### Quick Start
```bash
# bio/medRxiv papers
cd rxiv_harvest && python download.py

# clinical guidelines  
cd guideline_harvest && python crawl.py --sources all
```

### [Processing Module](./processing/README.md)
Converts PDFs to embeddings using Docling and ChromaDB.

### [RAG API](./rag_api/README.md)
FastAPI service for semantic search with Redis caching.

## Full Pipeline

1. **Harvest**: Download papers and guidelines
2. **Process**: Convert to embeddings  
3. **Serve**: Start API for search

```bash
# 1. harvest
cd rxiv_harvest && python download.py
cd ../guideline_harvest && python crawl.py --sources all

# 2. process  
cd ../processing && python process_papers.py ../rxiv_harvest/data/pdfs/biorxiv/ processed_papers/

# 3. serve
cd ../rag_api && docker-compose up -d
```

## Use Cases

- Academic research and literature review
- RAG-enhanced chatbots  
- Medical research support
- Literature trend analysis

## Requirements

- Python 3.10+
- NVIDIA GPU (recommended)
- Docker & Docker Compose

## Installation

```bash
git clone https://github.com/your-org/rxiv.git
cd rxiv

# Install each component
cd rxiv_harvest && pip install -r requirements.txt && cd ..
cd guideline_harvest && pip install -r requirements.txt && cd ..
cd processing && pip install -r requirements.txt && cd ..  
cd rag_api && pip install -r requirements.txt && cd ..
```

## Project Structure

```
delphea_kb/
├── rxiv_harvest/          # Academic papers
├── guideline_harvest/     # Clinical guidelines  
├── processing/            # PDF to embeddings
├── rag_api/              # Search API
└── README.md
```

## Contact

Haining Wang (hw56@iu.edu)
