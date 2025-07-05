# RAG API

FastAPI service for semantic search of academic papers and clinical guidelines.

## Features

- **FastAPI**: Async REST API
- **Vector search**: ChromaDB and FAISS backends
- **Redis caching**: Fast response times
- **GPU support**: Accelerated embeddings
- **Multi-model**: Different embedding models available

## Installation

### Local Development
```bash
cd rag_api
pip install -r requirements.txt

# install processing dependencies
pip install -r ../processing/requirements.txt

# start Redis (required for caching)
redis-server

# Start the API
python main.py
```

### Docker Deployment (Recommended)
```bash
# full stack with monitoring
docker-compose up -d

# API only
docker build -t rag-api .
docker run --gpus all -p 8000:8000 -v $(pwd)/../processed_papers:/app/processed_papers:ro rag-api
```

## Quick Start

### Basic API Usage

#### Start the service
```bash
# development mode
python main.py

# production mode  
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

#### Search for papers
```bash
# basic search
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "CRISPR gene editing applications",
    "model": "scientific",
    "top_k": 5
  }'

# advanced search with metadata and content
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning in medical diagnosis",
    "model": "primary", 
    "top_k": 10,
    "include_metadata": true,
    "include_content": true,
    "similarity_threshold": 0.2
  }'
```

### Python Client Example

```python
import asyncio
from client_example import RAGClient

async def search_papers():
    client = RAGClient("http://localhost:8000")
    
    # search with full metadata and content
    results = await client.search(
        query="COVID-19 vaccine effectiveness",
        model="scientific",
        top_k=10,
        include_metadata=True,
        include_content=True
    )
    
    print(f"Found {results['total_results']} papers")
    for result in results['results']:
        print(f"Score: {result['similarity_score']:.3f}")
        print(f"Title: {result['metadata'].get('title', 'N/A')}")
        print(f"Abstract: {result['text'][:200]}...")
        print("---")
    
    await client.close()

# run the example
asyncio.run(search_papers())
```

## API Reference

### Endpoints

#### `POST /search`
Semantic search across academic papers.

**Request Body:**
```json
{
  "query": "machine learning in healthcare",
  "model": "primary",
  "top_k": 10,
  "include_metadata": true,
  "include_content": true,  
  "similarity_threshold": 0.1
}
```

**Response:**
```json
{
  "query": "machine learning in healthcare",
  "model": "primary",
  "results": [
    {
      "chunk_id": "chunk_123",
      "similarity_score": 0.89,
      "text": "Machine learning algorithms have shown...",
      "chunk_type": "section",
      "metadata": {
        "section_title": "Results",
        "source_document": "paper_456.pdf",
        "chunk_length": 245,
        "type": "text"
      },
      "source_document": "paper_456.pdf",
      "original_content": {
        "sections": [...],
        "tables": [...],
        "figures": [...],
        "full_text": "..."
      }
    }
  ],
  "total_results": 10,
  "search_time": 0.045,
  "cached": false
}
```

#### `GET /health`
Service health check.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-05T14:30:22Z",
  "version": "1.0.0",
  "models_loaded": ["primary", "scientific", "dense"],
  "vector_stores": ["primary: 125,432 docs", "scientific: 125,432 docs"],
  "cache_status": "connected"
}
```

#### `GET /models`
List available embedding models.

**Response:**
```json
{
  "available_models": {
    "primary": {
      "document_count": 125432,
      "status": "available"
    },
    "scientific": {
      "document_count": 125432, 
      "status": "available"
    }
  },
  "default_model": "primary"
}
```

#### `GET /metrics`
Prometheus metrics endpoint.

### Configuration

#### API Configuration (`rag_config.json`)
```json
{
  "api": {
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 4
  },
  "embedding": {
    "models": {
      "primary": "all-MiniLM-L6-v2",
      "scientific": "allenai/scibert_scivocab_uncased",
      "dense": "sentence-transformers/all-mpnet-base-v2"
    },
    "batch_size": 128
  },
  "vector_stores": {
    "chromadb_path": "../processed_papers/embeddings"
  },
  "cache": {
    "redis_url": "redis://localhost:6379",
    "ttl": 3600
  }
}
```

## Architecture

### System Architecture
```
Client → Load Balancer → RAG API → Vector Store
   ↓                       ↓           ↓
Cache ←                GPU Models → Content Store
   ↓                       ↓           ↓  
Redis                   H100/A100    ChromaDB/FAISS
```

### File Structure
```
rag_api/
├── main.py                # FastAPI application
├── client_example.py      # Python client example
├── rag_config.json       # API configuration
├── requirements.txt      # API dependencies  
├── Dockerfile            # Container definition
├── docker-compose.yml    # Full stack deployment
├── prometheus.yml        # Metrics configuration
└── README.md            # This file
```

### Data Flow
1. **Query received** → API validates request
2. **Cache check** → Redis lookup for cached results  
3. **Embedding generation** → GPU-accelerated query embedding
4. **Vector search** → ChromaDB/FAISS similarity search
5. **Content retrieval** → Original document content lookup
6. **Response assembly** → Metadata + content + scoring
7. **Cache storage** → Store result for future requests

## Performance

### Benchmarks (H100 80GB)

| Operation | Latency | Throughput |
|-----------|---------|------------|
| Cached query | <10ms | 10,000+ QPS |
| Uncached query | 50-200ms | 500+ QPS |
| Embedding generation | 5-15ms | 5,000+ texts/sec |
| Vector search | 10-50ms | 1,000+ searches/sec |

### Optimization Tips

#### GPU Performance
```json
{
  "embedding": {
    "batch_size": 128,     // optimize for H100
    "normalize_embeddings": true,
    "models": {
      "primary": "all-MiniLM-L6-v2"  // fast model for production
    }
  }
}
```

#### Caching Strategy
```json
{
  "cache": {
    "ttl": 3600,          // 1 hour cache
    "key_prefix": "rag_api:",
    "max_memory": "2gb"
  }
}
```

#### Scaling Configuration
```bash
# multi-worker deployment
uvicorn main:app --workers 8 --worker-class uvicorn.workers.UvicornWorker

# load balancing with nginx
nginx -c nginx.conf
```

## Deployment

### Docker Compose (Recommended)
```bash
# full production stack
docker-compose up -d

# includes:
# - RAG API with GPU support
# - Redis cache
# - Prometheus monitoring  
# - Grafana dashboards
```

### Kubernetes
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-api
  template:
    spec:
      containers:
      - name: rag-api
        image: rag-api:latest
        resources:
          limits:
            nvidia.com/gpu: 1
          requests:
            memory: "8Gi"
            cpu: "4"
```

### Environment Variables
```bash
# production overrides
export RAG_CONFIG_PATH=/app/config/production.json
export REDIS_URL=redis://redis-cluster:6379  
export LOG_LEVEL=INFO
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

## Monitoring

### Prometheus Metrics
- `rag_api_requests_total`: Total requests by endpoint
- `rag_api_request_duration_seconds`: Request latency
- `rag_api_searches_total`: Search requests by model
- `rag_api_cache_hits_total`: Cache hit rate
- `rag_api_gpu_utilization`: GPU utilization percentage

### Grafana Dashboards
Access at `http://localhost:3000` (admin/admin):
- **API Performance**: Request rates, latency, error rates
- **Search Analytics**: Query patterns, model usage, result quality
- **System Health**: GPU utilization, memory usage, cache performance
- **Business Metrics**: Popular queries, user engagement, content coverage

### Alerting
```yaml
# example alert rules
groups:
- name: rag_api
  rules:
  - alert: HighErrorRate
    expr: rate(rag_api_requests_total{status=~"5.."}[5m]) > 0.1
    for: 2m
    annotations:
      summary: "High error rate detected"
      
  - alert: HighLatency  
    expr: histogram_quantile(0.95, rag_api_request_duration_seconds) > 1.0
    for: 5m
    annotations:
      summary: "High latency detected"
```

## Security

### API Security
- **Input validation**: Pydantic models with strict validation
- **Rate limiting**: Request rate limits per client
- **CORS**: Configurable cross-origin resource sharing
- **Authentication**: Ready for JWT/OAuth integration

### Infrastructure Security  
- **Non-root containers**: Security-hardened Docker images
- **Network isolation**: Container network security
- **Secret management**: Environment-based configuration
- **TLS encryption**: HTTPS termination at load balancer

## Troubleshooting

### Common Issues

**Issue**: API not finding vector stores
```bash
# check path configuration
ls -la ../processed_papers/embeddings/
# update rag_config.json paths
```

**Issue**: GPU not detected
```bash
# verify GPU access
nvidia-smi
# check Docker GPU support
docker run --gpus all nvidia/cuda:12.1-base nvidia-smi
```

**Issue**: Redis connection errors
```bash
# Check Redis status
redis-cli ping
# update Redis URL in config
```

**Issue**: High memory usage
```bash
# monitor memory usage
docker stats
# reduce batch sizes in config
```

### Performance Debugging

```bash
# API performance
curl -X GET "http://localhost:8000/metrics"

# Redis performance  
redis-cli --latency

# GPU utilization
nvidia-smi -l 1

# container resources
docker stats rag-api
```


## Integration Examples

### With Jupyter Notebooks
```python
import requests

# search from notebook
response = requests.post("http://localhost:8000/search", json={
    "query": "transformer architecture attention mechanism",
    "model": "scientific",
    "top_k": 5
})

results = response.json()
for result in results['results']:
    print(f"Score: {result['similarity_score']}")
    print(f"Content: {result['text'][:200]}...")
```

### With Web Applications
```javascript
// frontend integration
const searchPapers = async (query) => {
  const response = await fetch('http://localhost:8000/search', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query: query,
      model: 'primary',
      top_k: 10
    })
  });
  
  return await response.json();
};
```

### With ChatGPT/Claude
```python
# RAG-enhanced chat
async def enhanced_chat(user_query):
    # search relevant papers
    search_results = await rag_client.search(user_query, top_k=5)
    
    # build context
    context = "\n".join([r['text'] for r in search_results['results']])
    
    # send to LLM with context
    prompt = f"Context: {context}\n\nQuestion: {user_query}"
    return await llm_client.generate(prompt)
```

## Contact

For questions or issues with the RAG API:
- **Author**: Haining Wang (hw56@iu.edu)
- **Performance**: Optimized for NVIDIA H100/A100 GPUs
- **Repository**: https://github.com/your-org/rxiv

## License

This API is designed for academic research purposes. Please respect the licensing terms of the underlying models.