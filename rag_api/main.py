"""Production-ready RAG API service for academic paper search.

This module provides a high-performance REST API for retrieving and querying 
academic papers using industrial best practices:
- FastAPI with async/await for high concurrency
- Multiple embedding models and vector databases
- Comprehensive caching with Redis
- Production monitoring and logging
- Optimized for H100 GPU acceleration
- Full metadata and content preservation

Author: Haining Wang <hw56@iu.edu>
"""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

import chromadb
import redis.asyncio as aioredis
import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

# Import our processing modules
import sys
sys.path.append(str(Path(__file__).parent.parent / "processing"))

from create_embeddings import EmbeddingPipeline
from chunk_content import AcademicPaperChunker


# Prometheus metrics
REQUEST_COUNT = Counter('rag_api_requests_total', 'Total API requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('rag_api_request_duration_seconds', 'Request duration')
SEARCH_COUNT = Counter('rag_api_searches_total', 'Total search requests', ['model'])
SEARCH_DURATION = Histogram('rag_api_search_duration_seconds', 'Search duration', ['model'])


class SearchRequest(BaseModel):
    """Search request model with validation."""
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    model: str = Field(default="primary", description="Embedding model to use")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    include_metadata: bool = Field(default=True, description="Include full metadata in results")
    include_content: bool = Field(default=True, description="Include original content")
    similarity_threshold: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum similarity score")


class SearchResult(BaseModel):
    """Search result model."""
    chunk_id: str
    similarity_score: float
    text: str
    chunk_type: str
    metadata: Dict[str, Any]
    source_document: Optional[str] = None
    original_content: Optional[Dict[str, Any]] = None


class SearchResponse(BaseModel):
    """Search response model."""
    query: str
    model: str
    results: List[SearchResult]
    total_results: int
    search_time: float
    cached: bool = False


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    version: str
    models_loaded: List[str]
    vector_stores: List[str]
    cache_status: str


class RAGService:
    """High-performance RAG service with production features."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the RAG service.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.cache: Optional[aioredis.Redis] = None
        self.embedding_pipeline: Optional[EmbeddingPipeline] = None
        self.vector_stores: Dict[str, Any] = {}
        self.content_lookup: Dict[str, Dict[str, Any]] = {}
        
    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            "api": {
                "host": "0.0.0.0",
                "port": 8000,
                "workers": 1,
                "reload": False,
                "title": "Academic Paper RAG API",
                "version": "1.0.0"
            },
            "embedding": {
                "models": {
                    "primary": "all-MiniLM-L6-v2",
                    "scientific": "allenai/scibert_scivocab_uncased",
                    "dense": "sentence-transformers/all-mpnet-base-v2"
                },
                "batch_size": 64,  # Larger for H100
                "normalize_embeddings": True
            },
            "vector_stores": {
                "chromadb_path": "processed_papers/embeddings",
                "default_collection": "academic_papers"
            },
            "cache": {
                "redis_url": "redis://localhost:6379",
                "ttl": 3600,  # 1 hour
                "key_prefix": "rag_api:"
            },
            "content": {
                "content_dir": "processed_papers/extracted",
                "chunk_dir": "processed_papers/chunked"
            },
            "performance": {
                "max_concurrent_requests": 100,
                "request_timeout": 30.0,
                "cache_query_results": True
            },
            "logging": {
                "level": "INFO",
                "format": "json"
            }
        }
        
        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                # Deep merge configs
                self._merge_configs(default_config, user_config)
        
        return default_config
    
    def _merge_configs(self, default: Dict, user: Dict) -> None:
        """Deep merge user config into default config."""
        for key, value in user.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._merge_configs(default[key], value)
            else:
                default[key] = value
    
    def _setup_logging(self) -> structlog.stdlib.BoundLogger:
        """Setup structured logging."""
        log_level = self.config["logging"]["level"]
        
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer() if self.config["logging"]["format"] == "json" 
                else structlog.dev.ConsoleRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        logging.basicConfig(level=getattr(logging, log_level))
        return structlog.get_logger()
    
    async def initialize(self):
        """Initialize all service components."""
        self.logger.info("Initializing RAG service")
        
        # Initialize Redis cache
        try:
            self.cache = aioredis.from_url(
                self.config["cache"]["redis_url"],
                encoding="utf-8",
                decode_responses=True
            )
            await self.cache.ping()
            self.logger.info("Redis cache connected")
        except Exception as e:
            self.logger.warning("Redis not available, running without cache", error=str(e))
            self.cache = None
        
        # Initialize embedding pipeline
        try:
            self.embedding_pipeline = EmbeddingPipeline(self.config["embedding"])
            self.logger.info("Embedding pipeline initialized", 
                           models=list(self.config["embedding"]["models"].keys()))
        except Exception as e:
            self.logger.error("Failed to initialize embedding pipeline", error=str(e))
            raise
        
        # Load vector stores
        await self._load_vector_stores()
        
        # Load content lookup
        await self._load_content_lookup()
        
        self.logger.info("RAG service initialization complete")
    
    async def _load_vector_stores(self):
        """Load ChromaDB vector stores for each model."""
        chromadb_path = Path(self.config["vector_stores"]["chromadb_path"])
        
        for model_name in self.config["embedding"]["models"].keys():
            store_path = chromadb_path / f"chroma_{model_name}"
            
            if store_path.exists():
                try:
                    client = chromadb.PersistentClient(path=str(store_path))
                    collection = client.get_collection(f"academic_papers_{model_name}")
                    self.vector_stores[model_name] = collection
                    
                    count = collection.count()
                    self.logger.info("Loaded vector store", 
                                   model=model_name, 
                                   documents=count,
                                   path=str(store_path))
                except Exception as e:
                    self.logger.error("Failed to load vector store", 
                                    model=model_name, 
                                    error=str(e))
            else:
                self.logger.warning("Vector store not found", 
                                  model=model_name, 
                                  path=str(store_path))
    
    async def _load_content_lookup(self):
        """Load content lookup for retrieving original documents."""
        content_dir = Path(self.config["content"]["content_dir"])
        chunk_dir = Path(self.config["content"]["chunk_dir"])
        
        # Build lookup from chunk ID to original content
        if chunk_dir.exists():
            for chunk_file in chunk_dir.glob("**/*_chunked.json"):
                try:
                    with open(chunk_file, 'r', encoding='utf-8') as f:
                        chunk_data = json.load(f)
                    
                    document_id = chunk_data.get("document_id", chunk_file.stem)
                    
                    # Load original content
                    content_file = content_dir / f"{document_id}.json"
                    if content_file.exists():
                        with open(content_file, 'r', encoding='utf-8') as f:
                            original_content = json.load(f)
                        
                        # Map each chunk to its original content
                        for chunk in chunk_data.get("chunks", []):
                            chunk_id = chunk.get("id", "")
                            if chunk_id:
                                self.content_lookup[chunk_id] = {
                                    "original_content": original_content,
                                    "chunk_data": chunk_data,
                                    "document_id": document_id
                                }
                
                except Exception as e:
                    self.logger.warning("Failed to load content", 
                                      file=str(chunk_file), 
                                      error=str(e))
        
        self.logger.info("Content lookup loaded", 
                       total_chunks=len(self.content_lookup))
    
    async def search(self, request: SearchRequest) -> SearchResponse:
        """Perform semantic search on academic papers.
        
        Args:
            request: Search request parameters
            
        Returns:
            Search response with results
        """
        start_time = time.time()
        
        # Update metrics
        SEARCH_COUNT.labels(model=request.model).inc()
        
        # Check cache first
        cached_result = None
        if self.cache and self.config["performance"]["cache_query_results"]:
            cache_key = self._get_cache_key(request)
            cached_result = await self.cache.get(cache_key)
            
            if cached_result:
                try:
                    result = SearchResponse.parse_raw(cached_result)
                    result.cached = True
                    result.search_time = time.time() - start_time
                    return result
                except Exception as e:
                    self.logger.warning("Failed to parse cached result", error=str(e))
        
        # Validate model
        if request.model not in self.vector_stores:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Model '{request.model}' not available. Available models: {list(self.vector_stores.keys())}"
            )
        
        try:
            # Generate query embedding
            model = self.embedding_pipeline.embedding_models[request.model]
            query_embedding = model.encode([request.query], normalize_embeddings=True)[0]
            
            # Search in vector store
            collection = self.vector_stores[request.model]
            
            search_results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=request.top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Process results
            results = []
            for i in range(len(search_results["documents"][0])):
                similarity_score = 1.0 - search_results["distances"][0][i]
                
                # Apply similarity threshold
                if similarity_score < request.similarity_threshold:
                    continue
                
                metadata = search_results["metadatas"][0][i]
                chunk_id = metadata.get("chunk_id", f"chunk_{i}")
                
                # Build result
                result = SearchResult(
                    chunk_id=chunk_id,
                    similarity_score=similarity_score,
                    text=search_results["documents"][0][i],
                    chunk_type=metadata.get("type", "text"),
                    metadata=metadata if request.include_metadata else {},
                    source_document=metadata.get("source_document")
                )
                
                # Add original content if requested and available
                if request.include_content and chunk_id in self.content_lookup:
                    content_info = self.content_lookup[chunk_id]
                    result.original_content = content_info["original_content"]
                
                results.append(result)
            
            search_time = time.time() - start_time
            
            response = SearchResponse(
                query=request.query,
                model=request.model,
                results=results,
                total_results=len(results),
                search_time=search_time,
                cached=False
            )
            
            # Cache the result
            if self.cache and self.config["performance"]["cache_query_results"]:
                cache_key = self._get_cache_key(request)
                try:
                    await self.cache.setex(
                        cache_key,
                        self.config["cache"]["ttl"],
                        response.json()
                    )
                except Exception as e:
                    self.logger.warning("Failed to cache result", error=str(e))
            
            # Update metrics
            SEARCH_DURATION.labels(model=request.model).observe(search_time)
            
            return response
            
        except Exception as e:
            self.logger.error("Search failed", 
                            query=request.query, 
                            model=request.model, 
                            error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Search failed: {str(e)}"
            )
    
    def _get_cache_key(self, request: SearchRequest) -> str:
        """Generate cache key for search request."""
        key_data = {
            "query": request.query,
            "model": request.model,
            "top_k": request.top_k,
            "include_metadata": request.include_metadata,
            "include_content": request.include_content,
            "similarity_threshold": request.similarity_threshold
        }
        key_hash = hash(json.dumps(key_data, sort_keys=True))
        return f"{self.config['cache']['key_prefix']}search:{key_hash}"
    
    async def get_health(self) -> HealthResponse:
        """Get service health status."""
        cache_status = "connected" if self.cache else "unavailable"
        if self.cache:
            try:
                await self.cache.ping()
            except:
                cache_status = "error"
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            version=self.config["api"]["version"],
            models_loaded=list(self.vector_stores.keys()),
            vector_stores=[f"{k}: {v.count()} docs" for k, v in self.vector_stores.items()],
            cache_status=cache_status
        )
    
    async def shutdown(self):
        """Cleanup service resources."""
        self.logger.info("Shutting down RAG service")
        
        if self.cache:
            await self.cache.close()


# Global service instance
rag_service = RAGService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    await rag_service.initialize()
    yield
    # Shutdown
    await rag_service.shutdown()


# Create FastAPI app
app = FastAPI(
    title="Academic Paper RAG API",
    description="High-performance semantic search API for academic papers",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def metrics_middleware(request, call_next):
    """Middleware for collecting metrics."""
    start_time = time.time()
    
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path
    ).inc()
    
    response = await call_next(request)
    
    REQUEST_DURATION.observe(time.time() - start_time)
    
    return response


@app.post("/search", response_model=SearchResponse)
async def search_papers(request: SearchRequest) -> SearchResponse:
    """Search academic papers using semantic similarity.
    
    This endpoint performs semantic search across the academic paper corpus
    using pre-computed embeddings and vector stores.
    """
    return await rag_service.search(request)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint for monitoring."""
    return await rag_service.get_health()


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint."""
    return JSONResponse(
        content=generate_latest().decode(),
        media_type="text/plain"
    )


@app.get("/models")
async def list_models() -> Dict[str, Any]:
    """List available embedding models and their statistics."""
    models_info = {}
    
    for model_name, collection in rag_service.vector_stores.items():
        try:
            count = collection.count()
            models_info[model_name] = {
                "document_count": count,
                "status": "available"
            }
        except Exception as e:
            models_info[model_name] = {
                "status": "error",
                "error": str(e)
            }
    
    return {
        "available_models": models_info,
        "default_model": "primary"
    }


if __name__ == "__main__":
    # Load configuration
    config_path = Path("rag_config.json")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    api_config = config.get("api", {})
    
    uvicorn.run(
        "main:app",
        host=api_config.get("host", "0.0.0.0"),
        port=api_config.get("port", 8000),
        workers=api_config.get("workers", 1),
        reload=api_config.get("reload", False),
        log_level="info"
    )