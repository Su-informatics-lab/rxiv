"""Embedding pipeline for vector storage and retrieval.

This module creates high-quality embeddings optimized for academic paper retrieval:
- Multiple embedding models for different use cases
- Batch processing with memory optimization
- Vector database integration (ChromaDB, FAISS, etc.)
- Metadata-rich vector storage for enhanced retrieval
- Incremental updates and version management

Input:
    - Chunked content from academic papers
    - Embedding model configurations
    - Vector storage specifications

Output:
    - High-dimensional vector embeddings
    - Vector database with searchable indices
    - Metadata mappings for retrieval enhancement
    - Performance metrics and quality assessments

Author: Haining Wang <hw56@iu.edu>
"""

import json
import logging
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import chromadb
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


class EmbeddingPipeline:
    """Comprehensive embedding pipeline for academic paper retrieval."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the embedding pipeline.
        
        Args:
            config: Configuration dictionary for embedding settings
        """
        self.config = config or self._get_default_config()
        self.logger = self._setup_logging()
        self.embedding_models = self._load_embedding_models()
        self.vector_stores = {}
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for embedding pipeline.
        
        Returns:
            Default configuration dictionary
        """
        return {
            # Embedding models
            "models": {
                "primary": "all-MiniLM-L6-v2",  # Fast, good quality
                "scientific": "allenai/scibert_scivocab_uncased",  # Scientific domain
                "dense": "sentence-transformers/all-mpnet-base-v2",  # High quality
            },
            
            # Processing settings
            "batch_size": 32,
            "max_workers": 4,
            "normalize_embeddings": True,
            "cache_embeddings": True,
            
            # Vector storage
            "vector_stores": ["chromadb", "faiss"],
            "chromadb_path": "vector_db/chroma",
            "faiss_index_type": "IVF",
            "faiss_nlist": 100,
            
            # Quality settings
            "embedding_dimension": 384,  # For MiniLM
            "similarity_threshold": 0.5,
            "duplicate_detection": True,
            
            # Storage settings
            "save_embeddings": True,
            "embeddings_dir": "embeddings",
            "compression": True,
            
            # Logging
            "log_level": "INFO",
            "detailed_metrics": True,
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for embedding operations.
        
        Returns:
            Configured logger instance
        """
        log_dir = Path("logs/processing")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"embeddings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=getattr(logging, self.config["log_level"]),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def _load_embedding_models(self) -> Dict[str, SentenceTransformer]:
        """Load all configured embedding models with GPU optimization.
        
        Returns:
            Dictionary of loaded embedding models
        """
        models = {}
        
        # Check for GPU availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            self.logger.info(f"GPU detected: {torch.cuda.get_device_name()}")
            self.logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        else:
            self.logger.warning("No GPU detected, using CPU")
        
        for model_name, model_path in self.config["models"].items():
            try:
                self.logger.info(f"Loading embedding model: {model_name} ({model_path}) on {device}")
                model = SentenceTransformer(model_path, device=device)
                
                # Optimize for GPU if available
                if device == "cuda":
                    model = model.half()  # Use FP16 for faster inference
                    # Enable optimizations for H100
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cuda.matmul.allow_tf32 = True
                
                models[model_name] = model
                self.logger.info(f"Successfully loaded {model_name} on {device}")
            except Exception as e:
                self.logger.error(f"Failed to load {model_name}: {e}")
        
        if not models:
            raise RuntimeError("No embedding models could be loaded")
        
        return models
    
    def process_chunks(self, chunks_data: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
        """Process chunks to create embeddings and vector storage.
        
        Args:
            chunks_data: Chunked document data
            output_dir: Directory to save embeddings and vector stores
            
        Returns:
            Processing results and metrics
        """
        start_time = time.time()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            "document_id": chunks_data.get("document_id", "unknown"),
            "timestamp": datetime.now().isoformat(),
            "config": self.config,
            "embeddings": {},
            "vector_stores": {},
            "statistics": {}
        }
        
        chunks = chunks_data.get("chunks", [])
        if not chunks:
            self.logger.warning("No chunks found in input data")
            return results
        
        self.logger.info(f"Processing {len(chunks)} chunks with {len(self.embedding_models)} models")
        
        try:
            # Extract texts and metadata
            texts, metadata = self._prepare_chunk_data(chunks)
            
            # Generate embeddings with each model
            for model_name, model in self.embedding_models.items():
                self.logger.info(f"Generating embeddings with {model_name}")
                
                embeddings, model_stats = self._generate_embeddings(texts, model, model_name)
                
                # Save embeddings
                embeddings_file = output_dir / f"embeddings_{model_name}.npz"
                self._save_embeddings(embeddings, metadata, embeddings_file)
                
                results["embeddings"][model_name] = {
                    "file": str(embeddings_file),
                    "shape": embeddings.shape,
                    "statistics": model_stats
                }
                
                # Create vector stores
                if "chromadb" in self.config["vector_stores"]:
                    chroma_path = output_dir / f"chroma_{model_name}"
                    chroma_stats = self._create_chromadb_store(
                        embeddings, metadata, texts, chroma_path, model_name
                    )
                    results["vector_stores"][f"chromadb_{model_name}"] = chroma_stats
                
                if "faiss" in self.config["vector_stores"]:
                    faiss_path = output_dir / f"faiss_{model_name}.index"
                    faiss_stats = self._create_faiss_index(embeddings, faiss_path, model_name)
                    results["vector_stores"][f"faiss_{model_name}"] = faiss_stats
            
            # Generate overall statistics
            results["statistics"] = self._generate_processing_stats(
                chunks, results, start_time
            )
            
            # Save results summary
            results_file = output_dir / "embedding_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"Embedding processing completed in {time.time() - start_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Error in embedding processing: {e}")
            results["error"] = str(e)
        
        return results
    
    def _prepare_chunk_data(self, chunks: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Prepare chunk data for embedding generation.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Tuple of (texts, metadata)
        """
        texts = []
        metadata = []
        
        for i, chunk in enumerate(chunks):
            text = chunk.get("text", "").strip()
            if not text:
                continue
            
            texts.append(text)
            
            # Prepare metadata for storage
            chunk_metadata = {
                "chunk_id": chunk.get("id", f"chunk_{i}"),
                "type": chunk.get("type", "text"),
                "position": chunk.get("position", i),
                "length": len(text.split()),
                **chunk.get("metadata", {})
            }
            metadata.append(chunk_metadata)
        
        return texts, metadata
    
    def _generate_embeddings(self, texts: List[str], model: SentenceTransformer, 
                           model_name: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate embeddings for texts using specified model.
        
        Args:
            texts: List of texts to embed
            model: SentenceTransformer model
            model_name: Name of the model for logging
            
        Returns:
            Tuple of (embeddings array, statistics)
        """
        start_time = time.time()
        batch_size = self.config["batch_size"]
        
        embeddings = []
        
        # Process in batches with progress bar
        with tqdm(total=len(texts), desc=f"Embedding ({model_name})") as pbar:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                try:
                    batch_embeddings = model.encode(
                        batch_texts,
                        normalize_embeddings=self.config["normalize_embeddings"],
                        show_progress_bar=False
                    )
                    embeddings.append(batch_embeddings)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to embed batch {i}: {e}")
                    # Create zero embeddings for failed batch
                    zero_embeddings = np.zeros((len(batch_texts), model.get_sentence_embedding_dimension()))
                    embeddings.append(zero_embeddings)
                
                pbar.update(len(batch_texts))
        
        # Combine all embeddings
        all_embeddings = np.vstack(embeddings)
        
        # Generate statistics
        stats = {
            "model_name": model_name,
            "embedding_dimension": all_embeddings.shape[1],
            "total_embeddings": all_embeddings.shape[0],
            "processing_time": time.time() - start_time,
            "embeddings_per_second": len(texts) / (time.time() - start_time),
            "memory_usage_mb": all_embeddings.nbytes / (1024 * 1024),
        }
        
        # Quality metrics
        if self.config["detailed_metrics"]:
            stats.update(self._calculate_embedding_quality_metrics(all_embeddings))
        
        return all_embeddings, stats
    
    def _calculate_embedding_quality_metrics(self, embeddings: np.ndarray) -> Dict[str, float]:
        """Calculate quality metrics for embeddings.
        
        Args:
            embeddings: Numpy array of embeddings
            
        Returns:
            Dictionary of quality metrics
        """
        metrics = {}
        
        try:
            # Basic statistics
            metrics["mean_norm"] = float(np.mean(np.linalg.norm(embeddings, axis=1)))
            metrics["std_norm"] = float(np.std(np.linalg.norm(embeddings, axis=1)))
            
            # Diversity metrics (sample-based for large datasets)
            sample_size = min(1000, len(embeddings))
            sample_idx = np.random.choice(len(embeddings), sample_size, replace=False)
            sample_embeddings = embeddings[sample_idx]
            
            # Average pairwise similarity
            similarities = cosine_similarity(sample_embeddings)
            # Exclude diagonal (self-similarity)
            mask = ~np.eye(similarities.shape[0], dtype=bool)
            avg_similarity = float(np.mean(similarities[mask]))
            
            metrics["avg_pairwise_similarity"] = avg_similarity
            metrics["embedding_diversity"] = 1.0 - avg_similarity
            
        except Exception as e:
            self.logger.warning(f"Error calculating quality metrics: {e}")
        
        return metrics
    
    def _save_embeddings(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]], 
                        output_file: Path) -> None:
        """Save embeddings and metadata to file.
        
        Args:
            embeddings: Numpy array of embeddings
            metadata: List of metadata dictionaries
            output_file: Output file path
        """
        try:
            if self.config["compression"]:
                np.savez_compressed(
                    output_file,
                    embeddings=embeddings,
                    metadata=metadata
                )
            else:
                np.savez(
                    output_file,
                    embeddings=embeddings,
                    metadata=metadata
                )
            
            self.logger.info(f"Saved embeddings to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save embeddings: {e}")
    
    def _create_chromadb_store(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]], 
                              texts: List[str], output_path: Path, model_name: str) -> Dict[str, Any]:
        """Create ChromaDB vector store.
        
        Args:
            embeddings: Numpy array of embeddings
            metadata: List of metadata dictionaries
            texts: List of original texts
            output_path: Output directory path
            model_name: Name of the embedding model
            
        Returns:
            ChromaDB creation statistics
        """
        start_time = time.time()
        
        try:
            # Initialize ChromaDB client
            client = chromadb.PersistentClient(path=str(output_path))
            
            # Create or get collection
            collection_name = f"academic_papers_{model_name}"
            collection = client.get_or_create_collection(
                name=collection_name,
                metadata={"model": model_name, "created": datetime.now().isoformat()}
            )
            
            # Prepare data for ChromaDB
            ids = [f"chunk_{i}" for i in range(len(embeddings))]
            
            # Convert metadata to ChromaDB format (strings only)
            chroma_metadata = []
            for meta in metadata:
                chroma_meta = {}
                for key, value in meta.items():
                    if isinstance(value, (str, int, float, bool)):
                        chroma_meta[key] = str(value)
                    elif isinstance(value, dict):
                        chroma_meta[key] = json.dumps(value)
                    else:
                        chroma_meta[key] = str(value)
                chroma_metadata.append(chroma_meta)
            
            # Add embeddings to collection
            collection.add(
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=chroma_metadata,
                ids=ids
            )
            
            stats = {
                "store_type": "chromadb",
                "collection_name": collection_name,
                "path": str(output_path),
                "total_vectors": len(embeddings),
                "creation_time": time.time() - start_time,
                "vector_dimension": embeddings.shape[1]
            }
            
            self.logger.info(f"Created ChromaDB store at {output_path}")
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to create ChromaDB store: {e}")
            return {"error": str(e)}
    
    def _create_faiss_index(self, embeddings: np.ndarray, output_path: Path, 
                           model_name: str) -> Dict[str, Any]:
        """Create FAISS vector index.
        
        Args:
            embeddings: Numpy array of embeddings
            output_path: Output file path
            model_name: Name of the embedding model
            
        Returns:
            FAISS creation statistics
        """
        start_time = time.time()
        
        try:
            dimension = embeddings.shape[1]
            
            # Choose index type based on size
            if len(embeddings) < 1000:
                # Flat index for small datasets
                index = faiss.IndexFlatIP(dimension)
            else:
                # IVF index for larger datasets
                nlist = min(self.config["faiss_nlist"], len(embeddings) // 10)
                quantizer = faiss.IndexFlatIP(dimension)
                index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
                
                # Train the index
                index.train(embeddings.astype(np.float32))
            
            # Add embeddings to index
            index.add(embeddings.astype(np.float32))
            
            # Save index
            faiss.write_index(index, str(output_path))
            
            stats = {
                "store_type": "faiss",
                "index_type": type(index).__name__,
                "path": str(output_path),
                "total_vectors": index.ntotal,
                "creation_time": time.time() - start_time,
                "vector_dimension": dimension
            }
            
            self.logger.info(f"Created FAISS index at {output_path}")
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to create FAISS index: {e}")
            return {"error": str(e)}
    
    def _generate_processing_stats(self, chunks: List[Dict[str, Any]], 
                                 results: Dict[str, Any], start_time: float) -> Dict[str, Any]:
        """Generate comprehensive processing statistics.
        
        Args:
            chunks: Original chunks data
            results: Processing results
            start_time: Processing start timestamp
            
        Returns:
            Comprehensive statistics dictionary
        """
        total_time = time.time() - start_time
        
        stats = {
            "processing_summary": {
                "total_chunks": len(chunks),
                "total_models": len(self.embedding_models),
                "total_time_seconds": total_time,
                "chunks_per_second": len(chunks) / total_time if total_time > 0 else 0,
            },
            "chunk_statistics": {},
            "model_performance": {},
            "storage_summary": {}
        }
        
        # Chunk type analysis
        chunk_types = {}
        chunk_lengths = []
        
        for chunk in chunks:
            chunk_type = chunk.get("type", "text")
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
            
            text_length = len(chunk.get("text", "").split())
            chunk_lengths.append(text_length)
        
        stats["chunk_statistics"] = {
            "types": chunk_types,
            "length_stats": {
                "mean": float(np.mean(chunk_lengths)) if chunk_lengths else 0,
                "std": float(np.std(chunk_lengths)) if chunk_lengths else 0,
                "min": int(min(chunk_lengths)) if chunk_lengths else 0,
                "max": int(max(chunk_lengths)) if chunk_lengths else 0,
            }
        }
        
        # Model performance summary
        for model_name in results.get("embeddings", {}):
            model_stats = results["embeddings"][model_name].get("statistics", {})
            stats["model_performance"][model_name] = {
                "processing_time": model_stats.get("processing_time", 0),
                "embeddings_per_second": model_stats.get("embeddings_per_second", 0),
                "memory_usage_mb": model_stats.get("memory_usage_mb", 0),
            }
        
        # Storage summary
        storage_info = {}
        for store_name, store_info in results.get("vector_stores", {}).items():
            if "error" not in store_info:
                storage_info[store_name] = {
                    "total_vectors": store_info.get("total_vectors", 0),
                    "creation_time": store_info.get("creation_time", 0),
                }
        
        stats["storage_summary"] = storage_info
        
        return stats
    
    def load_embeddings(self, embeddings_file: Path) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Load embeddings and metadata from file.
        
        Args:
            embeddings_file: Path to embeddings file
            
        Returns:
            Tuple of (embeddings, metadata)
        """
        try:
            data = np.load(embeddings_file, allow_pickle=True)
            embeddings = data["embeddings"]
            metadata = data["metadata"].tolist()
            
            self.logger.info(f"Loaded {len(embeddings)} embeddings from {embeddings_file}")
            return embeddings, metadata
            
        except Exception as e:
            self.logger.error(f"Failed to load embeddings: {e}")
            raise
    
    def search_similar(self, query_text: str, model_name: str, top_k: int = 10, 
                      chroma_path: Optional[Path] = None) -> List[Dict[str, Any]]:
        """Search for similar chunks using embeddings.
        
        Args:
            query_text: Query text to search for
            model_name: Name of the embedding model to use
            top_k: Number of top results to return
            chroma_path: Path to ChromaDB store
            
        Returns:
            List of similar chunks with scores
        """
        try:
            if model_name not in self.embedding_models:
                raise ValueError(f"Model {model_name} not available")
            
            # Generate query embedding
            model = self.embedding_models[model_name]
            query_embedding = model.encode([query_text], normalize_embeddings=True)[0]
            
            # Search in ChromaDB if available
            if chroma_path and chroma_path.exists():
                client = chromadb.PersistentClient(path=str(chroma_path))
                collection = client.get_collection(f"academic_papers_{model_name}")
                
                results = collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"]
                )
                
                similar_chunks = []
                for i in range(len(results["documents"][0])):
                    chunk = {
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "similarity_score": 1.0 - results["distances"][0][i],  # Convert distance to similarity
                    }
                    similar_chunks.append(chunk)
                
                return similar_chunks
            
            else:
                self.logger.warning("ChromaDB not available for search")
                return []
                
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []


def main():
    """Main function for standalone embedding creation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create embeddings for chunked content")
    parser.add_argument("input_file", help="JSON file with chunked content")
    parser.add_argument("output_dir", help="Directory to save embeddings and vector stores")
    parser.add_argument("--config", help="Configuration file for embedding pipeline")
    parser.add_argument("--models", nargs="+", help="Specific models to use", 
                       default=["all-MiniLM-L6-v2"])
    
    args = parser.parse_args()
    
    # Load configuration
    config = None
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # Override models if specified
    if args.models:
        config["models"] = {f"model_{i}": model for i, model in enumerate(args.models)}
    
    # Initialize pipeline
    pipeline = EmbeddingPipeline(config)
    
    # Load chunked data
    with open(args.input_file, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)
    
    # Process chunks
    output_path = Path(args.output_dir)
    results = pipeline.process_chunks(chunks_data, output_path)
    
    # Print summary
    stats = results.get("statistics", {})
    print(f"Processing completed:")
    print(f"  Total chunks: {stats.get('processing_summary', {}).get('total_chunks', 0)}")
    print(f"  Total time: {stats.get('processing_summary', {}).get('total_time_seconds', 0):.2f}s")
    print(f"  Models used: {list(results.get('embeddings', {}).keys())}")
    print(f"  Vector stores: {list(results.get('vector_stores', {}).keys())}")


if __name__ == "__main__":
    main()