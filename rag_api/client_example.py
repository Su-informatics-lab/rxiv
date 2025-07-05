#!/usr/bin/env python3
"""Example client for the Academic Paper RAG API.

This script demonstrates how to interact with the RAG API for searching
academic papers with full metadata and content preservation.
"""

import asyncio
import json
import time
from typing import Dict, List, Any

import httpx


class RAGClient:
    """Client for interacting with the Academic Paper RAG API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the RAG client.
        
        Args:
            base_url: Base URL of the RAG API service
        """
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def search(
        self,
        query: str,
        model: str = "primary",
        top_k: int = 10,
        include_metadata: bool = True,
        include_content: bool = True,
        similarity_threshold: float = 0.0
    ) -> Dict[str, Any]:
        """Search for academic papers.
        
        Args:
            query: Search query text
            model: Embedding model to use
            top_k: Number of results to return
            include_metadata: Include full metadata in results
            include_content: Include original document content
            similarity_threshold: Minimum similarity score
            
        Returns:
            Search response with results
        """
        payload = {
            "query": query,
            "model": model,
            "top_k": top_k,
            "include_metadata": include_metadata,
            "include_content": include_content,
            "similarity_threshold": similarity_threshold
        }
        
        response = await self.client.post(f"{self.base_url}/search", json=payload)
        response.raise_for_status()
        
        return response.json()
    
    async def get_health(self) -> Dict[str, Any]:
        """Get API health status."""
        response = await self.client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    async def list_models(self) -> Dict[str, Any]:
        """List available embedding models."""
        response = await self.client.get(f"{self.base_url}/models")
        response.raise_for_status()
        return response.json()
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


async def main():
    """Example usage of the RAG client."""
    client = RAGClient()
    
    try:
        # Check API health
        print("🔍 Checking API health...")
        health = await client.get_health()
        print(f"✅ API Status: {health['status']}")
        print(f"📊 Models loaded: {', '.join(health['models_loaded'])}")
        print(f"💾 Cache status: {health['cache_status']}")
        print()
        
        # List available models
        print("🤖 Available models:")
        models = await client.list_models()
        for model_name, info in models["available_models"].items():
            if info["status"] == "available":
                print(f"  • {model_name}: {info['document_count']:,} documents")
            else:
                print(f"  • {model_name}: {info['status']}")
        print()
        
        # Example searches
        queries = [
            "COVID-19 vaccine effectiveness",
            "machine learning in medical diagnosis", 
            "CRISPR gene editing applications",
            "climate change and public health"
        ]
        
        for query in queries:
            print(f"🔎 Searching: '{query}'")
            start_time = time.time()
            
            results = await client.search(
                query=query,
                model="primary",
                top_k=5,
                include_metadata=True,
                include_content=False,  # Set to True to get original content
                similarity_threshold=0.1
            )
            
            search_time = time.time() - start_time
            
            print(f"⏱️  Search completed in {search_time:.3f}s (API: {results['search_time']:.3f}s)")
            print(f"📄 Found {results['total_results']} results")
            
            if results.get("cached"):
                print("💾 Results served from cache")
            
            # Display top results
            for i, result in enumerate(results["results"][:3], 1):
                print(f"\n  {i}. Score: {result['similarity_score']:.3f}")
                print(f"     Type: {result['chunk_type']}")
                print(f"     Source: {result.get('source_document', 'Unknown')}")
                print(f"     Text: {result['text'][:200]}...")
                
                # Show metadata example
                if result["metadata"]:
                    metadata_keys = list(result["metadata"].keys())[:3]
                    print(f"     Metadata: {', '.join(metadata_keys)}")
            
            print("-" * 80)
        
        # Example: Search with full content
        print("\n🔍 Example with full content retrieval:")
        content_results = await client.search(
            query="machine learning",
            model="scientific",
            top_k=2,
            include_metadata=True,
            include_content=True,
            similarity_threshold=0.2
        )
        
        for i, result in enumerate(content_results["results"], 1):
            print(f"\nResult {i}:")
            print(f"  Similarity: {result['similarity_score']:.3f}")
            print(f"  Chunk ID: {result['chunk_id']}")
            print(f"  Text: {result['text'][:150]}...")
            
            # Show original content structure if available
            if result.get("original_content"):
                original = result["original_content"]
                print(f"  Original document sections: {len(original.get('sections', []))}")
                print(f"  Original document tables: {len(original.get('tables', []))}")
                print(f"  Original document figures: {len(original.get('figures', []))}")
                print(f"  Full text length: {len(original.get('full_text', ''))} chars")
    
    except httpx.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())