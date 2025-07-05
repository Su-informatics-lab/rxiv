"""Intelligent content chunking for academic papers optimized for RAG.

This module implements advanced chunking strategies specifically designed for academic papers:
- Section-aware chunking respecting document structure
- Semantic chunking using embeddings for coherent segments
- Table and figure preservation with context
- Hierarchical chunking for multi-level retrieval
- Overlap management for context preservation

Input:
    - Structured JSON documents from content extraction
    - Chunking configuration and embedding models
    - Output specifications for vector storage

Output:
    - Semantically coherent text chunks with metadata
    - Preserved tables and figures as separate chunks
    - Hierarchical chunk relationships and references
    - Chunk embeddings ready for vector storage

Author: Haining Wang <hw56@iu.edu>
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class AcademicPaperChunker:
    """Intelligent chunker for academic papers with RAG optimization."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the academic paper chunker.
        
        Args:
            config: Configuration dictionary for chunking settings
        """
        self.config = config or self._get_default_config()
        self.logger = self._setup_logging()
        self.embedding_model = self._load_embedding_model()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for chunking.
        
        Returns:
            Default configuration dictionary
        """
        return {
            # Chunking parameters
            "chunk_size": 512,  # Target tokens per chunk
            "overlap_size": 50,  # Overlap between chunks
            "min_chunk_size": 100,  # Minimum chunk size
            "max_chunk_size": 1000,  # Maximum chunk size
            
            # Section handling
            "preserve_sections": True,
            "section_aware": True,
            "include_section_headers": True,
            
            # Table and figure handling
            "preserve_tables": True,
            "preserve_figures": True,
            "table_context_window": 200,  # Tokens before/after table
            
            # Semantic chunking
            "use_semantic_chunking": True,
            "semantic_threshold": 0.5,  # Cosine similarity threshold
            "embedding_model": "all-MiniLM-L6-v2",  # Lightweight model
            
            # Hierarchical chunking
            "create_hierarchy": True,
            "parent_chunk_size": 2000,  # Larger parent chunks
            
            # Processing settings
            "batch_size": 32,
            "log_level": "INFO",
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for chunking operations.
        
        Returns:
            Configured logger instance
        """
        log_dir = Path("logs/processing")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"chunking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=getattr(logging, self.config["log_level"]),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def _load_embedding_model(self) -> Optional[SentenceTransformer]:
        """Load embedding model for semantic chunking.
        
        Returns:
            Loaded SentenceTransformer model or None if disabled
        """
        if not self.config["use_semantic_chunking"]:
            return None
        
        try:
            model = SentenceTransformer(self.config["embedding_model"])
            self.logger.info(f"Loaded embedding model: {self.config['embedding_model']}")
            return model
        except Exception as e:
            self.logger.warning(f"Failed to load embedding model: {e}")
            self.logger.info("Falling back to non-semantic chunking")
            self.config["use_semantic_chunking"] = False
            return None
    
    def chunk_document(self, document_data: Dict[str, Any], source_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Chunk a single document with intelligent strategies.
        
        Args:
            document_data: Structured document content from extraction
            source_metadata: Source file metadata
            
        Returns:
            Dictionary containing chunked content and metadata
        """
        start_time = datetime.now()
        
        chunked_data = {
            "document_id": source_metadata.get("source_file", {}).get("name", "unknown"),
            "source_metadata": source_metadata,
            "chunking_metadata": {
                "timestamp": start_time.isoformat(),
                "config": self.config,
                "strategy": "academic_paper"
            },
            "chunks": [],
            "hierarchy": {},
            "statistics": {}
        }
        
        try:
            # Extract sections for section-aware chunking
            sections = document_data.get("sections", [])
            tables = document_data.get("tables", [])
            figures = document_data.get("figures", [])
            full_text = document_data.get("full_text", "")
            
            # Create chunks from sections
            if sections and self.config["section_aware"]:
                chunks = self._chunk_sections(sections)
            else:
                # Fallback to full text chunking
                chunks = self._chunk_text(full_text)
            
            # Add table chunks
            if tables and self.config["preserve_tables"]:
                table_chunks = self._chunk_tables(tables, chunks)
                chunks.extend(table_chunks)
            
            # Add figure chunks
            if figures and self.config["preserve_figures"]:
                figure_chunks = self._chunk_figures(figures, chunks)
                chunks.extend(figure_chunks)
            
            # Apply semantic chunking if enabled
            if self.config["use_semantic_chunking"] and self.embedding_model:
                chunks = self._apply_semantic_chunking(chunks)
            
            # Create hierarchical structure
            if self.config["create_hierarchy"]:
                hierarchy = self._create_hierarchy(chunks)
                chunked_data["hierarchy"] = hierarchy
            
            # Finalize chunks with metadata
            chunked_data["chunks"] = self._finalize_chunks(chunks, source_metadata)
            
            # Generate statistics
            chunked_data["statistics"] = self._generate_statistics(chunks, start_time)
            
            self.logger.info(f"Successfully chunked document with {len(chunks)} chunks")
            
        except Exception as e:
            self.logger.error(f"Error chunking document: {e}")
            chunked_data["error"] = str(e)
        
        return chunked_data
    
    def _chunk_sections(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk content by respecting section boundaries.
        
        Args:
            sections: List of document sections
            
        Returns:
            List of section-aware chunks
        """
        chunks = []
        
        for section in sections:
            section_title = section.get("title", "")
            section_content = section.get("content", "")
            
            if not section_content.strip():
                continue
            
            # Split section into chunks
            section_chunks = self._chunk_text(section_content, max_length=self.config["chunk_size"])
            
            for i, chunk_text in enumerate(section_chunks):
                chunk = {
                    "text": chunk_text,
                    "type": "section",
                    "section_title": section_title,
                    "section_index": i,
                    "metadata": {
                        "section": section_title,
                        "is_first_in_section": i == 0,
                        "is_last_in_section": i == len(section_chunks) - 1
                    }
                }
                
                # Include section header in first chunk if configured
                if i == 0 and self.config["include_section_headers"] and section_title:
                    chunk["text"] = f"{section_title}\n\n{chunk_text}"
                
                chunks.append(chunk)
        
        return chunks
    
    def _chunk_text(self, text: str, max_length: Optional[int] = None) -> List[str]:
        """Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            max_length: Maximum chunk length (defaults to config)
            
        Returns:
            List of text chunks
        """
        if not text.strip():
            return []
        
        max_length = max_length or self.config["chunk_size"]
        overlap = self.config["overlap_size"]
        min_length = self.config["min_chunk_size"]
        
        # Split into sentences first
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            # If adding this sentence would exceed max length, finalize current chunk
            if current_length + sentence_length > max_length and current_chunk:
                chunk_text = " ".join(current_chunk)
                if len(chunk_text.split()) >= min_length:
                    chunks.append(chunk_text)
                
                # Start new chunk with overlap
                overlap_sentences = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s.split()) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if len(chunk_text.split()) >= min_length:
                chunks.append(chunk_text)
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex patterns.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Academic paper specific sentence splitting
        # Handle citations, equations, and abbreviations
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        
        # Split on sentence boundaries while preserving citations
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)
        
        # Clean and filter sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _chunk_tables(self, tables: List[Dict[str, Any]], existing_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create chunks for tables with context.
        
        Args:
            tables: List of table data
            existing_chunks: Existing text chunks for context
            
        Returns:
            List of table chunks
        """
        table_chunks = []
        
        for i, table in enumerate(tables):
            # Create table content
            table_text = self._format_table_content(table)
            
            # Add context if available
            context = self._find_table_context(table, existing_chunks)
            
            chunk = {
                "text": table_text,
                "type": "table",
                "table_index": i,
                "metadata": {
                    "table_caption": table.get("caption", ""),
                    "context": context,
                    "headers": table.get("headers", []),
                    "row_count": len(table.get("rows", [])),
                }
            }
            
            table_chunks.append(chunk)
        
        return table_chunks
    
    def _format_table_content(self, table: Dict[str, Any]) -> str:
        """Format table data into readable text.
        
        Args:
            table: Table data dictionary
            
        Returns:
            Formatted table text
        """
        lines = []
        
        # Add caption
        caption = table.get("caption", "")
        if caption:
            lines.append(f"Table: {caption}")
            lines.append("")
        
        # Add headers
        headers = table.get("headers", [])
        if headers:
            lines.append(" | ".join(headers))
            lines.append("-" * (len(" | ".join(headers))))
        
        # Add rows
        rows = table.get("rows", [])
        for row in rows:
            if isinstance(row, list):
                lines.append(" | ".join(str(cell) for cell in row))
            else:
                lines.append(str(row))
        
        return "\n".join(lines)
    
    def _chunk_figures(self, figures: List[Dict[str, Any]], existing_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create chunks for figures with context.
        
        Args:
            figures: List of figure data
            existing_chunks: Existing text chunks for context
            
        Returns:
            List of figure chunks
        """
        figure_chunks = []
        
        for i, figure in enumerate(figures):
            # Create figure content
            figure_text = self._format_figure_content(figure)
            
            # Add context if available
            context = self._find_figure_context(figure, existing_chunks)
            
            chunk = {
                "text": figure_text,
                "type": "figure",
                "figure_index": i,
                "metadata": {
                    "figure_caption": figure.get("caption", ""),
                    "figure_description": figure.get("description", ""),
                    "context": context,
                }
            }
            
            figure_chunks.append(chunk)
        
        return figure_chunks
    
    def _format_figure_content(self, figure: Dict[str, Any]) -> str:
        """Format figure data into readable text.
        
        Args:
            figure: Figure data dictionary
            
        Returns:
            Formatted figure text
        """
        lines = []
        
        # Add caption
        caption = figure.get("caption", "")
        if caption:
            lines.append(f"Figure: {caption}")
        
        # Add description
        description = figure.get("description", "")
        if description:
            lines.append(f"Description: {description}")
        
        return "\n".join(lines) if lines else "Figure content"
    
    def _find_table_context(self, table: Dict[str, Any], chunks: List[Dict[str, Any]]) -> str:
        """Find contextual text around a table.
        
        Args:
            table: Table data
            chunks: List of text chunks
            
        Returns:
            Contextual text
        """
        # Simple implementation - can be enhanced with position-based matching
        caption = table.get("caption", "").lower()
        if not caption:
            return ""
        
        # Find chunks that might reference this table
        context_chunks = []
        for chunk in chunks:
            chunk_text = chunk.get("text", "").lower()
            if any(word in chunk_text for word in caption.split()[:3]):  # Match first 3 words
                context_chunks.append(chunk.get("text", ""))
        
        return " ".join(context_chunks[:2])  # Return first 2 relevant chunks
    
    def _find_figure_context(self, figure: Dict[str, Any], chunks: List[Dict[str, Any]]) -> str:
        """Find contextual text around a figure.
        
        Args:
            figure: Figure data
            chunks: List of text chunks
            
        Returns:
            Contextual text
        """
        # Similar to table context finding
        caption = figure.get("caption", "").lower()
        if not caption:
            return ""
        
        context_chunks = []
        for chunk in chunks:
            chunk_text = chunk.get("text", "").lower()
            if any(word in chunk_text for word in caption.split()[:3]):
                context_chunks.append(chunk.get("text", ""))
        
        return " ".join(context_chunks[:2])
    
    def _apply_semantic_chunking(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply semantic chunking to merge similar adjacent chunks.
        
        Args:
            chunks: List of initial chunks
            
        Returns:
            List of semantically optimized chunks
        """
        if not self.embedding_model or len(chunks) < 2:
            return chunks
        
        try:
            # Extract text for embedding
            chunk_texts = [chunk.get("text", "") for chunk in chunks]
            
            # Generate embeddings in batches
            embeddings = []
            batch_size = self.config["batch_size"]
            
            for i in range(0, len(chunk_texts), batch_size):
                batch = chunk_texts[i:i + batch_size]
                batch_embeddings = self.embedding_model.encode(batch)
                embeddings.extend(batch_embeddings)
            
            # Merge similar adjacent chunks
            merged_chunks = []
            current_chunk = chunks[0]
            current_embedding = embeddings[0]
            
            for i in range(1, len(chunks)):
                similarity = cosine_similarity([current_embedding], [embeddings[i]])[0][0]
                
                if similarity > self.config["semantic_threshold"]:
                    # Merge with current chunk
                    current_chunk["text"] += " " + chunks[i]["text"]
                    # Update embedding (simple average)
                    current_embedding = (current_embedding + embeddings[i]) / 2
                else:
                    # Start new chunk
                    merged_chunks.append(current_chunk)
                    current_chunk = chunks[i]
                    current_embedding = embeddings[i]
            
            # Add final chunk
            merged_chunks.append(current_chunk)
            
            self.logger.info(f"Semantic chunking: {len(chunks)} -> {len(merged_chunks)} chunks")
            return merged_chunks
            
        except Exception as e:
            self.logger.warning(f"Semantic chunking failed: {e}")
            return chunks
    
    def _create_hierarchy(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create hierarchical structure for chunks.
        
        Args:
            chunks: List of chunks
            
        Returns:
            Hierarchical structure mapping
        """
        hierarchy = {
            "parent_chunks": [],
            "child_mapping": {},
            "statistics": {}
        }
        
        parent_size = self.config["parent_chunk_size"]
        current_parent_text = ""
        current_children = []
        parent_id = 0
        
        for i, chunk in enumerate(chunks):
            chunk_text = chunk.get("text", "")
            chunk_length = len(chunk_text.split())
            
            # Check if adding this chunk would exceed parent size
            if len(current_parent_text.split()) + chunk_length > parent_size and current_children:
                # Create parent chunk
                parent_chunk = {
                    "id": f"parent_{parent_id}",
                    "text": current_parent_text.strip(),
                    "child_ids": [f"chunk_{child_idx}" for child_idx in current_children],
                    "type": "parent"
                }
                hierarchy["parent_chunks"].append(parent_chunk)
                
                # Map children to parent
                for child_idx in current_children:
                    hierarchy["child_mapping"][f"chunk_{child_idx}"] = f"parent_{parent_id}"
                
                # Reset for next parent
                parent_id += 1
                current_parent_text = chunk_text
                current_children = [i]
            else:
                current_parent_text += " " + chunk_text
                current_children.append(i)
        
        # Handle final parent
        if current_children:
            parent_chunk = {
                "id": f"parent_{parent_id}",
                "text": current_parent_text.strip(),
                "child_ids": [f"chunk_{child_idx}" for child_idx in current_children],
                "type": "parent"
            }
            hierarchy["parent_chunks"].append(parent_chunk)
            
            for child_idx in current_children:
                hierarchy["child_mapping"][f"chunk_{child_idx}"] = f"parent_{parent_id}"
        
        hierarchy["statistics"] = {
            "total_parents": len(hierarchy["parent_chunks"]),
            "total_children": len(chunks),
            "avg_children_per_parent": len(chunks) / max(1, len(hierarchy["parent_chunks"]))
        }
        
        return hierarchy
    
    def _finalize_chunks(self, chunks: List[Dict[str, Any]], source_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Finalize chunks with complete metadata.
        
        Args:
            chunks: List of chunks to finalize
            source_metadata: Source document metadata
            
        Returns:
            List of finalized chunks
        """
        finalized_chunks = []
        
        for i, chunk in enumerate(chunks):
            finalized_chunk = {
                "id": f"chunk_{i}",
                "text": chunk.get("text", ""),
                "type": chunk.get("type", "text"),
                "position": i,
                "metadata": {
                    **chunk.get("metadata", {}),
                    "source_document": source_metadata.get("source_file", {}).get("name", ""),
                    "chunk_length": len(chunk.get("text", "").split()),
                    "extraction_timestamp": source_metadata.get("extraction_info", {}).get("timestamp", ""),
                }
            }
            
            # Add type-specific metadata
            if chunk.get("type") == "section":
                finalized_chunk["metadata"]["section_title"] = chunk.get("section_title", "")
                finalized_chunk["metadata"]["section_index"] = chunk.get("section_index", 0)
            elif chunk.get("type") == "table":
                finalized_chunk["metadata"]["table_index"] = chunk.get("table_index", 0)
            elif chunk.get("type") == "figure":
                finalized_chunk["metadata"]["figure_index"] = chunk.get("figure_index", 0)
            
            finalized_chunks.append(finalized_chunk)
        
        return finalized_chunks
    
    def _generate_statistics(self, chunks: List[Dict[str, Any]], start_time: datetime) -> Dict[str, Any]:
        """Generate statistics for the chunking process.
        
        Args:
            chunks: List of processed chunks
            start_time: Processing start time
            
        Returns:
            Statistics dictionary
        """
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Count chunk types
        type_counts = {}
        chunk_lengths = []
        
        for chunk in chunks:
            chunk_type = chunk.get("type", "text")
            type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
            
            text_length = len(chunk.get("text", "").split())
            chunk_lengths.append(text_length)
        
        statistics = {
            "total_chunks": len(chunks),
            "processing_time_seconds": processing_time,
            "chunk_types": type_counts,
            "chunk_length_stats": {
                "mean": np.mean(chunk_lengths) if chunk_lengths else 0,
                "std": np.std(chunk_lengths) if chunk_lengths else 0,
                "min": min(chunk_lengths) if chunk_lengths else 0,
                "max": max(chunk_lengths) if chunk_lengths else 0,
            }
        }
        
        return statistics


def main():
    """Main function for standalone chunking."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Chunk extracted content for RAG")
    parser.add_argument("input_file", help="JSON file with extracted content")
    parser.add_argument("output_file", help="Output file for chunked content")
    parser.add_argument("--config", help="Configuration file for chunking")
    
    args = parser.parse_args()
    
    # Load configuration
    config = None
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Initialize chunker
    chunker = AcademicPaperChunker(config)
    
    # Load document
    with open(args.input_file, 'r', encoding='utf-8') as f:
        document_data = json.load(f)
    
    # Load metadata (assume same directory)
    metadata_file = args.input_file.replace('.json', '_metadata.json')
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        metadata = {}
    
    # Chunk document
    chunked_data = chunker.chunk_document(document_data, metadata)
    
    # Save results
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(chunked_data, f, indent=2, ensure_ascii=False)
    
    print(f"Chunked document into {len(chunked_data['chunks'])} chunks")
    print(f"Statistics: {chunked_data['statistics']}")


if __name__ == "__main__":
    main()