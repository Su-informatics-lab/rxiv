#!/usr/bin/env python3
"""Test script for the academic paper processing pipeline.

This script provides a simple CLI to test the processing pipeline components
with sample PDFs and configurations.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

def test_extraction(pdf_path: Path, output_dir: Path) -> Dict[str, Any]:
    """Test PDF content extraction."""
    from extract_content import PDFContentExtractor
    
    print(f"Testing extraction with: {pdf_path}")
    extractor = PDFContentExtractor()
    result = extractor.extract_single_pdf(pdf_path, output_dir)
    
    print(f"Extraction result: {result['status']}")
    if result['status'] == 'success':
        print(f"Content saved to: {result['json_output']}")
        print(f"Processing time: {result['processing_time']:.2f}s")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
    
    return result

def test_chunking(content_file: Path, output_file: Path) -> Dict[str, Any]:
    """Test content chunking."""
    from chunk_content import AcademicPaperChunker
    
    print(f"Testing chunking with: {content_file}")
    chunker = AcademicPaperChunker()
    
    # Load content
    with open(content_file, 'r', encoding='utf-8') as f:
        content = json.load(f)
    
    # Load metadata if available
    metadata_file = content_file.parent / f"{content_file.stem}_metadata.json"
    metadata = {}
    if metadata_file.exists():
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    
    result = chunker.chunk_document(content, metadata)
    
    # Save result
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"Chunking completed: {len(result['chunks'])} chunks created")
    print(f"Statistics: {result['statistics']}")
    print(f"Results saved to: {output_file}")
    
    return result

def test_embedding(chunks_file: Path, output_dir: Path) -> Dict[str, Any]:
    """Test embedding creation."""
    from create_embeddings import EmbeddingPipeline
    
    print(f"Testing embedding with: {chunks_file}")
    pipeline = EmbeddingPipeline()
    
    # Load chunks
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)
    
    result = pipeline.process_chunks(chunks_data, output_dir)
    
    print(f"Embedding completed for {len(result['embeddings'])} models")
    print(f"Statistics: {result['statistics']}")
    print(f"Results saved to: {output_dir}")
    
    return result

def test_full_pipeline(pdf_path: Path, output_dir: Path, config_file: Path = None) -> Dict[str, Any]:
    """Test the complete processing pipeline."""
    from process_papers import PaperProcessingPipeline
    
    print(f"Testing full pipeline with: {pdf_path}")
    
    # Load config if provided
    config = None
    if config_file and config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
    
    pipeline = PaperProcessingPipeline(config)
    
    # Create a temporary directory structure
    pdf_dir = pdf_path.parent
    
    result = pipeline.process_corpus(pdf_dir, output_dir)
    
    print(f"Pipeline completed")
    print(f"Statistics: {result.get('statistics', {})}")
    
    return result

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="Test the academic paper processing pipeline")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Extraction test
    extract_parser = subparsers.add_parser('extract', help='Test PDF extraction')
    extract_parser.add_argument('pdf_path', help='Path to PDF file')
    extract_parser.add_argument('--output-dir', default='test_output/extraction', help='Output directory')
    
    # Chunking test
    chunk_parser = subparsers.add_parser('chunk', help='Test content chunking')
    chunk_parser.add_argument('content_file', help='Path to extracted content JSON file')
    chunk_parser.add_argument('--output-file', default='test_output/chunked.json', help='Output file')
    
    # Embedding test
    embed_parser = subparsers.add_parser('embed', help='Test embedding creation')
    embed_parser.add_argument('chunks_file', help='Path to chunked content JSON file')
    embed_parser.add_argument('--output-dir', default='test_output/embeddings', help='Output directory')
    
    # Full pipeline test
    pipeline_parser = subparsers.add_parser('pipeline', help='Test full pipeline')
    pipeline_parser.add_argument('pdf_path', help='Path to PDF file')
    pipeline_parser.add_argument('--output-dir', default='test_output/pipeline', help='Output directory')
    pipeline_parser.add_argument('--config', help='Configuration file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'extract':
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            test_extraction(Path(args.pdf_path), output_dir)
        
        elif args.command == 'chunk':
            output_file = Path(args.output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            test_chunking(Path(args.content_file), output_file)
        
        elif args.command == 'embed':
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            test_embedding(Path(args.chunks_file), output_dir)
        
        elif args.command == 'pipeline':
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            config_file = Path(args.config) if args.config else None
            test_full_pipeline(Path(args.pdf_path), output_dir, config_file)
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()