"""PDF content extraction module using Docling.

This module provides robust PDF content extraction capabilities optimized for academic papers:
- High-quality layout preservation using Docling
- Structured content extraction (headers, paragraphs, tables, figures)
- Academic paper-specific parsing with section detection
- Comprehensive error handling and progress tracking
- Modular design for integration with RAG pipeline

Input:
    - PDF files from the downloaded corpus
    - Processing configuration and batch settings
    - Output directory specifications

Output:
    - Structured JSON documents with preserved layout
    - Extracted tables, figures, and metadata
    - Section-wise content organization
    - Processing logs and error tracking

Author: Haining Wang <hw56@iu.edu>
"""

import hashlib
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from docling import DocumentConverter
from docling.datamodel.base_models import ConversionStatus
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import PdfFormatOption
from tqdm import tqdm


class PDFContentExtractor:
    """High-performance PDF content extractor using Docling for academic papers."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the PDF content extractor.
        
        Args:
            config: Configuration dictionary for extraction settings
        """
        self.config = config or self._get_default_config()
        self.logger = self._setup_logging()
        self.converter = self._setup_converter()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for PDF extraction.
        
        Returns:
            Default configuration dictionary
        """
        return {
            "batch_size": 10,
            "max_workers": 4,
            "timeout": 300,  # 5 minutes per PDF
            "extract_tables": True,
            "extract_figures": True,
            "preserve_layout": True,
            "output_format": "json",
            "log_level": "INFO",
            "retry_attempts": 3,
            "skip_existing": True,
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for content extraction.
        
        Returns:
            Configured logger instance
        """
        log_dir = Path("logs/processing")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"content_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=getattr(logging, self.config["log_level"]),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def _setup_converter(self) -> DocumentConverter:
        """Setup Docling document converter with optimized settings.
        
        Returns:
            Configured DocumentConverter instance
        """
        # Configure pipeline options for academic papers
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True  # Enable OCR for scanned papers
        pipeline_options.do_table_structure = self.config["extract_tables"]
        pipeline_options.table_structure_options.do_cell_matching = True
        
        # Configure PDF format options
        pdf_options = PdfFormatOption(pipeline_options=pipeline_options)
        
        converter = DocumentConverter(
            format_options={
                "pdf": pdf_options,
            }
        )
        
        self.logger.info("Initialized Docling converter with academic paper optimization")
        return converter
    
    def extract_single_pdf(self, pdf_path: Path, output_dir: Path) -> Dict[str, Any]:
        """Extract content from a single PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save extracted content
            
        Returns:
            Dictionary containing extraction results and metadata
        """
        start_time = time.time()
        pdf_name = pdf_path.stem
        
        # Generate output paths
        json_output = output_dir / f"{pdf_name}.json"
        metadata_output = output_dir / f"{pdf_name}_metadata.json"
        
        # Skip if already processed and skip_existing is True
        if self.config["skip_existing"] and json_output.exists():
            self.logger.debug(f"Skipping {pdf_name} - already processed")
            return {
                "status": "skipped",
                "pdf_path": str(pdf_path),
                "reason": "already_exists"
            }
        
        try:
            self.logger.info(f"Processing {pdf_name}")
            
            # Convert PDF using Docling
            result = self.converter.convert(str(pdf_path))
            
            if result.status != ConversionStatus.SUCCESS:
                raise Exception(f"Conversion failed with status: {result.status}")
            
            # Extract structured content
            extracted_content = self._extract_structured_content(result.document)
            
            # Generate metadata
            metadata = self._generate_metadata(pdf_path, result.document, start_time)
            
            # Save extracted content
            with open(json_output, 'w', encoding='utf-8') as f:
                json.dump(extracted_content, f, indent=2, ensure_ascii=False)
            
            # Save metadata
            with open(metadata_output, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            processing_time = time.time() - start_time
            self.logger.info(f"Successfully processed {pdf_name} in {processing_time:.2f}s")
            
            return {
                "status": "success",
                "pdf_path": str(pdf_path),
                "json_output": str(json_output),
                "metadata_output": str(metadata_output),
                "processing_time": processing_time,
                "pages": len(result.document.pages) if hasattr(result.document, 'pages') else 0,
                "content_length": len(extracted_content.get("full_text", ""))
            }
            
        except Exception as e:
            self.logger.error(f"Failed to process {pdf_name}: {str(e)}")
            return {
                "status": "failed",
                "pdf_path": str(pdf_path),
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def _extract_structured_content(self, document) -> Dict[str, Any]:
        """Extract structured content from Docling document.
        
        Args:
            document: Docling document object
            
        Returns:
            Structured content dictionary
        """
        content = {
            "document_structure": {},
            "sections": [],
            "tables": [],
            "figures": [],
            "references": [],
            "full_text": "",
            "metadata": {}
        }
        
        try:
            # Extract main text content
            if hasattr(document, 'export_to_text'):
                content["full_text"] = document.export_to_text()
            
            # Extract structured elements using Docling's document model
            if hasattr(document, 'iterate_items'):
                current_section = None
                section_content = []
                
                for item in document.iterate_items():
                    item_type = item.label if hasattr(item, 'label') else 'text'
                    item_text = item.text if hasattr(item, 'text') else str(item)
                    
                    # Detect section headers
                    if item_type in ['title', 'section_header', 'subsection_header']:
                        # Save previous section
                        if current_section:
                            content["sections"].append({
                                "title": current_section,
                                "content": "\n".join(section_content),
                                "type": "section"
                            })
                        
                        current_section = item_text
                        section_content = []
                    
                    # Handle tables
                    elif item_type == 'table':
                        table_data = self._extract_table_data(item)
                        content["tables"].append(table_data)
                    
                    # Handle figures
                    elif item_type in ['figure', 'image']:
                        figure_data = self._extract_figure_data(item)
                        content["figures"].append(figure_data)
                    
                    # Regular text content
                    else:
                        section_content.append(item_text)
                
                # Add final section
                if current_section:
                    content["sections"].append({
                        "title": current_section,
                        "content": "\n".join(section_content),
                        "type": "section"
                    })
            
            # Extract document-level metadata
            content["metadata"] = self._extract_document_metadata(document)
            
        except Exception as e:
            self.logger.warning(f"Error in structured extraction: {e}")
            # Fallback to basic text extraction
            if hasattr(document, 'export_to_text'):
                content["full_text"] = document.export_to_text()
        
        return content
    
    def _extract_table_data(self, table_item) -> Dict[str, Any]:
        """Extract structured data from a table element.
        
        Args:
            table_item: Docling table item
            
        Returns:
            Structured table data
        """
        table_data = {
            "type": "table",
            "caption": "",
            "headers": [],
            "rows": [],
            "raw_content": str(table_item)
        }
        
        try:
            # Extract table structure if available
            if hasattr(table_item, 'table_data'):
                table_data["headers"] = table_item.table_data.get("headers", [])
                table_data["rows"] = table_item.table_data.get("rows", [])
            
            if hasattr(table_item, 'caption'):
                table_data["caption"] = table_item.caption
                
        except Exception as e:
            self.logger.debug(f"Table extraction error: {e}")
        
        return table_data
    
    def _extract_figure_data(self, figure_item) -> Dict[str, Any]:
        """Extract structured data from a figure element.
        
        Args:
            figure_item: Docling figure item
            
        Returns:
            Structured figure data
        """
        figure_data = {
            "type": "figure",
            "caption": "",
            "description": "",
            "raw_content": str(figure_item)
        }
        
        try:
            if hasattr(figure_item, 'caption'):
                figure_data["caption"] = figure_item.caption
            
            if hasattr(figure_item, 'alt_text'):
                figure_data["description"] = figure_item.alt_text
                
        except Exception as e:
            self.logger.debug(f"Figure extraction error: {e}")
        
        return figure_data
    
    def _extract_document_metadata(self, document) -> Dict[str, Any]:
        """Extract document-level metadata.
        
        Args:
            document: Docling document object
            
        Returns:
            Document metadata dictionary
        """
        metadata = {}
        
        try:
            # Extract available metadata from document
            if hasattr(document, 'metadata'):
                metadata.update(document.metadata)
            
            # Count content elements
            metadata["statistics"] = {
                "total_pages": len(document.pages) if hasattr(document, 'pages') else 0,
                "total_elements": len(list(document.iterate_items())) if hasattr(document, 'iterate_items') else 0
            }
            
        except Exception as e:
            self.logger.debug(f"Metadata extraction error: {e}")
        
        return metadata
    
    def _generate_metadata(self, pdf_path: Path, document, start_time: float) -> Dict[str, Any]:
        """Generate comprehensive metadata for the extraction.
        
        Args:
            pdf_path: Path to the original PDF
            document: Docling document object
            start_time: Processing start timestamp
            
        Returns:
            Comprehensive metadata dictionary
        """
        pdf_stat = pdf_path.stat()
        
        metadata = {
            "extraction_info": {
                "timestamp": datetime.now().isoformat(),
                "processing_time": time.time() - start_time,
                "extractor_version": "docling",
                "config": self.config
            },
            "source_file": {
                "path": str(pdf_path),
                "name": pdf_path.name,
                "size_bytes": pdf_stat.st_size,
                "modified_time": datetime.fromtimestamp(pdf_stat.st_mtime).isoformat(),
                "md5_hash": self._calculate_file_hash(pdf_path)
            },
            "document_info": {}
        }
        
        # Add document-specific metadata
        try:
            if hasattr(document, 'pages'):
                metadata["document_info"]["page_count"] = len(document.pages)
            
            # Add content statistics
            doc_metadata = self._extract_document_metadata(document)
            metadata["document_info"].update(doc_metadata)
            
        except Exception as e:
            self.logger.debug(f"Document info extraction error: {e}")
        
        return metadata
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            MD5 hash as hexadecimal string
        """
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def process_batch(self, pdf_files: List[Path], output_dir: Path) -> Dict[str, Any]:
        """Process a batch of PDF files.
        
        Args:
            pdf_files: List of PDF file paths to process
            output_dir: Directory to save extracted content
            
        Returns:
            Batch processing results
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            "batch_info": {
                "timestamp": datetime.now().isoformat(),
                "total_files": len(pdf_files),
                "output_directory": str(output_dir)
            },
            "processed_files": [],
            "statistics": {
                "successful": 0,
                "failed": 0,
                "skipped": 0
            }
        }
        
        self.logger.info(f"Starting batch processing of {len(pdf_files)} PDFs")
        
        with tqdm(total=len(pdf_files), desc="Extracting content") as pbar:
            for pdf_path in pdf_files:
                result = self.extract_single_pdf(pdf_path, output_dir)
                results["processed_files"].append(result)
                
                # Update statistics
                status = result["status"]
                if status in results["statistics"]:
                    results["statistics"][status] += 1
                
                pbar.set_postfix({
                    "success": results["statistics"]["successful"],
                    "failed": results["statistics"]["failed"],
                    "current": pdf_path.name[:20]
                })
                pbar.update(1)
        
        # Save batch results
        batch_results_file = output_dir / f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(batch_results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Batch processing completed. Results saved to {batch_results_file}")
        self.logger.info(f"Statistics: {results['statistics']}")
        
        return results


def main():
    """Main function for standalone content extraction."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract content from PDF files using Docling")
    parser.add_argument("input_dir", help="Directory containing PDF files")
    parser.add_argument("output_dir", help="Directory to save extracted content")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for processing")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum worker threads")
    
    args = parser.parse_args()
    
    # Setup configuration
    config = {
        "batch_size": args.batch_size,
        "max_workers": args.max_workers,
        "skip_existing": True,
    }
    
    # Initialize extractor
    extractor = PDFContentExtractor(config)
    
    # Find PDF files
    input_path = Path(args.input_dir)
    pdf_files = list(input_path.glob("**/*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {input_path}")
        return
    
    print(f"Found {len(pdf_files)} PDF files")
    
    # Process files
    output_path = Path(args.output_dir)
    results = extractor.process_batch(pdf_files, output_path)
    
    print(f"Processing completed: {results['statistics']}")


if __name__ == "__main__":
    main()