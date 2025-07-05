"""Main orchestrator for the academic paper processing pipeline.

This module coordinates the complete processing pipeline from PDF extraction to 
RAG-ready embeddings:
- Batch processing with configurable parallelism
- Pipeline orchestration with error recovery
- Progress tracking and comprehensive logging
- Modular design allowing individual pipeline stages
- Integration with the download pipeline

Input:
    - Downloaded PDF files from the corpus
    - Processing configuration and pipeline settings
    - Output directory specifications

Output:
    - Extracted structured content in JSON format
    - Intelligently chunked text ready for retrieval
    - High-quality embeddings and vector databases
    - Comprehensive processing reports and metrics

Author: Haining Wang <hw56@iu.edu>
"""

import argparse
import json
import logging
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from extract_content import PDFContentExtractor
from chunk_content import AcademicPaperChunker
from create_embeddings import EmbeddingPipeline


class PaperProcessingPipeline:
    """Complete pipeline orchestrator for academic paper processing."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the processing pipeline.
        
        Args:
            config: Configuration dictionary for the entire pipeline
        """
        self.config = config or self._get_default_config()
        self.logger = self._setup_logging()
        
        # Initialize pipeline components
        self.extractor = PDFContentExtractor(self.config.get("extraction", {}))
        self.chunker = AcademicPaperChunker(self.config.get("chunking", {}))
        self.embedder = EmbeddingPipeline(self.config.get("embedding", {}))
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for the entire pipeline.
        
        Returns:
            Default configuration dictionary
        """
        return {
            # Pipeline control
            "stages": ["extract", "chunk", "embed"],  # Stages to run
            "max_workers": max(1, multiprocessing.cpu_count() // 2),
            "batch_size": 10,
            "resume_mode": True,
            
            # Stage-specific configurations
            "extraction": {
                "skip_existing": True,
                "max_retries": 3,
                "timeout": 300,
            },
            "chunking": {
                "chunk_size": 512,
                "overlap_size": 50,
                "use_semantic_chunking": True,
            },
            "embedding": {
                "models": {
                    "primary": "all-MiniLM-L6-v2",
                    "scientific": "allenai/scibert_scivocab_uncased",
                },
                "vector_stores": ["chromadb"],
                "batch_size": 32,
            },
            
            # Output organization
            "output_structure": {
                "base_dir": "processed_papers",
                "extraction_dir": "extracted",
                "chunking_dir": "chunked", 
                "embedding_dir": "embeddings",
                "reports_dir": "reports",
            },
            
            # Quality control
            "quality_checks": True,
            "detailed_logging": True,
            "save_intermediate": True,
            
            # Error handling
            "fail_fast": False,
            "max_failures": 10,
            "retry_failed": True,
            
            # Performance
            "memory_limit_gb": 8,
            "disk_space_check": True,
            
            # Logging
            "log_level": "INFO",
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the processing pipeline.
        
        Returns:
            Configured logger instance
        """
        log_dir = Path("logs/pipeline")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=getattr(logging, self.config["log_level"]),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def process_corpus(self, input_dir: Path, output_dir: Path) -> Dict[str, Any]:
        """Process an entire corpus of PDF papers.
        
        Args:
            input_dir: Directory containing PDF files
            output_dir: Base output directory for processed files
            
        Returns:
            Complete processing results and statistics
        """
        start_time = time.time()
        
        # Setup output directory structure
        output_dirs = self._setup_output_directories(output_dir)
        
        # Find PDF files to process
        pdf_files = self._discover_pdf_files(input_dir)
        
        if not pdf_files:
            self.logger.warning(f"No PDF files found in {input_dir}")
            return {"error": "No PDF files found"}
        
        self.logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Initialize results tracking
        results = {
            "pipeline_info": {
                "start_time": datetime.now().isoformat(),
                "input_directory": str(input_dir),
                "output_directory": str(output_dir),
                "total_files": len(pdf_files),
                "config": self.config,
            },
            "stage_results": {},
            "file_results": [],
            "statistics": {},
            "errors": []
        }
        
        try:
            # Process files in batches
            batch_size = self.config["batch_size"]
            stages = self.config["stages"]
            
            for i in range(0, len(pdf_files), batch_size):
                batch_files = pdf_files[i:i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (len(pdf_files) + batch_size - 1) // batch_size
                
                self.logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_files)} files)")
                
                batch_results = self._process_batch(batch_files, output_dirs, stages)
                results["file_results"].extend(batch_results)
                
                # Check for too many failures
                if self.config["fail_fast"]:
                    failed_count = sum(1 for r in batch_results if "error" in r)
                    if failed_count > self.config["max_failures"]:
                        self.logger.error(f"Too many failures ({failed_count}), stopping")
                        break
            
            # Generate comprehensive statistics
            results["statistics"] = self._generate_corpus_statistics(results, start_time)
            
            # Save results report
            self._save_results_report(results, output_dirs["reports"])
            
            self.logger.info(f"Corpus processing completed in {time.time() - start_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Pipeline error: {e}")
            results["pipeline_error"] = str(e)
        
        return results
    
    def _setup_output_directories(self, base_dir: Path) -> Dict[str, Path]:
        """Setup organized output directory structure.
        
        Args:
            base_dir: Base output directory
            
        Returns:
            Dictionary mapping directory types to paths
        """
        structure = self.config["output_structure"]
        
        directories = {}
        for dir_type, dir_name in structure.items():
            if dir_type == "base_dir":
                continue
            
            dir_path = base_dir / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
            directories[dir_type.replace("_dir", "")] = dir_path
        
        return directories
    
    def _discover_pdf_files(self, input_dir: Path) -> List[Path]:
        """Discover all PDF files in the input directory.
        
        Args:
            input_dir: Directory to search for PDFs
            
        Returns:
            List of PDF file paths
        """
        pdf_files = []
        
        # Search recursively for PDF files
        for pattern in ["*.pdf", "**/*.pdf"]:
            pdf_files.extend(input_dir.glob(pattern))
        
        # Remove duplicates and sort
        pdf_files = sorted(list(set(pdf_files)))
        
        # Filter based on resume mode
        if self.config["resume_mode"]:
            pdf_files = self._filter_processed_files(pdf_files)
        
        return pdf_files
    
    def _filter_processed_files(self, pdf_files: List[Path]) -> List[Path]:
        """Filter out already processed files in resume mode.
        
        Args:
            pdf_files: List of PDF file paths
            
        Returns:
            List of unprocessed PDF file paths
        """
        # This is a simplified implementation
        # In practice, you'd check for existing output files
        unprocessed = []
        
        for pdf_file in pdf_files:
            # Check if final embedding output exists
            expected_output = Path(self.config["output_structure"]["base_dir"]) / \
                            self.config["output_structure"]["embedding_dir"] / \
                            pdf_file.stem / "embedding_results.json"
            
            if not expected_output.exists():
                unprocessed.append(pdf_file)
        
        if len(unprocessed) < len(pdf_files):
            self.logger.info(f"Resume mode: skipping {len(pdf_files) - len(unprocessed)} processed files")
        
        return unprocessed
    
    def _process_batch(self, pdf_files: List[Path], output_dirs: Dict[str, Path], 
                      stages: List[str]) -> List[Dict[str, Any]]:
        """Process a batch of PDF files through the pipeline.
        
        Args:
            pdf_files: List of PDF files to process
            output_dirs: Output directory mapping
            stages: List of pipeline stages to execute
            
        Returns:
            List of processing results for each file
        """
        batch_results = []
        
        # Process files in parallel or sequential based on config
        max_workers = min(self.config["max_workers"], len(pdf_files))
        
        if max_workers > 1:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(self._process_single_file, pdf_file, output_dirs, stages)
                    for pdf_file in pdf_files
                ]
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        batch_results.append(result)
                    except Exception as e:
                        self.logger.error(f"File processing failed: {e}")
                        batch_results.append({"error": str(e)})
        else:
            # Sequential processing for debugging or resource constraints
            for pdf_file in pdf_files:
                result = self._process_single_file(pdf_file, output_dirs, stages)
                batch_results.append(result)
        
        return batch_results
    
    def _process_single_file(self, pdf_file: Path, output_dirs: Dict[str, Path], 
                           stages: List[str]) -> Dict[str, Any]:
        """Process a single PDF file through all pipeline stages.
        
        Args:
            pdf_file: Path to PDF file
            output_dirs: Output directory mapping
            stages: List of pipeline stages to execute
            
        Returns:
            Processing result for the file
        """
        start_time = time.time()
        file_id = pdf_file.stem
        
        result = {
            "file_path": str(pdf_file),
            "file_id": file_id,
            "stages_completed": [],
            "stage_outputs": {},
            "statistics": {},
            "processing_time": 0
        }
        
        try:
            current_data = None
            
            # Stage 1: Content Extraction
            if "extract" in stages:
                extraction_dir = output_dirs["extraction"] / file_id
                extraction_dir.mkdir(exist_ok=True)
                
                extraction_result = self.extractor.extract_single_pdf(pdf_file, extraction_dir)
                
                if extraction_result["status"] == "success":
                    result["stages_completed"].append("extract")
                    result["stage_outputs"]["extract"] = extraction_result
                    
                    # Load extracted content for next stage
                    json_file = Path(extraction_result["json_output"])
                    with open(json_file, 'r', encoding='utf-8') as f:
                        current_data = json.load(f)
                else:
                    raise Exception(f"Extraction failed: {extraction_result.get('error', 'Unknown error')}")
            
            # Stage 2: Content Chunking
            if "chunk" in stages and current_data:
                chunking_dir = output_dirs["chunking"] / file_id
                chunking_dir.mkdir(exist_ok=True)
                
                # Load metadata
                metadata_file = extraction_dir / f"{file_id}_metadata.json"
                metadata = {}
                if metadata_file.exists():
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                
                chunked_data = self.chunker.chunk_document(current_data, metadata)
                
                # Save chunked data
                chunked_file = chunking_dir / f"{file_id}_chunked.json"
                with open(chunked_file, 'w', encoding='utf-8') as f:
                    json.dump(chunked_data, f, indent=2, ensure_ascii=False)
                
                result["stages_completed"].append("chunk")
                result["stage_outputs"]["chunk"] = {
                    "output_file": str(chunked_file),
                    "total_chunks": len(chunked_data.get("chunks", [])),
                    "statistics": chunked_data.get("statistics", {})
                }
                
                current_data = chunked_data
            
            # Stage 3: Embedding Creation
            if "embed" in stages and current_data:
                embedding_dir = output_dirs["embedding"] / file_id
                
                embedding_results = self.embedder.process_chunks(current_data, embedding_dir)
                
                result["stages_completed"].append("embed")
                result["stage_outputs"]["embed"] = {
                    "output_directory": str(embedding_dir),
                    "embeddings": embedding_results.get("embeddings", {}),
                    "vector_stores": embedding_results.get("vector_stores", {}),
                    "statistics": embedding_results.get("statistics", {})
                }
            
            # Calculate overall statistics
            result["processing_time"] = time.time() - start_time
            result["status"] = "success"
            
        except Exception as e:
            self.logger.error(f"Error processing {pdf_file}: {e}")
            result["error"] = str(e)
            result["status"] = "failed"
            result["processing_time"] = time.time() - start_time
        
        return result
    
    def _generate_corpus_statistics(self, results: Dict[str, Any], start_time: float) -> Dict[str, Any]:
        """Generate comprehensive statistics for the corpus processing.
        
        Args:
            results: Processing results
            start_time: Pipeline start timestamp
            
        Returns:
            Comprehensive statistics dictionary
        """
        file_results = results.get("file_results", [])
        total_time = time.time() - start_time
        
        # Basic counts
        successful_files = [r for r in file_results if r.get("status") == "success"]
        failed_files = [r for r in file_results if r.get("status") == "failed"]
        
        # Stage completion analysis
        stage_completion = {}
        for stage in self.config["stages"]:
            completed = [r for r in file_results if stage in r.get("stages_completed", [])]
            stage_completion[stage] = {
                "completed": len(completed),
                "percentage": len(completed) / len(file_results) * 100 if file_results else 0
            }
        
        # Performance metrics
        processing_times = [r.get("processing_time", 0) for r in file_results if "processing_time" in r]
        
        # Content statistics
        total_chunks = 0
        total_embeddings = 0
        
        for result in successful_files:
            chunk_info = result.get("stage_outputs", {}).get("chunk", {})
            total_chunks += chunk_info.get("total_chunks", 0)
            
            embed_info = result.get("stage_outputs", {}).get("embed", {})
            for model_info in embed_info.get("embeddings", {}).values():
                if isinstance(model_info, dict) and "shape" in model_info:
                    total_embeddings += model_info["shape"][0]
        
        statistics = {
            "summary": {
                "total_files": len(file_results),
                "successful_files": len(successful_files),
                "failed_files": len(failed_files),
                "success_rate": len(successful_files) / len(file_results) * 100 if file_results else 0,
                "total_processing_time": total_time,
            },
            "stage_completion": stage_completion,
            "performance": {
                "avg_processing_time": sum(processing_times) / len(processing_times) if processing_times else 0,
                "min_processing_time": min(processing_times) if processing_times else 0,
                "max_processing_time": max(processing_times) if processing_times else 0,
                "files_per_hour": len(file_results) / (total_time / 3600) if total_time > 0 else 0,
            },
            "content": {
                "total_chunks_created": total_chunks,
                "total_embeddings_created": total_embeddings,
                "avg_chunks_per_paper": total_chunks / len(successful_files) if successful_files else 0,
            }
        }
        
        return statistics
    
    def _save_results_report(self, results: Dict[str, Any], reports_dir: Path) -> None:
        """Save comprehensive results report.
        
        Args:
            results: Complete processing results
            reports_dir: Directory to save reports
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save full results
        full_report_file = reports_dir / f"pipeline_results_{timestamp}.json"
        with open(full_report_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        # Save summary report
        summary_report = {
            "pipeline_info": results.get("pipeline_info", {}),
            "statistics": results.get("statistics", {}),
            "errors": results.get("errors", [])
        }
        
        summary_file = reports_dir / f"pipeline_summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_report, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"Results saved to {full_report_file}")
        self.logger.info(f"Summary saved to {summary_file}")


def main():
    """Main function for running the complete processing pipeline."""
    parser = argparse.ArgumentParser(
        description="Process academic papers through extraction, chunking, and embedding pipeline"
    )
    
    parser.add_argument("input_dir", help="Directory containing PDF files")
    parser.add_argument("output_dir", help="Base output directory")
    parser.add_argument("--config", help="Configuration file for the pipeline")
    parser.add_argument("--stages", nargs="+", 
                       choices=["extract", "chunk", "embed"],
                       default=["extract", "chunk", "embed"],
                       help="Pipeline stages to run")
    parser.add_argument("--batch-size", type=int, default=10, 
                       help="Number of files to process in each batch")
    parser.add_argument("--max-workers", type=int, 
                       help="Maximum number of parallel workers")
    parser.add_argument("--resume", action="store_true", 
                       help="Resume processing, skip already processed files")
    
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Override config with command line arguments
    config["stages"] = args.stages
    config["batch_size"] = args.batch_size
    config["resume_mode"] = args.resume
    
    if args.max_workers:
        config["max_workers"] = args.max_workers
    
    # Initialize and run pipeline
    pipeline = PaperProcessingPipeline(config)
    
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory {input_path} does not exist")
        return
    
    print(f"Starting pipeline processing...")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Stages: {args.stages}")
    
    results = pipeline.process_corpus(input_path, output_path)
    
    # Print summary
    stats = results.get("statistics", {})
    summary = stats.get("summary", {})
    
    print(f"\nProcessing completed!")
    print(f"Total files: {summary.get('total_files', 0)}")
    print(f"Successful: {summary.get('successful_files', 0)}")
    print(f"Failed: {summary.get('failed_files', 0)}")
    print(f"Success rate: {summary.get('success_rate', 0):.1f}%")
    print(f"Total time: {summary.get('total_processing_time', 0):.2f}s")
    
    if "content" in stats:
        content = stats["content"]
        print(f"Total chunks: {content.get('total_chunks_created', 0)}")
        print(f"Total embeddings: {content.get('total_embeddings_created', 0)}")


if __name__ == "__main__":
    main()