#!/usr/bin/env python3
# main.py - Main orchestration script for medRxiv and bioRxiv crawler

import argparse
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from config import config
from download_metadata import fetch_all as fetch_metadata
from download_pdfs import load_entries, run_parallel as download_pdfs_parallel

def setup_argument_parser():
    """Setup command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Robust crawler for medRxiv and bioRxiv papers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --metadata-only                    # Download only metadata
  %(prog)s --pdfs-only                        # Download only PDFs
  %(prog)s --source biorxiv                   # Process only bioRxiv
  %(prog)s --resume                           # Resume interrupted downloads
  %(prog)s --start-date 2020-01-01            # Custom start date
  %(prog)s --end-date 2024-12-31              # Custom end date
        """
    )
    
    parser.add_argument(
        "--metadata-only", 
        action="store_true",
        help="Download only metadata, skip PDFs"
    )
    
    parser.add_argument(
        "--pdfs-only", 
        action="store_true",
        help="Download only PDFs, skip metadata"
    )
    
    parser.add_argument(
        "--source", 
        choices=["biorxiv", "medrxiv"],
        help="Process only specified source"
    )
    
    parser.add_argument(
        "--resume", 
        action="store_true",
        help="Resume interrupted downloads"
    )
    
    parser.add_argument(
        "--start-date",
        help="Start date for downloads (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--end-date",
        help="End date for downloads (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--threads",
        type=int,
        help="Number of threads for PDF downloads"
    )
    
    parser.add_argument(
        "--config",
        default="config.json",
        help="Configuration file path"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually doing it"
    )
    
    return parser

def get_sources_to_process(args) -> Dict[str, Dict]:
    """Get list of sources to process based on arguments"""
    sources = config.get_sources()
    
    if args.source:
        if args.source not in sources:
            print(f"Error: Source '{args.source}' not found or not enabled")
            sys.exit(1)
        return {args.source: sources[args.source]}
    
    return sources

def get_date_range(args) -> tuple:
    """Get start and end dates for processing"""
    end_date = args.end_date or config.get_end_date()
    return args.start_date, end_date

def print_summary(results: Dict[str, Dict]):
    """Print summary of operations"""
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    
    total_metadata = 0
    total_pdfs_downloaded = 0
    total_pdfs_existing = 0
    total_pdfs_failed = 0
    
    for source, data in results.items():
        print(f"\n{source.upper()}:")
        
        if "metadata" in data:
            metadata_count = data["metadata"]["successful"]
            failed_chunks = len(data["metadata"]["failed_chunks"])
            print(f"  Metadata: {metadata_count:,} entries, {failed_chunks} failed chunks")
            total_metadata += metadata_count
        
        if "pdfs" in data:
            pdf_results = data["pdfs"]
            downloaded = len([r for r in pdf_results if r["status"] == "downloaded"])
            existing = len([r for r in pdf_results if r["status"] == "exists"])
            failed = len([r for r in pdf_results if r["status"] not in ["downloaded", "exists"]])
            
            print(f"  PDFs: {downloaded:,} downloaded, {existing:,} existing, {failed:,} failed")
            total_pdfs_downloaded += downloaded
            total_pdfs_existing += existing
            total_pdfs_failed += failed
    
    print(f"\nTOTAL SUMMARY:")
    print(f"  Metadata entries: {total_metadata:,}")
    print(f"  PDFs downloaded: {total_pdfs_downloaded:,}")
    print(f"  PDFs existing: {total_pdfs_existing:,}")
    print(f"  PDFs failed: {total_pdfs_failed:,}")
    print("="*60)

def validate_arguments(args):
    """Validate command line arguments"""
    if args.metadata_only and args.pdfs_only:
        print("Error: Cannot specify both --metadata-only and --pdfs-only")
        sys.exit(1)
    
    if args.start_date:
        try:
            datetime.strptime(args.start_date, "%Y-%m-%d")
        except ValueError:
            print("Error: Invalid start date format. Use YYYY-MM-DD")
            sys.exit(1)
    
    if args.end_date:
        try:
            datetime.strptime(args.end_date, "%Y-%m-%d")
        except ValueError:
            print("Error: Invalid end date format. Use YYYY-MM-DD")
            sys.exit(1)
    
    if args.threads and args.threads <= 0:
        print("Error: Number of threads must be positive")
        sys.exit(1)

def main():
    """Main orchestration function"""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Validate arguments
    validate_arguments(args)
    
    # Load configuration
    global config
    if args.config != "config.json":
        from config import Config
        config = Config(args.config)
    
    # Override config with command line arguments
    if args.threads:
        config.set("download.num_threads", args.threads)
    
    # Create necessary directories
    config.create_directories()
    
    # Get sources and date range
    sources = get_sources_to_process(args)
    start_date_override, end_date = get_date_range(args)
    
    print(f"Starting rxiv crawler at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Sources: {', '.join(sources.keys())}")
    print(f"End date: {end_date}")
    if start_date_override:
        print(f"Start date override: {start_date_override}")
    print(f"Resume mode: {'Yes' if args.resume else 'No'}")
    
    if args.dry_run:
        print("\nDRY RUN MODE - No actual downloads will be performed")
        return
    
    results = {}
    
    # Process each source
    for source_name, source_config in sources.items():
        print(f"\n{'='*50}")
        print(f"Processing {source_name.upper()}")
        print(f"{'='*50}")
        
        start_date = start_date_override or source_config["start_date"]
        results[source_name] = {}
        
        # Download metadata
        if not args.pdfs_only:
            print(f"\nDownloading metadata for {source_name}...")
            metadata_file = Path(config.get("paths.data_dir")) / f"{source_name}_metadata.jsonl"
            
            try:
                successful, failed_chunks = fetch_metadata(
                    source_name, 
                    start_date, 
                    end_date, 
                    str(metadata_file),
                    resume=args.resume
                )
                results[source_name]["metadata"] = {
                    "successful": successful,
                    "failed_chunks": failed_chunks
                }
                print(f"Metadata download completed: {successful:,} entries, {len(failed_chunks)} failed chunks")
            except Exception as e:
                print(f"Error downloading metadata for {source_name}: {e}")
                results[source_name]["metadata"] = {"error": str(e)}
                continue
        
        # Download PDFs
        if not args.metadata_only:
            print(f"\nDownloading PDFs for {source_name}...")
            metadata_file = Path(config.get("paths.data_dir")) / f"{source_name}_metadata.jsonl"
            pdf_dir = Path(config.get("paths.pdf_dir")) / source_name
            
            if not metadata_file.exists():
                print(f"Metadata file not found: {metadata_file}")
                if not args.pdfs_only:
                    print("Skipping PDF download for this source")
                    continue
                else:
                    print("Cannot download PDFs without metadata file")
                    continue
            
            try:
                entries = load_entries(str(metadata_file))
                print(f"Loaded {len(entries):,} entries from metadata")
                
                pdf_results = download_pdfs_parallel(
                    entries,
                    source_name,
                    str(pdf_dir),
                    resume=args.resume
                )
                results[source_name]["pdfs"] = pdf_results
                
                downloaded = len([r for r in pdf_results if r["status"] == "downloaded"])
                existing = len([r for r in pdf_results if r["status"] == "exists"])
                failed = len([r for r in pdf_results if r["status"] not in ["downloaded", "exists"]])
                
                print(f"PDF download completed: {downloaded:,} downloaded, {existing:,} existing, {failed:,} failed")
                
            except Exception as e:
                print(f"Error downloading PDFs for {source_name}: {e}")
                results[source_name]["pdfs"] = {"error": str(e)}
                continue
    
    # Save overall results
    results_file = Path(config.get("paths.log_dir")) / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print_summary(results)
    print(f"\nResults saved to: {results_file}")
    print(f"Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)