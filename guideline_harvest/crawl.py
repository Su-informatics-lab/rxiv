"""Clinical Guidelines Harvester

This harvests clinical guidelines from major US medical organizations using crawl4ai.

Key Features:
- crawl4ai-powered intelligent web crawling
- Automatic PDF detection and download
- Rich metadata extraction
- Parallel processing
- Robust error handling and resume capability
"""

import asyncio
import json
import logging
import time
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
import click
from crawl4ai import AsyncWebCrawler
from metadata_extractor import MetadataExtractor
from pdf_processor import PDFProcessor
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from sources import (
    CLINICAL_GUIDELINE_SOURCES,
    get_high_priority_sources,
    get_sources_by_specialty,
)


class GuidelineHarvester:
    """Advanced clinical guidelines harvester using crawl4ai."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the guidelines harvester.

        Args:
            config: Configuration dictionary for harvesting settings
        """
        default_config = self._get_default_config()
        self.config = {**default_config, **(config or {})}
        self.console = Console()
        self.logger = self._setup_logging()
        self.metadata_extractor = MetadataExtractor()
        self.pdf_processor = PDFProcessor()
        self.session: Optional[aiohttp.ClientSession] = None
        self.crawler: Optional[AsyncWebCrawler] = None

        # statistics tracking
        self.stats = {
            "sources_processed": 0,
            "guidelines_found": 0,
            "pdfs_downloaded": 0,
            "metadata_extracted": 0,
            "errors": 0,
            "start_time": None,
        }

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for guidelines harvesting."""
        return {
            # crawling settings
            "max_concurrent_sources": 3,
            "max_concurrent_pages": 5,
            "request_delay": 1.0,
            "max_retries": 3,
            "timeout": 30.0,
            # content filtering
            "min_pdf_size": 50 * 1024,  # 50KB minimum
            "max_pdf_size": 50 * 1024 * 1024,  # 50MB maximum
            "allowed_file_types": [".pdf", ".doc", ".docx"],
            # crawl4ai settings
            "crawl4ai_config": {
                "verbose": False,
                "headless": True,
                "browser_type": "chromium",
                "user_agent": "Mozilla/5.0 (compatible; ClinicalGuidelinesBot/1.0; +https://example.com/bot)",
                "viewport": {"width": 1920, "height": 1080},
                "wait_time": 2.0,
                "css_selector": None,  # extract all content
                "word_count_threshold": 100,
                "exclude_external_images": True,
                "exclude_social_media_links": True,
            },
            # output settings
            "output_dir": "guidelines_data",
            "pdf_dir": "pdfs",
            "metadata_dir": "metadata",
            "logs_dir": "logs",
            "save_raw_html": False,
            "save_screenshots": False,
            # quality control
            "validate_pdfs": True,
            "extract_text_preview": True,
            "min_text_length": 500,
            # resume capability
            "resume_mode": True,
            "progress_file": "harvest_progress.json",
        }

    def _setup_logging(self) -> logging.Logger:
        """Setup rich logging with file and console handlers."""
        log_dir = Path(self.config["logs_dir"])
        log_dir.mkdir(exist_ok=True)

        log_file = (
            log_dir
            / f"guidelines_harvest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )

        # create logger
        logger = logging.getLogger("guidelines_harvester")

        # set logging level based on debug flag
        if self.config.get("debug", False):
            logger.setLevel(logging.DEBUG)
            console_level = logging.DEBUG
        else:
            logger.setLevel(logging.INFO)
            console_level = logging.INFO

        # file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)

        # rich console handler
        console_handler = RichHandler(console=self.console, rich_tracebacks=True)
        console_handler.setLevel(console_level)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    async def initialize(self):
        """Initialize the harvester components."""
        self.logger.info("🚀 Initializing Clinical Guidelines Harvester")

        # setup directories
        self._setup_directories()

        # initialize aiohttp session
        timeout = aiohttp.ClientTimeout(total=self.config["timeout"])
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            headers={"User-Agent": self.config["crawl4ai_config"]["user_agent"]},
        )

        # initialize crawler
        self.crawler = AsyncWebCrawler(**self.config["crawl4ai_config"])
        await self.crawler.start()

        self.logger.info("✅ Harvester initialization complete")

    def _setup_directories(self):
        """Create necessary directories for output."""
        base_dir = Path(self.config["output_dir"])

        directories = [
            base_dir,
            base_dir / self.config["pdf_dir"],
            base_dir / self.config["metadata_dir"],
            Path(self.config["logs_dir"]),
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    async def harvest_all_sources(
        self, source_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """Harvest guidelines from all configured sources.

        Args:
            source_filter: Optional filter for source names (e.g., 'high_priority',
            'cardiology')

        Returns:
            Comprehensive harvesting results
        """
        self.stats["start_time"] = time.time()

        # determine sources to process
        if source_filter == "high_priority":
            sources = get_high_priority_sources()
        elif source_filter:
            sources = get_sources_by_specialty(source_filter)
        else:
            sources = list(CLINICAL_GUIDELINE_SOURCES.values())

        self.logger.info(f"🎯 Processing {len(sources)} guideline sources")

        # display sources table
        self._display_sources_table(sources)

        results = {
            "harvest_info": {
                "start_time": datetime.now().isoformat(),
                "sources_count": len(sources),
                "filter": source_filter,
                "config": self.config,
            },
            "source_results": {},
            "statistics": {},
            "errors": [],
        }

        # process sources with controlled concurrency
        semaphore = asyncio.Semaphore(self.config["max_concurrent_sources"])

        tasks = [
            self._harvest_source_with_semaphore(semaphore, source, results)
            for source in sources
        ]

        # execute with progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console,
        ) as progress:

            task = progress.add_task("Harvesting guidelines...", total=len(sources))

            for coro in asyncio.as_completed(tasks):
                await coro
                progress.advance(task)

        # generate final statistics
        results["statistics"] = self._generate_final_statistics()

        # save results
        self._save_harvest_results(results)

        # display summary
        self._display_harvest_summary(results)

        return results

    def _display_sources_table(self, sources):
        """Display a table of sources to be processed."""
        table = Table(
            title="Clinical Guidelines Sources",
            show_header=True,
            header_style="bold magenta",
        )

        table.add_column("Organization", style="cyan", no_wrap=True)
        table.add_column("Abbreviation", style="green")
        table.add_column("Specialties", style="yellow")
        table.add_column("Priority", style="red")

        for source in sources:
            specialties = ", ".join(source.specialties[:3])
            if len(source.specialties) > 3:
                specialties += "..."

            table.add_row(
                source.name, source.abbreviation, specialties, str(source.priority)
            )

        self.console.print(table)

    async def _harvest_source_with_semaphore(
        self, semaphore: asyncio.Semaphore, source, results: Dict[str, Any]
    ):
        """Harvest a single source with concurrency control."""
        async with semaphore:
            try:
                source_result = await self._harvest_single_source(source)
                results["source_results"][source.abbreviation] = source_result
                self.stats["sources_processed"] += 1
            except Exception as e:
                error_msg = f"Failed to harvest {source.abbreviation}: {str(e)}"
                self.logger.error(error_msg)
                results["errors"].append(error_msg)
                self.stats["errors"] += 1

    async def _harvest_single_source(self, source) -> Dict[str, Any]:
        """Harvest guidelines from a single source.

        Args:
            source: GuidelineSource configuration

        Returns:
            Source-specific harvesting results
        """
        self.logger.info(f"🔍 Harvesting {source.name} ({source.abbreviation})")

        source_result = {
            "source_info": {
                "name": source.name,
                "abbreviation": source.abbreviation,
                "base_url": source.base_url,
                "start_time": datetime.now().isoformat(),
            },
            "guidelines_found": [],
            "pdfs_downloaded": [],
            "errors": [],
            "statistics": {},
        }

        try:
            # discover guideline pages
            guideline_urls = await self._discover_guideline_pages(source)
            self.logger.info(
                f"📄 Found {len(guideline_urls)} potential guidelines from {source.abbreviation}"
            )

            # process each guideline page
            for url in guideline_urls:
                try:
                    # handle direct PDF links
                    if url.lower().endswith(".pdf"):
                        # create minimal guideline info for direct PDF
                        guideline_result = {
                            "url": url,
                            "title": f"Direct PDF from {source.abbreviation}",
                            "metadata": {
                                "source": source.abbreviation,
                                "direct_pdf": True,
                            },
                            "pdf_urls": [url],
                            "content_preview": "",
                            "extraction_timestamp": datetime.now().isoformat(),
                            "word_count": 0,
                        }
                        source_result["guidelines_found"].append(guideline_result)

                        # download the PDF directly
                        self.logger.info(f"💾 Attempting to download direct PDF: {url}")
                        pdf_result = await self._download_pdf(
                            source, url, guideline_result
                        )
                        if pdf_result:
                            source_result["pdfs_downloaded"].append(pdf_result)
                            self.stats["pdfs_downloaded"] += 1
                            self.logger.info(
                                f"✅ Downloaded PDF: {pdf_result['filename']}"
                            )
                        else:
                            self.logger.warning(f"❌ Failed to download PDF: {url}")
                    else:
                        # analyze if this is a list page or single guideline page
                        page_analysis = await self._analyze_guideline_page(source, url)
                        
                        if page_analysis["is_list_page"]:
                            # This page contains multiple guidelines - extract them all
                            if self.config.get("debug", False):
                                self.logger.debug(f"📋 Processing list page: {url}")
                                self.logger.debug(f"  Found {len(page_analysis['guideline_links'])} guideline links")
                                self.logger.debug(f"  Found {len(page_analysis['pdf_links'])} direct PDF links")
                            
                            # Add direct PDF links
                            for pdf_url in page_analysis["pdf_links"]:
                                guideline_result = {
                                    "url": pdf_url,
                                    "title": f"PDF from {source.abbreviation} list",
                                    "metadata": {"source": source.abbreviation, "from_list_page": True},
                                    "pdf_urls": [pdf_url],
                                    "content_preview": "",
                                    "extraction_timestamp": datetime.now().isoformat(),
                                    "word_count": 0,
                                }
                                source_result["guidelines_found"].append(guideline_result)
                                
                                # download the PDF
                                pdf_result = await self._download_pdf(source, pdf_url, guideline_result)
                                if pdf_result:
                                    source_result["pdfs_downloaded"].append(pdf_result)
                                    self.stats["pdfs_downloaded"] += 1
                            
                            # Process individual guideline pages found on this list page
                            for guideline_url in page_analysis["guideline_links"][:10]:  # Limit to avoid overwhelming
                                try:
                                    guideline_result = await self._process_guideline_page(source, guideline_url)
                                    if guideline_result:
                                        source_result["guidelines_found"].append(guideline_result)
                                        
                                        # download PDFs if found
                                        if guideline_result.get("pdf_urls"):
                                            for pdf_url in guideline_result["pdf_urls"]:
                                                pdf_result = await self._download_pdf(source, pdf_url, guideline_result)
                                                if pdf_result:
                                                    source_result["pdfs_downloaded"].append(pdf_result)
                                                    self.stats["pdfs_downloaded"] += 1
                                    
                                    # Rate limit processing
                                    await asyncio.sleep(self.config["request_delay"])
                                    
                                except Exception as e:
                                    if self.config.get("debug", False):
                                        self.logger.debug(f"❌ Error processing guideline from list: {str(e)}")
                                    continue
                        else:
                            # Single guideline page - process normally
                            guideline_result = await self._process_guideline_page(source, url)
                            if guideline_result:
                                source_result["guidelines_found"].append(guideline_result)

                                # download PDFs if found
                                if guideline_result.get("pdf_urls"):
                                    for pdf_url in guideline_result["pdf_urls"]:
                                        pdf_result = await self._download_pdf(source, pdf_url, guideline_result)
                                        if pdf_result:
                                            source_result["pdfs_downloaded"].append(pdf_result)
                                            self.stats["pdfs_downloaded"] += 1

                    self.stats["guidelines_found"] += 1

                    # respect rate limiting
                    await asyncio.sleep(self.config["request_delay"])

                except Exception as e:
                    error_msg = f"Error processing {url}: {str(e)}"
                    source_result["errors"].append(error_msg)
                    self.logger.warning(error_msg)

            # generate source statistics
            source_result["statistics"] = {
                "total_pages_processed": len(guideline_urls),
                "guidelines_found": len(source_result["guidelines_found"]),
                "pdfs_downloaded": len(source_result["pdfs_downloaded"]),
                "error_count": len(source_result["errors"]),
                "processing_time": time.time()
                - time.time(),  # Will be calculated properly
            }

        except Exception as e:
            error_msg = f"Critical error harvesting {source.abbreviation}: {str(e)}"
            source_result["errors"].append(error_msg)
            self.logger.error(error_msg)

        return source_result

    async def _analyze_guideline_page(self, source, url: str) -> Dict[str, Any]:
        """Analyze if a page is a list page or single guideline page.
        
        Args:
            source: GuidelineSource configuration
            url: Page URL to analyze
            
        Returns:
            Analysis results with page type and found links
        """
        try:
            # Crawl the page
            result = await self.crawler.arun(
                url=url,
                word_count_threshold=self.config["crawl4ai_config"]["word_count_threshold"],
                bypass_cache=True,
            )
            
            if not result.success:
                return {
                    "is_list_page": False,
                    "pdf_links": [],
                    "guideline_links": [],
                    "confidence": 0.0
                }
            
            # Extract potential links
            pdf_links = self._find_pdf_urls(result, source.base_url)
            guideline_links = self._extract_guideline_links(result, source)
            
            # Remove PDF links from guideline links to avoid duplicates
            guideline_links = [link for link in guideline_links if not link.lower().endswith('.pdf')]
            
            # Heuristics to determine if this is a list page
            content = result.markdown or ""
            
            # Count guideline-related terms that suggest a list/index page
            list_indicators = [
                "clinical practice guidelines", "practice guidelines", "recommendations",
                "list of guidelines", "guideline library", "guideline index",
                "all guidelines", "guidelines and statements", "clinical standards",
                "consensus statements", "position statements", "advisory statements"
            ]
            
            list_score = sum(1 for indicator in list_indicators if indicator.lower() in content.lower())
            
            # Check for list-like HTML structures (many links, tables, etc.)
            html_content = getattr(result, 'html', '') or ''
            list_html_indicators = [
                '<table', '<ul', '<ol', '<dl',  # List structures
                'href=', 'download', 'pdf'      # Many links
            ]
            
            html_score = sum(html_content.lower().count(indicator) for indicator in list_html_indicators)
            
            # Decision logic
            is_list_page = (
                len(pdf_links) >= 3 or  # Many direct PDFs
                len(guideline_links) >= 5 or  # Many guideline links
                (list_score >= 2 and len(pdf_links) >= 1) or  # Strong text indicators + some PDFs
                (html_score >= 10 and len(guideline_links) >= 2)  # Many HTML structures + some links
            )
            
            confidence = min(1.0, (len(pdf_links) * 0.2 + len(guideline_links) * 0.1 + list_score * 0.3 + html_score * 0.01))
            
            if self.config.get("debug", False):
                self.logger.debug(f"📊 Page analysis for {url}:")
                self.logger.debug(f"  📄 PDFs found: {len(pdf_links)}")
                self.logger.debug(f"  🔗 Guideline links: {len(guideline_links)}")
                self.logger.debug(f"  📝 List text score: {list_score}")
                self.logger.debug(f"  🏗️ HTML structure score: {html_score}")
                self.logger.debug(f"  📋 Is list page: {is_list_page} (confidence: {confidence:.2f})")
            
            return {
                "is_list_page": is_list_page,
                "pdf_links": pdf_links,
                "guideline_links": guideline_links,
                "confidence": confidence
            }
            
        except Exception as e:
            if self.config.get("debug", False):
                self.logger.debug(f"💥 Error analyzing page {url}: {type(e).__name__}: {str(e)}")
            return {
                "is_list_page": False,
                "pdf_links": [],
                "guideline_links": [],
                "confidence": 0.0
            }

    async def _google_search(self, query: str, num_results: int = 15) -> List[str]:
        """Perform Google search and extract result URLs.
        
        Args:
            query: Search query
            num_results: Maximum number of results to return
            
        Returns:
            List of URLs from search results
        """
        try:
            # Use Google search via web scraping (careful to avoid blocking)
            search_url = f"https://www.google.com/search?q={urllib.parse.quote(query)}&num={num_results}"
            
            if self.config.get("debug", False):
                self.logger.debug(f"🔍 Google search URL: {search_url}")
            
            # Add delay and use proper headers to avoid blocking
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }
            
            async with self.session.get(search_url, headers=headers) as response:
                if response.status == 200:
                    html_content = await response.text()
                    
                    # Extract URLs from Google search results
                    urls = self._parse_google_results(html_content)
                    
                    if self.config.get("debug", False):
                        self.logger.debug(f"📋 Extracted {len(urls)} URLs from Google results")
                    
                    return urls[:num_results]
                else:
                    if self.config.get("debug", False):
                        self.logger.debug(f"❌ Google search failed with status {response.status}")
                    return []
                    
        except Exception as e:
            if self.config.get("debug", False):
                self.logger.debug(f"💥 Google search exception: {type(e).__name__}: {str(e)}")
            return []

    def _parse_google_results(self, html_content: str) -> List[str]:
        """Parse Google search results HTML to extract URLs.
        
        Args:
            html_content: Raw HTML from Google search results
            
        Returns:
            List of extracted URLs
        """
        import re
        
        urls = []
        
        # Regex patterns to extract URLs from Google search results
        # Google uses various patterns, so we try multiple
        patterns = [
            r'<a href="/url\?q=([^&"]+)',  # Standard Google result links
            r'<a[^>]+href="(https?://[^"]+)"[^>]*>',  # Direct links
            r'url\?q=([^&"]+)',  # URL parameter pattern
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, html_content)
            for match in matches:
                # Decode URL
                url = urllib.parse.unquote(match)
                
                # Filter out Google's own URLs and non-relevant results
                if (url.startswith('http') and 
                    'google.com' not in url and 
                    'youtube.com' not in url and
                    'wikipedia.org' not in url and
                    len(url) > 20):
                    urls.append(url)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_urls = []
        for url in urls:
            if url not in seen:
                seen.add(url)
                unique_urls.append(url)
        
        return unique_urls

    async def _discover_guideline_pages(self, source) -> List[str]:
        """Discover guideline pages using Google search.

        Args:
            source: GuidelineSource configuration

        Returns:
            List of URLs containing guidelines
        """
        if self.config.get("debug", False):
            self.logger.debug(
                f"🔍 Starting Google-based discovery for {source.abbreviation}"
            )
            self.logger.debug(f"📂 Base URL: {source.base_url}")

        discovered_urls = []
        
        # Google search queries for this organization
        domain = source.base_url.replace('https://', '').replace('http://', '')
        search_queries = [
            f'site:{domain} "clinical guidelines"',
            f'site:{domain} "practice guidelines"',
            f'site:{domain} "recommendations"',
            f'site:{domain} "consensus statement"',
            f'"{source.name}" guidelines filetype:pdf',
            f'"{source.abbreviation}" clinical guidelines',
        ]

        if self.config.get("debug", False):
            self.logger.debug(f"🔎 Google search queries: {search_queries}")

        try:
            for i, query in enumerate(search_queries):
                if self.config.get("debug", False):
                    self.logger.debug(f"🌐 Google search {i+1}/{len(search_queries)}: {query}")

                try:
                    # Perform Google search
                    search_results = await self._google_search(query, num_results=15)
                    
                    if search_results:
                        discovered_urls.extend(search_results)
                        
                        if self.config.get("debug", False):
                            self.logger.debug(f"📋 Found {len(search_results)} results for query: {query}")
                            for url in search_results[:3]:  # Show first 3 results
                                self.logger.debug(f"  📎 {url}")
                            if len(search_results) > 3:
                                self.logger.debug(f"  ... and {len(search_results) - 3} more")
                    else:
                        if self.config.get("debug", False):
                            self.logger.debug(f"❌ No results for query: {query}")

                    # Rate limit Google searches
                    await asyncio.sleep(2.0)
                    
                except Exception as e:
                    if self.config.get("debug", False):
                        self.logger.debug(f"💥 Error in Google search: {type(e).__name__}: {str(e)}")
                    self.logger.warning(f"Google search failed for query '{query}': {str(e)}")
                    continue

        except Exception as e:
            if self.config.get("debug", False):
                self.logger.debug(f"💥 Critical error in Google discovery: {type(e).__name__}: {str(e)}")
            self.logger.error(f"Error in Google discovery for {source.abbreviation}: {str(e)}")

        # Remove duplicates and filter to organization's domain
        unique_urls = []
        seen_urls = set()
        
        for url in discovered_urls:
            if url not in seen_urls and domain in url:
                unique_urls.append(url)
                seen_urls.add(url)
        
        # Limit results but be more generous since Google pre-filters relevance
        final_urls = unique_urls[:30]

        if self.config.get("debug", False):
            self.logger.debug(f"📊 Google discovery summary for {source.abbreviation}:")
            self.logger.debug(f"  🔗 Total URLs found: {len(discovered_urls)}")
            self.logger.debug(f"  🎯 Unique URLs: {len(unique_urls)}")
            self.logger.debug(f"  📋 Final list (limited): {len(final_urls)}")

        return final_urls

    def _extract_guideline_links(self, crawl_result, source) -> List[str]:
        """Extract guideline-related links from crawl result.

        Args:
            crawl_result: crawl4ai result object
            source: GuidelineSource configuration

        Returns:
            List of guideline URLs
        """
        guideline_links = []

        try:
            # parse links from the crawled content
            if hasattr(crawl_result, "links") and crawl_result.links:
                for link in crawl_result.links:
                    if isinstance(link, dict):
                        url = link.get("href", "")
                        text = link.get("text", "").lower()
                    elif isinstance(link, str):
                        url = link
                        text = ""
                    else:
                        continue

                    if self._is_guideline_link(url, text, source):
                        # convert relative URLs to absolute
                        if url.startswith("/"):
                            url = source.base_url + url
                        elif not url.startswith("http"):
                            continue
                        guideline_links.append(url)

            # check for PDF links directly in content
            if hasattr(crawl_result, "markdown") and crawl_result.markdown:
                pdf_links = self._extract_pdf_links_from_content(
                    crawl_result.markdown, source.base_url
                )
                guideline_links.extend(pdf_links)

        except Exception as e:
            self.logger.warning(f"Error extracting links: {str(e)}")

        return guideline_links

    def _is_guideline_link(self, url: str, text: str, source) -> bool:
        """Determine if a link is likely to contain guidelines.

        Args:
            url: Link URL
            text: Link text
            source: GuidelineSource configuration

        Returns:
            True if link is likely a guideline
        """
        # URL-based filtering
        url_lower = url.lower()
        guideline_indicators = [
            "guideline",
            "recommendation",
            "standard",
            "consensus",
            "practice",
            "clinical",
            "protocol",
            "pathway",
            "statement",
        ]

        if any(indicator in url_lower for indicator in guideline_indicators):
            return True

        # text-based filtering
        text_indicators = [
            "clinical practice guideline",
            "practice recommendation",
            "consensus statement",
            "clinical standard",
            "treatment guideline",
            "diagnostic guideline",
            "management guideline",
        ]

        if any(indicator in text for indicator in text_indicators):
            return True

        # PDF files are good candidates
        if url_lower.endswith(".pdf"):
            return True

        return False

    def _extract_pdf_links_from_content(self, content: str, base_url: str) -> List[str]:
        """Extract PDF links from page content.

        Args:
            content: Page content (markdown/text)
            base_url: Base URL for relative links

        Returns:
            List of PDF URLs
        """
        import re

        pdf_links = []

        # regex patterns for PDF links
        pdf_patterns = [
            r'href=["\']([^"\']*\.pdf[^"\']*)["\']',
            r"\[([^\]]*\.pdf[^\]]*)\]",
            r"(https?://[^\s\)]*\.pdf)",
        ]

        if self.config.get("debug", False):
            self.logger.debug(
                f"🔍 Extracting PDF links from content ({len(content)} chars)"
            )

        for pattern in pdf_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if self.config.get("debug", False) and matches:
                self.logger.debug(
                    f"📎 Pattern '{pattern}' found {len(matches)} matches"
                )

            for match in matches:
                url = match if isinstance(match, str) else match[0]

                # clean up URL - remove trailing punctuation
                original_url = url
                url = re.sub(r'[)\]\}>\'"]+$', "", url)

                if self.config.get("debug", False) and original_url != url:
                    self.logger.debug(f"🧹 Cleaned URL: '{original_url}' -> '{url}'")

                # convert relative URLs
                if url.startswith("/"):
                    url = base_url + url
                elif not url.startswith("http"):
                    if self.config.get("debug", False):
                        self.logger.debug(f"❌ Skipping invalid URL: '{url}'")
                    continue

                pdf_links.append(url)
                if self.config.get("debug", False):
                    self.logger.debug(f"✅ Added PDF link: {url}")

        if self.config.get("debug", False):
            self.logger.debug(f"📋 Total PDF links found: {len(pdf_links)}")

        return pdf_links

    async def _process_guideline_page(
        self, source, url: str
    ) -> Optional[Dict[str, Any]]:
        """Process a single guideline page to extract metadata and PDFs.

        Args:
            source: GuidelineSource configuration
            url: Page URL to process

        Returns:
            Guideline information dictionary
        """
        try:
            # Crawl the page
            result = await self.crawler.arun(
                url=url,
                word_count_threshold=self.config["crawl4ai_config"][
                    "word_count_threshold"
                ],
                bypass_cache=True,
            )

            if not result.success:
                self.logger.warning(f"Failed to crawl {url}: {result.error_message}")
                return None

            # extract metadata using our metadata extractor
            metadata = self.metadata_extractor.extract_metadata(result, source, url)

            # find PDF URLs
            pdf_urls = self._find_pdf_urls(result, source.base_url)

            guideline_info = {
                "url": url,
                "title": metadata.get("title", "Unknown Title"),
                "metadata": metadata,
                "pdf_urls": pdf_urls,
                "content_preview": result.markdown[:500] if result.markdown else "",
                "extraction_timestamp": datetime.now().isoformat(),
                "word_count": len(result.markdown.split()) if result.markdown else 0,
            }

            return guideline_info

        except Exception as e:
            self.logger.error(f"Error processing guideline page {url}: {str(e)}")
            return None

    def _find_pdf_urls(self, crawl_result, base_url: str) -> List[str]:
        """Find PDF URLs in the crawled content."""
        pdf_urls = []

        # check links
        if hasattr(crawl_result, "links") and crawl_result.links:
            for link in crawl_result.links:
                if isinstance(link, dict):
                    href = link.get("href", "")
                elif isinstance(link, str):
                    href = link
                else:
                    continue

                if href.lower().endswith(".pdf"):
                    if href.startswith("/"):
                        href = base_url + href
                    elif not href.startswith("http"):
                        continue
                    pdf_urls.append(href)

        # extract from content
        if hasattr(crawl_result, "markdown") and crawl_result.markdown:
            content_pdfs = self._extract_pdf_links_from_content(
                crawl_result.markdown, base_url
            )
            pdf_urls.extend(content_pdfs)

        return list(set(pdf_urls))  # remove duplicates

    async def _download_pdf(
        self, source, pdf_url: str, guideline_info: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Download and process a PDF file.

        Args:
            source: GuidelineSource configuration
            pdf_url: URL of the PDF to download
            guideline_info: Associated guideline information

        Returns:
            PDF processing result
        """
        try:
            if self.config.get("debug", False):
                self.logger.debug(f"📥 Attempting to download PDF: {pdf_url}")

            # generate filename
            filename = self._generate_pdf_filename(source, pdf_url, guideline_info)
            pdf_path = (
                Path(self.config["output_dir"]) / self.config["pdf_dir"] / filename
            )

            if self.config.get("debug", False):
                self.logger.debug(f"📁 Target filename: {filename}")

            # skip if already exists and resume mode is enabled
            if self.config["resume_mode"] and pdf_path.exists():
                if self.config.get("debug", False):
                    self.logger.debug(f"⏭️ PDF already exists (resume mode): {filename}")
                else:
                    self.logger.debug(f"PDF already exists: {filename}")
                return {
                    "url": pdf_url,
                    "filename": filename,
                    "status": "already_exists",
                    "file_path": str(pdf_path),
                }

            # download PDF
            if self.config.get("debug", False):
                self.logger.debug(f"🌐 Making HTTP request to: {pdf_url}")

            async with self.session.get(pdf_url) as response:
                if self.config.get("debug", False):
                    self.logger.debug(
                        f"📊 HTTP Response: {response.status} for {pdf_url}"
                    )
                    self.logger.debug(f"🔗 Response headers: {dict(response.headers)}")

                if response.status == 200:
                    content = await response.read()

                    if self.config.get("debug", False):
                        self.logger.debug(f"📦 Downloaded {len(content)} bytes")

                    # Validate PDF size
                    if len(content) < self.config["min_pdf_size"]:
                        if self.config.get("debug", False):
                            self.logger.debug(
                                f"❌ PDF too small: {len(content)} bytes < {self.config['min_pdf_size']} bytes"
                            )
                        self.logger.warning(f"PDF too small: {pdf_url}")
                        return None

                    if len(content) > self.config["max_pdf_size"]:
                        if self.config.get("debug", False):
                            self.logger.debug(
                                f"❌ PDF too large: {len(content)} bytes > {self.config['max_pdf_size']} bytes"
                            )
                        self.logger.warning(f"PDF too large: {pdf_url}")
                        return None

                    # save PDF
                    if self.config.get("debug", False):
                        self.logger.debug(f"💾 Saving PDF to: {pdf_path}")

                    with open(pdf_path, "wb") as f:
                        f.write(content)

                    # process PDF for additional metadata
                    pdf_metadata = None
                    if self.config["validate_pdfs"]:
                        if self.config.get("debug", False):
                            self.logger.debug(
                                f"🔍 Processing PDF metadata for: {filename}"
                            )
                        pdf_metadata = await self.pdf_processor.process_pdf(
                            pdf_path, guideline_info
                        )

                    result = {
                        "url": pdf_url,
                        "filename": filename,
                        "status": "downloaded",
                        "file_path": str(pdf_path),
                        "file_size": len(content),
                        "download_timestamp": datetime.now().isoformat(),
                        "pdf_metadata": pdf_metadata,
                    }

                    # save metadata
                    metadata_path = (
                        Path(self.config["output_dir"])
                        / self.config["metadata_dir"]
                        / f"{filename}.json"
                    )

                    if self.config.get("debug", False):
                        self.logger.debug(f"📄 Saving metadata to: {metadata_path}")

                    with open(metadata_path, "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "guideline_info": guideline_info,
                                "download_result": result,
                            },
                            f,
                            indent=2,
                            ensure_ascii=False,
                        )

                    self.stats["metadata_extracted"] += 1

                    if self.config.get("debug", False):
                        self.logger.debug(
                            f"✅ Successfully downloaded and processed: {filename}"
                        )

                    return result
                else:
                    if self.config.get("debug", False):
                        self.logger.debug(
                            f"❌ HTTP Error {response.status} for {pdf_url}"
                        )
                        if response.status == 403:
                            self.logger.debug(
                                f"🚫 403 Forbidden - Possible paywall or access restriction"
                            )
                        elif response.status == 404:
                            self.logger.debug(
                                f"🔍 404 Not Found - PDF may have been moved or deleted"
                            )
                        elif response.status == 429:
                            self.logger.debug(f"⏱️ 429 Rate Limited - Too many requests")

                    self.logger.warning(f"HTTP {response.status} for {pdf_url}")
                    return None

        except Exception as e:
            if self.config.get("debug", False):
                self.logger.debug(
                    f"💥 Exception during PDF download: {type(e).__name__}: {str(e)}"
                )
            self.logger.error(f"Error downloading PDF {pdf_url}: {str(e)}")
            return None

    def _generate_pdf_filename(
        self, source, pdf_url: str, guideline_info: Dict[str, Any]
    ) -> str:
        """Generate a standardized filename for the PDF."""
        import re
        from urllib.parse import urlparse

        # extract title from guideline info
        title = guideline_info.get("title", "unknown")

        # if title is generic or unknown, try to extract meaningful name from URL
        if title in ["unknown", "Unknown Title", "Direct PDF", ""] or title.startswith(
            "Direct PDF from"
        ):
            # extract filename from URL
            parsed_url = urlparse(pdf_url)
            url_filename = parsed_url.path.split("/")[-1]
            if url_filename and url_filename.endswith(".pdf"):
                # remove .pdf extension and use as title
                url_title = url_filename[:-4]
                # clean URL-derived title
                url_title = re.sub(r"[^a-zA-Z0-9\s_-]", "", url_title)
                url_title = re.sub(r"[-_]+", "_", url_title)
                if url_title and len(url_title) > 3:  # meaningful name
                    title = url_title

        # clean title for filename
        clean_title = re.sub(r"[^a-zA-Z0-9\s_-]", "", title)
        clean_title = re.sub(r"\s+", "_", clean_title.strip())
        clean_title = clean_title[:50]  # Limit length

        # extract year from metadata if available
        year = ""
        if "metadata" in guideline_info:
            for date_field in ["date", "publication_date", "year"]:
                if date_field in guideline_info["metadata"]:
                    date_val = str(guideline_info["metadata"][date_field])
                    year_match = re.search(r"(\d{4})", date_val)
                    if year_match:
                        year = f"_{year_match.group(1)}"
                        break

        # if still no meaningful title, extract from URL path
        if clean_title in ["unknown", ""] or len(clean_title) < 3:
            # try to get a meaningful identifier from URL
            url_parts = pdf_url.split("/")
            for part in reversed(url_parts):
                if part and not part.endswith(".pdf"):
                    clean_part = re.sub(r"[^a-zA-Z0-9_-]", "", part)
                    if clean_part and len(clean_part) > 3:
                        clean_title = clean_part[:20]
                        break

            # fallback to simple hash if still no good name
            if clean_title in ["unknown", ""] or len(clean_title) < 3:
                import hashlib

                url_hash = hashlib.md5(pdf_url.encode()).hexdigest()[:8]
                clean_title = f"pdf_{url_hash}"

        # construct filename
        filename = f"{source.abbreviation}_{clean_title}{year}.pdf"

        # ensure no duplicate names by adding timestamp if needed
        base_path = Path(self.config["output_dir"]) / self.config["pdf_dir"] / filename
        if base_path.exists():
            timestamp = datetime.now().strftime("%H%M%S")
            filename = f"{source.abbreviation}_{clean_title}{year}_{timestamp}.pdf"

        return filename

    def _generate_final_statistics(self) -> Dict[str, Any]:
        """Generate comprehensive harvesting statistics."""
        total_time = time.time() - self.stats["start_time"]

        return {
            "summary": {
                "total_time": total_time,
                "sources_processed": self.stats["sources_processed"],
                "guidelines_found": self.stats["guidelines_found"],
                "pdfs_downloaded": self.stats["pdfs_downloaded"],
                "metadata_extracted": self.stats["metadata_extracted"],
                "errors": self.stats["errors"],
            },
            "performance": {
                "sources_per_minute": (
                    self.stats["sources_processed"] / (total_time / 60)
                    if total_time > 0
                    else 0
                ),
                "pdfs_per_minute": (
                    self.stats["pdfs_downloaded"] / (total_time / 60)
                    if total_time > 0
                    else 0
                ),
                "success_rate": (
                    self.stats["pdfs_downloaded"]
                    / max(self.stats["guidelines_found"], 1)
                )
                * 100,
            },
        }

    def _save_harvest_results(self, results: Dict[str, Any]):
        """Save comprehensive harvest results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = (
            Path(self.config["output_dir"]) / f"harvest_results_{timestamp}.json"
        )

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        self.logger.info(f"📊 Results saved to {results_file}")

    def _display_harvest_summary(self, results: Dict[str, Any]):
        """Display a summary of the harvest results."""
        stats = results.get("statistics", {}).get("summary", {})

        summary_table = Table(
            title="🎯 Harvest Summary", show_header=True, header_style="bold green"
        )
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="magenta")

        summary_table.add_row(
            "Sources Processed", str(stats.get("sources_processed", 0))
        )
        summary_table.add_row("Guidelines Found", str(stats.get("guidelines_found", 0)))
        summary_table.add_row("PDFs Downloaded", str(stats.get("pdfs_downloaded", 0)))
        summary_table.add_row(
            "Metadata Extracted", str(stats.get("metadata_extracted", 0))
        )
        summary_table.add_row("Total Time", f"{stats.get('total_time', 0):.1f} seconds")

        self.console.print(summary_table)

    async def cleanup(self):
        """Cleanup resources."""
        if self.crawler:
            await self.crawler.close()

        if self.session:
            await self.session.close()


@click.command()
@click.option(
    "--sources",
    default="high_priority",
    help="Source filter: high_priority, cardiology, etc.",
)
@click.option("--max-concurrent", default=3, help="Maximum concurrent sources")
@click.option("--delay", default=1.0, help="Delay between requests")
@click.option("--output-dir", default="guidelines_data", help="Output directory")
@click.option("--resume/--no-resume", default=True, help="Resume interrupted downloads")
@click.option(
    "--validate-pdfs/--no-validate-pdfs", default=True, help="Validate downloaded PDFs"
)
@click.option("--debug", is_flag=True, help="Enable debug logging for troubleshooting")
def main(sources, max_concurrent, delay, output_dir, resume, validate_pdfs, debug):
    """Entry point for CLI, dispatches async runner."""
    config = {
        "max_concurrent_sources": max_concurrent,
        "request_delay": delay,
        "output_dir": output_dir,
        "resume_mode": resume,
        "validate_pdfs": validate_pdfs,
        "debug": debug,
    }
    asyncio.run(run(sources, config))


async def run(sources: str, config: Dict[str, Any]):
    """Clinical Guidelines Harvester - Async runner."""
    harvester = GuidelineHarvester(config)

    try:
        await harvester.initialize()
        results = await harvester.harvest_all_sources(sources)

        print(f"\n🎉 Harvest completed!")
        print(f"📁 Results saved to: {config['output_dir']}")
        print(
            f"📄 Guidelines found: {results['statistics']['summary']['guidelines_found']}"
        )
        print(
            f"📎 PDFs downloaded: {results['statistics']['summary']['pdfs_downloaded']}"
        )

    except KeyboardInterrupt:
        print("\n⚠️ Harvest interrupted by user")
    except Exception as e:
        print(f"\n❌ Harvest failed: {str(e)}")
    finally:
        await harvester.cleanup()


if __name__ == "__main__":
    main()
