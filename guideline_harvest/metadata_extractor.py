"""Metadata extraction for clinical guidelines

This module extracts metadata from clinical guideline web pages and documents using
multiple extraction strategies.

"""

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import spacy
from bs4 import BeautifulSoup
from dateparser import parse as parse_date


@dataclass
class ExtractedMetadata:
    """Structured clinical guideline metadata."""

    title: Optional[str] = None
    authors: List[str] = None
    publication_date: Optional[str] = None
    last_updated: Optional[str] = None
    organization: Optional[str] = None
    specialty: Optional[str] = None
    guideline_type: Optional[str] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None
    evidence_level: Optional[str] = None
    recommendation_grade: Optional[str] = None
    target_population: Optional[str] = None
    clinical_question: Optional[str] = None
    key_recommendations: List[str] = None
    methodology: Optional[str] = None
    version: Optional[str] = None
    status: Optional[str] = None  # draft, final, updated, withdrawn
    keywords: List[str] = None
    abstract: Optional[str] = None
    external_review: Optional[bool] = None
    funding_source: Optional[str] = None
    conflicts_of_interest: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {}
        for field, value in self.__dict__.items():
            if value is not None:
                result[field] = value
        return result


class MetadataExtractor:
    """Advanced metadata extractor for clinical guidelines."""

    def __init__(self):
        """Initialize the metadata extractor."""
        self.logger = logging.getLogger(__name__)

        # load spaCy model for NLP processing
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.logger.warning(
                "spaCy English model not found. Install with: python -m spacy download en_core_web_sm"
            )
            self.nlp = None

        # precompiled regex patterns for common metadata
        self.patterns = self._compile_patterns()

    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for metadata extraction."""
        return {
            # basic patterns
            "doi": re.compile(
                r"(?:doi:?\s*|DOI:?\s*)(10\.\d+/[^\s<>\"]+)", re.IGNORECASE
            ),
            "pmid": re.compile(r"(?:pmid:?\s*|PMID:?\s*)(\d{8,})", re.IGNORECASE),
            "version": re.compile(r"version\s*:?\s*(\d+\.?\d*)", re.IGNORECASE),
            # date patterns
            "date_patterns": [
                re.compile(
                    r"(\w+\s+\d{1,2},\s+\d{4})", re.IGNORECASE
                ),  # January 15, 2024
                re.compile(r"(\d{1,2}/\d{1,2}/\d{4})"),  # 01/15/2024
                re.compile(r"(\d{4}-\d{2}-\d{2})"),  # 2024-01-15
                re.compile(
                    r"(\d{1,2}\s+\w+\s+\d{4})", re.IGNORECASE
                ),  # 15 January 2024
            ],
            # evidence and recommendation grades
            "evidence_level": re.compile(
                r"(?:evidence\s+level|level\s+of\s+evidence)\s*:?\s*([A-D]|I{1,3}|1-4)",
                re.IGNORECASE,
            ),
            "recommendation_grade": re.compile(
                r"(?:grade|recommendation\s+grade|class)\s*:?\s*([A-D]|I{1,3}|1A?|2A?|3)",
                re.IGNORECASE,
            ),
            # clinical specialty patterns
            "specialty_indicators": re.compile(
                r"(cardiology|oncology|neurology|endocrinology|gastroenterology|pulmonology|nephrology|rheumatology|infectious\s+disease|emergency\s+medicine|internal\s+medicine|family\s+medicine|pediatrics|obstetrics|gynecology|psychiatry|dermatology|ophthalmology|otolaryngology|orthopedics|urology|anesthesiology|radiology|pathology|surgery)",
                re.IGNORECASE,
            ),
            # guideline type patterns
            "guideline_type": re.compile(
                r"(practice\s+guideline|clinical\s+practice\s+guideline|consensus\s+statement|position\s+statement|technical\s+review|systematic\s+review|meta-analysis|clinical\s+pathway|treatment\s+algorithm|diagnostic\s+criteria|screening\s+guideline)",
                re.IGNORECASE,
            ),
            # status patterns
            "status": re.compile(
                r"\b(draft|final|updated|revised|withdrawn|superseded|current|archived)\b",
                re.IGNORECASE,
            ),
            # author patterns
            "authors": re.compile(
                r"(?:authors?|by)\s*:?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*(?:\s*,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)*)",
                re.IGNORECASE,
            ),
            # organization patterns
            "organization": re.compile(
                r"(American\s+(?:Heart|Cancer|Diabetes|College|Academy|Association|Society)|Society\s+of|College\s+of|Association\s+of|Institute\s+of|Center\s+for|Centers\s+for|National\s+\w+|International\s+\w+)",
                re.IGNORECASE,
            ),
        }

    def extract_metadata(self, crawl_result, source, url: str) -> Dict[str, Any]:
        """Extract comprehensive metadata from a crawled guideline page.

        Args:
            crawl_result: crawl4ai result object
            source: GuidelineSource configuration
            url: Page URL

        Returns:
            Extracted metadata dictionary
        """
        metadata = ExtractedMetadata()

        try:
            # extract from HTML if available
            if hasattr(crawl_result, "html") and crawl_result.html:
                self._extract_from_html(crawl_result.html, metadata, source)

            # extract from markdown content
            if hasattr(crawl_result, "markdown") and crawl_result.markdown:
                self._extract_from_text(crawl_result.markdown, metadata, source)

            # extract from structured data if available
            if (
                hasattr(crawl_result, "extracted_content")
                and crawl_result.extracted_content
            ):
                self._extract_from_structured_data(
                    crawl_result.extracted_content, metadata
                )

            # apply source-specific patterns
            self._apply_source_patterns(crawl_result, metadata, source)

            # post-process and validate
            self._post_process_metadata(metadata, url, source)

        except Exception as e:
            self.logger.error(f"Error extracting metadata from {url}: {str(e)}")

        return metadata.to_dict()

    def _extract_from_html(self, html: str, metadata: ExtractedMetadata, source):
        """Extract metadata from HTML content."""
        try:
            soup = BeautifulSoup(html, "html.parser")

            # title extraction
            if not metadata.title:
                # try multiple title sources
                title_sources = [
                    soup.find("title"),
                    soup.find("h1"),
                    soup.find("meta", {"property": "og:title"}),
                    soup.find("meta", {"name": "title"}),
                ]

                for title_elem in title_sources:
                    if title_elem:
                        if title_elem.name == "meta":
                            title = title_elem.get("content", "").strip()
                        else:
                            title = title_elem.get_text(strip=True)

                        if title and len(title) > 10:  # reasonable title length
                            metadata.title = self._clean_title(title)
                            break

            # meta tags extraction
            meta_tags = soup.find_all("meta")
            for meta in meta_tags:
                name = meta.get("name", "").lower()
                property_name = meta.get("property", "").lower()
                content = meta.get("content", "").strip()

                if not content:
                    continue

                # date extraction from meta tags
                if name in [
                    "date",
                    "publication-date",
                    "article:published_time",
                ] or property_name in ["article:published_time"]:
                    parsed_date = parse_date(content)
                    if parsed_date:
                        metadata.publication_date = parsed_date.isoformat()

                # author extraction
                if name in ["author", "authors"] or property_name in ["article:author"]:
                    if not metadata.authors:
                        metadata.authors = []
                    metadata.authors.extend(self._parse_authors(content))

                # keywords
                if name in ["keywords", "article:tag"]:
                    if not metadata.keywords:
                        metadata.keywords = []
                    metadata.keywords.extend([k.strip() for k in content.split(",")])

                # description/abstract
                if name in ["description", "abstract"] and not metadata.abstract:
                    metadata.abstract = content

                # DOI
                if name in ["doi", "dc.identifier"] and not metadata.doi:
                    doi_match = self.patterns["doi"].search(content)
                    if doi_match:
                        metadata.doi = doi_match.group(1)

            # structured data extraction (JSON-LD, microdata, etc.)
            self._extract_structured_data_from_html(soup, metadata)

        except Exception as e:
            self.logger.warning(f"Error extracting from HTML: {str(e)}")

    def _extract_from_text(self, text: str, metadata: ExtractedMetadata, source):
        """Extract metadata from text content using NLP and regex."""
        try:
            # title extraction if not already found
            if not metadata.title:
                lines = text.split("\n")
                # often the first meaningful line is the title
                for line in lines[:10]:  # check first 10 lines
                    line = line.strip()
                    if len(line) > 20 and len(line) < 200:  # reasonable title length
                        # check if it looks like a title (not URL, not too many special chars)
                        if not line.startswith("http") and line.count("|") < 3:
                            metadata.title = self._clean_title(line)
                            break

            # DOI extraction
            if not metadata.doi:
                doi_match = self.patterns["doi"].search(text)
                if doi_match:
                    metadata.doi = doi_match.group(1)

            # PMID extraction
            if not metadata.pmid:
                pmid_match = self.patterns["pmid"].search(text)
                if pmid_match:
                    metadata.pmid = pmid_match.group(1)

            # version extraction
            if not metadata.version:
                version_match = self.patterns["version"].search(text)
                if version_match:
                    metadata.version = version_match.group(1)

            # date extraction
            if not metadata.publication_date:
                for pattern in self.patterns["date_patterns"]:
                    date_match = pattern.search(text)
                    if date_match:
                        parsed_date = parse_date(date_match.group(1))
                        if parsed_date:
                            metadata.publication_date = parsed_date.isoformat()
                            break

            # evidence level and recommendation grade
            if not metadata.evidence_level:
                evidence_match = self.patterns["evidence_level"].search(text)
                if evidence_match:
                    metadata.evidence_level = evidence_match.group(1)

            if not metadata.recommendation_grade:
                grade_match = self.patterns["recommendation_grade"].search(text)
                if grade_match:
                    metadata.recommendation_grade = grade_match.group(1)

            # specialty detection
            if not metadata.specialty:
                specialty_match = self.patterns["specialty_indicators"].search(text)
                if specialty_match:
                    metadata.specialty = (
                        specialty_match.group(1).lower().replace(" ", "_")
                    )

            # guideline type detection
            if not metadata.guideline_type:
                type_match = self.patterns["guideline_type"].search(text)
                if type_match:
                    metadata.guideline_type = (
                        type_match.group(1).lower().replace(" ", "_")
                    )

            # status detection
            if not metadata.status:
                status_match = self.patterns["status"].search(text)
                if status_match:
                    metadata.status = status_match.group(1).lower()

            # author extraction
            if not metadata.authors:
                author_match = self.patterns["authors"].search(text)
                if author_match:
                    metadata.authors = self._parse_authors(author_match.group(1))

            # organization detection
            if not metadata.organization:
                org_match = self.patterns["organization"].search(text)
                if org_match:
                    metadata.organization = org_match.group(1)

            # NLP-based extraction if spaCy is available
            if self.nlp:
                self._extract_with_nlp(text, metadata)

        except Exception as e:
            self.logger.warning(f"Error extracting from text: {str(e)}")

    def _extract_structured_data_from_html(
        self, soup: BeautifulSoup, metadata: ExtractedMetadata
    ):
        """Extract metadata from structured data in HTML."""
        try:
            # JSON-LD extraction
            json_ld_scripts = soup.find_all("script", type="application/ld+json")
            for script in json_ld_scripts:
                try:
                    data = json.loads(script.string)
                    self._extract_from_json_ld(data, metadata)
                except (json.JSONDecodeError, AttributeError):
                    continue

            # Microdata extraction
            # look for common microdata vocabularies
            schema_items = soup.find_all(attrs={"itemtype": True})
            for item in schema_items:
                itemtype = item.get("itemtype", "")
                if "schema.org" in itemtype:
                    self._extract_from_microdata(item, metadata)

        except Exception as e:
            self.logger.warning(f"Error extracting structured data: {str(e)}")

    def _extract_from_json_ld(
        self, data: Union[Dict, List], metadata: ExtractedMetadata
    ):
        """Extract metadata from JSON-LD structured data."""
        try:
            if isinstance(data, list):
                for item in data:
                    self._extract_from_json_ld(item, metadata)
                return

            if not isinstance(data, dict):
                return

            # handle different schema.org types
            schema_type = data.get("@type", "").lower()

            if schema_type in ["article", "medicalguidelinetext", "medicalwebpage"]:
                # title
                if not metadata.title and "headline" in data:
                    metadata.title = data["headline"]
                elif not metadata.title and "name" in data:
                    metadata.title = data["name"]

                # date
                if not metadata.publication_date and "datePublished" in data:
                    parsed_date = parse_date(data["datePublished"])
                    if parsed_date:
                        metadata.publication_date = parsed_date.isoformat()

                # authors
                if not metadata.authors and "author" in data:
                    authors = data["author"]
                    if isinstance(authors, list):
                        metadata.authors = [
                            self._extract_person_name(author) for author in authors
                        ]
                    else:
                        metadata.authors = [self._extract_person_name(authors)]

                # organization
                if not metadata.organization and "publisher" in data:
                    publisher = data["publisher"]
                    if isinstance(publisher, dict):
                        metadata.organization = publisher.get("name", "")
                    else:
                        metadata.organization = str(publisher)

                # abstract/description
                if not metadata.abstract and "description" in data:
                    metadata.abstract = data["description"]

                # keywords
                if not metadata.keywords and "keywords" in data:
                    keywords = data["keywords"]
                    if isinstance(keywords, list):
                        metadata.keywords = keywords
                    elif isinstance(keywords, str):
                        metadata.keywords = [k.strip() for k in keywords.split(",")]

        except Exception as e:
            self.logger.warning(f"Error extracting from JSON-LD: {str(e)}")

    def _extract_person_name(self, person_data: Union[Dict, str]) -> str:
        """Extract person name from structured data."""
        if isinstance(person_data, str):
            return person_data
        elif isinstance(person_data, dict):
            if "name" in person_data:
                return person_data["name"]
            elif "givenName" in person_data and "familyName" in person_data:
                return f"{person_data['givenName']} {person_data['familyName']}"
        return ""

    def _extract_from_microdata(self, item, metadata: ExtractedMetadata):
        """Extract metadata from microdata."""
        try:
            # find properties within this item
            properties = item.find_all(attrs={"itemprop": True})

            for prop in properties:
                prop_name = prop.get("itemprop", "").lower()

                if prop_name == "headline" and not metadata.title:
                    metadata.title = prop.get_text(strip=True)
                elif prop_name == "author" and not metadata.authors:
                    if not metadata.authors:
                        metadata.authors = []
                    metadata.authors.append(prop.get_text(strip=True))
                elif prop_name == "datepublished" and not metadata.publication_date:
                    date_text = prop.get("datetime") or prop.get_text(strip=True)
                    parsed_date = parse_date(date_text)
                    if parsed_date:
                        metadata.publication_date = parsed_date.isoformat()
                elif prop_name == "publisher" and not metadata.organization:
                    metadata.organization = prop.get_text(strip=True)
                elif prop_name == "description" and not metadata.abstract:
                    metadata.abstract = prop.get_text(strip=True)

        except Exception as e:
            self.logger.warning(f"Error extracting from microdata: {str(e)}")

    def _extract_with_nlp(self, text: str, metadata: ExtractedMetadata):
        """Use spaCy NLP for advanced metadata extraction."""
        try:
            doc = self.nlp(text[:10000])  # Limit text length for performance

            # extract organizations
            if not metadata.organization:
                orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
                if orgs:
                    # find the most likely organization (longest or most frequent)
                    metadata.organization = max(orgs, key=len)

            # extract persons (potential authors)
            if not metadata.authors:
                persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
                if persons:
                    metadata.authors = persons[:10]  # Limit to first 10

            # extract dates
            if not metadata.publication_date:
                dates = [ent.text for ent in doc.ents if ent.label_ == "DATE"]
                for date_text in dates:
                    parsed_date = parse_date(date_text)
                    if (
                        parsed_date and parsed_date.year >= 1990
                    ):  # reasonable date range
                        metadata.publication_date = parsed_date.isoformat()
                        break

        except Exception as e:
            self.logger.warning(f"Error in NLP extraction: {str(e)}")

    def _apply_source_patterns(self, crawl_result, metadata: ExtractedMetadata, source):
        """Apply source-specific metadata patterns."""
        try:
            content = ""
            if hasattr(crawl_result, "markdown"):
                content += crawl_result.markdown or ""
            if hasattr(crawl_result, "html"):
                content += crawl_result.html or ""

            # apply patterns from source configuration
            for pattern_name, pattern in source.metadata_patterns.items():
                if isinstance(pattern, str):
                    # convert string pattern to regex
                    regex_pattern = re.compile(pattern, re.IGNORECASE | re.DOTALL)
                    match = regex_pattern.search(content)

                    if match:
                        value = (
                            match.group(1).strip()
                            if match.groups()
                            else match.group().strip()
                        )

                        # map pattern names to metadata fields
                        if pattern_name == "title" and not metadata.title:
                            metadata.title = self._clean_title(value)
                        elif pattern_name == "authors" and not metadata.authors:
                            metadata.authors = self._parse_authors(value)
                        elif (
                            pattern_name in ["date", "publication_date"]
                            and not metadata.publication_date
                        ):
                            parsed_date = parse_date(value)
                            if parsed_date:
                                metadata.publication_date = parsed_date.isoformat()
                        elif (
                            pattern_name == "organization" and not metadata.organization
                        ):
                            metadata.organization = value
                        elif pattern_name == "category" and not metadata.specialty:
                            metadata.specialty = value.lower().replace(" ", "_")

            # set organization from source if not found
            if not metadata.organization:
                metadata.organization = source.name

        except Exception as e:
            self.logger.warning(f"Error applying source patterns: {str(e)}")

    def _post_process_metadata(self, metadata: ExtractedMetadata, url: str, source):
        """Post-process and validate extracted metadata."""
        try:
            # clean and validate title
            if metadata.title:
                metadata.title = self._clean_title(metadata.title)
                if len(metadata.title) < 10:  # Too short to be meaningful
                    metadata.title = None

            # deduplicate and clean authors
            if metadata.authors:
                seen = set()
                clean_authors = []
                for author in metadata.authors:
                    clean_author = self._clean_author_name(author)
                    if (
                        clean_author
                        and clean_author not in seen
                        and len(clean_author) > 3
                    ):
                        clean_authors.append(clean_author)
                        seen.add(clean_author)
                metadata.authors = clean_authors[:20]  # Limit to reasonable number

            # clean and deduplicate keywords
            if metadata.keywords:
                seen = set()
                clean_keywords = []
                for keyword in metadata.keywords:
                    clean_keyword = keyword.strip().lower()
                    if (
                        clean_keyword
                        and clean_keyword not in seen
                        and len(clean_keyword) > 2
                    ):
                        clean_keywords.append(clean_keyword)
                        seen.add(clean_keyword)
                metadata.keywords = clean_keywords[:30]  # Limit keywords

            # validate dates
            if metadata.publication_date:
                try:
                    parsed = parse_date(metadata.publication_date)
                    if (
                        parsed
                        and parsed.year < 1990
                        or parsed.year > datetime.now().year + 1
                    ):
                        metadata.publication_date = None
                except:
                    metadata.publication_date = None

            # set specialty from source if not detected
            if not metadata.specialty and source.specialties:
                metadata.specialty = source.specialties[0]

            # set guideline type from source if not detected
            if not metadata.guideline_type and source.guideline_types:
                metadata.guideline_type = source.guideline_types[0].value

            # add extraction metadata
            metadata.extraction_url = url
            metadata.extraction_timestamp = datetime.now().isoformat()
            metadata.source_organization = source.name

        except Exception as e:
            self.logger.warning(f"Error in post-processing: {str(e)}")

    def _clean_title(self, title: str) -> str:
        """Clean and normalize title text."""
        if not title:
            return ""

        # remove common prefixes and suffixes
        title = re.sub(
            r"^(.*?)\s*[-|:]\s*$", r"\1", title
        )  # remove trailing separators
        title = re.sub(r"^\s*[-|:]\s*(.*)", r"\1", title)  # remove leading separators

        # remove site names and common suffixes
        suffixes_to_remove = [
            r"\s*[-|]\s*(CDC|FDA|AHA|ACC|ASCO|IDSA|ADA|ACOG|AAP).*$",
            r"\s*[-|]\s*Home.*$",
            r"\s*[-|]\s*Guidelines.*$",
            r"\s*\|\s*.*$",  # remove everything after pipe
        ]

        for suffix_pattern in suffixes_to_remove:
            title = re.sub(suffix_pattern, "", title, flags=re.IGNORECASE)

        # clean up whitespace and special characters
        title = re.sub(r"\s+", " ", title)  # multiple spaces to single
        title = title.strip()

        return title

    def _clean_author_name(self, author: str) -> str:
        """Clean and normalize author name."""
        if not author:
            return ""

        # remove common suffixes
        author = re.sub(
            r"\s*(MD|PhD|MPH|DO|RN|PharmD|MS|MA|BSN)\.?\s*$",
            "",
            author,
            flags=re.IGNORECASE,
        )

        # remove email addresses
        author = re.sub(
            r"\s*[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\s*", "", author
        )

        # clean whitespace
        author = re.sub(r"\s+", " ", author).strip()

        return author

    def _parse_authors(self, authors_text: str) -> List[str]:
        """Parse author text into individual authors."""
        if not authors_text:
            return []

        # split by common separators
        separators = [",", ";", " and ", " & ", "\n"]
        authors = [authors_text]

        for sep in separators:
            new_authors = []
            for author in authors:
                new_authors.extend(author.split(sep))
            authors = new_authors

        # clean each author
        cleaned_authors = []
        for author in authors:
            clean_author = self._clean_author_name(author)
            if clean_author and len(clean_author) > 3:
                cleaned_authors.append(clean_author)

        return cleaned_authors

    def _extract_from_structured_data(
        self, structured_data: Dict, metadata: ExtractedMetadata
    ):
        """Extract metadata from pre-structured data."""
        try:
            # this method can be extended to handle specific structured data formats
            # that crawl4ai might provide

            if isinstance(structured_data, dict):
                # look for common fields
                if "title" in structured_data and not metadata.title:
                    metadata.title = structured_data["title"]

                if "authors" in structured_data and not metadata.authors:
                    authors = structured_data["authors"]
                    if isinstance(authors, list):
                        metadata.authors = authors
                    elif isinstance(authors, str):
                        metadata.authors = self._parse_authors(authors)

                if "date" in structured_data and not metadata.publication_date:
                    parsed_date = parse_date(str(structured_data["date"]))
                    if parsed_date:
                        metadata.publication_date = parsed_date.isoformat()

        except Exception as e:
            self.logger.warning(f"Error extracting from structured data: {str(e)}")
