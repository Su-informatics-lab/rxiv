"""PDF processing and validation for clinical guidelines

This module provides comprehensive PDF processing capabilities including
validation, metadata extraction, and text preview generation.
"""

import hashlib
import logging
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pdfplumber
import pymupdf  # fitz
import PyPDF2


class PDFProcessor:
    """Comprehensive PDF processor for clinical guidelines."""

    def __init__(self):
        """Initialize the PDF processor."""
        self.logger = logging.getLogger(__name__)

    async def process_pdf(
        self, pdf_path: Path, guideline_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process a PDF file to extract metadata and validate content.

        Args:
            pdf_path: Path to the PDF file
            guideline_info: Associated guideline information

        Returns:
            PDF processing results
        """
        try:
            result = {
                "file_path": str(pdf_path),
                "file_size": pdf_path.stat().st_size,
                "processing_timestamp": datetime.now().isoformat(),
                "validation": {},
                "metadata": {},
                "content_preview": {},
                "quality_assessment": {},
                "guideline_context": guideline_info.get("metadata", {}),
            }

            # basic file validation
            validation_result = self._validate_pdf_file(pdf_path)
            result["validation"] = validation_result

            if not validation_result["is_valid"]:
                self.logger.warning(f"PDF validation failed: {pdf_path}")
                return result

            # extract PDF metadata
            pdf_metadata = self._extract_pdf_metadata(pdf_path)
            result["metadata"] = pdf_metadata

            # generate content preview
            content_preview = self._generate_content_preview(pdf_path)
            result["content_preview"] = content_preview

            # assess PDF quality for guidelines (now using guideline context)
            quality_assessment = self._assess_pdf_quality(pdf_path, content_preview, guideline_info)
            result["quality_assessment"] = quality_assessment

            # generate file hash for deduplication
            result["file_hash"] = self._calculate_file_hash(pdf_path)

            return result

        except Exception as e:
            self.logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            return {
                "file_path": str(pdf_path),
                "error": str(e),
                "processing_timestamp": datetime.now().isoformat(),
            }

    def _validate_pdf_file(self, pdf_path: Path) -> Dict[str, Any]:
        """Validate PDF file integrity and format.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Validation results
        """
        validation = {
            "is_valid": False,
            "file_exists": False,
            "correct_mimetype": False,
            "readable": False,
            "page_count": 0,
            "file_size": 0,
            "errors": [],
        }

        try:
            # check file existence
            if not pdf_path.exists():
                validation["errors"].append("File does not exist")
                return validation

            validation["file_exists"] = True
            validation["file_size"] = pdf_path.stat().st_size

            # check file size
            if validation["file_size"] == 0:
                validation["errors"].append("File is empty")
                return validation

            if validation["file_size"] < 1024:  # Less than 1KB
                validation["errors"].append("File too small to be a valid PDF")
                return validation

            # check MIME type
            mime_type, _ = mimetypes.guess_type(str(pdf_path))
            if mime_type == "application/pdf":
                validation["correct_mimetype"] = True
            else:
                # also check by reading file header
                try:
                    with open(pdf_path, "rb") as f:
                        header = f.read(8)
                        if header.startswith(b"%PDF"):
                            validation["correct_mimetype"] = True
                        else:
                            validation["errors"].append("Invalid PDF header")
                except Exception as e:
                    validation["errors"].append(f"Cannot read file header: {str(e)}")

            # try to read PDF with PyPDF2
            try:
                with open(pdf_path, "rb") as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    validation["page_count"] = len(pdf_reader.pages)

                    if validation["page_count"] > 0:
                        validation["readable"] = True
                        # try to read first page to ensure it's not corrupted
                        first_page = pdf_reader.pages[0]
                        _ = first_page.extract_text()
                    else:
                        validation["errors"].append("PDF has no pages")

            except PyPDF2.errors.PdfReadError as e:
                validation["errors"].append(f"PyPDF2 read error: {str(e)}")
            except Exception as e:
                validation["errors"].append(f"Error reading PDF: {str(e)}")

            # overall validation
            validation["is_valid"] = (
                validation["file_exists"]
                and validation["correct_mimetype"]
                and validation["readable"]
                and validation["page_count"] > 0
                and len(validation["errors"]) == 0
            )

        except Exception as e:
            validation["errors"].append(f"Validation error: {str(e)}")

        return validation

    def _extract_pdf_metadata(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract metadata from PDF file.

        Args:
            pdf_path: Path to PDF file

        Returns:
            PDF metadata
        """
        metadata = {
            "extraction_method": [],
            "document_info": {},
            "structure_info": {},
            "text_info": {},
        }

        try:
            # PyPDF2 metadata extraction
            with open(pdf_path, "rb") as f:
                pdf_reader = PyPDF2.PdfReader(f)

                # document information
                if pdf_reader.metadata:
                    doc_info = {}
                    for key, value in pdf_reader.metadata.items():
                        # Clean up key names
                        clean_key = key.replace("/", "").lower()
                        if value:
                            doc_info[clean_key] = str(value)

                    metadata["document_info"] = doc_info
                    metadata["extraction_method"].append("PyPDF2")

                # structure information
                metadata["structure_info"] = {
                    "page_count": len(pdf_reader.pages),
                    "is_encrypted": pdf_reader.is_encrypted,
                    "has_outline": bool(pdf_reader.outline),
                }

        except Exception as e:
            self.logger.warning(f"PyPDF2 metadata extraction failed: {str(e)}")

        try:
            # PyMuPDF (fitz) metadata extraction
            pdf_doc = pymupdf.open(str(pdf_path))

            # document metadata
            pdf_metadata = pdf_doc.metadata
            if pdf_metadata:
                fitz_info = {}
                for key, value in pdf_metadata.items():
                    if value:
                        fitz_info[key.lower()] = str(value)

                # merge with existing metadata
                if not metadata["document_info"]:
                    metadata["document_info"] = fitz_info
                else:
                    metadata["document_info"].update(fitz_info)

                if "PyMuPDF" not in metadata["extraction_method"]:
                    metadata["extraction_method"].append("PyMuPDF")

            # additional structure info
            metadata["structure_info"].update(
                {
                    "page_count_fitz": pdf_doc.page_count,
                    "has_toc": bool(pdf_doc.get_toc()),
                    "is_pdf_a": pdf_doc.is_pdf,
                    "permissions": pdf_doc.permissions,
                }
            )

            pdf_doc.close()

        except Exception as e:
            self.logger.warning(f"PyMuPDF metadata extraction failed: {str(e)}")

        return metadata

    def _generate_content_preview(self, pdf_path: Path) -> Dict[str, Any]:
        """Generate content preview from PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Content preview information
        """
        preview = {
            "text_extraction_method": [],
            "first_page_text": "",
            "last_page_text": "",
            "total_text_length": 0,
            "word_count": 0,
            "has_meaningful_text": False,
            "detected_sections": [],
            "text_quality_score": 0.0,
        }

        try:
            # try pdfplumber first (better for structured documents)
            with pdfplumber.open(pdf_path) as pdf:
                all_text = ""

                # extract text from first few and last few pages
                total_pages = len(pdf.pages)
                pages_to_extract = min(5, total_pages)  # first and last 5 pages

                # first pages
                for i in range(min(pages_to_extract, total_pages)):
                    try:
                        page = pdf.pages[i]
                        page_text = page.extract_text() or ""
                        all_text += page_text + "\n"

                        if i == 0:
                            preview["first_page_text"] = page_text[
                                :1000
                            ]  # first 1000 chars
                    except Exception as e:
                        self.logger.debug(f"Error extracting page {i}: {str(e)}")

                # last pages (if document is long enough)
                if total_pages > pages_to_extract:
                    for i in range(
                        max(pages_to_extract, total_pages - pages_to_extract),
                        total_pages,
                    ):
                        try:
                            page = pdf.pages[i]
                            page_text = page.extract_text() or ""
                            all_text += page_text + "\n"

                            if i == total_pages - 1:
                                preview["last_page_text"] = page_text[
                                    -1000:
                                ]  # last 1000 chars
                        except Exception as e:
                            self.logger.debug(f"Error extracting page {i}: {str(e)}")

                preview["text_extraction_method"].append("pdfplumber")

                # analyze extracted text
                if all_text.strip():
                    preview["total_text_length"] = len(all_text)
                    preview["word_count"] = len(all_text.split())
                    preview["has_meaningful_text"] = self._has_meaningful_text(all_text)
                    preview["detected_sections"] = self._detect_sections(all_text)
                    preview["text_quality_score"] = self._calculate_text_quality_score(
                        all_text
                    )

        except Exception as e:
            self.logger.warning(f"pdfplumber extraction failed: {str(e)}")

        # fallback to PyMuPDF if pdfplumber failed or produced poor results
        if preview["total_text_length"] < 100:
            try:
                pdf_doc = pymupdf.open(str(pdf_path))
                all_text = ""

                total_pages = pdf_doc.page_count
                pages_to_extract = min(5, total_pages)

                # extract from first and last pages
                for i in range(min(pages_to_extract, total_pages)):
                    try:
                        page = pdf_doc[i]
                        page_text = page.get_text()
                        all_text += page_text + "\n"

                        if i == 0 and not preview["first_page_text"]:
                            preview["first_page_text"] = page_text[:1000]
                    except Exception as e:
                        self.logger.debug(
                            f"Error extracting page {i} with PyMuPDF: {str(e)}"
                        )

                if total_pages > pages_to_extract:
                    for i in range(
                        max(pages_to_extract, total_pages - pages_to_extract),
                        total_pages,
                    ):
                        try:
                            page = pdf_doc[i]
                            page_text = page.get_text()
                            all_text += page_text + "\n"

                            if i == total_pages - 1 and not preview["last_page_text"]:
                                preview["last_page_text"] = page_text[-1000:]
                        except Exception as e:
                            self.logger.debug(
                                f"Error extracting page {i} with PyMuPDF: {str(e)}"
                            )

                if all_text.strip() and len(all_text) > preview["total_text_length"]:
                    preview["total_text_length"] = len(all_text)
                    preview["word_count"] = len(all_text.split())
                    preview["has_meaningful_text"] = self._has_meaningful_text(all_text)
                    preview["detected_sections"] = self._detect_sections(all_text)
                    preview["text_quality_score"] = self._calculate_text_quality_score(
                        all_text
                    )
                    preview["text_extraction_method"].append("PyMuPDF")

                pdf_doc.close()

            except Exception as e:
                self.logger.warning(f"PyMuPDF extraction failed: {str(e)}")

        return preview

    def _has_meaningful_text(self, text: str) -> bool:
        """Determine if extracted text is meaningful.

        Args:
            text: Extracted text

        Returns:
            True if text appears meaningful
        """
        if len(text.strip()) < 100:
            return False

        # check for reasonable ratio of letters to other characters
        letter_count = sum(1 for c in text if c.isalpha())
        total_count = len(text)

        if total_count == 0:
            return False

        letter_ratio = letter_count / total_count

        # should have at least 40% letters for meaningful text
        if letter_ratio < 0.4:
            return False

        # check for common medical/clinical terms
        clinical_terms = [
            "patient",
            "treatment",
            "diagnosis",
            "clinical",
            "therapy",
            "medicine",
            "disease",
            "syndrome",
            "procedure",
            "guideline",
            "recommendation",
            "evidence",
            "study",
            "trial",
            "analysis",
        ]

        text_lower = text.lower()
        clinical_term_count = sum(1 for term in clinical_terms if term in text_lower)

        # should have at least some clinical terms for medical guidelines
        return clinical_term_count >= 3

    def _detect_sections(self, text: str) -> List[str]:
        """Detect common sections in clinical guidelines.

        Args:
            text: Full text content

        Returns:
            List of detected section types
        """
        sections = []
        text_lower = text.lower()

        # common guideline sections
        section_patterns = {
            "abstract": ["abstract", "summary", "executive summary"],
            "introduction": ["introduction", "background", "overview"],
            "methods": ["methods", "methodology", "approach"],
            "recommendations": [
                "recommendations",
                "clinical recommendations",
                "practice recommendations",
            ],
            "evidence": ["evidence", "evidence review", "literature review"],
            "implementation": ["implementation", "clinical implementation"],
            "conclusions": ["conclusions", "summary", "key points"],
            "references": ["references", "bibliography", "citations"],
            "appendix": ["appendix", "appendices", "supplementary"],
            "algorithm": ["algorithm", "flowchart", "decision tree"],
            "quality_measures": ["quality measures", "performance measures", "metrics"],
        }

        for section_type, patterns in section_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    sections.append(section_type)
                    break

        return sections

    def _calculate_text_quality_score(self, text: str) -> float:
        """Calculate a quality score for extracted text.

        Args:
            text: Extracted text

        Returns:
            Quality score between 0.0 and 1.0
        """
        if not text.strip():
            return 0.0

        score = 0.0

        # length score (longer is generally better for guidelines)
        word_count = len(text.split())
        if word_count > 1000:
            score += 0.3
        elif word_count > 500:
            score += 0.2
        elif word_count > 100:
            score += 0.1

        # coherence score (reasonable sentence structure)
        sentences = text.split(".")
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(
            len(sentences), 1
        )
        if 10 <= avg_sentence_length <= 30:  # Reasonable sentence length
            score += 0.2

        # clinical content score
        clinical_terms = [
            "patient",
            "treatment",
            "diagnosis",
            "clinical",
            "therapy",
            "medicine",
            "disease",
            "guideline",
            "recommendation",
            "evidence",
        ]
        text_lower = text.lower()
        clinical_density = (
            sum(text_lower.count(term) for term in clinical_terms) / word_count
        )
        if clinical_density > 0.01:  # At least 1% clinical terms
            score += 0.3

        # structure score (presence of sections)
        detected_sections = self._detect_sections(text)
        if len(detected_sections) >= 3:
            score += 0.2
        elif len(detected_sections) >= 1:
            score += 0.1

        return min(score, 1.0)

    def _assess_pdf_quality(
        self, pdf_path: Path, content_preview: Dict[str, Any], guideline_info: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Assess overall PDF quality for clinical guidelines.

        Args:
            pdf_path: Path to PDF file
            content_preview: Content preview information
            guideline_info: Associated guideline context for enhanced assessment

        Returns:
            Quality assessment
        """
        assessment = {
            "overall_score": 0.0,
            "is_likely_guideline": False,
            "quality_indicators": {},
            "issues": [],
            "recommendations": [],
        }

        try:
            score_components = {}

            # file size assessment
            file_size = pdf_path.stat().st_size
            if file_size > 1024 * 1024:  # > 1MB
                score_components["file_size"] = 0.3
            elif file_size > 100 * 1024:  # > 100KB
                score_components["file_size"] = 0.2
            else:
                score_components["file_size"] = 0.1
                assessment["issues"].append(
                    "File size may be too small for comprehensive guidelines"
                )

            # text quality assessment
            text_quality = content_preview.get("text_quality_score", 0.0)
            score_components["text_quality"] = text_quality * 0.4

            if text_quality < 0.3:
                assessment["issues"].append(
                    "Poor text extraction quality - may be scanned document"
                )
                assessment["recommendations"].append("Consider OCR processing")

            # content meaningfulness
            if content_preview.get("has_meaningful_text"):
                score_components["meaningful_content"] = 0.2
            else:
                assessment["issues"].append("No meaningful text detected")

            # section structure
            detected_sections = content_preview.get("detected_sections", [])
            if len(detected_sections) >= 3:
                score_components["structure"] = 0.2
            elif len(detected_sections) >= 1:
                score_components["structure"] = 0.1
            else:
                assessment["issues"].append("No clear section structure detected")

            # word count assessment
            word_count = content_preview.get("word_count", 0)
            if word_count >= 2000:
                score_components["length"] = 0.2
            elif word_count >= 1000:
                score_components["length"] = 0.15
            elif word_count >= 500:
                score_components["length"] = 0.1
            else:
                assessment["issues"].append(
                    "Document may be too short for comprehensive guidelines"
                )

            # guideline context bonus (use guideline_info to enhance assessment)
            if guideline_info:
                guideline_metadata = guideline_info.get("metadata", {})
                
                # bonus for having organization info
                if guideline_metadata.get("organization"):
                    score_components["guideline_context"] = 0.1
                
                # bonus for having specialty info
                if guideline_metadata.get("specialty"):
                    score_components["specialty_match"] = 0.1
                
                # bonus for having proper guideline type
                if guideline_metadata.get("guideline_type"):
                    score_components["type_validation"] = 0.1
                
                # title consistency check
                guideline_title = guideline_info.get("title", "").lower()
                if guideline_title and any(
                    term in guideline_title 
                    for term in ["guideline", "recommendation", "standard", "consensus"]
                ):
                    score_components["title_consistency"] = 0.1

            # calculate overall score
            assessment["overall_score"] = sum(score_components.values())
            assessment["quality_indicators"] = score_components

            # determine if likely a clinical guideline (enhanced with guideline context)
            base_criteria = (
                assessment["overall_score"] >= 0.6
                and content_preview.get("has_meaningful_text", False)
                and len(detected_sections) >= 2
                and word_count >= 500
            )
            
            # additional confidence from guideline context
            context_boost = bool(guideline_info and guideline_info.get("metadata", {}))
            
            assessment["is_likely_guideline"] = base_criteria or (
                assessment["overall_score"] >= 0.5 and context_boost
            )

            # generate recommendations
            if assessment["overall_score"] < 0.5:
                assessment["recommendations"].append(
                    "Review document manually for relevance"
                )

            if not assessment["is_likely_guideline"]:
                assessment["recommendations"].append(
                    "May not be a clinical guideline document"
                )

        except Exception as e:
            self.logger.error(f"Error assessing PDF quality: {str(e)}")
            assessment["error"] = str(e)

        return assessment

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file for deduplication.

        Args:
            file_path: Path to file

        Returns:
            SHA256 hash as hexadecimal string
        """
        hash_sha256 = hashlib.sha256()

        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            self.logger.error(f"Error calculating file hash: {str(e)}")
            return ""
