"""Major US Clinical Guidelines Sources Configuration

This module defines comprehensive sources for clinical guidelines from major
US medical organizations, government agencies, and professional societies.

"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List


class GuidelineType(Enum):
    """Types of clinical guidelines."""

    DIAGNOSIS = "diagnosis"
    TREATMENT = "treatment"
    SCREENING = "screening"
    PREVENTION = "prevention"
    MANAGEMENT = "management"
    PROCEDURE = "procedure"
    MEDICATION = "medication"
    EMERGENCY = "emergency"


class OrganizationType(Enum):
    """Types of organizations publishing guidelines."""

    GOVERNMENT = "government"
    PROFESSIONAL_SOCIETY = "professional_society"
    ACADEMIC = "academic"
    CONSORTIUM = "consortium"
    INTERNATIONAL = "international"


@dataclass
class GuidelineSource:
    """Configuration for a clinical guideline source."""

    name: str
    abbreviation: str
    base_url: str
    organization_type: OrganizationType
    specialties: List[str]
    guideline_types: List[GuidelineType]
    search_patterns: List[str]
    pdf_patterns: List[str]
    metadata_patterns: Dict[str, str]
    crawl_config: Dict[str, Any]
    priority: int  # 1=highest, 5=lowest


# Major US Clinical Guidelines Sources
CLINICAL_GUIDELINE_SOURCES = {
    # === GOVERNMENT AGENCIES ===
    "cdc": GuidelineSource(
        name="Centers for Disease Control and Prevention",
        abbreviation="CDC",
        base_url="https://www.cdc.gov",
        organization_type=OrganizationType.GOVERNMENT,
        specialties=[
            "infectious_disease",
            "epidemiology",
            "public_health",
            "prevention",
        ],
        guideline_types=[
            GuidelineType.PREVENTION,
            GuidelineType.SCREENING,
            GuidelineType.MANAGEMENT,
        ],
        search_patterns=[
            "/guidelines/",
            "/recommendations/",
            "/mmwr/",
            "/vaccines/",
            "/std/",
            "/cancer/",
            "/diabetes/",
            "/heart-disease/",
        ],
        pdf_patterns=[r"\.pdf$", r"guidelines.*\.pdf", r"recommendations.*\.pdf"],
        metadata_patterns={
            "title": r"<title>(.*?)</title>",
            "date": r"(\d{1,2}/\d{1,2}/\d{4}|\w+\s+\d{1,2},\s+\d{4})",
            "authors": r"Authors?:\s*(.*?)(?:\n|<)",
            "category": r"Category:\s*(.*?)(?:\n|<)",
        },
        crawl_config={
            "max_depth": 3,
            "follow_links": True,
            "respect_robots": True,
            "delay": 1.0,
        },
        priority=1,
    ),
    "cms": GuidelineSource(
        name="Centers for Medicare & Medicaid Services",
        abbreviation="CMS",
        base_url="https://www.cms.gov",
        organization_type=OrganizationType.GOVERNMENT,
        specialties=["healthcare_policy", "quality_measures", "coverage"],
        guideline_types=[GuidelineType.MANAGEMENT, GuidelineType.PROCEDURE],
        search_patterns=[
            "/medicare/coverage/",
            "/medicaid/",
            "/guidelines/",
            "/regulations/",
        ],
        pdf_patterns=[r"\.pdf$", r"coverage.*\.pdf", r"manual.*\.pdf"],
        metadata_patterns={
            "title": r"<title>(.*?)</title>",
            "effective_date": r"Effective Date:\s*(\d{1,2}/\d{1,2}/\d{4})",
            "policy": r"Policy:\s*(.*?)(?:\n|<)",
        },
        crawl_config={"max_depth": 2, "follow_links": True, "delay": 2.0},
        priority=2,
    ),
    "fda": GuidelineSource(
        name="Food and Drug Administration",
        abbreviation="FDA",
        base_url="https://www.fda.gov",
        organization_type=OrganizationType.GOVERNMENT,
        specialties=["drug_safety", "medical_devices", "food_safety"],
        guideline_types=[GuidelineType.MEDICATION, GuidelineType.PROCEDURE],
        search_patterns=[
            "/drugs/guidance/",
            "/medical-devices/guidance/",
            "/vaccines/",
        ],
        pdf_patterns=[r"\.pdf$", r"guidance.*\.pdf", r"draft.*\.pdf"],
        metadata_patterns={
            "title": r"<title>(.*?)</title>",
            "date": r"Date:\s*(\w+\s+\d{4})",
            "docket": r"Docket.*?:\s*(FDA-\d+-\w+-\d+)",
        },
        crawl_config={"max_depth": 2, "follow_links": True, "delay": 1.5},
        priority=2,
    ),
    # === PROFESSIONAL SOCIETIES ===
    "aha": GuidelineSource(
        name="American Heart Association",
        abbreviation="AHA",
        base_url="https://www.ahajournals.org",
        organization_type=OrganizationType.PROFESSIONAL_SOCIETY,
        specialties=["cardiology", "cardiovascular", "stroke"],
        guideline_types=[
            GuidelineType.DIAGNOSIS,
            GuidelineType.TREATMENT,
            GuidelineType.PREVENTION,
        ],
        search_patterns=["/guidelines/", "/statements/", "/clinical-statements/"],
        pdf_patterns=[r"\.pdf$", r"guidelines.*\.pdf", r"statement.*\.pdf"],
        metadata_patterns={
            "title": r"<title>(.*?)</title>",
            "doi": r"DOI:\s*(10\.\d+/.*?)(?:\s|<)",
            "authors": r"Authors:\s*(.*?)(?:\n|<)",
        },
        crawl_config={"max_depth": 2, "follow_links": True, "delay": 2.0},
        priority=1,
    ),
    "acc": GuidelineSource(
        name="American College of Cardiology",
        abbreviation="ACC",
        base_url="https://www.acc.org",
        organization_type=OrganizationType.PROFESSIONAL_SOCIETY,
        specialties=["cardiology", "cardiovascular"],
        guideline_types=[
            GuidelineType.DIAGNOSIS,
            GuidelineType.TREATMENT,
            GuidelineType.MANAGEMENT,
        ],
        search_patterns=["/guidelines/", "/clinical-guidance/", "/decision-pathways/"],
        pdf_patterns=[r"\.pdf$", r"guideline.*\.pdf", r"pathway.*\.pdf"],
        metadata_patterns={
            "title": r"<title>(.*?)</title>",
            "year": r"(\d{4})",
            "class": r"Class\s+(I{1,3}|IV)",
        },
        crawl_config={"max_depth": 2, "follow_links": True, "delay": 2.0},
        priority=1,
    ),
    "asco": GuidelineSource(
        name="American Society of Clinical Oncology",
        abbreviation="ASCO",
        base_url="https://ascopubs.org",
        organization_type=OrganizationType.PROFESSIONAL_SOCIETY,
        specialties=["oncology", "cancer", "hematology"],
        guideline_types=[
            GuidelineType.DIAGNOSIS,
            GuidelineType.TREATMENT,
            GuidelineType.SCREENING,
        ],
        search_patterns=[
            "/guidelines/",
            "/practice-guidelines/",
            "/clinical-practice-guidelines/",
        ],
        pdf_patterns=[r"\.pdf$", r"guideline.*\.pdf", r"clinical.*\.pdf"],
        metadata_patterns={
            "title": r"<title>(.*?)</title>",
            "doi": r"DOI:\s*(10\.\d+/.*?)(?:\s|<)",
            "cancer_type": r"Cancer Type:\s*(.*?)(?:\n|<)",
        },
        crawl_config={"max_depth": 2, "follow_links": True, "delay": 2.5},
        priority=1,
    ),
    "idsociety": GuidelineSource(
        name="Infectious Diseases Society of America",
        abbreviation="IDSA",
        base_url="https://www.idsociety.org",
        organization_type=OrganizationType.PROFESSIONAL_SOCIETY,
        specialties=["infectious_disease", "antimicrobial", "immunology"],
        guideline_types=[
            GuidelineType.DIAGNOSIS,
            GuidelineType.TREATMENT,
            GuidelineType.PREVENTION,
        ],
        search_patterns=[
            "/practice-guidelines/",
            "/guidelines/",
            "/clinical-practice/",
        ],
        pdf_patterns=[r"\.pdf$", r"guideline.*\.pdf", r"practice.*\.pdf"],
        metadata_patterns={
            "title": r"<title>(.*?)</title>",
            "pathogen": r"Pathogen:\s*(.*?)(?:\n|<)",
            "antimicrobial": r"Antimicrobial:\s*(.*?)(?:\n|<)",
        },
        crawl_config={"max_depth": 2, "follow_links": True, "delay": 2.0},
        priority=1,
    ),
    "diabetes": GuidelineSource(
        name="American Diabetes Association",
        abbreviation="ADA",
        base_url="https://diabetesjournals.org",
        organization_type=OrganizationType.PROFESSIONAL_SOCIETY,
        specialties=["endocrinology", "diabetes", "metabolism"],
        guideline_types=[
            GuidelineType.DIAGNOSIS,
            GuidelineType.TREATMENT,
            GuidelineType.MANAGEMENT,
        ],
        search_patterns=[
            "/care/",
            "/clinical-practice-recommendations/",
            "/standards/",
        ],
        pdf_patterns=[r"\.pdf$", r"standards.*\.pdf", r"recommendations.*\.pdf"],
        metadata_patterns={
            "title": r"<title>(.*?)</title>",
            "hba1c": r"HbA1c.*?(\d+\.?\d*%)",
            "glucose": r"glucose.*?(\d+\s*mg/dL)",
        },
        crawl_config={"max_depth": 2, "follow_links": True, "delay": 2.0},
        priority=1,
    ),
    "acog": GuidelineSource(
        name="American College of Obstetricians and Gynecologists",
        abbreviation="ACOG",
        base_url="https://www.acog.org",
        organization_type=OrganizationType.PROFESSIONAL_SOCIETY,
        specialties=["obstetrics", "gynecology", "womens_health"],
        guideline_types=[
            GuidelineType.DIAGNOSIS,
            GuidelineType.TREATMENT,
            GuidelineType.SCREENING,
        ],
        search_patterns=[
            "/clinical-guidance/",
            "/practice-bulletins/",
            "/committee-opinions/",
        ],
        pdf_patterns=[r"\.pdf$", r"bulletin.*\.pdf", r"opinion.*\.pdf"],
        metadata_patterns={
            "title": r"<title>(.*?)</title>",
            "number": r"No\.\s*(\d+)",
            "reaffirmed": r"Reaffirmed\s+(\d{4})",
        },
        crawl_config={"max_depth": 2, "follow_links": True, "delay": 2.0},
        priority=1,
    ),
    "aap": GuidelineSource(
        name="American Academy of Pediatrics",
        abbreviation="AAP",
        base_url="https://pediatrics.aappublications.org",
        organization_type=OrganizationType.PROFESSIONAL_SOCIETY,
        specialties=["pediatrics", "child_health", "adolescent_health"],
        guideline_types=[
            GuidelineType.DIAGNOSIS,
            GuidelineType.TREATMENT,
            GuidelineType.PREVENTION,
        ],
        search_patterns=[
            "/clinical-practice-guideline/",
            "/policy-statement/",
            "/technical-report/",
        ],
        pdf_patterns=[r"\.pdf$", r"guideline.*\.pdf", r"policy.*\.pdf"],
        metadata_patterns={
            "title": r"<title>(.*?)</title>",
            "age_group": r"Age.*?(\d+.*?years?)",
            "pediatric": r"Pediatric.*?(.*?)(?:\n|<)",
        },
        crawl_config={"max_depth": 2, "follow_links": True, "delay": 2.0},
        priority=1,
    ),
    # === SPECIALTY ORGANIZATIONS ===
    "asge": GuidelineSource(
        name="American Society for Gastrointestinal Endoscopy",
        abbreviation="ASGE",
        base_url="https://www.asge.org",
        organization_type=OrganizationType.PROFESSIONAL_SOCIETY,
        specialties=["gastroenterology", "endoscopy"],
        guideline_types=[GuidelineType.PROCEDURE, GuidelineType.DIAGNOSIS],
        search_patterns=["/guidelines/", "/clinical-practice/", "/technology/"],
        pdf_patterns=[r"\.pdf$", r"guideline.*\.pdf", r"technology.*\.pdf"],
        metadata_patterns={
            "title": r"<title>(.*?)</title>",
            "procedure": r"Procedure:\s*(.*?)(?:\n|<)",
            "indication": r"Indication:\s*(.*?)(?:\n|<)",
        },
        crawl_config={"max_depth": 2, "follow_links": True, "delay": 2.0},
        priority=2,
    ),
    "chest": GuidelineSource(
        name="American College of Chest Physicians",
        abbreviation="CHEST",
        base_url="https://journal.chestnet.org",
        organization_type=OrganizationType.PROFESSIONAL_SOCIETY,
        specialties=["pulmonology", "critical_care", "sleep_medicine"],
        guideline_types=[
            GuidelineType.DIAGNOSIS,
            GuidelineType.TREATMENT,
            GuidelineType.MANAGEMENT,
        ],
        search_patterns=["/guidelines/", "/clinical-practice/", "/antithrombotic/"],
        pdf_patterns=[r"\.pdf$", r"guideline.*\.pdf", r"antithrombotic.*\.pdf"],
        metadata_patterns={
            "title": r"<title>(.*?)</title>",
            "recommendation": r"Recommendation\s+(\d+\.?\d*)",
            "grade": r"Grade\s+([ABC])",
        },
        crawl_config={"max_depth": 2, "follow_links": True, "delay": 2.0},
        priority=2,
    ),
    # === QUALITY ORGANIZATIONS ===
    "nqf": GuidelineSource(
        name="National Quality Forum",
        abbreviation="NQF",
        base_url="https://www.qualityforum.org",
        organization_type=OrganizationType.CONSORTIUM,
        specialties=["quality_measures", "healthcare_quality"],
        guideline_types=[GuidelineType.MANAGEMENT],
        search_patterns=["/measures/", "/publications/", "/quality-measures/"],
        pdf_patterns=[r"\.pdf$", r"measure.*\.pdf", r"report.*\.pdf"],
        metadata_patterns={
            "title": r"<title>(.*?)</title>",
            "measure_id": r"NQF\s*#?\s*(\d+)",
            "domain": r"Domain:\s*(.*?)(?:\n|<)",
        },
        crawl_config={"max_depth": 2, "follow_links": True, "delay": 2.0},
        priority=3,
    ),
    "ahrq": GuidelineSource(
        name="Agency for Healthcare Research and Quality",
        abbreviation="AHRQ",
        base_url="https://www.ahrq.gov",
        organization_type=OrganizationType.GOVERNMENT,
        specialties=["healthcare_quality", "patient_safety", "evidence_based_medicine"],
        guideline_types=[GuidelineType.MANAGEMENT, GuidelineType.PREVENTION],
        search_patterns=["/guidelines/", "/tools/", "/research/", "/evidence/"],
        pdf_patterns=[r"\.pdf$", r"tool.*\.pdf", r"evidence.*\.pdf"],
        metadata_patterns={
            "title": r"<title>(.*?)</title>",
            "publication_date": r"Publication Date:\s*(\w+\s+\d{4})",
            "ahrq_number": r"AHRQ.*?(\d+-\w+\d*)",
        },
        crawl_config={"max_depth": 2, "follow_links": True, "delay": 1.5},
        priority=2,
    ),
}


def get_sources_by_specialty(specialty: str) -> List[GuidelineSource]:
    """Get guideline sources filtered by medical specialty."""
    return [
        source
        for source in CLINICAL_GUIDELINE_SOURCES.values()
        if specialty in source.specialties
    ]


def get_sources_by_priority(max_priority: int = 2) -> List[GuidelineSource]:
    """Get guideline sources filtered by priority level."""
    return [
        source
        for source in CLINICAL_GUIDELINE_SOURCES.values()
        if source.priority <= max_priority
    ]


def get_high_priority_sources() -> List[GuidelineSource]:
    """Get only the highest priority guideline sources."""
    return get_sources_by_priority(max_priority=1)
