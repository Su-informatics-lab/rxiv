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
    "uspstf": GuidelineSource(
        name="U.S. Preventive Services Task Force",
        abbreviation="USPSTF",
        base_url="https://www.uspreventiveservicestaskforce.org",
        organization_type=OrganizationType.GOVERNMENT,
        specialties=["preventive_care", "screening", "primary_care"],
        guideline_types=[GuidelineType.SCREENING, GuidelineType.PREVENTION],
        search_patterns=["/recommendations/", "/Page/", "/uspstf/"],
        pdf_patterns=[r"\.pdf$", r"recommendation.*\.pdf", r"final.*\.pdf"],
        metadata_patterns={
            "title": r"<title>(.*?)</title>",
            "grade": r"Grade\s+([A-DF])",
            "recommendation": r"Recommendation:\s*(.*?)(?:\n|<)",
        },
        crawl_config={"max_depth": 2, "follow_links": True, "delay": 1.5},
        priority=1,
    ),
    "nih": GuidelineSource(
        name="National Institutes of Health",
        abbreviation="NIH",
        base_url="https://www.nih.gov",
        organization_type=OrganizationType.GOVERNMENT,
        specialties=["research", "clinical_trials", "biomedical_research"],
        guideline_types=[GuidelineType.TREATMENT, GuidelineType.DIAGNOSIS],
        search_patterns=["/health-information/", "/clinical-trials/", "/research/"],
        pdf_patterns=[r"\.pdf$", r"clinical.*\.pdf", r"research.*\.pdf"],
        metadata_patterns={
            "title": r"<title>(.*?)</title>",
            "institute": r"(NHLBI|NIDDK|NIMH|NCI|NIAID|NICHD)",
            "date": r"(\d{1,2}/\d{1,2}/\d{4}|\w+\s+\d{1,2},\s+\d{4})",
        },
        crawl_config={"max_depth": 2, "follow_links": True, "delay": 1.5},
        priority=2,
    ),
    "acp": GuidelineSource(
        name="American College of Physicians",
        abbreviation="ACP",
        base_url="https://www.acponline.org",
        organization_type=OrganizationType.PROFESSIONAL_SOCIETY,
        specialties=["internal_medicine", "primary_care"],
        guideline_types=[
            GuidelineType.DIAGNOSIS,
            GuidelineType.TREATMENT,
            GuidelineType.MANAGEMENT,
        ],
        search_patterns=["/clinical-information/guidelines/", "/clinical-guidance/"],
        pdf_patterns=[r"\.pdf$", r"guideline.*\.pdf", r"clinical.*\.pdf"],
        metadata_patterns={
            "title": r"<title>(.*?)</title>",
            "recommendation": r"Recommendation\s+(\d+)",
            "grade": r"Grade:\s*(.*?)(?:\n|<)",
        },
        crawl_config={"max_depth": 2, "follow_links": True, "delay": 2.0},
        priority=1,
    ),
    "apa_psych": GuidelineSource(
        name="American Psychiatric Association",
        abbreviation="APA",
        base_url="https://www.psychiatry.org",
        organization_type=OrganizationType.PROFESSIONAL_SOCIETY,
        specialties=["psychiatry", "mental_health", "behavioral_health"],
        guideline_types=[GuidelineType.DIAGNOSIS, GuidelineType.TREATMENT],
        search_patterns=[
            "/psychiatrists/practice/clinical-practice-guidelines/",
            "/practice/",
        ],
        pdf_patterns=[r"\.pdf$", r"guideline.*\.pdf", r"practice.*\.pdf"],
        metadata_patterns={
            "title": r"<title>(.*?)</title>",
            "dsm": r"DSM-5.*?(\d+\.\d+)",
            "disorder": r"Disorder:\s*(.*?)(?:\n|<)",
        },
        crawl_config={"max_depth": 2, "follow_links": True, "delay": 2.0},
        priority=2,
    ),
    "nccn": GuidelineSource(
        name="National Comprehensive Cancer Network",
        abbreviation="NCCN",
        base_url="https://www.nccn.org",
        organization_type=OrganizationType.CONSORTIUM,
        specialties=["oncology", "cancer", "hematology"],
        guideline_types=[
            GuidelineType.DIAGNOSIS,
            GuidelineType.TREATMENT,
            GuidelineType.SCREENING,
        ],
        search_patterns=["/guidelines/", "/professionals/physician_gls/"],
        pdf_patterns=[r"\.pdf$", r"guideline.*\.pdf", r"nccn.*\.pdf"],
        metadata_patterns={
            "title": r"<title>(.*?)</title>",
            "version": r"Version\s+(\d+\.\d{4})",
            "cancer_type": r"NCCN.*?(.*?)\s+Guidelines",
        },
        crawl_config={"max_depth": 2, "follow_links": True, "delay": 2.5},
        priority=1,
    ),
    "aafp": GuidelineSource(
        name="American Academy of Family Physicians",
        abbreviation="AAFP",
        base_url="https://www.aafp.org",
        organization_type=OrganizationType.PROFESSIONAL_SOCIETY,
        specialties=["family_medicine", "primary_care"],
        guideline_types=[
            GuidelineType.DIAGNOSIS,
            GuidelineType.TREATMENT,
            GuidelineType.PREVENTION,
        ],
        search_patterns=["/family-physician/patient-care/clinical-recommendations/"],
        pdf_patterns=[r"\.pdf$", r"recommendation.*\.pdf", r"clinical.*\.pdf"],
        metadata_patterns={
            "title": r"<title>(.*?)</title>",
            "strength": r"Strength of Recommendation:\s*([ABC])",
            "evidence": r"Level of Evidence:\s*(\d+)",
        },
        crawl_config={"max_depth": 2, "follow_links": True, "delay": 2.0},
        priority=2,
    ),
    "aan": GuidelineSource(
        name="American Academy of Neurology",
        abbreviation="AAN",
        base_url="https://www.aan.com",
        organization_type=OrganizationType.PROFESSIONAL_SOCIETY,
        specialties=["neurology", "neurological_disorders"],
        guideline_types=[
            GuidelineType.DIAGNOSIS,
            GuidelineType.TREATMENT,
            GuidelineType.MANAGEMENT,
        ],
        search_patterns=["/Guidelines/", "/practice-guidelines/"],
        pdf_patterns=[r"\.pdf$", r"guideline.*\.pdf", r"practice.*\.pdf"],
        metadata_patterns={
            "title": r"<title>(.*?)</title>",
            "level": r"Level\s+([ABC])",
            "neurological": r"(epilepsy|dementia|stroke|multiple sclerosis|parkinson)",
        },
        crawl_config={"max_depth": 2, "follow_links": True, "delay": 2.0},
        priority=2,
    ),
    "asa": GuidelineSource(
        name="American Society of Anesthesiologists",
        abbreviation="ASA",
        base_url="https://www.asahq.org",
        organization_type=OrganizationType.PROFESSIONAL_SOCIETY,
        specialties=["anesthesiology", "perioperative_care", "pain_management"],
        guideline_types=[GuidelineType.PROCEDURE, GuidelineType.MANAGEMENT],
        search_patterns=["/standards-and-guidelines/", "/practice-management/"],
        pdf_patterns=[r"\.pdf$", r"guideline.*\.pdf", r"standard.*\.pdf"],
        metadata_patterns={
            "title": r"<title>(.*?)</title>",
            "standard": r"Standard\s+(\w+)",
            "effective": r"Effective:\s*(\d{1,2}/\d{1,2}/\d{4})",
        },
        crawl_config={"max_depth": 2, "follow_links": True, "delay": 2.0},
        priority=3,
    ),
    "clsi": GuidelineSource(
        name="Clinical and Laboratory Standards Institute",
        abbreviation="CLSI",
        base_url="https://clsi.org",
        organization_type=OrganizationType.CONSORTIUM,
        specialties=["laboratory_medicine", "diagnostics", "testing_standards"],
        guideline_types=[GuidelineType.PROCEDURE, GuidelineType.DIAGNOSIS],
        search_patterns=["/standards/", "/documents/"],
        pdf_patterns=[r"\.pdf$", r"standard.*\.pdf", r"guideline.*\.pdf"],
        metadata_patterns={
            "title": r"<title>(.*?)</title>",
            "document_id": r"(EP\d+|GP\d+|M\d+)",
            "edition": r"(\d+)(?:st|nd|rd|th)\s+Edition",
        },
        crawl_config={"max_depth": 2, "follow_links": True, "delay": 2.0},
        priority=3,
    ),
    "jointcommission": GuidelineSource(
        name="The Joint Commission",
        abbreviation="TJC",
        base_url="https://www.jointcommission.org",
        organization_type=OrganizationType.CONSORTIUM,
        specialties=["hospital_safety", "accreditation", "quality_improvement"],
        guideline_types=[GuidelineType.MANAGEMENT, GuidelineType.PROCEDURE],
        search_patterns=["/standards/", "/resources/", "/patient-safety/"],
        pdf_patterns=[r"\.pdf$", r"standard.*\.pdf", r"manual.*\.pdf"],
        metadata_patterns={
            "title": r"<title>(.*?)</title>",
            "element": r"Element of Performance\s*(\d+)",
            "effective": r"Effective:\s*(\w+\s+\d{1,2},\s+\d{4})",
        },
        crawl_config={"max_depth": 2, "follow_links": True, "delay": 2.0},
        priority=3,
    ),
    "ecri": GuidelineSource(
        name="ECRI Guidelines Trust",
        abbreviation="ECRI",
        base_url="https://guidelines.ecri.org",
        organization_type=OrganizationType.CONSORTIUM,
        specialties=["cross_specialty", "evidence_based_medicine"],
        guideline_types=[
            GuidelineType.DIAGNOSIS,
            GuidelineType.TREATMENT,
            GuidelineType.MANAGEMENT,
        ],
        search_patterns=["/", "/search/", "/browse/"],
        pdf_patterns=[r"\.pdf$", r"guideline.*\.pdf", r"evidence.*\.pdf"],
        metadata_patterns={
            "title": r"<title>(.*?)</title>",
            "organization": r"Source:\s*(.*?)(?:\n|<)",
            "updated": r"Updated:\s*(\w+\s+\d{4})",
        },
        crawl_config={"max_depth": 1, "follow_links": True, "delay": 2.5},
        priority=2,
    ),
    "icer": GuidelineSource(
        name="Institute for Clinical and Economic Review",
        abbreviation="ICER",
        base_url="https://icer.org",
        organization_type=OrganizationType.CONSORTIUM,
        specialties=["health_economics", "value_based_care", "cost_effectiveness"],
        guideline_types=[GuidelineType.MANAGEMENT],
        search_patterns=["/assessments/", "/reports/", "/evidence-reports/"],
        pdf_patterns=[r"\.pdf$", r"report.*\.pdf", r"assessment.*\.pdf"],
        metadata_patterns={
            "title": r"<title>(.*?)</title>",
            "therapeutic_area": r"Therapeutic Area:\s*(.*?)(?:\n|<)",
            "publication_date": r"Publication Date:\s*(\w+\s+\d{1,2},\s+\d{4})",
        },
        crawl_config={"max_depth": 2, "follow_links": True, "delay": 2.0},
        priority=3,
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


def get_government_sources() -> List[GuidelineSource]:
    """Get government agency sources."""
    return [
        source
        for source in CLINICAL_GUIDELINE_SOURCES.values()
        if source.organization_type == OrganizationType.GOVERNMENT
    ]


def get_professional_society_sources() -> List[GuidelineSource]:
    """Get professional medical society sources."""
    return [
        source
        for source in CLINICAL_GUIDELINE_SOURCES.values()
        if source.organization_type == OrganizationType.PROFESSIONAL_SOCIETY
    ]
