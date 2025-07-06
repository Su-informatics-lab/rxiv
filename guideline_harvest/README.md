# Clinical Guidelines Harvesting

Harvests clinical guidelines from major US medical organizations using crawl4ai web crawling.

> ⚙️ For HPC systems (e.g., IU Quartz), we recommend using Apptainer (formerly Singularity) for full browser support.

## Features

- **Web crawling**: Uses crawl4ai with JavaScript support
- **Sources**: 15+ major US medical organizations (CDC, FDA, AHA, ACC, ASCO, etc.)
- **Metadata extraction**: Automated extraction with spaCy NLP
- **PDF processing**: Multi-library validation and text extraction
- **Quality assessment**: Automatic relevance scoring

## Setup (with Apptainer)

First, set up a containerized environment with Playwright support:

```bash
# 1. load container
module load apptainer

# 2. pull container
apptainer pull playwright.sif docker://mcr.microsoft.com/playwright/python:v1.52.0-jammy

# 3. enter the container and install dependencies
apptainer shell --bind $(pwd) playwright.sif

# 4. set up environment variables
echo 'export PATH=$HOME/.local/bin:$PATH' > container_env.sh
echo 'export PLAYWRIGHT_BROWSERS_PATH=$HOME/.cache/ms-playwright' >> container_env.sh
source container_env.sh

# inside container:
pip install -r guideline_harvest/requirements.txt
python -m spacy download en_core_web_sm
playwright install  # we use isntall v1.52.0
```

## Usage

```bash
# basic use
python crawl.py

# cardiology guidelines only
python crawl.py --sources cardiology

# custom settings
python crawl.py --sources high_priority --max-concurrent 2 --delay 2.0
```

## Supported Organizations

| Organization | Abbrev | Type | Specialties | Guidelines/Recommendations |
|--------------|--------|------|-------------|---------------------------|
| **Government Agencies** |
| Centers for Disease Control and Prevention | CDC | Government | Infectious disease, epidemiology, public health | Prevention, screening, management guidelines |
| Food and Drug Administration | FDA | Government | Drug safety, medical devices | Medication, procedure guidance |
| Centers for Medicare & Medicaid Services | CMS | Government | Healthcare policy, quality measures | Coverage, reimbursement policies |
| Agency for Healthcare Research and Quality | AHRQ | Government | Healthcare quality, patient safety | Evidence-based medicine tools |
| U.S. Preventive Services Task Force | USPSTF | Government | Preventive care, screening | Screening recommendations with grades |
| National Institutes of Health | NIH | Government | Research, clinical trials | Disease-specific research guidance |
| **Professional Medical Societies** |
| American Heart Association | AHA | Professional Society | Cardiology, cardiovascular, stroke | Clinical practice guidelines, statements |
| American College of Cardiology | ACC | Professional Society | Cardiology, cardiovascular | Clinical guidance, decision pathways |
| American Society of Clinical Oncology | ASCO | Professional Society | Oncology, cancer, hematology | Cancer treatment guidelines |
| Infectious Diseases Society of America | IDSA | Professional Society | Infectious disease, antimicrobial | Treatment and prevention guidelines |
| American Diabetes Association | ADA | Professional Society | Endocrinology, diabetes | Standards of care, recommendations |
| American College of Obstetricians and Gynecologists | ACOG | Professional Society | Obstetrics, gynecology | Practice bulletins, committee opinions |
| American Academy of Pediatrics | AAP | Professional Society | Pediatrics, child health | Clinical practice guidelines, policies |
| American College of Physicians | ACP | Professional Society | Internal medicine, primary care | Clinical guidance recommendations |
| American Psychiatric Association | APA | Professional Society | Psychiatry, mental health | Practice guidelines, DSM-based |
| National Comprehensive Cancer Network | NCCN | Consortium | Oncology, cancer treatment | Evidence-based cancer guidelines |
| American Academy of Family Physicians | AAFP | Professional Society | Family medicine, primary care | Clinical recommendations |
| American Academy of Neurology | AAN | Professional Society | Neurology, neurological disorders | Practice guidelines for neurological conditions |
| American Society of Anesthesiologists | ASA | Professional Society | Anesthesiology, perioperative care | Practice standards, guidelines |
| **Specialty Organizations** |
| American Society for Gastrointestinal Endoscopy | ASGE | Professional Society | Gastroenterology, endoscopy | Procedure guidelines, technology assessments |
| American College of Chest Physicians | CHEST | Professional Society | Pulmonology, critical care | Clinical practice guidelines |
| National Quality Forum | NQF | Consortium | Quality measures, healthcare quality | Performance measures, quality standards |
| Clinical and Laboratory Standards Institute | CLSI | Consortium | Laboratory medicine, diagnostics | Testing standards, procedures |
| The Joint Commission | TJC | Consortium | Hospital safety, accreditation | Safety standards, accreditation requirements |
| ECRI Guidelines Trust | ECRI | Consortium | Cross-specialty, evidence-based | Guideline aggregation and review |
| Institute for Clinical and Economic Review | ICER | Consortium | Health economics, value-based care | Cost-effectiveness assessments |

## Configuration

```python
config = {
    "max_concurrent_sources": 3,
    "request_delay": 1.0,
    "min_pdf_size": 50 * 1024,
    "validate_pdfs": True
}
```

## Output

```
guidelines_data/
├── pdfs/                  # Downloaded PDFs
│   ├── CDC_diabetes_prevention_2024.pdf
│   └── AHA_heart_failure_management_2023.pdf
├── metadata/              # JSON metadata for each PDF
│   ├── CDC_diabetes_prevention_2024.pdf.json
│   └── AHA_heart_failure_management_2023.pdf.json
├── logs/                  # Harvesting logs
└── harvest_results.json   # Complete results
```

### Example Metadata
```json
{
  "guideline_info": {
    "title": "2024 Heart Failure Management Guidelines",
    "url": "https://ahajournals.org/guidelines/heart-failure-2024",
    "metadata": {
      "organization": "American Heart Association",
      "specialty": "cardiology",
      "guideline_type": "practice_guideline",
      "publication_date": "2024-01-15T00:00:00",
      "authors": ["Dr. John Smith", "Dr. Jane Doe"]
    }
  },
  "download_result": {
    "filename": "AHA_heart_failure_management_2024.pdf",
    "file_size": 2457600,
    "status": "downloaded"
  }
}
```

## Specialties

Available filters:
- `high_priority`: CDC, AHA, ACC, ASCO, etc.
- `cardiology`, `oncology`, `infectious_disease`, `endocrinology`
- `obstetrics`, `pediatrics`, `gastroenterology`, `pulmonology`

## Integration

Process with main pipeline:
```bash
cd ../processing
python process_papers.py ../guideline_harvest/guidelines_data/pdfs/ processed_guidelines/
```