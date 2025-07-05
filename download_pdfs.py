# download_pdfs.py
import os
import json
import time
from pathlib import Path
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from tqdm import tqdm

MAX_RETRIES = 3
NUM_THREADS = 16

def safe_doi_to_filename(doi):
    return quote(doi.replace("/", "_"), safe="")

def construct_pdf_url(source, doi, version):
    base = "https://www.biorxiv.org" if source == "biorxiv" else "https://www.medrxiv.org"
    return f"{base}/content/{doi}v{version}.full.pdf"

def download_pdf(entry, source, out_dir):
    doi = entry["doi"]
    version = entry["version"]
    out_path = Path(out_dir) / f"{safe_doi_to_filename(doi)}v{version}.pdf"
    if out_path.exists():
        return "exists"

    url = construct_pdf_url(source, doi, version)
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200 and r.headers["Content-Type"] == "application/pdf":
                with open(out_path, "wb") as f:
                    f.write(r.content)
                return "downloaded"
            else:
                time.sleep(2)
        except Exception:
            time.sleep(2)
    return f"failed: {doi}"

def load_entries(path):
    with open(path) as f:
        return [json.loads(l) for l in f]

def run_parallel(entries, source, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    results = []
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = [executor.submit(download_pdf, e, source, out_dir) for e in entries]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            results.append(fut.result())
    return results

if __name__ == "__main__":
    sources = ["biorxiv", "medrxiv"]
    for source in sources:
        entries = load_entries(f"data/{source}_metadata.jsonl")
        out_dir = f"pdfs/{source}"
        run_parallel(entries, source, out_dir)
