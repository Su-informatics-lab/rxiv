# download_metadata.py
import requests
import time
import json
from pathlib import Path
from tqdm import tqdm

API_BASE = "https://api.biorxiv.org/details"
CHUNK_SIZE = 100  # max allowed
SOURCES = {"biorxiv": "2013-01-01", "medrxiv": "2019-01-01"}
END_DATE = "2025-07-01"

def fetch_all(source, start_date, end_date, out_file):
    print(f"Fetching {source} metadata from {start_date} to {end_date}")
    url = f"{API_BASE}/{source}/{start_date}/{end_date}/0"
    res = requests.get(url).json()
    total = int(res["messages"][0]["total"])
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w") as f:
        for i in tqdm(range(0, total, CHUNK_SIZE)):
            url = f"{API_BASE}/{source}/{start_date}/{end_date}/{i}"
            for _ in range(3):  # retry up to 3 times
                try:
                    res = requests.get(url, timeout=10)
                    data = res.json()["collection"]
                    for entry in data:
                        f.write(json.dumps(entry) + "\n")
                    break
                except Exception as e:
                    print(f"Retrying ({source} offset {i}): {e}")
                    time.sleep(3)

if __name__ == "__main__":
    for src, start in SOURCES.items():
        out_file = f"data/{src}_metadata.jsonl"
        fetch_all(src, start, END_DATE, out_file)
