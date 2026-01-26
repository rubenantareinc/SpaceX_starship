"""
scrape_sources.py
-----------------
Very small helper to pull text from a URL.

This is intentionally minimal (and polite):
- It fetches a URL once
- Extracts visible paragraph text
- Saves a JSONL record into data/raw/

For serious scraping, prefer official APIs or pre-approved datasets.
"""

import argparse
import json
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup


def extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    # remove script/style
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    text = "\n".join([p for p in paras if p])
    # normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="Article/report URL")
    ap.add_argument("--incident_id", required=True)
    ap.add_argument("--out", default="data/raw/scraped.jsonl")
    ap.add_argument("--source_type", default="news", choices=["news", "official", "community"])
    args = ap.parse_args()

    resp = requests.get(args.url, timeout=30, headers={"User-Agent": "StarshipAnomalyExplainer/0.1"})
    resp.raise_for_status()
    text = extract_text(resp.text)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "incident_id": args.incident_id,
        "source_url": args.url,
        "source_type": args.source_type,
        "text": text,
    }
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved to {out_path} ({len(text)} chars).")


if __name__ == "__main__":
    main()
